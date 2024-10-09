import Mathlib

namespace metallic_sheet_dimension_l2299_229992

theorem metallic_sheet_dimension (x : ℝ) (h₁ : ∀ (l w h : ℝ), l = x - 8 → w = 28 → h = 4 → l * w * h = 4480) : x = 48 :=
sorry

end metallic_sheet_dimension_l2299_229992


namespace find_a_l2299_229974

def A (x : ℝ) : Prop := x^2 + 6 * x < 0
def B (a x : ℝ) : Prop := x^2 - (a - 2) * x - 2 * a < 0
def U (x : ℝ) : Prop := -6 < x ∧ x < 5

theorem find_a : (∀ x, A x ∨ ∃ a, B a x) = U x -> a = 5 :=
by
  sorry

end find_a_l2299_229974


namespace function_range_l2299_229969

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

theorem function_range : 
  ∀ x : ℝ, (0 < x ∧ x < Real.pi) → 1 ≤ f x ∧ f x ≤ 3 / 2 :=
by
  intro x
  sorry

end function_range_l2299_229969


namespace value_of_f_3_div_2_l2299_229986

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom even_f : ∀ x : ℝ, f (x) = f (-x)
axiom f_in_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = x + 1

theorem value_of_f_3_div_2 : f (3 / 2) = 3 / 2 := by
  sorry

end value_of_f_3_div_2_l2299_229986


namespace bc_money_l2299_229963

variables (A B C : ℕ)

theorem bc_money (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : C = 50) : B + C = 150 :=
sorry

end bc_money_l2299_229963


namespace line_equation_through_point_line_equation_sum_of_intercepts_l2299_229934

theorem line_equation_through_point (x y : ℝ) (h : y = 2 * x + 5)
  (hx : x = -2) (hy : y = 1) : 2 * x - y + 5 = 0 :=
by {
  sorry
}

theorem line_equation_sum_of_intercepts (x y : ℝ) (h : y = 2 * x + 6)
  (hx : x = -3) (hy : y = 3) : 2 * x - y + 6 = 0 :=
by {
  sorry
}

end line_equation_through_point_line_equation_sum_of_intercepts_l2299_229934


namespace option_d_is_correct_l2299_229976

theorem option_d_is_correct : (-2 : ℤ) ^ 3 = -8 := by
  sorry

end option_d_is_correct_l2299_229976


namespace relationship_between_number_and_value_l2299_229979

theorem relationship_between_number_and_value (n v : ℝ) (h1 : n = 7) (h2 : n - 4 = 21 * v) : v = 1 / 7 :=
  sorry

end relationship_between_number_and_value_l2299_229979


namespace solve_n_l2299_229949

/-
Define the condition for the problem.
Given condition: \(\frac{1}{n+1} + \frac{2}{n+1} + \frac{n}{n+1} = 4\)
-/

noncomputable def condition (n : ℚ) : Prop :=
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1)) = 4

/-
The theorem to prove: Value of \( n \) that satisfies the condition is \( n = -\frac{1}{3} \)
-/
theorem solve_n : ∃ n : ℚ, condition n ∧ n = -1 / 3 :=
by
  sorry

end solve_n_l2299_229949


namespace downstream_speed_is_40_l2299_229931

variable (Vu : ℝ) (Vs : ℝ) (Vd : ℝ)

theorem downstream_speed_is_40 (h1 : Vu = 26) (h2 : Vs = 33) :
  Vd = 40 :=
by
  sorry

end downstream_speed_is_40_l2299_229931


namespace painter_total_fence_painted_l2299_229900

theorem painter_total_fence_painted : 
  ∀ (L T W Th F : ℕ), 
  (T = W) → (W = Th) → 
  (L = T / 2) → 
  (F = 2 * T * (6 / 8)) → 
  (F = L + 300) → 
  (L + T + W + Th + F = 1500) :=
by
  sorry

end painter_total_fence_painted_l2299_229900


namespace rectangle_pairs_l2299_229960

theorem rectangle_pairs :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 = 18} = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
by { sorry }

end rectangle_pairs_l2299_229960


namespace original_time_taken_by_bullet_train_is_50_minutes_l2299_229933

-- Define conditions as assumptions
variables (T D : ℝ) (h0 : D = 48 * T) (h1 : D = 60 * (40 / 60))

-- Define the theorem we want to prove
theorem original_time_taken_by_bullet_train_is_50_minutes :
  T = 50 / 60 :=
by
  sorry

end original_time_taken_by_bullet_train_is_50_minutes_l2299_229933


namespace find_extra_digit_l2299_229981

theorem find_extra_digit (x y a : ℕ) (hx : x + y = 23456) (h10x : 10 * x + a + y = 55555) (ha : 0 ≤ a ∧ a ≤ 9) : a = 5 :=
by
  sorry

end find_extra_digit_l2299_229981


namespace number_of_bowls_l2299_229901

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l2299_229901


namespace stratified_sample_size_is_correct_l2299_229967

def workshop_A_produces : ℕ := 120
def workshop_B_produces : ℕ := 90
def workshop_C_produces : ℕ := 60
def sample_from_C : ℕ := 4

def total_products : ℕ := workshop_A_produces + workshop_B_produces + workshop_C_produces

noncomputable def sampling_ratio : ℚ := (sample_from_C:ℚ) / (workshop_C_produces:ℚ)

noncomputable def sample_size : ℚ := total_products * sampling_ratio

theorem stratified_sample_size_is_correct :
  sample_size = 18 := by
  sorry

end stratified_sample_size_is_correct_l2299_229967


namespace Einstein_sold_25_cans_of_soda_l2299_229935

def sell_snacks_proof : Prop :=
  let pizza_price := 12
  let fries_price := 0.30
  let soda_price := 2
  let goal := 500
  let pizza_boxes := 15
  let fries_packs := 40
  let still_needed := 258
  let earned_from_pizza := pizza_boxes * pizza_price
  let earned_from_fries := fries_packs * fries_price
  let total_earned := earned_from_pizza + earned_from_fries
  let total_have := goal - still_needed
  let earned_from_soda := total_have - total_earned
  let cans_of_soda_sold := earned_from_soda / soda_price
  cans_of_soda_sold = 25

theorem Einstein_sold_25_cans_of_soda : sell_snacks_proof := by
  sorry

end Einstein_sold_25_cans_of_soda_l2299_229935


namespace find_n_in_range_l2299_229932

theorem find_n_in_range : ∃ n, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [MOD 7] ∧ n = 8 := 
by
  sorry

end find_n_in_range_l2299_229932


namespace notebook_pre_tax_cost_eq_l2299_229905

theorem notebook_pre_tax_cost_eq :
  (∃ (n c X : ℝ), n + c = 3 ∧ n = 2 + c ∧ 1.1 * X = 3.3 ∧ X = n + c → n = 2.5) :=
by
  sorry

end notebook_pre_tax_cost_eq_l2299_229905


namespace average_monthly_sales_booster_club_l2299_229923

noncomputable def monthly_sales : List ℕ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

noncomputable def average_sales (sales : List ℕ) : ℝ :=
  (sales.foldr (λ x acc => x + acc) 0 : ℕ) / sales.length

theorem average_monthly_sales_booster_club : average_sales monthly_sales = 122.92 := by
  sorry

end average_monthly_sales_booster_club_l2299_229923


namespace fourth_is_20_fewer_than_third_l2299_229908

-- Definitions of the number of road signs at each intersection
def first_intersection := 40
def second_intersection := first_intersection + first_intersection / 4
def third_intersection := 2 * second_intersection
def total_signs := 270
def fourth_intersection := total_signs - (first_intersection + second_intersection + third_intersection)

-- Proving the fourth intersection has 20 fewer signs than the third intersection
theorem fourth_is_20_fewer_than_third : third_intersection - fourth_intersection = 20 :=
by
  -- This is a placeholder for the proof
  sorry

end fourth_is_20_fewer_than_third_l2299_229908


namespace find_m_value_l2299_229985

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)
def vector_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the condition that vector_sum is parallel to vector_b
def vectors_parallel (m : ℝ) : Prop :=
  let (x1, y1) := vector_sum m
  let (x2, y2) := vector_b
  x1 * y2 - x2 * y1 = 0

-- The statement to prove
theorem find_m_value : ∃ m : ℝ, vectors_parallel m ∧ m = -2 / 3 :=
by {
  sorry
}

end find_m_value_l2299_229985


namespace solution_set_inequality_l2299_229906

theorem solution_set_inequality (t : ℝ) (ht : 0 < t ∧ t < 1) :
  {x : ℝ | x^2 - (t + t⁻¹) * x + 1 < 0} = {x : ℝ | t < x ∧ x < t⁻¹} :=
sorry

end solution_set_inequality_l2299_229906


namespace goose_eggs_count_l2299_229929

theorem goose_eggs_count 
  (E : ℕ) 
  (hatch_rate : ℚ)
  (survive_first_month_rate : ℚ)
  (survive_first_year_rate : ℚ)
  (geese_survived_first_year : ℕ)
  (no_more_than_one_goose_per_egg : Prop) 
  (hatch_eq : hatch_rate = 2/3) 
  (survive_first_month_eq : survive_first_month_rate = 3/4) 
  (survive_first_year_eq : survive_first_year_rate = 2/5) 
  (geese_survived_eq : geese_survived_first_year = 130):
  E = 650 :=
by
  sorry

end goose_eggs_count_l2299_229929


namespace perimeter_shaded_area_is_942_l2299_229959

-- Definition involving the perimeter of the shaded area of the circles
noncomputable def perimeter_shaded_area (s : ℝ) : ℝ := 
  4 * 75 * 3.14

-- Main theorem stating that if the side length of the octagon is 100 cm,
-- then the perimeter of the shaded area is 942 cm.
theorem perimeter_shaded_area_is_942 :
  perimeter_shaded_area 100 = 942 := 
  sorry

end perimeter_shaded_area_is_942_l2299_229959


namespace probability_at_least_one_multiple_of_4_is_correct_l2299_229988

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 100
  let multiples_of_4 := 25
  let non_multiples_of_4 := total_numbers - multiples_of_4
  let p_non_multiple := (non_multiples_of_4 : ℚ) / total_numbers
  let p_both_non_multiples := p_non_multiple^2
  let p_at_least_one_multiple := 1 - p_both_non_multiples
  p_at_least_one_multiple

theorem probability_at_least_one_multiple_of_4_is_correct :
  probability_at_least_one_multiple_of_4 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_is_correct_l2299_229988


namespace find_m_range_l2299_229911

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)
variable (m : R)

-- Define that the function f is monotonically increasing
def monotonically_increasing (f : R → R) : Prop :=
  ∀ ⦃x y : R⦄, x ≤ y → f x ≤ f y

-- Lean statement for the proof problem
theorem find_m_range (h1 : monotonically_increasing f) (h2 : f (2 * m - 3) > f (-m)) : m > 1 :=
by
  sorry

end find_m_range_l2299_229911


namespace algebraic_expression_positive_l2299_229998

theorem algebraic_expression_positive (a b : ℝ) : 
  a^2 + b^2 + 4*b - 2*a + 6 > 0 :=
by sorry

end algebraic_expression_positive_l2299_229998


namespace solve_inequality_l2299_229990

def p (x : ℝ) : ℝ := x^2 - 5*x + 3

theorem solve_inequality (x : ℝ) : 
  abs (p x) < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
sorry

end solve_inequality_l2299_229990


namespace sequence_x_value_l2299_229922

theorem sequence_x_value (x : ℕ) (h1 : 3 - 1 = 2) (h2 : 6 - 3 = 3) (h3 : 10 - 6 = 4) (h4 : x - 10 = 5) : x = 15 :=
by
  sorry

end sequence_x_value_l2299_229922


namespace part1_part2_l2299_229995

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3 ^ x + 1) + a

theorem part1 (h : ∀ x : ℝ, f (-x) a = -f x a) : a = -1 :=
by sorry

noncomputable def f' (x : ℝ) : ℝ := 2 / (3 ^ x + 1) - 1

theorem part2 : ∀ t : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f' x + 1 = t ↔ 1 / 2 ≤ t ∧ t ≤ 1 :=
by sorry

end part1_part2_l2299_229995


namespace sum_of_undefined_values_l2299_229907

theorem sum_of_undefined_values (y : ℝ) :
  (y^2 - 7 * y + 12 = 0) → y = 3 ∨ y = 4 → (3 + 4 = 7) :=
by
  intro hy
  intro hy'
  sorry

end sum_of_undefined_values_l2299_229907


namespace raman_profit_percentage_l2299_229942

theorem raman_profit_percentage
  (cost1 weight1 rate1 : ℕ) (cost2 weight2 rate2 : ℕ) (total_cost_mix total_weight mixing_rate selling_rate profit profit_percentage : ℕ)
  (h_cost1 : cost1 = weight1 * rate1)
  (h_cost2 : cost2 = weight2 * rate2)
  (h_total_cost_mix : total_cost_mix = cost1 + cost2)
  (h_total_weight : total_weight = weight1 + weight2)
  (h_mixing_rate : mixing_rate = total_cost_mix / total_weight)
  (h_selling_price : selling_rate * total_weight = profit + total_cost_mix)
  (h_profit : profit = selling_rate * total_weight - total_cost_mix)
  (h_profit_percentage : profit_percentage = (profit * 100) / total_cost_mix)
  (h_weight1 : weight1 = 54)
  (h_rate1 : rate1 = 150)
  (h_weight2 : weight2 = 36)
  (h_rate2 : rate2 = 125)
  (h_selling_rate_value : selling_rate = 196) :
  profit_percentage = 40 :=
sorry

end raman_profit_percentage_l2299_229942


namespace prime_factors_sum_l2299_229947

theorem prime_factors_sum (w x y z t : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^t = 107100) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * t = 38 :=
sorry

end prime_factors_sum_l2299_229947


namespace andrea_needs_1500_sod_squares_l2299_229970

-- Define the measurements of the yard sections
def section1_length : ℕ := 30
def section1_width : ℕ := 40
def section2_length : ℕ := 60
def section2_width : ℕ := 80

-- Define the measurements of the sod square
def sod_length : ℕ := 2
def sod_width : ℕ := 2

-- Compute the areas
def area_section1 : ℕ := section1_length * section1_width
def area_section2 : ℕ := section2_length * section2_width
def total_area : ℕ := area_section1 + area_section2

-- Compute the area of one sod square
def area_sod : ℕ := sod_length * sod_width

-- Compute the number of sod squares needed
def num_sod_squares : ℕ := total_area / area_sod

-- Theorem and proof placeholder
theorem andrea_needs_1500_sod_squares : num_sod_squares = 1500 :=
by {
  -- Place proof here
  sorry
}

end andrea_needs_1500_sod_squares_l2299_229970


namespace problem_1_problem_2_l2299_229926

def f (a x : ℝ) : ℝ := |a - 3 * x| - |2 + x|

theorem problem_1 (x : ℝ) : f 2 x ≤ 3 ↔ -3 / 4 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

theorem problem_2 (a x : ℝ) : f a x ≥ 1 - a + 2 * |2 + x| → a ≥ -5 / 2 := by
  sorry

end problem_1_problem_2_l2299_229926


namespace remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l2299_229943

theorem remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2 :
  (x^15 - 1) % (x + 1) = -2 := 
sorry

end remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l2299_229943


namespace unit_cubes_with_paint_l2299_229961

/-- Conditions:
1. Cubes with each side one inch long are glued together to form a larger cube.
2. The larger cube's face is painted with red color and the entire assembly is taken apart.
3. 23 small cubes are found with no paints on them.
-/
theorem unit_cubes_with_paint (n : ℕ) (h1 : n^3 - (n - 2)^3 = 23) (h2 : n = 4) :
    n^3 - 23 = 41 :=
by
  sorry

end unit_cubes_with_paint_l2299_229961


namespace find_positive_integer_pairs_l2299_229917

theorem find_positive_integer_pairs :
  ∀ (m n : ℕ), m > 0 ∧ n > 0 → ∃ k : ℕ, (2^n - 13^m = k^3) ↔ (m = 2 ∧ n = 9) :=
by
  sorry

end find_positive_integer_pairs_l2299_229917


namespace pedal_triangle_angle_pedal_triangle_angle_equality_l2299_229965

variables {A B C T_A T_B T_C: Type*}
variables {α β γ : Real}
variables {triangle : ∀ (A B C : Type*) (α β γ : Real), α ≤ β ∧ β ≤ γ ∧ γ < 90}

theorem pedal_triangle_angle
  (h : α ≤ β ∧ β ≤ γ ∧ γ < 90)
  (angles : 180 - 2 * α ≥ γ) :
  true :=
sorry

theorem pedal_triangle_angle_equality
  (h : α = β)
  (angles : (45 < α ∧ α = β ∧ α ≤ 60) ∧ (60 ≤ γ ∧ γ < 90)) :
  true :=
sorry

end pedal_triangle_angle_pedal_triangle_angle_equality_l2299_229965


namespace original_price_of_sarees_l2299_229945

theorem original_price_of_sarees (P : ℝ) (h1 : 0.95 * 0.80 * P = 133) : P = 175 :=
sorry

end original_price_of_sarees_l2299_229945


namespace first_term_of_arithmetic_sequence_l2299_229997

theorem first_term_of_arithmetic_sequence (a : ℕ) (median last_term : ℕ) 
  (h_arithmetic_progression : true) (h_median : median = 1010) (h_last_term : last_term = 2015) :
  a = 5 :=
by
  have h1 : 2 * median = 2020 := by sorry
  have h2 : last_term + a = 2020 := by sorry
  have h3 : 2015 + a = 2020 := by sorry
  have h4 : a = 2020 - 2015 := by sorry
  have h5 : a = 5 := by sorry
  exact h5

end first_term_of_arithmetic_sequence_l2299_229997


namespace honey_teas_l2299_229962

-- Definitions corresponding to the conditions
def evening_cups := 2
def evening_servings_per_cup := 2
def morning_cups := 1
def morning_servings_per_cup := 1
def afternoon_cups := 1
def afternoon_servings_per_cup := 1
def servings_per_ounce := 6
def container_ounces := 16

-- Calculation for total servings of honey per day and total days until the container is empty
theorem honey_teas :
  (container_ounces * servings_per_ounce) / 
  (evening_cups * evening_servings_per_cup +
   morning_cups * morning_servings_per_cup +
   afternoon_cups * afternoon_servings_per_cup) = 16 :=
by
  sorry

end honey_teas_l2299_229962


namespace speed_conversion_l2299_229910

theorem speed_conversion (s : ℚ) (h : s = 13 / 48) : 
  ((13 / 48) * 3.6 = 0.975) :=
by
  sorry

end speed_conversion_l2299_229910


namespace tiles_needed_l2299_229983

def ft_to_inch (x : ℕ) : ℕ := x * 12

def height_ft : ℕ := 10
def length_ft : ℕ := 15
def tile_size_sq_inch : ℕ := 1

def height_inch : ℕ := ft_to_inch height_ft
def length_inch : ℕ := ft_to_inch length_ft
def area_sq_inch : ℕ := height_inch * length_inch

theorem tiles_needed : 
  height_ft = 10 ∧ length_ft = 15 ∧ tile_size_sq_inch = 1 →
  area_sq_inch = 21600 :=
by
  intro h
  exact sorry

end tiles_needed_l2299_229983


namespace max_value_g_eq_3_in_interval_l2299_229955

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end max_value_g_eq_3_in_interval_l2299_229955


namespace cube_side_length_l2299_229989

theorem cube_side_length (s : ℝ) (h : 6 * s^2 = 864) : s = 12 := by
  sorry

end cube_side_length_l2299_229989


namespace max_intersections_cos_circle_l2299_229914

theorem max_intersections_cos_circle :
  let circle := λ x y => (x - 4)^2 + y^2 = 25
  let cos_graph := λ x => (x, Real.cos x)
  ∀ x y, (circle x y ∧ y = Real.cos x) → (∃ (p : ℕ), p ≤ 8) := sorry

end max_intersections_cos_circle_l2299_229914


namespace M_inter_N_eq_interval_l2299_229925

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem M_inter_N_eq_interval : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} := 
  sorry

end M_inter_N_eq_interval_l2299_229925


namespace smallest_positive_integer_l2299_229996

theorem smallest_positive_integer (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 2)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 4) :
  a = 59 :=
sorry

end smallest_positive_integer_l2299_229996


namespace eggs_supplied_l2299_229918

-- Define the conditions
def daily_eggs_first_store (D : ℕ) : ℕ := 12 * D
def daily_eggs_second_store : ℕ := 30
def total_weekly_eggs (D : ℕ) : ℕ := 7 * (daily_eggs_first_store D + daily_eggs_second_store)

-- Statement: prove that if the total number of eggs supplied in a week is 630,
-- then Mark supplies 5 dozen eggs to the first store each day.
theorem eggs_supplied (D : ℕ) (h : total_weekly_eggs D = 630) : D = 5 :=
by
  sorry

end eggs_supplied_l2299_229918


namespace angle_sum_unique_l2299_229964

theorem angle_sum_unique (α β : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : β ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan α + Real.tan β - Real.tan α * Real.tan β + 1 = 0) : 
  α + β = 7 * π / 4 :=
sorry

end angle_sum_unique_l2299_229964


namespace P_work_time_l2299_229954

theorem P_work_time (T : ℝ) (hT : T > 0) : 
  (1 / T + 1 / 6 = 1 / 2.4) → T = 4 :=
by
  intros h
  sorry

end P_work_time_l2299_229954


namespace no_solutions_exists_unique_l2299_229982

def is_solution (a b c x y z : ℤ) : Prop :=
  2 * x - b * y + z = 2 * b ∧
  a * x + 5 * y - c * z = a

def no_solutions_for (a b c : ℤ) : Prop :=
  ∀ x y z : ℤ, ¬ is_solution a b c x y z

theorem no_solutions_exists_unique (a b c : ℤ) :
  (a = -2 ∧ b = 5 ∧ c = 1) ∨
  (a = 2 ∧ b = -5 ∧ c = -1) ∨
  (a = 10 ∧ b = -1 ∧ c = -5) ↔
  no_solutions_for a b c := 
sorry

end no_solutions_exists_unique_l2299_229982


namespace solve_for_x_l2299_229987

theorem solve_for_x (x : ℝ) (h : 0.60 * 500 = 0.50 * x) : x = 600 :=
  sorry

end solve_for_x_l2299_229987


namespace triangle_side_length_condition_l2299_229957

theorem triangle_side_length_condition (a : ℝ) (h₁ : a > 0) (h₂ : a + 2 > a + 5) (h₃ : a + 5 > a + 2) (h₄ : a + 2 + a + 5 > a) : a > 3 :=
by
  sorry

end triangle_side_length_condition_l2299_229957


namespace milk_water_ratio_l2299_229919

theorem milk_water_ratio (x y : ℝ) (h1 : 5 * x + 2 * y = 4 * x + 7 * y) :
  x / y = 5 :=
by 
  sorry

end milk_water_ratio_l2299_229919


namespace uniq_increasing_seq_l2299_229941

noncomputable def a (n : ℕ) : ℕ := n -- The correct sequence a_n = n

theorem uniq_increasing_seq (a : ℕ → ℕ)
  (h1 : a 2 = 2)
  (h2 : ∀ n m : ℕ, a (n * m) = a n * a m)
  (h_inc : ∀ n m : ℕ, n < m → a n < a m) : ∀ n : ℕ, a n = n := by
  -- Here we would place the proof, skipping it for now with sorry
  sorry

end uniq_increasing_seq_l2299_229941


namespace compute_x_squared_y_plus_x_y_squared_l2299_229973

open Real

theorem compute_x_squared_y_plus_x_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + 2 * x + 2 * y = 7) : 
  x^2 * y + x * y^2 = 245 / 121 := 
by 
  sorry

end compute_x_squared_y_plus_x_y_squared_l2299_229973


namespace weight_of_new_person_l2299_229913

def total_weight_increase (num_people : ℕ) (weight_increase_per_person : ℝ) : ℝ :=
  num_people * weight_increase_per_person

def new_person_weight (old_person_weight : ℝ) (total_weight_increase : ℝ) : ℝ :=
  old_person_weight + total_weight_increase

theorem weight_of_new_person :
  let old_person_weight := 50
  let num_people := 8
  let weight_increase_per_person := 2.5
  new_person_weight old_person_weight (total_weight_increase num_people weight_increase_per_person) = 70 := 
by
  sorry

end weight_of_new_person_l2299_229913


namespace dark_squares_exceed_light_squares_by_one_l2299_229904

theorem dark_squares_exceed_light_squares_by_one :
  let dark_squares := 25
  let light_squares := 24
  dark_squares - light_squares = 1 :=
by
  sorry

end dark_squares_exceed_light_squares_by_one_l2299_229904


namespace find_extrema_l2299_229993

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

theorem find_extrema :
  (∀ x, f x ≤ 17) ∧ (∃ x, f x = 17) ∧ (∀ x, f x ≥ -47) ∧ (∃ x, f x = -47) :=
by
  sorry

end find_extrema_l2299_229993


namespace earrings_ratio_l2299_229984

theorem earrings_ratio :
  ∀ (total_pairs : ℕ) (given_pairs : ℕ) (total_earrings : ℕ) (given_earrings : ℕ),
    total_pairs = 12 →
    given_pairs = total_pairs / 2 →
    total_earrings = total_pairs * 2 →
    given_earrings = total_earrings / 2 →
    total_earrings = 36 →
    given_earrings = 12 →
    (total_earrings / given_earrings = 3) :=
by
  sorry

end earrings_ratio_l2299_229984


namespace total_volume_structure_l2299_229972

theorem total_volume_structure (d : ℝ) (h_cone : ℝ) (h_cylinder : ℝ) 
  (r := d / 2) 
  (V_cone := (1 / 3) * π * r^2 * h_cone) 
  (V_cylinder := π * r^2 * h_cylinder) 
  (V_total := V_cone + V_cylinder) :
  d = 8 → h_cone = 9 → h_cylinder = 4 → V_total = 112 * π :=
by
  intros
  sorry

end total_volume_structure_l2299_229972


namespace cute_2020_all_integers_cute_l2299_229975

-- Definition of "cute" integer
def is_cute (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), n = a^2 + b^3 + c^3 + d^5

-- Proof problem 1: Assert that 2020 is cute
theorem cute_2020 : is_cute 2020 :=
sorry

-- Proof problem 2: Assert that every integer is cute
theorem all_integers_cute (n : ℤ) : is_cute n :=
sorry

end cute_2020_all_integers_cute_l2299_229975


namespace range_of_m_l2299_229939

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (m+1)*x^2 - m*x + m - 1 ≥ 0) ↔ m ≥ (2*Real.sqrt 3)/3 := by
  sorry

end range_of_m_l2299_229939


namespace num_ballpoint_pens_l2299_229952

-- Define the total number of school supplies
def total_school_supplies : ℕ := 60

-- Define the number of pencils
def num_pencils : ℕ := 5

-- Define the number of notebooks
def num_notebooks : ℕ := 10

-- Define the number of erasers
def num_erasers : ℕ := 32

-- Define the number of ballpoint pens and prove it equals 13
theorem num_ballpoint_pens : total_school_supplies - (num_pencils + num_notebooks + num_erasers) = 13 :=
by
sorry

end num_ballpoint_pens_l2299_229952


namespace find_percentage_l2299_229951

variable (X P : ℝ)

theorem find_percentage (h₁ : 0.20 * X = 400) (h₂ : (P / 100) * X = 2400) : P = 120 :=
by
  -- The proof is intentionally left out
  sorry

end find_percentage_l2299_229951


namespace max_min_values_of_g_l2299_229921

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^8 + 8 * (Real.cos x)^8

theorem max_min_values_of_g :
  (∀ x : ℝ, g x ≤ 8) ∧ (∀ x : ℝ, g x ≥ 8 / 27) :=
by
  sorry

end max_min_values_of_g_l2299_229921


namespace total_students_is_88_l2299_229948

def orchestra_students : Nat := 20
def band_students : Nat := 2 * orchestra_students
def choir_boys : Nat := 12
def choir_girls : Nat := 16
def choir_students : Nat := choir_boys + choir_girls

def total_students : Nat := orchestra_students + band_students + choir_students

theorem total_students_is_88 : total_students = 88 := by
  sorry

end total_students_is_88_l2299_229948


namespace program_final_value_l2299_229927

-- Define the program execution in a Lean function
def program_result (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S
  else program_result (i - 1) (S * i)

-- Initial conditions
def initial_i := 11
def initial_S := 1

-- The theorem to prove
theorem program_final_value : program_result initial_i initial_S = 990 := by
  sorry

end program_final_value_l2299_229927


namespace dishonest_shopkeeper_gain_l2299_229978

-- Conditions: false weight used by shopkeeper
def false_weight : ℚ := 930
def true_weight : ℚ := 1000

-- Correct answer: gain percentage
def gain_percentage (false_weight true_weight : ℚ) : ℚ :=
  ((true_weight - false_weight) / false_weight) * 100

theorem dishonest_shopkeeper_gain :
  gain_percentage false_weight true_weight = 7.53 := by
  sorry

end dishonest_shopkeeper_gain_l2299_229978


namespace liz_spent_total_l2299_229920

-- Definitions:
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def number_of_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

-- Total cost calculation:
def total_cost : ℕ :=
  recipe_book_cost + baking_dish_cost + (number_of_ingredients * ingredient_cost) + apron_cost

-- Theorem Statement:
theorem liz_spent_total : total_cost = 40 := by
  sorry

end liz_spent_total_l2299_229920


namespace solution_set_l2299_229902

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution_set (x : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_def : ∀ x : ℝ, x >= 0 → f x = x^2 - 4 * x) :
    f (x + 2) < 5 ↔ -7 < x ∧ x < 3 :=
sorry

end solution_set_l2299_229902


namespace right_triangle_side_length_l2299_229940

theorem right_triangle_side_length (a c b : ℕ) (h1 : a = 3) (h2 : c = 5) (h3 : c^2 = a^2 + b^2) : b = 4 :=
sorry

end right_triangle_side_length_l2299_229940


namespace relationship_among_a_b_c_l2299_229991

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.1 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.3 * Real.log 0.2)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  have a_neg : a < 0 :=
    by sorry
  have b_pos : b > 1 :=
    by sorry
  have c_pos : c < 1 :=
    by sorry
  sorry

end relationship_among_a_b_c_l2299_229991


namespace find_f_1002_l2299_229924

noncomputable def f : ℕ → ℝ :=
  sorry

theorem find_f_1002 (f : ℕ → ℝ) 
  (h : ∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) :
  f 1002 = 21 :=
sorry

end find_f_1002_l2299_229924


namespace trigonometric_identity_l2299_229937

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + Real.pi / 3) = 1 / 3) :
  Real.sin (5 * Real.pi / 3 - x) - Real.cos (2 * x - Real.pi / 3) = 4 / 9 :=
by
  sorry

end trigonometric_identity_l2299_229937


namespace perfect_square_values_l2299_229958

theorem perfect_square_values :
  ∀ n : ℕ, 0 < n → (∃ k : ℕ, (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_values_l2299_229958


namespace ratio_of_arithmetic_sequences_l2299_229930

-- Definitions for the conditions
variables {a_n b_n : ℕ → ℝ}
variables {S_n T_n : ℕ → ℝ}
variables (d_a d_b : ℝ)

-- Arithmetic sequences conditions
def is_arithmetic_sequence (u_n : ℕ → ℝ) (t : ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), u_n n = t + n * d

-- Sum of first n terms conditions
def sum_of_first_n_terms (u_n : ℕ → ℝ) (Sn : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), Sn n = n * (u_n 1 + u_n (n-1)) / 2

-- Main theorem statement
theorem ratio_of_arithmetic_sequences (h1 : is_arithmetic_sequence a_n (a_n 0) d_a)
                                     (h2 : is_arithmetic_sequence b_n (b_n 0) d_b)
                                     (h3 : sum_of_first_n_terms a_n S_n)
                                     (h4 : sum_of_first_n_terms b_n T_n)
                                     (h5 : ∀ n, (S_n n) / (T_n n) = (2 * n) / (3 * n + 1)) :
                                     ∀ n, (a_n n) / (b_n n) = (2 * n - 1) / (3 * n - 1) := sorry

end ratio_of_arithmetic_sequences_l2299_229930


namespace ab_value_l2299_229944

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a ^ 2 + b ^ 2 = 35) : a * b = 13 :=
by
  sorry

end ab_value_l2299_229944


namespace difference_between_multiplication_and_subtraction_l2299_229953

theorem difference_between_multiplication_and_subtraction (x : ℤ) (h1 : x = 11) :
  (3 * x) - (26 - x) = 18 := by
  sorry

end difference_between_multiplication_and_subtraction_l2299_229953


namespace arc_length_of_polar_curve_l2299_229977

noncomputable def arc_length (f : ℝ → ℝ) (df : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt ((f x)^2 + (df x)^2)

theorem arc_length_of_polar_curve :
  arc_length (λ φ => 3 * (1 + Real.sin φ)) (λ φ => 3 * Real.cos φ) (-Real.pi / 6) 0 = 
  6 * (Real.sqrt 3 - Real.sqrt 2) :=
by
  sorry -- Proof goes here

end arc_length_of_polar_curve_l2299_229977


namespace calculation_not_minus_one_l2299_229999

theorem calculation_not_minus_one :
  (-1 : ℤ) * 1 ≠ 1 ∧
  (-1 : ℤ) / (-1) = 1 ∧
  (-2015 : ℤ) / 2015 ≠ 1 ∧
  (-1 : ℤ)^9 * (-1 : ℤ)^2 ≠ 1 := by 
  sorry

end calculation_not_minus_one_l2299_229999


namespace area_BEIH_l2299_229912

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def area_quad (B E I H : ℝ × ℝ) : ℝ :=
  (1/2) * ((B.1 * E.2 + E.1 * I.2 + I.1 * H.2 + H.1 * B.2) - (B.2 * E.1 + E.2 * I.1 + I.2 * H.1 + H.2 * B.1))

theorem area_BEIH :
  let A : ℝ × ℝ := point 0 3
  let B : ℝ × ℝ := point 0 0
  let C : ℝ × ℝ := point 3 0
  let D : ℝ × ℝ := point 3 3
  let E : ℝ × ℝ := point 0 2
  let F : ℝ × ℝ := point 1 0
  let I : ℝ × ℝ := point (3/10) 2.1
  let H : ℝ × ℝ := point (3/4) (3/4)
  area_quad B E I H = 1.0125 :=
by
  sorry

end area_BEIH_l2299_229912


namespace find_large_number_l2299_229946

theorem find_large_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_large_number_l2299_229946


namespace second_field_full_rows_l2299_229938

theorem second_field_full_rows 
    (rows_field1 : ℕ) (cobs_per_row : ℕ) (total_cobs : ℕ)
    (H1 : rows_field1 = 13)
    (H2 : cobs_per_row = 4)
    (H3 : total_cobs = 116) : 
    (total_cobs - rows_field1 * cobs_per_row) / cobs_per_row = 16 :=
by sorry

end second_field_full_rows_l2299_229938


namespace fraction_of_girls_is_one_third_l2299_229966

-- Define the number of children and number of boys
def total_children : Nat := 45
def boys : Nat := 30

-- Calculate the number of girls
def girls : Nat := total_children - boys

-- Calculate the fraction of girls
def fraction_of_girls : Rat := (girls : Rat) / (total_children : Rat)

theorem fraction_of_girls_is_one_third : fraction_of_girls = 1 / 3 :=
by
  sorry -- Proof is not required

end fraction_of_girls_is_one_third_l2299_229966


namespace sufficient_not_necessary_l2299_229936

theorem sufficient_not_necessary (a b : ℝ) :
  (a^2 + b^2 = 0 → ab = 0) ∧ (ab = 0 → ¬(a^2 + b^2 = 0)) := 
by
  have h1 : (a^2 + b^2 = 0 → ab = 0) := sorry
  have h2 : (ab = 0 → ¬(a^2 + b^2 = 0)) := sorry
  exact ⟨h1, h2⟩

end sufficient_not_necessary_l2299_229936


namespace even_binomial_coefficients_l2299_229956

theorem even_binomial_coefficients (n : ℕ) (h_pos: 0 < n) : 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → 2 ∣ Nat.choose n k) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end even_binomial_coefficients_l2299_229956


namespace experts_win_probability_l2299_229915

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l2299_229915


namespace commuting_days_l2299_229980

theorem commuting_days 
  (a b c d x : ℕ)
  (cond1 : b + c = 12)
  (cond2 : a + c = 20)
  (cond3 : a + b + 2 * d = 14)
  (cond4 : d = 2) :
  a + b + c + d = 23 := sorry

end commuting_days_l2299_229980


namespace polynomial_roots_l2299_229928

theorem polynomial_roots :
  (∀ x : ℤ, (x^3 - 4*x^2 - 11*x + 24 = 0) ↔ (x = 4 ∨ x = 3 ∨ x = -1)) :=
sorry

end polynomial_roots_l2299_229928


namespace train_distance_difference_l2299_229971

theorem train_distance_difference:
  ∀ (D1 D2 : ℕ) (t : ℕ), 
    (D1 = 20 * t) →            -- Slower train's distance
    (D2 = 25 * t) →           -- Faster train's distance
    (D1 + D2 = 450) →         -- Total distance between stations
    (D2 - D1 = 50) := 
by
  intros D1 D2 t h1 h2 h3
  sorry

end train_distance_difference_l2299_229971


namespace lines_parallel_if_perpendicular_to_same_plane_l2299_229909

variable {Plane Line : Type}
variable {α β γ : Plane}
variable {m n : Line}

-- Define perpendicularity and parallelism as axioms for simplicity
axiom perp (L : Line) (P : Plane) : Prop
axiom parallel (L1 L2 : Line) : Prop

-- Assume conditions for the theorem
variables (h1 : perp m α) (h2 : perp n α)

-- The theorem proving the required relationship
theorem lines_parallel_if_perpendicular_to_same_plane : parallel m n := 
by
  sorry

end lines_parallel_if_perpendicular_to_same_plane_l2299_229909


namespace M_is_real_l2299_229994

open Complex

-- Define the condition that characterizes the set M
def M (Z : ℂ) : Prop := (Z - 1)^2 = abs (Z - 1)^2

-- Prove that M is exactly the set of real numbers
theorem M_is_real : ∀ (Z : ℂ), M Z ↔ Z.im = 0 :=
by
  sorry

end M_is_real_l2299_229994


namespace trader_loss_percent_l2299_229950

theorem trader_loss_percent :
  let SP1 : ℝ := 404415
  let SP2 : ℝ := 404415
  let gain_percent : ℝ := 15 / 100
  let loss_percent : ℝ := 15 / 100
  let CP1 : ℝ := SP1 / (1 + gain_percent)
  let CP2 : ℝ := SP2 / (1 - loss_percent)
  let TCP : ℝ := CP1 + CP2
  let TSP : ℝ := SP1 + SP2
  let overall_loss : ℝ := TSP - TCP
  let overall_loss_percent : ℝ := (overall_loss / TCP) * 100
  overall_loss_percent = -2.25 := 
sorry

end trader_loss_percent_l2299_229950


namespace triangle_cosines_identity_l2299_229903

theorem triangle_cosines_identity 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) / a) + 
  (c^2 * Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) / b) + 
  (a^2 * Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / c) = 
  (a^4 + b^4 + c^4) / (2 * a * b * c) :=
by
  sorry

end triangle_cosines_identity_l2299_229903


namespace gazprom_rnd_costs_calc_l2299_229968

theorem gazprom_rnd_costs_calc (R_D_t ΔAPL_t1 : ℝ) (h1 : R_D_t = 3157.61) (h2 : ΔAPL_t1 = 0.69) :
  R_D_t / ΔAPL_t1 = 4576 :=
by
  sorry

end gazprom_rnd_costs_calc_l2299_229968


namespace factorize_expression_l2299_229916

theorem factorize_expression (x y : ℝ) : x^2 - 1 + 2 * x * y + y^2 = (x + y + 1) * (x + y - 1) :=
by sorry

end factorize_expression_l2299_229916
