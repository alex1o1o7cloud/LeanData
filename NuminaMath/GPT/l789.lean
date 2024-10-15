import Mathlib

namespace NUMINAMATH_GPT_evaluate_f_at_3_l789_78996

theorem evaluate_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x + 3) : f 3 = 7 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_f_at_3_l789_78996


namespace NUMINAMATH_GPT_min_value_expression_l789_78988

variable (a b : ℝ)

theorem min_value_expression :
  0 < a →
  1 < b →
  a + b = 2 →
  (∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = (2 / a) + (1 / (b - 1)) → y ≥ x)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l789_78988


namespace NUMINAMATH_GPT_expand_product_l789_78919

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14 * x + 45 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l789_78919


namespace NUMINAMATH_GPT_number_of_girls_l789_78940

variable (G B : ℕ)

theorem number_of_girls (h1 : G + B = 2000)
    (h2 : 0.28 * (B : ℝ) + 0.32 * (G : ℝ) = 596) : 
    G = 900 := 
sorry

end NUMINAMATH_GPT_number_of_girls_l789_78940


namespace NUMINAMATH_GPT_find_x_l789_78989

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - q.1, p.2 + q.2)

theorem find_x (x y : ℤ) :
  star (3, 3) (0, 0) = star (x, y) (3, 2) → x = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l789_78989


namespace NUMINAMATH_GPT_total_watermelons_l789_78977

/-- Proof statement: Jason grew 37 watermelons and Sandy grew 11 watermelons. 
    Prove that they grew a total of 48 watermelons. -/
theorem total_watermelons (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ) 
                         (h1 : jason_watermelons = 37) (h2 : sandy_watermelons = 11) :
  total_watermelons = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_watermelons_l789_78977


namespace NUMINAMATH_GPT_greatest_product_sum_2000_l789_78987

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end NUMINAMATH_GPT_greatest_product_sum_2000_l789_78987


namespace NUMINAMATH_GPT_Eliza_first_more_than_300_paperclips_on_Thursday_l789_78979

theorem Eliza_first_more_than_300_paperclips_on_Thursday :
  ∃ k : ℕ, 5 * 3^k > 300 ∧ k = 4 := 
by
  sorry

end NUMINAMATH_GPT_Eliza_first_more_than_300_paperclips_on_Thursday_l789_78979


namespace NUMINAMATH_GPT_billy_distance_l789_78939

-- Definitions
def distance_billy_spit (b : ℝ) : ℝ := b
def distance_madison_spit (m : ℝ) (b : ℝ) : Prop := m = 1.20 * b
def distance_ryan_spit (r : ℝ) (m : ℝ) : Prop := r = 0.50 * m

-- Conditions
variables (m : ℝ) (b : ℝ) (r : ℝ)
axiom madison_farther: distance_madison_spit m b
axiom ryan_shorter: distance_ryan_spit r m
axiom ryan_distance: r = 18

-- Proof problem
theorem billy_distance : b = 30 := by
  sorry

end NUMINAMATH_GPT_billy_distance_l789_78939


namespace NUMINAMATH_GPT_infinite_power_tower_solution_l789_78967

theorem infinite_power_tower_solution : 
  ∃ x : ℝ, (∀ y, y = x ^ y → y = 4) → x = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_power_tower_solution_l789_78967


namespace NUMINAMATH_GPT_christine_stickers_l789_78955

theorem christine_stickers (stickers_has stickers_needs : ℕ) (h_has : stickers_has = 11) (h_needs : stickers_needs = 19) : 
  stickers_has + stickers_needs = 30 :=
by 
  sorry

end NUMINAMATH_GPT_christine_stickers_l789_78955


namespace NUMINAMATH_GPT_find_duration_l789_78991

noncomputable def machine_times (x : ℝ) : Prop :=
  let tP := x + 5
  let tQ := x + 3
  let tR := 2 * (x * (x + 3) / 3)
  (1 / tP + 1 / tQ + 1 / tR = 1 / x) ∧ (tP > 0) ∧ (tQ > 0) ∧ (tR > 0)

theorem find_duration {x : ℝ} (h : machine_times x) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_duration_l789_78991


namespace NUMINAMATH_GPT_count_isosceles_triangles_perimeter_25_l789_78963

theorem count_isosceles_triangles_perimeter_25 : 
  ∃ n : ℕ, (
    n = 6 ∧ 
    (∀ x b : ℕ, 
      2 * x + b = 25 → 
      b < 2 * x → 
      b > 0 →
      ∃ m : ℕ, 
        m = (x - 7) / 5
    ) 
  ) := sorry

end NUMINAMATH_GPT_count_isosceles_triangles_perimeter_25_l789_78963


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l789_78958

theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ≤ 999 ∧ n ≥ 100 ∧ (∃ k : ℕ, n = 17 * k) ∧ 
  (∀ m : ℕ, m ≤ 999 → m ≥ 100 → (∃ k : ℕ, m = 17 * k) → m ≤ n) ∧ n = 986 := 
sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l789_78958


namespace NUMINAMATH_GPT_quadratic_two_roots_l789_78943

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, (x = x₁ ∨ x = x₂) ↔ (x^2 + b*x - 3 = 0)) :=
by
  -- Indicate that a proof is required here
  sorry

end NUMINAMATH_GPT_quadratic_two_roots_l789_78943


namespace NUMINAMATH_GPT_four_integers_sum_6_7_8_9_l789_78992

theorem four_integers_sum_6_7_8_9 (a b c d : ℕ)
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  (a = 1) ∧ (b = 2) ∧ (c = 3) ∧ (d = 4) := 
by 
  sorry

end NUMINAMATH_GPT_four_integers_sum_6_7_8_9_l789_78992


namespace NUMINAMATH_GPT_cricket_team_right_handed_count_l789_78993

theorem cricket_team_right_handed_count 
  (total throwers non_throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h_total : total = 70)
  (h_throwers : throwers = 37)
  (h_non_throwers : non_throwers = total - throwers)
  (h_left_handed_non_throwers : left_handed_non_throwers = non_throwers / 3)
  (h_right_handed_non_throwers : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h_all_throwers_right_handed : ∀ (t : ℕ), t = throwers → t = right_handed_non_throwers + (total - throwers) - (non_throwers / 3)) :
  right_handed_non_throwers + throwers = 59 := 
by 
  sorry

end NUMINAMATH_GPT_cricket_team_right_handed_count_l789_78993


namespace NUMINAMATH_GPT_joan_books_l789_78926

theorem joan_books : 
  (33 - 26 = 7) :=
by
  sorry

end NUMINAMATH_GPT_joan_books_l789_78926


namespace NUMINAMATH_GPT_posters_total_l789_78937

-- Definitions based on conditions
def Mario_posters : Nat := 18
def Samantha_posters : Nat := Mario_posters + 15

-- Statement to prove: They made 51 posters altogether
theorem posters_total : Mario_posters + Samantha_posters = 51 := 
by sorry

end NUMINAMATH_GPT_posters_total_l789_78937


namespace NUMINAMATH_GPT_customers_sampling_candy_l789_78918

theorem customers_sampling_candy (total_customers caught fined not_caught : ℝ) 
    (h1 : total_customers = 100) 
    (h2 : caught = 0.22 * total_customers) 
    (h3 : not_caught / (caught / 0.9) = 0.1) :
    (not_caught + caught) / total_customers = 0.2444 := 
by sorry

end NUMINAMATH_GPT_customers_sampling_candy_l789_78918


namespace NUMINAMATH_GPT_function_identity_l789_78917

theorem function_identity
    (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x ≤ x)
    (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
    ∀ x : ℝ, f x = x :=
by
    sorry

end NUMINAMATH_GPT_function_identity_l789_78917


namespace NUMINAMATH_GPT_monotonically_increasing_iff_l789_78984

noncomputable def f (x : ℝ) (a : ℝ) := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) := 
sorry

end NUMINAMATH_GPT_monotonically_increasing_iff_l789_78984


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l789_78921

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h2 : a 1 + a 2 + a 3 = 34)
  (h3 : a n + a (n-1) + a (n-2) = 146)
  (h4 : S n = 390)
  (h5 : ∀ i j, a i + a j = a (i+1) + a (j-1)) :
  n = 13 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l789_78921


namespace NUMINAMATH_GPT_odd_integers_equality_l789_78972

-- Definitions
def is_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

def divides (d n : ℤ) := ∃ k : ℤ, n = d * k

-- Main statement
theorem odd_integers_equality (a b : ℤ) (ha_pos : 0 < a) (hb_pos : 0 < b)
 (ha_odd : is_odd a) (hb_odd : is_odd b)
 (h_div : divides (2 * a * b + 1) (a^2 + b^2 + 1))
 : a = b :=
by 
  sorry

end NUMINAMATH_GPT_odd_integers_equality_l789_78972


namespace NUMINAMATH_GPT_work_completion_time_for_A_l789_78997

theorem work_completion_time_for_A 
  (B_work_rate : ℝ)
  (combined_work_rate : ℝ)
  (x : ℝ) 
  (B_work_rate_def : B_work_rate = 1 / 6)
  (combined_work_rate_def : combined_work_rate = 3 / 10) :
  (1 / x) + B_work_rate = combined_work_rate →
  x = 7.5 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_time_for_A_l789_78997


namespace NUMINAMATH_GPT_find_value_of_expression_l789_78910

theorem find_value_of_expression (m n : ℝ) (h : |m - n - 5| + (2 * m + n - 4)^2 = 0) : 3 * m + n = 7 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l789_78910


namespace NUMINAMATH_GPT_probability_at_least_one_girl_l789_78951

theorem probability_at_least_one_girl (boys girls : ℕ) (total : ℕ) (choose_two : ℕ) : 
  boys = 3 → girls = 2 → total = boys + girls → choose_two = 2 → 
  1 - (Nat.choose boys choose_two) / (Nat.choose total choose_two) = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_girl_l789_78951


namespace NUMINAMATH_GPT_regular_polygon_sides_l789_78969

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l789_78969


namespace NUMINAMATH_GPT_slope_of_line_AF_parabola_l789_78981

theorem slope_of_line_AF_parabola (A : ℝ × ℝ)
  (hA_on_parabola : A.snd ^ 2 = 4 * A.fst)
  (h_dist_focus : Real.sqrt ((A.fst - 1) ^ 2 + A.snd ^ 2) = 4) :
  (A.snd / (A.fst - 1) = Real.sqrt 3 ∨ A.snd / (A.fst - 1) = -Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_slope_of_line_AF_parabola_l789_78981


namespace NUMINAMATH_GPT_smallest_positive_period_l789_78994

-- Define a predicate for a function to have a period
def is_periodic {α : Type*} [AddGroup α] (f : α → ℝ) (T : α) : Prop :=
  ∀ x, f (x) = f (x - T)

-- The actual problem statement
theorem smallest_positive_period {f : ℝ → ℝ} 
  (h : ∀ x : ℝ, f (3 * x) = f (3 * x - 3 / 2)) : 
  is_periodic f (1 / 2) ∧ 
  ¬ (∃ T : ℝ, 0 < T ∧ T < 1 / 2 ∧ is_periodic f T) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_l789_78994


namespace NUMINAMATH_GPT_minimum_value_of_f_l789_78933

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem minimum_value_of_f : ∃ x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l789_78933


namespace NUMINAMATH_GPT_sum_of_squares_not_divisible_by_17_l789_78922

theorem sum_of_squares_not_divisible_by_17
  (x y z : ℤ)
  (h_sum_div : 17 ∣ (x + y + z))
  (h_prod_div : 17 ∣ (x * y * z))
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_coprime_zx : Int.gcd z x = 1) :
  ¬ (17 ∣ (x^2 + y^2 + z^2)) := 
sorry

end NUMINAMATH_GPT_sum_of_squares_not_divisible_by_17_l789_78922


namespace NUMINAMATH_GPT_eel_cost_l789_78949

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 :=
by
  sorry

end NUMINAMATH_GPT_eel_cost_l789_78949


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l789_78953

theorem relationship_between_m_and_n
  (a : ℝ) (b : ℝ) (ha : a > 2) (hb : b ≠ 0)
  (m : ℝ := a + 1 / (a - 2))
  (n : ℝ := 2^(2 - b^2)) :
  m > n :=
sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l789_78953


namespace NUMINAMATH_GPT_roots_situation_depends_on_k_l789_78961

theorem roots_situation_depends_on_k (k : ℝ) : 
  let a := 1
  let b := -3
  let c := 2 - k
  let Δ := b^2 - 4 * a * c
  (Δ > 0) ∨ (Δ = 0) ∨ (Δ < 0) :=
by
  intros
  sorry

end NUMINAMATH_GPT_roots_situation_depends_on_k_l789_78961


namespace NUMINAMATH_GPT_factorization_1_factorization_2_factorization_3_factorization_4_l789_78976

-- Problem 1
theorem factorization_1 (a b : ℝ) : 
  4 * a^2 + 12 * a * b + 9 * b^2 = (2 * a + 3 * b)^2 :=
by sorry

-- Problem 2
theorem factorization_2 (a b : ℝ) : 
  16 * a^2 * (a - b) + 4 * b^2 * (b - a) = 4 * (a - b) * (2 * a - b) * (2 * a + b) :=
by sorry

-- Problem 3
theorem factorization_3 (m n : ℝ) : 
  25 * (m + n)^2 - 9 * (m - n)^2 = 4 * (4 * m + n) * (m + 4 * n) :=
by sorry

-- Problem 4
theorem factorization_4 (a b : ℝ) : 
  4 * a^2 - b^2 - 4 * a + 1 = (2 * a - 1 + b) * (2 * a - 1 - b) :=
by sorry

end NUMINAMATH_GPT_factorization_1_factorization_2_factorization_3_factorization_4_l789_78976


namespace NUMINAMATH_GPT_people_in_room_l789_78945

theorem people_in_room (people chairs : ℕ) (h1 : 5 / 8 * people = 4 / 5 * chairs)
  (h2 : chairs = 5 + 4 / 5 * chairs) : people = 32 :=
by
  sorry

end NUMINAMATH_GPT_people_in_room_l789_78945


namespace NUMINAMATH_GPT_dog_food_consumption_per_meal_l789_78909

theorem dog_food_consumption_per_meal
  (dogs : ℕ) (meals_per_day : ℕ) (total_food_kg : ℕ) (days : ℕ)
  (h_dogs : dogs = 4) (h_meals_per_day : meals_per_day = 2)
  (h_total_food_kg : total_food_kg = 100) (h_days : days = 50) :
  (total_food_kg * 1000 / days / meals_per_day / dogs) = 250 :=
by
  sorry

end NUMINAMATH_GPT_dog_food_consumption_per_meal_l789_78909


namespace NUMINAMATH_GPT_new_parabola_through_point_l789_78950

def original_parabola (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

theorem new_parabola_through_point : 
  (∃ b : ℝ, ∀ x : ℝ, (x ^ 2 + 2 * x - 1 + b) = (x ^ 2 + 2 * x + 3)) :=
by
  sorry

end NUMINAMATH_GPT_new_parabola_through_point_l789_78950


namespace NUMINAMATH_GPT_exponent_equality_l789_78913

theorem exponent_equality (n : ℕ) : (4^8 = 4^n) → (n = 8) := by
  intro h
  sorry

end NUMINAMATH_GPT_exponent_equality_l789_78913


namespace NUMINAMATH_GPT_average_math_chemistry_l789_78952

variables (M P C : ℕ)

axiom h1 : M + P = 60
axiom h2 : C = P + 20

theorem average_math_chemistry : (M + C) / 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_math_chemistry_l789_78952


namespace NUMINAMATH_GPT_cistern_empty_time_l789_78903

theorem cistern_empty_time
  (fill_time_without_leak : ℝ := 4)
  (additional_time_due_to_leak : ℝ := 2) :
  (1 / (fill_time_without_leak + additional_time_due_to_leak - fill_time_without_leak / fill_time_without_leak)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_cistern_empty_time_l789_78903


namespace NUMINAMATH_GPT_find_natural_numbers_l789_78971

theorem find_natural_numbers (n : ℕ) (p q : ℕ) (hp : p.Prime) (hq : q.Prime)
  (h : q = p + 2) (h1 : (2^n + p).Prime) (h2 : (2^n + q).Prime) :
    n = 1 ∨ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_numbers_l789_78971


namespace NUMINAMATH_GPT_max_fraction_l789_78936

theorem max_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1 / 3) : (a, b) = (25, 76) :=
sorry

end NUMINAMATH_GPT_max_fraction_l789_78936


namespace NUMINAMATH_GPT_average_new_data_set_is_5_l789_78968

variable {x1 x2 x3 x4 : ℝ}
variable (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0)
variable (var_sqr : ℝ) (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16))

theorem average_new_data_set_is_5 (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16)) : 
  (x1 + 3 + x2 + 3 + x3 + 3 + x4 + 3) / 4 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_average_new_data_set_is_5_l789_78968


namespace NUMINAMATH_GPT_present_age_of_son_l789_78916

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 25) (h2 : F + 2 = 2 * (S + 2)) : S = 23 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l789_78916


namespace NUMINAMATH_GPT_expected_number_of_draws_l789_78964

-- Given conditions
def redBalls : ℕ := 2
def blackBalls : ℕ := 5
def totalBalls : ℕ := redBalls + blackBalls

-- Definition of expected number of draws
noncomputable def expected_draws : ℚ :=
  (2 * (1/21) + 3 * (2/21) + 4 * (3/21) + 5 * (4/21) + 
   6 * (5/21) + 7 * (6/21))

-- The theorem statement to prove
theorem expected_number_of_draws :
  expected_draws = 16 / 3 := by
  sorry

end NUMINAMATH_GPT_expected_number_of_draws_l789_78964


namespace NUMINAMATH_GPT_Total_toys_l789_78946

-- Definitions from the conditions
def Mandy_toys : ℕ := 20
def Anna_toys : ℕ := 3 * Mandy_toys
def Amanda_toys : ℕ := Anna_toys + 2

-- The statement to be proven
theorem Total_toys : Mandy_toys + Anna_toys + Amanda_toys = 142 :=
by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_Total_toys_l789_78946


namespace NUMINAMATH_GPT_factor_sum_l789_78941

theorem factor_sum : 
  (∃ d e, x^2 + 9 * x + 20 = (x + d) * (x + e)) ∧ 
  (∃ e f, x^2 - x - 56 = (x + e) * (x - f)) → 
  ∃ d e f, d + e + f = 19 :=
by
  sorry

end NUMINAMATH_GPT_factor_sum_l789_78941


namespace NUMINAMATH_GPT_sum_of_ages_l789_78901

variable {P M Mo : ℕ}

-- Conditions
axiom ratio1 : 3 * M = 5 * P
axiom ratio2 : 3 * Mo = 5 * M
axiom age_difference : Mo - P = 80

-- Statement that needs to be proved
theorem sum_of_ages : P + M + Mo = 245 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l789_78901


namespace NUMINAMATH_GPT_parabola_equation_l789_78927

theorem parabola_equation (h k a : ℝ) (same_shape : ∀ x, -2 * x^2 + 2 = a * x^2 + k) (vertex : h = 4 ∧ k = -2) :
  ∀ x, -2 * (x - 4)^2 - 2 = a * (x - h)^2 + k :=
by
  -- This is where the actual proof would go
  simp
  sorry

end NUMINAMATH_GPT_parabola_equation_l789_78927


namespace NUMINAMATH_GPT_Georgie_prank_l789_78985

theorem Georgie_prank (w : ℕ) (condition1 : w = 8) : 
  ∃ (ways : ℕ), ways = 336 := 
by
  sorry

end NUMINAMATH_GPT_Georgie_prank_l789_78985


namespace NUMINAMATH_GPT_find_yellow_shells_l789_78934

-- Define the conditions
def total_shells : ℕ := 65
def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

-- Define the result as the proof goal
theorem find_yellow_shells (total_shells purple_shells pink_shells blue_shells orange_shells : ℕ) : 
  total_shells = 65 →
  purple_shells = 13 →
  pink_shells = 8 →
  blue_shells = 12 →
  orange_shells = 14 →
  65 - (13 + 8 + 12 + 14) = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_yellow_shells_l789_78934


namespace NUMINAMATH_GPT_hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l789_78986

-- Definitions for the given problem
def one_hundred_million : ℕ := 100000000
def ten_million : ℕ := 10000000
def one_million : ℕ := 1000000
def ten_thousand : ℕ := 10000

-- Proving the statements
theorem hundred_million_is_ten_times_ten_million :
  one_hundred_million = 10 * ten_million :=
by
  sorry

theorem one_million_is_hundred_times_ten_thousand :
  one_million = 100 * ten_thousand :=
by
  sorry

end NUMINAMATH_GPT_hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l789_78986


namespace NUMINAMATH_GPT_water_drain_rate_l789_78998

theorem water_drain_rate
  (total_volume : ℕ)
  (total_time : ℕ)
  (H1 : total_volume = 300)
  (H2 : total_time = 25) :
  total_volume / total_time = 12 := 
by
  sorry

end NUMINAMATH_GPT_water_drain_rate_l789_78998


namespace NUMINAMATH_GPT_sequence_remainder_zero_l789_78900

theorem sequence_remainder_zero :
  let a := 3
  let d := 8
  let n := 32
  let aₙ := a + (n - 1) * d
  let Sₙ := n * (a + aₙ) / 2
  aₙ = 251 → Sₙ % 8 = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sequence_remainder_zero_l789_78900


namespace NUMINAMATH_GPT_total_area_of_farm_l789_78956

-- Define the number of sections and area of each section
def number_of_sections : ℕ := 5
def area_of_each_section : ℕ := 60

-- State the problem as proving the total area of the farm
theorem total_area_of_farm : number_of_sections * area_of_each_section = 300 :=
by sorry

end NUMINAMATH_GPT_total_area_of_farm_l789_78956


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l789_78905

-- Define the quadratic equation and its coefficients
def a := 1
def b := -4
def c := -3

-- Define the discriminant function for a quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- State the problem in Lean: Prove that the quadratic equation x^2 - 4x - 3 = 0 has a positive discriminant.
theorem quadratic_has_two_distinct_real_roots : discriminant a b c > 0 :=
by
  sorry -- This is where the proof would go

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l789_78905


namespace NUMINAMATH_GPT_picture_area_l789_78930

theorem picture_area (x y : ℤ) (hx : 1 < x) (hy : 1 < y) (h : (x + 2) * (y + 4) = 45) : x * y = 15 := by
  sorry

end NUMINAMATH_GPT_picture_area_l789_78930


namespace NUMINAMATH_GPT_blueberry_picking_l789_78948

-- Define the amounts y1 and y2 as a function of x
variable (x : ℝ)
def y1 : ℝ := 60 + 18 * x
def y2 : ℝ := 150 + 15 * x

-- State the theorem about the relationships given the condition 
theorem blueberry_picking (hx : x > 10) : 
  y1 x = 60 + 18 * x ∧ y2 x = 150 + 15 * x :=
by
  sorry

end NUMINAMATH_GPT_blueberry_picking_l789_78948


namespace NUMINAMATH_GPT_wendy_distance_difference_l789_78907

-- Defining the distances ran and walked by Wendy
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- The theorem to prove the difference in distance
theorem wendy_distance_difference : distance_ran - distance_walked = 10.66 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_wendy_distance_difference_l789_78907


namespace NUMINAMATH_GPT_fish_in_pond_l789_78974

noncomputable def number_of_fish (marked_first: ℕ) (marked_second: ℕ) (catch_first: ℕ) (catch_second: ℕ) : ℕ :=
  (marked_first * catch_second) / marked_second

theorem fish_in_pond (h1 : marked_first = 30) (h2 : marked_second = 2) (h3 : catch_first = 30) (h4 : catch_second = 40) :
  number_of_fish marked_first marked_second catch_first catch_second = 600 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_fish_in_pond_l789_78974


namespace NUMINAMATH_GPT_quarterly_to_annual_rate_l789_78957

theorem quarterly_to_annual_rate (annual_rate : ℝ) (quarterly_rate : ℝ) (n : ℕ) (effective_annual_rate : ℝ) : 
  annual_rate = 4.5 →
  quarterly_rate = annual_rate / 4 →
  n = 4 →
  effective_annual_rate = (1 + quarterly_rate / 100)^n →
  effective_annual_rate * 100 = 4.56 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_quarterly_to_annual_rate_l789_78957


namespace NUMINAMATH_GPT_not_periodic_fraction_l789_78982

theorem not_periodic_fraction :
  ¬ ∃ (n k : ℕ), ∀ m ≥ n + k, ∃ l, 10^m + l = 10^(m+n) + l ∧ ((0.1234567891011121314 : ℝ) = (0.1234567891011121314 + l / (10^(m+n)))) :=
sorry

end NUMINAMATH_GPT_not_periodic_fraction_l789_78982


namespace NUMINAMATH_GPT_statement_true_when_b_le_a_div_5_l789_78959

theorem statement_true_when_b_le_a_div_5
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₀ : ∀ x : ℝ, f x = 5 * x + 3)
  (h₁ : ∀ x : ℝ, |f x + 7| < a ↔ |x + 2| < b)
  (h₂ : 0 < a)
  (h₃ : 0 < b) :
  b ≤ a / 5 :=
by
  sorry

end NUMINAMATH_GPT_statement_true_when_b_le_a_div_5_l789_78959


namespace NUMINAMATH_GPT_frog_jump_paths_l789_78914

noncomputable def φ : ℕ × ℕ → ℕ
| (0, 0) => 1
| (x, y) =>
  let φ_x1 := if x > 1 then φ (x - 1, y) else 0
  let φ_x2 := if x > 1 then φ (x - 2, y) else 0
  let φ_y1 := if y > 1 then φ (x, y - 1) else 0
  let φ_y2 := if y > 1 then φ (x, y - 2) else 0
  φ_x1 + φ_x2 + φ_y1 + φ_y2

theorem frog_jump_paths : φ (4, 4) = 556 := sorry

end NUMINAMATH_GPT_frog_jump_paths_l789_78914


namespace NUMINAMATH_GPT_factorize_3m2_minus_12_l789_78995

theorem factorize_3m2_minus_12 (m : ℤ) : 
  3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := 
sorry

end NUMINAMATH_GPT_factorize_3m2_minus_12_l789_78995


namespace NUMINAMATH_GPT_correct_statement_l789_78942

-- Definition of quadrants
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def is_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_third_quadrant (θ : ℝ) : Prop := -180 < θ ∧ θ < -90
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement of the problem
theorem correct_statement : is_obtuse_angle θ → is_second_quadrant θ :=
by sorry

end NUMINAMATH_GPT_correct_statement_l789_78942


namespace NUMINAMATH_GPT_abs_val_of_5_minus_e_l789_78912

theorem abs_val_of_5_minus_e : ∀ (e : ℝ), e = 2.718 → |5 - e| = 2.282 :=
by
  intros e he
  sorry

end NUMINAMATH_GPT_abs_val_of_5_minus_e_l789_78912


namespace NUMINAMATH_GPT_x_plus_y_equals_six_l789_78924

theorem x_plus_y_equals_six (x y : ℝ) (h₁ : y - x = 1) (h₂ : y^2 = x^2 + 6) : x + y = 6 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_y_equals_six_l789_78924


namespace NUMINAMATH_GPT_white_roses_per_table_decoration_l789_78915

theorem white_roses_per_table_decoration (x : ℕ) :
  let bouquets := 5
  let table_decorations := 7
  let roses_per_bouquet := 5
  let total_roses := 109
  5 * roses_per_bouquet + 7 * x = total_roses → x = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_white_roses_per_table_decoration_l789_78915


namespace NUMINAMATH_GPT_parallelogram_sides_are_parallel_l789_78925

theorem parallelogram_sides_are_parallel 
  {a b c : ℤ} (h_area : c * (a^2 + b^2) = 2011 * b) : 
  (∃ k : ℤ, a = 2011 * k ∧ (b = 2011 ∨ b = -2011)) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_sides_are_parallel_l789_78925


namespace NUMINAMATH_GPT_rectangle_width_l789_78931

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 750) 
  (h2 : 2 * L + 2 * W = 110) : 
  W = 25 :=
sorry

end NUMINAMATH_GPT_rectangle_width_l789_78931


namespace NUMINAMATH_GPT_valid_pairs_l789_78975

-- Define the target function and condition
def satisfies_condition (k l : ℤ) : Prop :=
  (7 * k - 5) * (4 * l - 3) = (5 * k - 3) * (6 * l - 1)

-- The theorem stating the exact pairs that satisfy the condition
theorem valid_pairs :
  ∀ (k l : ℤ), satisfies_condition k l ↔
    (k = 0 ∧ l = 6) ∨
    (k = 1 ∧ l = -1) ∨
    (k = 6 ∧ l = -6) ∨
    (k = 13 ∧ l = -7) ∨
    (k = -2 ∧ l = -22) ∨
    (k = -3 ∧ l = -15) ∨
    (k = -8 ∧ l = -10) ∨
    (k = -15 ∧ l = -9) :=
by
  sorry

end NUMINAMATH_GPT_valid_pairs_l789_78975


namespace NUMINAMATH_GPT_original_mixture_volume_l789_78928

theorem original_mixture_volume (x : ℝ) (h1 : 0.20 * x / (x + 3) = 1 / 6) : x = 15 :=
  sorry

end NUMINAMATH_GPT_original_mixture_volume_l789_78928


namespace NUMINAMATH_GPT_seventh_observation_l789_78935

theorem seventh_observation (avg6 : ℕ) (new_avg7 : ℕ) (old_avg : ℕ) (new_avg_diff : ℕ) (n : ℕ) (m : ℕ) (h1 : avg6 = 12) (h2 : new_avg_diff = 1) (h3 : n = 6) (h4 : m = 7) :
  ((n * old_avg = avg6 * old_avg) ∧ (m * new_avg7 = avg6 * old_avg + m - n)) →
  m * new_avg7 = 77 →
  avg6 * old_avg = 72 →
  77 - 72 = 5 :=
by
  sorry

end NUMINAMATH_GPT_seventh_observation_l789_78935


namespace NUMINAMATH_GPT_exists_a_satisfying_inequality_l789_78923

theorem exists_a_satisfying_inequality (x : ℝ) : 
  x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x → 
  ∃ a ∈ Set.Icc (-1 : ℝ) 2, (2 - a) * x^3 + (1 - 2 * a) * x^2 - 6 * x + 5 + 4 * a - a^2 < 0 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_exists_a_satisfying_inequality_l789_78923


namespace NUMINAMATH_GPT_paving_path_DE_time_l789_78999

-- Define the conditions
variable (v : ℝ) -- Speed of Worker 1
variable (x : ℝ) -- Total distance for Worker 1
variable (d2 : ℝ) -- Total distance for Worker 2
variable (AD DE EF FC : ℝ) -- Distances in the path of Worker 2

-- Define the statement
theorem paving_path_DE_time :
  (AD + DE + EF + FC) = d2 ∧
  x = 9 * v ∧
  d2 = 10.8 * v ∧
  d2 = AD + DE + EF + FC ∧
  (∀ t, t = (DE / (1.2 * v)) * 60) ∧
  t = 45 :=
by
  sorry

end NUMINAMATH_GPT_paving_path_DE_time_l789_78999


namespace NUMINAMATH_GPT_triangle_third_side_range_l789_78960

theorem triangle_third_side_range {x : ℤ} : 
  (7 < x ∧ x < 17) → (4 ≤ x ∧ x ≤ 16) :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_range_l789_78960


namespace NUMINAMATH_GPT_dice_sum_surface_l789_78906

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end NUMINAMATH_GPT_dice_sum_surface_l789_78906


namespace NUMINAMATH_GPT_tom_total_payment_l789_78929

variable (apples_kg : ℕ := 8)
variable (apples_rate : ℕ := 70)
variable (mangoes_kg : ℕ := 9)
variable (mangoes_rate : ℕ := 65)
variable (oranges_kg : ℕ := 5)
variable (oranges_rate : ℕ := 50)
variable (bananas_kg : ℕ := 3)
variable (bananas_rate : ℕ := 30)
variable (discount_apples : ℝ := 0.10)
variable (discount_oranges : ℝ := 0.15)

def total_cost_apple : ℝ := apples_kg * apples_rate
def total_cost_mango : ℝ := mangoes_kg * mangoes_rate
def total_cost_orange : ℝ := oranges_kg * oranges_rate
def total_cost_banana : ℝ := bananas_kg * bananas_rate
def discount_apples_amount : ℝ := discount_apples * total_cost_apple
def discount_oranges_amount : ℝ := discount_oranges * total_cost_orange
def apples_after_discount : ℝ := total_cost_apple - discount_apples_amount
def oranges_after_discount : ℝ := total_cost_orange - discount_oranges_amount

theorem tom_total_payment :
  apples_after_discount + total_cost_mango + oranges_after_discount + total_cost_banana = 1391.5 := by
  sorry

end NUMINAMATH_GPT_tom_total_payment_l789_78929


namespace NUMINAMATH_GPT_largest_of_consecutive_even_integers_l789_78980

theorem largest_of_consecutive_even_integers (x : ℤ) (h : 25 * (x + 24) = 10000) : x + 48 = 424 :=
sorry

end NUMINAMATH_GPT_largest_of_consecutive_even_integers_l789_78980


namespace NUMINAMATH_GPT_wall_width_l789_78962

theorem wall_width (area height : ℕ) (h1 : area = 16) (h2 : height = 4) : area / height = 4 :=
by
  sorry

end NUMINAMATH_GPT_wall_width_l789_78962


namespace NUMINAMATH_GPT_ascending_order_l789_78944

theorem ascending_order (a b c d : ℝ) (h1 : a = -6) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = Real.pi) :
  a < b ∧ b < c ∧ c < d :=
by
  sorry

end NUMINAMATH_GPT_ascending_order_l789_78944


namespace NUMINAMATH_GPT_add_ab_equals_four_l789_78966

theorem add_ab_equals_four (a b : ℝ) (h₁ : a * (a - 4) = 5) (h₂ : b * (b - 4) = 5) (h₃ : a ≠ b) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_add_ab_equals_four_l789_78966


namespace NUMINAMATH_GPT_difference_in_money_in_cents_l789_78911

theorem difference_in_money_in_cents (p : ℤ) (h₁ : ℤ) (h₂ : ℤ) 
  (h₁ : Linda_nickels = 7 * p - 2) (h₂ : Carol_nickels = 3 * p + 4) :
  5 * (Linda_nickels - Carol_nickels) = 20 * p - 30 := 
by sorry

end NUMINAMATH_GPT_difference_in_money_in_cents_l789_78911


namespace NUMINAMATH_GPT_find_p_l789_78983

variable (m n p : ℝ)

theorem find_p (h1 : m = n / 7 - 2 / 5)
               (h2 : m + p = (n + 21) / 7 - 2 / 5) : p = 3 := by
  sorry

end NUMINAMATH_GPT_find_p_l789_78983


namespace NUMINAMATH_GPT_mopping_time_is_30_l789_78954

def vacuuming_time := 45
def dusting_time := 60
def brushing_time_per_cat := 5
def number_of_cats := 3
def total_free_time := 180
def free_time_left := 30

def total_cleaning_time := total_free_time - free_time_left
def brushing_time := brushing_time_per_cat * number_of_cats
def time_other_tasks := vacuuming_time + dusting_time + brushing_time

theorem mopping_time_is_30 : total_cleaning_time - time_other_tasks = 30 := by
  -- Calculation proof would go here
  sorry

end NUMINAMATH_GPT_mopping_time_is_30_l789_78954


namespace NUMINAMATH_GPT_map_distance_l789_78970

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end NUMINAMATH_GPT_map_distance_l789_78970


namespace NUMINAMATH_GPT_square_side_length_l789_78902

theorem square_side_length 
  (A B C D E : Type) 
  (AB AC hypotenuse square_side_length : ℝ) 
  (h1: AB = 9) 
  (h2: AC = 12) 
  (h3: hypotenuse = Real.sqrt (9^2 + 12^2)) 
  (h4: square_side_length = 300 / 41) 
  : square_side_length = 300 / 41 := 
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l789_78902


namespace NUMINAMATH_GPT_find_a_b_l789_78978

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → a = 3 ∧ b = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_b_l789_78978


namespace NUMINAMATH_GPT_minimum_value_problem_l789_78920

open Real

theorem minimum_value_problem (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 6) :
  9 / x + 16 / y + 25 / z ≥ 24 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_problem_l789_78920


namespace NUMINAMATH_GPT_fraction_lt_sqrt2_bound_l789_78938

theorem fraction_lt_sqrt2_bound (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * (n * n))) :=
sorry

end NUMINAMATH_GPT_fraction_lt_sqrt2_bound_l789_78938


namespace NUMINAMATH_GPT_circle_center_sum_l789_78908

theorem circle_center_sum (h k : ℝ) :
  (∃ h k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = 6 * x + 8 * y - 15) → (h, k) = (3, 4)) →
  h + k = 7 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_sum_l789_78908


namespace NUMINAMATH_GPT_david_account_amount_l789_78904

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem david_account_amount : compound_interest 5000 0.06 2 1 = 5304.50 := by
  sorry

end NUMINAMATH_GPT_david_account_amount_l789_78904


namespace NUMINAMATH_GPT_julie_net_monthly_income_is_l789_78973

section JulieIncome

def starting_pay : ℝ := 5.00
def additional_experience_pay_per_year : ℝ := 0.50
def years_of_experience : ℕ := 3
def work_hours_per_day : ℕ := 8
def work_days_per_week : ℕ := 6
def bi_weekly_bonus : ℝ := 50.00
def tax_rate : ℝ := 0.12
def insurance_premium_per_month : ℝ := 40.00
def missed_days : ℕ := 1

-- Calculate Julie's net monthly income
def net_monthly_income : ℝ :=
    let hourly_wage := starting_pay + additional_experience_pay_per_year * years_of_experience
    let daily_earnings := hourly_wage * work_hours_per_day
    let weekly_earnings := daily_earnings * (work_days_per_week - missed_days)
    let bi_weekly_earnings := weekly_earnings * 2
    let gross_monthly_income := bi_weekly_earnings * 2 + bi_weekly_bonus * 2
    let tax_deduction := gross_monthly_income * tax_rate
    let total_deductions := tax_deduction + insurance_premium_per_month
    gross_monthly_income - total_deductions

theorem julie_net_monthly_income_is : net_monthly_income = 963.20 :=
    sorry

end JulieIncome

end NUMINAMATH_GPT_julie_net_monthly_income_is_l789_78973


namespace NUMINAMATH_GPT_clara_cookies_l789_78990

theorem clara_cookies (x : ℕ) :
  50 * 12 + x * 20 + 70 * 16 = 3320 → x = 80 :=
by
  sorry

end NUMINAMATH_GPT_clara_cookies_l789_78990


namespace NUMINAMATH_GPT_hyperbola_condition_l789_78947

theorem hyperbola_condition (k : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (4 + k) + y^2 / (1 - k) = 1)) ↔ (k < -4 ∨ k > 1) :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l789_78947


namespace NUMINAMATH_GPT_determine_a_range_l789_78965

noncomputable def single_element_intersection (a : ℝ) : Prop :=
  let A := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a * x + 1)}
  let B := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, |x|)}
  (∃ p : ℝ × ℝ, p ∈ A ∧ p ∈ B) ∧ 
  ∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ A ∧ p₁ ∈ B → p₂ ∈ A ∧ p₂ ∈ B → p₁ = p₂

theorem determine_a_range : 
  ∀ a : ℝ, single_element_intersection a ↔ a ∈ Set.Iic (-1) ∨ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_GPT_determine_a_range_l789_78965


namespace NUMINAMATH_GPT_math_proof_l789_78932

open Real

noncomputable def function (a b x : ℝ): ℝ := a * x^3 + b * x^2

theorem math_proof (a b : ℝ) :
  (function a b 1 = 3) ∧
  (deriv (function a b) 1 = 0) ∧
  (∃ (a b : ℝ), a = -6 ∧ b = 9 ∧ 
    function a b = -6 * (x^3) + 9 * (x^2)) ∧
  (∀ x, (0 < x ∧ x < 1) → deriv (function a b) x > 0) ∧
  (∀ x, (x < 0 ∨ x > 1) → deriv (function a b) x < 0) ∧
  (min (function a b (-2)) (function a b 2) = (-12)) ∧
  (max (function a b (-2)) (function a b 2) = 84) :=
by
  sorry

end NUMINAMATH_GPT_math_proof_l789_78932
