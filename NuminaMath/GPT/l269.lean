import Mathlib

namespace NUMINAMATH_GPT_vehicle_value_last_year_l269_26989

variable (v_this_year v_last_year : ℝ)

theorem vehicle_value_last_year:
  v_this_year = 16000 ∧ v_this_year = 0.8 * v_last_year → v_last_year = 20000 :=
by
  -- Proof steps can be added here, but replaced with sorry as per instructions.
  sorry

end NUMINAMATH_GPT_vehicle_value_last_year_l269_26989


namespace NUMINAMATH_GPT_minimum_value_k_eq_2_l269_26969

noncomputable def quadratic_function_min (a m k : ℝ) (h : 0 < a) : ℝ :=
  a * (-(k / 2)) * (-(k / 2) - k)

theorem minimum_value_k_eq_2 (a m : ℝ) (h : 0 < a) :
  quadratic_function_min a m 2 h = -a := 
by
  unfold quadratic_function_min
  sorry

end NUMINAMATH_GPT_minimum_value_k_eq_2_l269_26969


namespace NUMINAMATH_GPT_min_max_solution_A_l269_26923

theorem min_max_solution_A (x y z : ℕ) (h₁ : x + y + z = 100) (h₂ : 5 * x + 8 * y + 9 * z = 700) 
                           (h₃ : 0 ≤ x ∧ x ≤ 60) (h₄ : 0 ≤ y ∧ y ≤ 60) (h₅ : 0 ≤ z ∧ z ≤ 47) :
    35 ≤ x ∧ x ≤ 49 :=
by
  sorry

end NUMINAMATH_GPT_min_max_solution_A_l269_26923


namespace NUMINAMATH_GPT_total_pencils_l269_26986

   variables (n p t : ℕ)

   -- Condition 1: number of students
   def students := 12

   -- Condition 2: pencils per student
   def pencils_per_student := 3

   -- Theorem statement: Given the conditions, the total number of pencils given by the teacher is 36
   theorem total_pencils : t = students * pencils_per_student :=
   by
   sorry
   
end NUMINAMATH_GPT_total_pencils_l269_26986


namespace NUMINAMATH_GPT_least_integer_remainder_condition_l269_26995

def is_least_integer_with_remainder_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k ∈ [3, 4, 5, 6, 7, 10, 11], n % k = 1)

theorem least_integer_remainder_condition : ∃ (n : ℕ), is_least_integer_with_remainder_condition n ∧ n = 4621 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_least_integer_remainder_condition_l269_26995


namespace NUMINAMATH_GPT_polynomial_value_at_3_l269_26911

theorem polynomial_value_at_3 :
  ∃ (P : ℕ → ℚ), 
    (∀ (x : ℕ), P x = b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5 + b_6 * x^6) ∧ 
    (∀ (i : ℕ), i ≤ 6 → 0 ≤ b_i ∧ b_i < 5) ∧ 
    P (Nat.sqrt 5) = 35 + 26 * Nat.sqrt 5 -> 
    P 3 = 437 := 
by
  simp
  sorry

end NUMINAMATH_GPT_polynomial_value_at_3_l269_26911


namespace NUMINAMATH_GPT_value_of_y_l269_26982

variable {x y : ℝ}

theorem value_of_y (h1 : x > 2) (h2 : y > 2) (h3 : 1/x + 1/y = 3/4) (h4 : x * y = 8) : y = 4 :=
sorry

end NUMINAMATH_GPT_value_of_y_l269_26982


namespace NUMINAMATH_GPT_initial_population_l269_26922

/--
Suppose 5% of people in a village died by bombardment,
15% of the remaining population left the village due to fear,
and the population is now reduced to 3294.
Prove that the initial population was 4080.
-/
theorem initial_population (P : ℝ) 
  (H1 : 0.05 * P + 0.15 * (1 - 0.05) * P + 3294 = P) : P = 4080 :=
sorry

end NUMINAMATH_GPT_initial_population_l269_26922


namespace NUMINAMATH_GPT_distance_metric_l269_26958

noncomputable def d (x y : ℝ) : ℝ :=
  (|x - y|) / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem distance_metric (x y z : ℝ) :
  (d x x = 0) ∧
  (d x y = d y x) ∧
  (d x y + d y z ≥ d x z) := by
  sorry

end NUMINAMATH_GPT_distance_metric_l269_26958


namespace NUMINAMATH_GPT_least_sum_of_exponents_l269_26912

theorem least_sum_of_exponents (a b c d e : ℕ) (h : ℕ) (h_divisors : 225 ∣ h ∧ 216 ∣ h ∧ 847 ∣ h)
  (h_form : h = (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) * (11 ^ e)) : 
  a + b + c + d + e = 10 :=
sorry

end NUMINAMATH_GPT_least_sum_of_exponents_l269_26912


namespace NUMINAMATH_GPT_ratio_of_shirt_to_pants_l269_26951

theorem ratio_of_shirt_to_pants
    (total_cost : ℕ)
    (price_pants : ℕ)
    (price_shoes : ℕ)
    (price_shirt : ℕ)
    (h1 : total_cost = 340)
    (h2 : price_pants = 120)
    (h3 : price_shoes = price_pants + 10)
    (h4 : price_shirt = total_cost - (price_pants + price_shoes)) :
    price_shirt * 4 = price_pants * 3 := sorry

end NUMINAMATH_GPT_ratio_of_shirt_to_pants_l269_26951


namespace NUMINAMATH_GPT_hours_per_day_for_first_group_l269_26901

theorem hours_per_day_for_first_group (h : ℕ) :
  (39 * h * 12 = 30 * 6 * 26) → h = 10 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_for_first_group_l269_26901


namespace NUMINAMATH_GPT_inequality_holds_l269_26933

variable {x y : ℝ}

theorem inequality_holds (x : ℝ) (y : ℝ) (hy : y ≥ 5) : 
  x^2 - 2 * x * Real.sqrt (y - 5) + y^2 + y - 30 ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_holds_l269_26933


namespace NUMINAMATH_GPT_hexagon_angles_sum_l269_26909

theorem hexagon_angles_sum (mA mB mC : ℤ) (x y : ℤ)
  (hA : mA = 35) (hB : mB = 80) (hC : mC = 30)
  (hSum : (6 - 2) * 180 = 720)
  (hAdjacentA : 90 + 90 = 180)
  (hAdjacentC : 90 - mC = 60) :
  x + y = 95 := by
  sorry

end NUMINAMATH_GPT_hexagon_angles_sum_l269_26909


namespace NUMINAMATH_GPT_oblique_asymptote_l269_26953

theorem oblique_asymptote :
  ∀ x : ℝ, (∃ δ > 0, ∀ y > x, (abs (3 * y^2 + 8 * y + 12) / (3 * y + 4) - (y + 4 / 3)) < δ) :=
sorry

end NUMINAMATH_GPT_oblique_asymptote_l269_26953


namespace NUMINAMATH_GPT_triangle_angle_area_l269_26980

theorem triangle_angle_area
  (A B C : ℝ) (a b c : ℝ)
  (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
  (h2 : C = Real.pi / 3)
  (h3 : c = 2)
  (h4 : a + b + c = 2 * Real.sqrt 3 + 2) :
  ∃ (area : ℝ), area = (2 * Real.sqrt 3) / 3 :=
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_triangle_angle_area_l269_26980


namespace NUMINAMATH_GPT_sum_of_numerical_coefficients_binomial_l269_26993

theorem sum_of_numerical_coefficients_binomial (a b : ℕ) (n : ℕ) (h : n = 8) :
  let sum_num_coeff := (a + b)^n
  sum_num_coeff = 256 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_numerical_coefficients_binomial_l269_26993


namespace NUMINAMATH_GPT_geometric_sequence_condition_neither_necessary_nor_sufficient_l269_26964

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

noncomputable def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_condition_neither_necessary_nor_sufficient (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q → ¬( (is_monotonically_increasing a ↔ q > 1) ) :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_condition_neither_necessary_nor_sufficient_l269_26964


namespace NUMINAMATH_GPT_inequality_a_b_c_l269_26939

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_a_b_c_l269_26939


namespace NUMINAMATH_GPT_parabola_shift_units_l269_26946

theorem parabola_shift_units (h : ℝ) :
  (∃ h, (0 + 3 - h)^2 - 1 = 0) ↔ (h = 2 ∨ h = 4) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_shift_units_l269_26946


namespace NUMINAMATH_GPT_least_multiple_of_15_greater_than_500_l269_26944

theorem least_multiple_of_15_greater_than_500 : 
  ∃ (n : ℕ), n > 500 ∧ (∃ (k : ℕ), n = 15 * k) ∧ (n = 510) :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_15_greater_than_500_l269_26944


namespace NUMINAMATH_GPT_both_shots_hit_target_exactly_one_shot_hits_target_l269_26910

variable (p q : Prop)

theorem both_shots_hit_target : (p ∧ q) := sorry

theorem exactly_one_shot_hits_target : ((p ∧ ¬ q) ∨ (¬ p ∧ q)) := sorry

end NUMINAMATH_GPT_both_shots_hit_target_exactly_one_shot_hits_target_l269_26910


namespace NUMINAMATH_GPT_discount_percentage_l269_26907

theorem discount_percentage (marked_price sale_price cost_price : ℝ) (gain1 gain2 : ℝ)
  (h1 : gain1 = 0.35)
  (h2 : gain2 = 0.215)
  (h3 : sale_price = 30)
  (h4 : cost_price = marked_price / (1 + gain1))
  (h5 : marked_price = cost_price * (1 + gain2)) :
  ((sale_price - marked_price) / sale_price) * 100 = 10.009 :=
sorry

end NUMINAMATH_GPT_discount_percentage_l269_26907


namespace NUMINAMATH_GPT_number_of_oddly_powerful_integers_lt_500_l269_26962

noncomputable def count_oddly_powerful_integers_lt_500 : ℕ :=
  let count_cubes := 7 -- we counted cubes: 1^3, 2^3, 3^3, 4^3, 5^3, 6^3, 7^3
  let count_fifth_powers := 1 -- the additional fifth power not a cube: 3^5
  count_cubes + count_fifth_powers

theorem number_of_oddly_powerful_integers_lt_500 : count_oddly_powerful_integers_lt_500 = 8 :=
  sorry

end NUMINAMATH_GPT_number_of_oddly_powerful_integers_lt_500_l269_26962


namespace NUMINAMATH_GPT_find_m_direct_proportion_l269_26915

theorem find_m_direct_proportion (m : ℝ) (h1 : m^2 - 3 = 1) (h2 : m ≠ 2) : m = -2 :=
by {
  -- here would be the proof, but it's omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_find_m_direct_proportion_l269_26915


namespace NUMINAMATH_GPT_average_of_integers_l269_26990

theorem average_of_integers (A B C D : ℤ) (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D = 90) (h5 : 5 ≤ A) (h6 : A ≠ B ∧ B ≠ C ∧ C ≠ D) :
  (A + B + C + D) / 4 = 27 :=
by
  sorry

end NUMINAMATH_GPT_average_of_integers_l269_26990


namespace NUMINAMATH_GPT_proof_problem_l269_26917

variable {a b c : ℝ}

theorem proof_problem (h_cond : 0 < a ∧ a < b ∧ b < c) : 
  a * c < b * c ∧ a + b < b + c ∧ c / a > c / b := by
  sorry

end NUMINAMATH_GPT_proof_problem_l269_26917


namespace NUMINAMATH_GPT_expression_evaluation_l269_26966

theorem expression_evaluation (x y z : ℝ) (h : x = y + z) (h' : x = 2) :
  x^3 + 2 * y^3 + 2 * z^3 + 6 * x * y * z = 24 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l269_26966


namespace NUMINAMATH_GPT_total_water_intake_l269_26959

def theo_weekday := 8
def mason_weekday := 7
def roxy_weekday := 9
def zara_weekday := 10
def lily_weekday := 6

def theo_weekend := 10
def mason_weekend := 8
def roxy_weekend := 11
def zara_weekend := 12
def lily_weekend := 7

def total_cups_in_week (weekday_cups weekend_cups : ℕ) : ℕ :=
  5 * weekday_cups + 2 * weekend_cups

theorem total_water_intake :
  total_cups_in_week theo_weekday theo_weekend +
  total_cups_in_week mason_weekday mason_weekend +
  total_cups_in_week roxy_weekday roxy_weekend +
  total_cups_in_week zara_weekday zara_weekend +
  total_cups_in_week lily_weekday lily_weekend = 296 :=
by sorry

end NUMINAMATH_GPT_total_water_intake_l269_26959


namespace NUMINAMATH_GPT_erika_walked_distance_l269_26949

/-- Erika traveled to visit her cousin. She started on a scooter at an average speed of 
22 kilometers per hour. After completing three-fifths of the distance, the scooter's battery died, 
and she walked the rest of the way at 4 kilometers per hour. The total time it took her to reach her cousin's 
house was 2 hours. How far, in kilometers rounded to the nearest tenth, did Erika walk? -/
theorem erika_walked_distance (d : ℝ) (h1 : d > 0)
  (h2 : (3 / 5 * d) / 22 + (2 / 5 * d) / 4 = 2) : 
  (2 / 5 * d) = 6.3 :=
sorry

end NUMINAMATH_GPT_erika_walked_distance_l269_26949


namespace NUMINAMATH_GPT_negation_of_existential_proposition_l269_26940

theorem negation_of_existential_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 > Real.exp x_0) ↔ ∀ (x : ℝ), x^2 ≤ Real.exp x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_proposition_l269_26940


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l269_26935

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := -4, y := 2 }
def Q : Point := { x := 8, y := 2 }
def R : Point := { x := 6, y := -4 }

noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangle_area P Q R = 36 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l269_26935


namespace NUMINAMATH_GPT_superhero_vs_supervillain_distance_l269_26937

-- Definitions expressing the conditions
def superhero_speed (miles : ℕ) (minutes : ℕ) := (10 : ℕ) / (4 : ℕ)
def supervillain_speed (miles_per_hour : ℕ) := (100 : ℕ)

-- Distance calculation in 60 minutes
def superhero_distance_in_hour := 60 * superhero_speed 10 4
def supervillain_distance_in_hour := supervillain_speed 100

-- Proof statement
theorem superhero_vs_supervillain_distance :
  superhero_distance_in_hour - supervillain_distance_in_hour = (50 : ℕ) :=
by
  sorry

end NUMINAMATH_GPT_superhero_vs_supervillain_distance_l269_26937


namespace NUMINAMATH_GPT_div_polynomials_l269_26924

variable (a b : ℝ)

theorem div_polynomials :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b := 
by sorry

end NUMINAMATH_GPT_div_polynomials_l269_26924


namespace NUMINAMATH_GPT_problem1_problem2_l269_26903

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Define the conditions and questions as Lean statements

-- First problem: Prove that if A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem problem1 (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := 
  sorry

-- Second problem: Prove that if A a ⊆ B, then a ∈ (-∞, 0] ∪ [4, ∞)
theorem problem2 (a : ℝ) (h1 : A a ⊆ B) : a ≤ 0 ∨ a ≥ 4 := 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l269_26903


namespace NUMINAMATH_GPT_new_average_is_15_l269_26900

-- Definitions corresponding to the conditions
def avg_10_consecutive (seq : List ℤ) : Prop :=
  seq.length = 10 ∧ seq.sum = 200

def new_seq (seq : List ℤ) : List ℤ :=
  List.mapIdx (λ i x => x - ↑(9 - i)) seq

-- Statement of the proof problem
theorem new_average_is_15
  (seq : List ℤ)
  (h_seq : avg_10_consecutive seq) :
  (new_seq seq).sum = 150 := sorry

end NUMINAMATH_GPT_new_average_is_15_l269_26900


namespace NUMINAMATH_GPT_fraction_inequality_solution_l269_26931

open Set

theorem fraction_inequality_solution :
  {x : ℝ | 7 * x - 3 ≥ x^2 - x - 12 ∧ x ≠ 3 ∧ x ≠ -4} = Icc (-1 : ℝ) 3 ∪ Ioo (3 : ℝ) 4 ∪ Icc 4 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_solution_l269_26931


namespace NUMINAMATH_GPT_cos_double_beta_alpha_plus_double_beta_l269_26987

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = Real.sqrt 2 / 10)
variable (h2 : Real.sin β = Real.sqrt 10 / 10)

theorem cos_double_beta :
  Real.cos (2 * β) = 4 / 5 := by 
  sorry

theorem alpha_plus_double_beta :
  α + 2 * β = π / 4 := by 
  sorry

end NUMINAMATH_GPT_cos_double_beta_alpha_plus_double_beta_l269_26987


namespace NUMINAMATH_GPT_find_larger_number_l269_26970

theorem find_larger_number (a b : ℕ) (h1 : a + b = 96) (h2 : a = b + 12) : a = 54 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l269_26970


namespace NUMINAMATH_GPT_men_in_second_group_l269_26977

theorem men_in_second_group (W : ℝ)
  (h1 : W = 18 * 20)
  (h2 : W = M * 30) :
  M = 12 :=
by
  sorry

end NUMINAMATH_GPT_men_in_second_group_l269_26977


namespace NUMINAMATH_GPT_negation_proposition_l269_26926

theorem negation_proposition (x : ℝ) (hx : 0 < x) : x + 4 / x ≥ 4 :=
sorry

end NUMINAMATH_GPT_negation_proposition_l269_26926


namespace NUMINAMATH_GPT_sum_of_squares_transform_l269_26905

def isSumOfThreeSquaresDivByThree (N : ℕ) : Prop := 
  ∃ (a b c : ℤ), N = a^2 + b^2 + c^2 ∧ (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c)

def isSumOfThreeSquaresNotDivByThree (N : ℕ) : Prop := 
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z)

theorem sum_of_squares_transform {N : ℕ} :
  isSumOfThreeSquaresDivByThree N → isSumOfThreeSquaresNotDivByThree N :=
sorry

end NUMINAMATH_GPT_sum_of_squares_transform_l269_26905


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l269_26965

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1 → x^2 > 1) ∧ ¬(x^2 > 1 → x < -1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l269_26965


namespace NUMINAMATH_GPT_logarithmic_expression_max_value_l269_26916

theorem logarithmic_expression_max_value (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a / b = 3) :
  3 * Real.log (a / b) / Real.log a + 2 * Real.log (b / a) / Real.log b = -4 := 
sorry

end NUMINAMATH_GPT_logarithmic_expression_max_value_l269_26916


namespace NUMINAMATH_GPT_sum_of_first_1000_terms_l269_26913

def sequence_block_sum (n : ℕ) : ℕ :=
  1 + 3 * n

def sequence_sum_up_to (k : ℕ) : ℕ :=
  if k = 0 then 0 else (1 + 3 * (k * (k - 1) / 2)) + k

def nth_term_position (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + n

theorem sum_of_first_1000_terms : sequence_sum_up_to 43 + (1000 - nth_term_position 43) * 3 = 2912 :=
sorry

end NUMINAMATH_GPT_sum_of_first_1000_terms_l269_26913


namespace NUMINAMATH_GPT_linear_function_quadrants_l269_26938

theorem linear_function_quadrants (m : ℝ) :
  (∀ (x : ℝ), y = -3 * x + m →
  (x < 0 ∧ y > 0 ∨ x > 0 ∧ y < 0 ∨ x < 0 ∧ y < 0)) → m < 0 :=
sorry

end NUMINAMATH_GPT_linear_function_quadrants_l269_26938


namespace NUMINAMATH_GPT_range_of_p_nonnegative_range_of_p_all_values_range_of_p_l269_26902

def p (x : ℝ) : ℝ := x^4 - 6 * x^2 + 9

theorem range_of_p_nonnegative (x : ℝ) (hx : 0 ≤ x) : 
  ∃ y, y = p x ∧ 0 ≤ y := 
sorry

theorem range_of_p_all_values (y : ℝ) : 
  0 ≤ y → (∃ x, 0 ≤ x ∧ p x = y) :=
sorry

theorem range_of_p (x : ℝ) (hx : 0 ≤ x) : 
  ∀ y, (∃ x, 0 ≤ x ∧ p x = y) ↔ (0 ≤ y) :=
sorry

end NUMINAMATH_GPT_range_of_p_nonnegative_range_of_p_all_values_range_of_p_l269_26902


namespace NUMINAMATH_GPT_algebraic_expression_value_l269_26952

theorem algebraic_expression_value (p q : ℤ) 
  (h : 8 * p + 2 * q = -2023) : 
  (p * (-2) ^ 3 + q * (-2) + 1 = 2024) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l269_26952


namespace NUMINAMATH_GPT_max_value_fraction_l269_26947

theorem max_value_fraction (a b : ℝ) (h1 : ab = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  ∃ C, C = 30 / 97 ∧ (∀ x y : ℝ, (xy = 1) → (x > y) → (y ≥ 2/3) → (x - y) / (x^2 + y^2) ≤ C) :=
sorry

end NUMINAMATH_GPT_max_value_fraction_l269_26947


namespace NUMINAMATH_GPT_service_cost_is_correct_l269_26999

def service_cost_per_vehicle(cost_per_liter: ℝ)
                            (num_minivans: ℕ) 
                            (num_trucks: ℕ)
                            (total_cost: ℝ) 
                            (minivan_tank_liters: ℝ)
                            (truck_size_increase_pct: ℝ) 
                            (total_fuel: ℝ) 
                            (total_fuel_cost: ℝ) 
                            (total_service_cost: ℝ)
                            (num_vehicles: ℕ) 
                            (service_cost_per_vehicle: ℝ) : Prop :=
  cost_per_liter = 0.70 ∧
  num_minivans = 4 ∧
  num_trucks = 2 ∧
  total_cost = 395.4 ∧
  minivan_tank_liters = 65 ∧
  truck_size_increase_pct = 1.2 ∧
  total_fuel = (4 * minivan_tank_liters) + (2 * (minivan_tank_liters * (1 + truck_size_increase_pct))) ∧
  total_fuel_cost = total_fuel * cost_per_liter ∧
  total_service_cost = total_cost - total_fuel_cost ∧
  num_vehicles = num_minivans + num_trucks ∧
  service_cost_per_vehicle = total_service_cost / num_vehicles

-- Now, we state the theorem we want to prove.
theorem service_cost_is_correct :
  service_cost_per_vehicle 0.70 4 2 395.4 65 1.2 546 382.2 13.2 6 2.2 :=
by {
    sorry
}

end NUMINAMATH_GPT_service_cost_is_correct_l269_26999


namespace NUMINAMATH_GPT_total_cost_l269_26968

def c_teacher : ℕ := 60
def c_student : ℕ := 40

theorem total_cost (x : ℕ) : ∃ y : ℕ, y = c_student * x + c_teacher := by
  sorry

end NUMINAMATH_GPT_total_cost_l269_26968


namespace NUMINAMATH_GPT_moles_of_CO2_formed_l269_26908

-- Define the reaction
def reaction (HCl NaHCO3 CO2 : ℕ) : Prop :=
  HCl = NaHCO3 ∧ HCl + NaHCO3 = CO2

-- Given conditions
def given_conditions : Prop :=
  ∃ (HCl NaHCO3 CO2 : ℕ),
    reaction HCl NaHCO3 CO2 ∧ HCl = 3 ∧ NaHCO3 = 3

-- Prove the number of moles of CO2 formed is 3.
theorem moles_of_CO2_formed : given_conditions → ∃ CO2 : ℕ, CO2 = 3 :=
  by
    intros h
    sorry

end NUMINAMATH_GPT_moles_of_CO2_formed_l269_26908


namespace NUMINAMATH_GPT_largest_corner_sum_l269_26979

-- Definitions based on the given problem
def faces_labeled : List ℕ := [2, 3, 4, 5, 6, 7]
def opposite_faces : List (ℕ × ℕ) := [(2, 7), (3, 6), (4, 5)]

-- Condition that face 2 cannot be adjacent to face 4
def non_adjacent_faces : List (ℕ × ℕ) := [(2, 4)]

-- Function to check adjacency constraints
def adjacent_allowed (f1 f2 : ℕ) : Bool := 
  ¬ (f1, f2) ∈ non_adjacent_faces ∧ ¬ (f2, f1) ∈ non_adjacent_faces

-- Determine the largest sum of three numbers whose faces meet at a corner
theorem largest_corner_sum : ∃ (a b c : ℕ), a ∈ faces_labeled ∧ b ∈ faces_labeled ∧ c ∈ faces_labeled ∧ 
  (adjacent_allowed a b) ∧ (adjacent_allowed b c) ∧ (adjacent_allowed c a) ∧ 
  a + b + c = 18 := 
sorry

end NUMINAMATH_GPT_largest_corner_sum_l269_26979


namespace NUMINAMATH_GPT_total_revenue_correct_l269_26906

-- Define the conditions
def charge_per_slice : ℕ := 5
def slices_per_pie : ℕ := 4
def pies_sold : ℕ := 9

-- Prove the question: total revenue
theorem total_revenue_correct : charge_per_slice * slices_per_pie * pies_sold = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l269_26906


namespace NUMINAMATH_GPT_largest_int_with_remainder_5_lt_100_l269_26955

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end NUMINAMATH_GPT_largest_int_with_remainder_5_lt_100_l269_26955


namespace NUMINAMATH_GPT_inequality_proof_l269_26973

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l269_26973


namespace NUMINAMATH_GPT_area_change_l269_26942

variable (L B : ℝ)

def initial_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.20 * L

def new_breadth (B : ℝ) : ℝ := 0.95 * B

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

theorem area_change (L B : ℝ) : new_area L B = 1.14 * (initial_area L B) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_area_change_l269_26942


namespace NUMINAMATH_GPT_largest_k_consecutive_sum_l269_26960

theorem largest_k_consecutive_sum (k n : ℕ) :
  (5^7 = (k * (2 * n + k + 1)) / 2) → 1 ≤ k → k * (2 * n + k + 1) = 2 * 5^7 → k = 250 :=
sorry

end NUMINAMATH_GPT_largest_k_consecutive_sum_l269_26960


namespace NUMINAMATH_GPT_neg_one_exponent_difference_l269_26943

theorem neg_one_exponent_difference : (-1 : ℤ) ^ 2004 - (-1 : ℤ) ^ 2003 = 2 := by
  sorry

end NUMINAMATH_GPT_neg_one_exponent_difference_l269_26943


namespace NUMINAMATH_GPT_expression_value_l269_26998

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end NUMINAMATH_GPT_expression_value_l269_26998


namespace NUMINAMATH_GPT_eval_f_at_two_eval_f_at_neg_two_l269_26921

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x

theorem eval_f_at_two : f 2 = 14 :=
by
  sorry

theorem eval_f_at_neg_two : f (-2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_eval_f_at_two_eval_f_at_neg_two_l269_26921


namespace NUMINAMATH_GPT_simplify_and_evaluate_l269_26954

theorem simplify_and_evaluate (a : ℝ) (h : a = -3 / 2) : 
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l269_26954


namespace NUMINAMATH_GPT_a5_a6_value_l269_26983

def S (n : ℕ) : ℕ := n^3

theorem a5_a6_value : S 6 - S 4 = 152 :=
by
  sorry

end NUMINAMATH_GPT_a5_a6_value_l269_26983


namespace NUMINAMATH_GPT_tablet_battery_life_l269_26967

theorem tablet_battery_life :
  ∀ (active_usage_hours idle_usage_hours : ℕ),
  active_usage_hours + idle_usage_hours = 12 →
  active_usage_hours = 3 →
  ((active_usage_hours / 2) + (idle_usage_hours / 10)) > 1 →
  idle_usage_hours = 9 →
  0 = 0 := 
by
  intros active_usage_hours idle_usage_hours h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tablet_battery_life_l269_26967


namespace NUMINAMATH_GPT_sum_alternating_sequence_l269_26928

theorem sum_alternating_sequence : (Finset.range 2012).sum (λ k => (-1 : ℤ)^(k + 1)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_alternating_sequence_l269_26928


namespace NUMINAMATH_GPT_enlarged_poster_height_l269_26992

def original_poster_width : ℝ := 3
def original_poster_height : ℝ := 2
def new_poster_width : ℝ := 12

theorem enlarged_poster_height :
  new_poster_width / original_poster_width * original_poster_height = 8 := 
by
  sorry

end NUMINAMATH_GPT_enlarged_poster_height_l269_26992


namespace NUMINAMATH_GPT_rabbit_jump_lengths_order_l269_26978

theorem rabbit_jump_lengths_order :
  ∃ (R : ℕ) (G : ℕ) (P : ℕ) (F : ℕ),
    R = 2730 ∧
    R = P + 1100 ∧
    P = F + 150 ∧
    F = G - 200 ∧
    R > G ∧ G > P ∧ P > F :=
  by
  -- calculations
  sorry

end NUMINAMATH_GPT_rabbit_jump_lengths_order_l269_26978


namespace NUMINAMATH_GPT_correct_divisor_l269_26985

theorem correct_divisor (X D : ℕ) (h1 : X / 72 = 24) (h2 : X / D = 48) : D = 36 :=
sorry

end NUMINAMATH_GPT_correct_divisor_l269_26985


namespace NUMINAMATH_GPT_number_of_persons_l269_26945

theorem number_of_persons (P : ℕ) : 
  (P * 12 * 5 = 30 * 13 * 6) → P = 39 :=
by
  sorry

end NUMINAMATH_GPT_number_of_persons_l269_26945


namespace NUMINAMATH_GPT_ratio_wheelbarrow_to_earnings_l269_26981

theorem ratio_wheelbarrow_to_earnings :
  let duck_price := 10
  let chicken_price := 8
  let chickens_sold := 5
  let ducks_sold := 2
  let resale_earn := 60
  let total_earnings := chickens_sold * chicken_price + ducks_sold * duck_price
  let wheelbarrow_cost := resale_earn / 2
  (wheelbarrow_cost / total_earnings = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_wheelbarrow_to_earnings_l269_26981


namespace NUMINAMATH_GPT_sum_of_next_five_even_integers_l269_26984

theorem sum_of_next_five_even_integers (a : ℕ) (x : ℕ) 
  (h : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end NUMINAMATH_GPT_sum_of_next_five_even_integers_l269_26984


namespace NUMINAMATH_GPT_perfect_square_expression_l269_26956

theorem perfect_square_expression (x y : ℕ) (p : ℕ) [Fact (Nat.Prime p)]
    (h : 4 * x^2 + 8 * y^2 + (2 * x - 3 * y) * p - 12 * x * y = 0) :
    ∃ (n : ℕ), 4 * y + 1 = n^2 :=
sorry

end NUMINAMATH_GPT_perfect_square_expression_l269_26956


namespace NUMINAMATH_GPT_curves_intersect_at_l269_26914

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def g (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

theorem curves_intersect_at :
  (∃ x : ℝ, f x = g x) ↔ ([(0, 2), (6, 86)] = [(0, 2), (6, 86)]) :=
by
  sorry

end NUMINAMATH_GPT_curves_intersect_at_l269_26914


namespace NUMINAMATH_GPT_rosa_peaches_more_than_apples_l269_26934

def steven_peaches : ℕ := 17
def steven_apples  : ℕ := 16
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples  : ℕ := steven_apples + 8
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples  : ℕ := steven_apples / 2

theorem rosa_peaches_more_than_apples : rosa_peaches - rosa_apples = 25 := by
  sorry

end NUMINAMATH_GPT_rosa_peaches_more_than_apples_l269_26934


namespace NUMINAMATH_GPT_BC_length_47_l269_26988

theorem BC_length_47 (A B C D : ℝ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : B ≠ D)
  (h₄ : dist A C = 20) (h₅ : dist A D = 45) (h₆ : dist B D = 13)
  (h₇ : C = 0) (h₈ : D = 0) (h₉ : B = A + 43) :
  dist B C = 47 :=
sorry

end NUMINAMATH_GPT_BC_length_47_l269_26988


namespace NUMINAMATH_GPT_circle_area_l269_26932

theorem circle_area (r : ℝ) (h : 6 / (2 * π * r) = r / 2) : π * r^2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l269_26932


namespace NUMINAMATH_GPT_smallest_b_for_factorization_l269_26972

theorem smallest_b_for_factorization :
  ∃ (b : ℕ), (∀ r s : ℕ, (r * s = 3258) → (b = r + s)) ∧ (∀ c : ℕ, (∀ r' s' : ℕ, (r' * s' = 3258) → (c = r' + s')) → b ≤ c) :=
sorry

end NUMINAMATH_GPT_smallest_b_for_factorization_l269_26972


namespace NUMINAMATH_GPT_prank_helpers_combinations_l269_26919

theorem prank_helpers_combinations :
  let Monday := 1
  let Tuesday := 2
  let Wednesday := 3
  let Thursday := 4
  let Friday := 1
  (Monday * Tuesday * Wednesday * Thursday * Friday = 24) :=
by
  intros
  sorry

end NUMINAMATH_GPT_prank_helpers_combinations_l269_26919


namespace NUMINAMATH_GPT_stock_price_end_of_second_year_l269_26941

def initial_price : ℝ := 80
def first_year_increase_rate : ℝ := 1.2
def second_year_decrease_rate : ℝ := 0.3

theorem stock_price_end_of_second_year : 
  initial_price * (1 + first_year_increase_rate) * (1 - second_year_decrease_rate) = 123.2 := 
by sorry

end NUMINAMATH_GPT_stock_price_end_of_second_year_l269_26941


namespace NUMINAMATH_GPT_man_speed_in_still_water_l269_26929

theorem man_speed_in_still_water
  (speed_of_current_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_meters : ℝ)
  (speed_of_current_ms : ℝ := speed_of_current_kmph * (1000 / 3600))
  (speed_downstream : ℝ := distance_meters / time_seconds) :
  speed_of_current_kmph = 3 →
  time_seconds = 13.998880089592832 →
  distance_meters = 70 →
  (speed_downstream = (25 / 6)) →
  (speed_downstream - speed_of_current_ms) * (3600 / 1000) = 15 :=
by
  intros h_speed_current h_time h_distance h_downstream
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l269_26929


namespace NUMINAMATH_GPT_max_area_of_triangle_l269_26961

open Real

theorem max_area_of_triangle (a b c : ℝ) 
  (ha : 9 ≥ a) 
  (ha1 : a ≥ 8) 
  (hb : 8 ≥ b) 
  (hb1 : b ≥ 4) 
  (hc : 4 ≥ c) 
  (hc1 : c ≥ 3) : 
  ∃ A : ℝ, ∃ S : ℝ, S ≤ 16 ∧ S = max (1/2 * b * c * sin A) 16 := 
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l269_26961


namespace NUMINAMATH_GPT_find_m_l269_26957

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 - 2 * x + m) 
  (h2 : ∀ x ≥ (3 : ℝ), f x ≥ 1) : m = -2 := 
sorry

end NUMINAMATH_GPT_find_m_l269_26957


namespace NUMINAMATH_GPT_calculate_speed_l269_26994

-- Define the distance and time conditions
def distance : ℝ := 390
def time : ℝ := 4

-- Define the expected answer for speed
def expected_speed : ℝ := 97.5

-- Prove that speed equals expected_speed given the conditions
theorem calculate_speed : (distance / time) = expected_speed :=
by
  -- skipped proof steps
  sorry

end NUMINAMATH_GPT_calculate_speed_l269_26994


namespace NUMINAMATH_GPT_profit_sharing_l269_26975

theorem profit_sharing
  (A_investment B_investment C_investment total_profit : ℕ)
  (A_share : ℕ)
  (ratio_A ratio_B ratio_C : ℕ)
  (hA : A_investment = 6300)
  (hB : B_investment = 4200)
  (hC : C_investment = 10500)
  (hShare : A_share = 3810)
  (hRatio : ratio_A = 3 ∧ ratio_B = 2 ∧ ratio_C = 5)
  (hTotRatio : ratio_A + ratio_B + ratio_C = 10)
  (hShareCalc : A_share = (3/10) * total_profit) :
  total_profit = 12700 :=
sorry

end NUMINAMATH_GPT_profit_sharing_l269_26975


namespace NUMINAMATH_GPT_arccos_cos_eq_x_div_3_solutions_l269_26974

theorem arccos_cos_eq_x_div_3_solutions (x : ℝ) :
  (Real.arccos (Real.cos x) = x / 3) ∧ (-3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2) 
  ↔ x = -3 * Real.pi / 2 ∨ x = 0 ∨ x = 3 * Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arccos_cos_eq_x_div_3_solutions_l269_26974


namespace NUMINAMATH_GPT_sqrt_of_4_l269_26904

theorem sqrt_of_4 :
  {x | x * x = 4} = {2, -2} :=
sorry

end NUMINAMATH_GPT_sqrt_of_4_l269_26904


namespace NUMINAMATH_GPT_fraction_conversion_l269_26950

theorem fraction_conversion :
  let A := 4.5
  let B := 0.8
  let C := 80.0
  let D := 0.08
  let E := 0.45
  (4 / 5) = B :=
by
  sorry

end NUMINAMATH_GPT_fraction_conversion_l269_26950


namespace NUMINAMATH_GPT_triangle_obtuse_at_most_one_l269_26948

open Real -- Work within the Real number system

-- Definitions and main proposition
def is_obtuse (angle : ℝ) : Prop := angle > 90

def triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem triangle_obtuse_at_most_one (a b c : ℝ) (h : triangle a b c) :
  is_obtuse a ∧ is_obtuse b → false :=
by
  sorry

end NUMINAMATH_GPT_triangle_obtuse_at_most_one_l269_26948


namespace NUMINAMATH_GPT_find_extrema_l269_26996

noncomputable def function_extrema (x : ℝ) : ℝ :=
  (2 / 3) * Real.cos (3 * x - Real.pi / 6)

theorem find_extrema :
  (function_extrema (Real.pi / 18) = 2 / 3 ∧
   function_extrema (7 * Real.pi / 18) = -(2 / 3)) ∧
  (0 < Real.pi / 18 ∧ Real.pi / 18 < Real.pi / 2) ∧
  (0 < 7 * Real.pi / 18 ∧ 7 * Real.pi / 18 < Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_extrema_l269_26996


namespace NUMINAMATH_GPT_qr_length_is_correct_l269_26930

/-- Define points and segments in the triangle. -/
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(P Q R : Point)

def PQ_length (T : Triangle) : ℝ :=
(T.Q.x - T.P.x) * (T.Q.x - T.P.x) + (T.Q.y - T.P.y) * (T.Q.y - T.P.y)

def PR_length (T : Triangle) : ℝ :=
(T.R.x - T.P.x) * (T.R.x - T.P.x) + (T.R.y - T.P.y) * (T.R.y - T.P.y)

def QR_length (T : Triangle) : ℝ :=
(T.R.x - T.Q.x) * (T.R.x - T.Q.x) + (T.R.y - T.Q.y) * (T.R.y - T.Q.y)

noncomputable def XZ_length (T : Triangle) (X Y Z : Point) : ℝ :=
(PQ_length T)^(1/2) -- Assume the least length of XZ that follows the given conditions

theorem qr_length_is_correct (T : Triangle) :
  PQ_length T = 4*4 → 
  XZ_length T T.P T.Q T.R = 3.2 →
  QR_length T = 4*4 :=
sorry

end NUMINAMATH_GPT_qr_length_is_correct_l269_26930


namespace NUMINAMATH_GPT_cost_price_toy_l269_26936

theorem cost_price_toy (selling_price_total : ℝ) (total_toys : ℕ) (gain_toys : ℕ) (sp_per_toy : ℝ) (general_cost : ℝ) :
  selling_price_total = 27300 →
  total_toys = 18 →
  gain_toys = 3 →
  sp_per_toy = selling_price_total / total_toys →
  general_cost = sp_per_toy * total_toys - (sp_per_toy * gain_toys / total_toys) →
    general_cost = 1300 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_toy_l269_26936


namespace NUMINAMATH_GPT_perimeter_ratio_l269_26925

/-- Suppose we have a square piece of paper, 6 inches on each side, folded in half horizontally. 
The paper is then cut along the fold, and one of the halves is subsequently cut again horizontally 
through all layers. This results in one large rectangle and two smaller identical rectangles. 
Find the ratio of the perimeter of one smaller rectangle to the perimeter of the larger rectangle. -/
theorem perimeter_ratio (side_length : ℝ) (half_side_length : ℝ) (double_half_side_length : ℝ) :
    side_length = 6 →
    half_side_length = side_length / 2 →
    double_half_side_length = 1.5 * 2 →
    (2 * (half_side_length / 2 + side_length)) / (2 * (half_side_length + side_length)) = (5 / 6) :=
by
    -- Declare the side lengths
    intros h₁ h₂ h₃
    -- Insert the necessary algebra (proven manually earlier)
    sorry

end NUMINAMATH_GPT_perimeter_ratio_l269_26925


namespace NUMINAMATH_GPT_frog_weight_difference_l269_26927

theorem frog_weight_difference
  (large_frog_weight : ℕ)
  (small_frog_weight : ℕ)
  (h1 : large_frog_weight = 10 * small_frog_weight)
  (h2 : large_frog_weight = 120) :
  large_frog_weight - small_frog_weight = 108 :=
by
  sorry

end NUMINAMATH_GPT_frog_weight_difference_l269_26927


namespace NUMINAMATH_GPT_arithmetic_sequence_1000th_term_l269_26976

theorem arithmetic_sequence_1000th_term (a_1 : ℤ) (d : ℤ) (n : ℤ) (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 1000) : 
  a_1 + (n - 1) * d = 2998 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_1000th_term_l269_26976


namespace NUMINAMATH_GPT_quadrilateral_tile_angles_l269_26963

theorem quadrilateral_tile_angles :
  ∃ a b c d : ℝ, a + b + c + d = 360 ∧ a = 45 ∧ b = 60 ∧ c = 105 ∧ d = 150 := 
by {
  sorry
}

end NUMINAMATH_GPT_quadrilateral_tile_angles_l269_26963


namespace NUMINAMATH_GPT_cuboid_length_l269_26971

theorem cuboid_length (b h : ℝ) (A : ℝ) (l : ℝ) : b = 6 → h = 5 → A = 120 → 2 * (l * b + b * h + h * l) = A → l = 30 / 11 :=
by
  intros hb hh hA hSurfaceArea
  rw [hb, hh] at hSurfaceArea
  sorry

end NUMINAMATH_GPT_cuboid_length_l269_26971


namespace NUMINAMATH_GPT_monotonic_implies_m_l269_26997

noncomputable def cubic_function (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

theorem monotonic_implies_m (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + m) ≥ 0) → m ≥ 1 / 3 :=
  sorry

end NUMINAMATH_GPT_monotonic_implies_m_l269_26997


namespace NUMINAMATH_GPT_oliver_workout_hours_l269_26991

variable (x : ℕ)

theorem oliver_workout_hours :
  (x + (x - 2) + 2 * x + 2 * (x - 2) = 18) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_oliver_workout_hours_l269_26991


namespace NUMINAMATH_GPT_worth_of_each_gift_is_4_l269_26918

noncomputable def worth_of_each_gift
  (workers_per_block : ℕ)
  (total_blocks : ℕ)
  (total_amount : ℝ) : ℝ :=
  total_amount / (workers_per_block * total_blocks)

theorem worth_of_each_gift_is_4 (workers_per_block total_blocks : ℕ) (total_amount : ℝ)
  (h1 : workers_per_block = 100)
  (h2 : total_blocks = 10)
  (h3 : total_amount = 4000) :
  worth_of_each_gift workers_per_block total_blocks total_amount = 4 :=
by
  sorry

end NUMINAMATH_GPT_worth_of_each_gift_is_4_l269_26918


namespace NUMINAMATH_GPT_alberto_spent_2457_l269_26920

-- Define the expenses by Samara on each item
def oil_expense : ℕ := 25
def tires_expense : ℕ := 467
def detailing_expense : ℕ := 79

-- Define the additional amount Alberto spent more than Samara
def additional_amount : ℕ := 1886

-- Total amount spent by Samara
def samara_total_expense : ℕ := oil_expense + tires_expense + detailing_expense

-- The amount spent by Alberto
def alberto_expense := samara_total_expense + additional_amount

-- Theorem stating the amount spent by Alberto
theorem alberto_spent_2457 :
  alberto_expense = 2457 :=
by {
  -- Include the actual proof here if necessary
  sorry
}

end NUMINAMATH_GPT_alberto_spent_2457_l269_26920
