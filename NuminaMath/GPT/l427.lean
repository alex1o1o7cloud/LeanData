import Mathlib

namespace NUMINAMATH_GPT_min_value_frac_sum_l427_42777

variable {a b c : ℝ}

theorem min_value_frac_sum (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : a * b + b * c + c * a = 1) : 
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = (9 + 3 * Real.sqrt 3) / 2 :=
  sorry

end NUMINAMATH_GPT_min_value_frac_sum_l427_42777


namespace NUMINAMATH_GPT_train_passing_time_l427_42765

def train_distance_km : ℝ := 10
def train_time_min : ℝ := 15
def train_length_m : ℝ := 111.11111111111111

theorem train_passing_time : 
  let time_to_pass_signal_post := train_length_m / ((train_distance_km * 1000) / (train_time_min * 60))
  time_to_pass_signal_post = 10 :=
by
  sorry

end NUMINAMATH_GPT_train_passing_time_l427_42765


namespace NUMINAMATH_GPT_fraction_is_three_eights_l427_42780

-- The given number
def number := 48

-- The fraction 'x' by which the number exceeds by 30
noncomputable def fraction (x : ℝ) : Prop :=
number = number * x + 30

-- Our goal is to prove that the fraction is 3/8
theorem fraction_is_three_eights : fraction (3 / 8) :=
by
  -- We reduced the goal proof to a simpler form for illustration, you can solve it rigorously
  sorry

end NUMINAMATH_GPT_fraction_is_three_eights_l427_42780


namespace NUMINAMATH_GPT_fraction_of_red_marbles_after_tripling_blue_l427_42774

theorem fraction_of_red_marbles_after_tripling_blue (x : ℕ) (h₁ : ∃ y, y = (4 * x) / 7) (h₂ : ∃ z, z = (3 * x) / 7) :
  (3 * x / 7) / (((12 * x) / 7) + ((3 * x) / 7)) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_red_marbles_after_tripling_blue_l427_42774


namespace NUMINAMATH_GPT_equation_completing_square_l427_42787

theorem equation_completing_square :
  ∃ (a b c : ℤ), 64 * x^2 + 80 * x - 81 = 0 → 
  (a > 0) ∧ (2 * a * b = 80) ∧ (a^2 = 64) ∧ (a + b + c = 119) :=
sorry

end NUMINAMATH_GPT_equation_completing_square_l427_42787


namespace NUMINAMATH_GPT_proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l427_42762

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end NUMINAMATH_GPT_proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l427_42762


namespace NUMINAMATH_GPT_rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l427_42743

-- Case 1
theorem rt_triangle_case1
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 30) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = 4) (hb : b = 4 * Real.sqrt 3) (hc : c = 8)
  : b = 4 * Real.sqrt 3 ∧ c = 8 := by
  sorry

-- Case 2
theorem rt_triangle_case2
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : B = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = Real.sqrt 3 - 1) (hb : b = 3 - Real.sqrt 3) 
  (ha_b: A = 30)
  (h_c: c = 2 * Real.sqrt 3 - 2)
  : B = 60 ∧ A = 30 ∧ c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem rt_triangle_case3
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (hc : c = 2 + Real.sqrt 3)
  (ha : a = Real.sqrt 3 + 3/2) 
  (hb: b = (2 + Real.sqrt 3) / 2)
  : a = Real.sqrt 3 + 3/2 ∧ b = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l427_42743


namespace NUMINAMATH_GPT_problem_statement_l427_42724

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

theorem problem_statement (b : ℝ) (hb : 2^0 + 2*0 + b = 0) : f (-1) b = -3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l427_42724


namespace NUMINAMATH_GPT_polygon_sides_given_ratio_l427_42769

theorem polygon_sides_given_ratio (n : ℕ) 
  (h : (n - 2) * 180 / 360 = 9 / 2) : n = 11 :=
sorry

end NUMINAMATH_GPT_polygon_sides_given_ratio_l427_42769


namespace NUMINAMATH_GPT_smallest_number_is_28_l427_42797

theorem smallest_number_is_28 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 30) (h4 : b = 29) (h5 : c = b + 4) : a = 28 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_is_28_l427_42797


namespace NUMINAMATH_GPT_find_A_l427_42768

theorem find_A (A B C D E F G H I J : ℕ)
  (h1 : A > B ∧ B > C)
  (h2 : D > E ∧ E > F)
  (h3 : G > H ∧ H > I ∧ I > J)
  (h4 : (D = E + 2) ∧ (E = F + 2))
  (h5 : (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2))
  (h6 : A + B + C = 10) : A = 6 :=
sorry

end NUMINAMATH_GPT_find_A_l427_42768


namespace NUMINAMATH_GPT_total_kayaks_built_by_april_l427_42791

def kayaks_built_february : ℕ := 5
def kayaks_built_next_month (n : ℕ) : ℕ := 3 * n
def kayaks_built_march : ℕ := kayaks_built_next_month kayaks_built_february
def kayaks_built_april : ℕ := kayaks_built_next_month kayaks_built_march

theorem total_kayaks_built_by_april : 
  kayaks_built_february + kayaks_built_march + kayaks_built_april = 65 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_kayaks_built_by_april_l427_42791


namespace NUMINAMATH_GPT_fraction_subtraction_l427_42794

theorem fraction_subtraction :
  (8 / 23) - (5 / 46) = 11 / 46 := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l427_42794


namespace NUMINAMATH_GPT_calculate_xy_l427_42760

theorem calculate_xy (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 :=
by
  sorry

end NUMINAMATH_GPT_calculate_xy_l427_42760


namespace NUMINAMATH_GPT_christine_wander_time_l427_42750

noncomputable def distance : ℝ := 80
noncomputable def speed : ℝ := 20
noncomputable def time : ℝ := distance / speed

theorem christine_wander_time : time = 4 := 
by
  sorry

end NUMINAMATH_GPT_christine_wander_time_l427_42750


namespace NUMINAMATH_GPT_q_is_false_l427_42713

variable {p q : Prop}

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_GPT_q_is_false_l427_42713


namespace NUMINAMATH_GPT_range_of_k_l427_42764

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 ≤ k * x^2 + k * x + 3) :
  0 ≤ k ∧ k ≤ 12 :=
sorry

end NUMINAMATH_GPT_range_of_k_l427_42764


namespace NUMINAMATH_GPT_simplify_expression_l427_42739

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_simplify_expression_l427_42739


namespace NUMINAMATH_GPT_average_speed_l427_42736

theorem average_speed (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 85) (h₂ : s₂ = 45) (h₃ : s₃ = 60) (h₄ : s₄ = 75) (h₅ : s₅ = 50) : 
  (s₁ + s₂ + s₃ + s₄ + s₅) / 5 = 63 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_l427_42736


namespace NUMINAMATH_GPT_least_sum_of_variables_l427_42700

theorem least_sum_of_variables (x y z w : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)  
  (h : 2 * x^2 = 5 * y^3 ∧ 5 * y^3 = 8 * z^4 ∧ 8 * z^4 = 3 * w) : x + y + z + w = 54 := 
sorry

end NUMINAMATH_GPT_least_sum_of_variables_l427_42700


namespace NUMINAMATH_GPT_tan_11pi_over_6_l427_42749

theorem tan_11pi_over_6 : Real.tan (11 * Real.pi / 6) = - (Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_11pi_over_6_l427_42749


namespace NUMINAMATH_GPT_this_week_usage_less_next_week_usage_less_l427_42763

def last_week_usage : ℕ := 91

def usage_this_week : ℕ := (4 * 8) + (3 * 10)

def usage_next_week : ℕ := (5 * 5) + (2 * 12)

theorem this_week_usage_less : last_week_usage - usage_this_week = 29 := by
  -- proof goes here
  sorry

theorem next_week_usage_less : last_week_usage - usage_next_week = 42 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_this_week_usage_less_next_week_usage_less_l427_42763


namespace NUMINAMATH_GPT_bike_cost_l427_42789

theorem bike_cost (days_in_two_weeks : ℕ) 
  (bracelets_per_day : ℕ)
  (price_per_bracelet : ℕ)
  (total_bracelets : ℕ)
  (total_money : ℕ) 
  (h1 : days_in_two_weeks = 2 * 7)
  (h2 : bracelets_per_day = 8)
  (h3 : price_per_bracelet = 1)
  (h4 : total_bracelets = days_in_two_weeks * bracelets_per_day)
  (h5 : total_money = total_bracelets * price_per_bracelet) :
  total_money = 112 :=
sorry

end NUMINAMATH_GPT_bike_cost_l427_42789


namespace NUMINAMATH_GPT_car_time_interval_l427_42718

-- Define the conditions
def road_length := 3 -- in miles
def total_time := 10 -- in hours
def number_of_cars := 30

-- Define the conversion factor and the problem to prove
def hours_to_minutes (hours: ℕ) : ℕ := hours * 60
def time_interval_per_car (total_time_minutes: ℕ) (number_of_cars: ℕ) : ℕ := total_time_minutes / number_of_cars

-- The Lean 4 statement for the proof problem
theorem car_time_interval :
  time_interval_per_car (hours_to_minutes total_time) number_of_cars = 20 :=
by
  sorry

end NUMINAMATH_GPT_car_time_interval_l427_42718


namespace NUMINAMATH_GPT_material_for_one_pillowcase_l427_42761

def material_in_first_bale (x : ℝ) : Prop :=
  4 * x + 1100 = 5000

def material_in_third_bale : ℝ := 0.22 * 5000

def total_material_used_for_producing_items (x y : ℝ) : Prop :=
  150 * (y + 3.25) + 240 * y = x

theorem material_for_one_pillowcase :
  ∀ (x y : ℝ), 
    material_in_first_bale x → 
    material_in_third_bale = 1100 → 
    (x = 975) → 
    total_material_used_for_producing_items x y →
    y = 1.25 :=
by
  intro x y h1 h2 h3 h4
  rw [h3] at h4
  have : 150 * (y + 3.25) + 240 * y = 975 := h4
  sorry

end NUMINAMATH_GPT_material_for_one_pillowcase_l427_42761


namespace NUMINAMATH_GPT_total_price_purchase_l427_42793

variable (S T : ℝ)

theorem total_price_purchase (h1 : 2 * S + T = 2600) (h2 : 900 = 1200 * 0.75) : 2600 + 900 = 3500 := by
  sorry

end NUMINAMATH_GPT_total_price_purchase_l427_42793


namespace NUMINAMATH_GPT_median_length_range_l427_42719

/-- Define the structure of the triangle -/
structure Triangle :=
  (A B C : ℝ) -- vertices of the triangle
  (AD AE AF : ℝ) -- lengths of altitude, angle bisector, and median
  (angleA : AngleType) -- type of angle A (acute, orthogonal, obtuse)

-- Define the angle type as a custom type
inductive AngleType
| acute
| orthogonal
| obtuse

def m_range (t : Triangle) : Set ℝ :=
  match t.angleA with
  | AngleType.acute => {m : ℝ | 13 < m ∧ m < (2028 / 119)}
  | AngleType.orthogonal => {m : ℝ | m = (2028 / 119)}
  | AngleType.obtuse => {m : ℝ | (2028 / 119) < m}

-- Lean statement for proving the problem
theorem median_length_range (t : Triangle)
  (hAD : t.AD = 12)
  (hAE : t.AE = 13) : t.AF ∈ m_range t :=
by
  sorry

end NUMINAMATH_GPT_median_length_range_l427_42719


namespace NUMINAMATH_GPT_min_sum_of_dimensions_l427_42740

theorem min_sum_of_dimensions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 3003) :
  a + b + c = 45 := sorry

end NUMINAMATH_GPT_min_sum_of_dimensions_l427_42740


namespace NUMINAMATH_GPT_average_speed_l427_42711

   theorem average_speed (x : ℝ) : 
     let s1 := 40
     let s2 := 20
     let d1 := x
     let d2 := 2 * x
     let total_distance := d1 + d2
     let time1 := d1 / s1
     let time2 := d2 / s2
     let total_time := time1 + time2
     total_distance / total_time = 24 :=
   by
     sorry
   
end NUMINAMATH_GPT_average_speed_l427_42711


namespace NUMINAMATH_GPT_person_time_to_walk_without_walkway_l427_42733

def time_to_walk_without_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_against_walkway : ℝ) 
  (correct_time : ℝ) : Prop :=
  ∃ (vp vw : ℝ), 
    ((vp + vw) * time_with_walkway = walkway_length) ∧ 
    ((vp - vw) * time_against_walkway = walkway_length) ∧ 
     correct_time = walkway_length / vp

theorem person_time_to_walk_without_walkway : 
  time_to_walk_without_walkway 120 40 160 64 :=
sorry

end NUMINAMATH_GPT_person_time_to_walk_without_walkway_l427_42733


namespace NUMINAMATH_GPT_savings_same_l427_42734

theorem savings_same (A_salary B_salary total_salary : ℝ)
  (A_spend_perc B_spend_perc : ℝ)
  (h_total : A_salary + B_salary = total_salary)
  (h_A_salary : A_salary = 4500)
  (h_A_spend_perc : A_spend_perc = 0.95)
  (h_B_spend_perc : B_spend_perc = 0.85)
  (h_total_salary : total_salary = 6000) :
  ((1 - A_spend_perc) * A_salary) = ((1 - B_spend_perc) * B_salary) :=
by
  sorry

end NUMINAMATH_GPT_savings_same_l427_42734


namespace NUMINAMATH_GPT_remainder_when_divided_by_eleven_l427_42775

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_eleven_l427_42775


namespace NUMINAMATH_GPT_extrema_range_of_m_l427_42766

def has_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, (∀ z : ℝ, z ≤ x → f z ≤ f x) ∧ (∀ z : ℝ, z ≥ y → f z ≤ f y)

noncomputable def f (m x : ℝ) : ℝ :=
  x^3 + m * x^2 + (m + 6) * x + 1

theorem extrema_range_of_m (m : ℝ) :
  has_extrema (f m) ↔ (m ∈ Set.Iic (-3) ∪ Set.Ici 6) :=
by
  sorry

end NUMINAMATH_GPT_extrema_range_of_m_l427_42766


namespace NUMINAMATH_GPT_b11_eq_4_l427_42790

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d r : ℤ} {a1 : ℤ}

-- Define non-zero arithmetic sequence {a_n} with common difference d
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define geometric sequence {b_n} with common ratio r
def is_geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n * r

-- The given conditions
axiom a1_minus_a7_sq_plus_a13_eq_zero : a 1 - (a 7) ^ 2 + a 13 = 0
axiom b7_eq_a7 : b 7 = a 7

-- The problem statement to prove: b 11 = 4
theorem b11_eq_4
  (arith_seq : is_arithmetic_sequence a d)
  (geom_seq : is_geometric_sequence b r)
  (a1_non_zero : a1 ≠ 0) :
  b 11 = 4 :=
sorry

end NUMINAMATH_GPT_b11_eq_4_l427_42790


namespace NUMINAMATH_GPT_acute_triangle_properties_l427_42708

theorem acute_triangle_properties (A B C : ℝ) (AC BC : ℝ)
  (h_acute : ∀ {x : ℝ}, x = A ∨ x = B ∨ x = C → x < π / 2)
  (h_BC : BC = 1)
  (h_B_eq_2A : B = 2 * A) :
  (AC / Real.cos A = 2) ∧ (Real.sqrt 2 < AC ∧ AC < Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_properties_l427_42708


namespace NUMINAMATH_GPT_dog_food_weight_l427_42786

/-- 
 Mike has 2 dogs, each dog eats 6 cups of dog food twice a day.
 Mike buys 9 bags of 20-pound dog food a month.
 Prove that a cup of dog food weighs 0.25 pounds.
-/
theorem dog_food_weight :
  let dogs := 2
  let cups_per_meal := 6
  let meals_per_day := 2
  let bags_per_month := 9
  let weight_per_bag := 20
  let days_per_month := 30
  let total_cups_per_day := cups_per_meal * meals_per_day * dogs
  let total_cups_per_month := total_cups_per_day * days_per_month
  let total_weight_per_month := bags_per_month * weight_per_bag
  (total_weight_per_month / total_cups_per_month : ℝ) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_dog_food_weight_l427_42786


namespace NUMINAMATH_GPT_divisibility_condition_l427_42781

theorem divisibility_condition
  (a p q : ℕ) (hpq : p ≤ q) (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) :
  (p ∣ a^p ∨ p ∣ a^q) → (p ∣ a^p ∧ p ∣ a^q) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_condition_l427_42781


namespace NUMINAMATH_GPT_find_wrong_number_read_l427_42716

theorem find_wrong_number_read (avg_initial avg_correct num_total wrong_num : ℕ) 
    (h1 : avg_initial = 15)
    (h2 : avg_correct = 16)
    (h3 : num_total = 10)
    (h4 : wrong_num = 36) 
    : wrong_num - (avg_correct * num_total - avg_initial * num_total) = 26 := 
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_find_wrong_number_read_l427_42716


namespace NUMINAMATH_GPT_calories_in_300g_lemonade_proof_l427_42782

def g_lemon := 150
def g_sugar := 200
def g_water := 450

def c_lemon_per_100g := 30
def c_sugar_per_100g := 400
def c_water := 0

def total_calories :=
  g_lemon * c_lemon_per_100g / 100 +
  g_sugar * c_sugar_per_100g / 100 +
  g_water * c_water

def total_weight := g_lemon + g_sugar + g_water

def caloric_density := total_calories / total_weight

def calories_in_300g_lemonade := 300 * caloric_density

theorem calories_in_300g_lemonade_proof : calories_in_300g_lemonade = 317 := by
  sorry

end NUMINAMATH_GPT_calories_in_300g_lemonade_proof_l427_42782


namespace NUMINAMATH_GPT_maximize_profit_price_l427_42756

-- Definitions from the conditions
def initial_price : ℝ := 80
def initial_sales : ℝ := 200
def price_reduction_per_unit : ℝ := 1
def sales_increase_per_unit : ℝ := 20
def cost_price_per_helmet : ℝ := 50

-- Profit function
def profit (x : ℝ) : ℝ :=
  (x - cost_price_per_helmet) * (initial_sales + (initial_price - x) * sales_increase_per_unit)

-- The theorem statement
theorem maximize_profit_price : 
  ∃ x, (x = 70) ∧ (∀ y, profit y ≤ profit x) :=
sorry

end NUMINAMATH_GPT_maximize_profit_price_l427_42756


namespace NUMINAMATH_GPT_y_squared_plus_three_y_is_perfect_square_l427_42783

theorem y_squared_plus_three_y_is_perfect_square (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := 
by
  sorry

end NUMINAMATH_GPT_y_squared_plus_three_y_is_perfect_square_l427_42783


namespace NUMINAMATH_GPT_complex_expression_equals_zero_l427_42707

def i : ℂ := Complex.I

theorem complex_expression_equals_zero : 2 * i^5 + (1 - i)^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_complex_expression_equals_zero_l427_42707


namespace NUMINAMATH_GPT_simplify_expression_l427_42709

def E (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2)

theorem simplify_expression (x : ℝ) : E x = 9 * x^3 - 2 * x^2 + 9 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l427_42709


namespace NUMINAMATH_GPT_solve_for_y_l427_42748

theorem solve_for_y (y : ℝ) (h : (4/7) * (1/5) * y - 2 = 14) : y = 140 := 
sorry

end NUMINAMATH_GPT_solve_for_y_l427_42748


namespace NUMINAMATH_GPT_football_banquet_total_food_l427_42704

-- Definitions representing the conditions
def individual_max_food (n : Nat) := n ≤ 2
def min_guests (g : Nat) := g ≥ 160

-- The proof problem statement
theorem football_banquet_total_food : 
  ∀ (n g : Nat), (∀ i, i ≤ g → individual_max_food n) ∧ min_guests g → g * n = 320 := 
by
  intros n g h
  sorry

end NUMINAMATH_GPT_football_banquet_total_food_l427_42704


namespace NUMINAMATH_GPT_benny_picked_proof_l427_42712

-- Define the number of apples Dan picked
def dan_picked: ℕ := 9

-- Define the total number of apples picked
def total_apples: ℕ := 11

-- Define the number of apples Benny picked
def benny_picked (dan_picked total_apples: ℕ): ℕ :=
  total_apples - dan_picked

-- The theorem we need to prove
theorem benny_picked_proof: benny_picked dan_picked total_apples = 2 :=
by
  -- We calculate the number of apples Benny picked
  sorry

end NUMINAMATH_GPT_benny_picked_proof_l427_42712


namespace NUMINAMATH_GPT_fill_missing_digits_l427_42778

noncomputable def first_number (a : ℕ) : ℕ := a * 1000 + 2 * 100 + 5 * 10 + 7
noncomputable def second_number (b c : ℕ) : ℕ := 2 * 1000 + b * 100 + 9 * 10 + c

theorem fill_missing_digits (a b c : ℕ) : a = 1 ∧ b = 5 ∧ c = 6 → first_number a + second_number b c = 5842 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fill_missing_digits_l427_42778


namespace NUMINAMATH_GPT_quadratic_product_fact_l427_42776

def quadratic_factors_product : Prop :=
  let integer_pairs := [(-1, 24), (-2, 12), (-3, 8), (-4, 6), (-6, 4), (-8, 3), (-12, 2), (-24, 1)]
  let t_values := integer_pairs.map (fun (c, d) => c + d)
  let product_t := t_values.foldl (fun acc t => acc * t) 1
  product_t = -5290000

theorem quadratic_product_fact : quadratic_factors_product :=
by sorry

end NUMINAMATH_GPT_quadratic_product_fact_l427_42776


namespace NUMINAMATH_GPT_rectangle_division_impossible_l427_42755

theorem rectangle_division_impossible :
  ¬ ∃ n m : ℕ, n * 5 = 55 ∧ m * 11 = 39 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_division_impossible_l427_42755


namespace NUMINAMATH_GPT_tomatoes_cheaper_than_cucumbers_percentage_l427_42732

noncomputable def P_c := 5
noncomputable def two_T_three_P_c := 23
noncomputable def T := (two_T_three_P_c - 3 * P_c) / 2
noncomputable def percentage_by_which_tomatoes_cheaper_than_cucumbers := ((P_c - T) / P_c) * 100

theorem tomatoes_cheaper_than_cucumbers_percentage : 
  P_c = 5 → 
  (2 * T + 3 * P_c = 23) →
  T < P_c →
  percentage_by_which_tomatoes_cheaper_than_cucumbers = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tomatoes_cheaper_than_cucumbers_percentage_l427_42732


namespace NUMINAMATH_GPT_expression_value_l427_42746

/-- The value of the expression 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) is 1200. -/
theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l427_42746


namespace NUMINAMATH_GPT_complement_of_alpha_l427_42742

-- Define that the angle α is given as 44 degrees 36 minutes
def alpha : ℚ := 44 + 36 / 60  -- using rational numbers to represent the degrees and minutes

-- Define the complement function
def complement (angle : ℚ) : ℚ := 90 - angle

-- State the proposition to prove
theorem complement_of_alpha : complement alpha = 45 + 24 / 60 := 
by
  sorry

end NUMINAMATH_GPT_complement_of_alpha_l427_42742


namespace NUMINAMATH_GPT_points_on_circle_l427_42753

theorem points_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  x^2 + y^2 = 1 := 
by 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  sorry

end NUMINAMATH_GPT_points_on_circle_l427_42753


namespace NUMINAMATH_GPT_distance_between_planes_l427_42723

open Real

def plane1 (x y z : ℝ) : Prop := 3 * x - y + 2 * z - 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x - 2 * y + 4 * z + 4 = 0

theorem distance_between_planes :
  ∀ (x y z : ℝ), plane1 x y z →
  6 * x - 2 * y + 4 * z + 4 ≠ 0 →
  (∃ d : ℝ, d = abs (6 * x - 2 * y + 4 * z + 4) / sqrt (6^2 + (-2)^2 + 4^2) ∧ d = 5 * sqrt 14 / 14) :=
by
  intros x y z p1 p2
  sorry

end NUMINAMATH_GPT_distance_between_planes_l427_42723


namespace NUMINAMATH_GPT_ratio_B_C_l427_42758

def total_money := 595
def A_share := 420
def B_share := 105
def C_share := 70

-- The main theorem stating the expected ratio
theorem ratio_B_C : (B_share / C_share : ℚ) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_B_C_l427_42758


namespace NUMINAMATH_GPT_intersecting_parabolas_circle_radius_sq_l427_42726

theorem intersecting_parabolas_circle_radius_sq:
  (∀ (x y : ℝ), (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) → 
  ((x + 1/2)^2 + (y - 7/2)^2 = 13/2)) := sorry

end NUMINAMATH_GPT_intersecting_parabolas_circle_radius_sq_l427_42726


namespace NUMINAMATH_GPT_no_attention_prob_l427_42747

noncomputable def prob_no_attention (p1 p2 p3 : ℝ) : ℝ :=
  (1 - p1) * (1 - p2) * (1 - p3)

theorem no_attention_prob :
  let p1 := 0.9
  let p2 := 0.8
  let p3 := 0.6
  prob_no_attention p1 p2 p3 = 0.008 :=
by
  unfold prob_no_attention
  sorry

end NUMINAMATH_GPT_no_attention_prob_l427_42747


namespace NUMINAMATH_GPT_count_integers_l427_42799

def satisfies_conditions (n : ℤ) (r : ℤ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5

theorem count_integers (n : ℤ) (r : ℤ) :
  (satisfies_conditions n r) → ∃! n, 200 < n ∧ n < 300 ∧ ∃ r, n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_l427_42799


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l427_42767

theorem geometric_sequence_common_ratio (a1 a2 a3 a4 : ℝ)
  (h₁ : a1 = 32) (h₂ : a2 = -48) (h₃ : a3 = 72) (h₄ : a4 = -108)
  (h_geom : ∃ r, a2 = r * a1 ∧ a3 = r * a2 ∧ a4 = r * a3) :
  ∃ r, r = -3/2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l427_42767


namespace NUMINAMATH_GPT_product_divisible_by_8_probability_l427_42796

noncomputable def probability_product_divisible_by_8 (dice_rolls : Fin 6 → Fin 8) : ℚ :=
  -- Function to calculate the probability that the product of numbers is divisible by 8
  sorry

theorem product_divisible_by_8_probability :
  ∀ (dice_rolls : Fin 6 → Fin 8),
  probability_product_divisible_by_8 dice_rolls = 177 / 256 :=
sorry

end NUMINAMATH_GPT_product_divisible_by_8_probability_l427_42796


namespace NUMINAMATH_GPT_decreasing_population_density_l427_42729

def Population (t : Type) : Type := t

variable (stable_period: Prop)
variable (infertility: Prop)
variable (death_rate_exceeds_birth_rate: Prop)
variable (complex_structure: Prop)

theorem decreasing_population_density :
  death_rate_exceeds_birth_rate → true := sorry

end NUMINAMATH_GPT_decreasing_population_density_l427_42729


namespace NUMINAMATH_GPT_intersection_eq_l427_42728

def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem intersection_eq : A ∩ B = {x | -1 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l427_42728


namespace NUMINAMATH_GPT_area_of_given_triangle_is_8_l427_42730

-- Define the vertices of the triangle
def x1 := 2
def y1 := -3
def x2 := -1
def y2 := 6
def x3 := 4
def y3 := -5

-- Define the determinant formula for the area of the triangle
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℤ) : ℤ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem area_of_given_triangle_is_8 :
  area_of_triangle x1 y1 x2 y2 x3 y3 = 8 := by
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_is_8_l427_42730


namespace NUMINAMATH_GPT_value_of_t_plus_one_over_t_l427_42735

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_t_plus_one_over_t_l427_42735


namespace NUMINAMATH_GPT_range_of_g_l427_42784

noncomputable def g (a x : ℝ) : ℝ :=
  a * (Real.cos x)^4 - 2 * (Real.sin x) * (Real.cos x) + (Real.sin x)^4

theorem range_of_g (a : ℝ) (h : a > 0) :
  Set.range (g a) = Set.Icc (a - (3 - a) / 2) (a + (a + 1) / 2) :=
sorry

end NUMINAMATH_GPT_range_of_g_l427_42784


namespace NUMINAMATH_GPT_sequence_tenth_term_l427_42772

theorem sequence_tenth_term :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n / (1 + a n)) ∧ a 10 = 1 / 10 :=
sorry

end NUMINAMATH_GPT_sequence_tenth_term_l427_42772


namespace NUMINAMATH_GPT_number_of_monomials_l427_42773

-- Define the degree of a monomial
def degree (x_deg y_deg z_deg : ℕ) : ℕ := x_deg + y_deg + z_deg

-- Define a condition for the coefficient of the monomial
def monomial_coefficient (coeff : ℤ) : Prop := coeff = -3

-- Define a condition for the presence of the variables x, y, z
def contains_vars (x_deg y_deg z_deg : ℕ) : Prop := x_deg ≥ 1 ∧ y_deg ≥ 1 ∧ z_deg ≥ 1

-- Define the proof for the number of such monomials
theorem number_of_monomials :
  ∃ (x_deg y_deg z_deg : ℕ), contains_vars x_deg y_deg z_deg ∧ monomial_coefficient (-3) ∧ degree x_deg y_deg z_deg = 5 ∧ (6 = 6) :=
by
  sorry

end NUMINAMATH_GPT_number_of_monomials_l427_42773


namespace NUMINAMATH_GPT_positive_X_solution_l427_42759

def boxtimes (X Y : ℤ) : ℤ := X^2 - 2 * X + Y^2

theorem positive_X_solution (X : ℤ) (h : boxtimes X 7 = 164) : X = 13 :=
by
  sorry

end NUMINAMATH_GPT_positive_X_solution_l427_42759


namespace NUMINAMATH_GPT_smallest_even_n_sum_eq_l427_42722
  
theorem smallest_even_n_sum_eq (n : ℕ) (h_pos : n > 0) (h_even : n % 2 = 0) :
  n = 12 ↔ 
  let s₁ := n / 2 * (2 * 5 + (n - 1) * 6)
  let s₂ := n / 2 * (2 * 13 + (n - 1) * 3)
  s₁ = s₂ :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_n_sum_eq_l427_42722


namespace NUMINAMATH_GPT_distance_to_focus_F2_l427_42715

noncomputable def ellipse_foci_distance
  (x y : ℝ)
  (a b : ℝ) 
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1) 
  (a2 : a^2 = 9) 
  (b2 : b^2 = 2) 
  (F1 P : ℝ) 
  (h_P_on_ellipse : F1 = 3) 
  (h_PF1 : F1 = 4) 
: ℝ :=
  2

-- theorem to prove the problem statement
theorem distance_to_focus_F2
  (x y : ℝ)
  (a b : ℝ)
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1)
  (a2 : a^2 = 9)
  (b2 : b^2 = 2)
  (F1 P : ℝ)
  (h_P_on_ellipse : F1 = 3)
  (h_PF1 : F1 = 4)
: F2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_focus_F2_l427_42715


namespace NUMINAMATH_GPT_brian_video_watching_time_l427_42779

/--
Brian watches a 4-minute video of cats.
Then he watches a video twice as long as the cat video involving dogs.
Finally, he watches a video on gorillas that's twice as long as the combined duration of the first two videos.
Prove that Brian spends a total of 36 minutes watching animal videos.
-/
theorem brian_video_watching_time (cat_video dog_video gorilla_video : ℕ) 
  (h₁ : cat_video = 4) 
  (h₂ : dog_video = 2 * cat_video) 
  (h₃ : gorilla_video = 2 * (cat_video + dog_video)) : 
  cat_video + dog_video + gorilla_video = 36 := by
  sorry

end NUMINAMATH_GPT_brian_video_watching_time_l427_42779


namespace NUMINAMATH_GPT_meter_to_leap_l427_42702

theorem meter_to_leap
  (strides leaps bounds meters : ℝ)
  (h1 : 3 * strides = 4 * leaps)
  (h2 : 5 * bounds = 7 * strides)
  (h3 : 2 * bounds = 9 * meters) :
  1 * meters = (56 / 135) * leaps :=
by
  sorry

end NUMINAMATH_GPT_meter_to_leap_l427_42702


namespace NUMINAMATH_GPT_triangle_inequality_third_side_l427_42770

theorem triangle_inequality_third_side (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : 0 < x) (h₄ : x < a + b) (h₅ : a < b + x) (h₆ : b < a + x) :
  ¬(x = 9) := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_third_side_l427_42770


namespace NUMINAMATH_GPT_ratio_of_length_to_height_l427_42795

theorem ratio_of_length_to_height
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (vol_eq : 129024 = w * h * l)
  (w_eq : w = 8) :
  l / h = 7 := 
sorry

end NUMINAMATH_GPT_ratio_of_length_to_height_l427_42795


namespace NUMINAMATH_GPT_quadratic_real_roots_l427_42721

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m = 0) ↔ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l427_42721


namespace NUMINAMATH_GPT_max_product_sum_1976_l427_42727

theorem max_product_sum_1976 (a : ℕ) (P : ℕ → ℕ) (h : ∀ n, P n > 0 → a = 1976) :
  ∃ (k l : ℕ), (2 * k + 3 * l = 1976) ∧ (P 1976 = 2 * 3 ^ 658) := sorry

end NUMINAMATH_GPT_max_product_sum_1976_l427_42727


namespace NUMINAMATH_GPT_inequality_solution_l427_42785

theorem inequality_solution :
  {x : ℝ | (x - 3) * (x + 2) ≠ 0 ∧ (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0} = 
  {x : ℝ | x ≤ -2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l427_42785


namespace NUMINAMATH_GPT_value_of_a_plus_b_l427_42710

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l427_42710


namespace NUMINAMATH_GPT_max_a9_l427_42745

theorem max_a9 (a : Fin 18 → ℕ) (h_pos: ∀ i, 1 ≤ a i) (h_incr: ∀ i j, i < j → a i < a j) (h_sum: (Finset.univ : Finset (Fin 18)).sum a = 2001) : a 8 ≤ 192 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_a9_l427_42745


namespace NUMINAMATH_GPT_not_all_same_probability_l427_42701

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_not_all_same_probability_l427_42701


namespace NUMINAMATH_GPT_area_ADC_proof_l427_42706

-- Definitions for the given conditions and question
variables (BD DC : ℝ) (ABD_area ADC_area : ℝ)

-- Conditions
def ratio_condition := BD / DC = 3 / 2
def ABD_area_condition := ABD_area = 30

-- Question rewritten as proof problem
theorem area_ADC_proof (h1 : ratio_condition BD DC) (h2 : ABD_area_condition ABD_area) :
  ADC_area = 20 :=
sorry

end NUMINAMATH_GPT_area_ADC_proof_l427_42706


namespace NUMINAMATH_GPT_equiangular_polygon_angle_solution_l427_42725

-- Given two equiangular polygons P_1 and P_2 with different numbers of sides
-- Each angle of P_1 is x degrees
-- Each angle of P_2 is k * x degrees where k is an integer greater than 1
-- Prove that the number of valid pairs (x, k) is exactly 1

theorem equiangular_polygon_angle_solution : ∃ x k : ℕ, ( ∀ n m : ℕ, x = 180 - 360 / n ∧ k * x = 180 - 360 / m → (k > 1) → x = 60 ∧ k = 2) := sorry

end NUMINAMATH_GPT_equiangular_polygon_angle_solution_l427_42725


namespace NUMINAMATH_GPT_combined_average_score_l427_42757

theorem combined_average_score (M E : ℕ) (m e : ℕ) (h1 : M = 82) (h2 : E = 68) (h3 : m = 5 * e / 7) :
  ((m * M) + (e * E)) / (m + e) = 72 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_combined_average_score_l427_42757


namespace NUMINAMATH_GPT_simplify_expression_l427_42714

variable (x : ℝ)

theorem simplify_expression :
  (2 * x + 25) + (150 * x + 35) + (50 * x + 10) = 202 * x + 70 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l427_42714


namespace NUMINAMATH_GPT_ratio_n_over_p_l427_42720

-- Definitions and conditions from the problem
variables {m n p : ℝ}

-- The quadratic equation x^2 + mx + n = 0 has roots that are thrice those of x^2 + px + m = 0.
-- None of m, n, and p is zero.

-- Prove that n / p = 27 given these conditions.
theorem ratio_n_over_p (hmn0 : m ≠ 0) (hn : n = 9 * m) (hp : p = m / 3):
  n / p = 27 :=
  by
    sorry -- Formal proof will go here.

end NUMINAMATH_GPT_ratio_n_over_p_l427_42720


namespace NUMINAMATH_GPT_find_4_digit_number_l427_42705

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_4_digit_number_l427_42705


namespace NUMINAMATH_GPT_total_difference_is_correct_l427_42771

-- Define the harvest rates
def valencia_weekday_ripe := 90
def valencia_weekday_unripe := 38
def navel_weekday_ripe := 125
def navel_weekday_unripe := 65
def blood_weekday_ripe := 60
def blood_weekday_unripe := 42

def valencia_weekend_ripe := 75
def valencia_weekend_unripe := 33
def navel_weekend_ripe := 100
def navel_weekend_unripe := 57
def blood_weekend_ripe := 45
def blood_weekend_unripe := 36

-- Define the number of weekdays and weekend days
def weekdays := 5
def weekend_days := 2

-- Calculate the total harvests
def total_valencia_ripe := valencia_weekday_ripe * weekdays + valencia_weekend_ripe * weekend_days
def total_valencia_unripe := valencia_weekday_unripe * weekdays + valencia_weekend_unripe * weekend_days
def total_navel_ripe := navel_weekday_ripe * weekdays + navel_weekend_ripe * weekend_days
def total_navel_unripe := navel_weekday_unripe * weekdays + navel_weekend_unripe * weekend_days
def total_blood_ripe := blood_weekday_ripe * weekdays + blood_weekend_ripe * weekend_days
def total_blood_unripe := blood_weekday_unripe * weekdays + blood_weekend_unripe * weekend_days

-- Calculate the total differences
def valencia_difference := total_valencia_ripe - total_valencia_unripe
def navel_difference := total_navel_ripe - total_navel_unripe
def blood_difference := total_blood_ripe - total_blood_unripe

-- Define the total difference
def total_difference := valencia_difference + navel_difference + blood_difference

-- Theorem statement
theorem total_difference_is_correct :
  total_difference = 838 := by
  sorry

end NUMINAMATH_GPT_total_difference_is_correct_l427_42771


namespace NUMINAMATH_GPT_excess_calories_l427_42741

theorem excess_calories (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ)
  (run_minutes : ℕ) (calories_per_minute : ℕ)
  (h_bags : bags = 3) (h_ounces_per_bag : ounces_per_bag = 2)
  (h_calories_per_ounce : calories_per_ounce = 150)
  (h_run_minutes : run_minutes = 40)
  (h_calories_per_minute : calories_per_minute = 12) :
  (bags * ounces_per_bag * calories_per_ounce) - (run_minutes * calories_per_minute) = 420 := by
  sorry

end NUMINAMATH_GPT_excess_calories_l427_42741


namespace NUMINAMATH_GPT_find_matches_in_second_set_l427_42792

-- Conditions defined as Lean variables
variables (x : ℕ)
variables (avg_first_20 : ℚ := 40)
variables (avg_second_x : ℚ := 20)
variables (avg_all_30 : ℚ := 100 / 3)
variables (total_first_20 : ℚ := 20 * avg_first_20)
variables (total_all_30 : ℚ := 30 * avg_all_30)

-- Proof statement (question) along with conditions
theorem find_matches_in_second_set (x_value : x = 10) :
  avg_first_20 = 40 ∧ avg_second_x = 20 ∧ avg_all_30 = 100 / 3 →
  20 * avg_first_20 + x * avg_second_x = 30 * avg_all_30 → x = 10 := 
sorry

end NUMINAMATH_GPT_find_matches_in_second_set_l427_42792


namespace NUMINAMATH_GPT_time_lent_to_C_eq_l427_42731

variable (principal_B : ℝ := 5000)
variable (time_B : ℕ := 2)
variable (principal_C : ℝ := 3000)
variable (total_interest : ℝ := 1980)
variable (rate_of_interest_per_annum : ℝ := 0.09)

theorem time_lent_to_C_eq (n : ℝ) (H : principal_B * rate_of_interest_per_annum * time_B + principal_C * rate_of_interest_per_annum * n = total_interest) : 
  n = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_time_lent_to_C_eq_l427_42731


namespace NUMINAMATH_GPT_rectangle_ratio_l427_42717

theorem rectangle_ratio (a b : ℝ) (side : ℝ) (M N : ℝ → ℝ) (P Q : ℝ → ℝ)
  (h_side : side = 4)
  (h_M : M 0 = 4 / 3 ∧ M 4 = 8 / 3)
  (h_N : N 0 = 4 / 3 ∧ N 4 = 8 / 3)
  (h_perpendicular : P 0 = Q 0 ∧ P 4 = Q 4)
  (h_area : side * side = 16) :
  let UV := 6 / 5
  let VW := 40 / 3
  UV / VW = 9 / 100 :=
sorry

end NUMINAMATH_GPT_rectangle_ratio_l427_42717


namespace NUMINAMATH_GPT_face_value_is_100_l427_42754

-- Definitions based on conditions
def faceValue (F : ℝ) : Prop :=
  let discountedPrice := 0.92 * F
  let brokerageFee := 0.002 * discountedPrice
  let totalCostPrice := discountedPrice + brokerageFee
  totalCostPrice = 92.2

-- The proof statement in Lean
theorem face_value_is_100 : ∃ F : ℝ, faceValue F ∧ F = 100 :=
by
  use 100
  unfold faceValue
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_face_value_is_100_l427_42754


namespace NUMINAMATH_GPT_remaining_customers_is_13_l427_42744

-- Given conditions
def initial_customers : ℕ := 36
def half_left_customers : ℕ := initial_customers / 2  -- 50% of customers leaving
def remaining_customers_after_half_left : ℕ := initial_customers - half_left_customers

def thirty_percent_of_remaining : ℚ := remaining_customers_after_half_left * 0.30 
def thirty_percent_of_remaining_rounded : ℕ := thirty_percent_of_remaining.floor.toNat  -- rounding down

def final_remaining_customers : ℕ := remaining_customers_after_half_left - thirty_percent_of_remaining_rounded

-- Proof statement without proof
theorem remaining_customers_is_13 : final_remaining_customers = 13 := by
  sorry

end NUMINAMATH_GPT_remaining_customers_is_13_l427_42744


namespace NUMINAMATH_GPT_sin_x_sin_y_eq_sin_beta_sin_gamma_l427_42752

theorem sin_x_sin_y_eq_sin_beta_sin_gamma
  (A B C M : Type)
  (AM BM CM : ℝ)
  (alpha beta gamma x y : ℝ)
  (h1 : AM * AM = BM * CM)
  (h2 : BM ≠ 0)
  (h3 : CM ≠ 0)
  (hx : AM / BM = Real.sin beta / Real.sin x)
  (hy : AM / CM = Real.sin gamma / Real.sin y) :
  Real.sin x * Real.sin y = Real.sin beta * Real.sin gamma := 
sorry

end NUMINAMATH_GPT_sin_x_sin_y_eq_sin_beta_sin_gamma_l427_42752


namespace NUMINAMATH_GPT_puja_runs_distance_in_meters_l427_42737

noncomputable def puja_distance (time_in_seconds : ℝ) (speed_kmph : ℝ) : ℝ :=
  let time_in_hours := time_in_seconds / 3600
  let distance_km := speed_kmph * time_in_hours
  distance_km * 1000

theorem puja_runs_distance_in_meters :
  abs (puja_distance 59.995200383969284 30 - 499.96) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_puja_runs_distance_in_meters_l427_42737


namespace NUMINAMATH_GPT_chord_length_l427_42788

-- Define the key components.
structure Circle := 
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the initial conditions.
def circle1 : Circle := { center := (0, 0), radius := 5 }
def circle2 : Circle := { center := (2, 0), radius := 3 }

-- Define the chord and tangency condition.
def touches_internally (C1 C2 : Circle) : Prop :=
  C1.radius > C2.radius ∧ dist C1.center C2.center = C1.radius - C2.radius

def chord_divided_ratio (AB_length : ℝ) (r1 r2 : ℝ) : Prop :=
  ∃ (x : ℝ), AB_length = 4 * x ∧ r1 = x ∧ r2 = 3 * x

-- The theorem to prove the length of the chord AB.
theorem chord_length (h1 : touches_internally circle1 circle2)
                     (h2 : chord_divided_ratio 8 2 (6)) : ∃ (AB_length : ℝ), AB_length = 8 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l427_42788


namespace NUMINAMATH_GPT_ratio_of_autobiographies_to_fiction_l427_42751

theorem ratio_of_autobiographies_to_fiction (total_books fiction_books non_fiction_books picture_books autobiographies: ℕ) 
  (h1 : total_books = 35) 
  (h2 : fiction_books = 5) 
  (h3 : non_fiction_books = fiction_books + 4) 
  (h4 : picture_books = 11) 
  (h5 : autobiographies = total_books - (fiction_books + non_fiction_books + picture_books)) :
  autobiographies / fiction_books = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_autobiographies_to_fiction_l427_42751


namespace NUMINAMATH_GPT_inequality_solution_set_l427_42703

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom deriv_cond : ∀ (x : ℝ), x ≠ 0 → f' x < (2 * f x) / x
axiom zero_points : f (-2) = 0 ∧ f 1 = 0

theorem inequality_solution_set :
  {x : ℝ | x * f x < 0} = { x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) } :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l427_42703


namespace NUMINAMATH_GPT_glove_probability_correct_l427_42798

noncomputable def glove_probability : ℚ :=
  let red_pair := ("r1", "r2") -- pair of red gloves
  let black_pair := ("b1", "b2") -- pair of black gloves
  let white_pair := ("w1", "w2") -- pair of white gloves
  let all_pairs := [
    (red_pair.1, red_pair.2), 
    (black_pair.1, black_pair.2), 
    (white_pair.1, white_pair.2),
    (red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
    (red_pair.2, black_pair.1), (red_pair.2, white_pair.1),
    (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)
  ]
  let valid_pairs := [(red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
                      (red_pair.2, black_pair.1), (red_pair.2, white_pair.1), 
                      (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)]
  (valid_pairs.length : ℚ) / (all_pairs.length : ℚ)

theorem glove_probability_correct :
  glove_probability = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_glove_probability_correct_l427_42798


namespace NUMINAMATH_GPT_trams_to_add_l427_42738

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end NUMINAMATH_GPT_trams_to_add_l427_42738
