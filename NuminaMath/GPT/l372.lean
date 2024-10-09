import Mathlib

namespace symmetric_curve_wrt_line_l372_37246

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end symmetric_curve_wrt_line_l372_37246


namespace exists_arith_prog_5_primes_exists_arith_prog_6_primes_l372_37276

-- Define the condition of being an arithmetic progression
def is_arith_prog (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get! (i + 1) - seq.get! i = seq.get! 1 - seq.get! 0

-- Define the condition of being prime
def all_prime (seq : List ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ seq → Nat.Prime n

-- The main statements
theorem exists_arith_prog_5_primes :
  ∃ (seq : List ℕ), seq.length = 5 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

theorem exists_arith_prog_6_primes :
  ∃ (seq : List ℕ), seq.length = 6 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

end exists_arith_prog_5_primes_exists_arith_prog_6_primes_l372_37276


namespace repeating_decimal_as_fraction_l372_37234

/-- Define x as the repeating decimal 7.182182... -/
def x : ℚ := 
  7 + 182 / 999

/-- Define y as the fraction 7175/999 -/
def y : ℚ := 
  7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999 -/
theorem repeating_decimal_as_fraction : x = y :=
sorry

end repeating_decimal_as_fraction_l372_37234


namespace Jamie_liquid_limit_l372_37213

theorem Jamie_liquid_limit :
  let milk_ounces := 8
  let grape_juice_ounces := 16
  let water_bottle_limit := 8
  let already_consumed := milk_ounces + grape_juice_ounces
  let max_before_bathroom := already_consumed + water_bottle_limit
  max_before_bathroom = 32 :=
by
  sorry

end Jamie_liquid_limit_l372_37213


namespace quadratic_inequality_solution_l372_37221

open Real

theorem quadratic_inequality_solution :
    ∀ x : ℝ, -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 :=
by sorry

end quadratic_inequality_solution_l372_37221


namespace y_intercept_of_line_l372_37256

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 2 * y + 7 = 0) (hx : x = 0) : y = 7 / 2 :=
by
  sorry

end y_intercept_of_line_l372_37256


namespace range_of_a_l372_37285

variable (a : ℝ)
def proposition_p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def proposition_q := ∃ x₀ : ℝ, x₀^2 - x₀ + a = 0

theorem range_of_a (h1 : proposition_p a ∨ proposition_q a)
    (h2 : ¬ (proposition_p a ∧ proposition_q a)) :
    a < 0 ∨ (1 / 4) < a ∧ a < 4 :=
  sorry

end range_of_a_l372_37285


namespace clever_question_l372_37251

-- Define the conditions as predicates
def inhabitants_truthful (city : String) : Prop := 
  city = "Mars-Polis"

def inhabitants_lying (city : String) : Prop := 
  city = "Mars-City"

def responses (question : String) (city : String) : String :=
  if question = "Are we in Mars-City?" then
    if city = "Mars-City" then "No" else "Yes"
  else if question = "Do you live here?" then
    if city = "Mars-City" then "No" else "Yes"
  else "Unknown"

-- Define the main theorem
theorem clever_question (city : String) (initial_response : String) :
  (inhabitants_truthful city ∨ inhabitants_lying city) →
  responses "Are we in Mars-City?" city = initial_response →
  responses "Do you live here?" city = "Yes" ∨ responses "Do you live here?" city = "No" :=
by
  sorry

end clever_question_l372_37251


namespace half_angle_quadrant_l372_37277

theorem half_angle_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) :
    (∃ n : ℤ, (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)) :=
sorry

end half_angle_quadrant_l372_37277


namespace only_one_true_l372_37216

def statement_dong (xi: Prop) := ¬ xi
def statement_xi (nan: Prop) := ¬ nan
def statement_nan (dong: Prop) := ¬ dong
def statement_bei (nan: Prop) := ¬ (statement_nan nan) 

-- Define the main proof problem assuming all statements
theorem only_one_true : (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → true ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → true ∧ statement_nan dong → false ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → true ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∧ (statement_nan (statement_dong xi)) = true :=
sorry

end only_one_true_l372_37216


namespace height_of_original_triangle_l372_37215

variable (a b c : ℝ)

theorem height_of_original_triangle (a b c : ℝ) : 
  ∃ h : ℝ, h = a + b + c :=
  sorry

end height_of_original_triangle_l372_37215


namespace curve_equation_l372_37259

noncomputable def curve_passing_condition (x y : ℝ) : Prop :=
  (∃ (f : ℝ → ℝ), f 2 = 3 ∧ ∀ (t : ℝ), (f t) * t = 6 ∧ ((t ≠ 0 ∧ f t ≠ 0) → (t, f t) = (x, y)))

theorem curve_equation (x y : ℝ) (h1 : curve_passing_condition x y) : x * y = 6 :=
  sorry

end curve_equation_l372_37259


namespace min_percentage_both_physics_chemistry_l372_37290

/--
Given:
- A certain school conducted a survey.
- 68% of the students like physics.
- 72% of the students like chemistry.

Prove that the minimum percentage of students who like both physics and chemistry is 40%.
-/
theorem min_percentage_both_physics_chemistry (P C : ℝ)
(hP : P = 0.68) (hC : C = 0.72) :
  ∃ B, B = P + C - 1 ∧ B = 0.40 :=
by
  sorry

end min_percentage_both_physics_chemistry_l372_37290


namespace base_conversion_least_sum_l372_37211

theorem base_conversion_least_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3) : a + b = 10 :=
sorry

end base_conversion_least_sum_l372_37211


namespace probability_pink_second_marble_l372_37297

def bagA := (5, 5)  -- (red, green)
def bagB := (8, 2)  -- (pink, purple)
def bagC := (3, 7)  -- (pink, purple)

def P (success total : ℕ) := success / total

def probability_red := P 5 10
def probability_green := P 5 10

def probability_pink_given_red := P 8 10
def probability_pink_given_green := P 3 10

theorem probability_pink_second_marble :
  probability_red * probability_pink_given_red +
  probability_green * probability_pink_given_green = 11 / 20 :=
sorry

end probability_pink_second_marble_l372_37297


namespace expression_value_l372_37209

theorem expression_value (m n a b x : ℤ) (h1 : m = -n) (h2 : a * b = 1) (h3 : |x| = 3) :
  x = 3 ∨ x = -3 → (x = 3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = 26) ∧
                  (x = -3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = -28) := by
  sorry

end expression_value_l372_37209


namespace average_score_l372_37206

theorem average_score (classA_students classB_students : ℕ)
  (avg_score_classA avg_score_classB : ℕ)
  (h_classA : classA_students = 40)
  (h_classB : classB_students = 50)
  (h_avg_classA : avg_score_classA = 90)
  (h_avg_classB : avg_score_classB = 81) :
  (classA_students * avg_score_classA + classB_students * avg_score_classB) / 
  (classA_students + classB_students) = 85 := 
  by sorry

end average_score_l372_37206


namespace range_of_k_l372_37239

theorem range_of_k (k : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = 2 * (n:ℕ)^2 + k * (n:ℕ)) 
  (increasing : ∀ n : ℕ+, a n < a (n + 1)) : 
  k > -6 := 
by 
  sorry

end range_of_k_l372_37239


namespace range_of_x_l372_37248

theorem range_of_x (x : ℝ) (h : ∃ y : ℝ, y = (x - 3) ∧ y > 0) : x > 3 :=
sorry

end range_of_x_l372_37248


namespace range_of_m_decreasing_l372_37220

theorem range_of_m_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 3) * x₁ + 5 > (m - 3) * x₂ + 5) ↔ m < 3 :=
by
  sorry

end range_of_m_decreasing_l372_37220


namespace gold_stickers_for_second_student_l372_37201

theorem gold_stickers_for_second_student :
  (exists f : ℕ → ℕ,
      f 1 = 29 ∧
      f 3 = 41 ∧
      f 4 = 47 ∧
      f 5 = 53 ∧
      f 6 = 59 ∧
      (∀ n, f (n + 1) - f n = 6 ∨ f (n + 2) - f n = 12)) →
  (∃ f : ℕ → ℕ, f 2 = 35) :=
by
  sorry

end gold_stickers_for_second_student_l372_37201


namespace total_flowers_in_vases_l372_37269

theorem total_flowers_in_vases :
  let vase_count := 5
  let flowers_per_vase_4 := 5
  let flowers_per_vase_1 := 6
  let vases_with_5_flowers := 4
  let vases_with_6_flowers := 1
  (4 * 5 + 1 * 6 = 26) := by
  let total_flowers := 4 * 5 + 1 * 6
  show total_flowers = 26
  sorry

end total_flowers_in_vases_l372_37269


namespace largest_perimeter_l372_37263

-- Define the problem's conditions
def side1 := 7
def side2 := 9
def integer_side (x : ℕ) : Prop := (x > 2) ∧ (x < 16)

-- Define the perimeter calculation
def perimeter (a b c : ℕ) := a + b + c

-- The theorem statement which we want to prove
theorem largest_perimeter : ∃ x : ℕ, integer_side x ∧ perimeter side1 side2 x = 31 :=
by
  sorry

end largest_perimeter_l372_37263


namespace sum_of_x_values_proof_l372_37242

noncomputable def sum_of_x_values : ℝ := 
  (-(-4)) / 1 -- Sum of roots of x^2 - 4x - 7 = 0

theorem sum_of_x_values_proof (x : ℝ) (h : 7 = (x^3 - 2 * x^2 - 8 * x) / (x + 2)) : sum_of_x_values = 4 :=
sorry

end sum_of_x_values_proof_l372_37242


namespace all_roots_are_nth_roots_of_unity_l372_37208

noncomputable def smallest_positive_integer_n : ℕ :=
  5
  
theorem all_roots_are_nth_roots_of_unity :
  (∀ z : ℂ, (z^4 + z^3 + z^2 + z + 1 = 0) → z^(smallest_positive_integer_n) = 1) :=
  by
    sorry

end all_roots_are_nth_roots_of_unity_l372_37208


namespace miaCompletedAdditionalTasksOn6Days_l372_37238

def numDaysCompletingAdditionalTasks (n m : ℕ) : Prop :=
  n + m = 15 ∧ 4 * n + 7 * m = 78

theorem miaCompletedAdditionalTasksOn6Days (n m : ℕ): numDaysCompletingAdditionalTasks n m -> m = 6 :=
by
  intro h
  sorry

end miaCompletedAdditionalTasksOn6Days_l372_37238


namespace subset_ratio_l372_37299

theorem subset_ratio (S T : ℕ) (hS : S = 256) (hT : T = 56) :
  (T / S : ℚ) = 7 / 32 := by
sorry

end subset_ratio_l372_37299


namespace first_term_geometric_series_l372_37288

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end first_term_geometric_series_l372_37288


namespace solve_system_of_equations_l372_37247

theorem solve_system_of_equations
  (x y : ℚ)
  (h1 : 5 * x - 3 * y = -7)
  (h2 : 4 * x + 6 * y = 34) :
  x = 10 / 7 ∧ y = 33 / 7 :=
by
  sorry

end solve_system_of_equations_l372_37247


namespace length_of_platform_is_300_meters_l372_37271

-- Definitions used in the proof
def kmph_to_mps (v: ℕ) : ℕ := (v * 1000) / 3600

def speed := kmph_to_mps 72

def time_cross_man := 15

def length_train := speed * time_cross_man

def time_cross_platform := 30

def total_distance_cross_platform := speed * time_cross_platform

def length_platform := total_distance_cross_platform - length_train

theorem length_of_platform_is_300_meters :
  length_platform = 300 :=
by
  sorry

end length_of_platform_is_300_meters_l372_37271


namespace avg_two_expressions_l372_37244

theorem avg_two_expressions (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 84) : a = 32 := sorry

end avg_two_expressions_l372_37244


namespace abs_eq_sets_l372_37262

theorem abs_eq_sets (x : ℝ) : 
  (|x - 25| + |x - 15| = |2 * x - 40|) → (x ≤ 15 ∨ x ≥ 25) :=
by
  sorry

end abs_eq_sets_l372_37262


namespace max_convex_quadrilaterals_l372_37205

-- Define the points on the plane and the conditions
variable (A : Fin 7 → (ℝ × ℝ))

-- Hypothesis that any 3 given points are not collinear
def not_collinear (P Q R : (ℝ × ℝ)) : Prop :=
  (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1)

-- Hypothesis that the convex hull of all points is \triangle A1 A2 A3
def convex_hull_triangle (A : Fin 7 → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : Fin 7), i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)

-- The theorem to be proven
theorem max_convex_quadrilaterals :
  convex_hull_triangle A →
  (∀ i j k : Fin 7, i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)) →
  ∃ n, n = 17 := 
by
  sorry

end max_convex_quadrilaterals_l372_37205


namespace celer_tanks_dimensions_l372_37275

theorem celer_tanks_dimensions :
  ∃ (a v : ℕ), 
    (a * a * v = 200) ∧
    (2 * a ^ 3 + 50 = 300) ∧
    (a = 5) ∧
    (v = 8) :=
sorry

end celer_tanks_dimensions_l372_37275


namespace xiaotian_sep_usage_plan_cost_effectiveness_l372_37218

noncomputable def problem₁ (units : List Int) : Real :=
  units.sum / 1024 + 5 * 6

theorem xiaotian_sep_usage (units : List Int) (h : units = [200, -100, 100, -100, 212, 200]) :
  problem₁ units = 30.5 :=
sorry

def plan_cost_a (x : Int) : Real := 5 * x + 4

def plan_cost_b (x : Int) : Real :=
  if h : 20 < x ∧ x <= 23 then 5 * x - 1
  else 3 * x + 45

theorem plan_cost_effectiveness (x : Int) (h : x > 23) :
  plan_cost_a x > plan_cost_b x :=
sorry

end xiaotian_sep_usage_plan_cost_effectiveness_l372_37218


namespace base_7_to_base_10_equiv_l372_37258

theorem base_7_to_base_10_equiv : 
  ∀ (d2 d1 d0 : ℕ), 
      d2 = 3 → d1 = 4 → d0 = 6 → 
      (d2 * 7^2 + d1 * 7^1 + d0 * 7^0) = 181 := 
by 
  sorry

end base_7_to_base_10_equiv_l372_37258


namespace distinct_triangles_count_l372_37280

def num_points : ℕ := 8
def num_rows : ℕ := 2
def num_cols : ℕ := 4

-- Define the number of ways to choose 3 points from the 8 available points.
def combinations (n k : ℕ) := Nat.choose n k
def total_combinations := combinations num_points 3

-- Define the number of degenerate cases of collinear points in columns.
def degenerate_cases_per_column := combinations num_cols 3
def total_degenerate_cases := num_cols * degenerate_cases_per_column

-- The number of distinct triangles is the total combinations minus the degenerate cases.
def distinct_triangles := total_combinations - total_degenerate_cases

theorem distinct_triangles_count : distinct_triangles = 40 := by
  -- the proof goes here
  sorry

end distinct_triangles_count_l372_37280


namespace least_positive_integer_l372_37282

theorem least_positive_integer (n : ℕ) (h1 : n > 1)
  (h2 : n % 3 = 2) (h3 : n % 4 = 2) (h4 : n % 5 = 2) (h5 : n % 11 = 2) :
  n = 662 :=
sorry

end least_positive_integer_l372_37282


namespace harry_book_pages_correct_l372_37272

-- Define the total pages in Selena's book.
def selena_book_pages : ℕ := 400

-- Define Harry's book pages as 20 fewer than half of Selena's book pages.
def harry_book_pages : ℕ := (selena_book_pages / 2) - 20

-- The theorem to prove the number of pages in Harry's book.
theorem harry_book_pages_correct : harry_book_pages = 180 := by
  sorry

end harry_book_pages_correct_l372_37272


namespace solve_equation_l372_37293

theorem solve_equation:
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ 
    (x - y - x / y - (x^3 / y^3) + (x^4 / y^4) = 2017) ∧ 
    ((x = 2949 ∧ y = 983) ∨ (x = 4022 ∧ y = 2011)) :=
sorry

end solve_equation_l372_37293


namespace total_balloons_is_72_l372_37295

-- Definitions for the conditions from the problem
def fred_balloons : Nat := 10
def sam_balloons : Nat := 46
def dan_balloons : Nat := 16

-- The total number of red balloons is the sum of Fred's, Sam's, and Dan's balloons
def total_balloons (f s d : Nat) : Nat := f + s + d

-- The theorem stating the problem to be proved
theorem total_balloons_is_72 : total_balloons fred_balloons sam_balloons dan_balloons = 72 := by
  sorry

end total_balloons_is_72_l372_37295


namespace consecutive_integers_sum_l372_37207

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l372_37207


namespace range_of_a_l372_37225

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l372_37225


namespace total_balloons_l372_37219

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end total_balloons_l372_37219


namespace sin_360_eq_0_l372_37200

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end sin_360_eq_0_l372_37200


namespace summer_camp_activity_l372_37231

theorem summer_camp_activity :
  ∃ (a b c d e f : ℕ), 
  a + b + c + d + 3 * e + 4 * f = 12 ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  f = 1 := by
  sorry

end summer_camp_activity_l372_37231


namespace inequality_proof_l372_37235

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2) +
    (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2) +
    (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2)
  ) ≤ 8 := 
by
  sorry

end inequality_proof_l372_37235


namespace bacteria_growth_time_l372_37227

theorem bacteria_growth_time : 
  (∀ n : ℕ, 2 ^ n = 4096 → (n * 15) / 60 = 3) :=
by
  sorry

end bacteria_growth_time_l372_37227


namespace selling_price_for_loss_l372_37240

noncomputable def cp : ℝ := 640
def sp1 : ℝ := 768
def sp2 : ℝ := 448
def sp_profitable_sale : ℝ := 832

theorem selling_price_for_loss :
  sp_profitable_sale - cp = cp - sp2 :=
by
  sorry

end selling_price_for_loss_l372_37240


namespace option_C_correct_l372_37267

-- Define the base a and natural numbers m and n for exponents
variables {a : ℕ} {m n : ℕ}

-- Lean statement to prove (a^5)^3 = a^(5 * 3)
theorem option_C_correct : (a^5)^3 = a^(5 * 3) := 
by sorry

end option_C_correct_l372_37267


namespace ratio_of_times_gina_chooses_to_her_sister_l372_37253

theorem ratio_of_times_gina_chooses_to_her_sister (sister_shows : ℕ) (minutes_per_show : ℕ) (gina_minutes : ℕ) (ratio : ℕ × ℕ) :
  sister_shows = 24 →
  minutes_per_show = 50 →
  gina_minutes = 900 →
  ratio = (900 / Nat.gcd 900 1200, 1200 / Nat.gcd 900 1200) →
  ratio = (3, 4) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_times_gina_chooses_to_her_sister_l372_37253


namespace solve_equation_l372_37270

theorem solve_equation :
  ∃! (x y z : ℝ), 2 * x^4 + 2 * y^4 - 4 * x^3 * y + 6 * x^2 * y^2 - 4 * x * y^3 + 7 * y^2 + 7 * z^2 - 14 * y * z - 70 * y + 70 * z + 175 = 0 ∧ x = 0 ∧ y = 0 ∧ z = -5 :=
by
  sorry

end solve_equation_l372_37270


namespace age_product_difference_l372_37291

theorem age_product_difference (age_today : ℕ) (product_today : ℕ) (product_next_year : ℕ) :
  age_today = 7 →
  product_today = age_today * age_today →
  product_next_year = (age_today + 1) * (age_today + 1) →
  product_next_year - product_today = 15 :=
by
  sorry

end age_product_difference_l372_37291


namespace son_l372_37230

theorem son's_age (S F : ℕ) (h1: F = S + 27) (h2: F + 2 = 2 * (S + 2)) : S = 25 := by
  sorry

end son_l372_37230


namespace area_enclosed_by_curves_l372_37226

theorem area_enclosed_by_curves (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, (x + a * y)^2 = 16 * a^2) ∧ (∀ x y : ℝ, (a * x - y)^2 = 4 * a^2) →
  ∃ A : ℝ, A = 32 * a^2 / (1 + a^2) :=
by
  sorry

end area_enclosed_by_curves_l372_37226


namespace quarter_sector_area_l372_37273

theorem quarter_sector_area (d : ℝ) (h : d = 10) : (π * (d / 2)^2) / 4 = 6.25 * π :=
by 
  sorry

end quarter_sector_area_l372_37273


namespace correct_mark_proof_l372_37284

-- Define the conditions
def wrong_mark := 85
def increase_in_average : ℝ := 0.5
def number_of_pupils : ℕ := 104

-- Define the correct mark to be proven
noncomputable def correct_mark : ℕ := 33

-- Statement to be proven
theorem correct_mark_proof (x : ℝ) :
  (wrong_mark - x) / number_of_pupils = increase_in_average → x = correct_mark :=
by
  sorry

end correct_mark_proof_l372_37284


namespace Aiyanna_has_more_cookies_l372_37223

def Alyssa_cookies : ℕ := 129
def Aiyanna_cookies : ℕ := 140

theorem Aiyanna_has_more_cookies :
  Aiyanna_cookies - Alyssa_cookies = 11 := by
  sorry

end Aiyanna_has_more_cookies_l372_37223


namespace angle_B_of_triangle_l372_37224

theorem angle_B_of_triangle {A B C a b c : ℝ} (h1 : b^2 = a * c) (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) : 
  B = Real.pi / 3 :=
sorry

end angle_B_of_triangle_l372_37224


namespace part1_part2_part3_l372_37287

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 - Real.log x) * (x - Real.log x) + 1

variable {a : ℝ}

-- Prove that for all x > 0, if ax^2 > ln x, then f(x) ≥ ax^2 - ln x + 1
theorem part1 (h : ∀ x > 0, a*x^2 > Real.log x) (x : ℝ) (hx : x > 0) :
  f a x ≥ a*x^2 - Real.log x + 1 := sorry

-- Find the maximum value of a given there exists x₀ ∈ (0, +∞) where f(x₀) = 1 + x₀ ln x₀ - ln² x₀
theorem part2 (h : ∃ x₀ > 0, f a x₀ = 1 + x₀ * Real.log x₀ - (Real.log x₀)^2) :
  a ≤ 1 / Real.exp 1 := sorry

-- Prove that for all 1 < x < 2, we have f(x) > ax(2-ax)
theorem part3 (h : ∀ x, 1 < x ∧ x < 2) (x : ℝ) (hx1 : 1 < x) (hx2 : x < 2) :
  f a x > a * x * (2 - a * x) := sorry

end part1_part2_part3_l372_37287


namespace sphere_surface_area_l372_37265

theorem sphere_surface_area (a b c : ℝ) (r : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : r = (Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)) / 2):
    4 * Real.pi * r ^ 2 = 50 * Real.pi :=
by
  sorry

end sphere_surface_area_l372_37265


namespace books_total_l372_37292

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end books_total_l372_37292


namespace moles_of_H2_required_l372_37294

theorem moles_of_H2_required 
  (moles_C : ℕ) 
  (moles_O2 : ℕ) 
  (moles_CH4 : ℕ) 
  (moles_CO2 : ℕ) 
  (balanced_reaction_1 : ℕ → ℕ → ℕ → Prop)
  (balanced_reaction_2 : ℕ → ℕ → ℕ → ℕ → Prop)
  (H_balanced : balanced_reaction_2 2 4 2 1)
  (H_form_CO2 : balanced_reaction_1 1 1 1) :
  moles_C = 2 ∧ moles_O2 = 1 ∧ moles_CH4 = 2 ∧ moles_CO2 = 1 → (∃ moles_H2, moles_H2 = 4) :=
by sorry

end moles_of_H2_required_l372_37294


namespace annual_rent_per_sqft_l372_37260

theorem annual_rent_per_sqft
  (length width monthly_rent : ℕ)
  (H_length : length = 10)
  (H_width : width = 8)
  (H_monthly_rent : monthly_rent = 2400) :
  (12 * monthly_rent) / (length * width) = 360 := by
  sorry

end annual_rent_per_sqft_l372_37260


namespace inf_pos_integers_n_sum_two_squares_l372_37250

theorem inf_pos_integers_n_sum_two_squares:
  ∃ (s : ℕ → ℕ), (∀ (k : ℕ), ∃ (a₁ b₁ a₂ b₂ : ℕ),
   a₁ > 0 ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 ∧ s k = n ∧
   n = a₁^2 + b₁^2 ∧ n = a₂^2 + b₂^2 ∧ 
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂)) := sorry

end inf_pos_integers_n_sum_two_squares_l372_37250


namespace measure_time_with_hourglasses_l372_37255

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end measure_time_with_hourglasses_l372_37255


namespace combinations_of_coins_l372_37237

theorem combinations_of_coins (p n d : ℕ) (h₁ : p ≥ 0) (h₂ : n ≥ 0) (h₃ : d ≥ 0) 
  (value_eq : p + 5 * n + 10 * d = 25) : 
  ∃! c : ℕ, c = 12 :=
sorry

end combinations_of_coins_l372_37237


namespace travel_rate_on_foot_l372_37281

theorem travel_rate_on_foot
  (total_distance : ℝ)
  (total_time : ℝ)
  (distance_on_foot : ℝ)
  (rate_on_bicycle : ℝ)
  (rate_on_foot : ℝ) :
  total_distance = 80 ∧ total_time = 7 ∧ distance_on_foot = 32 ∧ rate_on_bicycle = 16 →
  rate_on_foot = 8 := by
  sorry

end travel_rate_on_foot_l372_37281


namespace expense_and_income_calculations_l372_37274

def alexander_salary : ℕ := 125000
def natalia_salary : ℕ := 61000
def utilities_transport_household : ℕ := 17000
def loan_repayment : ℕ := 15000
def theater_cost : ℕ := 5000
def cinema_cost_per_person : ℕ := 1000
def savings_crimea : ℕ := 20000
def dining_weekday_cost : ℕ := 1500
def dining_weekend_cost : ℕ := 3000
def weekdays : ℕ := 20
def weekends : ℕ := 10
def phone_A_cost : ℕ := 57000
def phone_B_cost : ℕ := 37000

def total_expenses : ℕ :=
  utilities_transport_household +
  loan_repayment +
  theater_cost + 2 * cinema_cost_per_person +
  savings_crimea +
  weekdays * dining_weekday_cost +
  weekends * dining_weekend_cost

def net_income : ℕ :=
  alexander_salary + natalia_salary

def can_buy_phones : Prop :=
  net_income - total_expenses < phone_A_cost + phone_B_cost

theorem expense_and_income_calculations :
  total_expenses = 119000 ∧
  net_income = 186000 ∧
  can_buy_phones :=
by
  sorry

end expense_and_income_calculations_l372_37274


namespace donna_total_episodes_per_week_l372_37296

-- Defining the conditions
def episodes_per_weekday : ℕ := 8
def weekday_count : ℕ := 5
def weekend_factor : ℕ := 3
def weekend_count : ℕ := 2

-- Theorem statement
theorem donna_total_episodes_per_week :
  (episodes_per_weekday * weekday_count) + ((episodes_per_weekday * weekend_factor) * weekend_count) = 88 := 
  by sorry

end donna_total_episodes_per_week_l372_37296


namespace pencils_per_child_l372_37279

theorem pencils_per_child (children : ℕ) (total_pencils : ℕ) (h1 : children = 2) (h2 : total_pencils = 12) :
  total_pencils / children = 6 :=
by 
  sorry

end pencils_per_child_l372_37279


namespace problem_solution_l372_37222

theorem problem_solution :
  ∀ x y : ℝ, 9 * y^2 + 6 * x * y + x + 12 = 0 → (x ≤ -3 ∨ x ≥ 4) :=
  sorry

end problem_solution_l372_37222


namespace willie_bananas_l372_37228

variable (W : ℝ) 

theorem willie_bananas (h1 : 35.0 - 14.0 = 21.0) (h2: W + 35.0 = 83.0) : 
  W = 48.0 :=
by
  sorry

end willie_bananas_l372_37228


namespace tan_sum_of_angles_eq_neg_sqrt_three_l372_37283

theorem tan_sum_of_angles_eq_neg_sqrt_three 
  (A B C : ℝ)
  (h1 : B - A = C - B)
  (h2 : A + B + C = Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 :=
sorry

end tan_sum_of_angles_eq_neg_sqrt_three_l372_37283


namespace elberta_has_22_dollars_l372_37243

theorem elberta_has_22_dollars (granny_smith : ℝ) (anjou : ℝ) (elberta : ℝ) 
  (h1 : granny_smith = 75) 
  (h2 : anjou = granny_smith / 4)
  (h3 : elberta = anjou + 3) : 
  elberta = 22 := 
by
  sorry

end elberta_has_22_dollars_l372_37243


namespace relationship_among_a_b_c_l372_37204

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_deriv : ∀ x ≠ 0, f'' x + f x / x > 0)

noncomputable def a : ℝ := (1 / Real.exp 1) * f (1 / Real.exp 1)
noncomputable def b : ℝ := -Real.exp 1 * f (-Real.exp 1)
noncomputable def c : ℝ := f 1

theorem relationship_among_a_b_c :
  a < c ∧ c < b :=
by
  -- sorry to skip the proof steps
  sorry

end relationship_among_a_b_c_l372_37204


namespace value_of_expression_l372_37245

theorem value_of_expression (m : ℝ) (h : 2 * m ^ 2 - 3 * m - 1 = 0) : 4 * m ^ 2 - 6 * m = 2 :=
sorry

end value_of_expression_l372_37245


namespace total_stars_l372_37233

-- Define the daily stars earned by Shelby
def shelby_monday : Nat := 4
def shelby_tuesday : Nat := 6
def shelby_wednesday : Nat := 3
def shelby_thursday : Nat := 5
def shelby_friday : Nat := 2
def shelby_saturday : Nat := 3
def shelby_sunday : Nat := 7

-- Define the daily stars earned by Alex
def alex_monday : Nat := 5
def alex_tuesday : Nat := 3
def alex_wednesday : Nat := 6
def alex_thursday : Nat := 4
def alex_friday : Nat := 7
def alex_saturday : Nat := 2
def alex_sunday : Nat := 5

-- Define the total stars earned by Shelby in a week
def total_shelby_stars : Nat := shelby_monday + shelby_tuesday + shelby_wednesday + shelby_thursday + shelby_friday + shelby_saturday + shelby_sunday

-- Define the total stars earned by Alex in a week
def total_alex_stars : Nat := alex_monday + alex_tuesday + alex_wednesday + alex_thursday + alex_friday + alex_saturday + alex_sunday

-- The proof problem statement
theorem total_stars (total_shelby_stars total_alex_stars : Nat) : total_shelby_stars + total_alex_stars = 62 := by
  sorry

end total_stars_l372_37233


namespace father_ate_8_brownies_l372_37210

noncomputable def brownies_initial := 24
noncomputable def brownies_mooney_ate := 4
noncomputable def brownies_after_mooney := brownies_initial - brownies_mooney_ate
noncomputable def brownies_mother_made_next_day := 24
noncomputable def brownies_total_expected := brownies_after_mooney + brownies_mother_made_next_day
noncomputable def brownies_actual_on_counter := 36

theorem father_ate_8_brownies :
  brownies_total_expected - brownies_actual_on_counter = 8 :=
by
  sorry

end father_ate_8_brownies_l372_37210


namespace square_area_from_diagonal_l372_37232

theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : (∃ A : ℝ, A = 392) :=
by
  sorry

end square_area_from_diagonal_l372_37232


namespace derivative_at_0_l372_37249

noncomputable def f (x : ℝ) := Real.exp x / (x + 2)

theorem derivative_at_0 : deriv f 0 = 1 / 4 := sorry

end derivative_at_0_l372_37249


namespace smallest_integer_with_remainders_l372_37212

theorem smallest_integer_with_remainders :
  ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) := 
sorry

end smallest_integer_with_remainders_l372_37212


namespace quadratic_has_one_solution_l372_37257

theorem quadratic_has_one_solution (n : ℤ) : 
  (n ^ 2 - 64 = 0) ↔ (n = 8 ∨ n = -8) := 
by
  sorry

end quadratic_has_one_solution_l372_37257


namespace arithmetic_sequence_sum_l372_37261

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d)
    (h_a5 : a 5 = 3)
    (h_a6 : a 6 = -2) :
  a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -49 :=
by
  sorry

end arithmetic_sequence_sum_l372_37261


namespace ray_steps_problem_l372_37217

theorem ray_steps_problem : ∃ n, n > 15 ∧ n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧ n = 71 :=
by
  sorry

end ray_steps_problem_l372_37217


namespace hour_hand_degrees_per_hour_l372_37203

-- Definitions based on the conditions
def number_of_rotations_in_6_days : ℕ := 12
def degrees_per_rotation : ℕ := 360
def hours_in_6_days : ℕ := 6 * 24

-- Statement to prove
theorem hour_hand_degrees_per_hour :
  (number_of_rotations_in_6_days * degrees_per_rotation) / hours_in_6_days = 30 :=
by sorry

end hour_hand_degrees_per_hour_l372_37203


namespace speed_of_current_l372_37214

-- Definitions for the conditions
variables (m c : ℝ)

-- Condition 1: man's speed with the current
def speed_with_current := m + c = 16

-- Condition 2: man's speed against the current
def speed_against_current := m - c = 9.6

-- The goal is to prove c = 3.2 given the conditions
theorem speed_of_current (h1 : speed_with_current m c) 
                         (h2 : speed_against_current m c) :
  c = 3.2 := 
sorry

end speed_of_current_l372_37214


namespace students_appeared_l372_37286

def passed (T : ℝ) : ℝ := 0.35 * T
def B_grade_range (T : ℝ) : ℝ := 0.25 * T
def failed (T : ℝ) : ℝ := T - passed T

theorem students_appeared (T : ℝ) (hp : passed T = 0.35 * T)
    (hb : B_grade_range T = 0.25 * T) (hf : failed T = 481) :
    T = 740 :=
by
  -- proof goes here
  sorry

end students_appeared_l372_37286


namespace num_passenger_cars_l372_37252

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l372_37252


namespace bert_money_problem_l372_37229

-- Define the conditions as hypotheses
theorem bert_money_problem
  (n : ℝ)
  (h1 : n > 0)  -- Since he can't have negative or zero dollars initially
  (h2 : (1/2) * ((3/4) * n - 9) = 15) :
  n = 52 :=
sorry

end bert_money_problem_l372_37229


namespace minimum_dimes_to_afford_sneakers_l372_37202

-- Define constants and conditions using Lean
def sneaker_cost : ℝ := 45.35
def ten_dollar_bills_count : ℕ := 3
def quarter_count : ℕ := 4
def dime_value : ℝ := 0.1
def quarter_value : ℝ := 0.25
def ten_dollar_bill_value : ℝ := 10.0

-- Define a function to calculate the total amount based on the number of dimes
def total_amount (dimes : ℕ) : ℝ :=
  (ten_dollar_bills_count * ten_dollar_bill_value) +
  (quarter_count * quarter_value) +
  (dimes * dime_value)

-- The main theorem to be proven
theorem minimum_dimes_to_afford_sneakers (n : ℕ) : total_amount n ≥ sneaker_cost ↔ n ≥ 144 :=
by
  sorry

end minimum_dimes_to_afford_sneakers_l372_37202


namespace eggs_left_l372_37268

def initial_eggs := 20
def mother_used := 5
def father_used := 3
def chicken1_laid := 4
def chicken2_laid := 3
def chicken3_laid := 2
def oldest_took := 2

theorem eggs_left :
  initial_eggs - (mother_used + father_used) + (chicken1_laid + chicken2_laid + chicken3_laid) - oldest_took = 19 := 
by
  sorry

end eggs_left_l372_37268


namespace emails_in_morning_and_evening_l372_37278

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 4
def evening_emails : ℕ := 8

theorem emails_in_morning_and_evening : morning_emails + evening_emails = 11 :=
by
  sorry

end emails_in_morning_and_evening_l372_37278


namespace factorize_expression_l372_37254

theorem factorize_expression (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 :=
by
  sorry

end factorize_expression_l372_37254


namespace distance_to_second_picture_edge_l372_37264

/-- Given a wall of width 25 feet, with a first picture 5 feet wide centered on the wall,
and a second picture 3 feet wide centered in the remaining space, the distance 
from the nearest edge of the second picture to the end of the wall is 13.5 feet. -/
theorem distance_to_second_picture_edge :
  let wall_width := 25
  let first_picture_width := 5
  let second_picture_width := 3
  let side_space := (wall_width - first_picture_width) / 2
  let remaining_space := side_space
  let second_picture_side_space := (remaining_space - second_picture_width) / 2
  10 + 3.5 = 13.5 :=
by
  sorry

end distance_to_second_picture_edge_l372_37264


namespace subset_implies_bound_l372_37266

def setA := {x : ℝ | x < 2}
def setB (m : ℝ) := {x : ℝ | x < m}

theorem subset_implies_bound (m : ℝ) (h : setB m ⊆ setA) : m ≤ 2 :=
by 
  sorry

end subset_implies_bound_l372_37266


namespace exp4_is_odd_l372_37241

-- Define the domain for n to be integers and the expressions used in the conditions
variable (n : ℤ)

-- Define the expressions
def exp1 := (n + 1) ^ 2
def exp2 := (n + 1) ^ 2 - (n - 1)
def exp3 := (n + 1) ^ 3
def exp4 := (n + 1) ^ 3 - n ^ 3

-- Prove that exp4 is always odd
theorem exp4_is_odd : ∀ n : ℤ, exp4 n % 2 = 1 := by {
  -- Lean code does not require a proof here, we'll put sorry to skip the proof
  sorry
}

end exp4_is_odd_l372_37241


namespace karlsson_candies_28_l372_37236

def karlsson_max_candies (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem karlsson_candies_28 : karlsson_max_candies 28 = 378 := by
  sorry

end karlsson_candies_28_l372_37236


namespace arithmetic_sequence_ninth_term_l372_37289

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) :=
  (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

theorem arithmetic_sequence_ninth_term
  (a: ℕ → ℕ)
  (h_arith: is_arithmetic_sequence a)
  (h_sum_5: sum_of_first_n_terms a 5 = 75)
  (h_a4: a 4 = 2 * a 2) :
  a 9 = 45 :=
sorry

end arithmetic_sequence_ninth_term_l372_37289


namespace numberOfBigBoats_l372_37298

-- Conditions
variable (students : Nat) (bigBoatCapacity : Nat) (smallBoatCapacity : Nat) (totalBoats : Nat)
variable (students_eq : students = 52)
variable (bigBoatCapacity_eq : bigBoatCapacity = 8)
variable (smallBoatCapacity_eq : smallBoatCapacity = 4)
variable (totalBoats_eq : totalBoats = 9)

theorem numberOfBigBoats : bigBoats + smallBoats = totalBoats → 
                         bigBoatCapacity * bigBoats + smallBoatCapacity * smallBoats = students → 
                         bigBoats = 4 := 
by
  intros h1 h2
  -- Proof steps
  sorry


end numberOfBigBoats_l372_37298
