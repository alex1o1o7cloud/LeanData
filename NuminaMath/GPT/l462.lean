import Mathlib

namespace NUMINAMATH_GPT_max_min_diff_of_c_l462_46202

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end NUMINAMATH_GPT_max_min_diff_of_c_l462_46202


namespace NUMINAMATH_GPT_ellipse_meets_sine_more_than_8_points_l462_46277

noncomputable def ellipse_intersects_sine_curve_more_than_8_times (a b : ℝ) (h k : ℝ) :=
  ∃ p : ℕ, p > 8 ∧ 
  ∃ (x y : ℝ), 
    (∃ (i : ℕ), y = Real.sin x ∧ 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)

theorem ellipse_meets_sine_more_than_8_points : 
  ∀ (a b h k : ℝ), ellipse_intersects_sine_curve_more_than_8_times a b h k := 
by sorry

end NUMINAMATH_GPT_ellipse_meets_sine_more_than_8_points_l462_46277


namespace NUMINAMATH_GPT_last_two_digits_of_1976_pow_100_l462_46225

theorem last_two_digits_of_1976_pow_100 :
  (1976 ^ 100) % 100 = 76 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_1976_pow_100_l462_46225


namespace NUMINAMATH_GPT_expectation_equality_variance_inequality_l462_46257

noncomputable def X1_expectation : ℚ :=
  2 * (2 / 5 : ℚ)

noncomputable def X1_variance : ℚ :=
  2 * (2 / 5) * (1 - 2 / 5)

noncomputable def P_X2_0 : ℚ :=
  (3 * 2) / (5 * 4)

noncomputable def P_X2_1 : ℚ :=
  (2 * 3) / (5 * 4)

noncomputable def P_X2_2 : ℚ :=
  (2 * 1) / (5 * 4)

noncomputable def X2_expectation : ℚ :=
  0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2

noncomputable def X2_variance : ℚ :=
  P_X2_0 * (0 - X2_expectation)^2 + P_X2_1 * (1 - X2_expectation)^2 + P_X2_2 * (2 - X2_expectation)^2

theorem expectation_equality : X1_expectation = X2_expectation :=
  by sorry

theorem variance_inequality : X1_variance > X2_variance :=
  by sorry

end NUMINAMATH_GPT_expectation_equality_variance_inequality_l462_46257


namespace NUMINAMATH_GPT_fence_cost_l462_46249

noncomputable def price_per_foot (total_cost : ℝ) (perimeter : ℝ) : ℝ :=
  total_cost / perimeter

theorem fence_cost (area : ℝ) (total_cost : ℝ) (price : ℝ) :
  area = 289 → total_cost = 4012 → price = price_per_foot 4012 (4 * (Real.sqrt 289)) → price = 59 :=
by
  intros h_area h_cost h_price
  sorry

end NUMINAMATH_GPT_fence_cost_l462_46249


namespace NUMINAMATH_GPT_letter_puzzle_l462_46287

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end NUMINAMATH_GPT_letter_puzzle_l462_46287


namespace NUMINAMATH_GPT_solution_set_of_inequality_l462_46268

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1 / 3 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l462_46268


namespace NUMINAMATH_GPT_tan_ratio_l462_46222

theorem tan_ratio (a b : ℝ) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : Real.sin (a + b) = 5/8) (h2 : Real.sin (a - b) = 3/8) :
  (Real.tan a) / (Real.tan b) = 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_ratio_l462_46222


namespace NUMINAMATH_GPT_calc_expression_l462_46267

-- Define the fractions and whole number in the problem
def frac1 : ℚ := 5/6
def frac2 : ℚ := 1 + 1/6
def whole : ℚ := 2

-- Define the expression to be proved
def expression : ℚ := (frac1) - (-whole) + (frac2)

-- The theorem to be proved
theorem calc_expression : expression = 4 :=
by { sorry }

end NUMINAMATH_GPT_calc_expression_l462_46267


namespace NUMINAMATH_GPT_sum_of_a_and_b_is_24_l462_46213

theorem sum_of_a_and_b_is_24 
  (a b : ℕ) 
  (h_a_pos : a > 0) 
  (h_b_gt_one : b > 1) 
  (h_maximal : ∀ (a' b' : ℕ), (a' > 0) → (b' > 1) → (a'^b' < 500) → (a'^b' ≤ a^b)) :
  a + b = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_a_and_b_is_24_l462_46213


namespace NUMINAMATH_GPT_polygon_sides_eight_l462_46295

theorem polygon_sides_eight {n : ℕ} (h : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eight_l462_46295


namespace NUMINAMATH_GPT_factorization_exists_l462_46243

-- Define the polynomial f(x)
def f (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 12

-- Definition for polynomial g(x)
def g (a : ℤ) (x : ℚ) : ℚ := x^2 + a*x + 3

-- Definition for polynomial h(x)
def h (b : ℤ) (x : ℚ) : ℚ := x^2 + b*x + 4

-- The main statement to prove
theorem factorization_exists :
  ∃ (a b : ℤ), (∀ x, f x = (g a x) * (h b x)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_exists_l462_46243


namespace NUMINAMATH_GPT_minimize_quadratic_expression_l462_46237

noncomputable def quadratic_expression (b : ℝ) : ℝ :=
  (1 / 3) * b^2 + 7 * b - 6

theorem minimize_quadratic_expression : ∃ b : ℝ, quadratic_expression b = -10.5 :=
  sorry

end NUMINAMATH_GPT_minimize_quadratic_expression_l462_46237


namespace NUMINAMATH_GPT_volume_of_Q_3_l462_46231

noncomputable def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 2       -- 1 + 1
  | 2 => 2 + 3 / 16
  | 3 => (2 + 3 / 16) + 3 / 64
  | _ => sorry -- This handles cases n >= 4, which we don't need.

theorem volume_of_Q_3 : Q 3 = 143 / 64 := by
  sorry

end NUMINAMATH_GPT_volume_of_Q_3_l462_46231


namespace NUMINAMATH_GPT_major_axis_length_l462_46241

def length_of_major_axis 
  (tangent_x : ℝ) (f1 : ℝ × ℝ) (f2 : ℝ × ℝ) : ℝ :=
  sorry

theorem major_axis_length 
  (hx_tangent : (4, 0) = (4, 0)) 
  (foci : (4, 2 + 2 * Real.sqrt 2) = (4, 2 + 2 * Real.sqrt 2) ∧ 
         (4, 2 - 2 * Real.sqrt 2) = (4, 2 - 2 * Real.sqrt 2)) :
  length_of_major_axis 4 
  (4, 2 + 2 * Real.sqrt 2) (4, 2 - 2 * Real.sqrt 2) = 4 :=
sorry

end NUMINAMATH_GPT_major_axis_length_l462_46241


namespace NUMINAMATH_GPT_ratio_of_fish_cat_to_dog_l462_46285

theorem ratio_of_fish_cat_to_dog (fish_dog : ℕ) (cost_per_fish : ℕ) (total_spent : ℕ)
  (h1 : fish_dog = 40)
  (h2 : cost_per_fish = 4)
  (h3 : total_spent = 240) :
  (total_spent / cost_per_fish - fish_dog) / fish_dog = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_fish_cat_to_dog_l462_46285


namespace NUMINAMATH_GPT_value_of_expression_l462_46209

theorem value_of_expression : (2 + 4 + 6) - (1 + 3 + 5) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l462_46209


namespace NUMINAMATH_GPT_geometric_sequence_sum_l462_46232

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (ha1 : q ≠ 0)
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 3 + a 4 = (a 1 + a 2) * q^2)
  : a 5 + a 6 = 48 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l462_46232


namespace NUMINAMATH_GPT_Kims_final_score_l462_46212

def easy_points : ℕ := 2
def average_points : ℕ := 3
def hard_points : ℕ := 5
def expert_points : ℕ := 7

def easy_correct : ℕ := 6
def average_correct : ℕ := 2
def hard_correct : ℕ := 4
def expert_correct : ℕ := 3

def complex_problems_bonus : ℕ := 1
def complex_problems_solved : ℕ := 2

def penalty_per_incorrect : ℕ := 1
def easy_incorrect : ℕ := 1
def average_incorrect : ℕ := 2
def hard_incorrect : ℕ := 2
def expert_incorrect : ℕ := 3

theorem Kims_final_score : 
  (easy_correct * easy_points + 
   average_correct * average_points + 
   hard_correct * hard_points + 
   expert_correct * expert_points + 
   complex_problems_solved * complex_problems_bonus) - 
   (easy_incorrect * penalty_per_incorrect + 
    average_incorrect * penalty_per_incorrect + 
    hard_incorrect * penalty_per_incorrect + 
    expert_incorrect * penalty_per_incorrect) = 53 :=
by 
  sorry

end NUMINAMATH_GPT_Kims_final_score_l462_46212


namespace NUMINAMATH_GPT_volume_of_tetrahedron_ABCD_l462_46219

noncomputable def tetrahedron_volume_proof (S: ℝ) (AB AD BD: ℝ) 
    (angle_ABD_DBC_CBA angle_ADB_BDC_CDA angle_ACB_ACD_BCD: ℝ) : ℝ :=
if h1 : S = 1 ∧ AB = AD ∧ BD = (Real.sqrt 2) / 2
    ∧ angle_ABD_DBC_CBA = 180 ∧ angle_ADB_BDC_CDA = 180 
    ∧ angle_ACB_ACD_BCD = 90 then
  (1 / 24)
else
  0

-- Statement to prove
theorem volume_of_tetrahedron_ABCD : tetrahedron_volume_proof 1 AB AD ((Real.sqrt 2) / 2) 180 180 90 = (1 / 24) :=
by sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_ABCD_l462_46219


namespace NUMINAMATH_GPT_problem_statement_l462_46223

theorem problem_statement :
  ∀ (x : ℝ),
    (5 * x - 10 = 15 * x + 5) →
    (5 * (x + 3) = 15 / 2) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_problem_statement_l462_46223


namespace NUMINAMATH_GPT_geometric_sum_of_first_four_terms_eq_120_l462_46292

theorem geometric_sum_of_first_four_terms_eq_120
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (ha2 : a 2 = 9)
  (ha5 : a 5 = 243) :
  a 1 * (1 - r^4) / (1 - r) = 120 := 
sorry

end NUMINAMATH_GPT_geometric_sum_of_first_four_terms_eq_120_l462_46292


namespace NUMINAMATH_GPT_maximum_radius_l462_46282

open Set Real

-- Definitions of sets M, N, and D_r.
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≥ 1 / 4 * p.fst^2}

def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≤ -1 / 4 * p.fst^2 + p.fst + 7}

def D_r (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - x₀)^2 + (p.snd - y₀)^2 ≤ r^2}

-- Theorem statement for the largest r
theorem maximum_radius {x₀ y₀ : ℝ} (H : D_r x₀ y₀ r ⊆ M ∩ N) :
  r = sqrt ((25 - 5 * sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_maximum_radius_l462_46282


namespace NUMINAMATH_GPT_part_a_part_b_l462_46220

noncomputable section

open Real

theorem part_a (x y z : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x-1)^2) + (y^2 / (y-1)^2) + (z^2 / (z-1)^2) ≥ 1 :=
sorry

theorem part_b : ∃ (infinitely_many : ℕ → (ℚ × ℚ × ℚ)), 
  ∀ n, ((infinitely_many n).1.1 ≠ 1) ∧ ((infinitely_many n).1.2 ≠ 1) ∧ ((infinitely_many n).2 ≠ 1) ∧ 
  ((infinitely_many n).1.1 * (infinitely_many n).1.2 * (infinitely_many n).2 = 1) ∧ 
  ((infinitely_many n).1.1^2 / ((infinitely_many n).1.1 - 1)^2 + 
   (infinitely_many n).1.2^2 / ((infinitely_many n).1.2 - 1)^2 + 
   (infinitely_many n).2^2 / ((infinitely_many n).2 - 1)^2 = 1) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l462_46220


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l462_46201

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 - 24 * Polynomial.X^2 + 143 * Polynomial.X - 210

theorem partial_fraction_decomposition (A B C p q r : ℝ) (h1 : Polynomial.roots polynomial = {p, q, r}) 
  (h2 : ∀ s : ℝ, 1 / (s^3 - 24 * s^2 + 143 * s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 243 :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l462_46201


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l462_46230

theorem quadratic_has_distinct_real_roots {k : ℝ} (hk : k < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ + k = 0) ∧ (x₂^2 - x₂ + k = 0) :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l462_46230


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l462_46252

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 1 = 3)
    (h2 : a 1 + a 3 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 * a 4 = 36 := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l462_46252


namespace NUMINAMATH_GPT_solve_inequality_range_of_m_l462_46217

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := - abs (x + 3) + m

theorem solve_inequality (x a : ℝ) :
  (f x + a - 1 > 0) ↔
  (a = 1 → x ≠ 2) ∧
  (a > 1 → true) ∧
  (a < 1 → x < a + 1 ∨ x > 3 - a) := by sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := by sorry

end NUMINAMATH_GPT_solve_inequality_range_of_m_l462_46217


namespace NUMINAMATH_GPT_correct_transformation_l462_46270

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l462_46270


namespace NUMINAMATH_GPT_integral_of_reciprocal_l462_46210

theorem integral_of_reciprocal (a b : ℝ) (h_eq : a = 1) (h_eb : b = Real.exp 1) : ∫ x in a..b, 1/x = 1 :=
by 
  rw [h_eq, h_eb]
  sorry

end NUMINAMATH_GPT_integral_of_reciprocal_l462_46210


namespace NUMINAMATH_GPT_julia_total_cost_l462_46299

theorem julia_total_cost
  (snickers_cost : ℝ := 1.5)
  (mm_cost : ℝ := 2 * snickers_cost)
  (pepsi_cost : ℝ := 2 * mm_cost)
  (bread_cost : ℝ := 3 * pepsi_cost)
  (snickers_qty : ℕ := 2)
  (mm_qty : ℕ := 3)
  (pepsi_qty : ℕ := 4)
  (bread_qty : ℕ := 5)
  (money_given : ℝ := 5 * 20) :
  ((snickers_qty * snickers_cost) + (mm_qty * mm_cost) + (pepsi_qty * pepsi_cost) + (bread_qty * bread_cost)) > money_given := 
by
  sorry

end NUMINAMATH_GPT_julia_total_cost_l462_46299


namespace NUMINAMATH_GPT_sequence_general_formula_l462_46203

theorem sequence_general_formula (a : ℕ → ℚ) (h₀ : a 1 = 3 / 5)
    (h₁ : ∀ n : ℕ, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n : ℕ, a n = 3 / (6 * n - 1) := 
by sorry

end NUMINAMATH_GPT_sequence_general_formula_l462_46203


namespace NUMINAMATH_GPT_dino_second_gig_hourly_rate_l462_46271

theorem dino_second_gig_hourly_rate (h1 : 20 * 10 = 200)
  (h2 : 5 * 40 = 200) (h3 : 500 + 500 = 1000) : 
  let total_income := 1000 
  let income_first_gig := 200 
  let income_third_gig := 200 
  let income_second_gig := total_income - income_first_gig - income_third_gig 
  let hours_second_gig := 30 
  let hourly_rate := income_second_gig / hours_second_gig 
  hourly_rate = 20 := 
by 
  sorry

end NUMINAMATH_GPT_dino_second_gig_hourly_rate_l462_46271


namespace NUMINAMATH_GPT_sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l462_46254

noncomputable def volume_of_spheres (V : ℝ) : ℝ :=
  V * (27 / 26)

noncomputable def volume_of_tetrahedrons (V : ℝ) : ℝ :=
  (3 * V * Real.sqrt 3) / (13 * Real.pi)

theorem sum_volumes_of_spheres (V : ℝ) : 
  (∑' n : ℕ, (V * (1/27)^n)) = volume_of_spheres V :=
sorry

theorem sum_volumes_of_tetrahedrons (V : ℝ) (r : ℝ) : 
  (∑' n : ℕ, (8/9 / Real.sqrt 3 * (r^3) * (1/27)^n * (1/26))) = volume_of_tetrahedrons V :=
sorry

end NUMINAMATH_GPT_sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l462_46254


namespace NUMINAMATH_GPT_mrs_randall_total_teaching_years_l462_46216

def years_teaching_third_grade : ℕ := 18
def years_teaching_second_grade : ℕ := 8

theorem mrs_randall_total_teaching_years : years_teaching_third_grade + years_teaching_second_grade = 26 :=
by
  sorry

end NUMINAMATH_GPT_mrs_randall_total_teaching_years_l462_46216


namespace NUMINAMATH_GPT_binom_150_1_l462_46298

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end NUMINAMATH_GPT_binom_150_1_l462_46298


namespace NUMINAMATH_GPT_sarah_house_units_digit_l462_46293

-- Sarah's house number has two digits
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- The four statements about Sarah's house number
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_digit_7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Exactly three out of the four statements are true
def exactly_three_true (n : ℕ) : Prop :=
  (is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ is_odd n ∧ ¬is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ ¬is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (¬is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n)

-- Main statement
theorem sarah_house_units_digit : ∃ n : ℕ, is_two_digit n ∧ exactly_three_true n ∧ n % 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sarah_house_units_digit_l462_46293


namespace NUMINAMATH_GPT_carlos_books_in_june_l462_46284

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end NUMINAMATH_GPT_carlos_books_in_june_l462_46284


namespace NUMINAMATH_GPT_bird_families_left_near_mountain_l462_46256

def total_bird_families : ℕ := 85
def bird_families_flew_to_africa : ℕ := 23
def bird_families_flew_to_asia : ℕ := 37

theorem bird_families_left_near_mountain : total_bird_families - (bird_families_flew_to_africa + bird_families_flew_to_asia) = 25 := by
  sorry

end NUMINAMATH_GPT_bird_families_left_near_mountain_l462_46256


namespace NUMINAMATH_GPT_fraction_addition_l462_46215

-- Definitions from conditions
def frac1 : ℚ := 18 / 42
def frac2 : ℚ := 2 / 9
def simplified_frac1 : ℚ := 3 / 7
def simplified_frac2 : ℚ := frac2
def common_denom_frac1 : ℚ := 27 / 63
def common_denom_frac2 : ℚ := 14 / 63

-- The problem statement to prove
theorem fraction_addition :
  frac1 + frac2 = 41 / 63 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l462_46215


namespace NUMINAMATH_GPT_fewer_mpg_in_city_l462_46206

theorem fewer_mpg_in_city
  (highway_miles : ℕ)
  (city_miles : ℕ)
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (tank_size : ℝ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 32 →
  tank_size = 336 / 32 →
  highway_mpg = 462 / tank_size →
  (highway_mpg - city_mpg) = 12 :=
by
  intros h_highway_miles h_city_miles h_city_mpg h_tank_size h_highway_mpg
  sorry

end NUMINAMATH_GPT_fewer_mpg_in_city_l462_46206


namespace NUMINAMATH_GPT_average_salary_correct_l462_46269

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 8800 := by
  sorry

end NUMINAMATH_GPT_average_salary_correct_l462_46269


namespace NUMINAMATH_GPT_measure_of_angle_D_in_scalene_triangle_l462_46224

-- Define the conditions
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ E ≠ F ∧ D ≠ F

-- Define the measure of angles based on the given conditions
def measure_of_angle_D (D E F : ℝ) : Prop :=
  E = 2 * D ∧ F = 40

-- Define the sum of angles in a triangle
def triangle_angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

theorem measure_of_angle_D_in_scalene_triangle (D E F : ℝ) (h_scalene : is_scalene D E F) 
  (h_measures : measure_of_angle_D D E F) (h_sum : triangle_angle_sum D E F) : D = 140 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_D_in_scalene_triangle_l462_46224


namespace NUMINAMATH_GPT_sandwich_cost_l462_46275

theorem sandwich_cost (soda_cost sandwich_cost total_cost : ℝ) (h1 : soda_cost = 0.87) (h2 : total_cost = 10.46) (h3 : 4 * soda_cost + 2 * sandwich_cost = total_cost) :
  sandwich_cost = 3.49 :=
by
  sorry

end NUMINAMATH_GPT_sandwich_cost_l462_46275


namespace NUMINAMATH_GPT_triplet_sums_to_two_l462_46239

theorem triplet_sums_to_two :
  (3 / 4 + 1 / 4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3 / 5 + 4 / 5 + 3 / 5 = 2) ∧
  (2 - 3 + 3 = 2) :=
by
  sorry

end NUMINAMATH_GPT_triplet_sums_to_two_l462_46239


namespace NUMINAMATH_GPT_rectangle_diagonals_equal_rhombus_not_l462_46289

/-- Define the properties for a rectangle -/
structure Rectangle :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- Define the properties for a rhombus -/
structure Rhombus :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- The property that distinguishes a rectangle from a rhombus is that the diagonals are equal. -/
theorem rectangle_diagonals_equal_rhombus_not
  (R : Rectangle)
  (H : Rhombus)
  (hR1 : R.sides_parallel)
  (hR2 : R.diagonals_equal)
  (hR3 : R.diagonals_bisect)
  (hR4 : R.angles_equal)
  (hH1 : H.sides_parallel)
  (hH2 : ¬H.diagonals_equal)
  (hH3 : H.diagonals_bisect)
  (hH4 : H.angles_equal) :
  (R.diagonals_equal) := by
  sorry

end NUMINAMATH_GPT_rectangle_diagonals_equal_rhombus_not_l462_46289


namespace NUMINAMATH_GPT_min_value_of_expression_l462_46211

noncomputable def minimum_value_expression : ℝ :=
  let f (a b : ℝ) := a^4 + b^4 + 16 / (a^2 + b^2)^2
  4

theorem min_value_of_expression (a b : ℝ) (h : 0 < a ∧ 0 < b) : 
  let f := a^4 + b^4 + 16 / (a^2 + b^2)^2
  ∃ c : ℝ, f = c ∧ c = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l462_46211


namespace NUMINAMATH_GPT_athlete_distance_l462_46240

theorem athlete_distance (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) (d : ℝ)
  (h1 : t = 24)
  (h2 : v_kmh = 30.000000000000004)
  (h3 : v_ms = v_kmh * 1000 / 3600)
  (h4 : d = v_ms * t) :
  d = 200 := 
sorry

end NUMINAMATH_GPT_athlete_distance_l462_46240


namespace NUMINAMATH_GPT_avg_weight_of_a_b_c_l462_46246

theorem avg_weight_of_a_b_c (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by
  sorry

end NUMINAMATH_GPT_avg_weight_of_a_b_c_l462_46246


namespace NUMINAMATH_GPT_sequence_periodic_a2014_l462_46259

theorem sequence_periodic_a2014 (a : ℕ → ℚ) 
  (h1 : a 1 = -1/4) 
  (h2 : ∀ n > 1, a n = 1 - (1 / (a (n - 1)))) : 
  a 2014 = -1/4 :=
sorry

end NUMINAMATH_GPT_sequence_periodic_a2014_l462_46259


namespace NUMINAMATH_GPT_hall_reunion_attendees_l462_46233

noncomputable def Oates : ℕ := 40
noncomputable def both : ℕ := 10
noncomputable def total : ℕ := 100
noncomputable def onlyOates := Oates - both
noncomputable def onlyHall := total - onlyOates - both
noncomputable def Hall := onlyHall + both

theorem hall_reunion_attendees : Hall = 70 := by {
  sorry
}

end NUMINAMATH_GPT_hall_reunion_attendees_l462_46233


namespace NUMINAMATH_GPT_no_triangular_sides_of_specific_a_b_l462_46274

theorem no_triangular_sides_of_specific_a_b (a b c : ℕ) (h1 : a = 10^100 + 1002) (h2 : b = 1001) (h3 : ∃ n : ℕ, c = n^2) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by sorry

end NUMINAMATH_GPT_no_triangular_sides_of_specific_a_b_l462_46274


namespace NUMINAMATH_GPT_cyrus_shots_percentage_l462_46283

theorem cyrus_shots_percentage (total_shots : ℕ) (missed_shots : ℕ) (made_shots : ℕ)
  (h_total : total_shots = 20)
  (h_missed : missed_shots = 4)
  (h_made : made_shots = total_shots - missed_shots) :
  (made_shots / total_shots : ℚ) * 100 = 80 := by
  sorry

end NUMINAMATH_GPT_cyrus_shots_percentage_l462_46283


namespace NUMINAMATH_GPT_compute_product_l462_46296

theorem compute_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 :=
sorry

end NUMINAMATH_GPT_compute_product_l462_46296


namespace NUMINAMATH_GPT_median_line_eqn_l462_46290

theorem median_line_eqn (A B C : ℝ × ℝ)
  (hA : A = (3, 7)) (hB : B = (5, -1)) (hC : C = (-2, -5)) :
  ∃ m b : ℝ, (4, -3, -7) = (m, b, 0) :=
by sorry

end NUMINAMATH_GPT_median_line_eqn_l462_46290


namespace NUMINAMATH_GPT_temp_neg_represents_below_zero_l462_46214

-- Definitions based on the conditions in a)
def above_zero (x: ℤ) : Prop := x > 0
def below_zero (x: ℤ) : Prop := x < 0

-- Proof problem derived from c)
theorem temp_neg_represents_below_zero (t1 t2: ℤ) 
  (h1: above_zero t1) (h2: t1 = 10) 
  (h3: below_zero t2) (h4: t2 = -3) : 
  -t2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_temp_neg_represents_below_zero_l462_46214


namespace NUMINAMATH_GPT_math_problem_solution_l462_46247

theorem math_problem_solution (x y : ℝ) : 
  abs x + x + 5 * y = 2 ∧ abs y - y + x = 7 → x + y + 2009 = 2012 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_solution_l462_46247


namespace NUMINAMATH_GPT_find_range_a_l462_46227

noncomputable def sincos_inequality (x a θ : ℝ) : Prop :=
  (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1 / 8

theorem find_range_a :
  (∀ (x : ℝ) (θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → sincos_inequality x a θ)
  ↔ a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_find_range_a_l462_46227


namespace NUMINAMATH_GPT_montague_fraction_l462_46294

noncomputable def fraction_montague (M C : ℝ) : Prop :=
  M + C = 1 ∧
  (0.70 * C) / (0.20 * M + 0.70 * C) = 7 / 11

theorem montague_fraction : ∃ M C : ℝ, fraction_montague M C ∧ M = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_montague_fraction_l462_46294


namespace NUMINAMATH_GPT_parkway_school_students_l462_46218

theorem parkway_school_students (total_boys total_soccer soccer_boys_percentage girls_not_playing_soccer : ℕ)
  (h1 : total_boys = 320)
  (h2 : total_soccer = 250)
  (h3 : soccer_boys_percentage = 86)
  (h4 : girls_not_playing_soccer = 95)
  (h5 : total_soccer * soccer_boys_percentage / 100 = 215) :
  total_boys + total_soccer - (total_soccer * soccer_boys_percentage / 100) + girls_not_playing_soccer = 450 :=
by
  sorry

end NUMINAMATH_GPT_parkway_school_students_l462_46218


namespace NUMINAMATH_GPT_intersection_M_N_l462_46273

def setM : Set ℝ := {x | x^2 - 1 ≤ 0}
def setN : Set ℝ := {x | x^2 - 3 * x > 0}

theorem intersection_M_N :
  {x | -1 ≤ x ∧ x < 0} = setM ∩ setN :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l462_46273


namespace NUMINAMATH_GPT_min_xy_value_l462_46234

theorem min_xy_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (hlog : Real.log x / Real.log 2 * Real.log y / Real.log 2 = 1) : x * y = 4 :=
by sorry

end NUMINAMATH_GPT_min_xy_value_l462_46234


namespace NUMINAMATH_GPT_geometric_sequence_sum_l462_46280

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h1 : a 1 + a 2 + a 3 = 7)
  (h2 : a 2 + a 3 + a 4 = 14) :
  a 4 + a 5 + a 6 = 56 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l462_46280


namespace NUMINAMATH_GPT_g_composed_g_has_exactly_two_distinct_real_roots_l462_46235

theorem g_composed_g_has_exactly_two_distinct_real_roots (d : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 + 4 * x + d) = 0 ∧ (y^2 + 4 * y + d) = 0) ↔ d = 8 :=
sorry

end NUMINAMATH_GPT_g_composed_g_has_exactly_two_distinct_real_roots_l462_46235


namespace NUMINAMATH_GPT_draw_two_green_marbles_probability_l462_46250

theorem draw_two_green_marbles_probability :
  let red := 5
  let green := 3
  let white := 7
  let total := red + green + white
  (green / total) * ((green - 1) / (total - 1)) = 1 / 35 :=
by
  sorry

end NUMINAMATH_GPT_draw_two_green_marbles_probability_l462_46250


namespace NUMINAMATH_GPT_imaginary_part_of_complex_num_l462_46205

-- Define the complex number and the imaginary part condition
def complex_num : ℂ := ⟨1, 2⟩

-- Define the theorem to prove the imaginary part is 2
theorem imaginary_part_of_complex_num : complex_num.im = 2 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_num_l462_46205


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l462_46229

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) 
  (h₁ : S 4 = 2) 
  (h₂ : S 8 = 6) 
  : S 12 = 12 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l462_46229


namespace NUMINAMATH_GPT_triangle_inequality_l462_46244

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_l462_46244


namespace NUMINAMATH_GPT_negation_prop_l462_46297

theorem negation_prop (p : Prop) : 
  (∀ (x : ℝ), x > 2 → x^2 - 1 > 0) → (¬(∀ (x : ℝ), x > 2 → x^2 - 1 > 0) ↔ (∃ (x : ℝ), x > 2 ∧ x^2 - 1 ≤ 0)) :=
by 
  sorry

end NUMINAMATH_GPT_negation_prop_l462_46297


namespace NUMINAMATH_GPT_boys_meet_time_is_correct_l462_46200

structure TrackMeetProblem where
  (track_length : ℕ) -- Track length in meters
  (speed_first_boy_kmh : ℚ) -- Speed of the first boy in km/hr
  (speed_second_boy_kmh : ℚ) -- Speed of the second boy in km/hr

noncomputable def time_to_meet (p : TrackMeetProblem) : ℚ :=
  let speed_first_boy_ms := (p.speed_first_boy_kmh * 1000) / 3600
  let speed_second_boy_ms := (p.speed_second_boy_kmh * 1000) / 3600
  let relative_speed := speed_first_boy_ms + speed_second_boy_ms
  (p.track_length : ℚ) / relative_speed

theorem boys_meet_time_is_correct (p : TrackMeetProblem) : 
  p.track_length = 4800 → 
  p.speed_first_boy_kmh = 61.3 → 
  p.speed_second_boy_kmh = 97.5 → 
  time_to_meet p = 108.8 := by
  intros
  sorry  

end NUMINAMATH_GPT_boys_meet_time_is_correct_l462_46200


namespace NUMINAMATH_GPT_geometric_series_problem_l462_46263

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_problem
  (c d : ℝ)
  (h : geometric_series_sum (c/d) (1/d) = 6) :
  geometric_series_sum (c/(c + 2 * d)) (1/(c + 2 * d)) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_geometric_series_problem_l462_46263


namespace NUMINAMATH_GPT_expected_teachers_with_masters_degree_l462_46255

theorem expected_teachers_with_masters_degree
  (prob: ℚ) (teachers: ℕ) (h_prob: prob = 1/4) (h_teachers: teachers = 320) :
  prob * teachers = 80 :=
by
  sorry

end NUMINAMATH_GPT_expected_teachers_with_masters_degree_l462_46255


namespace NUMINAMATH_GPT_find_k_l462_46278

theorem find_k (k : ℝ) : 
  (∀ α β : ℝ, (α * β = 15 ∧ α + β = -k ∧ (α + 3 + β + 3 = k)) → k = 3) :=
by 
  sorry

end NUMINAMATH_GPT_find_k_l462_46278


namespace NUMINAMATH_GPT_annabelle_savings_l462_46266

noncomputable def weeklyAllowance : ℕ := 30
noncomputable def junkFoodFraction : ℚ := 1 / 3
noncomputable def sweetsCost : ℕ := 8

theorem annabelle_savings :
  let junkFoodCost := weeklyAllowance * junkFoodFraction
  let totalSpent := junkFoodCost + sweetsCost
  let savings := weeklyAllowance - totalSpent
  savings = 12 := 
by
  sorry

end NUMINAMATH_GPT_annabelle_savings_l462_46266


namespace NUMINAMATH_GPT_number_of_ways_to_construct_cube_l462_46279

theorem number_of_ways_to_construct_cube :
  let num_white_cubes := 5
  let num_blue_cubes := 3
  let cube_size := (2, 2, 2)
  let num_rotations := 24
  let num_constructions := 4
  ∃ (num_constructions : ℕ), num_constructions = 4 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_construct_cube_l462_46279


namespace NUMINAMATH_GPT_max_real_solution_under_100_l462_46281

theorem max_real_solution_under_100 (k a b c r : ℕ) (h0 : ∃ (m n p : ℕ), a = k^m ∧ b = k^n ∧ c = k^p)
  (h1 : r < 100) (h2 : b^2 = 4 * a * c) (h3 : r = b / (2 * a)) : r ≤ 64 :=
sorry

end NUMINAMATH_GPT_max_real_solution_under_100_l462_46281


namespace NUMINAMATH_GPT_newspapers_sold_correct_l462_46260

def total_sales : ℝ := 425.0
def magazines_sold : ℝ := 150
def newspapers_sold : ℝ := total_sales - magazines_sold

theorem newspapers_sold_correct : newspapers_sold = 275.0 := by
  sorry

end NUMINAMATH_GPT_newspapers_sold_correct_l462_46260


namespace NUMINAMATH_GPT_find_b_l462_46265

theorem find_b (a b c : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 1 < c):
  (∀ N : ℝ, N ≠ 1 → (N^(3/a) * N^(2/(ab)) * N^(1/(abc)) = N^(39/48))) → b = 4 :=
  by
  sorry

end NUMINAMATH_GPT_find_b_l462_46265


namespace NUMINAMATH_GPT_women_at_each_table_l462_46236

/-- A waiter had 5 tables, each with 3 men and some women, and a total of 40 customers.
    Prove that there are 5 women at each table. -/
theorem women_at_each_table (W : ℕ) (total_customers : ℕ) (men_per_table : ℕ) (tables : ℕ)
  (h1 : total_customers = 40) (h2 : men_per_table = 3) (h3 : tables = 5) :
  (W * tables + men_per_table * tables = total_customers) → (W = 5) :=
by
  sorry

end NUMINAMATH_GPT_women_at_each_table_l462_46236


namespace NUMINAMATH_GPT_area_of_rectangle_l462_46264

theorem area_of_rectangle (side_small_squares : ℝ) (side_smaller_square : ℝ) (side_larger_square : ℝ) 
  (h_small_squares : side_small_squares ^ 2 = 4) 
  (h_smaller_square : side_smaller_square ^ 2 = 1) 
  (h_larger_square : side_larger_square = 2 * side_smaller_square) :
  let horizontal_length := 2 * side_small_squares
  let vertical_length := side_small_squares + side_smaller_square
  let area := horizontal_length * vertical_length
  area = 12 
:= by 
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l462_46264


namespace NUMINAMATH_GPT_circle_bisect_line_l462_46204

theorem circle_bisect_line (a : ℝ) :
  (∃ x y, (x - a) ^ 2 + (y + 1) ^ 2 = 3 ∧ 5 * x + 4 * y - a = 0) →
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_circle_bisect_line_l462_46204


namespace NUMINAMATH_GPT_border_material_correct_l462_46207

noncomputable def pi_approx := (22 : ℚ) / 7

def circle_radius (area : ℚ) (pi_value : ℚ) : ℚ :=
  (area * (7 / 22)).sqrt

def circumference (radius : ℚ) (pi_value : ℚ) : ℚ :=
  2 * pi_value * radius

def total_border_material (area : ℚ) (pi_value : ℚ) (extra : ℚ) : ℚ :=
  circumference (circle_radius area pi_value) pi_value + extra

theorem border_material_correct :
  total_border_material 616 pi_approx 3 = 91 :=
by
  sorry

end NUMINAMATH_GPT_border_material_correct_l462_46207


namespace NUMINAMATH_GPT_proof_problem_l462_46221

open Set Real

noncomputable def f (x : ℝ) : ℝ := sin x
noncomputable def g (x : ℝ) : ℝ := cos x
def U : Set ℝ := univ
def M : Set ℝ := {x | f x ≠ 0}
def N : Set ℝ := {x | g x ≠ 0}
def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem proof_problem :
  {x : ℝ | f x * g x = 0} = (C_U M) ∪ (C_U N) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l462_46221


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l462_46208

-- Problem 1: 1 / 0.25 = 4
theorem problem1 : 1 / 0.25 = 4 :=
by sorry

-- Problem 2: 0.25 / 0.1 = 2.5
theorem problem2 : 0.25 / 0.1 = 2.5 :=
by sorry

-- Problem 3: 1.2 / 1.2 = 1
theorem problem3 : 1.2 / 1.2 = 1 :=
by sorry

-- Problem 4: 4.01 * 1 = 4.01
theorem problem4 : 4.01 * 1 = 4.01 :=
by sorry

-- Problem 5: 0.25 * 2 = 0.5
theorem problem5 : 0.25 * 2 = 0.5 :=
by sorry

-- Problem 6: 0 / 2.76 = 0
theorem problem6 : 0 / 2.76 = 0 :=
by sorry

-- Problem 7: 0.8 / 1.25 = 0.64
theorem problem7 : 0.8 / 1.25 = 0.64 :=
by sorry

-- Problem 8: 3.5 * 2.7 = 9.45
theorem problem8 : 3.5 * 2.7 = 9.45 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l462_46208


namespace NUMINAMATH_GPT_fifty_percent_of_x_l462_46276

variable (x : ℝ)

theorem fifty_percent_of_x (h : 0.40 * x = 160) : 0.50 * x = 200 :=
by
  sorry

end NUMINAMATH_GPT_fifty_percent_of_x_l462_46276


namespace NUMINAMATH_GPT_candy_distribution_impossible_l462_46242

theorem candy_distribution_impossible :
  ∀ (candies : Fin 6 → ℕ),
  (candies 0 = 0 ∧ candies 1 = 1 ∧ candies 2 = 0 ∧ candies 3 = 0 ∧ candies 4 = 0 ∧ candies 5 = 1) →
  (∀ t, ∃ i, (i < 6) ∧ candies ((i+t)%6) = candies ((i+t+1)%6)) →
  ∃ (i : Fin 6), candies i ≠ candies ((i + 1) % 6) :=
by
  sorry

end NUMINAMATH_GPT_candy_distribution_impossible_l462_46242


namespace NUMINAMATH_GPT_ratio_elephants_to_others_l462_46261

theorem ratio_elephants_to_others (L P E : ℕ) (h1 : L = 2 * P) (h2 : L = 200) (h3 : L + P + E = 450) :
  E / (L + P) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_elephants_to_others_l462_46261


namespace NUMINAMATH_GPT_solutionY_materialB_correct_l462_46286

open Real

-- Definitions and conditions from step a
def solutionX_materialA : ℝ := 0.20
def solutionX_materialB : ℝ := 0.80
def solutionY_materialA : ℝ := 0.30
def mixture_materialA : ℝ := 0.22
def solutionX_in_mixture : ℝ := 0.80
def solutionY_in_mixture : ℝ := 0.20

-- The conjecture to prove
theorem solutionY_materialB_correct (B_Y : ℝ) 
  (h1 : solutionX_materialA = 0.20)
  (h2 : solutionX_materialB = 0.80) 
  (h3 : solutionY_materialA = 0.30) 
  (h4 : mixture_materialA = 0.22)
  (h5 : solutionX_in_mixture = 0.80)
  (h6 : solutionY_in_mixture = 0.20) :
  B_Y = 1 - solutionY_materialA := by 
  sorry

end NUMINAMATH_GPT_solutionY_materialB_correct_l462_46286


namespace NUMINAMATH_GPT_total_amount_paid_l462_46262

/-- Conditions -/
def days_in_may : Nat := 31
def rate_per_day : ℚ := 0.5
def days_book1_borrowed : Nat := 20
def days_book2_borrowed : Nat := 31
def days_book3_borrowed : Nat := 31

/-- Question and Proof -/
theorem total_amount_paid : rate_per_day * (days_book1_borrowed + days_book2_borrowed + days_book3_borrowed) = 41 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l462_46262


namespace NUMINAMATH_GPT_common_integer_solutions_l462_46245

theorem common_integer_solutions
    (y : ℤ)
    (h1 : -4 * y ≥ 2 * y + 10)
    (h2 : -3 * y ≤ 15)
    (h3 : -5 * y ≥ 3 * y + 24)
    (h4 : y ≤ -1) :
  y = -3 ∨ y = -4 ∨ y = -5 :=
by 
  sorry

end NUMINAMATH_GPT_common_integer_solutions_l462_46245


namespace NUMINAMATH_GPT_div_recurring_decimal_l462_46238

def recurringDecimalToFraction (q : ℚ) (h : q = 36/99) : ℚ := by
  sorry

theorem div_recurring_decimal : 12 / recurringDecimalToFraction 0.36 sorry = 33 :=
by
  sorry

end NUMINAMATH_GPT_div_recurring_decimal_l462_46238


namespace NUMINAMATH_GPT_geometric_sequence_product_l462_46253

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product 
  (h : is_geometric_sequence a r)
  (h_cond : a 4 * a 6 = 10) :
  a 2 * a 8 = 10 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l462_46253


namespace NUMINAMATH_GPT_find_f_six_l462_46248

theorem find_f_six (f : ℕ → ℤ) (h : ∀ (x : ℕ), f (x + 1) = x^2 - 4) : f 6 = 21 :=
by
sorry

end NUMINAMATH_GPT_find_f_six_l462_46248


namespace NUMINAMATH_GPT_discriminant_eq_perfect_square_l462_46228

variables (a b c t : ℝ)

-- Conditions
axiom a_nonzero : a ≠ 0
axiom t_root : a * t^2 + b * t + c = 0

-- Goal
theorem discriminant_eq_perfect_square :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 :=
by
  -- Conditions and goal are stated, proof to be filled.
  sorry

end NUMINAMATH_GPT_discriminant_eq_perfect_square_l462_46228


namespace NUMINAMATH_GPT_arithmetic_mean_fraction_l462_46251

theorem arithmetic_mean_fraction :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  let c := (9 : ℚ) / 10
  (1 / 3) * (a + b + c) = 149 / 180 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_fraction_l462_46251


namespace NUMINAMATH_GPT_coating_profit_l462_46258

theorem coating_profit (x y : ℝ) (h1 : 0.6 * x + 0.9 * (150 - x) ≤ 120)
  (h2 : 0.7 * x + 0.4 * (150 - x) ≤ 90) :
  (50 ≤ x ∧ x ≤ 100) → (y = -50 * x + 75000) → (x = 50 → y = 72500) :=
by
  intros hx hy hx_val
  sorry

end NUMINAMATH_GPT_coating_profit_l462_46258


namespace NUMINAMATH_GPT_value_of_n_l462_46288

theorem value_of_n (n : ℝ) : (∀ (x y : ℝ), x^2 + y^2 - 2 * n * x + 2 * n * y + 2 * n^2 - 8 = 0 → (x + 1)^2 + (y - 1)^2 = 2) → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_n_l462_46288


namespace NUMINAMATH_GPT_complete_work_in_12_days_l462_46291

def Ravi_rate_per_day : ℚ := 1 / 24
def Prakash_rate_per_day : ℚ := 1 / 40
def Suresh_rate_per_day : ℚ := 1 / 60
def combined_rate_per_day : ℚ := Ravi_rate_per_day + Prakash_rate_per_day + Suresh_rate_per_day

theorem complete_work_in_12_days : 
  (1 / combined_rate_per_day) = 12 := 
by
  sorry

end NUMINAMATH_GPT_complete_work_in_12_days_l462_46291


namespace NUMINAMATH_GPT_hyperbola_condition_l462_46226

theorem hyperbola_condition (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1 → (-1 < k ∧ k < 1)) ∧ 
  ((-1 < k ∧ k < 1) → ∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_condition_l462_46226


namespace NUMINAMATH_GPT_ellipse_standard_equation_l462_46272

theorem ellipse_standard_equation (a b : ℝ) (h1 : 2 * a = 2 * (2 * b)) (h2 : (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∨ (2, 0) ∈ {p : ℝ × ℝ | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}) :
  (∃ a b : ℝ, (a > b ∧ a > 0 ∧ b > 0 ∧ (2 * a = 2 * (2 * b)) ∧ (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧ (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} → (x^2 / 4 + y^2 / 1 = 1)) ∨ (x^2 / 16 + y^2 / 4 = 1))) :=
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l462_46272
