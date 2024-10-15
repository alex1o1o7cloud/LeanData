import Mathlib

namespace NUMINAMATH_CALUDE_bargain_bin_books_l2053_205397

theorem bargain_bin_books (initial_books : ℕ) : 
  initial_books - 3 + 10 = 11 → initial_books = 4 := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l2053_205397


namespace NUMINAMATH_CALUDE_rational_function_sum_l2053_205337

/-- A rational function with specific properties -/
def RationalFunction (p q : ℝ → ℝ) : Prop :=
  (∃ k a : ℝ, q = fun x ↦ k * (x + 3) * (x - 1) * (x - a)) ∧
  (∃ b : ℝ, p = fun x ↦ b * x + 2) ∧
  q 0 = -2

/-- The theorem statement -/
theorem rational_function_sum (p q : ℝ → ℝ) :
  RationalFunction p q →
  ∃! p, p + q = fun x ↦ (1/3) * x^3 - (1/3) * x^2 + (11/3) * x + 4 :=
by sorry

end NUMINAMATH_CALUDE_rational_function_sum_l2053_205337


namespace NUMINAMATH_CALUDE_weight_of_3_moles_HBrO3_l2053_205354

/-- The molecular weight of a single HBrO3 molecule in g/mol -/
def molecular_weight_HBrO3 : ℝ :=
  1.01 + 79.90 + 3 * 16.00

/-- The weight of 3 moles of HBrO3 in grams -/
def weight_3_moles_HBrO3 : ℝ :=
  3 * molecular_weight_HBrO3

theorem weight_of_3_moles_HBrO3 :
  weight_3_moles_HBrO3 = 386.73 := by sorry

end NUMINAMATH_CALUDE_weight_of_3_moles_HBrO3_l2053_205354


namespace NUMINAMATH_CALUDE_negative_roots_equation_reciprocal_roots_equation_l2053_205340

-- Part 1
theorem negative_roots_equation (r1 r2 : ℝ) :
  r1^2 + 3*r1 - 2 = 0 ∧ r2^2 + 3*r2 - 2 = 0 →
  (-r1)^2 - 3*(-r1) - 2 = 0 ∧ (-r2)^2 - 3*(-r2) - 2 = 0 := by sorry

-- Part 2
theorem reciprocal_roots_equation (a b c r1 r2 : ℝ) :
  a ≠ 0 ∧ r1 ≠ r2 ∧ r1 ≠ 0 ∧ r2 ≠ 0 ∧
  a*r1^2 - b*r1 + c = 0 ∧ a*r2^2 - b*r2 + c = 0 →
  c*(1/r1)^2 - b*(1/r1) + a = 0 ∧ c*(1/r2)^2 - b*(1/r2) + a = 0 := by sorry

end NUMINAMATH_CALUDE_negative_roots_equation_reciprocal_roots_equation_l2053_205340


namespace NUMINAMATH_CALUDE_parallelogram_area_l2053_205359

theorem parallelogram_area (base height : ℝ) (h1 : base = 60) (h2 : height = 16) :
  base * height = 960 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2053_205359


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_one_l2053_205302

theorem min_sum_reciprocal_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  x + y ≥ 4 ∧ (x + y = 4 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_one_l2053_205302


namespace NUMINAMATH_CALUDE_max_girls_for_five_boys_valid_arrangement_l2053_205396

/-- The maximum number of girls that can be arranged in a "Mathematical Ballet" -/
def max_girls (num_boys : ℕ) : ℕ :=
  (num_boys.choose 2) * 2

/-- Theorem stating the maximum number of girls for 5 boys -/
theorem max_girls_for_five_boys :
  max_girls 5 = 20 := by
  sorry

/-- Theorem proving the validity of the arrangement -/
theorem valid_arrangement (num_boys : ℕ) (num_girls : ℕ) :
  num_girls ≤ max_girls num_boys →
  ∃ (boy_positions : Fin num_boys → ℝ × ℝ)
    (girl_positions : Fin num_girls → ℝ × ℝ),
    ∀ (g : Fin num_girls),
      ∃ (b1 b2 : Fin num_boys),
        b1 ≠ b2 ∧
        dist (girl_positions g) (boy_positions b1) = 5 ∧
        dist (girl_positions g) (boy_positions b2) = 5 ∧
        ∀ (b : Fin num_boys),
          b ≠ b1 ∧ b ≠ b2 →
          dist (girl_positions g) (boy_positions b) ≠ 5 := by
  sorry


end NUMINAMATH_CALUDE_max_girls_for_five_boys_valid_arrangement_l2053_205396


namespace NUMINAMATH_CALUDE_choose_two_from_four_with_repetition_l2053_205313

/-- The number of ways to choose r items from n items with repetition allowed -/
def combinationsWithRepetition (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose (n + r - 1) r

theorem choose_two_from_four_with_repetition :
  combinationsWithRepetition 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_four_with_repetition_l2053_205313


namespace NUMINAMATH_CALUDE_hotel_bill_friends_count_prove_hotel_bill_friends_count_l2053_205388

theorem hotel_bill_friends_count : ℕ → Prop :=
  fun total_friends =>
    let standard_pay := 100
    let extra_pay := 100
    let actual_extra_pay := 220
    let standard_payers := 5
    let total_bill := standard_payers * standard_pay + extra_pay
    let share_per_friend := total_bill / total_friends
    total_friends = standard_payers + 1 ∧
    share_per_friend * total_friends = total_bill ∧
    share_per_friend + extra_pay = actual_extra_pay

theorem prove_hotel_bill_friends_count : 
  ∃ (n : ℕ), hotel_bill_friends_count n ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_hotel_bill_friends_count_prove_hotel_bill_friends_count_l2053_205388


namespace NUMINAMATH_CALUDE_problem_statement_l2053_205306

theorem problem_statement (x y z : ℝ) 
  (h1 : 2 * x - y - 2 * z - 6 = 0) 
  (h2 : x^2 + y^2 + z^2 ≤ 4) : 
  2 * x + y + z = 2/3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2053_205306


namespace NUMINAMATH_CALUDE_max_perimeter_of_rectangle_from_triangles_l2053_205338

theorem max_perimeter_of_rectangle_from_triangles :
  ∀ (L W : ℝ),
  L > 0 → W > 0 →
  L * W = 60 * (1/2 * 2 * 3) →
  2 * (L + W) ≤ 184 :=
by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_rectangle_from_triangles_l2053_205338


namespace NUMINAMATH_CALUDE_triangle_sin_b_l2053_205373

theorem triangle_sin_b (A B C : Real) (AC BC : Real) (h1 : AC = 2) (h2 : BC = 3) (h3 : Real.cos A = 3/5) :
  Real.sin B = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_b_l2053_205373


namespace NUMINAMATH_CALUDE_m_range_l2053_205364

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≠ 0 ∨ ∃ y : ℝ, y ≠ x ∧ y^2 + m*y + 1 = 0 → False) →
  (∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0) →
  1 < m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2053_205364


namespace NUMINAMATH_CALUDE_athlete_score_comparison_l2053_205344

theorem athlete_score_comparison 
  (p₁ p₂ p₃ : ℝ) 
  (hp₁ : p₁ > 0) 
  (hp₂ : p₂ > 0) 
  (hp₃ : p₃ > 0) : 
  (16/25) * p₁ + (9/25) * p₂ + (4/15) * p₃ > 
  (16/25) * p₁ + (1/4) * p₂ + (27/128) * p₃ :=
sorry

end NUMINAMATH_CALUDE_athlete_score_comparison_l2053_205344


namespace NUMINAMATH_CALUDE_symmetric_angle_set_l2053_205389

/-- Given α = π/6 and the terminal side of angle β is symmetric to the terminal side of α
    with respect to the line y=x, prove that the set of all possible values for β
    is {β | β = 2kπ + π/3, k ∈ ℤ}. -/
theorem symmetric_angle_set (α β : Real) (k : ℤ) :
  α = π / 6 →
  (∃ (f : Real → Real), f β = α ∧ f (π / 4) = π / 4 ∧ ∀ x, f (f x) = x) →
  (β = 2 * π * k + π / 3) :=
sorry

end NUMINAMATH_CALUDE_symmetric_angle_set_l2053_205389


namespace NUMINAMATH_CALUDE_specific_lot_volume_l2053_205328

/-- The volume of a rectangular lot -/
def lot_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem stating that the volume of the specific lot is 1600 cubic meters -/
theorem specific_lot_volume : lot_volume 40 20 2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_specific_lot_volume_l2053_205328


namespace NUMINAMATH_CALUDE_randy_pictures_l2053_205331

/-- Given that Peter drew 8 pictures, Quincy drew 20 more pictures than Peter,
    and they drew 41 pictures altogether, prove that Randy drew 5 pictures. -/
theorem randy_pictures (peter_pictures : ℕ) (quincy_pictures : ℕ) (total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + 5 := by
  sorry

end NUMINAMATH_CALUDE_randy_pictures_l2053_205331


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2053_205380

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : a^2 = 16) 
  (h2 : b^3 = -27) 
  (h3 : |a - b| = a - b) : 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2053_205380


namespace NUMINAMATH_CALUDE_power_sqrt_abs_calculation_l2053_205378

theorem power_sqrt_abs_calculation : 2^0 + Real.sqrt 9 - |(-4)| = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sqrt_abs_calculation_l2053_205378


namespace NUMINAMATH_CALUDE_people_dislike_radio_and_music_l2053_205330

theorem people_dislike_radio_and_music
  (total_people : ℕ)
  (radio_dislike_percent : ℚ)
  (music_dislike_percent : ℚ)
  (h_total : total_people = 1500)
  (h_radio : radio_dislike_percent = 40 / 100)
  (h_music : music_dislike_percent = 15 / 100) :
  (total_people : ℚ) * radio_dislike_percent * music_dislike_percent = 90 := by
  sorry

end NUMINAMATH_CALUDE_people_dislike_radio_and_music_l2053_205330


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l2053_205343

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l2053_205343


namespace NUMINAMATH_CALUDE_basketball_score_problem_l2053_205333

theorem basketball_score_problem (total_points hawks_points eagles_points : ℕ) : 
  total_points = 50 →
  hawks_points = eagles_points + 6 →
  hawks_points + eagles_points = total_points →
  eagles_points = 22 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_problem_l2053_205333


namespace NUMINAMATH_CALUDE_modular_inverse_15_l2053_205310

theorem modular_inverse_15 :
  (¬ ∃ x : ℤ, (15 * x) % 1105 = 1) ∧
  (∃ x : ℤ, (15 * x) % 221 = 1) ∧
  ((15 * 59) % 221 = 1) := by
sorry

end NUMINAMATH_CALUDE_modular_inverse_15_l2053_205310


namespace NUMINAMATH_CALUDE_money_distribution_l2053_205311

theorem money_distribution (total : ℕ) (p q r : ℕ) : 
  total = 9000 →
  p + q + r = total →
  r = 2 * (p + q) / 3 →
  r = 3600 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2053_205311


namespace NUMINAMATH_CALUDE_function_value_proof_l2053_205361

/-- Given a function f(x, z) = 2x^2 + y - z where f(2, 3) = 100, prove that f(5, 7) = 138 -/
theorem function_value_proof (y : ℝ) : 
  let f : ℝ → ℝ → ℝ := λ x z ↦ 2 * x^2 + y - z
  (f 2 3 = 100) → (f 5 7 = 138) := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l2053_205361


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2053_205329

/-- Proves that given a total of 8000 votes and a loss margin of 4000 votes,
    the percentage of votes received by the losing candidate is 25%. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h_total : total_votes = 8000)
  (h_margin : loss_margin = 4000) :
  (total_votes - loss_margin) / total_votes * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2053_205329


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2053_205384

/-- Given a point P(-3, 2), its symmetric point P' with respect to the origin O has coordinates (3, -2). -/
theorem symmetric_point_wrt_origin :
  let P : ℝ × ℝ := (-3, 2)
  let P' : ℝ × ℝ := (3, -2)
  let symmetric_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_wrt_origin P = P' := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2053_205384


namespace NUMINAMATH_CALUDE_triangle_problem_l2053_205314

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = Real.sqrt 7 →
  b = 2 →
  A = 60 * π / 180 →  -- Convert 60° to radians
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Sine law
  a / Real.sin A = b / Real.sin B →
  -- Cosine law
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Conclusions
  Real.sin B = Real.sqrt 21 / 7 ∧ c = 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l2053_205314


namespace NUMINAMATH_CALUDE_intersection_of_complements_l2053_205326

def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem intersection_of_complements :
  (U \ A) ∩ (U \ B) = {0, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_complements_l2053_205326


namespace NUMINAMATH_CALUDE_inverse_function_sum_l2053_205342

/-- Given two real numbers a and b, and functions f and f_inv defined as follows:
    f(x) = ax + 2b
    f_inv(x) = bx + 2a
    If f and f_inv are true functional inverses, then a + 2b = -3 -/
theorem inverse_function_sum (a b : ℝ) 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + 2 * b)
  (h2 : ∀ x, f_inv x = b * x + 2 * a)
  (h3 : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f) :
  a + 2 * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l2053_205342


namespace NUMINAMATH_CALUDE_kenya_peanuts_l2053_205324

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_difference : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_difference = 48) :
  jose_peanuts + kenya_difference = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l2053_205324


namespace NUMINAMATH_CALUDE_abs_plus_exp_zero_equals_three_l2053_205377

theorem abs_plus_exp_zero_equals_three :
  |(-2 : ℝ)| + (3 - Real.sqrt 5) ^ (0 : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_exp_zero_equals_three_l2053_205377


namespace NUMINAMATH_CALUDE_modulus_of_z_l2053_205325

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2053_205325


namespace NUMINAMATH_CALUDE_debbys_percentage_share_l2053_205332

theorem debbys_percentage_share (total : ℝ) (maggies_share : ℝ) 
  (h1 : total = 6000)
  (h2 : maggies_share = 4500) :
  (total - maggies_share) / total * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_debbys_percentage_share_l2053_205332


namespace NUMINAMATH_CALUDE_solution_difference_l2053_205335

theorem solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^(1/3 : ℝ) = -3 ∧ 9 - x₁^2 / 4 = (-3)^3) ∧
  (x₂^(1/3 : ℝ) = -3 ∧ 9 - x₂^2 / 4 = (-3)^3) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 24 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l2053_205335


namespace NUMINAMATH_CALUDE_least_n_divisibility_l2053_205367

theorem least_n_divisibility (n : ℕ) : n = 5 ↔ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2*n → 
    (∃ m : ℕ, m ≥ 1 ∧ m ≤ 2*n ∧ (n^2 - n + m) % m = 0) ∧ 
    (∃ l : ℕ, l ≥ 1 ∧ l ≤ 2*n ∧ (n^2 - n + l) % l ≠ 0)) ∧
  (∀ m : ℕ, m < n → 
    ¬(∀ k : ℕ, 1 ≤ k ∧ k ≤ 2*m → 
      (∃ p : ℕ, p ≥ 1 ∧ p ≤ 2*m ∧ (m^2 - m + p) % p = 0) ∧ 
      (∃ q : ℕ, q ≥ 1 ∧ q ≤ 2*m ∧ (m^2 - m + q) % q ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisibility_l2053_205367


namespace NUMINAMATH_CALUDE_remainder_171_pow_2147_mod_52_l2053_205322

theorem remainder_171_pow_2147_mod_52 : ∃ k : ℕ, 171^2147 = 52 * k + 7 := by sorry

end NUMINAMATH_CALUDE_remainder_171_pow_2147_mod_52_l2053_205322


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2053_205356

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y a : ℝ) : ℝ := x^2 + y^2 + 2*a*x + 2*y + (a-1)^2

/-- Predicate to check if a point (x, y) is outside the circle -/
def is_outside_circle (x y a : ℝ) : Prop := circle_equation x y a > 0

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) : 
  is_outside_circle 0 0 a :=
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2053_205356


namespace NUMINAMATH_CALUDE_ghee_mixture_volume_l2053_205339

/-- Prove that the volume of a mixture of two brands of vegetable ghee is 4 liters -/
theorem ghee_mixture_volume :
  ∀ (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ),
    weight_a = 900 →
    weight_b = 700 →
    ratio_a = 3 →
    ratio_b = 2 →
    total_weight = 3280 →
    ∃ (volume_a volume_b : ℝ),
      volume_a / volume_b = ratio_a / ratio_b ∧
      weight_a * volume_a + weight_b * volume_b = total_weight ∧
      volume_a + volume_b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ghee_mixture_volume_l2053_205339


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2053_205376

theorem gcd_of_three_numbers : Nat.gcd 13847 (Nat.gcd 21353 34691) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2053_205376


namespace NUMINAMATH_CALUDE_inequality_proof_l2053_205385

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (1/x + 1/y + 1/z) - (x + y + z) ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2053_205385


namespace NUMINAMATH_CALUDE_equation_is_linear_one_var_l2053_205307

/-- Predicate to check if an expression is linear in one variable -/
def IsLinearOneVar (e : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, e x = a * x + b ∧ a ≠ 0

/-- The specific equation we're checking -/
def equation (x : ℝ) : ℝ := 3 - 2*x

/-- Theorem stating that our equation is linear in one variable -/
theorem equation_is_linear_one_var : IsLinearOneVar equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_one_var_l2053_205307


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l2053_205319

theorem dress_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  d * ((100 - x) / 100) * 0.5 = 0.225 * d → x = 55 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l2053_205319


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sinusoid_l2053_205371

open Real

theorem axis_of_symmetry_sinusoid (x : ℝ) :
  let f := fun x => Real.sin (1/2 * x - π/6)
  ∃ k : ℤ, f (4*π/3 + x) = f (4*π/3 - x) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sinusoid_l2053_205371


namespace NUMINAMATH_CALUDE_continuous_function_image_interval_l2053_205320

open Set

theorem continuous_function_image_interval 
  (f : ℝ → ℝ) (hf : Continuous f) (a b : ℝ) (hab : a < b)
  (ha : a ∈ Set.range f) (hb : b ∈ Set.range f) :
  ∃ (I : Set ℝ), ∃ (s t : ℝ), I = Icc s t ∧ f '' I = Icc a b := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_image_interval_l2053_205320


namespace NUMINAMATH_CALUDE_circle_radius_spherical_coordinates_l2053_205312

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) is √2/2 --/
theorem circle_radius_spherical_coordinates :
  let r := Real.sqrt ((Real.sin (π/4))^2 + (Real.cos (π/4))^2)
  r = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_spherical_coordinates_l2053_205312


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2053_205350

theorem max_product_sum_300 : 
  (∃ a b : ℤ, a + b = 300 ∧ ∀ x y : ℤ, x + y = 300 → x * y ≤ a * b) ∧ 
  (∃ a b : ℤ, a + b = 300 ∧ a * b = 22500) := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2053_205350


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2053_205393

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/8 < 1 → x ≤ 5 ∧
  ((1/4 : ℚ) + (5 : ℚ)/8 < 1 ∧ ∀ y : ℤ, y > 5 → (1/4 : ℚ) + (y : ℚ)/8 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2053_205393


namespace NUMINAMATH_CALUDE_car_original_price_l2053_205365

/-- Given a car sold at a 15% loss and then resold with a 20% gain for Rs. 54000,
    prove that the original cost price of the car was Rs. 52,941.18 (rounded to two decimal places). -/
theorem car_original_price (loss_percent : ℝ) (gain_percent : ℝ) (final_price : ℝ) :
  loss_percent = 15 →
  gain_percent = 20 →
  final_price = 54000 →
  ∃ (original_price : ℝ),
    (1 - loss_percent / 100) * original_price * (1 + gain_percent / 100) = final_price ∧
    (round (original_price * 100) / 100 : ℝ) = 52941.18 := by
  sorry

end NUMINAMATH_CALUDE_car_original_price_l2053_205365


namespace NUMINAMATH_CALUDE_concentric_circles_angle_l2053_205392

theorem concentric_circles_angle (r₁ r₂ r₃ : ℝ) (shaded_area unshaded_area : ℝ) (θ : ℝ) : 
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_area = (3/4) * unshaded_area →
  shaded_area + unshaded_area = 29 * π →
  shaded_area = 11 * θ + 9 * π →
  θ = 6 * π / 77 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_angle_l2053_205392


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l2053_205386

/-- The number of cakes Baker initially made -/
def initial_cakes : ℕ := 48

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 44

/-- Theorem: Baker still has 4 cakes -/
theorem baker_remaining_cakes : initial_cakes - sold_cakes = 4 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l2053_205386


namespace NUMINAMATH_CALUDE_sqrt_720_simplified_l2053_205304

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplified_l2053_205304


namespace NUMINAMATH_CALUDE_congruence_solution_l2053_205301

theorem congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -437 [ZMOD 10] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2053_205301


namespace NUMINAMATH_CALUDE_labor_market_effects_l2053_205352

-- Define the labor market for doctors
structure LaborMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibriumWage : ℝ  -- Equilibrium wage

-- Define the commercial healthcare market
structure HealthcareMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibriumPrice : ℝ  -- Equilibrium price

-- Define the government policy
def governmentPolicy (minYears : ℕ) : Prop :=
  ∃ (requirement : ℕ), requirement ≥ minYears

-- Theorem statement
theorem labor_market_effects
  (initialMarket : LaborMarket)
  (initialHealthcare : HealthcareMarket)
  (policy : governmentPolicy 1)
  (newMarket : LaborMarket)
  (newHealthcare : HealthcareMarket) :
  (newMarket.equilibriumWage > initialMarket.equilibriumWage) ∧
  (newHealthcare.equilibriumPrice < initialHealthcare.equilibriumPrice) :=
sorry

end NUMINAMATH_CALUDE_labor_market_effects_l2053_205352


namespace NUMINAMATH_CALUDE_extremum_condition_l2053_205309

def f (a b x : ℝ) := x^3 - a*x^2 - b*x + a^2

theorem extremum_condition (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a + b = 7 := by sorry

end NUMINAMATH_CALUDE_extremum_condition_l2053_205309


namespace NUMINAMATH_CALUDE_triangle_third_side_l2053_205374

theorem triangle_third_side (a b : ℝ) (h₁ h₂ : ℝ) :
  a = 5 →
  b = 2 * Real.sqrt 6 →
  0 < h₁ →
  0 < h₂ →
  a * h₁ = b * h₂ →
  a + h₁ ≤ b + h₂ →
  ∃ c : ℝ, c * c = a * a + b * b ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2053_205374


namespace NUMINAMATH_CALUDE_smallest_three_digit_equal_sum_l2053_205360

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Proposition: 999 is the smallest three-digit number n such that
    Σ(n) = Σ(2n) = Σ(3n) = ... = Σ(n^2), where Σ(n) denotes the sum of the digits of n -/
theorem smallest_three_digit_equal_sum : 
  ∀ n : ℕ, 100 ≤ n → n < 999 → 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ sumOfDigits n ≠ sumOfDigits (k * n)) ∨
  sumOfDigits n ≠ sumOfDigits (n * n) :=
by sorry

#check smallest_three_digit_equal_sum

end NUMINAMATH_CALUDE_smallest_three_digit_equal_sum_l2053_205360


namespace NUMINAMATH_CALUDE_am_gm_inequality_l2053_205362

theorem am_gm_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l2053_205362


namespace NUMINAMATH_CALUDE_exactly_one_absent_probability_l2053_205341

theorem exactly_one_absent_probability (p_absent : ℝ) (h1 : p_absent = 1 / 20) :
  let p_present := 1 - p_absent
  2 * p_absent * p_present = 19 / 200 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_absent_probability_l2053_205341


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l2053_205321

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    a > 1 → b > 1 → c > 1 →
    (c ∣ (a * b + 1)) →
    (a ∣ (b * c + 1)) →
    (b ∣ (c * a + 1)) →
    ((a = 2 ∧ b = 3 ∧ c = 7) ∨
     (a = 2 ∧ b = 7 ∧ c = 3) ∨
     (a = 3 ∧ b = 2 ∧ c = 7) ∨
     (a = 3 ∧ b = 7 ∧ c = 2) ∨
     (a = 7 ∧ b = 2 ∧ c = 3) ∨
     (a = 7 ∧ b = 3 ∧ c = 2)) :=
by sorry


end NUMINAMATH_CALUDE_abc_divisibility_problem_l2053_205321


namespace NUMINAMATH_CALUDE_exists_negative_irrational_greater_than_neg_four_l2053_205369

theorem exists_negative_irrational_greater_than_neg_four :
  ∃ x : ℝ, x < 0 ∧ Irrational x ∧ -4 < x := by
sorry

end NUMINAMATH_CALUDE_exists_negative_irrational_greater_than_neg_four_l2053_205369


namespace NUMINAMATH_CALUDE_complex_exponent_calculation_l2053_205315

theorem complex_exponent_calculation : 3 * 3^6 - 9^60 / 9^58 + 4^3 = 2170 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponent_calculation_l2053_205315


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2053_205381

theorem root_difference_implies_k_value (k : ℝ) : 
  (∀ x₁ x₂, x₁^2 + k*x₁ + 10 = 0 → x₂^2 - k*x₂ + 10 = 0 → x₂ = x₁ + 3) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2053_205381


namespace NUMINAMATH_CALUDE_f_min_at_inv_e_l2053_205372

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem f_min_at_inv_e :
  ∀ x > 0, f (1 / Real.exp 1) ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_inv_e_l2053_205372


namespace NUMINAMATH_CALUDE_julia_total_food_expense_l2053_205308

/-- Represents the weekly food cost and number of weeks for an animal -/
structure AnimalExpense where
  weeklyFoodCost : ℕ
  numberOfWeeks : ℕ

/-- Calculates the total food expense for Julia's animals -/
def totalFoodExpense (animals : List AnimalExpense) : ℕ :=
  animals.map (fun a => a.weeklyFoodCost * a.numberOfWeeks) |>.sum

/-- The list of Julia's animals with their expenses -/
def juliaAnimals : List AnimalExpense := [
  ⟨15, 3⟩,  -- Parrot
  ⟨12, 5⟩,  -- Rabbit
  ⟨8, 2⟩,   -- Turtle
  ⟨5, 6⟩    -- Guinea pig
]

/-- Theorem stating that Julia's total food expense is $151 -/
theorem julia_total_food_expense :
  totalFoodExpense juliaAnimals = 151 := by
  sorry

end NUMINAMATH_CALUDE_julia_total_food_expense_l2053_205308


namespace NUMINAMATH_CALUDE_solve_euro_equation_l2053_205303

-- Define the operation €
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem solve_euro_equation : 
  ∀ x : ℝ, euro x (euro 4 5) = 720 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_euro_equation_l2053_205303


namespace NUMINAMATH_CALUDE_count_divisors_3240_multiple_of_three_l2053_205345

/-- The number of positive divisors of 3240 that are multiples of 3 -/
def num_divisors_multiple_of_three : ℕ := 32

/-- The prime factorization of 3240 -/
def factorization_3240 : List (ℕ × ℕ) := [(2, 3), (3, 4), (5, 1)]

/-- A function to count the number of positive divisors of 3240 that are multiples of 3 -/
def count_divisors_multiple_of_three (factorization : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem count_divisors_3240_multiple_of_three :
  count_divisors_multiple_of_three factorization_3240 = num_divisors_multiple_of_three :=
sorry

end NUMINAMATH_CALUDE_count_divisors_3240_multiple_of_three_l2053_205345


namespace NUMINAMATH_CALUDE_bens_debtor_payment_l2053_205398

/-- Calculates the amount paid by Ben's debtor given his financial transactions -/
theorem bens_debtor_payment (initial_amount cheque_amount maintenance_cost final_amount : ℕ) : 
  initial_amount = 2000 ∧ 
  cheque_amount = 600 ∧ 
  maintenance_cost = 1200 ∧ 
  final_amount = 1000 → 
  final_amount = initial_amount - cheque_amount - maintenance_cost + 800 := by
  sorry

#check bens_debtor_payment

end NUMINAMATH_CALUDE_bens_debtor_payment_l2053_205398


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2053_205399

/-- Given two circles in a 2D plane:
    Circle 1 with radius 3 and center (0, 0)
    Circle 2 with radius 5 and center (12, 0)
    The x-coordinate of the point where a line tangent to both circles
    intersects the x-axis (to the right of the origin) is 9/2. -/
theorem tangent_line_intersection (x : ℝ) : 
  (∃ (y : ℝ), (x^2 + y^2 = 3^2 ∧ ((x - 12)^2 + y^2 = 5^2))) → x = 9/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2053_205399


namespace NUMINAMATH_CALUDE_train_passing_length_l2053_205366

/-- The length of a train passing another train in opposite direction -/
theorem train_passing_length (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 50) (h2 : v2 = 62) (h3 : t = 9) :
  let relative_speed := (v1 + v2) * (1000 / 3600)
  let train_length := relative_speed * t
  ∃ ε > 0, |train_length - 280| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_passing_length_l2053_205366


namespace NUMINAMATH_CALUDE_p_plus_q_value_l2053_205383

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 12*p^2 + 25*p - 75 = 0)
  (hq : 10*q^3 - 75*q^2 - 375*q + 3750 = 0) : 
  p + q = -5/2 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_value_l2053_205383


namespace NUMINAMATH_CALUDE_max_sum_constrained_l2053_205391

theorem max_sum_constrained (x y : ℝ) : 
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l2053_205391


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l2053_205327

theorem min_k_for_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ k : ℝ, ∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (5 * x + y)) ↔
  (∃ k : ℝ, k ≥ Real.sqrt 30 / 5 ∧ 
    ∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (5 * x + y)) :=
by sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l2053_205327


namespace NUMINAMATH_CALUDE_option_d_most_suitable_for_comprehensive_survey_l2053_205347

/-- Represents a survey option -/
inductive SurveyOption
| A : SurveyOption  -- Investigating the service life of a batch of infrared thermometers
| B : SurveyOption  -- Investigating the travel methods of the people of Henan during the Spring Festival
| C : SurveyOption  -- Investigating the viewership of the Henan TV program "Li Yuan Chun"
| D : SurveyOption  -- Investigating the heights of all classmates

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  precision : ℝ

/-- Defines what makes a survey suitable for a comprehensive survey -/
def is_suitable_for_comprehensive_survey (s : SurveyCharacteristics) : Prop :=
  s.population_size ≤ 1000 ∧ s.precision ≥ 0.99

/-- Associates survey options with their characteristics -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.A => ⟨10000, 0.9⟩
| SurveyOption.B => ⟨20000000, 0.8⟩
| SurveyOption.C => ⟨5000000, 0.85⟩
| SurveyOption.D => ⟨50, 0.99⟩

/-- Theorem: Option D is the most suitable for a comprehensive survey -/
theorem option_d_most_suitable_for_comprehensive_survey :
  ∀ (o : SurveyOption), o ≠ SurveyOption.D →
    is_suitable_for_comprehensive_survey (survey_characteristics SurveyOption.D) ∧
    ¬is_suitable_for_comprehensive_survey (survey_characteristics o) :=
by sorry


end NUMINAMATH_CALUDE_option_d_most_suitable_for_comprehensive_survey_l2053_205347


namespace NUMINAMATH_CALUDE_interest_rate_increase_l2053_205357

/-- Proves that if an interest rate increases by 10 percent to become 11 percent,
    the original interest rate was 10 percent. -/
theorem interest_rate_increase (original_rate : ℝ) : 
  (original_rate * 1.1 = 0.11) → (original_rate = 0.1) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_increase_l2053_205357


namespace NUMINAMATH_CALUDE_smallest_x_for_quadratic_inequality_l2053_205355

theorem smallest_x_for_quadratic_inequality :
  ∃ x₀ : ℝ, x₀ = 3 ∧
  (∀ x : ℝ, x^2 - 8*x + 15 ≤ 0 → x ≥ x₀) ∧
  (x₀^2 - 8*x₀ + 15 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_quadratic_inequality_l2053_205355


namespace NUMINAMATH_CALUDE_orange_groups_indeterminate_philips_collection_valid_philips_orange_groups_indeterminate_l2053_205379

/-- Represents a fruit collection with oranges and bananas -/
structure FruitCollection where
  oranges : ℕ
  bananas : ℕ
  banana_groups : ℕ
  bananas_per_group : ℕ

/-- Predicate to check if the banana distribution is valid -/
def valid_banana_distribution (fc : FruitCollection) : Prop :=
  fc.bananas = fc.banana_groups * fc.bananas_per_group

/-- Theorem stating that the number of orange groups cannot be determined -/
theorem orange_groups_indeterminate (fc : FruitCollection) 
  (h1 : fc.oranges > 0)
  (h2 : valid_banana_distribution fc) :
  ¬ ∃ (orange_groups : ℕ), orange_groups > 0 ∧ ∀ (oranges_per_group : ℕ), fc.oranges = orange_groups * oranges_per_group :=
by
  sorry

/-- Philip's fruit collection -/
def philips_collection : FruitCollection :=
  { oranges := 87
  , bananas := 290
  , banana_groups := 2
  , bananas_per_group := 145 }

/-- Proof that Philip's collection satisfies the conditions -/
theorem philips_collection_valid :
  valid_banana_distribution philips_collection :=
by
  sorry

/-- Application of the theorem to Philip's collection -/
theorem philips_orange_groups_indeterminate :
  ¬ ∃ (orange_groups : ℕ), orange_groups > 0 ∧ ∀ (oranges_per_group : ℕ), philips_collection.oranges = orange_groups * oranges_per_group :=
by
  apply orange_groups_indeterminate
  · simp [philips_collection]
  · exact philips_collection_valid

end NUMINAMATH_CALUDE_orange_groups_indeterminate_philips_collection_valid_philips_orange_groups_indeterminate_l2053_205379


namespace NUMINAMATH_CALUDE_same_height_time_l2053_205395

/-- Represents the height of a ball as a function of time -/
def ball_height (a : ℝ) (h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2)^2 + h

theorem same_height_time :
  ∀ (a : ℝ) (h : ℝ),
  a ≠ 0 →
  ∃ (t : ℝ),
  t = 2.2 ∧
  ball_height a h t = ball_height a h (t - 2) :=
sorry

end NUMINAMATH_CALUDE_same_height_time_l2053_205395


namespace NUMINAMATH_CALUDE_solve_equation_l2053_205317

theorem solve_equation : ∃ x : ℝ, 0.5 * x + (0.3 * 0.2) = 0.26 ∧ x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2053_205317


namespace NUMINAMATH_CALUDE_fraction_modification_l2053_205370

theorem fraction_modification (p q r s x : ℚ) : 
  p ≠ q → q ≠ 0 → p = 3 → q = 5 → r = 7 → s = 9 → (p + x) / (q - x) = r / s → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_l2053_205370


namespace NUMINAMATH_CALUDE_stocking_price_calculation_l2053_205394

/-- The original price of a stocking before discount -/
def original_price : ℝ := 122.22

/-- The number of stockings ordered -/
def num_stockings : ℕ := 9

/-- The discount rate applied to the stockings -/
def discount_rate : ℝ := 0.1

/-- The cost of monogramming per stocking -/
def monogram_cost : ℝ := 5

/-- The total cost after discount and including monogramming -/
def total_cost : ℝ := 1035

/-- Theorem stating that the calculated original price satisfies the given conditions -/
theorem stocking_price_calculation :
  total_cost = num_stockings * (original_price * (1 - discount_rate) + monogram_cost) :=
by sorry

end NUMINAMATH_CALUDE_stocking_price_calculation_l2053_205394


namespace NUMINAMATH_CALUDE_net_change_in_cards_l2053_205300

def sold_cards : ℤ := 27
def received_cards : ℤ := 41
def bought_cards : ℤ := 20

theorem net_change_in_cards : -sold_cards + received_cards + bought_cards = 34 := by
  sorry

end NUMINAMATH_CALUDE_net_change_in_cards_l2053_205300


namespace NUMINAMATH_CALUDE_composite_representation_l2053_205382

theorem composite_representation (n : ℕ) (h1 : n > 3) (h2 : ¬ Nat.Prime n) :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a * b + b * c + c * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_composite_representation_l2053_205382


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_100011_l2053_205351

theorem smallest_six_digit_divisible_by_100011 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 → n % 100011 = 0 → n ≥ 100011 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_100011_l2053_205351


namespace NUMINAMATH_CALUDE_negative_a_exponent_division_l2053_205348

theorem negative_a_exponent_division (a : ℝ) : (-a)^6 / (-a)^3 = -a^3 := by sorry

end NUMINAMATH_CALUDE_negative_a_exponent_division_l2053_205348


namespace NUMINAMATH_CALUDE_max_value_theorem_l2053_205346

theorem max_value_theorem (x : ℝ) (h : x < -3) :
  x + 2 / (x + 3) ≤ -2 * Real.sqrt 2 - 3 ∧
  ∃ y, y < -3 ∧ y + 2 / (y + 3) = -2 * Real.sqrt 2 - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2053_205346


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2053_205323

theorem quadratic_equation_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   a * x₁^2 - (3*a + 1) * x₁ + 2*(a + 1) = 0 ∧
   a * x₂^2 - (3*a + 1) * x₂ + 2*(a + 1) = 0 ∧
   x₁ - x₁*x₂ + x₂ = 1 - a) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2053_205323


namespace NUMINAMATH_CALUDE_computer_room_arrangements_l2053_205358

/-- The number of different computer rooms -/
def n : ℕ := 6

/-- The minimum number of rooms that must be open -/
def k : ℕ := 2

/-- The number of arrangements for opening at least k out of n rooms -/
def num_arrangements (n k : ℕ) : ℕ := sorry

/-- Sum of combinations for opening 3 to 6 rooms, with 4 rooms counted twice -/
def sum_combinations (n : ℕ) : ℕ := 
  Nat.choose n 3 + 2 * Nat.choose n 4 + Nat.choose n 5 + Nat.choose n 6

/-- Total arrangements minus arrangements for 0 and 1 room -/
def power_minus_seven (n : ℕ) : ℕ := 2^n - 7

theorem computer_room_arrangements :
  num_arrangements n k = sum_combinations n ∧ 
  num_arrangements n k = power_minus_seven n := by sorry

end NUMINAMATH_CALUDE_computer_room_arrangements_l2053_205358


namespace NUMINAMATH_CALUDE_h_increasing_implies_k_range_l2053_205390

def h (k : ℝ) (x : ℝ) : ℝ := 2 * x - k

theorem h_increasing_implies_k_range (k : ℝ) :
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → h k x₁ < h k x₂) →
  k ∈ Set.Ici (-2) :=
sorry

end NUMINAMATH_CALUDE_h_increasing_implies_k_range_l2053_205390


namespace NUMINAMATH_CALUDE_quadratic_equation_with_irrational_root_l2053_205368

theorem quadratic_equation_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = 2 * Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -11 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_irrational_root_l2053_205368


namespace NUMINAMATH_CALUDE_sarah_homework_problem_l2053_205349

/-- The total number of problems Sarah has to complete given her homework assignments -/
def total_problems (math_pages reading_pages science_pages : ℕ) 
  (math_problems_per_page reading_problems_per_page science_problems_per_page : ℕ) : ℕ :=
  math_pages * math_problems_per_page + 
  reading_pages * reading_problems_per_page + 
  science_pages * science_problems_per_page

theorem sarah_homework_problem :
  total_problems 4 6 5 4 4 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sarah_homework_problem_l2053_205349


namespace NUMINAMATH_CALUDE_train_length_l2053_205318

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed : Real) (time : Real) (bridge_length : Real) :
  speed = 10 → -- 36 kmph converted to m/s
  time = 29.997600191984642 →
  bridge_length = 150 →
  speed * time - bridge_length = 149.97600191984642 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2053_205318


namespace NUMINAMATH_CALUDE_order_of_magnitude_l2053_205334

theorem order_of_magnitude (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (x : ℝ) (hx : x = Real.sqrt (a^2 + (b+c)^2))
  (y : ℝ) (hy : y = Real.sqrt (b^2 + (c+a)^2))
  (z : ℝ) (hz : z = Real.sqrt (c^2 + (a+b)^2)) :
  z > y ∧ y > x := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l2053_205334


namespace NUMINAMATH_CALUDE_march_first_is_monday_l2053_205316

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysBefore (match d with
    | DayOfWeek.Sunday => DayOfWeek.Saturday
    | DayOfWeek.Monday => DayOfWeek.Sunday
    | DayOfWeek.Tuesday => DayOfWeek.Monday
    | DayOfWeek.Wednesday => DayOfWeek.Tuesday
    | DayOfWeek.Thursday => DayOfWeek.Wednesday
    | DayOfWeek.Friday => DayOfWeek.Thursday
    | DayOfWeek.Saturday => DayOfWeek.Friday) n

theorem march_first_is_monday (march13 : DayOfWeek) 
    (h : march13 = DayOfWeek.Saturday) : 
    daysBefore march13 12 = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_march_first_is_monday_l2053_205316


namespace NUMINAMATH_CALUDE_bridget_initial_skittles_l2053_205363

/-- Proves that Bridget initially has 4 Skittles given the problem conditions. -/
theorem bridget_initial_skittles : 
  ∀ (bridget_initial henry_skittles bridget_final : ℕ),
  henry_skittles = 4 →
  bridget_final = bridget_initial + henry_skittles →
  bridget_final = 8 →
  bridget_initial = 4 := by sorry

end NUMINAMATH_CALUDE_bridget_initial_skittles_l2053_205363


namespace NUMINAMATH_CALUDE_ellipse_equation_equiv_standard_form_l2053_205336

/-- The equation of an ellipse given the sum of distances from any point to two fixed points -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10

/-- The standard form of an ellipse equation -/
def ellipse_standard_form (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 21 = 1

/-- Theorem stating that the ellipse equation is equivalent to its standard form -/
theorem ellipse_equation_equiv_standard_form :
  ∀ x y : ℝ, ellipse_equation x y ↔ ellipse_standard_form x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equiv_standard_form_l2053_205336


namespace NUMINAMATH_CALUDE_digit_sum_to_100_l2053_205387

def digits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def insert_operators (ds : List Nat) : List (Option Bool) :=
  [none, some true, some true, some false, some false, some false, some false, some true, some false]

def evaluate (ds : List Nat) (ops : List (Option Bool)) : Int :=
  match ds, ops with
  | [], _ => 0
  | d :: ds', none :: ops' => d * 100 + evaluate ds' ops'
  | d :: ds', some true :: ops' => d + evaluate ds' ops'
  | d :: ds', some false :: ops' => -d + evaluate ds' ops'
  | _, _ => 0

theorem digit_sum_to_100 :
  ∃ (ops : List (Option Bool)), evaluate digits ops = 100 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_to_100_l2053_205387


namespace NUMINAMATH_CALUDE_total_nailcutter_sounds_l2053_205375

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The number of customers -/
def number_of_customers : ℕ := 3

/-- The number of sounds produced per nail trimmed -/
def sounds_per_nail : ℕ := 1

/-- Theorem: The total number of nailcutter sounds produced for 3 customers is 60 -/
theorem total_nailcutter_sounds :
  nails_per_customer * number_of_customers * sounds_per_nail = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_nailcutter_sounds_l2053_205375


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2053_205353

theorem quadratic_roots_property (d e : ℝ) : 
  (2 * d^2 + 3 * d - 5 = 0) → 
  (2 * e^2 + 3 * e - 5 = 0) → 
  (d - 1) * (e - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2053_205353


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_of_squares_l2053_205305

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum_of_squares 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_sum : a 3 + a 5 = 5) 
  (h_prod : a 2 * a 6 = 4) : 
  a 3 ^ 2 + a 5 ^ 2 = 17 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_of_squares_l2053_205305
