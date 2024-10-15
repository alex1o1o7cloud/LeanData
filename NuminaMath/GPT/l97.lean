import Mathlib

namespace NUMINAMATH_GPT_circle_center_radius_l97_9748

theorem circle_center_radius :
  ∃ (h : ℝ × ℝ) (r : ℝ),
    (h = (1, -3)) ∧ (r = 2) ∧ ∀ x y : ℝ, 
    (x - h.1)^2 + (y - h.2)^2 = 4 → x^2 + y^2 - 2*x + 6*y + 6 = 0 :=
sorry

end NUMINAMATH_GPT_circle_center_radius_l97_9748


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l97_9739

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m : ℕ, (n = m → m > 0 → ∃ (a b : ℕ), n + 103 = 2^a * 5^b)) 
    ∧ n = 22 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l97_9739


namespace NUMINAMATH_GPT_union_M_N_l97_9704

namespace MyMath

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_M_N : M ∪ N = {-1, 1, 2} := sorry

end MyMath

end NUMINAMATH_GPT_union_M_N_l97_9704


namespace NUMINAMATH_GPT_total_amount_earned_l97_9733

-- Conditions
def avg_price_pair_rackets : ℝ := 9.8
def num_pairs_sold : ℕ := 60

-- Proof statement
theorem total_amount_earned :
  avg_price_pair_rackets * num_pairs_sold = 588 := by
    sorry

end NUMINAMATH_GPT_total_amount_earned_l97_9733


namespace NUMINAMATH_GPT_find_y_when_x_is_twelve_l97_9785

variables (x y k : ℝ)

theorem find_y_when_x_is_twelve
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = 12) :
  y = 56.25 :=
sorry

end NUMINAMATH_GPT_find_y_when_x_is_twelve_l97_9785


namespace NUMINAMATH_GPT_cos_sequence_next_coeff_sum_eq_28_l97_9789

theorem cos_sequence_next_coeff_sum_eq_28 (α : ℝ) :
  let u := 2 * Real.cos α
  2 * Real.cos (8 * α) = u ^ 8 - 8 * u ^ 6 + 20 * u ^ 4 - 16 * u ^ 2 + 2 → 
  8 + (-8) + 6 + 20 + 2 = 28 :=
by intros u; sorry

end NUMINAMATH_GPT_cos_sequence_next_coeff_sum_eq_28_l97_9789


namespace NUMINAMATH_GPT_existence_of_k_good_function_l97_9755

def is_k_good_function (f : ℕ+ → ℕ+) (k : ℕ) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem existence_of_k_good_function (k : ℕ) :
  (∃ f : ℕ+ → ℕ+, is_k_good_function f k) ↔ k ≥ 2 := sorry

end NUMINAMATH_GPT_existence_of_k_good_function_l97_9755


namespace NUMINAMATH_GPT_solve_for_z_l97_9738

theorem solve_for_z (x y : ℝ) (z : ℝ) (h : 2 / x - 1 / y = 3 / z) : 
  z = (2 * y - x) / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_z_l97_9738


namespace NUMINAMATH_GPT_problem_statement_l97_9714

-- Define the roots of the quadratic as r and s
variables (r s : ℝ)

-- Given conditions
def root_condition (r s : ℝ) := (r + s = 2 * Real.sqrt 6) ∧ (r * s = 3)

theorem problem_statement (h : root_condition r s) : r^8 + s^8 = 93474 :=
sorry

end NUMINAMATH_GPT_problem_statement_l97_9714


namespace NUMINAMATH_GPT_cone_volume_l97_9744

theorem cone_volume (slant_height : ℝ) (central_angle_deg : ℝ) (volume : ℝ) :
  slant_height = 1 ∧ central_angle_deg = 120 ∧ volume = (2 * Real.sqrt 2 / 81) * Real.pi →
  ∃ r h, h = Real.sqrt (slant_height^2 - r^2) ∧
    r = (1/3) ∧
    h = (2 * Real.sqrt 2 / 3) ∧
    volume = (1/3) * Real.pi * r^2 * h := 
by
  sorry

end NUMINAMATH_GPT_cone_volume_l97_9744


namespace NUMINAMATH_GPT_sides_of_polygon_l97_9734

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end NUMINAMATH_GPT_sides_of_polygon_l97_9734


namespace NUMINAMATH_GPT_problem_f3_is_neg2_l97_9790

theorem problem_f3_is_neg2 (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (1 + x) = -f (1 - x)) (h3 : f 1 = 2) : f 3 = -2 :=
sorry

end NUMINAMATH_GPT_problem_f3_is_neg2_l97_9790


namespace NUMINAMATH_GPT_travel_time_total_l97_9717

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_total_l97_9717


namespace NUMINAMATH_GPT_rectangle_area_proof_l97_9752

def rectangle_area (L W : ℝ) : ℝ := L * W

theorem rectangle_area_proof (L W : ℝ) (h1 : L + W = 23) (h2 : L^2 + W^2 = 289) : rectangle_area L W = 120 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_proof_l97_9752


namespace NUMINAMATH_GPT_integer_solution_unique_l97_9772

variable (x y : ℤ)

def nested_sqrt_1964_times (x : ℤ) : ℤ := 
  sorry -- (This should define the function for nested sqrt 1964 times, but we'll use sorry to skip the proof)

theorem integer_solution_unique : 
  nested_sqrt_1964_times x = y → x = 0 ∧ y = 0 :=
by
  intros h
  sorry -- Proof of the theorem goes here

end NUMINAMATH_GPT_integer_solution_unique_l97_9772


namespace NUMINAMATH_GPT_no_positive_a_b_for_all_primes_l97_9724

theorem no_positive_a_b_for_all_primes :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (p q : ℕ), p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ ¬Prime (a * p + b * q) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_a_b_for_all_primes_l97_9724


namespace NUMINAMATH_GPT_find_f_neg_eight_l97_9783

-- Conditions based on the given problem
variable (f : ℤ → ℤ)
axiom func_property : ∀ x y : ℤ, f (x + y) = f x + f y + x * y + 1
axiom f1_is_one : f 1 = 1

-- Main theorem
theorem find_f_neg_eight : f (-8) = 19 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_eight_l97_9783


namespace NUMINAMATH_GPT_polynomial_horner_method_l97_9763

-- Define the polynomial f
def f (x : ℕ) :=
  7 * x ^ 7 + 6 * x ^ 6 + 5 * x ^ 5 + 4 * x ^ 4 + 3 * x ^ 3 + 2 * x ^ 2 + x

-- Define x as given in the condition
def x : ℕ := 3

-- State that f(x) = 262 when x = 3
theorem polynomial_horner_method : f x = 262 :=
  by
  sorry

end NUMINAMATH_GPT_polynomial_horner_method_l97_9763


namespace NUMINAMATH_GPT_needed_people_l97_9781

theorem needed_people (n t t' k m : ℕ) (h1 : n = 6) (h2 : t = 8) (h3 : t' = 3) 
    (h4 : k = n * t) (h5 : k = m * t') : m - n = 10 :=
by
  sorry

end NUMINAMATH_GPT_needed_people_l97_9781


namespace NUMINAMATH_GPT_remainder_problem_l97_9775

theorem remainder_problem (f y z : ℤ) (k m n : ℤ) 
  (h1 : f % 5 = 3) 
  (h2 : y % 5 = 4)
  (h3 : z % 7 = 6)
  (h4 : (f + y) % 15 = 7)
  : (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l97_9775


namespace NUMINAMATH_GPT_find_n_l97_9740

theorem find_n (n : ℕ) : 
  Nat.lcm n 12 = 48 ∧ Nat.gcd n 12 = 8 → n = 32 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_l97_9740


namespace NUMINAMATH_GPT_solve_for_x_l97_9757

theorem solve_for_x (x : ℚ) (h : (1 / 3 - 1 / 4 = 4 / x)) : x = 48 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l97_9757


namespace NUMINAMATH_GPT_other_factor_of_936_mul_w_l97_9759

theorem other_factor_of_936_mul_w (w : ℕ) (h_w_pos : 0 < w)
  (h_factors_936w : ∃ k, 936 * w = k * (3^3)) 
  (h_factors_936w_2 : ∃ m, 936 * w = m * (10^2))
  (h_w : w = 120):
  ∃ n, n = 45 :=
by
  sorry

end NUMINAMATH_GPT_other_factor_of_936_mul_w_l97_9759


namespace NUMINAMATH_GPT_certain_number_exists_l97_9773

theorem certain_number_exists :
  ∃ x : ℤ, 55 * x % 7 = 6 ∧ x % 7 = 1 := by
  sorry

end NUMINAMATH_GPT_certain_number_exists_l97_9773


namespace NUMINAMATH_GPT_perfect_squares_of_k_l97_9711

theorem perfect_squares_of_k (k : ℕ) (h : ∃ (a : ℕ), k * (k + 1) = 3 * a^2) : 
  ∃ (m n : ℕ), k = 3 * m^2 ∧ k + 1 = n^2 := 
sorry

end NUMINAMATH_GPT_perfect_squares_of_k_l97_9711


namespace NUMINAMATH_GPT_time_to_cross_pole_l97_9718

-- Setting up the definitions
def speed_kmh : ℤ := 72
def length_m : ℤ := 180

-- Conversion function from km/hr to m/s
def convert_speed (v : ℤ) : ℚ :=
  v * (1000 : ℚ) / 3600

-- Given conditions in mathematics
def speed_ms : ℚ := convert_speed speed_kmh

-- Desired proposition
theorem time_to_cross_pole : 
  length_m / speed_ms = 9 := 
by
  -- Temporarily skipping the proof
  sorry

end NUMINAMATH_GPT_time_to_cross_pole_l97_9718


namespace NUMINAMATH_GPT_relationship_a_b_c_l97_9716

noncomputable def distinct_positive_numbers (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem relationship_a_b_c (a b c : ℝ) (h1 : distinct_positive_numbers a b c) (h2 : a^2 + c^2 = 2 * b * c) : b > a ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_relationship_a_b_c_l97_9716


namespace NUMINAMATH_GPT_ratio_third_to_second_year_l97_9707

-- Define the yearly production of the apple tree
def first_year_production : Nat := 40
def second_year_production : Nat := 2 * first_year_production + 8
def total_production_three_years : Nat := 194
def third_year_production : Nat := total_production_three_years - (first_year_production + second_year_production)

-- Define the ratio calculation
def ratio (a b : Nat) : (Nat × Nat) := 
  let gcd_ab := Nat.gcd a b 
  (a / gcd_ab, b / gcd_ab)

-- Prove the ratio of the third year's production to the second year's production
theorem ratio_third_to_second_year : 
  ratio third_year_production second_year_production = (3, 4) :=
  sorry

end NUMINAMATH_GPT_ratio_third_to_second_year_l97_9707


namespace NUMINAMATH_GPT_measure_angle_BAC_l97_9771

-- Define the elements in the problem
def triangle (A B C : Type) := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

-- Define the lengths and angles
variables {A B C X Y : Type}

-- Define the conditions given in the problem
def conditions (AX XY YB BC : ℝ) (angleABC : ℝ) : Prop :=
  AX = XY ∧ XY = YB ∧ YB = BC ∧ angleABC = 100

-- The Lean 4 statement (proof outline is not required)
theorem measure_angle_BAC {A B C X Y : Type} (hT : triangle A B C)
  (AX XY YB BC : ℝ) (angleABC : ℝ) (hC : conditions AX XY YB BC angleABC) :
  ∃ (t : ℝ), t = 25 :=
sorry
 
end NUMINAMATH_GPT_measure_angle_BAC_l97_9771


namespace NUMINAMATH_GPT_range_of_b_l97_9765

noncomputable def f (a x : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → a ∈ Set.Ico (-1 : ℝ) (0 : ℝ) → f a x < b) ↔ b > -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l97_9765


namespace NUMINAMATH_GPT_katya_sum_greater_than_masha_l97_9784

theorem katya_sum_greater_than_masha (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a+1)*(b+1) + (b+1)*(c+1) + (c+1)*(d+1) + (d+1)*(a+1)) - (a*b + b*c + c*d + d*a) = 4046 := by
  sorry

end NUMINAMATH_GPT_katya_sum_greater_than_masha_l97_9784


namespace NUMINAMATH_GPT_solve_system_eqns_l97_9715

theorem solve_system_eqns (x y z : ℝ) (h1 : x^3 + y^3 + z^3 = 8)
  (h2 : x^2 + y^2 + z^2 = 22)
  (h3 : 1/x + 1/y + 1/z + z/(x * y) = 0) :
  (x = 3 ∧ y = 2 ∧ z = -3) ∨ (x = -3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = -3) ∨ (x = 2 ∧ y = -3 ∧ z = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eqns_l97_9715


namespace NUMINAMATH_GPT_correct_propositions_l97_9777

-- Definitions of propositions
def prop1 (f : ℝ → ℝ) : Prop :=
  f (-2) ≠ f (2) → ∀ x : ℝ, f (-x) ≠ f (x)

def prop2 : Prop :=
  ∀ n : ℕ, n = 0 ∨ n = 1 → (∀ x : ℝ, x ≠ 0 → x ^ n ≠ 0)

def prop3 : Prop :=
  ∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) → (a * b ≠ 0) ∧ (a * b = 0 → a = 0 ∨ b = 0)

def prop4 (a b c d : ℝ) : Prop :=
  ∀ x : ℝ, ∃ k : ℝ, k = d → (3 * a * x ^ 2 + 2 * b * x + c ≠ 0 ∧ b ^ 2 - 3 * a * c ≥ 0)

-- Final proof statement
theorem correct_propositions (f : ℝ → ℝ) (a b c d : ℝ) :
  prop1 f ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 a b c d :=
sorry

end NUMINAMATH_GPT_correct_propositions_l97_9777


namespace NUMINAMATH_GPT_E_union_F_eq_univ_l97_9758

-- Define the given conditions
def E : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def F (a : ℝ) : Set ℝ := { x | x - 5 < a }
def I : Set ℝ := Set.univ
axiom a_gt_6 : ∃ a : ℝ, a > 6 ∧ 11 ∈ F a

-- State the theorem
theorem E_union_F_eq_univ (a : ℝ) (h₁ : a > 6) (h₂ : 11 ∈ F a) : E ∪ F a = I := by
  sorry

end NUMINAMATH_GPT_E_union_F_eq_univ_l97_9758


namespace NUMINAMATH_GPT_Razorback_tshirt_problem_l97_9725

theorem Razorback_tshirt_problem
  (A T : ℕ)
  (h1 : A + T = 186)
  (h2 : 78 * T = 1092) :
  A = 172 := by
  sorry

end NUMINAMATH_GPT_Razorback_tshirt_problem_l97_9725


namespace NUMINAMATH_GPT_parking_spots_l97_9761

def numberOfLevels := 5
def openSpotsOnLevel1 := 4
def openSpotsOnLevel2 := openSpotsOnLevel1 + 7
def openSpotsOnLevel3 := openSpotsOnLevel2 + 6
def openSpotsOnLevel4 := 14
def openSpotsOnLevel5 := openSpotsOnLevel4 + 5
def totalOpenSpots := openSpotsOnLevel1 + openSpotsOnLevel2 + openSpotsOnLevel3 + openSpotsOnLevel4 + openSpotsOnLevel5

theorem parking_spots :
  openSpotsOnLevel5 = 19 ∧ totalOpenSpots = 65 := by
  sorry

end NUMINAMATH_GPT_parking_spots_l97_9761


namespace NUMINAMATH_GPT_gcd_16_12_eq_4_l97_9730

theorem gcd_16_12_eq_4 : Nat.gcd 16 12 = 4 := by
  -- Skipping proof using sorry
  sorry

end NUMINAMATH_GPT_gcd_16_12_eq_4_l97_9730


namespace NUMINAMATH_GPT_determine_a_l97_9723

theorem determine_a (a : ℝ) 
  (h1 : (a - 1) * (0:ℝ)^2 + 0 + a^2 - 1 = 0)
  (h2 : a - 1 ≠ 0) : 
  a = -1 := 
sorry

end NUMINAMATH_GPT_determine_a_l97_9723


namespace NUMINAMATH_GPT_expression_divisible_by_11_l97_9760

theorem expression_divisible_by_11 (n : ℕ) : (3 ^ (2 * n + 2) + 2 ^ (6 * n + 1)) % 11 = 0 :=
sorry

end NUMINAMATH_GPT_expression_divisible_by_11_l97_9760


namespace NUMINAMATH_GPT_brass_players_10_l97_9721

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end NUMINAMATH_GPT_brass_players_10_l97_9721


namespace NUMINAMATH_GPT_smallest_boxes_l97_9751

theorem smallest_boxes (n : Nat) (h₁ : n % 5 = 0) (h₂ : n % 24 = 0) : n = 120 := 
  sorry

end NUMINAMATH_GPT_smallest_boxes_l97_9751


namespace NUMINAMATH_GPT_adjacent_books_probability_l97_9798

def chinese_books : ℕ := 2
def math_books : ℕ := 2
def physics_books : ℕ := 1
def total_books : ℕ := chinese_books + math_books + physics_books

theorem adjacent_books_probability :
  (total_books = 5) →
  (chinese_books = 2) →
  (math_books = 2) →
  (physics_books = 1) →
  (∃ p : ℚ, p = 1 / 5) :=
by
  intros h1 h2 h3 h4
  -- Proof omitted.
  exact ⟨1 / 5, rfl⟩

end NUMINAMATH_GPT_adjacent_books_probability_l97_9798


namespace NUMINAMATH_GPT_length_of_AB_l97_9729

noncomputable def hyperbola_conditions (a b : ℝ) (hac : a > 0) (hbc : b = 2 * a) :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def circle_intersection_condition (A B : ℝ × ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
  (A = (x1, y1)) ∧ (B = (x2, y2)) ∧ ((x1 - 2)^2 + (y1 - 3)^2 = 1 ∧ y1 = 2 * x1) ∧
  ((x2 - 2)^2 + (y2 - 3)^2 = 1 ∧ y2 = 2 * x2)

theorem length_of_AB {a b : ℝ} (hac : a > 0) (hb : b = 2 * a) :
  (hyperbola_conditions a b hac hb) →
  ∃ (A B : ℝ × ℝ), circle_intersection_condition A B → 
  dist A B = (4 * Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l97_9729


namespace NUMINAMATH_GPT_quadratic_m_leq_9_l97_9701

-- Define the quadratic equation
def quadratic_eq_has_real_roots (a b c : ℝ) : Prop := 
  b^2 - 4*a*c ≥ 0

-- Define the specific property we need to prove
theorem quadratic_m_leq_9 (m : ℝ) : (quadratic_eq_has_real_roots 1 (-6) m) → (m ≤ 9) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_m_leq_9_l97_9701


namespace NUMINAMATH_GPT_smallest_y_for_perfect_cube_l97_9788

theorem smallest_y_for_perfect_cube (x y : ℕ) (x_def : x = 11 * 36 * 54) : 
  (∃ y : ℕ, y > 0 ∧ ∀ (n : ℕ), (x * y = n^3 ↔ y = 363)) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_y_for_perfect_cube_l97_9788


namespace NUMINAMATH_GPT_t_lt_s_l97_9766

noncomputable def t : ℝ := Real.sqrt 11 - 3
noncomputable def s : ℝ := Real.sqrt 7 - Real.sqrt 5

theorem t_lt_s : t < s :=
by
  sorry

end NUMINAMATH_GPT_t_lt_s_l97_9766


namespace NUMINAMATH_GPT_rachel_milk_amount_l97_9720

theorem rachel_milk_amount : 
  let don_milk := (3 : ℚ) / 7
  let rachel_fraction := 4 / 5
  let rachel_milk := rachel_fraction * don_milk
  rachel_milk = 12 / 35 :=
by sorry

end NUMINAMATH_GPT_rachel_milk_amount_l97_9720


namespace NUMINAMATH_GPT_find_f_2015_l97_9756

noncomputable def f : ℝ → ℝ :=
sorry

lemma f_period : ∀ x : ℝ, f (x + 8) = f x :=
sorry

axiom f_func_eq : ∀ x : ℝ, f (x + 2) = (1 + f x) / (1 - f x)

axiom f_initial : f 1 = 1 / 4

theorem find_f_2015 : f 2015 = -3 / 5 :=
sorry

end NUMINAMATH_GPT_find_f_2015_l97_9756


namespace NUMINAMATH_GPT_find_m_l97_9746

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m + 1)

theorem find_m (m : ℝ) :
  (∀ x > 0, f m x < 0) → m = -2 := by
  sorry

end NUMINAMATH_GPT_find_m_l97_9746


namespace NUMINAMATH_GPT_remainder_of_product_mod_12_l97_9710

-- Define the given constants
def a := 1125
def b := 1127
def c := 1129
def d := 12

-- State the conditions as Lean hypotheses
lemma mod_eq_1125 : a % d = 9 := by sorry
lemma mod_eq_1127 : b % d = 11 := by sorry
lemma mod_eq_1129 : c % d = 1 := by sorry

-- Define the theorem to prove
theorem remainder_of_product_mod_12 : (a * b * c) % d = 3 := by
  -- Use the conditions stated above to prove the theorem
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_12_l97_9710


namespace NUMINAMATH_GPT_area_of_rectangle_l97_9762

theorem area_of_rectangle (x y : ℝ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 4) * (y + 3 / 2)) :
    x * y = 108 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l97_9762


namespace NUMINAMATH_GPT_number_of_people_l97_9731

theorem number_of_people (x : ℕ) : 
  (x % 10 = 1) ∧
  (x % 9 = 1) ∧
  (x % 8 = 1) ∧
  (x % 7 = 1) ∧
  (x % 6 = 1) ∧
  (x % 5 = 1) ∧
  (x % 4 = 1) ∧
  (x % 3 = 1) ∧
  (x % 2 = 1) ∧
  (x < 5000) →
  x = 2521 :=
sorry

end NUMINAMATH_GPT_number_of_people_l97_9731


namespace NUMINAMATH_GPT_find_integer_solutions_l97_9742

theorem find_integer_solutions :
  {p : ℤ × ℤ | 2 * p.1^3 + p.1 * p.2 = 7} = {(-7, -99), (-1, -9), (1, 5), (7, -97)} :=
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l97_9742


namespace NUMINAMATH_GPT_lifespan_of_bat_l97_9780

variable (B H F T : ℝ)

theorem lifespan_of_bat (h₁ : H = B - 6)
                        (h₂ : F = 4 * H)
                        (h₃ : T = 2 * B)
                        (h₄ : B + H + F + T = 62) :
  B = 11.5 :=
by
  sorry

end NUMINAMATH_GPT_lifespan_of_bat_l97_9780


namespace NUMINAMATH_GPT_lemonade_third_intermission_l97_9753

theorem lemonade_third_intermission (a b c T : ℝ) (h1 : a = 0.25) (h2 : b = 0.42) (h3 : T = 0.92) (h4 : T = a + b + c) : c = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_third_intermission_l97_9753


namespace NUMINAMATH_GPT_inequality_div_two_l97_9764

theorem inequality_div_two (x y : ℝ) (h : x > y) : x / 2 > y / 2 := sorry

end NUMINAMATH_GPT_inequality_div_two_l97_9764


namespace NUMINAMATH_GPT_hare_wins_l97_9743

def hare_wins_race : Prop :=
  let hare_speed := 10
  let hare_run_time := 30
  let hare_nap_time := 30
  let tortoise_speed := 4
  let tortoise_delay := 10
  let total_race_time := 60
  let hare_distance := hare_speed * hare_run_time
  let tortoise_total_time := total_race_time - tortoise_delay
  let tortoise_distance := tortoise_speed * tortoise_total_time
  hare_distance > tortoise_distance

theorem hare_wins : hare_wins_race := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_hare_wins_l97_9743


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l97_9737

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end NUMINAMATH_GPT_opposite_of_neg_2023_l97_9737


namespace NUMINAMATH_GPT_sequence_infinite_integers_l97_9786

theorem sequence_infinite_integers (x : ℕ → ℝ) (x1 x2 : ℝ) 
  (h1 : x 1 = x1) 
  (h2 : x 2 = x2) 
  (h3 : ∀ n ≥ 3, x n = x (n - 2) * x (n - 1) / (2 * x (n - 2) - x (n - 1))) : 
  (∃ k : ℤ, x1 = k ∧ x2 = k) ↔ (∀ n, ∃ m : ℤ, x n = m) :=
sorry

end NUMINAMATH_GPT_sequence_infinite_integers_l97_9786


namespace NUMINAMATH_GPT_ratio_add_b_l97_9700

theorem ratio_add_b (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_add_b_l97_9700


namespace NUMINAMATH_GPT_Expected_and_Variance_l97_9727

variables (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

def P (xi : ℕ) : ℝ := 
  if xi = 0 then p else if xi = 1 then 1 - p else 0

def E_xi : ℝ := 0 * P p 0 + 1 * P p 1

def D_xi : ℝ := (0 - E_xi p)^2 * P p 0 + (1 - E_xi p)^2 * P p 1

theorem Expected_and_Variance :
  (E_xi p = 1 - p) ∧ (D_xi p = p * (1 - p)) :=
sorry

end NUMINAMATH_GPT_Expected_and_Variance_l97_9727


namespace NUMINAMATH_GPT_probability_both_correct_given_any_correct_l97_9735

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_correct_given_any_correct_l97_9735


namespace NUMINAMATH_GPT_infinitely_many_m_l97_9787

theorem infinitely_many_m (r : ℕ) (n : ℕ) (h_r : r > 1) (h_n : n > 0) : 
  ∃ m, m = 4 * r ^ 4 ∧ ¬Prime (n^4 + m) :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_m_l97_9787


namespace NUMINAMATH_GPT_Minjeong_family_juice_consumption_l97_9749

theorem Minjeong_family_juice_consumption :
  (∀ (amount_per_time : ℝ) (times_per_day : ℕ) (days_per_week : ℕ),
  amount_per_time = 0.2 → times_per_day = 3 → days_per_week = 7 → 
  amount_per_time * times_per_day * days_per_week = 4.2) :=
by
  intros amount_per_time times_per_day days_per_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_Minjeong_family_juice_consumption_l97_9749


namespace NUMINAMATH_GPT_line_intercepts_l97_9728

-- Definitions
def point_on_axis (a b : ℝ) : Prop := a = b
def passes_through_point (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 1

theorem line_intercepts (a b x y : ℝ) (hx : x = -1) (hy : y = 2) (intercept_property : point_on_axis a b) (point_property : passes_through_point a b x y) :
  (2 * x + y = 0) ∨ (x + y - 1 = 0) :=
sorry

end NUMINAMATH_GPT_line_intercepts_l97_9728


namespace NUMINAMATH_GPT_older_friend_is_38_l97_9747

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end NUMINAMATH_GPT_older_friend_is_38_l97_9747


namespace NUMINAMATH_GPT_count_CONES_paths_l97_9736

def diagram : List (List Char) :=
  [[' ', ' ', 'C', ' ', ' ', ' '],
   [' ', 'C', 'O', 'C', ' ', ' '],
   ['C', 'O', 'N', 'O', 'C', ' '],
   [' ', 'N', 'E', 'N', ' ', ' '],
   [' ', ' ', 'S', ' ', ' ', ' ']]

def is_adjacent (pos1 pos2 : (Nat × Nat)) : Bool :=
  (pos1.1 = pos2.1 ∨ pos1.1 + 1 = pos2.1 ∨ pos1.1 = pos2.1 + 1) ∧
  (pos1.2 = pos2.2 ∨ pos1.2 + 1 = pos2.2 ∨ pos1.2 = pos2.2 + 1)

def valid_paths (diagram : List (List Char)) : Nat :=
  -- Implementation of counting paths that spell "CONES" skipped
  sorry

theorem count_CONES_paths (d : List (List Char)) 
  (h : d = [[' ', ' ', 'C', ' ', ' ', ' '],
            [' ', 'C', 'O', 'C', ' ', ' '],
            ['C', 'O', 'N', 'O', 'C', ' '],
            [' ', 'N', 'E', 'N', ' ', ' '],
            [' ', ' ', 'S', ' ', ' ', ' ']]): valid_paths d = 6 := 
by
  sorry

end NUMINAMATH_GPT_count_CONES_paths_l97_9736


namespace NUMINAMATH_GPT_intersection_of_sets_l97_9774

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ x : ℝ, y = 2^x - 1 }
def C : Set ℝ := { m | -1 < m ∧ m < 2 }

theorem intersection_of_sets : A ∩ B = C := 
by sorry

end NUMINAMATH_GPT_intersection_of_sets_l97_9774


namespace NUMINAMATH_GPT_arkansas_tshirts_sold_l97_9732

theorem arkansas_tshirts_sold (A T : ℕ) (h1 : A + T = 163) (h2 : 98 * A = 8722) : A = 89 := by
  -- We state the problem and add 'sorry' to skip the actual proof
  sorry

end NUMINAMATH_GPT_arkansas_tshirts_sold_l97_9732


namespace NUMINAMATH_GPT_original_price_correct_l97_9778

noncomputable def original_price (selling_price : ℝ) (gain_percent : ℝ) : ℝ :=
  selling_price / (1 + gain_percent / 100)

theorem original_price_correct :
  original_price 35 75 = 20 :=
by
  sorry

end NUMINAMATH_GPT_original_price_correct_l97_9778


namespace NUMINAMATH_GPT_line_intersects_midpoint_l97_9741

theorem line_intersects_midpoint (c : ℤ) : 
  (∃x y : ℤ, 2 * x - y = c ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → c = -1 := by
  sorry

end NUMINAMATH_GPT_line_intersects_midpoint_l97_9741


namespace NUMINAMATH_GPT_train_speed_second_part_l97_9754

-- Define conditions
def distance_first_part (x : ℕ) := x
def speed_first_part := 40
def distance_second_part (x : ℕ) := 2 * x
def total_distance (x : ℕ) := 5 * x
def average_speed := 40

-- Define the problem
theorem train_speed_second_part (x : ℕ) (v : ℕ) (h1 : total_distance x = 5 * x)
  (h2 : total_distance x / average_speed = distance_first_part x / speed_first_part + distance_second_part x / v) :
  v = 20 :=
  sorry

end NUMINAMATH_GPT_train_speed_second_part_l97_9754


namespace NUMINAMATH_GPT_greatest_possible_a_l97_9768

theorem greatest_possible_a (a : ℕ) :
  (∃ x : ℤ, x^2 + a * x = -28) → a = 29 :=
sorry

end NUMINAMATH_GPT_greatest_possible_a_l97_9768


namespace NUMINAMATH_GPT_angle_measure_l97_9782

theorem angle_measure (x : ℝ) 
  (h1 : 90 - x = (2 / 5) * (180 - x)) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l97_9782


namespace NUMINAMATH_GPT_cos_double_angle_l97_9702

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.cos (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l97_9702


namespace NUMINAMATH_GPT_mary_change_l97_9795

def cost_of_berries : ℝ := 7.19
def cost_of_peaches : ℝ := 6.83
def amount_paid : ℝ := 20.00

theorem mary_change : amount_paid - (cost_of_berries + cost_of_peaches) = 5.98 := by
  sorry

end NUMINAMATH_GPT_mary_change_l97_9795


namespace NUMINAMATH_GPT_find_B_l97_9703

theorem find_B (A B : ℕ) (h1 : Prime A) (h2 : Prime B) (h3 : A > 0) (h4 : B > 0) 
  (h5 : 1 / A - 1 / B = 192 / (2005^2 - 2004^2)) : B = 211 :=
sorry

end NUMINAMATH_GPT_find_B_l97_9703


namespace NUMINAMATH_GPT_pencils_problem_l97_9779

theorem pencils_problem (x : ℕ) :
  2 * x + 6 * 3 + 2 * 1 = 24 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_pencils_problem_l97_9779


namespace NUMINAMATH_GPT_alpha_lt_beta_of_acute_l97_9797

open Real

theorem alpha_lt_beta_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : 2 * sin α = sin α * cos β + cos α * sin β) : α < β :=
by
  sorry

end NUMINAMATH_GPT_alpha_lt_beta_of_acute_l97_9797


namespace NUMINAMATH_GPT_cos_diff_expression_eq_half_l97_9776

theorem cos_diff_expression_eq_half :
  (Real.cos (Real.pi * 24 / 180) * Real.cos (Real.pi * 36 / 180) -
   Real.cos (Real.pi * 66 / 180) * Real.cos (Real.pi * 54 / 180)) = 1 / 2 := by
sorry

end NUMINAMATH_GPT_cos_diff_expression_eq_half_l97_9776


namespace NUMINAMATH_GPT_probability_tile_from_ANGLE_l97_9794

def letters_in_ALGEBRA : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A']
def letters_in_ANGLE : List Char := ['A', 'N', 'G', 'L', 'E']
def count_matching_letters (letters: List Char) (target: List Char) : Nat :=
  letters.foldr (fun l acc => if l ∈ target then acc + 1 else acc) 0

theorem probability_tile_from_ANGLE :
  (count_matching_letters letters_in_ALGEBRA letters_in_ANGLE : ℚ) / (letters_in_ALGEBRA.length : ℚ) = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_tile_from_ANGLE_l97_9794


namespace NUMINAMATH_GPT_sequence_square_terms_l97_9705

theorem sequence_square_terms (k : ℤ) (y : ℕ → ℤ) 
  (h1 : y 1 = 1)
  (h2 : y 2 = 1)
  (h3 : ∀ n ≥ 1, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) :
  (∀ n, ∃ m : ℤ, y n = m ^ 2) ↔ k = 3 :=
by sorry

end NUMINAMATH_GPT_sequence_square_terms_l97_9705


namespace NUMINAMATH_GPT_represent_1917_as_sum_diff_of_squares_l97_9745

theorem represent_1917_as_sum_diff_of_squares : ∃ a b c : ℤ, 1917 = a^2 - b^2 + c^2 :=
by
  use 480, 478, 1
  sorry

end NUMINAMATH_GPT_represent_1917_as_sum_diff_of_squares_l97_9745


namespace NUMINAMATH_GPT_slices_per_pizza_l97_9722

def number_of_people : ℕ := 18
def slices_per_person : ℕ := 3
def number_of_pizzas : ℕ := 6
def total_slices : ℕ := number_of_people * slices_per_person

theorem slices_per_pizza : total_slices / number_of_pizzas = 9 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_slices_per_pizza_l97_9722


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l97_9708

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x >= 3) → (x^2 - 2*x - 3 >= 0) ∧ ¬((x^2 - 2*x - 3 >= 0) → (x >= 3)) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l97_9708


namespace NUMINAMATH_GPT_point_b_in_third_quadrant_l97_9712

-- Definitions of the points with their coordinates
def PointA : ℝ × ℝ := (2, 3)
def PointB : ℝ × ℝ := (-1, -4)
def PointC : ℝ × ℝ := (-4, 1)
def PointD : ℝ × ℝ := (5, -3)

-- Definition of a point being in the third quadrant
def inThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- The main Theorem to prove that PointB is in the third quadrant
theorem point_b_in_third_quadrant : inThirdQuadrant PointB :=
by sorry

end NUMINAMATH_GPT_point_b_in_third_quadrant_l97_9712


namespace NUMINAMATH_GPT_binomial_16_4_l97_9793

theorem binomial_16_4 : Nat.choose 16 4 = 1820 :=
  sorry

end NUMINAMATH_GPT_binomial_16_4_l97_9793


namespace NUMINAMATH_GPT_equilateral_triangles_formed_l97_9713

theorem equilateral_triangles_formed :
  ∀ k : ℤ, -8 ≤ k ∧ k ≤ 8 →
  (∃ triangles : ℕ, triangles = 426) :=
by sorry

end NUMINAMATH_GPT_equilateral_triangles_formed_l97_9713


namespace NUMINAMATH_GPT_general_formula_sequence_l97_9767

-- Define the sequence as an arithmetic sequence with the given first term and common difference
def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

-- Define given values
def a_1 : ℕ := 1
def d : ℕ := 2

-- State the theorem to be proved
theorem general_formula_sequence :
  ∀ n : ℕ, n > 0 → arithmetic_sequence a_1 d n = 2 * n - 1 :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_general_formula_sequence_l97_9767


namespace NUMINAMATH_GPT_parabola_vertex_l97_9769

theorem parabola_vertex {a b c : ℝ} (h₁ : ∃ b c, ∀ x, a * x^2 + b * x + c = a * (x + 3)^2) (h₂ : a * (2 + 3)^2 = -50) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l97_9769


namespace NUMINAMATH_GPT_number_of_jerseys_sold_l97_9750

-- Definitions based on conditions
def revenue_per_jersey : ℕ := 115
def revenue_per_tshirt : ℕ := 25
def tshirts_sold : ℕ := 113
def jersey_cost_difference : ℕ := 90

-- Main condition: Prove the number of jerseys sold is 113
theorem number_of_jerseys_sold : ∀ (J : ℕ), 
  (revenue_per_jersey = revenue_per_tshirt + jersey_cost_difference) →
  (J * revenue_per_jersey = tshirts_sold * revenue_per_tshirt) →
  J = 113 :=
by
  intros J h1 h2
  sorry

end NUMINAMATH_GPT_number_of_jerseys_sold_l97_9750


namespace NUMINAMATH_GPT_Monroe_spiders_l97_9706

theorem Monroe_spiders (S : ℕ) (h1 : 12 * 6 + S * 8 = 136) : S = 8 :=
by
  sorry

end NUMINAMATH_GPT_Monroe_spiders_l97_9706


namespace NUMINAMATH_GPT_minimum_soldiers_to_add_l97_9719

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_soldiers_to_add_l97_9719


namespace NUMINAMATH_GPT_matches_start_with_l97_9792

-- Let M be the number of matches Nate started with
variables (M : ℕ)

-- Given conditions
def dropped_creek (dropped : ℕ) := dropped = 10
def eaten_by_dog (eaten : ℕ) := eaten = 2 * 10
def matches_left (final_matches : ℕ) := final_matches = 40

-- Prove that the number of matches Nate started with is 70
theorem matches_start_with 
  (h1 : dropped_creek 10)
  (h2 : eaten_by_dog 20)
  (h3 : matches_left 40) 
  : M = 70 :=
sorry

end NUMINAMATH_GPT_matches_start_with_l97_9792


namespace NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l97_9770

-- Equation 1
theorem solve_quadratic1 (x : ℝ) :
  (x = 4 + 3 * Real.sqrt 2 ∨ x = 4 - 3 * Real.sqrt 2) ↔ x ^ 2 - 8 * x - 2 = 0 := by
  sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) :
  (x = 3 / 2 ∨ x = -1) ↔ 2 * x ^ 2 - x - 3 = 0 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l97_9770


namespace NUMINAMATH_GPT_tangent_line_circle_l97_9709

theorem tangent_line_circle {m : ℝ} (tangent : ∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 = m → false) : m = 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_l97_9709


namespace NUMINAMATH_GPT_expand_product_l97_9791

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l97_9791


namespace NUMINAMATH_GPT_constant_sequence_l97_9796

theorem constant_sequence (a : ℕ → ℕ) (h : ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → (i + j) ∣ (i * a i + j * a j)) :
  ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → a i = a j :=
by
  sorry

end NUMINAMATH_GPT_constant_sequence_l97_9796


namespace NUMINAMATH_GPT_katrina_cookies_sale_l97_9799

/-- 
Katrina has 120 cookies in the beginning.
She sells 36 cookies in the morning.
She sells 16 cookies in the afternoon.
She has 11 cookies left to take home at the end of the day.
Prove that she sold 57 cookies during the lunch rush.
-/
theorem katrina_cookies_sale :
  let total_cookies := 120
  let morning_sales := 36
  let afternoon_sales := 16
  let cookies_left := 11
  let cookies_sold_lunch_rush := total_cookies - morning_sales - afternoon_sales - cookies_left
  cookies_sold_lunch_rush = 57 :=
by
  sorry

end NUMINAMATH_GPT_katrina_cookies_sale_l97_9799


namespace NUMINAMATH_GPT_cos_sum_seventh_root_of_unity_l97_9726

theorem cos_sum_seventh_root_of_unity (z : ℂ) (α : ℝ) 
  (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : ∃ k : ℤ, α = (2 * k * π) / 7 ) :
  (Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)) = -1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_cos_sum_seventh_root_of_unity_l97_9726
