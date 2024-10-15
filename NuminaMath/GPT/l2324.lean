import Mathlib

namespace NUMINAMATH_GPT_binom_8_5_eq_56_l2324_232483

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_GPT_binom_8_5_eq_56_l2324_232483


namespace NUMINAMATH_GPT_bus_travel_time_l2324_232454

theorem bus_travel_time (D1 D2: ℝ) (T: ℝ) (h1: D1 + D2 = 250) (h2: D1 >= 0) (h3: D2 >= 0) :
  T = D1 / 40 + D2 / 60 ↔ D1 + D2 = 250 := 
by
  sorry

end NUMINAMATH_GPT_bus_travel_time_l2324_232454


namespace NUMINAMATH_GPT_calculate_expression_l2324_232452

theorem calculate_expression : 2 * Real.sin (60 * Real.pi / 180) + (-1/2)⁻¹ + abs (2 - Real.sqrt 3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2324_232452


namespace NUMINAMATH_GPT_triangle_largest_angle_l2324_232463

theorem triangle_largest_angle 
  (a1 a2 a3 : ℝ) 
  (h_sum : a1 + a2 + a3 = 180)
  (h_arith_seq : 2 * a2 = a1 + a3)
  (h_one_angle : a1 = 28) : 
  max a1 (max a2 a3) = 92 := 
by
  sorry

end NUMINAMATH_GPT_triangle_largest_angle_l2324_232463


namespace NUMINAMATH_GPT_train_length_l2324_232426

theorem train_length 
  (V : ℝ → ℝ) (L : ℝ) 
  (length_of_train : ∀ (t : ℝ), t = 8 → V t = L / 8) 
  (pass_platform : ∀ (d t : ℝ), d = L + 273 → t = 20 → V t = d / t) 
  : L = 182 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l2324_232426


namespace NUMINAMATH_GPT_total_workers_in_workshop_l2324_232432

theorem total_workers_in_workshop 
  (W : ℕ)
  (T : ℕ := 5)
  (avg_all : ℕ := 700)
  (avg_technicians : ℕ := 800)
  (avg_rest : ℕ := 650) 
  (total_salary_all : ℕ := W * avg_all)
  (total_salary_technicians : ℕ := T * avg_technicians)
  (total_salary_rest : ℕ := (W - T) * avg_rest) :
  total_salary_all = total_salary_technicians + total_salary_rest →
  W = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_workers_in_workshop_l2324_232432


namespace NUMINAMATH_GPT_mystical_mountain_creatures_l2324_232465

-- Definitions for conditions
def nineHeadedBirdHeads : Nat := 9
def nineHeadedBirdTails : Nat := 1
def nineTailedFoxHeads : Nat := 1
def nineTailedFoxTails : Nat := 9

-- Prove the number of Nine-Tailed Foxes
theorem mystical_mountain_creatures (x y : Nat)
  (h1 : 9 * x + (y - 1) = 36 * (y - 1) + 4 * x)
  (h2 : 9 * (x - 1) + y = 3 * (9 * y + (x - 1))) :
  x = 14 :=
by
  sorry

end NUMINAMATH_GPT_mystical_mountain_creatures_l2324_232465


namespace NUMINAMATH_GPT_simplify_fraction_l2324_232455

theorem simplify_fraction (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := 
sorry

end NUMINAMATH_GPT_simplify_fraction_l2324_232455


namespace NUMINAMATH_GPT_rabbits_clear_land_in_21_days_l2324_232441

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end NUMINAMATH_GPT_rabbits_clear_land_in_21_days_l2324_232441


namespace NUMINAMATH_GPT_student_arrangement_count_l2324_232496

theorem student_arrangement_count :
  let males := 4
  let females := 5
  let select_males := 2
  let select_females := 3
  let total_selected := select_males + select_females
  (Nat.choose males select_males) * (Nat.choose females select_females) * (Nat.factorial total_selected) = 7200 := 
by
  sorry

end NUMINAMATH_GPT_student_arrangement_count_l2324_232496


namespace NUMINAMATH_GPT_range_of_m_l2324_232478

theorem range_of_m (m : ℝ) :
  (∃ x y : ℤ, (x ≠ y) ∧ (x ≥ m ∧ y ≥ m) ∧ (3 - 2 * x ≥ 0) ∧ (3 - 2 * y ≥ 0)) ↔ (-1 < m ∧ m ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2324_232478


namespace NUMINAMATH_GPT_evaluate_expression_l2324_232459

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) : 
  z * (z - 4 * x) = 36 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2324_232459


namespace NUMINAMATH_GPT_find_number_l2324_232408

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2324_232408


namespace NUMINAMATH_GPT_divisibility_2_pow_a_plus_1_l2324_232409

theorem divisibility_2_pow_a_plus_1 (a b : ℕ) (h_b_pos : 0 < b) (h_b_ge_2 : 2 ≤ b) 
  (h_div : (2^a + 1) % (2^b - 1) = 0) : b = 2 := by
  sorry

end NUMINAMATH_GPT_divisibility_2_pow_a_plus_1_l2324_232409


namespace NUMINAMATH_GPT_flower_count_l2324_232493

theorem flower_count (R L T : ℕ) (h1 : R = L + 22) (h2 : R = T - 20) (h3 : L + R + T = 100) : R = 34 :=
by
  sorry

end NUMINAMATH_GPT_flower_count_l2324_232493


namespace NUMINAMATH_GPT_positive_difference_l2324_232488

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_l2324_232488


namespace NUMINAMATH_GPT_vacant_seats_calculation_l2324_232490

noncomputable def seats_vacant (total_seats : ℕ) (percentage_filled : ℚ) : ℚ := 
  total_seats * (1 - percentage_filled)

theorem vacant_seats_calculation: 
  seats_vacant 600 0.45 = 330 := 
by 
    -- sorry to skip the proof.
    sorry

end NUMINAMATH_GPT_vacant_seats_calculation_l2324_232490


namespace NUMINAMATH_GPT_S_2011_l2324_232402

variable {α : Type*}

-- Define initial term and sum function for arithmetic sequence
def a1 : ℤ := -2011
noncomputable def S (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * 2

-- Given conditions
def condition1 : a1 = -2011 := rfl
def condition2 : (S 2010 / 2010) - (S 2008 / 2008) = 2 := by sorry

-- Proof statement
theorem S_2011 : S 2011 = -2011 := by 
  -- Use the given conditions to prove the statement
  sorry

end NUMINAMATH_GPT_S_2011_l2324_232402


namespace NUMINAMATH_GPT_solution_to_problem_l2324_232433

theorem solution_to_problem (f : ℕ → ℕ) 
  (h1 : f 2 = 20)
  (h2 : ∀ n : ℕ, 0 < n → f (2 * n) + n * f 2 = f (2 * n + 2)) :
  f 10 = 220 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_problem_l2324_232433


namespace NUMINAMATH_GPT_find_k_l2324_232456

theorem find_k (k : ℕ) : (1 / 2) ^ 16 * (1 / 81) ^ k = 1 / 18 ^ 16 → k = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l2324_232456


namespace NUMINAMATH_GPT_symmetric_point_x_correct_l2324_232435

-- Define the Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry with respect to the x-axis
def symmetricPointX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point (-2, 1, 4)
def givenPoint : Point3D := { x := -2, y := 1, z := 4 }

-- Define the expected symmetric point
def expectedSymmetricPoint : Point3D := { x := -2, y := -1, z := -4 }

-- State the theorem to prove the expected symmetric point
theorem symmetric_point_x_correct :
  symmetricPointX givenPoint = expectedSymmetricPoint := by
  -- here the proof would go, but we leave it as sorry
  sorry

end NUMINAMATH_GPT_symmetric_point_x_correct_l2324_232435


namespace NUMINAMATH_GPT_sufficientButNotNecessary_l2324_232480

theorem sufficientButNotNecessary (x : ℝ) : ((x + 1) * (x - 3) < 0) → x < 3 ∧ ¬(x < 3 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficientButNotNecessary_l2324_232480


namespace NUMINAMATH_GPT_gcd_72_120_168_l2324_232416

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end NUMINAMATH_GPT_gcd_72_120_168_l2324_232416


namespace NUMINAMATH_GPT_evaluate_expression_l2324_232431

theorem evaluate_expression : (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (8 / 21) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2324_232431


namespace NUMINAMATH_GPT_price_of_second_variety_l2324_232477

-- Define prices and conditions
def price_first : ℝ := 126
def price_third : ℝ := 175.5
def mixture_price : ℝ := 153
def total_weight : ℝ := 4

-- Define unknown price
variable (x : ℝ)

-- Definition of the weighted mixture price
theorem price_of_second_variety :
  (1 * price_first) + (1 * x) + (2 * price_third) = total_weight * mixture_price →
  x = 135 :=
by
  sorry

end NUMINAMATH_GPT_price_of_second_variety_l2324_232477


namespace NUMINAMATH_GPT_find_m_l2324_232427

/-
Define the ellipse equation
-/
def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2) = 1

/-
Define the region R
-/
def region_R (x y : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (2*y = x) ∧ ellipse_eqn x y

/-
Define the region R'
-/
def region_R' (x y m : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (y = m*x) ∧ ellipse_eqn x y

/-
The statement we want to prove
-/
theorem find_m (m : ℝ) : (∃ (x y : ℝ), region_R x y) ∧ (∃ (x y : ℝ), region_R' x y m) →
(m = (2 : ℝ) / 9) := 
sorry

end NUMINAMATH_GPT_find_m_l2324_232427


namespace NUMINAMATH_GPT_horizontal_distance_travelled_l2324_232471

theorem horizontal_distance_travelled (r : ℝ) (θ : ℝ) (d : ℝ)
  (h_r : r = 2) (h_θ : θ = Real.pi / 6) :
  d = 2 * Real.sqrt 3 * Real.pi := sorry

end NUMINAMATH_GPT_horizontal_distance_travelled_l2324_232471


namespace NUMINAMATH_GPT_ellipse_hyperbola_equation_l2324_232444

-- Definitions for the Ellipse and Hyperbola
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / 10 + (y^2) / m = 1
def hyperbola (x y : ℝ) (b : ℝ) : Prop := (x^2) - (y^2) / b = 1

-- Conditions
def same_foci (c1 c2 : ℝ) : Prop := c1 = c2
def intersection_at_p (x y : ℝ) : Prop := x = (Real.sqrt 10) / 3 ∧ (ellipse x y 1 ∧ hyperbola x y 8)

-- Theorem stating the mathematically equivalent proof problem
theorem ellipse_hyperbola_equation :
  ∀ (m b : ℝ) (x y : ℝ), ellipse x y m ∧ hyperbola x y b ∧ same_foci (Real.sqrt (10 - m)) (Real.sqrt (1 + b)) ∧ intersection_at_p x y
  → (m = 1) ∧ (b = 8) := 
by
  intros m b x y h
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_equation_l2324_232444


namespace NUMINAMATH_GPT_missy_yells_at_obedient_dog_12_times_l2324_232467

theorem missy_yells_at_obedient_dog_12_times (x : ℕ) (h : x + 4 * x = 60) : x = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end NUMINAMATH_GPT_missy_yells_at_obedient_dog_12_times_l2324_232467


namespace NUMINAMATH_GPT_range_of_derivative_max_value_of_a_l2324_232422

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.cos x - (x - Real.pi / 2) * Real.sin x

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ :=
  -(1 + a) * Real.sin x - (x - Real.pi / 2) * Real.cos x

-- Part (1): Prove the range of the derivative when a = -1 is [0, π/2]
theorem range_of_derivative (x : ℝ) (h0 : 0 ≤ x) (hπ : x ≤ Real.pi / 2) :
  (0 ≤ f' (-1) x) ∧ (f' (-1) x ≤ Real.pi / 2) := 
sorry

-- Part (2): Prove the maximum value of 'a' when f(x) ≤ 0 always holds
theorem max_value_of_a (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ 0) :
  a ≤ -1 := 
sorry

end NUMINAMATH_GPT_range_of_derivative_max_value_of_a_l2324_232422


namespace NUMINAMATH_GPT_number_of_dress_designs_is_correct_l2324_232434

-- Define the number of choices for colors, patterns, and fabric types as conditions
def num_colors : Nat := 4
def num_patterns : Nat := 5
def num_fabric_types : Nat := 2

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_patterns * num_fabric_types

-- Prove that the total number of different dress designs is 40
theorem number_of_dress_designs_is_correct : total_dress_designs = 40 := by
  sorry

end NUMINAMATH_GPT_number_of_dress_designs_is_correct_l2324_232434


namespace NUMINAMATH_GPT_max_connected_stations_l2324_232468

theorem max_connected_stations (n : ℕ) 
  (h1 : ∀ s : ℕ, s ≤ n → s ≤ 3) 
  (h2 : ∀ x y : ℕ, x < y → ∃ z : ℕ, z < 3 ∧ z ≤ n) : 
  n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_max_connected_stations_l2324_232468


namespace NUMINAMATH_GPT_count_valid_pairs_l2324_232410

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 5 ∧ 
  ∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 40 →
  (5^j - 2^i) % 1729 = 0 →
  i = 0 ∧ j = 36 ∨ 
  i = 1 ∧ j = 37 ∨ 
  i = 2 ∧ j = 38 ∨ 
  i = 3 ∧ j = 39 ∨ 
  i = 4 ∧ j = 40 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_pairs_l2324_232410


namespace NUMINAMATH_GPT_find_number_l2324_232405

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 90) : x = 4000 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2324_232405


namespace NUMINAMATH_GPT_calen_more_pencils_l2324_232485

def calen_pencils (C B D: ℕ) :=
  D = 9 ∧
  B = 2 * D - 3 ∧
  C - 10 = 10

theorem calen_more_pencils (C B D : ℕ) (h : calen_pencils C B D) : C = B + 5 :=
by
  obtain ⟨hD, hB, hC⟩ := h
  simp only [hD, hB, hC]
  sorry

end NUMINAMATH_GPT_calen_more_pencils_l2324_232485


namespace NUMINAMATH_GPT_shooting_test_performance_l2324_232466

theorem shooting_test_performance (m n : ℝ)
    (h1 : m > 9.7)
    (h2 : n < 0.25) :
    (m = 9.9 ∧ n = 0.2) :=
sorry

end NUMINAMATH_GPT_shooting_test_performance_l2324_232466


namespace NUMINAMATH_GPT_second_discount_percentage_l2324_232404

theorem second_discount_percentage
    (original_price : ℝ)
    (first_discount : ℝ)
    (final_sale_price : ℝ)
    (second_discount : ℝ)
    (h1 : original_price = 390)
    (h2 : first_discount = 14)
    (h3 : final_sale_price = 285.09) :
    second_discount = 15 :=
by
  -- Since we are not providing the full proof, we assume the steps to be correct
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l2324_232404


namespace NUMINAMATH_GPT_eval_expr_l2324_232401

theorem eval_expr (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a > b) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a / b)^(b - a) :=
sorry

end NUMINAMATH_GPT_eval_expr_l2324_232401


namespace NUMINAMATH_GPT_calculate_minimal_total_cost_l2324_232486

structure GardenSection where
  area : ℕ
  flower_cost : ℚ

def garden := [
  GardenSection.mk 10 2.75, -- Orchids
  GardenSection.mk 14 2.25, -- Violets
  GardenSection.mk 14 1.50, -- Hyacinths
  GardenSection.mk 15 1.25, -- Tulips
  GardenSection.mk 25 0.75  -- Sunflowers
]

def total_cost (sections : List GardenSection) : ℚ :=
  sections.foldr (λ s acc => s.area * s.flower_cost + acc) 0

theorem calculate_minimal_total_cost :
  total_cost garden = 117.5 := by
  sorry

end NUMINAMATH_GPT_calculate_minimal_total_cost_l2324_232486


namespace NUMINAMATH_GPT_sum_of_tens_and_ones_digits_pow_l2324_232430

theorem sum_of_tens_and_ones_digits_pow : 
  let n := 7
  let exp := 12
  (n^exp % 100) / 10 + (n^exp % 10) = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tens_and_ones_digits_pow_l2324_232430


namespace NUMINAMATH_GPT_valid_triangles_from_10_points_l2324_232462

noncomputable def number_of_valid_triangles (n : ℕ) (h : n = 10) : ℕ :=
  if n = 10 then 100 else 0

theorem valid_triangles_from_10_points :
  number_of_valid_triangles 10 rfl = 100 := 
sorry

end NUMINAMATH_GPT_valid_triangles_from_10_points_l2324_232462


namespace NUMINAMATH_GPT_total_girls_in_circle_l2324_232414

theorem total_girls_in_circle (girls : Nat) 
  (h1 : (4 + 7) = girls + 2) : girls = 11 := 
by
  sorry

end NUMINAMATH_GPT_total_girls_in_circle_l2324_232414


namespace NUMINAMATH_GPT_solve_for_x_l2324_232476

theorem solve_for_x (b x : ℝ) (h1 : b > 1) (h2 : x > 0)
    (h3 : (4 * x) ^ (Real.log 4 / Real.log b) = (6 * x) ^ (Real.log 6 / Real.log b)) :
    x = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2324_232476


namespace NUMINAMATH_GPT_rational_squares_solution_l2324_232420

theorem rational_squares_solution {x y u v : ℕ} (x_pos : 0 < x) (y_pos : 0 < y) (u_pos : 0 < u) (v_pos : 0 < v) 
  (h1 : ∃ q : ℚ, q = (Real.sqrt (x * y) + Real.sqrt (u * v))) 
  (h2 : |(x / 9 : ℚ) - (y / 4 : ℚ)| = |(u / 3 : ℚ) - (v / 12 : ℚ)| ∧ |(u / 3 : ℚ) - (v / 12 : ℚ)| = u * v - x * y) :
  ∃ k : ℕ, x = 9 * k ∧ y = 4 * k ∧ u = 3 * k ∧ v = 12 * k := by
  sorry

end NUMINAMATH_GPT_rational_squares_solution_l2324_232420


namespace NUMINAMATH_GPT_initial_kittens_l2324_232413

theorem initial_kittens (kittens_given : ℕ) (kittens_left : ℕ) (initial_kittens : ℕ) :
  kittens_given = 4 → kittens_left = 4 → initial_kittens = kittens_given + kittens_left → initial_kittens = 8 :=
by
  intros hg hl hi
  rw [hg, hl] at hi
  -- Skipping proof detail
  sorry

end NUMINAMATH_GPT_initial_kittens_l2324_232413


namespace NUMINAMATH_GPT_make_up_set_money_needed_l2324_232474

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_make_up_set_money_needed_l2324_232474


namespace NUMINAMATH_GPT_inequality_geq_8_l2324_232482

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_geq_8_l2324_232482


namespace NUMINAMATH_GPT_negation_of_proposition_p_l2324_232484

variable (x : ℝ)

def proposition_p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0

theorem negation_of_proposition_p : ¬ proposition_p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_p_l2324_232484


namespace NUMINAMATH_GPT_laptop_price_l2324_232411

theorem laptop_price (x : ℝ) : 
  (0.8 * x - 120) = 0.9 * x - 64 → x = 560 :=
by
  sorry

end NUMINAMATH_GPT_laptop_price_l2324_232411


namespace NUMINAMATH_GPT_circle_center_radius_sum_l2324_232475

theorem circle_center_radius_sum :
  let D := { p : ℝ × ℝ | (p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34) }
  let c := 7
  let d := -5
  let s := 2 * Real.sqrt 10
  (c + d + s = 2 + 2 * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_sum_l2324_232475


namespace NUMINAMATH_GPT_circle_diameter_l2324_232449

theorem circle_diameter (r : ℝ) (h : π * r^2 = 9 * π) : 2 * r = 6 :=
by sorry

end NUMINAMATH_GPT_circle_diameter_l2324_232449


namespace NUMINAMATH_GPT_ratio_area_triangles_to_square_l2324_232498

theorem ratio_area_triangles_to_square (x : ℝ) :
  let A := (0, x)
  let B := (x, x)
  let C := (x, 0)
  let D := (0, 0)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let area_AMN := 1/2 * ((M.1 - A.1) * (N.2 - A.2) - (M.2 - A.2) * (N.1 - A.1))
  let area_MNP := 1/2 * ((N.1 - M.1) * (P.2 - M.2) - (N.2 - M.2) * (P.1 - M.1))
  let total_area_triangles := area_AMN + area_MNP
  let area_square := x * x
  total_area_triangles / area_square = 1/4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_area_triangles_to_square_l2324_232498


namespace NUMINAMATH_GPT_value_of_2022_plus_a_minus_b_l2324_232439

theorem value_of_2022_plus_a_minus_b (x a b : ℚ) (h_distinct : x ≠ a ∧ x ≠ b ∧ a ≠ b) 
  (h_gt : a > b) (h_min : ∀ y : ℚ, |y - a| + |y - b| ≥ 2 ∧ |x - a| + |x - b| = 2) :
  2022 + a - b = 2024 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_2022_plus_a_minus_b_l2324_232439


namespace NUMINAMATH_GPT_segment_length_of_absolute_value_l2324_232492

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end NUMINAMATH_GPT_segment_length_of_absolute_value_l2324_232492


namespace NUMINAMATH_GPT_total_amount_silver_l2324_232429

theorem total_amount_silver (x y : ℝ) (h₁ : y = 7 * x + 4) (h₂ : y = 9 * x - 8) : y = 46 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_amount_silver_l2324_232429


namespace NUMINAMATH_GPT_lucy_run_base10_eq_1878_l2324_232460

-- Define a function to convert a base-8 numeral to base-10.
def base8_to_base10 (n: Nat) : Nat :=
  (3 * 8^3) + (5 * 8^2) + (2 * 8^1) + (6 * 8^0)

-- Define the base-8 number.
def lucy_run (n : Nat) : Nat := n

-- Prove that the base-10 equivalent of the base-8 number 3526 is 1878.
theorem lucy_run_base10_eq_1878 : base8_to_base10 (lucy_run 3526) = 1878 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_lucy_run_base10_eq_1878_l2324_232460


namespace NUMINAMATH_GPT_sqrt_200_eq_l2324_232436

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_sqrt_200_eq_l2324_232436


namespace NUMINAMATH_GPT_annie_serious_accident_probability_l2324_232453

theorem annie_serious_accident_probability :
  (∀ temperature : ℝ, temperature < 32 → ∃ skid_chance_increase : ℝ, skid_chance_increase = 5 * ⌊ (32 - temperature) / 3 ⌋ / 100) →
  (∀ control_regain_chance : ℝ, control_regain_chance = 0.4) →
  (∀ control_loss_chance : ℝ, control_loss_chance = 1 - control_regain_chance) →
  (temperature = 8) →
  (serious_accident_probability = skid_chance_increase * control_loss_chance) →
  serious_accident_probability = 0.24 := by
  sorry

end NUMINAMATH_GPT_annie_serious_accident_probability_l2324_232453


namespace NUMINAMATH_GPT_cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l2324_232446

-- Definition of size-n tromino
def tromino_area (n : ℕ) := (4 * 4 * n - 1)

-- Problem (a): Can a size-5 tromino be tiled by size-1 trominos
theorem cannot_tile_size5_with_size1_trominos :
  ¬ (∃ (count : ℕ), count * 3 = tromino_area 5) :=
by sorry

-- Problem (b): Can a size-2013 tromino be tiled by size-1 trominos
theorem can_tile_size2013_with_size1_trominos :
  ∃ (count : ℕ), count * 3 = tromino_area 2013 :=
by sorry

end NUMINAMATH_GPT_cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l2324_232446


namespace NUMINAMATH_GPT_complement_union_eq_l2324_232479

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_union_eq : (U \ (A ∪ B)) = {6,8} := by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l2324_232479


namespace NUMINAMATH_GPT_polynomial_symmetric_equiv_l2324_232458

variable {R : Type*} [CommRing R]

def symmetric_about (P : R → R) (a b : R) : Prop :=
  ∀ x, P (2 * a - x) = 2 * b - P x

def polynomial_form (P : R → R) (a b : R) (Q : R → R) : Prop :=
  ∀ x, P x = b + (x - a) * Q ((x - a) * (x - a))

theorem polynomial_symmetric_equiv (P Q : R → R) (a b : R) :
  (symmetric_about P a b ↔ polynomial_form P a b Q) :=
sorry

end NUMINAMATH_GPT_polynomial_symmetric_equiv_l2324_232458


namespace NUMINAMATH_GPT_phil_quarters_l2324_232451

def initial_quarters : ℕ := 50

def quarters_after_first_year (initial : ℕ) : ℕ := 2 * initial

def quarters_collected_second_year : ℕ := 3 * 12

def quarters_collected_third_year : ℕ := 12 / 3

def total_quarters_before_loss (initial : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ := 
  quarters_after_first_year initial + second_year + third_year

def lost_quarters (total : ℕ) : ℕ := total / 4

def quarters_left (total : ℕ) (lost : ℕ) : ℕ := total - lost

theorem phil_quarters : 
  quarters_left 
    (total_quarters_before_loss 
      initial_quarters 
      quarters_collected_second_year 
      quarters_collected_third_year)
    (lost_quarters 
      (total_quarters_before_loss 
        initial_quarters 
        quarters_collected_second_year 
        quarters_collected_third_year))
  = 105 :=
by
  sorry

end NUMINAMATH_GPT_phil_quarters_l2324_232451


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l2324_232400

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_l2324_232400


namespace NUMINAMATH_GPT_sandy_friend_puppies_l2324_232495

theorem sandy_friend_puppies (original_puppies friend_puppies final_puppies : ℕ)
    (h1 : original_puppies = 8) (h2 : final_puppies = 12) :
    friend_puppies = final_puppies - original_puppies := by
    sorry

end NUMINAMATH_GPT_sandy_friend_puppies_l2324_232495


namespace NUMINAMATH_GPT_salmon_trip_l2324_232407

theorem salmon_trip (male_salmons : ℕ) (female_salmons : ℕ) : male_salmons = 712261 → female_salmons = 259378 → male_salmons + female_salmons = 971639 :=
  sorry

end NUMINAMATH_GPT_salmon_trip_l2324_232407


namespace NUMINAMATH_GPT_correct_calculation_l2324_232497

-- Definitions of calculations based on conditions
def calc_A (a : ℝ) := a^2 + a^2 = a^4
def calc_B (a : ℝ) := (a^2)^3 = a^5
def calc_C (a : ℝ) := a + 2 = 2 * a
def calc_D (a b : ℝ) := (a * b)^3 = a^3 * b^3

-- Theorem stating that only the fourth calculation is correct
theorem correct_calculation (a b : ℝ) :
  ¬(calc_A a) ∧ ¬(calc_B a) ∧ ¬(calc_C a) ∧ calc_D a b :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l2324_232497


namespace NUMINAMATH_GPT_trapezium_area_l2324_232481

theorem trapezium_area (a b h : ℝ) (h₁ : a = 20) (h₂ : b = 16) (h₃ : h = 15) : 
  (1/2 * (a + b) * h = 270) :=
by
  rw [h₁, h₂, h₃]
  -- The following lines of code are omitted as they serve as solving this proof, and the requirement is to provide the statement only. 
  sorry

end NUMINAMATH_GPT_trapezium_area_l2324_232481


namespace NUMINAMATH_GPT_min_director_games_l2324_232442

theorem min_director_games (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 325) (h2 : (26 * 25) / 2 = 325) : k = 0 :=
by {
  -- The conditions are provided in the hypothesis, and the goal is proving the minimum games by director equals 0.
  sorry
}

end NUMINAMATH_GPT_min_director_games_l2324_232442


namespace NUMINAMATH_GPT_inequality_proof_l2324_232417

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) : 
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l2324_232417


namespace NUMINAMATH_GPT_max_ratio_lemma_l2324_232491

theorem max_ratio_lemma (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn : ∀ n, S n = (n + 1) / 2 * a n)
  (hSn_minus_one : ∀ n, S (n - 1) = n / 2 * a (n - 1)) :
  ∀ n > 1, (a n / a (n - 1) ≤ 2) ∧ (a 2 / a 1 = 2) := sorry

end NUMINAMATH_GPT_max_ratio_lemma_l2324_232491


namespace NUMINAMATH_GPT_angle_CDE_proof_l2324_232470

theorem angle_CDE_proof
    (A B C D E : Type)
    (angle_A angle_B angle_C : ℝ)
    (angle_AEB : ℝ)
    (angle_BED : ℝ)
    (angle_BDE : ℝ) :
    angle_A = 90 ∧
    angle_B = 90 ∧
    angle_C = 90 ∧
    angle_AEB = 50 ∧
    angle_BED = 2 * angle_BDE →
    ∃ angle_CDE : ℝ, angle_CDE = 70 :=
by
  sorry

end NUMINAMATH_GPT_angle_CDE_proof_l2324_232470


namespace NUMINAMATH_GPT_expression_simplification_l2324_232406

-- Definitions for P and Q based on x and y
def P (x y : ℝ) := x + y
def Q (x y : ℝ) := x - y

-- The mathematical property to prove
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (x^2 - y^2) / (x * y) := 
by
  -- Sorry is used to skip the proof here
  sorry

end NUMINAMATH_GPT_expression_simplification_l2324_232406


namespace NUMINAMATH_GPT_shaded_region_area_l2324_232424

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2324_232424


namespace NUMINAMATH_GPT_triangle_angle_eq_pi_over_3_l2324_232487

theorem triangle_angle_eq_pi_over_3
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = a * b)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ C : ℝ, C = 2 * Real.pi / 3 ∧ 
            Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_triangle_angle_eq_pi_over_3_l2324_232487


namespace NUMINAMATH_GPT_Michelle_initial_crayons_l2324_232469

variable (M : ℕ)  -- M is the number of crayons Michelle initially has
variable (J : ℕ := 2)  -- Janet has 2 crayons
variable (final_crayons : ℕ := 4)  -- After Janet gives her crayons to Michelle, Michelle has 4 crayons

theorem Michelle_initial_crayons : M + J = final_crayons → M = 2 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_Michelle_initial_crayons_l2324_232469


namespace NUMINAMATH_GPT_express_An_l2324_232447

noncomputable def A_n (A : ℝ) (n : ℤ) : ℝ :=
  (1 / 2^n) * ((A + (A^2 - 4).sqrt)^n + (A - (A^2 - 4).sqrt)^n)

theorem express_An (a : ℝ) (A : ℝ) (n : ℤ) (h : a + a⁻¹ = A) :
  (a^n + a^(-n)) = A_n A n := 
sorry

end NUMINAMATH_GPT_express_An_l2324_232447


namespace NUMINAMATH_GPT_speed_plane_east_l2324_232457

-- Definitions of the conditions
def speed_west : ℕ := 275
def time_hours : ℝ := 3.5
def distance_apart : ℝ := 2100

-- Theorem statement to prove the speed of the plane traveling due East
theorem speed_plane_east (v: ℝ) 
  (h: (v + speed_west) * time_hours = distance_apart) : 
  v = 325 :=
  sorry

end NUMINAMATH_GPT_speed_plane_east_l2324_232457


namespace NUMINAMATH_GPT_double_pythagorean_triple_l2324_232425

theorem double_pythagorean_triple (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  (2*a)^2 + (2*b)^2 = (2*c)^2 :=
by
  sorry

end NUMINAMATH_GPT_double_pythagorean_triple_l2324_232425


namespace NUMINAMATH_GPT_teresa_speed_l2324_232419

def distance : ℝ := 25 -- kilometers
def time : ℝ := 5 -- hours

theorem teresa_speed :
  (distance / time) = 5 := by
  sorry

end NUMINAMATH_GPT_teresa_speed_l2324_232419


namespace NUMINAMATH_GPT_point_on_y_axis_l2324_232489

theorem point_on_y_axis (a : ℝ) 
  (h : (a - 2) = 0) : a = 2 := 
  by 
    sorry

end NUMINAMATH_GPT_point_on_y_axis_l2324_232489


namespace NUMINAMATH_GPT_min_birthdays_on_wednesday_l2324_232437

theorem min_birthdays_on_wednesday (n x w: ℕ) (h_n : n = 61) 
  (h_ineq : w > x) (h_sum : 6 * x + w = n) : w ≥ 13 :=
by
  sorry

end NUMINAMATH_GPT_min_birthdays_on_wednesday_l2324_232437


namespace NUMINAMATH_GPT_correct_answer_l2324_232450

noncomputable def sqrt_2 : ℝ := Real.sqrt 2

def P : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem correct_answer : {sqrt_2} ⊆ P :=
sorry

end NUMINAMATH_GPT_correct_answer_l2324_232450


namespace NUMINAMATH_GPT_min_soldiers_needed_l2324_232464

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end NUMINAMATH_GPT_min_soldiers_needed_l2324_232464


namespace NUMINAMATH_GPT_hcf_of_12_and_15_l2324_232412

-- Definitions of LCM and HCF
def LCM (a b : ℕ) : ℕ := sorry  -- Placeholder for actual LCM definition
def HCF (a b : ℕ) : ℕ := sorry  -- Placeholder for actual HCF definition

theorem hcf_of_12_and_15 :
  LCM 12 15 = 60 → HCF 12 15 = 3 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_12_and_15_l2324_232412


namespace NUMINAMATH_GPT_cubic_solution_l2324_232428

theorem cubic_solution (a b c : ℝ) (h_eq : ∀ x, x^3 - 4*x^2 + 7*x + 6 = 34 -> x = a ∨ x = b ∨ x = c)
(h_ge : a ≥ b ∧ b ≥ c) : 2 * a + b = 8 := 
sorry

end NUMINAMATH_GPT_cubic_solution_l2324_232428


namespace NUMINAMATH_GPT_donny_spent_total_on_friday_and_sunday_l2324_232448

noncomputable def daily_savings (initial: ℚ) (increase_rate: ℚ) (days: List ℚ) : List ℚ :=
days.scanl (λ acc day => acc * increase_rate + acc) initial

noncomputable def thursday_savings : ℚ := (daily_savings 15 (1 + 0.1) [15, 15, 15]).sum

noncomputable def friday_spent : ℚ := thursday_savings * 0.5

noncomputable def remaining_after_friday : ℚ := thursday_savings - friday_spent

noncomputable def saturday_savings (thursday: ℚ) : ℚ := thursday * (1 - 0.20)

noncomputable def total_savings_saturday : ℚ := remaining_after_friday + saturday_savings thursday_savings

noncomputable def sunday_spent : ℚ := total_savings_saturday * 0.40

noncomputable def total_spent : ℚ := friday_spent + sunday_spent

theorem donny_spent_total_on_friday_and_sunday : total_spent = 55.13 := by
  sorry

end NUMINAMATH_GPT_donny_spent_total_on_friday_and_sunday_l2324_232448


namespace NUMINAMATH_GPT_time_with_walkway_l2324_232473

theorem time_with_walkway (v w : ℝ) (t : ℕ) :
  (80 = 120 * (v - w)) → 
  (80 = 60 * v) → 
  t = 80 / (v + w) → 
  t = 40 :=
by
  sorry

end NUMINAMATH_GPT_time_with_walkway_l2324_232473


namespace NUMINAMATH_GPT_find_k_l2324_232418

theorem find_k (k : ℕ) (hk : k > 0) (h_coeff : 15 * k^4 < 120) : k = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_k_l2324_232418


namespace NUMINAMATH_GPT_range_of_x_l2324_232445

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x < -1/2 ∨ x > 1/4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l2324_232445


namespace NUMINAMATH_GPT_circle_line_intersection_symmetric_l2324_232438

theorem circle_line_intersection_symmetric (m n p x y : ℝ)
    (h_intersects : ∃ x y, x = m * y - 1 ∧ x^2 + y^2 + m * x + n * y + p = 0)
    (h_symmetric : ∀ A B : ℝ × ℝ, A = (x, y) ∧ B = (y, x) → y = x) :
    p < -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_line_intersection_symmetric_l2324_232438


namespace NUMINAMATH_GPT_total_pieces_of_paper_l2324_232421

/-- Definitions according to the problem's conditions -/
def pieces_after_first_cut : Nat := 10

def pieces_after_second_cut (initial_pieces : Nat) : Nat := initial_pieces + 9

def pieces_after_third_cut (after_second_cut_pieces : Nat) : Nat := after_second_cut_pieces + 9

def pieces_after_fourth_cut (after_third_cut_pieces : Nat) : Nat := after_third_cut_pieces + 9

/-- The main theorem stating the desired result -/
theorem total_pieces_of_paper : 
  pieces_after_fourth_cut (pieces_after_third_cut (pieces_after_second_cut pieces_after_first_cut)) = 37 := 
by 
  -- The proof would go here, but it's omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_total_pieces_of_paper_l2324_232421


namespace NUMINAMATH_GPT_train_speed_kmph_l2324_232472

-- The conditions
def speed_m_s : ℝ := 52.5042
def conversion_factor : ℝ := 3.6

-- The theorem we need to prove
theorem train_speed_kmph : speed_m_s * conversion_factor = 189.01512 := 
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l2324_232472


namespace NUMINAMATH_GPT_company_total_parts_l2324_232440

noncomputable def total_parts_made (planning_days : ℕ) (initial_rate : ℕ) (extra_rate : ℕ) (extra_parts : ℕ) (x_days : ℕ) : ℕ :=
  let initial_production := planning_days * initial_rate
  let increased_rate := initial_rate + extra_rate
  let actual_production := x_days * increased_rate
  initial_production + actual_production

def planned_production (planning_days : ℕ) (initial_rate : ℕ) (x_days : ℕ) : ℕ :=
  planning_days * initial_rate + x_days * initial_rate

theorem company_total_parts
  (planning_days : ℕ)
  (initial_rate : ℕ)
  (extra_rate : ℕ)
  (extra_parts : ℕ)
  (x_days : ℕ)
  (h1 : planning_days = 3)
  (h2 : initial_rate = 40)
  (h3 : extra_rate = 7)
  (h4 : extra_parts = 150)
  (h5 : x_days = 21)
  (h6 : 7 * x_days = extra_parts) :
  total_parts_made planning_days initial_rate extra_rate extra_parts x_days = 1107 := by
  sorry

end NUMINAMATH_GPT_company_total_parts_l2324_232440


namespace NUMINAMATH_GPT_no_unique_solution_l2324_232443

theorem no_unique_solution (d : ℝ) (x y : ℝ) :
  (3 * (3 * x + 4 * y) = 36) ∧ (9 * x + 12 * y = d) ↔ d ≠ 36 := sorry

end NUMINAMATH_GPT_no_unique_solution_l2324_232443


namespace NUMINAMATH_GPT_solve_color_problem_l2324_232423

variables (R B G C : Prop)

def color_problem (R B G C : Prop) : Prop :=
  (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C) → C ∧ (R ∨ B)

theorem solve_color_problem (R B G C : Prop) (h : (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C)) : C ∧ (R ∨ B) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_solve_color_problem_l2324_232423


namespace NUMINAMATH_GPT_car_Y_average_speed_l2324_232461

theorem car_Y_average_speed 
  (car_X_speed : ℝ)
  (car_X_time_before_Y : ℝ)
  (car_X_distance_when_Y_starts : ℝ)
  (car_X_total_distance : ℝ)
  (car_X_travel_time : ℝ)
  (car_Y_distance : ℝ)
  (car_Y_travel_time : ℝ)
  (h_car_X_speed : car_X_speed = 35)
  (h_car_X_time_before_Y : car_X_time_before_Y = 72 / 60)
  (h_car_X_distance_when_Y_starts : car_X_distance_when_Y_starts = car_X_speed * car_X_time_before_Y)
  (h_car_X_total_distance : car_X_total_distance = car_X_distance_when_Y_starts + car_X_distance_when_Y_starts)
  (h_car_X_travel_time : car_X_travel_time = car_X_total_distance / car_X_speed)
  (h_car_Y_distance : car_Y_distance = 490)
  (h_car_Y_travel_time : car_Y_travel_time = car_X_travel_time) :
  (car_Y_distance / car_Y_travel_time) = 32.24 := 
sorry

end NUMINAMATH_GPT_car_Y_average_speed_l2324_232461


namespace NUMINAMATH_GPT_revenue_95_percent_l2324_232415

-- Definitions based on the conditions
variables (C : ℝ) (n : ℝ)
def revenue_full : ℝ := 1.20 * C
def tickets_sold_percentage : ℝ := 0.95

-- Statement of the theorem based on the problem translation
theorem revenue_95_percent (C : ℝ) :
  (tickets_sold_percentage * revenue_full C) = 1.14 * C :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_revenue_95_percent_l2324_232415


namespace NUMINAMATH_GPT_total_houses_in_lincoln_county_l2324_232494

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (built_houses : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : built_houses = 97741) : 
  original_houses + built_houses = 118558 := 
by
  -- Sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_total_houses_in_lincoln_county_l2324_232494


namespace NUMINAMATH_GPT_simplify_expression_l2324_232499

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 2))) = (Real.sqrt 3 - 2 * Real.sqrt 5 - 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2324_232499


namespace NUMINAMATH_GPT_bus_dispatch_interval_l2324_232403

/--
Xiao Hua walks at a constant speed along the route of the "Chunlei Cup" bus.
He encounters a "Chunlei Cup" bus every 6 minutes head-on and is overtaken by a "Chunlei Cup" bus every 12 minutes.
Assume "Chunlei Cup" buses are dispatched at regular intervals, travel at a constant speed, and do not stop at any stations along the way.
Prove that the time interval between bus departures is 8 minutes.
-/
theorem bus_dispatch_interval
  (encounters_opposite_direction: ℕ)
  (overtakes_same_direction: ℕ)
  (constant_speed: Prop)
  (regular_intervals: Prop)
  (no_stops: Prop)
  (h1: encounters_opposite_direction = 6)
  (h2: overtakes_same_direction = 12)
  (h3: constant_speed)
  (h4: regular_intervals)
  (h5: no_stops) :
  True := 
sorry

end NUMINAMATH_GPT_bus_dispatch_interval_l2324_232403
