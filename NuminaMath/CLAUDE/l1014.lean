import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1014_101481

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_sum : 1/x + 1/y + 1/z = 2) : 8 * (x - 1) * (y - 1) * (z - 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1014_101481


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1014_101485

def is_valid (N : ℕ) : Prop :=
  ∀ k ∈ Finset.range 9, (N + k + 2) % (k + 2) = 0

theorem smallest_valid_number :
  ∃ N : ℕ, is_valid N ∧ ∀ M : ℕ, M < N → ¬ is_valid M :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1014_101485


namespace NUMINAMATH_CALUDE_collinear_vectors_k_value_l1014_101426

/-- Given two non-collinear vectors in a real vector space, 
    if certain conditions are met, then k = -8. -/
theorem collinear_vectors_k_value 
  (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e₁ e₂ : V) (k : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (AB CB CD : V) 
  (h_AB : AB = 2 • e₁ + k • e₂) 
  (h_CB : CB = e₁ + 3 • e₂) 
  (h_CD : CD = 2 • e₁ - e₂) 
  (h_collinear : ∃ (t : ℝ), AB = t • (CD - CB)) : 
  k = -8 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_k_value_l1014_101426


namespace NUMINAMATH_CALUDE_variety_show_probability_l1014_101442

/-- The probability of selecting exactly one boy who likes variety shows
    when randomly choosing two boys from a group of five, where two like
    variety shows and three do not. -/
theorem variety_show_probability :
  let total_boys : ℕ := 5
  let boys_like_shows : ℕ := 2
  let boys_dislike_shows : ℕ := 3
  let selected_boys : ℕ := 2
  
  boys_like_shows + boys_dislike_shows = total_boys →
  
  (Nat.choose total_boys selected_boys : ℚ) ≠ 0 →
  
  (Nat.choose boys_like_shows 1 * Nat.choose boys_dislike_shows 1 : ℚ) /
  (Nat.choose total_boys selected_boys : ℚ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_variety_show_probability_l1014_101442


namespace NUMINAMATH_CALUDE_total_loaves_served_l1014_101429

/-- Given that a restaurant served 0.5 loaf of wheat bread and 0.4 loaf of white bread,
    prove that the total number of loaves served is 0.9. -/
theorem total_loaves_served (wheat_bread : ℝ) (white_bread : ℝ)
    (h1 : wheat_bread = 0.5)
    (h2 : white_bread = 0.4) :
    wheat_bread + white_bread = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l1014_101429


namespace NUMINAMATH_CALUDE_record_cost_thomas_record_cost_l1014_101438

theorem record_cost (num_books : ℕ) (book_price : ℚ) (num_records : ℕ) (leftover : ℚ) : ℚ :=
  let total_sale := num_books * book_price
  let spent_on_records := total_sale - leftover
  spent_on_records / num_records

theorem thomas_record_cost :
  record_cost 200 1.5 75 75 = 3 := by
  sorry

end NUMINAMATH_CALUDE_record_cost_thomas_record_cost_l1014_101438


namespace NUMINAMATH_CALUDE_triangle_inequality_altitudes_l1014_101417

/-- Triangle inequality for side lengths and altitudes -/
theorem triangle_inequality_altitudes (a b c h_a h_b h_c Δ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ha : 0 < h_a) (h_pos_hb : 0 < h_b) (h_pos_hc : 0 < h_c)
  (h_area : 0 < Δ)
  (h_area_a : Δ = (a * h_a) / 2)
  (h_area_b : Δ = (b * h_b) / 2)
  (h_area_c : Δ = (c * h_c) / 2) :
  a * h_b + b * h_c + c * h_a ≥ 6 * Δ := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_altitudes_l1014_101417


namespace NUMINAMATH_CALUDE_max_value_of_z_l1014_101473

theorem max_value_of_z (x y : ℝ) (h1 : y ≤ 1) (h2 : x + y ≥ 0) (h3 : x - y - 2 ≤ 0) :
  ∃ (z : ℝ), z = x - 2*y ∧ z ≤ 3 ∧ ∀ (w : ℝ), w = x - 2*y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l1014_101473


namespace NUMINAMATH_CALUDE_expression_evaluation_l1014_101462

theorem expression_evaluation : 
  let x : ℝ := 3
  let expr := (2 * x^2 + 2*x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2*x + 1)
  expr / (x / (x + 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1014_101462


namespace NUMINAMATH_CALUDE_binomial_recursion_l1014_101448

theorem binomial_recursion (n k : ℕ) (h1 : k ≤ n) (h2 : ¬(n = 0 ∧ k = 0)) :
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_recursion_l1014_101448


namespace NUMINAMATH_CALUDE_no_real_roots_l1014_101498

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 7) - Real.sqrt (x - 5) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1014_101498


namespace NUMINAMATH_CALUDE_red_ball_probability_l1014_101455

theorem red_ball_probability (w r : ℕ+) 
  (h1 : r > w)
  (h2 : r < 2 * w)
  (h3 : 2 * w + 3 * r = 60) :
  (r : ℚ) / (w + r) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1014_101455


namespace NUMINAMATH_CALUDE_train_length_l1014_101440

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 18 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l1014_101440


namespace NUMINAMATH_CALUDE_flight_portion_cost_l1014_101413

theorem flight_portion_cost (total_cost ground_cost flight_additional_cost : ℕ) :
  total_cost = 1275 →
  flight_additional_cost = 625 →
  ground_cost = 325 →
  ground_cost + flight_additional_cost = 950 := by
  sorry

end NUMINAMATH_CALUDE_flight_portion_cost_l1014_101413


namespace NUMINAMATH_CALUDE_mikey_leaves_left_l1014_101405

/-- The number of leaves Mikey has left after some blow away -/
def leaves_left (initial : ℕ) (blown_away : ℕ) : ℕ :=
  initial - blown_away

/-- Theorem stating that Mikey has 112 leaves left -/
theorem mikey_leaves_left :
  leaves_left 356 244 = 112 := by
  sorry

end NUMINAMATH_CALUDE_mikey_leaves_left_l1014_101405


namespace NUMINAMATH_CALUDE_factorial_difference_l1014_101431

theorem factorial_difference (n : ℕ) (h : n.factorial = 362880) : 
  (n + 1).factorial - n.factorial = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1014_101431


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1014_101441

theorem quadratic_roots_property (a b : ℝ) : 
  a^2 - 2*a - 1 = 0 → b^2 - 2*b - 1 = 0 → a^2 + 2*b - a*b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1014_101441


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1014_101410

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 3 = Nat.choose (n - 1) 3 + Nat.choose (n - 1) 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1014_101410


namespace NUMINAMATH_CALUDE_max_value_cube_root_sum_max_value_achievable_l1014_101421

theorem max_value_cube_root_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) ≤ 2 :=
sorry

theorem max_value_achievable :
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_cube_root_sum_max_value_achievable_l1014_101421


namespace NUMINAMATH_CALUDE_sqrt_neg_three_squared_l1014_101403

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_squared_l1014_101403


namespace NUMINAMATH_CALUDE_ambiguous_dates_count_l1014_101463

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The minimum number of days in each month -/
def min_days_per_month : ℕ := 12

/-- The number of days with ambiguous date interpretation -/
def ambiguous_days : ℕ := months_in_year * min_days_per_month - months_in_year

theorem ambiguous_dates_count :
  ambiguous_days = 132 :=
sorry

end NUMINAMATH_CALUDE_ambiguous_dates_count_l1014_101463


namespace NUMINAMATH_CALUDE_sqrt_17_minus_1_gt_3_l1014_101415

theorem sqrt_17_minus_1_gt_3 : Real.sqrt 17 - 1 > 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_minus_1_gt_3_l1014_101415


namespace NUMINAMATH_CALUDE_binomial_variance_determines_n_l1014_101430

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_variance_determines_n (ξ : BinomialDistribution) 
  (h_p : ξ.p = 0.3) 
  (h_var : variance ξ = 2.1) : 
  ξ.n = 10 := by
sorry

end NUMINAMATH_CALUDE_binomial_variance_determines_n_l1014_101430


namespace NUMINAMATH_CALUDE_common_tangent_sum_l1014_101460

/-- Parabola Q₁ -/
def Q₁ (x y : ℝ) : Prop := y = x^2 + 2

/-- Parabola Q₂ -/
def Q₂ (x y : ℝ) : Prop := x = y^2 + 8

/-- Common tangent line M -/
def M (d e f : ℤ) (x y : ℝ) : Prop := d * x + e * y = f

/-- M has nonzero integer slope -/
def nonzero_integer_slope (d e : ℤ) : Prop := d ≠ 0 ∧ e ≠ 0

/-- d, e, f are coprime -/
def coprime (d e f : ℤ) : Prop := Nat.gcd (Nat.gcd d.natAbs e.natAbs) f.natAbs = 1

/-- Main theorem -/
theorem common_tangent_sum (d e f : ℤ) :
  (∃ x y : ℝ, Q₁ x y ∧ Q₂ x y ∧ M d e f x y) →
  nonzero_integer_slope d e →
  coprime d e f →
  d + e + f = 8 := by sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l1014_101460


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1014_101428

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax - b > 0 ↔ x < 1/3) → 
  (∀ x : ℝ, (a - b) * x - (a + b) > 0 ↔ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1014_101428


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1014_101478

theorem smallest_number_divisible (n : ℕ) : n = 34 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 10 = 2 * k ∧ m - 10 = 6 * k ∧ m - 10 = 12 * k ∧ m - 10 = 24 * k)) ∧
  (∃ k : ℕ, n - 10 = 2 * k ∧ n - 10 = 6 * k ∧ n - 10 = 12 * k ∧ n - 10 = 24 * k) ∧
  n > 10 :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l1014_101478


namespace NUMINAMATH_CALUDE_unique_f_two_l1014_101464

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x + y) + x * f y - 2 * x * y - x + 2

theorem unique_f_two (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∃! z : ℝ, f 2 = z ∧ z = 4 := by sorry

end NUMINAMATH_CALUDE_unique_f_two_l1014_101464


namespace NUMINAMATH_CALUDE_trajectory_and_line_equation_l1014_101477

/-- The trajectory of point P and the equation of line l -/
theorem trajectory_and_line_equation :
  ∀ (P : ℝ × ℝ) (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (M : ℝ × ℝ),
  F = (3 * Real.sqrt 3, 0) →
  l = {(x, y) | x = 4 * Real.sqrt 3} →
  M = (4, 2) →
  (∀ (x y : ℝ), P = (x, y) →
    Real.sqrt ((x - 3 * Real.sqrt 3)^2 + y^2) / |x - 4 * Real.sqrt 3| = Real.sqrt 3 / 2) →
  (∃ (B C : ℝ × ℝ), B ∈ l ∧ C ∈ l ∧ B ≠ C ∧ M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) →
  (∀ (x y : ℝ), P = (x, y) → x^2 / 36 + y^2 / 9 = 1) ∧
  (∃ (k : ℝ), k = -1/2 ∧ ∀ (x y : ℝ), y - 2 = k * (x - 4) ↔ x + 2*y - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_line_equation_l1014_101477


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1014_101458

/-- Given a geometric sequence {a_n} where a₂ = 2 and a₁₀ = 8, prove that a₆ = 4 -/
theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  (h_2 : a 2 = 2)  -- Second term is 2
  (h_10 : a 10 = 8)  -- Tenth term is 8
  : a 6 = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1014_101458


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1014_101483

/-- Given a square carpet with shaded squares, calculate the total shaded area -/
theorem shaded_area_calculation (carpet_side : ℝ) (S T : ℝ) 
  (h1 : carpet_side = 12)
  (h2 : carpet_side / S = 4)
  (h3 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1014_101483


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l1014_101436

theorem square_difference_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (product_eq : x * y = 120) :
  x^2 - y^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l1014_101436


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1014_101411

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the condition that the hyperbola passes through (2, 1)
def passes_through_point (a b : ℝ) : Prop := hyperbola a b 2 1

-- Define the condition that the hyperbola and ellipse share the same foci
def same_foci (a b : ℝ) : Prop := a^2 + b^2 = 3

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) 
  (h1 : passes_through_point a b) 
  (h2 : same_foci a b) : 
  ∀ x y : ℝ, hyperbola 2 1 x y := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1014_101411


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l1014_101419

theorem unique_quadratic_root (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → m = 0 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l1014_101419


namespace NUMINAMATH_CALUDE_simplify_expression_l1014_101471

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) - (1 / 4) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1014_101471


namespace NUMINAMATH_CALUDE_avg_people_moving_rounded_l1014_101418

/-- The number of people moving to California -/
def people_moving : ℕ := 4500

/-- The time period in days -/
def days : ℕ := 5

/-- The additional hours beyond full days -/
def extra_hours : ℕ := 12

/-- Function to calculate the average people per minute -/
def avg_people_per_minute (people : ℕ) (days : ℕ) (hours : ℕ) : ℚ :=
  people / (((days * 24 + hours) * 60) : ℚ)

/-- Function to round a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that the average number of people moving per minute, 
    when rounded to the nearest whole number, is 1 -/
theorem avg_people_moving_rounded : 
  round_to_nearest (avg_people_per_minute people_moving days extra_hours) = 1 := by
  sorry


end NUMINAMATH_CALUDE_avg_people_moving_rounded_l1014_101418


namespace NUMINAMATH_CALUDE_initial_files_correct_l1014_101482

/-- The number of files Megan initially had on her computer -/
def initial_files : ℕ := 93

/-- The number of files Megan deleted -/
def deleted_files : ℕ := 21

/-- The number of files in each folder -/
def files_per_folder : ℕ := 8

/-- The number of folders Megan ended up with -/
def num_folders : ℕ := 9

/-- Theorem stating that the initial number of files is correct -/
theorem initial_files_correct : 
  initial_files = deleted_files + num_folders * files_per_folder :=
by sorry

end NUMINAMATH_CALUDE_initial_files_correct_l1014_101482


namespace NUMINAMATH_CALUDE_exists_A_square_diff_two_l1014_101409

/-- The ceiling function -/
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.floor x + 1

/-- Main theorem -/
theorem exists_A_square_diff_two :
  ∃ A : ℝ, ∀ n : ℕ, ∃ m : ℤ, |A^n - m^2| = 2 :=
sorry

end NUMINAMATH_CALUDE_exists_A_square_diff_two_l1014_101409


namespace NUMINAMATH_CALUDE_fraction_problem_l1014_101490

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 6) = 3 / 4 → x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1014_101490


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l1014_101400

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 3}
def N : Set Nat := {1, 2}

theorem complement_of_union_equals_set (U M N : Set Nat) 
  (hU : U = {1, 2, 3, 4, 5}) 
  (hM : M = {1, 3}) 
  (hN : N = {1, 2}) : 
  (M ∪ N)ᶜ = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l1014_101400


namespace NUMINAMATH_CALUDE_sum_u_v_equals_negative_42_over_77_l1014_101420

theorem sum_u_v_equals_negative_42_over_77 
  (u v : ℚ) 
  (eq1 : 3 * u - 7 * v = 17) 
  (eq2 : 5 * u + 3 * v = 1) : 
  u + v = -42 / 77 := by
sorry

end NUMINAMATH_CALUDE_sum_u_v_equals_negative_42_over_77_l1014_101420


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1014_101423

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1014_101423


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_25_plus_13_25_l1014_101469

theorem sum_of_last_two_digits_of_7_25_plus_13_25 : 
  (7^25 + 13^25) % 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_25_plus_13_25_l1014_101469


namespace NUMINAMATH_CALUDE_pencil_distribution_l1014_101492

/-- Given a classroom with 4 children and 8 pencils to be distributed,
    prove that each child receives 2 pencils. -/
theorem pencil_distribution (num_children : ℕ) (num_pencils : ℕ) 
  (h1 : num_children = 4) 
  (h2 : num_pencils = 8) : 
  num_pencils / num_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1014_101492


namespace NUMINAMATH_CALUDE_probability_females_right_of_males_l1014_101457

theorem probability_females_right_of_males :
  let total_people : ℕ := 3 + 2
  let male_count : ℕ := 3
  let female_count : ℕ := 2
  let total_arrangements : ℕ := Nat.factorial total_people
  let favorable_arrangements : ℕ := Nat.factorial male_count * Nat.factorial female_count
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_females_right_of_males_l1014_101457


namespace NUMINAMATH_CALUDE_combined_population_theorem_l1014_101445

def wellington_population : ℕ := 900

def port_perry_population (wellington : ℕ) : ℕ := 7 * wellington

def lazy_harbor_population (port_perry : ℕ) : ℕ := port_perry - 800

theorem combined_population_theorem (wellington : ℕ) (port_perry : ℕ) (lazy_harbor : ℕ) :
  wellington = wellington_population →
  port_perry = port_perry_population wellington →
  lazy_harbor = lazy_harbor_population port_perry →
  port_perry + lazy_harbor = 11800 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_population_theorem_l1014_101445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1014_101475

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 2 = 4 →                     -- given condition
  ∃ r : ℝ,                      -- existence of common ratio for geometric sequence
    (1 + a 3) * r = a 6 ∧       -- geometric sequence conditions
    a 6 * r = 4 + a 10 →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1014_101475


namespace NUMINAMATH_CALUDE_complex_multiplication_l1014_101446

theorem complex_multiplication :
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := 2 - 3 * Complex.I
  z₁ * z₂ = 7 - 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1014_101446


namespace NUMINAMATH_CALUDE_medal_award_ways_l1014_101424

-- Define the total number of sprinters
def total_sprinters : ℕ := 10

-- Define the number of Spanish sprinters
def spanish_sprinters : ℕ := 4

-- Define the number of medals
def medals : ℕ := 3

-- Function to calculate the number of ways to award medals
def award_medals : ℕ := sorry

-- Theorem statement
theorem medal_award_ways :
  award_medals = 696 :=
sorry

end NUMINAMATH_CALUDE_medal_award_ways_l1014_101424


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1014_101443

def f (x : ℝ) := 3*x - x^3

theorem monotonic_increasing_interval_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, StrictMonoOn f (Set.Ioo (-1 : ℝ) 1) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1014_101443


namespace NUMINAMATH_CALUDE_inverse_proportion_l1014_101451

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 54) (h3 : x = 3 * y) :
  ∃ (y' : ℝ), 5 * y' = k ∧ y' = 109.35 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1014_101451


namespace NUMINAMATH_CALUDE_parabola_shift_l1014_101434

/-- Given a parabola y = x^2 + bx + c that is shifted 4 units to the right
    and 3 units down to become y = x^2 - 4x + 3, prove that b = 4 and c = 6. -/
theorem parabola_shift (b c : ℝ) : 
  (∀ x, (x - 4)^2 + b*(x - 4) + c - 3 = x^2 - 4*x + 3) → 
  b = 4 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_shift_l1014_101434


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l1014_101407

-- Define the complex cube root of unity
noncomputable def ω : ℂ := sorry

-- Define the theorem
theorem smallest_value_of_expression (a b c : ℤ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →  -- non-zero integers
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- distinct integers
  (Even a ∧ Even b ∧ Even c) →  -- even integers
  (ω^3 = 1 ∧ ω ≠ 1) →  -- properties of ω
  ∃ (min : ℝ), 
    min = Real.sqrt 12 ∧
    ∀ (x y z : ℤ), 
      (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) →
      (x ≠ y ∧ y ≠ z ∧ x ≠ z) →
      (Even x ∧ Even y ∧ Even z) →
      Complex.abs (x + y • ω + z • ω^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l1014_101407


namespace NUMINAMATH_CALUDE_nick_speed_l1014_101472

/-- Given the speeds of Alan, Maria, and Nick in relation to each other,
    prove that Nick's speed is 6 miles per hour. -/
theorem nick_speed (alan_speed : ℝ) (maria_speed : ℝ) (nick_speed : ℝ)
    (h1 : alan_speed = 6)
    (h2 : maria_speed = 3/4 * alan_speed)
    (h3 : nick_speed = 4/3 * maria_speed) :
    nick_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_nick_speed_l1014_101472


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_max_sum_value_max_sum_at_12_max_sum_is_144_l1014_101491

/-- The maximum sum of the first n terms of an arithmetic sequence with a_1 = 23 and d = -2 -/
theorem max_sum_arithmetic_sequence : ℕ → ℝ :=
  fun n => -n^2 + 24*n

/-- The maximum value of the sum of the first n terms is 144 -/
theorem max_sum_value : ∃ (n : ℕ), ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

/-- The value of n that maximizes the sum is 12 -/
theorem max_sum_at_12 : ∃ (n : ℕ), n = 12 ∧ ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

/-- The maximum sum value is 144 -/
theorem max_sum_is_144 : ∃ (n : ℕ), max_sum_arithmetic_sequence n = 144 ∧ ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_max_sum_value_max_sum_at_12_max_sum_is_144_l1014_101491


namespace NUMINAMATH_CALUDE_shaded_half_l1014_101456

/-- Represents a square divided into smaller squares with specific shading -/
structure DividedSquare where
  /-- The number of smaller squares the large square is divided into -/
  num_divisions : Nat
  /-- Whether a diagonal is drawn in one of the smaller squares -/
  has_diagonal : Bool
  /-- The number of quarters of a smaller square that are additionally shaded -/
  additional_shaded_quarters : Nat

/-- Calculates the fraction of the large square that is shaded -/
def shaded_fraction (s : DividedSquare) : Rat :=
  sorry

/-- Theorem stating that for a specific configuration, the shaded fraction is 1/2 -/
theorem shaded_half (s : DividedSquare) 
  (h1 : s.num_divisions = 4) 
  (h2 : s.has_diagonal = true)
  (h3 : s.additional_shaded_quarters = 2) : 
  shaded_fraction s = 1/2 :=
sorry

end NUMINAMATH_CALUDE_shaded_half_l1014_101456


namespace NUMINAMATH_CALUDE_triangle_area_l1014_101450

theorem triangle_area (a b c A B C : Real) (h1 : A = π/4) (h2 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) : 
  (1/2) * b * c * Real.sin A = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1014_101450


namespace NUMINAMATH_CALUDE_max_area_parabola_triangle_l1014_101487

/-- Given two points on a parabola, prove the maximum area of a triangle formed with a specific third point -/
theorem max_area_parabola_triangle (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ ≠ x₂ →
  x₁ + x₂ = 4 →
  y₁^2 = 6*x₁ →
  y₂^2 = 6*x₂ →
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let M := ((x₁ + x₂)/2, (y₁ + y₂)/2)
  let k_AB := (y₂ - y₁)/(x₂ - x₁)
  let C := (5, 0)
  let triangle_area := abs ((x₁ - 5)*(y₂ - 0) + (x₂ - x₁)*(0 - y₁) + (5 - x₂)*(y₁ - y₂)) / 2
  ∃ (max_area : ℝ), max_area = 14 * Real.sqrt 7 / 3 ∧ 
    ∀ (x₁' x₂' y₁' y₂' : ℝ), 
      x₁' ≠ x₂' → 
      x₁' + x₂' = 4 → 
      y₁'^2 = 6*x₁' → 
      y₂'^2 = 6*x₂' → 
      let A' := (x₁', y₁')
      let B' := (x₂', y₂')
      let triangle_area' := abs ((x₁' - 5)*(y₂' - 0) + (x₂' - x₁')*(0 - y₁') + (5 - x₂')*(y₁' - y₂')) / 2
      triangle_area' ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_max_area_parabola_triangle_l1014_101487


namespace NUMINAMATH_CALUDE_final_time_sum_l1014_101468

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define a function to add time
def addTime (start : Time) (elapsedHours elapsedMinutes elapsedSeconds : Nat) : Time :=
  sorry

-- Define a function to calculate the sum of time components
def sumTimeComponents (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

-- Theorem statement
theorem final_time_sum (startTime : Time) 
  (h1 : startTime.hours = 3) 
  (h2 : startTime.minutes = 0) 
  (h3 : startTime.seconds = 0) : 
  let finalTime := addTime startTime 240 58 30
  sumTimeComponents finalTime = 91 :=
sorry

end NUMINAMATH_CALUDE_final_time_sum_l1014_101468


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1014_101402

theorem binomial_coefficient_ratio (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) ↔ n = 43 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1014_101402


namespace NUMINAMATH_CALUDE_M_properties_M_remainder_l1014_101480

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + 
        d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    1 ≤ d1 ∧ d1 ≤ 9 ∧
    1 ≤ d2 ∧ d2 ≤ 9 ∧
    1 ≤ d3 ∧ d3 ≤ 9 ∧
    1 ≤ d4 ∧ d4 ≤ 9 ∧
    1 ≤ d5 ∧ d5 ≤ 9 ∧
    1 ≤ d6 ∧ d6 ≤ 9 ∧
    1 ≤ d7 ∧ d7 ≤ 9 ∧
    1 ≤ d8 ∧ d8 ≤ 9 ∧
    1 ≤ d9 ∧ d9 ≤ 9

def M : ℕ := sorry

theorem M_properties :
  is_valid_number M ∧ 
  M % 12 = 0 ∧
  ∀ n, is_valid_number n ∧ n % 12 = 0 → n ≤ M :=
by sorry

theorem M_remainder : M % 100 = 12 :=
by sorry

end NUMINAMATH_CALUDE_M_properties_M_remainder_l1014_101480


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1014_101479

theorem arithmetic_calculations :
  (5 + (-6) + 3 - 8 - (-4) = -2) ∧
  (-2^2 - 3 * (-1)^3 - (-1) / (-1/2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1014_101479


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1014_101494

theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (3*x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1014_101494


namespace NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l1014_101496

theorem normal_distribution_two_std_dev_below_mean 
  (μ : ℝ) (σ : ℝ) (h_μ : μ = 17.5) (h_σ : σ = 2.5) :
  μ - 2 * σ = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l1014_101496


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1014_101427

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ -1 →
  (2 * x / (x + 1) - (2 * x + 6) / (x^2 - 1) / ((x + 3) / (x^2 - 2 * x + 1))) = 2 / (x + 1) ∧
  (2 / (0 + 1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1014_101427


namespace NUMINAMATH_CALUDE_largest_certain_divisor_of_visible_product_l1014_101444

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem largest_certain_divisor_of_visible_product :
  ∀ (visible : Finset ℕ), visible ⊆ die_numbers → visible.card = 7 →
  ∃ (k : ℕ), (visible.prod id = 192 * k) := by
  sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_of_visible_product_l1014_101444


namespace NUMINAMATH_CALUDE_locus_of_point_P_l1014_101454

/-- The locus of points P(x, y) such that the product of slopes of AP and BP is -1/4,
    where A(-2, 0) and B(2, 0) are fixed points. -/
theorem locus_of_point_P (x y : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  (y / (x + 2)) * (y / (x - 2)) = -1/4 ↔ x^2 / 4 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l1014_101454


namespace NUMINAMATH_CALUDE_min_box_value_l1014_101414

/-- Given the equation (cx+d)(dx+c) = 15x^2 + ◻x + 15, where c, d, and ◻ are distinct integers,
    the minimum possible value of ◻ is 34. -/
theorem min_box_value (c d box : ℤ) : 
  (c * d + c = 15) →
  (c + d = box) →
  (c ≠ d) ∧ (c ≠ box) ∧ (d ≠ box) →
  (∀ (c' d' box' : ℤ), (c' * d' + c' = 15) → (c' + d' = box') → 
    (c' ≠ d') ∧ (c' ≠ box') ∧ (d' ≠ box') → box ≤ box') →
  box = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l1014_101414


namespace NUMINAMATH_CALUDE_percentage_calculation_l1014_101433

theorem percentage_calculation (P : ℝ) : 
  (3/5 : ℝ) * 120 * (P/100) = 36 → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1014_101433


namespace NUMINAMATH_CALUDE_vector_c_value_l1014_101467

/-- Given two planar vectors a and b, returns true if they are parallel -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_c_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, m)
  let c : ℝ × ℝ := (-7, 14)
  are_parallel a b →
  (3 • a.1 + 2 • b.1 + c.1 = 0 ∧ 3 • a.2 + 2 • b.2 + c.2 = 0) →
  c = (-7, 14) := by
  sorry


end NUMINAMATH_CALUDE_vector_c_value_l1014_101467


namespace NUMINAMATH_CALUDE_total_plans_is_180_l1014_101439

def male_teachers : ℕ := 4
def female_teachers : ℕ := 3
def schools : ℕ := 3

-- Function to calculate the number of ways to select and assign teachers
def selection_and_assignment_plans : ℕ :=
  (male_teachers.choose 1 * female_teachers.choose 2 +
   male_teachers.choose 2 * female_teachers.choose 1) *
  schools.factorial

-- Theorem to prove
theorem total_plans_is_180 :
  selection_and_assignment_plans = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_plans_is_180_l1014_101439


namespace NUMINAMATH_CALUDE_box_dimensions_l1014_101466

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17)
  (h2 : a + b = 13)
  (h3 : 2 * (b + c) = 40)
  (h4 : a < b)
  (h5 : b < c) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l1014_101466


namespace NUMINAMATH_CALUDE_sum_in_base5_l1014_101499

/-- Converts a number from base 10 to base 5 -/
def toBase5 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 5 to base 10 -/
def fromBase5 (n : ℕ) : ℕ := sorry

theorem sum_in_base5 : toBase5 (45 + 27) = 242 := by sorry

end NUMINAMATH_CALUDE_sum_in_base5_l1014_101499


namespace NUMINAMATH_CALUDE_evaluate_expression_l1014_101401

theorem evaluate_expression (x y z : ℚ) : 
  x = 1/4 → y = 3/4 → z = 3 → x^2 * y^3 * z = 81/1024 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1014_101401


namespace NUMINAMATH_CALUDE_rectangle_area_l1014_101404

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    prove that its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 8 * w + 2 * w = 200) : w * (4 * w) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1014_101404


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1014_101412

theorem triangle_angle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < π →
  a^2 + c^2 = b^2 + a*c →
  B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1014_101412


namespace NUMINAMATH_CALUDE_phil_remaining_books_pages_l1014_101465

def book_pages : List Nat := [120, 150, 80, 200, 90, 180, 75, 190, 110, 160, 130, 170, 100, 140, 210]

def misplaced_indices : List Nat := [1, 5, 9, 14]  -- 0-based indices

def remaining_pages : Nat := book_pages.sum - (misplaced_indices.map (λ i => book_pages.get! i)).sum

theorem phil_remaining_books_pages :
  remaining_pages = 1305 := by sorry

end NUMINAMATH_CALUDE_phil_remaining_books_pages_l1014_101465


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l1014_101488

/-- A line passing through three points in a rectangular coordinate system -/
structure Line where
  p1 : Prod ℝ ℝ
  p2 : Prod ℝ ℝ
  p3 : Prod ℝ ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- Theorem: The x-intercept of the line passing through (10, 3), (-10, -7), and (5, 1) is 4 -/
theorem x_intercept_of_specific_line :
  let l : Line := { p1 := (10, 3), p2 := (-10, -7), p3 := (5, 1) }
  x_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l1014_101488


namespace NUMINAMATH_CALUDE_compare_A_B_l1014_101422

theorem compare_A_B (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : (3/4) * A = (4/3) * B) : A > B := by
  sorry

end NUMINAMATH_CALUDE_compare_A_B_l1014_101422


namespace NUMINAMATH_CALUDE_function_property_l1014_101432

def positive_integer (n : ℕ) := n > 0

theorem function_property 
  (f : ℕ → ℕ) 
  (h : ∀ n, positive_integer n → f (f n) + f (n + 1) = n + 2) : 
  ∀ n, positive_integer n → f (f n + n) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1014_101432


namespace NUMINAMATH_CALUDE_committee_selection_with_president_l1014_101474

/-- The number of ways to choose a committee with a required member -/
def choose_committee_with_required (n m k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: Choosing a 5-person committee from 12 people with at least one being the president -/
theorem committee_selection_with_president :
  choose_committee_with_required 12 1 5 = 330 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_with_president_l1014_101474


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1014_101476

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1014_101476


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1014_101461

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a3 : a 3 = 2) :
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1014_101461


namespace NUMINAMATH_CALUDE_holdens_class_results_l1014_101453

/-- Proves the number of students who received an A in Mr. Holden's class exam
    and the number of students who did not receive an A in Mr. Holden's class quiz -/
theorem holdens_class_results (kennedy_total : ℕ) (kennedy_a : ℕ) (holden_total : ℕ)
    (h1 : kennedy_total = 20)
    (h2 : kennedy_a = 8)
    (h3 : holden_total = 30)
    (h4 : (kennedy_a : ℚ) / kennedy_total = (holden_a : ℚ) / holden_total)
    (h5 : (holden_total - holden_a : ℚ) / holden_total = 2 * (holden_not_a_quiz : ℚ) / holden_total) :
    holden_a = 12 ∧ holden_not_a_quiz = 9 := by
  sorry

#check holdens_class_results

end NUMINAMATH_CALUDE_holdens_class_results_l1014_101453


namespace NUMINAMATH_CALUDE_probability_white_ball_l1014_101459

/-- The probability of drawing a white ball from a bag with specified numbers of colored balls. -/
theorem probability_white_ball (white red black : ℕ) : 
  white = 3 → red = 4 → black = 5 → (white : ℚ) / (white + red + black) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l1014_101459


namespace NUMINAMATH_CALUDE_math_books_count_l1014_101435

theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℕ) (total_price : ℕ) :
  total_books = 250 →
  math_book_price = 7 →
  history_book_price = 9 →
  total_price = 1860 →
  ∃ (math_books history_books : ℕ),
    math_books + history_books = total_books ∧
    math_book_price * math_books + history_book_price * history_books = total_price ∧
    math_books = 195 :=
by sorry

end NUMINAMATH_CALUDE_math_books_count_l1014_101435


namespace NUMINAMATH_CALUDE_wall_decoration_thumbtack_fraction_l1014_101497

theorem wall_decoration_thumbtack_fraction :
  let total_decorations : ℕ := 50 * 3 / 2
  let nailed_decorations : ℕ := 50
  let remaining_decorations : ℕ := total_decorations - nailed_decorations
  let sticky_strip_decorations : ℕ := 15
  let thumbtack_decorations : ℕ := remaining_decorations - sticky_strip_decorations
  (thumbtack_decorations : ℚ) / remaining_decorations = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_wall_decoration_thumbtack_fraction_l1014_101497


namespace NUMINAMATH_CALUDE_last_digit_product_l1014_101486

/-- The last digit of (3^65 * 6^n * 7^71) is 4 for any non-negative integer n. -/
theorem last_digit_product (n : ℕ) : (3^65 * 6^n * 7^71) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_product_l1014_101486


namespace NUMINAMATH_CALUDE_onion_shelf_problem_l1014_101484

/-- Given the initial conditions of onions on a shelf, prove the final number of onions. -/
theorem onion_shelf_problem (initial : ℕ) (sold : ℕ) (added : ℕ) (given_away : ℕ) : 
  initial = 98 → sold = 65 → added = 20 → given_away = 10 → 
  initial - sold + added - given_away = 43 := by
sorry

end NUMINAMATH_CALUDE_onion_shelf_problem_l1014_101484


namespace NUMINAMATH_CALUDE_partnership_profit_l1014_101447

/-- The total profit of a partnership business given C's share and percentage -/
theorem partnership_profit (c_share : ℕ) (c_percentage : ℕ) (total_profit : ℕ) : 
  c_share = 60000 → c_percentage = 25 → total_profit = 240000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l1014_101447


namespace NUMINAMATH_CALUDE_initial_speed_is_40_l1014_101470

/-- A person's journey with varying speeds -/
def Journey (D T : ℝ) (initial_speed final_speed : ℝ) : Prop :=
  initial_speed > 0 ∧ final_speed > 0 ∧ D > 0 ∧ T > 0 ∧
  (2/3 * D) / (1/3 * T) = initial_speed ∧
  (1/3 * D) / (1/3 * T) = final_speed

/-- Theorem: Given the conditions, the initial speed is 40 kmph -/
theorem initial_speed_is_40 (D T : ℝ) :
  Journey D T initial_speed 20 → initial_speed = 40 := by
  sorry

#check initial_speed_is_40

end NUMINAMATH_CALUDE_initial_speed_is_40_l1014_101470


namespace NUMINAMATH_CALUDE_quadruplet_solution_l1014_101493

theorem quadruplet_solution (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ + x₂ = x₃^2 + x₄^2 + 6*x₃*x₄) ∧
  (x₁ + x₃ = x₂^2 + x₄^2 + 6*x₂*x₄) ∧
  (x₁ + x₄ = x₂^2 + x₃^2 + 6*x₂*x₃) ∧
  (x₂ + x₃ = x₁^2 + x₄^2 + 6*x₁*x₄) ∧
  (x₂ + x₄ = x₁^2 + x₃^2 + 6*x₁*x₃) ∧
  (x₃ + x₄ = x₁^2 + x₂^2 + 6*x₁*x₂) →
  (∃ c : ℝ, (x₁ = c ∧ x₂ = c ∧ x₃ = c ∧ x₄ = -3*c) ∨
            (x₁ = c ∧ x₂ = c ∧ x₃ = c ∧ x₄ = 1 - 3*c)) :=
by sorry


end NUMINAMATH_CALUDE_quadruplet_solution_l1014_101493


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l1014_101489

/-- The number of ways to distribute n distinct objects into k indistinguishable containers --/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 5 distinct objects into 3 indistinguishable containers
    results in 51 different arrangements --/
theorem distribute_five_into_three :
  distribute 5 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l1014_101489


namespace NUMINAMATH_CALUDE_total_berries_picked_l1014_101437

theorem total_berries_picked (total : ℕ) : 
  (total / 2 : ℚ) + (total / 3 : ℚ) + 7 = total → total = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_total_berries_picked_l1014_101437


namespace NUMINAMATH_CALUDE_trip_time_calculation_l1014_101406

theorem trip_time_calculation (distance : ℝ) (speed1 speed2 time1 : ℝ) 
  (h1 : speed1 = 100)
  (h2 : speed2 = 50)
  (h3 : time1 = 5)
  (h4 : distance = speed1 * time1) :
  distance / speed2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l1014_101406


namespace NUMINAMATH_CALUDE_complex_modulus_l1014_101495

theorem complex_modulus (z : ℂ) (h : z * Complex.I = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1014_101495


namespace NUMINAMATH_CALUDE_solve_problem_l1014_101416

/-- The number of Adidas shoes Alice sold to meet her quota -/
def problem : Prop :=
  let quota : ℕ := 1000
  let adidas_price : ℕ := 45
  let nike_price : ℕ := 60
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let reebok_sold : ℕ := 9
  let above_goal : ℕ := 65
  ∃ adidas_sold : ℕ,
    adidas_sold * adidas_price + nike_sold * nike_price + reebok_sold * reebok_price = quota + above_goal ∧
    adidas_sold = 6

theorem solve_problem : problem := by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l1014_101416


namespace NUMINAMATH_CALUDE_percent_problem_l1014_101452

theorem percent_problem (x : ℝ) : (0.25 * x = 0.12 * 1500 - 15) → x = 660 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l1014_101452


namespace NUMINAMATH_CALUDE_bryans_skittles_count_l1014_101449

theorem bryans_skittles_count (ben_mm : ℕ) (bryan_extra : ℕ) 
  (h1 : ben_mm = 20) 
  (h2 : bryan_extra = 30) : 
  ben_mm + bryan_extra = 50 := by
  sorry

end NUMINAMATH_CALUDE_bryans_skittles_count_l1014_101449


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l1014_101425

theorem initial_number_of_girls :
  ∀ (n : ℕ) (A : ℝ),
  n > 0 →
  (n * (A + 3) - n * A = 94 - 70) →
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l1014_101425


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_shared_foci_l1014_101408

-- Define the ellipse equation
def ellipse (x y a : ℝ) : Prop := x^2 / 6 + y^2 / a^2 = 1

-- Define the hyperbola equation
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 4 = 1

-- Define the property of shared foci
def shared_foci (a : ℝ) : Prop :=
  ∀ x y : ℝ, ellipse x y a ∧ hyperbola x y a → 
    (6 - a^2).sqrt = (a + 4).sqrt

-- Theorem statement
theorem ellipse_hyperbola_shared_foci :
  ∃ a : ℝ, a > 0 ∧ shared_foci a ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_shared_foci_l1014_101408
