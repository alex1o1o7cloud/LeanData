import Mathlib

namespace NUMINAMATH_GPT_min_abs_sum_l904_90438

theorem min_abs_sum (x y z : ℝ) (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 4) 
  (hy_eq : y^2 = x^2 + 2) (hz_eq : z^2 = y^2 + 2) : 
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_l904_90438


namespace NUMINAMATH_GPT_pages_left_to_be_read_l904_90415

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end NUMINAMATH_GPT_pages_left_to_be_read_l904_90415


namespace NUMINAMATH_GPT_equal_share_expense_l904_90421

theorem equal_share_expense (L B C X : ℝ) : 
  let T := L + B + C - X
  let share := T / 3 
  L + (share - L) == (B + C - X - 2 * L) / 3 := 
by
  sorry

end NUMINAMATH_GPT_equal_share_expense_l904_90421


namespace NUMINAMATH_GPT_platform_length_proof_l904_90410

noncomputable def train_length : ℝ := 1200
noncomputable def time_to_cross_tree : ℝ := 120
noncomputable def time_to_cross_platform : ℝ := 240
noncomputable def speed_of_train : ℝ := train_length / time_to_cross_tree
noncomputable def platform_length : ℝ := 2400 - train_length

theorem platform_length_proof (h1 : train_length = 1200) (h2 : time_to_cross_tree = 120) (h3 : time_to_cross_platform = 240) :
  platform_length = 1200 := by
  sorry

end NUMINAMATH_GPT_platform_length_proof_l904_90410


namespace NUMINAMATH_GPT_percentage_decrease_wages_l904_90463

theorem percentage_decrease_wages (W : ℝ) (P : ℝ) : 
  (0.20 * W * (1 - P / 100)) = 0.70 * (0.20 * W) → 
  P = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_wages_l904_90463


namespace NUMINAMATH_GPT_mechanic_hourly_rate_l904_90452

-- Definitions and conditions
def total_bill : ℕ := 450
def parts_charge : ℕ := 225
def hours_worked : ℕ := 5

-- The main theorem to prove
theorem mechanic_hourly_rate : (total_bill - parts_charge) / hours_worked = 45 := by
  sorry

end NUMINAMATH_GPT_mechanic_hourly_rate_l904_90452


namespace NUMINAMATH_GPT_smallest_possible_X_l904_90487

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end NUMINAMATH_GPT_smallest_possible_X_l904_90487


namespace NUMINAMATH_GPT_greatest_integer_b_l904_90402

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 12 ≠ 0) ↔ b = 6 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_b_l904_90402


namespace NUMINAMATH_GPT_triangle_altitude_from_rectangle_l904_90477

theorem triangle_altitude_from_rectangle (a b : ℕ) (A : ℕ) (h : ℕ) (H1 : a = 7) (H2 : b = 21) (H3 : A = 147) (H4 : a * b = A) (H5 : 2 * A = h * b) : h = 14 :=
sorry

end NUMINAMATH_GPT_triangle_altitude_from_rectangle_l904_90477


namespace NUMINAMATH_GPT_exp_monotonic_iff_l904_90467

theorem exp_monotonic_iff (a b : ℝ) : (a > b) ↔ (Real.exp a > Real.exp b) :=
sorry

end NUMINAMATH_GPT_exp_monotonic_iff_l904_90467


namespace NUMINAMATH_GPT_infinite_triangle_area_sum_l904_90459

noncomputable def rectangle_area_sum : ℝ :=
  let AB := 2
  let BC := 1
  let Q₁ := 0.5
  let base_area := (1/2) * Q₁ * (1/4)
  base_area * (1/(1 - 1/4))

theorem infinite_triangle_area_sum :
  rectangle_area_sum = 1/12 :=
by
  sorry

end NUMINAMATH_GPT_infinite_triangle_area_sum_l904_90459


namespace NUMINAMATH_GPT_train_length_l904_90417

theorem train_length (speed_kph : ℝ) (time_sec : ℝ) (speed_mps : ℝ) (length_m : ℝ) 
  (h1 : speed_kph = 60) 
  (h2 : time_sec = 42) 
  (h3 : speed_mps = speed_kph * 1000 / 3600) 
  (h4 : length_m = speed_mps * time_sec) :
  length_m = 700.14 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l904_90417


namespace NUMINAMATH_GPT_greatest_divisor_four_consecutive_squared_l904_90497

theorem greatest_divisor_four_consecutive_squared :
  ∀ (n: ℕ), ∃ m: ℕ, (∀ (n: ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) ∧ m = 144 := 
sorry

end NUMINAMATH_GPT_greatest_divisor_four_consecutive_squared_l904_90497


namespace NUMINAMATH_GPT_true_propositions_l904_90440

theorem true_propositions :
  (∀ x y, (x * y = 1 → x * y = (x * y))) ∧
  (¬ (∀ (a b : ℝ), (∀ (A B : ℝ), a = b → A = B) ∧ (A = B → a ≠ b))) ∧
  (∀ m : ℝ, (m ≤ 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0)) ↔
    (true ∧ true ∧ true) :=
by sorry

end NUMINAMATH_GPT_true_propositions_l904_90440


namespace NUMINAMATH_GPT_find_a2023_l904_90475

variable {a : ℕ → ℕ}
variable {x : ℕ}

def sequence_property (a: ℕ → ℕ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = 20

theorem find_a2023 (h1 : sequence_property a) 
                   (h2 : a 2 = 2 * x) 
                   (h3 : a 18 = 9 + x) 
                   (h4 : a 65 = 6 - x) : 
  a 2023 = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_a2023_l904_90475


namespace NUMINAMATH_GPT_garden_plant_count_l904_90488

theorem garden_plant_count :
  let rows := 52
  let columns := 15
  rows * columns = 780 := 
by
  sorry

end NUMINAMATH_GPT_garden_plant_count_l904_90488


namespace NUMINAMATH_GPT_area_of_square_with_diagonal_two_l904_90400

theorem area_of_square_with_diagonal_two {a d : ℝ} (h : d = 2) (h' : d = a * Real.sqrt 2) : a^2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_area_of_square_with_diagonal_two_l904_90400


namespace NUMINAMATH_GPT_people_remaining_at_end_l904_90414

def total_people_start : ℕ := 600
def girls_start : ℕ := 240
def boys_start : ℕ := total_people_start - girls_start
def boys_left_early : ℕ := boys_start / 4
def girls_left_early : ℕ := girls_start / 8
def total_left_early : ℕ := boys_left_early + girls_left_early
def people_remaining : ℕ := total_people_start - total_left_early

theorem people_remaining_at_end : people_remaining = 480 := by
  sorry

end NUMINAMATH_GPT_people_remaining_at_end_l904_90414


namespace NUMINAMATH_GPT_range_of_a_l904_90494

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 3 < x ∧ x < 4 ∧ ax^2 - 4*a*x - 2 > 0) ↔ a < -2/3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l904_90494


namespace NUMINAMATH_GPT_no_polygon_with_1974_diagonals_l904_90430

theorem no_polygon_with_1974_diagonals :
  ¬ ∃ N : ℕ, N * (N - 3) / 2 = 1974 :=
sorry

end NUMINAMATH_GPT_no_polygon_with_1974_diagonals_l904_90430


namespace NUMINAMATH_GPT_hidden_message_is_correct_l904_90466

def russian_alphabet_mapping : Char → Nat
| 'А' => 1
| 'Б' => 2
| 'В' => 3
| 'Г' => 4
| 'Д' => 5
| 'Е' => 6
| 'Ё' => 7
| 'Ж' => 8
| 'З' => 9
| 'И' => 10
| 'Й' => 11
| 'К' => 12
| 'Л' => 13
| 'М' => 14
| 'Н' => 15
| 'О' => 16
| 'П' => 17
| 'Р' => 18
| 'С' => 19
| 'Т' => 20
| 'У' => 21
| 'Ф' => 22
| 'Х' => 23
| 'Ц' => 24
| 'Ч' => 25
| 'Ш' => 26
| 'Щ' => 27
| 'Ъ' => 28
| 'Ы' => 29
| 'Ь' => 30
| 'Э' => 31
| 'Ю' => 32
| 'Я' => 33
| _ => 0

def prime_p : ℕ := 7 -- Assume some prime number p

def grid_position (p : ℕ) (k : ℕ) := p * k

theorem hidden_message_is_correct :
  ∃ m : String, m = "ПАРОЛЬ МЕДВЕЖАТА" :=
by
  let message := "ПАРОЛЬ МЕДВЕЖАТА"
  have h1 : russian_alphabet_mapping 'П' = 17 := by sorry
  have h2 : russian_alphabet_mapping 'А' = 1 := by sorry
  have h3 : russian_alphabet_mapping 'Р' = 18 := by sorry
  have h4 : russian_alphabet_mapping 'О' = 16 := by sorry
  have h5 : russian_alphabet_mapping 'Л' = 13 := by sorry
  have h6 : russian_alphabet_mapping 'Ь' = 29 := by sorry
  have h7 : russian_alphabet_mapping 'М' = 14 := by sorry
  have h8 : russian_alphabet_mapping 'Е' = 5 := by sorry
  have h9 : russian_alphabet_mapping 'Д' = 10 := by sorry
  have h10 : russian_alphabet_mapping 'В' = 3 := by sorry
  have h11 : russian_alphabet_mapping 'Ж' = 8 := by sorry
  have h12 : russian_alphabet_mapping 'Т' = 20 := by sorry
  have g1 : grid_position prime_p 17 = 119 := by sorry
  have g2 : grid_position prime_p 1 = 7 := by sorry
  have g3 : grid_position prime_p 18 = 126 := by sorry
  have g4 : grid_position prime_p 16 = 112 := by sorry
  have g5 : grid_position prime_p 13 = 91 := by sorry
  have g6 : grid_position prime_p 29 = 203 := by sorry
  have g7 : grid_position prime_p 14 = 98 := by sorry
  have g8 : grid_position prime_p 5 = 35 := by sorry
  have g9 : grid_position prime_p 10 = 70 := by sorry
  have g10 : grid_position prime_p 3 = 21 := by sorry
  have g11 : grid_position prime_p 8 = 56 := by sorry
  have g12 : grid_position prime_p 20 = 140 := by sorry
  existsi message
  rfl

end NUMINAMATH_GPT_hidden_message_is_correct_l904_90466


namespace NUMINAMATH_GPT_bob_cookie_price_same_as_jane_l904_90427

theorem bob_cookie_price_same_as_jane
  (r_jane : ℝ)
  (s_bob : ℝ)
  (dough_jane : ℝ)
  (num_jane_cookies : ℕ)
  (price_jane_cookie : ℝ)
  (total_earning_jane : ℝ)
  (num_cookies_bob : ℝ)
  (price_bob_cookie : ℝ) :
  r_jane = 4 ∧
  s_bob = 6 ∧
  dough_jane = 18 * (Real.pi * r_jane^2) ∧
  price_jane_cookie = 0.50 ∧
  total_earning_jane = 18 * 50 ∧
  num_cookies_bob = dough_jane / s_bob^2 ∧
  total_earning_jane = num_cookies_bob * price_bob_cookie →
  price_bob_cookie = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bob_cookie_price_same_as_jane_l904_90427


namespace NUMINAMATH_GPT_Kirill_is_69_l904_90485

/-- Kirill is 14 centimeters shorter than his brother.
    Their sister's height is twice the height of Kirill.
    Their cousin's height is 3 centimeters more than the sister's height.
    Together, their heights equal 432 centimeters.
    We aim to prove that Kirill's height is 69 centimeters.
-/
def Kirill_height (K : ℕ) : Prop :=
  let brother_height := K + 14
  let sister_height := 2 * K
  let cousin_height := 2 * K + 3
  K + brother_height + sister_height + cousin_height = 432

theorem Kirill_is_69 {K : ℕ} (h : Kirill_height K) : K = 69 :=
by
  sorry

end NUMINAMATH_GPT_Kirill_is_69_l904_90485


namespace NUMINAMATH_GPT_solution_interval_l904_90451

noncomputable def set_of_solutions : Set ℝ :=
  {x : ℝ | 4 * x - 3 < (x - 2) ^ 2 ∧ (x - 2) ^ 2 < 6 * x - 5}

theorem solution_interval :
  set_of_solutions = {x : ℝ | 7 < x ∧ x < 9} := by
  sorry

end NUMINAMATH_GPT_solution_interval_l904_90451


namespace NUMINAMATH_GPT_root_properties_of_polynomial_l904_90401

variables {r s t : ℝ}

def polynomial (x : ℝ) : ℝ := 6 * x^3 + 4 * x^2 + 1500 * x + 3000

theorem root_properties_of_polynomial :
  (∀ x : ℝ, polynomial x = 0 → (x = r ∨ x = s ∨ x = t)) →
  (r + s + t = -2 / 3) →
  (r * s + r * t + s * t = 250) →
  (r * s * t = -500) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992 / 27 :=
by
  sorry

end NUMINAMATH_GPT_root_properties_of_polynomial_l904_90401


namespace NUMINAMATH_GPT_find_extrema_on_interval_l904_90422

noncomputable def y (x : ℝ) := (10 * x + 10) / (x^2 + 2 * x + 2)

theorem find_extrema_on_interval :
  ∃ (min_val max_val : ℝ) (min_x max_x : ℝ), 
    min_val = 0 ∧ min_x = -1 ∧ max_val = 5 ∧ max_x = 0 ∧ 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≥ min_val) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≤ max_val) :=
by
  sorry

end NUMINAMATH_GPT_find_extrema_on_interval_l904_90422


namespace NUMINAMATH_GPT_circle_arc_and_circumference_l904_90416

theorem circle_arc_and_circumference (C_X : ℝ) (θ_YOZ : ℝ) (C_D : ℝ) (r_X r_D : ℝ) :
  C_X = 100 ∧ θ_YOZ = 150 ∧ r_X = 50 / π ∧ r_D = 25 / π ∧ C_D = 50 →
  (θ_YOZ / 360) * C_X = 500 / 12 ∧ 2 * π * r_D = C_D :=
by sorry

end NUMINAMATH_GPT_circle_arc_and_circumference_l904_90416


namespace NUMINAMATH_GPT_polynomial_factorization_l904_90423

theorem polynomial_factorization :
  ∀ (a b c : ℝ),
    a * (b - c) ^ 4 + b * (c - a) ^ 4 + c * (a - b) ^ 4 =
    (a - b) * (b - c) * (c - a) * (a + b + c) :=
  by
    intro a b c
    sorry

end NUMINAMATH_GPT_polynomial_factorization_l904_90423


namespace NUMINAMATH_GPT_winning_candidate_percentage_l904_90468

theorem winning_candidate_percentage (P: ℝ) (majority diff votes totalVotes : ℝ)
    (h1 : majority = 184)
    (h2 : totalVotes = 460)
    (h3 : diff = P * totalVotes / 100 - (100 - P) * totalVotes / 100)
    (h4 : majority = diff) : P = 70 :=
by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l904_90468


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l904_90489

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^3 - x^2 + 1 ≤ 0) ↔ ∃ (x₀ : ℝ), x₀^3 - x₀^2 + 1 > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_universal_proposition_l904_90489


namespace NUMINAMATH_GPT_solution_of_system_l904_90456

theorem solution_of_system : ∃ x y : ℝ, (2 * x + y = 2) ∧ (x - y = 1) ∧ (x = 1) ∧ (y = 0) := 
by
  sorry

end NUMINAMATH_GPT_solution_of_system_l904_90456


namespace NUMINAMATH_GPT_wall_clock_ring_interval_l904_90457

theorem wall_clock_ring_interval 
  (n : ℕ)                -- Number of rings in a day
  (total_minutes : ℕ)    -- Total minutes in a day
  (intervals : ℕ) :       -- Number of intervals
  n = 6 ∧ total_minutes = 1440 ∧ intervals = n - 1 ∧ intervals = 5
    → (1440 / intervals = 288 ∧ 288 / 60 = 4∧ 288 % 60 = 48) := sorry

end NUMINAMATH_GPT_wall_clock_ring_interval_l904_90457


namespace NUMINAMATH_GPT_parallelogram_fourth_vertex_distance_l904_90429

theorem parallelogram_fourth_vertex_distance (d1 d2 d3 d4 : ℝ) (h1 : d1 = 1) (h2 : d2 = 3) (h3 : d3 = 5) :
    d4 = 7 :=
sorry

end NUMINAMATH_GPT_parallelogram_fourth_vertex_distance_l904_90429


namespace NUMINAMATH_GPT_salary_january_l904_90428

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 :=
by 
  sorry

end NUMINAMATH_GPT_salary_january_l904_90428


namespace NUMINAMATH_GPT_find_p_of_probability_l904_90444

-- Define the conditions and the problem statement
theorem find_p_of_probability
  (A_red_prob : ℚ := 1/3) -- probability of drawing a red ball from bag A
  (A_to_B_ratio : ℚ := 1/2) -- ratio of number of balls in bag A to bag B
  (combined_red_prob : ℚ := 2/5) -- total probability of drawing a red ball after combining balls
  : p = 13 / 30 := by
  sorry

end NUMINAMATH_GPT_find_p_of_probability_l904_90444


namespace NUMINAMATH_GPT_sum_of_consecutive_even_integers_l904_90439

theorem sum_of_consecutive_even_integers (n : ℕ) (h1 : (n - 2) + (n + 2) = 162) (h2 : ∃ k : ℕ, n = k^2) :
  (n - 2) + n + (n + 2) = 243 :=
by
  -- no proof required
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_integers_l904_90439


namespace NUMINAMATH_GPT_quadratic_always_positive_l904_90464

theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) (hpos : a > 0) (hdisc : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_always_positive_l904_90464


namespace NUMINAMATH_GPT_deposit_time_l904_90478

theorem deposit_time (r t : ℕ) : 
  8000 + 8000 * r * t / 100 = 10200 → 
  8000 + 8000 * (r + 2) * t / 100 = 10680 → 
  t = 3 :=
by 
  sorry

end NUMINAMATH_GPT_deposit_time_l904_90478


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l904_90426

variable {x y z : ℝ}

def positive_reals (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0

theorem inequality_proof (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry -- Proof goes here

theorem equality_condition (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * z ∧ y = z :=
sorry -- Proof goes here

end NUMINAMATH_GPT_inequality_proof_equality_condition_l904_90426


namespace NUMINAMATH_GPT_sequence_eq_third_term_l904_90419

theorem sequence_eq_third_term 
  (p : ℤ → ℤ)
  (a : ℕ → ℤ)
  (n : ℕ) (h₁ : n > 2)
  (h₂ : a 2 = p (a 1))
  (h₃ : a 3 = p (a 2))
  (h₄ : ∀ k, 4 ≤ k ∧ k ≤ n → a k = p (a (k - 1)))
  (h₅ : a 1 = p (a n))
  : a 1 = a 3 :=
sorry

end NUMINAMATH_GPT_sequence_eq_third_term_l904_90419


namespace NUMINAMATH_GPT_largest_value_of_x_not_defined_l904_90407

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b*b - 4*a*c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2*a)
  let x2 := (-b - sqrt_discriminant) / (2*a)
  (x1, x2)

noncomputable def largest_root : ℝ :=
  let (x1, x2) := quadratic_formula 4 (-81) 49
  if x1 > x2 then x1 else x2

theorem largest_value_of_x_not_defined :
  largest_root = 19.6255 :=
by
  sorry

end NUMINAMATH_GPT_largest_value_of_x_not_defined_l904_90407


namespace NUMINAMATH_GPT_simplify_expression_l904_90476

theorem simplify_expression :
  ((5 ^ 7 + 2 ^ 8) * (1 ^ 5 - (-1) ^ 5) ^ 10) = 80263680 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l904_90476


namespace NUMINAMATH_GPT_probability_of_finding_last_defective_product_on_fourth_inspection_l904_90406

theorem probability_of_finding_last_defective_product_on_fourth_inspection :
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  probability = 1 / 5 :=
by
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  have : probability = 1 / 5 := sorry
  exact this

end NUMINAMATH_GPT_probability_of_finding_last_defective_product_on_fourth_inspection_l904_90406


namespace NUMINAMATH_GPT_nice_set_l904_90474

def nice (P : Set (ℤ × ℤ)) : Prop :=
  ∀ (a b c d : ℤ), (a, b) ∈ P ∧ (c, d) ∈ P → (b, a) ∈ P ∧ (a + c, b - d) ∈ P

def is_solution (p q : ℤ) : Prop :=
  Int.gcd p q = 1 ∧ p % 2 ≠ q % 2

theorem nice_set (p q : ℤ) (P : Set (ℤ × ℤ)) :
  nice P → (p, q) ∈ P → is_solution p q → P = Set.univ := 
  sorry

end NUMINAMATH_GPT_nice_set_l904_90474


namespace NUMINAMATH_GPT_exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l904_90455

theorem exists_five_integers_sum_fifth_powers (A B C D E : ℤ) : 
  ∃ (A B C D E : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 + E^5 :=
  by
    sorry

theorem no_four_integers_sum_fifth_powers (A B C D : ℤ) : 
  ¬ ∃ (A B C D : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 :=
  by
    sorry

end NUMINAMATH_GPT_exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l904_90455


namespace NUMINAMATH_GPT_square_must_rotate_at_least_5_turns_l904_90405

-- Define the square and pentagon as having equal side lengths
def square_sides : Nat := 4
def pentagon_sides : Nat := 5

-- The problem requires us to prove that the square needs to rotate at least 5 full turns
theorem square_must_rotate_at_least_5_turns :
  let lcm := Nat.lcm square_sides pentagon_sides
  lcm / square_sides = 5 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_square_must_rotate_at_least_5_turns_l904_90405


namespace NUMINAMATH_GPT_opposite_difference_five_times_l904_90432

variable (a b : ℤ) -- Using integers for this example

theorem opposite_difference_five_times (a b : ℤ) : (-a - 5 * b) = -(a) - (5 * b) := 
by
  -- The proof details would be filled in here
  sorry

end NUMINAMATH_GPT_opposite_difference_five_times_l904_90432


namespace NUMINAMATH_GPT_alex_shirts_count_l904_90460

theorem alex_shirts_count (j a b : ℕ) (h1 : j = a + 3) (h2 : b = j + 8) (h3 : b = 15) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_alex_shirts_count_l904_90460


namespace NUMINAMATH_GPT_cost_of_largest_pot_l904_90473

theorem cost_of_largest_pot
    (x : ℝ)
    (hx : 6 * x + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) = 8.25) :
    (x + 0.5) = 1.625 :=
sorry

end NUMINAMATH_GPT_cost_of_largest_pot_l904_90473


namespace NUMINAMATH_GPT_total_goals_scored_l904_90441

theorem total_goals_scored (g1 t1 g2 t2 : ℕ)
  (h1 : g1 = 2)
  (h2 : g1 = t1 - 3)
  (h3 : t2 = 6)
  (h4 : g2 = t2 - 2) :
  g1 + t1 + g2 + t2 = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_goals_scored_l904_90441


namespace NUMINAMATH_GPT_building_total_floors_l904_90495

def earl_final_floor (start : ℕ) : ℕ :=
  start + 5 - 2 + 7

theorem building_total_floors (start : ℕ) (current : ℕ) (remaining : ℕ) (total : ℕ) :
  earl_final_floor start = current →
  remaining = 9 →
  total = current + remaining →
  start = 1 →
  total = 20 := by
sorry

end NUMINAMATH_GPT_building_total_floors_l904_90495


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_q_l904_90471

theorem arithmetic_sequence_sum_q (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 2) + a (n + 1) = 2 * a n)
  (hq : q ≠ 1) :
  S 5 = 11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_q_l904_90471


namespace NUMINAMATH_GPT_survey_total_parents_l904_90469

theorem survey_total_parents (P : ℝ)
  (h1 : 0.15 * P + 0.60 * P + 0.20 * 0.25 * P + 0.05 * P = P)
  (h2 : 0.05 * P = 6) : 
  P = 120 :=
sorry

end NUMINAMATH_GPT_survey_total_parents_l904_90469


namespace NUMINAMATH_GPT_solution_set_inequality_l904_90482

theorem solution_set_inequality (a : ℝ) :
  ∀ x : ℝ,
    (12 * x^2 - a * x > a^2) →
    ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
     (a = 0 ∧ x ≠ 0) ∨
     (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l904_90482


namespace NUMINAMATH_GPT_find_z_l904_90408

theorem find_z 
  {x y z : ℕ}
  (hx : x = 4)
  (hy : y = 7)
  (h_least : x - y - z = 17) : 
  z = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_z_l904_90408


namespace NUMINAMATH_GPT_fraction_done_by_B_l904_90499

theorem fraction_done_by_B {A B : ℝ} (h : A = (2/5) * B) : (B / (A + B)) = (5/7) :=
by
  sorry

end NUMINAMATH_GPT_fraction_done_by_B_l904_90499


namespace NUMINAMATH_GPT_real_values_of_x_l904_90449

theorem real_values_of_x (x : ℝ) (h : x ≠ 4) :
  (x * (x + 1) / (x - 4)^2 ≥ 15) ↔ (x ≤ 3 ∨ (40/7 < x ∧ x < 4) ∨ x > 4) :=
by sorry

end NUMINAMATH_GPT_real_values_of_x_l904_90449


namespace NUMINAMATH_GPT_select_medical_team_l904_90491

open Nat

theorem select_medical_team : 
  let male_doctors := 5
  let female_doctors := 4
  let selected_doctors := 3
  (male_doctors.choose 1 * female_doctors.choose 2 + male_doctors.choose 2 * female_doctors.choose 1) = 70 :=
by
  sorry

end NUMINAMATH_GPT_select_medical_team_l904_90491


namespace NUMINAMATH_GPT_original_price_of_shoes_l904_90450

theorem original_price_of_shoes (P : ℝ) (h1 : 0.80 * P = 480) : P = 600 := 
by
  sorry

end NUMINAMATH_GPT_original_price_of_shoes_l904_90450


namespace NUMINAMATH_GPT_jaden_toy_cars_left_l904_90436

-- Definitions for each condition
def initial_toys : ℕ := 14
def purchased_toys : ℕ := 28
def birthday_toys : ℕ := 12
def given_to_sister : ℕ := 8
def given_to_vinnie : ℕ := 3
def traded_lost : ℕ := 5
def traded_received : ℕ := 7

-- The final number of toy cars Jaden has
def final_toys : ℕ :=
  initial_toys + purchased_toys + birthday_toys - given_to_sister - given_to_vinnie + (traded_received - traded_lost)

theorem jaden_toy_cars_left : final_toys = 45 :=
by
  -- The proof will be filled in here 
  sorry

end NUMINAMATH_GPT_jaden_toy_cars_left_l904_90436


namespace NUMINAMATH_GPT_triangle_is_obtuse_l904_90462

-- Definitions based on given conditions
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  if a ≥ b ∧ a ≥ c then a^2 > b^2 + c^2
  else if b ≥ a ∧ b ≥ c then b^2 > a^2 + c^2
  else c^2 > a^2 + b^2

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove
theorem triangle_is_obtuse : is_triangle 4 6 8 ∧ is_obtuse_triangle 4 6 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l904_90462


namespace NUMINAMATH_GPT_nora_nuts_problem_l904_90486

theorem nora_nuts_problem :
  ∃ n : ℕ, (∀ (a p c : ℕ), 30 * n = 18 * a ∧ 30 * n = 21 * p ∧ 30 * n = 16 * c) ∧ n = 34 :=
by
  -- Provided conditions and solution steps will go here.
  sorry

end NUMINAMATH_GPT_nora_nuts_problem_l904_90486


namespace NUMINAMATH_GPT_frac_abs_div_a_plus_one_l904_90404

theorem frac_abs_div_a_plus_one (a : ℝ) (h : a ≠ 0) : abs a / a + 1 = 0 ∨ abs a / a + 1 = 2 :=
by sorry

end NUMINAMATH_GPT_frac_abs_div_a_plus_one_l904_90404


namespace NUMINAMATH_GPT_apples_in_market_l904_90447

theorem apples_in_market (A O : ℕ) 
    (h1 : A = O + 27) 
    (h2 : A + O = 301) : 
    A = 164 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_market_l904_90447


namespace NUMINAMATH_GPT_greater_number_is_64_l904_90433

theorem greater_number_is_64
  (x y : ℕ)
  (h1 : x * y = 2048)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) :
  x = 64 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_greater_number_is_64_l904_90433


namespace NUMINAMATH_GPT_total_earnings_l904_90437

theorem total_earnings (L A J M : ℝ) 
  (hL : L = 2000) 
  (hA : A = 0.70 * L) 
  (hJ : J = 1.50 * A) 
  (hM : M = 0.40 * J) 
  : L + A + J + M = 6340 := 
  by 
    sorry

end NUMINAMATH_GPT_total_earnings_l904_90437


namespace NUMINAMATH_GPT_solution_set_l904_90435

-- Define determinant operation on 2x2 matrices
def determinant (a b c d : ℝ) := a * d - b * c

-- Define the condition inequality
def condition (x : ℝ) : Prop :=
  determinant x 3 (-x) x < determinant 2 0 1 2

-- Prove that the solution to the condition is -4 < x < 1
theorem solution_set : {x : ℝ | condition x} = {x : ℝ | -4 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l904_90435


namespace NUMINAMATH_GPT_probability_of_different_value_and_suit_l904_90470

theorem probability_of_different_value_and_suit :
  let total_cards := 52
  let first_card_choices := 52
  let remaining_cards := 51
  let different_suits := 3
  let different_values := 12
  let favorable_outcomes := different_suits * different_values
  let total_outcomes := remaining_cards
  let probability := favorable_outcomes / total_outcomes
  probability = 12 / 17 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_different_value_and_suit_l904_90470


namespace NUMINAMATH_GPT_soda_cost_proof_l904_90431

-- Define the main facts about the weeds
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32 / 2  -- Only half the weeds in the grass

-- Define the earning rate
def earning_per_weed : ℕ := 6

-- Define the total earnings and the remaining money conditions
def total_earnings : ℕ := (weeds_flower_bed + weeds_vegetable_patch + weeds_grass) * earning_per_weed
def remaining_money : ℕ := 147

-- Define the cost of the soda
def cost_of_soda : ℕ := total_earnings - remaining_money

-- Problem statement: Prove that the cost of the soda is 99 cents
theorem soda_cost_proof : cost_of_soda = 99 := by
  sorry

end NUMINAMATH_GPT_soda_cost_proof_l904_90431


namespace NUMINAMATH_GPT_number_of_girls_l904_90413

variable (N n g : ℕ)
variable (h1 : N = 1600)
variable (h2 : n = 200)
variable (h3 : g = 95)

theorem number_of_girls (G : ℕ) (h : g * N = G * n) : G = 760 :=
by sorry

end NUMINAMATH_GPT_number_of_girls_l904_90413


namespace NUMINAMATH_GPT_num_squares_sharing_two_vertices_l904_90472

-- Define the isosceles triangle and condition AB = AC
structure IsoscelesTriangle (A B C : Type) :=
  (AB AC : ℝ)
  (h_iso : AB = AC)

-- Define the problem statement in Lean
theorem num_squares_sharing_two_vertices 
  (A B C : Type) 
  (iso_tri : IsoscelesTriangle A B C) 
  (planeABC : ∀ P Q R : Type, P ≠ Q ∧ Q ≠ R ∧ P ≠ R) :
  ∃ n : ℕ, n = 4 := sorry

end NUMINAMATH_GPT_num_squares_sharing_two_vertices_l904_90472


namespace NUMINAMATH_GPT_range_of_a_l904_90424

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
sorry

end NUMINAMATH_GPT_range_of_a_l904_90424


namespace NUMINAMATH_GPT_find_a_extreme_value_l904_90496

theorem find_a_extreme_value (a : ℝ) :
  (f : ℝ → ℝ := λ x => x^3 + a*x^2 + 3*x - 9) →
  (f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + 3) →
  f' (-3) = 0 →
  a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_extreme_value_l904_90496


namespace NUMINAMATH_GPT_base9_sum_correct_l904_90448

def base9_addition (a b c : ℕ) : ℕ :=
  a + b + c

theorem base9_sum_correct :
  base9_addition (263) (452) (247) = 1073 :=
by sorry

end NUMINAMATH_GPT_base9_sum_correct_l904_90448


namespace NUMINAMATH_GPT_Daisy_vs_Bess_l904_90480

-- Define the conditions
def Bess_daily : ℕ := 2
def Brownie_multiple : ℕ := 3
def total_pails_per_week : ℕ := 77
def days_per_week : ℕ := 7

-- Define the weekly production for Bess
def Bess_weekly : ℕ := Bess_daily * days_per_week

-- Define the weekly production for Brownie
def Brownie_weekly : ℕ := Brownie_multiple * Bess_weekly

-- Farmer Red's total weekly milk production is the sum of Bess, Brownie, and Daisy's production
-- We need to prove the difference in weekly production between Daisy and Bess is 7 pails.
theorem Daisy_vs_Bess (Daisy_weekly : ℕ) (h : Bess_weekly + Brownie_weekly + Daisy_weekly = total_pails_per_week) :
  Daisy_weekly - Bess_weekly = 7 :=
by
  sorry

end NUMINAMATH_GPT_Daisy_vs_Bess_l904_90480


namespace NUMINAMATH_GPT_similar_triangles_y_value_l904_90484

theorem similar_triangles_y_value :
  ∀ (y : ℚ),
    (12 : ℚ) / y = (9 : ℚ) / 6 → 
    y = 8 :=
by
  intros y h
  sorry

end NUMINAMATH_GPT_similar_triangles_y_value_l904_90484


namespace NUMINAMATH_GPT_expression_value_l904_90434

theorem expression_value (a b c : ℚ) (h₁ : b = 8) (h₂ : c = 5) (h₃ : a * b * c = 2 * (a + b + c) + 14) : 
  (c - a) ^ 2 + b = 8513 / 361 := by 
  sorry

end NUMINAMATH_GPT_expression_value_l904_90434


namespace NUMINAMATH_GPT_wyatt_bought_4_cartons_of_juice_l904_90481

/-- 
Wyatt's mother gave him $74 to go to the store.
Wyatt bought 5 loaves of bread, each costing $5.
Each carton of orange juice cost $2.
Wyatt has $41 left.
We need to prove that Wyatt bought 4 cartons of orange juice.
-/
theorem wyatt_bought_4_cartons_of_juice (initial_money spent_money loaves_price juice_price loaves_qty money_left juice_qty : ℕ)
  (h1 : initial_money = 74)
  (h2 : money_left = 41)
  (h3 : loaves_price = 5)
  (h4 : juice_price = 2)
  (h5 : loaves_qty = 5)
  (h6 : spent_money = initial_money - money_left)
  (h7 : spent_money = loaves_qty * loaves_price + juice_qty * juice_price) :
  juice_qty = 4 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_wyatt_bought_4_cartons_of_juice_l904_90481


namespace NUMINAMATH_GPT_rate_of_interest_l904_90403

-- Given conditions
def P : ℝ := 1500
def SI : ℝ := 735
def r : ℝ := 7
def t := r  -- The time period in years is equal to the rate of interest

-- The formula for simple interest and the goal
theorem rate_of_interest : SI = P * r * t / 100 ↔ r = 7 := 
by
  -- We will use the given conditions and check if they support r = 7
  sorry

end NUMINAMATH_GPT_rate_of_interest_l904_90403


namespace NUMINAMATH_GPT_deviation_interpretation_l904_90445

variable (average_score : ℝ)
variable (x : ℝ)

-- Given condition
def higher_than_average : Prop := x = average_score + 5

-- To prove
def lower_than_average : Prop := x = average_score - 9

theorem deviation_interpretation (x : ℝ) (h : x = average_score + 5) : x - 14 = average_score - 9 :=
by
  sorry

end NUMINAMATH_GPT_deviation_interpretation_l904_90445


namespace NUMINAMATH_GPT_convert_binary_1101_to_decimal_l904_90493

theorem convert_binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by sorry

end NUMINAMATH_GPT_convert_binary_1101_to_decimal_l904_90493


namespace NUMINAMATH_GPT_solve_gcd_problem_l904_90490

def gcd_problem : Prop :=
  gcd 1337 382 = 191

theorem solve_gcd_problem : gcd_problem := 
by 
  sorry

end NUMINAMATH_GPT_solve_gcd_problem_l904_90490


namespace NUMINAMATH_GPT_triangle_AB_length_correct_l904_90442

theorem triangle_AB_length_correct (BC AC : Real) (A : Real) 
  (hBC : BC = Real.sqrt 7) 
  (hAC : AC = 2 * Real.sqrt 3) 
  (hA : A = Real.pi / 6) :
  ∃ (AB : Real), (AB = 5 ∨ AB = 1) :=
by
  sorry

end NUMINAMATH_GPT_triangle_AB_length_correct_l904_90442


namespace NUMINAMATH_GPT_inequality_am_gm_l904_90420

theorem inequality_am_gm (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2) + y / (x^2 + y^4)) ≤ (1 / (x * y)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l904_90420


namespace NUMINAMATH_GPT_hyperbola_focal_length_l904_90458

theorem hyperbola_focal_length : 
  (∃ (f : ℝ) (x y : ℝ), (3 * x^2 - y^2 = 3) ∧ (f = 4)) :=
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_focal_length_l904_90458


namespace NUMINAMATH_GPT_max_value_expr_l904_90443

theorem max_value_expr (a b c d : ℝ) (ha : -4 ≤ a ∧ a ≤ 4) (hb : -4 ≤ b ∧ b ≤ 4) (hc : -4 ≤ c ∧ c ≤ 4) (hd : -4 ≤ d ∧ d ≤ 4) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 72 :=
sorry

end NUMINAMATH_GPT_max_value_expr_l904_90443


namespace NUMINAMATH_GPT_largest_p_plus_q_l904_90453

-- All required conditions restated as Assumptions
def triangle {R : Type*} [LinearOrderedField R] (p q : R) : Prop :=
  let B : R × R := (10, 15)
  let C : R × R := (25, 15)
  let A : R × R := (p, q)
  let M : R × R := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let area : R := (1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))
  let median_slope : R := (A.2 - M.2) / (A.1 - M.1)
  area = 100 ∧ median_slope = -3

-- Statement to be proven
theorem largest_p_plus_q {R : Type*} [LinearOrderedField R] (p q : R) :
  triangle p q → p + q = 70 / 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_p_plus_q_l904_90453


namespace NUMINAMATH_GPT_evaluate_series_sum_l904_90465

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end NUMINAMATH_GPT_evaluate_series_sum_l904_90465


namespace NUMINAMATH_GPT_correct_statement_D_l904_90409

theorem correct_statement_D : (- 3 / 5 : ℚ) < (- 4 / 7 : ℚ) :=
  by
  -- The proof step is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_correct_statement_D_l904_90409


namespace NUMINAMATH_GPT_fraction_comparison_l904_90483

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) :=
by sorry

end NUMINAMATH_GPT_fraction_comparison_l904_90483


namespace NUMINAMATH_GPT_carrots_left_over_l904_90412

theorem carrots_left_over (c g : ℕ) (h₁ : c = 47) (h₂ : g = 4) : c % g = 3 :=
by
  sorry

end NUMINAMATH_GPT_carrots_left_over_l904_90412


namespace NUMINAMATH_GPT_find_f1_increasing_on_positive_solve_inequality_l904_90446

-- Given conditions
axiom f : ℝ → ℝ
axiom domain : ∀ x, 0 < x → true
axiom f4 : f 4 = 1
axiom multiplicative : ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y
axiom less_than_zero : ∀ x, 0 < x ∧ x < 1 → f x < 0

-- Required proofs
theorem find_f1 : f 1 = 0 := sorry

theorem increasing_on_positive : ∀ x y, 0 < x → 0 < y → x < y → f x < f y := sorry

theorem solve_inequality : {x : ℝ // 3 < x ∧ x ≤ 5} := sorry

end NUMINAMATH_GPT_find_f1_increasing_on_positive_solve_inequality_l904_90446


namespace NUMINAMATH_GPT_mms_pack_count_l904_90411

def mms_per_pack (sundaes_monday : Nat) (mms_monday : Nat) (sundaes_tuesday : Nat) (mms_tuesday : Nat) (packs : Nat) : Nat :=
  (sundaes_monday * mms_monday + sundaes_tuesday * mms_tuesday) / packs

theorem mms_pack_count 
  (sundaes_monday : Nat)
  (mms_monday : Nat)
  (sundaes_tuesday : Nat)
  (mms_tuesday : Nat)
  (packs : Nat)
  (monday_total_mms : sundaes_monday * mms_monday = 240)
  (tuesday_total_mms : sundaes_tuesday * mms_tuesday = 200)
  (total_packs : packs = 11)
  : mms_per_pack sundaes_monday mms_monday sundaes_tuesday mms_tuesday packs = 40 := by
  sorry

end NUMINAMATH_GPT_mms_pack_count_l904_90411


namespace NUMINAMATH_GPT_largest_number_is_A_l904_90418

theorem largest_number_is_A (x y z w: ℕ):
  x = (8 * 9 + 5) → -- 85 in base 9 to decimal
  y = (2 * 6 * 6) → -- 200 in base 6 to decimal
  z = ((6 * 11) + 8) → -- 68 in base 11 to decimal
  w = 70 → -- 70 in base 10 remains 70
  max (max x y) (max z w) = x := -- 77 is the maximum
by
  sorry

end NUMINAMATH_GPT_largest_number_is_A_l904_90418


namespace NUMINAMATH_GPT_ordered_triples_count_eq_4_l904_90479

theorem ordered_triples_count_eq_4 :
  ∃ (S : Finset (ℝ × ℝ × ℝ)), 
    (∀ x y z : ℝ, (x, y, z) ∈ S ↔ (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧ (xy + 1 = z) ∧ (yz + 1 = x) ∧ (zx + 1 = y)) ∧
    S.card = 4 :=
sorry

end NUMINAMATH_GPT_ordered_triples_count_eq_4_l904_90479


namespace NUMINAMATH_GPT_fraction_not_going_l904_90454

theorem fraction_not_going (S J : ℕ) (h1 : J = (2:ℕ)/3 * S) 
  (h_not_junior : 3/4 * J = 3/4 * (2/3 * S)) 
  (h_not_senior : 1/3 * S = (1:ℕ)/3 * S) :
  3/4 * (2/3 * S) + 1/3 * S = 5/6 * S :=
by 
  sorry

end NUMINAMATH_GPT_fraction_not_going_l904_90454


namespace NUMINAMATH_GPT_find_third_triangle_angles_l904_90425

-- Define the problem context
variables {A B C : ℝ} -- angles of the original triangle

-- Condition: The sum of the angles in a triangle is 180 degrees
axiom sum_of_angles (a b c : ℝ) : a + b + c = 180

-- Given conditions about the triangle and inscribed circles
def original_triangle (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

def inscribed_circle (a b c : ℝ) : Prop :=
original_triangle a b c

def second_triangle (a b c : ℝ) : Prop :=
inscribed_circle a b c

def third_triangle (a b c : ℝ) : Prop :=
second_triangle a b c

-- Goal: Prove that the angles in the third triangle are 60 degrees each
theorem find_third_triangle_angles (a b c : ℝ) (ha : original_triangle a b c)
  (h_inscribed : inscribed_circle a b c)
  (h_second : second_triangle a b c)
  (h_third : third_triangle a b c) : a = 60 ∧ b = 60 ∧ c = 60 := by
sorry

end NUMINAMATH_GPT_find_third_triangle_angles_l904_90425


namespace NUMINAMATH_GPT_gymnastics_team_l904_90461

def number_of_rows (n m k : ℕ) : Prop :=
  n = k * (2 * m + k - 1) / 2

def members_in_first_row (n m k : ℕ) : Prop :=
  number_of_rows n m k ∧ 16 < k

theorem gymnastics_team : ∃ m k : ℕ, members_in_first_row 1000 m k ∧ k = 25 ∧ m = 28 :=
by
  sorry

end NUMINAMATH_GPT_gymnastics_team_l904_90461


namespace NUMINAMATH_GPT_sum_of_reciprocals_l904_90492

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l904_90492


namespace NUMINAMATH_GPT_maxRegions_four_planes_maxRegions_n_planes_l904_90498

noncomputable def maxRegions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

theorem maxRegions_four_planes : maxRegions 4 = 11 := by
  sorry

theorem maxRegions_n_planes (n : ℕ) : maxRegions n = 1 + (n * (n + 1)) / 2 := by
  sorry

end NUMINAMATH_GPT_maxRegions_four_planes_maxRegions_n_planes_l904_90498
