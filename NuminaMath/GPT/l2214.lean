import Mathlib

namespace find_line_equation_l2214_221469

theorem find_line_equation (k m b : ℝ) :
  (∃ k, |(k^2 + 7*k + 10) - (m*k + b)| = 8) ∧ (8 = 2*m + b) ∧ (b ≠ 0) → (m = 5 ∧ b = 3) := 
by
  intro h
  sorry

end find_line_equation_l2214_221469


namespace circumscribed_circle_radius_of_rectangle_l2214_221426

theorem circumscribed_circle_radius_of_rectangle 
  (a b : ℝ) 
  (h1: a = 1) 
  (angle_between_diagonals : ℝ) 
  (h2: angle_between_diagonals = 60) : 
  ∃ R, R = 1 :=
by 
  sorry

end circumscribed_circle_radius_of_rectangle_l2214_221426


namespace rth_term_l2214_221492

-- Given arithmetic progression sum formula
def Sn (n : ℕ) : ℕ := 3 * n^2 + 4 * n + 5

-- Prove that the r-th term of the sequence is 6r + 1
theorem rth_term (r : ℕ) : (Sn r) - (Sn (r - 1)) = 6 * r + 1 :=
by
  sorry

end rth_term_l2214_221492


namespace arctan_sum_l2214_221486

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l2214_221486


namespace number_of_shirts_is_39_l2214_221480

-- Define the conditions as Lean definitions.
def washing_machine_capacity : ℕ := 8
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 9

-- Define the total number of pieces of clothing based on the conditions.
def total_pieces_of_clothing : ℕ :=
  number_of_loads * washing_machine_capacity

-- Define the number of shirts.
noncomputable def number_of_shirts : ℕ :=
  total_pieces_of_clothing - number_of_sweaters

-- The actual proof problem statement.
theorem number_of_shirts_is_39 :
  number_of_shirts = 39 := by
  sorry

end number_of_shirts_is_39_l2214_221480


namespace positive_divisors_multiple_of_15_l2214_221423

theorem positive_divisors_multiple_of_15 (a b c : ℕ) (n : ℕ) (divisor : ℕ) (h_factorization : n = 6480)
  (h_prime_factorization : n = 2^4 * 3^4 * 5^1)
  (h_divisor : divisor = 2^a * 3^b * 5^c)
  (h_a_range : 0 ≤ a ∧ a ≤ 4)
  (h_b_range : 1 ≤ b ∧ b ≤ 4)
  (h_c_range : 1 ≤ c ∧ c ≤ 1) : sorry :=
sorry

end positive_divisors_multiple_of_15_l2214_221423


namespace total_cost_of_one_pencil_and_eraser_l2214_221487

/-- Lila buys 15 pencils and 7 erasers for 170 cents. A pencil costs less than an eraser, 
neither item costs exactly half as much as the other, and both items cost a whole number of cents. 
Prove that the total cost of one pencil and one eraser is 16 cents. -/
theorem total_cost_of_one_pencil_and_eraser (p e : ℕ) (h1 : 15 * p + 7 * e = 170)
  (h2 : p < e) (h3 : p ≠ e / 2) : p + e = 16 :=
sorry

end total_cost_of_one_pencil_and_eraser_l2214_221487


namespace problem_1_problem_2_l2214_221403

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Problem I
theorem problem_1 (x : ℝ) : (f x 1 ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
by sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 0 → x ≤ -3) : a = 6 :=
by sorry

end problem_1_problem_2_l2214_221403


namespace polynomial_divisible_by_seven_l2214_221418

-- Define the theorem
theorem polynomial_divisible_by_seven (n : ℤ) : 7 ∣ (n + 7)^2 - n^2 :=
by sorry

end polynomial_divisible_by_seven_l2214_221418


namespace triangle_area_l2214_221462

theorem triangle_area {r : ℝ} (h_r : r = 6) {x : ℝ} 
  (h1 : 5 * x = 2 * r)
  (h2 : x = 12 / 5) : 
  (1 / 2 * (3 * x) * (4 * x) = 34.56) :=
by
  sorry

end triangle_area_l2214_221462


namespace hat_p_at_1_l2214_221432

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - (1 + 1)*x + 1

-- Definition of displeased polynomial
def isDispleased (p : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 x4 : ℝ), p (p x1) = 0 ∧ p (p x2) = 0 ∧ p (p x3) = 0 ∧ p (p x4) = 0

-- Define the specific polynomial hat_p
def hat_p (x : ℝ) : ℝ := p x

-- Theorem statement
theorem hat_p_at_1 : isDispleased hat_p → hat_p 1 = 0 :=
by
  sorry

end hat_p_at_1_l2214_221432


namespace find_unit_prices_l2214_221452

-- Define the prices of brush and chess set
variables (x y : ℝ)

-- Condition 1: Buying 5 brushes and 12 chess sets costs 315 yuan
def condition1 : Prop := 5 * x + 12 * y = 315

-- Condition 2: Buying 8 brushes and 6 chess sets costs 240 yuan
def condition2 : Prop := 8 * x + 6 * y = 240

-- Prove that the unit price of each brush is 15 yuan and each chess set is 20 yuan
theorem find_unit_prices (hx : condition1 x y) (hy : condition2 x y) :
  x = 15 ∧ y = 20 := 
sorry

end find_unit_prices_l2214_221452


namespace inequality_solution_l2214_221406

theorem inequality_solution (x : ℝ) :
  ((2 / (x - 1)) - (3 / (x - 3)) + (2 / (x - 4)) - (2 / (x - 5)) < (1 / 15)) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by
  sorry

end inequality_solution_l2214_221406


namespace negation_of_proposition_l2214_221464

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_proposition_l2214_221464


namespace total_wheels_is_90_l2214_221453

-- Defining the conditions
def number_of_bicycles := 20
def number_of_cars := 10
def number_of_motorcycles := 5

-- Calculating the total number of wheels
def total_wheels_in_garage : Nat :=
  (2 * number_of_bicycles) + (4 * number_of_cars) + (2 * number_of_motorcycles)

-- Statement to prove
theorem total_wheels_is_90 : total_wheels_in_garage = 90 := by
  sorry

end total_wheels_is_90_l2214_221453


namespace integer_solutions_l2214_221451

theorem integer_solutions (m n : ℤ) :
  m^3 - n^3 = 2 * m * n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
sorry

end integer_solutions_l2214_221451


namespace hal_paul_difference_l2214_221496

def halAnswer : Int := 12 - (3 * 2) + 4
def paulAnswer : Int := (12 - 3) * 2 + 4

theorem hal_paul_difference :
  halAnswer - paulAnswer = -12 := by
  sorry

end hal_paul_difference_l2214_221496


namespace simplify_expression_l2214_221446

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end simplify_expression_l2214_221446


namespace rise_in_height_of_field_l2214_221460

theorem rise_in_height_of_field
  (field_length : ℝ)
  (field_width : ℝ)
  (pit_length : ℝ)
  (pit_width : ℝ)
  (pit_depth : ℝ)
  (field_area : ℝ := field_length * field_width)
  (pit_area : ℝ := pit_length * pit_width)
  (remaining_area : ℝ := field_area - pit_area)
  (pit_volume : ℝ := pit_length * pit_width * pit_depth)
  (rise_in_height : ℝ := pit_volume / remaining_area) :
  field_length = 20 →
  field_width = 10 →
  pit_length = 8 →
  pit_width = 5 →
  pit_depth = 2 →
  rise_in_height = 0.5 :=
by
  intros
  sorry

end rise_in_height_of_field_l2214_221460


namespace marilyn_initial_bottle_caps_l2214_221420

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end marilyn_initial_bottle_caps_l2214_221420


namespace length_of_train_a_l2214_221476

theorem length_of_train_a
  (speed_train_a : ℝ) (speed_train_b : ℝ) 
  (clearing_time : ℝ) (length_train_b : ℝ)
  (h1 : speed_train_a = 42)
  (h2 : speed_train_b = 30)
  (h3 : clearing_time = 12.998960083193344)
  (h4 : length_train_b = 160) :
  ∃ length_train_a : ℝ, length_train_a = 99.9792016638669 :=
by 
  sorry

end length_of_train_a_l2214_221476


namespace hyperbola_satisfies_m_l2214_221458

theorem hyperbola_satisfies_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - m * y^2 = 1)
  (h2 : ∀ a b : ℝ, (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2*a = 2 * 2*b)) : 
  m = 4 := 
sorry

end hyperbola_satisfies_m_l2214_221458


namespace nat_pow_eq_sub_two_case_l2214_221433

theorem nat_pow_eq_sub_two_case (n : ℕ) : (∃ a k : ℕ, k ≥ 2 ∧ 2^n - 1 = a^k) ↔ (n = 0 ∨ n = 1) :=
by
  sorry

end nat_pow_eq_sub_two_case_l2214_221433


namespace sum_S19_is_190_l2214_221470

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ n m, a n + a m = a (n+1) + a (m-1)

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

-- Main theorem
theorem sum_S19_is_190 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_def : sum_of_first_n_terms a S)
  (h_condition : a 6 + a 14 = 20) :
  S 19 = 190 :=
sorry

end sum_S19_is_190_l2214_221470


namespace difference_students_l2214_221450

variables {A B AB : ℕ}

theorem difference_students (h1 : A + AB + B = 800)
  (h2 : AB = 20 * (A + AB) / 100)
  (h3 : AB = 25 * (B + AB) / 100) :
  A - B = 100 :=
sorry

end difference_students_l2214_221450


namespace school_travel_time_is_12_l2214_221468

noncomputable def time_to_school (T : ℕ) : Prop :=
  let extra_time := 6
  let total_distance_covered := 2 * extra_time
  T = total_distance_covered

theorem school_travel_time_is_12 :
  ∃ T : ℕ, time_to_school T ∧ T = 12 :=
by
  sorry

end school_travel_time_is_12_l2214_221468


namespace trigonometric_identity_l2214_221409

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α) ^ 2 = 11 / 5 :=
sorry

end trigonometric_identity_l2214_221409


namespace anna_discontinued_coaching_on_2nd_august_l2214_221447

theorem anna_discontinued_coaching_on_2nd_august
  (coaching_days : ℕ) (non_leap_year : ℕ) (first_day : ℕ) 
  (days_in_january : ℕ) (days_in_february : ℕ) (days_in_march : ℕ) 
  (days_in_april : ℕ) (days_in_may : ℕ) (days_in_june : ℕ) 
  (days_in_july : ℕ) (days_in_august : ℕ)
  (not_leap_year : non_leap_year = 365)
  (first_day_of_year : first_day = 1)
  (january_days : days_in_january = 31)
  (february_days : days_in_february = 28)
  (march_days : days_in_march = 31)
  (april_days : days_in_april = 30)
  (may_days : days_in_may = 31)
  (june_days : days_in_june = 30)
  (july_days : days_in_july = 31)
  (august_days : days_in_august = 31)
  (total_coaching_days : coaching_days = 245) :
  ∃ day, day = 2 ∧ month = "August" := 
sorry

end anna_discontinued_coaching_on_2nd_august_l2214_221447


namespace value_of_expr_l2214_221440

theorem value_of_expr (x : ℤ) (h : x = 3) : (2 * x + 6) ^ 2 = 144 := by
  sorry

end value_of_expr_l2214_221440


namespace same_function_l2214_221407

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (t : ℝ) : ℝ := (t^3 + t) / (t^2 + 1)

theorem same_function : ∀ x : ℝ, f x = g x :=
by sorry

end same_function_l2214_221407


namespace sum_exists_l2214_221472

theorem sum_exists 
  (n : ℕ) 
  (hn : n ≥ 5) 
  (k : ℕ) 
  (hk : k > (n + 1) / 2) 
  (a : ℕ → ℕ) 
  (ha1 : ∀ i, 1 ≤ a i) 
  (ha2 : ∀ i, a i < n) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j):
  ∃ i j l, i ≠ j ∧ a i + a j = a l := 
by 
  sorry

end sum_exists_l2214_221472


namespace alpha_plus_beta_l2214_221456

theorem alpha_plus_beta (α β : ℝ) (h : ∀ x, (x - α) / (x + β) = (x^2 - 116 * x + 2783) / (x^2 + 99 * x - 4080)) 
: α + β = 115 := 
sorry

end alpha_plus_beta_l2214_221456


namespace problem_1_problem_2_problem_3_l2214_221416

-- Definitions and conditions
def monomial_degree_condition (a : ℝ) : Prop := 2 + (1 + a) = 5

-- Proof goals
theorem problem_1 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = 9 := sorry
theorem problem_2 (a : ℝ) (h : monomial_degree_condition a) : (a + 1) * (a^2 - a + 1) = 9 := sorry
theorem problem_3 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = (a + 1) * (a^2 - a + 1) := sorry

end problem_1_problem_2_problem_3_l2214_221416


namespace angle_bisector_divides_longest_side_l2214_221448

theorem angle_bisector_divides_longest_side :
  ∀ (a b c : ℕ) (p q : ℕ), a = 12 → b = 15 → c = 18 →
  p + q = c → p * b = q * a → p = 8 ∧ q = 10 :=
by
  intros a b c p q ha hb hc hpq hprop
  rw [ha, hb, hc] at *
  sorry

end angle_bisector_divides_longest_side_l2214_221448


namespace disproves_proposition_b_l2214_221435

-- Definition and condition of complementary angles
def angles_complementary (angle1 angle2: ℝ) : Prop := angle1 + angle2 = 180

-- Proposition to disprove
def disprove (angle1 angle2: ℝ) : Prop := ¬ ((angle1 < 90 ∧ angle2 > 90 ∧ angle2 < 180) ∨ (angle2 < 90 ∧ angle1 > 90 ∧ angle1 < 180))

-- Definition of angles in sets
def set_a := (120, 60)
def set_b := (95.1, 84.9)
def set_c := (30, 60)
def set_d := (90, 90)

-- Statement to prove
theorem disproves_proposition_b : 
  (angles_complementary 95.1 84.9) ∧ (disprove 95.1 84.9) :=
by
  sorry

end disproves_proposition_b_l2214_221435


namespace value_of_expr_l2214_221454

theorem value_of_expr (a : Int) (h : a = -2) : a + 1 = -1 := by
  -- Placeholder for the proof, assuming it's correct
  sorry

end value_of_expr_l2214_221454


namespace problem1_problem2_problem3_l2214_221455

-- Definitions and conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Question 1: Prove the function is odd
theorem problem1 : ∀ x : ℝ, f (-x) = -f x := by
  sorry

-- Question 2: Prove the function is monotonically decreasing
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := by
  sorry

-- Question 3: Solve the inequality given f(2) = 1
theorem problem3 (h3 : f 2 = 1) : ∀ x : ℝ, f (-x^2) + 2*f x + 4 < 0 ↔ -2 < x ∧ x < 4 := by
  sorry

end problem1_problem2_problem3_l2214_221455


namespace find_x2_minus_x1_l2214_221441

theorem find_x2_minus_x1 (a x1 x2 d e : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x1 ≠ x2) (h_e : e = -d * x1)
  (h_y1 : ∀ x, y1 = a * (x - x1) * (x - x2)) (h_y2 : ∀ x, y2 = d * x + e)
  (h_intersect : ∀ x, y = a * (x - x1) * (x - x2) + (d * x + e)) 
  (h_single_point : ∀ x, y = a * (x - x1)^2) :
  x2 - x1 = d / a :=
sorry

end find_x2_minus_x1_l2214_221441


namespace rooms_in_house_l2214_221427

-- define the number of paintings
def total_paintings : ℕ := 32

-- define the number of paintings per room
def paintings_per_room : ℕ := 8

-- define the number of rooms
def number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) : ℕ := total_paintings / paintings_per_room

-- state the theorem
theorem rooms_in_house : number_of_rooms total_paintings paintings_per_room = 4 :=
by sorry

end rooms_in_house_l2214_221427


namespace min_value_quadratic_expr_l2214_221421

theorem min_value_quadratic_expr (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : x₁ ≠ x₂) 
  (h3 : x₁^2 - 4*a*x₁ + 3*a^2 < 0) 
  (h4 : x₂^2 - 4*a*x₂ + 3*a^2 < 0)
  (h5 : x₁ + x₂ = 4*a)
  (h6 : x₁ * x₂ = 3*a^2) : 
  x₁ + x₂ + a / (x₁ * x₂) = 4 * a + 1 / (3 * a) := 
sorry

end min_value_quadratic_expr_l2214_221421


namespace evaluate_expression_l2214_221488

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end evaluate_expression_l2214_221488


namespace count_three_digit_perfect_squares_divisible_by_4_l2214_221463

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l2214_221463


namespace students_attending_Harvard_l2214_221493

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end students_attending_Harvard_l2214_221493


namespace how_many_more_choc_chip_cookies_l2214_221479

-- Define the given conditions
def choc_chip_cookies_yesterday := 19
def raisin_cookies_this_morning := 231
def choc_chip_cookies_this_morning := 237

-- Define the total chocolate chip cookies
def total_choc_chip_cookies : ℕ := choc_chip_cookies_this_morning + choc_chip_cookies_yesterday

-- Define the proof statement
theorem how_many_more_choc_chip_cookies :
  total_choc_chip_cookies - raisin_cookies_this_morning = 25 :=
by
  -- Proof will go here
  sorry

end how_many_more_choc_chip_cookies_l2214_221479


namespace max_homework_ratio_l2214_221475

theorem max_homework_ratio 
  (H : ℕ) -- time spent on history tasks
  (biology_time : ℕ)
  (total_homework_time : ℕ)
  (geography_time : ℕ)
  (history_geography_relation : geography_time = 3 * H)
  (total_time_relation : total_homework_time = 180)
  (biology_time_known : biology_time = 20)
  (sum_time_relation : H + geography_time + biology_time = total_homework_time) :
  H / biology_time = 2 :=
by
  sorry

end max_homework_ratio_l2214_221475


namespace sum_of_squares_of_roots_eq_l2214_221457

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end sum_of_squares_of_roots_eq_l2214_221457


namespace surface_area_correct_l2214_221431

def radius_hemisphere : ℝ := 9
def height_cone : ℝ := 12
def radius_cone_base : ℝ := 9

noncomputable def total_surface_area : ℝ := 
  let base_area : ℝ := radius_hemisphere^2 * Real.pi
  let curved_area_hemisphere : ℝ := 2 * radius_hemisphere^2 * Real.pi
  let slant_height_cone : ℝ := Real.sqrt (radius_cone_base^2 + height_cone^2)
  let lateral_area_cone : ℝ := radius_cone_base * slant_height_cone * Real.pi
  base_area + curved_area_hemisphere + lateral_area_cone

theorem surface_area_correct : total_surface_area = 378 * Real.pi := by
  sorry

end surface_area_correct_l2214_221431


namespace no_solution_for_A_to_make_47A8_div_by_5_l2214_221491

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem no_solution_for_A_to_make_47A8_div_by_5 (A : ℕ) :
  ¬ (divisible_by_5 (47 * 1000 + A * 100 + 8)) :=
by
  sorry

end no_solution_for_A_to_make_47A8_div_by_5_l2214_221491


namespace find_digit_A_l2214_221445

open Nat

theorem find_digit_A :
  let n := 52
  let k := 13
  let number_of_hands := choose n k
  number_of_hands = 635013587600 → 0 = 0 := by
  suffices h: 635013587600 = 635013587600 by
    simp [h]
  sorry

end find_digit_A_l2214_221445


namespace work_done_by_gravity_l2214_221466

noncomputable def work_by_gravity (m g z_A z_B : ℝ) : ℝ :=
  m * g * (z_B - z_A)

theorem work_done_by_gravity (m g z_A z_B : ℝ) :
  work_by_gravity m g z_A z_B = m * g * (z_B - z_A) :=
by
  sorry

end work_done_by_gravity_l2214_221466


namespace find_divisor_l2214_221474

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161) 
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 16 :=
by
  sorry

end find_divisor_l2214_221474


namespace algebraic_expression_value_l2214_221422

theorem algebraic_expression_value (a b c : ℝ) (h : (∀ x : ℝ, (x - 1) * (x + 2) = a * x^2 + b * x + c)) :
  4 * a - 2 * b + c = 0 :=
sorry

end algebraic_expression_value_l2214_221422


namespace inverse_proportion_point_l2214_221415

theorem inverse_proportion_point (k : ℝ) (x1 y1 x2 y2 : ℝ)
  (h1 : y1 = k / x1) 
  (h2 : x1 = -2) 
  (h3 : y1 = 3)
  (h4 : x2 = 2) :
  y2 = -3 := 
by
  -- proof will be provided here
  sorry

end inverse_proportion_point_l2214_221415


namespace polynomial_pair_solution_l2214_221430

-- We define the problem in terms of polynomials over real numbers
open Polynomial

theorem polynomial_pair_solution (P Q : ℝ[X]) :
  (∀ x y : ℝ, P.eval (x + Q.eval y) = Q.eval (x + P.eval y)) →
  (P = Q ∨ (∃ a b : ℝ, P = X + C a ∧ Q = X + C b)) :=
by
  intro h
  sorry

end polynomial_pair_solution_l2214_221430


namespace factor_theorem_l2214_221410

theorem factor_theorem (h : ℤ) : (∀ m : ℤ, (m - 8) ∣ (m^2 - h * m - 24) ↔ h = 5) :=
  sorry

end factor_theorem_l2214_221410


namespace no_repair_needed_l2214_221482

def nominal_mass : ℝ := 370 -- Assign the nominal mass as determined in the problem solving.

def max_deviation (M : ℝ) : ℝ := 0.1 * M
def preserved_max_deviation : ℝ := 37
def unreadable_max_deviation : ℝ := 37

def within_max_deviation (dev : ℝ) := dev ≤ preserved_max_deviation

noncomputable def standard_deviation : ℝ := preserved_max_deviation

theorem no_repair_needed :
  ∀ (M : ℝ),
  max_deviation M = 0.1 * M →
  preserved_max_deviation ≤ max_deviation M →
  ∀ (dev : ℝ), within_max_deviation dev →
  standard_deviation ≤ preserved_max_deviation →
  preserved_max_deviation = 37 →
  "не требует" = "не требует" :=
by
  intros M h1 h2 h3 h4 h5
  sorry

end no_repair_needed_l2214_221482


namespace cats_added_l2214_221425

theorem cats_added (siamese_cats house_cats total_cats : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : total_cats = 28) : 
  total_cats - (siamese_cats + house_cats) = 10 := 
by 
  sorry

end cats_added_l2214_221425


namespace conference_duration_is_960_l2214_221400

-- The problem statement definition
def conference_sessions_duration_in_minutes (day1_hours : ℕ) (day1_minutes : ℕ) (day2_hours : ℕ) (day2_minutes : ℕ) : ℕ :=
  (day1_hours * 60 + day1_minutes) + (day2_hours * 60 + day2_minutes)

-- The theorem we want to prove given the above conditions
theorem conference_duration_is_960 :
  conference_sessions_duration_in_minutes 7 15 8 45 = 960 :=
by 
  -- The proof is omitted
  sorry

end conference_duration_is_960_l2214_221400


namespace max_area_triangle_l2214_221490

theorem max_area_triangle (a b c S : ℝ) (h₁ : S = a^2 - (b - c)^2) (h₂ : b + c = 8) :
  S ≤ 64 / 17 :=
sorry

end max_area_triangle_l2214_221490


namespace curve_transformation_l2214_221434

def matrix_transform (a : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (0 * x + 1 * y, a * x + 0 * y)

def curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 1

def transformed_curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + (y ^ 2) / 4 = 1

theorem curve_transformation (a : ℝ) 
  (h₁ : matrix_transform a 2 (-2) = (-2, 4))
  (h₂ : ∀ x y, curve_eq x y → transformed_curve_eq (matrix_transform a x y).fst (matrix_transform a x y).snd) :
  a = 2 ∧ ∀ x y, curve_eq x y → transformed_curve_eq (0 * x + 1 * y) (2 * x + 0 * y) :=
by
  sorry

end curve_transformation_l2214_221434


namespace squares_difference_l2214_221483

theorem squares_difference (x y z : ℤ) 
  (h1 : x + y = 10) 
  (h2 : x - y = 8) 
  (h3 : y + z = 15) : 
  x^2 - z^2 = -115 :=
by 
  sorry

end squares_difference_l2214_221483


namespace amount_left_after_pool_l2214_221449

def amount_left (total_earned : ℝ) (cost_per_person : ℝ) (num_people : ℕ) : ℝ :=
  total_earned - (cost_per_person * num_people)

theorem amount_left_after_pool :
  amount_left 30 2.5 10 = 5 :=
by
  sorry

end amount_left_after_pool_l2214_221449


namespace area_of_quadrilateral_ABCD_l2214_221428

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem area_of_quadrilateral_ABCD :
  let AB := 15 * sqrt 2
  let BE := 15 * sqrt 2
  let BC := 7.5 * sqrt 2
  let CE := 7.5 * sqrt 6
  let CD := 7.5 * sqrt 2
  let DE := 7.5 * sqrt 6
  (1/2 * AB * BE) + (1/2 * BC * CE) + (1/2 * CD * DE) = 225 + 112.5 * sqrt 12 :=
by
  sorry

end area_of_quadrilateral_ABCD_l2214_221428


namespace train_tunnel_length_l2214_221485

theorem train_tunnel_length 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (time_for_tail_to_exit : ℝ) 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 90) 
  (h_time_for_tail_to_exit : time_for_tail_to_exit = 2 / 60) :
  ∃ tunnel_length : ℝ, tunnel_length = 1 := 
by
  sorry

end train_tunnel_length_l2214_221485


namespace avg_age_team_proof_l2214_221414

-- Defining the known constants
def members : ℕ := 15
def avg_age_team : ℕ := 28
def captain_age : ℕ := avg_age_team + 4
def remaining_players : ℕ := members - 2
def avg_age_remaining : ℕ := avg_age_team - 2

-- Stating the problem to prove the average age remains 28
theorem avg_age_team_proof (W : ℕ) :
  28 = avg_age_team ∧
  members = 15 ∧
  captain_age = avg_age_team + 4 ∧
  remaining_players = members - 2 ∧
  avg_age_remaining = avg_age_team - 2 ∧
  28 * 15 = 26 * 13 + captain_age + W :=
sorry

end avg_age_team_proof_l2214_221414


namespace sin_x_cos_x_value_l2214_221405

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end sin_x_cos_x_value_l2214_221405


namespace overall_loss_amount_l2214_221495

theorem overall_loss_amount 
    (S : ℝ)
    (hS : S = 12499.99)
    (profit_percent : ℝ)
    (loss_percent : ℝ)
    (sold_at_profit : ℝ)
    (sold_at_loss : ℝ) 
    (condition1 : profit_percent = 0.2)
    (condition2 : loss_percent = -0.1)
    (condition3 : sold_at_profit = 0.2 * S * (1 + profit_percent))
    (condition4 : sold_at_loss = 0.8 * S * (1 + loss_percent))
    :
    S - (sold_at_profit + sold_at_loss) = 500 := 
by 
  sorry

end overall_loss_amount_l2214_221495


namespace basketball_classes_l2214_221443

theorem basketball_classes (x : ℕ) : (x * (x - 1)) / 2 = 10 :=
sorry

end basketball_classes_l2214_221443


namespace dog_revs_l2214_221494

theorem dog_revs (r₁ r₂ : ℝ) (n₁ : ℕ) (n₂ : ℕ) (h₁ : r₁ = 48) (h₂ : n₁ = 40) (h₃ : r₂ = 12) :
  n₂ = 160 := 
sorry

end dog_revs_l2214_221494


namespace question_1_question_2_l2214_221478

def f (x a : ℝ) := |x - a|

theorem question_1 :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

theorem question_2 (a : ℝ) (h : a = 2) :
  (∀ x, f x a + f (x + 5) a ≥ m) → m ≤ 5 :=
by
  sorry

end question_1_question_2_l2214_221478


namespace squared_distance_focus_product_tangents_l2214_221412

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end squared_distance_focus_product_tangents_l2214_221412


namespace mean_of_solutions_l2214_221497

theorem mean_of_solutions (x : ℝ) (h : x^3 + x^2 - 14 * x = 0) : 
  let a := (0 : ℝ)
  let b := (-1 + Real.sqrt 57) / 2
  let c := (-1 - Real.sqrt 57) / 2
  (a + b + c) / 3 = -2 / 3 :=
sorry

end mean_of_solutions_l2214_221497


namespace algebraic_expression_transformation_l2214_221437

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 + 4 * x + 3 = (x - 1)^2 + a * (x - 1) + b) → (a + b = 14) :=
by
  intros h
  sorry

end algebraic_expression_transformation_l2214_221437


namespace sequence_properties_l2214_221461

theorem sequence_properties :
  ∀ {a : ℕ → ℝ} {b : ℕ → ℝ},
  a 1 = 1 ∧ 
  (∀ n, b n > 4 / 3) ∧ 
  (∀ n, (∀ x, x^2 - b n * x + a n = 0 → (x = a (n + 1) ∨ x = 1 + a n))) →
  (a 2 = 1 / 2 ∧ ∃ n, b n > 4 / 3 ∧ n = 5) := by
  sorry

end sequence_properties_l2214_221461


namespace probability_event_proof_l2214_221411

noncomputable def probability_event_occur (deck_size : ℕ) (num_queens : ℕ) (num_jacks : ℕ) (num_reds : ℕ) : ℚ :=
  let prob_two_queens := (num_queens / deck_size) * ((num_queens - 1) / (deck_size - 1))
  let prob_at_least_one_jack := 
    (num_jacks / deck_size) * ((deck_size - num_jacks) / (deck_size - 1)) +
    ((deck_size - num_jacks) / deck_size) * (num_jacks / (deck_size - 1)) +
    (num_jacks / deck_size) * ((num_jacks - 1) / (deck_size - 1))
  let prob_both_red := (num_reds / deck_size) * ((num_reds - 1) / (deck_size - 1))
  prob_two_queens + prob_at_least_one_jack + prob_both_red

theorem probability_event_proof :
  probability_event_occur 52 4 4 26 = 89 / 221 :=
by
  sorry

end probability_event_proof_l2214_221411


namespace remainder_when_divided_by_9_l2214_221402

theorem remainder_when_divided_by_9 (z : ℤ) (k : ℤ) (h : z + 3 = 9 * k) :
  z % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l2214_221402


namespace max_value_of_expression_l2214_221438

open Real

theorem max_value_of_expression
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x^2 - 2 * x * y + 3 * y^2 = 10) 
  : x^2 + 2 * x * y + 3 * y^2 ≤ 10 * (45 + 42 * sqrt 3) := 
sorry

end max_value_of_expression_l2214_221438


namespace height_of_platform_l2214_221473

variables (l w h : ℕ)

theorem height_of_platform (hl1 : l + h - 2 * w = 36) (hl2 : w + h - l = 30) (hl3 : h = 2 * w) : h = 44 := 
sorry

end height_of_platform_l2214_221473


namespace expand_polynomial_l2214_221401

noncomputable def p (x : ℝ) : ℝ := 7 * x ^ 2 + 5
noncomputable def q (x : ℝ) : ℝ := 3 * x ^ 3 + 2 * x + 1

theorem expand_polynomial (x : ℝ) : 
  (p x) * (q x) = 21 * x ^ 5 + 29 * x ^ 3 + 7 * x ^ 2 + 10 * x + 5 := 
by sorry

end expand_polynomial_l2214_221401


namespace one_eighth_of_power_l2214_221484

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end one_eighth_of_power_l2214_221484


namespace yellow_ball_percentage_l2214_221439

theorem yellow_ball_percentage
  (yellow_balls : ℕ)
  (brown_balls : ℕ)
  (blue_balls : ℕ)
  (green_balls : ℕ)
  (total_balls : ℕ := yellow_balls + brown_balls + blue_balls + green_balls)
  (h_yellow : yellow_balls = 75)
  (h_brown : brown_balls = 120)
  (h_blue : blue_balls = 45)
  (h_green : green_balls = 60) :
  (yellow_balls * 100) / total_balls = 25 := 
by
  sorry

end yellow_ball_percentage_l2214_221439


namespace sqrt_5sq_4six_eq_320_l2214_221419

theorem sqrt_5sq_4six_eq_320 : Real.sqrt (5^2 * 4^6) = 320 :=
by sorry

end sqrt_5sq_4six_eq_320_l2214_221419


namespace range_of_a_l2214_221436

noncomputable def f (a x : ℝ) : ℝ := 
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  0 < a ∧ a ≤ 3/4 :=
by {
  sorry
}

end range_of_a_l2214_221436


namespace tigers_count_l2214_221442

theorem tigers_count (T C : ℝ) 
  (h1 : 12 + T + C = 39) 
  (h2 : C = 0.5 * (12 + T)) : 
  T = 14 := by
  sorry

end tigers_count_l2214_221442


namespace s_of_4_l2214_221465

noncomputable def t (x : ℚ) : ℚ := 5 * x - 14
noncomputable def s (y : ℚ) : ℚ := 
  let x := (y + 14) / 5
  x^2 + 5 * x - 4

theorem s_of_4 : s (4) = 674 / 25 := by
  sorry

end s_of_4_l2214_221465


namespace solve_poly_l2214_221404

open Real

-- Define the condition as a hypothesis
def prob_condition (x : ℝ) : Prop :=
  arctan (1 / x) + arctan (1 / (x^5)) = π / 6

-- Define the statement to be proven that x satisfies the polynomial equation
theorem solve_poly (x : ℝ) (h : prob_condition x) :
  x^6 - sqrt 3 * x^5 - sqrt 3 * x - 1 = 0 :=
sorry

end solve_poly_l2214_221404


namespace area_of_the_region_l2214_221429

noncomputable def region_area (C D : ℝ×ℝ) (rC rD : ℝ) (y : ℝ) : ℝ :=
  let rect_area := (D.1 - C.1) * y
  let sector_areaC := (1 / 2) * Real.pi * rC^2
  let sector_areaD := (1 / 2) * Real.pi * rD^2
  rect_area - (sector_areaC + sector_areaD)

theorem area_of_the_region :
  region_area (3, 5) (10, 5) 3 5 5 = 35 - 17 * Real.pi := by
  sorry

end area_of_the_region_l2214_221429


namespace central_angle_radian_measure_l2214_221471

-- Define the unit circle radius
def unit_circle_radius : ℝ := 1

-- Given an arc of length 1
def arc_length : ℝ := 1

-- Problem Statement: Prove that the radian measure of the central angle α is 1
theorem central_angle_radian_measure :
  ∀ (r : ℝ) (l : ℝ), r = unit_circle_radius → l = arc_length → |l / r| = 1 :=
by
  intros r l hr hl
  rw [hr, hl]
  sorry

end central_angle_radian_measure_l2214_221471


namespace students_like_apple_and_chocolate_not_blueberry_l2214_221444

variables (n A C B D : ℕ)

theorem students_like_apple_and_chocolate_not_blueberry
  (h1 : n = 50)
  (h2 : A = 25)
  (h3 : C = 20)
  (h4 : B = 5)
  (h5 : D = 15) :
  ∃ (x : ℕ), x = 10 ∧ x = n - D - (A + C - 2 * x) ∧ 0 ≤ 2 * x - A - C + B :=
sorry

end students_like_apple_and_chocolate_not_blueberry_l2214_221444


namespace length_of_uncovered_side_l2214_221413

-- Define the conditions of the problem
def area_condition (L W : ℝ) : Prop := L * W = 210
def fencing_condition (L W : ℝ) : Prop := L + 2 * W = 41

-- Define the proof statement
theorem length_of_uncovered_side (L W : ℝ) (h_area : area_condition L W) (h_fence : fencing_condition L W) : 
  L = 21 :=
  sorry

end length_of_uncovered_side_l2214_221413


namespace minimum_trucks_required_l2214_221459

-- Definitions for the problem
def total_weight_stones : ℝ := 10
def max_stone_weight : ℝ := 1
def truck_capacity : ℝ := 3

-- The theorem to prove
theorem minimum_trucks_required : ∃ (n : ℕ), n = 5 ∧ (n * truck_capacity) ≥ total_weight_stones := by
  sorry

end minimum_trucks_required_l2214_221459


namespace car_rental_cost_eq_800_l2214_221417

-- Define the number of people
def num_people : ℕ := 8

-- Define the cost of the Airbnb rental
def airbnb_cost : ℕ := 3200

-- Define each person's share
def share_per_person : ℕ := 500

-- Define the total contribution of all people
def total_contribution : ℕ := num_people * share_per_person

-- Define the car rental cost
def car_rental_cost : ℕ := total_contribution - airbnb_cost

-- State the theorem to be proved
theorem car_rental_cost_eq_800 : car_rental_cost = 800 :=
  by sorry

end car_rental_cost_eq_800_l2214_221417


namespace somu_present_age_l2214_221499

def Somu_Age_Problem (S F : ℕ) : Prop := 
  S = F / 3 ∧ S - 6 = (F - 6) / 5

theorem somu_present_age (S F : ℕ) 
  (h : Somu_Age_Problem S F) : S = 12 := 
by
  sorry

end somu_present_age_l2214_221499


namespace complete_square_l2214_221481

theorem complete_square (x : ℝ) : (x^2 + 4*x - 1 = 0) → ((x + 2)^2 = 5) :=
by
  intro h
  sorry

end complete_square_l2214_221481


namespace race_length_l2214_221467

theorem race_length (members : ℕ) (member_distance : ℕ) (ralph_multiplier : ℕ) 
    (h1 : members = 4) (h2 : member_distance = 3) (h3 : ralph_multiplier = 2) : 
    members * member_distance + ralph_multiplier * member_distance = 18 :=
by
  -- Start the proof with sorry to denote missing steps.
  sorry

end race_length_l2214_221467


namespace speed_in_still_water_l2214_221408

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) (h₁ : upstream_speed = 20) (h₂ : downstream_speed = 60) :
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end speed_in_still_water_l2214_221408


namespace trapezoid_leg_length_l2214_221477

theorem trapezoid_leg_length (S : ℝ) (h₁ : S > 0) : 
  ∃ x : ℝ, x = Real.sqrt (2 * S) ∧ x > 0 :=
by
  sorry

end trapezoid_leg_length_l2214_221477


namespace derivative_exp_l2214_221489

theorem derivative_exp (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x) : 
    ∀ x, deriv f x = Real.exp x :=
by 
  sorry

end derivative_exp_l2214_221489


namespace layla_earnings_l2214_221424

-- Define the hourly rates for each family
def rate_donaldson : ℕ := 15
def rate_merck : ℕ := 18
def rate_hille : ℕ := 20
def rate_johnson : ℕ := 22
def rate_ramos : ℕ := 25

-- Define the hours Layla worked for each family
def hours_donaldson : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def hours_johnson : ℕ := 4
def hours_ramos : ℕ := 2

-- Calculate the earnings for each family
def earnings_donaldson : ℕ := rate_donaldson * hours_donaldson
def earnings_merck : ℕ := rate_merck * hours_merck
def earnings_hille : ℕ := rate_hille * hours_hille
def earnings_johnson : ℕ := rate_johnson * hours_johnson
def earnings_ramos : ℕ := rate_ramos * hours_ramos

-- Calculate total earnings
def total_earnings : ℕ :=
  earnings_donaldson + earnings_merck + earnings_hille + earnings_johnson + earnings_ramos

-- The assertion that Layla's total earnings are $411
theorem layla_earnings : total_earnings = 411 := by
  sorry

end layla_earnings_l2214_221424


namespace smallest_multiplier_to_perfect_square_l2214_221498

-- Definitions for the conditions
def y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6

-- The theorem statement itself
theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, (∀ m : ℕ, (y * m = k) → (∃ n : ℕ, (k * y) = n^2)) :=
by
  let y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6
  let smallest_k := 70
  have h : y = 2^33 * 3^20 * 5^3 * 7^5 := by sorry
  use smallest_k
  intros m hm
  use (2^17 * 3^10 * 5 * 7)
  sorry

end smallest_multiplier_to_perfect_square_l2214_221498
