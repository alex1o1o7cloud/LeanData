import Mathlib

namespace minimum_y_squared_l181_181915

theorem minimum_y_squared :
  let consecutive_sum (x : ℤ) := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2
  ∃ y : ℤ, y^2 = 11 * (1^2 + 10) ∧ ∀ z : ℤ, z^2 = 11 * consecutive_sum z → y^2 ≤ z^2 := by
sorry

end minimum_y_squared_l181_181915


namespace math_problem_l181_181868

def is_polynomial (expr : String) : Prop := sorry
def is_monomial (expr : String) : Prop := sorry
def is_cubic (expr : String) : Prop := sorry
def is_quintic (expr : String) : Prop := sorry
def correct_option_C : String := "C"

theorem math_problem :
  ¬ is_polynomial "8 - 2 / z" ∧
  ¬ (is_monomial "-x^2yz" ∧ is_cubic "-x^2yz") ∧
  is_polynomial "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  is_quintic "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  ¬ is_monomial "5b / x" →
  correct_option_C = "C" := sorry

end math_problem_l181_181868


namespace find_f_g_3_l181_181383

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem find_f_g_3 :
  f (g 3) = -2 := by
  sorry

end find_f_g_3_l181_181383


namespace perfect_rectangle_squares_l181_181601

theorem perfect_rectangle_squares (squares : Finset ℕ) 
  (h₁ : 9 ∈ squares) 
  (h₂ : 2 ∈ squares) 
  (h₃ : squares.card = 9) 
  (h₄ : ∀ x ∈ squares, ∃ y ∈ squares, x ≠ y ∧ (gcd x y = 1)) :
  squares = {2, 5, 7, 9, 16, 25, 28, 33, 36} := 
sorry

end perfect_rectangle_squares_l181_181601


namespace quadratic_has_real_roots_range_l181_181635

noncomputable def has_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := 2
  let c := -1
  b^2 - 4 * a * c ≥ 0

theorem quadratic_has_real_roots_range (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 ∧ k ≠ 0 := by
sorry

end quadratic_has_real_roots_range_l181_181635


namespace surface_area_correct_l181_181494

def w := 3 -- width in cm
def l := 4 -- length in cm
def h := 5 -- height in cm

def surface_area : Nat := 
  2 * (h * w) + 2 * (l * w) + 2 * (l * h)

theorem surface_area_correct : surface_area = 94 := 
  by
    sorry

end surface_area_correct_l181_181494


namespace part1_part2_part3_l181_181393

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem part1 :
  ∀ x : ℝ, |f x| = |x - 1| → x = -2 ∨ x = 0 ∨ x = 1 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |f x1| = g a x1 ∧ |f x2| = g a x2) ↔ (a = 0 ∨ a = 2) :=
sorry

theorem part3 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ (a ≤ -2) :=
sorry

end part1_part2_part3_l181_181393


namespace gcd_of_2535_5929_11629_l181_181700

theorem gcd_of_2535_5929_11629 : Nat.gcd (Nat.gcd 2535 5929) 11629 = 1 := by
  sorry

end gcd_of_2535_5929_11629_l181_181700


namespace positive_integers_not_in_E_are_perfect_squares_l181_181143

open Set

def E : Set ℕ := {m | ∃ n : ℕ, m = Int.floor (n + Real.sqrt n + 0.5)}

theorem positive_integers_not_in_E_are_perfect_squares (m : ℕ) (h_pos : 0 < m) :
  m ∉ E ↔ ∃ t : ℕ, m = t^2 := 
by
    sorry

end positive_integers_not_in_E_are_perfect_squares_l181_181143


namespace a_range_l181_181779

theorem a_range (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1)
  (h_eq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
sorry

end a_range_l181_181779


namespace perpendicular_condition_sufficient_but_not_necessary_l181_181627

theorem perpendicular_condition_sufficient_but_not_necessary (m : ℝ) (h : m = -1) :
  (∀ x y : ℝ, mx + (2 * m - 1) * y + 1 = 0 ∧ 3 * x + m * y + 2 = 0) → (m = 0 ∨ m = -1) → (m = 0 ∨ m = -1) :=
by
  intro h1 h2
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l181_181627


namespace alcohol_percentage_new_mixture_l181_181057

theorem alcohol_percentage_new_mixture (initial_volume new_volume alcohol_initial : ℝ)
  (h1 : initial_volume = 15)
  (h2 : alcohol_initial = 0.20 * initial_volume)
  (h3 : new_volume = initial_volume + 5) :
  (alcohol_initial / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_new_mixture_l181_181057


namespace expected_winnings_is_minus_half_l181_181659

-- Define the given condition in Lean
noncomputable def prob_win_side_1 : ℚ := 1 / 4
noncomputable def prob_win_side_2 : ℚ := 1 / 4
noncomputable def prob_lose_side_3 : ℚ := 1 / 3
noncomputable def prob_no_change_side_4 : ℚ := 1 / 6

noncomputable def win_amount_side_1 : ℚ := 2
noncomputable def win_amount_side_2 : ℚ := 4
noncomputable def lose_amount_side_3 : ℚ := -6
noncomputable def no_change_amount_side_4 : ℚ := 0

-- Define the expected value function
noncomputable def expected_winnings : ℚ :=
  (prob_win_side_1 * win_amount_side_1) +
  (prob_win_side_2 * win_amount_side_2) +
  (prob_lose_side_3 * lose_amount_side_3) +
  (prob_no_change_side_4 * no_change_amount_side_4)

-- Statement to prove
theorem expected_winnings_is_minus_half : expected_winnings = -1 / 2 := 
by
  sorry

end expected_winnings_is_minus_half_l181_181659


namespace limit_sum_perimeters_l181_181247

theorem limit_sum_perimeters (a : ℝ) : ∑' n : ℕ, (4 * a) * (1 / 2) ^ n = 8 * a :=
by sorry

end limit_sum_perimeters_l181_181247


namespace problem_l181_181425

theorem problem (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z)
  (coprime : Nat.gcd X (Nat.gcd Y Z) = 1)
  (h : X * Real.log 3 / Real.log 100 + Y * Real.log 4 / Real.log 100 = Z):
  X + Y + Z = 4 :=
sorry

end problem_l181_181425


namespace num_people_watched_last_week_l181_181583

variable (s f t : ℕ)
variable (h1 : s = 80)
variable (h2 : f = s - 20)
variable (h3 : t = s + 15)
variable (total_last_week total_this_week : ℕ)
variable (h4 : total_this_week = f + s + t)
variable (h5 : total_this_week = total_last_week + 35)

theorem num_people_watched_last_week :
  total_last_week = 200 := sorry

end num_people_watched_last_week_l181_181583


namespace angle_B_of_right_triangle_l181_181695

theorem angle_B_of_right_triangle (B C : ℝ) (hA : A = 90) (hC : C = 3 * B) (h_sum : A + B + C = 180) : B = 22.5 :=
sorry

end angle_B_of_right_triangle_l181_181695


namespace LittleRed_system_of_eqns_l181_181427

theorem LittleRed_system_of_eqns :
  ∃ (x y : ℝ), (2/60) * x + (3/60) * y = 1.5 ∧ x + y = 18 :=
sorry

end LittleRed_system_of_eqns_l181_181427


namespace original_number_exists_l181_181891

theorem original_number_exists 
  (N: ℤ)
  (h1: ∃ (k: ℤ), N - 6 = 16 * k)
  (h2: ∀ (m: ℤ), (N - m) % 16 = 0 → m ≥ 6) : 
  N = 22 :=
sorry

end original_number_exists_l181_181891


namespace cheenu_time_difference_l181_181226

theorem cheenu_time_difference :
  let boy_distance : ℝ := 18
  let boy_time_hours : ℝ := 4
  let old_man_distance : ℝ := 12
  let old_man_time_hours : ℝ := 5
  let hour_to_minute : ℝ := 60
  
  let boy_time_minutes := boy_time_hours * hour_to_minute
  let old_man_time_minutes := old_man_time_hours * hour_to_minute

  let boy_time_per_mile := boy_time_minutes / boy_distance
  let old_man_time_per_mile := old_man_time_minutes / old_man_distance
  
  old_man_time_per_mile - boy_time_per_mile = 12 :=
by sorry

end cheenu_time_difference_l181_181226


namespace number_of_apples_l181_181686

theorem number_of_apples (C : ℝ) (A : ℕ) (total_cost : ℝ) (price_diff : ℝ) (num_oranges : ℕ)
  (h_price : C = 0.26)
  (h_price_diff : price_diff = 0.28)
  (h_num_oranges : num_oranges = 7)
  (h_total_cost : total_cost = 4.56) :
  A * C + num_oranges * (C + price_diff) = total_cost → A = 3 := 
by
  sorry

end number_of_apples_l181_181686


namespace forty_percent_of_number_l181_181453

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : (40/100) * N = 192 :=
by
  sorry

end forty_percent_of_number_l181_181453


namespace simplify_expression_l181_181366

theorem simplify_expression :
  (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 :=
by
  sorry

end simplify_expression_l181_181366


namespace student_B_speed_l181_181334

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l181_181334


namespace total_accepted_cartons_l181_181559

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end total_accepted_cartons_l181_181559


namespace division_of_neg6_by_3_l181_181332

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end division_of_neg6_by_3_l181_181332


namespace tricycles_count_l181_181570

theorem tricycles_count {s t : Nat} (h1 : s + t = 10) (h2 : 2 * s + 3 * t = 26) : t = 6 :=
sorry

end tricycles_count_l181_181570


namespace triangle_is_isosceles_l181_181530

open Real

variables (α β γ : ℝ) (a b : ℝ)

theorem triangle_is_isosceles
(h1 : a + b = tan (γ / 2) * (a * tan α + b * tan β)) :
α = β :=
by
  sorry

end triangle_is_isosceles_l181_181530


namespace exists_smallest_n_l181_181833

theorem exists_smallest_n :
  ∃ n : ℕ, (n^2 + 20 * n + 19) % 2019 = 0 ∧ n = 2000 :=
sorry

end exists_smallest_n_l181_181833


namespace max_comic_books_l181_181454

namespace JasmineComicBooks

-- Conditions
def total_money : ℝ := 12.50
def comic_book_cost : ℝ := 1.15

-- Statement of the theorem
theorem max_comic_books (n : ℕ) (h : n * comic_book_cost ≤ total_money) : n ≤ 10 := by
  sorry

end JasmineComicBooks

end max_comic_books_l181_181454


namespace total_cost_shoes_and_jerseys_l181_181469

theorem total_cost_shoes_and_jerseys 
  (shoes : ℕ) (jerseys : ℕ) (cost_shoes : ℕ) (cost_jersey : ℕ) 
  (cost_total_shoes : ℕ) (cost_per_shoe : ℕ) (cost_per_jersey : ℕ) 
  (h1 : shoes = 6)
  (h2 : jerseys = 4) 
  (h3 : cost_per_jersey = cost_per_shoe / 4)
  (h4 : cost_total_shoes = 480)
  (h5 : cost_per_shoe = cost_total_shoes / shoes)
  (h6 : cost_per_jersey = cost_per_shoe / 4)
  (total_cost : ℕ) 
  (h7 : total_cost = cost_total_shoes + cost_per_jersey * jerseys) :
  total_cost = 560 :=
sorry

end total_cost_shoes_and_jerseys_l181_181469


namespace vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l181_181752

open Real

-- Problem 1: Prove the vertex of the parabola is at (1, -a)
theorem vertex_of_parabola (a : ℝ) (h : a ≠ 0) : 
  ∀ x : ℝ, y = a * x^2 - 2 * a * x → (1, -a) = ((1 : ℝ), - a) := 
sorry

-- Problem 2: Prove x_0 = 3 if m = n for given points on the parabola
theorem point_symmetry_on_parabola (a : ℝ) (h : a ≠ 0) (m n : ℝ) :
  m = n → ∀ (x0 : ℝ), y = a * x0 ^ 2 - 2 * a * x0 → x0 = 3 :=
sorry

-- Problem 3: Prove the conditions for y1 < y2 ≤ -a and the range of m
theorem range_of_m (a : ℝ) (h : a < 0) : 
  ∀ (m y1 y2 : ℝ), (y1 < y2) ∧ (y2 ≤ -a) → m < (1 / 2) := 
sorry

end vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l181_181752


namespace sin_C_of_right_triangle_l181_181637

theorem sin_C_of_right_triangle (A B C: ℝ) (sinA: ℝ) (sinB: ℝ) (sinC: ℝ) :
  (sinA = 8/17) →
  (sinB = 1) →
  (A + B + C = π) →
  (B = π / 2) →
  (sinC = 15/17) :=
  
by
  intro h_sinA h_sinB h_triangle h_B
  sorry -- Proof is not required

end sin_C_of_right_triangle_l181_181637


namespace typing_time_together_l181_181910

theorem typing_time_together 
  (jonathan_time : ℝ)
  (susan_time : ℝ)
  (jack_time : ℝ)
  (document_pages : ℝ)
  (combined_time : ℝ) :
  jonathan_time = 40 →
  susan_time = 30 →
  jack_time = 24 →
  document_pages = 10 →
  combined_time = document_pages / ((document_pages / jonathan_time) + (document_pages / susan_time) + (document_pages / jack_time)) →
  combined_time = 10 :=
by sorry

end typing_time_together_l181_181910


namespace pawns_left_l181_181577

-- Definitions of the initial conditions
def initial_pawns : ℕ := 8
def kennedy_lost_pawns : ℕ := 4
def riley_lost_pawns : ℕ := 1

-- Definition of the total pawns left function
def total_pawns_left (initial_pawns kennedy_lost_pawns riley_lost_pawns : ℕ) : ℕ :=
  (initial_pawns - kennedy_lost_pawns) + (initial_pawns - riley_lost_pawns)

-- The statement to prove
theorem pawns_left : total_pawns_left initial_pawns kennedy_lost_pawns riley_lost_pawns = 11 := by
  -- Proof omitted
  sorry

end pawns_left_l181_181577


namespace pat_more_hours_than_jane_l181_181151

theorem pat_more_hours_than_jane (H P K M J : ℝ) 
  (h_total : H = P + K + M + J)
  (h_pat : P = 2 * K)
  (h_mark : M = (1/3) * P)
  (h_jane : J = (1/2) * M)
  (H290 : H = 290) :
  P - J = 120.83 := 
by
  sorry

end pat_more_hours_than_jane_l181_181151


namespace rose_age_l181_181951

variable {R M : ℝ}

theorem rose_age (h1 : R = (1/3) * M) (h2 : R + M = 100) : R = 25 :=
sorry

end rose_age_l181_181951


namespace range_of_m_l181_181068

-- Definitions and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_eccentricity (e a b : ℝ) : Prop :=
  e = Real.sqrt (1 - (b^2 / a^2))

def is_semi_latus_rectum (d a b : ℝ) : Prop :=
  d = 2 * b^2 / a

-- Main theorem statement
theorem range_of_m (a b m : ℝ) (x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : is_eccentricity (Real.sqrt (3) / 2) a b)
  (h4 : is_semi_latus_rectum 1 a b)
  (h_ellipse : ellipse a b x y) : 
  m ∈ Set.Ioo (-3 / 2 : ℝ) (3 / 2 : ℝ) := 
sorry

end range_of_m_l181_181068


namespace cost_price_of_article_l181_181205

-- Define the conditions and goal as a Lean 4 statement
theorem cost_price_of_article (M C : ℝ) (h1 : 0.95 * M = 75) (h2 : 1.25 * C = 75) : 
  C = 60 := 
by 
  sorry

end cost_price_of_article_l181_181205


namespace harry_change_l181_181602

theorem harry_change (a : ℕ) :
  (∃ k : ℕ, a = 50 * k + 2 ∧ a < 100) ∧ (∃ m : ℕ, a = 5 * m + 4 ∧ a < 100) →
  a = 52 :=
by sorry

end harry_change_l181_181602


namespace fewerCansCollected_l181_181059

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l181_181059


namespace fraction_ratio_l181_181825

theorem fraction_ratio
  (m n p q r : ℚ)
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 :=
by
  sorry

end fraction_ratio_l181_181825


namespace find_x_l181_181421

theorem find_x (x : ℚ) (h : (35 / 100) * x = (40 / 100) * 50) : 
  x = 400 / 7 :=
sorry

end find_x_l181_181421


namespace largest_among_abc_l181_181958

theorem largest_among_abc
  (x : ℝ) 
  (hx : 0 < x) 
  (hx1 : x < 1)
  (a : ℝ)
  (ha : a = 2 * Real.sqrt x )
  (b : ℝ)
  (hb : b = 1 + x)
  (c : ℝ)
  (hc : c = 1 / (1 - x)) 
  : a < b ∧ b < c :=
by
  sorry

end largest_among_abc_l181_181958


namespace pat_earns_per_photo_l181_181533

-- Defining conditions
def minutes_per_shark := 10
def fuel_cost_per_hour := 50
def hunting_hours := 5
def expected_profit := 200

-- Defining intermediate calculations based on the conditions
def sharks_per_hour := 60 / minutes_per_shark
def total_sharks := sharks_per_hour * hunting_hours
def total_fuel_cost := fuel_cost_per_hour * hunting_hours
def total_earnings := expected_profit + total_fuel_cost
def earnings_per_photo := total_earnings / total_sharks

-- Main theorem: Prove that Pat earns $15 for each photo
theorem pat_earns_per_photo : earnings_per_photo = 15 := by
  -- The proof would be here
  sorry

end pat_earns_per_photo_l181_181533


namespace purely_imaginary_complex_number_l181_181652

theorem purely_imaginary_complex_number (a : ℝ) :
  (∃ b : ℝ, (a^2 - 3 * a + 2) = 0 ∧ a ≠ 1) → a = 2 :=
by
  sorry

end purely_imaginary_complex_number_l181_181652


namespace john_needs_to_add_empty_cans_l181_181578

theorem john_needs_to_add_empty_cans :
  ∀ (num_full_cans : ℕ) (weight_per_full_can total_weight weight_per_empty_can required_weight : ℕ),
  num_full_cans = 6 →
  weight_per_full_can = 14 →
  total_weight = 88 →
  weight_per_empty_can = 2 →
  required_weight = total_weight - (num_full_cans * weight_per_full_can) →
  required_weight / weight_per_empty_can = 2 :=
by
  intros
  sorry

end john_needs_to_add_empty_cans_l181_181578


namespace parabola_intersects_y_axis_l181_181131

theorem parabola_intersects_y_axis (m n : ℝ) :
  (∃ (x y : ℝ), y = x^2 + m * x + n ∧ 
  ((x = -1 ∧ y = -6) ∨ (x = 1 ∧ y = 0))) →
  (0, (-4)) = (0, n) :=
by
  sorry

end parabola_intersects_y_axis_l181_181131


namespace photos_in_gallery_l181_181200

theorem photos_in_gallery (P : ℕ) 
  (h1 : P / 2 + (P / 2 + 120) + P = 920) : P = 400 :=
by
  sorry

end photos_in_gallery_l181_181200


namespace max_partitioned_test_plots_is_78_l181_181811

def field_length : ℕ := 52
def field_width : ℕ := 24
def total_fence : ℕ := 1994
def gcd_field_dimensions : ℕ := Nat.gcd field_length field_width

-- Since gcd_field_dimensions divides both 52 and 24 and gcd_field_dimensions = 4
def possible_side_lengths : List ℕ := [1, 2, 4]

noncomputable def max_square_plots : ℕ :=
  let max_plots (a : ℕ) : ℕ := (field_length / a) * (field_width / a)
  let valid_fence (a : ℕ) : Bool :=
    let vertical_fence := (field_length / a - 1) * field_width
    let horizontal_fence := (field_width / a - 1) * field_length
    vertical_fence + horizontal_fence ≤ total_fence
  let valid_lengths := possible_side_lengths.filter valid_fence
  valid_lengths.map max_plots |>.maximum? |>.getD 0

theorem max_partitioned_test_plots_is_78 : max_square_plots = 78 := by
  sorry

end max_partitioned_test_plots_is_78_l181_181811


namespace no_positive_integer_solution_l181_181286

/-- Let \( p \) be a prime greater than 3 and \( x \) be an integer such that \( p \) divides \( x \).
    Then the equation \( x^2 - 1 = y^p \) has no positive integer solutions for \( y \). -/
theorem no_positive_integer_solution {p x y : ℕ} (hp : Nat.Prime p) (hgt : 3 < p) (hdiv : p ∣ x) :
  ¬∃ y : ℕ, (x^2 - 1 = y^p) ∧ (0 < y) :=
by
  sorry

end no_positive_integer_solution_l181_181286


namespace range_x_sub_cos_y_l181_181075

theorem range_x_sub_cos_y (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) : 
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 :=
sorry

end range_x_sub_cos_y_l181_181075


namespace find_initial_divisor_l181_181890

theorem find_initial_divisor (N D : ℤ) (h1 : N = 2 * D) (h2 : N % 4 = 2) : D = 3 :=
by
  sorry

end find_initial_divisor_l181_181890


namespace prove_a₈_l181_181447

noncomputable def first_term (a : ℕ → ℝ) : Prop := a 1 = 3
noncomputable def arithmetic_b (a b : ℕ → ℝ) : Prop := ∀ n, b n = a (n + 1) - a n
noncomputable def b_conditions (b : ℕ → ℝ) : Prop := b 3 = -2 ∧ b 10 = 12

theorem prove_a₈ (a b : ℕ → ℝ) (h1 : first_term a) (h2 : arithmetic_b a b) (h3 : b_conditions b) :
  a 8 = 3 :=
sorry

end prove_a₈_l181_181447


namespace dhoni_spent_300_dollars_l181_181288

theorem dhoni_spent_300_dollars :
  ∀ (L S X : ℝ),
  L = 6 →
  S = L - 2 →
  (X / S) - (X / L) = 25 →
  X = 300 :=
by
intros L S X hL hS hEquation
sorry

end dhoni_spent_300_dollars_l181_181288


namespace common_roots_cubic_polynomials_l181_181574

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ (r^3 + a * r^2 + 17 * r + 10 = 0) ∧ (s^3 + a * s^2 + 17 * s + 10 = 0) ∧ 
               (r^3 + b * r^2 + 20 * r + 12 = 0) ∧ (s^3 + b * s^2 + 20 * s + 12 = 0)) →
  (a, b) = (-6, -7) :=
by sorry

end common_roots_cubic_polynomials_l181_181574


namespace fraction_increases_by_3_l181_181185

-- Define initial fraction
def initial_fraction (x y : ℕ) : ℕ :=
  2 * x * y / (3 * x - y)

-- Define modified fraction
def modified_fraction (x y : ℕ) (m : ℕ) : ℕ :=
  2 * (m * x) * (m * y) / (m * (3 * x) - (m * y))

-- State the theorem to prove the value of modified fraction is 3 times the initial fraction
theorem fraction_increases_by_3 (x y : ℕ) : modified_fraction x y 3 = 3 * initial_fraction x y :=
by sorry

end fraction_increases_by_3_l181_181185


namespace chocolates_initial_count_l181_181552

theorem chocolates_initial_count (remaining_chocolates: ℕ) 
    (daily_percentage: ℝ) (days: ℕ) 
    (final_chocolates: ℝ) 
    (remaining_fraction_proof: remaining_fraction = 0.7) 
    (days_proof: days = 3) 
    (final_chocolates_proof: final_chocolates = 28): 
    (remaining_fraction^days * (initial_chocolates:ℝ) = final_chocolates) → 
    (initial_chocolates = 82) := 
by 
  sorry

end chocolates_initial_count_l181_181552


namespace Robie_gave_away_boxes_l181_181108

theorem Robie_gave_away_boxes :
  ∀ (total_cards cards_per_box boxes_with_him remaining_cards : ℕ)
  (h_total_cards : total_cards = 75)
  (h_cards_per_box : cards_per_box = 10)
  (h_boxes_with_him : boxes_with_him = 5)
  (h_remaining_cards : remaining_cards = 5),
  (total_cards / cards_per_box) - boxes_with_him = 2 :=
by
  intros total_cards cards_per_box boxes_with_him remaining_cards
  intros h_total_cards h_cards_per_box h_boxes_with_him h_remaining_cards
  sorry

end Robie_gave_away_boxes_l181_181108


namespace solve_system_of_equations_l181_181295

theorem solve_system_of_equations (x y : ℚ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : x + 3 * y = 9) : 
  x = 42 / 11 ∧ y = 19 / 11 :=
by {
  sorry
}

end solve_system_of_equations_l181_181295


namespace train_passing_platform_time_l181_181031

-- Conditions
variable (l t : ℝ) -- Length of the train and time to pass the pole
variable (v : ℝ) -- Velocity of the train
variable (n : ℝ) -- Multiple of t seconds to pass the platform
variable (d_platform : ℝ) -- Length of the platform

-- Theorem statement
theorem train_passing_platform_time (h1 : d_platform = 3 * l) (h2 : v = l / t) (h3 : n = (l + d_platform) / l) :
  n = 4 := by
  sorry

end train_passing_platform_time_l181_181031


namespace find_b_l181_181990

theorem find_b (b : ℝ) (h1 : 0 < b) (h2 : b < 6)
  (h_ratio : ∃ (QRS QOP : ℝ), QRS / QOP = 4 / 25) : b = 6 :=
sorry

end find_b_l181_181990


namespace parabola_passes_through_fixed_point_l181_181967

theorem parabola_passes_through_fixed_point:
  ∀ t : ℝ, ∃ x y : ℝ, (y = 4 * x^2 + 2 * t * x - 3 * t ∧ (x = 3 ∧ y = 36)) :=
by
  intro t
  use 3
  use 36
  sorry

end parabola_passes_through_fixed_point_l181_181967


namespace sum_of_squares_not_square_l181_181649

theorem sum_of_squares_not_square (a : ℕ) : 
  ¬ ∃ b : ℕ, (a - 1)^2 + a^2 + (a + 1)^2 = b^2 := 
by {
  sorry
}

end sum_of_squares_not_square_l181_181649


namespace find_a_l181_181873

theorem find_a (a : ℝ) (h₁ : a > 1) (h₂ : (∀ x : ℝ, a^3 = 8)) : a = 2 :=
by
  sorry

end find_a_l181_181873


namespace greatest_product_sum_300_l181_181591

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l181_181591


namespace hyperbola_asymptote_y_eq_1_has_m_neg_3_l181_181273

theorem hyperbola_asymptote_y_eq_1_has_m_neg_3
    (m : ℝ)
    (h1 : ∀ x y, (x^2 / (2 * m)) - (y^2 / m) = 1)
    (h2 : ∀ x, 1 = (x^2 / (2 * m))): m = -3 :=
by
  sorry

end hyperbola_asymptote_y_eq_1_has_m_neg_3_l181_181273


namespace concentric_circles_ratio_l181_181138

theorem concentric_circles_ratio
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : π * b^2 - π * a^2 = 4 * (π * a^2)) :
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end concentric_circles_ratio_l181_181138


namespace percentage_of_filled_seats_l181_181871

theorem percentage_of_filled_seats (total_seats vacant_seats : ℕ) (h_total : total_seats = 600) (h_vacant : vacant_seats = 240) :
  (total_seats - vacant_seats) * 100 / total_seats = 60 :=
by
  sorry

end percentage_of_filled_seats_l181_181871


namespace calculate_ab_plus_cd_l181_181787

theorem calculate_ab_plus_cd (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 12) :
  a * b + c * d = 27 :=
by
  sorry -- Proof to be filled in.

end calculate_ab_plus_cd_l181_181787


namespace determine_functions_l181_181245

noncomputable def functional_eq_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x) ^ 2

theorem determine_functions (f : ℝ → ℝ) (h : functional_eq_condition f) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
sorry

end determine_functions_l181_181245


namespace solve_first_train_length_l181_181426

noncomputable def first_train_length (time: ℝ) (speed1_kmh: ℝ) (speed2_kmh: ℝ) (length2: ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * 1000 / 3600
  let speed2_ms := speed2_kmh * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem solve_first_train_length :
  first_train_length 7.0752960452818945 80 65 165 = 120.28 :=
by
  simp [first_train_length]
  norm_num
  sorry

end solve_first_train_length_l181_181426


namespace solve_for_y_l181_181778

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 5 * (y / y ^ (3 / 4)) = 2 + y ^ (1 / 4)) : y = 16 / 81 :=
by
  sorry

end solve_for_y_l181_181778


namespace percentage_of_180_equation_l181_181701

theorem percentage_of_180_equation (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * ((P / 100) * 180) = 36) : P = 30 :=
sorry

end percentage_of_180_equation_l181_181701


namespace amount_saved_l181_181056

theorem amount_saved (list_price : ℝ) (tech_deals_discount : ℝ) (electro_bargains_discount : ℝ)
    (tech_deals_price : ℝ) (electro_bargains_price : ℝ) (amount_saved : ℝ) :
  tech_deals_discount = 0.15 →
  list_price = 120 →
  tech_deals_price = list_price * (1 - tech_deals_discount) →
  electro_bargains_discount = 20 →
  electro_bargains_price = list_price - electro_bargains_discount →
  amount_saved = tech_deals_price - electro_bargains_price →
  amount_saved = 2 :=
by
  -- proof steps would go here
  sorry

end amount_saved_l181_181056


namespace find_intersection_l181_181823

noncomputable def setM : Set ℝ := {x : ℝ | x^2 ≤ 9}
noncomputable def setN : Set ℝ := {x : ℝ | x ≤ 1}
noncomputable def intersection : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

theorem find_intersection (x : ℝ) : (x ∈ setM ∧ x ∈ setN) ↔ (x ∈ intersection) := 
by sorry

end find_intersection_l181_181823


namespace fraction_value_sin_cos_value_l181_181357

open Real

-- Let alpha be an angle in radians satisfying the given condition
variable (α : ℝ)

-- Given condition
def condition  : Prop := sin α = 2 * cos α

-- First question
theorem fraction_value (h : condition α) : 
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1 / 6 :=
sorry

-- Second question
theorem sin_cos_value (h : condition α) : 
  sin α ^ 2 + 2 * sin α * cos α = 8 / 5 :=
sorry

end fraction_value_sin_cos_value_l181_181357


namespace mandy_yoga_time_l181_181560

theorem mandy_yoga_time (G B Y : ℕ) (h1 : 2 * B = 3 * G) (h2 : 3 * Y = 2 * (G + B)) (h3 : Y = 30) : Y = 30 := by
  sorry

end mandy_yoga_time_l181_181560


namespace g_g_g_3_eq_71_l181_181290

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1 else 2 * n + 5

theorem g_g_g_3_eq_71 : g (g (g 3)) = 71 := 
by
  sorry

end g_g_g_3_eq_71_l181_181290


namespace car_fuel_efficiency_in_city_l181_181924

theorem car_fuel_efficiency_in_city 
    (H C T : ℝ) 
    (h1 : H * T = 462) 
    (h2 : (H - 15) * T = 336) : 
    C = 40 :=
by 
    sorry

end car_fuel_efficiency_in_city_l181_181924


namespace trains_meet_at_10_am_l181_181984

def distance (speed time : ℝ) : ℝ := speed * time

theorem trains_meet_at_10_am
  (distance_pq : ℝ)
  (speed_train_from_p : ℝ)
  (start_time_from_p : ℝ)
  (speed_train_from_q : ℝ)
  (start_time_from_q : ℝ)
  (meeting_time : ℝ) :
  distance_pq = 110 → 
  speed_train_from_p = 20 → 
  start_time_from_p = 7 → 
  speed_train_from_q = 25 → 
  start_time_from_q = 8 → 
  meeting_time = 10 :=
by
  sorry

end trains_meet_at_10_am_l181_181984


namespace eval_expression_at_minus_3_l181_181798

theorem eval_expression_at_minus_3 :
  (5 + 2 * x * (x + 2) - 4^2) / (x - 4 + x^2) = -5 / 2 :=
by
  let x := -3
  sorry

end eval_expression_at_minus_3_l181_181798


namespace hide_and_seek_l181_181608

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l181_181608


namespace retail_profit_percent_l181_181112

variable (CP : ℝ) (MP : ℝ) (SP : ℝ)
variable (h_marked : MP = CP + 0.60 * CP)
variable (h_discount : SP = MP - 0.25 * MP)

theorem retail_profit_percent : CP = 100 → MP = CP + 0.60 * CP → SP = MP - 0.25 * MP → 
       (SP - CP) / CP * 100 = 20 := 
by
  intros h1 h2 h3
  sorry

end retail_profit_percent_l181_181112


namespace distance_p_ran_l181_181985

variable (d t v : ℝ)
-- d: head start distance in meters
-- t: time in minutes
-- v: speed of q in meters per minute

theorem distance_p_ran (h1 : d = 0.3 * v * t) : 1.3 * v * t = 1.3 * v * t :=
by
  sorry

end distance_p_ran_l181_181985


namespace maximum_distance_product_l181_181596

theorem maximum_distance_product (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let ρ1 := 4 * Real.cos α
  let ρ2 := 2 * Real.sin α
  |ρ1 * ρ2| ≤ 4 :=
by
  -- The proof would go here
  sorry

end maximum_distance_product_l181_181596


namespace smallest_n_for_gcd_lcm_l181_181531

theorem smallest_n_for_gcd_lcm (n a b : ℕ) (h_gcd : Nat.gcd a b = 999) (h_lcm : Nat.lcm a b = Nat.factorial n) :
  n = 37 := sorry

end smallest_n_for_gcd_lcm_l181_181531


namespace number_of_solutions_l181_181004

noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

theorem number_of_solutions (h : -1 ≤ x ∧ x ≤ 1) : 
  (∃ s : ℕ, s = 21 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g (g (g x)) = g x) :=
sorry

end number_of_solutions_l181_181004


namespace white_marbles_multiple_of_8_l181_181581

-- Definitions based on conditions
def blue_marbles : ℕ := 16
def num_groups : ℕ := 8

-- Stating the problem
theorem white_marbles_multiple_of_8 (white_marbles : ℕ) :
  (blue_marbles + white_marbles) % num_groups = 0 → white_marbles % num_groups = 0 :=
by
  sorry

end white_marbles_multiple_of_8_l181_181581


namespace symmetric_point_y_axis_l181_181620

theorem symmetric_point_y_axis (A B : ℝ × ℝ) (hA : A = (2, 5)) (h_symm : B = (-A.1, A.2)) :
  B = (-2, 5) :=
sorry

end symmetric_point_y_axis_l181_181620


namespace value_of_m_l181_181832

theorem value_of_m (m : ℤ) (h1 : abs m = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end value_of_m_l181_181832


namespace range_of_a_for_quadratic_eq_l181_181612

theorem range_of_a_for_quadratic_eq (a : ℝ) (h : ∀ x : ℝ, ax^2 = (x+1)*(x-1)) : a ≠ 1 :=
by
  sorry

end range_of_a_for_quadratic_eq_l181_181612


namespace candles_left_in_room_l181_181299

-- Define the variables and conditions
def total_candles : ℕ := 40
def alyssa_used : ℕ := total_candles / 2
def remaining_candles_after_alyssa : ℕ := total_candles - alyssa_used
def chelsea_used : ℕ := (7 * remaining_candles_after_alyssa) / 10
def final_remaining_candles : ℕ := remaining_candles_after_alyssa - chelsea_used

-- The theorem we need to prove
theorem candles_left_in_room : final_remaining_candles = 6 := by
  sorry

end candles_left_in_room_l181_181299


namespace cos_double_angle_l181_181875

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -Real.sqrt 3 / 2) : Real.cos (2 * α) = 1 / 2 :=
by
  sorry

end cos_double_angle_l181_181875


namespace pairs_satisfy_equation_l181_181255

theorem pairs_satisfy_equation :
  ∀ (x n : ℕ), (x > 0 ∧ n > 0) ∧ 3 * 2 ^ x + 4 = n ^ 2 → (x, n) = (2, 4) ∨ (x, n) = (5, 10) ∨ (x, n) = (6, 14) :=
by
  sorry

end pairs_satisfy_equation_l181_181255


namespace probability_selecting_girl_l181_181827

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end probability_selecting_girl_l181_181827


namespace find_possible_values_l181_181527

def real_number_y (y : ℝ) := (3 < y ∧ y < 4)

theorem find_possible_values (y : ℝ) (h : real_number_y y) : 
  42 < (y^2 + 7*y + 12) ∧ (y^2 + 7*y + 12) < 56 := 
sorry

end find_possible_values_l181_181527


namespace solve_for_x_l181_181918

-- Define the new operation m ※ n
def operation (m n : ℤ) : ℤ :=
  if m ≥ 0 then m + n else m / n

-- Define the condition given in the problem
def condition (x : ℤ) : Prop :=
  operation (-9) (-x) = x

-- The main theorem to prove
theorem solve_for_x (x : ℤ) : condition x ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end solve_for_x_l181_181918


namespace f_g_3_value_l181_181592

def f (x : ℝ) := x^3 + 1
def g (x : ℝ) := 3 * x + 2

theorem f_g_3_value : f (g 3) = 1332 := by
  sorry

end f_g_3_value_l181_181592


namespace sum_of_integers_is_23_l181_181534

theorem sum_of_integers_is_23
  (x y : ℕ) (x_pos : 0 < x) (y_pos : 0 < y) (h : x * y + x + y = 155) 
  (rel_prime : Nat.gcd x y = 1) (x_lt_30 : x < 30) (y_lt_30 : y < 30) :
  x + y = 23 :=
by
  sorry

end sum_of_integers_is_23_l181_181534


namespace jason_commute_with_detour_l181_181196

theorem jason_commute_with_detour (d1 d2 d3 d4 d5 : ℝ) 
  (h1 : d1 = 4)     -- Distance from house to first store
  (h2 : d2 = 6)     -- Distance between first and second store
  (h3 : d3 = d2 + (2/3) * d2) -- Distance between second and third store without detour
  (h4 : d4 = 3)     -- Additional distance due to detour
  (h5 : d5 = d1)    -- Distance from third store to work
  : d1 + d2 + (d3 + d4) + d5 = 27 :=
by
  sorry

end jason_commute_with_detour_l181_181196


namespace greatest_monthly_drop_is_march_l181_181227

-- Define the price changes for each month
def price_change_january : ℝ := -0.75
def price_change_february : ℝ := 1.50
def price_change_march : ℝ := -3.00
def price_change_april : ℝ := 2.50
def price_change_may : ℝ := -1.00
def price_change_june : ℝ := 0.50
def price_change_july : ℝ := -2.50

-- Prove that the month with the greatest drop in price is March
theorem greatest_monthly_drop_is_march :
  (price_change_march = -3.00) →
  (∀ m, m ≠ price_change_march → m ≥ price_change_march) :=
by
  intros h1 h2
  sorry

end greatest_monthly_drop_is_march_l181_181227


namespace prime_factor_of_T_l181_181860

-- Define constants and conditions
def x : ℕ := 2021
def T : ℕ := Nat.sqrt ((x + x) + (x - x) + (x * x) + (x / x))

-- Define what needs to be proved
theorem prime_factor_of_T : ∃ p : ℕ, Nat.Prime p ∧ Nat.factorization T p > 0 ∧ (∀ q : ℕ, Nat.Prime q ∧ Nat.factorization T q > 0 → q ≤ p) :=
sorry

end prime_factor_of_T_l181_181860


namespace initial_milk_quantity_l181_181529

theorem initial_milk_quantity 
  (milk_left_in_tank : ℕ) -- the remaining milk in the tank
  (pumping_rate : ℕ) -- the rate at which milk was pumped out
  (pumping_hours : ℕ) -- hours during which milk was pumped out
  (adding_rate : ℕ) -- the rate at which milk was added
  (adding_hours : ℕ) -- hours during which milk was added 
  (initial_milk : ℕ) -- initial milk collected
  (h1 : milk_left_in_tank = 28980) -- condition 3
  (h2 : pumping_rate = 2880) -- condition 1 (rate)
  (h3 : pumping_hours = 4) -- condition 1 (hours)
  (h4 : adding_rate = 1500) -- condition 2 (rate)
  (h5 : adding_hours = 7) -- condition 2 (hours)
  : initial_milk = 30000 :=
by
  sorry

end initial_milk_quantity_l181_181529


namespace product_of_inverses_l181_181717

theorem product_of_inverses : 
  ((1 - 1 / (3^2)) * (1 - 1 / (5^2)) * (1 - 1 / (7^2)) * (1 - 1 / (11^2)) * (1 - 1 / (13^2)) * (1 - 1 / (17^2))) = 210 / 221 := 
by {
  sorry
}

end product_of_inverses_l181_181717


namespace min_value_a_l181_181979

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ (Real.sqrt 2) / 2 → x^3 - 2 * x * Real.log x / Real.log a ≤ 0) ↔ a ≥ 1 / 4 := 
sorry

end min_value_a_l181_181979


namespace image_center_coordinates_l181_181763

-- Define the point reflecting across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the point translation by adding some units to the y-coordinate
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the initial point and translation
def initial_point : ℝ × ℝ := (3, -4)
def translation_units : ℝ := 5

-- Prove the final coordinates of the image of the center of circle Q
theorem image_center_coordinates : translate_y (reflect_x initial_point) translation_units = (3, 9) :=
  sorry

end image_center_coordinates_l181_181763


namespace find_n_l181_181838

noncomputable def tangent_line_problem (x0 : ℝ) (n : ℕ) : Prop :=
(x0 ∈ Set.Ioo (Real.sqrt n) (Real.sqrt (n + 1))) ∧
(∃ m : ℝ, 0 < m ∧ m < 1 ∧ (2 * x0 = 1 / m) ∧ (x0^2 = (Real.log m - 1)))

theorem find_n (x0 : ℝ) (n : ℕ) :
  tangent_line_problem x0 n → n = 2 :=
sorry

end find_n_l181_181838


namespace sum_opposite_sign_zero_l181_181264

def opposite_sign (a b : ℝ) : Prop :=
(a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem sum_opposite_sign_zero {a b : ℝ} (h : opposite_sign a b) : a + b = 0 :=
sorry

end sum_opposite_sign_zero_l181_181264


namespace rod_length_is_38_point_25_l181_181209

noncomputable def length_of_rod (n : ℕ) (l : ℕ) (conversion_factor : ℕ) : ℝ :=
  (n * l : ℝ) / conversion_factor

theorem rod_length_is_38_point_25 :
  length_of_rod 45 85 100 = 38.25 :=
by
  sorry

end rod_length_is_38_point_25_l181_181209


namespace find_number_l181_181545

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 100) : N = 192 :=
sorry

end find_number_l181_181545


namespace simon_fraction_of_alvin_l181_181645

theorem simon_fraction_of_alvin (alvin_age simon_age : ℕ) (h_alvin : alvin_age = 30)
  (h_simon : simon_age = 10) (h_fraction : ∃ f : ℚ, simon_age + 5 = f * (alvin_age + 5)) :
  ∃ f : ℚ, f = 3 / 7 := by
  sorry

end simon_fraction_of_alvin_l181_181645


namespace number_of_Sunzi_books_l181_181598

theorem number_of_Sunzi_books
    (num_books : ℕ) (total_cost : ℕ)
    (price_Zhuangzi price_Kongzi price_Mengzi price_Laozi price_Sunzi : ℕ)
    (num_Zhuangzi num_Kongzi num_Mengzi num_Laozi num_Sunzi : ℕ) :
  num_books = 300 →
  total_cost = 4500 →
  price_Zhuangzi = 10 →
  price_Kongzi = 20 →
  price_Mengzi = 15 →
  price_Laozi = 30 →
  price_Sunzi = 12 →
  num_Zhuangzi = num_Kongzi →
  num_Sunzi = 4 * num_Laozi + 15 →
  num_Zhuangzi + num_Kongzi + num_Mengzi + num_Laozi + num_Sunzi = num_books →
  price_Zhuangzi * num_Zhuangzi +
  price_Kongzi * num_Kongzi +
  price_Mengzi * num_Mengzi +
  price_Laozi * num_Laozi +
  price_Sunzi * num_Sunzi = total_cost →
  num_Sunzi = 75 :=
by
  intros h_nb h_tc h_pZ h_pK h_pM h_pL h_pS h_nZ h_nS h_books h_cost
  sorry

end number_of_Sunzi_books_l181_181598


namespace hoseok_subtraction_result_l181_181972

theorem hoseok_subtraction_result:
  ∃ x : ℤ, 15 * x = 45 ∧ x - 1 = 2 :=
by
  sorry

end hoseok_subtraction_result_l181_181972


namespace sequence_general_term_and_sum_sum_tn_bound_l181_181844

theorem sequence_general_term_and_sum (c : ℝ) (h₁ : c = 1) 
  (f : ℕ → ℝ) (hf : ∀ x, f x = (1 / 3) ^ x) :
  (∀ n, a_n = -2 / 3 ^ n) ∧ (∀ n, b_n = 2 * n - 1) :=
by {
  sorry
}

theorem sum_tn_bound (h₂ : ∀ n > 0, T_n = (1 / 2) * (1 - 1 / (2 * n + 1))) :
  ∃ n, T_n > 1005 / 2014 ∧ n = 252 :=
by {
  sorry
}

end sequence_general_term_and_sum_sum_tn_bound_l181_181844


namespace neg_ex_proposition_l181_181615

open Classical

theorem neg_ex_proposition :
  ¬ (∃ n : ℕ, n^2 > 2^n) ↔ ∀ n : ℕ, n^2 ≤ 2^n :=
by sorry

end neg_ex_proposition_l181_181615


namespace first_group_size_l181_181674

theorem first_group_size
  (x : ℕ)
  (h1 : 2 * x + 22 + 16 + 14 = 68) : 
  x = 8 :=
by
  sorry

end first_group_size_l181_181674


namespace max_tickets_sold_l181_181368

theorem max_tickets_sold (bus_capacity : ℕ) (num_stations : ℕ) (max_capacity : bus_capacity = 25) 
  (total_stations : num_stations = 14) : 
  ∃ (tickets : ℕ), tickets = 67 :=
by 
  sorry

end max_tickets_sold_l181_181368


namespace james_farmer_walk_distance_l181_181859

theorem james_farmer_walk_distance (d : ℝ) :
  ∃ d : ℝ,
    (∀ w : ℝ, (w = 300 + 50 → d = 20) ∧ 
             (w' = w * 1.30 ∧ w'' = w' * 1.20 → w'' = 546)) :=
by
  sorry

end james_farmer_walk_distance_l181_181859


namespace last_two_digits_of_sum_l181_181389

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum :
  last_two_digits (factorial 4 + factorial 5 + factorial 6 + factorial 7 + factorial 8 + factorial 9) = 4 :=
by
  sorry

end last_two_digits_of_sum_l181_181389


namespace birds_find_more_than_half_millet_on_sunday_l181_181848

noncomputable def seed_millet_fraction : ℕ → ℚ
| 0 => 2 * 0.2 -- initial amount on Day 1 (Monday)
| (n+1) => 0.7 * seed_millet_fraction n + 0.4

theorem birds_find_more_than_half_millet_on_sunday :
  let dayMillets : ℕ := 7
  let total_seeds : ℚ := 2
  let half_seeds : ℚ := total_seeds / 2
  (seed_millet_fraction dayMillets > half_seeds) := by
    sorry

end birds_find_more_than_half_millet_on_sunday_l181_181848


namespace four_digit_multiples_of_13_and_7_l181_181584

theorem four_digit_multiples_of_13_and_7 : 
  (∃ n : ℕ, 
    (∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 91 = 0 → k = 1001 + 91 * (n - 11)) 
    ∧ n - 11 + 1 = 99) :=
by
  sorry

end four_digit_multiples_of_13_and_7_l181_181584


namespace equalize_money_l181_181144

theorem equalize_money (ann_money : ℕ) (bill_money : ℕ) : 
  ann_money = 777 → 
  bill_money = 1111 → 
  ∃ x, bill_money - x = ann_money + x :=
by
  sorry

end equalize_money_l181_181144


namespace intersection_is_empty_l181_181751

-- Define sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 3, 4}

-- Define isolated elements for a set
def is_isolated (A : Set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

-- Define isolated sets
def isolated_set (A : Set ℕ) : Set ℕ :=
  {x | is_isolated A x}

-- Define isolated sets for M and N
def M' := isolated_set M
def N' := isolated_set N

-- The intersection of the isolated sets
theorem intersection_is_empty : M' ∩ N' = ∅ := 
  sorry

end intersection_is_empty_l181_181751


namespace valentines_given_l181_181678

theorem valentines_given (original current given : ℕ) (h1 : original = 58) (h2 : current = 16) (h3 : given = original - current) : given = 42 := by
  sorry

end valentines_given_l181_181678


namespace reciprocal_of_5_is_1_div_5_l181_181707

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- The problem statement: Prove that the reciprocal of 5 is 1/5
theorem reciprocal_of_5_is_1_div_5 : is_reciprocal 5 (1 / 5) :=
by
  sorry

end reciprocal_of_5_is_1_div_5_l181_181707


namespace parameter_a_values_l181_181797

theorem parameter_a_values (a : ℝ) :
  (∃ x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ ((|x| - 8)^2 + (|y| - 15)^2 = a) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, |x₁ + y₁ + 8| + |x₁ - y₁ + 8| = 16 →
      (|x₁| - 8)^2 + (|y₁| - 15)^2 = a →
      |x₂ + y₂ + 8| + |x₂ - y₂ + 8| = 16 →
      (|x₂| - 8)^2 + (|y₂| - 15)^2 = a →
      (x₁, y₁) = (x₂, y₂) ∨ (x₁, y₁) = (y₂, x₂))) ↔ a = 49 ∨ a = 289 :=
by sorry

end parameter_a_values_l181_181797


namespace day_90_N_minus_1_is_Thursday_l181_181300

/-- 
    Given that the 150th day of year N is a Sunday, 
    and the 220th day of year N+2 is also a Sunday,
    prove that the 90th day of year N-1 is a Thursday.
-/
theorem day_90_N_minus_1_is_Thursday (N : ℕ)
    (h1 : (150 % 7 = 0))  -- 150th day of year N is Sunday
    (h2 : (220 % 7 = 0))  -- 220th day of year N + 2 is Sunday
    : ((90 + 366) % 7 = 4) := -- 366 days in a leap year (N-1), 90th day modulo 7 is Thursday
by
  sorry

end day_90_N_minus_1_is_Thursday_l181_181300


namespace race_length_l181_181165

theorem race_length (covered_meters remaining_meters race_length : ℕ)
  (h_covered : covered_meters = 721)
  (h_remaining : remaining_meters = 279)
  (h_race_length : race_length = covered_meters + remaining_meters) :
  race_length = 1000 :=
by
  rw [h_covered, h_remaining] at h_race_length
  exact h_race_length

end race_length_l181_181165


namespace smallest_solution_of_equation_l181_181557

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (9 * x^2 - 45 * x + 50 = 0) ∧ (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) :=
sorry

end smallest_solution_of_equation_l181_181557


namespace sin_cos_from_tan_l181_181506

variable {α : Real} (hα : α > 0) -- Assume α is an acute angle

theorem sin_cos_from_tan (h : Real.tan α = 2) : 
  Real.sin α = 2 / Real.sqrt 5 ∧ Real.cos α = 1 / Real.sqrt 5 := 
by sorry

end sin_cos_from_tan_l181_181506


namespace Shekar_biology_marks_l181_181021

theorem Shekar_biology_marks 
  (math_marks : ℕ := 76) 
  (science_marks : ℕ := 65) 
  (social_studies_marks : ℕ := 82) 
  (english_marks : ℕ := 47) 
  (average_marks : ℕ := 71) 
  (num_subjects : ℕ := 5) 
  (biology_marks : ℕ) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks → biology_marks = 85 := 
by 
  sorry

end Shekar_biology_marks_l181_181021


namespace waiter_customer_count_l181_181032

def initial_customers := 33
def customers_left := 31
def new_customers := 26

theorem waiter_customer_count :
  (initial_customers - customers_left) + new_customers = 28 :=
by
  -- This is a placeholder for the proof that can be filled later.
  sorry

end waiter_customer_count_l181_181032


namespace sum_of_five_integers_l181_181513

-- Definitions of the five integers based on the conditions given in the problem
def a := 12345
def b := 23451
def c := 34512
def d := 45123
def e := 51234

-- Statement of the proof problem
theorem sum_of_five_integers :
  a + b + c + d + e = 166665 :=
by
  -- The proof is omitted
  sorry

end sum_of_five_integers_l181_181513


namespace find_a_range_l181_181794

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then -(x - 1) ^ 2 else (3 - a) * x + 4 * a

theorem find_a_range (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ a - f x₂ a) / (x₁ - x₂) > 0) ↔ (-1 ≤ a ∧ a < 3) :=
sorry

end find_a_range_l181_181794


namespace place_value_ratio_l181_181168

theorem place_value_ratio :
  let val_6 := 1000
  let val_2 := 0.1
  val_6 / val_2 = 10000 :=
by
  -- the proof would go here
  sorry

end place_value_ratio_l181_181168


namespace ScarlettsDishCost_l181_181046

theorem ScarlettsDishCost (L P : ℝ) (tip_rate tip_amount : ℝ) (x : ℝ) 
  (hL : L = 10) (hP : P = 17) (htip_rate : tip_rate = 0.10) (htip_amount : tip_amount = 4) 
  (h : tip_rate * (L + P + x) = tip_amount) : x = 13 :=
by
  sorry

end ScarlettsDishCost_l181_181046


namespace range_of_a_l181_181399

theorem range_of_a
  (P : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) :
  ¬P → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l181_181399


namespace explicit_formula_l181_181303

variable (f : ℝ → ℝ)
variable (is_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
variable (max_value : ∀ x, f x ≤ 13)
variable (value_at_3 : f 3 = 5)
variable (value_at_neg1 : f (-1) = 5)

theorem explicit_formula :
  (∀ x, f x = -2 * x^2 + 4 * x + 11) :=
by
  sorry

end explicit_formula_l181_181303


namespace seashells_found_l181_181130

theorem seashells_found (C B : ℤ) (h1 : 9 * B = 7 * C) (h2 : B = C - 12) : C = 54 :=
by
  sorry

end seashells_found_l181_181130


namespace units_digit_of_result_l181_181412

def tens_plus_one (a b : ℕ) : Prop := a = b + 1

theorem units_digit_of_result (a b : ℕ) (h : tens_plus_one a b) :
  ((10 * a + b) - (10 * b + a)) % 10 = 9 :=
by
  -- Let's mark this part as incomplete using sorry.
  sorry

end units_digit_of_result_l181_181412


namespace triangle_incircle_ratio_l181_181224

theorem triangle_incircle_ratio (r s q : ℝ) (h1 : r + s = 8) (h2 : r < s) (h3 : r + q = 13) (h4 : s + q = 17) (h5 : 8 + 13 > 17 ∧ 8 + 17 > 13 ∧ 13 + 17 > 8):
  r / s = 1 / 3 := by sorry

end triangle_incircle_ratio_l181_181224


namespace irreducible_fraction_l181_181076

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l181_181076


namespace total_coins_constant_l181_181712

-- Definitions based on the conditions
def stack1 := 12
def stack2 := 17
def stack3 := 23
def stack4 := 8

def totalCoins := stack1 + stack2 + stack3 + stack4 -- 60 coins
def is_divisor (x: ℕ) := x ∣ totalCoins

-- The theorem statement
theorem total_coins_constant {x: ℕ} (h: is_divisor x) : totalCoins = 60 :=
by
  -- skip the proof steps
  sorry

end total_coins_constant_l181_181712


namespace fraction_of_earth_surface_humans_can_inhabit_l181_181567

theorem fraction_of_earth_surface_humans_can_inhabit :
  (1 / 3) * (2 / 3) = (2 / 9) :=
by
  sorry

end fraction_of_earth_surface_humans_can_inhabit_l181_181567


namespace area_of_rectangle_l181_181738

theorem area_of_rectangle (l w : ℝ) (h_perimeter : 2 * (l + w) = 126) (h_difference : l - w = 37) : l * w = 650 :=
sorry

end area_of_rectangle_l181_181738


namespace tan_diff_l181_181623

variables {α β : ℝ}

theorem tan_diff (h1 : Real.tan α = -3/4) (h2 : Real.tan (Real.pi - β) = 1/2) :
  Real.tan (α - β) = -2/11 :=
by
  sorry

end tan_diff_l181_181623


namespace trigonometric_identity_l181_181519

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.sin (α + π / 3) = 12 / 13) 
  : Real.cos (π / 6 - α) = 12 / 13 := 
sorry

end trigonometric_identity_l181_181519


namespace p_n_divisible_by_5_l181_181630

noncomputable def p_n (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

theorem p_n_divisible_by_5 (n : ℕ) (h : n ≠ 0) : p_n n % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end p_n_divisible_by_5_l181_181630


namespace animals_percentage_monkeys_l181_181278

theorem animals_percentage_monkeys (initial_monkeys : ℕ) (initial_birds : ℕ) (birds_eaten : ℕ) (final_monkeys : ℕ) (final_birds : ℕ) : 
  initial_monkeys = 6 → 
  initial_birds = 6 → 
  birds_eaten = 2 → 
  final_monkeys = initial_monkeys → 
  final_birds = initial_birds - birds_eaten → 
  (final_monkeys * 100 / (final_monkeys + final_birds) = 60) := 
by intros
   sorry

end animals_percentage_monkeys_l181_181278


namespace highest_possible_N_l181_181298

/--
In a football tournament with 15 teams, each team played exactly once against every other team.
A win earns 3 points, a draw earns 1 point, and a loss earns 0 points.
We need to prove that the highest possible integer \( N \) such that there are at least 6 teams with at least \( N \) points is 34.
-/
theorem highest_possible_N : 
  ∃ (N : ℤ) (teams : Fin 15 → ℤ) (successfulTeams : Fin 6 → Fin 15),
    (∀ i j, i ≠ j → teams i + teams j ≤ 207) ∧ 
    (∀ k, k < 6 → teams (successfulTeams k) ≥ 34) ∧ 
    (∀ k, 0 ≤ teams k) ∧ 
    N = 34 := sorry

end highest_possible_N_l181_181298


namespace volunteer_assigned_probability_l181_181292

theorem volunteer_assigned_probability :
  let volunteers := ["A", "B", "C", "D"]
  let areas := ["Beijing", "Zhangjiakou"]
  let total_ways := 14
  let favorable_ways := 6
  ∃ (p : ℚ), p = 6/14 → (1 / total_ways) * favorable_ways = 3/7
:= sorry

end volunteer_assigned_probability_l181_181292


namespace area_enclosed_by_graph_l181_181358

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l181_181358


namespace average_score_correct_l181_181724

-- Define the conditions
def simplified_scores : List Int := [10, -5, 0, 8, -3]
def base_score : Int := 90

-- Translate simplified score to actual score
def actual_score (s : Int) : Int :=
  base_score + s

-- Calculate the average of the actual scores
def average_score : Int :=
  (simplified_scores.map actual_score).sum / simplified_scores.length

-- The proof statement
theorem average_score_correct : average_score = 92 := 
by 
  -- Steps to compute the average score
  -- sorry is used since the proof steps are not required
  sorry

end average_score_correct_l181_181724


namespace car_mpg_in_city_l181_181932

theorem car_mpg_in_city 
    (miles_per_tank_highway : Real)
    (miles_per_tank_city : Real)
    (mpg_difference : Real)
    : True := by
  let H := 21.05
  let T := 720 / H
  let C := H - 10
  have h1 : 720 = H * T := by
    sorry
  have h2 : 378 = C * T := by
    sorry
  exact True.intro

end car_mpg_in_city_l181_181932


namespace pqrs_product_l181_181737

noncomputable def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
noncomputable def Q : ℝ := -Real.sqrt 2012 - Real.sqrt 2013
noncomputable def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
noncomputable def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

theorem pqrs_product : P * Q * R * S = 1 := 
by 
  sorry

end pqrs_product_l181_181737


namespace roots_sum_product_l181_181999

variable {a b : ℝ}

theorem roots_sum_product (ha : a + b = 6) (hp : a * b = 8) : 
  a^4 + b^4 + a^3 * b + a * b^3 = 432 :=
by
  sorry

end roots_sum_product_l181_181999


namespace tan_of_trig_eq_l181_181994

theorem tan_of_trig_eq (x : Real) (h : (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2) : Real.tan x = 4 / 3 :=
by sorry

end tan_of_trig_eq_l181_181994


namespace find_z_l181_181500

theorem find_z (z : ℝ) (v : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ)
  (h_v : v = (4, 1, z)) (h_u : u = (2, -3, 4))
  (h_eq : (4 * 2 + 1 * -3 + z * 4) / (2 * 2 + -3 * -3 + 4 * 4) = 5 / 29) :
  z = 0 :=
by
  sorry

end find_z_l181_181500


namespace base7_to_base10_l181_181594

open Nat

theorem base7_to_base10 : (3 * 7^2 + 5 * 7^1 + 1 * 7^0 = 183) :=
by
  sorry

end base7_to_base10_l181_181594


namespace polynomial_divisibility_l181_181189

noncomputable def polynomial_with_positive_int_coeffs : Type :=
{ f : ℕ → ℕ // ∀ m n : ℕ, f m < f n ↔ m < n }

theorem polynomial_divisibility
  (f : polynomial_with_positive_int_coeffs)
  (n : ℕ) (hn : n > 0) :
  f.1 n ∣ f.1 (f.1 n + 1) ↔ n = 1 :=
sorry

end polynomial_divisibility_l181_181189


namespace min_value_is_8_plus_4_sqrt_3_l181_181756

noncomputable def min_value_of_expression (a b : ℝ) : ℝ :=
  2 / a + 1 / b

theorem min_value_is_8_plus_4_sqrt_3 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + 2 * b = 1) :
  min_value_of_expression a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_is_8_plus_4_sqrt_3_l181_181756


namespace sum_of_opposites_is_zero_l181_181603

theorem sum_of_opposites_is_zero (a b : ℚ) (h : a = -b) : a + b = 0 := 
by sorry

end sum_of_opposites_is_zero_l181_181603


namespace exists_solution_l181_181118

noncomputable def smallest_c0 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) : ℕ :=
  a * b - a - b + 1

theorem exists_solution (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) :
  ∃ c0, (c0 = smallest_c0 a b ha hb h) ∧ ∀ c : ℕ, c ≥ c0 → ∃ x y : ℕ, a * x + b * y = c :=
sorry

end exists_solution_l181_181118


namespace john_can_fix_l181_181986

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l181_181986


namespace circle_diameter_l181_181307

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l181_181307


namespace midpoint_sum_l181_181829

theorem midpoint_sum (x y : ℝ) (h1 : (x + 0) / 2 = 2) (h2 : (y + 9) / 2 = 4) : x + y = 3 := by
  sorry

end midpoint_sum_l181_181829


namespace sort_mail_together_time_l181_181782

-- Definitions of work rates
def mail_handler_work_rate : ℚ := 1 / 3
def assistant_work_rate : ℚ := 1 / 6

-- Definition to calculate combined work time
def combined_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

-- Statement to prove
theorem sort_mail_together_time :
  combined_time mail_handler_work_rate assistant_work_rate = 2 := by
  -- Proof goes here
  sorry

end sort_mail_together_time_l181_181782


namespace geometric_sequence_first_term_l181_181150

theorem geometric_sequence_first_term (a b c : ℕ) 
    (h1 : 16 = a * (2^3)) 
    (h2 : 32 = a * (2^4)) : 
    a = 2 := 
sorry

end geometric_sequence_first_term_l181_181150


namespace find_ellipse_parameters_l181_181340

noncomputable def ellipse_centers_and_axes (F1 F2 : ℝ × ℝ) (d : ℝ) (tangent_slope : ℝ) :=
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  let a := d / 2
  let c := (Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  (h, k, a, b)

theorem find_ellipse_parameters :
  let F1 := (-1, 1)
  let F2 := (5, 1)
  let d := 10
  let tangent_at_x_axis_slope := 1
  let (h, k, a, b) := ellipse_centers_and_axes F1 F2 d tangent_at_x_axis_slope
  h + k + a + b = 12 :=
by
  sorry

end find_ellipse_parameters_l181_181340


namespace integer_solutions_inequality_system_l181_181766

theorem integer_solutions_inequality_system :
  {x : ℤ | 2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1} = {3, 4, 5} :=
by
  sorry

end integer_solutions_inequality_system_l181_181766


namespace XiaoMing_selection_l181_181939

def final_positions (n : Nat) : List Nat :=
  if n <= 2 then
    List.range n
  else
    final_positions (n / 2) |>.filter (λ k => k % 2 = 0) |>.map (λ k => k / 2)

theorem XiaoMing_selection (n : Nat) (h : n = 32) : final_positions n = [16, 32] :=
  by
  sorry

end XiaoMing_selection_l181_181939


namespace total_trucks_l181_181673

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end total_trucks_l181_181673


namespace how_much_money_per_tshirt_l181_181669

def money_made_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) : Prop :=
  total_money_tshirts / number_tshirts = 62

theorem how_much_money_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) 
  (h1 : total_money_tshirts = 11346) 
  (h2 : number_tshirts = 183) : 
  money_made_per_tshirt total_money_tshirts number_tshirts := 
by 
  sorry

end how_much_money_per_tshirt_l181_181669


namespace multiplication_problem_solution_l181_181965

theorem multiplication_problem_solution (a b c : ℕ) 
  (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1) 
  (h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h3 : (a * 100 + b * 10 + b) * c = b * 1000 + c * 100 + b * 10 + 1) : 
  a = 5 ∧ b = 3 ∧ c = 7 := 
sorry

end multiplication_problem_solution_l181_181965


namespace blue_marble_difference_l181_181803

theorem blue_marble_difference :
  ∃ a b : ℕ, (10 * a = 10 * b) ∧ (3 * a + b = 80) ∧ (7 * a - 9 * b = 40) := by
  sorry

end blue_marble_difference_l181_181803


namespace range_of_inverse_proportion_l181_181768

theorem range_of_inverse_proportion (x : ℝ) (h : 3 < x) :
    -1 < -3 / x ∧ -3 / x < 0 :=
by
  sorry

end range_of_inverse_proportion_l181_181768


namespace solve_x_values_l181_181884

theorem solve_x_values : ∀ (x : ℝ), (x + 45 / (x - 4) = -10) ↔ (x = -1 ∨ x = -5) :=
by
  intro x
  sorry

end solve_x_values_l181_181884


namespace angle_bisector_theorem_l181_181937

noncomputable def ratio_of_segments (x y z p q : ℝ) :=
  q / x = y / (y + x)

theorem angle_bisector_theorem (x y z p q : ℝ) (h1 : p / x = q / y)
  (h2 : p + q = z) : ratio_of_segments x y z p q :=
by
  sorry

end angle_bisector_theorem_l181_181937


namespace factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l181_181111

theorem factorization_A (x y : ℝ) : x^2 - 2 * x * y = x * (x - 2 * y) :=
  by sorry

theorem factorization_B (x y : ℝ) : x^2 - 25 * y^2 = (x - 5 * y) * (x + 5 * y) :=
  by sorry

theorem factorization_C (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 :=
  by sorry

theorem factorization_D_incorrect (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) :=
  by sorry

theorem factorization_D_correct (x : ℝ) : x^2 + x - 2 = (x + 2) * (x - 1) :=
  by sorry

end factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l181_181111


namespace determine_b_l181_181703

noncomputable def f (x b : ℝ) : ℝ := 1 / (3 * x + b)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, f (f_inv x) b = x) -> b = 3 :=
by
  intro h
  sorry

end determine_b_l181_181703


namespace price_and_max_units_proof_l181_181448

/-- 
Given the conditions of purchasing epidemic prevention supplies: 
- 60 units of type A and 45 units of type B costing 1140 yuan
- 45 units of type A and 30 units of type B costing 840 yuan
- A total of 600 units with a cost not exceeding 8000 yuan

Prove:
1. The price of each unit of type A is 16 yuan, and type B is 4 yuan.
2. The maximum number of units of type A that can be purchased is 466.
--/
theorem price_and_max_units_proof 
  (x y : ℕ) 
  (m : ℕ)
  (h1 : 60 * x + 45 * y = 1140) 
  (h2 : 45 * x + 30 * y = 840) 
  (h3 : 16 * m + 4 * (600 - m) ≤ 8000) 
  (h4 : m ≤ 600) :
  x = 16 ∧ y = 4 ∧ m = 466 := 
by 
  sorry

end price_and_max_units_proof_l181_181448


namespace hyperbola_equation_l181_181748

-- Definitions based on the conditions:
def hyperbola (x y a b : ℝ) : Prop := (y^2 / a^2) - (x^2 / b^2) = 1

def point_on_hyperbola (a b : ℝ) : Prop := hyperbola 2 (-2) a b

def asymptotes (a b : ℝ) : Prop := a / b = (Real.sqrt 2) / 2

-- Prove the equation of the hyperbola
theorem hyperbola_equation :
  ∃ a b, a = Real.sqrt 2 ∧ b = 2 ∧ hyperbola y x (Real.sqrt 2) 2 :=
by
  -- Placeholder for the actual proof
  sorry

end hyperbola_equation_l181_181748


namespace distance_between_towns_l181_181275

variables (x y z : ℝ)

theorem distance_between_towns
  (h1 : x / 24 + y / 16 + z / 12 = 2)
  (h2 : x / 12 + y / 16 + z / 24 = 2.25) :
  x + y + z = 34 :=
sorry

end distance_between_towns_l181_181275


namespace fraction_to_decimal_l181_181067

theorem fraction_to_decimal : (7 / 50 : ℝ) = 0.14 := by
  sorry

end fraction_to_decimal_l181_181067


namespace f_max_min_l181_181327

def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom cauchy_f : ∀ x y : ℝ, f (x + y) = f x + f y
axiom less_than_zero : ∀ x : ℝ, x > 0 → f x < 0
axiom f_one : f 1 = -2

theorem f_max_min : (∀ x ∈ [-3, 3], f (-3) = 6 ∧ f 3 = -6) :=
by sorry

end f_max_min_l181_181327


namespace right_triangle_one_leg_div_by_3_l181_181881

theorem right_triangle_one_leg_div_by_3 {a b c : ℕ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (h : a^2 + b^2 = c^2) : 3 ∣ a ∨ 3 ∣ b := 
by 
  sorry

end right_triangle_one_leg_div_by_3_l181_181881


namespace andrew_paid_total_l181_181249

-- Define the quantities and rates
def quantity_grapes : ℕ := 14
def rate_grapes : ℕ := 54
def quantity_mangoes : ℕ := 10
def rate_mangoes : ℕ := 62

-- Define the cost calculations
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes
def total_cost : ℕ := cost_grapes + cost_mangoes

-- Prove the total amount paid is as expected
theorem andrew_paid_total : total_cost = 1376 := by
  sorry 

end andrew_paid_total_l181_181249


namespace jodi_walked_miles_per_day_l181_181495

theorem jodi_walked_miles_per_day (x : ℕ) 
  (h1 : 6 * x + 12 + 18 + 24 = 60) : 
  x = 1 :=
by
  sorry

end jodi_walked_miles_per_day_l181_181495


namespace chocolates_not_in_box_initially_l181_181899

theorem chocolates_not_in_box_initially 
  (total_chocolates : ℕ) 
  (chocolates_friend_brought : ℕ) 
  (initial_boxes : ℕ) 
  (additional_boxes : ℕ)
  (total_after_friend : ℕ)
  (chocolates_each_box : ℕ)
  (total_chocolates_initial : ℕ) :
  total_chocolates = 50 ∧ initial_boxes = 3 ∧ chocolates_friend_brought = 25 ∧ total_after_friend = 75 
  ∧ additional_boxes = 2 ∧ chocolates_each_box = 15 ∧ total_chocolates_initial = 50
  → (total_chocolates_initial - (initial_boxes * chocolates_each_box)) = 5 :=
by
  sorry

end chocolates_not_in_box_initially_l181_181899


namespace quadratic_real_roots_condition_l181_181216

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m-1) * x₁^2 - 4 * x₁ + 1 = 0 ∧ (m-1) * x₂^2 - 4 * x₂ + 1 = 0) ↔ (m < 5 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_condition_l181_181216


namespace boy_speed_in_kmph_l181_181109

-- Define the conditions
def side_length : ℕ := 35
def time_seconds : ℕ := 56

-- Perimeter of the square field
def perimeter : ℕ := 4 * side_length

-- Speed in meters per second
def speed_mps : ℚ := perimeter / time_seconds

-- Speed in kilometers per hour
def speed_kmph : ℚ := speed_mps * (3600 / 1000)

-- Theorem stating the boy's speed is 9 km/hr
theorem boy_speed_in_kmph : speed_kmph = 9 :=
by
  sorry

end boy_speed_in_kmph_l181_181109


namespace distance_from_original_position_l181_181441

/-- Definition of initial problem conditions and parameters --/
def square_area (l : ℝ) : Prop :=
  l * l = 18

def folded_area_relation (x : ℝ) : Prop :=
  0.5 * x^2 = 2 * (18 - 0.5 * x^2)

/-- The main statement that needs to be proved --/
theorem distance_from_original_position :
  ∃ (A_initial A_folded_dist : ℝ),
    square_area A_initial ∧
    (∃ x : ℝ, folded_area_relation x ∧ A_folded_dist = 2 * Real.sqrt 6 * Real.sqrt 2) ∧
    A_folded_dist = 4 * Real.sqrt 3 :=
by
  -- The proof is omitted here; providing structure for the problem.
  sorry

end distance_from_original_position_l181_181441


namespace rook_placement_l181_181962

theorem rook_placement : 
  let n := 8
  let k := 6
  let binom := Nat.choose
  binom 8 6 * binom 8 6 * Nat.factorial 6 = 564480 := by
    sorry

end rook_placement_l181_181962


namespace estimate_total_fish_in_pond_l181_181306

theorem estimate_total_fish_in_pond :
  ∀ (total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample : ℕ),
  initial_sample_size = 100 →
  second_sample_size = 200 →
  tagged_in_second_sample = 10 →
  total_tagged_fish = 100 →
  (total_tagged_fish : ℚ) / (total_fish : ℚ) = tagged_in_second_sample / second_sample_size →
  total_fish = 2000 := by
  intros total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample
  intro h1 h2 h3 h4 h5
  sorry

end estimate_total_fish_in_pond_l181_181306


namespace calculate_sum_and_difference_l181_181465

theorem calculate_sum_and_difference : 0.5 - 0.03 + 0.007 = 0.477 := sorry

end calculate_sum_and_difference_l181_181465


namespace range_of_a_l181_181047

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x > 3 → x > a)) ↔ (a ≤ 3) :=
sorry

end range_of_a_l181_181047


namespace geometric_seq_common_ratio_l181_181477

theorem geometric_seq_common_ratio (a_n : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = a_n 1 * (1 - q ^ 3) / (1 - q))
  (hS2 : S 2 = a_n 1 * (1 - q ^ 2) / (1 - q))
  (h : S 3 + 3 * S 2 = 0) 
  (hq_not_one : q ≠ 1) :
  q = -2 :=
by sorry

end geometric_seq_common_ratio_l181_181477


namespace total_distance_is_correct_l181_181063

noncomputable def magic_ball_total_distance : ℕ := sorry

theorem total_distance_is_correct : magic_ball_total_distance = 80 := sorry

end total_distance_is_correct_l181_181063


namespace student_most_stable_l181_181259

theorem student_most_stable (A B C : ℝ) (hA : A = 0.024) (hB : B = 0.08) (hC : C = 0.015) : C < A ∧ C < B := by
  sorry

end student_most_stable_l181_181259


namespace parallelogram_area_leq_half_triangle_area_l181_181625

-- Definition of a triangle and a parallelogram inside it.
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

structure Parallelogram (α : Type) [LinearOrderedField α] :=
(P Q R S : α × α)

-- Function to calculate the area of a triangle
def triangle_area {α : Type} [LinearOrderedField α] (T : Triangle α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Function to calculate the area of a parallelogram
def parallelogram_area {α : Type} [LinearOrderedField α] (P : Parallelogram α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Statement of the problem
theorem parallelogram_area_leq_half_triangle_area {α : Type} [LinearOrderedField α]
(T : Triangle α) (P : Parallelogram α) (inside : P.P.1 < T.A.1 ∧ P.P.2 < T.C.1) : 
  parallelogram_area P ≤ 1 / 2 * triangle_area T :=
sorry

end parallelogram_area_leq_half_triangle_area_l181_181625


namespace age_30_years_from_now_l181_181134

variables (ElderSonAge : ℕ) (DeclanAgeDiff : ℕ) (YoungerSonAgeDiff : ℕ) (ThirdSiblingAgeDiff : ℕ)

-- Given conditions
def elder_son_age : ℕ := 40
def declan_age : ℕ := elder_son_age + 25
def younger_son_age : ℕ := elder_son_age - 10
def third_sibling_age : ℕ := younger_son_age - 5

-- To prove the ages 30 years from now
def younger_son_age_30_years_from_now : ℕ := younger_son_age + 30
def third_sibling_age_30_years_from_now : ℕ := third_sibling_age + 30

-- The proof statement
theorem age_30_years_from_now : 
  younger_son_age_30_years_from_now = 60 ∧ 
  third_sibling_age_30_years_from_now = 55 :=
by
  sorry

end age_30_years_from_now_l181_181134


namespace payal_finished_fraction_l181_181502

-- Define the conditions
variables (x : ℕ)

-- Given conditions
-- 1. Total pages in the book
def total_pages : ℕ := 60
-- 2. Payal has finished 20 more pages than she has yet to read.
def pages_yet_to_read (x : ℕ) : ℕ := x - 20

-- Main statement to prove: the fraction of the pages finished is 2/3
theorem payal_finished_fraction (h : x + (x - 20) = 60) : (x : ℚ) / 60 = 2 / 3 :=
sorry

end payal_finished_fraction_l181_181502


namespace find_k_from_roots_ratio_l181_181126

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l181_181126


namespace dan_initial_money_l181_181204

theorem dan_initial_money (cost_candy : ℕ) (cost_chocolate : ℕ) (total_spent: ℕ) (hc : cost_candy = 7) (hch : cost_chocolate = 6) (hs : total_spent = 13) 
  (h : total_spent = cost_candy + cost_chocolate) : total_spent = 13 := by
  sorry

end dan_initial_money_l181_181204


namespace John_overall_profit_l181_181297

theorem John_overall_profit :
  let CP_grinder := 15000
  let Loss_percentage_grinder := 0.04
  let CP_mobile_phone := 8000
  let Profit_percentage_mobile_phone := 0.10
  let CP_refrigerator := 24000
  let Profit_percentage_refrigerator := 0.08
  let CP_television := 12000
  let Loss_percentage_television := 0.06
  let SP_grinder := CP_grinder * (1 - Loss_percentage_grinder)
  let SP_mobile_phone := CP_mobile_phone * (1 + Profit_percentage_mobile_phone)
  let SP_refrigerator := CP_refrigerator * (1 + Profit_percentage_refrigerator)
  let SP_television := CP_television * (1 - Loss_percentage_television)
  let Total_CP := CP_grinder + CP_mobile_phone + CP_refrigerator + CP_television
  let Total_SP := SP_grinder + SP_mobile_phone + SP_refrigerator + SP_television
  let Overall_profit := Total_SP - Total_CP
  Overall_profit = 1400 := by
  sorry

end John_overall_profit_l181_181297


namespace find_building_block_width_l181_181192

noncomputable def building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40)
: ℕ :=
(8 * 10 * 12) / 40 / (3 * 4)

theorem find_building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40) :
  building_block_width box_height box_width box_length building_block_height building_block_length num_building_blocks box_height_eq box_width_eq box_length_eq building_block_height_eq building_block_length_eq num_building_blocks_eq = 2 := 
sorry

end find_building_block_width_l181_181192


namespace rectangular_region_area_l181_181826

-- Definitions based on conditions
variable (w : ℝ) -- length of the shorter sides
variable (l : ℝ) -- length of the longer side
variable (total_fence_length : ℝ) -- total length of the fence

-- Given conditions as hypotheses
theorem rectangular_region_area
  (h1 : l = 2 * w) -- The length of the side opposite the wall is twice the length of each of the other two fenced sides
  (h2 : w + w + l = total_fence_length) -- The total length of the fence is 40 feet
  (h3 : total_fence_length = 40) -- total fence length of 40 feet
: (w * l) = 200 := -- The area of the rectangular region is 200 square feet
sorry

end rectangular_region_area_l181_181826


namespace word_limit_correct_l181_181727

-- Definition for the conditions
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650
def exceeded_amount : ℕ := 100

-- The total words written
def total_words : ℕ := saturday_words + sunday_words

-- The word limit which we need to prove
def word_limit : ℕ := total_words - exceeded_amount

theorem word_limit_correct : word_limit = 1000 := by
  unfold word_limit total_words saturday_words sunday_words exceeded_amount
  sorry

end word_limit_correct_l181_181727


namespace speed_of_stream_l181_181190

-- Definitions based on the conditions provided
def speed_still_water : ℝ := 15
def upstream_time_ratio := 2

-- Proof statement
theorem speed_of_stream (v : ℝ) 
  (h1 : ∀ d t_up t_down, (15 - v) * t_up = d ∧ (15 + v) * t_down = d ∧ t_up = upstream_time_ratio * t_down) : 
  v = 5 :=
sorry

end speed_of_stream_l181_181190


namespace power_of_i_l181_181195

theorem power_of_i (i : ℂ) 
  (h1: i^1 = i) 
  (h2: i^2 = -1) 
  (h3: i^3 = -i) 
  (h4: i^4 = 1)
  (h5: i^5 = i) 
  : i^2016 = 1 :=
by {
  sorry
}

end power_of_i_l181_181195


namespace simple_interest_rate_l181_181285

-- Define the conditions
def S : ℚ := 2500
def P : ℚ := 5000
def T : ℚ := 5

-- Define the proof problem
theorem simple_interest_rate (R : ℚ) (h : S = P * R * T / 100) : R = 10 := by
  sorry

end simple_interest_rate_l181_181285


namespace smallest_k_satisfies_l181_181353

noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

theorem smallest_k_satisfies (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (sqrt (x * y)) + (1 / 2) * (sqrt (abs (x - y))) ≥ (x + y) / 2 :=
by
  sorry

end smallest_k_satisfies_l181_181353


namespace cost_price_of_computer_table_l181_181461

variable (C : ℝ) (SP : ℝ)
variable (h1 : SP = 5400)
variable (h2 : SP = C * 1.32)

theorem cost_price_of_computer_table : C = 5400 / 1.32 :=
by
  -- We are required to prove C = 5400 / 1.32
  sorry

end cost_price_of_computer_table_l181_181461


namespace car_speed_travel_l181_181405

theorem car_speed_travel (v : ℝ) :
  600 = 3600 / 6 ∧
  (6 : ℝ) = (3600 / v) + 2 →
  v = 900 :=
by
  sorry

end car_speed_travel_l181_181405


namespace total_people_in_church_l181_181274

def c : ℕ := 80
def m : ℕ := 60
def f : ℕ := 60

theorem total_people_in_church : c + m + f = 200 :=
by
  sorry

end total_people_in_church_l181_181274


namespace fifth_term_is_67_l181_181236

noncomputable def satisfies_sequence (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :=
  (a = 3) ∧ (d = 27) ∧ 
  (a = (1/3 : ℚ) * (3 + b)) ∧
  (b = (1/3 : ℚ) * (a + 27)) ∧
  (27 = (1/3 : ℚ) * (b + e))

theorem fifth_term_is_67 :
  ∃ (e : ℕ), satisfies_sequence 3 a b 27 e ∧ e = 67 :=
sorry

end fifth_term_is_67_l181_181236


namespace evaluate_expression_l181_181837

theorem evaluate_expression :
  (2 * 10^3)^3 = 8 * 10^9 :=
by
  sorry

end evaluate_expression_l181_181837


namespace students_total_l181_181432

def num_girls : ℕ := 11
def num_boys : ℕ := num_girls + 5

theorem students_total : num_girls + num_boys = 27 := by
  sorry

end students_total_l181_181432


namespace probability_red_second_draw_l181_181338

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_red_second_draw_l181_181338


namespace max_markers_with_20_dollars_l181_181976

theorem max_markers_with_20_dollars (single_marker_cost : ℕ) (four_pack_cost : ℕ) (eight_pack_cost : ℕ) :
  single_marker_cost = 2 → four_pack_cost = 6 → eight_pack_cost = 10 → (∃ n, n = 16) := by
    intros h1 h2 h3
    existsi 16
    sorry

end max_markers_with_20_dollars_l181_181976


namespace alpha_range_in_first_quadrant_l181_181460

open Real

theorem alpha_range_in_first_quadrant (k : ℤ) (α : ℝ) 
  (h1 : cos α ≤ sin α) : 
  (2 * k * π + π / 4) ≤ α ∧ α < (2 * k * π + π / 2) :=
sorry

end alpha_range_in_first_quadrant_l181_181460


namespace sum_of_coefficients_l181_181940

def polynomial (x y : ℕ) : ℕ := (x^2 - 3*x*y + y^2)^8

theorem sum_of_coefficients : polynomial 1 1 = 1 :=
sorry

end sum_of_coefficients_l181_181940


namespace expression_evaluation_l181_181995

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by
  sorry

end expression_evaluation_l181_181995


namespace find_x_l181_181817

def Hiram_age := 40
def Allyson_age := 28
def Twice_Allyson_age := 2 * Allyson_age
def Four_less_than_twice_Allyson_age := Twice_Allyson_age - 4

theorem find_x (x : ℤ) : Hiram_age + x = Four_less_than_twice_Allyson_age → x = 12 := 
by
  intros h -- introducing the assumption 
  sorry

end find_x_l181_181817


namespace parabola_circle_intersection_radius_squared_l181_181093

theorem parabola_circle_intersection_radius_squared :
  (∀ x y, y = (x - 2)^2 → x + 1 = (y + 2)^2 → (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end parabola_circle_intersection_radius_squared_l181_181093


namespace number_of_ways_to_choose_a_pair_of_socks_l181_181470

-- Define the number of socks of each color
def white_socks := 5
def brown_socks := 5
def blue_socks := 5
def green_socks := 5

-- Define the total number of socks
def total_socks := white_socks + brown_socks + blue_socks + green_socks

-- Define the number of ways to choose 2 blue socks from 5 blue socks
def num_ways_choose_two_blue_socks : ℕ := Nat.choose blue_socks 2

-- The proof statement
theorem number_of_ways_to_choose_a_pair_of_socks :
  num_ways_choose_two_blue_socks = 10 :=
sorry

end number_of_ways_to_choose_a_pair_of_socks_l181_181470


namespace ratio_of_bubbles_l181_181981

def bubbles_dawn_per_ounce : ℕ := 200000

def mixture_bubbles (bubbles_other_per_ounce : ℕ) : ℕ :=
  let half_ounce_dawn := bubbles_dawn_per_ounce / 2
  let half_ounce_other := bubbles_other_per_ounce / 2
  half_ounce_dawn + half_ounce_other

noncomputable def find_ratio (bubbles_other_per_ounce : ℕ) : ℚ :=
  (bubbles_other_per_ounce : ℚ) / bubbles_dawn_per_ounce

theorem ratio_of_bubbles
  (bubbles_other_per_ounce : ℕ)
  (h_mixture : mixture_bubbles bubbles_other_per_ounce = 150000) :
  find_ratio bubbles_other_per_ounce = 1 / 2 :=
by
  sorry

end ratio_of_bubbles_l181_181981


namespace friends_attended_birthday_l181_181293

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l181_181293


namespace books_added_l181_181765

theorem books_added (initial_books sold_books current_books added_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : sold_books = 3)
  (h3 : current_books = 11)
  (h4 : added_books = current_books - (initial_books - sold_books)) :
  added_books = 10 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end books_added_l181_181765


namespace train_speed_l181_181912

theorem train_speed (D T : ℝ) (h1 : D = 160) (h2 : T = 16) : D / T = 10 :=
by 
  -- given D = 160 and T = 16, we need to prove D / T = 10
  sorry

end train_speed_l181_181912


namespace number_of_pencils_is_11_l181_181022

noncomputable def numberOfPencils (A B : ℕ) :  ℕ :=
  2 * A + 1 * B

theorem number_of_pencils_is_11 (A B : ℕ) (h1 : A + 2 * B = 16) (h2 : A + B = 9) : numberOfPencils A B = 11 :=
  sorry

end number_of_pencils_is_11_l181_181022


namespace flat_fee_l181_181566

theorem flat_fee (f n : ℝ) (h1 : f + 3 * n = 215) (h2 : f + 6 * n = 385) : f = 45 :=
  sorry

end flat_fee_l181_181566


namespace bridge_length_proof_l181_181586

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 49.9960003199744
noncomputable def train_speed_kmph : ℝ := 18
noncomputable def conversion_factor : ℝ := 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * conversion_factor
noncomputable def total_distance : ℝ := train_speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_proof : bridge_length = 149.980001599872 := 
by 
  sorry

end bridge_length_proof_l181_181586


namespace linear_coefficient_of_quadratic_term_is_negative_five_l181_181239

theorem linear_coefficient_of_quadratic_term_is_negative_five (a b c : ℝ) (x : ℝ) :
  (2 * x^2 = 5 * x - 3) →
  (a = 2) →
  (b = -5) →
  (c = 3) →
  (a * x^2 + b * x + c = 0) :=
by
  sorry

end linear_coefficient_of_quadratic_term_is_negative_five_l181_181239


namespace range_of_b_div_c_l181_181588

theorem range_of_b_div_c (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : b^2 = c^2 + a * c) :
  1 < b / c ∧ b / c < 2 := 
sorry

end range_of_b_div_c_l181_181588


namespace find_number_l181_181017

theorem find_number (x : ℝ) (h : 0.8 * x = (4/5 : ℝ) * 25 + 16) : x = 45 :=
by
  sorry

end find_number_l181_181017


namespace an_geometric_l181_181400

-- Define the functions and conditions
def f (x : ℝ) (b : ℝ) : ℝ := b * x + 1

def g (n : ℕ) (b : ℝ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => f (g n b) b

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ :=
  g (n + 1) b - g n b

-- Prove that a_n is a geometric sequence
theorem an_geometric (b : ℝ) (h : b ≠ 1) : 
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) b = q * a n b :=
sorry

end an_geometric_l181_181400


namespace part1_part2_l181_181451

variable (a b : ℝ)

theorem part1 : ((-a)^2 * (a^2)^2 / a^3) = a^3 := sorry

theorem part2 : (a + b) * (a - b) - (a - b)^2 = 2 * a * b - 2 * b^2 := sorry

end part1_part2_l181_181451


namespace find_x_eq_e_l181_181532

noncomputable def f (x : ℝ) : ℝ := x + x * (Real.log x) ^ 2

noncomputable def f' (x : ℝ) : ℝ :=
  1 + (Real.log x) ^ 2 + 2 * Real.log x

theorem find_x_eq_e : ∃ (x : ℝ), (x * f' x = 2 * f x) ∧ (x = Real.exp 1) :=
by
  sorry

end find_x_eq_e_l181_181532


namespace triangle_BC_range_l181_181969

open Real

variable {a C : ℝ} (A : ℝ) (ABC : Triangle A C)

/-- Proof problem statement -/
theorem triangle_BC_range (A C : ℝ) (h0 : 0 < A) (h1 : A < π) (c : ℝ) (h2 : c = sqrt 2) (h3 : a * cos C = c * sin A): 
  ∃ (BC : ℝ), sqrt 2 < BC ∧ BC < 2 :=
sorry

end triangle_BC_range_l181_181969


namespace avg_speed_BC_60_mph_l181_181850

theorem avg_speed_BC_60_mph 
  (d_AB : ℕ) (d_BC : ℕ) (avg_speed_total : ℚ) (time_ratio : ℚ) (t_AB : ℕ) :
  d_AB = 120 ∧ d_BC = 60 ∧ avg_speed_total = 45 ∧ time_ratio = 3 ∧
  t_AB = 3 → (d_BC / (t_AB / time_ratio) = 60) :=
by
  sorry

end avg_speed_BC_60_mph_l181_181850


namespace positive_integer_solutions_of_inequality_l181_181791

theorem positive_integer_solutions_of_inequality :
  {x : ℕ | 2 * (x - 1) < 7 - x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_l181_181791


namespace value_of_r_l181_181069

theorem value_of_r (n : ℕ) (h : n = 3) : 
  let s := 2^n - 1
  let r := 4^s - s
  r = 16377 := by
  let s := 2^3 - 1
  let r := 4^s - s
  sorry

end value_of_r_l181_181069


namespace olympic_medals_l181_181160

theorem olympic_medals (total_sprinters british_sprinters non_british_sprinters ways_case1 ways_case2 ways_case3 : ℕ)
  (h_total : total_sprinters = 10)
  (h_british : british_sprinters = 4)
  (h_non_british : non_british_sprinters = 6)
  (h_case1 : ways_case1 = 6 * 5 * 4)
  (h_case2 : ways_case2 = 4 * 3 * (6 * 5))
  (h_case3 : ways_case3 = (4 * 3) * (3 * 2) * 6) :
  ways_case1 + ways_case2 + ways_case3 = 912 := by
  sorry

end olympic_medals_l181_181160


namespace arithmetic_sequence_problem_l181_181105

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + (n - 1) * d

-- Given condition
def given_condition (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
3 * a 9 - a 15 - a 3 = 20

-- Question to prove
def question (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
2 * a 8 - a 7 = 20

-- Main theorem
theorem arithmetic_sequence_problem (a: ℕ → ℝ) (a1 d: ℝ):
  arithmetic_sequence a a1 d →
  given_condition a a1 d →
  question a a1 d :=
by
  sorry

end arithmetic_sequence_problem_l181_181105


namespace factorize_expression_l181_181742

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by sorry

end factorize_expression_l181_181742


namespace chimney_bricks_l181_181876

variable (h : ℕ)

/-- Brenda would take 8 hours to build a chimney alone. 
    Brandon would take 12 hours to build it alone. 
    When they work together, their efficiency is diminished by 15 bricks per hour due to their chatting. 
    If they complete the chimney in 6 hours when working together, then the total number of bricks in the chimney is 360. -/
theorem chimney_bricks
  (h : ℕ)
  (Brenda_rate : ℕ)
  (Brandon_rate : ℕ)
  (effective_rate : ℕ)
  (completion_time : ℕ)
  (h_eq : Brenda_rate = h / 8)
  (h_eq_alt : Brandon_rate = h / 12)
  (effective_rate_eq : effective_rate = (Brenda_rate + Brandon_rate) - 15)
  (completion_eq : 6 * effective_rate = h) :
  h = 360 := by 
  sorry

end chimney_bricks_l181_181876


namespace total_time_assignment_l181_181644

-- Define the time taken for each part
def time_first_part : ℕ := 25
def time_second_part : ℕ := 2 * time_first_part
def time_third_part : ℕ := 45

-- Define the total time taken for the assignment
def total_time : ℕ := time_first_part + time_second_part + time_third_part

-- The theorem stating that the total time is 120 minutes
theorem total_time_assignment : total_time = 120 := by
  sorry

end total_time_assignment_l181_181644


namespace nina_money_l181_181661

variable (C : ℝ)

theorem nina_money (h1: 6 * C = 8 * (C - 1.15)) : 6 * C = 27.6 := by
  have h2: C = 4.6 := sorry
  rw [h2]
  norm_num
  done

end nina_money_l181_181661


namespace find_table_price_l181_181369

noncomputable def chair_price (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
noncomputable def chair_table_sum (C T : ℝ) : Prop := C + T = 64

theorem find_table_price (C T : ℝ) (h1 : chair_price C T) (h2 : chair_table_sum C T) : T = 56 :=
by sorry

end find_table_price_l181_181369


namespace problem_A_problem_B_problem_C_problem_D_l181_181053

theorem problem_A (a b: ℝ) (h : b > 0 ∧ a > b) : ¬(1/a > 1/b) := 
by {
  sorry
}

theorem problem_B (a b: ℝ) (h : a < b ∧ b < 0): (a^2 > a*b) := 
by {
  sorry
}

theorem problem_C (a b: ℝ) (h : a > b): ¬(|a| > |b|) := 
by {
  sorry
}

theorem problem_D (a: ℝ) (h : a > 2): (a + 4/(a-2) ≥ 6) := 
by {
  sorry
}

end problem_A_problem_B_problem_C_problem_D_l181_181053


namespace find_m_l181_181230

-- Definitions of the given vectors and their properties
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Condition that vectors a and b are parallel
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 = 0

-- Goal: Find the value of m such that vectors a and b are parallel
theorem find_m (m : ℝ) : 
  are_parallel a (b m) → m = 6 :=
by
  sorry

end find_m_l181_181230


namespace split_tips_evenly_l181_181666

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l181_181666


namespace father_l181_181240

theorem father's_age : 
  ∀ (M F : ℕ), 
  (M = (2 : ℚ) / 5 * F) → 
  (M + 10 = (1 : ℚ) / 2 * (F + 10)) → 
  F = 50 :=
by
  intros M F h1 h2
  sorry

end father_l181_181240


namespace inequality_add_l181_181729

theorem inequality_add {a b c : ℝ} (h : a > b) : a + c > b + c :=
sorry

end inequality_add_l181_181729


namespace always_true_inequality_l181_181683

theorem always_true_inequality (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by
  sorry

end always_true_inequality_l181_181683


namespace kennedy_lost_pawns_l181_181217

-- Definitions based on conditions
def initial_pawns_per_player := 8
def total_pawns := 2 * initial_pawns_per_player -- Total pawns in the game initially
def pawns_lost_by_Riley := 1 -- Riley lost 1 pawn
def pawns_remaining := 11 -- 11 pawns left in the game

-- Translations of conditions to Lean
theorem kennedy_lost_pawns : 
  initial_pawns_per_player - (pawns_remaining - (initial_pawns_per_player - pawns_lost_by_Riley)) = 4 := 
by 
  sorry

end kennedy_lost_pawns_l181_181217


namespace Nancy_folders_l181_181320

def n_initial : ℕ := 43
def n_deleted : ℕ := 31
def n_per_folder : ℕ := 6
def n_folders : ℕ := (n_initial - n_deleted) / n_per_folder

theorem Nancy_folders : n_folders = 2 := by
  sorry

end Nancy_folders_l181_181320


namespace triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l181_181977

-- Define the properties and variables of the given obtuse triangle
variables (a b c : ℝ) (A C : ℝ)
-- Given conditions
axiom ha : a = 7
axiom hb : b = 3
axiom hcosC : Real.cos C = 11 / 14

-- Prove the values of c and angle A
theorem triangle_ABC_c_and_A_value (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : c = 5 ∧ A = 2 * Real.pi / 3 :=
sorry

-- Prove the value of sin(2C - π / 6)
theorem sin_2C_minus_pi_6 (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : Real.sin (2 * C - Real.pi / 6) = 71 / 98 :=
sorry

end triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l181_181977


namespace seats_per_bus_correct_l181_181187

-- Define the conditions given in the problem
def students : ℕ := 28
def buses : ℕ := 4

-- Define the number of seats per bus
def seats_per_bus : ℕ := students / buses

-- State the theorem that proves the number of seats per bus
theorem seats_per_bus_correct : seats_per_bus = 7 := by
  -- conditions are used as definitions, the goal is to prove seats_per_bus == 7
  sorry

end seats_per_bus_correct_l181_181187


namespace sum_of_possible_values_l181_181745

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 5) :
  ∃ s : ℝ, s = (x - 2) * (y - 2) ∧ (s = -3 ∨ s = 9) :=
sorry

end sum_of_possible_values_l181_181745


namespace speed_of_second_train_correct_l181_181313

noncomputable def length_first_train : ℝ := 140 -- in meters
noncomputable def length_second_train : ℝ := 160 -- in meters
noncomputable def time_to_cross : ℝ := 10.799136069114471 -- in seconds
noncomputable def speed_first_train : ℝ := 60 -- in km/hr
noncomputable def speed_second_train : ℝ := 40 -- in km/hr

theorem speed_of_second_train_correct :
  (length_first_train + length_second_train)/time_to_cross - (speed_first_train * (5/18)) = speed_second_train * (5/18) :=
by
  sorry

end speed_of_second_train_correct_l181_181313


namespace trapezoid_base_ratio_l181_181689

-- Define the context of the problem
variables (AB CD : ℝ) (h : AB < CD)

-- Define the main theorem to be proved
theorem trapezoid_base_ratio (h : AB / CD = 1 / 2) :
  ∃ (E F G H I J : ℝ), 
    EJ - EI = FI - FH / 5 ∧ -- These points create segments that divide equally as per the conditions 
    FI - FH = GH / 5 ∧
    GH - GI = HI / 5 ∧
    HI - HJ = JI / 5 ∧
    JI - JE = EJ / 5 :=
sorry

end trapezoid_base_ratio_l181_181689


namespace abs_condition_l181_181052

theorem abs_condition (x : ℝ) : |2 * x - 7| ≤ 0 ↔ x = 7 / 2 := 
by
  sorry

end abs_condition_l181_181052


namespace friends_bought_color_box_l181_181648

variable (total_pencils : ℕ) (pencils_per_box : ℕ) (chloe_pencils : ℕ)

theorem friends_bought_color_box : 
  (total_pencils = 42) → 
  (pencils_per_box = 7) → 
  (chloe_pencils = pencils_per_box) → 
  (total_pencils - chloe_pencils) / pencils_per_box = 5 := 
by 
  intros ht hb hc
  sorry

end friends_bought_color_box_l181_181648


namespace num_valid_arrangements_l181_181535

-- Define the people and the days of the week
inductive Person := | A | B | C | D | E
inductive DayOfWeek := | Monday | Tuesday | Wednesday | Thursday | Friday

-- Define the arrangement function type
def Arrangement := DayOfWeek → Person

/-- The total number of valid arrangements for 5 people
    (A, B, C, D, E) on duty from Monday to Friday such that:
    - A and B are not on duty on adjacent days,
    - B and C are on duty on adjacent days,
    is 36.
-/
theorem num_valid_arrangements : 
  ∃ (arrangements : Finset (Arrangement)), arrangements.card = 36 ∧
  (∀ (x : Arrangement), x ∈ arrangements →
    (∀ (d1 d2 : DayOfWeek), 
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) →
      ¬(x d1 = Person.A ∧ x d2 = Person.B)) ∧
    (∃ (d1 d2 : DayOfWeek),
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) ∧
      (x d1 = Person.B ∧ x d2 = Person.C)))
  := sorry

end num_valid_arrangements_l181_181535


namespace square_sides_product_l181_181719

theorem square_sides_product (a : ℝ) : 
  (∃ s : ℝ, s = 5 ∧ (a = -3 + s ∨ a = -3 - s)) → (a = 2 ∨ a = -8) → -8 * 2 = -16 :=
by
  intro _ _
  exact rfl

end square_sides_product_l181_181719


namespace total_beetles_eaten_each_day_l181_181436

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end total_beetles_eaten_each_day_l181_181436


namespace torn_out_sheets_count_l181_181522

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l181_181522


namespace set_of_points_l181_181330

theorem set_of_points (x y : ℝ) (h : x^2 * y - y ≥ 0) :
  (y ≥ 0 ∧ |x| ≥ 1) ∨ (y ≤ 0 ∧ |x| ≤ 1) :=
sorry

end set_of_points_l181_181330


namespace evaluate_g_neg5_l181_181013

def g (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_g_neg5 : g (-5) = -22 := 
  by sorry

end evaluate_g_neg5_l181_181013


namespace division_addition_correct_l181_181902

-- Define a function that performs the arithmetic operations described
def calculateResult : ℕ :=
  let division := 12 * 4 -- dividing 12 by 1/4 is the same as multiplying by 4
  division + 5 -- then add 5 to the result

-- The theorem statement to prove
theorem division_addition_correct : calculateResult = 53 := by
  sorry

end division_addition_correct_l181_181902


namespace sum_of_triangles_l181_181001

def triangle (a b c : ℕ) : ℕ :=
  (a * b) + c

theorem sum_of_triangles : 
  triangle 4 2 3 + triangle 5 3 2 = 28 :=
by
  sorry

end sum_of_triangles_l181_181001


namespace swapped_two_digit_number_l181_181456

variable (a : ℕ)

theorem swapped_two_digit_number (h : a < 10) (sum_digits : ∃ t : ℕ, t + a = 13) : 
    ∃ n : ℕ, n = 9 * a + 13 :=
by
  sorry

end swapped_two_digit_number_l181_181456


namespace divisibility_by_100_l181_181956

theorem divisibility_by_100 (n : ℕ) (k : ℕ) (h : n = 5 * k + 2) :
    100 ∣ (5^n + 12*n^2 + 12*n + 3) :=
sorry

end divisibility_by_100_l181_181956


namespace find_total_roses_l181_181919

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l181_181919


namespace buttons_ratio_l181_181498

theorem buttons_ratio
  (initial_buttons : ℕ)
  (shane_multiplier : ℕ)
  (final_buttons : ℕ)
  (total_buttons_after_shane : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  total_buttons_after_shane = initial_buttons + shane_multiplier * initial_buttons →
  (total_buttons_after_shane - final_buttons) / total_buttons_after_shane = 1 / 2 :=
by
  intros
  sorry

end buttons_ratio_l181_181498


namespace exists_non_degenerate_triangle_l181_181642

theorem exists_non_degenerate_triangle
  (l : Fin 7 → ℝ)
  (h_ordered : ∀ i j, i ≤ j → l i ≤ l j)
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) :
  ∃ i j k : Fin 7, i < j ∧ j < k ∧ l i + l j > l k ∧ l j + l k > l i ∧ l k + l i > l j := 
sorry

end exists_non_degenerate_triangle_l181_181642


namespace tank_capacity_l181_181198

variable (c : ℕ) -- Total capacity of the tank in liters.
variable (w_0 : ℕ := c / 3) -- Initial volume of water in the tank in liters.

theorem tank_capacity (h1 : w_0 = c / 3) (h2 : (w_0 + 5) / c = 2 / 5) : c = 75 :=
by
  -- Proof steps would be here.
  sorry

end tank_capacity_l181_181198


namespace successful_experimental_operation_l181_181523

/-- Problem statement:
Given the following biological experimental operations:
1. spreading diluted E. coli culture on solid medium,
2. introducing sterile air into freshly inoculated grape juice with yeast,
3. inoculating soil leachate on beef extract peptone medium,
4. using slightly opened rose flowers as experimental material for anther culture.

Prove that spreading diluted E. coli culture on solid medium can successfully achieve the experimental objective of obtaining single colonies.
-/
theorem successful_experimental_operation :
  ∃ objective_result,
    (objective_result = "single_colonies" →
     let operation_A := "spreading diluted E. coli culture on solid medium"
     let operation_B := "introducing sterile air into freshly inoculated grape juice with yeast"
     let operation_C := "inoculating soil leachate on beef extract peptone medium"
     let operation_D := "slightly opened rose flowers as experimental material for anther culture"
     ∃ successful_operation,
       successful_operation = operation_A
       ∧ (successful_operation = operation_A → objective_result = "single_colonies")
       ∧ (successful_operation = operation_B → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_C → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_D → objective_result ≠ "single_colonies")) :=
sorry

end successful_experimental_operation_l181_181523


namespace find_y_l181_181708

theorem find_y (y : ℝ) : 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := 
by
  sorry

end find_y_l181_181708


namespace polynomial_root_l181_181959

theorem polynomial_root (x0 : ℝ) (z : ℝ) 
  (h1 : x0^3 - x0 - 1 = 0) 
  (h2 : z = x0^2 + 3 * x0 + 1) : 
  z^3 - 5 * z^2 - 10 * z - 11 = 0 := 
sorry

end polynomial_root_l181_181959


namespace minimum_value_of_quadratic_function_l181_181874

def quadratic_function (a x : ℝ) : ℝ :=
  4 * x ^ 2 - 4 * a * x + (a ^ 2 - 2 * a + 2)

def min_value_in_interval (f : ℝ → ℝ) (a : ℝ) (interval : Set ℝ) (min_val : ℝ) : Prop :=
  ∀ x ∈ interval, f x ≥ min_val ∧ ∃ y ∈ interval, f y = min_val

theorem minimum_value_of_quadratic_function :
  ∃ a : ℝ, min_value_in_interval (quadratic_function a) a {x | 0 ≤ x ∧ x ≤ 1} 2 ↔ (a = 0 ∨ a = 3 + Real.sqrt 5) :=
by
  sorry

end minimum_value_of_quadratic_function_l181_181874


namespace least_N_l181_181170

theorem least_N :
  ∃ N : ℕ, 
    (N % 2 = 1) ∧ 
    (N % 3 = 2) ∧ 
    (N % 5 = 3) ∧ 
    (N % 7 = 4) ∧ 
    (∀ M : ℕ, 
      (M % 2 = 1) ∧ 
      (M % 3 = 2) ∧ 
      (M % 5 = 3) ∧ 
      (M % 7 = 4) → 
      N ≤ M) :=
  sorry

end least_N_l181_181170


namespace peaches_eaten_l181_181009

theorem peaches_eaten (P B Baskets P_each R Boxes P_box : ℕ) 
  (h1 : B = 5) 
  (h2 : P_each = 25)
  (h3 : Baskets = B * P_each)
  (h4 : R = 8) 
  (h5 : P_box = 15)
  (h6 : Boxes = R * P_box)
  (h7 : P = Baskets - Boxes) : P = 5 :=
by sorry

end peaches_eaten_l181_181009


namespace juicy_12_juicy_20_l181_181351

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 1 = (1 / a) + (1 / b) + (1 / c) + (1 / d) ∧ a * b * c * d = n

theorem juicy_12 : is_juicy 12 :=
sorry

theorem juicy_20 : is_juicy 20 :=
sorry

end juicy_12_juicy_20_l181_181351


namespace John_writing_years_l181_181569

def books_written (total_earnings per_book_earning : ℕ) : ℕ :=
  total_earnings / per_book_earning

def books_per_year (months_in_year months_per_book : ℕ) : ℕ :=
  months_in_year / months_per_book

def years_writing (total_books books_per_year : ℕ) : ℕ :=
  total_books / books_per_year

theorem John_writing_years :
  let total_earnings := 3600000
  let per_book_earning := 30000
  let months_in_year := 12
  let months_per_book := 2
  let total_books := books_written total_earnings per_book_earning
  let books_per_year := books_per_year months_in_year months_per_book
  years_writing total_books books_per_year = 20 := by
sorry

end John_writing_years_l181_181569


namespace sum_of_distinct_digits_base6_l181_181538

theorem sum_of_distinct_digits_base6 (A B C : ℕ) (hA : A < 6) (hB : B < 6) (hC : C < 6) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_first_col : C + C % 6 = 4)
  (h_second_col : B + B % 6 = C)
  (h_third_col : A + B % 6 = A) :
  A + B + C = 6 := by
  sorry

end sum_of_distinct_digits_base6_l181_181538


namespace integer_solutions_l181_181697

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end integer_solutions_l181_181697


namespace monotonicity_of_f_sum_of_squares_of_roots_l181_181687

noncomputable def f (x a : Real) : Real := Real.log x - a * x^2

theorem monotonicity_of_f (a : Real) :
  (a ≤ 0 → ∀ x y : Real, 0 < x → x < y → f x a < f y a) ∧
  (a > 0 → ∀ x y : Real, 0 < x → x < Real.sqrt (1/(2 * a)) → Real.sqrt (1/(2 * a)) < y → f x a < f (Real.sqrt (1/(2 * a))) a ∧ f (Real.sqrt (1/(2 * a))) a > f y a) :=
by sorry

theorem sum_of_squares_of_roots (a x1 x2 : Real) (h1 : f x1 a = 0) (h2 : f x2 a = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 > 2 * Real.exp 1 :=
by sorry

end monotonicity_of_f_sum_of_squares_of_roots_l181_181687


namespace sophist_statements_correct_l181_181253

-- Definitions based on conditions
def num_knights : ℕ := 40
def num_liars : ℕ := 25

-- Statements made by the sophist
def sophist_statement1 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_knights = 40
def sophist_statement2 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_liars + 1 = 26

-- Theorem to be proved
theorem sophist_statements_correct :
  sophist_statement1 ∧ sophist_statement2 :=
by
  -- Placeholder for the actual proof
  sorry

end sophist_statements_correct_l181_181253


namespace David_squats_l181_181646

theorem David_squats (h1: ∀ d z: ℕ, d = 3 * 58) : d = 174 :=
by
  sorry

end David_squats_l181_181646


namespace sum_of_first_n_terms_l181_181429

-- Define the sequence aₙ
def a (n : ℕ) : ℕ := 2 * n - 1

-- Prove that the sum of the first n terms of the sequence is n²
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range (n+1)).sum a = n^2 :=
by sorry -- Proof is skipped

end sum_of_first_n_terms_l181_181429


namespace remainder_of_452867_div_9_l181_181183

theorem remainder_of_452867_div_9 : (452867 % 9) = 5 := by
  sorry

end remainder_of_452867_div_9_l181_181183


namespace max_in_circle_eqn_l181_181435

theorem max_in_circle_eqn : 
  ∀ (x y : ℝ), (x ≥ 0) → (y ≥ 0) → (4 * x + 3 * y ≤ 12) → (x - 1)^2 + (y - 1)^2 = 1 :=
by
  intros x y hx hy hineq
  sorry

end max_in_circle_eqn_l181_181435


namespace first_proof_l181_181870

def triangular (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def covers_all_columns (k : ℕ) : Prop :=
  ∀ c : ℕ, (c < 10) → (∃ m : ℕ, m ≤ k ∧ (triangular m) % 10 = c)

theorem first_proof (k : ℕ) (h : covers_all_columns 28) : 
  triangular k = 435 :=
sorry

end first_proof_l181_181870


namespace circle_equation_l181_181743

-- Definitions of the conditions
def passes_through (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (r : ℝ) : Prop :=
  (c - a) ^ 2 + (d - b) ^ 2 = r ^ 2

def center_on_line (a : ℝ) (b : ℝ) : Prop :=
  a - b - 4 = 0

-- Statement of the problem to be proved
theorem circle_equation 
  (a b r : ℝ) 
  (h1 : passes_through a b (-1) (-4) r)
  (h2 : passes_through a b 6 3 r)
  (h3 : center_on_line a b) :
  -- Equation of the circle
  (a = 3 ∧ b = -1 ∧ r = 5) → ∀ x y : ℝ, 
    (x - 3)^2 + (y + 1)^2 = 25 :=
sorry

end circle_equation_l181_181743


namespace solve_for_star_l181_181410

theorem solve_for_star 
  (x : ℝ) 
  (h : 45 - (28 - (37 - (15 - x))) = 58) : 
  x = 19 :=
by
  -- Proof goes here. Currently incomplete, so we use sorry.
  sorry

end solve_for_star_l181_181410


namespace area_of_triangle_from_squares_l181_181555

theorem area_of_triangle_from_squares :
  ∃ (a b c : ℕ), (a = 15 ∧ b = 15 ∧ c = 6 ∧ (1/2 : ℚ) * a * c = 45) :=
by
  let a := 15
  let b := 15
  let c := 6
  have h1 : (1/2 : ℚ) * a * c = 45 := sorry
  exact ⟨a, b, c, ⟨rfl, rfl, rfl, h1⟩⟩

end area_of_triangle_from_squares_l181_181555


namespace value_of_f_2012_1_l181_181072

noncomputable def f : ℝ → ℝ :=
sorry

-- Condition 1: f is even
axiom even_f : ∀ x : ℝ, f x = f (-x)

-- Condition 2: f(x + 3) = -f(x)
axiom periodicity_f : ∀ x : ℝ, f (x + 3) = -f x

-- Condition 3: f(x) = 2x + 3 for -3 ≤ x ≤ 0
axiom defined_f_on_interval : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x = 2 * x + 3

-- Assertion to prove
theorem value_of_f_2012_1 : f 2012.1 = -1.2 :=
by sorry

end value_of_f_2012_1_l181_181072


namespace sequence_general_formula_l181_181954

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- because sequences in the solution are 1-indexed.
  | 1 => 2
  | k+2 => sequence (k+1) + 3 * (k+1)

theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : 
  sequence n = 2 + 3 * n * (n - 1) / 2 :=
by
  sorry

#eval sequence 1  -- should output 2
#eval sequence 2  -- should output 5
#eval sequence 3  -- should output 11
#eval sequence 4  -- should output 20
#eval sequence 5  -- should output 32
#eval sequence 6  -- should output 47

end sequence_general_formula_l181_181954


namespace shortest_side_of_similar_triangle_l181_181003

theorem shortest_side_of_similar_triangle (a1 a2 h1 h2 : ℝ)
  (h1_eq : a1 = 24)
  (h2_eq : h1 = 37)
  (h2_eq' : h2 = 74)
  (h_similar : h2 / h1 = 2)
  (h_a2_eq : a2 = 2 * Real.sqrt 793):
  a2 = 2 * Real.sqrt 793 := by
  sorry

end shortest_side_of_similar_triangle_l181_181003


namespace james_profit_correct_l181_181725

noncomputable def jamesProfit : ℝ :=
  let tickets_bought := 200
  let cost_per_ticket := 2
  let winning_ticket_percentage := 0.20
  let percentage_one_dollar := 0.50
  let percentage_three_dollars := 0.30
  let percentage_four_dollars := 0.20
  let percentage_five_dollars := 0.80
  let grand_prize_ticket_count := 1
  let average_remaining_winner := 15
  let tax_percentage := 0.10
  let total_cost := tickets_bought * cost_per_ticket
  let winning_tickets := tickets_bought * winning_ticket_percentage
  let tickets_five_dollars := winning_tickets * percentage_five_dollars
  let other_winning_tickets := winning_tickets - tickets_five_dollars - grand_prize_ticket_count
  let total_winnings_before_tax := (tickets_five_dollars * 5) + (grand_prize_ticket_count * 5000) + (other_winning_tickets * average_remaining_winner)
  let total_tax := total_winnings_before_tax * tax_percentage
  let total_winnings_after_tax := total_winnings_before_tax - total_tax
  total_winnings_after_tax - total_cost

theorem james_profit_correct : jamesProfit = 4338.50 := by
  sorry

end james_profit_correct_l181_181725


namespace value_of_r_minus_p_l181_181117

-- Define the arithmetic mean conditions
def arithmetic_mean1 (p q : ℝ) : Prop :=
  (p + q) / 2 = 10

def arithmetic_mean2 (q r : ℝ) : Prop :=
  (q + r) / 2 = 27

-- Prove that r - p = 34 based on the conditions
theorem value_of_r_minus_p (p q r : ℝ)
  (h1 : arithmetic_mean1 p q)
  (h2 : arithmetic_mean2 q r) :
  r - p = 34 :=
by
  sorry

end value_of_r_minus_p_l181_181117


namespace velocity_at_3_velocity_at_4_l181_181457

-- Define the distance as a function of time
def s (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Define the velocity as the derivative of the distance
noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem velocity_at_3 : v 3 = 20 :=
by
  sorry

theorem velocity_at_4 : v 4 = 26 :=
by
  sorry

end velocity_at_3_velocity_at_4_l181_181457


namespace simplify_expression_l181_181812

theorem simplify_expression :
  (3^4 + 3^2) / (3^3 - 3) = 15 / 4 :=
by {
  sorry
}

end simplify_expression_l181_181812


namespace geom_sequence_a7_l181_181807

theorem geom_sequence_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n+1) = a n * r) 
  (h_a1 : a 1 = 8) 
  (h_a4_eq : a 4 = a 3 * a 5) : 
  a 7 = 1 / 8 :=
by
  sorry

end geom_sequence_a7_l181_181807


namespace least_number_to_add_l181_181710

theorem least_number_to_add (n : ℕ) (sum_digits : ℕ) (next_multiple : ℕ) 
  (h1 : n = 51234) 
  (h2 : sum_digits = 5 + 1 + 2 + 3 + 4) 
  (h3 : next_multiple = 18) :
  ∃ k, (k = next_multiple - sum_digits) ∧ (n + k) % 9 = 0 :=
sorry

end least_number_to_add_l181_181710


namespace same_solution_set_l181_181472

theorem same_solution_set :
  (∀ x : ℝ, (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0) :=
sorry

end same_solution_set_l181_181472


namespace diamond_4_3_l181_181731

def diamond (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem diamond_4_3 : diamond 4 3 = 1 :=
by
  -- The proof will go here.
  sorry

end diamond_4_3_l181_181731


namespace room_length_l181_181497

theorem room_length (L : ℝ) (width : ℝ := 4) (total_cost : ℝ := 20900) (rate : ℝ := 950) :
  L * width = total_cost / rate → L = 5.5 :=
by
  sorry

end room_length_l181_181497


namespace bananas_needed_to_make_yogurts_l181_181786

theorem bananas_needed_to_make_yogurts 
    (slices_per_yogurt : ℕ) 
    (slices_per_banana: ℕ) 
    (number_of_yogurts: ℕ) 
    (total_needed_slices: ℕ) 
    (bananas_needed: ℕ) 
    (h1: slices_per_yogurt = 8)
    (h2: slices_per_banana = 10)
    (h3: number_of_yogurts = 5)
    (h4: total_needed_slices = number_of_yogurts * slices_per_yogurt)
    (h5: bananas_needed = total_needed_slices / slices_per_banana): 
    bananas_needed = 4 := 
by
    sorry

end bananas_needed_to_make_yogurts_l181_181786


namespace min_moves_to_emit_all_colors_l181_181235

theorem min_moves_to_emit_all_colors :
  ∀ (colors : Fin 7 → Prop) (room : Fin 4 → Fin 7)
  (h : ∀ i j, i ≠ j → room i ≠ room j) (moves : ℕ),
  (∀ (n : ℕ) (i : Fin 4), n < moves → ∃ c : Fin 7, colors c ∧ room i = c ∧
    (∀ j, j ≠ i → room j ≠ c)) →
  (∃ n, n = 8) :=
by
  sorry

end min_moves_to_emit_all_colors_l181_181235


namespace calculate_nabla_l181_181397

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calculate_nabla : nabla (nabla 2 3) 4 = 11 / 9 :=
by
  sorry

end calculate_nabla_l181_181397


namespace goose_eggs_count_l181_181019

theorem goose_eggs_count (E : ℕ) 
  (h1 : (1/2 : ℝ) * E = E/2)
  (h2 : (3/4 : ℝ) * (E/2) = (3 * E) / 8)
  (h3 : (2/5 : ℝ) * ((3 * E) / 8) = (3 * E) / 20)
  (h4 : (3 * E) / 20 = 120) :
  E = 400 :=
sorry

end goose_eggs_count_l181_181019


namespace find_g_75_l181_181636

variable (g : ℝ → ℝ)

def prop_1 := ∀ x y : ℝ, x > 0 → y > 0 → g (x * y) = g x / y
def prop_2 := g 50 = 30

theorem find_g_75 (h1 : prop_1 g) (h2 : prop_2 g) : g 75 = 20 :=
by
  sorry

end find_g_75_l181_181636


namespace only_nonneg_solution_l181_181030

theorem only_nonneg_solution :
  ∀ (x y : ℕ), 2^x = y^2 + y + 1 → (x, y) = (0, 0) := by
  intros x y h
  sorry

end only_nonneg_solution_l181_181030


namespace remaining_oranges_l181_181424

/-- Define the conditions of the problem. -/
def oranges_needed_Michaela : ℕ := 20
def oranges_needed_Cassandra : ℕ := 2 * oranges_needed_Michaela
def total_oranges_picked : ℕ := 90

/-- State the proof problem. -/
theorem remaining_oranges : total_oranges_picked - (oranges_needed_Michaela + oranges_needed_Cassandra) = 30 := 
sorry

end remaining_oranges_l181_181424


namespace arithmetic_seq_formula_sum_first_n_terms_l181_181819

/-- Define the given arithmetic sequence an -/
def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0       => a1
| (n + 1) => arithmetic_seq a1 d n + d

variable {a3 a7 : ℤ}
variable (a3_eq : arithmetic_seq 1 2 2 = 5)
variable (a7_eq : arithmetic_seq 1 2 6 = 13)

/-- Define the sequence bn -/
def b_seq (n : ℕ) : ℚ :=
  1 / ((2 * n + 1) * (arithmetic_seq 1 2 n))

/-- Define the sum of the first n terms of the sequence bn -/
def sum_b_seq : ℕ → ℚ
| 0       => 0
| (n + 1) => sum_b_seq n + b_seq (n + 1)
          
theorem arithmetic_seq_formula:
  ∀ (n : ℕ), arithmetic_seq 1 2 n = 2 * n - 1 :=
by
  intros
  sorry

theorem sum_first_n_terms:
  ∀ (n : ℕ), sum_b_seq n = n / (2 * n + 1) :=
by
  intros
  sorry

end arithmetic_seq_formula_sum_first_n_terms_l181_181819


namespace calvin_haircut_goal_percentage_l181_181622

theorem calvin_haircut_goal_percentage :
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  (completed_haircuts / total_haircuts_needed) * 100 = 80 :=
by
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  show (completed_haircuts / total_haircuts_needed) * 100 = 80
  sorry

end calvin_haircut_goal_percentage_l181_181622


namespace find_f_2017_l181_181801

theorem find_f_2017 (f : ℕ → ℕ) (H1 : ∀ x y : ℕ, f (x * y + 1) = f x * f y - f y - x + 2) (H2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end find_f_2017_l181_181801


namespace find_C_l181_181409

theorem find_C
  (A B C : ℕ)
  (h1 : A + B + C = 1000)
  (h2 : A + C = 700)
  (h3 : B + C = 600) :
  C = 300 := by
  sorry

end find_C_l181_181409


namespace interest_rate_A_l181_181672

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end interest_rate_A_l181_181672


namespace age_of_other_replaced_man_l181_181179

theorem age_of_other_replaced_man (A B C D : ℕ) (h1 : A = 23) (h2 : ((52 + C + D) / 4 > (A + B + C + D) / 4)) :
  B < 29 := 
by
  sorry

end age_of_other_replaced_man_l181_181179


namespace car_travel_distance_l181_181043

theorem car_travel_distance 
  (v_train : ℝ) (h_train_speed : v_train = 90) 
  (v_car : ℝ) (h_car_speed : v_car = (2 / 3) * v_train) 
  (t : ℝ) (h_time : t = 0.5) :
  ∃ d : ℝ, d = v_car * t ∧ d = 30 := 
sorry

end car_travel_distance_l181_181043


namespace probability_red_or_white_correct_l181_181934

-- Define the conditions
def totalMarbles : ℕ := 30
def blueMarbles : ℕ := 5
def redMarbles : ℕ := 9
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the calculated probability
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- Verify the probability is equal to 5 / 6
theorem probability_red_or_white_correct :
  probabilityRedOrWhite = 5 / 6 := by
  sorry

end probability_red_or_white_correct_l181_181934


namespace common_measure_angle_l181_181935

theorem common_measure_angle (α β : ℝ) (m n : ℕ) (h : α = β * (m / n)) : α / m = β / n :=
by 
  sorry

end common_measure_angle_l181_181935


namespace sammy_remaining_problems_l181_181044

variable (total_problems : Nat)
variable (fraction_problems : Nat) (decimal_problems : Nat) (multiplication_problems : Nat) (division_problems : Nat)
variable (completed_fraction_problems : Nat) (completed_decimal_problems : Nat)
variable (completed_multiplication_problems : Nat) (completed_division_problems : Nat)
variable (remaining_problems : Nat)

theorem sammy_remaining_problems
  (h₁ : total_problems = 115)
  (h₂ : fraction_problems = 35)
  (h₃ : decimal_problems = 40)
  (h₄ : multiplication_problems = 20)
  (h₅ : division_problems = 20)
  (h₆ : completed_fraction_problems = 11)
  (h₇ : completed_decimal_problems = 17)
  (h₈ : completed_multiplication_problems = 9)
  (h₉ : completed_division_problems = 5)
  (h₁₀ : remaining_problems =
    fraction_problems - completed_fraction_problems +
    decimal_problems - completed_decimal_problems +
    multiplication_problems - completed_multiplication_problems +
    division_problems - completed_division_problems) :
  remaining_problems = 73 :=
  by
    -- proof to be written
    sorry

end sammy_remaining_problems_l181_181044


namespace don_travel_time_to_hospital_l181_181617

noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def time_to_travel (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem don_travel_time_to_hospital :
  let speed_mary := 60
  let speed_don := 30
  let time_mary_minutes := 15
  let time_mary_hours := time_mary_minutes / 60
  let distance := distance_traveled speed_mary time_mary_hours
  let time_don_hours := time_to_travel distance speed_don
  time_don_hours * 60 = 30 :=
by
  sorry

end don_travel_time_to_hospital_l181_181617


namespace max_log_sum_value_l181_181507

noncomputable def max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) : ℝ :=
  Real.log x + Real.log y

theorem max_log_sum_value : ∀ (x y : ℝ), x > 0 → y > 0 → x + 4 * y = 40 → max_log_sum x y sorry sorry sorry = 2 :=
by
  intro x y h1 h2 h3
  sorry

end max_log_sum_value_l181_181507


namespace ratio_copper_zinc_l181_181201

theorem ratio_copper_zinc (total_mass zinc_mass : ℕ) (h1 : total_mass = 100) (h2 : zinc_mass = 35) : 
  ∃ (copper_mass : ℕ), 
    copper_mass = total_mass - zinc_mass ∧ (copper_mass / 5, zinc_mass / 5) = (13, 7) :=
by {
  sorry
}

end ratio_copper_zinc_l181_181201


namespace total_handshakes_at_convention_l181_181212

def number_of_gremlins := 30
def number_of_imps := 20
def disagreeing_imps := 5
def specific_gremlins := 10

theorem total_handshakes_at_convention : 
  (number_of_gremlins * (number_of_gremlins - 1) / 2) +
  ((number_of_imps - disagreeing_imps) * number_of_gremlins) + 
  (disagreeing_imps * (number_of_gremlins - specific_gremlins)) = 985 :=
by 
  sorry

end total_handshakes_at_convention_l181_181212


namespace place_circle_no_overlap_l181_181289

theorem place_circle_no_overlap 
    (rect_width rect_height : ℝ) (num_squares : ℤ) (square_size square_diameter : ℝ)
    (h_rect_dims : rect_width = 20 ∧ rect_height = 25)
    (h_num_squares : num_squares = 120)
    (h_square_size : square_size = 1)
    (h_circle_diameter : square_diameter = 1) : 
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ rect_width ∧ 0 ≤ y ∧ y ≤ rect_height ∧ 
    ∀ (square_x square_y : ℝ), 
      0 ≤ square_x ∧ square_x ≤ rect_width - square_size ∧ 
      0 ≤ square_y ∧ square_y ≤ rect_height - square_size → 
      (x - square_x)^2 + (y - square_y)^2 ≥ (square_diameter / 2)^2 :=
sorry

end place_circle_no_overlap_l181_181289


namespace supreme_sports_package_channels_l181_181730

theorem supreme_sports_package_channels (c_start : ℕ) (c_removed1 : ℕ) (c_added1 : ℕ)
                                         (c_removed2 : ℕ) (c_added2 : ℕ)
                                         (c_final : ℕ)
                                         (net1 : ℕ) (net2 : ℕ) (c_mid : ℕ) :
  c_start = 150 →
  c_removed1 = 20 →
  c_added1 = 12 →
  c_removed2 = 10 →
  c_added2 = 8 →
  c_final = 147 →
  net1 = c_removed1 - c_added1 →
  net2 = c_removed2 - c_added2 →
  c_mid = c_start - net1 - net2 →
  c_final - c_mid = 7 :=
by
  intros
  sorry

end supreme_sports_package_channels_l181_181730


namespace reciprocal_relation_l181_181423

theorem reciprocal_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := 
by
  sorry

end reciprocal_relation_l181_181423


namespace truck_capacity_l181_181760

theorem truck_capacity
  (x y : ℝ)
  (h1 : 2 * x + 3 * y = 15.5)
  (h2 : 5 * x + 6 * y = 35) :
  3 * x + 5 * y = 24.5 :=
sorry

end truck_capacity_l181_181760


namespace line_through_center_and_perpendicular_l181_181718

theorem line_through_center_and_perpendicular 
(C : ℝ × ℝ) 
(HC : ∀ (x y : ℝ), x ^ 2 + (y - 1) ^ 2 = 4 → C = (0, 1))
(l : ℝ → ℝ)
(Hl : ∀ x y : ℝ, 3 * x + 2 * y + 1 = 0 → y = l x)
: ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b ↔ 2 * x - 3 * y + 3 = 0) :=
by 
  sorry

end line_through_center_and_perpendicular_l181_181718


namespace Mary_and_Sandra_solution_l181_181024

theorem Mary_and_Sandra_solution (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  (2 * 40 + 3 * 60) * n / (5 * n) = (4 * 30 * n + 80 * m) / (4 * n + m) →
  m + n = 29 :=
by
  intro h
  sorry

end Mary_and_Sandra_solution_l181_181024


namespace gcm_less_than_90_l181_181694

theorem gcm_less_than_90 (a b : ℕ) (h1 : a = 8) (h2 : b = 12) : 
  ∃ x : ℕ, x < 90 ∧ ∀ y : ℕ, y < 90 → (a ∣ y) ∧ (b ∣ y) → y ≤ x → x = 72 :=
sorry

end gcm_less_than_90_l181_181694


namespace range_of_a_l181_181964

noncomputable def domain_f (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + a ≥ 0
noncomputable def range_g (a : ℝ) : Prop := ∀ x : ℝ, x ≤ 2 → 2^x - a ∈ Set.Ioi (0 : ℝ)

theorem range_of_a (a : ℝ) : (domain_f a ∨ range_g a) ∧ ¬(domain_f a ∧ range_g a) → (a ≥ 1 ∨ a ≤ 0) := by
  sorry

end range_of_a_l181_181964


namespace ratio_of_amounts_l181_181450

theorem ratio_of_amounts (B J P : ℝ) (hB : B = 60) (hP : P = (1 / 3) * B) (hJ : J = B - 20) : J / P = 2 :=
by
  have hP_val : P = 20 := by sorry
  have hJ_val : J = 40 := by sorry
  have ratio : J / P = 40 / 20 := by sorry
  show J / P = 2
  sorry

end ratio_of_amounts_l181_181450


namespace problem_statement_l181_181917

variables {Line Plane : Type}
variables {m n : Line} {alpha beta : Plane}

-- Define parallel and perpendicular relations
def parallel (l1 l2 : Line) : Prop := sorry
def perp (l : Line) (p : Plane) : Prop := sorry

-- Define that m and n are different lines
axiom diff_lines (m n : Line) : m ≠ n 

-- Define that alpha and beta are different planes
axiom diff_planes (alpha beta : Plane) : alpha ≠ beta

-- Statement to prove: If m ∥ n and m ⟂ α, then n ⟂ α
theorem problem_statement (h1 : parallel m n) (h2 : perp m alpha) : perp n alpha := 
sorry

end problem_statement_l181_181917


namespace maximize_pasture_area_l181_181638

theorem maximize_pasture_area
  (barn_length fence_cost budget : ℕ)
  (barn_length_eq : barn_length = 400)
  (fence_cost_eq : fence_cost = 5)
  (budget_eq : budget = 1500) :
  ∃ x y : ℕ, y = 150 ∧
  x > 0 ∧
  2 * x + y = budget / fence_cost ∧
  y = barn_length - 2 * x ∧
  (x * y) = (75 * 150) :=
by
  sorry

end maximize_pasture_area_l181_181638


namespace find_ab_l181_181656

theorem find_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_area_9 : (1/2) * (12 / a) * (12 / b) = 9) : 
  a * b = 8 := 
by 
  sorry

end find_ab_l181_181656


namespace min_cubes_needed_proof_l181_181604

noncomputable def min_cubes_needed_to_form_30_digit_number : ℕ :=
  sorry

theorem min_cubes_needed_proof : min_cubes_needed_to_form_30_digit_number = 50 :=
  sorry

end min_cubes_needed_proof_l181_181604


namespace probability_all_selected_l181_181901

theorem probability_all_selected (P_Ram P_Ravi P_Ritu : ℚ) 
  (h1 : P_Ram = 3 / 7) 
  (h2 : P_Ravi = 1 / 5) 
  (h3 : P_Ritu = 2 / 9) : 
  P_Ram * P_Ravi * P_Ritu = 2 / 105 := 
by
  sorry

end probability_all_selected_l181_181901


namespace triangle_angle_sum_l181_181916

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l181_181916


namespace floor_neg_sqrt_eval_l181_181094

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l181_181094


namespace trucks_transport_l181_181220

variables {x y : ℝ}

theorem trucks_transport (h1 : 2 * x + 3 * y = 15.5)
                         (h2 : 5 * x + 6 * y = 35) :
  3 * x + 2 * y = 17 :=
sorry

end trucks_transport_l181_181220


namespace jane_last_day_vases_l181_181788

theorem jane_last_day_vases (vases_per_day : ℕ) (total_vases : ℕ) (days : ℕ) (day_arrange_total: days = 17) (vases_per_day_is_25 : vases_per_day = 25) (total_vases_is_378 : total_vases = 378) :
  (vases_per_day * (days - 1) >= total_vases) → (total_vases - vases_per_day * (days - 1)) = 0 :=
by
  intros h
  -- adding this line below to match condition ": (total_vases - vases_per_day * (days - 1)) = 0"
  sorry

end jane_last_day_vases_l181_181788


namespace find_n_value_l181_181100

theorem find_n_value (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 9) : n = 210 := sorry

end find_n_value_l181_181100


namespace quadratic_other_x_intercept_l181_181993

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end quadratic_other_x_intercept_l181_181993


namespace probability_log2_x_between_1_and_2_l181_181023

noncomputable def probability_log_between : ℝ :=
  let favorable_range := (4:ℝ) - (2:ℝ)
  let total_range := (6:ℝ) - (0:ℝ)
  favorable_range / total_range

theorem probability_log2_x_between_1_and_2 :
  probability_log_between = 1 / 3 :=
sorry

end probability_log2_x_between_1_and_2_l181_181023


namespace circle_possible_m_values_l181_181525

theorem circle_possible_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + m * x - m * y + 2 = 0) ↔ m > 2 ∨ m < -2 :=
by
  sorry

end circle_possible_m_values_l181_181525


namespace arithmetic_sequence_problem_l181_181414

theorem arithmetic_sequence_problem : 
  ∀ (a : ℕ → ℕ) (d : ℕ), 
  a 1 = 1 →
  (a 3 + a 4 + a 5 + a 6 = 20) →
  a 8 = 9 :=
by
  intros a d h₁ h₂
  -- We skip the proof, leaving a placeholder.
  sorry

end arithmetic_sequence_problem_l181_181414


namespace point_P_outside_circle_l181_181849

theorem point_P_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end point_P_outside_circle_l181_181849


namespace height_difference_l181_181810

theorem height_difference :
  let janet_height := 3.6666666666666665
  let sister_height := 2.3333333333333335
  janet_height - sister_height = 1.333333333333333 :=
by
  sorry

end height_difference_l181_181810


namespace smallest_Y_74_l181_181085

def isDigitBin (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d = 0 ∨ d = 1

def smallest_Y (Y : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ isDigitBin T ∧ T % 15 = 0 ∧ Y = T / 15

theorem smallest_Y_74 : smallest_Y 74 := by
  sorry

end smallest_Y_74_l181_181085


namespace percentage_difference_y_less_than_z_l181_181028

-- Define the variables and the conditions
variables (x y z : ℝ)
variables (h₁ : x = 12 * y)
variables (h₂ : z = 1.2 * x)

-- Define the theorem statement
theorem percentage_difference_y_less_than_z (h₁ : x = 12 * y) (h₂ : z = 1.2 * x) :
  ((z - y) / z) * 100 = 93.06 := by
  sorry

end percentage_difference_y_less_than_z_l181_181028


namespace sum_of_reciprocals_l181_181696

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 55) :
  (1 / x + 1 / y) = 16 / 55 :=
by
  sorry

end sum_of_reciprocals_l181_181696


namespace find_g9_l181_181759

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g3_value : g 3 = 4

theorem find_g9 : g 9 = 64 := sorry

end find_g9_l181_181759


namespace multiple_of_bees_l181_181749

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l181_181749


namespace find_pairs_l181_181033

theorem find_pairs (n p : ℕ) (hp : Prime p) (hnp : n ≤ 2 * p) (hdiv : (p - 1) * n + 1 % n^(p-1) = 0) :
  (n = 1 ∧ Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end find_pairs_l181_181033


namespace find_m_value_l181_181257

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def is_perpendicular (v1 v2 : vector) : Prop :=
  dot_product v1 v2 = 0

theorem find_m_value (a b : vector) (m : ℝ) (h: a = (2, -1)) (h2: b = (1, 3))
  (h3: is_perpendicular a (a.1 + m * b.1, a.2 + m * b.2)) : m = 5 :=
sorry

end find_m_value_l181_181257


namespace wood_rope_length_equivalence_l181_181709

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l181_181709


namespace isosceles_triangle_perimeter_l181_181128

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4 ∨ a = 7) (h2 : b = 4 ∨ b = 7) (h3 : a ≠ b) :
  (a + a + b = 15 ∨ a + a + b = 18) ∨ (a + b + b = 15 ∨ a + b + b = 18) :=
sorry

end isosceles_triangle_perimeter_l181_181128


namespace max_area_rect_bamboo_fence_l181_181431

theorem max_area_rect_bamboo_fence (a b : ℝ) (h : a + b = 10) : a * b ≤ 24 :=
by
  sorry

end max_area_rect_bamboo_fence_l181_181431


namespace product_of_y_coordinates_on_line_l181_181691

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem product_of_y_coordinates_on_line (y1 y2 : ℝ) (h1 : distance (4, -1) (-2, y1) = 8) (h2 : distance (4, -1) (-2, y2) = 8) :
  y1 * y2 = -27 :=
sorry

end product_of_y_coordinates_on_line_l181_181691


namespace least_number_of_cans_l181_181955

theorem least_number_of_cans (maaza pepsi sprite : ℕ) (h_maaza : maaza = 80) (h_pepsi : pepsi = 144) (h_sprite : sprite = 368) :
  ∃ n, n = 37 := sorry

end least_number_of_cans_l181_181955


namespace polar_line_through_center_perpendicular_to_axis_l181_181382

-- We define our conditions
def circle_in_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

def center_of_circle (C : ℝ × ℝ) : Prop := C = (2, 0)

def line_in_rectangular (x : ℝ) : Prop := x = 2

-- We now state the proof problem
theorem polar_line_through_center_perpendicular_to_axis (ρ θ : ℝ) : 
  (∃ C, center_of_circle C ∧ (∃ x, line_in_rectangular x)) →
  (circle_in_polar ρ θ → ρ * Real.cos θ = 2) :=
by
  sorry

end polar_line_through_center_perpendicular_to_axis_l181_181382


namespace wendy_percentage_accounting_related_jobs_l181_181662

noncomputable def wendy_accountant_years : ℝ := 25.5
noncomputable def wendy_accounting_manager_years : ℝ := 15.5 -- Including 6 months as 0.5 years
noncomputable def wendy_financial_consultant_years : ℝ := 10.25 -- Including 3 months as 0.25 years
noncomputable def wendy_tax_advisor_years : ℝ := 4
noncomputable def wendy_lifespan : ℝ := 80

theorem wendy_percentage_accounting_related_jobs :
  ((wendy_accountant_years + wendy_accounting_manager_years + wendy_financial_consultant_years + wendy_tax_advisor_years) / wendy_lifespan) * 100 = 69.0625 :=
by
  sorry

end wendy_percentage_accounting_related_jobs_l181_181662


namespace integer_solutions_inequality_system_l181_181720

theorem integer_solutions_inequality_system :
  {x : ℤ | (x + 2 > 0) ∧ (2 * x - 1 ≤ 0)} = {-1, 0} := 
by
  -- proof goes here
  sorry

end integer_solutions_inequality_system_l181_181720


namespace original_useful_item_is_pencil_l181_181548

def code_language (x : String) : String :=
  if x = "item" then "pencil"
  else if x = "pencil" then "mirror"
  else if x = "mirror" then "board"
  else x

theorem original_useful_item_is_pencil : 
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") ∧
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") 
  → "mirror" = "pencil" :=
by sorry

end original_useful_item_is_pencil_l181_181548


namespace seconds_in_12_5_minutes_l181_181314

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end seconds_in_12_5_minutes_l181_181314


namespace find_a_if_f_is_odd_l181_181020

noncomputable def f (a x : ℝ) : ℝ := (Real.logb 2 ((a - x) / (1 + x))) 

theorem find_a_if_f_is_odd (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

end find_a_if_f_is_odd_l181_181020


namespace find_positive_solutions_l181_181524

noncomputable def satisfies_eq1 (x y : ℝ) : Prop :=
  2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0

noncomputable def satisfies_eq2 (x y : ℝ) : Prop :=
  2 * x^2 + x^2 * y^4 = 18 * y^2

theorem find_positive_solutions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  satisfies_eq1 x y ∧ satisfies_eq2 x y ↔ 
  (x = 2 ∧ y = 2) ∨ 
  (x = Real.sqrt 286^(1/4) / 4 ∧ y = Real.sqrt 286^(1/4)) :=
sorry

end find_positive_solutions_l181_181524


namespace right_triangle_area_l181_181781

theorem right_triangle_area (x : ℝ) (h : 3 * x + 4 * x = 10) : 
  (1 / 2) * (3 * x) * (4 * x) = 24 :=
sorry

end right_triangle_area_l181_181781


namespace shape_described_by_theta_eq_c_is_plane_l181_181260

-- Definitions based on conditions in the problem
def spherical_coordinates (ρ θ φ : ℝ) := true

def is_plane_condition (θ c : ℝ) := θ = c

-- Statement to prove
theorem shape_described_by_theta_eq_c_is_plane (c : ℝ) :
  ∀ ρ θ φ : ℝ, spherical_coordinates ρ θ φ → is_plane_condition θ c → "Plane" = "Plane" :=
by sorry

end shape_described_by_theta_eq_c_is_plane_l181_181260


namespace triangle_side_length_uniqueness_l181_181388

-- Define the conditions as axioms
variable (n : ℕ)
variable (h : n > 0)
variable (A1 : 3 * n + 9 > 5 * n - 4)
variable (A2 : 5 * n - 4 > 4 * n + 6)

-- The theorem stating the constraints and expected result
theorem triangle_side_length_uniqueness :
  (4 * n + 6) + (3 * n + 9) > (5 * n - 4) ∧
  (3 * n + 9) + (5 * n - 4) > (4 * n + 6) ∧
  (5 * n - 4) + (4 * n + 6) > (3 * n + 9) ∧
  3 * n + 9 > 5 * n - 4 ∧
  5 * n - 4 > 4 * n + 6 → 
  n = 11 :=
by {
  -- Proof steps can be filled here
  sorry
}

end triangle_side_length_uniqueness_l181_181388


namespace ratio_of_tetrahedrons_volume_l181_181367

theorem ratio_of_tetrahedrons_volume (d R s s' V_ratio m n : ℕ) (h1 : d = 4)
  (h2 : R = 2)
  (h3 : s = 4 * R / Real.sqrt 6)
  (h4 : s' = s / Real.sqrt 8)
  (h5 : V_ratio = (s' / s) ^ 3)
  (hm : m = 1)
  (hn : n = 32)
  (h_ratio : V_ratio = m / n) :
  m + n = 33 :=
by
  sorry

end ratio_of_tetrahedrons_volume_l181_181367


namespace quadratic_roots_l181_181347

-- Define the condition for the quadratic equation
def quadratic_eq (x m : ℝ) : Prop := x^2 - 4*x + m + 2 = 0

-- Define the discriminant condition
def discriminant_pos (m : ℝ) : Prop := (4^2 - 4 * (m + 2)) > 0

-- Define the condition range for m
def m_range (m : ℝ) : Prop := m < 2

-- Define the condition for m as a positive integer
def m_positive_integer (m : ℕ) : Prop := m = 1

-- The main theorem stating the problem
theorem quadratic_roots : 
  (∀ (m : ℝ), discriminant_pos m → m_range m) ∧ 
  (∀ m : ℕ, m_positive_integer m → (∃ x1 x2 : ℝ, quadratic_eq x1 m ∧ quadratic_eq x2 m ∧ x1 = 1 ∧ x2 = 3)) := 
by 
  sorry

end quadratic_roots_l181_181347


namespace brick_width_l181_181011

theorem brick_width (l_brick : ℕ) (w_courtyard l_courtyard : ℕ) (num_bricks : ℕ) (w_brick : ℕ)
  (H1 : l_courtyard = 24) 
  (H2 : w_courtyard = 14) 
  (H3 : num_bricks = 8960) 
  (H4 : l_brick = 25) 
  (H5 : (w_courtyard * 100 * l_courtyard * 100 = (num_bricks * (l_brick * w_brick)))) :
  w_brick = 15 :=
by
  sorry

end brick_width_l181_181011


namespace pages_per_sheet_is_one_l181_181739

-- Definition of conditions
def stories_per_week : Nat := 3
def pages_per_story : Nat := 50
def num_weeks : Nat := 12
def reams_bought : Nat := 3
def sheets_per_ream : Nat := 500

-- Calculate total pages written over num_weeks (short stories only)
def total_pages : Nat := stories_per_week * pages_per_story * num_weeks

-- Calculate total sheets available
def total_sheets : Nat := reams_bought * sheets_per_ream

-- Calculate pages per sheet, rounding to nearest whole number
def pages_per_sheet : Nat := (total_pages / total_sheets)

-- The main statement to prove
theorem pages_per_sheet_is_one : pages_per_sheet = 1 :=
by
  sorry

end pages_per_sheet_is_one_l181_181739


namespace tenth_term_is_correct_l181_181520

-- Definitions corresponding to the problem conditions
def sequence_term (n : ℕ) : ℚ := (-1)^n * (2 * n + 1) / (n^2 + 1)

-- Theorem statement for the equivalent proof problem
theorem tenth_term_is_correct : sequence_term 10 = 21 / 101 := by sorry

end tenth_term_is_correct_l181_181520


namespace compute_ab_val_l181_181966

variables (a b : ℝ)

theorem compute_ab_val
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
sorry

end compute_ab_val_l181_181966


namespace overall_average_score_l181_181203

-- Definitions based on given conditions
def n_m : ℕ := 8   -- number of male students
def avg_m : ℚ := 87  -- average score of male students
def n_f : ℕ := 12  -- number of female students
def avg_f : ℚ := 92  -- average score of female students

-- The target statement to prove
theorem overall_average_score (n_m : ℕ) (avg_m : ℚ) (n_f : ℕ) (avg_f : ℚ) (overall_avg : ℚ) :
  n_m = 8 ∧ avg_m = 87 ∧ n_f = 12 ∧ avg_f = 92 → overall_avg = 90 :=
by
  sorry

end overall_average_score_l181_181203


namespace calculate_P_AB_l181_181982

section Probability
-- Define the given probabilities
variables (P_B_given_A : ℚ) (P_A : ℚ)
-- Given conditions
def given_conditions := P_B_given_A = 3/10 ∧ P_A = 1/5

-- Prove that P(AB) = 3/50
theorem calculate_P_AB (h : given_conditions P_B_given_A P_A) : (P_A * P_B_given_A) = 3/50 :=
by
  rcases h with ⟨h1, h2⟩
  simp [h1, h2]
  -- Here we would include the steps leading to the conclusion; this part just states the theorem
  sorry

end Probability

end calculate_P_AB_l181_181982


namespace cafeteria_extra_fruits_l181_181512

theorem cafeteria_extra_fruits (red_apples green_apples bananas oranges students : ℕ) (fruits_per_student : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : bananas = 17)
  (h4 : oranges = 12)
  (h5 : students = 21)
  (h6 : fruits_per_student = 2) :
  (red_apples + green_apples + bananas + oranges) - (students * fruits_per_student) = 43 :=
by
  sorry

end cafeteria_extra_fruits_l181_181512


namespace squares_in_50th_ring_l181_181968

noncomputable def number_of_squares_in_nth_ring (n : ℕ) : ℕ :=
  8 * n + 6

theorem squares_in_50th_ring : number_of_squares_in_nth_ring 50 = 406 := 
  by
  sorry

end squares_in_50th_ring_l181_181968


namespace isosceles_triangle_base_angle_l181_181771

-- Define the problem and the given conditions
theorem isosceles_triangle_base_angle (A B C : ℝ)
(h_triangle : A + B + C = 180)
(h_isosceles : (A = B ∨ B = C ∨ C = A))
(h_ratio : (A = B / 2 ∨ B = C / 2 ∨ C = A / 2)) :
(A = 45 ∨ A = 72) ∨ (B = 45 ∨ B = 72) ∨ (C = 45 ∨ C = 72) :=
sorry

end isosceles_triangle_base_angle_l181_181771


namespace simplify_expression_l181_181137

noncomputable def p (a b c x k : ℝ) := 
  k * (((x + a) ^ 2 / ((a - b) * (a - c))) +
       ((x + b) ^ 2 / ((b - a) * (b - c))) +
       ((x + c) ^ 2 / ((c - a) * (c - b))))

theorem simplify_expression (a b c k : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : b ≠ c) (h₃ : k ≠ 0) :
  p a b c x k = k :=
sorry

end simplify_expression_l181_181137


namespace solve_for_x_l181_181455

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = 1 / 3 * (6 * x + 45)) : x = 0 :=
sorry

end solve_for_x_l181_181455


namespace find_missing_number_l181_181754

theorem find_missing_number (square boxplus boxtimes boxminus : ℕ) :
  square = 423 / 47 ∧
  1448 = 282 * boxminus + (boxminus * 10 + boxtimes) ∧
  423 * (boxplus / 3) = 282 →
  square = 9 ∧
  boxminus = 5 ∧
  boxtimes = 8 ∧
  boxplus = 2 ∧
  9 = 9 :=
by
  intro h
  sorry

end find_missing_number_l181_181754


namespace maximum_value_of_f_l181_181261

def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem maximum_value_of_f :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x a ≥ -2) → f 2 a = 25 :=
by
  intro a h
  -- sorry to skip the proof
  sorry

end maximum_value_of_f_l181_181261


namespace negation_of_p_l181_181726

def p := ∀ x : ℝ, x^2 ≥ 0

theorem negation_of_p : ¬p = (∃ x : ℝ, x^2 < 0) :=
  sorry

end negation_of_p_l181_181726


namespace root_in_interval_l181_181005

theorem root_in_interval (a b c : ℝ) (h_a : a ≠ 0)
    (h_table : ∀ x y, (x = 1.2 ∧ y = -1.16) ∨ (x = 1.3 ∧ y = -0.71) ∨ (x = 1.4 ∧ y = -0.24) ∨ (x = 1.5 ∧ y = 0.25) ∨ (x = 1.6 ∧ y = 0.76) → y = a * x^2 + b * x + c ) :
  ∃ x₁, 1.4 < x₁ ∧ x₁ < 1.5 ∧ a * x₁^2 + b * x₁ + c = 0 :=
by sorry

end root_in_interval_l181_181005


namespace fraction_value_l181_181213

theorem fraction_value : (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 := by
  sorry

end fraction_value_l181_181213


namespace linear_function_through_origin_l181_181510

theorem linear_function_through_origin (k : ℝ) (h : ∃ x y : ℝ, (x = 0 ∧ y = 0) ∧ y = (k - 2) * x + (k^2 - 4)) : k = -2 :=
by
  sorry

end linear_function_through_origin_l181_181510


namespace problem_solution_l181_181263

variable {a b x y : ℝ}

-- Define the conditions as Lean assumptions
axiom cond1 : a * x + b * y = 3
axiom cond2 : a * x^2 + b * y^2 = 7
axiom cond3 : a * x^3 + b * y^3 = 16
axiom cond4 : a * x^4 + b * y^4 = 42

-- The main theorem statement: under these conditions, prove a * x^5 + b * y^5 = 99
theorem problem_solution : a * x^5 + b * y^5 = 99 := 
sorry -- proof omitted

end problem_solution_l181_181263


namespace min_k_l181_181516

noncomputable 
def f (k : ℕ) (x : ℝ) : ℝ := 
  (Real.sin (k * x / 10)) ^ 4 + (Real.cos (k * x / 10)) ^ 4

theorem min_k (k : ℕ) 
    (h : (∀ a : ℝ, {y | ∃ x : ℝ, a < x ∧ x < a+1 ∧ y = f k x} = 
                  {y | ∃ x : ℝ, y = f k x})) 
    : k ≥ 16 :=
by
  sorry

end min_k_l181_181516


namespace lisa_flight_time_l181_181124

noncomputable def distance : ℝ := 519.5
noncomputable def speed : ℝ := 54.75
noncomputable def time : ℝ := 9.49

theorem lisa_flight_time : distance / speed = time :=
by
  sorry

end lisa_flight_time_l181_181124


namespace other_number_l181_181291

theorem other_number (x : ℕ) (h : 27 + x = 62) : x = 35 :=
by
  sorry

end other_number_l181_181291


namespace weighted_valid_votes_l181_181499

theorem weighted_valid_votes :
  let total_votes := 10000
  let invalid_vote_rate := 0.25
  let valid_votes := total_votes * (1 - invalid_vote_rate)
  let v_b := (valid_votes - 2 * (valid_votes * 0.15 + valid_votes * 0.07) + valid_votes * 0.05) / 4
  let v_a := v_b + valid_votes * 0.15
  let v_c := v_a + valid_votes * 0.07
  let v_d := v_b - valid_votes * 0.05
  let weighted_votes_A := v_a * 3.0
  let weighted_votes_B := v_b * 2.5
  let weighted_votes_C := v_c * 2.75
  let weighted_votes_D := v_d * 2.25
  weighted_votes_A = 7200 ∧
  weighted_votes_B = 3187.5 ∧
  weighted_votes_C = 8043.75 ∧
  weighted_votes_D = 2025 :=
by
  sorry

end weighted_valid_votes_l181_181499


namespace complex_number_problem_l181_181401

open Complex -- Open the complex numbers namespace

theorem complex_number_problem 
  (z1 z2 : ℂ) 
  (h_z1 : z1 = 2 - I) 
  (h_z2 : z2 = -I) : 
  z1 / z2 + Complex.abs z2 = 2 + 2 * I := by
-- Definitions and conditions directly from (a)
  rw [h_z1, h_z2] -- Replace z1 and z2 with their given values
  sorry -- Proof to be filled in place of the solution steps

end complex_number_problem_l181_181401


namespace hexagon_diagonals_l181_181308

theorem hexagon_diagonals (n : ℕ) (h : n = 6) : (n * (n - 3)) / 2 = 9 := by
  sorry

end hexagon_diagonals_l181_181308


namespace onion_rings_cost_l181_181336

variable (hamburger_cost smoothie_cost total_payment change_received : ℕ)

theorem onion_rings_cost (h_hamburger : hamburger_cost = 4) 
                         (h_smoothie : smoothie_cost = 3) 
                         (h_total_payment : total_payment = 20) 
                         (h_change_received : change_received = 11) :
                         total_payment - change_received - hamburger_cost - smoothie_cost = 2 :=
by
  sorry

end onion_rings_cost_l181_181336


namespace solve_g_eq_5_l181_181250

noncomputable def g (x : ℝ) : ℝ :=
if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solve_g_eq_5 : {x : ℝ | g x = 5} = {-3/4, 20/3} :=
by
  sorry

end solve_g_eq_5_l181_181250


namespace josef_game_l181_181488

theorem josef_game : 
  ∃ S : Finset ℕ, 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 1440 ∧ 1440 % n = 0 ∧ n % 5 = 0) ∧ 
    S.card = 18 := sorry

end josef_game_l181_181488


namespace regression_is_appropriate_l181_181892

-- Definitions for the different analysis methods
inductive AnalysisMethod
| ResidualAnalysis : AnalysisMethod
| RegressionAnalysis : AnalysisMethod
| IsoplethBarChart : AnalysisMethod
| IndependenceTest : AnalysisMethod

-- Relating height and weight with an appropriate analysis method
def appropriateMethod (method : AnalysisMethod) : Prop :=
  method = AnalysisMethod.RegressionAnalysis

-- Stating the theorem that regression analysis is the appropriate method
theorem regression_is_appropriate : appropriateMethod AnalysisMethod.RegressionAnalysis :=
by sorry

end regression_is_appropriate_l181_181892


namespace xy_product_l181_181903

noncomputable def f (t : ℝ) : ℝ := Real.sqrt (t^2 + 1) - t + 1

theorem xy_product (x y : ℝ)
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) :
  x * y = 1 := by
  sorry

end xy_product_l181_181903


namespace cannot_form_right_triangle_l181_181971

theorem cannot_form_right_triangle : ¬∃ a b c : ℕ, a = 4 ∧ b = 6 ∧ c = 11 ∧ (a^2 + b^2 = c^2) :=
by
  sorry

end cannot_form_right_triangle_l181_181971


namespace solve_system_of_inequalities_l181_181911

variable {R : Type*} [LinearOrderedField R]

theorem solve_system_of_inequalities (x1 x2 x3 x4 x5 : R)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h5 : x5 > 0) :
  (x1^2 - x3^2) * (x2^2 - x3^2) ≤ 0 ∧ 
  (x3^2 - x1^2) * (x3^2 - x1^2) ≤ 0 ∧ 
  (x3^2 - x3 * x2) * (x1^2 - x3 * x2) ≤ 0 ∧ 
  (x1^2 - x1 * x3) * (x3^2 - x1 * x3) ≤ 0 ∧ 
  (x3^2 - x2 * x1) * (x1^2 - x2 * x1) ≤ 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 :=
sorry

end solve_system_of_inequalities_l181_181911


namespace equilateral_triangle_data_l181_181415

theorem equilateral_triangle_data
  (A : ℝ)
  (b : ℝ)
  (ha : A = 450)
  (hb : b = 25)
  (equilateral : ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a) :
  ∃ (h P : ℝ), h = 36 ∧ P = 75 := by
  sorry

end equilateral_triangle_data_l181_181415


namespace probability_intersection_l181_181775

variables (A B : Type → Prop)

-- Assuming we have a measure space (probability) P
variables {P : Type → Prop}

-- Given probabilities
def p_A := 0.65
def p_B := 0.55
def p_Ac_Bc := 0.20

-- The theorem to be proven
theorem probability_intersection :
  (p_A + p_B - (1 - p_Ac_Bc) = 0.40) :=
by
  sorry

end probability_intersection_l181_181775


namespace troll_problem_l181_181950

theorem troll_problem (T : ℕ) (h : 6 + T + T / 2 = 33) : 4 * 6 - T = 6 :=
by sorry

end troll_problem_l181_181950


namespace word_count_proof_l181_181104

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l181_181104


namespace largest_common_term_l181_181385

theorem largest_common_term (n m : ℕ) (k : ℕ) (a : ℕ) 
  (h1 : a = 7 + 7 * n) 
  (h2 : a = 8 + 12 * m) 
  (h3 : 56 + 84 * k < 500) : a = 476 :=
  sorry

end largest_common_term_l181_181385


namespace solution_y_amount_l181_181693

theorem solution_y_amount :
  ∀ (y : ℝ) (volume_x volume_y : ℝ),
    volume_x = 200 ∧
    volume_y = y ∧
    10 / 100 * volume_x = 20 ∧
    30 / 100 * volume_y = 0.3 * y ∧
    (20 + 0.3 * y) / (volume_x + y) = 0.25 →
    y = 600 :=
by 
  intros y volume_x volume_y
  intros H
  sorry

end solution_y_amount_l181_181693


namespace arrival_time_difference_l181_181610

-- Define the times in minutes, with 600 representing 10:00 AM.
def my_watch_time_planned := 600
def my_watch_fast := 5
def my_watch_slow := 10

def friend_watch_time_planned := 600
def friend_watch_fast := 5

-- Calculate actual arrival times.
def my_actual_arrival_time := my_watch_time_planned - my_watch_fast + my_watch_slow
def friend_actual_arrival_time := friend_watch_time_planned - friend_watch_fast

-- Prove the arrival times and difference.
theorem arrival_time_difference :
  friend_actual_arrival_time < my_actual_arrival_time ∧
  my_actual_arrival_time - friend_actual_arrival_time = 20 :=
by
  -- Proof terms can be filled in later.
  sorry

end arrival_time_difference_l181_181610


namespace tangent_line_intersects_y_axis_at_10_l181_181764

-- Define the curve y = x^2 + 11
def curve (x : ℝ) : ℝ := x^2 + 11

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 12)

-- Define the tangent line at point_of_tangency
def tangent_line (x : ℝ) : ℝ :=
  let slope := curve_derivative point_of_tangency.1
  let y_intercept := point_of_tangency.2 - slope * point_of_tangency.1
  slope * x + y_intercept

-- Theorem stating the y-coordinate of the intersection of the tangent line with the y-axis
theorem tangent_line_intersects_y_axis_at_10 :
  tangent_line 0 = 10 :=
by
  sorry

end tangent_line_intersects_y_axis_at_10_l181_181764


namespace composite_integer_divisors_l181_181333

theorem composite_integer_divisors (n : ℕ) (k : ℕ) (d : ℕ → ℕ) 
  (h_composite : 1 < n ∧ ¬Prime n)
  (h_divisors : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
  (h_distinct : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j)
  (h_range : d 1 = 1 ∧ d k = n)
  (h_ratio : ∀ i, 1 ≤ i ∧ i < k → (d (i + 1) - d i) = (i * (d 2 - d 1))) : n = 6 :=
by sorry

end composite_integer_divisors_l181_181333


namespace min_value_3x_plus_4y_l181_181952

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_plus_4y_l181_181952


namespace slower_speed_percentage_l181_181928

noncomputable def usual_speed_time : ℕ := 16  -- usual time in minutes
noncomputable def additional_time : ℕ := 24  -- additional time in minutes

theorem slower_speed_percentage (S S_slow : ℝ) (D : ℝ) 
  (h1 : D = S * usual_speed_time) 
  (h2 : D = S_slow * (usual_speed_time + additional_time)) : 
  (S_slow / S) * 100 = 40 :=
by 
  sorry

end slower_speed_percentage_l181_181928


namespace eagles_score_l181_181957

variables (F E : ℕ)

theorem eagles_score (h1 : F + E = 56) (h2 : F = E + 8) : E = 24 := 
sorry

end eagles_score_l181_181957


namespace largest_prime_factor_5985_l181_181492

theorem largest_prime_factor_5985 : ∃ p, Nat.Prime p ∧ p ∣ 5985 ∧ ∀ q, Nat.Prime q ∧ q ∣ 5985 → q ≤ p :=
sorry

end largest_prime_factor_5985_l181_181492


namespace remainder_is_zero_l181_181381

theorem remainder_is_zero (D R r : ℕ) (h1 : D = 12 * 42 + R)
                           (h2 : D = 21 * 24 + r)
                           (h3 : r < 21) :
                           r = 0 :=
by 
  sorry

end remainder_is_zero_l181_181381


namespace lunch_break_is_48_minutes_l181_181302

noncomputable def lunch_break_duration (L : ℝ) (p a : ℝ) : Prop :=
  (8 - L) * (p + a) = 0.6 ∧ 
  (9 - L) * p = 0.35 ∧
  (5 - L) * a = 0.1

theorem lunch_break_is_48_minutes :
  ∃ L p a, lunch_break_duration L p a ∧ L * 60 = 48 :=
by
  -- proof steps would go here
  sorry

end lunch_break_is_48_minutes_l181_181302


namespace max_colors_404_max_colors_406_l181_181664

theorem max_colors_404 (n k : ℕ) (h1 : n = 404) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

theorem max_colors_406 (n k : ℕ) (h1 : n = 406) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

end max_colors_404_max_colors_406_l181_181664


namespace distribution_scheme_count_l181_181605

noncomputable def NumberOfDistributionSchemes : Nat :=
  let plumbers := 5
  let residences := 4
  Nat.choose plumbers (residences - 1) * Nat.factorial residences

theorem distribution_scheme_count :
  NumberOfDistributionSchemes = 240 :=
by
  sorry

end distribution_scheme_count_l181_181605


namespace find_x_l181_181090

/-- Let x be a real number such that the square roots of a positive number are given by x - 4 and 3. 
    Prove that x equals 1. -/
theorem find_x (x : ℝ) 
  (h₁ : ∃ n : ℝ, n > 0 ∧ n.sqrt = x - 4 ∧ n.sqrt = 3) : 
  x = 1 :=
by
  sorry

end find_x_l181_181090


namespace probability_of_B_l181_181010

-- Define the events and their probabilities according to the problem description
def A₁ := "Event where a red ball is taken from bag A"
def A₂ := "Event where a white ball is taken from bag A"
def A₃ := "Event where a black ball is taken from bag A"
def B := "Event where a red ball is taken from bag B"

-- Types of bags A and B containing balls
structure Bag where
  red : Nat
  white : Nat
  black : Nat

-- Initial bags
def bagA : Bag := ⟨ 3, 2, 5 ⟩
def bagB : Bag := ⟨ 3, 3, 4 ⟩

-- Probabilities of each event in bagA
def P_A₁ : ℚ := 3 / 10
def P_A₂ : ℚ := 2 / 10
def P_A₃ : ℚ := 5 / 10

-- Probability of event B under conditions A₁, A₂, A₃
def P_B_given_A₁ : ℚ := 4 / 11
def P_B_given_A₂ : ℚ := 3 / 11
def P_B_given_A₃ : ℚ := 3 / 11

-- Goal: Prove that the probability of drawing a red ball from bag B (P(B)) is 3/10
theorem probability_of_B : 
  (P_A₁ * P_B_given_A₁ + P_A₂ * P_B_given_A₂ + P_A₃ * P_B_given_A₃) = (3 / 10) :=
by
  -- Placeholder for the proof
  sorry

end probability_of_B_l181_181010


namespace number_of_students_increased_l181_181978

theorem number_of_students_increased
  (original_number_of_students : ℕ) (increase_in_expenses : ℕ) (diminshed_average_expenditure : ℕ)
  (original_expenditure : ℕ) (increase_in_students : ℕ) :
  original_number_of_students = 35 →
  increase_in_expenses = 42 →
  diminshed_average_expenditure = 1 →
  original_expenditure = 420 →
  (35 + increase_in_students) * (12 - 1) - 420 = 42 →
  increase_in_students = 7 :=
by
  intros
  sorry

end number_of_students_increased_l181_181978


namespace sum_of_a_b_l181_181081

def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem sum_of_a_b (a b : ℝ) (h : symmetric_x_axis (3, a) (b, 4)) : a + b = -1 :=
by
  sorry

end sum_of_a_b_l181_181081


namespace total_distance_crawled_l181_181229

theorem total_distance_crawled :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let pos4 := 0
  abs (pos2 - pos1) + abs (pos3 - pos2) + abs (pos4 - pos3) = 29 :=
by
  sorry

end total_distance_crawled_l181_181229


namespace problem_statement_l181_181463

variable {a b c d : ℚ}

-- Conditions
axiom h1 : a / b = 3
axiom h2 : b / c = 3 / 4
axiom h3 : c / d = 2 / 3

-- Goal
theorem problem_statement : d / a = 2 / 3 := by
  sorry

end problem_statement_l181_181463


namespace largest_packet_size_gcd_l181_181800

theorem largest_packet_size_gcd:
    ∀ (n1 n2 : ℕ), n1 = 36 → n2 = 60 → Nat.gcd n1 n2 = 12 :=
by
  intros n1 n2 h1 h2
  -- Sorry is added because the proof is not required as per the instructions
  sorry

end largest_packet_size_gcd_l181_181800


namespace sequence_value_a8_b8_l181_181152

theorem sequence_value_a8_b8
(a b : ℝ) 
(h1 : a + b = 1) 
(h2 : a^2 + b^2 = 3) 
(h3 : a^3 + b^3 = 4) 
(h4 : a^4 + b^4 = 7) 
(h5 : a^5 + b^5 = 11) 
(h6 : a^6 + b^6 = 18) : 
a^8 + b^8 = 47 :=
sorry

end sequence_value_a8_b8_l181_181152


namespace tangent_points_l181_181493

noncomputable def f (x : ℝ) : ℝ := x^3 + 1
def P : ℝ × ℝ := (-2, 1)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_points (x0 : ℝ) (y0 : ℝ) (hP : P = (-2, 1)) (hf : y0 = f x0) :
  (3 * x0^2 = (y0 - 1) / (x0 + 2)) → (x0 = 0 ∨ x0 = -3) :=
by
  sorry

end tangent_points_l181_181493


namespace trapezoid_AD_BC_ratio_l181_181202

variables {A B C D M N K : Type} {AD BC CM MD NA CN : ℝ}

-- Definition of the trapezoid and the ratio conditions
def is_trapezoid (A B C D : Type) : Prop := sorry -- Assume existence of a trapezoid for lean to accept the statement
def ratio_CM_MD (CM MD : ℝ) : Prop := CM / MD = 4 / 3
def ratio_NA_CN (NA CN : ℝ) : Prop := NA / CN = 4 / 3

-- Proof statement for the given problem
theorem trapezoid_AD_BC_ratio 
  (h_trapezoid: is_trapezoid A B C D)
  (h_CM_MD: ratio_CM_MD CM MD)
  (h_NA_CN: ratio_NA_CN NA CN) :
  AD / BC = 7 / 12 :=
sorry

end trapezoid_AD_BC_ratio_l181_181202


namespace prairie_total_area_l181_181167

theorem prairie_total_area (dust : ℕ) (untouched : ℕ) (total : ℕ) 
  (h1 : dust = 64535) (h2 : untouched = 522) : total = dust + untouched :=
by
  sorry

end prairie_total_area_l181_181167


namespace problem_l181_181037

def m (x : ℝ) : ℝ := (x + 2) * (x + 3)
def n (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 9

theorem problem (x : ℝ) : m x < n x :=
by sorry

end problem_l181_181037


namespace triangle_angle_contradiction_l181_181064

theorem triangle_angle_contradiction (A B C : ℝ) (hA : 60 < A) (hB : 60 < B) (hC : 60 < C) (h_sum : A + B + C = 180) : false :=
by {
  -- This would be the proof part, which we don't need to detail according to the instructions.
  sorry
}

end triangle_angle_contradiction_l181_181064


namespace area_of_rectangular_plot_l181_181384

theorem area_of_rectangular_plot (B L : ℕ) (h1 : L = 3 * B) (h2 : B = 18) : L * B = 972 := by
  sorry

end area_of_rectangular_plot_l181_181384


namespace arithmetic_geometric_condition_l181_181670

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n-1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def sum_arith_seq (a₁ d n : ℕ) : ℕ := n * a₁ + (n * (n-1) / 2) * d

-- Given conditions and required proofs
theorem arithmetic_geometric_condition {d a₁ : ℕ} (h : d ≠ 0) (S₃ : sum_arith_seq a₁ d 3 = 9)
  (geometric_seq : (arithmetic_seq a₁ d 5)^2 = (arithmetic_seq a₁ d 3) * (arithmetic_seq a₁ d 8)) :
  d = 1 ∧ ∀ n, sum_arith_seq 2 1 n = (n^2 + 3 * n) / 2 :=
by
  sorry

end arithmetic_geometric_condition_l181_181670


namespace rightmost_three_digits_of_7_pow_2023_l181_181845

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l181_181845


namespace tan_2alpha_value_beta_value_l181_181430

variable (α β : ℝ)
variable (h1 : 0 < β ∧ β < α ∧ α < π / 2)
variable (h2 : Real.cos α = 1 / 7)
variable (h3 : Real.cos (α - β) = 13 / 14)

theorem tan_2alpha_value : Real.tan (2 * α) = - (8 * Real.sqrt 3 / 47) :=
by
  sorry

theorem beta_value : β = π / 3 :=
by
  sorry

end tan_2alpha_value_beta_value_l181_181430


namespace inequality_pow_gt_linear_l181_181501

theorem inequality_pow_gt_linear {a : ℝ} (n : ℕ) (h₁ : a > -1) (h₂ : a ≠ 0) (h₃ : n ≥ 2) :
  (1 + a:ℝ)^n > 1 + n * a :=
sorry

end inequality_pow_gt_linear_l181_181501


namespace hyperbola_eccentricity_l181_181471

theorem hyperbola_eccentricity (a b : ℝ) (h : a^2 = 4 ∧ b^2 = 3) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 7 / 2 :=
    by
  sorry

end hyperbola_eccentricity_l181_181471


namespace quadratic_max_value_l181_181374

open Real

variables (a b c x : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_max_value (h₀ : a < 0) (x₀ : ℝ) (h₁ : 2 * a * x₀ + b = 0) : 
  ∀ x : ℝ, f a b c x ≤ f a b c x₀ := sorry

end quadratic_max_value_l181_181374


namespace functional_eq_solution_l181_181342

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_eq_solution_l181_181342


namespace inequality_proof_equality_case_l181_181080

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) ≥ c / b - (c^2) / a) :=
sorry

theorem equality_case (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) = c / b - c^2 / a) ↔ (a = b * c) :=
sorry

end inequality_proof_equality_case_l181_181080


namespace batch_of_pizza_dough_makes_three_pizzas_l181_181793

theorem batch_of_pizza_dough_makes_three_pizzas
  (pizza_dough_time : ℕ)
  (baking_time : ℕ)
  (total_time_minutes : ℕ)
  (oven_capacity : ℕ)
  (total_pizzas : ℕ) 
  (number_of_batches : ℕ)
  (one_batch_pizzas : ℕ) :
  pizza_dough_time = 30 →
  baking_time = 30 →
  total_time_minutes = 300 →
  oven_capacity = 2 →
  total_pizzas = 12 →
  total_time_minutes = total_pizzas / oven_capacity * baking_time + number_of_batches * pizza_dough_time →
  number_of_batches = total_time_minutes / 30 →
  one_batch_pizzas = total_pizzas / number_of_batches →
  one_batch_pizzas = 3 :=
by
  intros
  sorry

end batch_of_pizza_dough_makes_three_pizzas_l181_181793


namespace recording_time_is_one_hour_l181_181571

-- Define the recording interval and number of instances
def recording_interval : ℕ := 5 -- The device records data every 5 seconds
def number_of_instances : ℕ := 720 -- The device recorded 720 instances of data

-- Prove that the total recording time is 1 hour
theorem recording_time_is_one_hour : (recording_interval * number_of_instances) / 3600 = 1 := by
  sorry

end recording_time_is_one_hour_l181_181571


namespace tree_height_is_12_l181_181041

-- Let h be the height of the tree in meters.
def height_of_tree (h : ℝ) : Prop :=
  ∃ h, (h / 8 = 150 / 100) → h = 12

theorem tree_height_is_12 : ∃ h : ℝ, height_of_tree h :=
by {
  sorry
}

end tree_height_is_12_l181_181041


namespace x_pow_n_plus_inv_x_pow_n_l181_181983

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end x_pow_n_plus_inv_x_pow_n_l181_181983


namespace odd_difference_even_odd_l181_181757

theorem odd_difference_even_odd (a b : ℤ) (ha : a % 2 = 0) (hb : b % 2 = 1) : (a - b) % 2 = 1 :=
sorry

end odd_difference_even_odd_l181_181757


namespace mod_exp_l181_181590

theorem mod_exp (n : ℕ) : (5^303) % 11 = 4 :=
  by sorry

end mod_exp_l181_181590


namespace vacation_days_proof_l181_181933

-- Define the conditions
def family_vacation (total_days rain_days clear_afternoons : ℕ) : Prop :=
  total_days = 18 ∧ rain_days = 13 ∧ clear_afternoons = 12

-- State the theorem to be proved
theorem vacation_days_proof : family_vacation 18 13 12 → 18 = 18 :=
by
  -- Skip the proof
  intro h
  sorry

end vacation_days_proof_l181_181933


namespace boys_girls_relationship_l181_181391

theorem boys_girls_relationship (b g : ℕ): (4 + 2 * b = g) → (b = (g - 4) / 2) :=
by
  intros h
  sorry

end boys_girls_relationship_l181_181391


namespace weight_of_dried_grapes_l181_181343

/-- The weight of dried grapes available from 20 kg of fresh grapes given the water content in fresh and dried grapes. -/
theorem weight_of_dried_grapes (W_fresh W_dried : ℝ) (fresh_weight : ℝ) (weight_dried : ℝ) :
  W_fresh = 0.9 → 
  W_dried = 0.2 → 
  fresh_weight = 20 →
  weight_dried = (0.1 * fresh_weight) / (1 - W_dried) → 
  weight_dried = 2.5 :=
by sorry

end weight_of_dried_grapes_l181_181343


namespace prob_one_AB_stuck_prob_at_least_two_stuck_l181_181582

-- Define the events and their probabilities.
def prob_traffic_I := 1 / 10
def prob_no_traffic_I := 9 / 10
def prob_traffic_II := 3 / 5
def prob_no_traffic_II := 2 / 5

-- Define the events
def event_A := prob_traffic_I
def not_event_A := prob_no_traffic_I
def event_B := prob_traffic_I
def not_event_B := prob_no_traffic_I
def event_C := prob_traffic_II
def not_event_C := prob_no_traffic_II

-- Define the probabilities as required in the problem
def prob_exactly_one_of_A_B_in_traffic :=
  event_A * not_event_B + not_event_A * event_B

def prob_at_least_two_in_traffic :=
  event_A * event_B * not_event_C +
  event_A * not_event_B * event_C +
  not_event_A * event_B * event_C +
  event_A * event_B * event_C

-- Proofs (statements only)
theorem prob_one_AB_stuck :
  prob_exactly_one_of_A_B_in_traffic = 9 / 50 := sorry

theorem prob_at_least_two_stuck :
  prob_at_least_two_in_traffic = 59 / 500 := sorry

end prob_one_AB_stuck_prob_at_least_two_stuck_l181_181582


namespace twenty_eight_is_seventy_percent_of_what_number_l181_181210

theorem twenty_eight_is_seventy_percent_of_what_number (x : ℝ) (h : 28 / x = 70 / 100) : x = 40 :=
by
  sorry

end twenty_eight_is_seventy_percent_of_what_number_l181_181210


namespace alpha_value_l181_181613

-- Define the conditions in Lean
variables (α β γ k : ℝ)

-- Mathematically equivalent problem statements translated to Lean
theorem alpha_value :
  (∀ β γ, α = (k * γ) / β) → -- proportionality condition
  (α = 4) →
  (β = 27) →
  (γ = 3) →
  (∀ β γ, β = -81 → γ = 9 → α = -4) :=
by
  sorry

end alpha_value_l181_181613


namespace sum_of_numbers_eq_l181_181621

theorem sum_of_numbers_eq (a b : ℕ) (h1 : a = 64) (h2 : b = 32) (h3 : a = 2 * b) : a + b = 96 := 
by 
  sorry

end sum_of_numbers_eq_l181_181621


namespace solve_equation_l181_181244

theorem solve_equation (x : ℝ) :
  x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3 * x + 2 > 0 ↔ x = -3 ∨ x = 0 ∨ x = 5 :=
by
  sorry

end solve_equation_l181_181244


namespace max_plus_min_l181_181483

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 2016
axiom condition2 (x : ℝ) : x > 0 → f x > 2016

theorem max_plus_min (M N : ℝ) (hM : M = f 2016) (hN : N = f (-2016)) : M + N = 4032 :=
by
  sorry

end max_plus_min_l181_181483


namespace Alyssa_has_37_balloons_l181_181042

variable (Sandy_balloons : ℕ) (Sally_balloons : ℕ) (Total_balloons : ℕ)

-- Conditions
axiom Sandy_Condition : Sandy_balloons = 28
axiom Sally_Condition : Sally_balloons = 39
axiom Total_Condition : Total_balloons = 104

-- Definition of Alyssa's balloons
def Alyssa_balloons : ℕ := Total_balloons - (Sandy_balloons + Sally_balloons)

-- The proof statement 
theorem Alyssa_has_37_balloons 
: Alyssa_balloons Sandy_balloons Sally_balloons Total_balloons = 37 :=
by
  -- The proof body will be placed here, but we will leave it as a placeholder for now
  sorry

end Alyssa_has_37_balloons_l181_181042


namespace sequences_correct_l181_181550

def arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def geometric_sequence (b a₁ b₁ : ℕ) : Prop :=
  a₁ * a₁ = b * b₁

noncomputable def sequence_a (n : ℕ) :=
  (n * (n + 1)) / 2

noncomputable def sequence_b (n : ℕ) :=
  ((n + 1) * (n + 1)) / 2

theorem sequences_correct :
  (∀ n : ℕ,
    n ≥ 1 →
    arithmetic_sequence (sequence_a n) (sequence_b n) (sequence_a (n + 1)) ∧
    geometric_sequence (sequence_b n) (sequence_a (n + 1)) (sequence_b (n + 1))) ∧
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (sequence_a 2 = 3) :=
by
  sorry

end sequences_correct_l181_181550


namespace correct_calculation_l181_181077

theorem correct_calculation (m n : ℝ) :
  3 * m^2 * n - 3 * m^2 * n = 0 ∧
  ¬ (3 * m^2 - 2 * m^2 = 1) ∧
  ¬ (3 * m^2 + 2 * m^2 = 5 * m^4) ∧
  ¬ (3 * m + 2 * n = 5 * m * n) := by
  sorry

end correct_calculation_l181_181077


namespace streetlights_each_square_l181_181514

-- Define the conditions
def total_streetlights : Nat := 200
def total_squares : Nat := 15
def unused_streetlights : Nat := 20

-- State the question mathematically
def streetlights_installed := total_streetlights - unused_streetlights
def streetlights_per_square := streetlights_installed / total_squares

-- The theorem we need to prove
theorem streetlights_each_square : streetlights_per_square = 12 := sorry

end streetlights_each_square_l181_181514


namespace imaginaria_city_population_l181_181862

theorem imaginaria_city_population (a b c : ℕ) (h₁ : a^2 + 225 = b^2 + 1) (h₂ : b^2 + 1 + 75 = c^2) : 5 ∣ a^2 :=
by
  sorry

end imaginaria_city_population_l181_181862


namespace prism_faces_l181_181631

-- Define the conditions of the problem
def prism (E : ℕ) : Prop :=
  ∃ (L : ℕ), 3 * L = E

-- Define the main proof statement
theorem prism_faces (E : ℕ) (hE : prism E) : E = 27 → 2 + E / 3 = 11 :=
by
  sorry -- Proof is not required

end prism_faces_l181_181631


namespace coefficients_verification_l181_181680

theorem coefficients_verification :
  let a0 := -3
  let a1 := -13 -- Not required as part of the proof but shown for completeness
  let a2 := 6
  let a3 := 0 -- Filler value to ensure there is a6 value
  let a4 := 0 -- Filler value to ensure there is a6 value
  let a5 := 0 -- Filler value to ensure there is a6 value
  let a6 := 0 -- Filler value to ensure there is a6 value
  (1 + 2*x) * (x - 2)^5 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5 + a6 * (1 - x)^6 ->
  a0 = -3 ∧
  a0 + a1 + a2 + a3 + a4 + a5 + a6 = -32 :=
by
  intro a0 a1 a2 a3 a4 a5 a6 h
  exact ⟨rfl, sorry⟩

end coefficients_verification_l181_181680


namespace combined_age_l181_181946

-- Define the conditions as Lean assumptions
def avg_age_three_years_ago := 19
def number_of_original_members := 6
def number_of_years_passed := 3
def current_avg_age := 19

-- Calculate the total age three years ago
def total_age_three_years_ago := number_of_original_members * avg_age_three_years_ago 

-- Calculate the increase in total age over three years
def total_increase_in_age := number_of_original_members * number_of_years_passed 

-- Calculate the current total age of the original members
def current_total_age_of_original_members := total_age_three_years_ago + total_increase_in_age

-- Define the number of current total members and the current total age
def number_of_current_members := 8
def current_total_age := number_of_current_members * current_avg_age

-- Formally state the problem and proof
theorem combined_age : 
  (current_total_age - current_total_age_of_original_members = 20) := 
by
  sorry

end combined_age_l181_181946


namespace total_teachers_correct_l181_181348

-- Define the number of departments and the total number of teachers
def num_departments : ℕ := 7
def total_teachers : ℕ := 140

-- Proving that the total number of teachers is 140
theorem total_teachers_correct : total_teachers = 140 := 
by
  sorry

end total_teachers_correct_l181_181348


namespace student_correct_answers_l181_181893

theorem student_correct_answers (C I : ℕ) (h₁ : C + I = 100) (h₂ : C - 2 * I = 61) : C = 87 :=
sorry

end student_correct_answers_l181_181893


namespace time_before_Car_Y_started_in_minutes_l181_181897

noncomputable def timeBeforeCarYStarted (speedX speedY distanceX : ℝ) : ℝ :=
  let t := distanceX / speedX
  (speedY * t - distanceX) / speedX

theorem time_before_Car_Y_started_in_minutes 
  (speedX speedY distanceX : ℝ)
  (h_speedX : speedX = 35)
  (h_speedY : speedY = 70)
  (h_distanceX : distanceX = 42) : 
  (timeBeforeCarYStarted speedX speedY distanceX) * 60 = 72 :=
by
  sorry

end time_before_Car_Y_started_in_minutes_l181_181897


namespace positive_diff_40_x_l181_181323

theorem positive_diff_40_x
  (x : ℝ)
  (h : (40 + x + 15) / 3 = 35) :
  abs (x - 40) = 10 :=
sorry

end positive_diff_40_x_l181_181323


namespace part_1_part_2_part_3_l181_181119

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem part_1 (h : f 1 m = 5) : m = 4 :=
sorry

theorem part_2 (m : ℝ) (h : m = 4) : ∀ x : ℝ, f (-x) m = -f x m :=
sorry

theorem part_3 (m : ℝ) (h : m = 4) : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 m < f x2 m :=
sorry

end part_1_part_2_part_3_l181_181119


namespace log_sum_correct_l181_181761

noncomputable def log_sum : ℝ := 
  Real.log 8 / Real.log 10 + 
  3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 2 / Real.log 10 +
  2 * Real.log 5 / Real.log 10 +
  5 * Real.log 25 / Real.log 10

theorem log_sum_correct : abs (log_sum - 12.301) < 0.001 :=
by sorry

end log_sum_correct_l181_181761


namespace find_n_l181_181923

variable {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
variable (h : 1/a + 1/b + 1/c = 1/(a + b + c))

theorem find_n (n : ℤ) : (∃ k : ℕ, n = 2 * k - 1) → 
  (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) :=
by
  sorry

end find_n_l181_181923


namespace min_value_inequality_l181_181878

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 * x / (2 * y + z) + 3 * y / (x + 2 * z) + 9 * z / (x + y) ≥ 83 :=
sorry

end min_value_inequality_l181_181878


namespace cone_central_angle_l181_181980

theorem cone_central_angle (l : ℝ) (α : ℝ) (h : (30 : ℝ) * π / 180 > 0) :
  α = π := 
sorry

end cone_central_angle_l181_181980


namespace slope_of_line_AB_l181_181344

-- Define the points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (2, 4)

-- State the proposition that we need to prove
theorem slope_of_line_AB :
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 5 / 2 := by
  sorry

end slope_of_line_AB_l181_181344


namespace initial_customers_l181_181110

theorem initial_customers (x : ℕ) (h1 : x - 31 + 26 = 28) : x = 33 := 
by 
  sorry

end initial_customers_l181_181110


namespace general_rule_equation_l181_181173

theorem general_rule_equation (n : ℕ) (hn : n > 0) : (n + 1) / n + (n + 1) = (n + 2) + 1 / n :=
by
  sorry

end general_rule_equation_l181_181173


namespace find_A_l181_181515

theorem find_A (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 2 * x ^ 2 - 13 * x + 10 ≠ 0 → 1 / (x ^ 3 - 2 * x ^ 2 - 13 * x + 10) = A / (x + 2) + B / (x - 1) + C / (x - 1) ^ 2)
  → A = 1 / 9 := 
sorry

end find_A_l181_181515


namespace max_chord_length_of_parabola_l181_181172

-- Definitions based on the problem conditions
def parabola (x y : ℝ) : Prop := x^2 = 8 * y
def y_midpoint_condition (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 4

-- The theorem to prove that the maximum length of the chord AB is 12
theorem max_chord_length_of_parabola (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h_mid : y_midpoint_condition y1 y2) : 
  abs ((y1 + y2) + 2 * 2) = 12 :=
sorry

end max_chord_length_of_parabola_l181_181172


namespace linear_transformation_proof_l181_181840

theorem linear_transformation_proof (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) :
  ∃ (k b : ℝ), k = 4 ∧ b = -1 ∧ (y = k * x + b ∧ -1 ≤ y ∧ y ≤ 3) :=
by
  sorry

end linear_transformation_proof_l181_181840


namespace geometric_progression_coincides_arithmetic_l181_181481

variables (a d q : ℝ)
variables (ap : ℕ → ℝ) (gp : ℕ → ℝ)

-- Define the N-th term of the arithmetic progression
def nth_term_ap (n : ℕ) : ℝ := a + n * d

-- Define the N-th term of the geometric progression
def nth_term_gp (n : ℕ) : ℝ := a * q^n

theorem geometric_progression_coincides_arithmetic :
  gp 3 = ap 10 →
  gp 4 = ap 74 :=
by
  intro h
  sorry

end geometric_progression_coincides_arithmetic_l181_181481


namespace tom_cheaper_than_jane_l181_181518

-- Define constants for Store A
def store_a_full_price : ℝ := 125
def store_a_discount_one : ℝ := 0.08
def store_a_discount_two : ℝ := 0.12
def store_a_tax : ℝ := 0.07

-- Define constants for Store B
def store_b_full_price : ℝ := 130
def store_b_discount_one : ℝ := 0.10
def store_b_discount_three : ℝ := 0.15
def store_b_tax : ℝ := 0.05

-- Define the number of smartphones bought by Tom and Jane
def tom_quantity : ℕ := 2
def jane_quantity : ℕ := 3

-- Define the final amount Tom pays
def final_amount_tom : ℝ :=
  let full_price := tom_quantity * store_a_full_price
  let discount := store_a_discount_two * full_price
  let discounted_price := full_price - discount
  let tax := store_a_tax * discounted_price
  discounted_price + tax

-- Define the final amount Jane pays
def final_amount_jane : ℝ :=
  let full_price := jane_quantity * store_b_full_price
  let discount := store_b_discount_three * full_price
  let discounted_price := full_price - discount
  let tax := store_b_tax * discounted_price
  discounted_price + tax

-- Prove that Tom's total cost is $112.68 cheaper than Jane's total cost
theorem tom_cheaper_than_jane : final_amount_jane - final_amount_tom = 112.68 :=
by
  have tom := final_amount_tom
  have jane := final_amount_jane
  sorry

end tom_cheaper_than_jane_l181_181518


namespace point_not_in_second_quadrant_l181_181929

theorem point_not_in_second_quadrant (a : ℝ) :
  (∃ b : ℝ, b = 2 * a - 1) ∧ ¬(a < 0 ∧ (2 * a - 1 > 0)) := 
by sorry

end point_not_in_second_quadrant_l181_181929


namespace lines_divide_circle_into_four_arcs_l181_181841

theorem lines_divide_circle_into_four_arcs (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → y = x + a ∨ y = x + b) →
  a^2 + b^2 = 2 :=
by
  sorry

end lines_divide_circle_into_four_arcs_l181_181841


namespace lemon_count_l181_181714

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end lemon_count_l181_181714


namespace total_modules_in_stock_l181_181242

-- Given conditions
def module_cost_high : ℝ := 10
def module_cost_low : ℝ := 3.5
def total_stock_value : ℝ := 45
def low_module_count : ℕ := 10

-- To be proved: total number of modules in stock
theorem total_modules_in_stock (x : ℕ) (y : ℕ) (h1 : y = low_module_count) 
  (h2 : module_cost_high * x + module_cost_low * y = total_stock_value) : 
  x + y = 11 := 
sorry

end total_modules_in_stock_l181_181242


namespace fraction_red_marbles_after_doubling_l181_181922

theorem fraction_red_marbles_after_doubling (x : ℕ) (h : x > 0) :
  let blue_fraction : ℚ := 3 / 5
  let red_fraction := 1 - blue_fraction
  let initial_blue_marbles := blue_fraction * x
  let initial_red_marbles := red_fraction * x
  let new_red_marbles := 2 * initial_red_marbles
  let new_total_marbles := initial_blue_marbles + new_red_marbles
  let new_red_fraction := new_red_marbles / new_total_marbles
  new_red_fraction = 4 / 7 :=
sorry

end fraction_red_marbles_after_doubling_l181_181922


namespace coffee_cost_per_week_l181_181178

def num_people: ℕ := 4
def cups_per_person_per_day: ℕ := 2
def ounces_per_cup: ℝ := 0.5
def cost_per_ounce: ℝ := 1.25

theorem coffee_cost_per_week : 
  (num_people * cups_per_person_per_day * ounces_per_cup * 7 * cost_per_ounce) = 35 :=
by
  sorry

end coffee_cost_per_week_l181_181178


namespace children_ticket_price_l181_181045

theorem children_ticket_price
  (C : ℝ)
  (adult_ticket_price : ℝ)
  (total_payment : ℝ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (H1 : adult_ticket_price = 8)
  (H2 : total_payment = 201)
  (H3 : total_tickets = 33)
  (H4 : children_tickets = 21)
  : C = 5 :=
by
  sorry

end children_ticket_price_l181_181045


namespace percentage_of_music_students_l181_181234

theorem percentage_of_music_students 
  (total_students : ℕ) 
  (dance_students : ℕ) 
  (art_students : ℕ) 
  (drama_students : ℕ)
  (h_total : total_students = 2000) 
  (h_dance : dance_students = 450) 
  (h_art : art_students = 680) 
  (h_drama : drama_students = 370) 
  : (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 
:= by 
  sorry

end percentage_of_music_students_l181_181234


namespace shopkeeper_percentage_above_cost_l181_181279

theorem shopkeeper_percentage_above_cost (CP MP SP : ℚ) 
  (h1 : CP = 100) 
  (h2 : SP = CP * 1.02)
  (h3 : SP = MP * 0.85) : 
  (MP - CP) / CP * 100 = 20 :=
by sorry

end shopkeeper_percentage_above_cost_l181_181279


namespace augmented_matrix_solution_l181_181857

theorem augmented_matrix_solution (c1 c2 : ℚ) 
    (h1 : 2 * (3 : ℚ) + 3 * (5 : ℚ) = c1)
    (h2 : (5 : ℚ) = c2) : 
    c1 - c2 = 16 := 
by 
  sorry

end augmented_matrix_solution_l181_181857


namespace add_three_to_both_sides_l181_181223

variable {a b : ℝ}

theorem add_three_to_both_sides (h : a < b) : 3 + a < 3 + b :=
by
  sorry

end add_three_to_both_sides_l181_181223


namespace minimize_folded_area_l181_181027

-- defining the problem as statements in Lean
variables (a M N : ℝ) (M_on_AB : M > 0 ∧ M < a) (N_on_CD : N > 0 ∧ N < a)

-- main theorem statement
theorem minimize_folded_area :
  BM = 5 * a / 8 →
  CN = a / 8 →
  S = 3 * a ^ 2 / 8 := sorry

end minimize_folded_area_l181_181027


namespace simplify_expression_l181_181363

theorem simplify_expression (x y : ℝ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 :=
by
  sorry

end simplify_expression_l181_181363


namespace dog_nails_per_foot_l181_181377

-- Definitions from conditions
def number_of_dogs := 4
def number_of_parrots := 8
def total_nails_to_cut := 113
def parrots_claws := 8

-- Derived calculations from the solution but only involving given conditions
def dogs_claws (nails_per_foot : ℕ) := 16 * nails_per_foot
def parrots_total_claws := number_of_parrots * parrots_claws

-- The main theorem to prove the number of nails per dog foot
theorem dog_nails_per_foot :
  ∃ x : ℚ, 16 * x + parrots_total_claws = total_nails_to_cut :=
by {
  -- Directly state the expected answer
  use 3.0625,
  -- Placeholder for proof
  sorry
}

end dog_nails_per_foot_l181_181377


namespace area_of_isosceles_trapezoid_l181_181073

theorem area_of_isosceles_trapezoid (R α : ℝ) (hR : R > 0) (hα1 : 0 < α) (hα2 : α < π) :
  let a := 2 * R
  let b := 2 * R * Real.sin (α / 2)
  let h := R * Real.cos (α / 2)
  (1 / 2) * (a + b) * h = R^2 * (1 + Real.sin (α / 2)) * Real.cos (α / 2) :=
by
  sorry

end area_of_isosceles_trapezoid_l181_181073


namespace sufficient_but_not_necessary_condition_l181_181256

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → (m = 1) ∨ (m = -2)) → (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → false) :=
by
  intros hm h_perp h
  sorry

end sufficient_but_not_necessary_condition_l181_181256


namespace reciprocal_geometric_sum_l181_181156

variable (n : ℕ) (r s : ℝ)
variable (h_r_nonzero : r ≠ 0)
variable (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3)

theorem reciprocal_geometric_sum (n : ℕ) (r s : ℝ) (h_r_nonzero : r ≠ 0)
  (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3) :
  ((1 - (1 / r^2)^n) / (1 - 1 / r^2)) = s^3 / r^2 :=
sorry

end reciprocal_geometric_sum_l181_181156


namespace part1_part2_l181_181485

noncomputable def triangle_area (A B C : ℝ) (a b c : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem part1 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  triangle_area A B C a b c = Real.sqrt 3 :=
sorry

theorem part2 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  ∃ B, 
  (B = 2 * π / 3) ∧ (4 * Real.sin C^2 + 3 * Real.sin A^2 + 2) / (Real.sin B^2) = 5 :=
sorry

end part1_part2_l181_181485


namespace find_m_n_l181_181466

theorem find_m_n :
  ∀ (m n : ℤ), (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m * x + n) → 
  (m = 4 ∧ n = -32) :=
by
  intros m n h
  let x := 0
  sorry

end find_m_n_l181_181466


namespace city_renumbering_not_possible_l181_181091

-- Defining the problem conditions
def city_renumbering_invalid (city_graph : Type) (connected : city_graph → city_graph → Prop) : Prop :=
  ∃ (M N : city_graph), ∀ (renumber : city_graph → city_graph),
  (renumber M = N ∧ renumber N = M) → ¬(
    ∀ x y : city_graph,
    connected x y ↔ connected (renumber x) (renumber y)
  )

-- Statement of the problem
theorem city_renumbering_not_possible (city_graph : Type) (connected : city_graph → city_graph → Prop) :
  city_renumbering_invalid city_graph connected :=
sorry

end city_renumbering_not_possible_l181_181091


namespace sum_of_three_numbers_l181_181154

theorem sum_of_three_numbers :
  ∃ (a b c : ℕ), 
    (a ≤ b ∧ b ≤ c) ∧ 
    (b = 8) ∧ 
    ((a + b + c) / 3 = a + 8) ∧ 
    ((a + b + c) / 3 = c - 20) ∧ 
    (a + b + c = 60) :=
by
  sorry

end sum_of_three_numbers_l181_181154


namespace lives_per_player_l181_181352

-- Definitions based on the conditions
def initial_players : Nat := 2
def joined_players : Nat := 2
def total_lives : Nat := 24

-- Derived condition
def total_players : Nat := initial_players + joined_players

-- Proof statement
theorem lives_per_player : total_lives / total_players = 6 :=
by
  sorry

end lives_per_player_l181_181352


namespace cost_of_fencing_each_side_l181_181907

theorem cost_of_fencing_each_side (x : ℝ) (h : 4 * x = 316) : x = 79 :=
by
  sorry

end cost_of_fencing_each_side_l181_181907


namespace factorization1_factorization2_factorization3_l181_181445

-- (1) Prove x^3 - 6x^2 + 9x == x(x-3)^2
theorem factorization1 (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

-- (2) Prove (x-2)^2 - x + 2 == (x-2)(x-3)
theorem factorization2 (x : ℝ) : (x - 2)^2 - x + 2 = (x - 2) * (x - 3) :=
by sorry

-- (3) Prove (x^2 + y^2)^2 - 4x^2*y^2 == (x + y)^2(x - y)^2
theorem factorization3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
by sorry

end factorization1_factorization2_factorization3_l181_181445


namespace smallest_sum_of_squares_l181_181061

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 175) : 
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 + y^2 = 625 :=
sorry

end smallest_sum_of_squares_l181_181061


namespace replaced_solution_percentage_l181_181776

theorem replaced_solution_percentage (y x z w : ℝ) 
  (h1 : x = 0.5)
  (h2 : y = 80)
  (h3 : z = 0.5 * y)
  (h4 : w = 50) 
  :
  (40 + 0.5 * x) = 50 → x = 20 :=
by
  sorry

end replaced_solution_percentage_l181_181776


namespace pete_total_miles_l181_181913

-- Definitions based on conditions
def flip_step_count : ℕ := 89999
def steps_full_cycle : ℕ := 90000
def total_flips : ℕ := 52
def end_year_reading : ℕ := 55555
def steps_per_mile : ℕ := 1900

-- Total steps Pete walked
def total_steps_pete_walked (flips : ℕ) (end_reading : ℕ) : ℕ :=
  flips * steps_full_cycle + end_reading

-- Total miles Pete walked
def total_miles_pete_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

-- Given the parameters, closest number of miles Pete walked should be 2500
theorem pete_total_miles : total_miles_pete_walked (total_steps_pete_walked total_flips end_year_reading) steps_per_mile = 2500 :=
by
  sorry

end pete_total_miles_l181_181913


namespace least_subtracted_number_correct_l181_181997

noncomputable def least_subtracted_number (n : ℕ) : ℕ :=
  n - 13

theorem least_subtracted_number_correct (n : ℕ) : 
  least_subtracted_number 997 = 997 - 13 ∧
  (least_subtracted_number 997 % 5 = 3) ∧
  (least_subtracted_number 997 % 9 = 3) ∧
  (least_subtracted_number 997 % 11 = 3) :=
by
  let x := 997 - 13
  have : x = 984 := rfl
  have h5 : x % 5 = 3 := by sorry
  have h9 : x % 9 = 3 := by sorry
  have h11 : x % 11 = 3 := by sorry
  exact ⟨rfl, h5, h9, h11⟩

end least_subtracted_number_correct_l181_181997


namespace hakimi_age_is_40_l181_181804

variable (H : ℕ)
variable (Jared_age : ℕ) (Molly_age : ℕ := 30)
variable (total_age : ℕ := 120)

theorem hakimi_age_is_40 (h1 : Jared_age = H + 10) (h2 : H + Jared_age + Molly_age = total_age) : H = 40 :=
by
  sorry

end hakimi_age_is_40_l181_181804


namespace select_team_l181_181452

-- Definition of the problem conditions 
def boys : Nat := 10
def girls : Nat := 12
def team_size : Nat := 8
def boys_in_team : Nat := 4
def girls_in_team : Nat := 4

-- Given conditions reflect in the Lean statement that needs proof
theorem select_team : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 103950 :=
by
  sorry

end select_team_l181_181452


namespace sale_price_lower_by_2_5_percent_l181_181153

open Real

theorem sale_price_lower_by_2_5_percent (x : ℝ) : 
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  sale_price = 0.975 * x :=
by
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  show sale_price = 0.975 * x
  sorry

end sale_price_lower_by_2_5_percent_l181_181153


namespace find_number_of_two_dollar_pairs_l181_181614

noncomputable def pairs_of_two_dollars (x y z : ℕ) : Prop :=
  x + y + z = 15 ∧ 2 * x + 4 * y + 5 * z = 38 ∧ x >= 1 ∧ y >= 1 ∧ z >= 1

theorem find_number_of_two_dollar_pairs (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : 2 * x + 4 * y + 5 * z = 38) 
  (hx : x >= 1) 
  (hy : y >= 1) 
  (hz : z >= 1) :
  pairs_of_two_dollars x y z → x = 12 :=
by
  intros
  sorry

end find_number_of_two_dollar_pairs_l181_181614


namespace compute_one_plus_i_power_four_l181_181155

theorem compute_one_plus_i_power_four (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end compute_one_plus_i_power_four_l181_181155


namespace angle_BCM_in_pentagon_l181_181175

-- Definitions of the conditions
structure Pentagon (A B C D E : Type) :=
  (is_regular : ∀ (x y : Type), ∃ (angle : ℝ), angle = 108)

structure EquilateralTriangle (A B M : Type) :=
  (is_equilateral : ∀ (x y : Type), ∃ (angle : ℝ), angle = 60)

-- Problem statement
theorem angle_BCM_in_pentagon (A B C D E M : Type) (P : Pentagon A B C D E) (T : EquilateralTriangle A B M) :
  ∃ (angle : ℝ), angle = 66 :=
by
  sorry

end angle_BCM_in_pentagon_l181_181175


namespace center_cell_value_l181_181371

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l181_181371


namespace speed_of_second_part_l181_181975

theorem speed_of_second_part
  (total_distance : ℝ)
  (distance_part1 : ℝ)
  (speed_part1 : ℝ)
  (average_speed : ℝ)
  (speed_part2 : ℝ) :
  total_distance = 70 →
  distance_part1 = 35 →
  speed_part1 = 48 →
  average_speed = 32 →
  speed_part2 = 24 :=
by
  sorry

end speed_of_second_part_l181_181975


namespace parametric_to_ordinary_eq_l181_181146

-- Define the parametric equations and the domain of the parameter t
def parametric_eqns (t : ℝ) : ℝ × ℝ := (t + 1, 3 - t^2)

-- Define the target equation to be proved
def target_eqn (x y : ℝ) : Prop := y = -x^2 + 2*x + 2

-- Prove that, given the parametric equations, the target ordinary equation holds
theorem parametric_to_ordinary_eq :
  ∃ (t : ℝ) (x y : ℝ), parametric_eqns t = (x, y) ∧ target_eqn x y :=
by
  sorry

end parametric_to_ordinary_eq_l181_181146


namespace kolya_win_l181_181861

theorem kolya_win : ∀ stones : ℕ, stones = 100 → (∃ strategy : (ℕ → ℕ × ℕ), ∀ opponent_strategy : (ℕ → ℕ × ℕ), true → true) :=
by
  sorry

end kolya_win_l181_181861


namespace remainder_x_plus_3uy_plus_u_div_y_l181_181018

theorem remainder_x_plus_3uy_plus_u_div_y (x y u v : ℕ) (hx : x = u * y + v) (hu : 0 ≤ v) (hv : v < y) (huv : u + v < y) : 
  (x + 3 * u * y + u) % y = u + v :=
by
  sorry

end remainder_x_plus_3uy_plus_u_div_y_l181_181018


namespace dot_product_example_l181_181858

def vector := ℝ × ℝ

-- Define the dot product function
def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example : dot_product (-1, 0) (0, 2) = 0 := by
  sorry

end dot_product_example_l181_181858


namespace infant_weight_in_4th_month_l181_181847

-- Given conditions
def a : ℕ := 3000
def x : ℕ := 4
def y : ℕ := a + 700 * x

-- Theorem stating the weight of the infant in the 4th month equals 5800 grams
theorem infant_weight_in_4th_month : y = 5800 := by
  sorry

end infant_weight_in_4th_month_l181_181847


namespace algebraic_expression_value_l181_181025

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), (∃ x, x = -2 ∧ a * x - b = 1) → 4 * a + 2 * b + 7 = 5 :=
by
  intros a b h
  cases' h with x hx
  cases' hx with hx1 hx2
  rw [hx1] at hx2
  sorry

end algebraic_expression_value_l181_181025


namespace emily_initial_marbles_l181_181356

open Nat

theorem emily_initial_marbles (E : ℕ) (h : 3 * E - (3 * E / 2 + 1) = 8) : E = 6 :=
sorry

end emily_initial_marbles_l181_181356


namespace evaluate_expression_l181_181193

theorem evaluate_expression : 8^3 + 4 * 8^2 + 6 * 8 + 3 = 1000 := by
  sorry

end evaluate_expression_l181_181193


namespace max_students_equal_distribution_l181_181284

-- Define the number of pens and pencils
def pens : ℕ := 1008
def pencils : ℕ := 928

-- Define the problem statement which asks for the GCD of the given numbers
theorem max_students_equal_distribution : Nat.gcd pens pencils = 16 :=
by 
  -- Lean's gcd computation can be used to confirm the result
  sorry

end max_students_equal_distribution_l181_181284


namespace sum_of_x_and_y_greater_equal_twice_alpha_l181_181842

theorem sum_of_x_and_y_greater_equal_twice_alpha (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) :
  x + y ≥ 2 * α :=
sorry

end sum_of_x_and_y_greater_equal_twice_alpha_l181_181842


namespace bug_total_distance_l181_181365

def total_distance (p1 p2 p3 p4 : ℤ) : ℤ :=
  abs (p2 - p1) + abs (p3 - p2) + abs (p4 - p3)

theorem bug_total_distance : total_distance (-3) (-8) 0 6 = 19 := 
by sorry

end bug_total_distance_l181_181365


namespace sin_225_eq_neg_sqrt2_over_2_l181_181863

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l181_181863


namespace simplify_expression_l181_181885

theorem simplify_expression (x : ℝ) : (3 * x) ^ 5 - (4 * x) * (x ^ 4) = 239 * x ^ 5 := 
by
  sorry

end simplify_expression_l181_181885


namespace non_adjacent_ball_arrangements_l181_181402

-- Statement only, proof is omitted
theorem non_adjacent_ball_arrangements :
  let n := (3: ℕ) -- Number of identical yellow balls
  let white_red_positions := (4: ℕ) -- Positions around the yellow unit
  let choose_positions := Nat.choose white_red_positions 2
  let arrange_balls := (2: ℕ) -- Ways to arrange the white and red balls in the chosen positions
  let total_arrangements := choose_positions * arrange_balls
  total_arrangements = 12 := 
by
  sorry

end non_adjacent_ball_arrangements_l181_181402


namespace change_received_l181_181895

def totalCostBeforeDiscount : ℝ :=
  5.75 + 2.50 + 3.25 + 3.75 + 4.20

def discount : ℝ :=
  (3.75 + 4.20) * 0.10

def totalCostAfterDiscount : ℝ :=
  totalCostBeforeDiscount - discount

def salesTax : ℝ :=
  totalCostAfterDiscount * 0.06

def finalTotalCost : ℝ :=
  totalCostAfterDiscount + salesTax

def amountPaid : ℝ :=
  50.00

def change : ℝ :=
  amountPaid - finalTotalCost

theorem change_received (h : change = 30.34) : change = 30.34 := by
  sorry

end change_received_l181_181895


namespace number_of_days_same_l181_181665

-- Defining volumes as given in the conditions.
def volume_project1 : ℕ := 100 * 25 * 30
def volume_project2 : ℕ := 75 * 20 * 50

-- The mathematical statement we want to prove.
theorem number_of_days_same : volume_project1 = volume_project2 → ∀ d : ℕ, d > 0 → d = d :=
by
  sorry

end number_of_days_same_l181_181665


namespace triangle_area_l181_181114

theorem triangle_area (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 := 
by 
  sorry

end triangle_area_l181_181114


namespace number_of_candies_l181_181241

theorem number_of_candies (n : ℕ) (h1 : 11 ≤ n) (h2 : n ≤ 100) (h3 : n % 18 = 0) (h4 : n % 7 = 1) : n = 36 :=
by
  sorry

end number_of_candies_l181_181241


namespace sum_of_consecutive_integers_product_l181_181681

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end sum_of_consecutive_integers_product_l181_181681


namespace unique_solution_abs_eq_l181_181324

theorem unique_solution_abs_eq : ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  sorry

end unique_solution_abs_eq_l181_181324


namespace range_of_a_if_monotonic_l181_181576

theorem range_of_a_if_monotonic :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 3 * a * x^2 - 2 * x + 1 ≥ 0) → a > 1 / 3 :=
by
  sorry

end range_of_a_if_monotonic_l181_181576


namespace dog_has_fewer_lives_than_cat_l181_181679

noncomputable def cat_lives : ℕ := 9
noncomputable def mouse_lives : ℕ := 13
noncomputable def dog_lives : ℕ := mouse_lives - 7
noncomputable def dog_less_lives : ℕ := cat_lives - dog_lives

theorem dog_has_fewer_lives_than_cat : dog_less_lives = 3 := by
  sorry

end dog_has_fewer_lives_than_cat_l181_181679


namespace smallest_value_of_Q_l181_181181

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

noncomputable def A := Q (-1)
noncomputable def B := Q (0)
noncomputable def C := (2 : ℝ)^2
def D := 1 - 2 + 3 - 4 + 5
def E := 2 -- assuming all zeros are real

theorem smallest_value_of_Q :
  min (min (min (min A B) C) D) E = 2 :=
by sorry

end smallest_value_of_Q_l181_181181


namespace grace_wins_probability_l181_181948

def probability_grace_wins : ℚ :=
  let total_possible_outcomes := 36
  let losing_combinations := 6
  let winning_combinations := total_possible_outcomes - losing_combinations
  winning_combinations / total_possible_outcomes

theorem grace_wins_probability :
    probability_grace_wins = 5 / 6 := by
  sorry

end grace_wins_probability_l181_181948


namespace f_g_of_neg2_l181_181411

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x - 1)^2

theorem f_g_of_neg2 : f (g (-2)) = 29 := by
  -- We need to show f(g(-2)) = 29 given the definitions of f and g
  sorry

end f_g_of_neg2_l181_181411


namespace sequence_all_ones_l181_181888

theorem sequence_all_ones (k : ℕ) (n : ℕ → ℕ) (h_k : 2 ≤ k)
  (h1 : ∀ i, 1 ≤ i → i ≤ k → 1 ≤ n i) 
  (h2 : n 2 ∣ 2^(n 1) - 1) 
  (h3 : n 3 ∣ 2^(n 2) - 1) 
  (h4 : n 4 ∣ 2^(n 3) - 1)
  (h5 : ∀ i, 2 ≤ i → i < k → n (i + 1) ∣ 2^(n i) - 1)
  (h6 : n 1 ∣ 2^(n k) - 1) : 
  ∀ i, 1 ≤ i → i ≤ k → n i = 1 := 
by 
  sorry

end sequence_all_ones_l181_181888


namespace solve_fractional_equation_l181_181772

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end solve_fractional_equation_l181_181772


namespace gcd_9125_4277_l181_181705

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 :=
by
  -- proof by Euclidean algorithm steps
  sorry

end gcd_9125_4277_l181_181705


namespace sachin_borrowed_amount_l181_181346

variable (P : ℝ) (gain : ℝ)
variable (interest_rate_borrow : ℝ := 4 / 100)
variable (interest_rate_lend : ℝ := 25 / 4 / 100)
variable (time_period : ℝ := 2)
variable (gain_provided : ℝ := 112.5)

theorem sachin_borrowed_amount (h : gain = 0.0225 * P) : P = 5000 :=
by sorry

end sachin_borrowed_amount_l181_181346


namespace find_radius_of_sphere_l181_181921

noncomputable def radius_of_sphere (R : ℝ) : Prop :=
  ∃ a b c : ℝ, 
  (R = |a| ∧ R = |b| ∧ R = |c|) ∧ 
  ((3 - R)^2 + (2 - R)^2 + (1 - R)^2 = R^2)

theorem find_radius_of_sphere : radius_of_sphere (3 + Real.sqrt 2) ∨ radius_of_sphere (3 - Real.sqrt 2) :=
sorry

end find_radius_of_sphere_l181_181921


namespace unique_7tuple_exists_l181_181162

theorem unique_7tuple_exists 
  (x : Fin 7 → ℝ) 
  (h : (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7) 
  : ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7 :=
sorry

end unique_7tuple_exists_l181_181162


namespace podium_height_l181_181265

theorem podium_height (l w h : ℝ) (r s : ℝ) (H1 : r = l + h - w) (H2 : s = w + h - l) 
  (Hr : r = 40) (Hs : s = 34) : h = 37 :=
by
  sorry

end podium_height_l181_181265


namespace original_class_size_l181_181927

theorem original_class_size
  (N : ℕ)
  (h1 : 40 * N = T)
  (h2 : T + 15 * 32 = 36 * (N + 15)) :
  N = 15 := by
  sorry

end original_class_size_l181_181927


namespace values_of_x0_l181_181467

noncomputable def x_seq (x_0 : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x_0
  | n + 1 => if 3 * (x_seq x_0 n) < 1 then 3 * (x_seq x_0 n)
             else if 3 * (x_seq x_0 n) < 2 then 3 * (x_seq x_0 n) - 1
             else 3 * (x_seq x_0 n) - 2

theorem values_of_x0 (x_0 : ℝ) (h : 0 ≤ x_0 ∧ x_0 < 1) :
  (∃! x_0, x_0 = x_seq x_0 6) → (x_seq x_0 6 = x_0) :=
  sorry

end values_of_x0_l181_181467


namespace means_imply_sum_of_squares_l181_181692

noncomputable def arithmetic_mean (x y z : ℝ) : ℝ :=
(x + y + z) / 3

noncomputable def geometric_mean (x y z : ℝ) : ℝ :=
(x * y * z) ^ (1/3)

noncomputable def harmonic_mean (x y z : ℝ) : ℝ :=
3 / ((1/x) + (1/y) + (1/z))

theorem means_imply_sum_of_squares (x y z : ℝ) :
  arithmetic_mean x y z = 10 →
  geometric_mean x y z = 6 →
  harmonic_mean x y z = 4 →
  x^2 + y^2 + z^2 = 576 :=
by
  -- Proof is omitted for now
  exact sorry

end means_imply_sum_of_squares_l181_181692


namespace eighth_term_of_geometric_sequence_l181_181723

def geometric_sequence_term (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_geometric_sequence : 
  geometric_sequence_term 12 (1 / 3) 8 = 4 / 729 :=
by 
  sorry

end eighth_term_of_geometric_sequence_l181_181723


namespace rectangular_solid_surface_area_l181_181521

-- Definitions based on conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rectangular_solid (a b c : ℕ) :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a * b * c = 231

noncomputable def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- Main theorem based on question and answer
theorem rectangular_solid_surface_area :
  ∃ (a b c : ℕ), rectangular_solid a b c ∧ surface_area a b c = 262 := by
  sorry

end rectangular_solid_surface_area_l181_181521


namespace point_reflection_correct_l181_181796

def point_reflection_y_axis (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (-x, y, -z)

theorem point_reflection_correct :
  point_reflection_y_axis (-3) 5 2 = (3, 5, -2) :=
by
  -- The proof would go here
  sorry

end point_reflection_correct_l181_181796


namespace prob_one_tails_in_three_consecutive_flips_l181_181433

-- Define the probability of heads and tails
def P_H : ℝ := 0.5
def P_T : ℝ := 0.5

-- Define the probability of a sequence of coin flips resulting in exactly one tails in three flips
def P_one_tails_in_three_flips : ℝ :=
  P_H * P_H * P_T + P_H * P_T * P_H + P_T * P_H * P_H

-- The statement we need to prove
theorem prob_one_tails_in_three_consecutive_flips :
  P_one_tails_in_three_flips = 0.375 :=
by
  sorry

end prob_one_tails_in_three_consecutive_flips_l181_181433


namespace fraction_product_108_l181_181904

theorem fraction_product_108 : (1/2 : ℚ) * (1/3) * (1/6) * 108 = 3 := by
  sorry

end fraction_product_108_l181_181904


namespace mixed_number_division_l181_181632

theorem mixed_number_division :
  (4 + 2 / 3 + 5 + 1 / 4) / (3 + 1 / 2 - 2 + 3 / 5) = 11 + 1 / 54 :=
by
  sorry

end mixed_number_division_l181_181632


namespace opposite_of_negative_2023_l181_181553

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l181_181553


namespace emily_chairs_count_l181_181231

theorem emily_chairs_count 
  (C : ℕ) 
  (T : ℕ) 
  (time_per_furniture : ℕ)
  (total_time : ℕ) 
  (hT : T = 2) 
  (h_time : time_per_furniture = 8) 
  (h_total : 8 * C + 8 * T = 48) : 
  C = 4 := by
    sorry

end emily_chairs_count_l181_181231


namespace max_lessons_l181_181676

-- Declaring noncomputable variables for the number of shirts, pairs of pants, and pairs of shoes.
noncomputable def s : ℕ := sorry
noncomputable def p : ℕ := sorry
noncomputable def b : ℕ := sorry

lemma conditions_satisfied :
  2 * (s + 1) * p * b = 2 * s * p * b + 36 ∧
  2 * s * (p + 1) * b = 2 * s * p * b + 72 ∧
  2 * s * p * (b + 1) = 2 * s * p * b + 54 ∧
  s * p * b = 27 ∧
  s * b = 36 ∧
  p * b = 18 := by
  sorry

theorem max_lessons : (2 * s * p * b) = 216 :=
by
  have h := conditions_satisfied
  sorry

end max_lessons_l181_181676


namespace total_price_increase_percentage_l181_181728

theorem total_price_increase_percentage 
    (P : ℝ) 
    (h1 : P > 0) 
    (P_after_first_increase : ℝ := P * 1.2) 
    (P_after_second_increase : ℝ := P_after_first_increase * 1.15) :
    ((P_after_second_increase - P) / P) * 100 = 38 :=
by
  sorry

end total_price_increase_percentage_l181_181728


namespace twentieth_term_is_78_l181_181034

-- Define the arithmetic sequence parameters
def first_term : ℤ := 2
def common_difference : ℤ := 4

-- Define the function to compute the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- Formulate the theorem to prove
theorem twentieth_term_is_78 : nth_term 20 = 78 :=
by
  sorry

end twentieth_term_is_78_l181_181034


namespace solve_equation_one_solve_equation_two_l181_181473

theorem solve_equation_one (x : ℝ) : 3 * x + 7 = 32 - 2 * x → x = 5 :=
by
  intro h
  sorry

theorem solve_equation_two (x : ℝ) : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 → x = -1 :=
by
  intro h
  sorry

end solve_equation_one_solve_equation_two_l181_181473


namespace Debby_bought_bottles_l181_181941

theorem Debby_bought_bottles :
  (5 : ℕ) * (71 : ℕ) = 355 :=
by
  -- Math proof goes here
  sorry

end Debby_bought_bottles_l181_181941


namespace weeks_to_save_l181_181232

-- Define the conditions as given in the problem
def cost_of_bike : ℕ := 600
def gift_from_parents : ℕ := 60
def gift_from_uncle : ℕ := 40
def gift_from_sister : ℕ := 20
def gift_from_friend : ℕ := 30
def weekly_earnings : ℕ := 18

-- Total gift money
def total_gift_money : ℕ := gift_from_parents + gift_from_uncle + gift_from_sister + gift_from_friend

-- Total money after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_gift_money + weekly_earnings * x

-- Main theorem statement
theorem weeks_to_save (x : ℕ) : total_money_after_weeks x = cost_of_bike → x = 25 := by
  sorry

end weeks_to_save_l181_181232


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l181_181359

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ p2 = p1 + 1 ∧ is_prime p3 ∧ p3 = p2 + 1

def sum_divisible_by_5 (p1 p2 p3 : ℕ) : Prop :=
  (p1 + p2 + p3) % 5 = 0

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (p1 p2 p3 : ℕ), consecutive_primes p1 p2 p3 ∧ sum_divisible_by_5 p1 p2 p3 ∧ p1 + p2 + p3 = 10 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l181_181359


namespace factorial_div_eq_l181_181647

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l181_181647


namespace sum_of_midpoint_coordinates_l181_181556

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := -1
  let x2 := 11
  let y2 := 21
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end sum_of_midpoint_coordinates_l181_181556


namespace john_ultramarathon_distance_l181_181036

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l181_181036


namespace square_of_distance_is_82_l181_181641

noncomputable def square_distance_from_B_to_center (a b : ℝ) : ℝ := a^2 + b^2

theorem square_of_distance_is_82
  (a b : ℝ)
  (r : ℝ := 11)
  (ha : a^2 + (b + 7)^2 = r^2)
  (hc : (a + 3)^2 + b^2 = r^2) :
  square_distance_from_B_to_center a b = 82 := by
  -- Proof steps omitted
  sorry

end square_of_distance_is_82_l181_181641


namespace rectangle_perimeter_l181_181856

theorem rectangle_perimeter {y x : ℝ} (hxy : x < y) : 
  2 * (y - x) + 2 * x = 2 * y :=
by
  sorry

end rectangle_perimeter_l181_181856


namespace tailwind_speed_l181_181305

-- Define the given conditions
def plane_speed_with_wind (P W : ℝ) : Prop := P + W = 460
def plane_speed_against_wind (P W : ℝ) : Prop := P - W = 310

-- Theorem stating the proof problem
theorem tailwind_speed (P W : ℝ) 
  (h1 : plane_speed_with_wind P W) 
  (h2 : plane_speed_against_wind P W) : 
  W = 75 :=
sorry

end tailwind_speed_l181_181305


namespace g_sum_1_2_3_2_l181_181931

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then
    (a * b + a - 3) / (3 * a)
  else
    (a * b + b + 3) / (-3 * b)

theorem g_sum_1_2_3_2 : g 1 2 + g 3 2 = -11 / 6 :=
by sorry

end g_sum_1_2_3_2_l181_181931


namespace factor_1_factor_2_factor_3_l181_181113

-- Consider the variables a, b, x, y
variable (a b x y : ℝ)

-- Statement 1: Factorize 3a^3 - 6a^2 + 3a
theorem factor_1 : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
by
  sorry
  
-- Statement 2: Factorize a^2(x - y) + b^2(y - x)
theorem factor_2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a^2 - b^2) :=
by
  sorry

-- Statement 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factor_3 : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
by
  sorry

end factor_1_factor_2_factor_3_l181_181113


namespace max_value_f_l181_181579

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

theorem max_value_f (x : ℝ) (h : -4 < x ∧ x < 1) : ∃ y, f y = -1 ∧ (∀ z, f z ≤ f y) :=
by 
  sorry

end max_value_f_l181_181579


namespace no_constant_term_l181_181026

theorem no_constant_term (n : ℕ) (hn : ∀ r : ℕ, ¬(n = (4 * r) / 3)) : n ≠ 8 :=
by 
  intro h
  sorry

end no_constant_term_l181_181026


namespace uphill_flat_road_system_l181_181008

variables {x y : ℝ}

theorem uphill_flat_road_system :
  (3 : ℝ)⁻¹ * x + (4 : ℝ)⁻¹ * y = 70 / 60 ∧
  (4 : ℝ)⁻¹ * y + (5 : ℝ)⁻¹ * x = 54 / 60 :=
sorry

end uphill_flat_road_system_l181_181008


namespace root_intervals_l181_181820

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem root_intervals (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ r1 r2 : ℝ, (a < r1 ∧ r1 < b ∧ f a b c r1 = 0) ∧ (b < r2 ∧ r2 < c ∧ f a b c r2 = 0) :=
sorry

end root_intervals_l181_181820


namespace find_m_l181_181149

theorem find_m 
  (h : ∀ x, (0 < x ∧ x < 2) ↔ ( - (1 / 2) * x^2 + 2 * x > m * x )) :
  m = 1 :=
sorry

end find_m_l181_181149


namespace find_m_l181_181789

theorem find_m (a0 a1 a2 a3 a4 a5 a6 m : ℝ) 
  (h1 : (1 + m) ^ 6 = a0 + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  m = 1 ∨ m = -3 := 
  sorry

end find_m_l181_181789


namespace find_multiplier_l181_181926

theorem find_multiplier (n x : ℝ) (h1 : n = 1.0) (h2 : 3 * n - 1 = x * n) : x = 2 :=
by
  sorry

end find_multiplier_l181_181926


namespace marching_band_formations_l181_181015

open Nat

theorem marching_band_formations :
  ∃ g, (g = 9) ∧ ∀ s t : ℕ, (s * t = 480 ∧ 15 ≤ t ∧ t ≤ 60) ↔ 
    (t = 15 ∨ t = 16 ∨ t = 20 ∨ t = 24 ∨ t = 30 ∨ t = 32 ∨ t = 40 ∨ t = 48 ∨ t = 60) :=
by
  -- Skipped proof.
  sorry

end marching_band_formations_l181_181015


namespace interest_calculation_years_l181_181936

theorem interest_calculation_years (P n : ℝ) (r : ℝ) (SI CI : ℝ)
  (h₁ : SI = P * r * n / 100)
  (h₂ : r = 5)
  (h₃ : SI = 50)
  (h₄ : CI = P * ((1 + r / 100)^n - 1))
  (h₅ : CI = 51.25) :
  n = 2 := by
  sorry

end interest_calculation_years_l181_181936


namespace fifth_friend_contribution_l181_181736

variables (a b c d e : ℕ)

theorem fifth_friend_contribution:
  a + b + c + d + e = 120 ∧
  a = 2 * b ∧
  b = (c + d) / 3 ∧
  c = 2 * e →
  e = 12 :=
sorry

end fifth_friend_contribution_l181_181736


namespace ice_cream_depth_l181_181563

noncomputable def volume_sphere (r : ℝ) := (4/3) * Real.pi * r^3
noncomputable def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h

theorem ice_cream_depth
  (radius_sphere : ℝ)
  (radius_cylinder : ℝ)
  (density_constancy : volume_sphere radius_sphere = volume_cylinder radius_cylinder (h : ℝ)) :
  h = 9 / 25 := by
  sorry

end ice_cream_depth_l181_181563


namespace modulo_arithmetic_l181_181480

theorem modulo_arithmetic :
  (222 * 15 - 35 * 9 + 2^3) % 18 = 17 :=
by
  sorry

end modulo_arithmetic_l181_181480


namespace general_formula_arithmetic_sequence_l181_181490

variable (a : ℕ → ℤ)

def isArithmeticSequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_formula_arithmetic_sequence :
  isArithmeticSequence a →
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  intros h_seq h_a5 h_a17
  sorry

end general_formula_arithmetic_sequence_l181_181490


namespace symmetric_angle_set_l181_181541

theorem symmetric_angle_set (α β : ℝ) (k : ℤ) 
  (h1 : β = 2 * (k : ℝ) * Real.pi + Real.pi / 12)
  (h2 : α = -Real.pi / 3)
  (symmetric : α + β = -Real.pi / 4) :
  ∃ k : ℤ, β = 2 * (k : ℝ) * Real.pi + Real.pi / 12 :=
sorry

end symmetric_angle_set_l181_181541


namespace rectangle_dimensions_l181_181836

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 3 * w) 
  (h2 : 2 * (l + w) = 2 * l * w) : 
  w = 4 / 3 ∧ l = 4 := 
by
  sorry

end rectangle_dimensions_l181_181836


namespace decrease_in_profit_due_to_looms_breakdown_l181_181095

theorem decrease_in_profit_due_to_looms_breakdown :
  let num_looms := 70
  let month_days := 30
  let total_sales := 1000000
  let total_expenses := 150000
  let daily_sales_per_loom := total_sales / (num_looms * month_days)
  let daily_expenses_per_loom := total_expenses / (num_looms * month_days)
  let loom1_days := 10
  let loom2_days := 5
  let loom3_days := 15
  let loom_repair_cost := 2000
  let loom1_loss := daily_sales_per_loom * loom1_days
  let loom2_loss := daily_sales_per_loom * loom2_days
  let loom3_loss := daily_sales_per_loom * loom3_days
  let total_loss_sales := loom1_loss + loom2_loss + loom3_loss
  let total_repair_cost := loom_repair_cost * 3
  let decrease_in_profit := total_loss_sales + total_repair_cost
  decrease_in_profit = 20285.70 := by
  sorry

end decrease_in_profit_due_to_looms_breakdown_l181_181095


namespace sheep_count_l181_181528

theorem sheep_count (S H : ℕ) (h1 : S / H = 3 / 7) (h2 : H * 230 = 12880) : S = 24 :=
by
  sorry

end sheep_count_l181_181528


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l181_181762

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l181_181762


namespace partial_fraction_decomposition_product_l181_181132

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℚ,
    (A + 2) * (A - 3) *
    (B - 2) * (B - 3) *
    (C - 2) * (C + 2) = x^2 - 12 ∧
    (A = -2) ∧
    (B = 2/5) ∧
    (C = 3/5) ∧
    (A * B * C = -12/25) :=
  sorry

end partial_fraction_decomposition_product_l181_181132


namespace num_12_digit_with_consecutive_ones_l181_181906

theorem num_12_digit_with_consecutive_ones :
  let total := 3^12
  let F12 := 985
  total - F12 = 530456 :=
by
  let total := 3^12
  let F12 := 985
  have h : total - F12 = 530456
  sorry
  exact h

end num_12_digit_with_consecutive_ones_l181_181906


namespace cube_volume_correct_l181_181671

-- Define the height and base dimensions of the pyramid
def pyramid_height := 15
def pyramid_base_length := 12
def pyramid_base_width := 8

-- Define the side length of the cube-shaped box
def cube_side_length := max pyramid_height pyramid_base_length

-- Define the volume of the cube-shaped box
def cube_volume := cube_side_length ^ 3

-- Theorem statement: the volume of the smallest cube-shaped box that can fit the pyramid is 3375 cubic inches
theorem cube_volume_correct : cube_volume = 3375 := by
  sorry

end cube_volume_correct_l181_181671


namespace equilibrium_table_n_max_l181_181246

theorem equilibrium_table_n_max (table : Fin 2010 → Fin 2010 → ℕ) :
  (∃ n, ∀ (i j k l : Fin 2010),
      table i j + table k l = table i l + table k j ∧
      ∀ m ≤ n, (m = 0 ∨ m = 1)
  ) → n = 1 ∧ table (Fin.mk 0 (by norm_num)) (Fin.mk 0 (by norm_num)) = 2 :=
by
  sorry

end equilibrium_table_n_max_l181_181246


namespace average_time_correct_l181_181660

-- Define the times for each runner
def y_time : ℕ := 58
def z_time : ℕ := 26
def w_time : ℕ := 2 * z_time

-- Define the number of runners
def num_runners : ℕ := 3

-- Calculate the summed time of all runners
def total_time : ℕ := y_time + z_time + w_time

-- Calculate the average time
def average_time : ℚ := total_time / num_runners

-- Statement of the proof problem
theorem average_time_correct : average_time = 45.33 := by
  -- The proof would go here
  sorry

end average_time_correct_l181_181660


namespace total_amount_spent_l181_181486

-- Define the prices of the CDs
def price_life_journey : ℕ := 100
def price_day_life : ℕ := 50
def price_when_rescind : ℕ := 85

-- Define the discounted price for The Life Journey CD
def discount_life_journey : ℕ := 20 -- 20% discount equivalent to $20
def discounted_price_life_journey : ℕ := price_life_journey - discount_life_journey

-- Define the number of CDs bought
def num_life_journey : ℕ := 3
def num_day_life : ℕ := 4
def num_when_rescind : ℕ := 2

-- Define the function to calculate money spent on each type with offers in consideration
def cost_life_journey : ℕ := num_life_journey * discounted_price_life_journey
def cost_day_life : ℕ := (num_day_life / 2) * price_day_life -- Buy one get one free offer
def cost_when_rescind : ℕ := num_when_rescind * price_when_rescind

-- Calculate the total cost
def total_cost := cost_life_journey + cost_day_life + cost_when_rescind

-- Define Lean theorem to prove the total cost
theorem total_amount_spent : total_cost = 510 :=
  by
    -- Skipping the actual proof as the prompt specifies
    sorry

end total_amount_spent_l181_181486


namespace max_sum_of_multiplication_table_l181_181667

theorem max_sum_of_multiplication_table :
  let numbers := [3, 5, 7, 11, 17, 19]
  let repeated_num := 19
  ∃ d e f, d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧ d ≠ e ∧ e ≠ f ∧ d ≠ f ∧
  3 * repeated_num * (d + e + f) = 1995 := 
by {
  sorry
}

end max_sum_of_multiplication_table_l181_181667


namespace jerry_painting_hours_l181_181626

-- Define the variables and conditions
def time_painting (P : ℕ) : ℕ := P
def time_counter (P : ℕ) : ℕ := 3 * P
def time_lawn : ℕ := 6
def hourly_rate : ℕ := 15
def total_paid : ℕ := 570

-- Hypothesize that the total hours spent leads to the total payment
def total_hours (P : ℕ) : ℕ := time_painting P + time_counter P + time_lawn

-- Prove that the solution for P matches the conditions
theorem jerry_painting_hours (P : ℕ) 
  (h1 : hourly_rate * total_hours P = total_paid) : 
  P = 8 :=
by
  sorry

end jerry_painting_hours_l181_181626


namespace largest_r_l181_181086

theorem largest_r (a : ℕ → ℕ) (h : ∀ n, 0 < a n ∧ a n ≤ a (n + 2) ∧ a (n + 2) ≤ Int.sqrt (a n ^ 2 + 2 * a (n + 1))) :
  ∃ M, ∀ n ≥ M, a (n + 2) = a n :=
sorry

end largest_r_l181_181086


namespace sky_color_changes_l181_181380

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l181_181380


namespace find_certain_number_l181_181337

theorem find_certain_number (x : ℤ) (h : x - 5 = 4) : x = 9 :=
sorry

end find_certain_number_l181_181337


namespace smallest_gcd_qr_l181_181315

theorem smallest_gcd_qr {p q r : ℕ} (hpq : Nat.gcd p q = 300) (hpr : Nat.gcd p r = 450) : 
  ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 150 :=
by
  sorry

end smallest_gcd_qr_l181_181315


namespace inscribed_quadrilateral_inradius_l181_181706

noncomputable def calculate_inradius (a b c d: ℝ) (A: ℝ) : ℝ := (A / ((a + c + b + d) / 2))

theorem inscribed_quadrilateral_inradius {a b c d: ℝ} (h1: a + c = 10) (h2: b + d = 10) (h3: a + b + c + d = 20) (hA: 12 = 12):
  calculate_inradius a b c d 12 = 6 / 5 :=
by
  sorry

end inscribed_quadrilateral_inradius_l181_181706


namespace production_today_l181_181988

theorem production_today (n : ℕ) (P T : ℕ) 
  (h1 : n = 4) 
  (h2 : (P + T) / (n + 1) = 58) 
  (h3 : P = n * 50) : 
  T = 90 := 
by
  sorry

end production_today_l181_181988


namespace prime_pairs_satisfying_conditions_l181_181174

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q : ℕ) : Prop :=
  (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_satisfying_conditions :
  { (p, q) | is_prime p ∧ is_prime q ∧ satisfies_conditions p q } = {(2, 3), (2, 5), (3, 11)} := 
sorry

end prime_pairs_satisfying_conditions_l181_181174


namespace irrational_neg_pi_lt_neg_two_l181_181281

theorem irrational_neg_pi_lt_neg_two (h1 : Irrational π) (h2 : π > 2) : Irrational (-π) ∧ -π < -2 := by
  sorry

end irrational_neg_pi_lt_neg_two_l181_181281


namespace max_beds_120_l181_181482

/-- The dimensions of the park. --/
def park_length : ℕ := 60
def park_width : ℕ := 30

/-- The dimensions of each flower bed. --/
def bed_length : ℕ := 3
def bed_width : ℕ := 5

/-- The available fencing length. --/
def total_fencing : ℕ := 2400

/-- Calculate the largest number of flower beds that can be created. --/
def max_flower_beds (park_length park_width bed_length bed_width total_fencing : ℕ) : ℕ := 
  let n := park_width / bed_width  -- number of beds per column
  let m := park_length / bed_length  -- number of beds per row
  let vertical_fencing := bed_width * (n - 1) * m
  let horizontal_fencing := bed_length * (m - 1) * n
  if vertical_fencing + horizontal_fencing <= total_fencing then n * m else 0

theorem max_beds_120 : max_flower_beds 60 30 3 5 2400 = 120 := by
  unfold max_flower_beds
  rfl

end max_beds_120_l181_181482


namespace range_cos_A_l181_181633

theorem range_cos_A {A B C : ℚ} (h : 1 / (Real.tan B) + 1 / (Real.tan C) = 1 / (Real.tan A))
  (h_non_neg_A: 0 ≤ A) (h_less_pi_A: A ≤ π): 
  (Real.cos A ∈ Set.Ico (2 / 3) 1) :=
sorry

end range_cos_A_l181_181633


namespace greatest_divisor_l181_181252

theorem greatest_divisor :
  ∃ x, (∀ y : ℕ, y > 0 → x ∣ (7^y + 12*y - 1)) ∧ (∀ z, (∀ y : ℕ, y > 0 → z ∣ (7^y + 12*y - 1)) → z ≤ x) ∧ x = 18 :=
sorry

end greatest_divisor_l181_181252


namespace time_to_park_l181_181505

-- distance from house to market in miles
def d_market : ℝ := 5

-- distance from house to park in miles
def d_park : ℝ := 3

-- time to market in minutes
def t_market : ℝ := 30

-- assuming constant speed, calculate time to park
theorem time_to_park : (3 / 5) * 30 = 18 := by
  sorry

end time_to_park_l181_181505


namespace directors_dividends_correct_l181_181898

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l181_181898


namespace total_distance_hiked_l181_181597

theorem total_distance_hiked
  (a b c d e : ℕ)
  (h1 : a + b + c = 34)
  (h2 : b + c = 24)
  (h3 : c + d + e = 40)
  (h4 : a + c + e = 38)
  (h5 : d = 14) :
  a + b + c + d + e = 48 :=
by
  sorry

end total_distance_hiked_l181_181597


namespace double_rooms_booked_l181_181855

theorem double_rooms_booked (S D : ℕ) 
  (h1 : S + D = 260) 
  (h2 : 35 * S + 60 * D = 14000) : 
  D = 196 :=
by
  sorry

end double_rooms_booked_l181_181855


namespace number_of_free_ranging_chickens_is_105_l181_181489

namespace ChickenProblem

-- Conditions as definitions
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def free_ranging_chickens : ℕ := 2 * run_chickens - 4
def total_coop_run_chickens : ℕ := coop_chickens + run_chickens

-- The ratio condition
def ratio_condition : Prop :=
  (coop_chickens + run_chickens) * 5 = free_ranging_chickens * 2

-- Proof Statement
theorem number_of_free_ranging_chickens_is_105 :
  free_ranging_chickens = 105 :=
by {
  sorry
}

end ChickenProblem

end number_of_free_ranging_chickens_is_105_l181_181489


namespace number_of_stickers_after_losing_page_l181_181054

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l181_181054


namespace common_difference_is_3_l181_181599

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop := 
  a 3 + a 11 = 24

def condition2 (a : ℕ → ℝ) : Prop := 
  a 4 = 3

theorem common_difference_is_3 (h_arith : is_arithmetic a d) (h1 : condition1 a) (h2 : condition2 a) : 
  d = 3 := 
sorry

end common_difference_is_3_l181_181599


namespace odd_function_periodic_example_l181_181813

theorem odd_function_periodic_example (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_segment : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (10 * Real.sqrt 3) = 36 - 20 * Real.sqrt 3 := 
sorry

end odd_function_periodic_example_l181_181813


namespace projection_of_vectors_l181_181361

variables {a b : ℝ}

noncomputable def vector_projection (a b : ℝ) : ℝ :=
  (a * b) / b^2 * b

theorem projection_of_vectors
  (ha : abs a = 6)
  (hb : abs b = 3)
  (hab : a * b = -12) : vector_projection a b = -4 :=
sorry

end projection_of_vectors_l181_181361


namespace total_area_of_frequency_histogram_l181_181830

theorem total_area_of_frequency_histogram (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ f x ∧ f x ≤ 1) (integral_f_one : ∫ x, f x = 1) :
  ∫ x, f x = 1 := 
sorry

end total_area_of_frequency_histogram_l181_181830


namespace max_prime_product_l181_181378

theorem max_prime_product : 
  ∃ (x y z : ℕ), 
    Prime x ∧ Prime y ∧ Prime z ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    x + y + z = 49 ∧ 
    x * y * z = 4199 := 
by
  sorry

end max_prime_product_l181_181378


namespace three_x_minus_five_y_l181_181270

noncomputable def F : ℝ × ℝ :=
  let D := (15, 3)
  let E := (6, 8)
  ((D.1 + E.1) / 2, (D.2 + E.2) / 2)

theorem three_x_minus_five_y : (3 * F.1 - 5 * F.2) = 4 := by
  sorry

end three_x_minus_five_y_l181_181270


namespace initial_number_divisible_by_15_l181_181865

theorem initial_number_divisible_by_15 (N : ℕ) (h : (N - 7) % 15 = 0) : N = 22 := 
by
  sorry

end initial_number_divisible_by_15_l181_181865


namespace largest_constant_c_l181_181987

theorem largest_constant_c (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 1) : 
  x^6 + y^6 ≥ (1 / 2) * x * y :=
sorry

end largest_constant_c_l181_181987


namespace find_r_l181_181097

theorem find_r 
  (r RB QC : ℝ)
  (angleA : ℝ)
  (h0 : RB = 6)
  (h1 : QC = 4)
  (h2 : angleA = 90) :
  (r + 6) ^ 2 + (r + 4) ^ 2 = 10 ^ 2 → r = 2 := 
by 
  sorry

end find_r_l181_181097


namespace find_number_l181_181440

theorem find_number (x n : ℤ) (h1 : |x| = 9 * x - n) (h2 : x = 2) : n = 16 := by 
  sorry

end find_number_l181_181440


namespace sport_formulation_water_quantity_l181_181050

theorem sport_formulation_water_quantity (flavoring : ℝ) (corn_syrup : ℝ) (water : ℝ)
    (hs : flavoring / corn_syrup = 1 / 12) 
    (hw : flavoring / water = 1 / 30) 
    (sport_fs_ratio : flavoring / corn_syrup = 3 * (1 / 12)) 
    (sport_fw_ratio : flavoring / water = (1 / 2) * (1 / 30)) 
    (cs_sport : corn_syrup = 1) : 
    water = 15 :=
by
  sorry

end sport_formulation_water_quantity_l181_181050


namespace cuckoo_chime_78_l181_181417

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end cuckoo_chime_78_l181_181417


namespace sequence_is_arithmetic_l181_181815

theorem sequence_is_arithmetic 
  (a_n : ℕ → ℤ) 
  (h : ∀ n : ℕ, a_n n = n + 1) 
  : ∀ n : ℕ, a_n (n + 1) - a_n n = 1 :=
by
  sorry

end sequence_is_arithmetic_l181_181815


namespace sum_eighth_row_interior_numbers_l181_181145

-- Define the sum of the interior numbers in the nth row of Pascal's Triangle.
def sum_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Problem statement: Prove the sum of the interior numbers of Pascal's Triangle in the eighth row is 126,
-- given the sums for the fifth and sixth rows.
theorem sum_eighth_row_interior_numbers :
  sum_interior_numbers 5 = 14 →
  sum_interior_numbers 6 = 30 →
  sum_interior_numbers 8 = 126 :=
by
  sorry

end sum_eighth_row_interior_numbers_l181_181145


namespace unique_triplet_l181_181866

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end unique_triplet_l181_181866


namespace y_intercept_of_line_l181_181496

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by sorry

end y_intercept_of_line_l181_181496


namespace inequality_range_l181_181184

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem inequality_range (a b x: ℝ) (h : a ≠ 0) :
  (|a + b| + |a - b|) ≥ |a| * f x → 1 ≤ x ∧ x ≤ 2 :=
by
  intro h1
  unfold f at h1
  sorry

end inequality_range_l181_181184


namespace Peter_vacation_l181_181624

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end Peter_vacation_l181_181624


namespace cash_price_of_tablet_l181_181938

-- Define the conditions
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 4 * 40
def next_4_months_payment : ℕ := 4 * 35
def last_4_months_payment : ℕ := 4 * 30
def savings : ℕ := 70

-- Define the total installment payments
def total_installment_payments : ℕ := down_payment + first_4_months_payment + next_4_months_payment + last_4_months_payment

-- The statement to prove
theorem cash_price_of_tablet : total_installment_payments - savings = 450 := by
  -- proof goes here
  sorry

end cash_price_of_tablet_l181_181938


namespace solution_set_bf_x2_solution_set_g_l181_181296

def f (x : ℝ) := x^2 - 5 * x + 6

theorem solution_set_bf_x2 (x : ℝ) : (2 < x ∧ x < 3) ↔ f x < 0 := sorry

noncomputable def g (x : ℝ) := 6 * x^2 - 5 * x + 1

theorem solution_set_g (x : ℝ) : (1 / 3 < x ∧ x < 1 / 2) ↔ g x < 0 := sorry

end solution_set_bf_x2_solution_set_g_l181_181296


namespace sum_of_solutions_l181_181107

theorem sum_of_solutions : 
  ∃ x1 x2 x3 : ℝ, (x1 = 10 ∧ x2 = 50/7 ∧ x3 = 50 ∧ (x1 + x2 + x3 = 470 / 7) ∧ 
  (∀ x : ℝ, x = abs (3 * x - abs (50 - 3 * x)) → (x = x1 ∨ x = x2 ∨ x = x3))) := 
sorry

end sum_of_solutions_l181_181107


namespace hike_on_saturday_l181_181158

-- Define the conditions
variables (x : Real) -- distance hiked on Saturday
variables (y : Real) -- distance hiked on Sunday
variables (z : Real) -- total distance hiked

-- Define given values
def hiked_on_sunday : Real := 1.6
def total_hiked : Real := 9.8

-- The hypothesis: y + x = z
axiom hike_total : y + x = z

theorem hike_on_saturday : x = 8.2 :=
by
  sorry

end hike_on_saturday_l181_181158


namespace tan_eq_example_l181_181822

theorem tan_eq_example (x : ℝ) (hx : Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x)) : x = 30 * Real.pi / 180 :=
  sorry

end tan_eq_example_l181_181822


namespace MaryAddedCandy_l181_181102

-- Definitions based on the conditions
def MaryInitialCandyCount (MeganCandyCount : ℕ) : ℕ :=
  3 * MeganCandyCount

-- Given conditions
def MeganCandyCount : ℕ := 5
def MaryTotalCandyCount : ℕ := 25

-- Proof statement
theorem MaryAddedCandy : 
  let MaryInitialCandy := MaryInitialCandyCount MeganCandyCount
  MaryTotalCandyCount - MaryInitialCandy = 10 :=
by 
  sorry

end MaryAddedCandy_l181_181102


namespace georgia_vs_texas_license_plates_l181_181040

theorem georgia_vs_texas_license_plates :
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 :=
by
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  show georgia_plates - texas_plates = 731161600
  sorry

end georgia_vs_texas_license_plates_l181_181040


namespace proof_least_sum_l181_181120

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end proof_least_sum_l181_181120


namespace delivery_newspapers_15_houses_l181_181896

-- State the problem using Lean 4 syntax

noncomputable def delivery_sequences (n : ℕ) : ℕ :=
  if h : n < 3 then 2^n
  else if n = 3 then 6
  else delivery_sequences (n-1) + delivery_sequences (n-2) + delivery_sequences (n-3)

theorem delivery_newspapers_15_houses :
  delivery_sequences 15 = 849 :=
sorry

end delivery_newspapers_15_houses_l181_181896


namespace jed_correct_speed_l181_181103

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end jed_correct_speed_l181_181103


namespace projection_of_a_onto_b_l181_181049

namespace VectorProjection

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def scalar_projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_onto_b :
  scalar_projection (1, -2) (3, 4) = -1 := by
    sorry

end VectorProjection

end projection_of_a_onto_b_l181_181049


namespace equal_segments_l181_181611

-- Given a triangle ABC and D as the foot of the bisector from B
variables (A B C D E F : Point) (ABC : Triangle A B C) (Dfoot : BisectorFoot B A C D) 

-- Given that the circumcircles of triangles ABD and BCD intersect sides AB and BC at E and F respectively
variables (circABD : Circumcircle A B D) (circBCD : Circumcircle B C D)
variables (intersectAB : Intersect circABD A B E) (intersectBC : Intersect circBCD B C F)

-- The theorem to prove that AE = CF
theorem equal_segments : AE = CF :=
by
  sorry

end equal_segments_l181_181611


namespace floor_eq_l181_181589

theorem floor_eq (r : ℝ) (h : ⌊r⌋ + r = 12.4) : r = 6.4 := by
  sorry

end floor_eq_l181_181589


namespace hexagon_perimeter_of_intersecting_triangles_l181_181176

/-- Given two equilateral triangles with parallel sides, where the perimeter of the blue triangle 
    is 4 and the perimeter of the green triangle is 5, prove that the perimeter of the hexagon 
    formed by their intersection is 3. -/
theorem hexagon_perimeter_of_intersecting_triangles 
    (P_blue P_green P_hexagon : ℝ)
    (h_blue : P_blue = 4)
    (h_green : P_green = 5) :
    P_hexagon = 3 := 
sorry

end hexagon_perimeter_of_intersecting_triangles_l181_181176


namespace commute_distance_l181_181773

theorem commute_distance (D : ℝ)
  (h1 : ∀ t : ℝ, t > 0 → t = D / 45)
  (h2 : ∀ t : ℝ, t > 0 → t = D / 30)
  (h3 : D / 45 + D / 30 = 1) :
  D = 18 :=
by
  sorry

end commute_distance_l181_181773


namespace integer_equality_condition_l181_181653

theorem integer_equality_condition
  (x y z : ℤ)
  (h : x * (x - y) + y * (y - z) + z * (z - x) = 0) :
  x = y ∧ y = z :=
sorry

end integer_equality_condition_l181_181653


namespace complex_division_l181_181115

theorem complex_division (i : ℂ) (h_i : i * i = -1) : (3 - 4 * i) / i = 4 - 3 * i :=
by
  sorry

end complex_division_l181_181115


namespace trigonometric_identity_l181_181141

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin α ^ 2 + 2 * Real.cos α ^ 2 = 6 / 5 := 
by 
  sorry

end trigonometric_identity_l181_181141


namespace equilateral_triangle_area_percentage_l181_181491

noncomputable def percentage_area_of_triangle_in_pentagon (s : ℝ) : ℝ :=
  ((4 * Real.sqrt 3 - 3) / 13) * 100

theorem equilateral_triangle_area_percentage
  (s : ℝ) :
  let pentagon_area := s^2 * (1 + Real.sqrt 3 / 4)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  (triangle_area / pentagon_area) * 100 = percentage_area_of_triangle_in_pentagon s :=
by
  sorry

end equilateral_triangle_area_percentage_l181_181491


namespace depth_of_well_l181_181573

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ)
  (h1 : d = 15 * t1^2)
  (h2 : t2 = d / 1100)
  (h3 : t1 + t2 = 9.5) :
  d = 870.25 := 
sorry

end depth_of_well_l181_181573


namespace probability_all_digits_distinct_probability_all_digits_odd_l181_181074

-- Definitions to be used in the proof
def total_possibilities : ℕ := 10^5
def all_distinct_possibilities : ℕ := 10 * 9 * 8 * 7 * 6
def all_odd_possibilities : ℕ := 5^5

-- Probabilities
def prob_all_distinct : ℚ := all_distinct_possibilities / total_possibilities
def prob_all_odd : ℚ := all_odd_possibilities / total_possibilities

-- Lean 4 Statements to Prove
theorem probability_all_digits_distinct :
  prob_all_distinct = 30240 / 100000 := by
  sorry

theorem probability_all_digits_odd :
  prob_all_odd = 3125 / 100000 := by
  sorry

end probability_all_digits_distinct_probability_all_digits_odd_l181_181074


namespace min_value_l181_181989

variable (a b c : ℝ)

theorem min_value (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) : 
  (a - b) ^ 2 + (b - c) ^ 2 = 25 / 2 := 
sorry

end min_value_l181_181989


namespace factorize_expression_l181_181721

variable (a b : ℝ)

theorem factorize_expression : (a - b)^2 + 6 * (b - a) + 9 = (a - b - 3)^2 :=
by
  sorry

end factorize_expression_l181_181721


namespace find_c_l181_181517

theorem find_c (x y c : ℝ) (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = c^x * 27^y)
  (h2 : x + y = 4) : c = 49 :=
by
  sorry

end find_c_l181_181517


namespace max_sinA_cosB_cosC_l181_181082

theorem max_sinA_cosB_cosC (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A ∧ A < 180) (h3 : 0 < B ∧ B < 180) (h4 : 0 < C ∧ C < 180) : 
  ∃ M : ℝ, M = (1 + Real.sqrt 5) / 2 ∧ ∀ a b c : ℝ, a + b + c = 180 → 0 < a ∧ a < 180 → 0 < b ∧ b < 180 → 0 < c ∧ c < 180 → (Real.sin a + Real.cos b * Real.cos c) ≤ M :=
by sorry

end max_sinA_cosB_cosC_l181_181082


namespace power_mod_l181_181157

theorem power_mod (h : 5 ^ 200 ≡ 1 [MOD 1000]) : 5 ^ 6000 ≡ 1 [MOD 1000] :=
by
  sorry

end power_mod_l181_181157


namespace quadruplet_babies_l181_181802

variable (a b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : a = 5 * b)
variable (h3 : 2 * a + 3 * b + 4 * c = 1500)

theorem quadruplet_babies : 4 * c = 136 := by
  sorry

end quadruplet_babies_l181_181802


namespace interest_rate_l181_181609

-- Define the sum of money
def P : ℝ := 1800

-- Define the time period in years
def T : ℝ := 2

-- Define the difference in interests
def interest_difference : ℝ := 18

-- Define the relationship between simple interest, compound interest, and the interest rate
theorem interest_rate (R : ℝ) 
  (h1 : SI = P * R * T / 100)
  (h2 : CI = P * (1 + R/100)^2 - P)
  (h3 : CI - SI = interest_difference) :
  R = 10 :=
by
  sorry

end interest_rate_l181_181609


namespace angle_F_calculation_l181_181083

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end angle_F_calculation_l181_181083


namespace maximum_value_of_f_on_interval_l181_181558

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3

theorem maximum_value_of_f_on_interval :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 3) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 57 :=
by
  sorry

end maximum_value_of_f_on_interval_l181_181558


namespace neg_abs_nonneg_l181_181503

theorem neg_abs_nonneg :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by
  sorry

end neg_abs_nonneg_l181_181503


namespace elapsed_time_l181_181713

variable (totalDistance : ℕ) (runningSpeed : ℕ) (distanceRemaining : ℕ)

theorem elapsed_time (h1 : totalDistance = 120) (h2 : runningSpeed = 4) (h3 : distanceRemaining = 20) :
  (totalDistance - distanceRemaining) / runningSpeed = 25 := by
sorry

end elapsed_time_l181_181713


namespace box_volume_correct_l181_181774

-- Define the dimensions of the obelisk
def obelisk_height : ℕ := 15
def base_length : ℕ := 8
def base_width : ℕ := 10

-- Define the dimension and volume goal for the cube-shaped box
def box_side_length : ℕ := obelisk_height
def box_volume : ℕ := box_side_length ^ 3

-- The proof goal
theorem box_volume_correct : box_volume = 3375 := 
by sorry

end box_volume_correct_l181_181774


namespace annual_donation_amount_l181_181386

-- Define the conditions
variables (age_start age_end : ℕ)
variables (total_donations : ℕ)

-- Define the question (prove the annual donation amount) given these conditions
theorem annual_donation_amount (h1 : age_start = 13) (h2 : age_end = 33) (h3 : total_donations = 105000) :
  total_donations / (age_end - age_start) = 5250 :=
by
   sorry

end annual_donation_amount_l181_181386


namespace ending_number_of_multiples_l181_181758

theorem ending_number_of_multiples (n : ℤ) (h : 991 = (n - 100) / 10 + 1) : n = 10000 :=
by
  sorry

end ending_number_of_multiples_l181_181758


namespace jack_travel_total_hours_l181_181007

theorem jack_travel_total_hours :
  (20 + 14 * 24) + (15 + 10 * 24) + (10 + 7 * 24) = 789 := by
  sorry

end jack_travel_total_hours_l181_181007


namespace arithmetic_sequence_sum_S9_l181_181287

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = n * (a 1 + a n) / 2

-- Problem statement in Lean
theorem arithmetic_sequence_sum_S9 (h_seq : ∃ d, arithmetic_sequence a d) (h_a2 : a 2 = -2) (h_a8 : a 8 = 6) (h_S_def : sum_of_first_n_terms a S) : S 9 = 18 := 
by {
  sorry
}

end arithmetic_sequence_sum_S9_l181_181287


namespace min_value_frac_sqrt_l181_181129

theorem min_value_frac_sqrt (x : ℝ) (h : x > 1) : 
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 :=
sorry

end min_value_frac_sqrt_l181_181129


namespace new_member_money_l181_181238

variable (T M : ℝ)
variable (H1 : T / 7 = 20)
variable (H2 : (T + M) / 8 = 14)

theorem new_member_money : M = 756 :=
by
  sorry

end new_member_money_l181_181238


namespace m_condition_sufficient_not_necessary_l181_181816

-- Define the function f(x) and its properties
def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

-- Define the condition for the function being increasing on (0, ∞)
def is_increasing_on_positives (m : ℝ) :=
  ∀ x y : ℝ, 0 < x → x < y → f m x < f m y

-- Prove that if m > 0, then the function is increasing on (0, ∞)
lemma m_gt_0_sufficient (m : ℝ) (h : 0 < m) : is_increasing_on_positives m :=
sorry

-- Show that the condition is indeed sufficient but not necessary
theorem m_condition_sufficient_not_necessary :
  ∀ m : ℝ, (0 < m → is_increasing_on_positives m) ∧ (is_increasing_on_positives m → 0 < m) :=
sorry

end m_condition_sufficient_not_necessary_l181_181816


namespace sum_of_interior_angles_quadrilateral_l181_181458

-- Define the function for the sum of the interior angles
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Theorem that the sum of the interior angles of a quadrilateral is 360 degrees
theorem sum_of_interior_angles_quadrilateral : sum_of_interior_angles 4 = 360 :=
by
  sorry

end sum_of_interior_angles_quadrilateral_l181_181458


namespace average_marks_five_subjects_l181_181214

theorem average_marks_five_subjects 
  (P total_marks : ℕ)
  (h1 : total_marks = P + 350) :
  (total_marks - P) / 5 = 70 :=
by
  sorry

end average_marks_five_subjects_l181_181214


namespace none_of_these_true_l181_181702

variable (s r p q : ℝ)
variable (hs : s > 0) (hr : r > 0) (hpq : p * q ≠ 0) (h : s * (p * r) > s * (q * r))

theorem none_of_these_true : ¬ (-p > -q) ∧ ¬ (-p > q) ∧ ¬ (1 > -q / p) ∧ ¬ (1 < q / p) :=
by
  -- The hypothetical theorem to be proven would continue here
  sorry

end none_of_these_true_l181_181702


namespace cylinder_height_decrease_l181_181996

/--
Two right circular cylinders have the same volume. The radius of the second cylinder is 20% more than the radius
of the first. Prove that the height of the second cylinder is approximately 30.56% less than the first one's height.
-/
theorem cylinder_height_decrease (r1 h1 r2 h2 : ℝ) (hradius : r2 = 1.2 * r1) (hvolumes : π * r1^2 * h1 = π * r2^2 * h2) :
  h2 = 25 / 36 * h1 :=
by
  sorry

end cylinder_height_decrease_l181_181996


namespace distance_between_foci_of_ellipse_l181_181643

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l181_181643


namespace part_I_part_II_l181_181438

noncomputable def f (x : ℝ) := Real.sin x
noncomputable def f' (x : ℝ) := Real.cos x

theorem part_I (x : ℝ) (h : 0 < x) : f' x > 1 - x^2 / 2 := sorry

theorem part_II (a : ℝ) : (∀ x, 0 < x ∧ x < Real.pi / 2 → f x + f x / f' x > a * x) ↔ a ≤ 2 := sorry

end part_I_part_II_l181_181438


namespace min_value_expr_l181_181886

open Real

theorem min_value_expr(p q r : ℝ)(hp : 0 < p)(hq : 0 < q)(hr : 0 < r) :
  (5 * r / (3 * p + q) + 5 * p / (q + 3 * r) + 4 * q / (2 * p + 2 * r)) ≥ 5 / 2 :=
sorry

end min_value_expr_l181_181886


namespace radius_of_circle_l181_181508

theorem radius_of_circle (r : ℝ) (h : 6 * Real.pi * r + 6 = 2 * Real.pi * r^2) : 
  r = (3 + Real.sqrt 21) / 2 :=
by
  sorry

end radius_of_circle_l181_181508


namespace sequence_general_term_correct_l181_181387

open Nat

def S (n : ℕ) : ℤ := 3 * (n : ℤ) * (n : ℤ) - 2 * (n : ℤ) + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2
  else 6 * (n : ℤ) - 5

theorem sequence_general_term_correct : ∀ n, (S n - S (n - 1) = a n) :=
by
  intros
  sorry

end sequence_general_term_correct_l181_181387


namespace determine_OP_l181_181657

variable (a b c d : ℝ)
variable (O A B C D P : ℝ)
variable (p : ℝ)

def OnLine (O A B C D P : ℝ) : Prop := O < A ∧ A < B ∧ B < C ∧ C < D ∧ B < P ∧ P < C

theorem determine_OP (h : OnLine O A B C D P) 
(hAP : P - A = p - a) 
(hPD : D - P = d - p) 
(hBP : P - B = p - b) 
(hPC : C - P = c - p) 
(hAP_PD_BP_PC : (p - a) / (d - p) = (p - b) / (c - p)) :
  p = (a * c - b * d) / (a - b + c - d) :=
sorry

end determine_OP_l181_181657


namespace distinct_solutions_difference_eq_sqrt29_l181_181947

theorem distinct_solutions_difference_eq_sqrt29 :
  (∃ a b : ℝ, a > b ∧
    (∀ x : ℝ, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ 
      x = a ∨ x = b) ∧ 
    a - b = Real.sqrt 29) :=
sorry

end distinct_solutions_difference_eq_sqrt29_l181_181947


namespace least_possible_value_of_m_plus_n_l181_181186

noncomputable def least_possible_sum (m n : ℕ) : ℕ :=
m + n

theorem least_possible_value_of_m_plus_n (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0)
  (h3 : Nat.gcd (m + n) 330 = 1)
  (h4 : m^m % n^n = 0)
  (h5 : m % n ≠ 0) : 
  least_possible_sum m n = 98 := 
sorry

end least_possible_value_of_m_plus_n_l181_181186


namespace inequality_proof_l181_181349

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end inequality_proof_l181_181349


namespace omega_terms_sum_to_zero_l181_181446

theorem omega_terms_sum_to_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 :=
by sorry

end omega_terms_sum_to_zero_l181_181446


namespace complex_point_quadrant_l181_181629

theorem complex_point_quadrant 
  (i : Complex) 
  (h_i_unit : i = Complex.I) : 
  (Complex.re ((i - 3) / (1 + i)) < 0) ∧ (Complex.im ((i - 3) / (1 + i)) > 0) :=
by {
  sorry
}

end complex_point_quadrant_l181_181629


namespace egg_production_difference_l181_181312

-- Define the conditions
def last_year_production : ℕ := 1416
def this_year_production : ℕ := 4636

-- Define the theorem statement
theorem egg_production_difference :
  this_year_production - last_year_production = 3220 :=
by
  sorry

end egg_production_difference_l181_181312


namespace find_a_l181_181806

theorem find_a (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 = 180)
  (h2 : x2 = 182)
  (h3 : x3 = 173)
  (h4 : x4 = 175)
  (h6 : x6 = 178)
  (h7 : x7 = 176)
  (h_avg : (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 178) : x5 = 182 := by
  sorry

end find_a_l181_181806


namespace number_of_clients_l181_181511

theorem number_of_clients (num_cars num_selections_per_car num_cars_per_client total_selections num_clients : ℕ)
  (h1 : num_cars = 15)
  (h2 : num_selections_per_car = 3)
  (h3 : num_cars_per_client = 3)
  (h4 : total_selections = num_cars * num_selections_per_car)
  (h5 : num_clients = total_selections / num_cars_per_client) :
  num_clients = 15 := 
by
  sorry

end number_of_clients_l181_181511


namespace find_x_l181_181479

-- Definitions based directly on conditions
def vec_a : ℝ × ℝ := (2, 4)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 3)
def vec_c (x : ℝ) : ℝ × ℝ := (2 - x, 1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematically equivalent proof problem statement
theorem find_x (x : ℝ) : dot_product (vec_c x) (vec_b x) = 0 → (x = -1 ∨ x = 3) :=
by
  -- Placeholder for the proof
  sorry

end find_x_l181_181479


namespace equal_cubes_l181_181795

theorem equal_cubes (a : ℤ) : -(a ^ 3) = (-a) ^ 3 :=
by
  sorry

end equal_cubes_l181_181795


namespace workers_together_time_l181_181839

theorem workers_together_time (hA : ℝ) (hB : ℝ) (jobA_time : hA = 10) (jobB_time : hB = 12) : 
  1 / ((1 / hA) + (1 / hB)) = (60 / 11) :=
by
  -- skipping the proof details
  sorry

end workers_together_time_l181_181839


namespace angle_A_value_l181_181682

/-- 
In triangle ABC, the sides opposite to angles A, B, C are a, b, and c respectively.
Given:
  - C = π / 3,
  - b = √6,
  - c = 3,
Prove that A = 5π / 12.
-/
theorem angle_A_value (a b c : ℝ) (A B C : ℝ) (hC : C = Real.pi / 3) (hb : b = Real.sqrt 6) (hc : c = 3) :
  A = 5 * Real.pi / 12 :=
sorry

end angle_A_value_l181_181682


namespace total_cost_of_suits_l181_181879

theorem total_cost_of_suits : 
    ∃ o t : ℕ, o = 300 ∧ t = 3 * o + 200 ∧ o + t = 1400 :=
by
  sorry

end total_cost_of_suits_l181_181879


namespace ratio_seniors_to_juniors_l181_181943

variable (j s : ℕ)

-- Condition: \(\frac{3}{7}\) of the juniors participated is equal to \(\frac{6}{7}\) of the seniors participated
def participation_condition (j s : ℕ) : Prop :=
  3 * j = 6 * s

-- Theorem to be proved: the ratio of seniors to juniors is \( \frac{1}{2} \)
theorem ratio_seniors_to_juniors (j s : ℕ) (h : participation_condition j s) : s / j = 1 / 2 :=
  sorry

end ratio_seniors_to_juniors_l181_181943


namespace smallest_n_in_range_l181_181101

theorem smallest_n_in_range : ∃ n : ℕ, n > 1 ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 8 = 2) ∧ 120 ≤ n ∧ n ≤ 149 :=
by
  sorry

end smallest_n_in_range_l181_181101


namespace high_heels_height_l181_181449

theorem high_heels_height (x : ℝ) :
  let height := 157
  let lower_limbs := 95
  let golden_ratio := 0.618
  (95 + x) / (157 + x) = 0.618 → x = 5.3 :=
sorry

end high_heels_height_l181_181449


namespace initial_quarters_l181_181593

-- Define the conditions
def quartersAfterLoss (x : ℕ) : ℕ := (4 * x) / 3
def quartersAfterThirdYear (x : ℕ) : ℕ := x - 4
def quartersAfterSecondYear (x : ℕ) : ℕ := x - 36
def quartersAfterFirstYear (x : ℕ) : ℕ := x * 2

-- The main theorem
theorem initial_quarters (x : ℕ) (h1 : quartersAfterLoss x = 140)
    (h2 : quartersAfterThirdYear 140 = 136)
    (h3 : quartersAfterSecondYear 136 = 100)
    (h4 : quartersAfterFirstYear 50 = 100) :
  x = 50 := by
  simp [quartersAfterFirstYear, quartersAfterSecondYear,
        quartersAfterThirdYear, quartersAfterLoss] at *
  sorry

end initial_quarters_l181_181593


namespace circumscribed_sphere_radius_is_3_l181_181741

noncomputable def radius_of_circumscribed_sphere (SA SB SC : ℝ) : ℝ :=
  let space_diagonal := Real.sqrt (SA^2 + SB^2 + SC^2)
  space_diagonal / 2

theorem circumscribed_sphere_radius_is_3 : radius_of_circumscribed_sphere 2 4 4 = 3 :=
by
  unfold radius_of_circumscribed_sphere
  simp
  apply sorry

end circumscribed_sphere_radius_is_3_l181_181741


namespace area_triangle_FQH_l181_181006

open Set

structure Point where
  x : ℝ
  y : ℝ

def Rectangle (A B C D : Point) : Prop :=
  A.x = B.x ∧ C.x = D.x ∧ A.y = D.y ∧ B.y = C.y

def IsMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def AreaTrapezoid (A B C D : Point) : ℝ :=
  0.5 * (B.x - A.x + D.x - C.x) * (A.y - C.y)

def AreaTriangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

variables (E P R H F Q G : Point)

-- Conditions
axiom h1 : Rectangle E F G H
axiom h2 : E.y - P.y = 8
axiom h3 : R.y - H.y = 8
axiom h4 : F.x - E.x = 16
axiom h5 : AreaTrapezoid P R H G = 160

-- Target to prove
theorem area_triangle_FQH : AreaTriangle F Q H = 80 :=
sorry

end area_triangle_FQH_l181_181006


namespace blue_pieces_correct_l181_181755

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end blue_pieces_correct_l181_181755


namespace maximum_special_points_l181_181462

theorem maximum_special_points (n : ℕ) (h : n = 11) : 
  ∃ p : ℕ, p = 91 := 
sorry

end maximum_special_points_l181_181462


namespace sum_of_altitudes_less_than_sum_of_sides_l181_181780

theorem sum_of_altitudes_less_than_sum_of_sides 
  (a b c h_a h_b h_c K : ℝ) 
  (triangle_area : K = (1/2) * a * h_a)
  (h_a_def : h_a = 2 * K / a) 
  (h_b_def : h_b = 2 * K / b)
  (h_c_def : h_c = 2 * K / c) : 
  h_a + h_b + h_c < a + b + c := by
  sorry

end sum_of_altitudes_less_than_sum_of_sides_l181_181780


namespace find_g_neg3_l181_181537

def f (x : ℚ) : ℚ := 4 * x - 6
def g (u : ℚ) : ℚ := 3 * (f u)^2 + 4 * (f u) - 2

theorem find_g_neg3 : g (-3) = 43 / 16 := by
  sorry

end find_g_neg3_l181_181537


namespace multiplier_for_deans_height_l181_181547

theorem multiplier_for_deans_height (h_R : ℕ) (h_R_eq : h_R = 13) (d : ℕ) (d_eq : d = 255) (h_D : ℕ) (h_D_eq : h_D = h_R + 4) : 
  d / h_D = 15 := by
  sorry

end multiplier_for_deans_height_l181_181547


namespace carl_weight_l181_181079

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end carl_weight_l181_181079


namespace keith_apples_correct_l181_181060

def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def total_apples : ℕ := 16
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_apples_correct : keith_apples = 6 := by
  -- the actual proof would go here
  sorry

end keith_apples_correct_l181_181060


namespace min_value_correct_l181_181276

noncomputable def min_value (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)] : ℝ :=
  if x + y = 1 then (a / x + b / y) else 0

theorem min_value_correct (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)]
  (h : x + y = 1) : min_value a b x y = (Real.sqrt a + Real.sqrt b)^2 :=
by
  sorry

end min_value_correct_l181_181276


namespace monotonicity_and_extrema_l181_181688

noncomputable def f (x : ℝ) := (2 * x) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2) ∧
  (f 3 = 5 / 4) ∧
  (f 5 = 3 / 2) :=
by
  sorry

end monotonicity_and_extrema_l181_181688


namespace find_z_l181_181390

variable {x y z w : ℝ}

theorem find_z (h : (1/x) + (1/y) = (1/z) + w) : z = (x * y) / (x + y - w * x * y) :=
by sorry

end find_z_l181_181390


namespace max_median_value_l181_181096

theorem max_median_value (x : ℕ) (h : 198 + x ≤ 392) : x ≤ 194 :=
by {
  sorry
}

end max_median_value_l181_181096


namespace james_total_beverages_l181_181872

-- Define the initial quantities
def initial_sodas := 4 * 10 + 12
def initial_juice_boxes := 3 * 8 + 5
def initial_water_bottles := 2 * 15
def initial_energy_drinks := 7

-- Define the consumption rates
def mon_to_wed_sodas := 3 * 3
def mon_to_wed_juice_boxes := 2 * 3
def mon_to_wed_water_bottles := 1 * 3

def thu_to_sun_sodas := 2 * 4
def thu_to_sun_juice_boxes := 4 * 4
def thu_to_sun_water_bottles := 1 * 4
def thu_to_sun_energy_drinks := 1 * 4

-- Define total beverages consumed
def total_consumed_sodas := mon_to_wed_sodas + thu_to_sun_sodas
def total_consumed_juice_boxes := mon_to_wed_juice_boxes + thu_to_sun_juice_boxes
def total_consumed_water_bottles := mon_to_wed_water_bottles + thu_to_sun_water_bottles
def total_consumed_energy_drinks := thu_to_sun_energy_drinks

-- Define total beverages consumed by the end of the week
def total_beverages_consumed := total_consumed_sodas + total_consumed_juice_boxes + total_consumed_water_bottles + total_consumed_energy_drinks

-- The theorem statement to prove
theorem james_total_beverages : total_beverages_consumed = 50 :=
  by sorry

end james_total_beverages_l181_181872


namespace range_of_m_l181_181335

variable (x y m : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : 2/x + 1/y = 1)
variable (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m)

theorem range_of_m (h1 : 0 < x) (h2 : 0 < y) (h3 : 2/x + 1/y = 1) (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m) : -4 < m ∧ m < 2 := 
sorry

end range_of_m_l181_181335


namespace max_value_of_N_l181_181191

def I_k (k : Nat) : Nat :=
  10^(k + 1) + 32

def N (k : Nat) : Nat :=
  (Nat.factors (I_k k)).count 2

theorem max_value_of_N :
  ∃ k : Nat, N k = 6 ∧ (∀ m : Nat, N m ≤ 6) :=
by
  sorry

end max_value_of_N_l181_181191


namespace monomial_2015_l181_181504

def a (n : ℕ) : ℤ := (-1 : ℤ)^n * (2 * n - 1)

theorem monomial_2015 :
  a 2015 * (x : ℤ) ^ 2015 = -4029 * (x : ℤ) ^ 2015 :=
by
  sorry

end monomial_2015_l181_181504


namespace exponentiation_product_rule_l181_181318

theorem exponentiation_product_rule (a : ℝ) : (3 * a) ^ 2 = 9 * a ^ 2 :=
by
  sorry

end exponentiation_product_rule_l181_181318


namespace bounded_expression_l181_181565

theorem bounded_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := 
sorry

end bounded_expression_l181_181565


namespace sum_of_xy_is_1289_l181_181218

-- Define the variables and conditions
def internal_angle1 (x y : ℕ) : ℕ := 5 * x + 3 * y
def internal_angle2 (x y : ℕ) : ℕ := 3 * x + 20
def internal_angle3 (x y : ℕ) : ℕ := 10 * y + 30

-- Definition of the sum of angles of a triangle
def sum_of_angles (x y : ℕ) : ℕ := internal_angle1 x y + internal_angle2 x y + internal_angle3 x y

-- Define the theorem statement
theorem sum_of_xy_is_1289 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : sum_of_angles x y = 180) : x + y = 1289 :=
by sorry

end sum_of_xy_is_1289_l181_181218


namespace joe_flight_expense_l181_181309

theorem joe_flight_expense
  (initial_amount : ℕ)
  (hotel_expense : ℕ)
  (food_expense : ℕ)
  (remaining_amount : ℕ)
  (flight_expense : ℕ)
  (h1 : initial_amount = 6000)
  (h2 : hotel_expense = 800)
  (h3 : food_expense = 3000)
  (h4 : remaining_amount = 1000)
  (h5 : flight_expense = initial_amount - remaining_amount - hotel_expense - food_expense) :
  flight_expense = 1200 :=
by
  sorry

end joe_flight_expense_l181_181309


namespace fraction_of_white_surface_area_is_11_16_l181_181733

theorem fraction_of_white_surface_area_is_11_16 :
  let cube_surface_area := 6 * 4^2
  let total_surface_faces := 96
  let corner_black_faces := 8 * 3
  let center_black_faces := 6 * 1
  let total_black_faces := corner_black_faces + center_black_faces
  let white_faces := total_surface_faces - total_black_faces
  (white_faces : ℚ) / total_surface_faces = 11 / 16 := 
by sorry

end fraction_of_white_surface_area_is_11_16_l181_181733


namespace weight_loss_percentage_l181_181821

theorem weight_loss_percentage 
  (weight_before weight_after : ℝ) 
  (h_before : weight_before = 800) 
  (h_after : weight_after = 640) : 
  (weight_before - weight_after) / weight_before * 100 = 20 := 
by
  sorry

end weight_loss_percentage_l181_181821


namespace arithmetic_sequence_n_equals_100_l181_181322

theorem arithmetic_sequence_n_equals_100
  (a₁ : ℕ) (d : ℕ) (a_n : ℕ)
  (h₁ : a₁ = 1)
  (h₂ : d = 3)
  (h₃ : a_n = 298) :
  ∃ n : ℕ, a_n = a₁ + (n - 1) * d ∧ n = 100 :=
by
  sorry

end arithmetic_sequence_n_equals_100_l181_181322


namespace quadruple_perimeter_l181_181199

-- Define the rectangle's original and expanded dimensions and perimeters
def original_perimeter (a b : ℝ) := 2 * (a + b)
def new_perimeter (a b : ℝ) := 2 * ((4 * a) + (4 * b))

-- Statement to be proved
theorem quadruple_perimeter (a b : ℝ) : new_perimeter a b = 4 * original_perimeter a b :=
  sorry

end quadruple_perimeter_l181_181199


namespace smallest_sum_l181_181254

-- First, we define the conditions as assumptions:
def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  2 * y = x + z

def is_geometric_sequence (x y z : ℕ) : Prop :=
  y ^ 2 = x * z

-- Given conditions
variables (A B C D : ℕ)
variables (hABC : is_arithmetic_sequence A B C) (hBCD : is_geometric_sequence B C D)
variables (h_ratio : 4 * C = 7 * B)

-- The main theorem to prove
theorem smallest_sum : A + B + C + D = 97 :=
sorry

end smallest_sum_l181_181254


namespace infinite_solutions_l181_181316

theorem infinite_solutions (a : ℤ) (h_a : a > 1) 
  (h_sol : ∃ x y : ℤ, x^2 - a * y^2 = -1) : 
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ, (f n).fst^2 - a * (f n).snd^2 = -1 :=
sorry

end infinite_solutions_l181_181316


namespace floor_add_frac_eq_154_l181_181767

theorem floor_add_frac_eq_154 (r : ℝ) (h : ⌊r⌋ + r = 15.4) : r = 7.4 := 
sorry

end floor_add_frac_eq_154_l181_181767


namespace find_d1_over_d2_l181_181770

variables {k c1 c2 d1 d2 : ℝ}
variables (c1_nonzero : c1 ≠ 0) (c2_nonzero : c2 ≠ 0) 
variables (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0)
variables (h1 : c1 * d1 = k) (h2 : c2 * d2 = k)
variables (h3 : c1 / c2 = 3 / 4)

theorem find_d1_over_d2 : d1 / d2 = 4 / 3 :=
sorry

end find_d1_over_d2_l181_181770


namespace machine_working_time_l181_181744

theorem machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) (h1 : shirts_per_minute = 3) (h2 : total_shirts = 6) :
  (total_shirts / shirts_per_minute) = 2 :=
by
  -- Begin the proof
  sorry

end machine_working_time_l181_181744


namespace trajectory_of_P_l181_181328

def point := ℝ × ℝ

-- Definitions for points A and F, and the circle equation
def A : point := (-1, 0)
def F (x y : ℝ) := (x - 1) ^ 2 + y ^ 2 = 16

-- Main theorem statement: proving the trajectory equation of point P
theorem trajectory_of_P : 
  (∀ (B : point), F B.1 B.2 → 
  (∃ P : point, ∃ (k : ℝ), (P.1 - B.1) * k = -(P.2 - B.2) ∧ (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0)) →
  (∃ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) :=
sorry

end trajectory_of_P_l181_181328


namespace part_a_part_b_l181_181140

/- Part (a) -/
theorem part_a (a b c d : ℝ) (h1 : (a + b ≠ c + d)) (h2 : (a + c ≠ b + d)) (h3 : (a + d ≠ b + c)) :
  ∃ (spheres : ℕ), spheres = 8 := sorry

/- Part (b) -/
theorem part_b (a b c d : ℝ) (h : (a + b = c + d) ∨ (a + c = b + d) ∨ (a + d = b + c)) :
  ∃ (spheres : ℕ), ∀ (n : ℕ), n > 0 → spheres = n := sorry

end part_a_part_b_l181_181140


namespace basin_more_than_tank2_l181_181038

/-- Define the water volumes in milliliters -/
def volume_bottle1 : ℕ := 1000 -- 1 liter = 1000 milliliters
def volume_bottle2 : ℕ := 400  -- 400 milliliters
def volume_tank : ℕ := 2800    -- 2800 milliliters
def volume_basin : ℕ := volume_bottle1 + volume_bottle2 + volume_tank -- total volume in basin
def volume_tank2 : ℕ := 4000 + 100 -- 4 liters 100 milliliters tank

/-- Theorem: The basin can hold 100 ml more water than the 4-liter 100-milliliter tank -/
theorem basin_more_than_tank2 : volume_basin = volume_tank2 + 100 :=
by
  -- This is where the proof would go, but it is not required for this exercise
  sorry

end basin_more_than_tank2_l181_181038


namespace binomial_coeff_sum_l181_181685

theorem binomial_coeff_sum 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h1 : (1 - 2 * 0 : ℝ)^(7) = a_0 + a_1 * 0 + a_2 * 0^2 + a_3 * 0^3 + a_4 * 0^4 + a_5 * 0^5 + a_6 * 0^6 + a_7 * 0^7)
  (h2 : (1 - 2 * 1 : ℝ)^(7) = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5 + a_6 * 1^6 + a_7 * 1^7) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 := 
sorry

end binomial_coeff_sum_l181_181685


namespace fraction_complex_z_l181_181065

theorem fraction_complex_z (z : ℂ) (hz : z = 1 - I) : 2 / z = 1 + I := by
    sorry

end fraction_complex_z_l181_181065


namespace gcd_a_b_eq_one_l181_181098

def a : ℕ := 130^2 + 240^2 + 350^2
def b : ℕ := 131^2 + 241^2 + 349^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l181_181098


namespace plane_equation_l181_181851

variable (x y z : ℝ)

def pointA : ℝ × ℝ × ℝ := (3, 0, 0)
def normalVector : ℝ × ℝ × ℝ := (2, -3, 1)

theorem plane_equation : 
  ∃ a b c d, normalVector = (a, b, c) ∧ pointA = (x, y, z) ∧ a * (x - 3) + b * y + c * z = d ∧ d = -6 := 
  sorry

end plane_equation_l181_181851


namespace difference_of_triangular_2010_2009_l181_181066

def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_of_triangular_2010_2009 :
  triangular 2010 - triangular 2009 = 2010 :=
by
  sorry

end difference_of_triangular_2010_2009_l181_181066


namespace perfect_squares_count_in_range_l181_181914

theorem perfect_squares_count_in_range :
  ∃ (n : ℕ), (
    (∀ (k : ℕ), (50 < k^2 ∧ k^2 < 500) → (8 ≤ k ∧ k ≤ 22)) ∧
    (15 = 22 - 8 + 1)
  ) := sorry

end perfect_squares_count_in_range_l181_181914


namespace fruit_bowl_oranges_l181_181395

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end fruit_bowl_oranges_l181_181395


namespace table_tennis_matches_l181_181182

def num_players : ℕ := 8

def total_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem table_tennis_matches : total_matches num_players = 28 := by
  sorry

end table_tennis_matches_l181_181182


namespace sum_of_factors_36_eq_91_l181_181809

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l181_181809


namespace determine_ω_and_φ_l181_181628

noncomputable def f (x : ℝ) (ω φ : ℝ) := 2 * Real.sin (ω * x + φ)
def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) := (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ d > 0, d < T ∧ ∀ m n : ℤ, m ≠ n → f (m * d) ≠ f (n * d))

theorem determine_ω_and_φ :
  ∃ ω φ : ℝ,
    (0 < ω) ∧
    (|φ| < Real.pi / 2) ∧
    (smallest_positive_period (f ω φ) Real.pi) ∧
    (f 0 ω φ = Real.sqrt 3) ∧
    (ω = 2 ∧ φ = Real.pi / 3) :=
by
  sorry

end determine_ω_and_φ_l181_181628


namespace sum_squares_reciprocal_l181_181883

variable (x y : ℝ)

theorem sum_squares_reciprocal (h₁ : x + y = 12) (h₂ : x * y = 32) :
  (1/x)^2 + (1/y)^2 = 5/64 := by
  sorry

end sum_squares_reciprocal_l181_181883


namespace sandy_spent_home_currency_l181_181148

variable (A B C D : ℝ)

def total_spent_home_currency (A B C D : ℝ) : ℝ :=
  let total_foreign := A + B + C
  total_foreign * D

theorem sandy_spent_home_currency (D : ℝ) : 
  total_spent_home_currency 13.99 12.14 7.43 D = 33.56 * D := 
by
  sorry

end sandy_spent_home_currency_l181_181148


namespace hyperbola_product_slopes_constant_l181_181607

theorem hyperbola_product_slopes_constant (a b x0 y0 : ℝ) (h_a : a > 0) (h_b : b > 0) (hP : (x0 / a) ^ 2 - (y0 / b) ^ 2 = 1) (h_diff_a1_a2 : x0 ≠ a ∧ x0 ≠ -a) :
  (y0 / (x0 + a)) * (y0 / (x0 - a)) = b^2 / a^2 :=
by sorry

end hyperbola_product_slopes_constant_l181_181607


namespace days_in_month_l181_181122

-- The number of days in the month
variable (D : ℕ)

-- The conditions provided in the problem
def mean_daily_profit (D : ℕ) := 350
def mean_first_fifteen_days := 225
def mean_last_fifteen_days := 475
def total_profit := mean_first_fifteen_days * 15 + mean_last_fifteen_days * 15

-- The Lean statement to prove the number of days in the month
theorem days_in_month : D = 30 :=
by
  -- mean_daily_profit(D) * D should be equal to total_profit
  have h : mean_daily_profit D * D = total_profit := sorry
  -- solve for D
  sorry

end days_in_month_l181_181122


namespace fraction_relevant_quarters_l181_181908

-- Define the total number of quarters and the number of relevant quarters
def total_quarters : ℕ := 50
def relevant_quarters : ℕ := 10

-- Define the theorem that states the fraction of relevant quarters is 1/5
theorem fraction_relevant_quarters : (relevant_quarters : ℚ) / total_quarters = 1 / 5 := by
  sorry

end fraction_relevant_quarters_l181_181908


namespace double_x_value_l181_181690

theorem double_x_value (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end double_x_value_l181_181690


namespace smallest_fourth_number_l181_181177

theorem smallest_fourth_number :
  ∃ (a b : ℕ), 145 + 10 * a + b = 4 * (28 + a + b) ∧ 10 * a + b = 35 :=
by
  sorry

end smallest_fourth_number_l181_181177


namespace carrie_pays_l181_181364

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l181_181364


namespace solve_for_x_l181_181716

theorem solve_for_x (x : ℝ) (h : x / 6 = 15 / 10) : x = 9 :=
by
  sorry

end solve_for_x_l181_181716


namespace canoe_stream_speed_l181_181542

theorem canoe_stream_speed (C S : ℝ) (h1 : C - S = 9) (h2 : C + S = 12) : S = 1.5 :=
by
  sorry

end canoe_stream_speed_l181_181542


namespace exists_sequence_l181_181092

theorem exists_sequence (n : ℕ) : ∃ (a : ℕ → ℕ), 
  (∀ i, 1 ≤ i → i < n → (a i > a (i + 1))) ∧
  (∀ i, 1 ≤ i → i < n → (a i ∣ a (i + 1)^2)) ∧
  (∀ i j, 1 ≤ i → 1 ≤ j → i < n → j < n → (i ≠ j → ¬(a i ∣ a j))) :=
sorry

end exists_sequence_l181_181092


namespace x3_plus_y3_values_l181_181894

noncomputable def x_y_satisfy_eqns (x y : ℝ) : Prop :=
  y^2 - 3 = (x - 3)^3 ∧ x^2 - 3 = (y - 3)^2 ∧ x ≠ y

theorem x3_plus_y3_values (x y : ℝ) (h : x_y_satisfy_eqns x y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 :=
  sorry

end x3_plus_y3_values_l181_181894


namespace tan_double_alpha_l181_181222

theorem tan_double_alpha (α : ℝ) (h : ∀ x : ℝ, (3 * Real.sin x + Real.cos x) ≤ (3 * Real.sin α + Real.cos α)) :
  Real.tan (2 * α) = -3 / 4 :=
sorry

end tan_double_alpha_l181_181222


namespace quadrilateral_inequality_l181_181225

theorem quadrilateral_inequality
  (AB AC BD CD: ℝ)
  (h1 : AB + BD ≤ AC + CD)
  (h2 : AB + CD < AC + BD) :
  AB < AC := by
  sorry

end quadrilateral_inequality_l181_181225


namespace number_of_correct_answers_l181_181329

def total_questions := 30
def correct_points := 3
def incorrect_points := -1
def total_score := 78

theorem number_of_correct_answers (x : ℕ) :
  3 * x + incorrect_points * (total_questions - x) = total_score → x = 27 :=
by
  sorry

end number_of_correct_answers_l181_181329


namespace baseball_cards_l181_181321

theorem baseball_cards (cards_per_page new_cards pages : ℕ) (h1 : cards_per_page = 8) (h2 : new_cards = 3) (h3 : pages = 2) : 
  (pages * cards_per_page - new_cards = 13) := by
  sorry

end baseball_cards_l181_181321


namespace episodes_per_season_before_loss_l181_181846

-- Define the given conditions
def initial_total_seasons : ℕ := 12 + 14
def episodes_lost_per_season : ℕ := 2
def remaining_episodes : ℕ := 364
def total_episodes_lost : ℕ := 12 * episodes_lost_per_season + 14 * episodes_lost_per_season
def initial_total_episodes : ℕ := remaining_episodes + total_episodes_lost

-- Define the theorem to prove
theorem episodes_per_season_before_loss : initial_total_episodes / initial_total_seasons = 16 :=
by
  sorry

end episodes_per_season_before_loss_l181_181846


namespace intersection_A_B_l181_181654

def A : Set ℝ := {y | ∃ x : ℝ, y = x ^ (1 / 3)}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_B :
  A ∩ B = {x | x > 1} :=
sorry

end intersection_A_B_l181_181654


namespace algebraic_sum_of_coefficients_l181_181595

open Nat

theorem algebraic_sum_of_coefficients
  (u : ℕ → ℤ)
  (h1 : u 1 = 5)
  (hrec : ∀ n : ℕ, n > 0 → u (n + 1) - u n = 3 + 4 * (n - 1)) :
  (∃ P : ℕ → ℤ, (∀ n, u n = P n) ∧ (P 1 + P 0 = 5)) :=
sorry

end algebraic_sum_of_coefficients_l181_181595


namespace smallest_perfect_square_greater_than_x_l181_181161

theorem smallest_perfect_square_greater_than_x (x : ℤ)
  (h₁ : ∃ k : ℤ, k^2 ≠ x)
  (h₂ : x ≥ 0) :
  ∃ n : ℤ, n^2 > x ∧ ∀ m : ℤ, m^2 > x → n^2 ≤ m^2 :=
sorry

end smallest_perfect_square_greater_than_x_l181_181161


namespace sodium_chloride_solution_l181_181394

theorem sodium_chloride_solution (n y : ℝ) (h1 : n > 30) 
  (h2 : 0.01 * n * n = 0.01 * (n - 8) * (n + y)) : 
  y = 8 * n / (n + 8) :=
sorry

end sodium_chloride_solution_l181_181394


namespace width_of_cistern_is_6_l181_181051

-- Length of the cistern
def length : ℝ := 8

-- Breadth of the water surface
def breadth : ℝ := 1.85

-- Total wet surface area
def total_wet_surface_area : ℝ := 99.8

-- Let w be the width of the cistern
def width (w : ℝ) : Prop :=
  total_wet_surface_area = (length * w) + 2 * (length * breadth) + 2 * (w * breadth)

theorem width_of_cistern_is_6 : width 6 :=
  by
    -- This proof is omitted. The statement asserts that the width is 6 meters.
    sorry

end width_of_cistern_is_6_l181_181051


namespace average_percentage_l181_181392

theorem average_percentage (x : ℝ) : (60 + x + 80) / 3 = 70 → x = 70 :=
by
  intro h
  sorry

end average_percentage_l181_181392


namespace best_model_based_on_R_squared_l181_181331

theorem best_model_based_on_R_squared:
  ∀ (R2_1 R2_2 R2_3 R2_4: ℝ), 
  R2_1 = 0.98 → R2_2 = 0.80 → R2_3 = 0.54 → R2_4 = 0.35 → 
  R2_1 ≥ R2_2 ∧ R2_1 ≥ R2_3 ∧ R2_1 ≥ R2_4 :=
by
  intros R2_1 R2_2 R2_3 R2_4 h1 h2 h3 h4
  sorry

end best_model_based_on_R_squared_l181_181331


namespace required_moles_of_H2O_l181_181418

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end required_moles_of_H2O_l181_181418


namespace Cathy_wins_l181_181526

theorem Cathy_wins (n k : ℕ) (hn : n > 0) (hk : k > 0) : (∃ box_count : ℕ, box_count = 1) :=
  if h : n ≤ 2^(k-1) then
    sorry
  else
    sorry

end Cathy_wins_l181_181526


namespace steve_pencils_left_l181_181221

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end steve_pencils_left_l181_181221


namespace second_percentage_increase_l181_181319

theorem second_percentage_increase :
  ∀ (P : ℝ) (x : ℝ), (P * 1.30 * (1 + x) = P * 1.5600000000000001) → x = 0.2 :=
by
  intros P x h
  sorry

end second_percentage_increase_l181_181319


namespace pears_for_twenty_apples_l181_181753

-- Definitions based on given conditions
variables (a o p : ℕ) -- represent the number of apples, oranges, and pears respectively
variables (k1 k2 : ℕ) -- scaling factors 

-- Conditions as given
axiom ten_apples_five_oranges : 10 * a = 5 * o
axiom three_oranges_four_pears : 3 * o = 4 * p

-- Proving the number of pears Mia can buy for 20 apples
theorem pears_for_twenty_apples : 13 * p ≤ (20 * a) :=
by
  -- Actual proof would go here
  sorry

end pears_for_twenty_apples_l181_181753


namespace sum_of_first_three_cards_l181_181949

theorem sum_of_first_three_cards :
  ∀ (G Y : ℕ → ℕ) (cards : ℕ → ℕ),
  (∀ n, G n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) →
  (∀ n, Y n ∈ ({4, 5, 6, 7, 8} : Set ℕ)) →
  (∀ n, cards (2 * n) = G (cards n) → cards (2 * n + 1) = Y (cards n + 1)) →
  (∀ n, Y n = G (n + 1) ∨ ∃ k, Y n = k * G (n + 1)) →
  (cards 0 + cards 1 + cards 2 = 14) :=
by
  sorry

end sum_of_first_three_cards_l181_181949


namespace intersection_complement_l181_181464

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {1, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_complement_l181_181464


namespace min_value_of_expression_l181_181843

theorem min_value_of_expression (n : ℕ) (h_pos : n > 0) : n = 8 → (n / 2 + 32 / n) = 8 :=
by sorry

end min_value_of_expression_l181_181843


namespace solve_for_x_l181_181360

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end solve_for_x_l181_181360


namespace percentage_x_eq_six_percent_y_l181_181121

variable {x y : ℝ}

theorem percentage_x_eq_six_percent_y (h1 : ∃ P : ℝ, (P / 100) * x = (6 / 100) * y)
  (h2 : (18 / 100) * x = (9 / 100) * y) : 
  ∃ P : ℝ, P = 12 := 
sorry

end percentage_x_eq_six_percent_y_l181_181121


namespace jesse_stamps_l181_181084

variable (A E : Nat)

theorem jesse_stamps :
  E = 3 * A ∧ E + A = 444 → E = 333 :=
by
  sorry

end jesse_stamps_l181_181084


namespace find_x_plus_y_l181_181877

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l181_181877


namespace triangle_area_eq_e_div_4_l181_181243

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

noncomputable def tangent_line (x : ℝ) : ℝ :=
  let k := (Real.exp 1) * (x + 1)
  k * (x - 1) + Real.exp 1

theorem triangle_area_eq_e_div_4 :
  let area := (1 / 2) * Real.exp 1 * (1 / 2)
  area = (Real.exp 1) / 4 :=
by
  sorry

end triangle_area_eq_e_div_4_l181_181243


namespace construct_right_triangle_l181_181408

theorem construct_right_triangle (hypotenuse : ℝ) (ε : ℝ) (h_positive : 0 < ε) (h_less_than_ninety : ε < 90) :
    ∃ α β : ℝ, α + β = 90 ∧ α - β = ε ∧ 45 < α ∧ α < 90 :=
by
  sorry

end construct_right_triangle_l181_181408


namespace point_transformation_l181_181824

theorem point_transformation (a b : ℝ) :
  let P := (a, b)
  let P₁ := (2 * 2 - a, 2 * 3 - b) -- Rotate P 180° counterclockwise around (2, 3)
  let P₂ := (P₁.2, P₁.1)           -- Reflect P₁ about the line y = x
  P₂ = (5, -4) → a - b = 7 :=
by
  intros
  sorry

end point_transformation_l181_181824


namespace sum_of_angles_of_circumscribed_quadrilateral_l181_181310

theorem sum_of_angles_of_circumscribed_quadrilateral
  (EF GH : ℝ)
  (EF_central_angle : EF = 100)
  (GH_central_angle : GH = 120) :
  (EF / 2 + GH / 2) = 70 :=
by
  sorry

end sum_of_angles_of_circumscribed_quadrilateral_l181_181310


namespace range_of_x_for_expression_meaningful_l181_181854

theorem range_of_x_for_expression_meaningful (x : ℝ) :
  (x - 1 > 0 ∧ x ≠ 1) ↔ x > 1 :=
by
  sorry

end range_of_x_for_expression_meaningful_l181_181854


namespace sum_of_remainders_l181_181089

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5) = 5 :=
by
  sorry

end sum_of_remainders_l181_181089


namespace remainder_of_h_x6_l181_181961

def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

noncomputable def remainder_when_h_x6_divided_by_h (x : ℝ) : ℝ :=
  let hx := h x
  let hx6 := h (x^6)
  hx6 - 6 * hx

theorem remainder_of_h_x6 (x : ℝ) : remainder_when_h_x6_divided_by_h x = 6 :=
  sorry

end remainder_of_h_x6_l181_181961


namespace multiplicative_inverse_of_550_mod_4319_l181_181326

theorem multiplicative_inverse_of_550_mod_4319 :
  (48^2 + 275^2 = 277^2) → ((550 * 2208) % 4319 = 1) := by
  intro h
  sorry

end multiplicative_inverse_of_550_mod_4319_l181_181326


namespace douglas_votes_percentage_l181_181106

theorem douglas_votes_percentage 
  (V : ℝ)
  (hx : 0.62 * 2 * V + 0.38 * V = 1.62 * V)
  (hy : 3 * V > 0) : 
  ((1.62 * V) / (3 * V)) * 100 = 54 := 
by
  sorry

end douglas_votes_percentage_l181_181106


namespace B_visits_A_l181_181650

/-- Students A, B, and C were surveyed on whether they have visited cities A, B, and C -/
def student_visits_city (student : Type) (city : Type) : Prop := sorry -- assume there's a definition

variables (A_student B_student C_student : Type) (city_A city_B city_C : Type)

variables 
  -- A's statements
  (A_visits_more_than_B : student_visits_city A_student city_A → ¬ student_visits_city A_student city_B → ∃ city, student_visits_city B_student city ∧ ¬ student_visits_city A_student city)
  (A_not_visit_B : ¬ student_visits_city A_student city_B)
  -- B's statement
  (B_not_visit_C : ¬ student_visits_city B_student city_C)
  -- C's statement
  (all_three_same_city : student_visits_city A_student city_A → student_visits_city B_student city_A → student_visits_city C_student city_A)

theorem B_visits_A : student_visits_city B_student city_A :=
by
  sorry

end B_visits_A_l181_181650


namespace find_n_positive_integers_l181_181998

theorem find_n_positive_integers :
  ∀ n : ℕ, 0 < n →
  (∃ k : ℕ, (n^2 + 11 * n - 4) * n! + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by
  sorry

end find_n_positive_integers_l181_181998


namespace option_b_is_incorrect_l181_181675

theorem option_b_is_incorrect : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end option_b_is_incorrect_l181_181675


namespace arithmetic_sequence_nth_term_l181_181818

theorem arithmetic_sequence_nth_term (a₁ a₂ a₃ : ℤ) (x : ℤ) (n : ℕ)
  (h₁ : a₁ = 3 * x - 4)
  (h₂ : a₂ = 6 * x - 14)
  (h₃ : a₃ = 4 * x + 3)
  (h₄ : ∀ k : ℕ, a₁ + (k - 1) * ((a₂ - a₁) + (a₃ - a₂) / 2) = 3012) :
  n = 247 :=
by {
  -- Proof to be provided
  sorry
}

end arithmetic_sequence_nth_term_l181_181818


namespace class_8_1_total_score_l181_181207

noncomputable def total_score (spirit neatness standard_of_movements : ℝ) 
(weights_spirit weights_neatness weights_standard : ℝ) : ℝ :=
  (spirit * weights_spirit + neatness * weights_neatness + standard_of_movements * weights_standard) / 
  (weights_spirit + weights_neatness + weights_standard)

theorem class_8_1_total_score :
  total_score 8 9 10 2 3 5 = 9.3 :=
by
  sorry

end class_8_1_total_score_l181_181207


namespace tile_D_is_IV_l181_181600

structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

def Tile_I : Tile := ⟨3, 1, 4, 2⟩
def Tile_II : Tile := ⟨2, 3, 1, 5⟩
def Tile_III : Tile := ⟨4, 0, 3, 1⟩
def Tile_IV : Tile := ⟨5, 4, 2, 0⟩

def is_tile_D (t : Tile) : Prop :=
  t.left = 0 ∧ t.top = 5

theorem tile_D_is_IV : is_tile_D Tile_IV :=
  by
    -- skip proof here
    sorry

end tile_D_is_IV_l181_181600


namespace opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l181_181169

def improper_fraction : ℚ := -4/3

theorem opposite_of_fraction : -improper_fraction = 4/3 :=
by sorry

theorem reciprocal_of_fraction : (improper_fraction⁻¹) = -3/4 :=
by sorry

theorem absolute_value_of_fraction : |improper_fraction| = 4/3 :=
by sorry

end opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l181_181169


namespace average_book_width_correct_l181_181992

noncomputable def average_book_width 
  (widths : List ℚ) (number_of_books : ℕ) : ℚ :=
(widths.sum) / number_of_books

theorem average_book_width_correct :
  average_book_width [5, 3/4, 1.5, 3, 7.25, 12] 6 = 59 / 12 := 
  by 
  sorry

end average_book_width_correct_l181_181992


namespace inequality_solution_set_l181_181251

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end inequality_solution_set_l181_181251


namespace probability_A_given_B_l181_181655

namespace ProbabilityProof

def total_parts : ℕ := 100
def A_parts_produced : ℕ := 0
def A_parts_qualified : ℕ := 35
def B_parts_produced : ℕ := 60
def B_parts_qualified : ℕ := 50

def event_A (x : ℕ) : Prop := x ≤ B_parts_qualified + A_parts_qualified
def event_B (x : ℕ) : Prop := x ≤ A_parts_produced

-- Formalizing the probability condition P(A | B) = 7/8, logically this should be revised with practical events.
theorem probability_A_given_B : (event_B x → event_A x) := sorry

end ProbabilityProof

end probability_A_given_B_l181_181655


namespace students_in_johnsons_class_l181_181379

-- Define the conditions as constants/variables
def studentsInFinleysClass : ℕ := 24
def studentsAdditionalInJohnsonsClass : ℕ := 10

-- State the problem as a theorem
theorem students_in_johnsons_class : 
  let halfFinleysClass := studentsInFinleysClass / 2
  let johnsonsClass := halfFinleysClass + studentsAdditionalInJohnsonsClass
  johnsonsClass = 22 :=
by
  sorry

end students_in_johnsons_class_l181_181379


namespace simplest_radical_form_l181_181944

def is_simplest_radical_form (r : ℝ) : Prop :=
  ∀ x : ℝ, x * x = r → ∃ y : ℝ, y * y ≠ r

theorem simplest_radical_form :
   (is_simplest_radical_form 6) :=
by
  sorry

end simplest_radical_form_l181_181944


namespace unique_solution_for_log_problem_l181_181639

noncomputable def log_problem (x : ℝ) :=
  let a := Real.log (x / 2 - 1) / Real.log (x - 11 / 4).sqrt
  let b := 2 * Real.log (x - 11 / 4) / Real.log (x / 2 - 1 / 4)
  let c := Real.log (x / 2 - 1 / 4) / (2 * Real.log (x / 2 - 1))
  a * b * c = 2 ∧ (a = b ∧ c = a + 1)

theorem unique_solution_for_log_problem :
  ∃! x, log_problem x = true := sorry

end unique_solution_for_log_problem_l181_181639


namespace evaluate_expression_l181_181048

theorem evaluate_expression 
    (a b c : ℕ) 
    (ha : a = 7)
    (hb : b = 11)
    (hc : c = 13) :
  let numerator := a^3 * (1 / b - 1 / c) + b^3 * (1 / c - 1 / a) + c^3 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  numerator / denominator = 31 := 
by {
  sorry
}

end evaluate_expression_l181_181048


namespace line_exists_l181_181317

theorem line_exists (x y x' y' : ℝ)
  (h1 : x' = 3 * x + 2 * y + 1)
  (h2 : y' = x + 4 * y - 3) : 
  (∃ A B C : ℝ, A * x + B * y + C = 0 ∧ A * x' + B * y' + C = 0 ∧ 
  ((A = 1 ∧ B = -1 ∧ C = 4) ∨ (A = 4 ∧ B = -8 ∧ C = -5))) :=
sorry

end line_exists_l181_181317


namespace calculation_A_B_l181_181180

theorem calculation_A_B :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 :=
by
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  sorry

end calculation_A_B_l181_181180


namespace number_of_groups_of_oranges_l181_181889

-- Defining the conditions
def total_oranges : ℕ := 356
def oranges_per_group : ℕ := 2

-- The proof statement
theorem number_of_groups_of_oranges : total_oranges / oranges_per_group = 178 := 
by 
  sorry

end number_of_groups_of_oranges_l181_181889


namespace net_profit_calc_l181_181882

theorem net_profit_calc (purchase_price : ℕ) (overhead_percentage : ℝ) (markup : ℝ) 
  (h_pp : purchase_price = 48) (h_op : overhead_percentage = 0.10) (h_markup : markup = 35) :
  let overhead := overhead_percentage * purchase_price
  let net_profit := markup - overhead
  net_profit = 30.20 := by
    sorry

end net_profit_calc_l181_181882


namespace sum_seven_terms_l181_181887

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) - a n = d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- Given condition
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 42

-- Proof statement
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms a S) 
  (h_cond : given_condition a) : 
  S 7 = 98 := 
sorry

end sum_seven_terms_l181_181887


namespace train_number_of_cars_l181_181618

theorem train_number_of_cars (lena_cars : ℕ) (time_counted : ℕ) (total_time : ℕ) 
  (cars_in_train : ℕ)
  (h1 : lena_cars = 8) 
  (h2 : time_counted = 15)
  (h3 : total_time = 210)
  (h4 : (8 / 15 : ℚ) * 210 = 112)
  : cars_in_train = 112 :=
sorry

end train_number_of_cars_l181_181618


namespace edward_spent_13_l181_181792

-- Define the initial amount of money Edward had
def initial_amount : ℕ := 19
-- Define the current amount of money Edward has now
def current_amount : ℕ := 6
-- Define the amount of money Edward spent
def amount_spent : ℕ := initial_amount - current_amount

-- The proof we need to show
theorem edward_spent_13 : amount_spent = 13 := by
  -- The proof goes here.
  sorry

end edward_spent_13_l181_181792


namespace intersection_M_N_l181_181777

open Set

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 0 < x}
def intersection := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l181_181777


namespace certain_fraction_ratio_l181_181407

theorem certain_fraction_ratio :
  (∃ (x y : ℚ), (x / y) / (6 / 5) = (2 / 5) / 0.14285714285714288) →
  (∃ (x y : ℚ), x / y = 84 / 25) := 
  by
    intros h_ratio
    have h_rat := h_ratio
    sorry

end certain_fraction_ratio_l181_181407


namespace sin_C_value_l181_181062

theorem sin_C_value (A B C : Real) (AC BC : Real) (h_AC : AC = 3) (h_BC : BC = 2 * Real.sqrt 3) (h_A : A = 2 * B) :
    let C : Real := Real.pi - A - B
    Real.sin C = Real.sqrt 6 / 9 :=
  sorry

end sin_C_value_l181_181062


namespace number_of_integers_l181_181814

theorem number_of_integers (n : ℤ) : 
  (16 < n^2) → (n^2 < 121) → n = -10 ∨ n = -9 ∨ n = -8 ∨ n = -7 ∨ n = -6 ∨ n = -5 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 := 
by
  sorry

end number_of_integers_l181_181814


namespace correct_calculated_value_l181_181769

theorem correct_calculated_value (x : ℝ) (h : 3 * x - 5 = 103) : x / 3 - 5 = 7 := 
by 
  sorry

end correct_calculated_value_l181_181769


namespace karlsson_weight_l181_181476

variable {F K M : ℕ}

theorem karlsson_weight (h1 : F + K = M + 120) (h2 : K + M = F + 60) : K = 90 := by
  sorry

end karlsson_weight_l181_181476


namespace mark_deposit_amount_l181_181233

-- Define the conditions
def bryans_deposit (M : ℝ) : ℝ := 5 * M - 40
def total_deposit (M : ℝ) : ℝ := M + bryans_deposit M

-- State the theorem
theorem mark_deposit_amount (M : ℝ) (h1: total_deposit M = 400) : M = 73.33 :=
by
  sorry

end mark_deposit_amount_l181_181233


namespace product_consecutive_even_div_48_l181_181704

theorem product_consecutive_even_div_48 (k : ℤ) : 
  (2 * k) * (2 * k + 2) * (2 * k + 4) % 48 = 0 :=
by
  sorry

end product_consecutive_even_div_48_l181_181704


namespace equal_serving_weight_l181_181215

theorem equal_serving_weight (total_weight : ℝ) (num_family_members : ℕ)
  (h1 : total_weight = 13) (h2 : num_family_members = 5) :
  total_weight / num_family_members = 2.6 :=
by
  sorry

end equal_serving_weight_l181_181215


namespace average_and_variance_of_new_data_set_l181_181078

theorem average_and_variance_of_new_data_set
  (avg : ℝ) (var : ℝ) (constant : ℝ)
  (h_avg : avg = 2.8)
  (h_var : var = 3.6)
  (h_const : constant = 60) :
  (avg + constant = 62.8) ∧ (var = 3.6) :=
sorry

end average_and_variance_of_new_data_set_l181_181078


namespace prove_all_perfect_squares_l181_181437

noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k^2 = n

noncomputable def all_distinct (l : List ℕ) : Prop :=
l.Nodup

noncomputable def pairwise_products_are_perfect_squares (l : List ℕ) : Prop :=
∀ i j, i < l.length → j < l.length → i ≠ j → is_perfect_square (l.nthLe i sorry * l.nthLe j sorry)

theorem prove_all_perfect_squares :
  ∀ l : List ℕ, l.length = 25 →
  (∀ x ∈ l, x ≤ 1000 ∧ 0 < x) →
  all_distinct l →
  pairwise_products_are_perfect_squares l →
  ∀ x ∈ l, is_perfect_square x := 
by
  intros l h1 h2 h3 h4
  sorry

end prove_all_perfect_squares_l181_181437


namespace problem_statement_l181_181419

def product_of_first_n (n : ℕ) : ℕ := List.prod (List.range' 1 n)

def sum_of_first_n (n : ℕ) : ℕ := List.sum (List.range' 1 n)

theorem problem_statement : 
  let numerator := product_of_first_n 9  -- product of numbers 1 through 8
  let denominator := sum_of_first_n 9  -- sum of numbers 1 through 8
  numerator / denominator = 1120 :=
by {
  sorry
}

end problem_statement_l181_181419


namespace A_doubles_after_6_months_l181_181228

variable (x : ℕ)

def A_investment_share (x : ℕ) := (3000 * x) + (6000 * (12 - x))
def B_investment_share := 4500 * 12

theorem A_doubles_after_6_months (h : A_investment_share x = B_investment_share) : x = 6 :=
by
  sorry

end A_doubles_after_6_months_l181_181228


namespace hyperbola_intersection_l181_181211

variable (a b c : ℝ) -- positive constants
variables (F1 F2 : (ℝ × ℝ)) -- foci of the hyperbola

-- The positive constants a and b
axiom a_pos : a > 0
axiom b_pos : b > 0

-- The foci are at (-c, 0) and (c, 0)
axiom F1_def : F1 = (-c, 0)
axiom F2_def : F2 = (c, 0)

-- We want to prove that the points (-c, b^2 / a) and (-c, -b^2 / a) are on the hyperbola
theorem hyperbola_intersection :
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ a > 0 ∧ b > 0) →
  ∀ y : ℝ, ∃ y1 y2 : ℝ, (y1 = b^2 / a ∧ y2 = -b^2 / a ∧ 
  ( ( (-c)^2 / a^2) - (y1^2 / b^2) = 1 ∧  (-c)^2 / a^2 - y2^2 / b^2 = 1 ) ) :=
by
  intros h
  sorry

end hyperbola_intersection_l181_181211


namespace f_monotonic_increasing_l181_181439

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

theorem f_monotonic_increasing :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 > x2 → f x1 > f x2 :=
by
  intros x1 x2 hx1 hx2 h
  sorry

end f_monotonic_increasing_l181_181439


namespace students_who_like_yellow_l181_181732

theorem students_who_like_yellow (total_students girls students_like_green girls_like_pink students_like_yellow : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_green = total_students / 2)
  (h3 : girls_like_pink = girls / 3)
  (h4 : girls = 18)
  (h5 : students_like_yellow = total_students - (students_like_green + girls_like_pink)) :
  students_like_yellow = 9 :=
by
  sorry

end students_who_like_yellow_l181_181732


namespace first_place_friend_distance_friend_running_distance_l181_181572

theorem first_place_friend_distance (distance_mina_finish : ℕ) (halfway_condition : ∀ x, x = distance_mina_finish / 2) :
  (∃ y, y = distance_mina_finish / 2) :=
by
  sorry

-- Given conditions
def distance_mina_finish : ℕ := 200
noncomputable def first_place_friend_position := distance_mina_finish / 2

-- The theorem we need to prove
theorem friend_running_distance : first_place_friend_position = 100 :=
by
  sorry

end first_place_friend_distance_friend_running_distance_l181_181572


namespace heath_plants_per_hour_l181_181268

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end heath_plants_per_hour_l181_181268


namespace not_all_pieces_found_l181_181325

theorem not_all_pieces_found (N : ℕ) (petya_tore : ℕ → ℕ) (vasya_tore : ℕ → ℕ) : 
  (∀ n, petya_tore n = n * 5 - n) →
  (∀ n, vasya_tore n = n * 9 - n) →
  1988 = N ∧ (N % 2 = 1) → false :=
by
  intros h_petya h_vasya h
  sorry

end not_all_pieces_found_l181_181325


namespace atomic_number_order_l181_181619

-- Define that elements A, B, C, D, and E are in the same period
variable (A B C D E : Type)

-- Define conditions based on the problem
def highest_valence_oxide_basic (x : Type) : Prop := sorry
def basicity_greater (x y : Type) : Prop := sorry
def gaseous_hydride_stability (x y : Type) : Prop := sorry
def smallest_ionic_radius (x : Type) : Prop := sorry

-- Assume conditions given in the problem
axiom basic_oxides : highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
axiom basicity_order : basicity_greater B A
axiom hydride_stabilities : gaseous_hydride_stability C D
axiom smallest_radius : smallest_ionic_radius E

-- Prove that the order of atomic numbers from smallest to largest is B, A, E, D, C
theorem atomic_number_order :
  ∃ (A B C D E : Type), highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
  ∧ basicity_greater B A ∧ gaseous_hydride_stability C D ∧ smallest_ionic_radius E
  ↔ B = B ∧ A = A ∧ E = E ∧ D = D ∧ C = C := sorry

end atomic_number_order_l181_181619


namespace simplify_expression_l181_181580

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l181_181580


namespace constant_term_equality_l181_181540

theorem constant_term_equality (a : ℝ) 
  (h1 : ∃ T, T = (x : ℝ)^2 + 2 / x ∧ T^9 = 64 * ↑(Nat.choose 9 6)) 
  (h2 : ∃ T, T = (x : ℝ) + a / (x^2) ∧ T^9 = a^3 * ↑(Nat.choose 9 3)):
  a = 4 := 
sorry

end constant_term_equality_l181_181540


namespace carl_teaches_periods_l181_181663

theorem carl_teaches_periods (cards_per_student : ℕ) (students_per_class : ℕ) (pack_cost : ℕ) (amount_spent : ℕ) (cards_per_pack : ℕ) :
  cards_per_student = 10 →
  students_per_class = 30 →
  pack_cost = 3 →
  amount_spent = 108 →
  cards_per_pack = 50 →
  (amount_spent / pack_cost) * cards_per_pack / (cards_per_student * students_per_class) = 6 :=
by
  intros hc hs hp ha hpkg
  /- proof steps would go here -/
  sorry

end carl_teaches_periods_l181_181663


namespace solve_system_of_equations_l181_181509

theorem solve_system_of_equations :
  (∃ x y : ℝ, (x / y + y / x = 173 / 26) ∧ (1 / x + 1 / y = 15 / 26) ∧ ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13))) :=
by
  sorry

end solve_system_of_equations_l181_181509


namespace value_of_a_l181_181808

theorem value_of_a (a : ℕ) (h : ∀ x, ((a - 2) * x > a - 2) ↔ (x < 1)) : a = 0 ∨ a = 1 := by
  sorry

end value_of_a_l181_181808


namespace mary_pizza_order_l181_181963

theorem mary_pizza_order (p e r n : ℕ) (h1 : p = 8) (h2 : e = 7) (h3 : r = 9) :
  n = (r + e) / p → n = 2 :=
by
  sorry

end mary_pizza_order_l181_181963


namespace bicycle_final_price_l181_181283

theorem bicycle_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (h1 : original_price = 200) (h2 : discount1 = 0.4) (h3 : discount2 = 0.2) :
  (original_price * (1 - discount1) * (1 - discount2)) = 96 :=
by
  -- sorry proof here
  sorry

end bicycle_final_price_l181_181283


namespace total_boxes_correct_l181_181136

def boxes_chocolate : ℕ := 2
def boxes_sugar : ℕ := 5
def boxes_gum : ℕ := 2
def total_boxes : ℕ := boxes_chocolate + boxes_sugar + boxes_gum

theorem total_boxes_correct : total_boxes = 9 := by
  sorry

end total_boxes_correct_l181_181136


namespace hillary_climbing_rate_l181_181564

theorem hillary_climbing_rate :
  ∀ (H : ℕ) (Eddy_rate : ℕ) (Hillary_climb : ℕ) (Hillary_descend_rate : ℕ) (pass_time : ℕ) (start_to_summit : ℕ),
    Eddy_rate = 500 →
    Hillary_climb = 4000 →
    Hillary_descend_rate = 1000 →
    pass_time = 6 →
    start_to_summit = 5000 →
    (Hillary_climb + Eddy_rate * pass_time = Hillary_climb + (pass_time - Hillary_climb / H) * Hillary_descend_rate) →
    H = 800 :=
by
  intros H Eddy_rate Hillary_climb Hillary_descend_rate pass_time start_to_summit
  intro h1 h2 h3 h4 h5 h6
  sorry

end hillary_climbing_rate_l181_181564


namespace sqrt_81_eq_9_l181_181973

theorem sqrt_81_eq_9 : Real.sqrt 81 = 9 :=
by
  sorry

end sqrt_81_eq_9_l181_181973


namespace range_of_m_l181_181442

-- Define the discriminant of a quadratic equation
def discriminant(a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Proposition p: The equation x^2 - 2x + m = 0 has two distinct real roots
def p (m : ℝ) : Prop := discriminant 1 (-2) m > 0

-- Proposition q: The function y = (m + 2)x - 1 is monotonically increasing
def q (m : ℝ) : Prop := m + 2 > 0

-- The main theorem stating the conditions and proving the range of m
theorem range_of_m (m : ℝ) (hpq : p m ∨ q m) (hpnq : ¬(p m ∧ q m)) : m ≤ -2 ∨ m ≥ 1 := sorry

end range_of_m_l181_181442


namespace clearance_sale_gain_percent_l181_181269

theorem clearance_sale_gain_percent
  (SP : ℝ := 30)
  (gain_percent : ℝ := 25)
  (discount_percent : ℝ := 10)
  (CP : ℝ := SP/(1 + gain_percent/100)) :
  let Discount := discount_percent / 100 * SP
  let SP_sale := SP - Discount
  let Gain_during_sale := SP_sale - CP
  let Gain_percent_during_sale := (Gain_during_sale / CP) * 100
  Gain_percent_during_sale = 12.5 := 
by
  sorry

end clearance_sale_gain_percent_l181_181269


namespace tank_ratio_two_l181_181920

variable (T1 : ℕ) (F1 : ℕ) (F2 : ℕ) (T2 : ℕ)

-- Assume the given conditions
axiom h1 : T1 = 48
axiom h2 : F1 = T1 / 3
axiom h3 : F1 - 1 = F2 + 3
axiom h4 : T2 = F2 * 2

-- The theorem to prove
theorem tank_ratio_two (h1 : T1 = 48) (h2 : F1 = T1 / 3) (h3 : F1 - 1 = F2 + 3) (h4 : T2 = F2 * 2) : T1 / T2 = 2 := by
  sorry

end tank_ratio_two_l181_181920


namespace exists_x1_x2_l181_181828

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem exists_x1_x2 (a : ℝ) (h : a < 0) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 ≥ f a x2 :=
by
  sorry

end exists_x1_x2_l181_181828


namespace ratio_boysGradeA_girlsGradeB_l181_181735

variable (S G B : ℕ)

-- Given conditions
axiom h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S
axiom h2 : S = B + G

-- Definitions based on conditions
def boys_in_GradeA (B : ℕ) := (2 / 5 : ℚ) * B
def girls_in_GradeB (G : ℕ) := (3 / 5 : ℚ) * G

-- The proof goal
theorem ratio_boysGradeA_girlsGradeB (S G B : ℕ) (h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S) (h2 : S = B + G) :
    boys_in_GradeA B / girls_in_GradeB G = 2 / 9 :=
by
  sorry

end ratio_boysGradeA_girlsGradeB_l181_181735


namespace dot_product_calculation_l181_181135

def vec_a : ℝ × ℝ := (1, 0)
def vec_b : ℝ × ℝ := (2, 3)
def vec_s : ℝ × ℝ := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2)
def vec_t : ℝ × ℝ := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_calculation :
  dot_product vec_s vec_t = -9 := by
  sorry

end dot_product_calculation_l181_181135


namespace jasper_time_l181_181443

theorem jasper_time {omar_time : ℕ} {omar_height : ℕ} {jasper_height : ℕ} 
  (h1 : omar_time = 12)
  (h2 : omar_height = 240)
  (h3 : jasper_height = 600)
  (h4 : ∃ t : ℕ, t = (jasper_height * omar_time) / (3 * omar_height))
  : t = 10 :=
by sorry

end jasper_time_l181_181443


namespace oranges_in_total_l181_181294

def number_of_boxes := 3
def oranges_per_box := 8
def total_oranges := 24

theorem oranges_in_total : number_of_boxes * oranges_per_box = total_oranges := 
by {
  -- sorry skips the proof part
  sorry 
}

end oranges_in_total_l181_181294


namespace cost_of_traveling_roads_is_2600_l181_181123

-- Define the lawn, roads, and the cost parameters
def width_lawn : ℝ := 80
def length_lawn : ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 2

-- Area calculations
def area_road_1 : ℝ := road_width * length_lawn
def area_road_2 : ℝ := road_width * width_lawn
def area_intersection : ℝ := road_width * road_width

def total_area_roads : ℝ := area_road_1 + area_road_2 - area_intersection

def total_cost : ℝ := total_area_roads * cost_per_sq_meter

theorem cost_of_traveling_roads_is_2600 :
  total_cost = 2600 :=
by
  sorry

end cost_of_traveling_roads_is_2600_l181_181123


namespace evaluate_expression_l181_181127

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 7) :
  (x^5 + 3 * y^3) / 9 = 141 :=
by
  sorry

end evaluate_expression_l181_181127


namespace total_enemies_l181_181554

theorem total_enemies (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) (enemies_defeated : ℕ) :  
  (3 = points_per_enemy) → 
  (12 = points_earned) → 
  (2 = enemies_left) → 
  (points_earned / points_per_enemy = enemies_defeated) → 
  (enemies_defeated + enemies_left = 6) := 
by
  intros
  sorry

end total_enemies_l181_181554


namespace sequence_periodic_l181_181188

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodic (a : ℕ → ℝ) (m_0 : ℕ) (h : sequence_condition a) :
  ∀ m ≥ m_0, a (m + 9) = a m := 
sorry

end sequence_periodic_l181_181188


namespace minimum_value_l181_181016

variable {x : ℝ}

theorem minimum_value (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
by
  sorry

end minimum_value_l181_181016


namespace price_per_ton_max_tons_l181_181271

variable (x y m : ℝ)

def conditions := x = y + 100 ∧ 2 * x + y = 1700

theorem price_per_ton (h : conditions x y) : x = 600 ∧ y = 500 :=
  sorry

def budget_conditions := 10 * (600 - 100) + 1 * 500 ≤ 5600

theorem max_tons (h : budget_conditions) : 600 * m + 500 * (10 - m) ≤ 5600 → m ≤ 6 :=
  sorry

end price_per_ton_max_tons_l181_181271


namespace find_k_l181_181311

theorem find_k (x : ℝ) (k : ℝ) (h : 2 * x - 3 = 3 * x - 2 + k) (h_solution : x = 2) : k = -3 := by
  sorry

end find_k_l181_181311


namespace find_a_2016_l181_181282

-- Given definition for the sequence sum
def sequence_sum (n : ℕ) : ℕ := n * n

-- Definition for a_n using the given sequence sum
def term (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

-- Stating the theorem that we need to prove
theorem find_a_2016 : term 2016 = 4031 := 
by 
  sorry

end find_a_2016_l181_181282


namespace colin_speed_l181_181142

noncomputable def B : Real := 1
noncomputable def T : Real := 2 * B
noncomputable def Br : Real := (1/3) * T
noncomputable def C : Real := 6 * Br

theorem colin_speed : C = 4 := by
  sorry

end colin_speed_l181_181142


namespace island_of_misfortune_l181_181116

def statement (n : ℕ) (knight : ℕ → Prop) (liar : ℕ → Prop) : Prop :=
  ∀ k : ℕ, k < n → (
    if k = 0 then ∀ m : ℕ, (m % 2 = 1) ↔ liar m
    else if k = 1 then ∀ m : ℕ, (m % 3 = 1) ↔ liar m
    else ∀ m : ℕ, (m % (k + 1) = 1) ↔ liar m
  )

theorem island_of_misfortune :
  ∃ n : ℕ, n >= 2 ∧ statement n knight liar
:= sorry

end island_of_misfortune_l181_181116


namespace max_nine_multiple_l181_181058

theorem max_nine_multiple {a b c n : ℕ} (h1 : Prime a) (h2 : Prime b) (h3 : Prime c) (h4 : 3 < a) (h5 : 3 < b) (h6 : 3 < c) (h7 : 2 * a + 5 * b = c) : 9 ∣ (a + b + c) :=
sorry

end max_nine_multiple_l181_181058


namespace log_base_property_l181_181867

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x / log a

theorem log_base_property
  (a : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (hf9 : f a 9 = 2) :
  f a (3^a) = 3 :=
by
  sorry

end log_base_property_l181_181867


namespace fewest_four_dollar_frisbees_l181_181014

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 196) : y = 4 :=
by
  sorry

end fewest_four_dollar_frisbees_l181_181014


namespace eiffel_tower_vs_burj_khalifa_l181_181304

-- Define the heights of the structures
def height_eiffel_tower : ℕ := 324
def height_burj_khalifa : ℕ := 830

-- Define the statement to be proven
theorem eiffel_tower_vs_burj_khalifa :
  height_burj_khalifa - height_eiffel_tower = 506 :=
by
  sorry

end eiffel_tower_vs_burj_khalifa_l181_181304


namespace natives_cannot_obtain_910_rupees_with_50_coins_l181_181711

theorem natives_cannot_obtain_910_rupees_with_50_coins (x y z : ℤ) : 
  x + y + z = 50 → 
  10 * x + 34 * y + 62 * z = 910 → 
  false :=
by
  sorry

end natives_cannot_obtain_910_rupees_with_50_coins_l181_181711


namespace find_real_numbers_l181_181698

theorem find_real_numbers (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end find_real_numbers_l181_181698


namespace circle_radius_triple_area_l181_181403

noncomputable def circle_radius (n : ℝ) : ℝ :=
  let r := (n * (Real.sqrt 3 + 1)) / 2
  r

theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) :
  r = (n * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end circle_radius_triple_area_l181_181403


namespace room_length_l181_181422

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sq_meter : ℝ) (length : ℝ) : 
  width = 3.75 ∧ total_cost = 14437.5 ∧ cost_per_sq_meter = 700 → length = 5.5 :=
by
  sorry

end room_length_l181_181422


namespace water_flow_into_sea_per_minute_l181_181546

noncomputable def river_flow_rate_kmph : ℝ := 4
noncomputable def river_depth_m : ℝ := 5
noncomputable def river_width_m : ℝ := 19
noncomputable def hours_to_minutes : ℝ := 60
noncomputable def km_to_m : ℝ := 1000

noncomputable def flow_rate_m_per_min : ℝ := (river_flow_rate_kmph * km_to_m) / hours_to_minutes
noncomputable def cross_sectional_area_m2 : ℝ := river_depth_m * river_width_m
noncomputable def volume_per_minute_m3 : ℝ := cross_sectional_area_m2 * flow_rate_m_per_min

theorem water_flow_into_sea_per_minute :
  volume_per_minute_m3 = 6333.65 := by 
  -- Proof would go here
  sorry

end water_flow_into_sea_per_minute_l181_181546


namespace highest_elevation_l181_181585

-- Define the function for elevation as per the conditions
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2

-- Prove that the highest elevation reached is 500 meters
theorem highest_elevation : (exists t : ℝ, elevation t = 500) ∧ (∀ t : ℝ, elevation t ≤ 500) := sorry

end highest_elevation_l181_181585


namespace johns_weekly_allowance_l181_181029

theorem johns_weekly_allowance
    (A : ℝ)
    (h1 : ∃ A, (4/15) * A = 0.64) :
    A = 2.40 :=
by
  sorry

end johns_weekly_allowance_l181_181029


namespace find_blue_balls_l181_181208

theorem find_blue_balls 
  (B : ℕ)
  (red_balls : ℕ := 7)
  (green_balls : ℕ := 4)
  (prob_red_red : ℚ := 7 / 40) -- 0.175 represented as a rational number
  (h : (21 / ((11 + B) * (10 + B) / 2 : ℚ)) = prob_red_red) :
  B = 5 :=
sorry

end find_blue_balls_l181_181208


namespace angle_B_in_triangle_tan_A_given_c_eq_3a_l181_181434

theorem angle_B_in_triangle (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) : B = π / 3 := 
sorry

theorem tan_A_given_c_eq_3a (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) (h2 : c = 3 * a) : 
(Real.tan A) = Real.sqrt 3 / 5 :=
sorry

end angle_B_in_triangle_tan_A_given_c_eq_3a_l181_181434


namespace successful_multiplications_in_one_hour_l181_181159

variable (multiplications_per_second : ℕ)
variable (error_rate_percentage : ℕ)

theorem successful_multiplications_in_one_hour
  (h1 : multiplications_per_second = 15000)
  (h2 : error_rate_percentage = 5)
  : (multiplications_per_second * 3600 * (100 - error_rate_percentage) / 100) 
    + (multiplications_per_second * 3600 * error_rate_percentage / 100) = 54000000 := by
  sorry

end successful_multiplications_in_one_hour_l181_181159


namespace solve_equation_l181_181740

theorem solve_equation (x : ℝ) : 2 * (x - 2)^2 = 6 - 3 * x ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_equation_l181_181740


namespace ellipse_foci_distance_l181_181640

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l181_181640


namespace side_salad_cost_l181_181147

theorem side_salad_cost (T S : ℝ)
  (h1 : T + S + 4 + 2 = 2 * T) 
  (h2 : (T + S + 4 + 2) + T = 24) : S = 2 :=
by
  sorry

end side_salad_cost_l181_181147


namespace division_of_fractions_l181_181634

theorem division_of_fractions : (1 / 6) / (1 / 3) = 1 / 2 :=
by
  sorry

end division_of_fractions_l181_181634


namespace cleaning_task_sequences_correct_l181_181272

section ChemistryClass

-- Total number of students
def total_students : ℕ := 15

-- Number of classes in a week
def classes_per_week : ℕ := 5

-- Calculate the number of valid sequences of task assignments
def num_valid_sequences : ℕ := total_students * (total_students - 1) * (total_students - 2) * (total_students - 3) * (total_students - 4)

theorem cleaning_task_sequences_correct :
  num_valid_sequences = 360360 :=
by
  unfold num_valid_sequences
  norm_num
  sorry

end ChemistryClass

end cleaning_task_sequences_correct_l181_181272


namespace odd_phone_calls_are_even_l181_181805

theorem odd_phone_calls_are_even (n : ℕ) : Even (2 * n) :=
by
  sorry

end odd_phone_calls_are_even_l181_181805


namespace find_abc_l181_181035

theorem find_abc
  (a b c : ℝ)
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := 
by 
sorry

end find_abc_l181_181035


namespace standard_equation_of_circle_l181_181606

-- Definitions based on problem conditions
def center : ℝ × ℝ := (-1, 2)
def radius : ℝ := 2

-- Lean statement of the problem
theorem standard_equation_of_circle :
  ∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = radius ^ 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end standard_equation_of_circle_l181_181606


namespace trig_eqn_solution_l181_181237

open Real

theorem trig_eqn_solution (x : ℝ) (n : ℤ) :
  sin x ≠ 0 →
  cos x ≠ 0 →
  sin x + cos x ≥ 0 →
  (sqrt (1 + tan x) = sin x + cos x) →
  ∃ k : ℤ, (x = k * π + π / 4) ∨ (x = k * π - π / 4) ∨ (x = (2 * k * π + 3 * π / 4)) :=
by
  sorry

end trig_eqn_solution_l181_181237


namespace interest_rate_is_five_percent_l181_181055

-- Define the problem parameters
def principal : ℝ := 1200
def amount_after_period : ℝ := 1344
def time_period : ℝ := 2.4

-- Define the simple interest formula
def interest (P R T : ℝ) : ℝ := P * R * T

-- The goal is to prove that the rate of interest is 5% per year
theorem interest_rate_is_five_percent :
  ∃ R, interest principal R time_period = amount_after_period - principal ∧ R = 0.05 :=
by
  sorry

end interest_rate_is_five_percent_l181_181055


namespace fraction_sum_identity_l181_181953

theorem fraction_sum_identity (p q r : ℝ) (h₀ : p ≠ q) (h₁ : p ≠ r) (h₂ : q ≠ r) 
(h : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := 
sorry

end fraction_sum_identity_l181_181953


namespace arun_brother_weight_upper_limit_l181_181722

theorem arun_brother_weight_upper_limit (w : ℝ) (X : ℝ) 
  (h1 : 61 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < X)
  (h3 : w ≤ 64)
  (h4 : ((62 + 63 + 64) / 3) = 63) :
  X = 64 :=
by
  sorry

end arun_brother_weight_upper_limit_l181_181722


namespace ratio_of_doctors_to_nurses_l181_181536

def total_staff : ℕ := 250
def nurses : ℕ := 150
def doctors : ℕ := total_staff - nurses

theorem ratio_of_doctors_to_nurses : 
  (doctors : ℚ) / (nurses : ℚ) = 2 / 3 := by
  sorry

end ratio_of_doctors_to_nurses_l181_181536


namespace average_last_4_matches_l181_181905

theorem average_last_4_matches (avg_10: ℝ) (avg_6: ℝ) (total_matches: ℕ) (first_matches: ℕ) :
  avg_10 = 38.9 → avg_6 = 42 → total_matches = 10 → first_matches = 6 → 
  (avg_10 * total_matches - avg_6 * first_matches) / (total_matches - first_matches) = 34.25 :=
by 
  intros h1 h2 h3 h4
  sorry

end average_last_4_matches_l181_181905


namespace johnny_needs_45_planks_l181_181942

theorem johnny_needs_45_planks
  (legs_per_table : ℕ)
  (planks_per_leg : ℕ)
  (surface_planks_per_table : ℕ)
  (number_of_tables : ℕ)
  (h1 : legs_per_table = 4)
  (h2 : planks_per_leg = 1)
  (h3 : surface_planks_per_table = 5)
  (h4 : number_of_tables = 5) :
  number_of_tables * (legs_per_table * planks_per_leg + surface_planks_per_table) = 45 :=
by
  sorry

end johnny_needs_45_planks_l181_181942


namespace problem1_problem2_l181_181372

theorem problem1 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : Real.exp x + Real.exp y > 2 * Real.exp 1 :=
by {
  sorry -- Proof goes here
}

theorem problem2 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : x * Real.log x + y * Real.log y > 0 :=
by {
  sorry -- Proof goes here
}

end problem1_problem2_l181_181372


namespace find_p_value_l181_181164

open Set

/-- Given the parabola C: y^2 = 2px with p > 0, point A(0, sqrt(3)),
    and point B on the parabola such that AB is perpendicular to AF,
    and |BF| = 4. Determine the value of p. -/
theorem find_p_value (p : ℝ) (h : p > 0) :
  ∃ p, p = 2 ∨ p = 6 :=
sorry

end find_p_value_l181_181164


namespace multiply_fractions_l181_181790

theorem multiply_fractions :
  (1 / 3 : ℚ) * (3 / 5) * (5 / 6) = 1 / 6 :=
by
  sorry

end multiply_fractions_l181_181790


namespace largest_angle_in_triangle_l181_181668

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end largest_angle_in_triangle_l181_181668


namespace min_sum_real_possible_sums_int_l181_181376

-- Lean 4 statement for the real numbers case
theorem min_sum_real (x y : ℝ) (hx : x + y + 2 * x * y = 5) (hx_pos : x > 0) (hy_pos : y > 0) :
  x + y ≥ Real.sqrt 11 - 1 := 
sorry

-- Lean 4 statement for the integers case
theorem possible_sums_int (x y : ℤ) (hx : x + y + 2 * x * y = 5) :
  x + y = 5 ∨ x + y = -7 :=
sorry

end min_sum_real_possible_sums_int_l181_181376


namespace problem_divisible_by_480_l181_181420

theorem problem_divisible_by_480 (a : ℕ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) : ∃ k : ℕ, a * (a^2 - 1) * (a^2 - 4) = 480 * k :=
by
  sorry

end problem_divisible_by_480_l181_181420


namespace range_of_a_l181_181267

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + 4

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = 0 → (f a x = 0 → x > 0)) ↔ a > 3 := sorry

end range_of_a_l181_181267


namespace angle_D_in_triangle_DEF_l181_181373

theorem angle_D_in_triangle_DEF 
  (E F D : ℝ) 
  (hEF : F = 3 * E) 
  (hE : E = 15) 
  (h_sum_angles : D + E + F = 180) : D = 120 :=
by
  -- Proof goes here
  sorry

end angle_D_in_triangle_DEF_l181_181373


namespace exchanges_divisible_by_26_l181_181139

variables (p a d : ℕ) -- Define the variables for the number of exchanges

theorem exchanges_divisible_by_26 (t : ℕ) (h1 : p = 4 * a + d) (h2 : p = a + 5 * d) :
  ∃ k : ℕ, a + p + d = 26 * k :=
by {
  -- Replace these sorry placeholders with the actual proof where needed
  sorry
}

end exchanges_divisible_by_26_l181_181139


namespace find_a_l181_181747

-- Define the hyperbola equation and the asymptote conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / 9) = 1

def asymptote1 (x y : ℝ) : Prop := 3 * x + 2 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Prove that if asymptote conditions hold, a = 2
theorem find_a (a : ℝ) (ha : a > 0) :
  (∀ x y, asymptote1 x y) ∧ (∀ x y, asymptote2 x y) → a = 2 :=
sorry

end find_a_l181_181747


namespace remainder_of_E_div_88_l181_181544

-- Define the given expression E and the binomial coefficient 
noncomputable def E : ℤ :=
  1 - 90 * Nat.choose 10 1 + 90 ^ 2 * Nat.choose 10 2 - 90 ^ 3 * Nat.choose 10 3 + 
  90 ^ 4 * Nat.choose 10 4 - 90 ^ 5 * Nat.choose 10 5 + 90 ^ 6 * Nat.choose 10 6 - 
  90 ^ 7 * Nat.choose 10 7 + 90 ^ 8 * Nat.choose 10 8 - 90 ^ 9 * Nat.choose 10 9 + 
  90 ^ 10 * Nat.choose 10 10

-- The theorem that we need to prove
theorem remainder_of_E_div_88 : E % 88 = 1 := by
  sorry

end remainder_of_E_div_88_l181_181544


namespace sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l181_181197

theorem sufficient_but_not_necessary_condition_x_gt_5_x_gt_3 :
  ∀ x : ℝ, (x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ x ≤ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l181_181197


namespace f_monotonicity_g_min_l181_181125

-- Definitions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * a ^ x - 2 * a ^ (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a ^ (2 * x) + a ^ (-2 * x) - 2 * f x a

-- Conditions
variable {a : ℝ} 
variable (a_pos : 0 < a) (a_ne_one : a ≠ 1) (f_one : f 1 a = 3) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3)

-- Monotonicity of f(x)
theorem f_monotonicity : 
  (∀ x y, x < y → f x a < f y a) ∨ (∀ x y, x < y → f y a < f x a) :=
sorry

-- Minimum value of g(x)
theorem g_min : ∃ x' : ℝ, 0 ≤ x' ∧ x' ≤ 3 ∧ g x' a = -2 :=
sorry

end f_monotonicity_g_min_l181_181125


namespace daily_shoppers_correct_l181_181562

noncomputable def daily_shoppers (P : ℝ) : Prop :=
  let weekly_taxes : ℝ := 6580
  let daily_taxes := weekly_taxes / 7
  let percent_taxes := 0.94
  percent_taxes * P = daily_taxes

theorem daily_shoppers_correct : ∃ P : ℝ, daily_shoppers P ∧ P = 1000 :=
by
  sorry

end daily_shoppers_correct_l181_181562


namespace alec_votes_l181_181900

variable (students totalVotes goalVotes neededVotes : ℕ)

theorem alec_votes (h1 : students = 60)
                   (h2 : goalVotes = 3 * students / 4)
                   (h3 : totalVotes = students / 2 + 5 + (students - (students / 2 + 5)) / 5)
                   (h4 : neededVotes = goalVotes - totalVotes) :
                   neededVotes = 5 :=
by sorry

end alec_votes_l181_181900


namespace car_speed_l181_181375

theorem car_speed (d t : ℝ) (h_d : d = 624) (h_t : t = 3) : d / t = 208 := by
  sorry

end car_speed_l181_181375


namespace sqrt_nested_expression_l181_181864

theorem sqrt_nested_expression : 
  Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := 
by
  sorry

end sqrt_nested_expression_l181_181864


namespace cost_of_cookies_l181_181785

theorem cost_of_cookies (diane_has : ℕ) (needs_more : ℕ) (cost : ℕ) :
  diane_has = 27 → needs_more = 38 → cost = 65 :=
by
  sorry

end cost_of_cookies_l181_181785


namespace cost_per_pie_eq_l181_181651

-- We define the conditions
def price_per_piece : ℝ := 4
def pieces_per_pie : ℕ := 3
def pies_per_hour : ℕ := 12
def actual_revenue : ℝ := 138

-- Lean theorem statement
theorem cost_per_pie_eq : (price_per_piece * pieces_per_pie * pies_per_hour - actual_revenue) / pies_per_hour = 0.50 := by
  -- Proof would go here
  sorry

end cost_per_pie_eq_l181_181651


namespace range_of_a_l181_181000

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ -x^2 + 4*x + a = 0) ↔ (-3 ≤ a ∧ a ≤ 21) :=
by
  sorry

end range_of_a_l181_181000


namespace coordinate_relationship_l181_181350

theorem coordinate_relationship (x y : ℝ) (h : |x| - |y| = 0) : (|x| - |y| = 0) :=
by
    sorry

end coordinate_relationship_l181_181350


namespace positive_integers_divisors_of_2_to_the_n_plus_1_l181_181831

theorem positive_integers_divisors_of_2_to_the_n_plus_1:
  ∀ n : ℕ, 0 < n → (n^2 ∣ 2^n + 1) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end positive_integers_divisors_of_2_to_the_n_plus_1_l181_181831


namespace find_number_l181_181475

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l181_181475


namespace monthly_interest_payment_l181_181362

theorem monthly_interest_payment (P : ℝ) (R : ℝ) (monthly_payment : ℝ)
  (hP : P = 28800) (hR : R = 0.09) : 
  monthly_payment = (P * R) / 12 :=
by
  sorry

end monthly_interest_payment_l181_181362


namespace Sophie_donuts_l181_181539

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end Sophie_donuts_l181_181539


namespace big_cows_fewer_than_small_cows_l181_181070

theorem big_cows_fewer_than_small_cows (b s : ℕ) (h1 : b = 6) (h2 : s = 7) : 
  (s - b) / s = 1 / 7 :=
by
  sorry

end big_cows_fewer_than_small_cows_l181_181070


namespace compute_expression_l181_181354

theorem compute_expression :
  (4 + 8 - 16 + 32 + 64 - 128 + 256) / (8 + 16 - 32 + 64 + 128 - 256 + 512) = 1 / 2 :=
by
  sorry

end compute_expression_l181_181354


namespace binding_cost_is_correct_l181_181568

-- Definitions for the conditions used in the problem
def total_cost : ℝ := 250      -- Total cost to copy and bind 10 manuscripts
def copy_cost_per_page : ℝ := 0.05   -- Cost per page to copy
def pages_per_manuscript : ℕ := 400  -- Number of pages in each manuscript
def num_manuscripts : ℕ := 10      -- Number of manuscripts

-- The target value we want to prove
def binding_cost_per_manuscript : ℝ := 5 

-- The theorem statement proving the binding cost per manuscript
theorem binding_cost_is_correct :
  let copy_cost_per_manuscript := pages_per_manuscript * copy_cost_per_page
  let total_copy_cost := num_manuscripts * copy_cost_per_manuscript
  let total_binding_cost := total_cost - total_copy_cost
  (total_binding_cost / num_manuscripts) = binding_cost_per_manuscript :=
by
  sorry

end binding_cost_is_correct_l181_181568


namespace range_of_m_l181_181684

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + m * x + 2 * m - 3 < 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l181_181684


namespace probability_of_shaded_triangle_l181_181163

def total_triangles : ℕ := 9
def shaded_triangles : ℕ := 3

theorem probability_of_shaded_triangle :
  total_triangles > 5 →
  (shaded_triangles : ℚ) / total_triangles = 1 / 3 :=
by
  intros h
  -- proof here
  sorry

end probability_of_shaded_triangle_l181_181163


namespace determine_p_range_l181_181575

theorem determine_p_range :
  ∀ (p : ℝ), (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = (x + 9 / 8) * (x + 9 / 8) ∧ (f x) = (8*x^2 + 18*x + 4*p)/8 ) →
  2.5 < p ∧ p < 2.6 :=
by
  sorry

end determine_p_range_l181_181575


namespace common_difference_divisible_by_p_l181_181783

variable (a : ℕ → ℕ) (p : ℕ)

-- Define that the sequence a is an arithmetic progression with common difference d
def is_arithmetic_progression (d : ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

-- Define that the sequence a is strictly increasing
def is_increasing_arithmetic_progression : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

-- Define that all elements a_i are prime numbers
def all_primes : Prop :=
  ∀ i : ℕ, Nat.Prime (a i)

-- Define that the first element of the sequence is greater than p
def first_element_greater_than_p : Prop :=
  a 1 > p

-- Combining all conditions
def conditions (d : ℕ) : Prop :=
  is_arithmetic_progression a d ∧ is_increasing_arithmetic_progression a ∧ all_primes a ∧ first_element_greater_than_p a p ∧ Nat.Prime p

-- Statement to prove: common difference is divisible by p
theorem common_difference_divisible_by_p (d : ℕ) (h : conditions a p d) : p ∣ d :=
sorry

end common_difference_divisible_by_p_l181_181783


namespace fraction_meaningful_l181_181345

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ∃ y, y = 1 / (x - 1) :=
by
  sorry

end fraction_meaningful_l181_181345


namespace road_network_possible_l181_181039

theorem road_network_possible (n : ℕ) :
  (n = 6 → true) ∧ (n = 1986 → false) :=
by {
  -- Proof of the statement goes here.
  sorry
}

end road_network_possible_l181_181039


namespace min_value_of_squares_l181_181002

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : a^2 + b^2 + c^2 ≥ t^2 / 3 :=
sorry

end min_value_of_squares_l181_181002


namespace find_n_eq_l181_181945

theorem find_n_eq : 
  let a := 2^4
  let b := 3^3
  ∃ (n : ℤ), a - 7 = b + n :=
by
  let a := 2^4
  let b := 3^3
  use -18
  sorry

end find_n_eq_l181_181945


namespace opposite_of_neg_four_l181_181715

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end opposite_of_neg_four_l181_181715


namespace leaks_drain_time_l181_181474

-- Definitions from conditions
def pump_rate : ℚ := 1 / 2 -- tanks per hour
def leak1_rate : ℚ := 1 / 6 -- tanks per hour
def leak2_rate : ℚ := 1 / 9 -- tanks per hour

-- Proof statement
theorem leaks_drain_time : (leak1_rate + leak2_rate)⁻¹ = 3.6 :=
by
  sorry

end leaks_drain_time_l181_181474


namespace interval_a_b_l181_181677

noncomputable def f (x : ℝ) : ℝ := |Real.log (x - 1)|

theorem interval_a_b (a b : ℝ) (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : x1 < x2) (h3 : x2 < b) (h4 : f x1 > f x2) :
  a < 2 := 
sorry

end interval_a_b_l181_181677


namespace students_taking_neither_l181_181853

def total_students : ℕ := 1200
def music_students : ℕ := 60
def art_students : ℕ := 80
def sports_students : ℕ := 30
def music_and_art_students : ℕ := 25
def music_and_sports_students : ℕ := 15
def art_and_sports_students : ℕ := 20
def all_three_students : ℕ := 10

theorem students_taking_neither :
  total_students - (music_students + art_students + sports_students 
  - music_and_art_students - music_and_sports_students - art_and_sports_students 
  + all_three_students) = 1080 := sorry

end students_taking_neither_l181_181853


namespace average_mpg_correct_l181_181413

noncomputable def average_mpg (initial_miles final_miles : ℕ) (refill1 refill2 refill3 : ℕ) : ℚ :=
  let distance := final_miles - initial_miles
  let total_gallons := refill1 + refill2 + refill3
  distance / total_gallons

theorem average_mpg_correct :
  average_mpg 32000 33100 15 10 22 = 23.4 :=
by
  sorry

end average_mpg_correct_l181_181413


namespace triangle_XYZ_median_inequalities_l181_181699

theorem triangle_XYZ_median_inequalities :
  ∀ (XY XZ : ℝ), 
  (∀ (YZ : ℝ), YZ = 10 → 
  ∀ (XM : ℝ), XM = 6 → 
  ∃ (x : ℝ), x = (XY + XZ - 20)/4 → 
  ∃ (N n : ℝ), 
  N = 192 ∧ n = 92 → 
  N - n = 100) :=
by sorry

end triangle_XYZ_median_inequalities_l181_181699


namespace min_value_f_l181_181799

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 1/2 * Real.cos (2 * x) - 1

theorem min_value_f : ∃ x : ℝ, f x = -5/2 := sorry

end min_value_f_l181_181799


namespace time_after_2023_minutes_l181_181970

def start_time : Nat := 1 * 60 -- Start time is 1:00 a.m. in minutes from midnight, which is 60 minutes.
def elapsed_time : Nat := 2023 -- The elapsed time is 2023 minutes.

theorem time_after_2023_minutes : (start_time + elapsed_time) % 1440 = 643 := 
by
  -- 1440 represents the total minutes in a day (24 hours * 60 minutes).
  -- 643 represents the time 10:43 a.m. in minutes from midnight. This is obtained as 10 * 60 + 43 = 643.
  sorry

end time_after_2023_minutes_l181_181970


namespace bounds_of_F_and_G_l181_181784

noncomputable def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem bounds_of_F_and_G {a b c : ℝ}
  (hF0 : |F a b c 0| ≤ 1)
  (hF1 : |F a b c 1| ≤ 1)
  (hFm1 : |F a b c (-1)| ≤ 1) :
  (∀ x, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by
  sorry

end bounds_of_F_and_G_l181_181784


namespace trevor_eggs_l181_181484

theorem trevor_eggs :
  let gertrude := 4
  let blanche := 3
  let nancy := 2
  let martha := 2
  let ophelia := 5
  let penelope := 1
  let quinny := 3
  let dropped := 2
  let gifted := 3
  let total_collected := gertrude + blanche + nancy + martha + ophelia + penelope + quinny
  let remaining_after_drop := total_collected - dropped
  let final_eggs := remaining_after_drop - gifted
  final_eggs = 15 := by
    sorry

end trevor_eggs_l181_181484


namespace units_digit_sum_cubes_l181_181930

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l181_181930


namespace triangle_number_placement_l181_181587

theorem triangle_number_placement
  (A B C D E F : ℕ)
  (h1 : A + B + C = 6)
  (h2 : D = 5)
  (h3 : E = 6)
  (h4 : D + E + F = 14)
  (h5 : B = 3) : 
  (A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4) :=
by {
  sorry
}

end triangle_number_placement_l181_181587


namespace find_angle_and_perimeter_l181_181266

open Real

variables {A B C a b c : ℝ}

/-- If (2a - c)sinA + (2c - a)sinC = 2bsinB in triangle ABC -/
theorem find_angle_and_perimeter
  (h1 : (2 * a - c) * sin A + (2 * c - a) * sin C = 2 * b * sin B)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (b_eq : b = 1) :
  B = π / 3 ∧ (sqrt 3 + 1 < a + b + c ∧ a + b + c ≤ 3) :=
sorry

end find_angle_and_perimeter_l181_181266


namespace p_necessary_for_q_l181_181280

-- Definitions
def p (a b : ℝ) : Prop := (a + b = 2) ∨ (a + b = -2)
def q (a b : ℝ) : Prop := a + b = 2

-- Statement of the problem
theorem p_necessary_for_q (a b : ℝ) : (p a b → q a b) ∧ ¬(q a b → p a b) := 
sorry

end p_necessary_for_q_l181_181280


namespace total_cookies_baked_l181_181012

def cookies_baked_yesterday : ℕ := 435
def cookies_baked_today : ℕ := 139

theorem total_cookies_baked : cookies_baked_yesterday + cookies_baked_today = 574 := by
  sorry

end total_cookies_baked_l181_181012


namespace solve_x_y_l181_181339

theorem solve_x_y (x y : ℝ) (h1 : x^2 + y^2 = 16 * x - 10 * y + 14) (h2 : x - y = 6) : 
  x + y = 3 := 
by 
  sorry

end solve_x_y_l181_181339


namespace simplify_expression_l181_181658

theorem simplify_expression (c : ℤ) : (3 * c + 6 - 6 * c) / 3 = -c + 2 := by
  sorry

end simplify_expression_l181_181658


namespace polygon_area_is_14_l181_181099

def vertices : List (ℕ × ℕ) :=
  [(1, 2), (2, 2), (3, 3), (3, 4), (4, 5), (5, 5), (6, 5), (6, 4), (5, 3),
   (4, 3), (4, 2), (3, 1), (2, 1), (1, 1)]

noncomputable def area_of_polygon (vs : List (ℕ × ℕ)) : ℝ := sorry

theorem polygon_area_is_14 :
  area_of_polygon vertices = 14 := sorry

end polygon_area_is_14_l181_181099


namespace area_of_triangle_l181_181166

noncomputable def circumradius (a b c : ℝ) (α : ℝ) : ℝ := a / (2 * Real.sin α)

theorem area_of_triangle (A B C a b c R : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * R)
  (h₂ : a = 2) (h₃ : b + c = 4) : 
  1 / 2 * b * (c * Real.sin A) = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l181_181166


namespace dans_average_rate_l181_181355

/-- Dan's average rate for the entire trip, given the conditions, equals 0.125 miles per minute --/
theorem dans_average_rate :
  ∀ (d_run d_swim : ℝ) (r_run r_swim : ℝ) (time_run time_swim : ℝ),
  d_run = 3 ∧ d_swim = 3 ∧ r_run = 10 ∧ r_swim = 6 ∧ 
  time_run = (d_run / r_run) * 60 ∧ time_swim = (d_swim / r_swim) * 60 →
  ((d_run + d_swim) / (time_run + time_swim)) = 0.125 :=
by
  intros d_run d_swim r_run r_swim time_run time_swim h
  sorry

end dans_average_rate_l181_181355


namespace stickers_after_loss_l181_181880

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l181_181880


namespace trigonometric_identity_l181_181428

theorem trigonometric_identity (α : ℝ) : 
  - (Real.sin α) + (Real.sqrt 3) * (Real.cos α) = 2 * (Real.sin (α + 2 * Real.pi / 3)) :=
by
  sorry

end trigonometric_identity_l181_181428


namespace find_positive_integers_divisors_l181_181852

theorem find_positive_integers_divisors :
  ∃ n_list : List ℕ, 
    (∀ n ∈ n_list, n > 0 ∧ (n * (n + 1)) / 2 ∣ 10 * n) ∧ n_list.length = 5 :=
sorry

end find_positive_integers_divisors_l181_181852


namespace cos_double_angle_l181_181468

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 1/4) : Real.cos (2 * theta) = -7/8 :=
by
  sorry

end cos_double_angle_l181_181468


namespace taco_castle_num_dodge_trucks_l181_181869

theorem taco_castle_num_dodge_trucks
  (D F T V H C : ℕ)
  (hV : V = 5)
  (h1 : F = D / 3)
  (h2 : F = 2 * T)
  (h3 : V = T / 2)
  (h4 : H = 3 * F / 4)
  (h5 : C = 2 * H / 3) :
  D = 60 :=
by
  sorry

end taco_castle_num_dodge_trucks_l181_181869


namespace evaluate_expression_l181_181406

variable (b x : ℝ)

theorem evaluate_expression (h : x = b + 9) : x - b + 4 = 13 := by
  sorry

end evaluate_expression_l181_181406


namespace tangent_identity_l181_181925

theorem tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2)
  = ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) :=
sorry

end tangent_identity_l181_181925


namespace find_special_three_digit_numbers_l181_181543

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l181_181543


namespace exists_permutation_with_large_neighbor_difference_l181_181478

theorem exists_permutation_with_large_neighbor_difference :
  ∃ (σ : Fin 100 → Fin 100), 
    (∀ (i : Fin 99), (|σ i.succ - σ i| ≥ 50)) :=
sorry

end exists_permutation_with_large_neighbor_difference_l181_181478


namespace inequality_solution_set_l181_181746

theorem inequality_solution_set (a b : ℝ) (h1 : a = -2) (h2 : b = 1) :
  {x : ℝ | |2 * x + a| + |x - b| < 6} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_set_l181_181746


namespace simplify_expression_l181_181974

theorem simplify_expression :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90 - 18) * (90 + 18)) / ((120 - 9) * (120 + 9)) = 1 := by
  sorry

end simplify_expression_l181_181974


namespace inequality_relation_l181_181404

open Real

theorem inequality_relation (x : ℝ) :
  ¬ ((∀ x, (x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
     (∀ x, (x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) := 
by
  sorry

end inequality_relation_l181_181404


namespace power_difference_divisible_by_35_l181_181734

theorem power_difference_divisible_by_35 (n : ℕ) : (3^(6*n) - 2^(6*n)) % 35 = 0 := 
by sorry

end power_difference_divisible_by_35_l181_181734


namespace janet_more_siblings_than_carlos_l181_181171

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end janet_more_siblings_than_carlos_l181_181171


namespace probability_correct_l181_181750

-- Definition for the total number of ways to select topics
def total_ways : ℕ := 6 * 6

-- Definition for the number of ways two students select different topics
def different_topics_ways : ℕ := 6 * 5

-- Definition for the probability of selecting different topics
def probability_different_topics : ℚ := different_topics_ways / total_ways

-- The statement to be proved in Lean
theorem probability_correct :
  probability_different_topics = 5 / 6 := 
sorry

end probability_correct_l181_181750


namespace circle_equation_l181_181341

-- Definitions for the given conditions
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-1, 1)
def line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 2 = 0

-- Theorem statement for the proof problem
theorem circle_equation :
  ∃ (h k : ℝ), line (h, k) ∧ (h = 1) ∧ (k = 1) ∧
  ((h - 1)^2 + (k - 1)^2 = 4) :=
sorry

end circle_equation_l181_181341


namespace trips_and_weights_l181_181133

theorem trips_and_weights (x : ℕ) (w : ℕ) (trips_Bill Jean_total limit_total: ℕ)
  (h1 : x + (x + 6) = 40)
  (h2 : trips_Bill = x)
  (h3 : Jean_total = x + 6)
  (h4 : w = 7850)
  (h5 : limit_total = 8000)
  : 
  trips_Bill = 17 ∧ 
  Jean_total = 23 ∧ 
  (w : ℝ) / 40 = 196.25 := 
by 
  sorry

end trips_and_weights_l181_181133


namespace rational_root_theorem_l181_181194

theorem rational_root_theorem :
  (∃ x : ℚ, 3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0)
  → (x = 1 ∨ x = 1/3) := by
  sorry

end rational_root_theorem_l181_181194


namespace trapezoid_lower_side_length_l181_181219

variable (U L : ℝ) (height area : ℝ)

theorem trapezoid_lower_side_length
  (h1 : L = U - 3.4)
  (h2 : height = 5.2)
  (h3 : area = 100.62)
  (h4 : area = (1 / 2) * (U + L) * height) :
  L = 17.65 :=
by
  sorry

end trapezoid_lower_side_length_l181_181219


namespace condition_sufficiency_l181_181087

theorem condition_sufficiency (x₁ x₂ : ℝ) :
  (x₁ > 4 ∧ x₂ > 4) → (x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧ ¬ ((x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) → (x₁ > 4 ∧ x₂ > 4)) :=
by 
  sorry

end condition_sufficiency_l181_181087


namespace sum_of_three_consecutive_even_nums_l181_181991

theorem sum_of_three_consecutive_even_nums : 80 + 82 + 84 = 246 := by
  sorry

end sum_of_three_consecutive_even_nums_l181_181991


namespace exists_multiple_of_prime_with_all_nines_digits_l181_181487

theorem exists_multiple_of_prime_with_all_nines_digits (p : ℕ) (hp_prime : Nat.Prime p) (h2 : p ≠ 2) (h5 : p ≠ 5) :
  ∃ n : ℕ, (∀ d ∈ (n.digits 10), d = 9) ∧ p ∣ n :=
by
  sorry

end exists_multiple_of_prime_with_all_nines_digits_l181_181487


namespace triangle_inequality_equality_condition_l181_181834

theorem triangle_inequality (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 :=
sorry

theorem equality_condition (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  (a = b) ∧ (b = c) :=
sorry

end triangle_inequality_equality_condition_l181_181834


namespace six_by_six_board_partition_l181_181561

theorem six_by_six_board_partition (P : Prop) (Q : Prop) 
(board : ℕ × ℕ) (domino : ℕ × ℕ) 
(h1 : board = (6, 6)) 
(h2 : domino = (2, 1)) 
(h3 : P → Q ∧ Q → P) :
  ∃ R₁ R₂ : ℕ × ℕ, (R₁ = (p, q) ∧ R₂ = (r, s) ∧ ((R₁.1 * R₁.2 + R₂.1 * R₂.2) = 36)) :=
sorry

end six_by_six_board_partition_l181_181561


namespace find_particular_number_l181_181301

def particular_number (x : ℕ) : Prop :=
  (2 * (67 - (x / 23))) = 102

theorem find_particular_number : particular_number 2714 :=
by {
  sorry
}

end find_particular_number_l181_181301


namespace sequence_type_l181_181551

-- Definitions based on the conditions
def Sn (a : ℝ) (n : ℕ) : ℝ := a^n - 1

def sequence_an (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a - 1 else (Sn a n - Sn a (n - 1))

-- Proving the mathematical statement
theorem sequence_type (a : ℝ) (h : a ≠ 0) : 
  (∀ n > 1, (sequence_an a n = sequence_an a 1 + (n - 1) * (sequence_an a 2 - sequence_an a 1)) ∨
  (∀ n > 2, sequence_an a n / sequence_an a (n-1) = a)) :=
sorry

end sequence_type_l181_181551


namespace min_product_sum_l181_181398

theorem min_product_sum (a : Fin 7 → ℕ) (b : Fin 7 → ℕ) 
  (h2 : ∀ i, 2 ≤ a i) 
  (h3 : ∀ i, a i ≤ 166) 
  (h4 : ∀ i, a i ^ b i % 167 = a (i + 1) % 7 + 1 ^ 2 % 167) : 
  b 0 * b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * (b 0 + b 1 + b 2 + b 3 + b 4 + b 5 + b 6) = 675 := sorry

end min_product_sum_l181_181398


namespace total_workers_count_l181_181549

theorem total_workers_count 
  (W N : ℕ)
  (h1 : (W : ℝ) * 9000 = 7 * 12000 + N * 6000)
  (h2 : W = 7 + N) 
  : W = 14 :=
sorry

end total_workers_count_l181_181549


namespace intersection_P_Q_intersection_complementP_Q_l181_181444

-- Define the universal set U
def U := Set.univ (ℝ)

-- Define set P
def P := {x : ℝ | |x| > 2}

-- Define set Q
def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Complement of P with respect to U
def complement_P : Set ℝ := {x : ℝ | |x| ≤ 2}

theorem intersection_P_Q : P ∩ Q = ({x : ℝ | 2 < x ∧ x < 3}) :=
by {
  sorry
}

theorem intersection_complementP_Q : complement_P ∩ Q = ({x : ℝ | 1 < x ∧ x ≤ 2}) :=
by {
  sorry
}

end intersection_P_Q_intersection_complementP_Q_l181_181444


namespace eva_total_marks_correct_l181_181071

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l181_181071


namespace least_expensive_trip_is_1627_44_l181_181835

noncomputable def least_expensive_trip_cost : ℝ :=
  let distance_DE := 4500
  let distance_DF := 4000
  let distance_EF := Real.sqrt (distance_DE ^ 2 - distance_DF ^ 2)
  let cost_bus (distance : ℝ) : ℝ := distance * 0.20
  let cost_plane (distance : ℝ) : ℝ := distance * 0.12 + 120
  let cost_DE := min (cost_bus distance_DE) (cost_plane distance_DE)
  let cost_EF := min (cost_bus distance_EF) (cost_plane distance_EF)
  let cost_DF := min (cost_bus distance_DF) (cost_plane distance_DF)
  cost_DE + cost_EF + cost_DF

theorem least_expensive_trip_is_1627_44 :
  least_expensive_trip_cost = 1627.44 := sorry

end least_expensive_trip_is_1627_44_l181_181835


namespace smallest_number_condition_l181_181248

theorem smallest_number_condition 
  (x : ℕ) 
  (h1 : ∃ k : ℕ, x - 6 = k * 12)
  (h2 : ∃ k : ℕ, x - 6 = k * 16)
  (h3 : ∃ k : ℕ, x - 6 = k * 18)
  (h4 : ∃ k : ℕ, x - 6 = k * 21)
  (h5 : ∃ k : ℕ, x - 6 = k * 28)
  (h6 : ∃ k : ℕ, x - 6 = k * 35)
  (h7 : ∃ k : ℕ, x - 6 = k * 39) 
  : x = 65526 :=
sorry

end smallest_number_condition_l181_181248


namespace find_p_when_q_is_1_l181_181262

-- Define the proportionality constant k and the relationship
variables {k p q : ℝ}
def inversely_proportional (k q p : ℝ) : Prop := p = k / (q + 2)

-- Given conditions
theorem find_p_when_q_is_1 (h1 : inversely_proportional k 4 1) : 
  inversely_proportional k 1 2 :=
by 
  sorry

end find_p_when_q_is_1_l181_181262


namespace gcd_triples_l181_181206

theorem gcd_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  gcd a 20 = b ∧ gcd b 15 = c ∧ gcd a c = 5 ↔
  ∃ t : ℕ, t > 0 ∧ 
    ((a = 20 * t ∧ b = 20 ∧ c = 5) ∨ 
     (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨ 
     (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by
  sorry

end gcd_triples_l181_181206


namespace find_c_value_l181_181396

theorem find_c_value (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 4) (h3 : x2 = 5) (h4 : y2 = 0) (c : ℝ)
  (h5 : 3 * ((x1 + x2) / 2) - 2 * ((y1 + y2) / 2) = c) : c = 5 :=
sorry

end find_c_value_l181_181396


namespace union_A_B_union_complement_A_B_l181_181909

open Set

-- Definitions for sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {3, 5}

-- Statement 1: Prove that A ∪ B = {1, 3, 5, 7}
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by
  sorry

-- Definition for complement of A in U
def complement_A_U : Set ℕ := {x ∈ U | x ∉ A}

-- Statement 2: Prove that (complement of A in U) ∪ B = {2, 3, 4, 5, 6}
theorem union_complement_A_B : complement_A_U ∪ B = {2, 3, 4, 5, 6} := by
  sorry

end union_A_B_union_complement_A_B_l181_181909


namespace point_not_on_line_l181_181416

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : ¬ ((2023, 0) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) :=
by
  -- proof is omitted
  sorry

end point_not_on_line_l181_181416


namespace tangent_addition_l181_181370

open Real

theorem tangent_addition (x : ℝ) (h : tan x = 3) :
  tan (x + π / 6) = - (5 * (sqrt 3 + 3)) / 3 := by
  -- Providing a brief outline of the proof steps is not necessary for the statement
  sorry

end tangent_addition_l181_181370


namespace eagles_points_l181_181277

theorem eagles_points (x y : ℕ) (h₁ : x + y = 82) (h₂ : x - y = 18) : y = 32 :=
sorry

end eagles_points_l181_181277


namespace sufficient_condition_for_perpendicular_l181_181960

variables {Plane : Type} {Line : Type} 
variables (α β γ : Plane) (m n : Line)

-- Definitions based on conditions
variables (perpendicular : Plane → Plane → Prop)
variables (perpendicular_line : Line → Plane → Prop)
variables (intersection : Plane → Plane → Line)

-- Conditions from option D
variable (h1 : perpendicular_line n α)
variable (h2 : perpendicular_line n β)
variable (h3 : perpendicular_line m α)

-- Statement to prove
theorem sufficient_condition_for_perpendicular (h1 : perpendicular_line n α)
  (h2 : perpendicular_line n β) (h3 : perpendicular_line m α) : 
  perpendicular_line m β := 
sorry

end sufficient_condition_for_perpendicular_l181_181960


namespace video_game_map_width_l181_181616

theorem video_game_map_width (volume length height : ℝ) (h1 : volume = 50)
                            (h2 : length = 5) (h3 : height = 2) :
  ∃ width : ℝ, volume = length * width * height ∧ width = 5 :=
by
  sorry

end video_game_map_width_l181_181616


namespace goods_train_passes_man_in_10_seconds_l181_181459

def goods_train_pass_time (man_speed_kmph goods_speed_kmph goods_length_m : ℕ) : ℕ :=
  let relative_speed_mps := (man_speed_kmph + goods_speed_kmph) * 1000 / 3600
  goods_length_m / relative_speed_mps

theorem goods_train_passes_man_in_10_seconds :
  goods_train_pass_time 55 60 320 = 10 := sorry

end goods_train_passes_man_in_10_seconds_l181_181459


namespace converse_and_inverse_l181_181088

-- Definitions
def is_circle (s : Type) : Prop := sorry
def has_no_corners (s : Type) : Prop := sorry

-- Converse Statement
def converse_false (s : Type) : Prop :=
  has_no_corners s → is_circle s → False

-- Inverse Statement
def inverse_true (s : Type) : Prop :=
  ¬ is_circle s → ¬ has_no_corners s

-- Main Proof Problem
theorem converse_and_inverse (s : Type) :
  (converse_false s) ∧ (inverse_true s) := sorry

end converse_and_inverse_l181_181088


namespace min_value_fraction_l181_181258

theorem min_value_fraction (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) : 
  ∃ a, (∀ x y, (x - 1 ≥ 0) ∧ (x - y + 1 ≤ 0) ∧ (x + y - 4 ≤ 0) → (x / (y + 1)) ≥ a) ∧ 
      (a = 1 / 4) :=
sorry

end min_value_fraction_l181_181258
