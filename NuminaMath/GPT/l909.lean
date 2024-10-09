import Mathlib

namespace actual_revenue_percentage_l909_90921

def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.25 * R
def actual_revenue (R : ℝ) := 0.75 * R

theorem actual_revenue_percentage (R : ℝ) : 
  (actual_revenue R / projected_revenue R) * 100 = 60 :=
by
  sorry

end actual_revenue_percentage_l909_90921


namespace tony_fish_after_ten_years_l909_90998

theorem tony_fish_after_ten_years :
  let initial_fish := 6
  let x := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  let y := [4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
  (List.foldl (fun acc ⟨add, die⟩ => acc + add - die) initial_fish (List.zip x y)) = 34 := 
by
  sorry

end tony_fish_after_ten_years_l909_90998


namespace five_twos_make_24_l909_90951

theorem five_twos_make_24 :
  ∃ a b c d e : ℕ, a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  ((a + b + c) * (d + e) = 24) :=
by
  sorry

end five_twos_make_24_l909_90951


namespace program_arrangement_possible_l909_90953

theorem program_arrangement_possible (initial_programs : ℕ) (additional_programs : ℕ) 
  (h_initial: initial_programs = 6) (h_additional: additional_programs = 2) : 
  ∃ arrangements, arrangements = 56 :=
by
  sorry

end program_arrangement_possible_l909_90953


namespace alternative_plan_cost_is_eleven_l909_90944

-- Defining current cost
def current_cost : ℕ := 12

-- Defining the alternative plan cost in terms of current cost
def alternative_cost : ℕ := current_cost - 1

-- Theorem stating the alternative cost is $11
theorem alternative_plan_cost_is_eleven : alternative_cost = 11 :=
by
  -- This is the proof, which we are skipping with sorry
  sorry

end alternative_plan_cost_is_eleven_l909_90944


namespace value_of_g_l909_90911

-- Defining the function g and its property
def g (x : ℝ) : ℝ := 5

-- Theorem to prove g(x - 3) = 5 for any real number x
theorem value_of_g (x : ℝ) : g (x - 3) = 5 := by
  sorry

end value_of_g_l909_90911


namespace chess_tournament_participants_l909_90974

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 := by
  sorry

end chess_tournament_participants_l909_90974


namespace abs_value_expression_l909_90906

theorem abs_value_expression (m n : ℝ) (h1 : m < 0) (h2 : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 :=
sorry

end abs_value_expression_l909_90906


namespace contrapositive_of_square_inequality_l909_90992

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x^2 > y^2 → x > y) ↔ (x ≤ y → x^2 ≤ y^2) :=
by
  sorry

end contrapositive_of_square_inequality_l909_90992


namespace axis_of_symmetry_parabola_l909_90964

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), ∃ y : ℝ, y = (x - 5)^2 → x = 5 := 
by 
  sorry

end axis_of_symmetry_parabola_l909_90964


namespace correct_option_l909_90972

-- Define the variable 'a' as a real number
variable (a : ℝ)

-- Define propositions for each option
def option_A : Prop := 5 * a ^ 2 - 4 * a ^ 2 = 1
def option_B : Prop := (a ^ 7) / (a ^ 4) = a ^ 3
def option_C : Prop := (a ^ 3) ^ 2 = a ^ 5
def option_D : Prop := a ^ 2 * a ^ 3 = a ^ 6

-- State the main proposition asserting that option B is correct and others are incorrect
theorem correct_option :
  option_B a ∧ ¬option_A a ∧ ¬option_C a ∧ ¬option_D a :=
  by sorry

end correct_option_l909_90972


namespace sequence_a_100_l909_90988

theorem sequence_a_100 : 
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + 2 * n) ∧ a 100 = 9902) :=
by
  sorry

end sequence_a_100_l909_90988


namespace smallest_k_for_polygon_l909_90943

-- Definitions and conditions
def equiangular_decagon_interior_angle : ℝ := 144

-- Question transformation into a proof problem
theorem smallest_k_for_polygon (k : ℕ) (hk : k > 1) :
  (∀ (n2 : ℕ), n2 = 10 * k → ∃ (interior_angle : ℝ), interior_angle = k * equiangular_decagon_interior_angle ∧
  n2 ≥ 3) → k = 2 :=
by
  sorry

end smallest_k_for_polygon_l909_90943


namespace four_distinct_real_roots_l909_90979

noncomputable def f (x c : ℝ) : ℝ := x^2 + 4 * x + c

-- We need to prove that if c is in the interval (-1, 3), f(f(x)) has exactly 4 distinct real roots
theorem four_distinct_real_roots (c : ℝ) : (-1 < c) ∧ (c < 3) → 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) 
  ∧ (f (f x₁ c) c = 0 ∧ f (f x₂ c) c = 0 ∧ f (f x₃ c) c = 0 ∧ f (f x₄ c) c = 0) :=
by sorry

end four_distinct_real_roots_l909_90979


namespace value_of_expression_l909_90914

theorem value_of_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) : x + 2 * y = 10 :=
by
  -- Proof goes here
  sorry

end value_of_expression_l909_90914


namespace distance_not_all_odd_l909_90952

theorem distance_not_all_odd (A B C D : ℝ × ℝ) : 
  ∃ (P Q : ℝ × ℝ), dist P Q % 2 = 0 := by sorry

end distance_not_all_odd_l909_90952


namespace largest_two_digit_divisible_by_6_ending_in_4_l909_90954

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l909_90954


namespace solve_inequality_l909_90929

theorem solve_inequality 
  (k_0 k b m n : ℝ)
  (hM1 : -1 = k_0 * m + b) (hM2 : -1 = k^2 / m)
  (hN1 : 2 = k_0 * n + b) (hN2 : 2 = k^2 / n) :
  {x : ℝ | x^2 > k_0 * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
  sorry

end solve_inequality_l909_90929


namespace smallest_pos_integer_n_l909_90924

theorem smallest_pos_integer_n 
  (x y : ℤ)
  (hx: ∃ k : ℤ, x = 8 * k - 2)
  (hy : ∃ l : ℤ, y = 8 * l + 2) :
  ∃ n : ℤ, n > 0 ∧ ∃ (m : ℤ), x^2 - x*y + y^2 + n = 8 * m ∧ n = 4 := by
  sorry

end smallest_pos_integer_n_l909_90924


namespace morio_current_age_l909_90960

-- Given conditions
def teresa_current_age : ℕ := 59
def morio_age_when_michiko_born : ℕ := 38
def teresa_age_when_michiko_born : ℕ := 26

-- Definitions derived from the conditions
def michiko_age : ℕ := teresa_current_age - teresa_age_when_michiko_born

-- Statement to prove Morio's current age
theorem morio_current_age : (michiko_age + morio_age_when_michiko_born) = 71 :=
by
  sorry

end morio_current_age_l909_90960


namespace bee_loss_rate_l909_90923

theorem bee_loss_rate (initial_bees : ℕ) (days : ℕ) (remaining_bees : ℕ) :
  initial_bees = 80000 → 
  days = 50 → 
  remaining_bees = initial_bees / 4 → 
  (initial_bees - remaining_bees) / days = 1200 :=
by
  intros h₁ h₂ h₃
  sorry

end bee_loss_rate_l909_90923


namespace breadth_of_boat_l909_90937

theorem breadth_of_boat :
  ∀ (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (rho : ℝ),
    L = 8 → h = 0.01 → m = 160 → g = 9.81 → rho = 1000 →
    (L * 2 * h = (m * g) / (rho * g)) :=
by
  intros L h m g rho hL hh hm hg hrho
  sorry

end breadth_of_boat_l909_90937


namespace relationship_between_y_values_l909_90931

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_values_l909_90931


namespace system_of_equations_solution_l909_90982

theorem system_of_equations_solution :
  ∃ x y : ℝ, 7 * x - 3 * y = 2 ∧ 2 * x + y = 8 ∧ x = 2 ∧ y = 4 :=
by
  use 2
  use 4
  sorry

end system_of_equations_solution_l909_90982


namespace intersection_in_quadrants_I_and_II_l909_90983

open Set

def in_quadrants_I_and_II (x y : ℝ) : Prop :=
  (0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)

theorem intersection_in_quadrants_I_and_II :
  ∀ (x y : ℝ),
    y > 3 * x → y > -2 * x + 3 → in_quadrants_I_and_II x y :=
by
  intros x y h1 h2
  sorry

end intersection_in_quadrants_I_and_II_l909_90983


namespace percentage_honda_red_l909_90938

theorem percentage_honda_red (total_cars : ℕ) (honda_cars : ℕ) (percentage_red_total : ℚ)
  (percentage_red_non_honda : ℚ) (percentage_red_honda : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  percentage_red_total = 0.60 →
  percentage_red_non_honda = 0.225 →
  percentage_red_honda = 0.90 →
  ((honda_cars * percentage_red_honda) / total_cars) * 100 = ((total_cars * percentage_red_total - (total_cars - honda_cars) * percentage_red_non_honda) / honda_cars) * 100 :=
by
  sorry

end percentage_honda_red_l909_90938


namespace simplify_expression_l909_90912

theorem simplify_expression (x y : ℝ) : 
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 :=
by
  sorry

end simplify_expression_l909_90912


namespace total_pages_in_book_l909_90917

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end total_pages_in_book_l909_90917


namespace max_sum_clock_digits_l909_90934

theorem max_sum_clock_digits : ∃ t : ℕ, 0 ≤ t ∧ t < 24 ∧ 
  (∃ h1 h2 m1 m2 : ℕ, t = h1 * 10 + h2 + m1 * 10 + m2 ∧ 
   (0 ≤ h1 ∧ h1 ≤ 2) ∧ (0 ≤ h2 ∧ h2 ≤ 9) ∧ (0 ≤ m1 ∧ m1 ≤ 5) ∧ (0 ≤ m2 ∧ m2 ≤ 9) ∧ 
   h1 + h2 + m1 + m2 = 24) := sorry

end max_sum_clock_digits_l909_90934


namespace jerry_remaining_money_l909_90941

-- Define initial money
def initial_money := 18

-- Define amount spent on video games
def spent_video_games := 6

-- Define amount spent on a snack
def spent_snack := 3

-- Define total amount spent
def total_spent := spent_video_games + spent_snack

-- Define remaining money after spending
def remaining_money := initial_money - total_spent

theorem jerry_remaining_money : remaining_money = 9 :=
by
  sorry

end jerry_remaining_money_l909_90941


namespace yunjeong_locker_problem_l909_90905

theorem yunjeong_locker_problem
  (l r f b : ℕ)
  (h_l : l = 7)
  (h_r : r = 13)
  (h_f : f = 8)
  (h_b : b = 14)
  (same_rows : ∀ pos1 pos2 : ℕ, pos1 = pos2) :
  (l - 1) + (r - 1) + (f - 1) + (b - 1) = 399 := sorry

end yunjeong_locker_problem_l909_90905


namespace proof_area_of_squares_l909_90913

noncomputable def area_of_squares : Prop :=
  let side_C := 48
  let side_D := 60
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  (area_C / area_D = (16 / 25)) ∧ 
  ((area_D - area_C) / area_C = (36 / 100))

theorem proof_area_of_squares : area_of_squares := sorry

end proof_area_of_squares_l909_90913


namespace find_price_per_craft_l909_90986

-- Definitions based on conditions
def price_per_craft (x : ℝ) : Prop :=
  let crafts_sold := 3
  let extra_money := 7
  let deposit := 18
  let remaining_money := 25
  let total_before_deposit := 43
  3 * x + extra_money = total_before_deposit

-- Statement of the problem to prove x = 12 given conditions
theorem find_price_per_craft : ∃ x : ℝ, price_per_craft x ∧ x = 12 :=
by
  sorry

end find_price_per_craft_l909_90986


namespace find_increase_x_l909_90925

noncomputable def initial_radius : ℝ := 7
noncomputable def initial_height : ℝ := 5
variable (x : ℝ)

theorem find_increase_x (hx : x > 0)
  (volume_eq : π * (initial_radius + x) ^ 2 * initial_height =
               π * initial_radius ^ 2 * (initial_height + 2 * x)) :
  x = 28 / 5 :=
by
  sorry

end find_increase_x_l909_90925


namespace slope_of_line_l909_90946

theorem slope_of_line : ∀ (x y : ℝ), (x - y + 1 = 0) → (1 = 1) :=
by
  intros x y h
  sorry

end slope_of_line_l909_90946


namespace complement_of_intersection_eq_l909_90989

-- Definitions of sets with given conditions
def U : Set ℝ := {x | 0 ≤ x ∧ x < 10}
def A : Set ℝ := {x | 2 < x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Complement of a set with respect to U
def complement_U (S : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ S}

-- Intersect two sets
def intersection (S1 S2 : Set ℝ) : Set ℝ := {x | x ∈ S1 ∧ x ∈ S2}

theorem complement_of_intersection_eq :
  complement_U (intersection A B) = {x | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := 
by
  sorry

end complement_of_intersection_eq_l909_90989


namespace base_7_minus_base_8_to_decimal_l909_90968

theorem base_7_minus_base_8_to_decimal : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) - (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 8190 :=
by sorry

end base_7_minus_base_8_to_decimal_l909_90968


namespace probability_of_forming_triangle_l909_90927

def segment_lengths : List ℕ := [1, 3, 5, 7, 9]
def valid_combinations : List (ℕ × ℕ × ℕ) := [(3, 5, 7), (3, 7, 9), (5, 7, 9)]
def total_combinations := Nat.choose 5 3

theorem probability_of_forming_triangle :
  (valid_combinations.length : ℚ) / total_combinations = 3 / 10 := 
by
  sorry

end probability_of_forming_triangle_l909_90927


namespace original_numbers_geometric_sequence_l909_90966

theorem original_numbers_geometric_sequence (a q : ℝ) :
  (2 * (a * q + 8) = a + a * q^2) →
  ((a * q + 8) ^ 2 = a * (a * q^2 + 64)) →
  (a, a * q, a * q^2) = (4, 12, 36) ∨ (a, a * q, a * q^2) = (4 / 9, -20 / 9, 100 / 9) :=
by {
  sorry
}

end original_numbers_geometric_sequence_l909_90966


namespace expansion_coefficient_a2_l909_90935

theorem expansion_coefficient_a2 : 
  (∃ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    (1 - 2*x)^7 = a + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6 + a_7*x^7 -> 
    a_2 = 84) :=
sorry

end expansion_coefficient_a2_l909_90935


namespace certain_number_sum_421_l909_90955

theorem certain_number_sum_421 :
  ∃ n, (∃ k, n = 423 * k) ∧ k = 2 →
  n + 421 = 1267 :=
by
  sorry

end certain_number_sum_421_l909_90955


namespace collinear_points_sum_xy_solution_l909_90919

theorem collinear_points_sum_xy_solution (x y : ℚ)
  (h1 : (B : ℚ × ℚ) = (-2, y))
  (h2 : (A : ℚ × ℚ) = (x, 5))
  (h3 : (C : ℚ × ℚ) = (1, 1))
  (h4 : dist (B.1, B.2) (C.1, C.2) = 2 * dist (A.1, A.2) (C.1, C.2))
  (h5 : (y - 5) / (-2 - x) = (1 - 5) / (1 - x)) :
  x + y = -9 / 2 ∨ x + y = 17 / 2 :=
by sorry

end collinear_points_sum_xy_solution_l909_90919


namespace total_calories_burned_l909_90996

def base_distance : ℝ := 15
def records : List ℝ := [0.1, -0.8, 0.9, 16.5 - base_distance, 2.0, -1.5, 14.1 - base_distance, 1.0, 0.8, -1.1]
def calorie_burn_rate : ℝ := 20

theorem total_calories_burned :
  (base_distance * 10 + (List.sum records)) * calorie_burn_rate = 3040 :=
by
  sorry

end total_calories_burned_l909_90996


namespace divisors_remainder_5_l909_90932

theorem divisors_remainder_5 (d : ℕ) : d ∣ 2002 ∧ d > 5 ↔ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 14 ∨ 
                                      d = 22 ∨ d = 26 ∨ d = 77 ∨ d = 91 ∨ 
                                      d = 143 ∨ d = 154 ∨ d = 182 ∨ d = 286 ∨ 
                                      d = 1001 ∨ d = 2002 :=
by sorry

end divisors_remainder_5_l909_90932


namespace largest_constant_inequality_equality_condition_l909_90987

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆) ^ 2 ≥
    3 * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end largest_constant_inequality_equality_condition_l909_90987


namespace no_natural_numbers_satisfy_conditions_l909_90910

theorem no_natural_numbers_satisfy_conditions : 
  ¬ ∃ (a b : ℕ), 
    (∃ (k : ℕ), k^2 = a^2 + 2 * b^2) ∧ 
    (∃ (m : ℕ), m^2 = b^2 + 2 * a) :=
by {
  -- Proof steps and logical deductions can be written here.
  sorry
}

end no_natural_numbers_satisfy_conditions_l909_90910


namespace weight_of_dry_grapes_l909_90990

theorem weight_of_dry_grapes (w_fresh : ℝ) (perc_water_fresh perc_water_dried : ℝ) (w_non_water : ℝ) (w_dry : ℝ) :
  w_fresh = 5 →
  perc_water_fresh = 0.90 →
  perc_water_dried = 0.20 →
  w_non_water = w_fresh * (1 - perc_water_fresh) →
  w_non_water = w_dry * (1 - perc_water_dried) →
  w_dry = 0.625 :=
by sorry

end weight_of_dry_grapes_l909_90990


namespace triangle_cosine_theorem_l909_90908

def triangle_sums (a b c : ℝ) : ℝ := 
  b^2 + c^2 - a^2 + a^2 + c^2 - b^2 + a^2 + b^2 - c^2

theorem triangle_cosine_theorem (a b c : ℝ) (cos_A cos_B cos_C : ℝ) :
  a = 2 → b = 3 → c = 4 → 2 * b * c * cos_A + 2 * c * a * cos_B + 2 * a * b * cos_C = 29 :=
by
  intros h₁ h₂ h₃
  sorry

end triangle_cosine_theorem_l909_90908


namespace sum_of_transformed_roots_equals_one_l909_90950

theorem sum_of_transformed_roots_equals_one 
  {α β γ : ℝ} 
  (hα : α^3 - α - 1 = 0) 
  (hβ : β^3 - β - 1 = 0) 
  (hγ : γ^3 - γ - 1 = 0) : 
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
sorry

end sum_of_transformed_roots_equals_one_l909_90950


namespace distance_between_stripes_l909_90961

theorem distance_between_stripes
  (curb_distance : ℝ) (length_curb : ℝ) (stripe_length : ℝ) (distance_stripes : ℝ)
  (h1 : curb_distance = 60)
  (h2 : length_curb = 20)
  (h3 : stripe_length = 50)
  (h4 : distance_stripes = (length_curb * curb_distance) / stripe_length) :
  distance_stripes = 24 :=
by
  sorry

end distance_between_stripes_l909_90961


namespace unique_two_digit_u_l909_90973

theorem unique_two_digit_u:
  ∃! u : ℤ, 10 ≤ u ∧ u < 100 ∧ 
            (15 * u) % 100 = 45 ∧ 
            u % 17 = 7 :=
by
  -- To be completed in proof
  sorry

end unique_two_digit_u_l909_90973


namespace find_k_value_l909_90949

theorem find_k_value (k : ℕ) :
  3 * 6 * 4 * k = Nat.factorial 8 → k = 560 :=
by
  sorry

end find_k_value_l909_90949


namespace zero_in_interval_l909_90997

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → (f 0.25 < 0) → ∃ x, 0.25 < x ∧ x < 0.5 ∧ f x = 0 :=
by
  intro h0 h05 h025
  -- This is just the statement; the proof is not required as per instructions
  sorry

end zero_in_interval_l909_90997


namespace no_such_primes_l909_90916

theorem no_such_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_three : p > 3) (hq_gt_three : q > 3) (hq_div_p2_minus_1 : q ∣ (p^2 - 1)) 
  (hp_div_q2_minus_1 : p ∣ (q^2 - 1)) : false := 
sorry

end no_such_primes_l909_90916


namespace payment_to_C_l909_90962

-- Work rates definition
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 8
def combined_work_rate_A_B : ℚ := work_rate_A + work_rate_B
def combined_work_rate_A_B_C : ℚ := 1 / 3

-- C's work rate calculation
def work_rate_C : ℚ := combined_work_rate_A_B_C - combined_work_rate_A_B

-- Payment calculation
def total_payment : ℚ := 3200
def C_payment_ratio : ℚ := work_rate_C / combined_work_rate_A_B_C
def C_payment : ℚ := total_payment * C_payment_ratio

-- Theorem stating the result
theorem payment_to_C : C_payment = 400 := by
  sorry

end payment_to_C_l909_90962


namespace intersection_of_sets_union_of_complement_and_set_l909_90994

def set1 := { x : ℝ | -1 < x ∧ x < 2 }
def set2 := { x : ℝ | x > 0 }
def complement_set2 := { x : ℝ | x ≤ 0 }
def intersection_set := { x : ℝ | 0 < x ∧ x < 2 }
def union_set := { x : ℝ | x < 2 }

theorem intersection_of_sets : 
  { x : ℝ | x ∈ set1 ∧ x ∈ set2 } = intersection_set := 
by 
  sorry

theorem union_of_complement_and_set : 
  { x : ℝ | x ∈ complement_set2 ∨ x ∈ set1 } = union_set := 
by 
  sorry

end intersection_of_sets_union_of_complement_and_set_l909_90994


namespace expand_and_simplify_l909_90958

theorem expand_and_simplify :
  ∀ (x : ℝ), 5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 :=
by
  intro x
  sorry

end expand_and_simplify_l909_90958


namespace polynomial_coefficients_sum_even_odd_coefficients_difference_square_l909_90970

theorem polynomial_coefficients_sum (a : Fin 8 → ℝ):
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 3^7 - 1 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

theorem even_odd_coefficients_difference_square (a : Fin 8 → ℝ):
  (a 0 + a 2 + a 4 + a 6)^2 - (a 1 + a 3 + a 5 + a 7)^2 = -3^7 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

end polynomial_coefficients_sum_even_odd_coefficients_difference_square_l909_90970


namespace fraction_part_of_twenty_five_l909_90947

open Nat

def eighty_percent (x : ℕ) : ℕ := (85 * x) / 100

theorem fraction_part_of_twenty_five (x y : ℕ) (h1 : eighty_percent 40 = 34) (h2 : 34 - y = 14) (h3 : y = (4 * 25) / 5) : y = 20 :=
by 
  -- Given h1: eighty_percent 40 = 34
  -- And h2: 34 - y = 14
  -- And h3: y = (4 * 25) / 5
  -- Show y = 20
  sorry

end fraction_part_of_twenty_five_l909_90947


namespace fifteenth_term_arithmetic_sequence_l909_90922

theorem fifteenth_term_arithmetic_sequence (a d : ℤ) : 
  (a + 20 * d = 17) ∧ (a + 21 * d = 20) → (a + 14 * d = -1) := by
  sorry

end fifteenth_term_arithmetic_sequence_l909_90922


namespace correct_statement_about_K_l909_90936

-- Defining the possible statements about the chemical equilibrium constant K
def K (n : ℕ) : String :=
  match n with
  | 1 => "The larger the K, the smaller the conversion rate of the reactants."
  | 2 => "K is related to the concentration of the reactants."
  | 3 => "K is related to the concentration of the products."
  | 4 => "K is related to temperature."
  | _ => "Invalid statement"

-- Given that the correct answer is that K is related to temperature
theorem correct_statement_about_K : K 4 = "K is related to temperature." :=
by
  rfl

end correct_statement_about_K_l909_90936


namespace calculate_coeffs_l909_90933

noncomputable def quadratic_coeffs (p q : ℝ) : Prop :=
  if p = 1 then true else if p = -2 then q = -1 else false

theorem calculate_coeffs (p q : ℝ) :
    (∃ p q, (x^2 + p * x + q = 0) ∧ (x^2 - p^2 * x + p * q = 0)) →
    quadratic_coeffs p q :=
by sorry

end calculate_coeffs_l909_90933


namespace problem_conditions_l909_90928

noncomputable def f (x : ℝ) : ℝ := -x - x^3

variables (x₁ x₂ : ℝ)

theorem problem_conditions (h₁ : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧
  (¬ (f x₂ * f (-x₂) > 0)) ∧
  (¬ (f x₁ + f x₂ ≤ f (-x₁) + f (-x₂))) ∧
  (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :=
sorry

end problem_conditions_l909_90928


namespace right_triangle_hypotenuse_l909_90976

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 3) (h' : b = 4) (hc : c^2 = a^2 + b^2) : c = 5 := 
by
  -- proof goes here
  sorry

end right_triangle_hypotenuse_l909_90976


namespace num_sets_B_l909_90977

open Set

theorem num_sets_B (A B : Set ℕ) (hA : A = {1, 2}) (h_union : A ∪ B = {1, 2, 3}) : ∃ n, n = 4 :=
by
  sorry

end num_sets_B_l909_90977


namespace unique_integer_solution_l909_90900

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2 * n^2 + m^2 + n^2 + 6 * m * n ↔ m = 0 ∧ n = 0 :=
by
  sorry

end unique_integer_solution_l909_90900


namespace difference_square_consecutive_l909_90920

theorem difference_square_consecutive (x : ℕ) (h : x * (x + 1) = 812) : (x + 1)^2 - x = 813 :=
sorry

end difference_square_consecutive_l909_90920


namespace problem_solution_l909_90991

open Set

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem problem_solution :
  A ∩ B = {1, 2, 3} ∧
  A ∩ C = {3, 4, 5, 6} ∧
  A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} ∧
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8} :=
by
  sorry

end problem_solution_l909_90991


namespace greatest_positive_integer_x_l909_90985

theorem greatest_positive_integer_x (x : ℕ) (h₁ : x^2 < 12) (h₂ : ∀ y: ℕ, y^2 < 12 → y ≤ x) : 
  x = 3 := 
by
  sorry

end greatest_positive_integer_x_l909_90985


namespace value_of_f_at_7_l909_90903

theorem value_of_f_at_7
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = 2 :=
by
  -- Proof will be filled here
  sorry

end value_of_f_at_7_l909_90903


namespace dave_guitar_strings_l909_90930

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end dave_guitar_strings_l909_90930


namespace domain_of_func_l909_90902

noncomputable def func (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem domain_of_func :
  ∀ x : ℝ, x ≠ 1 / 2 ↔ ∃ y : ℝ, y = func x := sorry

end domain_of_func_l909_90902


namespace rectangle_cut_dimensions_l909_90959

-- Define the original dimensions of the rectangle as constants.
def original_length : ℕ := 12
def original_height : ℕ := 6

-- Define the dimensions of the new rectangle after slicing parallel to the longer side.
def new_length := original_length / 2
def new_height := original_height

-- The theorem statement.
theorem rectangle_cut_dimensions :
  new_length = 6 ∧ new_height = 6 :=
by
  sorry

end rectangle_cut_dimensions_l909_90959


namespace increased_percentage_l909_90907

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l909_90907


namespace gasoline_added_l909_90948

theorem gasoline_added (total_capacity : ℝ) (initial_fraction final_fraction : ℝ) 
(h1 : initial_fraction = 3 / 4)
(h2 : final_fraction = 9 / 10)
(h3 : total_capacity = 29.999999999999996) : 
(final_fraction * total_capacity - initial_fraction * total_capacity = 4.499999999999999) :=
by sorry

end gasoline_added_l909_90948


namespace tim_income_less_than_juan_l909_90945

-- Definitions of the conditions
variables {T J M : ℝ}
def mart_income_condition1 (M T : ℝ) : Prop := M = 1.40 * T
def mart_income_condition2 (M J : ℝ) : Prop := M = 0.84 * J

-- The proof goal
theorem tim_income_less_than_juan (T J M : ℝ) 
(h1: mart_income_condition1 M T) 
(h2: mart_income_condition2 M J) : 
T = 0.60 * J :=
by
  sorry

end tim_income_less_than_juan_l909_90945


namespace area_of_closed_shape_l909_90975

theorem area_of_closed_shape :
  ∫ y in (-2 : ℝ)..3, ((2:ℝ)^y + 2 - (2:ℝ)^y) = 10 := by
  sorry

end area_of_closed_shape_l909_90975


namespace inequality_satisfied_for_a_l909_90993

theorem inequality_satisfied_for_a (a : ℝ) :
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry

end inequality_satisfied_for_a_l909_90993


namespace simplify_expression_l909_90939

variable (b : ℝ)

theorem simplify_expression :
  (2 * b + 6 - 5 * b) / 2 = -3 / 2 * b + 3 :=
sorry

end simplify_expression_l909_90939


namespace monthly_expenses_last_month_was_2888_l909_90957

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end monthly_expenses_last_month_was_2888_l909_90957


namespace gasoline_price_percentage_increase_l909_90915

theorem gasoline_price_percentage_increase 
  (price_month1_euros : ℝ) (price_month3_dollars : ℝ) (exchange_rate : ℝ) 
  (price_month1 : ℝ) (percent_increase : ℝ):
  price_month1_euros = 20 →
  price_month3_dollars = 15 →
  exchange_rate = 1.2 →
  price_month1 = price_month1_euros * exchange_rate →
  percent_increase = ((price_month1 - price_month3_dollars) / price_month3_dollars) * 100 →
  percent_increase = 60 :=
by intros; sorry

end gasoline_price_percentage_increase_l909_90915


namespace geometric_progression_fourth_term_l909_90971

theorem geometric_progression_fourth_term (x : ℚ)
  (h : (3 * x + 3) / x = (5 * x + 5) / (3 * x + 3)) :
  (5 / 3) * (5 * x + 5) = -125/12 :=
by
  sorry

end geometric_progression_fourth_term_l909_90971


namespace average_interest_rate_equal_4_09_percent_l909_90981

-- Define the given conditions
def investment_total : ℝ := 5000
def interest_rate_at_3_percent : ℝ := 0.03
def interest_rate_at_5_percent : ℝ := 0.05
def return_relationship (x : ℝ) : Prop := 
  interest_rate_at_5_percent * x = 2 * interest_rate_at_3_percent * (investment_total - x)

-- Define the final statement
theorem average_interest_rate_equal_4_09_percent :
  ∃ x : ℝ, return_relationship x ∧ 
  ((interest_rate_at_5_percent * x + interest_rate_at_3_percent * (investment_total - x)) / investment_total) = 0.04091 := 
by
  sorry

end average_interest_rate_equal_4_09_percent_l909_90981


namespace tan_A_value_l909_90942

open Real

theorem tan_A_value (A : ℝ) (h1 : sin A * (sin A + sqrt 3 * cos A) = -1 / 2) (h2 : 0 < A ∧ A < π) :
  tan A = -sqrt 3 / 3 :=
sorry

end tan_A_value_l909_90942


namespace total_games_across_leagues_l909_90999

-- Defining the conditions for the leagues
def leagueA_teams := 20
def leagueB_teams := 25
def leagueC_teams := 30

-- Function to calculate the number of games in a round-robin tournament
def number_of_games (n : ℕ) := n * (n - 1) / 2

-- Proposition to prove total games across all leagues
theorem total_games_across_leagues :
  number_of_games leagueA_teams + number_of_games leagueB_teams + number_of_games leagueC_teams = 925 := by
  sorry

end total_games_across_leagues_l909_90999


namespace roots_opposite_signs_l909_90969

theorem roots_opposite_signs (p : ℝ) (hp : p > 0) :
  ( ∃ (x₁ x₂ : ℝ), (x₁ * x₂ < 0) ∧ (5 * x₁^2 - 4 * (p + 3) * x₁ + 4 = p^2) ∧  
      (5 * x₂^2 - 4 * (p + 3) * x₂ + 4 = p^2) ) ↔ p > 2 :=
by {
  sorry
}

end roots_opposite_signs_l909_90969


namespace max_value_frac_l909_90980
noncomputable section

open Real

variables (a b x y : ℝ)

theorem max_value_frac :
  a > 1 → b > 1 → 
  a^x = 2 → b^y = 2 →
  a + sqrt b = 4 →
  (2/x + 1/y) ≤ 4 :=
by
  intros ha hb hax hby hab
  sorry

end max_value_frac_l909_90980


namespace quadratic_graph_nature_l909_90967

theorem quadratic_graph_nature (a b : Real) (h : a ≠ 0) :
  ∀ (x : Real), (a * x^2 + b * x + (b^2 / (2 * a)) > 0) ∨ (a * x^2 + b * x + (b^2 / (2 * a)) < 0) :=
by
  sorry

end quadratic_graph_nature_l909_90967


namespace line_eq_l909_90965

theorem line_eq (P : ℝ × ℝ) (hP : P = (1, 2)) (h_perp : ∀ x y : ℝ, 2 * x + y - 1 = 0 → x - 2 * y + c = 0) : 
  ∃ c : ℝ, (x - 2 * y + c = 0 ∧ P ∈ {(x, y) | x - 2 * y + c = 0}) ∧ c = 3 :=
  sorry

end line_eq_l909_90965


namespace average_speed_of_train_l909_90978

theorem average_speed_of_train (d1 d2: ℝ) (t1 t2: ℝ) (h_d1: d1 = 250) (h_d2: d2 = 350) (h_t1: t1 = 2) (h_t2: t2 = 4) :
  (d1 + d2) / (t1 + t2) = 100 := by
  sorry

end average_speed_of_train_l909_90978


namespace find_pairs_1984_l909_90995

theorem find_pairs_1984 (m n : ℕ) :
  19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by
  sorry

end find_pairs_1984_l909_90995


namespace log2_125_eq_9y_l909_90926

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end log2_125_eq_9y_l909_90926


namespace cloth_sold_worth_l909_90984

-- Define the commission rate and commission received
def commission_rate := 0.05
def commission_received := 12.50

-- State the theorem to be proved
theorem cloth_sold_worth : commission_received / commission_rate = 250 :=
by
  sorry

end cloth_sold_worth_l909_90984


namespace gcd_impossible_l909_90963

-- Define the natural numbers a, b, and c
variable (a b c : ℕ)

-- Define the factorial values
def fact_30 := Nat.factorial 30
def fact_40 := Nat.factorial 40
def fact_50 := Nat.factorial 50

-- Define the gcd values to be checked
def gcd_ab := fact_30 + 111
def gcd_bc := fact_40 + 234
def gcd_ca := fact_50 + 666

-- The main theorem to prove the impossibility
theorem gcd_impossible (h1 : Nat.gcd a b = gcd_ab) (h2 : Nat.gcd b c = gcd_bc) (h3 : Nat.gcd c a = gcd_ca) : False :=
by
  -- Proof omitted
  sorry

end gcd_impossible_l909_90963


namespace arithmetic_sequence_S7_geometric_sequence_k_l909_90904

noncomputable def S_n (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_S7 (a_4 : ℕ) (h : a_4 = 8) : S_n a_4 1 7 = 56 := by
  sorry

def Sn_formula (k : ℕ) : ℕ := k^2 + k
def a (i d : ℕ) := i * d

theorem geometric_sequence_k (a_1 k : ℕ) (h1 : a_1 = 2) (h2 : (2 * k + 2)^2 = 6 * (k^2 + k)) :
  k = 2 := by
  sorry

end arithmetic_sequence_S7_geometric_sequence_k_l909_90904


namespace min_cuts_for_payment_7_days_l909_90940

theorem min_cuts_for_payment_7_days (n : ℕ) (h : n = 7) : ∃ k, k = 1 :=
by sorry

end min_cuts_for_payment_7_days_l909_90940


namespace apothem_comparison_l909_90918

noncomputable def pentagon_side_length : ℝ := 4 / Real.tan (54 * Real.pi / 180)

noncomputable def pentagon_apothem : ℝ := pentagon_side_length / (2 * Real.tan (54 * Real.pi / 180))

noncomputable def hexagon_side_length : ℝ := 4 / Real.sqrt 3

noncomputable def hexagon_apothem : ℝ := (Real.sqrt 3 / 2) * hexagon_side_length

theorem apothem_comparison : pentagon_apothem = 1.06 * hexagon_apothem :=
by
  sorry

end apothem_comparison_l909_90918


namespace acute_angles_sine_relation_l909_90901

theorem acute_angles_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : α < β :=
by
  sorry

end acute_angles_sine_relation_l909_90901


namespace keith_score_l909_90956

theorem keith_score (K : ℕ) (h : K + 3 * K + (3 * K + 5) = 26) : K = 3 :=
by
  sorry

end keith_score_l909_90956


namespace matrix_solution_l909_90909

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, -3], ![4, -1]]
noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := ![![ -8,  5], ![ 11, -7]]

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![ -1.2, -1.4], ![1.7, 1.9]]

theorem matrix_solution : M * A = B :=
by sorry

end matrix_solution_l909_90909
