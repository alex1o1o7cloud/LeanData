import Mathlib

namespace NUMINAMATH_GPT_find_x_l1937_193703

noncomputable def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x + 1)^(1 / 3) / 5^(1/3)

theorem find_x (x : ℝ) :
  h (3 * x) = 3 * h x ↔ x = -1 + (10^(1/2)) / 3 ∨ x = -1 - (10^(1/2)) / 3 := by
  sorry

end NUMINAMATH_GPT_find_x_l1937_193703


namespace NUMINAMATH_GPT_bananas_to_oranges_l1937_193730

theorem bananas_to_oranges :
  (3 / 4) * 12 * b = 9 * o →
  ((3 / 5) * 15 * b) = 9 * o := 
by
  sorry

end NUMINAMATH_GPT_bananas_to_oranges_l1937_193730


namespace NUMINAMATH_GPT_valid_expression_l1937_193720

theorem valid_expression (x : ℝ) : 
  (x - 1 ≥ 0 ∧ x - 2 ≠ 0) ↔ (x ≥ 1 ∧ x ≠ 2) := 
by
  sorry

end NUMINAMATH_GPT_valid_expression_l1937_193720


namespace NUMINAMATH_GPT_not_monotonic_in_interval_l1937_193704

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a * x - 5

theorem not_monotonic_in_interval (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f a x ≠ (1/3) * x^3 - x^2 + a * x - 5) → a ≥ 1 ∨ a ≤ -3 :=
sorry

end NUMINAMATH_GPT_not_monotonic_in_interval_l1937_193704


namespace NUMINAMATH_GPT_sum_of_cubes_consecutive_divisible_by_9_l1937_193723

theorem sum_of_cubes_consecutive_divisible_by_9 (n : ℤ) : 9 ∣ (n-1)^3 + n^3 + (n+1)^3 :=
  sorry

end NUMINAMATH_GPT_sum_of_cubes_consecutive_divisible_by_9_l1937_193723


namespace NUMINAMATH_GPT_final_population_correct_l1937_193712

noncomputable def initialPopulation : ℕ := 300000
noncomputable def immigration : ℕ := 50000
noncomputable def emigration : ℕ := 30000

noncomputable def populationAfterImmigration : ℕ := initialPopulation + immigration
noncomputable def populationAfterEmigration : ℕ := populationAfterImmigration - emigration

noncomputable def pregnancies : ℕ := populationAfterEmigration / 8
noncomputable def twinPregnancies : ℕ := pregnancies / 4
noncomputable def singlePregnancies : ℕ := pregnancies - twinPregnancies

noncomputable def totalBirths : ℕ := twinPregnancies * 2 + singlePregnancies
noncomputable def finalPopulation : ℕ := populationAfterEmigration + totalBirths

theorem final_population_correct : finalPopulation = 370000 :=
by
  sorry

end NUMINAMATH_GPT_final_population_correct_l1937_193712


namespace NUMINAMATH_GPT_football_match_goals_even_likely_l1937_193700

noncomputable def probability_even_goals (p_1 : ℝ) (q_1 : ℝ) : Prop :=
  let p := p_1^2 + q_1^2
  let q := 2 * p_1 * q_1
  p >= q

theorem football_match_goals_even_likely (p_1 : ℝ) (h : p_1 >= 0 ∧ p_1 <= 1) : probability_even_goals p_1 (1 - p_1) :=
by sorry

end NUMINAMATH_GPT_football_match_goals_even_likely_l1937_193700


namespace NUMINAMATH_GPT_betty_cookies_and_brownies_difference_l1937_193749

-- Definitions based on the conditions
def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10
def cookies_per_day : ℕ := 3
def brownies_per_day : ℕ := 1
def days : ℕ := 7

-- The proof statement
theorem betty_cookies_and_brownies_difference :
  initial_cookies - (cookies_per_day * days) - (initial_brownies - (brownies_per_day * days)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_betty_cookies_and_brownies_difference_l1937_193749


namespace NUMINAMATH_GPT_polygon_sides_l1937_193755

-- Define the given condition formally
def sum_of_internal_and_external_angle (n : ℕ) : ℕ :=
  (n - 2) * 180 + (1) -- This represents the sum of internal angles plus an external angle

theorem polygon_sides (n : ℕ) : 
  sum_of_internal_and_external_angle n = 1350 → n = 9 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1937_193755


namespace NUMINAMATH_GPT_area_of_one_cookie_l1937_193770

theorem area_of_one_cookie (L W : ℝ)
    (W_eq_15 : W = 15)
    (circumference_condition : 4 * L + 2 * W = 70) :
    L * W = 150 :=
by
  sorry

end NUMINAMATH_GPT_area_of_one_cookie_l1937_193770


namespace NUMINAMATH_GPT_no_valid_k_exists_l1937_193763

theorem no_valid_k_exists {k : ℕ} : ¬(∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p + q = 41 ∧ p * q = k) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_k_exists_l1937_193763


namespace NUMINAMATH_GPT_min_xy_min_a_b_l1937_193733

-- Problem 1 Lean Statement
theorem min_xy {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / (4 * y) = 1) : xy ≥ 2 := sorry

-- Problem 2 Lean Statement
theorem min_a_b {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : ab = a + 2 * b + 4) : a + b ≥ 3 + 2 * Real.sqrt 6 := sorry

end NUMINAMATH_GPT_min_xy_min_a_b_l1937_193733


namespace NUMINAMATH_GPT_triangle_inequality_l1937_193736

theorem triangle_inequality (A B C : ℝ) :
  ∀ (a b c : ℝ), (a = 2 * Real.sin (A / 2) * Real.cos (A / 2)) ∧
                 (b = 2 * Real.sin (B / 2) * Real.cos (B / 2)) ∧
                 (c = Real.cos ((A + B) / 2)) ∧
                 (x = Real.sqrt (Real.tan (A / 2) * Real.tan (B / 2)))
                 → (Real.sqrt (a * b) / Real.sin (C / 2) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2)) := by {
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_l1937_193736


namespace NUMINAMATH_GPT_geometric_sum_n_equals_4_l1937_193739

def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def S (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))
def sum_value : ℚ := 26 / 81

theorem geometric_sum_n_equals_4 (n : ℕ) (h : S n = sum_value) : n = 4 :=
by sorry

end NUMINAMATH_GPT_geometric_sum_n_equals_4_l1937_193739


namespace NUMINAMATH_GPT_standard_eq_of_parabola_l1937_193738

-- Conditions:
-- The point (1, -2) lies on the parabola.
def point_on_parabola : Prop := ∃ p : ℝ, (1, -2).2^2 = 2 * p * (1, -2).1 ∨ (1, -2).1^2 = 2 * p * (1, -2).2

-- Question to be proved:
-- The standard equation of the parabola passing through the point (1, -2) is y^2 = 4x or x^2 = - (1/2) y.
theorem standard_eq_of_parabola : point_on_parabola → (y^2 = 4*x ∨ x^2 = -(1/(2:ℝ)) * y) :=
by
  sorry -- proof to be provided

end NUMINAMATH_GPT_standard_eq_of_parabola_l1937_193738


namespace NUMINAMATH_GPT_determine_coefficients_l1937_193702

theorem determine_coefficients (a b c : ℝ) (x y : ℝ) :
  (x = 3/4 ∧ y = 5/8) →
  (a * (x - 1) + 2 * y = 1) →
  (b * |x - 1| + c * y = 3) →
  (a = 1 ∧ b = 2 ∧ c = 4) := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_determine_coefficients_l1937_193702


namespace NUMINAMATH_GPT_bobs_total_profit_l1937_193711

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end NUMINAMATH_GPT_bobs_total_profit_l1937_193711


namespace NUMINAMATH_GPT_range_of_a_l1937_193774

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then x^2 + 2 * a else -x

theorem range_of_a (a : ℝ) (h : a < 0) (hf : f a (1 - a) ≥ f a (1 + a)) : -2 ≤ a ∧ a ≤ -1 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1937_193774


namespace NUMINAMATH_GPT_area_and_cost_of_path_l1937_193761

-- Define the dimensions of the rectangular grass field
def length_field : ℝ := 75
def width_field : ℝ := 55

-- Define the width of the path around the field
def path_width : ℝ := 2.8

-- Define the cost per square meter for constructing the path
def cost_per_sq_m : ℝ := 2

-- Define the total length and width including the path
def total_length : ℝ := length_field + 2 * path_width
def total_width : ℝ := width_field + 2 * path_width

-- Define the area of the entire field including the path
def area_total : ℝ := total_length * total_width

-- Define the area of the grass field alone
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_total - area_field

-- Define the cost of constructing the path
def cost_path : ℝ := area_path * cost_per_sq_m

-- The statement to be proved
theorem area_and_cost_of_path :
  area_path = 759.36 ∧ cost_path = 1518.72 := by
  sorry

end NUMINAMATH_GPT_area_and_cost_of_path_l1937_193761


namespace NUMINAMATH_GPT_circle_radius_l1937_193759

theorem circle_radius (M N r : ℝ) (h1 : M = Real.pi * r^2) (h2 : N = 2 * Real.pi * r) (h3 : M / N = 25) : r = 50 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1937_193759


namespace NUMINAMATH_GPT_ratio_a_c_l1937_193781

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_a_c_l1937_193781


namespace NUMINAMATH_GPT_total_plates_l1937_193746

-- define the variables for the number of plates
def plates_lobster_rolls : Nat := 25
def plates_spicy_hot_noodles : Nat := 14
def plates_seafood_noodles : Nat := 16

-- state the problem as a theorem
theorem total_plates :
  plates_lobster_rolls + plates_spicy_hot_noodles + plates_seafood_noodles = 55 := by
  sorry

end NUMINAMATH_GPT_total_plates_l1937_193746


namespace NUMINAMATH_GPT_max_oranges_donated_l1937_193767

theorem max_oranges_donated (N : ℕ) : ∃ n : ℕ, n < 7 ∧ (N % 7 = n) ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_oranges_donated_l1937_193767


namespace NUMINAMATH_GPT_compute_expression_l1937_193742

theorem compute_expression : (-3) * 2 + 4 = -2 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1937_193742


namespace NUMINAMATH_GPT_impossible_to_form_triangle_l1937_193786

theorem impossible_to_form_triangle 
  (a b c : ℝ)
  (h1 : a = 9) 
  (h2 : b = 4) 
  (h3 : c = 3) 
  : ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  rw [h1, h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_impossible_to_form_triangle_l1937_193786


namespace NUMINAMATH_GPT_books_total_l1937_193776

theorem books_total (Tim_books Sam_books : ℕ) (h1 : Tim_books = 44) (h2 : Sam_books = 52) : Tim_books + Sam_books = 96 := 
by
  sorry

end NUMINAMATH_GPT_books_total_l1937_193776


namespace NUMINAMATH_GPT_count_divisors_divisible_exactly_2007_l1937_193728

-- Definitions and conditions
def prime_factors_2006 : List Nat := [2, 17, 59]

def prime_factors_2006_pow_2006 : List (Nat × Nat) := [(2, 2006), (17, 2006), (59, 2006)]

def number_of_divisors (n : Nat) : Nat :=
  prime_factors_2006_pow_2006.foldl (λ acc ⟨p, exp⟩ => acc * (exp + 1)) 1

theorem count_divisors_divisible_exactly_2007 : 
  (number_of_divisors (2^2006 * 17^2006 * 59^2006) = 3) :=
  sorry

end NUMINAMATH_GPT_count_divisors_divisible_exactly_2007_l1937_193728


namespace NUMINAMATH_GPT_evaporate_water_l1937_193731

theorem evaporate_water (M : ℝ) (W_i W_f x : ℝ) (d : ℝ)
  (h_initial_mass : M = 500)
  (h_initial_water_content : W_i = 0.85 * M)
  (h_final_water_content : W_f = 0.75 * (M - x))
  (h_desired_fraction : d = 0.75) :
  x = 200 := 
  sorry

end NUMINAMATH_GPT_evaporate_water_l1937_193731


namespace NUMINAMATH_GPT_sum_of_100th_row_l1937_193750

def triangularArraySum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2^(n+1) - 3*n

theorem sum_of_100th_row :
  triangularArraySum 100 = 2^100 - 297 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_100th_row_l1937_193750


namespace NUMINAMATH_GPT_stick_segments_l1937_193701

theorem stick_segments (L : ℕ) (L_nonzero : L > 0) :
  let red_segments := 8
  let blue_segments := 12
  let black_segments := 18
  let total_segments := (red_segments + blue_segments + black_segments) 
                       - (lcm red_segments blue_segments / blue_segments) 
                       - (lcm blue_segments black_segments / black_segments)
                       - (lcm red_segments black_segments / black_segments)
                       + (lcm red_segments (lcm blue_segments black_segments) / (lcm blue_segments black_segments))
  let shortest_segment_length := L / lcm red_segments (lcm blue_segments black_segments)
  (total_segments = 28) ∧ (shortest_segment_length = L / 72) := by
  sorry

end NUMINAMATH_GPT_stick_segments_l1937_193701


namespace NUMINAMATH_GPT_quadratic_min_n_l1937_193794

theorem quadratic_min_n (m n : ℝ) : 
  (∃ x : ℝ, (x^2 + (m - 2023) * x + (n - 1)) = 0) ∧ 
  (m - 2023)^2 - 4 * (n - 1) = 0 → 
  n = 1 := 
sorry

end NUMINAMATH_GPT_quadratic_min_n_l1937_193794


namespace NUMINAMATH_GPT_min_guesses_correct_l1937_193706

def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  (min_guesses n k = 2 ↔ n = 2 * k) ∧ (min_guesses n k = 1 ↔ n ≠ 2 * k) := by
  sorry

end NUMINAMATH_GPT_min_guesses_correct_l1937_193706


namespace NUMINAMATH_GPT_total_students_in_middle_school_l1937_193796

/-- Given that 20% of the students are in the band and there are 168 students in the band,
    prove that the total number of students in the middle school is 840. -/
theorem total_students_in_middle_school (total_students : ℕ) (band_students : ℕ) 
  (h1 : 20 ≤ 100)
  (h2 : band_students = 168)
  (h3 : band_students = 20 * total_students / 100) 
  : total_students = 840 :=
sorry

end NUMINAMATH_GPT_total_students_in_middle_school_l1937_193796


namespace NUMINAMATH_GPT_age_difference_l1937_193795

theorem age_difference (A B : ℕ) (h1 : B = 34) (h2 : A + 10 = 2 * (B - 10)) : A - B = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1937_193795


namespace NUMINAMATH_GPT_pseudoprime_pow_minus_one_l1937_193713

theorem pseudoprime_pow_minus_one (n : ℕ) (hpseudo : 2^n ≡ 2 [MOD n]) : 
  ∃ m : ℕ, 2^(2^n - 1) ≡ 1 [MOD (2^n - 1)] :=
by
  sorry

end NUMINAMATH_GPT_pseudoprime_pow_minus_one_l1937_193713


namespace NUMINAMATH_GPT_maximum_value_of_f_inequality_holds_for_all_x_l1937_193791

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

theorem maximum_value_of_f (a : ℝ) (h : 0 ≤ a) : 
  (∀ x, f a x ≤ f a 1) → f a 1 = 3 / Real.exp 1 → a = 1 := 
by 
  sorry

theorem inequality_holds_for_all_x (b : ℝ) : 
  (∀ a ≤ 0, ∀ x, 0 ≤ x → f a x ≤ b * Real.log (x + 1)) → 1 ≤ b := 
by 
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_inequality_holds_for_all_x_l1937_193791


namespace NUMINAMATH_GPT_volume_of_set_l1937_193762

theorem volume_of_set (m n p : ℕ) (h_rel_prime : Nat.gcd n p = 1) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_p : 0 < p) 
  (h_volume : (m + n * Real.pi) / p = (324 + 37 * Real.pi) / 3) : 
  m + n + p = 364 := 
  sorry

end NUMINAMATH_GPT_volume_of_set_l1937_193762


namespace NUMINAMATH_GPT_C_finishes_job_in_days_l1937_193775

theorem C_finishes_job_in_days :
  ∀ (A B C : ℚ),
    (A + B = 1 / 15) →
    (A + B + C = 1 / 3) →
    1 / C = 3.75 :=
by
  intros A B C hab habc
  sorry

end NUMINAMATH_GPT_C_finishes_job_in_days_l1937_193775


namespace NUMINAMATH_GPT_ordered_triples_2022_l1937_193782

theorem ordered_triples_2022 :
  ∃ n : ℕ, n = 13 ∧ (∃ a c : ℕ, a ≤ c ∧ (a * c = 2022^2)) := by
  sorry

end NUMINAMATH_GPT_ordered_triples_2022_l1937_193782


namespace NUMINAMATH_GPT_customers_left_l1937_193758

theorem customers_left (initial_customers : ℝ) (first_left : ℝ) (second_left : ℝ) : initial_customers = 36.0 ∧ first_left = 19.0 ∧ second_left = 14.0 → initial_customers - first_left - second_left = 3.0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_customers_left_l1937_193758


namespace NUMINAMATH_GPT_minimum_boys_needed_l1937_193780

theorem minimum_boys_needed (k n m : ℕ) (hn : n > 0) (hm : m > 0) (h : 100 * n + m * k = 10 * k) : n + m = 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_boys_needed_l1937_193780


namespace NUMINAMATH_GPT_xinjiang_arable_land_increase_reason_l1937_193772

theorem xinjiang_arable_land_increase_reason
  (global_climate_warm: Prop)
  (annual_rainfall_increase: Prop)
  (reserve_arable_land_development: Prop)
  (national_land_policies_adjustment: Prop)
  (arable_land_increased: Prop) :
  (arable_land_increased → reserve_arable_land_development) :=
sorry

end NUMINAMATH_GPT_xinjiang_arable_land_increase_reason_l1937_193772


namespace NUMINAMATH_GPT_fraction_four_or_older_l1937_193740

theorem fraction_four_or_older (total_students : ℕ) (under_three : ℕ) (not_between_three_and_four : ℕ)
  (h_total : total_students = 300) (h_under_three : under_three = 20) (h_not_between_three_and_four : not_between_three_and_four = 50) :
  (not_between_three_and_four - under_three) / total_students = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_four_or_older_l1937_193740


namespace NUMINAMATH_GPT_x_is_integer_l1937_193769

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ k1 : ℤ, x^2 - x = k1)
  (h2 : ∃ (n : ℕ) (_ : n > 2) (k2 : ℤ), x^n - x = k2) : 
  ∃ (m : ℤ), x = m := 
sorry

end NUMINAMATH_GPT_x_is_integer_l1937_193769


namespace NUMINAMATH_GPT_no_solution_for_x_l1937_193721

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_x_l1937_193721


namespace NUMINAMATH_GPT_extreme_points_sum_gt_l1937_193748

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem extreme_points_sum_gt (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 8)
    {x₁ x₂ : ℝ} (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) (h₄ : x₁ < x₂)
    (h₅ : 0 < x₁) (h₆ : 0 < x₂) : f x₁ a + f x₂ a > 3 - 2 * Real.log 2 := sorry

end NUMINAMATH_GPT_extreme_points_sum_gt_l1937_193748


namespace NUMINAMATH_GPT_car_travel_l1937_193784

namespace DistanceTravel

/- Define the conditions -/
def distance_initial : ℕ := 120
def car_speed : ℕ := 80

/- Define the relationship between y and x -/
def y (x : ℝ) : ℝ := distance_initial - car_speed * x

/- Prove that y is a linear function and verify the value of y at x = 0.8 -/
theorem car_travel (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1.5) : 
  (y x = distance_initial - car_speed * x) ∧ 
  (y x = 120 - 80 * x) ∧ 
  (x = 0.8 → y x = 56) :=
sorry

end DistanceTravel

end NUMINAMATH_GPT_car_travel_l1937_193784


namespace NUMINAMATH_GPT_factor_expression_eq_l1937_193745

-- Define the given expression
def given_expression (x : ℝ) : ℝ :=
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6)

-- Define the correct factored form
def factored_expression (x : ℝ) : ℝ :=
  5 * x * (3 * x^2 + 17)

-- The theorem stating the equality of the given expression and its factored form
theorem factor_expression_eq (x : ℝ) : given_expression x = factored_expression x :=
  by
  sorry

end NUMINAMATH_GPT_factor_expression_eq_l1937_193745


namespace NUMINAMATH_GPT_two_people_paint_time_l1937_193752

theorem two_people_paint_time (h : 5 * 7 = 35) :
  ∃ t : ℝ, 2 * t = 35 ∧ t = 17.5 := 
sorry

end NUMINAMATH_GPT_two_people_paint_time_l1937_193752


namespace NUMINAMATH_GPT_train_travel_distance_l1937_193743

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end NUMINAMATH_GPT_train_travel_distance_l1937_193743


namespace NUMINAMATH_GPT_price_of_each_lemon_square_l1937_193765

-- Given
def brownies_sold : Nat := 4
def price_per_brownie : Nat := 3
def lemon_squares_sold : Nat := 5
def goal_amount : Nat := 50
def cookies_sold : Nat := 7
def price_per_cookie : Nat := 4

-- Prove
theorem price_of_each_lemon_square :
  (brownies_sold * price_per_brownie + lemon_squares_sold * L + cookies_sold * price_per_cookie = goal_amount) →
  L = 2 :=
by
  sorry

end NUMINAMATH_GPT_price_of_each_lemon_square_l1937_193765


namespace NUMINAMATH_GPT_diamond_value_l1937_193783

def diamond (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem diamond_value : diamond 6 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_diamond_value_l1937_193783


namespace NUMINAMATH_GPT_emily_total_points_l1937_193708

def score_round_1 : ℤ := 16
def score_round_2 : ℤ := 33
def score_round_3 : ℤ := -25
def score_round_4 : ℤ := 46
def score_round_5 : ℤ := 12
def score_round_6 : ℤ := 30 - (2 * score_round_5 / 3)

def total_score : ℤ :=
  score_round_1 + score_round_2 + score_round_3 + score_round_4 + score_round_5 + score_round_6

theorem emily_total_points : total_score = 104 := by
  sorry

end NUMINAMATH_GPT_emily_total_points_l1937_193708


namespace NUMINAMATH_GPT_percentage_singing_l1937_193788

def total_rehearsal_time : ℕ := 75
def warmup_time : ℕ := 6
def notes_time : ℕ := 30
def words_time (t : ℕ) : ℕ := t
def singing_time (t : ℕ) : ℕ := total_rehearsal_time - warmup_time - notes_time - words_time t
def singing_percentage (t : ℕ) : ℕ := (singing_time t * 100) / total_rehearsal_time

theorem percentage_singing (t : ℕ) : (singing_percentage t) = (4 * (39 - t)) / 3 :=
by
  sorry

end NUMINAMATH_GPT_percentage_singing_l1937_193788


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1937_193714

theorem hyperbola_eccentricity (a : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y = (1 / 8) * x^2 → x^2 = 8 * y) →
  (∀ y x : ℝ, y^2 / a - x^2 = 1 → a + 1 = 4) →
  e^2 = 4 / 3 →
  e = 2 * Real.sqrt 3 / 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1937_193714


namespace NUMINAMATH_GPT_find_m_l1937_193768

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

def C_UA : Set ℕ := {1, 2}

theorem find_m (m : ℝ) (hA : A m = {0, 3}) (hCUA : U \ A m = C_UA) : m = -3 := 
  sorry

end NUMINAMATH_GPT_find_m_l1937_193768


namespace NUMINAMATH_GPT_rank_siblings_l1937_193760

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end NUMINAMATH_GPT_rank_siblings_l1937_193760


namespace NUMINAMATH_GPT_cost_of_toaster_l1937_193719

-- Definitions based on the conditions
def initial_spending : ℕ := 3000
def tv_return : ℕ := 700
def returned_bike_cost : ℕ := 500
def sold_bike_cost : ℕ := returned_bike_cost + (returned_bike_cost / 5)
def selling_price : ℕ := (4 * sold_bike_cost) / 5
def total_out_of_pocket : ℕ := 2020

-- Proving the cost of the toaster
theorem cost_of_toaster : initial_spending - (tv_return + returned_bike_cost) + selling_price - total_out_of_pocket = 260 := by
  sorry

end NUMINAMATH_GPT_cost_of_toaster_l1937_193719


namespace NUMINAMATH_GPT_perpendicular_k_value_parallel_k_value_l1937_193724

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)
def u (k : ℝ) : ℝ × ℝ := (k - 1, 2 * k + 2)
def v : ℝ × ℝ := (4, -4)

noncomputable def is_perpendicular (x y : ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

noncomputable def is_parallel (x y : ℝ × ℝ) : Prop :=
  x.1 * y.2 = x.2 * y.1

theorem perpendicular_k_value :
  is_perpendicular (u (-3)) v :=
by sorry

theorem parallel_k_value :
  is_parallel (u (-1/3)) v :=
by sorry

end NUMINAMATH_GPT_perpendicular_k_value_parallel_k_value_l1937_193724


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1937_193716

theorem negation_of_universal_proposition :
  (∃ x : ℤ, x % 5 = 0 ∧ ¬ (x % 2 = 1)) ↔ ¬ (∀ x : ℤ, x % 5 = 0 → (x % 2 = 1)) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1937_193716


namespace NUMINAMATH_GPT_maximize_profit_l1937_193729

noncomputable def profit (x : ℕ) : ℝ :=
  if x ≤ 200 then
    (0.40 - 0.24) * 30 * x
  else if x ≤ 300 then
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * x - (0.24 - 0.08) * 10 * (x - 200)
  else
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * 300 - (0.24 - 0.08) * 10 * (x - 200) - (0.24 - 0.08) * 20 * (x - 300)

theorem maximize_profit : ∀ x : ℕ, 
  profit 300 = 1120 ∧ (∀ y : ℕ, profit y ≤ 1120) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l1937_193729


namespace NUMINAMATH_GPT_div_fact_l1937_193727

-- Conditions
def fact_10 : ℕ := 3628800
def fact_4 : ℕ := 4 * 3 * 2 * 1

-- Question and Correct Answer
theorem div_fact (h : fact_10 = 3628800) : fact_10 / fact_4 = 151200 :=
by
  sorry

end NUMINAMATH_GPT_div_fact_l1937_193727


namespace NUMINAMATH_GPT_smallest_positive_debt_l1937_193778

theorem smallest_positive_debt :
  ∃ (D : ℕ) (p g : ℤ), 0 < D ∧ D = 350 * p + 240 * g ∧ D = 10 := sorry

end NUMINAMATH_GPT_smallest_positive_debt_l1937_193778


namespace NUMINAMATH_GPT_Alyssa_spent_on_marbles_l1937_193715

def total_spent_on_toys : ℝ := 12.30
def cost_of_football : ℝ := 5.71
def amount_spent_on_marbles : ℝ := 12.30 - 5.71

theorem Alyssa_spent_on_marbles :
  total_spent_on_toys - cost_of_football = amount_spent_on_marbles :=
by
  sorry

end NUMINAMATH_GPT_Alyssa_spent_on_marbles_l1937_193715


namespace NUMINAMATH_GPT_factorial_comparison_l1937_193789

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end NUMINAMATH_GPT_factorial_comparison_l1937_193789


namespace NUMINAMATH_GPT_find_m_l1937_193756

theorem find_m 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (h_f : ∀ x, f x = x^2 - 4*x + m)
  (h_g : ∀ x, g x = x^2 - 2*x + 2*m)
  (h_cond : 3 * f 3 = g 3)
  : m = 12 := 
sorry

end NUMINAMATH_GPT_find_m_l1937_193756


namespace NUMINAMATH_GPT_find_b_l1937_193741

def passesThrough (b c : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 + b * P.1 + c

theorem find_b (b c : ℝ)
  (H1 : passesThrough b c (1, 2))
  (H2 : passesThrough b c (5, 2)) :
  b = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1937_193741


namespace NUMINAMATH_GPT_card_statements_are_false_l1937_193725

theorem card_statements_are_false :
  ¬( ( (statements: ℕ) →
        (statements = 1 ↔ ¬statements = 1 ∧ ¬statements = 2 ∧ ¬statements = 3 ∧ ¬statements = 4 ∧ ¬statements = 5) ∧
        ( statements = 2 ↔ (statements = 1 ∨ statements = 3 ∨ statements = 4 ∨ statements = 5)) ∧
        (statements = 3 ↔ (statements = 1 ∧ statements = 2 ∧ (statements = 4 ∨ statements = 5) ) ) ∧
        (statements = 4 ↔ (statements = 1 ∧ statements = 2 ∧ statements = 3 ∧ statements != 5 ) ) ∧
        (statements = 5 ↔ (statements = 4 ) )
)) :=
sorry

end NUMINAMATH_GPT_card_statements_are_false_l1937_193725


namespace NUMINAMATH_GPT_quadratic_roots_eq1_quadratic_roots_eq2_l1937_193735

theorem quadratic_roots_eq1 :
  ∀ x : ℝ, (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
by
  intros x
  sorry

theorem quadratic_roots_eq2 :
  ∀ x : ℝ, ((x + 2)^2 = (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_quadratic_roots_eq1_quadratic_roots_eq2_l1937_193735


namespace NUMINAMATH_GPT_frank_problems_each_type_l1937_193799

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end NUMINAMATH_GPT_frank_problems_each_type_l1937_193799


namespace NUMINAMATH_GPT_grandfather_age_l1937_193777

variable (F S G : ℕ)

theorem grandfather_age (h1 : F = 58) (h2 : F - S = S) (h3 : S - 5 = (1 / 2) * G) : G = 48 := by
  sorry

end NUMINAMATH_GPT_grandfather_age_l1937_193777


namespace NUMINAMATH_GPT_ratio_of_incomes_l1937_193779

theorem ratio_of_incomes 
  (E1 E2 I1 I2 : ℕ)
  (h1 : E1 / E2 = 3 / 2)
  (h2 : E1 = I1 - 1200)
  (h3 : E2 = I2 - 1200)
  (h4 : I1 = 3000) :
  I1 / I2 = 5 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_incomes_l1937_193779


namespace NUMINAMATH_GPT_lines_symmetric_about_y_axis_l1937_193707

theorem lines_symmetric_about_y_axis (m n p : ℝ) :
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0)
  ↔ (m = -n ∧ p = -5) :=
sorry

end NUMINAMATH_GPT_lines_symmetric_about_y_axis_l1937_193707


namespace NUMINAMATH_GPT_simplify_fraction_l1937_193726

theorem simplify_fraction : (140 / 9800) * 35 = 1 / 70 := 
by
  -- Proof steps would go here.
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1937_193726


namespace NUMINAMATH_GPT_both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l1937_193709

variables (p1 p2 : Prop)

theorem both_shots_hit (p1 p2 : Prop) : (p1 ∧ p2) ↔ (p1 ∧ p2) :=
by sorry

theorem both_shots_missed (p1 p2 : Prop) : (¬p1 ∧ ¬p2) ↔ (¬p1 ∧ ¬p2) :=
by sorry

theorem exactly_one_shot_hit (p1 p2 : Prop) : ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) ↔ ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) :=
by sorry

theorem at_least_one_shot_hit (p1 p2 : Prop) : (p1 ∨ p2) ↔ (p1 ∨ p2) :=
by sorry

end NUMINAMATH_GPT_both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l1937_193709


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1937_193753

def range_of_k (k : ℝ) : Prop := (k ≥ 4) ∨ (k ≤ 2)

theorem quadratic_inequality_solution (k : ℝ) (x : ℝ) (h : x = 1) :
  k^2*x^2 - 6*k*x + 8 ≥ 0 → range_of_k k := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1937_193753


namespace NUMINAMATH_GPT_factorial_plus_one_div_prime_l1937_193798

theorem factorial_plus_one_div_prime (n : ℕ) (h : (n! + 1) % (n + 1) = 0) : Nat.Prime (n + 1) := 
sorry

end NUMINAMATH_GPT_factorial_plus_one_div_prime_l1937_193798


namespace NUMINAMATH_GPT_pentagon_PT_length_l1937_193757

theorem pentagon_PT_length (QR RS ST : ℝ) (angle_T right_angle_QRS T : Prop) (length_PT := (fun (a b : ℝ) => a + 3 * Real.sqrt b)) :
  QR = 3 →
  RS = 3 →
  ST = 3 →
  angle_T →
  right_angle_QRS →
  (angle_Q angle_R angle_S : ℝ) →
  angle_Q = 135 →
  angle_R = 135 →
  angle_S = 135 →
  ∃ (a b : ℝ), length_PT a b = 6 * Real.sqrt 2 ∧ a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_PT_length_l1937_193757


namespace NUMINAMATH_GPT_exists_y_square_divisible_by_five_btw_50_and_120_l1937_193787

theorem exists_y_square_divisible_by_five_btw_50_and_120 : ∃ y : ℕ, (∃ k : ℕ, y = k^2) ∧ (y % 5 = 0) ∧ (50 ≤ y ∧ y ≤ 120) ∧ y = 100 :=
by
  sorry

end NUMINAMATH_GPT_exists_y_square_divisible_by_five_btw_50_and_120_l1937_193787


namespace NUMINAMATH_GPT_diamond_value_l1937_193751

def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem diamond_value : diamond 7 3 = 22 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_diamond_value_l1937_193751


namespace NUMINAMATH_GPT_determine_values_of_a_and_b_l1937_193793

namespace MathProofProblem

variables (a b : ℤ)

theorem determine_values_of_a_and_b :
  (b + 1 = 2) ∧ (a - 1 ≠ -3) ∧ (a - 1 = -3) ∧ (b + 1 ≠ 2) ∧ (a - 1 = 2) ∧ (b + 1 = -3) →
  a = 3 ∧ b = -4 := by
  sorry

end MathProofProblem

end NUMINAMATH_GPT_determine_values_of_a_and_b_l1937_193793


namespace NUMINAMATH_GPT_pole_length_is_5_l1937_193790

theorem pole_length_is_5 (x : ℝ) (gate_width gate_height : ℝ) 
  (h_gate_wide : gate_width = 3) 
  (h_pole_taller : gate_height = x - 1) 
  (h_diagonal : x^2 = gate_height^2 + gate_width^2) : 
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_pole_length_is_5_l1937_193790


namespace NUMINAMATH_GPT_fish_lifespan_proof_l1937_193732

def hamster_lifespan : ℝ := 2.5

def dog_lifespan : ℝ := 4 * hamster_lifespan

def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_proof :
  fish_lifespan = 12 := 
  by
  sorry

end NUMINAMATH_GPT_fish_lifespan_proof_l1937_193732


namespace NUMINAMATH_GPT_mouse_seed_hiding_l1937_193744

theorem mouse_seed_hiding : 
  ∀ (h_m h_r x : ℕ), 
  4 * h_m = x →
  7 * h_r = x →
  h_m = h_r + 3 →
  x = 28 :=
by
  intros h_m h_r x H1 H2 H3
  sorry

end NUMINAMATH_GPT_mouse_seed_hiding_l1937_193744


namespace NUMINAMATH_GPT_stratified_sampling_correct_l1937_193785

-- Define the total number of employees
def total_employees : ℕ := 100

-- Define the number of employees in each age group
def under_30 : ℕ := 20
def between_30_and_40 : ℕ := 60
def over_40 : ℕ := 20

-- Define the number of people to be drawn
def total_drawn : ℕ := 20

-- Function to calculate number of people to be drawn from each group
def stratified_draw (group_size : ℕ) (total_size : ℕ) (drawn : ℕ) : ℕ :=
  (group_size * drawn) / total_size

-- The proof problem statement
theorem stratified_sampling_correct :
  stratified_draw under_30 total_employees total_drawn = 4 ∧
  stratified_draw between_30_and_40 total_employees total_drawn = 12 ∧
  stratified_draw over_40 total_employees total_drawn = 4 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l1937_193785


namespace NUMINAMATH_GPT_AdvancedVowelSoup_l1937_193764

noncomputable def AdvancedVowelSoup.sequence_count : ℕ :=
  let total_sequences := 7^7
  let vowel_only_sequences := 5^7
  let consonant_only_sequences := 2^7
  total_sequences - vowel_only_sequences - consonant_only_sequences

theorem AdvancedVowelSoup.valid_sequences : AdvancedVowelSoup.sequence_count = 745290 := by
  sorry

end NUMINAMATH_GPT_AdvancedVowelSoup_l1937_193764


namespace NUMINAMATH_GPT_sequence_linear_constant_l1937_193718

open Nat

theorem sequence_linear_constant (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a 1 ∧ a (n + 1) > a n)
  (h2 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := 
sorry

end NUMINAMATH_GPT_sequence_linear_constant_l1937_193718


namespace NUMINAMATH_GPT_milk_processing_days_required_l1937_193754

variable (a m x : ℝ) (n : ℝ)

theorem milk_processing_days_required
  (h1 : (n - a) * (x + m) = nx)
  (h2 : ax + (10 * a / 9) * x + (5 * a / 9) * m = 2 / 3)
  (h3 : nx = 1 / 2) :
  n = 2 * a :=
by sorry

end NUMINAMATH_GPT_milk_processing_days_required_l1937_193754


namespace NUMINAMATH_GPT_range_of_a_sqrt10_e_bounds_l1937_193773

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≤ g x) ↔ a ≤ 1 :=
by
  sorry

theorem sqrt10_e_bounds : 
  (1095 / 1000 : ℝ) < Real.exp (1/10 : ℝ) ∧ Real.exp (1/10 : ℝ) < (2000 / 1791 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_sqrt10_e_bounds_l1937_193773


namespace NUMINAMATH_GPT_max_marks_is_400_l1937_193797

theorem max_marks_is_400 :
  ∃ M : ℝ, (0.30 * M = 120) ∧ (M = 400) := 
by 
  sorry

end NUMINAMATH_GPT_max_marks_is_400_l1937_193797


namespace NUMINAMATH_GPT_sum_of_digits_of_number_of_rows_l1937_193705

theorem sum_of_digits_of_number_of_rows :
  ∃ N, (3 * (N * (N + 1) / 2) = 1575) ∧ (Nat.digits 10 N).sum = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_number_of_rows_l1937_193705


namespace NUMINAMATH_GPT_circle_tangent_parabola_height_difference_l1937_193792

theorem circle_tangent_parabola_height_difference
  (a b r : ℝ)
  (point_of_tangency_left : a ≠ 0)
  (points_of_tangency_on_parabola : (2 * a^2) = (2 * (-a)^2))
  (center_y_coordinate : ∃ c , c = b)
  (circle_equation_tangent_parabola : ∀ x, (x^2 + (2*x^2 - b)^2 = r^2))
  (quartic_double_root : ∀ x, (x = a ∨ x = -a) → (x^2 + (4 - 2*b)*x^2 + b^2 - r^2 = 0)) :
  b - 2 * a^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_parabola_height_difference_l1937_193792


namespace NUMINAMATH_GPT_quadrilateral_perimeter_l1937_193747

theorem quadrilateral_perimeter
  (EF FG HG : ℝ)
  (h1 : EF = 7)
  (h2 : FG = 15)
  (h3 : HG = 3)
  (perp1 : EF * FG = 0)
  (perp2 : HG * FG = 0) :
  EF + FG + HG + Real.sqrt (4^2 + 15^2) = 25 + Real.sqrt 241 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_perimeter_l1937_193747


namespace NUMINAMATH_GPT_sculpture_cost_in_INR_l1937_193766

def USD_per_NAD := 1 / 5
def INR_per_USD := 8
def cost_in_NAD := 200
noncomputable def cost_in_INR := (cost_in_NAD * USD_per_NAD) * INR_per_USD

theorem sculpture_cost_in_INR :
  cost_in_INR = 320 := by
  sorry

end NUMINAMATH_GPT_sculpture_cost_in_INR_l1937_193766


namespace NUMINAMATH_GPT_proof_problem_l1937_193737

noncomputable def f (x : ℝ) : ℝ :=
  Real.log ((1 + Real.sqrt x) / (1 - Real.sqrt x))

theorem proof_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  f ( (5 * x + 2 * x^2) / (1 + 5 * x + 3 * x^2) ) = Real.sqrt 5 * f x :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1937_193737


namespace NUMINAMATH_GPT_train_pass_jogger_in_40_seconds_l1937_193710

noncomputable def time_to_pass_jogger (jogger_speed_kmh : ℝ) (train_speed_kmh : ℝ) (initial_distance_m : ℝ) (train_length_m : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - jogger_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)  -- Conversion from km/hr to m/s
  let total_distance_m := initial_distance_m + train_length_m
  total_distance_m / relative_speed_ms

theorem train_pass_jogger_in_40_seconds :
  time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end NUMINAMATH_GPT_train_pass_jogger_in_40_seconds_l1937_193710


namespace NUMINAMATH_GPT_expression_value_l1937_193722

theorem expression_value :
  ∀ (x y : ℚ), (x = -5/4) → (y = -3/2) → -2 * x - y^2 = 1/4 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_expression_value_l1937_193722


namespace NUMINAMATH_GPT_not_lengths_of_external_diagonals_l1937_193734

theorem not_lengths_of_external_diagonals (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) :
  (¬ (a = 5 ∧ b = 6 ∧ c = 9)) :=
by
  sorry

end NUMINAMATH_GPT_not_lengths_of_external_diagonals_l1937_193734


namespace NUMINAMATH_GPT_invalid_perimeters_l1937_193771

theorem invalid_perimeters (x : ℕ) (h1 : 18 < x) (h2 : x < 42) :
  (42 + x ≠ 58) ∧ (42 + x ≠ 85) :=
by
  sorry

end NUMINAMATH_GPT_invalid_perimeters_l1937_193771


namespace NUMINAMATH_GPT_gas_and_maintenance_money_l1937_193717

theorem gas_and_maintenance_money
  (income : ℝ := 3200)
  (rent : ℝ := 1250)
  (utilities : ℝ := 150)
  (retirement_savings : ℝ := 400)
  (groceries : ℝ := 300)
  (insurance : ℝ := 200)
  (miscellaneous_expenses : ℝ := 200)
  (car_payment : ℝ := 350) :
  income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous_expenses + car_payment) = 350 :=
by
  sorry

end NUMINAMATH_GPT_gas_and_maintenance_money_l1937_193717
