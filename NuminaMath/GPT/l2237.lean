import Mathlib

namespace max_sqrt_distance_l2237_223746

theorem max_sqrt_distance (x y : ℝ) 
  (h : x^2 + y^2 - 4 * x - 4 * y + 6 = 0) : 
  ∃ z, z = 3 * Real.sqrt 2 ∧ ∀ w, w = Real.sqrt (x^2 + y^2) → w ≤ z :=
sorry

end max_sqrt_distance_l2237_223746


namespace total_distance_of_relay_race_l2237_223767

theorem total_distance_of_relay_race 
    (fraction_siwon : ℝ := 3/10) 
    (fraction_dawon : ℝ := 4/10) 
    (distance_together : ℝ := 140) :
    (fraction_siwon + fraction_dawon) * 200 = distance_together :=
by
    sorry

end total_distance_of_relay_race_l2237_223767


namespace consistent_values_l2237_223793

theorem consistent_values (a x: ℝ) :
    (12 * x^2 + 48 * x - a + 36 = 0) ∧ ((a + 60) * x - 3 * (a - 20) = 0) ↔
    ((a = -12 ∧ x = -2) ∨ (a = 0 ∧ x = -1) ∨ (a = 180 ∧ x = 2)) := 
by
  -- proof steps should be filled here
  sorry

end consistent_values_l2237_223793


namespace range_of_a_l2237_223794

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 ≤ 0

theorem range_of_a :
  ¬ quadratic_inequality a ↔ -1 < a ∧ a < 3 :=
  by
  sorry

end range_of_a_l2237_223794


namespace number_of_pear_trees_l2237_223795

theorem number_of_pear_trees (A P : ℕ) (h1 : A + P = 46)
  (h2 : ∀ (s : Finset (Fin 46)), s.card = 28 → ∃ (i : Fin 46), i ∈ s ∧ i < A)
  (h3 : ∀ (s : Finset (Fin 46)), s.card = 20 → ∃ (i : Fin 46), i ∈ s ∧ A ≤ i) :
  P = 27 :=
by
  sorry

end number_of_pear_trees_l2237_223795


namespace vendelin_pastels_l2237_223791

theorem vendelin_pastels (M V W : ℕ) (h1 : M = 5) (h2 : V < 5) (h3 : W = M + V) (h4 : M + V + W = 7 * V) : W = 7 := 
sorry

end vendelin_pastels_l2237_223791


namespace sum_of_three_numbers_l2237_223778

theorem sum_of_three_numbers (x y z : ℕ) (h1 : x ≤ y) (h2 : y ≤ z) (h3 : y = 7) 
    (h4 : (x + y + z) / 3 = x + 12) (h5 : (x + y + z) / 3 = z - 18) : 
    x + y + z = 39 :=
by
  sorry

end sum_of_three_numbers_l2237_223778


namespace no_integer_solutions_l2237_223740

theorem no_integer_solutions (w l : ℕ) (hw_pos : 0 < w) (hl_pos : 0 < l) : 
  (w * l = 24 ∧ (w = l ∨ 2 * l = w)) → false :=
by 
  sorry

end no_integer_solutions_l2237_223740


namespace qualified_products_correct_l2237_223789

def defect_rate : ℝ := 0.005
def total_produced : ℝ := 18000

theorem qualified_products_correct :
  total_produced * (1 - defect_rate) = 17910 := by
  sorry

end qualified_products_correct_l2237_223789


namespace exists_matrices_B_C_not_exists_matrices_commute_l2237_223728

-- Equivalent proof statement for part (a)
theorem exists_matrices_B_C (A : Matrix (Fin 2) (Fin 2) ℝ): 
  ∃ (B C : Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 :=
by
  sorry

-- Equivalent proof statement for part (b)
theorem not_exists_matrices_commute (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = ![![0, 1], ![1, 0]]) :
  ¬∃ (B C: Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 ∧ B * C = C * B :=
by
  sorry

end exists_matrices_B_C_not_exists_matrices_commute_l2237_223728


namespace airplane_seats_theorem_l2237_223788

def airplane_seats_proof : Prop :=
  ∀ (s : ℝ),
  (∃ (first_class business_class economy premium_economy : ℝ),
    first_class = 30 ∧
    business_class = 0.4 * s ∧
    economy = 0.6 * s ∧
    premium_economy = s - (first_class + business_class + economy)) →
  s = 150

theorem airplane_seats_theorem : airplane_seats_proof :=
sorry

end airplane_seats_theorem_l2237_223788


namespace total_cows_is_108_l2237_223742

-- Definitions of the sons' shares and the number of cows the fourth son received
def first_son_share : ℚ := 2 / 3
def second_son_share : ℚ := 1 / 6
def third_son_share : ℚ := 1 / 9
def fourth_son_cows : ℕ := 6

-- The total number of cows in the herd
def total_cows (n : ℕ) : Prop :=
  first_son_share + second_son_share + third_son_share + (fourth_son_cows / n) = 1

-- Prove that given the number of cows the fourth son received, the total number of cows in the herd is 108
theorem total_cows_is_108 : total_cows 108 :=
by
  sorry

end total_cows_is_108_l2237_223742


namespace proof_problem_1_proof_problem_2_l2237_223705

noncomputable def problem_1 (a b : ℝ) : Prop :=
  ((2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3))) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6)

noncomputable def problem_2 : Prop :=
  ((2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 2^(3/4 - 1) - (-2005)^0) = 100

theorem proof_problem_1 (a b : ℝ) : problem_1 a b := 
  sorry

theorem proof_problem_2 : problem_2 := 
  sorry

end proof_problem_1_proof_problem_2_l2237_223705


namespace opposite_of_neg5_l2237_223766

-- Define the concept of the opposite of a number
def opposite (x : Int) : Int :=
  -x

-- The proof problem: Prove that the opposite of -5 is 5
theorem opposite_of_neg5 : opposite (-5) = 5 :=
by
  sorry

end opposite_of_neg5_l2237_223766


namespace tom_savings_l2237_223738

theorem tom_savings :
  let insurance_cost_per_month := 20
  let total_months := 24
  let procedure_cost := 5000
  let insurance_coverage := 0.80
  let total_insurance_cost := total_months * insurance_cost_per_month
  let insurance_cover_amount := procedure_cost * insurance_coverage
  let out_of_pocket_cost := procedure_cost - insurance_cover_amount
  let savings := procedure_cost - total_insurance_cost - out_of_pocket_cost
  savings = 3520 :=
by
  sorry

end tom_savings_l2237_223738


namespace number_of_integer_values_l2237_223720

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 9 * x^2 + 2 * x + 17

theorem number_of_integer_values :
  (∃ xs : List ℤ, xs.length = 4 ∧ ∀ x ∈ xs, Nat.Prime (Int.natAbs (Q x))) :=
by
  sorry

end number_of_integer_values_l2237_223720


namespace quadratic_inequality_solution_l2237_223701

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x - 21 ≤ 0 ↔ -3 ≤ x ∧ x ≤ 7 :=
sorry

end quadratic_inequality_solution_l2237_223701


namespace hyperbola_properties_l2237_223757

theorem hyperbola_properties (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (c := Real.sqrt (a^2 + b^2))
  (F2 := (c, 0)) (P : ℝ × ℝ)
  (h_perpendicular : ∃ (x y : ℝ), P = (x, y) ∧ y = -a/b * (x - c))
  (h_distance : Real.sqrt ((P.1 - c)^2 + P.2^2) = 2)
  (h_slope : P.2 / (P.1 - c) = -1/2) :
  
  b = 2 ∧
  (∀ x y, x^2 - y^2 / 4 = 1 ↔ x^2 - y^2 / b^2 = 1) ∧
  P = (Real.sqrt (5) / 5, 2 * Real.sqrt (5) / 5) :=
sorry

end hyperbola_properties_l2237_223757


namespace xyz_value_l2237_223796

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (xy + xz + yz) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3)
  : xyz = 5 :=
by
  sorry

end xyz_value_l2237_223796


namespace ratio_of_distances_l2237_223786

theorem ratio_of_distances (d_5 d_4 : ℝ) (h1 : d_5 + d_4 ≤ 26.67) (h2 : d_5 / 5 + d_4 / 4 = 6) : 
  d_5 / (d_5 + d_4) = 1 / 2 :=
sorry

end ratio_of_distances_l2237_223786


namespace fraction_inequality_l2237_223721

theorem fraction_inequality (a : ℝ) (h : a ≠ 2) : (1 / (a^2 - 4 * a + 4) > 2 / (a^3 - 8)) :=
by sorry

end fraction_inequality_l2237_223721


namespace total_sleep_correct_l2237_223777

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l2237_223777


namespace fraction_of_work_left_l2237_223753

variable (A_work_days : ℕ) (B_work_days : ℕ) (work_days_together: ℕ)

theorem fraction_of_work_left (hA : A_work_days = 15) (hB : B_work_days = 20) (hT : work_days_together = 4):
  (1 - 4 * (1 / 15 + 1 / 20)) = (8 / 15) :=
sorry

end fraction_of_work_left_l2237_223753


namespace min_solutions_f_eq_zero_l2237_223732

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 3) = f x)
variable (h_zero_at_2 : f 2 = 0)

theorem min_solutions_f_eq_zero : ∃ S : Finset ℝ, (∀ x ∈ S, f x = 0) ∧ 7 ≤ S.card ∧ (∀ x ∈ S, x > 0 ∧ x < 6) := 
sorry

end min_solutions_f_eq_zero_l2237_223732


namespace inverse_variation_solution_l2237_223783

noncomputable def const_k (x y : ℝ) := (x^2) * (y^4)

theorem inverse_variation_solution (x y : ℝ) (k : ℝ) (h1 : x = 8) (h2 : y = 2) (h3 : k = const_k x y) :
  ∀ y' : ℝ, y' = 4 → const_k x y' = 1024 → x^2 = 4 := by
  intros
  sorry

end inverse_variation_solution_l2237_223783


namespace books_bought_at_bookstore_l2237_223764

-- Define the initial count of books
def initial_books : ℕ := 72

-- Define the number of books received each month from the book club
def books_from_club (months : ℕ) : ℕ := months

-- Number of books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Number of books bought
def books_from_yard_sales : ℕ := 2

-- Number of books donated and sold
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final total count of books
def final_books : ℕ := 81

-- Calculate the number of books acquired and then removed, and prove 
-- the number of books bought at the bookstore halfway through the year
theorem books_bought_at_bookstore (months : ℕ) (b : ℕ) :
  initial_books + books_from_club months + books_from_daughter + books_from_mother + books_from_yard_sales + b - books_donated - books_sold = final_books → b = 5 :=
by sorry

end books_bought_at_bookstore_l2237_223764


namespace suji_age_problem_l2237_223727

theorem suji_age_problem (x : ℕ) 
  (h1 : 5 * x + 6 = 13 * (4 * x + 6) / 11)
  (h2 : 11 * (4 * x + 6) = 9 * (3 * x + 6)) :
  4 * x = 16 :=
by
  sorry

end suji_age_problem_l2237_223727


namespace rain_on_first_day_l2237_223782

theorem rain_on_first_day (x : ℝ) (h1 : x >= 0)
  (h2 : (2 * x) + 50 / 100 * (2 * x) = 3 * x) 
  (h3 : 6 * 12 = 72)
  (h4 : 3 * 3 = 9)
  (h5 : x + 2 * x + 3 * x = 6 * x)
  (h6 : 6 * x + 21 - 9 = 72) : x = 10 :=
by 
  -- Proof would go here, but we skip it according to instructions
  sorry

end rain_on_first_day_l2237_223782


namespace price_of_first_candy_l2237_223731

theorem price_of_first_candy (P: ℝ) 
  (total_weight: ℝ) (price_per_lb_mixture: ℝ) 
  (weight_first: ℝ) (weight_second: ℝ) 
  (price_per_lb_second: ℝ) :
  total_weight = 30 →
  price_per_lb_mixture = 3 →
  weight_first = 20 →
  weight_second = 10 →
  price_per_lb_second = 3.1 →
  20 * P + 10 * price_per_lb_second = total_weight * price_per_lb_mixture →
  P = 2.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_of_first_candy_l2237_223731


namespace find_required_school_year_hours_l2237_223768

-- Define constants for the problem
def summer_hours_per_week : ℕ := 40
def summer_weeks : ℕ := 12
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 36
def school_year_earnings : ℕ := 9000

-- Calculate total summer hours, hourly rate, total school year hours, and required school year weekly hours
def total_summer_hours := summer_hours_per_week * summer_weeks
def hourly_rate := summer_earnings / total_summer_hours
def total_school_year_hours := school_year_earnings / hourly_rate
def required_school_year_hours_per_week := total_school_year_hours / school_year_weeks

-- Prove the required hours per week is 20
theorem find_required_school_year_hours : required_school_year_hours_per_week = 20 := by
  sorry

end find_required_school_year_hours_l2237_223768


namespace solve_eq_l2237_223735

theorem solve_eq : ∃ x : ℝ, 6 * x - 4 * x = 380 - 10 * (x + 2) ∧ x = 30 := 
by
  sorry

end solve_eq_l2237_223735


namespace even_function_f_D_l2237_223784

noncomputable def f_A (x : ℝ) : ℝ := 2 * |x| - 1
def D_f_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

def f_B (x : ℕ) : ℕ := x^2 + x

def f_C (x : ℝ) : ℝ := x ^ 3

noncomputable def f_D (x : ℝ) : ℝ := x^2
def D_f_D := {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)}

theorem even_function_f_D : 
  ∀ x ∈ D_f_D, f_D (-x) = f_D (x) :=
sorry

end even_function_f_D_l2237_223784


namespace tangency_condition_l2237_223723

-- Define the equation for the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 = 9

-- Define the equation for the hyperbola
def hyperbola_eq (x y m : ℝ) : Prop :=
  (x - 2)^2 - m * (y + 1)^2 = 1

-- Prove that for the ellipse and hyperbola to be tangent, m must equal 3
theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y ∧ hyperbola_eq x y m) → m = 3 :=
by
  sorry

end tangency_condition_l2237_223723


namespace fraction_inhabitable_earth_surface_l2237_223743

theorem fraction_inhabitable_earth_surface 
  (total_land_fraction: ℚ) 
  (inhabitable_land_fraction: ℚ) 
  (h1: total_land_fraction = 1/3) 
  (h2: inhabitable_land_fraction = 2/3) 
  : (total_land_fraction * inhabitable_land_fraction) = 2/9 :=
by
  sorry

end fraction_inhabitable_earth_surface_l2237_223743


namespace joshua_total_payment_is_correct_l2237_223708

noncomputable def total_cost : ℝ := 
  let t_shirt_price := 8
  let sweater_price := 18
  let jacket_price := 80
  let jeans_price := 35
  let shoes_price := 60
  let jacket_discount := 0.10
  let shoes_discount := 0.15
  let clothing_tax_rate := 0.05
  let shoes_tax_rate := 0.08

  let t_shirt_count := 6
  let sweater_count := 4
  let jacket_count := 5
  let jeans_count := 3
  let shoes_count := 2

  let t_shirts_subtotal := t_shirt_price * t_shirt_count
  let sweaters_subtotal := sweater_price * sweater_count
  let jackets_subtotal := jacket_price * jacket_count
  let jeans_subtotal := jeans_price * jeans_count
  let shoes_subtotal := shoes_price * shoes_count

  let jackets_discounted := jackets_subtotal * (1 - jacket_discount)
  let shoes_discounted := shoes_subtotal * (1 - shoes_discount)

  let total_before_tax := t_shirts_subtotal + sweaters_subtotal + jackets_discounted + jeans_subtotal + shoes_discounted

  let t_shirts_tax := t_shirts_subtotal * clothing_tax_rate
  let sweaters_tax := sweaters_subtotal * clothing_tax_rate
  let jackets_tax := jackets_discounted * clothing_tax_rate
  let jeans_tax := jeans_subtotal * clothing_tax_rate
  let shoes_tax := shoes_discounted * shoes_tax_rate

  total_before_tax + t_shirts_tax + sweaters_tax + jackets_tax + jeans_tax + shoes_tax

theorem joshua_total_payment_is_correct : total_cost = 724.41 := by
  sorry

end joshua_total_payment_is_correct_l2237_223708


namespace jamie_cherry_pies_l2237_223733

theorem jamie_cherry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36) (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) : 
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := 
by {
  sorry
}

end jamie_cherry_pies_l2237_223733


namespace probability_major_A_less_than_25_l2237_223771

def total_students : ℕ := 100 -- assuming a total of 100 students for simplicity

def male_percent : ℝ := 0.40
def major_A_percent : ℝ := 0.50
def major_B_percent : ℝ := 0.30
def major_C_percent : ℝ := 0.20
def major_A_25_or_older_percent : ℝ := 0.60
def major_A_less_than_25_percent : ℝ := 1 - major_A_25_or_older_percent

theorem probability_major_A_less_than_25 :
  (major_A_percent * major_A_less_than_25_percent) = 0.20 :=
by
  sorry

end probability_major_A_less_than_25_l2237_223771


namespace sphere_surface_area_l2237_223785

theorem sphere_surface_area (a : ℝ) (d : ℝ) (S : ℝ) : 
  a = 3 → d = Real.sqrt 7 → S = 40 * Real.pi := by
  sorry

end sphere_surface_area_l2237_223785


namespace problem_value_l2237_223707

theorem problem_value :
  (1 / 3 * 9 * 1 / 27 * 81 * 1 / 243 * 729 * 1 / 2187 * 6561 * 1 / 19683 * 59049) = 243 := 
sorry

end problem_value_l2237_223707


namespace percentage_profit_is_35_l2237_223718

-- Define the conditions
def initial_cost_price : ℝ := 100
def markup_percentage : ℝ := 0.5
def discount_percentage : ℝ := 0.1
def marked_price : ℝ := initial_cost_price * (1 + markup_percentage)
def selling_price : ℝ := marked_price * (1 - discount_percentage)

-- Define the statement/proof problem
theorem percentage_profit_is_35 :
  (selling_price - initial_cost_price) / initial_cost_price * 100 = 35 := by 
  sorry

end percentage_profit_is_35_l2237_223718


namespace solve_for_k_l2237_223726

theorem solve_for_k (k : ℤ) : (∃ x : ℤ, x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 :=
by
  sorry

end solve_for_k_l2237_223726


namespace arithmetic_sequence_inequality_l2237_223712

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

-- All terms are positive
def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem arithmetic_sequence_inequality
  (h_arith_seq : is_arithmetic_sequence a a1 d)
  (h_non_zero_diff : d ≠ 0)
  (h_positive : all_positive a) :
  (a 1) * (a 8) < (a 4) * (a 5) :=
by
  sorry

end arithmetic_sequence_inequality_l2237_223712


namespace sum_max_min_f_l2237_223722

noncomputable def f (x : ℝ) : ℝ :=
  1 + (Real.sin x / (2 + Real.cos x))

theorem sum_max_min_f {a b : ℝ} (ha : ∀ x, f x ≤ a) (hb : ∀ x, b ≤ f x) (h_max : ∃ x, f x = a) (h_min : ∃ x, f x = b) :
  a + b = 2 :=
sorry

end sum_max_min_f_l2237_223722


namespace find_point_W_coordinates_l2237_223775

theorem find_point_W_coordinates 
(O U S V : ℝ × ℝ)
(hO : O = (0, 0))
(hU : U = (3, 3))
(hS : S = (3, 0))
(hV : V = (0, 3))
(hSquare : (O.1 - U.1)^2 + (O.2 - U.2)^2 = 18)
(hArea_Square : 3 * 3 = 9) :
  ∃ W : ℝ × ℝ, W = (3, 9) ∧ 1 / 2 * (abs (S.1 - V.1) * abs (W.2 - S.2)) = 9 :=
by
  sorry

end find_point_W_coordinates_l2237_223775


namespace sampling_probabilities_equal_l2237_223747

noncomputable def populationSize (N : ℕ) := N
noncomputable def sampleSize (n : ℕ) := n

def P1 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P2 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P3 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)

theorem sampling_probabilities_equal (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  P1 N n = P2 N n ∧ P2 N n = P3 N n :=
by
  -- Proof steps will go here
  sorry

end sampling_probabilities_equal_l2237_223747


namespace smallest_two_ks_l2237_223781

theorem smallest_two_ks (k : ℕ) (h : ℕ → Prop) : 
  (∀ k, (k^2 + 36) % 180 = 0 → k = 12 ∨ k = 18) :=
by {
 sorry
}

end smallest_two_ks_l2237_223781


namespace stickers_per_friend_l2237_223792

variable (d: ℕ) (h_d : d > 0)

theorem stickers_per_friend (h : 72 % d = 0) : 72 / d = 72 / d := by
  sorry

end stickers_per_friend_l2237_223792


namespace weighted_mean_calculation_l2237_223739

/-- Prove the weighted mean of the numbers 16, 28, and 45 with weights 2, 3, and 5 is 34.1 -/
theorem weighted_mean_calculation :
  let numbers := [16, 28, 45]
  let weights := [2, 3, 5]
  let total_weight := (2 + 3 + 5 : ℝ)
  let weighted_sum := ((16 * 2 + 28 * 3 + 45 * 5) : ℝ)
  (weighted_sum / total_weight) = 34.1 :=
by
  -- We only state the theorem without providing the proof
  sorry

end weighted_mean_calculation_l2237_223739


namespace range_of_x_l2237_223752

theorem range_of_x {a : ℝ} : 
  (∀ a : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (x = 0 ∨ x = -2) :=
by sorry

end range_of_x_l2237_223752


namespace ellipse_foci_coordinates_l2237_223751

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) → ∃ c : ℝ, (c = 4) ∧ (x = c ∨ x = -c) ∧ (y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l2237_223751


namespace find_integer_solutions_l2237_223763

theorem find_integer_solutions :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    (a * b - 2 * c * d = 3) ∧ (a * c + b * d = 1) } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end find_integer_solutions_l2237_223763


namespace base2_to_base4_conversion_l2237_223724

/-- Definition of base conversion from binary to quaternary. -/
def bin_to_quat (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  if n = 10 then 2 else
  if n = 11 then 3 else
  0 -- (more cases can be added as necessary)

theorem base2_to_base4_conversion :
  bin_to_quat 1 * 4^4 + bin_to_quat 1 * 4^3 + bin_to_quat 10 * 4^2 + bin_to_quat 11 * 4^1 + bin_to_quat 10 * 4^0 = 11232 :=
by sorry

end base2_to_base4_conversion_l2237_223724


namespace lauren_time_8_miles_l2237_223745

-- Conditions
def time_alex_run_6_miles : ℕ := 36
def time_lauren_run_5_miles : ℕ := time_alex_run_6_miles / 3
def time_per_mile_lauren : ℚ := time_lauren_run_5_miles / 5

-- Proof statement
theorem lauren_time_8_miles : 8 * time_per_mile_lauren = 19.2 := by
  sorry

end lauren_time_8_miles_l2237_223745


namespace meeting_lamppost_l2237_223700

-- Define the initial conditions of the problem
def lampposts : ℕ := 400
def start_alla : ℕ := 1
def start_boris : ℕ := 400
def meet_alla : ℕ := 55
def meet_boris : ℕ := 321

-- Define a theorem that we need to prove: Alla and Boris will meet at the 163rd lamppost
theorem meeting_lamppost : ∃ (n : ℕ), n = 163 := 
by {
  sorry -- Proof goes here
}

end meeting_lamppost_l2237_223700


namespace largest_common_term_l2237_223755

-- Definitions for the first arithmetic sequence
def arithmetic_seq1 (n : ℕ) : ℕ := 2 + 5 * n

-- Definitions for the second arithmetic sequence
def arithmetic_seq2 (m : ℕ) : ℕ := 5 + 8 * m

-- Main statement of the problem
theorem largest_common_term (n m k : ℕ) (a : ℕ) :
  (a = arithmetic_seq1 n) ∧ (a = arithmetic_seq2 m) ∧ (1 ≤ a) ∧ (a ≤ 150) →
  a = 117 :=
by {
  sorry
}

end largest_common_term_l2237_223755


namespace range_of_r_l2237_223711

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem range_of_r (r : ℝ) (hr: 0 < r) : (M ∩ N r = N r) → r ≤ 2 - Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_r_l2237_223711


namespace total_flowers_bouquets_l2237_223779

-- Define the number of tulips Lana picked
def tulips : ℕ := 36

-- Define the number of roses Lana picked
def roses : ℕ := 37

-- Define the number of extra flowers Lana picked
def extra_flowers : ℕ := 3

-- Prove that the total number of flowers used by Lana for the bouquets is 76
theorem total_flowers_bouquets : (tulips + roses + extra_flowers) = 76 :=
by
  sorry

end total_flowers_bouquets_l2237_223779


namespace thirtieth_triangular_number_is_465_l2237_223787

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem thirtieth_triangular_number_is_465 : triangular_number 30 = 465 :=
by
  sorry

end thirtieth_triangular_number_is_465_l2237_223787


namespace even_numbers_set_l2237_223762

-- Define the set of all even numbers in set-builder notation
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Theorem stating that this set is the set of all even numbers
theorem even_numbers_set :
  ∀ x : ℤ, (x ∈ even_set ↔ ∃ n : ℤ, x = 2 * n) := by
  sorry

end even_numbers_set_l2237_223762


namespace frank_initial_money_l2237_223717

theorem frank_initial_money (X : ℝ) (h1 : X * (4 / 5) * (3 / 4) * (6 / 7) * (2 / 3) = 600) : X = 2333.33 :=
sorry

end frank_initial_money_l2237_223717


namespace Mabel_gave_away_daisies_l2237_223706

-- Setting up the conditions
variables (d_total : ℕ) (p_per_daisy : ℕ) (p_remaining : ℕ)

-- stating the assumptions
def initial_petals (d_total p_per_daisy : ℕ) := d_total * p_per_daisy
def petals_given_away (d_total p_per_daisy p_remaining : ℕ) := initial_petals d_total p_per_daisy - p_remaining
def daisies_given_away (d_total p_per_daisy p_remaining : ℕ) := petals_given_away d_total p_per_daisy p_remaining / p_per_daisy

-- The main theorem
theorem Mabel_gave_away_daisies 
  (h1 : d_total = 5)
  (h2 : p_per_daisy = 8)
  (h3 : p_remaining = 24) :
  daisies_given_away d_total p_per_daisy p_remaining = 2 :=
sorry

end Mabel_gave_away_daisies_l2237_223706


namespace coaches_needed_l2237_223773

theorem coaches_needed (x : ℕ) : 44 * x + 64 = 328 := by
  sorry

end coaches_needed_l2237_223773


namespace no_integers_satisfy_l2237_223709

theorem no_integers_satisfy (a b c d : ℤ) : ¬ (a^4 + b^4 + c^4 + 2016 = 10 * d) :=
sorry

end no_integers_satisfy_l2237_223709


namespace minimum_value_a5_a6_l2237_223749

-- Defining the arithmetic geometric sequence relational conditions.
def arithmetic_geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6) ∧ (∀ n, a n > 0)

-- The mathematical problem to prove:
theorem minimum_value_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h : arithmetic_geometric_sequence_condition a q) :
  a 5 + a 6 = 48 :=
sorry

end minimum_value_a5_a6_l2237_223749


namespace problem_part1_problem_part2_l2237_223730

-- Define the sequences and conditions
variable {a b : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}
variable {d q : ℕ}
variable {b_initial : ℕ}

axiom geom_seq (n : ℕ) : b n = b_initial * q^n
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Problem conditions
axiom cond_geom_seq : b_initial = 2
axiom cond_geom_b2_b3 : b 2 + b 3 = 12
axiom cond_geom_ratio : q > 0
axiom cond_relation_b3_a4 : b 3 = a 4 - 2 * a 1
axiom cond_sum_S_11_b4 : S 11 = 11 * b 4

-- Theorem statement
theorem problem_part1 :
  (a n = 3 * n - 2) ∧ (b n = 2 ^ n) :=
  sorry

theorem problem_part2 :
  (T n = (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) :=
  sorry

end problem_part1_problem_part2_l2237_223730


namespace new_sequence_69th_term_l2237_223716

-- Definitions and conditions
def original_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ := a n

def new_sequence (a : ℕ → ℕ) (k : ℕ) : ℕ :=
if k % 4 = 1 then a (k / 4 + 1) else 0  -- simplified modeling, the inserted numbers are denoted arbitrarily as 0

-- The statement to be proven
theorem new_sequence_69th_term (a : ℕ → ℕ) : new_sequence a 69 = a 18 :=
by
  sorry

end new_sequence_69th_term_l2237_223716


namespace arithmetic_mean_two_digit_multiples_of_8_l2237_223780

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l2237_223780


namespace isabel_spending_ratio_l2237_223759

theorem isabel_spending_ratio :
  ∀ (initial_amount toy_cost remaining_amount : ℝ),
    initial_amount = 204 ∧
    toy_cost = initial_amount / 2 ∧
    remaining_amount = 51 →
    ((initial_amount - toy_cost - remaining_amount) / remaining_amount) = 1 / 2 :=
by
  intros
  sorry

end isabel_spending_ratio_l2237_223759


namespace carpet_length_l2237_223710

theorem carpet_length (percent_covered : ℝ) (width : ℝ) (floor_area : ℝ) (carpet_length : ℝ) :
  percent_covered = 0.30 → width = 4 → floor_area = 120 → carpet_length = 9 :=
by
  sorry

end carpet_length_l2237_223710


namespace remainder_of_poly_div_l2237_223798

theorem remainder_of_poly_div (x : ℤ) : 
  (x + 1)^2009 % (x^2 + x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_div_l2237_223798


namespace total_books_is_10_l2237_223760

def total_books (B : ℕ) : Prop :=
  (2 / 5 : ℚ) * B + (3 / 10 : ℚ) * B + ((3 / 10 : ℚ) * B - 1) + 1 = B

theorem total_books_is_10 : total_books 10 := by
  sorry

end total_books_is_10_l2237_223760


namespace sum_k1_k2_k3_l2237_223714

theorem sum_k1_k2_k3 :
  ∀ (k1 k2 k3 t1 t2 t3 : ℝ),
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  t1 = (5 / 9) * (k1 - 32) →
  t2 = (5 / 9) * (k2 - 32) →
  t3 = (5 / 9) * (k3 - 32) →
  k1 + k2 + k3 = 510 :=
by
  intros k1 k2 k3 t1 t2 t3 ht1 ht2 ht3 ht1k1 ht2k2 ht3k3
  sorry

end sum_k1_k2_k3_l2237_223714


namespace money_left_after_shopping_l2237_223741

-- Conditions
def cost_mustard_oil : ℤ := 2 * 13
def cost_pasta : ℤ := 3 * 4
def cost_sauce : ℤ := 1 * 5
def total_cost : ℤ := cost_mustard_oil + cost_pasta + cost_sauce
def total_money : ℤ := 50

-- Theorem to prove
theorem money_left_after_shopping : total_money - total_cost = 7 := by
  sorry

end money_left_after_shopping_l2237_223741


namespace lisa_speed_l2237_223758

-- Define conditions
def distance : ℕ := 256
def time : ℕ := 8

-- Define the speed calculation theorem
theorem lisa_speed : (distance / time) = 32 := 
by {
  sorry
}

end lisa_speed_l2237_223758


namespace equation_is_point_l2237_223774

-- Definition of the condition in the problem
def equation (x y : ℝ) := x^2 + 36*y^2 - 12*x - 72*y + 36 = 0

-- The theorem stating the equivalence to the point (6, 1)
theorem equation_is_point :
  ∀ (x y : ℝ), equation x y → (x = 6 ∧ y = 1) :=
by
  intros x y h
  -- The proof steps would go here
  sorry

end equation_is_point_l2237_223774


namespace rationalize_denominator_l2237_223765

theorem rationalize_denominator (a b c : Real) (h : b*c*c = a) :
  2 / (b + c) = (c*c) / (3 * 2) :=
by
  sorry

end rationalize_denominator_l2237_223765


namespace expression_value_l2237_223756

theorem expression_value : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end expression_value_l2237_223756


namespace product_of_numbers_l2237_223770

variable {x y : ℝ}

theorem product_of_numbers (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 40 * k) : 
  x * y = 6400 / 63 := by
  sorry

end product_of_numbers_l2237_223770


namespace projection_of_b_onto_a_l2237_223737

open Real

noncomputable def e1 : ℝ × ℝ := (1, 0)
noncomputable def e2 : ℝ × ℝ := (0, 1)

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude u)

theorem projection_of_b_onto_a : projection b a = 2 * sqrt 5 / 5 := by
  sorry

end projection_of_b_onto_a_l2237_223737


namespace invalid_speed_against_stream_l2237_223713

theorem invalid_speed_against_stream (rate_still_water speed_with_stream : ℝ) (h1 : rate_still_water = 6) (h2 : speed_with_stream = 20) :
  ∃ (v : ℝ), speed_with_stream = rate_still_water + v ∧ (rate_still_water - v < 0) → false :=
by
  sorry

end invalid_speed_against_stream_l2237_223713


namespace not_diff_of_squares_2022_l2237_223725

theorem not_diff_of_squares_2022 :
  ¬ ∃ a b : ℤ, a^2 - b^2 = 2022 :=
by
  sorry

end not_diff_of_squares_2022_l2237_223725


namespace point_B_coordinates_sum_l2237_223769

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l2237_223769


namespace total_guppies_correct_l2237_223715

-- Define the initial conditions as variables
def initial_guppies : ℕ := 7
def baby_guppies_1 : ℕ := 3 * 12
def baby_guppies_2 : ℕ := 9

-- Define the total number of guppies
def total_guppies : ℕ := initial_guppies + baby_guppies_1 + baby_guppies_2

-- Theorem: Proving the total number of guppies is 52
theorem total_guppies_correct : total_guppies = 52 :=
by
  sorry

end total_guppies_correct_l2237_223715


namespace triplet_zero_solution_l2237_223703

theorem triplet_zero_solution (x y z : ℝ) 
  (h1 : x^3 + y = z^2) 
  (h2 : y^3 + z = x^2) 
  (h3 : z^3 + x = y^2) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end triplet_zero_solution_l2237_223703


namespace find_point_P_l2237_223702

noncomputable def f (x : ℝ) : ℝ := x^2 - x

theorem find_point_P :
  (∃ x y : ℝ, f x = y ∧ (2 * x - 1 = 1) ∧ (y = x^2 - x)) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end find_point_P_l2237_223702


namespace minimum_expression_value_l2237_223748

theorem minimum_expression_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := 
by
  sorry

end minimum_expression_value_l2237_223748


namespace solve_inequality_l2237_223790

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end solve_inequality_l2237_223790


namespace min_time_needed_l2237_223729

-- Define the conditions and required time for shoeing horses
def num_blacksmiths := 48
def num_horses := 60
def hooves_per_horse := 4
def time_per_hoof := 5
def total_hooves := num_horses * hooves_per_horse
def total_time_one_blacksmith := total_hooves * time_per_hoof
def min_time (num_blacksmiths : Nat) (total_time_one_blacksmith : Nat) : Nat :=
  total_time_one_blacksmith / num_blacksmiths

-- Prove that the minimum time needed is 25 minutes
theorem min_time_needed : min_time num_blacksmiths total_time_one_blacksmith = 25 :=
by
  sorry

end min_time_needed_l2237_223729


namespace value_of_product_l2237_223736

theorem value_of_product : (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by sorry

end value_of_product_l2237_223736


namespace sum_xy_l2237_223719

theorem sum_xy (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 10) : x + y = 14 ∨ x + y = -2 :=
sorry

end sum_xy_l2237_223719


namespace trailing_zeros_a6_l2237_223776

theorem trailing_zeros_a6:
  (∃ a : ℕ+ → ℚ, 
    a 1 = 3 / 2 ∧ 
    (∀ n : ℕ+, a (n + 1) = (1 / 2) * (a n + (1 / a n))) ∧
    (∃ k, 10^k ≤ a 6 ∧ a 6 < 10^(k + 1))) →
  (∃ m, m = 22) :=
sorry

end trailing_zeros_a6_l2237_223776


namespace cube_painting_l2237_223734

theorem cube_painting (n : ℕ) (h₁ : n > 4) 
  (h₂ : (2 * (n - 2)) = (n^2 - 2*n + 1)) : n = 5 :=
sorry

end cube_painting_l2237_223734


namespace fred_balloons_l2237_223744

variable (initial_balloons : ℕ := 709)
variable (balloons_given : ℕ := 221)
variable (remaining_balloons : ℕ := 488)

theorem fred_balloons :
  initial_balloons - balloons_given = remaining_balloons :=
  by
    sorry

end fred_balloons_l2237_223744


namespace tourist_total_value_l2237_223761

theorem tourist_total_value
    (tax_rate : ℝ)
    (V : ℝ)
    (tax_paid : ℝ)
    (exempt_amount : ℝ) :
    exempt_amount = 600 ∧
    tax_rate = 0.07 ∧
    tax_paid = 78.4 →
    (tax_rate * (V - exempt_amount) = tax_paid) →
    V = 1720 :=
by
  intros h1 h2
  have h_exempt : exempt_amount = 600 := h1.left
  have h_tax_rate : tax_rate = 0.07 := h1.right.left
  have h_tax_paid : tax_paid = 78.4 := h1.right.right
  sorry

end tourist_total_value_l2237_223761


namespace find_x_l2237_223797

-- Define the conditions according to the problem statement
variables {C x : ℝ} -- C is the cost per liter of pure spirit, x is the volume of water in the first solution

-- Condition 1: The cost for the first solution
def cost_first_solution (C : ℝ) (x : ℝ) : Prop := 0.50 = C * (1 / (1 + x))

-- Condition 2: The cost for the second solution (approximating 0.4999999999999999 as 0.50)
def cost_second_solution (C : ℝ) : Prop := 0.50 = C * (1 / 3)

-- The theorem to prove: x = 2 given the two conditions
theorem find_x (C : ℝ) (x : ℝ) (h1 : cost_first_solution C x) (h2 : cost_second_solution C) : x = 2 := 
sorry

end find_x_l2237_223797


namespace nat_le_two_pow_million_l2237_223772

theorem nat_le_two_pow_million (n : ℕ) (h : n ≤ 2^1000000) : 
  ∃ (x : ℕ → ℕ) (k : ℕ), k ≤ 1100000 ∧ x 0 = 1 ∧ x k = n ∧ 
  ∀ (i : ℕ), 1 ≤ i → i ≤ k → ∃ (r s : ℕ), 0 ≤ r ∧ r ≤ s ∧ s < i ∧ x i = x r + x s :=
sorry

end nat_le_two_pow_million_l2237_223772


namespace animal_costs_l2237_223754

theorem animal_costs :
  ∃ (C G S P : ℕ),
      C + G + S + P = 1325 ∧
      G + S + P = 425 ∧
      C + S + P = 1225 ∧
      G + P = 275 ∧
      C = 900 ∧
      G = 100 ∧
      S = 150 ∧
      P = 175 :=
by
  sorry

end animal_costs_l2237_223754


namespace range_of_m_l2237_223799

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l2237_223799


namespace complement_intersection_l2237_223704

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {2, 4}

theorem complement_intersection :
  ((U \ A) ∩ B) = {2} :=
sorry

end complement_intersection_l2237_223704


namespace divisible_by_12_l2237_223750

theorem divisible_by_12 (a b c d : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (hpos_d : 0 < d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := 
by
  sorry

end divisible_by_12_l2237_223750
