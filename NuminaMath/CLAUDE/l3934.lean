import Mathlib

namespace alloy_density_proof_l3934_393499

/-- The specific gravity of gold relative to water -/
def gold_specific_gravity : ℝ := 19

/-- The specific gravity of copper relative to water -/
def copper_specific_gravity : ℝ := 9

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 4

/-- The specific gravity of the resulting alloy -/
def alloy_specific_gravity : ℝ := 17

/-- Theorem stating that mixing gold and copper in the given ratio results in the specified alloy density -/
theorem alloy_density_proof :
  (gold_copper_ratio * gold_specific_gravity + copper_specific_gravity) / (gold_copper_ratio + 1) = alloy_specific_gravity :=
by sorry

end alloy_density_proof_l3934_393499


namespace sum_of_digits_of_number_l3934_393428

/-- The sum of the digits of 10^100 - 57 -/
def sum_of_digits : ℕ := 889

/-- The number we're considering -/
def number : ℕ := 10^100 - 57

/-- Theorem stating that the sum of the digits of our number is equal to sum_of_digits -/
theorem sum_of_digits_of_number : 
  (number.digits 10).sum = sum_of_digits := by sorry

end sum_of_digits_of_number_l3934_393428


namespace power_difference_evaluation_l3934_393421

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16245775 := by
  sorry

end power_difference_evaluation_l3934_393421


namespace all_terms_even_l3934_393484

theorem all_terms_even (m n : ℤ) (hm : Even m) (hn : Even n) :
  ∀ k : Fin 9, Even ((Finset.range 9).sum (λ i => (Nat.choose 8 i : ℤ) * m^(8 - i) * n^i)) := by
  sorry

end all_terms_even_l3934_393484


namespace brads_cookies_brads_cookies_solution_l3934_393454

theorem brads_cookies (total_cookies : ℕ) (greg_ate : ℕ) (leftover : ℕ) : ℕ :=
  let total_halves := total_cookies * 2
  let after_greg := total_halves - greg_ate
  after_greg - leftover

theorem brads_cookies_solution :
  brads_cookies 14 4 18 = 6 := by
  sorry

end brads_cookies_brads_cookies_solution_l3934_393454


namespace pablo_candy_cost_l3934_393411

/-- The cost of candy given Pablo's reading and spending habits -/
def candy_cost (pages_per_book : ℕ) (books_read : ℕ) (earnings_per_page : ℚ) (money_left : ℚ) : ℚ :=
  (pages_per_book * books_read : ℕ) * earnings_per_page - money_left

/-- Theorem stating the cost of candy given Pablo's specific situation -/
theorem pablo_candy_cost :
  candy_cost 150 12 (1 / 100) 3 = 15 := by
  sorry

end pablo_candy_cost_l3934_393411


namespace max_value_of_f_on_interval_l3934_393478

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 4 ∧
  f x = 16 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-4 : ℝ) 4 → f y ≤ f x :=
by sorry

end max_value_of_f_on_interval_l3934_393478


namespace tv_price_change_l3934_393467

theorem tv_price_change (P : ℝ) (x : ℝ) : 
  (P - (x / 100) * P) * (1 + 30 / 100) = P * (1 + 4 / 100) → x = 20 := by
  sorry

end tv_price_change_l3934_393467


namespace expansion_coefficient_sum_l3934_393465

theorem expansion_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (m * x - 1)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 33 →
  m = 3 := by
sorry

end expansion_coefficient_sum_l3934_393465


namespace complex_number_location_l3934_393470

theorem complex_number_location (z : ℂ) (h : z / (4 + 2*I) = I) :
  (z.re < 0) ∧ (z.im > 0) := by sorry

end complex_number_location_l3934_393470


namespace expression_evaluation_l3934_393479

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + 2*x + 2) / x * (y^2 + 2*y + 2) / y + (x^2 - 3*x + 2) / y * (y^2 - 3*y + 2) / x =
  2*x*y - x/y - y/x + 13 + 10/x + 4/y + 8/(x*y) := by sorry

end expression_evaluation_l3934_393479


namespace shovel_time_closest_to_17_l3934_393435

/-- Represents the snow shoveling problem --/
structure SnowShoveling where
  /-- Initial shoveling rate in cubic yards per hour --/
  initial_rate : ℕ
  /-- Decrease in shoveling rate per hour --/
  rate_decrease : ℕ
  /-- Break duration in hours --/
  break_duration : ℚ
  /-- Hours of shoveling before a break --/
  hours_before_break : ℕ
  /-- Driveway width in yards --/
  driveway_width : ℕ
  /-- Driveway length in yards --/
  driveway_length : ℕ
  /-- Snow depth in yards --/
  snow_depth : ℕ

/-- Calculates the time taken to shovel the driveway clean, including breaks --/
def time_to_shovel (problem : SnowShoveling) : ℚ :=
  sorry

/-- Theorem stating that the time taken to shovel the driveway is closest to 17 hours --/
theorem shovel_time_closest_to_17 (problem : SnowShoveling) 
  (h1 : problem.initial_rate = 25)
  (h2 : problem.rate_decrease = 1)
  (h3 : problem.break_duration = 1/2)
  (h4 : problem.hours_before_break = 2)
  (h5 : problem.driveway_width = 5)
  (h6 : problem.driveway_length = 12)
  (h7 : problem.snow_depth = 4) :
  ∃ (t : ℚ), time_to_shovel problem = t ∧ abs (t - 17) < abs (t - 14) ∧ 
             abs (t - 17) < abs (t - 15) ∧ abs (t - 17) < abs (t - 16) ∧ 
             abs (t - 17) < abs (t - 18) :=
  sorry

end shovel_time_closest_to_17_l3934_393435


namespace complex_power_simplification_l3934_393418

theorem complex_power_simplification :
  ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 2000 = 1 := by
  sorry

end complex_power_simplification_l3934_393418


namespace remainder_theorem_l3934_393483

theorem remainder_theorem (k : ℤ) : (1125 * 1127 * (12 * k + 1)) % 12 = 3 := by
  sorry

end remainder_theorem_l3934_393483


namespace multiplication_value_proof_l3934_393462

theorem multiplication_value_proof (x : ℝ) : (7.5 / 6) * x = 15 → x = 12 := by
  sorry

end multiplication_value_proof_l3934_393462


namespace quadratic_real_root_condition_l3934_393424

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end quadratic_real_root_condition_l3934_393424


namespace boys_in_school_l3934_393482

/-- The number of boys in a school, given the initial number of girls, 
    the number of new girls who joined, and the total number of pupils after new girls joined. -/
def number_of_boys (initial_girls new_girls total_pupils : ℕ) : ℕ :=
  total_pupils - (initial_girls + new_girls)

/-- Theorem stating that the number of boys in the school is 222 -/
theorem boys_in_school : number_of_boys 706 418 1346 = 222 := by
  sorry

end boys_in_school_l3934_393482


namespace line_equation_l3934_393422

/-- A line with slope -2 and sum of x and y intercepts equal to 12 has the general equation 2x + y - 8 = 0 -/
theorem line_equation (l : Set (ℝ × ℝ)) (slope : ℝ) (intercept_sum : ℝ) : 
  slope = -2 →
  intercept_sum = 12 →
  ∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -8 ∧
  l = {(x, y) | a * x + b * y + c = 0} :=
by sorry

end line_equation_l3934_393422


namespace product_from_lcm_and_gcd_l3934_393416

theorem product_from_lcm_and_gcd (a b : ℕ+) : 
  Nat.lcm a b = 72 → Nat.gcd a b = 6 → a * b = 432 := by
  sorry

end product_from_lcm_and_gcd_l3934_393416


namespace remainder_17_45_mod_5_l3934_393480

theorem remainder_17_45_mod_5 : 17^45 % 5 = 2 := by sorry

end remainder_17_45_mod_5_l3934_393480


namespace triangle_ABC_properties_l3934_393417

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a - b) / c = (Real.sin B + Real.sin C) / (Real.sin B + Real.sin A) ∧
  a = Real.sqrt 7 ∧
  b = 2 * c

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  A = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end triangle_ABC_properties_l3934_393417


namespace games_from_friend_l3934_393432

theorem games_from_friend (games_from_garage_sale : ℕ) 
  (non_working_games : ℕ) (good_games : ℕ) : ℕ :=
  by
  have h1 : games_from_garage_sale = 8 := by sorry
  have h2 : non_working_games = 23 := by sorry
  have h3 : good_games = 6 := by sorry
  
  let total_games := non_working_games + good_games
  
  have h4 : total_games = 29 := by sorry
  
  let games_from_friend := total_games - games_from_garage_sale
  
  have h5 : games_from_friend = 21 := by sorry
  
  exact games_from_friend

end games_from_friend_l3934_393432


namespace sum_floor_equals_217_l3934_393438

theorem sum_floor_equals_217 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_squares : x^2 + y^2 = 4050 ∧ z^2 + w^2 = 4050)
  (products : x*z = 2040 ∧ y*w = 2040) : 
  ⌊x + y + z + w⌋ = 217 := by
sorry

end sum_floor_equals_217_l3934_393438


namespace symmetry_f_and_f_inv_symmetry_f_and_f_swap_same_curve_f_and_f_inv_l3934_393426

-- Define a function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Statement 1
theorem symmetry_f_and_f_inv :
  ∀ x y, y = f x ↔ x = f_inv y :=
sorry

-- Statement 2
theorem symmetry_f_and_f_swap :
  ∀ x y, y = f x ↔ x = f y :=
sorry

-- Statement 4
theorem same_curve_f_and_f_inv :
  ∀ x y, y = f x ↔ x = f_inv y :=
sorry

end symmetry_f_and_f_inv_symmetry_f_and_f_swap_same_curve_f_and_f_inv_l3934_393426


namespace complex_quadrant_range_l3934_393496

theorem complex_quadrant_range (z : ℂ) (a : ℝ) :
  z * (a + Complex.I) = 2 + 3 * Complex.I →
  (z.re * z.im < 0 ↔ -3/2 < a ∧ a < 2/3) :=
by sorry

end complex_quadrant_range_l3934_393496


namespace K_bounds_l3934_393497

/-- The number of triples in a given system for a natural number n -/
noncomputable def K (n : ℕ) : ℝ := sorry

/-- Theorem stating the bounds for K(n) -/
theorem K_bounds (n : ℕ) : n / 6 - 1 < K n ∧ K n < 2 * n / 9 := by sorry

end K_bounds_l3934_393497


namespace compound_molecular_weight_l3934_393431

/-- Atomic weight of Barium in g/mol -/
def Ba_weight : ℝ := 137.33

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Number of Barium atoms in the compound -/
def Ba_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 2

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ := Ba_count * Ba_weight + O_count * O_weight + H_count * H_weight

/-- Theorem stating that the molecular weight of the compound is 171.35 g/mol -/
theorem compound_molecular_weight : molecular_weight = 171.35 := by
  sorry

end compound_molecular_weight_l3934_393431


namespace difference_of_squares_l3934_393400

theorem difference_of_squares (a b : ℝ) (h1 : a + b = 75) (h2 : a - b = 15) :
  a^2 - b^2 = 1125 := by
  sorry

end difference_of_squares_l3934_393400


namespace salt_merchant_problem_l3934_393460

/-- The salt merchant problem -/
theorem salt_merchant_problem (x y : ℝ) (a : ℝ) 
  (h1 : a * (y - x) = 100)  -- Profit from first transaction
  (h2 : a * y * (y / x - 1) = 120)  -- Profit from second transaction
  (h3 : x > 0)  -- Price in Tver is positive
  (h4 : y > x)  -- Price in Moscow is higher than in Tver
  : a * x = 500 := by
  sorry

end salt_merchant_problem_l3934_393460


namespace group_a_trees_l3934_393486

theorem group_a_trees (group_a_plots : ℕ) (group_b_plots : ℕ) : 
  (4 * group_a_plots = 5 * group_b_plots) →  -- Both groups planted the same total number of trees
  (group_b_plots = group_a_plots - 3) →      -- Group B worked on 3 fewer plots than Group A
  (4 * group_a_plots = 60) :=                -- Group A planted 60 trees in total
by
  sorry

#check group_a_trees

end group_a_trees_l3934_393486


namespace complex_point_on_line_l3934_393429

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (a - Complex.I)⁻¹
  (z.im = 2 * z.re) → a = 1/2 := by
  sorry

end complex_point_on_line_l3934_393429


namespace light_reflection_l3934_393414

/-- Given a light ray emitted from point P (6, 4) intersecting the x-axis at point Q (2, 0)
    and reflecting off the x-axis, prove that the equations of the lines on which the
    incident and reflected rays lie are x - y - 2 = 0 and x + y - 2 = 0, respectively. -/
theorem light_reflection (P Q : ℝ × ℝ) : 
  P = (6, 4) → Q = (2, 0) → 
  ∃ (incident_ray reflected_ray : ℝ → ℝ → Prop),
    (∀ x y, incident_ray x y ↔ x - y - 2 = 0) ∧
    (∀ x y, reflected_ray x y ↔ x + y - 2 = 0) :=
by sorry

end light_reflection_l3934_393414


namespace stamps_theorem_l3934_393439

/-- Given denominations 3, n, and n+1, this function checks if k cents can be formed -/
def can_form (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), k = 3 * a + n * b + (n + 1) * c

/-- The main theorem -/
theorem stamps_theorem :
  ∃! (n : ℕ), 
    n > 0 ∧ 
    (∀ (k : ℕ), k ≤ 115 → ¬(can_form n k)) ∧
    (∀ (k : ℕ), k > 115 → can_form n k) ∧
    n = 59 := by
  sorry

end stamps_theorem_l3934_393439


namespace worker_wage_increase_l3934_393444

theorem worker_wage_increase (original_wage : ℝ) : 
  (original_wage * 1.5 = 42) → original_wage = 28 := by
  sorry

end worker_wage_increase_l3934_393444


namespace whitney_whale_books_l3934_393436

/-- The number of whale books Whitney bought -/
def whale_books : ℕ := sorry

/-- The number of fish books Whitney bought -/
def fish_books : ℕ := 7

/-- The number of magazines Whitney bought -/
def magazines : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 11

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 1

/-- The total amount Whitney spent in dollars -/
def total_spent : ℕ := 179

/-- Theorem stating that Whitney bought 9 whale books -/
theorem whitney_whale_books : 
  whale_books * book_cost + fish_books * book_cost + magazines * magazine_cost = total_spent ∧
  whale_books = 9 := by sorry

end whitney_whale_books_l3934_393436


namespace valid_numbers_l3934_393441

def is_valid_number (n : ℕ) : Prop :=
  30 ∣ n ∧ (Finset.card (Nat.divisors n) = 30)

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {11250, 4050, 7500, 1620, 1200, 720} := by
  sorry

end valid_numbers_l3934_393441


namespace peanut_seed_sprouting_probability_l3934_393407

/-- The probability of exactly k successes in n independent trials,
    where p is the probability of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem peanut_seed_sprouting_probability :
  let n : ℕ := 3  -- total number of seeds
  let k : ℕ := 2  -- number of seeds we want to sprout
  let p : ℝ := 3/5  -- probability of each seed sprouting
  binomial_probability n k p = 54/125 := by
sorry

end peanut_seed_sprouting_probability_l3934_393407


namespace integer_quadruple_solution_l3934_393466

theorem integer_quadruple_solution :
  ∃! (S : Set (ℕ × ℕ × ℕ × ℕ)),
    S.Nonempty ∧
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔
      (1 < a ∧ a < b ∧ b < c ∧ c < d) ∧
      (∃ k : ℕ, a * b * c * d - 1 = k * ((a - 1) * (b - 1) * (c - 1) * (d - 1)))) ∧
    S = {(3, 5, 17, 255), (2, 4, 10, 80)} :=
by sorry

end integer_quadruple_solution_l3934_393466


namespace smallest_three_digit_mod_congruence_l3934_393464

theorem smallest_three_digit_mod_congruence :
  ∃ n : ℕ, 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    45 * n % 315 = 90 ∧ 
    ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 45 * m % 315 = 90 → m ≥ n :=
by sorry

end smallest_three_digit_mod_congruence_l3934_393464


namespace rationalize_denominator_l3934_393427

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end rationalize_denominator_l3934_393427


namespace distance_between_stations_l3934_393459

/-- The distance between two stations given three cars with different speeds --/
theorem distance_between_stations (speed_A speed_B speed_C : ℝ) (time_diff : ℝ) : 
  speed_A = 90 →
  speed_B = 80 →
  speed_C = 60 →
  time_diff = 1/3 →
  (speed_A + speed_B) * ((speed_A + speed_C) * time_diff / (speed_B - speed_C)) = 425 := by
  sorry

end distance_between_stations_l3934_393459


namespace power_two_divides_odd_power_minus_one_l3934_393477

theorem power_two_divides_odd_power_minus_one (k : ℕ) (h : Odd k) :
  ∀ n : ℕ, n ≥ 1 → (2^(n+2) : ℕ) ∣ k^(2^n) - 1 :=
by sorry

end power_two_divides_odd_power_minus_one_l3934_393477


namespace grocery_store_lite_soda_l3934_393455

/-- Given a grocery store with soda bottles, proves that the number of lite soda bottles is 60 -/
theorem grocery_store_lite_soda (regular : ℕ) (diet : ℕ) (lite : ℕ) 
  (h1 : regular = 81)
  (h2 : diet = 60)
  (h3 : diet = lite) : 
  lite = 60 := by
  sorry

end grocery_store_lite_soda_l3934_393455


namespace cross_placements_count_l3934_393481

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)

/-- Represents a rectangle --/
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a cross shape --/
structure Cross :=
  (size : ℕ)

/-- Function to calculate the number of ways to place a cross in a grid with a rectangle removed --/
def count_cross_placements (g : Grid) (r : Rectangle) (c : Cross) : ℕ :=
  sorry

/-- Theorem stating the number of ways to place a 5-cell cross in a 40x40 grid with a 36x37 rectangle removed --/
theorem cross_placements_count :
  let g := Grid.mk 40
  let r := Rectangle.mk 36 37
  let c := Cross.mk 5
  count_cross_placements g r c = 113 := by
  sorry

end cross_placements_count_l3934_393481


namespace absolute_value_equation_solution_product_l3934_393456

theorem absolute_value_equation_solution_product : ∃ (x₁ x₂ : ℝ),
  (|2 * x₁ - 1| + 4 = 24) ∧
  (|2 * x₂ - 1| + 4 = 24) ∧
  (x₁ ≠ x₂) ∧
  (x₁ * x₂ = -99.75) := by
  sorry

end absolute_value_equation_solution_product_l3934_393456


namespace smallest_n_for_candy_removal_l3934_393405

theorem smallest_n_for_candy_removal : ∃ n : ℕ, 
  (∀ k : ℕ, k > 0 → k * (k + 1) / 2 ≥ 64 → n ≤ k) ∧ 
  n * (n + 1) / 2 ≥ 64 :=
by sorry

end smallest_n_for_candy_removal_l3934_393405


namespace birds_joining_fence_l3934_393453

theorem birds_joining_fence (initial_storks initial_birds joining_birds : ℕ) : 
  initial_storks = 6 →
  initial_birds = 2 →
  initial_storks = initial_birds + joining_birds + 1 →
  joining_birds = 3 := by
sorry

end birds_joining_fence_l3934_393453


namespace arithmetic_sequence_20th_term_l3934_393412

/-- An arithmetic sequence {a_n} satisfying given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 18)
  (h_sum2 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 := by
    sorry

end arithmetic_sequence_20th_term_l3934_393412


namespace angle_measure_proof_l3934_393445

theorem angle_measure_proof :
  ∀ (A B : ℝ),
  A + B = 180 →
  A = 5 * B →
  A = 150 :=
by
  sorry

end angle_measure_proof_l3934_393445


namespace area_to_paint_l3934_393442

-- Define the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_width : ℝ := 5
def door_height : ℝ := 2
def door_width : ℝ := 7

-- Define the theorem
theorem area_to_paint :
  wall_height * wall_length - (window_height * window_width + door_height * door_width) = 121 := by
  sorry

end area_to_paint_l3934_393442


namespace number_equal_to_square_plus_opposite_l3934_393452

theorem number_equal_to_square_plus_opposite :
  ∀ x : ℝ, x = x^2 + (-x) → x = 0 ∨ x = 2 := by
sorry

end number_equal_to_square_plus_opposite_l3934_393452


namespace min_phase_shift_l3934_393415

theorem min_phase_shift (x φ : ℝ) : 
  (∀ x, 2 * Real.sin (x + π/6 - φ) = 2 * Real.sin (x - π/3)) →
  (φ > 0 → φ ≥ π/2) ∧ 
  (∃ φ₀ > 0, ∀ x, 2 * Real.sin (x + π/6 - φ₀) = 2 * Real.sin (x - π/3) ∧ φ₀ = π/2) := by
  sorry

#check min_phase_shift

end min_phase_shift_l3934_393415


namespace main_theorem_l3934_393402

/-- A nondecreasing function satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧
  (∀ x y : ℝ, f (f x) + f y = f (x + f y) + 1)

/-- The set of all solutions to the functional equation. -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | FunctionalEquation f ∧
    (∀ x, f x = 1) ∨
    (∀ x, f x = x + 1) ∨
    (∃ n : ℕ+, ∃ α : ℝ, 0 ≤ α ∧ α < 1 ∧ 
      (∀ x, f x = (1 / n) * ⌊n * x + α⌋ + 1)) ∨
    (∃ n : ℕ+, ∃ α : ℝ, 0 ≤ α ∧ α < 1 ∧ 
      (∀ x, f x = (1 / n) * ⌈n * x - α⌉ + 1))}

/-- The main theorem stating that the SolutionSet contains all solutions to the functional equation. -/
theorem main_theorem : ∀ f : ℝ → ℝ, FunctionalEquation f → f ∈ SolutionSet := by
  sorry

end main_theorem_l3934_393402


namespace replaced_tomatoes_cost_is_2_20_l3934_393488

/-- Represents the grocery order with item prices and total costs -/
structure GroceryOrder where
  original_total : ℝ
  original_tomatoes : ℝ
  original_lettuce : ℝ
  original_celery : ℝ
  new_lettuce : ℝ
  new_celery : ℝ
  delivery_tip : ℝ
  new_total : ℝ

/-- Calculates the cost of the replaced can of tomatoes -/
def replaced_tomatoes_cost (order : GroceryOrder) : ℝ :=
  order.new_total - order.original_total - order.delivery_tip -
  (order.new_lettuce - order.original_lettuce) -
  (order.new_celery - order.original_celery) +
  order.original_tomatoes

/-- Theorem stating that the cost of the replaced can of tomatoes is $2.20 -/
theorem replaced_tomatoes_cost_is_2_20 (order : GroceryOrder)
  (h1 : order.original_total = 25)
  (h2 : order.original_tomatoes = 0.99)
  (h3 : order.original_lettuce = 1)
  (h4 : order.original_celery = 1.96)
  (h5 : order.new_lettuce = 1.75)
  (h6 : order.new_celery = 2)
  (h7 : order.delivery_tip = 8)
  (h8 : order.new_total = 35) :
  replaced_tomatoes_cost order = 2.20 := by
  sorry


end replaced_tomatoes_cost_is_2_20_l3934_393488


namespace graveyard_skeletons_l3934_393434

/-- Represents the number of skeletons in the graveyard -/
def S : ℕ := sorry

/-- The number of bones in an adult woman's skeleton -/
def womanBones : ℕ := 20

/-- The number of bones in an adult man's skeleton -/
def manBones : ℕ := womanBones + 5

/-- The number of bones in a child's skeleton -/
def childBones : ℕ := womanBones / 2

/-- The total number of bones in the graveyard -/
def totalBones : ℕ := 375

theorem graveyard_skeletons :
  (S / 2 * womanBones + S / 4 * manBones + S / 4 * childBones = totalBones) →
  S = 20 := by sorry

end graveyard_skeletons_l3934_393434


namespace color_theorem_l3934_393430

theorem color_theorem :
  ∃ (f : ℕ → ℕ),
    (∀ x, x ∈ Finset.range 2013 → f x ∈ Finset.range 7) ∧
    (∀ y, y ∈ Finset.range 7 → ∃ x ∈ Finset.range 2013, f x = y) ∧
    (∀ a b c, a ∈ Finset.range 2013 → b ∈ Finset.range 2013 → c ∈ Finset.range 2013 →
      a ≠ b → b ≠ c → a ≠ c → f a = f b → f b = f c →
        ¬(2014 ∣ (a * b * c)) ∧
        f ((a * b * c) % 2014) = f a) :=
by sorry

end color_theorem_l3934_393430


namespace f_properties_l3934_393450

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ ((x < 0 ∧ y ≤ 0) ∨ (x ≥ 2 ∧ y > 2)) → f x < f y) ∧
  (∃ δ₁ > 0, ∀ x, 0 < |x| ∧ |x| < δ₁ → f x < f 0) ∧
  (∃ δ₂ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ₂ → f x > f 2) :=
by sorry


end f_properties_l3934_393450


namespace deaf_to_blind_ratio_l3934_393451

theorem deaf_to_blind_ratio (total : ℕ) (deaf : ℕ) (h1 : total = 240) (h2 : deaf = 180) :
  (deaf : ℚ) / (total - deaf) = 3 / 1 := by
  sorry

end deaf_to_blind_ratio_l3934_393451


namespace chris_age_is_17_l3934_393404

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℕ
  ben : ℕ
  chris : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Six years ago, Chris was the same age as Amy is now
  ages.chris - 6 = ages.amy ∧
  -- In 3 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 3 = (3 * (ages.amy + 3)) / 4

/-- The theorem stating that Chris's age is 17 -/
theorem chris_age_is_17 :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.chris = 17 := by
  sorry


end chris_age_is_17_l3934_393404


namespace farm_work_hourly_rate_l3934_393490

theorem farm_work_hourly_rate 
  (total_amount : ℕ) 
  (tips : ℕ) 
  (hours_worked : ℕ) 
  (h1 : total_amount = 240)
  (h2 : tips = 50)
  (h3 : hours_worked = 19) :
  (total_amount - tips) / hours_worked = 10 := by
sorry

end farm_work_hourly_rate_l3934_393490


namespace mouse_breeding_problem_l3934_393475

theorem mouse_breeding_problem (initial_mice : ℕ) (first_round_pups : ℕ) (eaten_pups : ℕ) (final_mice : ℕ) :
  initial_mice = 8 →
  first_round_pups = 6 →
  eaten_pups = 2 →
  final_mice = 280 →
  ∃ (second_round_pups : ℕ),
    final_mice = initial_mice + initial_mice * first_round_pups +
      (initial_mice + initial_mice * first_round_pups) * second_round_pups -
      (initial_mice + initial_mice * first_round_pups) * eaten_pups ∧
    second_round_pups = 6 :=
by sorry

end mouse_breeding_problem_l3934_393475


namespace five_in_range_of_quadratic_l3934_393403

theorem five_in_range_of_quadratic (b : ℝ) : ∃ x : ℝ, x^2 + b*x + 3 = 5 := by
  sorry

end five_in_range_of_quadratic_l3934_393403


namespace first_day_over_200_paperclips_l3934_393401

def paperclips (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_over_200_paperclips :
  (∀ j : ℕ, j < 8 → paperclips j ≤ 200) ∧ paperclips 8 > 200 :=
by sorry

end first_day_over_200_paperclips_l3934_393401


namespace digit_equation_solution_l3934_393448

theorem digit_equation_solution : ∃! (X : ℕ), X < 10 ∧ (510 : ℚ) / X = 40 + 3 * X :=
by sorry

end digit_equation_solution_l3934_393448


namespace pipe_A_fill_time_l3934_393476

/-- The time (in hours) taken by pipe B to empty the full cistern -/
def time_B : ℝ := 25

/-- The time (in hours) taken to fill the cistern when both pipes are opened -/
def time_both : ℝ := 99.99999999999999

/-- The time (in hours) taken by pipe A to fill the cistern -/
def time_A : ℝ := 20

/-- Theorem stating that the time taken by pipe A to fill the cistern is 20 hours -/
theorem pipe_A_fill_time :
  (1 / time_A - 1 / time_B) * time_both = 1 :=
sorry

end pipe_A_fill_time_l3934_393476


namespace central_square_side_length_l3934_393458

/-- Given a rectangular hallway and total flooring area, calculates the side length of a central square area --/
theorem central_square_side_length 
  (hallway_length : ℝ) 
  (hallway_width : ℝ) 
  (total_area : ℝ) 
  (h1 : hallway_length = 6)
  (h2 : hallway_width = 4)
  (h3 : total_area = 124) :
  let hallway_area := hallway_length * hallway_width
  let central_area := total_area - hallway_area
  let side_length := Real.sqrt central_area
  side_length = 10 := by sorry

end central_square_side_length_l3934_393458


namespace routes_on_3x2_grid_l3934_393413

/-- The number of routes on a grid from top-left to bottom-right -/
def num_routes (width : ℕ) (height : ℕ) : ℕ :=
  Nat.choose (width + height) height

/-- The theorem stating that the number of routes on a 3x2 grid is 10 -/
theorem routes_on_3x2_grid : num_routes 3 2 = 10 := by sorry

end routes_on_3x2_grid_l3934_393413


namespace number_relationships_l3934_393489

theorem number_relationships : 
  (100000000 = 10 * 10000000) ∧ (1000000 = 100 * 10000) := by
  sorry

#check number_relationships

end number_relationships_l3934_393489


namespace unique_solution_for_power_sum_l3934_393469

theorem unique_solution_for_power_sum : 
  ∃! (x y z : ℕ), x < y ∧ y < z ∧ 3^x + 3^y + 3^z = 179415 ∧ x = 4 ∧ y = 7 ∧ z = 11 := by
  sorry

end unique_solution_for_power_sum_l3934_393469


namespace smallest_num_rectangles_l3934_393410

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Whether two natural numbers are in the ratio 5:4 -/
def in_ratio_5_4 (a b : ℕ) : Prop := 5 * b = 4 * a

/-- The number of small rectangles needed to cover a larger rectangle -/
def num_small_rectangles (small large : Rectangle) : ℕ :=
  large.area / small.area

theorem smallest_num_rectangles :
  let small_rectangle : Rectangle := ⟨2, 3⟩
  ∃ (large_rectangle : Rectangle),
    in_ratio_5_4 large_rectangle.width large_rectangle.height ∧
    num_small_rectangles small_rectangle large_rectangle = 30 ∧
    ∀ (other_rectangle : Rectangle),
      in_ratio_5_4 other_rectangle.width other_rectangle.height →
      num_small_rectangles small_rectangle other_rectangle ≥ 30 :=
by sorry

end smallest_num_rectangles_l3934_393410


namespace no_integer_solution_l3934_393493

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end no_integer_solution_l3934_393493


namespace circle_theorem_part1_circle_theorem_part2_l3934_393437

-- Define the points A and B
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Define the circle equation for part 1
def circle_eq1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10

-- Define the circle equation for part 2
def circle_eq2 (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 5

-- Part 1: Circle passing through A and B with center on the line
theorem circle_theorem_part1 :
  ∃ (center : ℝ × ℝ), 
    (line_eq center.1 center.2) ∧
    (∀ (x y : ℝ), circle_eq1 x y ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (A.1 - center.1)^2 + (A.2 - center.2)^2) ∧
      ((x - center.1)^2 + (y - center.2)^2 = (B.1 - center.1)^2 + (B.2 - center.2)^2)) :=
sorry

-- Part 2: Circle passing through A and B with minimum area
theorem circle_theorem_part2 :
  ∃ (center : ℝ × ℝ),
    (∀ (other_center : ℝ × ℝ),
      (A.1 - center.1)^2 + (A.2 - center.2)^2 ≤ (A.1 - other_center.1)^2 + (A.2 - other_center.2)^2) ∧
    (∀ (x y : ℝ), circle_eq2 x y ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (A.1 - center.1)^2 + (A.2 - center.2)^2) ∧
      ((x - center.1)^2 + (y - center.2)^2 = (B.1 - center.1)^2 + (B.2 - center.2)^2)) :=
sorry

end circle_theorem_part1_circle_theorem_part2_l3934_393437


namespace complex_sum_real_imag_parts_l3934_393420

theorem complex_sum_real_imag_parts (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : 
  z.re + z.im = 0 := by sorry

end complex_sum_real_imag_parts_l3934_393420


namespace cube_volume_l3934_393408

/-- Given a box with dimensions 8 cm x 15 cm x 5 cm that can be built using a minimum of 60 cubes,
    the volume of each cube is 10 cm³. -/
theorem cube_volume (length width height min_cubes : ℕ) : 
  length = 8 → width = 15 → height = 5 → min_cubes = 60 →
  (length * width * height : ℚ) / min_cubes = 10 := by
  sorry

end cube_volume_l3934_393408


namespace B_is_smallest_l3934_393498

def A : ℤ := 32 + 7
def B : ℤ := 3 * 10 + 3
def C : ℤ := 50 - 9

theorem B_is_smallest : B ≤ A ∧ B ≤ C := by
  sorry

end B_is_smallest_l3934_393498


namespace triangle_properties_l3934_393461

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.C = 5 * Real.pi / 6) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) 
  (h4 : t.B = Real.pi / 3) : 
  (t.c = Real.sqrt 13) ∧ 
  (-Real.sqrt 3 < 2 * t.c - t.a) ∧ 
  (2 * t.c - t.a < 2 * Real.sqrt 3) := by
sorry


end triangle_properties_l3934_393461


namespace right_triangle_circle_theorem_l3934_393494

/-- A right triangle with a circle inscribed on one side --/
structure RightTriangleWithCircle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Point where the circle meets AC
  D : ℝ × ℝ
  -- B is a right angle
  right_angle_at_B : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  -- BC is the diameter of the circle
  BC_is_diameter : ∃ (center : ℝ × ℝ), 
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = (center.1 - C.1)^2 + (center.2 - C.2)^2
  -- D lies on AC
  D_on_AC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t*(C.1 - A.1), A.2 + t*(C.2 - A.2))
  -- D lies on the circle
  D_on_circle : ∃ (center : ℝ × ℝ), 
    (center.1 - D.1)^2 + (center.2 - D.2)^2 = (center.1 - B.1)^2 + (center.2 - B.2)^2

/-- The theorem to be proved --/
theorem right_triangle_circle_theorem (t : RightTriangleWithCircle) 
  (h1 : Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) = 3)
  (h2 : Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 6) :
  Real.sqrt ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2) = 12 := by
sorry

end right_triangle_circle_theorem_l3934_393494


namespace factor_w4_minus_81_l3934_393433

theorem factor_w4_minus_81 (w : ℂ) : w^4 - 81 = (w-3)*(w+3)*(w-3*I)*(w+3*I) := by
  sorry

end factor_w4_minus_81_l3934_393433


namespace election_votes_difference_l3934_393495

theorem election_votes_difference (total_votes : ℕ) (winner_votes : ℕ) (second_votes : ℕ) (third_votes : ℕ) (fourth_votes : ℕ) 
  (h_total : total_votes = 979)
  (h_candidates : winner_votes + second_votes + third_votes + fourth_votes = total_votes)
  (h_winner_second : winner_votes = second_votes + 53)
  (h_winner_fourth : winner_votes = fourth_votes + 105)
  (h_fourth : fourth_votes = 199) :
  winner_votes - third_votes = 79 := by
sorry

end election_votes_difference_l3934_393495


namespace car_distance_formula_l3934_393474

/-- The distance traveled by a car after time t -/
def distance (t : ℝ) : ℝ :=
  10 + 60 * t

/-- The initial distance traveled by the car -/
def initial_distance : ℝ := 10

/-- The constant speed of the car after the initial distance -/
def speed : ℝ := 60

theorem car_distance_formula (t : ℝ) :
  distance t = initial_distance + speed * t :=
by sorry

end car_distance_formula_l3934_393474


namespace wuzhen_conference_impact_l3934_393419

/-- Represents the cultural impact of the World Internet Conference in Wuzhen -/
structure CulturalImpact where
  promote_chinese_culture : Bool
  innovate_world_culture : Bool
  enhance_chinese_influence : Bool

/-- The World Internet Conference venue -/
def Wuzhen : String := "Wuzhen, China"

/-- Characteristics of Wuzhen -/
structure WuzhenCharacteristics where
  tradition_modernity_blend : Bool
  chinese_foreign_embrace : Bool

/-- The cultural impact of the World Internet Conference -/
def conference_impact (venue : String) (characteristics : WuzhenCharacteristics) : CulturalImpact :=
  { promote_chinese_culture := true,
    innovate_world_culture := true,
    enhance_chinese_influence := true }

/-- Theorem stating the cultural impact of the World Internet Conference in Wuzhen -/
theorem wuzhen_conference_impact :
  let venue := Wuzhen
  let characteristics := { tradition_modernity_blend := true, chinese_foreign_embrace := true }
  let impact := conference_impact venue characteristics
  impact.promote_chinese_culture ∧ impact.innovate_world_culture ∧ impact.enhance_chinese_influence :=
by
  sorry

end wuzhen_conference_impact_l3934_393419


namespace fourth_root_difference_l3934_393492

theorem fourth_root_difference : (81 : ℝ) ^ (1/4) - (1296 : ℝ) ^ (1/4) = -3 := by
  sorry

end fourth_root_difference_l3934_393492


namespace triangle_angle_problem_l3934_393409

/-- Given a triangle with angles 40°, 3x, and x + 10°, prove that x = 32.5° --/
theorem triangle_angle_problem (x : ℝ) : 
  (40 : ℝ) + 3 * x + (x + 10) = 180 → x = 32.5 := by
  sorry

end triangle_angle_problem_l3934_393409


namespace three_player_cooperation_strategy_l3934_393473

/-- Represents the dimensions of the game board -/
def boardSize : Nat := 1000

/-- Represents the possible rectangle shapes that can be painted -/
inductive Rectangle
  | twoByOne
  | oneByTwo
  | oneByThree
  | threeByOne

/-- Represents a player in the game -/
inductive Player
  | Andy
  | Bess
  | Charley
  | Dick

/-- Represents a position on the board -/
structure Position where
  x : Fin boardSize
  y : Fin boardSize

/-- Represents a move in the game -/
structure Move where
  player : Player
  rectangle : Rectangle
  position : Position

/-- The game state -/
structure GameState where
  board : Fin boardSize → Fin boardSize → Bool
  currentPlayer : Player

/-- Function to check if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Bool := sorry

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Function to check if a player has a valid move -/
def hasValidMove (state : GameState) (player : Player) : Bool := sorry

/-- Theorem: There exists a strategy for three players to make the fourth player lose -/
theorem three_player_cooperation_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      ∃ (losingPlayer : Player),
        ¬(hasValidMove (applyMove initialState (strategy initialState)) losingPlayer) :=
sorry

end three_player_cooperation_strategy_l3934_393473


namespace power_fraction_evaluation_l3934_393449

theorem power_fraction_evaluation :
  ((5^2014)^2 - (5^2012)^2) / ((5^2013)^2 - (5^2011)^2) = 25 := by
  sorry

end power_fraction_evaluation_l3934_393449


namespace cats_not_liking_catnip_or_tuna_l3934_393443

/-- Given a pet shop with cats, prove the number of cats that don't like catnip or tuna -/
theorem cats_not_liking_catnip_or_tuna
  (total_cats : ℕ)
  (cats_like_catnip : ℕ)
  (cats_like_tuna : ℕ)
  (cats_like_both : ℕ)
  (h1 : total_cats = 80)
  (h2 : cats_like_catnip = 15)
  (h3 : cats_like_tuna = 60)
  (h4 : cats_like_both = 10) :
  total_cats - (cats_like_catnip + cats_like_tuna - cats_like_both) = 15 :=
by sorry

end cats_not_liking_catnip_or_tuna_l3934_393443


namespace trig_identity_l3934_393471

theorem trig_identity (θ a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^4 / a + (Real.cos θ)^4 / b = 1 / (2 * (a + b)) →
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b)^2 :=
by sorry

end trig_identity_l3934_393471


namespace logical_equivalence_l3934_393447

theorem logical_equivalence (P Q R : Prop) :
  (¬P ∧ ¬Q → ¬R) ↔ (R → P ∨ Q) := by
  sorry

end logical_equivalence_l3934_393447


namespace triangle_abc_theorem_l3934_393491

-- Define the triangle ABC
theorem triangle_abc_theorem (a b c A B C : ℝ) 
  (h1 : a / Real.tan A = b / (2 * Real.sin B))
  (h2 : a = 6)
  (h3 : b = 2 * c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) : 
  A = π / 3 ∧ 
  (1/2 * b * c * Real.sin A : ℝ) = 6 * Real.sqrt 3 := by
  sorry

end triangle_abc_theorem_l3934_393491


namespace age_ratio_l3934_393487

def cody_age : ℕ := 14
def grandmother_age : ℕ := 84

theorem age_ratio : (grandmother_age : ℚ) / (cody_age : ℚ) = 6 := by
  sorry

end age_ratio_l3934_393487


namespace clara_climbs_96_blocks_l3934_393423

/-- The number of stone blocks Clara climbs past in the historical tower -/
def total_blocks (levels : ℕ) (steps_per_level : ℕ) (blocks_per_step : ℕ) : ℕ :=
  levels * steps_per_level * blocks_per_step

/-- Theorem stating that Clara climbs past 96 blocks of stone -/
theorem clara_climbs_96_blocks :
  total_blocks 4 8 3 = 96 := by
  sorry

end clara_climbs_96_blocks_l3934_393423


namespace power_multiplication_l3934_393406

theorem power_multiplication (a : ℝ) : 4 * a^2 * a = 4 * a^3 := by
  sorry

end power_multiplication_l3934_393406


namespace sin_cos_sum_equals_sqrt_three_half_l3934_393472

theorem sin_cos_sum_equals_sqrt_three_half : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (130 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_cos_sum_equals_sqrt_three_half_l3934_393472


namespace geometric_progression_sum_relation_l3934_393463

theorem geometric_progression_sum_relation 
  (a : ℝ) (p q : ℝ) (S S₁ : ℝ) 
  (ha : 0 < a ∧ a < 1) (hp : p > 0) (hq : q > 0)
  (hS : S = (1 - a^p)⁻¹) (hS₁ : S₁ = (1 - a^q)⁻¹) :
  S^q * (S₁ - 1)^p = S₁^p * (S - 1)^q := by sorry

end geometric_progression_sum_relation_l3934_393463


namespace negation_of_existence_squared_less_than_one_l3934_393425

theorem negation_of_existence_squared_less_than_one :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by
  sorry

end negation_of_existence_squared_less_than_one_l3934_393425


namespace coin_division_problem_l3934_393440

theorem coin_division_problem (n : ℕ) : 
  (n > 0 ∧ 
   n % 8 = 7 ∧ 
   n % 7 = 5 ∧ 
   ∀ m : ℕ, (m > 0 ∧ m % 8 = 7 ∧ m % 7 = 5) → n ≤ m) →
  (n = 47 ∧ n % 9 = 2) :=
by sorry

end coin_division_problem_l3934_393440


namespace inequality_system_solution_l3934_393457

theorem inequality_system_solution (x : ℝ) : 
  (x - 2 > 1 ∧ -2 * x ≤ 4) ↔ x > 3 := by
sorry

end inequality_system_solution_l3934_393457


namespace sequence_range_l3934_393485

-- Define the sequence a_n
def a (n : ℕ+) (p : ℝ) : ℝ := 2 * (n : ℝ)^2 + p * (n : ℝ)

-- State the theorem
theorem sequence_range (p : ℝ) :
  (∀ n : ℕ+, a n p < a (n + 1) p) ↔ p > -6 :=
sorry

end sequence_range_l3934_393485


namespace arithmetic_sequence_property_l3934_393468

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 4 + a 7 + a 10 = 30 →
  a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 := by
  sorry

end arithmetic_sequence_property_l3934_393468


namespace roses_cut_proof_l3934_393446

/-- Given a vase with an initial number of roses and a final number of roses,
    calculate the number of roses that were added. -/
def roses_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 2 initial roses and 23 final roses,
    the number of roses added is 21. -/
theorem roses_cut_proof :
  roses_added 2 23 = 21 := by
  sorry

end roses_cut_proof_l3934_393446
