import Mathlib

namespace NUMINAMATH_GPT_value_of_m_l1765_176534

theorem value_of_m :
  ∀ m : ℝ, (x : ℝ) → (x^2 - 5 * x + m = (x - 3) * (x - 2)) → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1765_176534


namespace NUMINAMATH_GPT_boys_in_other_communities_l1765_176596

theorem boys_in_other_communities (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℕ)
  (H_total : total_boys = 400)
  (H_muslim : muslim_percent = 44)
  (H_hindu : hindu_percent = 28)
  (H_sikh : sikh_percent = 10) :
  total_boys * (1 - (muslim_percent + hindu_percent + sikh_percent) / 100) = 72 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_other_communities_l1765_176596


namespace NUMINAMATH_GPT_unique_solution_l1765_176530

def system_of_equations (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) :=
  a11 * x1 + a12 * x2 + a13 * x3 = 0 ∧
  a21 * x1 + a22 * x2 + a23 * x3 = 0 ∧
  a31 * x1 + a32 * x2 + a33 * x3 = 0

theorem unique_solution
  (x1 x2 x3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h_pos: 0 < a11 ∧ 0 < a22 ∧ 0 < a33)
  (h_neg: a12 < 0 ∧ a13 < 0 ∧ a21 < 0 ∧ a23 < 0 ∧ a31 < 0 ∧ a32 < 0)
  (h_sum_pos: 0 < a11 + a12 + a13 ∧ 0 < a21 + a22 + a23 ∧ 0 < a31 + a32 + a33)
  (h_system: system_of_equations a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3):
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := sorry

end NUMINAMATH_GPT_unique_solution_l1765_176530


namespace NUMINAMATH_GPT_john_age_is_24_l1765_176569

noncomputable def john_age_condition (j d b : ℕ) : Prop :=
  j = d - 28 ∧
  j + d = 76 ∧
  j + 5 = 2 * (b + 5)

theorem john_age_is_24 (d b : ℕ) : ∃ j, john_age_condition j d b ∧ j = 24 :=
by
  use 24
  unfold john_age_condition
  sorry

end NUMINAMATH_GPT_john_age_is_24_l1765_176569


namespace NUMINAMATH_GPT_average_rate_of_trip_l1765_176514

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_average_rate_of_trip_l1765_176514


namespace NUMINAMATH_GPT_range_of_k_l1765_176510

theorem range_of_k (k : ℝ) : (∀ (x : ℝ), k * x ^ 2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l1765_176510


namespace NUMINAMATH_GPT_spring_length_relationship_l1765_176550

def spring_length (x : ℝ) : ℝ := 6 + 0.3 * x

theorem spring_length_relationship (x : ℝ) : spring_length x = 0.3 * x + 6 :=
by sorry

end NUMINAMATH_GPT_spring_length_relationship_l1765_176550


namespace NUMINAMATH_GPT_prime_exists_solution_l1765_176548

theorem prime_exists_solution (p : ℕ) [hp : Fact p.Prime] :
  ∃ n : ℕ, (6 * n^2 + 5 * n + 1) % p = 0 :=
by
  sorry

end NUMINAMATH_GPT_prime_exists_solution_l1765_176548


namespace NUMINAMATH_GPT_soldiers_height_order_l1765_176594

theorem soldiers_height_order {n : ℕ} (a b : Fin n → ℝ) 
  (ha : ∀ i j, i ≤ j → a i ≥ a j) 
  (hb : ∀ i j, i ≤ j → b i ≥ b j) 
  (h : ∀ i, a i ≤ b i) :
  ∀ i, a i ≤ b i :=
  by sorry

end NUMINAMATH_GPT_soldiers_height_order_l1765_176594


namespace NUMINAMATH_GPT_oil_amount_correct_l1765_176536

-- Definitions based on the conditions in the problem
def initial_amount : ℝ := 0.16666666666666666
def additional_amount : ℝ := 0.6666666666666666
def final_amount : ℝ := 0.8333333333333333

-- Lean 4 statement to prove the given problem
theorem oil_amount_correct :
  initial_amount + additional_amount = final_amount :=
by
  sorry

end NUMINAMATH_GPT_oil_amount_correct_l1765_176536


namespace NUMINAMATH_GPT_speed_of_mother_minimum_running_time_l1765_176584

namespace XiaotongTravel

def distance_to_binjiang : ℝ := 4320
def time_diff : ℝ := 12
def speed_rate : ℝ := 1.2

theorem speed_of_mother : 
  ∃ (x : ℝ), (distance_to_binjiang / x - distance_to_binjiang / (speed_rate * x) = time_diff) → (speed_rate * x = 72) :=
sorry

def distance_to_company : ℝ := 2940
def running_speed : ℝ := 150
def total_time : ℝ := 30

theorem minimum_running_time :
  ∃ (y : ℝ), ((distance_to_company - running_speed * y) / 72 + y ≤ total_time) → (y ≥ 10) :=
sorry

end XiaotongTravel

end NUMINAMATH_GPT_speed_of_mother_minimum_running_time_l1765_176584


namespace NUMINAMATH_GPT_sequence_property_l1765_176575

theorem sequence_property
  (b : ℝ) (h₀ : b > 0)
  (u : ℕ → ℝ)
  (h₁ : u 1 = b)
  (h₂ : ∀ n ≥ 1, u (n + 1) = 1 / (2 - u n)) :
  u 10 = (4 * b - 3) / (6 * b - 5) :=
by
  sorry

end NUMINAMATH_GPT_sequence_property_l1765_176575


namespace NUMINAMATH_GPT_linear_equation_in_x_l1765_176558

theorem linear_equation_in_x (m : ℤ) (h : |m| = 1) (h₂ : m - 1 ≠ 0) : m = -1 :=
sorry

end NUMINAMATH_GPT_linear_equation_in_x_l1765_176558


namespace NUMINAMATH_GPT_max_x_plus_y_l1765_176505

-- Define the conditions as hypotheses in a Lean statement
theorem max_x_plus_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^4 = (x - 1) * (y^3 - 23) - 1) :
  x + y ≤ 7 ∧ (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^4 = (x - 1) * (y^3 - 23) - 1 ∧ x + y = 7) :=
by
  sorry

end NUMINAMATH_GPT_max_x_plus_y_l1765_176505


namespace NUMINAMATH_GPT_range_of_a_l1765_176533

theorem range_of_a (a : ℝ) :
  (a + 1)^2 > (3 - 2 * a)^2 ↔ (2 / 3) < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1765_176533


namespace NUMINAMATH_GPT_maximize_product_l1765_176589

theorem maximize_product (x y z : ℝ) (h1 : x ≥ 20) (h2 : y ≥ 40) (h3 : z ≥ 1675) (h4 : x + y + z = 2015) :
  x * y * z ≤ 721480000 / 27 :=
by sorry

end NUMINAMATH_GPT_maximize_product_l1765_176589


namespace NUMINAMATH_GPT_area_S3_l1765_176516

theorem area_S3 {s1 s2 s3 : ℝ} (h1 : s1^2 = 25)
  (h2 : s2 = s1 / Real.sqrt 2)
  (h3 : s3 = s2 / Real.sqrt 2)
  : s3^2 = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_area_S3_l1765_176516


namespace NUMINAMATH_GPT_problem1_problem2_l1765_176555

-- Problem 1
theorem problem1 : ((2 / 3 - 1 / 12 - 1 / 15) * -60) = -31 := by
  sorry

-- Problem 2
theorem problem2 : ((-7 / 8) / ((7 / 4) - 7 / 8 - 7 / 12)) = -3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1765_176555


namespace NUMINAMATH_GPT_sphere_radius_proportional_l1765_176525

theorem sphere_radius_proportional
  (k : ℝ)
  (r1 r2 : ℝ)
  (W1 W2 : ℝ)
  (h_weight_area : ∀ (r : ℝ), W1 = k * (4 * π * r^2))
  (h_given1: W2 = 32)
  (h_given2: r2 = 0.3)
  (h_given3: W1 = 8):
  r1 = 0.15 := 
by
  sorry

end NUMINAMATH_GPT_sphere_radius_proportional_l1765_176525


namespace NUMINAMATH_GPT_jaxon_toys_l1765_176500

-- Definitions as per the conditions
def toys_jaxon : ℕ := sorry
def toys_gabriel : ℕ := 2 * toys_jaxon
def toys_jerry : ℕ := 2 * toys_jaxon + 8
def total_toys : ℕ := toys_jaxon + toys_gabriel + toys_jerry

-- Theorem to prove
theorem jaxon_toys : total_toys = 83 → toys_jaxon = 15 := sorry

end NUMINAMATH_GPT_jaxon_toys_l1765_176500


namespace NUMINAMATH_GPT_ice_cream_sandwiches_each_l1765_176503

theorem ice_cream_sandwiches_each (total_ice_cream_sandwiches : ℕ) (number_of_nieces : ℕ) 
  (h1 : total_ice_cream_sandwiches = 143) (h2 : number_of_nieces = 11) : 
  total_ice_cream_sandwiches / number_of_nieces = 13 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_sandwiches_each_l1765_176503


namespace NUMINAMATH_GPT_table_price_l1765_176568

theorem table_price (C T : ℝ) (h1 : 2 * C + T = 0.6 * (C + 2 * T)) (h2 : C + T = 96) : T = 84 := by
  sorry

end NUMINAMATH_GPT_table_price_l1765_176568


namespace NUMINAMATH_GPT_part_I_part_II_l1765_176560

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part_I (a : ℝ) (h1 : 0 < a) (h2 : ∀ x : ℝ, 0 < x → f x a ≥ 0) : a = 1 := 
sorry

theorem part_II (n : ℕ) (hn : 0 < n) : 
  let an := (1 + 1 / (n : ℝ)) ^ n
  let bn := (1 + 1 / (n : ℝ)) ^ (n + 1)
  an < Real.exp 1 ∧ Real.exp 1 < bn := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l1765_176560


namespace NUMINAMATH_GPT_line_passes_through_parabola_vertex_l1765_176540

theorem line_passes_through_parabola_vertex : 
  ∃ (c : ℝ), (∀ (x : ℝ), y = 2 * x + c → ∃ (x0 : ℝ), (x0 = 0 ∧ y = c^2)) ∧ 
  (∀ (c1 c2 : ℝ), (y = 2 * x + c1 ∧ y = 2 * x + c2 → c1 = c2)) → 
  ∃ c : ℝ, c = 0 ∨ c = 1 :=
by 
  -- Proof should be inserted here
  sorry

end NUMINAMATH_GPT_line_passes_through_parabola_vertex_l1765_176540


namespace NUMINAMATH_GPT_number_of_pairs_divisible_by_five_l1765_176571

theorem number_of_pairs_divisible_by_five :
  (∃ n : ℕ, n = 864) ↔
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 80) ∧ (1 ≤ b ∧ b ≤ 30) →
  (a * b) % 5 = 0 → (∃ n : ℕ, n = 864) := 
sorry

end NUMINAMATH_GPT_number_of_pairs_divisible_by_five_l1765_176571


namespace NUMINAMATH_GPT_find_a2_geometric_sequence_l1765_176553

theorem find_a2_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) 
  (h_a1 : a 1 = 1 / 4) (h_eq : a 3 * a 5 = 4 * (a 4 - 1)) : a 2 = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_geometric_sequence_l1765_176553


namespace NUMINAMATH_GPT_man_and_son_work_together_l1765_176537

-- Define the rates at which the man and his son can complete the work
def man_work_rate := 1 / 5
def son_work_rate := 1 / 20

-- Define the combined work rate when they work together
def combined_work_rate := man_work_rate + son_work_rate

-- Define the total time taken to complete the work together
def days_to_complete_together := 1 / combined_work_rate

-- The theorem stating that they will complete the work in 4 days
theorem man_and_son_work_together : days_to_complete_together = 4 := by
  sorry

end NUMINAMATH_GPT_man_and_son_work_together_l1765_176537


namespace NUMINAMATH_GPT_percentage_more_research_l1765_176597

-- Defining the various times spent
def acclimation_period : ℝ := 1
def learning_basics_period : ℝ := 2
def dissertation_fraction : ℝ := 0.5
def total_time : ℝ := 7

-- Defining the time spent on each activity
def dissertation_period := dissertation_fraction * acclimation_period
def research_period := total_time - acclimation_period - learning_basics_period - dissertation_period

-- The main theorem to prove
theorem percentage_more_research : 
  ((research_period - learning_basics_period) / learning_basics_period) * 100 = 75 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_percentage_more_research_l1765_176597


namespace NUMINAMATH_GPT_probability_red_or_green_is_two_thirds_l1765_176535

-- Define the conditions
def total_balls := 2 + 3 + 4
def favorable_outcomes := 2 + 4

-- Define the probability calculation
def probability_red_or_green := (favorable_outcomes : ℚ) / total_balls

-- The theorem statement
theorem probability_red_or_green_is_two_thirds : probability_red_or_green = 2 / 3 := by
  -- This part will contain the proof using Lean, but we skip it with "sorry" for now.
  sorry

end NUMINAMATH_GPT_probability_red_or_green_is_two_thirds_l1765_176535


namespace NUMINAMATH_GPT_warriors_wins_count_l1765_176518

variable {wins : ℕ → ℕ}
variable (raptors hawks warriors spurs lakers : ℕ)

def conditions (wins : ℕ → ℕ) (raptors hawks warriors spurs lakers : ℕ) : Prop :=
  wins raptors > wins hawks ∧
  wins warriors > wins spurs ∧ wins warriors < wins lakers ∧
  wins spurs > 25

theorem warriors_wins_count
  (wins : ℕ → ℕ)
  (raptors hawks warriors spurs lakers : ℕ)
  (h : conditions wins raptors hawks warriors spurs lakers) :
  wins warriors = 37 := sorry

end NUMINAMATH_GPT_warriors_wins_count_l1765_176518


namespace NUMINAMATH_GPT_fib_fact_last_two_sum_is_five_l1765_176578

def fib_fact_last_two_sum (s : List (Fin 100)) : Fin 100 :=
  s.sum

theorem fib_fact_last_two_sum_is_five :
  fib_fact_last_two_sum [1, 1, 2, 6, 20, 20, 0] = 5 :=
by 
  sorry

end NUMINAMATH_GPT_fib_fact_last_two_sum_is_five_l1765_176578


namespace NUMINAMATH_GPT_smallest_c_no_real_root_l1765_176572

theorem smallest_c_no_real_root (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 5) ↔ c = -4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_no_real_root_l1765_176572


namespace NUMINAMATH_GPT_lucas_initial_pet_beds_l1765_176513

-- Definitions from the problem conditions
def additional_beds := 8
def beds_per_pet := 2
def pets := 10

-- Statement to prove
theorem lucas_initial_pet_beds :
  (pets * beds_per_pet) - additional_beds = 12 := 
by
  sorry

end NUMINAMATH_GPT_lucas_initial_pet_beds_l1765_176513


namespace NUMINAMATH_GPT_focal_length_of_lens_l1765_176523

-- Define the conditions
def initial_screen_distance : ℝ := 80
def moved_screen_distance : ℝ := 40
def lens_formula (f v u : ℝ) : Prop := (1 / f) = (1 / v) + (1 / u)

-- Define the proof goal
theorem focal_length_of_lens :
  ∃ f : ℝ, (f = 100 ∨ f = 60) ∧
  lens_formula f f (1 / 0) ∧  -- parallel beam implies object at infinity u = 1/0
  initial_screen_distance = 80 ∧
  moved_screen_distance = 40 :=
sorry

end NUMINAMATH_GPT_focal_length_of_lens_l1765_176523


namespace NUMINAMATH_GPT_directrix_of_parabola_l1765_176541

-- Define the given condition
def parabola_eq (x y : ℝ) : Prop := y = -4 * x^2

-- The problem we need to prove
theorem directrix_of_parabola :
  ∃ y : ℝ, (∀ x : ℝ, parabola_eq x y) ↔ y = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1765_176541


namespace NUMINAMATH_GPT_calculate_neg_three_minus_one_l1765_176586

theorem calculate_neg_three_minus_one : -3 - 1 = -4 := by
  sorry

end NUMINAMATH_GPT_calculate_neg_three_minus_one_l1765_176586


namespace NUMINAMATH_GPT_product_of_square_roots_l1765_176580

theorem product_of_square_roots (a b : ℝ) (h₁ : a^2 = 9) (h₂ : b^2 = 9) (h₃ : a ≠ b) : a * b = -9 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_product_of_square_roots_l1765_176580


namespace NUMINAMATH_GPT_division_of_composite_products_l1765_176549

noncomputable def product_of_first_seven_composites : ℕ :=
  4 * 6 * 8 * 9 * 10 * 12 * 14

noncomputable def product_of_next_seven_composites : ℕ :=
  15 * 16 * 18 * 20 * 21 * 22 * 24

noncomputable def divided_product_composites : ℚ :=
  product_of_first_seven_composites / product_of_next_seven_composites

theorem division_of_composite_products : divided_product_composites = 1 / 176 := by
  sorry

end NUMINAMATH_GPT_division_of_composite_products_l1765_176549


namespace NUMINAMATH_GPT_prob_factor_less_than_nine_l1765_176598

theorem prob_factor_less_than_nine : 
  (∃ (n : ℕ), n = 72) ∧ (∃ (total_factors : ℕ), total_factors = 12) ∧ 
  (∃ (factors_lt_9 : ℕ), factors_lt_9 = 6) → 
  (6 / 12 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_prob_factor_less_than_nine_l1765_176598


namespace NUMINAMATH_GPT_permutations_mississippi_l1765_176508

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end NUMINAMATH_GPT_permutations_mississippi_l1765_176508


namespace NUMINAMATH_GPT_product_of_four_consecutive_odd_numbers_is_perfect_square_l1765_176539

theorem product_of_four_consecutive_odd_numbers_is_perfect_square (n : ℤ) :
    (n + 0) * (n + 2) * (n + 4) * (n + 6) = 9 :=
sorry

end NUMINAMATH_GPT_product_of_four_consecutive_odd_numbers_is_perfect_square_l1765_176539


namespace NUMINAMATH_GPT_tangent_line_at_origin_l1765_176574

-- Define the function f(x) = x^3 + ax with an extremum at x = 1
def f (x a : ℝ) : ℝ := x^3 + a * x

-- Define the condition for a local extremum at x = 1: f'(1) = 0
def extremum_condition (a : ℝ) : Prop := (3 * 1^2 + a = 0)

-- Define the derivative of f at x = 0
def derivative_at_origin (a : ℝ) : ℝ := 3 * 0^2 + a

-- Define the value of function at x = 0
def value_at_origin (a : ℝ) : ℝ := f 0 a

-- The main theorem to prove
theorem tangent_line_at_origin (a : ℝ) (ha : extremum_condition a) :
    (value_at_origin a = 0) ∧ (derivative_at_origin a = -3) → ∀ x, (3 * x + (f x a - f 0 a) / (x - 0) = 0) := by
  sorry

end NUMINAMATH_GPT_tangent_line_at_origin_l1765_176574


namespace NUMINAMATH_GPT_sumata_family_miles_driven_per_day_l1765_176561

theorem sumata_family_miles_driven_per_day :
  let total_miles := 1837.5
  let number_of_days := 13.5
  let miles_per_day := total_miles / number_of_days
  (miles_per_day : Real) = 136.1111 :=
by
  sorry

end NUMINAMATH_GPT_sumata_family_miles_driven_per_day_l1765_176561


namespace NUMINAMATH_GPT_simplify_frac_l1765_176532

variable (b c : ℕ)
variable (b_val : b = 2)
variable (c_val : c = 3)

theorem simplify_frac : (15 * b ^ 4 * c ^ 2) / (45 * b ^ 3 * c) = 2 :=
by
  rw [b_val, c_val]
  sorry

end NUMINAMATH_GPT_simplify_frac_l1765_176532


namespace NUMINAMATH_GPT_cake_cost_is_20_l1765_176563

-- Define the given conditions
def total_budget : ℕ := 50
def additional_needed : ℕ := 11
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

-- Define the derived conditions
def total_cost : ℕ := total_budget + additional_needed
def combined_bouquet_balloons_cost : ℕ := bouquet_cost + balloons_cost
def cake_cost : ℕ := total_cost - combined_bouquet_balloons_cost

-- The theorem to be proved
theorem cake_cost_is_20 : cake_cost = 20 :=
by
  -- proof steps are not required
  sorry

end NUMINAMATH_GPT_cake_cost_is_20_l1765_176563


namespace NUMINAMATH_GPT_no_real_solutions_l1765_176543

theorem no_real_solutions : ¬ ∃ (r s : ℝ),
  (r - 50) / 3 = (s - 2 * r) / 4 ∧
  r^2 + 3 * s = 50 :=
by {
  -- sorry, proof steps would go here
  sorry
}

end NUMINAMATH_GPT_no_real_solutions_l1765_176543


namespace NUMINAMATH_GPT_simplify_fraction_l1765_176517

variable {a b c : ℝ} -- assuming a, b, c are real numbers

theorem simplify_fraction (hc : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2 * a * b) / (a^2 + c^2 - b^2 + 2 * a * c) = (a + b - c) / (a - b + c) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1765_176517


namespace NUMINAMATH_GPT_shopping_problem_l1765_176520

theorem shopping_problem
  (D S H N : ℝ)
  (h1 : (D - (D / 2 - 10)) + (S - 0.85 * S) + (H - (H - 30)) + (N - N) = 120)
  (T_sale : ℝ := (D / 2 - 10) + 0.85 * S + (H - 30) + N) :
  (120 + 0.10 * T_sale = 0.10 * 1200) →
  D + S + H + N = 1200 :=
by
  sorry

end NUMINAMATH_GPT_shopping_problem_l1765_176520


namespace NUMINAMATH_GPT_total_players_is_28_l1765_176581

def total_players (A B C AB BC AC ABC : ℕ) : ℕ :=
  A + B + C - (AB + BC + AC) + ABC

theorem total_players_is_28 :
  total_players 10 15 18 8 6 4 3 = 28 :=
by
  -- as per inclusion-exclusion principle
  -- T = A + B + C - (AB + BC + AC) + ABC
  -- substituting given values we repeatedly perform steps until final answer
  -- take user inputs to build your final answer.
  sorry

end NUMINAMATH_GPT_total_players_is_28_l1765_176581


namespace NUMINAMATH_GPT_sound_frequency_and_speed_glass_proof_l1765_176579

def length_rod : ℝ := 1.10 -- Length of the glass rod, l in meters
def nodal_distance_air : ℝ := 0.12 -- Distance between nodal points in air, l' in meters
def speed_sound_air : ℝ := 340 -- Speed of sound in air, V in meters per second

-- Frequency of the sound produced
def frequency_sound_produced : ℝ := 1416.67

-- Speed of longitudinal waves in the glass
def speed_longitudinal_glass : ℝ := 3116.67

theorem sound_frequency_and_speed_glass_proof :
  (2 * nodal_distance_air = 0.24) ∧
  (frequency_sound_produced * (2 * length_rod) = speed_longitudinal_glass) :=
by
  -- Here we will include real equivalent math proof in the future
  sorry

end NUMINAMATH_GPT_sound_frequency_and_speed_glass_proof_l1765_176579


namespace NUMINAMATH_GPT_monkeys_bananas_minimum_l1765_176595

theorem monkeys_bananas_minimum (b1 b2 b3 : ℕ) (x y z : ℕ) : 
  (x = 2 * y) ∧ (z = (2 * y) / 3) ∧ 
  (x = (2 * b1) / 3 + (b2 / 3) + (5 * b3) / 12) ∧ 
  (y = (b1 / 6) + (b2 / 3) + (5 * b3) / 12) ∧ 
  (z = (b1 / 6) + (b2 / 3) + (b3 / 6)) →
  b1 = 324 ∧ b2 = 162 ∧ b3 = 72 ∧ (b1 + b2 + b3 = 558) :=
sorry

end NUMINAMATH_GPT_monkeys_bananas_minimum_l1765_176595


namespace NUMINAMATH_GPT_age_multiplier_l1765_176531

theorem age_multiplier (S F M X : ℕ) (h1 : S = 27) (h2 : F = 48) (h3 : S + F = 75)
  (h4 : 27 - X = F - S) (h5 : F = M * X) : M = 8 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_age_multiplier_l1765_176531


namespace NUMINAMATH_GPT_binom_20_10_eq_184756_l1765_176521

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end NUMINAMATH_GPT_binom_20_10_eq_184756_l1765_176521


namespace NUMINAMATH_GPT_locus_of_midpoint_l1765_176557

theorem locus_of_midpoint
  (x y : ℝ)
  (h : ∃ (A : ℝ × ℝ), A = (2*x, 2*y) ∧ (A.1)^2 + (A.2)^2 - 8*A.1 = 0) :
  x^2 + y^2 - 4*x = 0 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_midpoint_l1765_176557


namespace NUMINAMATH_GPT_sum_of_n_and_k_l1765_176506

theorem sum_of_n_and_k (n k : ℕ) 
  (h1 : (n.choose k) * 3 = (n.choose (k + 1)))
  (h2 : (n.choose (k + 1)) * 2 = (n.choose (k + 2))) :
  n + k = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_n_and_k_l1765_176506


namespace NUMINAMATH_GPT_ratio_of_sums_l1765_176591

open Nat

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 8 - 2 * a 3) / 7)

def arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem ratio_of_sums
    (a : ℕ → ℝ)
    (S : ℕ → ℝ)
    (a_arith : arithmetic_sequence_property a 1)
    (s_def : ∀ n, S n = sum_of_first_n_terms a n)
    (a8_eq_2a3 : a 8 = 2 * a 3) :
  S 15 / S 5 = 6 :=
sorry

end NUMINAMATH_GPT_ratio_of_sums_l1765_176591


namespace NUMINAMATH_GPT_purely_imaginary_sufficient_but_not_necessary_l1765_176501

theorem purely_imaginary_sufficient_but_not_necessary (a b : ℝ) (h : ¬(b = 0)) : 
  (a = 0 → p ∧ q) → (q ∧ ¬p) :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_sufficient_but_not_necessary_l1765_176501


namespace NUMINAMATH_GPT_goteborg_to_stockholm_distance_l1765_176547

/-- 
Given that the distance from Goteborg to Jonkoping on a map is 100 cm 
and the distance from Jonkoping to Stockholm is 150 cm, with a map scale of 1 cm: 20 km,
prove that the total distance from Goteborg to Stockholm passing through Jonkoping is 5000 km.
-/
theorem goteborg_to_stockholm_distance :
  let distance_G_to_J := 100 -- distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- distance from Jonkoping to Stockholm in cm
  let scale := 20 -- scale of the map, 1 cm : 20 km
  distance_G_to_J * scale + distance_J_to_S * scale = 5000 := 
by 
  let distance_G_to_J := 100 -- defining the distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- defining the distance from Jonkoping to Stockholm in cm
  let scale := 20 -- defining the scale of the map, 1 cm : 20 km
  sorry

end NUMINAMATH_GPT_goteborg_to_stockholm_distance_l1765_176547


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1765_176570

-- Definitions
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c

-- The condition we are given
axiom m : ℝ

-- The quadratic equation specific condition
axiom quadratic_condition : quadratic_eq 1 2 m = 0

-- The necessary but not sufficient condition for real solutions
theorem necessary_but_not_sufficient (h : m < 2) : 
  ∃ x : ℝ, quadratic_eq 1 2 m x = 0 ∧ quadratic_eq 1 2 m x = 0 → m ≤ 1 ∨ m > 1 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1765_176570


namespace NUMINAMATH_GPT_no_valid_formation_l1765_176524

-- Define the conditions related to the formation:
-- s : number of rows
-- t : number of musicians per row
-- Total musicians = s * t = 400
-- t is divisible by 4
-- 10 ≤ t ≤ 50
-- Additionally, the brass section needs to form a triangle in the first three rows
-- while maintaining equal distribution of musicians from each section in every row.

theorem no_valid_formation (s t : ℕ) (h_mul : s * t = 400) 
  (h_div : t % 4 = 0) 
  (h_range : 10 ≤ t ∧ t ≤ 50) 
  (h_triangle : ∀ (r1 r2 r3 : ℕ), r1 < r2 ∧ r2 < r3 → r1 + r2 + r3 = 100 → false) : 
  x = 0 := by
  sorry

end NUMINAMATH_GPT_no_valid_formation_l1765_176524


namespace NUMINAMATH_GPT_triangle_side_length_l1765_176593

open Real

/-- Given a triangle ABC with the incircle touching side AB at point D,
where AD = 5 and DB = 3, and given that the angle A is 60 degrees,
prove that the length of side BC is 13. -/
theorem triangle_side_length
  (A B C D : Point)
  (AD DB : ℝ)
  (hAD : AD = 5)
  (hDB : DB = 3)
  (angleA : Real)
  (hangleA : angleA = π / 3) : 
  ∃ BC : ℝ, BC = 13 :=
sorry

end NUMINAMATH_GPT_triangle_side_length_l1765_176593


namespace NUMINAMATH_GPT_total_vessels_l1765_176509

open Nat

theorem total_vessels (x y z w : ℕ) (hx : x > 0) (hy : y > x) (hz : z > y) (hw : w > z) :
  ∃ total : ℕ, total = x * (2 * y + 1) + z * (1 + 1 / w) := sorry

end NUMINAMATH_GPT_total_vessels_l1765_176509


namespace NUMINAMATH_GPT_min_time_to_cross_river_l1765_176590

-- Definitions for the time it takes each horse to cross the river
def timeA : ℕ := 2
def timeB : ℕ := 3
def timeC : ℕ := 7
def timeD : ℕ := 6

-- Definition for the minimum time required for all horses to cross the river
def min_crossing_time : ℕ := 18

-- Theorem stating the problem: 
theorem min_time_to_cross_river :
  ∀ (timeA timeB timeC timeD : ℕ), timeA = 2 → timeB = 3 → timeC = 7 → timeD = 6 →
  min_crossing_time = 18 :=
sorry

end NUMINAMATH_GPT_min_time_to_cross_river_l1765_176590


namespace NUMINAMATH_GPT_find_f_x_sq_minus_2_l1765_176577

-- Define the polynomial and its given condition
def f (x : ℝ) : ℝ := sorry  -- f is some polynomial, we'll leave it unspecified for now

-- Assume the given condition
axiom f_condition : ∀ x : ℝ, f (x^2 + 2) = x^4 + 6 * x^2 + 4

-- Prove the desired result
theorem find_f_x_sq_minus_2 (x : ℝ) : f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
sorry

end NUMINAMATH_GPT_find_f_x_sq_minus_2_l1765_176577


namespace NUMINAMATH_GPT_gear_p_revolutions_per_minute_l1765_176592

theorem gear_p_revolutions_per_minute (r : ℝ) 
  (cond2 : ℝ := 40) 
  (cond3 : 1.5 * r + 45 = 1.5 * 40) :
  r = 10 :=
by
  sorry

end NUMINAMATH_GPT_gear_p_revolutions_per_minute_l1765_176592


namespace NUMINAMATH_GPT_number_of_x_for_P_eq_zero_l1765_176582

noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x) - Complex.exp (4 * Complex.I * x)

theorem number_of_x_for_P_eq_zero : 
  ∃ (n : ℕ), n = 4 ∧ ∃ (xs : Fin n → ℝ), (∀ i, 0 ≤ xs i ∧ xs i < 2 * Real.pi ∧ P (xs i) = 0) ∧ Function.Injective xs := 
sorry

end NUMINAMATH_GPT_number_of_x_for_P_eq_zero_l1765_176582


namespace NUMINAMATH_GPT_completion_time_is_midnight_next_day_l1765_176507

-- Define the initial start time
def start_time : ℕ := 9 -- 9:00 AM in hours

-- Define the completion time for 1/4th of the mosaic
def partial_completion_time : ℕ := 3 * 60 + 45  -- 3 hours and 45 minutes in minutes

-- Calculate total_time needed to complete the whole mosaic
def total_time : ℕ := 4 * partial_completion_time -- total time in minutes

-- Define the time at which the artist should finish the entire mosaic
def end_time : ℕ := start_time * 60 + total_time -- end time in minutes

-- Assuming 24 hours in a day, calculate 12:00 AM next day in minutes from midnight
def midnight_next_day : ℕ := 24 * 60

-- Theorem proving the artist will finish at 12:00 AM next day
theorem completion_time_is_midnight_next_day :
  end_time = midnight_next_day := by
    sorry -- proof not required

end NUMINAMATH_GPT_completion_time_is_midnight_next_day_l1765_176507


namespace NUMINAMATH_GPT_circle_center_is_21_l1765_176562

theorem circle_center_is_21 : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 5 = 0 →
                                      ∃ h k : ℝ, h = 2 ∧ k = 1 ∧ (x - h)^2 + (y - k)^2 = 10 :=
by
  intro x y h_eq
  sorry

end NUMINAMATH_GPT_circle_center_is_21_l1765_176562


namespace NUMINAMATH_GPT_rectangle_perimeter_l1765_176515

theorem rectangle_perimeter (a b : ℕ) : 
  (2 * a + b = 6 ∨ a + 2 * b = 6 ∨ 2 * a + b = 9 ∨ a + 2 * b = 9) → 
  2 * a + 2 * b = 10 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1765_176515


namespace NUMINAMATH_GPT_percentage_increase_l1765_176599

variable (x r : ℝ)

theorem percentage_increase (h_x : x = 78.4) (h_r : x = 70 * (1 + r)) : r = 0.12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_increase_l1765_176599


namespace NUMINAMATH_GPT_cube_volume_l1765_176529

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end NUMINAMATH_GPT_cube_volume_l1765_176529


namespace NUMINAMATH_GPT_hyperbola_focal_length_l1765_176522

noncomputable def a : ℝ := Real.sqrt 10
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt (a ^ 2 + b ^ 2)
noncomputable def focal_length : ℝ := 2 * c

theorem hyperbola_focal_length :
  focal_length = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l1765_176522


namespace NUMINAMATH_GPT_percentage_decrease_of_larger_angle_l1765_176554

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end NUMINAMATH_GPT_percentage_decrease_of_larger_angle_l1765_176554


namespace NUMINAMATH_GPT_log_domain_eq_l1765_176527

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 2 * x - 3

def log_domain (x : ℝ) : Prop := quadratic_expr x > 0

theorem log_domain_eq :
  {x : ℝ | log_domain x} = 
  {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_log_domain_eq_l1765_176527


namespace NUMINAMATH_GPT_intersection_M_N_l1765_176567

def M : Set ℝ := { x | x ≤ 4 }
def N : Set ℝ := { x | 0 < x }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x ≤ 4 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1765_176567


namespace NUMINAMATH_GPT_alfred_gain_percent_l1765_176519

theorem alfred_gain_percent :
  let purchase_price := 4700
  let repair_costs := 800
  let selling_price := 5800
  let total_cost := purchase_price + repair_costs
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 5.45 := 
by
  sorry

end NUMINAMATH_GPT_alfred_gain_percent_l1765_176519


namespace NUMINAMATH_GPT_building_height_is_74_l1765_176585

theorem building_height_is_74
  (building_shadow : ℚ)
  (flagpole_height : ℚ)
  (flagpole_shadow : ℚ)
  (ratio_valid : building_shadow / flagpole_shadow = 21 / 8)
  (flagpole_height_value : flagpole_height = 28)
  (building_shadow_value : building_shadow = 84)
  (flagpole_shadow_value : flagpole_shadow = 32) :
  ∃ (h : ℚ), h = 74 := by
  sorry

end NUMINAMATH_GPT_building_height_is_74_l1765_176585


namespace NUMINAMATH_GPT_pyramid_pattern_l1765_176542

theorem pyramid_pattern
  (R : ℕ → ℕ)  -- a function representing the number of blocks in each row
  (R₁ : R 1 = 9)  -- the first row has 9 blocks
  (sum_eq : R 1 + R 2 + R 3 + R 4 + R 5 = 25)  -- the total number of blocks is 25
  (pattern : ∀ n, 1 ≤ n ∧ n < 5 → R (n + 1) = R n - 2) : ∃ d, d = 2 :=
by
  have pattern_valid : R 1 = 9 ∧ R 2 = 7 ∧ R 3 = 5 ∧ R 4 = 3 ∧ R 5 = 1 :=
    sorry  -- Proof omitted
  exact ⟨2, rfl⟩

end NUMINAMATH_GPT_pyramid_pattern_l1765_176542


namespace NUMINAMATH_GPT_battery_lasts_12_more_hours_l1765_176544

-- Define initial conditions
def standby_battery_life : ℕ := 36
def active_battery_life : ℕ := 4
def total_time_on : ℕ := 12
def active_usage_time : ℕ := 90  -- in minutes

-- Conversion and calculation functions
def active_usage_hours : ℚ := active_usage_time / 60
def standby_consumption_rate : ℚ := 1 / standby_battery_life
def active_consumption_rate : ℚ := 1 / active_battery_life
def battery_used_standby : ℚ := (total_time_on - active_usage_hours) * standby_consumption_rate
def battery_used_active : ℚ := active_usage_hours * active_consumption_rate
def total_battery_used : ℚ := battery_used_standby + battery_used_active
def remaining_battery : ℚ := 1 - total_battery_used
def additional_hours_standby : ℚ := remaining_battery / standby_consumption_rate

-- Proof statement
theorem battery_lasts_12_more_hours : additional_hours_standby = 12 := by
  sorry

end NUMINAMATH_GPT_battery_lasts_12_more_hours_l1765_176544


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1765_176566

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 2)
  (h3 : a 5 = 1/4) :
  q = 1/2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1765_176566


namespace NUMINAMATH_GPT_determine_m_with_opposite_roots_l1765_176511

theorem determine_m_with_opposite_roots (c d k : ℝ) (h : c + d ≠ 0):
  (∃ m : ℝ, ∀ x : ℝ, (x^2 - d * x) / (c * x - k) = (m - 2) / (m + 2) ∧ 
            (x = -y ∧ y = -x)) ↔ m = 2 * (c - d) / (c + d) :=
sorry

end NUMINAMATH_GPT_determine_m_with_opposite_roots_l1765_176511


namespace NUMINAMATH_GPT_largest_domain_of_f_l1765_176526

theorem largest_domain_of_f (f : ℝ → ℝ) (dom : ℝ → Prop) :
  (∀ x : ℝ, dom x → dom (1 / x)) →
  (∀ x : ℝ, dom x → (f x + f (1 / x) = x)) →
  (∀ x : ℝ, dom x ↔ x = 1 ∨ x = -1) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_largest_domain_of_f_l1765_176526


namespace NUMINAMATH_GPT_prob_sum_seven_prob_two_fours_l1765_176546

-- Definitions and conditions
def total_outcomes : ℕ := 36
def outcomes_sum_seven : ℕ := 6
def outcomes_two_fours : ℕ := 1

-- Proof problem for question 1
theorem prob_sum_seven : outcomes_sum_seven / total_outcomes = 1 / 6 :=
by
  sorry

-- Proof problem for question 2
theorem prob_two_fours : outcomes_two_fours / total_outcomes = 1 / 36 :=
by
  sorry

end NUMINAMATH_GPT_prob_sum_seven_prob_two_fours_l1765_176546


namespace NUMINAMATH_GPT_tires_in_parking_lot_l1765_176573

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end NUMINAMATH_GPT_tires_in_parking_lot_l1765_176573


namespace NUMINAMATH_GPT_simplify_and_evaluate_correct_l1765_176564

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_correct_l1765_176564


namespace NUMINAMATH_GPT_percent_decrease_l1765_176545

theorem percent_decrease (P S : ℝ) (h₀ : P = 100) (h₁ : S = 70) :
  ((P - S) / P) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_l1765_176545


namespace NUMINAMATH_GPT_inequality_holds_for_all_l1765_176552

theorem inequality_holds_for_all (m : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 8 * x + 20) / (m * x^2 - m * x - 1) < 0) : -4 < m ∧ m ≤ 0 := 
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_l1765_176552


namespace NUMINAMATH_GPT_range_of_a_l1765_176587

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a (a : ℝ) (h : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∨ a ∈ Set.Ioi 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1765_176587


namespace NUMINAMATH_GPT_cube_root_of_8_l1765_176512

theorem cube_root_of_8 : (∃ x : ℝ, x * x * x = 8) ∧ (∃ y : ℝ, y * y * y = 8 → y = 2) :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_8_l1765_176512


namespace NUMINAMATH_GPT_initial_puppies_l1765_176528

-- Definitions based on the conditions in the problem
def sold : ℕ := 21
def puppies_per_cage : ℕ := 9
def number_of_cages : ℕ := 9

-- The statement to prove
theorem initial_puppies : sold + (puppies_per_cage * number_of_cages) = 102 := by
  sorry

end NUMINAMATH_GPT_initial_puppies_l1765_176528


namespace NUMINAMATH_GPT_two_digit_numbers_div_quotient_remainder_l1765_176538

theorem two_digit_numbers_div_quotient_remainder (x y : ℕ) (N : ℕ) (h1 : N = 10 * x + y) (h2 : N = 7 * (x + y) + 6) (hx_range : 1 ≤ x ∧ x ≤ 9) (hy_range : 0 ≤ y ∧ y ≤ 9) :
  N = 62 ∨ N = 83 := sorry

end NUMINAMATH_GPT_two_digit_numbers_div_quotient_remainder_l1765_176538


namespace NUMINAMATH_GPT_diana_owes_amount_l1765_176551

def principal : ℝ := 60
def rate : ℝ := 0.06
def time : ℝ := 1
def interest := principal * rate * time
def original_amount := principal
def total_amount := original_amount + interest

theorem diana_owes_amount :
  total_amount = 63.60 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_diana_owes_amount_l1765_176551


namespace NUMINAMATH_GPT_chess_match_duration_l1765_176502

def time_per_move_polly := 28
def time_per_move_peter := 40
def total_moves := 30
def moves_per_player := total_moves / 2

def Polly_time := moves_per_player * time_per_move_polly
def Peter_time := moves_per_player * time_per_move_peter
def total_time_seconds := Polly_time + Peter_time
def total_time_minutes := total_time_seconds / 60

theorem chess_match_duration : total_time_minutes = 17 := by
  sorry

end NUMINAMATH_GPT_chess_match_duration_l1765_176502


namespace NUMINAMATH_GPT_A_star_B_eq_l1765_176556

def A : Set ℝ := {x | ∃ y, y = 2 * x - x^2}
def B : Set ℝ := {y | ∃ x, y = 2^x ∧ x > 0}
def A_star_B : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem A_star_B_eq : A_star_B = {x | x ≤ 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_A_star_B_eq_l1765_176556


namespace NUMINAMATH_GPT_arith_seq_ratio_l1765_176588

theorem arith_seq_ratio (a_2 a_3 S_4 S_5 : ℕ) 
  (arithmetic_seq : ∀ n : ℕ, ℕ)
  (sum_of_first_n_terms : ∀ n : ℕ, ℕ)
  (h1 : (a_2 : ℚ) / a_3 = 1 / 3) 
  (h2 : S_4 = 4 * (a_2 - (a_3 - a_2)) + ((4 * 3 * (a_3 - a_2)) / 2)) 
  (h3 : S_5 = 5 * (a_2 - (a_3 - a_2)) + ((5 * 4 * (a_3 - a_2)) / 2)) :
  (S_4 : ℚ) / S_5 = 8 / 15 :=
by sorry

end NUMINAMATH_GPT_arith_seq_ratio_l1765_176588


namespace NUMINAMATH_GPT_problem_i31_problem_i32_problem_i33_problem_i34_l1765_176583

-- Problem I3.1
theorem problem_i31 (a : ℝ) :
  a = 1.8 * 5.0865 + 1 - 0.0865 * 1.8 → a = 10 :=
by sorry

-- Problem I3.2
theorem problem_i32 (a b : ℕ) (oh ok : ℕ) (OABC : Prop) :
  oh = ok ∧ oh = a ∧ ok = a ∧ OABC ∧ (b = AC) → b = 10 :=
by sorry

-- Problem I3.3
theorem problem_i33 (b c : ℕ) :
  b = 10 → c = (10 - 2) :=
by sorry

-- Problem I3.4
theorem problem_i34 (c d : ℕ) :
  c = 30 → d = 3 * c → d = 90 :=
by sorry

end NUMINAMATH_GPT_problem_i31_problem_i32_problem_i33_problem_i34_l1765_176583


namespace NUMINAMATH_GPT_net_loss_is_1_percent_l1765_176565

noncomputable def net_loss_percent (CP SP1 SP2 SP3 SP4 : ℝ) : ℝ :=
  let TCP := 4 * CP
  let TSP := SP1 + SP2 + SP3 + SP4
  ((TCP - TSP) / TCP) * 100

theorem net_loss_is_1_percent
  (CP : ℝ)
  (HCP : CP = 1000)
  (SP1 : ℝ)
  (HSP1 : SP1 = CP * 1.1 * 0.95)
  (SP2 : ℝ)
  (HSP2 : SP2 = (CP * 0.9) * 1.02)
  (SP3 : ℝ)
  (HSP3 : SP3 = (CP * 1.2) * 1.03)
  (SP4 : ℝ)
  (HSP4 : SP4 = (CP * 0.75) * 1.01) :
  abs (net_loss_percent CP SP1 SP2 SP3 SP4 + 1.09) < 0.01 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_net_loss_is_1_percent_l1765_176565


namespace NUMINAMATH_GPT_smallest_sum_of_digits_l1765_176559

noncomputable def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_sum_of_digits (n : ℕ) (h : sum_of_digits n = 2017) : sum_of_digits (n + 1) = 2 := 
sorry

end NUMINAMATH_GPT_smallest_sum_of_digits_l1765_176559


namespace NUMINAMATH_GPT_Tyrone_total_money_l1765_176576

theorem Tyrone_total_money :
  let usd_bills := 4 * 1 + 1 * 10 + 2 * 5 + 30 * 0.25 + 5 * 0.5 + 48 * 0.1 + 12 * 0.05 + 4 * 1 + 64 * 0.01 + 3 * 2 + 5 * 0.5
  let euro_to_usd := 20 * 1.1
  let pound_to_usd := 15 * 1.32
  let cad_to_usd := 6 * 0.76
  let total_usd_currency := usd_bills
  let total_foreign_usd_currency := euro_to_usd + pound_to_usd + cad_to_usd
  let total_money := total_usd_currency + total_foreign_usd_currency
  total_money = 98.90 :=
by
  sorry

end NUMINAMATH_GPT_Tyrone_total_money_l1765_176576


namespace NUMINAMATH_GPT_sum_leq_six_of_quadratic_roots_l1765_176504

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_leq_six_of_quadratic_roots_l1765_176504
