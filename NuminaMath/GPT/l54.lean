import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l54_5488

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l54_5488


namespace NUMINAMATH_GPT_final_percentage_is_46_l54_5450

def initial_volume : ℚ := 50
def initial_concentration : ℚ := 0.60
def drained_volume : ℚ := 35
def replacement_concentration : ℚ := 0.40

def initial_chemical_amount : ℚ := initial_volume * initial_concentration
def drained_chemical_amount : ℚ := drained_volume * initial_concentration
def remaining_chemical_amount : ℚ := initial_chemical_amount - drained_chemical_amount
def added_chemical_amount : ℚ := drained_volume * replacement_concentration
def final_chemical_amount : ℚ := remaining_chemical_amount + added_chemical_amount
def final_volume : ℚ := initial_volume

def final_percentage : ℚ := (final_chemical_amount / final_volume) * 100

theorem final_percentage_is_46 :
  final_percentage = 46 := by
  sorry

end NUMINAMATH_GPT_final_percentage_is_46_l54_5450


namespace NUMINAMATH_GPT_movie_box_office_growth_l54_5495

theorem movie_box_office_growth 
  (x : ℝ) 
  (r₁ r₃ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₃ : r₃ = 2.4) 
  (growth : r₃ = (1 + x) ^ 2) : 
  (1 + x) ^ 2 = 2.4 :=
by sorry

end NUMINAMATH_GPT_movie_box_office_growth_l54_5495


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l54_5419

-- Definition of the infinite geometric series with given first term and common ratio
def infinite_geometric_series (a : ℚ) (r : ℚ) : ℚ := a / (1 - r)

-- Problem statement
theorem infinite_geometric_series_sum :
  infinite_geometric_series (5 / 3) (-2 / 9) = 15 / 11 :=
sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l54_5419


namespace NUMINAMATH_GPT_range_of_n_l54_5474

theorem range_of_n (m n : ℝ) (h₁ : n = m^2 + 2 * m + 2) (h₂ : |m| < 2) : -1 ≤ n ∧ n < 10 :=
sorry

end NUMINAMATH_GPT_range_of_n_l54_5474


namespace NUMINAMATH_GPT_each_person_gets_equal_share_l54_5480

-- Definitions based on the conditions
def number_of_friends: Nat := 4
def initial_chicken_wings: Nat := 9
def additional_chicken_wings: Nat := 7

-- The proof statement
theorem each_person_gets_equal_share (total_chicken_wings := initial_chicken_wings + additional_chicken_wings) : 
       total_chicken_wings / number_of_friends = 4 := 
by 
  sorry

end NUMINAMATH_GPT_each_person_gets_equal_share_l54_5480


namespace NUMINAMATH_GPT_principal_trebled_after_5_years_l54_5446

-- Definitions of the conditions
def original_simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100
def total_simple_interest (P R n T : ℕ) : ℕ := (P * R * n) / 100 + (3 * P * R * (T - n)) / 100

-- The theorem statement
theorem principal_trebled_after_5_years :
  ∀ (P R : ℕ), original_simple_interest P R 10 = 800 →
              total_simple_interest P R 5 10 = 1600 →
              5 = 5 :=
by
  intros P R h1 h2
  sorry

end NUMINAMATH_GPT_principal_trebled_after_5_years_l54_5446


namespace NUMINAMATH_GPT_sequence_value_l54_5467

theorem sequence_value (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 :=
by
  -- The proof is not required, so we add sorry to indicate that
  sorry

end NUMINAMATH_GPT_sequence_value_l54_5467


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l54_5418

theorem system1_solution :
  ∃ (x y : ℤ), (4 * x - y = 1) ∧ (y = 2 * x + 3) ∧ (x = 2) ∧ (y = 7) :=
by
  sorry

theorem system2_solution :
  ∃ (x y : ℤ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l54_5418


namespace NUMINAMATH_GPT_segment_AB_length_l54_5454

-- Define the problem conditions
variables (AB CD h : ℝ)
variables (x : ℝ)
variables (AreaRatio : ℝ)
variable (k : ℝ := 5 / 2)

-- The given conditions
def condition1 : Prop := AB = 5 * x ∧ CD = 2 * x
def condition2 : Prop := AB + CD = 280
def condition3 : Prop := h = AB - 20
def condition4 : Prop := AreaRatio = k

-- The statement to prove
theorem segment_AB_length (h k : ℝ) (x : ℝ) :
  (AB = 5 * x ∧ CD = 2 * x) ∧ (AB + CD = 280) ∧ (h = AB - 20) ∧ (AreaRatio = k) → AB = 200 :=
by 
  sorry

end NUMINAMATH_GPT_segment_AB_length_l54_5454


namespace NUMINAMATH_GPT_fraction_to_decimal_representation_l54_5466

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end NUMINAMATH_GPT_fraction_to_decimal_representation_l54_5466


namespace NUMINAMATH_GPT_part1_proof_part2_proof_l54_5402

-- Given conditions
variables (a b x : ℝ)
def y (a b x : ℝ) := a*x^2 + (b-2)*x + 3

-- The initial conditions
noncomputable def conditions := 
  (∀ x, -1 < x ∧ x < 3 → y a b x > 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ y a b 1 = 2)

-- Part (1): Prove that the solution set of y >= 4 is {1}
theorem part1_proof :
  conditions a b →
  {x | y a b x ≥ 4} = {1} :=
  by
    sorry

-- Part (2): Prove that the minimum value of (1/a + 4/b) is 9
theorem part2_proof :
  conditions a b →
  ∃ x, x = 1/a + 4/b ∧ x = 9 :=
  by
    sorry

end NUMINAMATH_GPT_part1_proof_part2_proof_l54_5402


namespace NUMINAMATH_GPT_ferris_wheel_seat_capacity_l54_5435

theorem ferris_wheel_seat_capacity
  (total_seats : ℕ)
  (broken_seats : ℕ)
  (total_people : ℕ)
  (seats_available : ℕ)
  (people_per_seat : ℕ)
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : total_people = 120)
  (h4 : seats_available = total_seats - broken_seats)
  (h5 : people_per_seat = total_people / seats_available) :
  people_per_seat = 15 := 
by sorry

end NUMINAMATH_GPT_ferris_wheel_seat_capacity_l54_5435


namespace NUMINAMATH_GPT_least_positive_integer_x_l54_5477

theorem least_positive_integer_x :
  ∃ x : ℕ, (x > 0) ∧ (∃ k : ℕ, (2 * x + 51) = k * 59) ∧ x = 4 :=
by
  -- Lean statement
  sorry

end NUMINAMATH_GPT_least_positive_integer_x_l54_5477


namespace NUMINAMATH_GPT_perfect_square_of_polynomial_l54_5473

theorem perfect_square_of_polynomial (k : ℝ) (h : ∃ (p : ℝ), ∀ x : ℝ, x^2 + 6*x + k^2 = (x + p)^2) : k = 3 ∨ k = -3 := 
sorry

end NUMINAMATH_GPT_perfect_square_of_polynomial_l54_5473


namespace NUMINAMATH_GPT_num_ways_distinct_letters_l54_5420

def letters : List String := ["A₁", "A₂", "A₃", "N₁", "N₂", "N₃", "B₁", "B₂"]

theorem num_ways_distinct_letters : (letters.permutations.length = 40320) := by
  sorry

end NUMINAMATH_GPT_num_ways_distinct_letters_l54_5420


namespace NUMINAMATH_GPT_sequence_1234_to_500_not_divisible_by_9_l54_5472

-- Definition for the sum of the digits of concatenated sequence
def sum_of_digits (n : ℕ) : ℕ :=
  -- This is a placeholder for the actual function calculating the sum of digits
  -- of all numbers from 1 to n concatenated together.
  sorry 

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem sequence_1234_to_500_not_divisible_by_9 : ¬ is_divisible_by_9 (sum_of_digits 500) :=
by
  -- Placeholder indicating the solution facts and methods should go here.
  sorry

end NUMINAMATH_GPT_sequence_1234_to_500_not_divisible_by_9_l54_5472


namespace NUMINAMATH_GPT_option_c_correct_l54_5449

theorem option_c_correct (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end NUMINAMATH_GPT_option_c_correct_l54_5449


namespace NUMINAMATH_GPT_g_func_eq_l54_5414

theorem g_func_eq (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → g (x / y) = y * g x)
  (h2 : g 50 = 10) :
  g 25 = 20 :=
sorry

end NUMINAMATH_GPT_g_func_eq_l54_5414


namespace NUMINAMATH_GPT_susan_correct_question_percentage_l54_5423

theorem susan_correct_question_percentage (y : ℕ) : 
  (75 * (2 * y - 1) / y) = 
  ((6 * y - 3) / (8 * y) * 100)  :=
sorry

end NUMINAMATH_GPT_susan_correct_question_percentage_l54_5423


namespace NUMINAMATH_GPT_minimize_travel_expense_l54_5492

noncomputable def travel_cost_A (x : ℕ) : ℝ := 2000 * x * 0.75
noncomputable def travel_cost_B (x : ℕ) : ℝ := 2000 * (x - 1) * 0.8

theorem minimize_travel_expense (x : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x ≤ 15 → travel_cost_B x < travel_cost_A x) ∧
  (x = 16 → travel_cost_A x = travel_cost_B x) ∧
  (17 ≤ x ∧ x ≤ 25 → travel_cost_A x < travel_cost_B x) :=
by
  sorry

end NUMINAMATH_GPT_minimize_travel_expense_l54_5492


namespace NUMINAMATH_GPT_complex_fraction_value_l54_5441

theorem complex_fraction_value :
  (Complex.mk 1 2) * (Complex.mk 1 2) / Complex.mk 3 (-4) = -1 :=
by
  -- Here we would provide the proof, but as per instructions,
  -- we will insert sorry to skip it.
  sorry

end NUMINAMATH_GPT_complex_fraction_value_l54_5441


namespace NUMINAMATH_GPT_parallel_conditions_l54_5460

-- Definitions of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y + 2 = 0

-- Definition of parallel lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → l2 x y

-- Proof statement
theorem parallel_conditions (m : ℝ) :
  parallel (l1 m) (l2 m) ↔ (m = 1 ∨ m = -6) :=
by
  intros
  sorry

end NUMINAMATH_GPT_parallel_conditions_l54_5460


namespace NUMINAMATH_GPT_correct_propositions_l54_5475

variable (P1 P2 P3 P4 : Prop)

-- Proposition 1: The negation of ∀ x ∈ ℝ, cos(x) > 0 is ∃ x ∈ ℝ such that cos(x) ≤ 0. 
def prop1 : Prop := 
  (¬ (∀ x : ℝ, Real.cos x > 0)) ↔ (∃ x : ℝ, Real.cos x ≤ 0)

-- Proposition 2: If 0 < a < 1, then the equation x^2 + a^x - 3 = 0 has only one real root.
def prop2 : Prop := 
  ∀ a : ℝ, (0 < a ∧ a < 1) → (∃! x : ℝ, x^2 + a^x - 3 = 0)

-- Proposition 3: For any real number x, if f(-x) = f(x) and f'(x) > 0 when x > 0, then f'(x) < 0 when x < 0.
def prop3 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x > 0 → deriv f x > 0) →
  (∀ x : ℝ, x < 0 → deriv f x < 0)

-- Proposition 4: For a rectangle with area S and perimeter l, the pair of real numbers (6, 8) is a valid (S, l) pair.
def prop4 : Prop :=
  ∃ (a b : ℝ), (a * b = 6) ∧ (2 * (a + b) = 8)

theorem correct_propositions (P1_def : prop1)
                            (P3_def : ∀ f : ℝ → ℝ, prop3 f) :
                          P1 ∧ P3 :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l54_5475


namespace NUMINAMATH_GPT_total_num_animals_l54_5491

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end NUMINAMATH_GPT_total_num_animals_l54_5491


namespace NUMINAMATH_GPT_gifted_subscribers_l54_5416

theorem gifted_subscribers (initial_subs : ℕ) (revenue_per_sub : ℕ) (total_revenue : ℕ) (h1 : initial_subs = 150) (h2 : revenue_per_sub = 9) (h3 : total_revenue = 1800) :
  total_revenue / revenue_per_sub - initial_subs = 50 :=
by
  sorry

end NUMINAMATH_GPT_gifted_subscribers_l54_5416


namespace NUMINAMATH_GPT_diagonal_of_square_l54_5459

theorem diagonal_of_square (d : ℝ) (s : ℝ) (h : d = 2) (h_eq : s * Real.sqrt 2 = d) : s = Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_diagonal_of_square_l54_5459


namespace NUMINAMATH_GPT_selecting_female_probability_l54_5447

theorem selecting_female_probability (female male : ℕ) (total : ℕ)
  (h_female : female = 4)
  (h_male : male = 6)
  (h_total : total = female + male) :
  (female / total : ℚ) = 2 / 5 := 
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_selecting_female_probability_l54_5447


namespace NUMINAMATH_GPT_candies_count_l54_5428

theorem candies_count :
  ∃ n, (n = 35 ∧ ∃ x, x ≥ 11 ∧ n = 3 * (x - 1) + 2) ∧ ∃ y, y ≤ 9 ∧ n = 4 * (y - 1) + 3 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_candies_count_l54_5428


namespace NUMINAMATH_GPT_smallest_base_10_integer_l54_5462

theorem smallest_base_10_integer :
  ∃ (c d : ℕ), 3 < c ∧ 3 < d ∧ (3 * c + 4 = 4 * d + 3) ∧ (3 * c + 4 = 19) :=
by {
 sorry
}

end NUMINAMATH_GPT_smallest_base_10_integer_l54_5462


namespace NUMINAMATH_GPT_correctly_calculated_value_l54_5494

theorem correctly_calculated_value :
  ∀ (x : ℕ), (x * 15 = 45) → ((x * 5) * 10 = 150) := 
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_correctly_calculated_value_l54_5494


namespace NUMINAMATH_GPT_maximum_overtakes_l54_5457

-- Definitions based on problem conditions
structure Team where
  members : List ℕ
  speed_const : ℕ → ℝ -- Speed of each member is constant but different
  run_segment : ℕ → ℕ -- Each member runs exactly one segment
  
def relay_race_condition (team1 team2 : Team) : Prop :=
  team1.members.length = 20 ∧
  team2.members.length = 20 ∧
  ∀ i, (team1.speed_const i ≠ team2.speed_const i)

def transitions (team : Team) : ℕ :=
  team.members.length - 1

-- The theorem to be proved
theorem maximum_overtakes (team1 team2 : Team) (hcond : relay_race_condition team1 team2) : 
  ∃ n, n = 38 :=
by
  sorry

end NUMINAMATH_GPT_maximum_overtakes_l54_5457


namespace NUMINAMATH_GPT_natural_numbers_divisors_l54_5434

theorem natural_numbers_divisors (n : ℕ) : 
  n + 1 ∣ n^2 + 1 → n = 0 ∨ n = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_natural_numbers_divisors_l54_5434


namespace NUMINAMATH_GPT_diameter_circle_C_inscribed_within_D_l54_5465

noncomputable def circle_diameter_C (d_D : ℝ) (ratio : ℝ) : ℝ :=
  let R := d_D / 2
  let r := (R : ℝ) / (Real.sqrt 5)
  2 * r

theorem diameter_circle_C_inscribed_within_D 
  (d_D : ℝ) (ratio : ℝ) (h_dD_pos : 0 < d_D) (h_ratio : ratio = 4)
  (h_dD : d_D = 24) : 
  circle_diameter_C d_D ratio = 24 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_diameter_circle_C_inscribed_within_D_l54_5465


namespace NUMINAMATH_GPT_no_fixed_point_implies_no_double_fixed_point_l54_5439

theorem no_fixed_point_implies_no_double_fixed_point (f : ℝ → ℝ) 
  (hf : Continuous f)
  (h : ∀ x : ℝ, f x ≠ x) :
  ∀ x : ℝ, f (f x) ≠ x :=
sorry

end NUMINAMATH_GPT_no_fixed_point_implies_no_double_fixed_point_l54_5439


namespace NUMINAMATH_GPT_height_of_shorter_tree_l54_5440

theorem height_of_shorter_tree (H h : ℝ) (h_difference : H = h + 20) (ratio : h / H = 5 / 7) : h = 50 := 
by
  sorry

end NUMINAMATH_GPT_height_of_shorter_tree_l54_5440


namespace NUMINAMATH_GPT_triangle_area_l54_5470

theorem triangle_area :
  ∃ (A : ℝ),
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a = 65 ∧ b = 60 ∧ c = 25 ∧ s = 75 ∧  area = 750 :=
by
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  use Real.sqrt (s * (s - a) * (s - b) * (s - c))
  -- We would prove the conditions and calculations here, but we skip the proof parts
  sorry

end NUMINAMATH_GPT_triangle_area_l54_5470


namespace NUMINAMATH_GPT_N_is_perfect_square_l54_5413

def N (n : ℕ) : ℕ :=
  (10^(2*n+1) - 1) / 9 * 10 + 
  2 * (10^(n+1) - 1) / 9 + 25

theorem N_is_perfect_square (n : ℕ) : ∃ k, k^2 = N n :=
  sorry

end NUMINAMATH_GPT_N_is_perfect_square_l54_5413


namespace NUMINAMATH_GPT_lobster_distribution_l54_5482

theorem lobster_distribution :
  let HarborA := 50
  let HarborB := 70.5
  let HarborC := (2 / 3) * HarborB
  let HarborD := HarborA - 0.15 * HarborA
  let Sum := HarborA + HarborB + HarborC + HarborD
  let HooperBay := 3 * Sum
  let Total := HooperBay + Sum
  Total = 840 := by
  sorry

end NUMINAMATH_GPT_lobster_distribution_l54_5482


namespace NUMINAMATH_GPT_extremum_condition_l54_5424

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * a * x^2 + 1 = 0

theorem extremum_condition (a : ℝ) : has_extremum a ↔ a < 0 := 
  sorry

end NUMINAMATH_GPT_extremum_condition_l54_5424


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l54_5403

-- Option A
theorem option_A (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 
  (x-1)^2 + x*(x-4) + (x-2)*(x+2) ≠ 0 := 
sorry

-- Option B
theorem option_B (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^3 + (1/x)^3 - 3 = 15 := 
sorry

-- Option C
theorem option_C (x : ℝ) (a b c : ℝ) (h_a : a = 1 / 20 * x + 20) (h_b : b = 1 / 20 * x + 19) (h_c : c = 1 / 20 * x + 21) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := 
sorry

-- Option D
theorem option_D (x m n : ℝ) (h : 2*x^2 - 8*x + 7 = 0) (h_roots : m + n = 4 ∧ m * n = 7/2) : 
  Real.sqrt (m^2 + n^2) = 3 := 
sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l54_5403


namespace NUMINAMATH_GPT_max_sum_of_factors_of_48_l54_5461

theorem max_sum_of_factors_of_48 (d Δ : ℕ) (h : d * Δ = 48) : d + Δ ≤ 49 :=
sorry

end NUMINAMATH_GPT_max_sum_of_factors_of_48_l54_5461


namespace NUMINAMATH_GPT_number_of_k_for_lcm_l54_5400

theorem number_of_k_for_lcm (a b : ℕ) :
  (∀ a b, k = 2^a * 3^b) → 
  (∀ (a : ℕ), 0 ≤ a ∧ a ≤ 24) →
  (∃ b, b = 12) →
  (∀ k, k = 2^a * 3^b) →
  (Nat.lcm (Nat.lcm (6^6) (8^8)) k = 12^12) :=
sorry

end NUMINAMATH_GPT_number_of_k_for_lcm_l54_5400


namespace NUMINAMATH_GPT_jason_current_cards_l54_5497

-- Define the initial number of Pokemon cards Jason had.
def initial_cards : ℕ := 9

-- Define the number of Pokemon cards Jason gave to his friends.
def given_away : ℕ := 4

-- Prove that the number of Pokemon cards he has now is 5.
theorem jason_current_cards : initial_cards - given_away = 5 := by
  sorry

end NUMINAMATH_GPT_jason_current_cards_l54_5497


namespace NUMINAMATH_GPT_percentage_of_ore_contains_alloy_l54_5436

def ore_contains_alloy_iron (weight_ore weight_iron : ℝ) (P : ℝ) : Prop :=
  (P / 100 * weight_ore) * 0.9 = weight_iron

theorem percentage_of_ore_contains_alloy (w_ore : ℝ) (w_iron : ℝ) (P : ℝ) 
    (h_w_ore : w_ore = 266.6666666666667) (h_w_iron : w_iron = 60) 
    (h_ore_contains : ore_contains_alloy_iron w_ore w_iron P) 
    : P = 25 :=
by
  rw [h_w_ore, h_w_iron] at h_ore_contains
  sorry

end NUMINAMATH_GPT_percentage_of_ore_contains_alloy_l54_5436


namespace NUMINAMATH_GPT_abc_divides_sum_exp21_l54_5445

theorem abc_divides_sum_exp21
  (a b c : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ∣ b^4)
  (hbc : b ∣ c^4)
  (hca : c ∣ a^4)
  : abc ∣ (a + b + c)^21 :=
by
sorry

end NUMINAMATH_GPT_abc_divides_sum_exp21_l54_5445


namespace NUMINAMATH_GPT_ordered_pairs_sum_reciprocal_l54_5421

theorem ordered_pairs_sum_reciprocal (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (1 / a + 1 / b : ℚ) = 1 / 6) → ∃ n : ℕ, n = 9 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_sum_reciprocal_l54_5421


namespace NUMINAMATH_GPT_find_a_value_l54_5443

theorem find_a_value (a x : ℝ) (h1 : 6 * (x + 8) = 18 * x) (h2 : 6 * x - 2 * (a - x) = 2 * a + x) : a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l54_5443


namespace NUMINAMATH_GPT_watermelon_percentage_l54_5481

theorem watermelon_percentage (total_drink : ℕ)
  (orange_percentage : ℕ)
  (grape_juice : ℕ)
  (watermelon_amount : ℕ)
  (W : ℕ) :
  total_drink = 300 →
  orange_percentage = 25 →
  grape_juice = 105 →
  watermelon_amount = total_drink - (orange_percentage * total_drink) / 100 - grape_juice →
  W = (watermelon_amount * 100) / total_drink →
  W = 40 :=
sorry

end NUMINAMATH_GPT_watermelon_percentage_l54_5481


namespace NUMINAMATH_GPT_savannah_rolls_l54_5433

-- Definitions and conditions
def total_gifts := 12
def gifts_per_roll_1 := 3
def gifts_per_roll_2 := 5
def gifts_per_roll_3 := 4

-- Prove the number of rolls
theorem savannah_rolls :
  gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3 = total_gifts →
  3 + 5 + 4 = 12 →
  3 = total_gifts / (gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_savannah_rolls_l54_5433


namespace NUMINAMATH_GPT_rectangle_area_ratio_l54_5408

theorem rectangle_area_ratio (s x y : ℝ) (h_square : s > 0)
    (h_side_ae : x > 0) (h_side_ag : y > 0)
    (h_ratio_area : x * y = (1 / 4) * s^2) :
    ∃ (r : ℝ), r > 0 ∧ r = x / y := 
sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l54_5408


namespace NUMINAMATH_GPT_minimum_value_ineq_l54_5463

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  (1 : ℝ) ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_ineq_l54_5463


namespace NUMINAMATH_GPT_strategy2_is_better_final_cost_strategy2_correct_l54_5417

def initial_cost : ℝ := 12000

def strategy1_discount : ℝ := 
  let after_first_discount := initial_cost * 0.70
  let after_second_discount := after_first_discount * 0.85
  let after_third_discount := after_second_discount * 0.95
  after_third_discount

def strategy2_discount : ℝ := 
  let after_first_discount := initial_cost * 0.55
  let after_second_discount := after_first_discount * 0.90
  let after_third_discount := after_second_discount * 0.90
  let final_cost := after_third_discount + 150
  final_cost

theorem strategy2_is_better : strategy2_discount < strategy1_discount :=
by {
  sorry -- proof goes here
}

theorem final_cost_strategy2_correct : strategy2_discount = 5496 :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_strategy2_is_better_final_cost_strategy2_correct_l54_5417


namespace NUMINAMATH_GPT_range_of_2m_plus_n_l54_5415

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x / Real.log 3)

theorem range_of_2m_plus_n {m n : ℝ} (hmn : 0 < m ∧ m < n) (heq : f m = f n) :
  ∃ y, y ∈ Set.Ici (2 * Real.sqrt 2) ∧ (2 * m + n = y) :=
sorry

end NUMINAMATH_GPT_range_of_2m_plus_n_l54_5415


namespace NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l54_5405

theorem least_integer_greater_than_sqrt_500 (x: ℕ) (h1: 22^2 = 484) (h2: 23^2 = 529) (h3: 484 < 500 ∧ 500 < 529) : x = 23 :=
  sorry

end NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l54_5405


namespace NUMINAMATH_GPT_problem1_problem2_l54_5451

theorem problem1 (x : ℝ) (h1 : x * (x + 4) = -5 * (x + 4)) : x = -4 ∨ x = -5 := 
by 
  sorry

theorem problem2 (x : ℝ) (h2 : (x + 2) ^ 2 = (2 * x - 1) ^ 2) : x = 3 ∨ x = -1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l54_5451


namespace NUMINAMATH_GPT_intersection_x_axis_l54_5464

theorem intersection_x_axis (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, 3)) (h2 : (x2, y2) = (3, -1)) :
  ∃ x : ℝ, (x, 0) = (4, 0) :=
by sorry

end NUMINAMATH_GPT_intersection_x_axis_l54_5464


namespace NUMINAMATH_GPT_percentage_of_non_defective_products_l54_5487

-- Define the conditions
def totalProduction : ℕ := 100
def M1_production : ℕ := 25
def M2_production : ℕ := 35
def M3_production : ℕ := 40

def M1_defective_rate : ℝ := 0.02
def M2_defective_rate : ℝ := 0.04
def M3_defective_rate : ℝ := 0.05

-- Calculate the total defective units
noncomputable def total_defective_units : ℝ := 
  (M1_defective_rate * M1_production) + 
  (M2_defective_rate * M2_production) + 
  (M3_defective_rate * M3_production)

-- Calculate the percentage of defective products
noncomputable def defective_percentage : ℝ := (total_defective_units / totalProduction) * 100

-- Calculate the percentage of non-defective products
noncomputable def non_defective_percentage : ℝ := 100 - defective_percentage

-- The statement to prove
theorem percentage_of_non_defective_products :
  non_defective_percentage = 96.1 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_non_defective_products_l54_5487


namespace NUMINAMATH_GPT_total_cows_l54_5478

/-- A farmer divides his herd of cows among his four sons.
The first son receives 1/3 of the herd, the second son receives 1/6,
the third son receives 1/9, and the rest goes to the fourth son,
who receives 12 cows. Calculate the total number of cows in the herd
-/
theorem total_cows (n : ℕ) (h1 : (n : ℚ) * (1 / 3) + (n : ℚ) * (1 / 6) + (n : ℚ) * (1 / 9) + 12 = n) : n = 54 := by
  sorry

end NUMINAMATH_GPT_total_cows_l54_5478


namespace NUMINAMATH_GPT_set_complement_union_l54_5458

namespace ProblemOne

def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem set_complement_union :
  (Aᶜ ∪ B) = {x : ℝ | -3 < x ∧ x < 5} := sorry

end ProblemOne

end NUMINAMATH_GPT_set_complement_union_l54_5458


namespace NUMINAMATH_GPT_unit_circle_arc_length_l54_5471

theorem unit_circle_arc_length (r : ℝ) (A : ℝ) (θ : ℝ) : r = 1 ∧ A = 1 ∧ A = (1 / 2) * r^2 * θ → r * θ = 2 :=
by
  -- Given r = 1 (radius of unit circle) and area A = 1
  -- A = (1 / 2) * r^2 * θ is the formula for the area of the sector
  sorry

end NUMINAMATH_GPT_unit_circle_arc_length_l54_5471


namespace NUMINAMATH_GPT_solve_system_l54_5427

section system_equations

variable (x y : ℤ)

def equation1 := 2 * x - y = 5
def equation2 := 5 * x + 2 * y = 8
def solution := x = 2 ∧ y = -1

theorem solve_system : (equation1 x y) ∧ (equation2 x y) ↔ solution x y := by
  sorry

end system_equations

end NUMINAMATH_GPT_solve_system_l54_5427


namespace NUMINAMATH_GPT_acid_volume_16_liters_l54_5406

theorem acid_volume_16_liters (V A_0 B_0 A_1 B_1 : ℝ) 
  (h_initial_ratio : 4 * B_0 = A_0)
  (h_initial_volume : A_0 + B_0 = V)
  (h_remove_mixture : 10 * A_0 / V = A_1)
  (h_remove_mixture_base : 10 * B_0 / V = B_1)
  (h_new_A : A_1 = A_0 - 8)
  (h_new_B : B_1 = B_0 - 2 + 10)
  (h_new_ratio : 2 * B_1 = 3 * A_1) :
  A_0 = 16 :=
by {
  -- Here we will have the proof steps, which are omitted.
  sorry
}

end NUMINAMATH_GPT_acid_volume_16_liters_l54_5406


namespace NUMINAMATH_GPT_compute_expression_l54_5410

theorem compute_expression : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l54_5410


namespace NUMINAMATH_GPT_leak_empty_time_l54_5425

theorem leak_empty_time :
  let A := (1:ℝ)/6
  let AL := A - L
  ∀ L: ℝ, (A - L = (1:ℝ)/8) → (1 / L = 24) :=
by
  intros A AL L h
  sorry

end NUMINAMATH_GPT_leak_empty_time_l54_5425


namespace NUMINAMATH_GPT_part1_part2_l54_5456

theorem part1 (a b c C : ℝ) (h : b - 1/2 * c = a * Real.cos C) (h1 : ∃ (A B : ℝ), Real.sin B - 1/2 * Real.sin C = Real.sin A * Real.cos C) :
  ∃ A : ℝ, A = 60 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 4 * (b + c) = 3 * b * c) (h2 : a = 2 * Real.sqrt 3) (h3 : b - 1/2 * c = a * Real.cos 60)
  (h4 : ∀ (A : ℝ), A = 60) : ∃ S : ℝ, S = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l54_5456


namespace NUMINAMATH_GPT_price_of_10_pound_bag_l54_5431

variables (P : ℝ) -- price of the 10-pound bag
def cost (n5 n10 n25 : ℕ) := n5 * 13.85 + n10 * P + n25 * 32.25

theorem price_of_10_pound_bag (h : ∃ (n5 n10 n25 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 ≥ 65
  ∧ n5 * 5 + n10 * 10 + n25 * 25 ≤ 80 
  ∧ cost P n5 n10 n25 = 98.77) : 
  P = 20.42 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_price_of_10_pound_bag_l54_5431


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l54_5448

theorem lcm_of_two_numbers (x y : ℕ) (h1 : Nat.gcd x y = 12) (h2 : x * y = 2460) : Nat.lcm x y = 205 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l54_5448


namespace NUMINAMATH_GPT_num_triangles_correct_num_lines_correct_l54_5444

-- Definition for the first proof problem: Number of triangles
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem num_triangles_correct :
  num_triangles 9 = 84 :=
by
  sorry

-- Definition for the second proof problem: Number of lines
def num_lines (n : ℕ) : ℕ := Nat.choose n 2

theorem num_lines_correct :
  num_lines 9 = 36 :=
by
  sorry

end NUMINAMATH_GPT_num_triangles_correct_num_lines_correct_l54_5444


namespace NUMINAMATH_GPT_domain_range_sum_l54_5407

theorem domain_range_sum (m n : ℝ) 
  (h1 : ∀ x, m ≤ x ∧ x ≤ n → 3 * m ≤ -x ^ 2 + 2 * x ∧ -x ^ 2 + 2 * x ≤ 3 * n)
  (h2 : -m ^ 2 + 2 * m = 3 * m)
  (h3 : -n ^ 2 + 2 * n = 3 * n) :
  m = -1 ∧ n = 0 ∧ m + n = -1 := 
by 
  sorry

end NUMINAMATH_GPT_domain_range_sum_l54_5407


namespace NUMINAMATH_GPT_quotient_ab_solution_l54_5432

noncomputable def a : Real := sorry
noncomputable def b : Real := sorry

def condition1 (a b : Real) : Prop :=
  (1/(3 * a) + 1/b = 2011)

def condition2 (a b : Real) : Prop :=
  (1/a + 1/(3 * b) = 1)

theorem quotient_ab_solution (a b : Real) 
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  (a + b) / (a * b) = 1509 :=
sorry

end NUMINAMATH_GPT_quotient_ab_solution_l54_5432


namespace NUMINAMATH_GPT_probability_abc_plus_ab_plus_a_divisible_by_4_l54_5455

noncomputable def count_multiples_of (n m : ℕ) : ℕ := (m / n)

noncomputable def probability_divisible_by_4 : ℚ := 
  let total_numbers := 2008
  let multiples_of_4 := count_multiples_of 4 total_numbers
  -- Probability that 'a' is divisible by 4
  let p_a := (multiples_of_4 : ℚ) / total_numbers
  -- Probability that 'a' is not divisible by 4
  let p_not_a := 1 - p_a
  -- Considering specific cases for b and c modulo 4
  let p_bc_cases := (2 * ((1 / 4) * (1 / 4)))  -- Probabilities for specific cases noted as 2 * (1/16)
  -- Adjusting probabilities for non-divisible 'a'
  let p_not_a_cases := p_bc_cases * p_not_a
  -- Total Probability
  p_a + p_not_a_cases

theorem probability_abc_plus_ab_plus_a_divisible_by_4 :
  probability_divisible_by_4 = 11 / 32 :=
sorry

end NUMINAMATH_GPT_probability_abc_plus_ab_plus_a_divisible_by_4_l54_5455


namespace NUMINAMATH_GPT_running_speed_equiv_l54_5484

variable (R : ℝ)
variable (walking_speed : ℝ) (total_distance : ℝ) (total_time: ℝ) (distance_walked : ℝ) (distance_ran : ℝ)

theorem running_speed_equiv :
  walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4 →
  1 + (4 / R) = 1.5 →
  R = 8 :=
by
  intros H1 H2
  -- H1: Condition set (walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4)
  -- H2: Equation (1 + (4 / R) = 1.5)
  sorry

end NUMINAMATH_GPT_running_speed_equiv_l54_5484


namespace NUMINAMATH_GPT_apps_added_eq_sixty_l54_5498

-- Definitions derived from the problem conditions
def initial_apps : ℕ := 50
def removed_apps : ℕ := 10
def final_apps : ℕ := 100

-- Intermediate calculation based on the problem
def apps_after_removal : ℕ := initial_apps - removed_apps

-- The main theorem stating the mathematically equivalent proof problem
theorem apps_added_eq_sixty : final_apps - apps_after_removal = 60 :=
by
  sorry

end NUMINAMATH_GPT_apps_added_eq_sixty_l54_5498


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l54_5437

noncomputable def a_n (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_problem :
  ∃ d : ℕ, a_n 1 2 d = 2 ∧ a_n 2 2 d + a_n 3 2 d = 13 ∧ (a_n 4 2 d + a_n 5 2 d + a_n 6 2 d = 42) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l54_5437


namespace NUMINAMATH_GPT_problem_statement_l54_5496

noncomputable def f (x : ℝ) := Real.log 9 * (Real.log x / Real.log 3)

theorem problem_statement : deriv f 2 + deriv f 2 = 1 := sorry

end NUMINAMATH_GPT_problem_statement_l54_5496


namespace NUMINAMATH_GPT_reciprocal_of_neg3_l54_5499

theorem reciprocal_of_neg3 : 1 / (-3: ℝ) = -1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg3_l54_5499


namespace NUMINAMATH_GPT_product_of_digits_in_base7_7891_is_zero_l54_5442

/-- The function to compute the base 7 representation. -/
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else 
    let rest := to_base7 (n / 7)
    rest ++ [n % 7]

/-- The function to compute the product of the digits of a list. -/
def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * d) 1

theorem product_of_digits_in_base7_7891_is_zero :
  product_of_digits (to_base7 7891) = 0 := by
  sorry

end NUMINAMATH_GPT_product_of_digits_in_base7_7891_is_zero_l54_5442


namespace NUMINAMATH_GPT_terminating_decimal_representation_l54_5409

theorem terminating_decimal_representation : 
  (67 / (2^3 * 5^4) : ℝ) = 0.0134 :=
    sorry

end NUMINAMATH_GPT_terminating_decimal_representation_l54_5409


namespace NUMINAMATH_GPT_scientific_notation_correct_l54_5429

def num_people : ℝ := 2580000
def scientific_notation_form : ℝ := 2.58 * 10^6

theorem scientific_notation_correct : num_people = scientific_notation_form :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l54_5429


namespace NUMINAMATH_GPT_unique_p_value_l54_5452

theorem unique_p_value (p : Nat) (h₁ : Nat.Prime (p+10)) (h₂ : Nat.Prime (p+14)) : p = 3 := by
  sorry

end NUMINAMATH_GPT_unique_p_value_l54_5452


namespace NUMINAMATH_GPT_division_multiplication_eval_l54_5486

theorem division_multiplication_eval : (18 / (5 + 2 - 3)) * 4 = 18 := 
by
  sorry

end NUMINAMATH_GPT_division_multiplication_eval_l54_5486


namespace NUMINAMATH_GPT_total_hoodies_l54_5469

def Fiona_hoodies : ℕ := 3
def Casey_hoodies : ℕ := Fiona_hoodies + 2

theorem total_hoodies : (Fiona_hoodies + Casey_hoodies) = 8 := by
  sorry

end NUMINAMATH_GPT_total_hoodies_l54_5469


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l54_5468

theorem equilateral_triangle_side_length (c : ℕ) (h : c = 4 * 21) : c / 3 = 28 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l54_5468


namespace NUMINAMATH_GPT_maximize_profit_l54_5476

noncomputable def profit_function (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 40 then
  -2 * x^2 + 120 * x - 300
else if 40 < x ∧ x ≤ 100 then
  -x - 3600 / x + 1800
else
  0

theorem maximize_profit :
  profit_function 60 = 1680 ∧
  ∀ x, 0 < x ∧ x ≤ 100 → profit_function x ≤ 1680 := 
sorry

end NUMINAMATH_GPT_maximize_profit_l54_5476


namespace NUMINAMATH_GPT_evaluate_expression_l54_5404

theorem evaluate_expression :
  54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l54_5404


namespace NUMINAMATH_GPT_university_diploma_percentage_l54_5401

theorem university_diploma_percentage
  (A : ℝ) (B : ℝ) (C : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.10)
  (hC : C = 0.15) :
  A - B + C * (1 - A) = 0.39 := 
sorry

end NUMINAMATH_GPT_university_diploma_percentage_l54_5401


namespace NUMINAMATH_GPT_number_of_girls_l54_5453

theorem number_of_girls (B G : ℕ) (h1 : B + G = 400) 
  (h2 : 0.60 * B = (6 / 10 : ℝ) * B) 
  (h3 : 0.80 * G = (8 / 10 : ℝ) * G) 
  (h4 : (6 / 10 : ℝ) * B + (8 / 10 : ℝ) * G = (65 / 100 : ℝ) * 400) : G = 100 := by
sorry

end NUMINAMATH_GPT_number_of_girls_l54_5453


namespace NUMINAMATH_GPT_no_common_points_range_a_l54_5411

theorem no_common_points_range_a (a k : ℝ) (hl : ∃ k, ∀ x y : ℝ, k * x - y - k + 2 = 0) :
  (∀ x y : ℝ, x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) → (-7 < a ∧ a < -2) ∨ (1 < a) := by
  sorry

end NUMINAMATH_GPT_no_common_points_range_a_l54_5411


namespace NUMINAMATH_GPT_smallest_possible_denominator_l54_5489

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_denominator_l54_5489


namespace NUMINAMATH_GPT_Martha_improvement_in_lap_time_l54_5430

theorem Martha_improvement_in_lap_time 
  (initial_laps : ℕ) (initial_time : ℕ) 
  (first_month_laps : ℕ) (first_month_time : ℕ) 
  (second_month_laps : ℕ) (second_month_time : ℕ)
  (sec_per_min : ℕ)
  (conds : initial_laps = 15 ∧ initial_time = 30 ∧ first_month_laps = 18 ∧ first_month_time = 27 ∧ 
           second_month_laps = 20 ∧ second_month_time = 27 ∧ sec_per_min = 60)
  : ((initial_time / initial_laps : ℚ) - (second_month_time / second_month_laps)) * sec_per_min = 39 :=
by
  sorry

end NUMINAMATH_GPT_Martha_improvement_in_lap_time_l54_5430


namespace NUMINAMATH_GPT_domain_of_c_is_all_reals_l54_5412

theorem domain_of_c_is_all_reals (k : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_domain_of_c_is_all_reals_l54_5412


namespace NUMINAMATH_GPT_correct_average_and_variance_l54_5426

theorem correct_average_and_variance
  (n : ℕ) (avg incorrect_variance correct_variance : ℝ)
  (incorrect_score1 actual_score1 incorrect_score2 actual_score2 : ℝ)
  (H1 : n = 48)
  (H2 : avg = 70)
  (H3 : incorrect_variance = 75)
  (H4 : incorrect_score1 = 50)
  (H5 : actual_score1 = 80)
  (H6 : incorrect_score2 = 100)
  (H7 : actual_score2 = 70)
  (Havg : avg = (n * avg - incorrect_score1 - incorrect_score2 + actual_score1 + actual_score2) / n)
  (Hvar : correct_variance = incorrect_variance + (actual_score1 - avg) ^ 2 + (actual_score2 - avg) ^ 2
                     - (incorrect_score1 - avg) ^ 2 - (incorrect_score2 - avg) ^ 2 / n) :
  avg = 70 ∧ correct_variance = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_average_and_variance_l54_5426


namespace NUMINAMATH_GPT_liters_pepsi_144_l54_5438

/-- A drink vendor has 50 liters of Maaza, some liters of Pepsi, and 368 liters of Sprite. -/
def liters_maaza : ℕ := 50
def liters_sprite : ℕ := 368
def num_cans : ℕ := 281

/-- The total number of liters of drinks the vendor has -/
def total_liters (lit_pepsi: ℕ) : ℕ := liters_maaza + lit_pepsi + liters_sprite

/-- Given that the least number of cans required is 281, prove that the liters of Pepsi is 144. -/
theorem liters_pepsi_144 (P : ℕ) (h: total_liters P % num_cans = 0) : P = 144 :=
by
  sorry

end NUMINAMATH_GPT_liters_pepsi_144_l54_5438


namespace NUMINAMATH_GPT_distribute_pencils_l54_5483

def number_of_ways_to_distribute_pencils (pencils friends : ℕ) : ℕ :=
  Nat.choose (pencils - friends + friends - 1) (friends - 1)

theorem distribute_pencils :
  number_of_ways_to_distribute_pencils 4 4 = 35 :=
by
  sorry

end NUMINAMATH_GPT_distribute_pencils_l54_5483


namespace NUMINAMATH_GPT_stratified_sampling_probability_l54_5490

open Finset Nat

noncomputable def combin (n k : ℕ) : ℕ := choose n k

theorem stratified_sampling_probability :
  let total_balls := 40
  let red_balls := 16
  let blue_balls := 12
  let white_balls := 8
  let yellow_balls := 4
  let n_draw := 10
  let red_draw := 4
  let blue_draw := 3
  let white_draw := 2
  let yellow_draw := 1
  
  combin yellow_balls yellow_draw * combin white_balls white_draw * combin blue_balls blue_draw * combin red_balls red_draw = combin total_balls n_draw :=
sorry

end NUMINAMATH_GPT_stratified_sampling_probability_l54_5490


namespace NUMINAMATH_GPT_min_n_for_binomial_constant_term_l54_5485

theorem min_n_for_binomial_constant_term : ∃ (n : ℕ), n > 0 ∧ 3 * n - 7 * ((3 * n) / 7) = 0 ∧ n = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_n_for_binomial_constant_term_l54_5485


namespace NUMINAMATH_GPT_warriors_can_defeat_dragon_l54_5493

theorem warriors_can_defeat_dragon (n : ℕ) (h : n = 20^20) :
  (∀ n, n % 2 = 0 ∨ n % 3 = 0) → (∃ m, m = 0) := 
sorry

end NUMINAMATH_GPT_warriors_can_defeat_dragon_l54_5493


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l54_5422

theorem smallest_whole_number_larger_than_perimeter (c : ℝ) (h1 : 13 < c) (h2 : c < 25) : 50 = Nat.ceil (6 + 19 + c) :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l54_5422


namespace NUMINAMATH_GPT_f_injective_on_restricted_domain_l54_5479

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- Define the restricted domain
def f_restricted (x : ℝ) (h : -2 <= x) : ℝ := f x

-- The main statement to be proved
theorem f_injective_on_restricted_domain : 
  (∀ x1 x2 : {x // -2 <= x}, f_restricted x1.val x1.property = f_restricted x2.val x2.property → x1 = x2) := 
sorry

end NUMINAMATH_GPT_f_injective_on_restricted_domain_l54_5479
