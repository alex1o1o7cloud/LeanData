import Mathlib

namespace lowest_test_score_dropped_l1705_170586

theorem lowest_test_score_dropped (S L : ℕ)
  (h1 : S = 5 * 42) 
  (h2 : S - L = 4 * 48) : 
  L = 18 :=
by
  sorry

end lowest_test_score_dropped_l1705_170586


namespace max_value_l1705_170594

-- Definitions for the given conditions
def point_A := (3, 1)
def line_equation (m n : ℝ) := 3 * m + n + 1 = 0
def positive_product (m n : ℝ) := m * n > 0

-- The main statement to be proved
theorem max_value (m n : ℝ) (h1 : line_equation m n) (h2 : positive_product m n) : 
  (3 / m + 1 / n) ≤ -16 :=
sorry

end max_value_l1705_170594


namespace hyperbola_equation_through_point_l1705_170525

theorem hyperbola_equation_through_point
  (hyp_passes_through : ∀ (x y : ℝ), (x, y) = (1, 1) → ∃ (a b t : ℝ), (y^2 / a^2 - x^2 / b^2 = t))
  (asymptotes : ∀ (x y : ℝ), (y / x = Real.sqrt 2 ∨ y / x = -Real.sqrt 2) → ∃ (a b t : ℝ), (a = b * Real.sqrt 2)) :
  ∃ (a b t : ℝ), (2 * (1:ℝ)^2 - (1:ℝ)^2 = 1) :=
by
  sorry

end hyperbola_equation_through_point_l1705_170525


namespace remainder_is_nine_l1705_170514

-- Define the dividend and divisor
def n : ℕ := 4039
def d : ℕ := 31

-- Prove that n mod d equals 9
theorem remainder_is_nine : n % d = 9 := by
  sorry

end remainder_is_nine_l1705_170514


namespace workman_problem_l1705_170515

theorem workman_problem
    (total_work : ℝ)
    (B_rate : ℝ)
    (A_rate : ℝ)
    (days_together : ℝ)
    (W : total_work = 8 * (A_rate + B_rate))
    (A_2B : A_rate = 2 * B_rate) :
    total_work = 24 * B_rate :=
by
  sorry

end workman_problem_l1705_170515


namespace competition_sequences_l1705_170516

-- Define the problem conditions
def team_size : Nat := 7

-- Define the statement to prove
theorem competition_sequences :
  (Nat.choose (2 * team_size) team_size) = 3432 :=
by
  -- Proof will go here
  sorry

end competition_sequences_l1705_170516


namespace circle_equation_l1705_170523

theorem circle_equation (x y : ℝ) (h_eq : x = 0) (k_eq : y = -2) (r_eq : y = 4) :
  (x - 0)^2 + (y - (-2))^2 = 16 := 
by
  sorry

end circle_equation_l1705_170523


namespace ladybugs_total_total_ladybugs_is_5_l1705_170552

def num_ladybugs (x y : ℕ) : ℕ :=
  x + y

theorem ladybugs_total (x y n : ℕ) 
    (h_spot_calc_1: 6 * x + 4 * y = 30 ∨ 6 * x + 4 * y = 26)
    (h_total_spots_30: (6 * x + 4 * y = 30) ↔ 3 * x + 2 * y = 15)
    (h_total_spots_26: (6 * x + 4 * y = 26) ↔ 3 * x + 2 * y = 13)
    (h_truth_only_one: 
       (6 * x + 4 * y = 30 ∧ ¬(6 * x + 4 * y = 26)) ∨
       (¬(6 * x + 4 * y = 30) ∧ 6 * x + 4 * y = 26))
    : n = x + y :=
by 
  sorry

theorem total_ladybugs_is_5 : ∃ x y : ℕ, num_ladybugs x y = 5 :=
  ⟨3, 2, rfl⟩

end ladybugs_total_total_ladybugs_is_5_l1705_170552


namespace min_days_to_find_poisoned_apple_l1705_170535

theorem min_days_to_find_poisoned_apple (n : ℕ) (n_pos : 0 < n) : 
  ∀ k : ℕ, 2^k ≥ 2021 → k ≥ 11 :=
  sorry

end min_days_to_find_poisoned_apple_l1705_170535


namespace sum_first_20_terms_arithmetic_seq_l1705_170512

theorem sum_first_20_terms_arithmetic_seq :
  ∃ (a d : ℤ) (S_20 : ℤ), d > 0 ∧
  (a + 2 * d) * (a + 6 * d) = -12 ∧
  (a + 3 * d) + (a + 5 * d) = -4 ∧
  S_20 = 20 * a + (20 * 19 / 2) * d ∧
  S_20 = 180 :=
by
  sorry

end sum_first_20_terms_arithmetic_seq_l1705_170512


namespace range_of_m_l1705_170576

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 2^|x| + m = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l1705_170576


namespace positive_root_exists_iff_p_range_l1705_170531

theorem positive_root_exists_iff_p_range (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^4 + 4 * p * x^3 + x^2 + 4 * p * x + 4 = 0) ↔ 
  p ∈ Set.Iio (-Real.sqrt 2 / 2) ∪ Set.Ioi (Real.sqrt 2 / 2) :=
by
  sorry

end positive_root_exists_iff_p_range_l1705_170531


namespace sqrt_expression_is_869_l1705_170543

theorem sqrt_expression_is_869 :
  (31 * 30 * 29 * 28 + 1) = 869 := 
sorry

end sqrt_expression_is_869_l1705_170543


namespace total_balloons_l1705_170536

theorem total_balloons (F S M : ℕ) (hF : F = 5) (hS : S = 6) (hM : M = 7) : F + S + M = 18 :=
by 
  sorry

end total_balloons_l1705_170536


namespace lionsAfterOneYear_l1705_170505

-- Definitions based on problem conditions
def initialLions : Nat := 100
def birthRate : Nat := 5
def deathRate : Nat := 1
def monthsInYear : Nat := 12

-- Theorem statement
theorem lionsAfterOneYear :
  initialLions + birthRate * monthsInYear - deathRate * monthsInYear = 148 :=
by
  sorry

end lionsAfterOneYear_l1705_170505


namespace interest_rate_correct_l1705_170556

noncomputable def annual_interest_rate : ℝ :=
  4^(1/10) - 1

theorem interest_rate_correct (P A₁₀ A₁₅ : ℝ) (h₁ : P = 6000) (h₂ : A₁₀ = 24000) (h₃ : A₁₅ = 48000) :
  (P * (1 + annual_interest_rate)^10 = A₁₀) ∧ (P * (1 + annual_interest_rate)^15 = A₁₅) :=
by
  sorry

end interest_rate_correct_l1705_170556


namespace english_textbook_cost_l1705_170544

variable (cost_english_book : ℝ)

theorem english_textbook_cost :
  let geography_book_cost := 10.50
  let num_books := 35
  let total_order_cost := 630
  (num_books * cost_english_book + num_books * geography_book_cost = total_order_cost) →
  cost_english_book = 7.50 :=
by {
sorry
}

end english_textbook_cost_l1705_170544


namespace hall_area_l1705_170519

theorem hall_area (L W : ℝ) 
  (h1 : W = (1/2) * L)
  (h2 : L - W = 8) : 
  L * W = 128 := 
  sorry

end hall_area_l1705_170519


namespace quadratic_roots_equal_integral_l1705_170584

theorem quadratic_roots_equal_integral (c : ℝ) (h : (6^2 - 4 * 3 * c) = 0) : 
  ∃ x : ℝ, (3 * x^2 - 6 * x + c = 0) ∧ (x = 1) := 
by sorry

end quadratic_roots_equal_integral_l1705_170584


namespace sum_of_relatively_prime_integers_l1705_170567

theorem sum_of_relatively_prime_integers (n : ℕ) (h : n ≥ 7) :
  ∃ a b : ℕ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 :=
by
  sorry

end sum_of_relatively_prime_integers_l1705_170567


namespace youngest_child_age_l1705_170526

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) : 
  x = 4 := by
  sorry

end youngest_child_age_l1705_170526


namespace trigonometric_expression_result_l1705_170592

variable (α : ℝ)
variable (line_eq : ∀ x y : ℝ, 6 * x - 2 * y - 5 = 0)
variable (tan_alpha : Real.tan α = 3)

theorem trigonometric_expression_result :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 := 
by
  sorry

end trigonometric_expression_result_l1705_170592


namespace trapezoid_base_solutions_l1705_170577

theorem trapezoid_base_solutions (A h : ℕ) (d : ℕ) (bd : ℕ → Prop)
  (hA : A = 1800) (hH : h = 60) (hD : d = 10) (hBd : ∀ (x : ℕ), bd x ↔ ∃ (k : ℕ), x = d * k) :
  ∃ m n : ℕ, bd (10 * m) ∧ bd (10 * n) ∧ 10 * (m + n) = 60 ∧ m + n = 6 :=
by
  simp [hA, hH, hD, hBd]
  sorry

end trapezoid_base_solutions_l1705_170577


namespace polygon_sides_l1705_170510

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end polygon_sides_l1705_170510


namespace percent_problem_l1705_170561

variable (x : ℝ)

theorem percent_problem (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 :=
by sorry

end percent_problem_l1705_170561


namespace josh_and_fred_age_l1705_170500

theorem josh_and_fred_age
    (a b k : ℕ)
    (h1 : 10 * a + b > 10 * b + a)
    (h2 : 99 * (a^2 - b^2) = k^2)
    (ha : a ≥ 0 ∧ a ≤ 9)
    (hb : b ≥ 0 ∧ b ≤ 9) : 
    10 * a + b = 65 ∧ 
    10 * b + a = 56 := 
sorry

end josh_and_fred_age_l1705_170500


namespace sequence_finite_l1705_170548

def sequence_terminates (a_0 : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (a 0 = a_0) ∧ 
                  (∀ n, ((a n > 5) ∧ (a n % 10 ≤ 5) → a (n + 1) = a n / 10)) ∧
                  (∀ n, ((a n > 5) ∧ (a n % 10 > 5) → a (n + 1) = 9 * a n)) → 
                  ∃ n, a n ≤ 5 

theorem sequence_finite (a_0 : ℕ) : sequence_terminates a_0 :=
sorry

end sequence_finite_l1705_170548


namespace bell_pepper_slices_l1705_170590

theorem bell_pepper_slices :
  ∀ (num_peppers : ℕ) (slices_per_pepper : ℕ) (total_slices_pieces : ℕ) (half_slices : ℕ),
  num_peppers = 5 → slices_per_pepper = 20 → total_slices_pieces = 200 →
  half_slices = (num_peppers * slices_per_pepper) / 2 →
  (total_slices_pieces - (num_peppers * slices_per_pepper)) / half_slices = 2 :=
by
  intros num_peppers slices_per_pepper total_slices_pieces half_slices h1 h2 h3 h4
  -- skip the proof with sorry as instructed
  sorry

end bell_pepper_slices_l1705_170590


namespace number_of_perfect_squares_between_50_and_200_l1705_170565

theorem number_of_perfect_squares_between_50_and_200 :
  ∃ n: ℕ, 50 < n^2 ∧ n^2 < 200 ∧ (14 - 8 + 1 = 7) := sorry

end number_of_perfect_squares_between_50_and_200_l1705_170565


namespace martha_flower_cost_l1705_170507

theorem martha_flower_cost :
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  total_cost = 2700 :=
by
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  -- Proof to be added here
  sorry

end martha_flower_cost_l1705_170507


namespace negative_root_m_positive_l1705_170575

noncomputable def is_negative_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 + m * x - 4 = 0

theorem negative_root_m_positive : ∀ m : ℝ, is_negative_root m → m > 0 :=
by
  intro m
  intro h
  sorry

end negative_root_m_positive_l1705_170575


namespace robotics_club_neither_l1705_170506

theorem robotics_club_neither (total students programming electronics both: ℕ) 
  (h1: total = 120)
  (h2: programming = 80)
  (h3: electronics = 50)
  (h4: both = 15) : 
  total - ((programming - both) + (electronics - both) + both) = 5 :=
by
  sorry

end robotics_club_neither_l1705_170506


namespace women_with_fair_hair_percentage_l1705_170559

theorem women_with_fair_hair_percentage
  (A : ℝ) (B : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.25) :
  A * B = 0.10 := 
by
  rw [hA, hB]
  norm_num

end women_with_fair_hair_percentage_l1705_170559


namespace ratio_of_pants_to_shirts_l1705_170527

noncomputable def cost_shirt : ℝ := 6
noncomputable def cost_pants : ℝ := 8
noncomputable def num_shirts : ℝ := 10
noncomputable def total_cost : ℝ := 100

noncomputable def num_pants : ℝ :=
  (total_cost - (num_shirts * cost_shirt)) / cost_pants

theorem ratio_of_pants_to_shirts : num_pants / num_shirts = 1 / 2 := by
  sorry

end ratio_of_pants_to_shirts_l1705_170527


namespace national_park_sightings_l1705_170520

def january_sightings : ℕ := 26

def february_sightings : ℕ := 3 * january_sightings

def march_sightings : ℕ := february_sightings / 2

def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem national_park_sightings : total_sightings = 143 := by
  sorry

end national_park_sightings_l1705_170520


namespace distance_walked_is_18_miles_l1705_170566

-- Defining the variables for speed, time, and distance
variables (x t d : ℕ)

-- Declaring the conditions given in the problem
def walked_distance_at_usual_rate : Prop :=
  d = x * t

def walked_distance_at_increased_rate : Prop :=
  d = (x + 1) * (3 * t / 4)

def walked_distance_at_decreased_rate : Prop :=
  d = (x - 1) * (t + 3)

-- The proof problem statement to show the distance walked is 18 miles
theorem distance_walked_is_18_miles
  (hx : walked_distance_at_usual_rate x t d)
  (hz : walked_distance_at_increased_rate x t d)
  (hy : walked_distance_at_decreased_rate x t d) :
  d = 18 := by
  sorry

end distance_walked_is_18_miles_l1705_170566


namespace percentage_fertilizer_in_second_solution_l1705_170583

theorem percentage_fertilizer_in_second_solution 
    (v1 v2 v3 : ℝ) 
    (p1 p2 p3 : ℝ) 
    (h1 : v1 = 20) 
    (h2 : v2 + v1 = 42) 
    (h3 : p1 = 74 / 100) 
    (h4 : p2 = 63 / 100) 
    (h5 : v3 = (63 * 42 - 74 * 20) / 22) 
    : p3 = (53 / 100) :=
by
  sorry

end percentage_fertilizer_in_second_solution_l1705_170583


namespace smallest_positive_integer_problem_l1705_170564

theorem smallest_positive_integer_problem
  (n : ℕ) 
  (h1 : 50 ∣ n) 
  (h2 : (∃ e1 e2 e3 : ℕ, n = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100)) 
  (h3 : ∀ m : ℕ, (50 ∣ m) → ((∃ e1 e2 e3 : ℕ, m = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100) → (n ≤ m))) :
  n / 50 = 8100 := 
sorry

end smallest_positive_integer_problem_l1705_170564


namespace ways_to_divide_day_l1705_170595

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end ways_to_divide_day_l1705_170595


namespace track_length_l1705_170570

theorem track_length
  (meet1_dist : ℝ)
  (meet2_sally_additional_dist : ℝ)
  (constant_speed : ∀ (b_speed s_speed : ℝ), b_speed = s_speed)
  (opposite_start : true)
  (brenda_first_meet : meet1_dist = 100)
  (sally_second_meet : meet2_sally_additional_dist = 200) :
  ∃ L : ℝ, L = 200 :=
by
  sorry

end track_length_l1705_170570


namespace find_integer_l1705_170532

theorem find_integer (n : ℤ) (h1 : n ≥ 50) (h2 : n ≤ 100) (h3 : n % 7 = 0) (h4 : n % 9 = 3) (h5 : n % 6 = 3) : n = 84 := 
by 
  sorry

end find_integer_l1705_170532


namespace total_distance_walked_l1705_170582

-- Define the conditions
def home_to_school : ℕ := 750
def half_distance : ℕ := home_to_school / 2
def return_home : ℕ := half_distance
def home_to_school_again : ℕ := home_to_school

-- Define the theorem statement
theorem total_distance_walked : 
  half_distance + return_home + home_to_school_again = 1500 := by
  sorry

end total_distance_walked_l1705_170582


namespace hamburgers_leftover_l1705_170547

-- Define the number of hamburgers made and served
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove the number of leftover hamburgers
theorem hamburgers_leftover : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_leftover_l1705_170547


namespace james_total_socks_l1705_170509

-- Definitions based on conditions
def red_pairs : ℕ := 20
def black_pairs : ℕ := red_pairs / 2
def white_pairs : ℕ := 2 * (red_pairs + black_pairs)
def green_pairs : ℕ := (red_pairs + black_pairs + white_pairs) + 5

-- Total number of pairs
def total_pairs := red_pairs + black_pairs + white_pairs + green_pairs

-- Total number of socks
def total_socks := total_pairs * 2

-- The main theorem to prove the total number of socks
theorem james_total_socks : total_socks = 370 :=
  by
  -- proof is skipped
  sorry

end james_total_socks_l1705_170509


namespace paving_stone_size_l1705_170528

theorem paving_stone_size (length_courtyard width_courtyard : ℕ) (num_paving_stones : ℕ) (area_courtyard : ℕ) (s : ℕ)
  (h₁ : length_courtyard = 30) 
  (h₂ : width_courtyard = 18)
  (h₃ : num_paving_stones = 135)
  (h₄ : area_courtyard = length_courtyard * width_courtyard)
  (h₅ : area_courtyard = num_paving_stones * s * s) :
  s = 2 := 
by
  sorry

end paving_stone_size_l1705_170528


namespace find_first_number_l1705_170538

/-- The lcm of two numbers is 2310 and hcf (gcd) is 26. One of the numbers is 286. What is the other number? --/
theorem find_first_number (A : ℕ) 
  (h_lcm : Nat.lcm A 286 = 2310) 
  (h_gcd : Nat.gcd A 286 = 26) : 
  A = 210 := 
by
  sorry

end find_first_number_l1705_170538


namespace optimal_selling_price_maximizes_profit_l1705_170539

/-- The purchase price of a certain product is 40 yuan. -/
def cost_price : ℝ := 40

/-- At a selling price of 50 yuan, 50 units can be sold. -/
def initial_selling_price : ℝ := 50
def initial_quantity_sold : ℝ := 50

/-- If the selling price increases by 1 yuan, the sales volume decreases by 1 unit. -/
def price_increase_effect (x : ℝ) : ℝ := initial_selling_price + x
def quantity_decrease_effect (x : ℝ) : ℝ := initial_quantity_sold - x

/-- The revenue function. -/
def revenue (x : ℝ) : ℝ := (price_increase_effect x) * (quantity_decrease_effect x)

/-- The cost function. -/
def cost (x : ℝ) : ℝ := cost_price * (quantity_decrease_effect x)

/-- The profit function. -/
def profit (x : ℝ) : ℝ := revenue x - cost x

/-- The proof that the optimal selling price to maximize profit is 70 yuan. -/
theorem optimal_selling_price_maximizes_profit : price_increase_effect 20 = 70 :=
by
  sorry

end optimal_selling_price_maximizes_profit_l1705_170539


namespace evaluate_expression_l1705_170560

/-- Given conditions: -/
def a : ℕ := 3998
def b : ℕ := 3999

theorem evaluate_expression :
  b^3 - 2 * a * b^2 - 2 * a^2 * b + (b - 2)^3 = 95806315 :=
  sorry

end evaluate_expression_l1705_170560


namespace original_price_of_tshirt_l1705_170529

theorem original_price_of_tshirt :
  ∀ (P : ℝ), 
    (∀ discount quantity_sold revenue : ℝ, discount = 8 ∧ quantity_sold = 130 ∧ revenue = 5590 ∧
      revenue = quantity_sold * (P - discount)) → P = 51 := 
by
  intros P
  intro h
  sorry

end original_price_of_tshirt_l1705_170529


namespace max_value_of_y_l1705_170533

open Real

theorem max_value_of_y (x : ℝ) (h1 : 0 < x) (h2 : x < sqrt 3) : x * sqrt (3 - x^2) ≤ 9 / 4 :=
sorry

end max_value_of_y_l1705_170533


namespace simplify_expression_l1705_170541

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end simplify_expression_l1705_170541


namespace noemi_initial_money_l1705_170569

variable (money_lost_roulette : ℕ := 400)
variable (money_lost_blackjack : ℕ := 500)
variable (money_left : ℕ)
variable (money_started : ℕ)

axiom money_left_condition : money_left > 0
axiom total_loss_condition : money_lost_roulette + money_lost_blackjack = 900

theorem noemi_initial_money (h1 : money_lost_roulette = 400) (h2 : money_lost_blackjack = 500)
    (h3 : money_started - 900 = money_left) (h4 : money_left > 0) :
    money_started > 900 := by
  sorry

end noemi_initial_money_l1705_170569


namespace log_w_u_value_l1705_170579

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_w_u_value (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu1 : u ≠ 1) (hv1 : v ≠ 1) (hw1 : w ≠ 1)
    (h1 : log u (v * w) + log v w = 5) (h2 : log v u + log w v = 3) : 
    log w u = 4 / 5 := 
sorry

end log_w_u_value_l1705_170579


namespace jame_initial_gold_bars_l1705_170568

theorem jame_initial_gold_bars (X : ℝ) (h1 : X * 0.1 + 0.5 * (X * 0.9) = 0.5 * (X * 0.9) - 27) :
  X = 60 :=
by
-- Placeholder for proof
sorry

end jame_initial_gold_bars_l1705_170568


namespace correct_operation_l1705_170573

theorem correct_operation (a : ℝ) : (a^3)^3 = a^9 := 
sorry

end correct_operation_l1705_170573


namespace trajectory_of_center_of_moving_circle_l1705_170571

noncomputable def center_trajectory (x y : ℝ) : Prop :=
  0 < y ∧ y ≤ 1 ∧ x^2 = 4 * (y - 1)

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  0 ≤ y ∧ y ≤ 2 ∧ x^2 + y^2 = 4 ∧ 0 < y → center_trajectory x y :=
by
  sorry

end trajectory_of_center_of_moving_circle_l1705_170571


namespace radius_of_tangent_circle_l1705_170578

theorem radius_of_tangent_circle (a b : ℕ) (r1 r2 r3 : ℚ) (R : ℚ)
  (h1 : a = 6) (h2 : b = 8)
  (h3 : r1 = a / 2) (h4 : r2 = b / 2) (h5 : r3 = (Real.sqrt (a^2 + b^2)) / 2) :
  R = 144 / 23 := sorry

end radius_of_tangent_circle_l1705_170578


namespace total_dogs_l1705_170521

variable (U : Type) [Fintype U]
variable (jump fetch shake : U → Prop)
variable [DecidablePred jump] [DecidablePred fetch] [DecidablePred shake]

theorem total_dogs (h_jump : Fintype.card {u | jump u} = 70)
  (h_jump_and_fetch : Fintype.card {u | jump u ∧ fetch u} = 30)
  (h_fetch : Fintype.card {u | fetch u} = 40)
  (h_fetch_and_shake : Fintype.card {u | fetch u ∧ shake u} = 20)
  (h_shake : Fintype.card {u | shake u} = 50)
  (h_jump_and_shake : Fintype.card {u | jump u ∧ shake u} = 25)
  (h_all_three : Fintype.card {u | jump u ∧ fetch u ∧ shake u} = 15)
  (h_none : Fintype.card {u | ¬jump u ∧ ¬fetch u ∧ ¬shake u} = 15) :
  Fintype.card U = 115 :=
by
  sorry

end total_dogs_l1705_170521


namespace shapes_identification_l1705_170534

theorem shapes_identification :
  (∃ x y: ℝ, (x - 1/2)^2 + y^2 = 1/4) ∧ (∃ t: ℝ, x = -t ∧ y = 2 + t → x + y + 1 = 0) :=
by
  sorry

end shapes_identification_l1705_170534


namespace solution_set_l1705_170580

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem solution_set :
  {x : ℝ | x + (x + 2) * f (x + 2) ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by {
  sorry
}

end solution_set_l1705_170580


namespace cost_of_used_cd_l1705_170503

theorem cost_of_used_cd (N U : ℝ) 
    (h1 : 6 * N + 2 * U = 127.92) 
    (h2 : 3 * N + 8 * U = 133.89) :
    U = 9.99 :=
by 
  sorry

end cost_of_used_cd_l1705_170503


namespace circles_intersect_line_l1705_170530

theorem circles_intersect_line (m c : ℝ)
  (hA : (1 : ℝ) - 3 + c = 0)
  (hB : 1 = -(m - 1) / (-4)) :
  m + c = -1 :=
by
  sorry

end circles_intersect_line_l1705_170530


namespace isosceles_triangle_roots_l1705_170524

theorem isosceles_triangle_roots (k : ℝ) (a b : ℝ) 
  (h1 : a = 2 ∨ b = 2)
  (h2 : a^2 - 6 * a + k = 0)
  (h3 : b^2 - 6 * b + k = 0) :
  k = 9 :=
by
  sorry

end isosceles_triangle_roots_l1705_170524


namespace johan_painted_green_fraction_l1705_170562

theorem johan_painted_green_fraction :
  let total_rooms := 10
  let walls_per_room := 8
  let purple_walls := 32
  let purple_rooms := purple_walls / walls_per_room
  let green_rooms := total_rooms - purple_rooms
  (green_rooms : ℚ) / total_rooms = 3 / 5 := by
  sorry

end johan_painted_green_fraction_l1705_170562


namespace algebra_expression_value_l1705_170598

theorem algebra_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 11) : 3 * x^2 + 9 * x + 12 = 30 := 
by
  sorry

end algebra_expression_value_l1705_170598


namespace min_length_GH_l1705_170549

theorem min_length_GH :
  let ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
  let A := (-2, 0)
  let B := (2, 0)
  ∀ P G H : ℝ × ℝ,
    (P.1^2 / 4 + P.2^2 = 1) →
    P.2 > 0 →
    (G.2 = 3) →
    (H.2 = 3) →
    ∃ k : ℝ, k > 0 ∧ G.1 = 3 / k - 2 ∧ H.1 = -12 * k + 2 →
    |G.1 - H.1| = 8 :=
sorry

end min_length_GH_l1705_170549


namespace proof_smallest_integer_proof_sum_of_integers_l1705_170542

def smallest_integer (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ n = 98

def sum_of_integers (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ a + b + c + d + e = 510

theorem proof_smallest_integer : ∃ n : Int, smallest_integer n := by
  sorry

theorem proof_sum_of_integers : ∃ n : Int, sum_of_integers n := by
  sorry

end proof_smallest_integer_proof_sum_of_integers_l1705_170542


namespace probability_increase_l1705_170563

theorem probability_increase:
  let P_win1 := 0.30
  let P_lose1 := 0.70
  let P_win2 := 0.50
  let P_lose2 := 0.50
  let P_win3 := 0.40
  let P_lose3 := 0.60
  let P_win4 := 0.25
  let P_lose4 := 0.75
  let P_win_all := P_win1 * P_win2 * P_win3 * P_win4
  let P_lose_all := P_lose1 * P_lose2 * P_lose3 * P_lose4
  (P_lose_all - P_win_all) / P_win_all = 9.5 :=
by
  sorry

end probability_increase_l1705_170563


namespace pool_filling_time_l1705_170596

theorem pool_filling_time (rate_jim rate_sue rate_tony : ℝ) (h1 : rate_jim = 1 / 30) (h2 : rate_sue = 1 / 45) (h3 : rate_tony = 1 / 90) : 
     1 / (rate_jim + rate_sue + rate_tony) = 15 := by
  sorry

end pool_filling_time_l1705_170596


namespace largest_c_in_range_of_f_l1705_170589

theorem largest_c_in_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 - 6 * x + c = 2) -> c ≤ 11 :=
by
  sorry

end largest_c_in_range_of_f_l1705_170589


namespace first_bag_brown_mms_l1705_170555

theorem first_bag_brown_mms :
  ∀ (x : ℕ),
  (12 + 8 + 8 + 3 + x) / 5 = 8 → x = 9 :=
by
  intros x h
  sorry

end first_bag_brown_mms_l1705_170555


namespace gcd_p4_minus_1_eq_240_l1705_170550

theorem gcd_p4_minus_1_eq_240 (p : ℕ) (hp : Prime p) (h_gt_5 : p > 5) :
  gcd (p^4 - 1) 240 = 240 :=
by sorry

end gcd_p4_minus_1_eq_240_l1705_170550


namespace number_of_round_table_arrangements_l1705_170513

theorem number_of_round_table_arrangements : (Nat.factorial 5) / 5 = 24 := 
by
  sorry

end number_of_round_table_arrangements_l1705_170513


namespace last_digit_of_sum_edges_l1705_170508

def total_edges (n : ℕ) : ℕ := (n + 1) * n * 2

def internal_edges (n : ℕ) : ℕ := (n - 1) * n * 2

def dominoes (n : ℕ) : ℕ := (n * n) / 2

def perfect_matchings (n : ℕ) : ℕ := if n = 8 then 12988816 else 0  -- specific to 8x8 chessboard

def sum_internal_edges_contribution (n : ℕ) : ℕ := perfect_matchings n * (dominoes n * 2)

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_sum_edges {n : ℕ} (h : n = 8) :
  last_digit (sum_internal_edges_contribution n) = 4 :=
by
  rw [h]
  sorry

end last_digit_of_sum_edges_l1705_170508


namespace people_present_l1705_170588

-- Number of parents, pupils, teachers, staff members, and volunteers
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_teachers : ℕ := 35
def num_staff_members : ℕ := 20
def num_volunteers : ℕ := 50

-- The total number of people present in the program
def total_people : ℕ := num_parents + num_pupils + num_teachers + num_staff_members + num_volunteers

-- Proof statement
theorem people_present : total_people = 908 := by
  -- Proof goes here, but adding sorry for now
  sorry

end people_present_l1705_170588


namespace find_b_l1705_170581

def h (x : ℝ) : ℝ := 5 * x + 6

theorem find_b : ∃ b : ℝ, h b = 0 ∧ b = -6 / 5 :=
by
  sorry

end find_b_l1705_170581


namespace a_b_condition_l1705_170501

theorem a_b_condition (a b : ℂ) (h : (a + b) / a = b / (a + b)) :
  (∃ x y : ℂ, x = a ∧ y = b ∧ ((¬ x.im = 0 ∧ y.im = 0) ∨ (x.im = 0 ∧ ¬ y.im = 0) ∨ (¬ x.im = 0 ∧ ¬ y.im = 0))) :=
by
  sorry

end a_b_condition_l1705_170501


namespace star_comm_star_assoc_star_id_exists_star_not_dist_add_l1705_170572

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Statement 1: Commutativity
theorem star_comm : ∀ x y : ℝ, star x y = star y x := 
by sorry

-- Statement 2: Associativity
theorem star_assoc : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := 
by sorry

-- Statement 3: Identity Element
theorem star_id_exists : ∃ e : ℝ, ∀ x : ℝ, star x e = x := 
by sorry

-- Statement 4: Distributivity Over Addition
theorem star_not_dist_add : ∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z := 
by sorry

end star_comm_star_assoc_star_id_exists_star_not_dist_add_l1705_170572


namespace sarah_initial_bake_l1705_170502

theorem sarah_initial_bake (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (initial_cupcakes : ℕ)
  (h1 : todd_ate = 14)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 8)
  (h4 : packages * cupcakes_per_package + todd_ate = initial_cupcakes) :
  initial_cupcakes = 38 :=
by sorry

end sarah_initial_bake_l1705_170502


namespace twenty_percent_of_x_l1705_170557

noncomputable def x := 1800 / 1.2

theorem twenty_percent_of_x (h : 1.2 * x = 1800) : 0.2 * x = 300 :=
by
  -- The proof would go here, but we'll replace it with sorry.
  sorry

end twenty_percent_of_x_l1705_170557


namespace simplify_vector_eq_l1705_170511

-- Define points A, B, C
variables {A B C O : Type} [AddGroup A]

-- Define vector operations corresponding to overrightarrow.
variables (AB OC OB AC AO BO : A)

-- Conditions in Lean definitions
-- Assuming properties like vector addition and subtraction, and associative properties
def vector_eq : Prop := AB + OC - OB = AC

theorem simplify_vector_eq :
  AB + OC - OB = AC :=
by
  -- Proof steps go here
  sorry

end simplify_vector_eq_l1705_170511


namespace triangle_side_sum_l1705_170546

def sum_of_remaining_sides_of_triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α = 40 ∧ β = 50 ∧ γ = 180 - α - β ∧ c = 8 * Real.sqrt 3 →
  (a + b) = 34.3

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :
  sum_of_remaining_sides_of_triangle A B C a b c α β γ :=
sorry

end triangle_side_sum_l1705_170546


namespace ice_cream_flavors_l1705_170504

theorem ice_cream_flavors : (Nat.choose 8 3) = 56 := 
by {
    sorry
}

end ice_cream_flavors_l1705_170504


namespace problem_bound_l1705_170540

theorem problem_bound (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by 
  sorry

end problem_bound_l1705_170540


namespace travel_distance_l1705_170545

-- Define the average speed of the car
def speed : ℕ := 68

-- Define the duration of the trip in hours
def time : ℕ := 12

-- Define the distance formula for constant speed
def distance (speed time : ℕ) : ℕ := speed * time

-- Proof statement
theorem travel_distance : distance speed time = 756 := by
  -- Provide a placeholder for the proof
  sorry

end travel_distance_l1705_170545


namespace reported_length_correct_l1705_170558

def length_in_yards := 80
def conversion_factor := 3 -- 1 yard is 3 feet
def length_in_feet := 240

theorem reported_length_correct :
  length_in_feet = length_in_yards * conversion_factor :=
by rfl

end reported_length_correct_l1705_170558


namespace roots_bounds_if_and_only_if_conditions_l1705_170518

theorem roots_bounds_if_and_only_if_conditions (a b c : ℝ) (h : a > 0) (x1 x2 : ℝ) (hr : ∀ {x : ℝ}, a * x^2 + b * x + c = 0 → x = x1 ∨ x = x2) :
  (|x1| ≤ 1 ∧ |x2| ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) :=
sorry

end roots_bounds_if_and_only_if_conditions_l1705_170518


namespace checkered_rectangles_containing_one_gray_cell_l1705_170597

theorem checkered_rectangles_containing_one_gray_cell 
  (num_gray_cells : ℕ) 
  (num_blue_cells : ℕ) 
  (num_red_cells : ℕ)
  (blue_containing_rectangles : ℕ) 
  (red_containing_rectangles : ℕ) :
  num_gray_cells = 40 →
  num_blue_cells = 36 →
  num_red_cells = 4 →
  blue_containing_rectangles = 4 →
  red_containing_rectangles = 8 →
  num_blue_cells * blue_containing_rectangles + num_red_cells * red_containing_rectangles = 176 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end checkered_rectangles_containing_one_gray_cell_l1705_170597


namespace max_sum_seq_l1705_170599

theorem max_sum_seq (a : ℕ → ℝ) (h1 : a 1 = 0)
  (h2 : abs (a 2) = abs (a 1 - 1)) 
  (h3 : abs (a 3) = abs (a 2 - 1)) 
  (h4 : abs (a 4) = abs (a 3 - 1)) 
  : ∃ M, (∀ (b : ℕ → ℝ), b 1 = 0 → abs (b 2) = abs (b 1 - 1) → abs (b 3) = abs (b 2 - 1) → abs (b 4) = abs (b 3 - 1) → (b 1 + b 2 + b 3 + b 4) ≤ M) 
    ∧ (a 1 + a 2 + a 3 + a 4 = M) :=
  sorry

end max_sum_seq_l1705_170599


namespace roberto_raise_percentage_l1705_170574

theorem roberto_raise_percentage
    (starting_salary : ℝ)
    (previous_salary : ℝ)
    (current_salary : ℝ)
    (h1 : starting_salary = 80000)
    (h2 : previous_salary = starting_salary * 1.40)
    (h3 : current_salary = 134400) :
    ((current_salary - previous_salary) / previous_salary) * 100 = 20 :=
by sorry

end roberto_raise_percentage_l1705_170574


namespace paul_diner_total_cost_l1705_170517

/-- At Paul's Diner, sandwiches cost $5 each and sodas cost $3 each. If a customer buys
more than 4 sandwiches, they receive a $10 discount on the total bill. Calculate the total
cost if a customer purchases 6 sandwiches and 3 sodas. -/
def totalCost (num_sandwiches num_sodas : ℕ) : ℕ :=
  let sandwich_cost := 5
  let soda_cost := 3
  let discount := if num_sandwiches > 4 then 10 else 0
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) - discount

theorem paul_diner_total_cost : totalCost 6 3 = 29 :=
by
  sorry

end paul_diner_total_cost_l1705_170517


namespace Lauryn_employs_80_men_l1705_170522

theorem Lauryn_employs_80_men (W M : ℕ) 
  (h1 : M = W - 20) 
  (h2 : M + W = 180) : 
  M = 80 := 
by 
  sorry

end Lauryn_employs_80_men_l1705_170522


namespace rational_quotient_of_arith_geo_subseq_l1705_170551

theorem rational_quotient_of_arith_geo_subseq (A d : ℝ) (h_d_nonzero : d ≠ 0)
    (h_contains_geo : ∃ (q : ℝ) (k m n : ℕ), q ≠ 1 ∧ q ≠ 0 ∧ 
        A + k * d = (A + m * d) * q ∧ A + m * d = (A + n * d) * q)
    : ∃ (r : ℚ), A / d = r :=
  sorry

end rational_quotient_of_arith_geo_subseq_l1705_170551


namespace farmer_ducks_sold_l1705_170585

theorem farmer_ducks_sold (D : ℕ) (earnings : ℕ) :
  (earnings = (10 * D) + (5 * 8)) →
  ((earnings / 2) * 2 = 60) →
  D = 2 := by
  sorry

end farmer_ducks_sold_l1705_170585


namespace smallest_number_is_minus_three_l1705_170593

theorem smallest_number_is_minus_three :
  ∀ (a b c d : ℤ), (a = 0) → (b = -3) → (c = 1) → (d = -1) → b < d ∧ d < a ∧ a < c → b = -3 :=
by
  intros a b c d ha hb hc hd h
  exact hb

end smallest_number_is_minus_three_l1705_170593


namespace max_area_right_triangle_l1705_170553

def right_triangle_max_area (l : ℝ) (p : ℝ) (h : ℝ) : ℝ :=
  l + p + h

noncomputable def maximal_area (x y : ℝ) : ℝ :=
  (1/2) * x * y

theorem max_area_right_triangle (x y : ℝ) (h : ℝ) (hp : h = Real.sqrt (x^2 + y^2)) (hp2: x + y + h = 60) :
  maximal_area 30 30 = 450 :=
by
  sorry

end max_area_right_triangle_l1705_170553


namespace giant_slide_wait_is_15_l1705_170537

noncomputable def wait_time_for_giant_slide
  (hours_at_carnival : ℕ) 
  (roller_coaster_wait : ℕ)
  (tilt_a_whirl_wait : ℕ)
  (rides_roller_coaster : ℕ)
  (rides_tilt_a_whirl : ℕ)
  (rides_giant_slide : ℕ) : ℕ :=
  
  (hours_at_carnival * 60 - (roller_coaster_wait * rides_roller_coaster + tilt_a_whirl_wait * rides_tilt_a_whirl)) / rides_giant_slide

theorem giant_slide_wait_is_15 :
  wait_time_for_giant_slide 4 30 60 4 1 4 = 15 := 
sorry

end giant_slide_wait_is_15_l1705_170537


namespace speed_in_still_water_l1705_170587

theorem speed_in_still_water (u d s : ℝ) (hu : u = 20) (hd : d = 60) (hs : s = (u + d) / 2) : s = 40 := 
by 
  sorry

end speed_in_still_water_l1705_170587


namespace hyperbola_dot_product_zero_l1705_170591

theorem hyperbola_dot_product_zero
  (a b x y : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_ecc : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 2) :
  let B := (-x, y)
  let C := (x, y)
  let A := (a, 0)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 0 :=
by
  sorry

end hyperbola_dot_product_zero_l1705_170591


namespace pentagon_area_sol_l1705_170554

theorem pentagon_area_sol (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (3 * b + a) = 792) : a + b = 45 :=
sorry

end pentagon_area_sol_l1705_170554
