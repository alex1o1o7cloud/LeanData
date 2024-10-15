import Mathlib

namespace NUMINAMATH_GPT_angles_of_terminal_side_on_line_y_equals_x_l824_82474

noncomputable def set_of_angles_on_y_equals_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 180 + 45

theorem angles_of_terminal_side_on_line_y_equals_x (α : ℝ) :
  (∃ k : ℤ, α = k * 360 + 45) ∨ (∃ k : ℤ, α = k * 360 + 225) ↔ set_of_angles_on_y_equals_x α :=
by
  sorry

end NUMINAMATH_GPT_angles_of_terminal_side_on_line_y_equals_x_l824_82474


namespace NUMINAMATH_GPT_min_value_4x2_plus_y2_l824_82456

theorem min_value_4x2_plus_y2 {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) : 
  4 * x^2 + y^2 ≥ 18 := by
  sorry

end NUMINAMATH_GPT_min_value_4x2_plus_y2_l824_82456


namespace NUMINAMATH_GPT_bad_carrots_count_l824_82424

def total_carrots (vanessa_carrots : ℕ) (mother_carrots : ℕ) : ℕ := 
vanessa_carrots + mother_carrots

def bad_carrots (total_carrots : ℕ) (good_carrots : ℕ) : ℕ := 
total_carrots - good_carrots

theorem bad_carrots_count : 
  ∀ (vanessa_carrots mother_carrots good_carrots : ℕ), 
  vanessa_carrots = 17 → 
  mother_carrots = 14 → 
  good_carrots = 24 → 
  bad_carrots (total_carrots vanessa_carrots mother_carrots) good_carrots = 7 := 
by 
  intros; 
  sorry

end NUMINAMATH_GPT_bad_carrots_count_l824_82424


namespace NUMINAMATH_GPT_total_games_played_l824_82465

noncomputable def win_ratio : ℝ := 5.5
noncomputable def lose_ratio : ℝ := 4.5
noncomputable def tie_ratio : ℝ := 2.5
noncomputable def rained_out_ratio : ℝ := 1
noncomputable def higher_league_ratio : ℝ := 3.5
noncomputable def lost_games : ℝ := 13.5

theorem total_games_played :
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  total_parts * games_per_part = 51 :=
by
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  have : total_parts * games_per_part = 51 := sorry
  exact this

end NUMINAMATH_GPT_total_games_played_l824_82465


namespace NUMINAMATH_GPT_red_candies_remain_percentage_l824_82473

noncomputable def percent_red_candies_remain (N : ℝ) : ℝ :=
let total_initial_candies : ℝ := 5 * N
let green_candies_eat : ℝ := N
let remaining_after_green : ℝ := total_initial_candies - green_candies_eat

let half_orange_candies_eat : ℝ := N / 2
let remaining_after_half_orange : ℝ := remaining_after_green - half_orange_candies_eat

let half_all_remaining_candies_eat : ℝ := (N / 2) + (N / 4) + (N / 2) + (N / 2)
let remaining_after_half_all : ℝ := remaining_after_half_orange - half_all_remaining_candies_eat

let final_remaining_candies : ℝ := 0.32 * total_initial_candies
let candies_to_eat_finally : ℝ := remaining_after_half_all - final_remaining_candies
let each_color_final_eat : ℝ := candies_to_eat_finally / 2

let remaining_red_candies : ℝ := (N / 2) - each_color_final_eat

(remaining_red_candies / N) * 100

theorem red_candies_remain_percentage (N : ℝ) : percent_red_candies_remain N = 42.5 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_red_candies_remain_percentage_l824_82473


namespace NUMINAMATH_GPT_quadratic_roots_condition_l824_82480

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k*x^2 + 2*x + 1 = 0 ∧ k*y^2 + 2*y + 1 = 0) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l824_82480


namespace NUMINAMATH_GPT_Zoe_siblings_l824_82420

structure Child where
  eyeColor : String
  hairColor : String
  height : String

def Emma : Child := { eyeColor := "Green", hairColor := "Red", height := "Tall" }
def Zoe : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Short" }
def Liam : Child := { eyeColor := "Green", hairColor := "Brown", height := "Short" }
def Noah : Child := { eyeColor := "Gray", hairColor := "Red", height := "Tall" }
def Mia : Child := { eyeColor := "Green", hairColor := "Red", height := "Short" }
def Lucas : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Tall" }

def sibling (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.height = c2.height

theorem Zoe_siblings : sibling Zoe Noah ∧ sibling Zoe Lucas ∧ ∃ x, sibling Noah x ∧ sibling Lucas x :=
by
  sorry

end NUMINAMATH_GPT_Zoe_siblings_l824_82420


namespace NUMINAMATH_GPT_range_of_a_l824_82454

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l824_82454


namespace NUMINAMATH_GPT_bradley_travel_time_l824_82436

theorem bradley_travel_time (T : ℕ) (h1 : T / 4 = 20) (h2 : T / 3 = 45) : T - 20 = 280 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_bradley_travel_time_l824_82436


namespace NUMINAMATH_GPT_first_snail_time_proof_l824_82429

-- Define the conditions
def first_snail_speed := 2 -- speed in feet per minute
def second_snail_speed := 2 * first_snail_speed
def third_snail_speed := 5 * second_snail_speed
def third_snail_time := 2 -- time in minutes
def distance := third_snail_speed * third_snail_time

-- Define the time it took the first snail
def first_snail_time := distance / first_snail_speed

-- Define the theorem to be proven
theorem first_snail_time_proof : first_snail_time = 20 := 
by
  -- Proof should be filled here
  sorry

end NUMINAMATH_GPT_first_snail_time_proof_l824_82429


namespace NUMINAMATH_GPT_geometric_sequence_property_l824_82481

-- Define the sequence and the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the main property we are considering
def given_property (a: ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n - 1) = (a n) ^ 2

-- State the theorem
theorem geometric_sequence_property {a : ℕ → ℝ} (n : ℕ) (hn : n ≥ 2) :
  (is_geometric_sequence a → given_property a n ∧ ∀ a, given_property a n → ¬ is_geometric_sequence a) := sorry

end NUMINAMATH_GPT_geometric_sequence_property_l824_82481


namespace NUMINAMATH_GPT_updated_mean_l824_82407

-- Definitions
def initial_mean := 200
def number_of_observations := 50
def decrement_per_observation := 9

-- Theorem stating the updated mean after decrementing each observation
theorem updated_mean : 
  (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 191 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_updated_mean_l824_82407


namespace NUMINAMATH_GPT_scholars_number_l824_82493

theorem scholars_number (n : ℕ) : n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 → n = 509 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_scholars_number_l824_82493


namespace NUMINAMATH_GPT_y_n_is_square_of_odd_integer_l824_82427

-- Define the sequences and the initial conditions
def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

-- Helper function to check if a number is odd
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The theorem to prove
theorem y_n_is_square_of_odd_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, y n = k ^ 2 ∧ is_odd k := by
  sorry

end NUMINAMATH_GPT_y_n_is_square_of_odd_integer_l824_82427


namespace NUMINAMATH_GPT_reflection_image_l824_82408

theorem reflection_image (m b : ℝ) 
  (h1 : ∀ x y : ℝ, (x, y) = (0, 1) → (4, 5) = (2 * ((x + (m * y - y + b))/ (1 + m^2)) - x, 2 * ((y + (m * x - x + b)) / (1 + m^2)) - y))
  : m + b = 4 :=
sorry

end NUMINAMATH_GPT_reflection_image_l824_82408


namespace NUMINAMATH_GPT_number_of_integers_l824_82414

theorem number_of_integers (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2020) (h3 : ∃ k : ℕ, n^n = k^2) : n = 1032 :=
sorry

end NUMINAMATH_GPT_number_of_integers_l824_82414


namespace NUMINAMATH_GPT_find_t_l824_82477

theorem find_t (s t : ℤ) (h1 : 9 * s + 5 * t = 108) (h2 : s = t - 2) : t = 9 :=
sorry

end NUMINAMATH_GPT_find_t_l824_82477


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l824_82405

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 := by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l824_82405


namespace NUMINAMATH_GPT_trigonometric_unique_solution_l824_82488

theorem trigonometric_unique_solution :
  (∃ x : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ 0 ≤ y ∧ y < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 ∧
    Real.sin y = 0.6 ∧ Real.cos y = 0.8 → x = y) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_unique_solution_l824_82488


namespace NUMINAMATH_GPT_find_first_number_l824_82464

theorem find_first_number (HCF LCM number2 number1 : ℕ) 
    (hcf_condition : HCF = 12) 
    (lcm_condition : LCM = 396) 
    (number2_condition : number2 = 198) 
    (number1_condition : number1 * number2 = HCF * LCM) : 
    number1 = 24 := 
by 
    sorry

end NUMINAMATH_GPT_find_first_number_l824_82464


namespace NUMINAMATH_GPT_Alyssa_cookie_count_l824_82432

/--
  Alyssa had some cookies.
  Aiyanna has 140 cookies.
  Aiyanna has 11 more cookies than Alyssa.
  How many cookies does Alyssa have? 
-/
theorem Alyssa_cookie_count 
  (aiyanna_cookies : ℕ) 
  (more_cookies : ℕ)
  (h1 : aiyanna_cookies = 140)
  (h2 : more_cookies = 11)
  (h3 : aiyanna_cookies = alyssa_cookies + more_cookies) :
  alyssa_cookies = 129 := 
sorry

end NUMINAMATH_GPT_Alyssa_cookie_count_l824_82432


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l824_82409

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l824_82409


namespace NUMINAMATH_GPT_coin_problem_l824_82494

theorem coin_problem : ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5) ∧ n % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_coin_problem_l824_82494


namespace NUMINAMATH_GPT_encounter_count_l824_82489

theorem encounter_count (vA vB d : ℝ) (h₁ : 5 * d / vA = 9 * d / vB) :
  ∃ encounters : ℝ, encounters = 3023 :=
by
  sorry

end NUMINAMATH_GPT_encounter_count_l824_82489


namespace NUMINAMATH_GPT_power_equivalence_l824_82497

theorem power_equivalence (L : ℕ) : 32^4 * 4^5 = 2^L → L = 30 :=
by
  sorry

end NUMINAMATH_GPT_power_equivalence_l824_82497


namespace NUMINAMATH_GPT_next_special_year_after_2009_l824_82416

def is_special_year (n : ℕ) : Prop :=
  ∃ d1 d2 d3 d4 : ℕ,
    (2000 ≤ n) ∧ (n < 10000) ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n) ∧
    (d1 ≠ 0) ∧
    ∀ (p q r s : ℕ),
    (p * 1000 + q * 100 + r * 10 + s < n) →
    (p ≠ d1 ∨ q ≠ d2 ∨ r ≠ d3 ∨ s ≠ d4)

theorem next_special_year_after_2009 : ∃ y : ℕ, is_special_year y ∧ y > 2009 ∧ y = 2022 :=
  sorry

end NUMINAMATH_GPT_next_special_year_after_2009_l824_82416


namespace NUMINAMATH_GPT_no_difference_of_squares_equals_222_l824_82469

theorem no_difference_of_squares_equals_222 (a b : ℤ) : a^2 - b^2 ≠ 222 := 
  sorry

end NUMINAMATH_GPT_no_difference_of_squares_equals_222_l824_82469


namespace NUMINAMATH_GPT_at_least_two_same_books_l824_82441

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def satisfied (n : Nat) : Prop :=
  n / sum_of_digits n = 13

theorem at_least_two_same_books (n1 n2 n3 n4 : Nat) (h1 : satisfied n1) (h2 : satisfied n2) (h3 : satisfied n3) (h4 : satisfied n4) :
  n1 = n2 ∨ n1 = n3 ∨ n1 = n4 ∨ n2 = n3 ∨ n2 = n4 ∨ n3 = n4 :=
sorry

end NUMINAMATH_GPT_at_least_two_same_books_l824_82441


namespace NUMINAMATH_GPT_value_of_a_l824_82402

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end NUMINAMATH_GPT_value_of_a_l824_82402


namespace NUMINAMATH_GPT_triangle_PZQ_area_is_50_l824_82484

noncomputable def area_triangle_PZQ (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) : ℝ :=
  let RS := PQ -- since PQRS is a rectangle, RS = PQ
  let XY := RS - RX - SY
  let height := 2 * QR -- height is doubled due to triangle similarity ratio
  let area := 0.5 * PQ * height
  area

theorem triangle_PZQ_area_is_50 (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) :
  area_triangle_PZQ PQ QR RX SY hPQ hQR hRX hSY = 50 :=
  sorry

end NUMINAMATH_GPT_triangle_PZQ_area_is_50_l824_82484


namespace NUMINAMATH_GPT_find_original_number_l824_82428

theorem find_original_number : ∃ (N : ℤ), (∃ (k : ℤ), N - 30 = 87 * k) ∧ N = 117 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l824_82428


namespace NUMINAMATH_GPT_find_ratios_sum_l824_82490
noncomputable def Ana_biking_rate : ℝ := 8.6
noncomputable def Bob_biking_rate : ℝ := 6.2
noncomputable def CAO_biking_rate : ℝ := 5

variable (a b c : ℝ)

-- Conditions  
def Ana_distance := 2 * a + b + c = Ana_biking_rate
def Bob_distance := b + c = Bob_biking_rate
def Cao_distance := Real.sqrt (b^2 + c^2) = CAO_biking_rate

-- Main statement
theorem find_ratios_sum : 
  Ana_distance a b c ∧ 
  Bob_distance b c ∧ 
  Cao_distance b c →
  ∃ (p q r : ℕ), p + q + r = 37 ∧ Nat.gcd p q = 1 ∧ ((a / c) = p / r) ∧ ((b / c) = q / r) ∧ ((a / b) = p / q) :=
sorry

end NUMINAMATH_GPT_find_ratios_sum_l824_82490


namespace NUMINAMATH_GPT_lines_condition_l824_82421

-- Assume x and y are real numbers representing coordinates on the lines l1 and l2
variables (x y : ℝ)

-- Points on the lines l1 and l2 satisfy the condition |x| - |y| = 0.
theorem lines_condition (x y : ℝ) (h : abs x = abs y) : abs x - abs y = 0 :=
by
  sorry

end NUMINAMATH_GPT_lines_condition_l824_82421


namespace NUMINAMATH_GPT_f_2009_is_one_l824_82404

   -- Define the properties of the function f
   variables (f : ℤ → ℤ)
   variable (h_even : ∀ x : ℤ, f x = f (-x))
   variable (h1 : f 1 = 1)
   variable (h2008 : f 2008 ≠ 1)
   variable (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b))

   -- Prove that f(2009) = 1
   theorem f_2009_is_one : f 2009 = 1 :=
   sorry
   
end NUMINAMATH_GPT_f_2009_is_one_l824_82404


namespace NUMINAMATH_GPT_mens_wages_l824_82419

variable (M : ℕ) (wages_of_men : ℕ)

-- Conditions based on the problem
axiom eq1 : 15 * M = 90
axiom def_wages_of_men : wages_of_men = 5 * M

-- Prove that the total wages of the men are Rs. 30
theorem mens_wages : wages_of_men = 30 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_mens_wages_l824_82419


namespace NUMINAMATH_GPT_find_b_for_smallest_c_l824_82453

theorem find_b_for_smallest_c (c b : ℝ) (h_c_pos : 0 < c) (h_b_pos : 0 < b)
  (polynomial_condition : ∀ x : ℝ, (x^4 - c*x^3 + b*x^2 - c*x + 1 = 0) → real) :
  c = 4 → b = 6 :=
by
  intros h_c_eq_4
  sorry

end NUMINAMATH_GPT_find_b_for_smallest_c_l824_82453


namespace NUMINAMATH_GPT_young_fish_per_pregnant_fish_l824_82470

-- Definitions based on conditions
def tanks := 3
def fish_per_tank := 4
def total_young_fish := 240

-- Calculations based on conditions
def total_pregnant_fish := tanks * fish_per_tank

-- The proof statement
theorem young_fish_per_pregnant_fish : total_young_fish / total_pregnant_fish = 20 := by
  sorry

end NUMINAMATH_GPT_young_fish_per_pregnant_fish_l824_82470


namespace NUMINAMATH_GPT_probability_same_color_l824_82487

-- Define the total combinations function
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- The given values from the problem
def whiteBalls := 2
def blackBalls := 3
def totalBalls := whiteBalls + blackBalls
def drawnBalls := 2

-- Calculate combinations
def comb_white_2 := comb whiteBalls drawnBalls
def comb_black_2 := comb blackBalls drawnBalls
def comb_total_2 := comb totalBalls drawnBalls

-- The correct answer given in the solution
def correct_probability := 2 / 5

-- Statement for the proof in Lean
theorem probability_same_color : (comb_white_2 + comb_black_2) / comb_total_2 = correct_probability := by
  sorry

end NUMINAMATH_GPT_probability_same_color_l824_82487


namespace NUMINAMATH_GPT_arc_length_of_curve_l824_82446

noncomputable def arc_length : ℝ :=
∫ t in (0 : ℝ)..(Real.pi / 3),
  (Real.sqrt ((t^2 * Real.cos t)^2 + (t^2 * Real.sin t)^2))

theorem arc_length_of_curve :
  arc_length = (Real.pi^3 / 81) :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_curve_l824_82446


namespace NUMINAMATH_GPT_smaller_of_two_digit_product_l824_82467

theorem smaller_of_two_digit_product (a b : ℕ) (h1 : a * b = 4896) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 32 :=
sorry

end NUMINAMATH_GPT_smaller_of_two_digit_product_l824_82467


namespace NUMINAMATH_GPT_infinite_sequence_exists_l824_82471

noncomputable def has_k_distinct_positive_divisors (n k : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ d ∈ S, d ∣ n

theorem infinite_sequence_exists :
    ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, 0 < k → ∃ n : ℕ, (a n > 0) ∧ has_k_distinct_positive_divisors (a n ^ 2 + a n + 2023) k) :=
  sorry

end NUMINAMATH_GPT_infinite_sequence_exists_l824_82471


namespace NUMINAMATH_GPT_leo_class_girls_l824_82444

theorem leo_class_girls (g b : ℕ) 
  (h_ratio : 3 * b = 4 * g) 
  (h_total : g + b = 35) : g = 15 := 
by
  sorry

end NUMINAMATH_GPT_leo_class_girls_l824_82444


namespace NUMINAMATH_GPT_value_of_fraction_pow_l824_82447

theorem value_of_fraction_pow (a b : ℤ) 
  (h1 : ∀ x, (x^2 + (a + 1)*x + a*b) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 4) : 
  ((1 / 2 : ℚ) ^ (a + 2*b) = 4) :=
sorry

end NUMINAMATH_GPT_value_of_fraction_pow_l824_82447


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l824_82430

theorem arithmetic_sequence_term (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 4 = 6)
    (h2 : 2 * (a 3) - (a 2) = 6)
    (h_sum : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  a 1 = -3 := 
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l824_82430


namespace NUMINAMATH_GPT_arithmetic_sequence_m_value_l824_82479

theorem arithmetic_sequence_m_value (m : ℝ) (h : 2 + 6 = 2 * m) : m = 4 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_m_value_l824_82479


namespace NUMINAMATH_GPT_remainder_8354_11_l824_82458

theorem remainder_8354_11 : 8354 % 11 = 6 := sorry

end NUMINAMATH_GPT_remainder_8354_11_l824_82458


namespace NUMINAMATH_GPT_amount_of_silver_l824_82491

-- Definitions
def total_silver (x : ℕ) : Prop :=
  (x - 4) % 7 = 0 ∧ (x + 8) % 9 = 1

-- Theorem to be proven
theorem amount_of_silver (x : ℕ) (h : total_silver x) : (x - 4)/7 = (x + 8)/9 :=
by sorry

end NUMINAMATH_GPT_amount_of_silver_l824_82491


namespace NUMINAMATH_GPT_simplify_expression_l824_82452

theorem simplify_expression (n : ℤ) :
  (2 : ℝ) ^ (-(3 * n + 1)) + (2 : ℝ) ^ (-(3 * n - 2)) - 3 * (2 : ℝ) ^ (-3 * n) = (3 / 2) * (2 : ℝ) ^ (-3 * n) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l824_82452


namespace NUMINAMATH_GPT_row_time_to_100_yards_l824_82411

theorem row_time_to_100_yards :
  let init_width_yd := 50
  let final_width_yd := 100
  let increase_width_yd_per_10m := 2
  let rowing_speed_mps := 5
  let current_speed_mps := 1
  let yard_to_meter := 0.9144
  let init_width_m := init_width_yd * yard_to_meter
  let final_width_m := final_width_yd * yard_to_meter
  let width_increase_m_per_10m := increase_width_yd_per_10m * yard_to_meter
  let total_width_increase := (final_width_m - init_width_m)
  let num_segments := total_width_increase / width_increase_m_per_10m
  let total_distance := num_segments * 10
  let effective_speed := rowing_speed_mps + current_speed_mps
  let time := total_distance / effective_speed
  time = 41.67 := by
  sorry

end NUMINAMATH_GPT_row_time_to_100_yards_l824_82411


namespace NUMINAMATH_GPT_cougar_ratio_l824_82439

theorem cougar_ratio (lions tigers total_cats cougars : ℕ) 
  (h_lions : lions = 12) 
  (h_tigers : tigers = 14) 
  (h_total : total_cats = 39) 
  (h_cougars : cougars = total_cats - (lions + tigers)) 
  : cougars * 2 = lions + tigers := 
by 
  rw [h_lions, h_tigers] 
  norm_num at * 
  sorry

end NUMINAMATH_GPT_cougar_ratio_l824_82439


namespace NUMINAMATH_GPT_parametric_equations_solution_l824_82417

theorem parametric_equations_solution (t₁ t₂ : ℝ) : 
  (1 = 1 + 2 * t₁ ∧ 2 = 2 - 3 * t₁) ∧
  (-1 = 1 + 2 * t₂ ∧ 5 = 2 - 3 * t₂) ↔
  (t₁ = 0 ∧ t₂ = -1) :=
by
  sorry

end NUMINAMATH_GPT_parametric_equations_solution_l824_82417


namespace NUMINAMATH_GPT_school_allocation_methods_l824_82450

-- Define the conditions
def doctors : ℕ := 3
def nurses : ℕ := 6
def schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- The combinatorial function for binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Verify the number of allocation methods
theorem school_allocation_methods : 
  C doctors doctors_per_school * C nurses nurses_per_school *
  C (doctors - 1) doctors_per_school * C (nurses - 2) nurses_per_school *
  C (doctors - 2) doctors_per_school * C (nurses - 4) nurses_per_school = 540 := 
sorry

end NUMINAMATH_GPT_school_allocation_methods_l824_82450


namespace NUMINAMATH_GPT_daniel_total_worth_l824_82422

theorem daniel_total_worth
    (sales_tax_paid : ℝ)
    (sales_tax_rate : ℝ)
    (cost_tax_free_items : ℝ)
    (tax_rate_pos : 0 < sales_tax_rate) :
    sales_tax_paid = 0.30 →
    sales_tax_rate = 0.05 →
    cost_tax_free_items = 18.7 →
    ∃ (x : ℝ), 0.05 * x = 0.30 ∧ (x + cost_tax_free_items = 24.7) := by
    sorry

end NUMINAMATH_GPT_daniel_total_worth_l824_82422


namespace NUMINAMATH_GPT_first_storm_duration_l824_82495

theorem first_storm_duration
  (x y : ℕ)
  (h1 : 30 * x + 15 * y = 975)
  (h2 : x + y = 45) :
  x = 20 :=
by sorry

end NUMINAMATH_GPT_first_storm_duration_l824_82495


namespace NUMINAMATH_GPT_uma_income_l824_82466

theorem uma_income
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 5000)
  (h2 : 3 * x - 2 * y = 5000) :
  4 * x = 20000 :=
by
  sorry

end NUMINAMATH_GPT_uma_income_l824_82466


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l824_82459

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15)
  (h2 : a + 10 * d = 18) : 
  a + 4 * d = 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l824_82459


namespace NUMINAMATH_GPT_smallest_among_5_neg7_0_neg53_l824_82462

-- Define the rational numbers involved as constants
def a : ℚ := 5
def b : ℚ := -7
def c : ℚ := 0
def d : ℚ := -5 / 3

-- Define the conditions as separate lemmas
lemma positive_greater_than_zero (x : ℚ) (hx : x > 0) : x > c := by sorry
lemma zero_greater_than_negative (x : ℚ) (hx : x < 0) : c > x := by sorry
lemma compare_negative_by_absolute_value (x y : ℚ) (hx : x < 0) (hy : y < 0) (habs : |x| > |y|) : x < y := by sorry

-- Prove the main assertion
theorem smallest_among_5_neg7_0_neg53 : 
    b < a ∧ b < c ∧ b < d := by
    -- Here we apply the defined conditions to show b is the smallest
    sorry

end NUMINAMATH_GPT_smallest_among_5_neg7_0_neg53_l824_82462


namespace NUMINAMATH_GPT_integral_even_odd_l824_82486

open Real

theorem integral_even_odd (a : ℝ) :
  (∫ x in -a..a, x^2 + sin x) = 18 → a = 3 :=
by
  intros h
  -- We'll skip the proof
  sorry

end NUMINAMATH_GPT_integral_even_odd_l824_82486


namespace NUMINAMATH_GPT_nonagon_angles_l824_82478

/-- Determine the angles of the nonagon given specified conditions -/
theorem nonagon_angles (a : ℝ) (x : ℝ) 
  (h_angle_eq : ∀ (AIH BCD HGF : ℝ), AIH = x → BCD = x → HGF = x)
  (h_internal_sum : 7 * 180 = 1260)
  (h_tessellation : x + x + x + (360 - x) + (360 - x) + (360 - x) = 1080) :
  True := sorry

end NUMINAMATH_GPT_nonagon_angles_l824_82478


namespace NUMINAMATH_GPT_find_q_l824_82482

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 17) (h2 : 6 * p + 5 * q = 20) : q = 2 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l824_82482


namespace NUMINAMATH_GPT_math_proof_problem_l824_82443

-- Define the function and its properties
variable (f : ℝ → ℝ)
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity : ∀ x : ℝ, f (x + 1) = -f x
axiom increasing_on_interval : ∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y

-- Theorem statement expressing the questions and answers
theorem math_proof_problem :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧
  (f 2 = f 0) :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l824_82443


namespace NUMINAMATH_GPT_range_of_a_l824_82460

theorem range_of_a (a : ℝ) : (∀ x : ℤ, x > 2 * a - 3 ∧ 2 * (x : ℝ) ≥ 3 * ((x : ℝ) - 2) + 5) ↔ (1 / 2 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l824_82460


namespace NUMINAMATH_GPT_moles_of_NaNO3_formed_l824_82449

/- 
  Define the reaction and given conditions.
  The following assumptions and definitions will directly come from the problem's conditions.
-/

/-- 
  Represents a chemical reaction: 1 molecule of AgNO3,
  1 molecule of NaOH producing 1 molecule of NaNO3 and 1 molecule of AgOH.
-/
def balanced_reaction (agNO3 naOH naNO3 agOH : ℕ) := agNO3 = 1 ∧ naOH = 1 ∧ naNO3 = 1 ∧ agOH = 1

/-- 
  Proves that the number of moles of NaNO3 formed is 1,
  given 1 mole of AgNO3 and 1 mole of NaOH.
-/
theorem moles_of_NaNO3_formed (agNO3 naOH naNO3 agOH : ℕ)
  (h : balanced_reaction agNO3 naOH naNO3 agOH) :
  naNO3 = 1 := 
by
  sorry  -- Proof will be added here later

end NUMINAMATH_GPT_moles_of_NaNO3_formed_l824_82449


namespace NUMINAMATH_GPT_probability_neither_prime_nor_composite_lemma_l824_82410

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬ is_prime n ∧ ¬ is_composite n

def probability_of_neither_prime_nor_composite (n : ℕ) : ℚ :=
  if 1 ≤ n ∧ n ≤ 97 then 1 / 97 else 0

theorem probability_neither_prime_nor_composite_lemma :
  probability_of_neither_prime_nor_composite 1 = 1 / 97 := by
  sorry

end NUMINAMATH_GPT_probability_neither_prime_nor_composite_lemma_l824_82410


namespace NUMINAMATH_GPT_cone_from_sector_l824_82461

def cone_can_be_formed (θ : ℝ) (r_sector : ℝ) (r_cone_base : ℝ) (l_slant_height : ℝ) : Prop :=
  θ = 270 ∧ r_sector = 12 ∧ ∃ L, L = θ / 360 * (2 * Real.pi * r_sector) ∧ 2 * Real.pi * r_cone_base = L ∧ l_slant_height = r_sector

theorem cone_from_sector (base_radius slant_height : ℝ) :
  cone_can_be_formed 270 12 base_radius slant_height ↔ base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end NUMINAMATH_GPT_cone_from_sector_l824_82461


namespace NUMINAMATH_GPT_find_m_l824_82406

theorem find_m (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((1/3 : ℝ) * x1^3 - 3 * x1 + m = 0) ∧ ((1/3 : ℝ) * x2^3 - 3 * x2 + m = 0)) ↔ (m = -2 * Real.sqrt 3 ∨ m = 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_find_m_l824_82406


namespace NUMINAMATH_GPT_part1_part2_l824_82476

def y (x : ℝ) : ℝ := -x^2 + 8*x - 7

-- Part (1) Lean statement
theorem part1 : ∀ x : ℝ, x < 4 → y x < y (x + 1) := sorry

-- Part (2) Lean statement
theorem part2 : ∀ x : ℝ, (x < 1 ∨ x > 7) → y x < 0 := sorry

end NUMINAMATH_GPT_part1_part2_l824_82476


namespace NUMINAMATH_GPT_possible_slopes_of_line_intersects_ellipse_l824_82472

/-- 
A line whose y-intercept is (0, 3) intersects the ellipse 4x^2 + 9y^2 = 36. 
Find all possible slopes of this line. 
-/
theorem possible_slopes_of_line_intersects_ellipse :
  (∀ m : ℝ, ∃ x : ℝ, 4 * x^2 + 9 * (m * x + 3)^2 = 36) ↔ 
  (m <= - (Real.sqrt 5) / 3 ∨ m >= (Real.sqrt 5) / 3) :=
sorry

end NUMINAMATH_GPT_possible_slopes_of_line_intersects_ellipse_l824_82472


namespace NUMINAMATH_GPT_length_of_diagonal_EG_l824_82423

theorem length_of_diagonal_EG (EF FG GH HE : ℕ) (hEF : EF = 7) (hFG : FG = 15) 
  (hGH : GH = 7) (hHE : HE = 7) (primeEG : Prime EG) : EG = 11 ∨ EG = 13 :=
by
  -- Apply conditions and proof steps here
  sorry

end NUMINAMATH_GPT_length_of_diagonal_EG_l824_82423


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_value_l824_82442

theorem arithmetic_sequence_a6_value (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_roots : ∀ x, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) :
  a 6 = -6 :=
by
  -- Definitions and given conditions would go here in a fully elaborated proof.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_value_l824_82442


namespace NUMINAMATH_GPT_fruit_costs_l824_82415

theorem fruit_costs (
    A O B : ℝ
) (h1 : O = A + 0.28)
  (h2 : B = A - 0.15)
  (h3 : 3 * A + 7 * O + 5 * B = 7.84) :
  A = 0.442 ∧ O = 0.722 ∧ B = 0.292 :=
by
  -- The proof is omitted here; replacing with sorry for now
  sorry

end NUMINAMATH_GPT_fruit_costs_l824_82415


namespace NUMINAMATH_GPT_inequality_solution_l824_82463

theorem inequality_solution (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x > 1) ∧ 
  (0 < a ∧ a < 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x < 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_l824_82463


namespace NUMINAMATH_GPT_first_place_prize_is_200_l824_82492

-- Define the conditions from the problem
def total_prize_money : ℤ := 800
def num_winners : ℤ := 18
def second_place_prize : ℤ := 150
def third_place_prize : ℤ := 120
def fourth_to_eighteenth_prize : ℤ := 22
def fourth_to_eighteenth_winners : ℤ := num_winners - 3

-- Define the amount awarded to fourth to eighteenth place winners
def total_fourth_to_eighteenth_prize : ℤ := fourth_to_eighteenth_winners * fourth_to_eighteenth_prize

-- Define the total amount awarded to second and third place winners
def total_second_and_third_prize : ℤ := second_place_prize + third_place_prize

-- Define the total amount awarded to second to eighteenth place winners
def total_second_to_eighteenth_prize : ℤ := total_fourth_to_eighteenth_prize + total_second_and_third_prize

-- Define the amount awarded to first place
def first_place_prize : ℤ := total_prize_money - total_second_to_eighteenth_prize

-- Statement for proof required
theorem first_place_prize_is_200 : first_place_prize = 200 :=
by
  -- Assuming the conditions are correct
  sorry

end NUMINAMATH_GPT_first_place_prize_is_200_l824_82492


namespace NUMINAMATH_GPT_length_of_platform_l824_82440

theorem length_of_platform (L : ℕ) :
  (∀ (V : ℚ), V = 600 / 52 → V = (600 + L) / 78) → L = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l824_82440


namespace NUMINAMATH_GPT_race_runners_l824_82434

theorem race_runners (n : ℕ) (h1 : 5 * 8 + (n - 5) * 10 = 70) : n = 8 :=
sorry

end NUMINAMATH_GPT_race_runners_l824_82434


namespace NUMINAMATH_GPT_cricket_player_average_l824_82426

theorem cricket_player_average
  (A : ℕ)
  (h1 : 8 * A + 96 = 9 * (A + 8)) :
  A = 24 :=
by
  sorry

end NUMINAMATH_GPT_cricket_player_average_l824_82426


namespace NUMINAMATH_GPT_find_general_formula_l824_82475

theorem find_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h₀ : n > 0)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, S (n + 1) = 2 * S n + n + 1)
  (h₃ : ∀ n, S (n + 1) - S n = a (n + 1)) :
  a n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_find_general_formula_l824_82475


namespace NUMINAMATH_GPT_problem_solution_l824_82425

theorem problem_solution :
  ∀ p q : ℝ, (3 * p ^ 2 - 5 * p - 21 = 0) → (3 * q ^ 2 - 5 * q - 21 = 0) →
  (9 * p ^ 3 - 9 * q ^ 3) * (p - q)⁻¹ = 88 :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l824_82425


namespace NUMINAMATH_GPT_correct_graph_for_race_l824_82412

-- Define the conditions for the race.
def tortoise_constant_speed (d t : ℝ) := 
  ∃ k : ℝ, k > 0 ∧ d = k * t

def hare_behavior (d t t_nap t_end d_nap : ℝ) :=
  ∃ k1 k2 : ℝ, k1 > 0 ∧ k2 > 0 ∧ t_nap > 0 ∧ t_end > t_nap ∧
  (d = k1 * t ∨ (t_nap < t ∧ t < t_end ∧ d = d_nap) ∨ (t_end ≥ t ∧ d = d_nap + k2 * (t - t_end)))

-- Define the competition outcome.
def tortoise_wins (d_tortoise d_hare : ℝ) :=
  d_tortoise > d_hare

-- Proof that the graph which describes the race is Option (B).
theorem correct_graph_for_race :
  ∃ d_t d_h t t_nap t_end d_nap, 
    tortoise_constant_speed d_t t ∧ hare_behavior d_h t t_nap t_end d_nap ∧ tortoise_wins d_t d_h → "Option B" = "correct" :=
sorry -- Proof omitted.

end NUMINAMATH_GPT_correct_graph_for_race_l824_82412


namespace NUMINAMATH_GPT_yoongi_age_l824_82433

theorem yoongi_age
  (H Y : ℕ)
  (h1 : Y = H - 2)
  (h2 : Y + H = 18) :
  Y = 8 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_age_l824_82433


namespace NUMINAMATH_GPT_boots_cost_more_l824_82445

theorem boots_cost_more (S B : ℝ) 
  (h1 : 22 * S + 16 * B = 460) 
  (h2 : 8 * S + 32 * B = 560) : B - S = 5 :=
by
  -- Here we provide the statement only, skipping the proof
  sorry

end NUMINAMATH_GPT_boots_cost_more_l824_82445


namespace NUMINAMATH_GPT_geometric_sequence_s6_s4_l824_82499

section GeometricSequence

variables {a : ℕ → ℝ} {a1 : ℝ} {q : ℝ}
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = a1 * (1 - q^(n + 1)) / (1 - q))
variable (h_ratio : S 4 / S 2 = 3)

theorem geometric_sequence_s6_s4 :
  S 6 / S 4 = 7 / 3 :=
sorry

end GeometricSequence

end NUMINAMATH_GPT_geometric_sequence_s6_s4_l824_82499


namespace NUMINAMATH_GPT_sum4_l824_82437

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := sorry
noncomputable def gamma : ℂ := sorry

axiom sum1 : alpha + beta + gamma = 1
axiom sum2 : alpha^2 + beta^2 + gamma^2 = 5
axiom sum3 : alpha^3 + beta^3 + gamma^3 = 9

theorem sum4 : alpha^4 + beta^4 + gamma^4 = 56 := by
  sorry

end NUMINAMATH_GPT_sum4_l824_82437


namespace NUMINAMATH_GPT_sum_min_values_eq_zero_l824_82468

-- Definitions of the polynomials
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c*x + d

-- Main theorem statement
theorem sum_min_values_eq_zero (b d : ℝ) :
  let a := -16
  let c := -8
  (-64 + b = 0) ∧ (-16 + d = 0) → (-64 + b + (-16 + d) = 0) :=
by
  intros
  rw [add_assoc]
  sorry

end NUMINAMATH_GPT_sum_min_values_eq_zero_l824_82468


namespace NUMINAMATH_GPT_patty_heavier_before_losing_weight_l824_82403

theorem patty_heavier_before_losing_weight {w_R w_P w_P' x : ℝ}
  (h1 : w_R = 100)
  (h2 : w_P = 100 * x)
  (h3 : w_P' = w_P - 235)
  (h4 : w_P' = w_R + 115) :
  x = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_patty_heavier_before_losing_weight_l824_82403


namespace NUMINAMATH_GPT_fraction_of_students_with_partner_l824_82457

theorem fraction_of_students_with_partner
  (a b : ℕ)
  (condition1 : ∀ seventh, seventh ≠ 0 → ∀ tenth, tenth ≠ 0 → a * b = 0)
  (condition2 : b / 4 = (3 * a) / 7) :
  (b / 4 + 3 * a / 7) / (b + a) = 6 / 19 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_partner_l824_82457


namespace NUMINAMATH_GPT_platform_length_proof_l824_82418

noncomputable def train_length : ℝ := 480

noncomputable def speed_kmph : ℝ := 55

noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

noncomputable def crossing_time : ℝ := 71.99424046076314

noncomputable def total_distance_covered : ℝ := speed_mps * crossing_time

noncomputable def platform_length : ℝ := total_distance_covered - train_length

theorem platform_length_proof : platform_length = 620 := by
  sorry

end NUMINAMATH_GPT_platform_length_proof_l824_82418


namespace NUMINAMATH_GPT_factor_polynomial_l824_82448

theorem factor_polynomial : 
  (∀ x : ℝ, (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)) :=
by
  intro x
  -- Left-hand side
  let lhs := x^2 + 6 * x + 9 - 64 * x^4
  -- Right-hand side after factorization
  let rhs := (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)
  -- Prove the equality
  show lhs = rhs
  sorry

end NUMINAMATH_GPT_factor_polynomial_l824_82448


namespace NUMINAMATH_GPT_find_covered_number_l824_82413

theorem find_covered_number (a x : ℤ) (h : (x - a) / 2 = x + 3) (hx : x = -7) : a = 1 := by
  sorry

end NUMINAMATH_GPT_find_covered_number_l824_82413


namespace NUMINAMATH_GPT_find_a_of_ellipse_foci_l824_82483

theorem find_a_of_ellipse_foci (a : ℝ) :
  (∀ x y : ℝ, a^2 * x^2 - (a / 2) * y^2 = 1) →
  (a^2 - (2 / a) = 4) →
  a = (1 - Real.sqrt 5) / 4 :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_a_of_ellipse_foci_l824_82483


namespace NUMINAMATH_GPT_largest_possible_integer_in_list_l824_82498

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
  (a = 6) ∧ 
  (b = 6) ∧ 
  (c = 7) ∧ 
  (∀ x, x ≠ a ∨ x ≠ b ∨ x ≠ c → x ≠ 6) ∧ 
  (d > 7) ∧ 
  (12 = (a + b + c + d + e) / 5) ∧ 
  (max a (max b (max c (max d e))) = 33) := by
  sorry

end NUMINAMATH_GPT_largest_possible_integer_in_list_l824_82498


namespace NUMINAMATH_GPT_length_of_AB_l824_82496

theorem length_of_AB 
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 8)
  (hB : B.1 ^ 2 + B.2 ^ 2 = 8)
  (lA : A.1 - 2 * A.2 + 5 = 0)
  (lB : B.1 - 2 * B.2 + 5 = 0) :
  dist A B = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_length_of_AB_l824_82496


namespace NUMINAMATH_GPT_bears_in_stock_initially_l824_82455

theorem bears_in_stock_initially 
  (shipment_bears : ℕ) (shelf_bears : ℕ) (shelves_used : ℕ)
  (total_bears_shelved : shipment_bears + shelf_bears * shelves_used = 24) : 
  (24 - shipment_bears = 6) :=
by
  exact sorry

end NUMINAMATH_GPT_bears_in_stock_initially_l824_82455


namespace NUMINAMATH_GPT_XF_XG_value_l824_82451

-- Define the given conditions
noncomputable def AB := 4
noncomputable def BC := 3
noncomputable def CD := 7
noncomputable def DA := 9

noncomputable def DX (BD : ℚ) := (1 / 3) * BD
noncomputable def BY (BD : ℚ) := (1 / 4) * BD

-- Variables and points in the problem
variables (BD p q : ℚ)
variables (A B C D X Y E F G : Point)

-- Proof statement
theorem XF_XG_value 
(AB_eq : AB = 4) (BC_eq : BC = 3) (CD_eq : CD = 7) (DA_eq : DA = 9)
(DX_eq : DX BD = (1 / 3) * BD) (BY_eq : BY BD = (1 / 4) * BD)
(AC_BD_prod : p * q = 55) :
  XF * XG = (110 / 9) := 
by
  sorry

end NUMINAMATH_GPT_XF_XG_value_l824_82451


namespace NUMINAMATH_GPT_geometric_sequence_a8_value_l824_82485

variable {a : ℕ → ℕ}

-- Assuming a is a geometric sequence, provide the condition a_3 * a_9 = 4 * a_4
def geometric_sequence_condition (a : ℕ → ℕ) :=
  (a 3) * (a 9) = 4 * (a 4)

-- Prove that a_8 = 4 under the given condition
theorem geometric_sequence_a8_value (a : ℕ → ℕ) (h : geometric_sequence_condition a) : a 8 = 4 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_a8_value_l824_82485


namespace NUMINAMATH_GPT_difference_of_M_and_m_l824_82401

-- Define the variables and conditions
def total_students : ℕ := 2500
def min_G : ℕ := 1750
def max_G : ℕ := 1875
def min_R : ℕ := 1000
def max_R : ℕ := 1125

-- The statement to prove
theorem difference_of_M_and_m : 
  ∃ G R m M, 
  (G = total_students - R + m) ∧ 
  (min_G ≤ G ∧ G ≤ max_G) ∧
  (min_R ≤ R ∧ R ≤ max_R) ∧
  (m = min_G + min_R - total_students) ∧
  (M = max_G + max_R - total_students) ∧
  (M - m = 250) :=
sorry

end NUMINAMATH_GPT_difference_of_M_and_m_l824_82401


namespace NUMINAMATH_GPT_fraction_of_64_l824_82435

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end NUMINAMATH_GPT_fraction_of_64_l824_82435


namespace NUMINAMATH_GPT_cube_of_99999_is_correct_l824_82400

theorem cube_of_99999_is_correct : (99999 : ℕ)^3 = 999970000299999 :=
by
  sorry

end NUMINAMATH_GPT_cube_of_99999_is_correct_l824_82400


namespace NUMINAMATH_GPT_consecutive_integer_sum_l824_82438

theorem consecutive_integer_sum (a b c : ℕ) 
  (h1 : b = a + 2) 
  (h2 : c = a + 4) 
  (h3 : a + c = 140) 
  (h4 : b - a = 2) : a + b + c = 210 := 
sorry

end NUMINAMATH_GPT_consecutive_integer_sum_l824_82438


namespace NUMINAMATH_GPT_same_answer_l824_82431

structure Person :=
(name : String)
(tellsTruth : Bool)

def Fedya : Person :=
{ name := "Fedya",
  tellsTruth := true }

def Vadim : Person :=
{ name := "Vadim",
  tellsTruth := false }

def question (p : Person) (q : String) : Bool :=
if p.tellsTruth then q = p.name else q ≠ p.name

theorem same_answer (q : String) :
  (question Fedya q = question Vadim q) :=
sorry

end NUMINAMATH_GPT_same_answer_l824_82431
