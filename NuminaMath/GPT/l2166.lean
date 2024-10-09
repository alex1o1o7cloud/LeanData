import Mathlib

namespace find_second_number_l2166_216675

theorem find_second_number (a b c : ℝ) (h1 : a + b + c = 3.622) (h2 : a = 3.15) (h3 : c = 0.458) : b = 0.014 :=
sorry

end find_second_number_l2166_216675


namespace probability_of_green_ball_is_2_over_5_l2166_216689

noncomputable def container_probabilities : ℚ :=
  let prob_A_selected : ℚ := 1/2
  let prob_B_selected : ℚ := 1/2
  let prob_green_in_A : ℚ := 5/10
  let prob_green_in_B : ℚ := 3/10

  prob_A_selected * prob_green_in_A + prob_B_selected * prob_green_in_B

theorem probability_of_green_ball_is_2_over_5 :
  container_probabilities = 2 / 5 := by
  sorry

end probability_of_green_ball_is_2_over_5_l2166_216689


namespace count_possible_pairs_l2166_216616

/-- There are four distinct mystery novels, three distinct fantasy novels, and three distinct biographies.
I want to choose two books with one of them being a specific mystery novel, "Mystery Masterpiece".
Prove that the number of possible pairs that include this mystery novel and one book from a different genre
is 6. -/
theorem count_possible_pairs (mystery_novels : Fin 4)
                            (fantasy_novels : Fin 3)
                            (biographies : Fin 3)
                            (MysteryMasterpiece : Fin 4):
                            (mystery_novels ≠ MysteryMasterpiece) →
                            ∀ genre : Fin 2, genre ≠ 0 ∧ genre ≠ 1 →
                            (genre = 1 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            (genre = 2 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            ∃ total_pairs : Nat, total_pairs = 6 :=
by
  intros h_ne_genres h_genres h_counts1 h_counts2
  sorry

end count_possible_pairs_l2166_216616


namespace gwen_total_books_l2166_216676

theorem gwen_total_books
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (mystery_shelves_count : mystery_shelves = 3)
  (picture_shelves_count : picture_shelves = 5)
  (each_shelf_books : books_per_shelf = 9) :
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf) = 72 := by
  sorry

end gwen_total_books_l2166_216676


namespace find_m_n_l2166_216696

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := x^3 + m * x^2 + n * x + 1

theorem find_m_n (m n : ℝ) (x : ℝ) (hx : x ≠ 0 ∧ f x m n = 1 ∧ (3 * x^2 + 2 * m * x + n = 0) ∧ (∀ y, f y m n ≥ -31 ∧ f (-2) m n = -31)) :
  m = 12 ∧ n = 36 :=
sorry

end find_m_n_l2166_216696


namespace speed_of_stream_l2166_216699

/-- Given Athul's rowing conditions, prove the speed of the stream is 1 km/h. -/
theorem speed_of_stream 
  (A S : ℝ)
  (h1 : 16 = (A - S) * 4)
  (h2 : 24 = (A + S) * 4) : 
  S = 1 := 
sorry

end speed_of_stream_l2166_216699


namespace quadratic_points_order_l2166_216674

theorem quadratic_points_order (c y1 y2 : ℝ) 
  (hA : y1 = 0^2 - 6 * 0 + c)
  (hB : y2 = 4^2 - 6 * 4 + c) : 
  y1 > y2 := 
by 
  sorry

end quadratic_points_order_l2166_216674


namespace worker_total_amount_l2166_216658

-- Definitions of the conditions
def pay_per_day := 20
def deduction_per_idle_day := 3
def total_days := 60
def idle_days := 40
def worked_days := total_days - idle_days
def earnings := worked_days * pay_per_day
def deductions := idle_days * deduction_per_idle_day

-- Statement of the problem
theorem worker_total_amount : earnings - deductions = 280 := by
  sorry

end worker_total_amount_l2166_216658


namespace difference_of_sums_1500_l2166_216668

def sum_of_first_n_odd_numbers (n : ℕ) : ℕ :=
  n * n

def sum_of_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_of_sums_1500 :
  sum_of_first_n_even_numbers 1500 - sum_of_first_n_odd_numbers 1500 = 1500 :=
by
  sorry

end difference_of_sums_1500_l2166_216668


namespace integer_triangle_cosines_rational_l2166_216633

theorem integer_triangle_cosines_rational (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  ∃ (cos_α cos_β cos_γ : ℚ), 
    cos_γ = (a^2 + b^2 - c^2) / (2 * a * b) ∧
    cos_β = (a^2 + c^2 - b^2) / (2 * a * c) ∧
    cos_α = (b^2 + c^2 - a^2) / (2 * b * c) :=
by
  sorry

end integer_triangle_cosines_rational_l2166_216633


namespace calculate_savings_l2166_216620

theorem calculate_savings :
  let plane_cost : ℕ := 600
  let boat_cost : ℕ := 254
  plane_cost - boat_cost = 346 := by
    let plane_cost : ℕ := 600
    let boat_cost : ℕ := 254
    sorry

end calculate_savings_l2166_216620


namespace problem_part1_problem_part2_l2166_216656

noncomputable def arithmetic_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + 2

theorem problem_part1 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) :
  a 2 = 4 := 
sorry

theorem problem_part2 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) 
  (h4 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2) :
  S 10 = 110 :=
sorry

end problem_part1_problem_part2_l2166_216656


namespace gum_total_l2166_216623

theorem gum_total (initial_gum : ℝ) (additional_gum : ℝ) : initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 :=
by
  intros
  sorry

end gum_total_l2166_216623


namespace sum_mnp_l2166_216606

noncomputable def volume_of_parallelepiped := 2 * 3 * 4
noncomputable def volume_of_extended_parallelepipeds := 
  2 * (1 * 2 * 3 + 1 * 2 * 4 + 1 * 3 * 4)
noncomputable def volume_of_quarter_cylinders := 
  4 * (1 / 4 * Real.pi * 1^2 * (2 + 3 + 4))
noncomputable def volume_of_spherical_octants := 
  8 * (1 / 8 * (4 / 3) * Real.pi * 1^3)

noncomputable def total_volume := 
  volume_of_parallelepiped + volume_of_extended_parallelepipeds + 
  volume_of_quarter_cylinders + volume_of_spherical_octants

theorem sum_mnp : 228 + 85 + 3 = 316 := by
  sorry

end sum_mnp_l2166_216606


namespace ratio_not_necessarily_constant_l2166_216602

theorem ratio_not_necessarily_constant (x y : ℝ) : ¬ (∃ k : ℝ, ∀ x y, x / y = k) :=
by
  sorry

end ratio_not_necessarily_constant_l2166_216602


namespace sum_of_three_consecutive_odd_integers_l2166_216654

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end sum_of_three_consecutive_odd_integers_l2166_216654


namespace gcd_294_84_l2166_216619

theorem gcd_294_84 : gcd 294 84 = 42 :=
by
  sorry

end gcd_294_84_l2166_216619


namespace mrs_sheridan_fish_distribution_l2166_216698

theorem mrs_sheridan_fish_distribution :
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium
  fish_in_large_aquarium = 225 :=
by {
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium

  have : fish_in_large_aquarium = 225 := by sorry
  exact this
}

end mrs_sheridan_fish_distribution_l2166_216698


namespace total_trash_pieces_l2166_216629

theorem total_trash_pieces (classroom_trash : ℕ) (outside_trash : ℕ)
  (h1 : classroom_trash = 344) (h2 : outside_trash = 1232) : 
  classroom_trash + outside_trash = 1576 :=
by
  sorry

end total_trash_pieces_l2166_216629


namespace bacteria_population_l2166_216626

theorem bacteria_population (initial_population : ℕ) (tripling_factor : ℕ) (hours_per_tripling : ℕ) (target_population : ℕ) 
(initial_population_eq : initial_population = 300)
(tripling_factor_eq : tripling_factor = 3)
(hours_per_tripling_eq : hours_per_tripling = 5)
(target_population_eq : target_population = 87480) :
∃ n : ℕ, (hours_per_tripling * n = 30) ∧ (initial_population * (tripling_factor ^ n) ≥ target_population) := sorry

end bacteria_population_l2166_216626


namespace arithmetic_sequence_common_difference_l2166_216621

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Conditions
def condition1 : Prop := ∀ n, S n = (n * (2*a 1 + (n-1) * d)) / 2
def condition2 : Prop := S 3 = 6
def condition3 : Prop := a 3 = 0

-- Question
def question : ℝ := d

-- Correct Answer
def correct_answer : ℝ := -2

-- Proof Problem Statement
theorem arithmetic_sequence_common_difference : 
  condition1 a S d ∧ condition2 S ∧ condition3 a →
  question d = correct_answer :=
sorry

end arithmetic_sequence_common_difference_l2166_216621


namespace find_k_l2166_216691

theorem find_k : ∀ (x y k : ℤ), (x = -y) → (2 * x + 5 * y = k) → (x - 3 * y = 16) → (k = -12) :=
by
  intros x y k h1 h2 h3
  sorry

end find_k_l2166_216691


namespace four_people_pairing_l2166_216687

theorem four_people_pairing
    (persons : Fin 4 → Type)
    (common_language : ∀ (i j : Fin 4), Prop)
    (communicable : ∀ (i j k : Fin 4), common_language i j ∨ common_language j k ∨ common_language k i)
    : ∃ (i j : Fin 4) (k l : Fin 4), i ≠ j ∧ k ≠ l ∧ common_language i j ∧ common_language k l := 
sorry

end four_people_pairing_l2166_216687


namespace tank_capacity_l2166_216690

noncomputable def inflow_A (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_B (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_C (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_X (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_Y (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

theorem tank_capacity
  (fA : ℕ := inflow_A 8 7)
  (fB : ℕ := inflow_B 12 3)
  (fC : ℕ := inflow_C 6 4)
  (oX : ℕ := outflow_X 20 7)
  (oY : ℕ := outflow_Y 15 5) :
  fA + fB + fC = 6960 ∧ oX + oY = 12900 ∧ 12900 - 6960 = 5940 :=
by
  sorry

end tank_capacity_l2166_216690


namespace solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l2166_216624

-- Define the function f(x) and g(x)
def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Define the inequality problem when a = 2
theorem solution_set_for_f_when_a_2 : 
  { x : ℝ | f x 2 ≤ 6 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

-- Prove the range of values for a when f(x) + g(x) ≥ 3
theorem range_of_a_for_f_plus_g_ge_3 : 
  ∀ a : ℝ, (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a :=
by
  sorry

end solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l2166_216624


namespace tetrahedron_ratio_l2166_216660

theorem tetrahedron_ratio (a b c d : ℝ) (h₁ : a^2 = b^2 + c^2) (h₂ : b^2 = a^2 + d^2) (h₃ : c^2 = a^2 + b^2) : 
  a / d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end tetrahedron_ratio_l2166_216660


namespace compare_negative_fractions_l2166_216697

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l2166_216697


namespace count_three_digit_perfect_squares_l2166_216695

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_perfect_squares : 
  ∃ (count : ℕ), count = 22 ∧
  ∀ (n : ℕ), is_three_digit_number n → is_perfect_square n → true :=
sorry

end count_three_digit_perfect_squares_l2166_216695


namespace perfect_cubes_in_range_l2166_216646

theorem perfect_cubes_in_range (K : ℤ) (hK_pos : K > 1) (Z : ℤ) 
  (hZ_eq : Z = K ^ 3) (hZ_range: 600 < Z ∧ Z < 2000) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 :=
by
  sorry

end perfect_cubes_in_range_l2166_216646


namespace remainder_of_3_pow_500_mod_17_l2166_216612

theorem remainder_of_3_pow_500_mod_17 : (3 ^ 500) % 17 = 13 := 
by
  sorry

end remainder_of_3_pow_500_mod_17_l2166_216612


namespace least_possible_integral_QR_l2166_216671

theorem least_possible_integral_QR (PQ PR SR SQ QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 10) (hSR : SR = 15) (hSQ : SQ = 24) :
  9 ≤ QR ∧ QR < 17 :=
by
  sorry

end least_possible_integral_QR_l2166_216671


namespace probability_point_between_C_and_E_l2166_216677

noncomputable def length_between_points (total_length : ℝ) (ratio : ℝ) : ℝ :=
ratio * total_length

theorem probability_point_between_C_and_E
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : C < E)
  (h3 : B - A = 4 * (D - A))
  (h4 : B - A = 8 * (B - C))
  (h5 : B - E = 2 * (E - C)) :
  (E - C) / (B - A) = 1 / 24 :=
by 
  sorry

end probability_point_between_C_and_E_l2166_216677


namespace print_shop_cost_difference_l2166_216625

theorem print_shop_cost_difference :
  let cost_per_copy_X := 1.25
  let cost_per_copy_Y := 2.75
  let num_copies := 40
  let total_cost_X := cost_per_copy_X * num_copies
  let total_cost_Y := cost_per_copy_Y * num_copies
  total_cost_Y - total_cost_X = 60 :=
by 
  dsimp only []
  sorry

end print_shop_cost_difference_l2166_216625


namespace max_x1_x2_squares_l2166_216628

noncomputable def x1_x2_squares_eq_max : Prop :=
  ∃ k : ℝ, (∀ x1 x2 : ℝ, (x1 + x2 = k - 2) ∧ (x1 * x2 = k^2 + 3 * k + 5) → x1^2 + x2^2 = 18)

theorem max_x1_x2_squares : x1_x2_squares_eq_max :=
by sorry

end max_x1_x2_squares_l2166_216628


namespace sum_of_squares_transform_l2166_216638

def isSumOfThreeSquaresDivByThree (N : ℕ) : Prop := 
  ∃ (a b c : ℤ), N = a^2 + b^2 + c^2 ∧ (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c)

def isSumOfThreeSquaresNotDivByThree (N : ℕ) : Prop := 
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z)

theorem sum_of_squares_transform {N : ℕ} :
  isSumOfThreeSquaresDivByThree N → isSumOfThreeSquaresNotDivByThree N :=
sorry

end sum_of_squares_transform_l2166_216638


namespace car_miles_per_gallon_in_city_l2166_216610

-- Define the conditions and the problem
theorem car_miles_per_gallon_in_city :
  ∃ C H T : ℝ, 
    H = 462 / T ∧ 
    C = 336 / T ∧ 
    C = H - 12 ∧ 
    C = 32 :=
by
  sorry

end car_miles_per_gallon_in_city_l2166_216610


namespace problem1_problem2_l2166_216609

noncomputable section

theorem problem1 :
  (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 :=
  sorry

theorem problem2 :
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1 / 2) = -6 * Real.sqrt 5 :=
  sorry

end problem1_problem2_l2166_216609


namespace last_digit_square_of_second_l2166_216666

def digit1 := 1
def digit2 := 3
def digit3 := 4
def digit4 := 9

theorem last_digit_square_of_second :
  digit4 = digit2 ^ 2 :=
by
  -- Conditions
  have h1 : digit1 = digit2 / 3 := by sorry
  have h2 : digit3 = digit1 + digit2 := by sorry
  sorry

end last_digit_square_of_second_l2166_216666


namespace minimum_value_sum_l2166_216692

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / (3 * b) + b / (5 * c) + c / (6 * a)) >= (3 / (90^(1/3))) :=
by 
  sorry

end minimum_value_sum_l2166_216692


namespace correct_operation_l2166_216630

theorem correct_operation (x : ℝ) (hx : x ≠ 0) :
  (x^3 / x^2 = x) :=
by {
  sorry
}

end correct_operation_l2166_216630


namespace percentage_of_girls_after_change_l2166_216618

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end percentage_of_girls_after_change_l2166_216618


namespace negation_of_homework_submission_l2166_216614

variable {S : Type} -- S is the set of all students in this class
variable (H : S → Prop) -- H(x) means "student x has submitted the homework"

theorem negation_of_homework_submission :
  (¬ ∀ x, H x) ↔ (∃ x, ¬ H x) :=
by
  sorry

end negation_of_homework_submission_l2166_216614


namespace count_integer_values_l2166_216683

theorem count_integer_values (π : Real) (hπ : Real.pi = π):
  ∃ n : ℕ, n = 27 ∧ ∀ x : ℤ, |(x:Real)| < 4 * π + 1 ↔ -13 ≤ x ∧ x ≤ 13 :=
by sorry

end count_integer_values_l2166_216683


namespace Robert_books_read_in_six_hours_l2166_216663

theorem Robert_books_read_in_six_hours (P H T: ℕ)
    (h1: P = 270)
    (h2: H = 90)
    (h3: T = 6):
    T * H / P = 2 :=
by 
    -- sorry placeholder to indicate that this is where the proof goes.
    sorry

end Robert_books_read_in_six_hours_l2166_216663


namespace set_intersection_example_l2166_216640

def universal_set := Set ℝ

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1 ∧ -2 ≤ x ∧ x ≤ 1}

def C : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

def complement (A : Set ℝ) : Set ℝ := {x : ℝ | x ∉ A}

def difference (A B : Set ℝ) : Set ℝ := A \ B

def union (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∨ x ∈ B}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection_example :
  intersection (complement A) (union B C) = {x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 4)} :=
by
  sorry

end set_intersection_example_l2166_216640


namespace Richard_walked_10_miles_third_day_l2166_216613

def distance_to_NYC := 70
def day1 := 20
def day2 := (day1 / 2) - 6
def remaining_distance := 36
def day3 := 70 - (day1 + day2 + remaining_distance)

theorem Richard_walked_10_miles_third_day (h : day3 = 10) : day3 = 10 :=
by {
    sorry
}

end Richard_walked_10_miles_third_day_l2166_216613


namespace find_original_number_l2166_216670

-- Define the given conditions
def increased_by_twenty_percent (x : ℝ) : ℝ := x * 1.20

-- State the theorem
theorem find_original_number (x : ℝ) (h : increased_by_twenty_percent x = 480) : x = 400 :=
by
  sorry

end find_original_number_l2166_216670


namespace evaluate_expression_l2166_216645

noncomputable def a : ℕ := 3^2 + 5^2 + 7^2
noncomputable def b : ℕ := 2^2 + 4^2 + 6^2

theorem evaluate_expression : (a / b : ℚ) - (b / a : ℚ) = 3753 / 4656 :=
by
  sorry

end evaluate_expression_l2166_216645


namespace seats_usually_taken_l2166_216681

def total_tables : Nat := 15
def seats_per_table : Nat := 10
def proportion_left_unseated : Rat := 1 / 10
def proportion_taken : Rat := 1 - proportion_left_unseated

theorem seats_usually_taken :
  proportion_taken * (total_tables * seats_per_table) = 135 := by
  sorry

end seats_usually_taken_l2166_216681


namespace sequence_general_formula_l2166_216672

theorem sequence_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 2 = 4 →
  S 4 = 30 →
  (∀ n, n ≥ 2 → a (n + 1) + a (n - 1) = 2 * (a n + 1)) →
  ∀ n, a n = n^2 :=
by
  intros h1 h2 h3
  sorry

end sequence_general_formula_l2166_216672


namespace sum_digits_n_plus_one_l2166_216657

/-- 
Let S(n) be the sum of the digits of a positive integer n.
Given S(n) = 29, prove that the possible values of S(n + 1) are 3, 12, or 30.
-/
theorem sum_digits_n_plus_one (S : ℕ → ℕ) (n : ℕ) (h : S n = 29) :
  S (n + 1) = 3 ∨ S (n + 1) = 12 ∨ S (n + 1) = 30 := 
sorry

end sum_digits_n_plus_one_l2166_216657


namespace solve_fraction_eq_l2166_216652

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x = -1) ↔ ((x^2 + 2 * x + 3) / (x + 2) = x + 3) := 
by 
  sorry

end solve_fraction_eq_l2166_216652


namespace problem1_problem2_l2166_216642

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Define the conditions and questions as Lean statements

-- First problem: Prove that if A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem problem1 (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := 
  sorry

-- Second problem: Prove that if A a ⊆ B, then a ∈ (-∞, 0] ∪ [4, ∞)
theorem problem2 (a : ℝ) (h1 : A a ⊆ B) : a ≤ 0 ∨ a ≥ 4 := 
  sorry

end problem1_problem2_l2166_216642


namespace ford_younger_than_christopher_l2166_216639

variable (G C F Y : ℕ)

-- Conditions
axiom h1 : G = C + 8
axiom h2 : F = C - Y
axiom h3 : G + C + F = 60
axiom h4 : C = 18

-- Target statement
theorem ford_younger_than_christopher : Y = 2 :=
sorry

end ford_younger_than_christopher_l2166_216639


namespace profit_percent_300_l2166_216686

theorem profit_percent_300 (SP : ℝ) (h : SP ≠ 0) (CP : ℝ) (h1 : CP = 0.25 * SP) : 
  (SP - CP) / CP * 100 = 300 := 
  sorry

end profit_percent_300_l2166_216686


namespace smallest_number_append_l2166_216604

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l2166_216604


namespace hours_per_day_for_first_group_l2166_216607

theorem hours_per_day_for_first_group (h : ℕ) :
  (39 * h * 12 = 30 * 6 * 26) → h = 10 :=
by
  sorry

end hours_per_day_for_first_group_l2166_216607


namespace earnings_bc_l2166_216667

variable (A B C : ℕ)

theorem earnings_bc :
  A + B + C = 600 →
  A + C = 400 →
  C = 100 →
  B + C = 300 :=
by
  intros h1 h2 h3
  sorry

end earnings_bc_l2166_216667


namespace find_t_l2166_216627

variable {x y z w t : ℝ}

theorem find_t (hx : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
               (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
               (hxy : x + 1/y = t)
               (hyz : y + 1/z = t)
               (hzw : z + 1/w = t)
               (hwx : w + 1/x = t) : 
               t = Real.sqrt 2 :=
by
  sorry

end find_t_l2166_216627


namespace new_average_is_15_l2166_216634

-- Definitions corresponding to the conditions
def avg_10_consecutive (seq : List ℤ) : Prop :=
  seq.length = 10 ∧ seq.sum = 200

def new_seq (seq : List ℤ) : List ℤ :=
  List.mapIdx (λ i x => x - ↑(9 - i)) seq

-- Statement of the proof problem
theorem new_average_is_15
  (seq : List ℤ)
  (h_seq : avg_10_consecutive seq) :
  (new_seq seq).sum = 150 := sorry

end new_average_is_15_l2166_216634


namespace feathers_per_crown_l2166_216631

theorem feathers_per_crown (total_feathers total_crowns feathers_per_crown : ℕ) 
  (h₁ : total_feathers = 6538) 
  (h₂ : total_crowns = 934) 
  (h₃ : feathers_per_crown = total_feathers / total_crowns) : 
  feathers_per_crown = 7 := 
by 
  sorry

end feathers_per_crown_l2166_216631


namespace max_value_a_plus_2b_l2166_216661

theorem max_value_a_plus_2b {a b : ℝ} (h_positive : 0 < a ∧ 0 < b) (h_eqn : a^2 + 2 * a * b + 4 * b^2 = 6) :
  a + 2 * b ≤ 2 * Real.sqrt 2 :=
sorry

end max_value_a_plus_2b_l2166_216661


namespace max_value_expr_l2166_216664

open Real

noncomputable def expr (x : ℝ) : ℝ :=
  (x^4 + 3 * x^2 - sqrt (x^8 + 9)) / x^2

theorem max_value_expr : ∀ (x y : ℝ), (0 < x) → (y = x + 1 / x) → expr x = 15 / 7 :=
by
  intros x y hx hy
  sorry

end max_value_expr_l2166_216664


namespace chef_earns_less_than_manager_l2166_216649

noncomputable def manager_wage : ℝ := 6.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage + 0.2 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 2.60 :=
by
  sorry

end chef_earns_less_than_manager_l2166_216649


namespace square_field_diagonal_l2166_216653

theorem square_field_diagonal (a : ℝ) (d : ℝ) (h : a^2 = 800) : d = 40 :=
by
  sorry

end square_field_diagonal_l2166_216653


namespace total_time_for_5_smoothies_l2166_216650

-- Definitions for the conditions
def freeze_time : ℕ := 40
def blend_time_per_smoothie : ℕ := 3
def chop_time_apples_per_smoothie : ℕ := 2
def chop_time_bananas_per_smoothie : ℕ := 3
def chop_time_strawberries_per_smoothie : ℕ := 4
def chop_time_mangoes_per_smoothie : ℕ := 5
def chop_time_pineapples_per_smoothie : ℕ := 6
def number_of_smoothies : ℕ := 5

-- Total chopping time per smoothie
def chop_time_per_smoothie : ℕ := chop_time_apples_per_smoothie + 
                                  chop_time_bananas_per_smoothie + 
                                  chop_time_strawberries_per_smoothie + 
                                  chop_time_mangoes_per_smoothie + 
                                  chop_time_pineapples_per_smoothie

-- Total chopping time for 5 smoothies
def total_chop_time : ℕ := chop_time_per_smoothie * number_of_smoothies

-- Total blending time for 5 smoothies
def total_blend_time : ℕ := blend_time_per_smoothie * number_of_smoothies

-- Total time to make 5 smoothies
def total_time : ℕ := total_chop_time + total_blend_time

-- Theorem statement
theorem total_time_for_5_smoothies : total_time = 115 := by
  sorry

end total_time_for_5_smoothies_l2166_216650


namespace negation_of_existence_statement_l2166_216603

theorem negation_of_existence_statement :
  ¬ (∃ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 ≤ 0)) ↔ ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 > 0) :=
by
  sorry

end negation_of_existence_statement_l2166_216603


namespace angle_C_is_108_l2166_216622

theorem angle_C_is_108
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : B < C)
  (h3 : C < D)
  (h4 : D < E)
  (h5 : B - A = C - B)
  (h6 : C - B = D - C)
  (h7 : D - C = E - D)
  (angle_sum : A + B + C + D + E = 540) :
  C = 108 := 
sorry

end angle_C_is_108_l2166_216622


namespace bryden_receives_amount_l2166_216665

variable (q : ℝ) (p : ℝ) (num_quarters : ℝ)

-- Define the conditions
def face_value_of_quarter : Prop := q = 0.25
def percentage_offer : Prop := p = 25 * q
def number_of_quarters : Prop := num_quarters = 5

-- Define the theorem to be proved
theorem bryden_receives_amount (h1 : face_value_of_quarter q) (h2 : percentage_offer q p) (h3 : number_of_quarters num_quarters) :
  (p * num_quarters * q) = 31.25 :=
by
  sorry

end bryden_receives_amount_l2166_216665


namespace range_of_p_nonnegative_range_of_p_all_values_range_of_p_l2166_216608

def p (x : ℝ) : ℝ := x^4 - 6 * x^2 + 9

theorem range_of_p_nonnegative (x : ℝ) (hx : 0 ≤ x) : 
  ∃ y, y = p x ∧ 0 ≤ y := 
sorry

theorem range_of_p_all_values (y : ℝ) : 
  0 ≤ y → (∃ x, 0 ≤ x ∧ p x = y) :=
sorry

theorem range_of_p (x : ℝ) (hx : 0 ≤ x) : 
  ∀ y, (∃ x, 0 ≤ x ∧ p x = y) ↔ (0 ≤ y) :=
sorry

end range_of_p_nonnegative_range_of_p_all_values_range_of_p_l2166_216608


namespace probability_two_digit_between_15_25_l2166_216611

-- Define a type for standard six-sided dice rolls
def is_standard_six_sided_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Define the set of valid two-digit numbers
def valid_two_digit_number (n : ℕ) : Prop := n ≥ 15 ∧ n ≤ 25

-- Function to form a two-digit number from two dice rolls
def form_two_digit_number (d1 d2 : ℕ) : ℕ := 10 * d1 + d2

-- The main statement of the problem
theorem probability_two_digit_between_15_25 :
  (∃ (n : ℚ), n = 5/9) ∧
  (∀ (d1 d2 : ℕ), is_standard_six_sided_die d1 → is_standard_six_sided_die d2 →
  valid_two_digit_number (form_two_digit_number d1 d2)) :=
sorry

end probability_two_digit_between_15_25_l2166_216611


namespace bees_flew_in_l2166_216673

theorem bees_flew_in (initial_bees : ℕ) (total_bees : ℕ) (new_bees : ℕ) (h1 : initial_bees = 16) (h2 : total_bees = 23) (h3 : total_bees = initial_bees + new_bees) : new_bees = 7 :=
by
  sorry

end bees_flew_in_l2166_216673


namespace simplify_expression_l2166_216641

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a)^(2 : ℝ) :=
by sorry

end simplify_expression_l2166_216641


namespace total_revenue_correct_l2166_216605

-- Define the conditions
def charge_per_slice : ℕ := 5
def slices_per_pie : ℕ := 4
def pies_sold : ℕ := 9

-- Prove the question: total revenue
theorem total_revenue_correct : charge_per_slice * slices_per_pie * pies_sold = 180 :=
by
  sorry

end total_revenue_correct_l2166_216605


namespace possible_values_x_l2166_216693

theorem possible_values_x : 
  let x := Nat.gcd 112 168 
  ∃ d : Finset ℕ, d.card = 8 ∧ ∀ y ∈ d, y ∣ 112 ∧ y ∣ 168 := 
by
  let x := Nat.gcd 112 168
  have : x = 56 := by norm_num
  use Finset.filter (fun n => 56 % n = 0) (Finset.range 57)
  sorry

end possible_values_x_l2166_216693


namespace correct_statements_proof_l2166_216662

theorem correct_statements_proof :
  (∀ (a b : ℤ), a - 3 = b - 3 → a = b) ∧
  ¬ (∀ (a b c : ℤ), a = b → a + c = b - c) ∧
  (∀ (a b m : ℤ), m ≠ 0 → (a / m) = (b / m) → a = b) ∧
  ¬ (∀ (a : ℤ), a^2 = 2 * a → a = 2) :=
by
  -- Here we would prove the statements individually:
  -- sorry is a placeholder suggesting that the proofs need to be filled in.
  sorry

end correct_statements_proof_l2166_216662


namespace total_employees_in_company_l2166_216601

-- Given facts and conditions
def ratio_A_B_C : Nat × Nat × Nat := (5, 4, 1)
def sample_size : Nat := 20
def prob_sel_A_B_from_C : ℚ := 1 / 45

-- Number of group C individuals, calculated from probability constraint
def num_persons_group_C := 10

theorem total_employees_in_company (x : Nat) :
  x = 10 * (5 + 4 + 1) :=
by
  -- Since the sample size is 20, and the ratio of sampling must be consistent with the population ratio,
  -- it can be derived that the total number of employees in the company must be 100.
  -- Adding sorry to skip the actual detailed proof.
  sorry

end total_employees_in_company_l2166_216601


namespace sum_of_three_integers_l2166_216637

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end sum_of_three_integers_l2166_216637


namespace machine_A_production_l2166_216679

-- Definitions based on the conditions
def machine_production (A B: ℝ) (TA TB: ℝ) : Prop :=
  B = 1.10 * A ∧
  TA = TB + 10 ∧
  A * TA = 660 ∧
  B * TB = 660

-- The main statement to be proved: Machine A produces 6 sprockets per hour.
theorem machine_A_production (A B: ℝ) (TA TB: ℝ) 
  (h : machine_production A B TA TB) : 
  A = 6 := 
by sorry

end machine_A_production_l2166_216679


namespace hausdorff_dimension_union_sup_l2166_216655

open Set

noncomputable def Hausdorff_dimension (A : Set ℝ) : ℝ :=
sorry -- Definition for Hausdorff dimension is nontrivial and can be added here

theorem hausdorff_dimension_union_sup {A : ℕ → Set ℝ} :
  Hausdorff_dimension (⋃ i, A i) = ⨆ i, Hausdorff_dimension (A i) :=
sorry

end hausdorff_dimension_union_sup_l2166_216655


namespace find_quotient_l2166_216635

theorem find_quotient
    (dividend divisor remainder : ℕ)
    (h1 : dividend = 136)
    (h2 : divisor = 15)
    (h3 : remainder = 1)
    (h4 : dividend = divisor * quotient + remainder) :
    quotient = 9 :=
by
  sorry

end find_quotient_l2166_216635


namespace value_of_x_l2166_216688

theorem value_of_x (x : ℚ) (h : (x + 10 + 17 + 3 * x + 15 + 3 * x + 6) / 5 = 26) : x = 82 / 7 :=
by
  sorry

end value_of_x_l2166_216688


namespace canvas_bag_lower_carbon_solution_l2166_216680

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end canvas_bag_lower_carbon_solution_l2166_216680


namespace largest_of_five_consecutive_sum_l2166_216651

theorem largest_of_five_consecutive_sum (n : ℕ) 
  (h : n + (n+1) + (n+2) + (n+3) + (n+4) = 90) : 
  n + 4 = 20 :=
sorry

end largest_of_five_consecutive_sum_l2166_216651


namespace find_x_if_vectors_parallel_l2166_216669

theorem find_x_if_vectors_parallel (x : ℝ)
  (a : ℝ × ℝ := (x - 1, 2))
  (b : ℝ × ℝ := (2, 1)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → x = 5 :=
by sorry

end find_x_if_vectors_parallel_l2166_216669


namespace tenth_term_arithmetic_sequence_l2166_216684

theorem tenth_term_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 23)
  (h2 : a + 7 * d = 55) :
  a + 9 * d = 71 :=
sorry

end tenth_term_arithmetic_sequence_l2166_216684


namespace tank_fraction_after_adding_water_l2166_216643

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (full_capacity : ℚ) 
  (added_water : ℚ) 
  (final_fraction : ℚ) 
  (h1 : initial_fraction = 3/4) 
  (h2 : full_capacity = 56) 
  (h3 : added_water = 7) 
  (h4 : final_fraction = (initial_fraction * full_capacity + added_water) / full_capacity) : 
  final_fraction = 7 / 8 := 
by 
  sorry

end tank_fraction_after_adding_water_l2166_216643


namespace one_python_can_eat_per_week_l2166_216647

-- Definitions based on the given conditions
def burmese_pythons := 5
def alligators_eaten := 15
def weeks := 3

-- Theorem statement to prove the number of alligators one python can eat per week
theorem one_python_can_eat_per_week : (alligators_eaten / burmese_pythons) / weeks = 1 := 
by 
-- sorry is used to skip the actual proof
sorry

end one_python_can_eat_per_week_l2166_216647


namespace find_n_values_l2166_216644

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def A_n_k (n k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

def every_A_n_k_prime (n : ℕ) : Prop :=
  ∀ k, k < n → is_prime (A_n_k n k)

theorem find_n_values :
  ∀ n : ℕ, every_A_n_k_prime n → n = 1 ∨ n = 2 := sorry

end find_n_values_l2166_216644


namespace cars_in_garage_l2166_216694

/-
Conditions:
1. Total wheels in the garage: 22
2. Riding lawnmower wheels: 4
3. Timmy's bicycle wheels: 2
4. Each of Timmy's parents' bicycles: 2 wheels, and there are 2 bicycles.
5. Joey's tricycle wheels: 3
6. Timmy's dad's unicycle wheels: 1

Question: How many cars are inside the garage?

Correct Answer: The number of cars is 2.
-/
theorem cars_in_garage (total_wheels : ℕ) (lawnmower_wheels : ℕ)
  (timmy_bicycle_wheels : ℕ) (parents_bicycles_wheels : ℕ)
  (joey_tricycle_wheels : ℕ) (dad_unicycle_wheels : ℕ) 
  (cars_wheels : ℕ) (cars : ℕ) :
  total_wheels = 22 →
  lawnmower_wheels = 4 →
  timmy_bicycle_wheels = 2 →
  parents_bicycles_wheels = 2 * 2 →
  joey_tricycle_wheels = 3 →
  dad_unicycle_wheels = 1 →
  cars_wheels = total_wheels - (lawnmower_wheels + timmy_bicycle_wheels + parents_bicycles_wheels + joey_tricycle_wheels + dad_unicycle_wheels) →
  cars = cars_wheels / 4 →
  cars = 2 := by
  sorry

end cars_in_garage_l2166_216694


namespace min_value_frac_eq_nine_halves_l2166_216682

theorem min_value_frac_eq_nine_halves {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 2) :
  ∃ (x y : ℝ), 2 / x + 1 / y = 9 / 2 := by
  sorry

end min_value_frac_eq_nine_halves_l2166_216682


namespace find_f_prime_at_1_l2166_216600

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^2 + 3 * x * f_prime_at_1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) :
  (∀ x, deriv (λ x => f x f_prime_at_1) x = 2 * x + 3 * f_prime_at_1) → 
  deriv (λ x => f x f_prime_at_1) 1 = -1 := 
by
exact sorry

end find_f_prime_at_1_l2166_216600


namespace charlie_age_when_jenny_twice_as_old_as_bobby_l2166_216632

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end charlie_age_when_jenny_twice_as_old_as_bobby_l2166_216632


namespace find_g_of_3_l2166_216659

theorem find_g_of_3 (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) :
  g 3 = 5 :=
sorry

end find_g_of_3_l2166_216659


namespace least_students_with_brown_eyes_and_lunch_box_l2166_216685

variable (U : Finset ℕ) (B L : Finset ℕ)
variables (hU : U.card = 25) (hB : B.card = 15) (hL : L.card = 18)

theorem least_students_with_brown_eyes_and_lunch_box : 
  (B ∩ L).card ≥ 8 := by
  sorry

end least_students_with_brown_eyes_and_lunch_box_l2166_216685


namespace exists_distinct_group_and_country_selection_l2166_216636

theorem exists_distinct_group_and_country_selection 
  (n m : ℕ) 
  (h_nm1 : n > m) 
  (h_m1 : m > 1) 
  (groups : Fin n → Fin m → Fin n → Prop) 
  (group_conditions : ∀ i j : Fin n, ∀ k : Fin m, ∀ l : Fin m, (i ≠ j) → (groups i k j = false)) 
  : 
  ∃ (selected : Fin n → Fin (m * n)), 
    (∀ i j: Fin n, i ≠ j → selected i ≠ selected j) ∧ 
    (∀ i j: Fin n, selected i / m ≠ selected j / m) := sorry

end exists_distinct_group_and_country_selection_l2166_216636


namespace sqrt_of_4_l2166_216617

theorem sqrt_of_4 :
  {x | x * x = 4} = {2, -2} :=
sorry

end sqrt_of_4_l2166_216617


namespace steve_took_4_berries_l2166_216615

theorem steve_took_4_berries (s t : ℕ) (H1 : s = 32) (H2 : t = 21) (H3 : s - 7 = t + x) :
  x = 4 :=
by
  sorry

end steve_took_4_berries_l2166_216615


namespace john_roommates_multiple_of_bob_l2166_216678

theorem john_roommates_multiple_of_bob (bob_roommates john_roommates : ℕ) (multiple : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) 
  (h3 : john_roommates = multiple * bob_roommates + 5) : 
  multiple = 2 :=
by
  sorry

end john_roommates_multiple_of_bob_l2166_216678


namespace solve_x_l2166_216648

theorem solve_x (x : ℝ) (h : (4 * x + 3) / (3 * x ^ 2 + 4 * x - 4) = 3 * x / (3 * x - 2)) :
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_x_l2166_216648
