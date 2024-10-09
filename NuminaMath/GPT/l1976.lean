import Mathlib

namespace simplify_sum_of_squares_roots_l1976_197618

theorem simplify_sum_of_squares_roots :
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 :=
by
  sorry

end simplify_sum_of_squares_roots_l1976_197618


namespace least_pos_int_with_ten_factors_l1976_197619

theorem least_pos_int_with_ten_factors : ∃ (n : ℕ), n > 0 ∧ (∀ m, (m > 0 ∧ ∃ d : ℕ, d∣n → d = 1 ∨ d = n) → m < n) ∧ ( ∃! n, ∃ d : ℕ, d∣n ) := sorry

end least_pos_int_with_ten_factors_l1976_197619


namespace trig_identity_t_half_l1976_197647

theorem trig_identity_t_half (a t : ℝ) (ht : t = Real.tan (a / 2)) :
  Real.sin a = (2 * t) / (1 + t^2) ∧
  Real.cos a = (1 - t^2) / (1 + t^2) ∧
  Real.tan a = (2 * t) / (1 - t^2) := 
sorry

end trig_identity_t_half_l1976_197647


namespace paving_stone_proof_l1976_197635

noncomputable def paving_stone_width (length_court : ℝ) (width_court : ℝ) 
                                      (num_stones: ℕ) (stone_length: ℝ) : ℝ :=
  let area_court := length_court * width_court
  let area_stone := stone_length * (area_court / (num_stones * stone_length))
  area_court / area_stone

theorem paving_stone_proof :
  paving_stone_width 50 16.5 165 2.5 = 2 :=
sorry

end paving_stone_proof_l1976_197635


namespace correct_division_l1976_197682

theorem correct_division (x : ℝ) (h : 8 * x + 8 = 56) : x / 8 = 0.75 :=
by
  sorry

end correct_division_l1976_197682


namespace min_value_f_l1976_197641

def f (x : ℝ) : ℝ := |2 * x - 1| + |3 * x - 2| + |4 * x - 3| + |5 * x - 4|

theorem min_value_f : (∃ x : ℝ, ∀ y : ℝ, f y ≥ f x) := 
sorry

end min_value_f_l1976_197641


namespace percentage_difference_l1976_197623

-- Define the quantities involved
def milk_in_A : ℕ := 1264
def transferred_milk : ℕ := 158

-- Define the quantities of milk in container B and C after transfer
noncomputable def quantity_in_B : ℕ := milk_in_A / 2
noncomputable def quantity_in_C : ℕ := quantity_in_B

-- Prove that the percentage difference between the quantity of milk in container B
-- and the capacity of container A is 50%
theorem percentage_difference :
  ((milk_in_A - quantity_in_B) * 100 / milk_in_A) = 50 := sorry

end percentage_difference_l1976_197623


namespace rectangle_ratio_l1976_197673

theorem rectangle_ratio (A L : ℝ) (hA : A = 100) (hL : L = 20) :
  ∃ W : ℝ, A = L * W ∧ (L / W) = 4 :=
by
  sorry

end rectangle_ratio_l1976_197673


namespace max_min_P_l1976_197654

theorem max_min_P (a b c : ℝ) (h : |a + b| + |b + c| + |c + a| = 8) :
  (a^2 + b^2 + c^2 = 48) ∨ (a^2 + b^2 + c^2 = 16 / 3) :=
sorry

end max_min_P_l1976_197654


namespace sum_of_first_20_primes_l1976_197652

theorem sum_of_first_20_primes :
  ( [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71].sum = 639 ) :=
by
  sorry

end sum_of_first_20_primes_l1976_197652


namespace ratio_of_side_lengths_l1976_197693

theorem ratio_of_side_lengths
  (pentagon_perimeter square_perimeter : ℕ)
  (pentagon_sides square_sides : ℕ)
  (pentagon_perimeter_eq : pentagon_perimeter = 100)
  (square_perimeter_eq : square_perimeter = 100)
  (pentagon_sides_eq : pentagon_sides = 5)
  (square_sides_eq : square_sides = 4) :
  (pentagon_perimeter / pentagon_sides) / (square_perimeter / square_sides) = 4 / 5 :=
by
  sorry

end ratio_of_side_lengths_l1976_197693


namespace max_diff_units_digit_l1976_197607

theorem max_diff_units_digit (n : ℕ) (h1 : n = 850 ∨ n = 855) : ∃ d, d = 5 :=
by 
  sorry

end max_diff_units_digit_l1976_197607


namespace problem1_problem2_l1976_197614

noncomputable def f (x a b : ℝ) : ℝ := 2 * x ^ 2 - 2 * a * x + b

noncomputable def set_A (a b : ℝ) : Set ℝ := {x | f x a b > 0 }

noncomputable def set_B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1 }

theorem problem1 (a b : ℝ) (h : f (-1) a b = -8) :
  (∀ x, x ∈ (set_A a b)ᶜ ∪ set_B 1 ↔ -3 ≤ x ∧ x ≤ 2) :=
  sorry

theorem problem2 (a b : ℝ) (t : ℝ) (h : f (-1) a b = -8) (h_not_P : (set_A a b) ∩ (set_B t) = ∅) :
  -2 ≤ t ∧ t ≤ 0 :=
  sorry

end problem1_problem2_l1976_197614


namespace parabola_vertex_x_coord_l1976_197664

theorem parabola_vertex_x_coord (a b c : ℝ)
  (h1 : 5 = a * 2^2 + b * 2 + c)
  (h2 : 5 = a * 8^2 + b * 8 + c)
  (h3 : 11 = a * 9^2 + b * 9 + c) :
  5 = (2 + 8) / 2 := 
sorry

end parabola_vertex_x_coord_l1976_197664


namespace redistributed_gnomes_l1976_197617

def WestervilleWoods : ℕ := 20
def RavenswoodForest := 4 * WestervilleWoods
def GreenwoodGrove := (5 * RavenswoodForest) / 4
def OwnerTakes (f: ℕ) (p: ℚ) := p * f

def RemainingGnomes (initial: ℕ) (p: ℚ) := initial - (OwnerTakes initial p)

def TotalRemainingGnomes := 
  (RemainingGnomes RavenswoodForest (40 / 100)) + 
  (RemainingGnomes WestervilleWoods (30 / 100)) + 
  (RemainingGnomes GreenwoodGrove (50 / 100))

def GnomesPerForest := TotalRemainingGnomes / 3

theorem redistributed_gnomes : 
  2 * 37 + 38 = TotalRemainingGnomes := by
  sorry

end redistributed_gnomes_l1976_197617


namespace kyler_wins_zero_l1976_197640

-- Definitions based on conditions provided
def peter_games_won : ℕ := 5
def peter_games_lost : ℕ := 3
def emma_games_won : ℕ := 4
def emma_games_lost : ℕ := 4
def kyler_games_lost : ℕ := 4

-- Number of games each player played
def peter_total_games : ℕ := peter_games_won + peter_games_lost
def emma_total_games : ℕ := emma_games_won + emma_games_lost
def kyler_total_games (k : ℕ) : ℕ := k + kyler_games_lost

-- Step 1: total number of games in the tournament
def total_games (k : ℕ) : ℕ := (peter_total_games + emma_total_games + kyler_total_games k) / 2

-- Step 2: Total games equation
def games_equation (k : ℕ) : Prop := 
  (peter_games_won + emma_games_won + k = total_games k)

-- The proof problem, we need to prove Kyler's wins
theorem kyler_wins_zero : games_equation 0 := by
  -- proof omitted
  sorry

end kyler_wins_zero_l1976_197640


namespace valid_5_digit_numbers_l1976_197632

noncomputable def num_valid_numbers (d : ℕ) (h : d ≠ 7) (h_valid : d < 10) (h_pos : d ≠ 0) : ℕ :=
  let choices_first_place := 7   -- choices for the first digit (1-9, excluding d and 7)
  let choices_other_places := 8  -- choices for other digits (0-9, excluding d and 7)
  choices_first_place * choices_other_places ^ 4

theorem valid_5_digit_numbers (d : ℕ) (h_d_ne_7 : d ≠ 7) (h_d_valid : d < 10) (h_d_pos : d ≠ 0) :
  num_valid_numbers d h_d_ne_7 h_d_valid h_d_pos = 28672 := sorry

end valid_5_digit_numbers_l1976_197632


namespace percent_decrease_area_pentagon_l1976_197681

open Real

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * s ^ 2

noncomputable def area_pentagon (s : ℝ) : ℝ :=
  (sqrt (5 * (5 + 2 * sqrt 5)) / 4) * s ^ 2

noncomputable def diagonal_pentagon (s : ℝ) : ℝ :=
  (1 + sqrt 5) / 2 * s

theorem percent_decrease_area_pentagon :
  let s_p := sqrt (400 / sqrt (5 * (5 + 2 * sqrt 5)))
  let d := diagonal_pentagon s_p
  let new_d := 0.9 * d
  let new_s := new_d / ((1 + sqrt 5) / 2)
  let new_area := area_pentagon new_s
  (100 - new_area) / 100 * 100 = 20 :=
by
  sorry

end percent_decrease_area_pentagon_l1976_197681


namespace instantaneous_velocity_at_t3_l1976_197615

open Real

noncomputable def displacement (t : ℝ) : ℝ := 4 - 2 * t + t ^ 2

theorem instantaneous_velocity_at_t3 : deriv displacement 3 = 4 := 
by
  sorry

end instantaneous_velocity_at_t3_l1976_197615


namespace ratio_adults_children_l1976_197667

-- Definitions based on conditions
def children := 45
def total_adults (A : ℕ) : Prop := (2 / 3 : ℚ) * A = 10

-- The theorem stating the problem
theorem ratio_adults_children :
  ∃ A, total_adults A ∧ (A : ℚ) / children = (1 / 3 : ℚ) :=
by {
  sorry
}

end ratio_adults_children_l1976_197667


namespace purchasing_plans_count_l1976_197674

theorem purchasing_plans_count :
  (∃ (x y : ℕ), 15 * x + 20 * y = 360) ∧ ∀ (x y : ℕ), 15 * x + 20 * y = 360 → (x % 4 = 0) ∧ (y = 18 - (3 / 4) * x) := sorry

end purchasing_plans_count_l1976_197674


namespace total_whipped_cream_l1976_197636

theorem total_whipped_cream (cream_from_farm : ℕ) (cream_to_buy : ℕ) (total_cream : ℕ) 
  (h1 : cream_from_farm = 149) 
  (h2 : cream_to_buy = 151) 
  (h3 : total_cream = cream_from_farm + cream_to_buy) : 
  total_cream = 300 :=
sorry

end total_whipped_cream_l1976_197636


namespace interval_length_condition_l1976_197602

theorem interval_length_condition (c : ℝ) (x : ℝ) (H1 : 3 ≤ 5 * x - 4) (H2 : 5 * x - 4 ≤ c) 
                                  (H3 : (c + 4) / 5 - 7 / 5 = 15) : c - 3 = 75 := 
sorry

end interval_length_condition_l1976_197602


namespace samuel_faster_l1976_197631

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end samuel_faster_l1976_197631


namespace a6_value_l1976_197656

variable (a_n : ℕ → ℤ)

/-- Given conditions in the arithmetic sequence -/
def arithmetic_sequence_property (a_n : ℕ → ℤ) :=
  ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)

/-- Given sum condition a_4 + a_5 + a_6 + a_7 + a_8 = 150 -/
def sum_condition :=
  a_n 4 + a_n 5 + a_n 6 + a_n 7 + a_n 8 = 150

theorem a6_value (h : arithmetic_sequence_property a_n) (hsum : sum_condition a_n) :
  a_n 6 = 30 := 
by
  sorry

end a6_value_l1976_197656


namespace hazel_salmon_caught_l1976_197646

-- Define the conditions
def father_salmon_caught : Nat := 27
def total_salmon_caught : Nat := 51

-- Define the main statement to be proved
theorem hazel_salmon_caught : total_salmon_caught - father_salmon_caught = 24 := by
  sorry

end hazel_salmon_caught_l1976_197646


namespace last_two_digits_of_large_exponent_l1976_197622

theorem last_two_digits_of_large_exponent :
  (9 ^ (8 ^ (7 ^ (6 ^ (5 ^ (4 ^ (3 ^ 2))))))) % 100 = 21 :=
by
  sorry

end last_two_digits_of_large_exponent_l1976_197622


namespace positive_even_representation_l1976_197608

theorem positive_even_representation (k : ℕ) (h : k > 0) :
  ∃ (a b : ℤ), (2 * k : ℤ) = a * b ∧ a + b = 0 := 
by
  sorry

end positive_even_representation_l1976_197608


namespace sum_of_digits_of_square_99999_l1976_197699

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_square_99999 : sum_of_digits ((99999 : ℕ)^2) = 45 := by
  sorry

end sum_of_digits_of_square_99999_l1976_197699


namespace simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l1976_197650

-- Problem 1
theorem simplify_expression1 (a b : ℝ) : 
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := 
sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := 
sorry

-- Problem 3
theorem simplify_expression3 (a b : ℝ) : 
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := 
sorry

-- Problem 4
theorem simplify_expression4 (x y : ℝ) : 
  6 * x * y^2 - (2 * x - (1 / 2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := 
sorry

end simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l1976_197650


namespace probability_of_Y_l1976_197687

theorem probability_of_Y (P_X P_both : ℝ) (h1 : P_X = 1/5) (h2 : P_both = 0.13333333333333333) : 
    (0.13333333333333333 / (1 / 5)) = 0.6666666666666667 :=
by sorry

end probability_of_Y_l1976_197687


namespace tom_books_after_transactions_l1976_197688

-- Define the initial conditions as variables
def initial_books : ℕ := 5
def sold_books : ℕ := 4
def new_books : ℕ := 38

-- Define the property we need to prove
theorem tom_books_after_transactions : initial_books - sold_books + new_books = 39 := by
  sorry

end tom_books_after_transactions_l1976_197688


namespace crystal_meal_combinations_l1976_197690

-- Definitions for conditions:
def entrees := 4
def drinks := 4
def desserts := 3 -- includes two desserts and the option of no dessert

-- Statement of the problem as a theorem:
theorem crystal_meal_combinations : entrees * drinks * desserts = 48 := by
  sorry

end crystal_meal_combinations_l1976_197690


namespace compute_fraction_l1976_197620

theorem compute_fraction : (2015 : ℝ) / ((2015 : ℝ)^2 - (2016 : ℝ) * (2014 : ℝ)) = 2015 :=
by {
  sorry
}

end compute_fraction_l1976_197620


namespace sun_city_population_correct_l1976_197684

noncomputable def willowdale_population : Nat := 2000
noncomputable def roseville_population : Nat := 3 * willowdale_population - 500
noncomputable def sun_city_population : Nat := 2 * roseville_population + 1000

theorem sun_city_population_correct : sun_city_population = 12000 := by
  sorry

end sun_city_population_correct_l1976_197684


namespace solve_equation_1_solve_equation_2_l1976_197695

theorem solve_equation_1 (x : ℚ) : 1 - (1 / (x - 5)) = (x / (x + 5)) → x = 15 / 2 := 
by
  sorry

theorem solve_equation_2 (x : ℚ) : (3 / (x - 1)) - (2 / (x + 1)) = (1 / (x^2 - 1)) → x = -4 := 
by
  sorry

end solve_equation_1_solve_equation_2_l1976_197695


namespace car_speed_on_local_roads_l1976_197651

theorem car_speed_on_local_roads
    (v : ℝ) -- Speed of the car on local roads
    (h1 : v > 0) -- The speed is positive
    (h2 : 40 / v + 3 = 5) -- Given equation based on travel times and distances
    : v = 20 := 
sorry

end car_speed_on_local_roads_l1976_197651


namespace robin_piano_highest_before_lowest_l1976_197629

def probability_reach_highest_from_middle_C : ℚ :=
  let p_k (k : ℕ) (p_prev : ℚ) (p_next : ℚ) : ℚ := (1/2 : ℚ) * p_prev + (1/2 : ℚ) * p_next
  let p_1 := 0
  let p_88 := 1
  let A := -1/87
  let B := 1/87
  A + B * 40

theorem robin_piano_highest_before_lowest :
  probability_reach_highest_from_middle_C = 13 / 29 :=
by
  sorry

end robin_piano_highest_before_lowest_l1976_197629


namespace unique_sequence_l1976_197638

theorem unique_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) ^ 2 = 1 + (n + 2021) * a n) →
  (∀ n : ℕ, a n = n + 2019) :=
by
  sorry

end unique_sequence_l1976_197638


namespace max_value_t_min_value_y_l1976_197653

open Real

-- Maximum value of t for ∀ x ∈ ℝ, |3x + 2| + |3x - 1| ≥ t
theorem max_value_t :
  ∃ t, (∀ x : ℝ, |3 * x + 2| + |3 * x - 1| ≥ t) ∧ t = 3 :=
by
  sorry

-- Minimum value of y for 4m + 5n = 3
theorem min_value_y (m n: ℝ) (hm : m > 0) (hn: n > 0) (h: 4 * m + 5 * n = 3) :
  ∃ y, (y = (1 / (m + 2 * n)) + (4 / (3 * m + 3 * n))) ∧ y = 3 :=
by
  sorry

end max_value_t_min_value_y_l1976_197653


namespace tech_gadget_cost_inr_l1976_197661

def conversion_ratio (a b : ℝ) : Prop := a = b

theorem tech_gadget_cost_inr :
  (forall a b c : ℝ, conversion_ratio (a / b) c) →
  (forall a b c d : ℝ, conversion_ratio (a / b) c → conversion_ratio (a / d) c) →
  ∀ (n_usd : ℝ) (n_inr : ℝ) (cost_n : ℝ), 
    n_usd = 8 →
    n_inr = 5 →
    cost_n = 160 →
    cost_n / n_usd * n_inr = 100 :=
by
  sorry

end tech_gadget_cost_inr_l1976_197661


namespace smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l1976_197633

theorem smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 
             (n % 9 = 0) ∧ 
             (∃ d1 d2 d3 d4 : ℕ, 
               d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧ 
               d1 % 2 = 1 ∧ 
               d2 % 2 = 0 ∧ 
               d3 % 2 = 0 ∧ 
               d4 % 2 = 0) ∧ 
             (∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ m % 9 = 0 ∧ 
               ∃ e1 e2 e3 e4 : ℕ, 
                 e1 * 1000 + e2 * 100 + e3 * 10 + e4 = m ∧ 
                 e1 % 2 = 1 ∧ 
                 e2 % 2 = 0 ∧ 
                 e3 % 2 = 0 ∧ 
                 e4 % 2 = 0) → n ≤ m) ∧ 
             n = 1026 :=
sorry

end smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l1976_197633


namespace number_of_lattice_points_in_triangle_l1976_197606

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end number_of_lattice_points_in_triangle_l1976_197606


namespace find_N_l1976_197634

variable (a b c N : ℕ)

theorem find_N (h1 : a + b + c = 90) (h2 : a - 7 = N) (h3 : b + 7 = N) (h4 : 5 * c = N) : N = 41 := 
by
  sorry

end find_N_l1976_197634


namespace bridgette_has_4_birds_l1976_197680

/-
Conditions:
1. Bridgette has 2 dogs.
2. Bridgette has 3 cats.
3. Bridgette has some birds.
4. She gives the dogs a bath twice a month.
5. She gives the cats a bath once a month.
6. She gives the birds a bath once every 4 months.
7. In a year, she gives a total of 96 baths.
-/

def num_birds (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ) : ℕ :=
  let yearly_dog_baths := num_dogs * dog_baths_per_month * 12
  let yearly_cat_baths := num_cats * cat_baths_per_month * 12
  let birds_baths := total_baths_per_year - (yearly_dog_baths + yearly_cat_baths)
  let baths_per_bird_per_year := 12 / bird_baths_per_4_months
  birds_baths / baths_per_bird_per_year

theorem bridgette_has_4_birds :
  ∀ (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ),
    num_dogs = 2 →
    num_cats = 3 →
    dog_baths_per_month = 2 →
    cat_baths_per_month = 1 →
    bird_baths_per_4_months = 4 →
    total_baths_per_year = 96 →
    num_birds num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year = 4 :=
by
  intros
  sorry


end bridgette_has_4_birds_l1976_197680


namespace total_hotdogs_sold_l1976_197663

-- Define the number of small and large hotdogs
def small_hotdogs : ℕ := 58
def large_hotdogs : ℕ := 21

-- Define the total hotdogs
def total_hotdogs : ℕ := small_hotdogs + large_hotdogs

-- The Main Statement to prove the total number of hotdogs sold
theorem total_hotdogs_sold : total_hotdogs = 79 :=
by
  -- Proof is skipped using sorry
  sorry

end total_hotdogs_sold_l1976_197663


namespace Homer_first_try_points_l1976_197644

variable (x : ℕ)
variable (h1 : x + (x - 70) + 2 * (x - 70) = 1390)

theorem Homer_first_try_points : x = 400 := by
  sorry

end Homer_first_try_points_l1976_197644


namespace sum_of_areas_of_circles_l1976_197678

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l1976_197678


namespace find_average_after_17th_inning_l1976_197694

def initial_average_after_16_inns (A : ℕ) : Prop :=
  let total_runs := 16 * A
  let new_total_runs := total_runs + 87
  let new_average := new_total_runs / 17
  new_average = A + 4

def runs_in_17th_inning := 87

noncomputable def average_after_17th_inning (A : ℕ) : Prop :=
  A + 4 = 23

theorem find_average_after_17th_inning (A : ℕ) :
  initial_average_after_16_inns A →
  average_after_17th_inning A :=
  sorry

end find_average_after_17th_inning_l1976_197694


namespace product_of_three_integers_sum_l1976_197697
-- Import necessary libraries

-- Define the necessary conditions and the goal
theorem product_of_three_integers_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
(h4 : a * b * c = 11^3) : a + b + c = 133 :=
sorry

end product_of_three_integers_sum_l1976_197697


namespace minimum_value_of_k_l1976_197658

theorem minimum_value_of_k (x y : ℝ) (h : x * (x - 1) ≤ y * (1 - y)) : x^2 + y^2 ≤ 2 :=
sorry

end minimum_value_of_k_l1976_197658


namespace square_side_length_l1976_197610

theorem square_side_length (s : ℝ) (h : s^2 = 1/9) : s = 1/3 :=
sorry

end square_side_length_l1976_197610


namespace convert_speed_kmh_to_ms_l1976_197689

-- Define the given speed in km/h
def speed_kmh : ℝ := 1.1076923076923078

-- Define the conversion factor from km/h to m/s
def conversion_factor : ℝ := 3.6

-- State the theorem
theorem convert_speed_kmh_to_ms (s : ℝ) (h : s = speed_kmh) : (s / conversion_factor) = 0.3076923076923077 := by
  -- Skip the proof as instructed
  sorry

end convert_speed_kmh_to_ms_l1976_197689


namespace Nina_total_problems_l1976_197624

def Ruby_math_problems := 12
def Ruby_reading_problems := 4
def Ruby_science_problems := 5

def Nina_math_problems := 5 * Ruby_math_problems
def Nina_reading_problems := 9 * Ruby_reading_problems
def Nina_science_problems := 3 * Ruby_science_problems

def total_problems := Nina_math_problems + Nina_reading_problems + Nina_science_problems

theorem Nina_total_problems : total_problems = 111 :=
by
  sorry

end Nina_total_problems_l1976_197624


namespace calum_disco_ball_budget_l1976_197676

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end calum_disco_ball_budget_l1976_197676


namespace volleyball_team_ways_l1976_197600

def num_ways_choose_starers : ℕ :=
  3 * (Nat.choose 12 6 + Nat.choose 12 5)

theorem volleyball_team_ways :
  num_ways_choose_starers = 5148 := by
  sorry

end volleyball_team_ways_l1976_197600


namespace numberOfAntiPalindromes_l1976_197630

-- Define what it means for a number to be an anti-palindrome in base 3
def isAntiPalindrome (n : ℕ) : Prop :=
  ∀ (a b : ℕ), a + b = 2 → a ≠ b

-- Define the constraint of no two consecutive digits being the same
def noConsecutiveDigits (digits : List ℕ) : Prop :=
  ∀ (i : ℕ), i < digits.length - 1 → digits.nthLe i sorry ≠ digits.nthLe (i + 1) sorry

-- We want to find the number of anti-palindromes less than 3^12 fulfilling both conditions
def countAntiPalindromes (m : ℕ) (base : ℕ) : ℕ :=
  sorry -- Placeholder definition for the count, to be implemented

-- The main theorem to prove
theorem numberOfAntiPalindromes : countAntiPalindromes (3^12) 3 = 126 :=
  sorry -- Proof to be filled

end numberOfAntiPalindromes_l1976_197630


namespace simplify_fractions_l1976_197686

theorem simplify_fractions :
  (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 :=
by sorry

end simplify_fractions_l1976_197686


namespace fractional_part_of_students_who_walk_home_l1976_197603

theorem fractional_part_of_students_who_walk_home 
  (students_by_bus : ℚ)
  (students_by_car : ℚ)
  (students_by_bike : ℚ)
  (students_by_skateboard : ℚ)
  (h_bus : students_by_bus = 1/3)
  (h_car : students_by_car = 1/5)
  (h_bike : students_by_bike = 1/8)
  (h_skateboard : students_by_skateboard = 1/15)
  : 1 - (students_by_bus + students_by_car + students_by_bike + students_by_skateboard) = 11/40 := 
by
  sorry

end fractional_part_of_students_who_walk_home_l1976_197603


namespace max_band_members_l1976_197616

theorem max_band_members (n : ℤ) (h1 : 20 * n % 31 = 11) (h2 : 20 * n < 1200) : 20 * n = 1100 :=
sorry

end max_band_members_l1976_197616


namespace shadow_length_building_l1976_197613

theorem shadow_length_building:
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  (height_flagstaff / shadow_flagstaff = height_building / expected_shadow_building) := by
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  sorry

end shadow_length_building_l1976_197613


namespace problem_statement_l1976_197601

def T : Set ℤ :=
  {n^2 + (n+2)^2 + (n+4)^2 | n : ℤ }

theorem problem_statement :
  (∀ x ∈ T, ¬ (4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) :=
by
  sorry

end problem_statement_l1976_197601


namespace parabola_vertex_x_coordinate_l1976_197677

theorem parabola_vertex_x_coordinate (a b c : ℝ) (h1 : c = 0) (h2 : 16 * a + 4 * b = 0) (h3 : 9 * a + 3 * b = 9) : 
    -b / (2 * a) = 2 :=
by 
  -- You can start by adding a proof here
  sorry

end parabola_vertex_x_coordinate_l1976_197677


namespace probability_manu_wins_l1976_197657

theorem probability_manu_wins :
  ∑' (n : ℕ), (1 / 2)^(4 * (n + 1)) = 1 / 15 :=
by
  sorry

end probability_manu_wins_l1976_197657


namespace speed_of_current_l1976_197675

variables (b c : ℝ)

theorem speed_of_current (h1 : b + c = 12) (h2 : b - c = 4) : c = 4 :=
sorry

end speed_of_current_l1976_197675


namespace projection_matrix_solution_l1976_197692

theorem projection_matrix_solution (a c : ℚ) (Q : Matrix (Fin 2) (Fin 2) ℚ) 
  (hQ : Q = !![a, 18/45; c, 27/45] ) 
  (proj_Q : Q * Q = Q) : 
  (a, c) = (2/5, 3/5) :=
by
  sorry

end projection_matrix_solution_l1976_197692


namespace comparison_abc_l1976_197609

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := 0.5 * (Real.log 2023 / Real.log 2022 + Real.log 2022 / Real.log 2023)

theorem comparison_abc : c > b ∧ b > a := by
  sorry

end comparison_abc_l1976_197609


namespace fraction_value_l1976_197621

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l1976_197621


namespace base_of_exponent_l1976_197698

theorem base_of_exponent (b x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) (h3 : b^x * 4^y = 531441) : b = 3 :=
by
  sorry

end base_of_exponent_l1976_197698


namespace ratio_sandra_amy_ruth_l1976_197625

/-- Given the amounts received by Sandra and Amy, and an unknown amount received by Ruth,
    the ratio of the money shared between Sandra, Amy, and Ruth is 2:1:R/50. -/
theorem ratio_sandra_amy_ruth (R : ℝ) (hAmy : 50 > 0) (hSandra : 100 > 0) :
  (100 : ℝ) / 50 = 2 ∧ (50 : ℝ) / 50 = 1 ∧ ∃ (R : ℝ), (100/50 : ℝ) = 2 ∧ (50/50 : ℝ) = 1 ∧ (R / 50 : ℝ) = (R / 50 : ℝ) :=
by
  sorry

end ratio_sandra_amy_ruth_l1976_197625


namespace borrowed_nickels_l1976_197665

def n_original : ℕ := 87
def n_left : ℕ := 12
def n_borrowed : ℕ := n_original - n_left

theorem borrowed_nickels : n_borrowed = 75 := by
  sorry

end borrowed_nickels_l1976_197665


namespace number_square_l1976_197645

-- Define conditions.
def valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d * d ≤ 9

-- Main statement.
theorem number_square (n : ℕ) (valid_digits : ∀ d, d ∈ [n / 100, (n / 10) % 10, n % 10] → valid_digit d) : 
  n = 233 :=
by
  -- Proof goes here
  sorry

end number_square_l1976_197645


namespace find_three_digit_integers_mod_l1976_197612

theorem find_three_digit_integers_mod (n : ℕ) :
  (n % 7 = 3) ∧ (n % 8 = 6) ∧ (n % 5 = 2) ∧ (100 ≤ n) ∧ (n < 1000) :=
sorry

end find_three_digit_integers_mod_l1976_197612


namespace football_games_total_l1976_197628

def total_football_games_per_season (games_per_month : ℝ) (num_months : ℝ) : ℝ :=
  games_per_month * num_months

theorem football_games_total (games_per_month : ℝ) (num_months : ℝ) (total_games : ℝ) :
  games_per_month = 323.0 ∧ num_months = 17.0 ∧ total_games = 5491.0 →
  total_football_games_per_season games_per_month num_months = total_games :=
by
  intros h
  have h1 : games_per_month = 323.0 := h.1
  have h2 : num_months = 17.0 := h.2.1
  have h3 : total_games = 5491.0 := h.2.2
  rw [h1, h2, h3]
  sorry

end football_games_total_l1976_197628


namespace sufficient_not_necessary_condition_l1976_197639

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h_pos : x > 0) :
  (a = 4 → x + a / x ≥ 4) ∧ (∃ b : ℝ, b ≠ 4 ∧ ∃ x : ℝ, x > 0 ∧ x + b / x ≥ 4) :=
by
  sorry

end sufficient_not_necessary_condition_l1976_197639


namespace quadrilateral_areas_product_l1976_197648

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end quadrilateral_areas_product_l1976_197648


namespace arithmetic_progression_number_of_terms_l1976_197683

variable (a d : ℕ)
variable (n : ℕ) (h_n_even : n % 2 = 0)
variable (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 60)
variable (h_sum_even : (n / 2) * (2 * (a + d) + (n - 2) * d) = 80)
variable (h_diff : (n - 1) * d = 16)

theorem arithmetic_progression_number_of_terms : n = 8 :=
by
  sorry

end arithmetic_progression_number_of_terms_l1976_197683


namespace number_line_steps_l1976_197611

theorem number_line_steps (total_steps : ℕ) (total_distance : ℕ) (steps_taken : ℕ) (result_distance : ℕ) 
  (h1 : total_distance = 36) (h2 : total_steps = 9) (h3 : steps_taken = 6) : 
  result_distance = (steps_taken * (total_distance / total_steps)) → result_distance = 24 :=
by
  intros H
  sorry

end number_line_steps_l1976_197611


namespace two_dice_sum_greater_than_four_l1976_197685
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l1976_197685


namespace probability_both_red_l1976_197691

-- Definitions for the problem conditions
def total_balls := 16
def red_balls := 7
def blue_balls := 5
def green_balls := 4
def first_red_prob := (red_balls : ℚ) / total_balls
def second_red_given_first_red_prob := (red_balls - 1 : ℚ) / (total_balls - 1)

-- The statement to be proved
theorem probability_both_red : (first_red_prob * second_red_given_first_red_prob) = (7 : ℚ) / 40 :=
by 
  -- Proof goes here
  sorry

end probability_both_red_l1976_197691


namespace delta_f_l1976_197666

open BigOperators

def f (n : ℕ) : ℕ := ∑ i in Finset.range n, (i + 1) * (n - i)

theorem delta_f (k : ℕ) : f (k + 1) - f k = ∑ i in Finset.range (k + 1), (i + 1) :=
by
  sorry

end delta_f_l1976_197666


namespace intersection_complement_P_Q_l1976_197637

def P (x : ℝ) : Prop := x - 1 ≤ 0
def Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

def complement_P (x : ℝ) : Prop := ¬ P x

theorem intersection_complement_P_Q :
  {x : ℝ | complement_P x} ∩ {x : ℝ | Q x} = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_P_Q_l1976_197637


namespace convert_to_general_form_l1976_197662

theorem convert_to_general_form (x : ℝ) :
  5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 :=
by
  sorry

end convert_to_general_form_l1976_197662


namespace sock_problem_l1976_197660

def sock_pair_count (total_socks : Nat) (socks_distribution : List (String × Nat)) (target_color : String) (different_color : String) : Nat :=
  if target_color = different_color then 0
  else match socks_distribution with
    | [] => 0
    | (color, count) :: tail =>
        if color = target_color then count * socks_distribution.foldl (λ acc (col_count : String × Nat) =>
          if col_count.fst ≠ target_color then acc + col_count.snd else acc) 0
        else sock_pair_count total_socks tail target_color different_color

theorem sock_problem : sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "white" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "brown" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "blue" =
                        48 :=
by sorry

end sock_problem_l1976_197660


namespace problem_proof_l1976_197669

-- Problem statement
variable (f : ℕ → ℕ)

-- Condition: if f(k) ≥ k^2 then f(k+1) ≥ (k+1)^2
variable (h : ∀ k, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)

-- Additional condition: f(4) ≥ 25
variable (h₀ : f 4 ≥ 25)

-- To prove: ∀ k ≥ 4, f(k) ≥ k^2
theorem problem_proof : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_proof_l1976_197669


namespace evaluate_expression_l1976_197679

theorem evaluate_expression : 
  - (16 / 2 * 8 - 72 + 4^2) = -8 :=
by 
  -- here, the proof would typically go
  sorry

end evaluate_expression_l1976_197679


namespace total_bill_cost_l1976_197643

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end total_bill_cost_l1976_197643


namespace divisor_increase_by_10_5_l1976_197668

def condition_one (n t : ℕ) : Prop :=
  n * (t + 7) = t * (n + 2)

def condition_two (n t z : ℕ) : Prop :=
  n * (t + z) = t * (n + 3)

theorem divisor_increase_by_10_5 (n t : ℕ) (hz : ℕ) (nz : n ≠ 0) (tz : t ≠ 0)
  (h1 : condition_one n t) (h2 : condition_two n t hz) : hz = 21 / 2 :=
by {
  sorry
}

end divisor_increase_by_10_5_l1976_197668


namespace linear_equation_solution_l1976_197672

theorem linear_equation_solution (x : ℝ) (h : 1 - x = -3) : x = 4 :=
by
  sorry

end linear_equation_solution_l1976_197672


namespace final_answer_l1976_197659

theorem final_answer : (848 / 8) - 100 = 6 := 
by
  sorry

end final_answer_l1976_197659


namespace find_f_60_l1976_197670

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition.

axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_48 : f 48 = 36

theorem find_f_60 : f 60 = 28.8 := by 
  sorry

end find_f_60_l1976_197670


namespace simplify_fraction_l1976_197696

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 :=
by
  have h1 : Real.sqrt 75 = 5 * Real.sqrt 3 := by sorry
  have h2 : Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry
  sorry

end simplify_fraction_l1976_197696


namespace minimum_bailing_rate_l1976_197649

-- Conditions as formal definitions.
def distance_from_shore : ℝ := 3
def intake_rate : ℝ := 20 -- gallons per minute
def sinking_threshold : ℝ := 120 -- gallons
def speed_first_half : ℝ := 6 -- miles per hour
def speed_second_half : ℝ := 3 -- miles per hour

-- Formal translation of the problem using definitions.
theorem minimum_bailing_rate : (distance_from_shore = 3) →
                             (intake_rate = 20) →
                             (sinking_threshold = 120) →
                             (speed_first_half = 6) →
                             (speed_second_half = 3) →
                             (∃ r : ℝ, 18 ≤ r) :=
by
  sorry

end minimum_bailing_rate_l1976_197649


namespace mmobile_additional_line_cost_l1976_197604

noncomputable def cost_tmobile (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * 16

noncomputable def cost_mmobile (x : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * x

theorem mmobile_additional_line_cost
  (x : ℕ)
  (ht : cost_tmobile 5 = 98)
  (hm : cost_tmobile 5 - cost_mmobile x 5 = 11) :
  x = 14 :=
by
  sorry

end mmobile_additional_line_cost_l1976_197604


namespace problem1_problem2_problem3_problem4_l1976_197655

section
  variable (a b c d : Int)

  theorem problem1 : -27 + (-32) + (-8) + 72 = 5 := by
    sorry

  theorem problem2 : -4 - 2 * 32 + (-2 * 32) = -132 := by
    sorry

  theorem problem3 : (-48 : Int) / (-2 : Int)^3 - (-25 : Int) * (-4 : Int) + (-2 : Int)^3 = -102 := by
    sorry

  theorem problem4 : (-3 : Int)^2 - (3 / 2)^3 * (2 / 9) - 6 / (-(2 / 3))^3 = -12 := by
    sorry
end

end problem1_problem2_problem3_problem4_l1976_197655


namespace middle_number_is_14_5_l1976_197605

theorem middle_number_is_14_5 (x y z : ℝ) (h1 : x + y = 24) (h2 : x + z = 29) (h3 : y + z = 34) : y = 14.5 :=
sorry

end middle_number_is_14_5_l1976_197605


namespace gift_wrapping_combinations_l1976_197627

theorem gift_wrapping_combinations 
  (wrapping_varieties : ℕ)
  (ribbon_colors : ℕ)
  (gift_card_types : ℕ)
  (H_wrapping_varieties : wrapping_varieties = 8)
  (H_ribbon_colors : ribbon_colors = 3)
  (H_gift_card_types : gift_card_types = 4) : 
  wrapping_varieties * ribbon_colors * gift_card_types = 96 := 
by
  sorry

end gift_wrapping_combinations_l1976_197627


namespace all_positive_l1976_197642

theorem all_positive (a1 a2 a3 a4 a5 a6 a7 : ℝ)
  (h1 : a1 + a2 + a3 + a4 > a5 + a6 + a7)
  (h2 : a1 + a2 + a3 + a5 > a4 + a6 + a7)
  (h3 : a1 + a2 + a3 + a6 > a4 + a5 + a7)
  (h4 : a1 + a2 + a3 + a7 > a4 + a5 + a6)
  (h5 : a1 + a2 + a4 + a5 > a3 + a6 + a7)
  (h6 : a1 + a2 + a4 + a6 > a3 + a5 + a7)
  (h7 : a1 + a2 + a4 + a7 > a3 + a5 + a6)
  (h8 : a1 + a2 + a5 + a6 > a3 + a4 + a7)
  (h9 : a1 + a2 + a5 + a7 > a3 + a4 + a6)
  (h10 : a1 + a2 + a6 + a7 > a3 + a4 + a5)
  (h11 : a1 + a3 + a4 + a5 > a2 + a6 + a7)
  (h12 : a1 + a3 + a4 + a6 > a2 + a5 + a7)
  (h13 : a1 + a3 + a4 + a7 > a2 + a5 + a6)
  (h14 : a1 + a3 + a5 + a6 > a2 + a4 + a7)
  (h15 : a1 + a3 + a5 + a7 > a2 + a4 + a6)
  (h16 : a1 + a3 + a6 + a7 > a2 + a4 + a5)
  (h17 : a1 + a4 + a5 + a6 > a2 + a3 + a7)
  (h18 : a1 + a4 + a5 + a7 > a2 + a3 + a6)
  (h19 : a1 + a4 + a6 + a7 > a2 + a3 + a5)
  (h20 : a1 + a5 + a6 + a7 > a2 + a3 + a4)
  (h21 : a2 + a3 + a4 + a5 > a1 + a6 + a7)
  (h22 : a2 + a3 + a4 + a6 > a1 + a5 + a7)
  (h23 : a2 + a3 + a4 + a7 > a1 + a5 + a6)
  (h24 : a2 + a3 + a5 + a6 > a1 + a4 + a7)
  (h25 : a2 + a3 + a5 + a7 > a1 + a4 + a6)
  (h26 : a2 + a3 + a6 + a7 > a1 + a4 + a5)
  (h27 : a2 + a4 + a5 + a6 > a1 + a3 + a7)
  (h28 : a2 + a4 + a5 + a7 > a1 + a3 + a6)
  (h29 : a2 + a4 + a6 + a7 > a1 + a3 + a5)
  (h30 : a2 + a5 + a6 + a7 > a1 + a3 + a4)
  (h31 : a3 + a4 + a5 + a6 > a1 + a2 + a7)
  (h32 : a3 + a4 + a5 + a7 > a1 + a2 + a6)
  (h33 : a3 + a4 + a6 + a7 > a1 + a2 + a5)
  (h34 : a3 + a5 + a6 + a7 > a1 + a2 + a4)
  (h35 : a4 + a5 + a6 + a7 > a1 + a2 + a3)
: a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0 ∧ a6 > 0 ∧ a7 > 0 := 
sorry

end all_positive_l1976_197642


namespace pears_equivalence_l1976_197626

theorem pears_equivalence :
  (3 / 4 : ℚ) * 16 * (5 / 6) = 10 → 
  (2 / 5 : ℚ) * 20 * (5 / 6) = 20 / 3 := 
by
  intros h
  sorry

end pears_equivalence_l1976_197626


namespace vasya_max_pencils_l1976_197671

theorem vasya_max_pencils (money_for_pencils : ℕ) (rebate_20 : ℕ) (rebate_5 : ℕ) :
  money_for_pencils = 30 → rebate_20 = 25 → rebate_5 = 10 → ∃ max_pencils, max_pencils = 36 :=
by
  intros h_money h_r20 h_r5
  sorry

end vasya_max_pencils_l1976_197671
