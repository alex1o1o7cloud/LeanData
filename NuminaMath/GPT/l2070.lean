import Mathlib

namespace walnut_price_l2070_207084

theorem walnut_price {total_weight total_value walnut_price hazelnut_price : ℕ} 
  (h1 : total_weight = 55)
  (h2 : total_value = 1978)
  (h3 : walnut_price > hazelnut_price)
  (h4 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ hazelnut_price = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9)
  (h5 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ b = a - 1) : 
  walnut_price = 43 := 
sorry

end walnut_price_l2070_207084


namespace correct_sampling_method_order_l2070_207022

-- Definitions for sampling methods
def simple_random_sampling (method : ℕ) : Bool :=
  method = 1

def systematic_sampling (method : ℕ) : Bool :=
  method = 2

def stratified_sampling (method : ℕ) : Bool :=
  method = 3

-- Main theorem stating the correct method order
theorem correct_sampling_method_order : simple_random_sampling 1 ∧ stratified_sampling 3 ∧ systematic_sampling 2 :=
by
  sorry

end correct_sampling_method_order_l2070_207022


namespace bus_speed_including_stoppages_l2070_207058

-- Definitions based on conditions
def speed_excluding_stoppages : ℝ := 50 -- kmph
def stoppage_time_per_hour : ℝ := 18 -- minutes

-- Lean statement of the problem
theorem bus_speed_including_stoppages :
  (speed_excluding_stoppages * (1 - stoppage_time_per_hour / 60)) = 35 := by
  sorry

end bus_speed_including_stoppages_l2070_207058


namespace sally_total_spent_l2070_207033

-- Define the prices paid by Sally for peaches after the coupon and for cherries
def P_peaches : ℝ := 12.32
def C_cherries : ℝ := 11.54

-- State the problem to prove that the total amount Sally spent is 23.86
theorem sally_total_spent : P_peaches + C_cherries = 23.86 := by
  sorry

end sally_total_spent_l2070_207033


namespace find_A_l2070_207048

theorem find_A :
  ∃ A B C : ℝ, 
  (1 : ℝ) / (x^3 - 7 * x^2 + 11 * x + 15) = 
  A / (x - 5) + B / (x + 3) + C / ((x + 3)^2) → 
  A = 1 / 64 := 
by 
  sorry

end find_A_l2070_207048


namespace quadratic_solution_factoring_solution_l2070_207034

-- Define the first problem: Solve 2x^2 - 6x - 5 = 0
theorem quadratic_solution (x : ℝ) : 2 * x^2 - 6 * x - 5 = 0 ↔ x = (3 + Real.sqrt 19) / 2 ∨ x = (3 - Real.sqrt 19) / 2 :=
by
  sorry

-- Define the second problem: Solve 3x(4-x) = 2(x-4)
theorem factoring_solution (x : ℝ) : 3 * x * (4 - x) = 2 * (x - 4) ↔ x = 4 ∨ x = -2 / 3 :=
by
  sorry

end quadratic_solution_factoring_solution_l2070_207034


namespace value_of_f_g_10_l2070_207027

def g (x : ℤ) : ℤ := 4 * x + 6
def f (x : ℤ) : ℤ := 6 * x - 10

theorem value_of_f_g_10 : f (g 10) = 266 :=
by
  sorry

end value_of_f_g_10_l2070_207027


namespace range_of_a_l2070_207042

theorem range_of_a (a : ℝ) (h : ∀ t : ℝ, 0 < t → t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) : 
  (2 / 13) ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l2070_207042


namespace cyclist_traveled_18_miles_l2070_207019

noncomputable def cyclist_distance (v t d : ℕ) : Prop :=
  (d = v * t) ∧ 
  (d = (v + 1) * (3 * t / 4)) ∧ 
  (d = (v - 1) * (t + 3))

theorem cyclist_traveled_18_miles : ∃ (d : ℕ), cyclist_distance 3 6 d ∧ d = 18 :=
by
  sorry

end cyclist_traveled_18_miles_l2070_207019


namespace value_of_a_when_b_is_24_l2070_207095

variable (a b k : ℝ)

theorem value_of_a_when_b_is_24 (h1 : a = k / b^2) (h2 : 40 = k / 12^2) (h3 : b = 24) : a = 10 :=
by
  sorry

end value_of_a_when_b_is_24_l2070_207095


namespace combined_weight_of_elephant_and_donkey_l2070_207029

theorem combined_weight_of_elephant_and_donkey 
  (tons_to_pounds : ℕ → ℕ)
  (elephant_weight_tons : ℕ) 
  (donkey_percentage : ℕ) : 
  tons_to_pounds elephant_weight_tons * (1 + donkey_percentage / 100) = 6600 :=
by
  let tons_to_pounds (t : ℕ) := 2000 * t
  let elephant_weight_tons := 3
  let donkey_percentage := 10
  sorry

end combined_weight_of_elephant_and_donkey_l2070_207029


namespace range_of_x_l2070_207011

variable (x : ℝ)

-- Conditions used in the problem
def sqrt_condition : Prop := x + 2 ≥ 0
def non_zero_condition : Prop := x + 1 ≠ 0

-- The statement to be proven
theorem range_of_x : sqrt_condition x ∧ non_zero_condition x ↔ (x ≥ -2 ∧ x ≠ -1) :=
by
  sorry

end range_of_x_l2070_207011


namespace solution_correct_l2070_207039

-- Conditions of the problem
variable (f : ℝ → ℝ)
variable (h_f_domain : ∀ (x : ℝ), 0 < x → 0 < f x)
variable (h_f_eq : ∀ (x y : ℝ), 0 < x → 0 < y → f x * f (y * f x) = f (x + y))

-- Correct answer to be proven
theorem solution_correct :
  ∃ b : ℝ, 0 ≤ b ∧ ∀ t : ℝ, 0 < t → f t = 1 / (1 + b * t) :=
sorry

end solution_correct_l2070_207039


namespace smallest_perimeter_is_23_l2070_207072

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def are_consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧ b = a + 2 ∧ c = b + 2

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_perimeter_is_23 : 
  ∃ (a b c : ℕ), are_consecutive_odd_primes a b c ∧ satisfies_triangle_inequality a b c ∧ is_prime (a + b + c) ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_perimeter_is_23_l2070_207072


namespace number_of_seasons_l2070_207023

theorem number_of_seasons 
        (episodes_per_season : ℕ) 
        (fraction_watched : ℚ) 
        (remaining_episodes : ℕ) 
        (h_episodes_per_season : episodes_per_season = 20) 
        (h_fraction_watched : fraction_watched = 1 / 3) 
        (h_remaining_episodes : remaining_episodes = 160) : 
        ∃ (seasons : ℕ), seasons = 12 :=
by
  sorry

end number_of_seasons_l2070_207023


namespace largest_class_students_l2070_207025

theorem largest_class_students (x : ℕ) (h1 : 8 * x - (4 + 8 + 12 + 16 + 20 + 24 + 28) = 380) : x = 61 :=
by
  sorry

end largest_class_students_l2070_207025


namespace eq_has_one_integral_root_l2070_207065

theorem eq_has_one_integral_root :
  ∀ x : ℝ, (x - (9 / (x - 5)) = 4 - (9 / (x-5))) → x = 4 := by
  intros x h
  sorry

end eq_has_one_integral_root_l2070_207065


namespace max_abc_value_l2070_207071

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end max_abc_value_l2070_207071


namespace ancient_chinese_problem_l2070_207077

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end ancient_chinese_problem_l2070_207077


namespace range_of_values_l2070_207056

theorem range_of_values (x : ℝ) : (x^2 - 5 * x + 6 < 0) ↔ (2 < x ∧ x < 3) :=
sorry

end range_of_values_l2070_207056


namespace square_of_cube_plus_11_l2070_207008

def third_smallest_prime : ℕ := 5

theorem square_of_cube_plus_11 : (third_smallest_prime ^ 3)^2 + 11 = 15636 := by
  -- We will provide a proof later
  sorry

end square_of_cube_plus_11_l2070_207008


namespace cos_value_of_inclined_line_l2070_207017

variable (α : ℝ)
variable (l : ℝ) -- representing line as real (though we handle angles here)
variable (h_tan_line : ∃ α, tan α * (-1/2) = -1)

theorem cos_value_of_inclined_line (h_perpendicular : h_tan_line) :
  cos (2015 * Real.pi / 2 + 2 * α) = 4 / 5 := 
sorry

end cos_value_of_inclined_line_l2070_207017


namespace min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l2070_207089

-- Statements for minimum questions required for different number of cards 

theorem min_questions_30_cards (cards : Fin 30 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 10 :=
by
  sorry

theorem min_questions_31_cards (cards : Fin 31 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 11 :=
by
  sorry

theorem min_questions_32_cards (cards : Fin 32 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 12 :=
by
  sorry

theorem min_questions_50_cards_circle (cards : Fin 50 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 50 :=
by
  sorry

end min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l2070_207089


namespace transformed_parabola_l2070_207086

theorem transformed_parabola (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 3) → (y = 2 * (x + 1)^2 + 2) :=
by
  sorry

end transformed_parabola_l2070_207086


namespace min_value_reciprocals_l2070_207016

theorem min_value_reciprocals (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h_sum : x + y = 8) (h_prod : x * y = 12) : 
  (1/x + 1/y) = 2/3 :=
sorry

end min_value_reciprocals_l2070_207016


namespace original_grape_jelly_beans_l2070_207044

namespace JellyBeans

-- Definition of the problem conditions
variables (g c : ℕ)
axiom h1 : g = 3 * c
axiom h2 : g - 15 = 5 * (c - 5)

-- Proof goal statement
theorem original_grape_jelly_beans : g = 15 :=
by
  sorry

end JellyBeans

end original_grape_jelly_beans_l2070_207044


namespace special_even_diff_regular_l2070_207079

def first_n_even_sum (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def special_even_sum (n : ℕ) : ℕ :=
  let sum_cubes := (n * (n + 1) / 2) ^ 2
  let sum_squares := n * (n + 1) * (2 * n + 1) / 6
  2 * (sum_cubes + sum_squares)

theorem special_even_diff_regular : 
  let n := 100
  special_even_sum n - first_n_even_sum n = 51403900 :=
by
  sorry

end special_even_diff_regular_l2070_207079


namespace livestock_allocation_l2070_207041

theorem livestock_allocation :
  ∃ (x y z : ℕ), x + y + z = 100 ∧ 20 * x + 6 * y + z = 200 ∧ x = 5 ∧ y = 1 ∧ z = 94 :=
by
  sorry

end livestock_allocation_l2070_207041


namespace f_is_decreasing_on_interval_l2070_207024

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 - 2

theorem f_is_decreasing_on_interval :
  ∀ x y : ℝ, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f y < f x :=
by
  sorry

end f_is_decreasing_on_interval_l2070_207024


namespace tire_radius_increase_l2070_207087

noncomputable def radius_increase (initial_radius : ℝ) (odometer_initial : ℝ) (odometer_winter : ℝ) : ℝ :=
  let rotations := odometer_initial / ((2 * Real.pi * initial_radius) / 63360)
  let winter_circumference := (odometer_winter / rotations) * 63360
  let new_radius := winter_circumference / (2 * Real.pi)
  new_radius - initial_radius

theorem tire_radius_increase : radius_increase 16 520 505 = 0.32 := by
  sorry

end tire_radius_increase_l2070_207087


namespace alexis_pants_l2070_207057

theorem alexis_pants (P D : ℕ) (A_p : ℕ)
  (h1 : P + D = 13)
  (h2 : 3 * D = 18)
  (h3 : A_p = 3 * P) : A_p = 21 :=
  sorry

end alexis_pants_l2070_207057


namespace ann_hill_length_l2070_207002

/-- Given the conditions:
1. Mary slides down a hill that is 630 feet long at a speed of 90 feet/minute.
2. Ann slides down a hill at a rate of 40 feet/minute.
3. Ann's trip takes 13 minutes longer than Mary's.
Prove that the length of the hill Ann slides down is 800 feet. -/
theorem ann_hill_length
    (distance_Mary : ℕ) (speed_Mary : ℕ) 
    (speed_Ann : ℕ) (time_diff : ℕ)
    (h1 : distance_Mary = 630)
    (h2 : speed_Mary = 90)
    (h3 : speed_Ann = 40)
    (h4 : time_diff = 13) :
    speed_Ann * ((distance_Mary / speed_Mary) + time_diff) = 800 := 
by
    sorry

end ann_hill_length_l2070_207002


namespace claire_shirts_proof_l2070_207014

theorem claire_shirts_proof : 
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := 
by
  intro brian_shirts andrew_shirts steven_shirts claire_shirts
  intros h_brian h_andrew h_steven h_claire
  sorry

end claire_shirts_proof_l2070_207014


namespace smaller_acute_angle_is_20_degrees_l2070_207007

noncomputable def smaller_acute_angle (x : ℝ) : Prop :=
  let θ1 := 7 * x
  let θ2 := 2 * x
  θ1 + θ2 = 90 ∧ θ2 = 20

theorem smaller_acute_angle_is_20_degrees : ∃ x : ℝ, smaller_acute_angle x :=
  sorry

end smaller_acute_angle_is_20_degrees_l2070_207007


namespace pirate_coins_l2070_207075

def coins_remain (k : ℕ) (x : ℕ) : ℕ :=
  if k = 0 then x else coins_remain (k - 1) x * (15 - k) / 15

theorem pirate_coins (x : ℕ) :
  (∀ k < 15, (k + 1) * coins_remain k x % 15 = 0) → 
  coins_remain 14 x = 8442 :=
sorry

end pirate_coins_l2070_207075


namespace increasing_on_iff_decreasing_on_periodic_even_l2070_207028

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f x = f (x + p)
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

theorem increasing_on_iff_decreasing_on_periodic_even :
  (is_even f ∧ is_periodic f 2 ∧ is_increasing_on f 0 1) ↔ is_decreasing_on f 3 4 := 
by
  sorry

end increasing_on_iff_decreasing_on_periodic_even_l2070_207028


namespace jimin_shared_fruits_total_l2070_207066

-- Define the quantities given in the conditions
def persimmons : ℕ := 2
def apples : ℕ := 7

-- State the theorem to be proved
theorem jimin_shared_fruits_total : persimmons + apples = 9 := by
  sorry

end jimin_shared_fruits_total_l2070_207066


namespace part1_part2_l2070_207036

theorem part1 (x y : ℝ) (h1 : (1, 0) = (x, y)) (h2 : (0, 2) = (x, y)): 
    ∃ k b : ℝ, k = -2 ∧ b = 2 ∧ y = k * x + b := 
by 
  sorry

theorem part2 (m n : ℝ) (h : n = -2 * m + 2) (hm : -2 < m ∧ m ≤ 3):
    -4 ≤ n ∧ n < 6 := 
by 
  sorry

end part1_part2_l2070_207036


namespace pencils_evenly_distributed_l2070_207001

-- Define the initial number of pencils Eric had
def initialPencils : Nat := 150

-- Define the additional pencils brought by another teacher
def additionalPencils : Nat := 30

-- Define the total number of containers
def numberOfContainers : Nat := 5

-- Define the total number of pencils after receiving additional pencils
def totalPencils := initialPencils + additionalPencils

-- Define the number of pencils per container after even distribution
def pencilsPerContainer := totalPencils / numberOfContainers

-- Statement of the proof problem
theorem pencils_evenly_distributed :
  pencilsPerContainer = 36 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end pencils_evenly_distributed_l2070_207001


namespace tangent_parallel_to_line_l2070_207092

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_to_line :
  ∃ a b : ℝ, (f a = b) ∧ (3 * a^2 + 1 = 4) ∧ (P = (1, 0) ∨ P = (-1, -4)) :=
by
  sorry

end tangent_parallel_to_line_l2070_207092


namespace problem_sol_l2070_207076

open Complex

theorem problem_sol (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : (a + i) / i = 1 + b * i) : a + b = 0 :=
sorry

end problem_sol_l2070_207076


namespace three_digit_numbers_not_multiple_of_3_5_7_l2070_207063

theorem three_digit_numbers_not_multiple_of_3_5_7 : 
  let total_three_digit_numbers := 900
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_5 := (995 - 100) / 5 + 1
  let multiples_of_7 := (994 - 105) / 7 + 1
  let multiples_of_15 := (990 - 105) / 15 + 1
  let multiples_of_21 := (987 - 105) / 21 + 1
  let multiples_of_35 := (980 - 105) / 35 + 1
  let multiples_of_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_of_3 + multiples_of_5 + multiples_of_7 - multiples_of_15 - multiples_of_21 - multiples_of_35 + multiples_of_105
  let non_multiples_total := total_three_digit_numbers - total_multiples
  non_multiples_total = 412 :=
by
  sorry

end three_digit_numbers_not_multiple_of_3_5_7_l2070_207063


namespace emily_curtains_purchase_l2070_207074

theorem emily_curtains_purchase 
    (c : ℕ) 
    (curtain_cost : ℕ := 30)
    (print_count : ℕ := 9)
    (print_cost_per_unit : ℕ := 15)
    (installation_cost : ℕ := 50)
    (total_cost : ℕ := 245) :
    (curtain_cost * c + print_count * print_cost_per_unit + installation_cost = total_cost) → c = 2 :=
by
  sorry

end emily_curtains_purchase_l2070_207074


namespace problem_statement_l2070_207040

noncomputable def f (x : ℝ) : ℝ := x / Real.cos x

theorem problem_statement (x1 x2 x3 : ℝ) (h1 : abs x1 < Real.pi / 2)
                         (h2 : abs x2 < Real.pi / 2) (h3 : abs x3 < Real.pi / 2)
                         (c1 : f x1 + f x2 ≥ 0) (c2 : f x2 + f x3 ≥ 0) (c3 : f x3 + f x1 ≥ 0) :
  f (x1 + x2 + x3) ≥ 0 :=
sorry

end problem_statement_l2070_207040


namespace straws_to_adult_pigs_l2070_207090

theorem straws_to_adult_pigs (total_straws : ℕ) (num_piglets : ℕ) (straws_per_piglet : ℕ)
  (straws_adult_pigs : ℕ) (straws_piglets : ℕ) :
  total_straws = 300 →
  num_piglets = 20 →
  straws_per_piglet = 6 →
  (straws_piglets = num_piglets * straws_per_piglet) →
  (straws_adult_pigs = straws_piglets) →
  straws_adult_pigs = 120 :=
by
  intros h_total h_piglets h_straws_per_piglet h_straws_piglets h_equal
  subst h_total
  subst h_piglets
  subst h_straws_per_piglet
  subst h_straws_piglets
  subst h_equal
  sorry

end straws_to_adult_pigs_l2070_207090


namespace total_income_l2070_207005

theorem total_income (I : ℝ) 
  (h1 : I * 0.225 = 40000) : 
  I = 177777.78 :=
by
  sorry

end total_income_l2070_207005


namespace number_of_insects_l2070_207097

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h : total_legs = 54) (k : legs_per_insect = 6) :
  total_legs / legs_per_insect = 9 := by
  sorry

end number_of_insects_l2070_207097


namespace pages_for_ten_dollars_l2070_207078

theorem pages_for_ten_dollars (p c pages_per_cent : ℕ) (dollars cents : ℕ) (h1 : p = 5) (h2 : c = 10) (h3 : pages_per_cent = p / c) (h4 : dollars = 10) (h5 : cents = 100 * dollars) :
  (cents * pages_per_cent) = 500 :=
by
  sorry

end pages_for_ten_dollars_l2070_207078


namespace minimum_expr_value_l2070_207099

noncomputable def expr_min_value (a : ℝ) (h : a > 1) : ℝ :=
  a + 2 / (a - 1)

theorem minimum_expr_value (a : ℝ) (h : a > 1) :
  expr_min_value a h = 1 + 2 * Real.sqrt 2 :=
sorry

end minimum_expr_value_l2070_207099


namespace parallel_line_distance_l2070_207010

-- Definition of a line
structure Line where
  m : ℚ -- slope
  c : ℚ -- y-intercept

-- Given conditions
def given_line : Line :=
  { m := 3 / 4, c := 6 }

-- Prove that there exist lines parallel to the given line and 5 units away from it
theorem parallel_line_distance (L : Line)
  (h_parallel : L.m = given_line.m)
  (h_distance : abs (L.c - given_line.c) = 25 / 4) :
  (L.c = 12.25) ∨ (L.c = -0.25) :=
sorry

end parallel_line_distance_l2070_207010


namespace problem1_problem2_problem3_l2070_207068

section problem

variable (m : ℝ)

-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions
def p : Prop := (16 * m^2 - 4) ≥ 0

-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀^2 - 2x₀ - 1 > 0
def q : Prop := ∃ (x₀ : ℝ), (m * x₀^2 - 2 * x₀ - 1) > 0

-- Solution to (1): If p is true, the range of values for m
theorem problem1 (hp : p m) : m ≥ 1/2 ∨ m ≤ -1/2 := sorry

-- Solution to (2): If q is true, the range of values for m
theorem problem2 (hq : q m) : m > -1 := sorry

-- Solution to (3): If both p and q are false but either p or q is true,
-- find the range of values for m
theorem problem3 (hnp : ¬p m) (hnq : ¬q m) (hpq : p m ∨ q m) : -1 < m ∧ m < 1/2 := sorry

end problem

end problem1_problem2_problem3_l2070_207068


namespace Madelyn_daily_pizza_expense_l2070_207050

theorem Madelyn_daily_pizza_expense (total_expense : ℕ) (days_in_may : ℕ) 
  (h1 : total_expense = 465) (h2 : days_in_may = 31) : 
  total_expense / days_in_may = 15 := 
by
  sorry

end Madelyn_daily_pizza_expense_l2070_207050


namespace mark_theater_expense_l2070_207098

noncomputable def price_per_performance (hours_per_performance : ℕ) (price_per_hour : ℕ) : ℕ :=
  hours_per_performance * price_per_hour

noncomputable def total_cost (num_weeks : ℕ) (num_visits_per_week : ℕ) (price_per_performance : ℕ) : ℕ :=
  num_weeks * num_visits_per_week * price_per_performance

theorem mark_theater_expense :
  ∀(num_weeks num_visits_per_week hours_per_performance price_per_hour : ℕ),
  num_weeks = 6 →
  num_visits_per_week = 1 →
  hours_per_performance = 3 →
  price_per_hour = 5 →
  total_cost num_weeks num_visits_per_week (price_per_performance hours_per_performance price_per_hour) = 90 :=
by
  intros num_weeks num_visits_per_week hours_per_performance price_per_hour
  intro h_num_weeks h_num_visits_per_week h_hours_per_performance h_price_per_hour
  rw [h_num_weeks, h_num_visits_per_week, h_hours_per_performance, h_price_per_hour]
  sorry

end mark_theater_expense_l2070_207098


namespace find_a_range_l2070_207052

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else (a + 1) / x

theorem find_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 
  - (7 / 2) ≤ a ∧ a ≤ -2 :=
by
  sorry

end find_a_range_l2070_207052


namespace push_mower_cuts_one_acre_per_hour_l2070_207091

noncomputable def acres_per_hour_push_mower : ℕ :=
  let total_acres := 8
  let fraction_riding := 3 / 4
  let riding_mower_rate := 2
  let mowing_hours := 5
  let acres_riding := fraction_riding * total_acres
  let time_riding_mower := acres_riding / riding_mower_rate
  let remaining_hours := mowing_hours - time_riding_mower
  let remaining_acres := total_acres - acres_riding
  remaining_acres / remaining_hours

theorem push_mower_cuts_one_acre_per_hour :
  acres_per_hour_push_mower = 1 := 
by 
  -- Detailed proof steps would go here.
  sorry

end push_mower_cuts_one_acre_per_hour_l2070_207091


namespace rabbit_prob_top_or_bottom_l2070_207030

-- Define the probability function for the rabbit to hit the top or bottom border from a given point
noncomputable def prob_reach_top_or_bottom (start : ℕ × ℕ) (board_end : ℕ × ℕ) : ℚ :=
  sorry -- Detailed probability computation based on recursive and symmetry argument

-- The proof statement for the starting point (2, 3) on a rectangular board extending to (6, 5)
theorem rabbit_prob_top_or_bottom : prob_reach_top_or_bottom (2, 3) (6, 5) = 17 / 24 :=
  sorry

end rabbit_prob_top_or_bottom_l2070_207030


namespace complement_union_eq_zero_or_negative_l2070_207043

def U : Set ℝ := Set.univ

def P : Set ℝ := { x | x > 1 }

def Q : Set ℝ := { x | x * (x - 2) < 0 }

theorem complement_union_eq_zero_or_negative :
  (U \ (P ∪ Q)) = { x | x ≤ 0 } := by
  sorry

end complement_union_eq_zero_or_negative_l2070_207043


namespace domain_transform_l2070_207070

variable (f : ℝ → ℝ)

theorem domain_transform (h : ∀ x, -1 ≤ x ∧ x ≤ 4 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f y = 2 * x - 1 :=
sorry

end domain_transform_l2070_207070


namespace inequalities_of_function_nonneg_l2070_207018

theorem inequalities_of_function_nonneg (a b A B : ℝ)
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := sorry

end inequalities_of_function_nonneg_l2070_207018


namespace hot_dog_cost_l2070_207059

variable {Real : Type} [LinearOrderedField Real]

-- Define the cost of a hamburger and a hot dog
variables (h d : Real)

-- Arthur's buying conditions
def condition1 := 3 * h + 4 * d = 10
def condition2 := 2 * h + 3 * d = 7

-- Problem statement: Proving that the cost of a hot dog is 1 dollar
theorem hot_dog_cost
    (h d : Real)
    (hc1 : condition1 h d)
    (hc2 : condition2 h d) : 
    d = 1 :=
sorry

end hot_dog_cost_l2070_207059


namespace value_of_abs_m_minus_n_l2070_207055

theorem value_of_abs_m_minus_n  (m n : ℝ) (h_eq : ∀ x, (x^2 - 2 * x + m) * (x^2 - 2 * x + n) = 0)
  (h_arith_seq : ∀ x₁ x₂ x₃ x₄ : ℝ, x₁ + x₂ = 2 ∧ x₃ + x₄ = 2 ∧ x₁ = 1 / 4 ∧ x₂ = 3 / 4 ∧ x₃ = 5 / 4 ∧ x₄ = 7 / 4) :
  |m - n| = 1 / 2 :=
by
  sorry

end value_of_abs_m_minus_n_l2070_207055


namespace fencing_cost_l2070_207069

theorem fencing_cost (w : ℝ) (h : ℝ) (p : ℝ) (cost_per_meter : ℝ) 
  (hw : h = w + 10) (perimeter : p = 220) (cost_rate : cost_per_meter = 6.5) : 
  ((p * cost_per_meter) = 1430) := by 
  sorry

end fencing_cost_l2070_207069


namespace maximum_rubles_received_l2070_207096

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l2070_207096


namespace solve_quadratic_l2070_207060

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l2070_207060


namespace rubles_greater_than_seven_l2070_207061

theorem rubles_greater_than_seven (x : ℕ) (h : x > 7) : ∃ a b : ℕ, x = 3 * a + 5 * b :=
sorry

end rubles_greater_than_seven_l2070_207061


namespace tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l2070_207000

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

theorem tangent_line_eq_at_1 : 
  ∃ c : ℝ, ∀ x y : ℝ, y = f x → (x = 1 → y = 0) → y = 2 * (x - 1) → 2 * x - y - 2 = 0 := 
by sorry

theorem max_value_on_interval :
  ∃ xₘ : ℝ, (0 ≤ xₘ ∧ xₘ ≤ 2) ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 2) → f x ≤ 6 :=
by sorry

theorem unique_solution_exists :
  ∃! x₀ : ℝ, f x₀ = g x₀ :=
by sorry

end tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l2070_207000


namespace other_root_of_equation_l2070_207021

theorem other_root_of_equation (m : ℤ) (h₁ : (2 : ℤ) ∈ {x : ℤ | x ^ 2 - 3 * x - m = 0}) : 
  ∃ x, x ≠ 2 ∧ (x ^ 2 - 3 * x - m = 0) ∧ x = 1 :=
by {
  sorry
}

end other_root_of_equation_l2070_207021


namespace hyperbola_parabola_focus_l2070_207003

theorem hyperbola_parabola_focus (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, (1/k^2) * y^2 = 0 ∧ x^2 - (y^2 / k^2) = 1) ∧ (∃ x : ℝ, y^2 = 8 * x) →
  k = Real.sqrt 3 :=
by sorry

end hyperbola_parabola_focus_l2070_207003


namespace original_deck_size_l2070_207037

-- Define the conditions
def boys_kept_away (remaining_cards kept_away_cards : ℕ) : Prop :=
  remaining_cards + kept_away_cards = 52

-- Define the problem
theorem original_deck_size (remaining_cards : ℕ) (kept_away_cards := 2) :
  boys_kept_away remaining_cards kept_away_cards → remaining_cards + kept_away_cards = 52 :=
by
  intro h
  exact h

end original_deck_size_l2070_207037


namespace line_through_vertex_has_two_a_values_l2070_207047

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end line_through_vertex_has_two_a_values_l2070_207047


namespace prove_AF_eq_l2070_207031

-- Definitions
variables {A B C E F : Type*}
variables [Field A] [Field B] [Field C] [Field E] [Field F]

-- Conditions
def triangle_ABC (AB AC : ℝ) (h : AB > AC) : Prop := true

def external_bisector (angleA : ℝ) (circumcircle_meets : ℝ) : Prop := true

def foot_perpendicular (E AB : ℝ) : Prop := true

-- Theorem statement
theorem prove_AF_eq (AB AC AF : ℝ) (h_triangle : triangle_ABC AB AC (by sorry))
  (h_external_bisector : external_bisector (by sorry) (by sorry))
  (h_foot_perpendicular : foot_perpendicular (by sorry) AB) :
  2 * AF = AB - AC := by
  sorry

end prove_AF_eq_l2070_207031


namespace jose_investment_l2070_207020

theorem jose_investment 
  (T_investment : ℕ := 30000) -- Tom's investment in Rs.
  (J_months : ℕ := 10)        -- Jose's investment period in months
  (T_months : ℕ := 12)        -- Tom's investment period in months
  (total_profit : ℕ := 72000) -- Total profit in Rs.
  (jose_profit : ℕ := 40000)  -- Jose's share of profit in Rs.
  : ∃ X : ℕ, (jose_profit * (T_investment * T_months)) = ((total_profit - jose_profit) * (X * J_months)) ∧ X = 45000 :=
  sorry

end jose_investment_l2070_207020


namespace triangle_inequality_l2070_207013

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) ≥ 3 :=
by
  sorry

end triangle_inequality_l2070_207013


namespace find_locus_of_p_l2070_207062

noncomputable def locus_of_point_p (a b : ℝ) : Set (ℝ × ℝ) :=
{p | (p.snd = 0 ∧ -a < p.fst ∧ p.fst < a) ∨ (p.fst = a^2 / Real.sqrt (a^2 + b^2))}

theorem find_locus_of_p (a b : ℝ) (P : ℝ × ℝ) :
  (∃ (x0 y0: ℝ),
      P = (x0, y0) ∧
      ( ∃ (x1 y1 x2 y2 : ℝ),
        (x0 ≠ 0 ∨ y0 ≠ 0) ∧
        (x1 ≠ x2 ∨ y1 ≠ y2) ∧
        (y0 = 0 ∨ (b^2 * x0 = -a^2 * (x0 - Real.sqrt (a^2 + b^2)))) ∧
        ((y0 = 0 ∧ -a < x0 ∧ x0 < a) ∨ x0 = a^2 / Real.sqrt (a^2 + b^2)))) ↔ 
  P ∈ locus_of_point_p a b :=
sorry

end find_locus_of_p_l2070_207062


namespace range_of_m_l2070_207073

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < 3) ↔ (x / 3 < 1 - (x - 3) / 6 ∧ x < m)) → m ≥ 3 :=
by
  sorry

end range_of_m_l2070_207073


namespace cost_per_serving_in_cents_after_coupon_l2070_207085

def oz_per_serving : ℝ := 1
def price_per_bag : ℝ := 25
def bag_weight : ℝ := 40
def coupon : ℝ := 5
def dollars_to_cents (d : ℝ) : ℝ := d * 100

theorem cost_per_serving_in_cents_after_coupon : 
  dollars_to_cents ((price_per_bag - coupon) / bag_weight) = 50 := by
  sorry

end cost_per_serving_in_cents_after_coupon_l2070_207085


namespace cubic_inequality_solution_l2070_207009

theorem cubic_inequality_solution (x : ℝ) :
  (x^3 - 2 * x^2 - x + 2 > 0) ∧ (x < 3) ↔ (x < -1 ∨ (1 < x ∧ x < 3)) := 
sorry

end cubic_inequality_solution_l2070_207009


namespace parallel_lines_chords_distance_l2070_207012

theorem parallel_lines_chords_distance
  (r d : ℝ)
  (h1 : ∀ (P Q : ℝ), P = Q + d / 2 → Q = P - d / 2)
  (h2 : ∀ (A B : ℝ), A = B + 3 * d / 2 → B = A - 3 * d / 2)
  (chords : ∀ (l1 l2 l3 l4 : ℝ), (l1 = 40 ∧ l2 = 40 ∧ l3 = 36 ∧ l4 = 36)) :
  d = 1.46 :=
sorry

end parallel_lines_chords_distance_l2070_207012


namespace solution_set_inequality_range_of_m_l2070_207026

def f (x : ℝ) : ℝ := |2 * x + 1| + 2 * |x - 3|

theorem solution_set_inequality :
  ∀ x : ℝ, f x ≤ 7 * x ↔ x ≥ 1 :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x = |m|) ↔ (m ≥ 7 ∨ m ≤ -7) :=
by sorry

end solution_set_inequality_range_of_m_l2070_207026


namespace max_n_divisor_l2070_207094

theorem max_n_divisor (k n : ℕ) (h1 : 81849 % n = k) (h2 : 106392 % n = k) (h3 : 124374 % n = k) : n = 243 := by
  sorry

end max_n_divisor_l2070_207094


namespace find_phi_l2070_207046

theorem find_phi :
  ∀ φ : ℝ, 0 < φ ∧ φ < 90 → 
    (∃θ : ℝ, θ = 144 ∧ θ = 2 * φ ∧ (144 - θ) = 72) → φ = 81 :=
by
  intros φ h1 h2
  sorry

end find_phi_l2070_207046


namespace cos_double_angle_l2070_207004

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 2/3) : Real.cos (2 * θ) = -1/9 := 
  sorry

end cos_double_angle_l2070_207004


namespace common_focus_hyperbola_ellipse_l2070_207015

theorem common_focus_hyperbola_ellipse (p : ℝ) (c : ℝ) :
  (0 < p ∧ p < 8) →
  (c = Real.sqrt (3 + 1)) →
  (c = Real.sqrt (8 - p)) →
  p = 4 := by
sorry

end common_focus_hyperbola_ellipse_l2070_207015


namespace tan_C_value_b_value_l2070_207093

-- Define variables and conditions
variable (A B C a b c : ℝ)
variable (A_eq : A = Real.pi / 4)
variable (cond : b^2 - a^2 = 1 / 4 * c^2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 5 / 2)

-- First part: Prove tan(C) = 4 given the conditions
theorem tan_C_value : A = Real.pi / 4 ∧ b^2 - a^2 = 1 / 4 * c^2 → Real.tan C = 4 := by
  intro h
  sorry

-- Second part: Prove b = 5 / 2 given the area condition
theorem b_value : (1 / 2 * b * c * Real.sin (Real.pi / 4) = 5 / 2) → b = 5 / 2 := by
  intro h
  sorry

end tan_C_value_b_value_l2070_207093


namespace min_chord_length_l2070_207006

variable (α : ℝ)

def curve_eq (x y α : ℝ) :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line_eq (x : ℝ) :=
  x = Real.pi / 4

theorem min_chord_length :
  ∃ d, (∀ α : ℝ, ∃ y1 y2 : ℝ, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ d = |y2 - y1|) ∧
  (∀ α : ℝ, ∃ y1 y2, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ |y2 - y1| ≥ d) :=
sorry

end min_chord_length_l2070_207006


namespace ratio_of_c_and_d_l2070_207054

theorem ratio_of_c_and_d
  (x y c d : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c)
  (h2 : 9 * y - 12 * x = d) :
  c / d = -2 / 3 := 
  sorry

end ratio_of_c_and_d_l2070_207054


namespace growth_comparison_l2070_207080

theorem growth_comparison (x : ℝ) (h : ℝ) (hx : x > 0) : 
  (0 < x ∧ x < 1 / 2 → (x + h) - x > ((x + h)^2 - x^2)) ∧
  (x > 1 / 2 → ((x + h)^2 - x^2) > (x + h) - x) :=
by
  sorry

end growth_comparison_l2070_207080


namespace intersection_A_B_l2070_207081

def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 4 + 3 * y^2 / 4 = 1) }
def B : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (y = x^2) }

theorem intersection_A_B :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2} = 
  {x : ℝ | ∃ y : ℝ, ((x, y) ∈ A ∧ (x, y) ∈ B)} :=
by
  sorry

end intersection_A_B_l2070_207081


namespace solve_coin_problem_l2070_207053

def coin_problem : Prop :=
  ∃ (x y z : ℕ), 
  1 * x + 2 * y + 5 * z = 71 ∧ 
  x = y ∧ 
  x + y + z = 31 ∧ 
  x = 12 ∧ 
  y = 12 ∧ 
  z = 7

theorem solve_coin_problem : coin_problem :=
  sorry

end solve_coin_problem_l2070_207053


namespace total_feet_l2070_207035

theorem total_feet (H C F : ℕ) (h1 : H + C = 48) (h2 : H = 28) :
  F = 2 * H + 4 * C → F = 136 :=
by
  -- substitute H = 28 and perform the calculations
  sorry

end total_feet_l2070_207035


namespace harriet_speed_l2070_207083

-- Define the conditions
def return_speed := 140 -- speed from B-town to A-ville in km/h
def total_trip_time := 5 -- total trip time in hours
def trip_time_to_B := 2.8 -- trip time from A-ville to B-town in hours

-- Define the theorem to prove
theorem harriet_speed {r_speed : ℝ} {t_time : ℝ} {t_time_B : ℝ} 
  (h1 : r_speed = 140) 
  (h2 : t_time = 5) 
  (h3 : t_time_B = 2.8) : 
  ((r_speed * (t_time - t_time_B)) / t_time_B) = 110 :=
by 
  -- Assume we have completed proof steps here.
  sorry

end harriet_speed_l2070_207083


namespace probZ_eq_1_4_l2070_207051

noncomputable def probX : ℚ := 1/4
noncomputable def probY : ℚ := 1/3
noncomputable def probW : ℚ := 1/6

theorem probZ_eq_1_4 :
  let probZ : ℚ := 1 - (probX + probY + probW)
  probZ = 1/4 :=
by
  sorry

end probZ_eq_1_4_l2070_207051


namespace determine_weights_of_balls_l2070_207038

theorem determine_weights_of_balls (A B C D E m1 m2 m3 m4 m5 m6 m7 m8 m9 : ℝ)
  (h1 : m1 = A)
  (h2 : m2 = B)
  (h3 : m3 = C)
  (h4 : m4 = A + D)
  (h5 : m5 = A + E)
  (h6 : m6 = B + D)
  (h7 : m7 = B + E)
  (h8 : m8 = C + D)
  (h9 : m9 = C + E) :
  ∃ (A' B' C' D' E' : ℝ), 
    ((A' = A ∨ B' = B ∨ C' = C ∨ D' = D ∨ E' = E) ∧
     (A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ E' ∧
      B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ E' ∧
      C' ≠ D' ∧ C' ≠ E' ∧
      D' ≠ E')) :=
sorry

end determine_weights_of_balls_l2070_207038


namespace boat_stream_speed_l2070_207032

/-- A boat can travel with a speed of 22 km/hr in still water. 
If the speed of the stream is unknown, the boat takes 7 hours 
to go 189 km downstream. What is the speed of the stream?
-/
theorem boat_stream_speed (v : ℝ) : (22 + v) * 7 = 189 → v = 5 :=
by
  intro h
  sorry

end boat_stream_speed_l2070_207032


namespace polynomial_multiplication_identity_l2070_207067

-- Statement of the problem
theorem polynomial_multiplication_identity (x : ℝ) : 
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = (12 / 5) * x^2 :=
by
  sorry

end polynomial_multiplication_identity_l2070_207067


namespace find_x_given_scores_l2070_207064

theorem find_x_given_scores : 
  ∃ x : ℝ, (9.1 + 9.3 + x + 9.2 + 9.4) / 5 = 9.3 ∧ x = 9.5 :=
by {
  sorry
}

end find_x_given_scores_l2070_207064


namespace polynomial_difference_square_l2070_207045

theorem polynomial_difference_square (a : Fin 11 → ℝ) (x : ℝ) (sqrt2 : ℝ)
  (h_eq : (sqrt2 - x)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + 
          a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10) : 
  ((a 0 + a 2 + a 4 + a 6 + a 8 + a 10)^2 - (a 1 + a 3 + a 5 + a 7 + a 9)^2 = 1) :=
by
  sorry

end polynomial_difference_square_l2070_207045


namespace evaluate_expression_l2070_207049

-- Defining the primary condition
def condition (x : ℝ) : Prop := x > 3

-- Definition of the expression we need to evaluate
def expression (x : ℝ) : ℝ := abs (1 - abs (x - 3))

-- Stating the theorem
theorem evaluate_expression (x : ℝ) (h : condition x) : expression x = abs (4 - x) := 
by 
  -- Since the problem only asks for the statement, the proof is left as sorry.
  sorry

end evaluate_expression_l2070_207049


namespace robie_initial_cards_l2070_207082

-- Definitions of the problem conditions
def each_box_cards : ℕ := 25
def extra_cards : ℕ := 11
def given_away_boxes : ℕ := 6
def remaining_boxes : ℕ := 12

-- The final theorem we need to prove
theorem robie_initial_cards : 
  (given_away_boxes + remaining_boxes) * each_box_cards + extra_cards = 461 :=
by
  sorry

end robie_initial_cards_l2070_207082


namespace monkey_height_37_minutes_l2070_207088

noncomputable def monkey_climb (minutes : ℕ) : ℕ :=
if minutes = 37 then 60 else 0

theorem monkey_height_37_minutes : (monkey_climb 37) = 60 := 
by
  sorry

end monkey_height_37_minutes_l2070_207088
