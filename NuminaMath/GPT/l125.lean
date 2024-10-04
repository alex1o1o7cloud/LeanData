import Mathlib

namespace balloon_arrangements_l125_125848

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l125_125848


namespace find_fraction_squares_l125_125661

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end find_fraction_squares_l125_125661


namespace non_increasing_condition_l125_125940

variable {a b : ℝ} (f : ℝ → ℝ)

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem non_increasing_condition (h₀ : ∀ x y, a ≤ x → x < y → y ≤ b → ¬ (f x > f y)) :
  ¬ increasing_on_interval f a b :=
by
  intro h1
  have : ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y := h1
  exact sorry

end non_increasing_condition_l125_125940


namespace distinct_arrangements_balloon_l125_125823

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l125_125823


namespace distinct_arrangements_balloon_l125_125815

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125815


namespace product_of_integers_is_eight_l125_125723

-- Define three different positive integers a, b, c such that they sum to 7
def sum_to_seven (a b c : ℕ) : Prop := a + b + c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Prove that the product of these integers is 8
theorem product_of_integers_is_eight (a b c : ℕ) (h : sum_to_seven a b c) : a * b * c = 8 := by sorry

end product_of_integers_is_eight_l125_125723


namespace fg_minus_gf_eq_zero_l125_125941

noncomputable def f (x : ℝ) : ℝ := 4 * x + 6

noncomputable def g (x : ℝ) : ℝ := x / 2 - 1

theorem fg_minus_gf_eq_zero (x : ℝ) : (f (g x)) - (g (f x)) = 0 :=
by
  sorry

end fg_minus_gf_eq_zero_l125_125941


namespace om_4_2_eq_18_l125_125385

def om (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem om_4_2_eq_18 : om 4 2 = 18 :=
by
  sorry

end om_4_2_eq_18_l125_125385


namespace probability_non_adjacent_zeros_l125_125358

-- Define the conditions
def num_ones : ℕ := 4
def num_zeros : ℕ := 2

-- Calculate combinations
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Calculate total arrangements
def total_arrangements : ℕ := combination 6 2

-- Calculate non-adjacent arrangements
def non_adjacent_arrangements : ℕ := combination 5 2

-- Define the probability and prove the equality
theorem probability_non_adjacent_zeros :
  (non_adjacent_arrangements.toRat / total_arrangements.toRat) = (2 / 3) := by
  sorry

end probability_non_adjacent_zeros_l125_125358


namespace spa_polish_total_digits_l125_125758

theorem spa_polish_total_digits (girls : ℕ) (digits_per_girl : ℕ) (total_digits : ℕ)
  (h1 : girls = 5) (h2 : digits_per_girl = 20) : total_digits = 100 :=
by
  sorry

end spa_polish_total_digits_l125_125758


namespace kyunghwan_spent_the_most_l125_125097

-- Define initial pocket money for everyone
def initial_money : ℕ := 20000

-- Define remaining money
def remaining_S : ℕ := initial_money / 4
def remaining_K : ℕ := initial_money / 8
def remaining_D : ℕ := initial_money / 5

-- Calculate spent money
def spent_S : ℕ := initial_money - remaining_S
def spent_K : ℕ := initial_money - remaining_K
def spent_D : ℕ := initial_money - remaining_D

theorem kyunghwan_spent_the_most 
  (h1 : remaining_S = initial_money / 4)
  (h2 : remaining_K = initial_money / 8)
  (h3 : remaining_D = initial_money / 5) :
  spent_K > spent_S ∧ spent_K > spent_D :=
by
  -- Proof skipped
  sorry

end kyunghwan_spent_the_most_l125_125097


namespace sin_lt_alpha_lt_tan_l125_125932

variable {α : ℝ}

theorem sin_lt_alpha_lt_tan (h1 : 0 < α) (h2 : α < π / 2) : sin α < α ∧ α < tan α :=
  sorry

end sin_lt_alpha_lt_tan_l125_125932


namespace cost_price_of_article_l125_125587

theorem cost_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) 
    (h1 : SP = 100) 
    (h2 : profit_percent = 0.20) 
    (h3 : SP = CP * (1 + profit_percent)) : 
    CP = 83.33 :=
by
  sorry

end cost_price_of_article_l125_125587


namespace part1_part2_l125_125339

def f (x : ℝ) : ℝ := x^2 - 1
def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, |f x| = g x a → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) → a ≤ -2 :=
sorry

end part1_part2_l125_125339


namespace solve_linear_equation_l125_125095

theorem solve_linear_equation (x : ℝ) (h : 2 * x - 1 = 1) : x = 1 :=
sorry

end solve_linear_equation_l125_125095


namespace belindas_age_l125_125106

theorem belindas_age (T B : ℕ) (h1 : T + B = 56) (h2 : B = 2 * T + 8) (h3 : T = 16) : B = 40 :=
by
  sorry

end belindas_age_l125_125106


namespace solve_inequality_l125_125466

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem solve_inequality (x : ℝ) : (otimes (x-2) (x+2) < 2) ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by
  sorry

end solve_inequality_l125_125466


namespace find_mean_l125_125087

noncomputable def mean_of_normal_distribution (σ : ℝ) (value : ℝ) (std_devs : ℝ) : ℝ :=
value + std_devs * σ

theorem find_mean
  (σ : ℝ := 1.5)
  (value : ℝ := 12)
  (std_devs : ℝ := 2)
  (h : value = mean_of_normal_distribution σ (value - std_devs * σ) std_devs) :
  mean_of_normal_distribution σ value std_devs = 15 :=
sorry

end find_mean_l125_125087


namespace geometric_sequence_sum_S8_l125_125518

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l125_125518


namespace twenty_cows_twenty_days_l125_125370

-- Defining the initial conditions as constants
def num_cows : ℕ := 20
def days_one_cow_eats_one_bag : ℕ := 20
def bags_eaten_by_one_cow_in_days (d : ℕ) : ℕ := if d = days_one_cow_eats_one_bag then 1 else 0

-- Defining the total bags eaten by all cows
def total_bags_eaten_by_cows (cows : ℕ) (days : ℕ) : ℕ :=
  cows * (days / days_one_cow_eats_one_bag)

-- Statement to be proved: In 20 days, 20 cows will eat 20 bags of husk
theorem twenty_cows_twenty_days :
  total_bags_eaten_by_cows num_cows days_one_cow_eats_one_bag = 20 := sorry

end twenty_cows_twenty_days_l125_125370


namespace balloon_permutations_l125_125821

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l125_125821


namespace parallel_lines_l125_125034

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end parallel_lines_l125_125034


namespace solve_system_of_equations_l125_125396

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), (3 * x + 4 * y = 16) ∧ (5 * x - 6 * y = 33) ∧ x = 6 ∧ y = -1/2 :=
by
  have h1 : 3 * 6 + 4 * (-1/2) = 16 := by norm_num
  have h2 : 5 * 6 - 6 * (-1/2) = 33 := by norm_num
  use 6, -1/2
  exact ⟨h1, h2, rfl, rfl⟩

end solve_system_of_equations_l125_125396


namespace donny_spent_total_on_friday_and_sunday_l125_125783

noncomputable def daily_savings (initial: ℚ) (increase_rate: ℚ) (days: List ℚ) : List ℚ :=
days.scanl (λ acc day => acc * increase_rate + acc) initial

noncomputable def thursday_savings : ℚ := (daily_savings 15 (1 + 0.1) [15, 15, 15]).sum

noncomputable def friday_spent : ℚ := thursday_savings * 0.5

noncomputable def remaining_after_friday : ℚ := thursday_savings - friday_spent

noncomputable def saturday_savings (thursday: ℚ) : ℚ := thursday * (1 - 0.20)

noncomputable def total_savings_saturday : ℚ := remaining_after_friday + saturday_savings thursday_savings

noncomputable def sunday_spent : ℚ := total_savings_saturday * 0.40

noncomputable def total_spent : ℚ := friday_spent + sunday_spent

theorem donny_spent_total_on_friday_and_sunday : total_spent = 55.13 := by
  sorry

end donny_spent_total_on_friday_and_sunday_l125_125783


namespace min_value_expression_l125_125493

theorem min_value_expression (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 1 / (x - 2) ∧ y = 4 := 
sorry

end min_value_expression_l125_125493


namespace regular_polygon_sides_l125_125440

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l125_125440


namespace lollipops_given_l125_125177

theorem lollipops_given (initial_people later_people : ℕ) (total_people groups_of_five : ℕ) :
  initial_people = 45 →
  later_people = 15 →
  total_people = initial_people + later_people →
  groups_of_five = total_people / 5 →
  total_people = 60 →
  groups_of_five = 12 :=
by intros; sorry

end lollipops_given_l125_125177


namespace general_term_seq_l125_125193

theorem general_term_seq 
  (a : ℕ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 5/3) 
  (h_rec : ∀ n, n > 0 → a (n + 2) = (5 / 3) * a (n + 1) - (2 / 3) * a n) : 
  ∀ n, a n = 2 - (3 / 2) * (2 / 3)^n :=
by
  sorry

end general_term_seq_l125_125193


namespace problem1_l125_125751

variable {x : ℝ} {b c : ℝ}

theorem problem1 (hb : b = 9) (hc : c = -11) :
  b + c = -2 := 
by
  simp [hb, hc]
  sorry

end problem1_l125_125751


namespace distinct_arrangements_balloon_l125_125826

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l125_125826


namespace geometric_seq_sum_l125_125528

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l125_125528


namespace probability_negative_product_l125_125776

open_locale classical -- Use classical logic

theorem probability_negative_product :
  let S := {-3, -1, 2, 6, 5, -4}
  let total_ways := nat.choose (fintype.card S) 2
  let favorable_ways := 3 * 3 -- 3 ways to choose a negative from 3, 3 ways to choose a positive from 3
  (favorable_ways : ℝ) / total_ways = (3 : ℝ) / 5 :=
by
  sorry

end probability_negative_product_l125_125776


namespace complex_division_l125_125292

-- Conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Question: Prove the complex division
theorem complex_division (h : i = Complex.I) : (8 - i) / (2 + i) = 3 - 2 * i :=
by sorry

end complex_division_l125_125292


namespace English_family_information_l125_125591

-- Define the statements given by the family members.
variables (father_statement : Prop)
          (mother_statement : Prop)
          (daughter_statement : Prop)

-- Conditions provided in the problem
variables (going_to_Spain : Prop)
          (coming_from_Newcastle : Prop)
          (stopped_in_Paris : Prop)

-- Define what each family member said
axiom Father : father_statement ↔ (going_to_Spain ∨ coming_from_Newcastle)
axiom Mother : mother_statement ↔ ((¬going_to_Spain ∧ coming_from_Newcastle) ∨ (stopped_in_Paris ∧ ¬going_to_Spain))
axiom Daughter : daughter_statement ↔ (¬coming_from_Newcastle ∨ stopped_in_Paris)

-- The final theorem to be proved:
theorem English_family_information : (¬going_to_Spain ∧ coming_from_Newcastle ∧ stopped_in_Paris) :=
by
  -- steps to prove the theorem should go here, but they are skipped with sorry
  sorry

end English_family_information_l125_125591


namespace find_x_l125_125053

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) : hash x 7 = 63 → x = 3 :=
by
  sorry

end find_x_l125_125053


namespace singer_arrangements_l125_125283

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end singer_arrangements_l125_125283


namespace Borgnine_total_legs_l125_125997

def numChimps := 12
def numLions := 8
def numLizards := 5
def numTarantulas := 125

def chimpLegsEach := 2
def lionLegsEach := 4
def lizardLegsEach := 4
def tarantulaLegsEach := 8

def legsSeen := numChimps * chimpLegsEach +
                numLions * lionLegsEach +
                numLizards * lizardLegsEach

def legsToSee := numTarantulas * tarantulaLegsEach

def totalLegs := legsSeen + legsToSee

theorem Borgnine_total_legs : totalLegs = 1076 := by
  sorry

end Borgnine_total_legs_l125_125997


namespace rate_of_interest_increase_l125_125934

noncomputable def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

noncomputable def percentage_increase_in_rate (P A1 A2 T : ℝ) : ℝ :=
  let SI1 := A1 - P in
  let R1 := (SI1 * 100) / (P * T) in
  let SI2 := A2 - P in
  let R2 := (SI2 * 100) / (P * T) in
  ((R2 - R1) / R1) * 100

theorem rate_of_interest_increase :
  percentage_increase_in_rate 800 956 1052 3 ≈ 61.54 := by
    sorry

end rate_of_interest_increase_l125_125934


namespace johns_age_l125_125892

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l125_125892


namespace min_value_a4b3c2_l125_125219

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l125_125219


namespace toilet_paper_squares_per_roll_l125_125152

theorem toilet_paper_squares_per_roll
  (trips_per_day : ℕ)
  (squares_per_trip : ℕ)
  (num_rolls : ℕ)
  (supply_days : ℕ)
  (total_squares : ℕ)
  (squares_per_roll : ℕ)
  (h1 : trips_per_day = 3)
  (h2 : squares_per_trip = 5)
  (h3 : num_rolls = 1000)
  (h4 : supply_days = 20000)
  (h5 : total_squares = trips_per_day * squares_per_trip * supply_days)
  (h6 : squares_per_roll = total_squares / num_rolls) :
  squares_per_roll = 300 :=
by sorry

end toilet_paper_squares_per_roll_l125_125152


namespace largest_angle_90_degrees_l125_125605

def triangle_altitudes (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (9 * a = 12 * b) ∧ (9 * a = 18 * c)

theorem largest_angle_90_degrees (a b c : ℝ) 
  (h : triangle_altitudes a b c) : 
  exists (A B C : ℝ) (hApos : A > 0) (hBpos : B > 0) (hCpos : C > 0),
    (A^2 = B^2 + C^2) ∧ (B * C / 2 = 9 * a / 2 ∨ 
                         B * A / 2 = 12 * b / 2 ∨ 
                         C * A / 2 = 18 * c / 2) :=
sorry

end largest_angle_90_degrees_l125_125605


namespace solve_for_x_l125_125555

theorem solve_for_x : 
  let x := (√(7^2 + 24^2)) / (√(49 + 16)) in 
  x = 25 * √65 / 65 := 
by
  -- Step 1: expand the terms inside the square roots
  let a := 7^2 + 24^2 
  let b := 49 + 16

  have a_eq : a = 625 := by
    calc
      a = 7^2 + 24^2 : rfl
      ... = 49 + 576 : rfl
      ... = 625 : rfl

  have b_eq : b = 65 := by
    calc
      b = 49 + 16 : rfl
      ... = 65 : rfl

  -- Step 2: Simplify the square roots
  let sqrt_a := √a
  have sqrt_a_eq : sqrt_a = 25 := by
    rw [a_eq]
    norm_num

  let sqrt_b := √b
  have sqrt_b_eq : sqrt_b = √65 := by
    rw [b_eq]

  -- Step 3: Simplify x
  let x := sqrt_a / sqrt_b

  show x = 25 * √65 / 65
  rw [sqrt_a_eq, sqrt_b_eq]
  field_simp
  norm_num
  rw [mul_div_cancel_left 25 (sqrt_ne_zero.2 (ne_of_gt (by norm_num : √65 ≠ 0))) ]
  sorry

end solve_for_x_l125_125555


namespace hexagon_height_correct_l125_125085

-- Define the dimensions of the original rectangle
def original_rectangle_width := 16
def original_rectangle_height := 9
def original_rectangle_area := original_rectangle_width * original_rectangle_height

-- Define the dimensions of the new rectangle formed by the hexagons
def new_rectangle_width := 12
def new_rectangle_height := 12
def new_rectangle_area := new_rectangle_width * new_rectangle_height

-- Define the parameter x, which is the height of the hexagons
def hexagon_height := 6

-- Theorem stating the equivalence of the areas and the specific height x
theorem hexagon_height_correct :
  original_rectangle_area = new_rectangle_area ∧
  hexagon_height * 2 = new_rectangle_height :=
by
  sorry

end hexagon_height_correct_l125_125085


namespace unique_zero_property_l125_125413

theorem unique_zero_property (x : ℝ) (h1 : ∀ a : ℝ, x * a = x) (h2 : ∀ (a : ℝ), a ≠ 0 → x / a = x) :
  x = 0 :=
sorry

end unique_zero_property_l125_125413


namespace zeros_not_adjacent_probability_l125_125355

-- Definitions based on the conditions
def total_arrangements : ℕ := Nat.choose 6 2
def non_adjacent_zero_arrangements : ℕ := Nat.choose 5 2

-- The probability that the 2 zeros are not adjacent
def probability_non_adjacent_zero : ℚ :=
  (non_adjacent_zero_arrangements : ℚ) / (total_arrangements : ℚ)

-- The theorem statement
theorem zeros_not_adjacent_probability :
  probability_non_adjacent_zero = 2 / 3 :=
by
  -- The proof would go here
  sorry

end zeros_not_adjacent_probability_l125_125355


namespace average_temperature_l125_125258

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l125_125258


namespace squirrel_acorns_l125_125374

theorem squirrel_acorns (S A : ℤ) 
  (h1 : A = 4 * S + 3) 
  (h2 : A = 5 * S - 6) : 
  A = 39 :=
by sorry

end squirrel_acorns_l125_125374


namespace volume_of_water_displaced_l125_125135

theorem volume_of_water_displaced (r h s : ℝ) (V : ℝ) 
  (r_eq : r = 5) (h_eq : h = 12) (s_eq : s = 6) :
  V = s^3 :=
by
  have cube_volume : V = s^3 := by sorry
  show V = s^3
  exact cube_volume

end volume_of_water_displaced_l125_125135


namespace faye_age_l125_125000

theorem faye_age (D E C F : ℤ)
  (h1 : D = E - 4)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (hD : D = 18) :
  F = 21 :=
by
  sorry

end faye_age_l125_125000


namespace grilled_cheese_sandwiches_l125_125882

theorem grilled_cheese_sandwiches (h_cheese : ℕ) (g_cheese : ℕ) (total_cheese : ℕ) (ham_sandwiches : ℕ) (grilled_cheese_sandwiches : ℕ) :
  h_cheese = 2 →
  g_cheese = 3 →
  total_cheese = 50 →
  ham_sandwiches = 10 →
  total_cheese - (ham_sandwiches * h_cheese) = grilled_cheese_sandwiches * g_cheese →
  grilled_cheese_sandwiches = 10 :=
by {
  intros,
  sorry
}

end grilled_cheese_sandwiches_l125_125882


namespace exists_positive_integer_m_such_that_sqrt_8m_is_integer_l125_125707

theorem exists_positive_integer_m_such_that_sqrt_8m_is_integer :
  ∃ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), 8 * m = k^2 :=
by
  sorry

end exists_positive_integer_m_such_that_sqrt_8m_is_integer_l125_125707


namespace green_paint_quarts_l125_125012

theorem green_paint_quarts (blue green white : ℕ) (h_ratio : 3 = blue ∧ 2 = green ∧ 4 = white) 
  (h_white_paint : white = 12) : green = 6 := 
by
  sorry

end green_paint_quarts_l125_125012


namespace sum_of_two_digit_factors_is_162_l125_125246

-- Define the number
def num := 6545

-- Define the condition: num can be written as a product of two two-digit numbers
def are_two_digit_numbers (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = num

-- The theorem to prove
theorem sum_of_two_digit_factors_is_162 : ∃ a b : ℕ, are_two_digit_numbers a b ∧ a + b = 162 :=
sorry

end sum_of_two_digit_factors_is_162_l125_125246


namespace product_of_two_numbers_l125_125112

theorem product_of_two_numbers :
  ∀ x y: ℝ, 
  ((x - y)^2) / ((x + y)^3) = 4 / 27 → 
  x + y = 5 * (x - y) + 3 → 
  x * y = 15.75 :=
by 
  intro x y
  sorry

end product_of_two_numbers_l125_125112


namespace smallest_a_for_polynomial_l125_125952

theorem smallest_a_for_polynomial (a b x₁ x₂ x₃ : ℕ) 
    (h1 : x₁ * x₂ * x₃ = 2730)
    (h2 : x₁ + x₂ + x₃ = a)
    (h3 : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
    (h4 : ∀ y₁ y₂ y₃ : ℕ, y₁ * y₂ * y₃ = 2730 ∧ y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + y₂ + y₃ ≥ a) :
  a = 54 :=
  sorry

end smallest_a_for_polynomial_l125_125952


namespace percentage_increase_correct_l125_125862

variable {R1 E1 P1 R2 E2 P2 R3 E3 P3 : ℝ}

-- Conditions
axiom H1 : P1 = R1 - E1
axiom H2 : R2 = 1.20 * R1
axiom H3 : E2 = 1.10 * E1
axiom H4 : P2 = R2 - E2
axiom H5 : P2 = 1.15 * P1
axiom H6 : R3 = 1.25 * R2
axiom H7 : E3 = 1.20 * E2
axiom H8 : P3 = R3 - E3
axiom H9 : P3 = 1.35 * P2

theorem percentage_increase_correct :
  ((P3 - P1) / P1) * 100 = 55.25 :=
by sorry

end percentage_increase_correct_l125_125862


namespace algebra_expression_value_l125_125026

theorem algebra_expression_value (m : ℝ) (hm : m^2 - m - 1 = 0) : m^2 - m + 2008 = 2009 :=
by
  sorry

end algebra_expression_value_l125_125026


namespace distinct_arrangements_balloon_l125_125808

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l125_125808


namespace john_age_proof_l125_125902

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l125_125902


namespace min_sum_squares_l125_125066

variable {a b c t : ℝ}

def min_value_of_sum_squares (a b c : ℝ) (t : ℝ) : ℝ :=
  a^2 + b^2 + c^2

theorem min_sum_squares (h : a + b + c = t) : min_value_of_sum_squares a b c t ≥ t^2 / 3 :=
by
  sorry

end min_sum_squares_l125_125066


namespace intersectionDistance_l125_125678

noncomputable def polarCurve (ρ θ : ℝ) : Prop :=
ρ^2 = 12 / (2 + cos θ ^ 2)

noncomputable def polarLine (ρ θ : ℝ) : Prop :=
2 * ρ * cos (θ - π / 6) = sqrt 3

noncomputable def parametricLine (t : ℝ) : ℝ × ℝ :=
(-0.5 * t, sqrt 3 + (sqrt 3 / 2) * t)

noncomputable def cartesianCurve (x y : ℝ) : Prop :=
(x ^ 2) / 4 + (y ^ 2) / 6 = 1

theorem intersectionDistance :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, parametricLine t₁ = A ∧ parametricLine t₂ = B ∧
                  cartesianCurve A.fst A.snd ∧ cartesianCurve B.fst B.snd) ∧
    dist A B = 4 * sqrt 10 / 3 := by
{ sorry }

end intersectionDistance_l125_125678


namespace eight_term_sum_l125_125520

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l125_125520


namespace value_of_sum_l125_125199

theorem value_of_sum (x y z : ℝ) 
    (h1 : x + 2*y + 3*z = 10) 
    (h2 : 4*x + 3*y + 2*z = 15) : 
    x + y + z = 5 :=
by
    sorry

end value_of_sum_l125_125199


namespace squared_difference_l125_125049

theorem squared_difference (x y : ℝ) (h₁ : (x + y)^2 = 49) (h₂ : x * y = 8) : (x - y)^2 = 17 := 
by
  -- Proof omitted
  sorry

end squared_difference_l125_125049


namespace distinct_arrangements_balloon_l125_125806

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l125_125806


namespace probability_complement_given_A_l125_125017

theorem probability_complement_given_A :
  (∀ (A B : Type) [MeasureTheory.ProbabilityMeasure A] [MeasureTheory.ProbabilityMeasure B],
  let PA := MeasureTheory.Measure.probability
  let PB := MeasureTheory.Measure.probability
  let PAB := PA * PB in
  PA = 1/3 → PB = 1/4 → PA (B | A) = 3/4 →
  PA (¬ B | A) = 7/16) :=
by
  intros A B _ _ PA PB PAB hPA hPB hPA_B
  sorry

end probability_complement_given_A_l125_125017


namespace heather_total_oranges_l125_125342

--Definition of the problem conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

--Statement of the theorem
theorem heather_total_oranges : initial_oranges + additional_oranges = 95.0 := by
  sorry

end heather_total_oranges_l125_125342


namespace emily_weight_l125_125039

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end emily_weight_l125_125039


namespace poly_division_l125_125164

noncomputable def A := 1
noncomputable def B := 3
noncomputable def C := 2
noncomputable def D := -1

theorem poly_division :
  (∀ x : ℝ, x ≠ -1 → (x^3 + 4*x^2 + 5*x + 2) / (x+1) = x^2 + 3*x + 2) ∧
  (A + B + C + D = 5) :=
by
  sorry

end poly_division_l125_125164


namespace abs_lt_inequality_solution_l125_125406

theorem abs_lt_inequality_solution (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 :=
by sorry

end abs_lt_inequality_solution_l125_125406


namespace domain_of_h_l125_125468

noncomputable def h (x : ℝ) : ℝ :=
  (x^2 - 9) / (abs (x - 4) + x^2 - 1)

theorem domain_of_h :
  ∀ (x : ℝ), x ≠ (1 + Real.sqrt 13) / 2 → (abs (x - 4) + x^2 - 1) ≠ 0 :=
sorry

end domain_of_h_l125_125468


namespace length_of_P1P2_segment_l125_125427

theorem length_of_P1P2_segment (x : ℝ) (h₀ : 0 < x ∧ x < π / 2) (h₁ : 6 * Real.cos x = 9 * Real.tan x) :
  Real.sin x = 1 / 2 :=
by
  sorry

end length_of_P1P2_segment_l125_125427


namespace find_f_2_find_f_neg2_l125_125465

noncomputable def f : ℝ → ℝ := sorry -- This is left to be defined as a function on ℝ

axiom f_property : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_at_1 : f 1 = 2

theorem find_f_2 : f 2 = 6 := by
  sorry

theorem find_f_neg2 : f (-2) = 2 := by
  sorry

end find_f_2_find_f_neg2_l125_125465


namespace total_cost_of_bicycles_is_2000_l125_125711

noncomputable def calculate_total_cost_of_bicycles (SP1 SP2 : ℝ) (profit1 profit2 : ℝ) : ℝ :=
  let C1 := SP1 / (1 + profit1)
  let C2 := SP2 / (1 - profit2)
  C1 + C2

theorem total_cost_of_bicycles_is_2000 :
  calculate_total_cost_of_bicycles 990 990 0.10 0.10 = 2000 :=
by
  -- Proof will be provided here
  sorry

end total_cost_of_bicycles_is_2000_l125_125711


namespace sum_of_products_of_roots_l125_125540

theorem sum_of_products_of_roots (p q r : ℂ) (h : 4 * (p^3) - 2 * (p^2) + 13 * p - 9 = 0 ∧ 4 * (q^3) - 2 * (q^2) + 13 * q - 9 = 0 ∧ 4 * (r^3) - 2 * (r^2) + 13 * r - 9 = 0) :
  p*q + p*r + q*r = 13 / 4 :=
  sorry

end sum_of_products_of_roots_l125_125540


namespace prob_four_questions_to_advance_l125_125204

open ProbabilityTheory

/-- Definition of probability of answering a question correctly -/
def prob_correct : ℝ := 0.8

/-- Definition of probability of answering a question incorrectly -/
def prob_incorrect : ℝ := 1 - prob_correct

/-- Probability that the contestant correctly answers the first question -/
def p1 : ℝ := prob_correct

/-- Probability that the contestant correctly answers the second question -/
def p2 : ℝ := prob_correct

/-- Probability that the contestant incorrectly answers the third question -/
def p3 : ℝ := prob_incorrect

/-- Probability that the contestant correctly answers the fourth question -/
def p4 : ℝ := prob_correct

/-- Theorem stating the probability that the contestant answers exactly four questions before advancing -/
theorem prob_four_questions_to_advance :
  p1 * p2 * p3 * p4 = 0.128 :=
by sorry

end prob_four_questions_to_advance_l125_125204


namespace Crimson_Valley_skirts_l125_125159

theorem Crimson_Valley_skirts
  (Azure_Valley_skirts : ℕ)
  (Seafoam_Valley_skirts : ℕ)
  (Purple_Valley_skirts : ℕ)
  (Crimson_Valley_skirts : ℕ)
  (h1 : Azure_Valley_skirts = 90)
  (h2 : Seafoam_Valley_skirts = (2/3 : ℚ) * Azure_Valley_skirts)
  (h3 : Purple_Valley_skirts = (1/4 : ℚ) * Seafoam_Valley_skirts)
  (h4 : Crimson_Valley_skirts = (1/3 : ℚ) * Purple_Valley_skirts)
  : Crimson_Valley_skirts = 5 := 
sorry

end Crimson_Valley_skirts_l125_125159


namespace max_quotient_l125_125857

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  ∃ max_val, max_val = 225 ∧ ∀ (x y : ℝ), (100 ≤ x ∧ x ≤ 300) ∧ (500 ≤ y ∧ y ≤ 1500) → (y^2 / x^2) ≤ max_val := 
by
  use 225
  sorry

end max_quotient_l125_125857


namespace amina_wins_is_21_over_32_l125_125991

/--
Amina and Bert alternate turns tossing a fair coin. Amina goes first and each player takes three turns.
The first player to toss a tail wins. If neither Amina nor Bert tosses a tail, then neither wins.
Prove that the probability that Amina wins is \( \frac{21}{32} \).
-/
def amina_wins_probability : ℚ :=
  let p_first_turn := 1 / 2
  let p_second_turn := (1 / 2) ^ 3
  let p_third_turn := (1 / 2) ^ 5
  p_first_turn + p_second_turn + p_third_turn

theorem amina_wins_is_21_over_32 :
  amina_wins_probability = 21 / 32 :=
sorry

end amina_wins_is_21_over_32_l125_125991


namespace closest_to_10_l125_125307

theorem closest_to_10
  (A B C D : ℝ)
  (hA : A = 9.998)
  (hB : B = 10.1)
  (hC : C = 10.09)
  (hD : D = 10.001) :
  abs (10 - D) < abs (10 - A) ∧ abs (10 - D) < abs (10 - B) ∧ abs (10 - D) < abs (10 - C) :=
by
  sorry

end closest_to_10_l125_125307


namespace rope_length_total_l125_125733

theorem rope_length_total :
  let length1 := 24
  let length2 := 20
  let length3 := 14
  let length4 := 12
  length1 + length2 + length3 + length4 = 70 :=
by
  sorry

end rope_length_total_l125_125733


namespace probability_non_adjacent_zeros_l125_125357

-- Define the conditions
def num_ones : ℕ := 4
def num_zeros : ℕ := 2

-- Calculate combinations
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Calculate total arrangements
def total_arrangements : ℕ := combination 6 2

-- Calculate non-adjacent arrangements
def non_adjacent_arrangements : ℕ := combination 5 2

-- Define the probability and prove the equality
theorem probability_non_adjacent_zeros :
  (non_adjacent_arrangements.toRat / total_arrangements.toRat) = (2 / 3) := by
  sorry

end probability_non_adjacent_zeros_l125_125357


namespace claire_earnings_l125_125611

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end claire_earnings_l125_125611


namespace function_value_sum_l125_125690

namespace MathProof

variable {f : ℝ → ℝ}

theorem function_value_sum :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 5) = f x) →
  f (1 / 3) = 2022 →
  f (1 / 2) = 17 →
  f (-7) + f 12 + f (16 / 3) + f (9 / 2) = 2005 :=
by
  intros h_odd h_periodic h_f13 h_f12
  sorry

end MathProof

end function_value_sum_l125_125690


namespace toms_initial_investment_l125_125968

theorem toms_initial_investment (t j k : ℕ) (hj_neq_ht : t ≠ j) (hk_neq_ht : t ≠ k) (hj_neq_hk : j ≠ k) 
  (h1 : t + j + k = 1200) 
  (h2 : t - 150 + 3 * j + 3 * k = 1800) : 
  t = 825 := 
sorry

end toms_initial_investment_l125_125968


namespace ratio_of_width_perimeter_is_3_16_l125_125602

-- We define the conditions
def length_of_room : ℕ := 25
def width_of_room : ℕ := 15

-- We define the calculation and verification of the ratio
theorem ratio_of_width_perimeter_is_3_16 :
  let P := 2 * (length_of_room + width_of_room)
  let ratio := width_of_room / P
  let a := 15 / Nat.gcd 15 80
  let b := 80 / Nat.gcd 15 80
  (a, b) = (3, 16) :=
by 
  -- The proof is skipped with sorry
  sorry

end ratio_of_width_perimeter_is_3_16_l125_125602


namespace find_angle_C_find_side_c_l125_125875

noncomputable def triangle := Type

structure Triangle (A B C : ℝ) :=
  (side_a : ℝ)
  (side_b : ℝ)
  (side_c : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom law_of_cosines (T : Triangle A B C) : 
  2 * T.side_c * Math.cos T.angle_C = T.side_b * Math.cos T.angle_A + T.side_a * Math.cos T.angle_B

theorem find_angle_C (A B C a b c : ℝ) (h : Triangle A B C) (h1 : law_of_cosines h) :
  h.angle_C = Real.pi / 3 := 
sorry

theorem find_side_c (A B C : ℝ) (a : ℝ := 6) (cos_A : ℝ := -4 / 5) : 
  ∃ (h : Triangle A B C), law_of_cosines h → h.side_c = 5 * Real.sqrt(3) :=
sorry

end find_angle_C_find_side_c_l125_125875


namespace jaden_toy_cars_l125_125208

variable (initial_cars bought_cars birthday_cars gave_sister gave_vinnie : ℕ)

theorem jaden_toy_cars :
  let final_cars := initial_cars + bought_cars + birthday_cars - gave_sister - gave_vinnie in
  initial_cars = 14 → bought_cars = 28 → birthday_cars = 12 → gave_sister = 8 → gave_vinnie = 3 →
  final_cars = 43 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jaden_toy_cars_l125_125208


namespace average_temperature_l125_125254

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l125_125254


namespace problem1_no_solution_problem2_solution_l125_125234

theorem problem1_no_solution (x : ℝ) 
  (h : (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1) : false :=
by
  -- The original equation turns out to have no solution
  sorry

theorem problem2_solution (x : ℝ) 
  (h : 1 - (x - 2)/(2 + x) = 16/(x^2 - 4)) : x = 6 :=
by
  -- The equation has a solution x = 6
  sorry

end problem1_no_solution_problem2_solution_l125_125234


namespace rectangle_measurement_error_l125_125500

theorem rectangle_measurement_error 
  (L W : ℝ)
  (measured_length : ℝ := 1.05 * L)
  (measured_width : ℝ := 0.96 * W)
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  (error : ℝ := calculated_area - actual_area) :
  ((error / actual_area) * 100) = 0.8 :=
sorry

end rectangle_measurement_error_l125_125500


namespace jaden_toy_cars_l125_125207

theorem jaden_toy_cars :
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  initial + bought + birthday - to_sister - to_friend = 43 :=
by
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  show initial + bought + birthday - to_sister - to_friend = 43
  sorry

end jaden_toy_cars_l125_125207


namespace minimum_knights_l125_125264

-- Definitions based on the conditions
def total_people := 1001
def is_knight (person : ℕ) : Prop := sorry -- Assume definition of knight
def is_liar (person : ℕ) : Prop := sorry    -- Assume definition of liar

-- Conditions
axiom next_to_each_knight_is_liar : ∀ (p : ℕ), is_knight p → is_liar (p + 1) ∨ is_liar (p - 1)
axiom next_to_each_liar_is_knight : ∀ (p : ℕ), is_liar p → is_knight (p + 1) ∨ is_knight (p - 1)

-- Proving the minimum number of knights
theorem minimum_knights : ∃ (k : ℕ), k ≤ total_people ∧ k ≥ 502 ∧ (∀ (n : ℕ), n ≥ k → is_knight n) :=
  sorry

end minimum_knights_l125_125264


namespace johns_age_l125_125904

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125904


namespace number_of_ways_to_form_divisible_number_l125_125685

def valid_digits : List ℕ := [0, 2, 4, 7, 8, 9]

def is_divisible_by_4 (d1 d2 : ℕ) : Prop :=
  (d1 * 10 + d2) % 4 = 0

def is_divisible_by_3 (sum_of_digits : ℕ) : Prop :=
  sum_of_digits % 3 = 0

def replace_asterisks_to_form_divisible_number : Prop :=
  ∃ (a1 a2 a3 a4 a5 l : ℕ), a1 ∈ valid_digits ∧ a2 ∈ valid_digits ∧ a3 ∈ valid_digits ∧ a4 ∈ valid_digits ∧ a5 ∈ valid_digits ∧
  l ∈ [0, 2, 4, 8] ∧
  is_divisible_by_4 0 l ∧
  is_divisible_by_3 (11 + a1 + a2 + a3 + a4 + a5) ∧
  (4 * 324 = 1296)

theorem number_of_ways_to_form_divisible_number :
  replace_asterisks_to_form_divisible_number :=
  sorry

end number_of_ways_to_form_divisible_number_l125_125685


namespace johnny_marble_combinations_l125_125378

/-- 
Johnny has 10 different colored marbles. 
The number of ways he can choose four different marbles from his bag is 210.
-/
theorem johnny_marble_combinations : (Nat.choose 10 4) = 210 := by
  sorry

end johnny_marble_combinations_l125_125378


namespace angle_CAB_in_regular_hexagon_l125_125372

-- Define a regular hexagon
structure regular_hexagon (A B C D E F : Type) :=
  (interior_angle : ℝ)
  (all_sides_equal : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = F)
  (all_angles_equal : interior_angle = 120)

-- Define the problem of finding the angle CAB
theorem angle_CAB_in_regular_hexagon 
  (A B C D E F : Type)
  (hex : regular_hexagon A B C D E F)
  (diagonal_AC : A = C)
  : ∃ (CAB : ℝ), CAB = 30 :=
sorry

end angle_CAB_in_regular_hexagon_l125_125372


namespace solve_m_n_l125_125712

theorem solve_m_n (m n : ℝ) (h : m^2 + 2 * m + n^2 - 6 * n + 10 = 0) :
  m = -1 ∧ n = 3 :=
sorry

end solve_m_n_l125_125712


namespace sqrt_seven_irrational_l125_125739

theorem sqrt_seven_irrational : irrational (Real.sqrt 7) :=
sorry

end sqrt_seven_irrational_l125_125739


namespace probability_non_adjacent_zeros_l125_125356

-- Define the conditions
def num_ones : ℕ := 4
def num_zeros : ℕ := 2

-- Calculate combinations
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Calculate total arrangements
def total_arrangements : ℕ := combination 6 2

-- Calculate non-adjacent arrangements
def non_adjacent_arrangements : ℕ := combination 5 2

-- Define the probability and prove the equality
theorem probability_non_adjacent_zeros :
  (non_adjacent_arrangements.toRat / total_arrangements.toRat) = (2 / 3) := by
  sorry

end probability_non_adjacent_zeros_l125_125356


namespace temperature_on_friday_is_72_l125_125724

-- Define the temperatures on specific days
def temp_sunday := 40
def temp_monday := 50
def temp_tuesday := 65
def temp_wednesday := 36
def temp_thursday := 82
def temp_saturday := 26

-- Average temperature over the week
def average_temp := 53

-- Total number of days in a week
def days_in_week := 7

-- Calculate the total sum of temperatures given the average temperature
def total_sum_temp : ℤ := average_temp * days_in_week

-- Sum of known temperatures from specific days
def known_sum_temp : ℤ := temp_sunday + temp_monday + temp_tuesday + temp_wednesday + temp_thursday + temp_saturday

-- Define the temperature on Friday
def temp_friday : ℤ := total_sum_temp - known_sum_temp

theorem temperature_on_friday_is_72 : temp_friday = 72 :=
by
  -- Placeholder for the proof
  sorry

end temperature_on_friday_is_72_l125_125724


namespace probability_sum_is_3_l125_125961

theorem probability_sum_is_3 (die : Type) [Fintype die] [DecidableEq die] 
  (dice_faces : die → ℕ) (h : ∀ d, dice_faces d ∈ {1, 2, 3, 4, 5, 6}) :
  (∑ i in finset.range 3, (die →₀ ℕ).single 1) = 3 → 
  (1 / (finset.card univ) ^ 3) = 1 / 216 :=
by
  sorry

end probability_sum_is_3_l125_125961


namespace pencils_left_l125_125100

theorem pencils_left (initial_pencils : ℕ := 79) (pencils_taken : ℕ := 4) : initial_pencils - pencils_taken = 75 :=
by
  sorry

end pencils_left_l125_125100


namespace sum_of_ages_l125_125718

theorem sum_of_ages (a b c d : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b = 24 ∨ a * c = 24 ∨ a * d = 24 ∨ b * c = 24 ∨ b * d = 24 ∨ c * d = 24)
  (h8 : a * b = 35 ∨ a * c = 35 ∨ a * d = 35 ∨ b * c = 35 ∨ b * d = 35 ∨ c * d = 35)
  (h9 : a < 10) (h10 : b < 10) (h11 : c < 10) (h12 : d < 10)
  (h13 : 0 < a) (h14 : 0 < b) (h15 : 0 < c) (h16 : 0 < d) :
  a + b + c + d = 23 := sorry

end sum_of_ages_l125_125718


namespace conic_section_hyperbola_l125_125469

theorem conic_section_hyperbola (x y : ℝ) : 
  (2 * x - 7)^2 - 4 * (y + 3)^2 = 169 → 
  -- Explain that this equation is of a hyperbola
  true := 
sorry

end conic_section_hyperbola_l125_125469


namespace train_speed_proof_l125_125755

noncomputable def train_speed (L : ℕ) (t : ℝ) (v_m : ℝ) : ℝ :=
  let v_m_m_s := v_m * (1000 / 3600)
  let v_rel := L / t
  v_rel + v_m_m_s

theorem train_speed_proof
  (L : ℕ)
  (t : ℝ)
  (v_m : ℝ)
  (hL : L = 900)
  (ht : t = 53.99568034557235)
  (hv_m : v_m = 3)
  : train_speed L t v_m = 63.0036 :=
  by sorry

end train_speed_proof_l125_125755


namespace cylinder_volume_l125_125364

theorem cylinder_volume (r h : ℝ) (hr : r = 1) (hh : h = 1) : (π * r^2 * h) = π :=
by
  sorry

end cylinder_volume_l125_125364


namespace deans_height_l125_125306

theorem deans_height
  (D : ℕ) 
  (h1 : 10 * D = D + 81) : 
  D = 9 := sorry

end deans_height_l125_125306


namespace sqrt_fraction_expression_l125_125002

theorem sqrt_fraction_expression : 
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + (Real.sqrt (9 / 4) + Real.sqrt (4 / 9))^2) = (199 / 36) := 
by
  sorry

end sqrt_fraction_expression_l125_125002


namespace calculate_visits_to_water_fountain_l125_125927

-- Define the distance from the desk to the fountain
def distance_desk_to_fountain : ℕ := 30

-- Define the total distance Mrs. Hilt walked
def total_distance_walked : ℕ := 120

-- Define the distance of a round trip (desk to fountain and back)
def round_trip_distance : ℕ := 2 * distance_desk_to_fountain

-- Define the number of round trips and hence the number of times to water fountain
def number_of_visits : ℕ := total_distance_walked / round_trip_distance

theorem calculate_visits_to_water_fountain:
    number_of_visits = 2 := 
by
    sorry

end calculate_visits_to_water_fountain_l125_125927


namespace area_of_large_rectangle_l125_125790

noncomputable def areaEFGH : ℕ :=
  let shorter_side := 3
  let longer_side := 2 * shorter_side
  let width_EFGH := shorter_side + shorter_side
  let length_EFGH := longer_side + longer_side
  width_EFGH * length_EFGH

theorem area_of_large_rectangle :
  areaEFGH = 72 := by
  sorry

end area_of_large_rectangle_l125_125790


namespace ella_spent_on_video_games_last_year_l125_125001

theorem ella_spent_on_video_games_last_year 
  (new_salary : ℝ) 
  (raise : ℝ) 
  (percentage_spent_on_video_games : ℝ) 
  (h_new_salary : new_salary = 275) 
  (h_raise : raise = 0.10) 
  (h_percentage_spent : percentage_spent_on_video_games = 0.40) :
  (new_salary / (1 + raise) * percentage_spent_on_video_games = 100) :=
by
  sorry

end ella_spent_on_video_games_last_year_l125_125001


namespace geometric_sequence_sum_l125_125530

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l125_125530


namespace intersection_A_B_l125_125340

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  sorry

end intersection_A_B_l125_125340


namespace evaluate_expression_l125_125629

theorem evaluate_expression : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 :=
by
  sorry

end evaluate_expression_l125_125629


namespace rectangle_perimeter_from_square_l125_125547

theorem rectangle_perimeter_from_square (d : ℝ)
  (h : d = 6) :
  ∃ (p : ℝ), p = 12 :=
by
  sorry

end rectangle_perimeter_from_square_l125_125547


namespace loan_period_l125_125138

theorem loan_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) (years : ℝ) :
  principal = 3500 ∧ rate_A = 0.1 ∧ rate_C = 0.12 ∧ gain = 210 →
  (rate_C * principal * years - rate_A * principal * years) = gain →
  years = 3 :=
by
  sorry

end loan_period_l125_125138


namespace exp_fixed_point_l125_125657

theorem exp_fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : a^0 = 1 :=
by
  exact one_pow 0

end exp_fixed_point_l125_125657


namespace cannot_bisect_abs_function_l125_125584

theorem cannot_bisect_abs_function 
  (f : ℝ → ℝ)
  (hf1 : ∀ x, f x = |x|) :
  ¬ (∃ a b, a < b ∧ f a * f b < 0) :=
by
  sorry

end cannot_bisect_abs_function_l125_125584


namespace solve_for_y_l125_125667

theorem solve_for_y (y : ℤ) : (2 / 3 - 3 / 5 : ℚ) = 5 / y → y = 75 :=
by
  sorry

end solve_for_y_l125_125667


namespace trigonometric_identity_l125_125022

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 8 :=
by 
  sorry

end trigonometric_identity_l125_125022


namespace citric_acid_molecular_weight_l125_125452

noncomputable def molecularWeightOfCitricAcid : ℝ :=
  let weight_C := 12.01
  let weight_H := 1.008
  let weight_O := 16.00
  let num_C := 6
  let num_H := 8
  let num_O := 7
  (num_C * weight_C) + (num_H * weight_H) + (num_O * weight_O)

theorem citric_acid_molecular_weight :
  molecularWeightOfCitricAcid = 192.124 :=
by
  -- the step-by-step proof will go here
  sorry

end citric_acid_molecular_weight_l125_125452


namespace claire_earning_l125_125614

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end claire_earning_l125_125614


namespace find_p_and_q_l125_125341

theorem find_p_and_q :
  (∀ p q: ℝ, (∃ x : ℝ, x^2 + p * x + q = 0 ∧ q * x^2 + p * x + 1 = 0) ∧ (-2) ^ 2 + p * (-2) + q = 0 ∧ p ≠ 0 ∧ q ≠ 0 → 
    (p, q) = (1, -2) ∨ (p, q) = (3, 2) ∨ (p, q) = (5/2, 1)) :=
sorry

end find_p_and_q_l125_125341


namespace number_of_people_on_boats_l125_125419

def boats := 5
def people_per_boat := 3

theorem number_of_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end number_of_people_on_boats_l125_125419


namespace work_completion_time_l125_125585

-- Let's define the initial conditions
def total_days := 100
def initial_people := 10
def days1 := 20
def work_done1 := 1 / 4
def days2 (remaining_work_per_person: ℚ) := (3/4) / remaining_work_per_person
def remaining_people := initial_people - 2
def remaining_work_per_person_per_day := remaining_people * (work_done1 / (initial_people * days1))

-- Theorem stating that the total number of days to complete the work is 95
theorem work_completion_time : 
  days1 + days2 remaining_work_per_person_per_day = 95 := 
  by
    sorry -- Proof to be filled in

end work_completion_time_l125_125585


namespace sales_on_second_street_l125_125229

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end sales_on_second_street_l125_125229


namespace geometric_sequence_sum_S8_l125_125519

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l125_125519


namespace rotation_matrix_inverse_and_area_l125_125795

theorem rotation_matrix_inverse_and_area :
  let M := ![
    [real.cos (-real.pi / 4), -real.sin (-real.pi / 4)], 
    [real.sin (-real.pi / 4), real.cos (-real.pi / 4)]
  ] in
  let M_inv := ![
    [real.cos (real.pi / 4), -real.sin (real.pi / 4)], 
    [real.sin (real.pi / 4), real.cos (real.pi / 4)]
  ] in
  let A := (1, 0) in
  let B := (2, 2) in
  let C := (3, 0) in
  let area_ABC := (1 / 2) * (3 - 1) * 2 in
  let area_A1B1C1 := area_ABC in
  M = ![
    [real.sqrt 2 / 2, real.sqrt 2 / 2], 
    [-real.sqrt 2 / 2, real.sqrt 2 / 2]
  ] ∧
  M_inv = ![
    [real.sqrt 2 / 2, -real.sqrt 2 / 2], 
    [real.sqrt 2 / 2, real.sqrt 2 / 2]
  ] ∧
  area_A1B1C1 = 2 :=
by {
  sorry
}

end rotation_matrix_inverse_and_area_l125_125795


namespace no_solution_inequality_l125_125293

theorem no_solution_inequality (m : ℝ) : (¬ ∃ x : ℝ, |x + 1| + |x - 5| ≤ m) ↔ m < 6 :=
sorry

end no_solution_inequality_l125_125293


namespace min_x_minus_y_l125_125638

theorem min_x_minus_y {x y : ℝ} (hx : 0 ≤ x) (hx2 : x ≤ 2 * Real.pi) (hy : 0 ≤ y) (hy2 : y ≤ 2 * Real.pi)
    (h : 2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1 / 2) : 
    x - y = -Real.pi / 2 := 
sorry

end min_x_minus_y_l125_125638


namespace custom_dollar_five_neg3_l125_125462

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end custom_dollar_five_neg3_l125_125462


namespace total_square_footage_after_expansion_l125_125101

-- Definitions from the conditions
def size_smaller_house_initial : ℕ := 5200
def size_larger_house : ℕ := 7300
def expansion_smaller_house : ℕ := 3500

-- The new size of the smaller house after expansion
def size_smaller_house_after_expansion : ℕ :=
  size_smaller_house_initial + expansion_smaller_house

-- The new total square footage
def new_total_square_footage : ℕ :=
  size_smaller_house_after_expansion + size_larger_house

-- Goal statement: Prove the total new square footage is 16000 sq. ft.
theorem total_square_footage_after_expansion : new_total_square_footage = 16000 := by
  sorry

end total_square_footage_after_expansion_l125_125101


namespace divisibility_of_f_by_cubic_factor_l125_125077

noncomputable def f (x : ℂ) (m n : ℕ) : ℂ := x^(3 * m + 2) + (-x^2 - 1)^(3 * n + 1) + 1

theorem divisibility_of_f_by_cubic_factor (m n : ℕ) : ∀ x : ℂ, x^2 + x + 1 = 0 → f x m n = 0 :=
by
  sorry

end divisibility_of_f_by_cubic_factor_l125_125077


namespace club_committee_selections_l125_125756

theorem club_committee_selections : (Nat.choose 18 3) = 816 := by
  sorry

end club_committee_selections_l125_125756


namespace students_without_A_l125_125675

theorem students_without_A (total_students : ℕ) (students_english : ℕ) 
  (students_math : ℕ) (students_both : ℕ) (students_only_math : ℕ) :
  total_students = 30 → students_english = 6 → students_math = 15 → 
  students_both = 3 → students_only_math = 1 →
  (total_students - (students_math - students_only_math + 
                     students_english - students_both + 
                     students_both) = 12) :=
by sorry

end students_without_A_l125_125675


namespace sarahs_total_problems_l125_125714

def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def science_pages : ℕ := 5
def math_problems_per_page : ℕ := 4
def reading_problems_per_page : ℕ := 4
def science_problems_per_page : ℕ := 6

def total_math_problems : ℕ := math_pages * math_problems_per_page
def total_reading_problems : ℕ := reading_pages * reading_problems_per_page
def total_science_problems : ℕ := science_pages * science_problems_per_page

def total_problems : ℕ := total_math_problems + total_reading_problems + total_science_problems

theorem sarahs_total_problems :
  total_problems = 70 :=
by
  -- proof will be inserted here
  sorry

end sarahs_total_problems_l125_125714


namespace find_k_l125_125021

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the condition for vectors to be parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Translate the problem condition
def problem_condition (k : ℝ) : Prop :=
  let lhs := (k * a.1 + b.1, k * a.2 + b.2)
  let rhs := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  is_parallel lhs rhs

-- The goal is to find k such that the condition holds
theorem find_k : problem_condition (-1/3) :=
by
  sorry

end find_k_l125_125021


namespace largest_k_inequality_l125_125018

theorem largest_k_inequality {a b c : ℝ} (h1 : a ≤ b) (h2 : b ≤ c) (h3 : ab + bc + ca = 0) (h4 : abc = 1) :
  |a + b| ≥ 4 * |c| :=
sorry

end largest_k_inequality_l125_125018


namespace chris_age_l125_125945

theorem chris_age (a b c : ℤ) (h1 : a + b + c = 45) (h2 : c - 5 = a)
  (h3 : c + 4 = 3 * (b + 4) / 4) : c = 15 :=
by
  sorry

end chris_age_l125_125945


namespace correct_operation_l125_125286

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l125_125286


namespace geometric_sequence_sum_l125_125529

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l125_125529


namespace Margo_total_distance_walked_l125_125925

theorem Margo_total_distance_walked :
  ∀ (d : ℝ),
  (5 * (d / 5) + 3 * (d / 3) = 1) →
  (2 * d = 3.75) :=
by
  sorry

end Margo_total_distance_walked_l125_125925


namespace polynomial_equality_l125_125610

def P (x : ℝ) : ℝ := x ^ 3 - 3 * x ^ 2 - 3 * x - 1

noncomputable def x1 : ℝ := 1 - Real.sqrt 2
noncomputable def x2 : ℝ := 1 + Real.sqrt 2
noncomputable def x3 : ℝ := 1 - 2 * Real.sqrt 2
noncomputable def x4 : ℝ := 1 + 2 * Real.sqrt 2

theorem polynomial_equality :
  P x1 + P x2 = P x3 + P x4 :=
sorry

end polynomial_equality_l125_125610


namespace zoo_guides_children_total_l125_125311

theorem zoo_guides_children_total :
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  total_children = 1674 :=
by
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  sorry

end zoo_guides_children_total_l125_125311


namespace total_students_is_45_l125_125058

-- Define the initial conditions with the definitions provided
def drunk_drivers : Nat := 6
def speeders : Nat := 7 * drunk_drivers - 3
def total_students : Nat := drunk_drivers + speeders

-- The theorem to prove that the total number of students is 45
theorem total_students_is_45 : total_students = 45 :=
by
  sorry

end total_students_is_45_l125_125058


namespace zeros_not_adjacent_probability_l125_125354

-- Definitions based on the conditions
def total_arrangements : ℕ := Nat.choose 6 2
def non_adjacent_zero_arrangements : ℕ := Nat.choose 5 2

-- The probability that the 2 zeros are not adjacent
def probability_non_adjacent_zero : ℚ :=
  (non_adjacent_zero_arrangements : ℚ) / (total_arrangements : ℚ)

-- The theorem statement
theorem zeros_not_adjacent_probability :
  probability_non_adjacent_zero = 2 / 3 :=
by
  -- The proof would go here
  sorry

end zeros_not_adjacent_probability_l125_125354


namespace average_temperature_l125_125259

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l125_125259


namespace gain_percent_is_sixty_l125_125599

-- Definitions based on the conditions
def costPrice : ℝ := 675
def sellingPrice : ℝ := 1080
def gain : ℝ := sellingPrice - costPrice
def gainPercent : ℝ := (gain / costPrice) * 100

-- Proof statement
theorem gain_percent_is_sixty (h1 : costPrice = 675) (h2 : sellingPrice = 1080) :
  gainPercent = 60 :=
by
  rw [h1, h2]
  -- Additional steps to prove the equality can be abstracted here
  sorry

end gain_percent_is_sixty_l125_125599


namespace number_of_possible_values_for_a_l125_125929

theorem number_of_possible_values_for_a 
  (a b c d : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a > b) (h6 : b > c) (h7 : c > d)
  (h8 : a + b + c + d = 2004)
  (h9 : a^2 - b^2 - c^2 + d^2 = 1004) : 
  ∃ n : ℕ, n = 500 :=
  sorry

end number_of_possible_values_for_a_l125_125929


namespace largest_unattainable_sum_l125_125684

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end largest_unattainable_sum_l125_125684


namespace largest_unattainable_sum_l125_125681

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end largest_unattainable_sum_l125_125681


namespace Jolene_raised_total_money_l125_125509

-- Definitions for the conditions
def babysits_earning_per_family : ℤ := 30
def number_of_families : ℤ := 4
def cars_earning_per_car : ℤ := 12
def number_of_cars : ℤ := 5

-- Calculation of total earnings
def babysitting_earnings : ℤ := babysits_earning_per_family * number_of_families
def car_washing_earnings : ℤ := cars_earning_per_car * number_of_cars
def total_earnings : ℤ := babysitting_earnings + car_washing_earnings

-- The proof statement
theorem Jolene_raised_total_money : total_earnings = 180 := by
  sorry

end Jolene_raised_total_money_l125_125509


namespace count_odd_numbers_distinct_digits_l125_125593

theorem count_odd_numbers_distinct_digits : 
  ∃ n : ℕ, (∀ x : ℕ, 200 ≤ x ∧ x ≤ 999 ∧ x % 2 = 1 ∧ (∀ d ∈ [digit1, digit2, digit3], d ≤ 7) ∧ (digit1 ≠ digit2 ∧ digit2 ≠ digit3 ∧ digit1 ≠ digit3) → True) ∧
  n = 120 :=
sorry

end count_odd_numbers_distinct_digits_l125_125593


namespace Randy_used_blocks_l125_125226

theorem Randy_used_blocks (initial_blocks blocks_left used_blocks : ℕ) 
  (h1 : initial_blocks = 97) 
  (h2 : blocks_left = 72) 
  (h3 : used_blocks = initial_blocks - blocks_left) : 
  used_blocks = 25 :=
by
  sorry

end Randy_used_blocks_l125_125226


namespace min_value_a4b3c2_l125_125217

noncomputable def a (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def b (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def c (x : ℝ) : ℝ := if x > 0 then x else 0

theorem min_value_a4b3c2 (a b c : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 :=
by sorry

example : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (1/a + 1/b + 1/c = 9) ∧ a^4 * b^3 * c^2 = 1/1152 :=
by 
  use [1/4, 1/3, 1/2]
  split
  norm_num -- 0 < 1/4
  split
  norm_num -- 0 < 1/3
  split
  norm_num -- 0 < 1/2
  split
  norm_num -- 1/(1/4) + 1/(1/3) + 1/(1/2) = 9
  norm_num -- (1/4)^4 * (1/3)^3 * (1/2)^2 = 1/1152

end min_value_a4b3c2_l125_125217


namespace lines_parallel_l125_125111

-- Definitions based on conditions
variable (line1 line2 : ℝ → ℝ → Prop) -- Assuming lines as relations for simplicity
variable (plane : ℝ → ℝ → ℝ → Prop) -- Assuming plane as a relation for simplicity

-- Condition: Both lines are perpendicular to the same plane
def perpendicular_to_plane (line : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ (x y z : ℝ), plane x y z → line x y

axiom line1_perpendicular : perpendicular_to_plane line1 plane
axiom line2_perpendicular : perpendicular_to_plane line2 plane

-- Theorem: Both lines are parallel
theorem lines_parallel : ∀ (line1 line2 : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop),
  (perpendicular_to_plane line1 plane) →
  (perpendicular_to_plane line2 plane) →
  (∀ x y : ℝ, line1 x y → line2 x y) := sorry

end lines_parallel_l125_125111


namespace time_to_read_18_pages_l125_125446

-- Definitions based on the conditions
def reading_rate : ℚ := 2 / 4 -- Amalia reads 4 pages in 2 minutes
def pages_to_read : ℕ := 18 -- Number of pages Amalia needs to read

-- Goal: Total time required to read 18 pages
theorem time_to_read_18_pages (r : ℚ := reading_rate) (p : ℕ := pages_to_read) :
  p * r = 9 := by
  sorry

end time_to_read_18_pages_l125_125446


namespace part_i_part_ii_l125_125800

noncomputable def f (x a : ℝ) := |x - a|

theorem part_i :
  (∀ (x : ℝ), (f x 1) ≥ (|x + 1| + 1) ↔ x ≤ -0.5) :=
sorry

theorem part_ii :
  (∀ (x a : ℝ), (f x a) + 3 * x ≤ 0 → { x | x ≤ -1 } ⊆ { x | (f x a) + 3 * x ≤ 0 }) →
  (∀ (a : ℝ), (0 ≤ a ∧ a ≤ 2) ∨ (-4 ≤ a ∧ a < 0)) :=
sorry

end part_i_part_ii_l125_125800


namespace final_inventory_is_correct_l125_125315

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct_l125_125315


namespace maximum_possible_en_value_l125_125387

def bn (n : ℕ) : ℤ :=
  (10^n - 1) / 7

def en (n : ℕ) : ℤ :=
  Int.gcd (bn n) (bn (n + 2))

theorem maximum_possible_en_value : ∃ n : ℕ, en n = 99 :=
by
  sorry

end maximum_possible_en_value_l125_125387


namespace function_quadrants_l125_125873

theorem function_quadrants (n : ℝ) (h: ∀ x : ℝ, x ≠ 0 → ((n-1)*x * x > 0)) : n > 1 :=
sorry

end function_quadrants_l125_125873


namespace smallest_positive_omega_l125_125108

theorem smallest_positive_omega (f g : ℝ → ℝ) (ω : ℝ) 
  (hf : ∀ x, f x = Real.cos (ω * x)) 
  (hg : ∀ x, g x = Real.sin (ω * x - π / 4)) 
  (heq : ∀ x, f (x - π / 2) = g x) :
  ω = 3 / 2 :=
sorry

end smallest_positive_omega_l125_125108


namespace circle_tangent_radius_l125_125109

noncomputable def R : ℝ := 4
noncomputable def r : ℝ := 3
noncomputable def O1O2 : ℝ := R + r
noncomputable def r_inscribed : ℝ := (R * r) / O1O2

theorem circle_tangent_radius :
  r_inscribed = (24 : ℝ) / 7 :=
by
  -- The proof would go here
  sorry

end circle_tangent_radius_l125_125109


namespace find_expression_l125_125594

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_x2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem find_expression (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : symmetric_about_x2 f)
  (h3 : ∀ x, -2 < x ∧ x ≤ 2 → f x = -x^2 + 1) :
  ∀ x, -6 < x ∧ x < -2 → f x = -(x + 4)^2 + 1 :=
by
  sorry

end find_expression_l125_125594


namespace B_should_be_paid_2307_69_l125_125423

noncomputable def A_work_per_day : ℚ := 1 / 15
noncomputable def B_work_per_day : ℚ := 1 / 10
noncomputable def C_work_per_day : ℚ := 1 / 20
noncomputable def combined_work_per_day : ℚ := A_work_per_day + B_work_per_day + C_work_per_day
noncomputable def total_work : ℚ := 1
noncomputable def total_wages : ℚ := 5000
noncomputable def time_taken : ℚ := total_work / combined_work_per_day
noncomputable def B_share_of_work : ℚ := B_work_per_day / combined_work_per_day
noncomputable def B_share_of_wages : ℚ := B_share_of_work * total_wages

theorem B_should_be_paid_2307_69 : B_share_of_wages = 2307.69 := by
  sorry

end B_should_be_paid_2307_69_l125_125423


namespace solve_inequality_l125_125937

theorem solve_inequality (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ x ∈ Set.Ioo (-1 / 3 : ℝ) 1 :=
sorry

end solve_inequality_l125_125937


namespace balloon_permutations_l125_125818

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l125_125818


namespace johns_age_l125_125906

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125906


namespace proof_l125_125420

-- Define the universal set U.
def U : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set M.
def M : Set ℕ := {1, 2, 3}

-- Define set N.
def N : Set ℕ := {3, 4, 5, 6}

-- The complement of M with respect to U.
def compl_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- The intersection of complement of M and N.
def result : Set ℕ := compl_U_M ∩ N

-- The theorem to be proven.
theorem proof : result = {4, 5, 6} := by
  -- This is where the proof would go.
  sorry

end proof_l125_125420


namespace max_value_of_expression_l125_125337

def f (x : ℝ) : ℝ := x^3 + x

theorem max_value_of_expression
  (a b : ℝ)
  (h : f (a^2) + f (2 * b^2 - 3) = 0) :
  a * Real.sqrt (1 + b^2) ≤ 5 * Real.sqrt 2 / 4 := sorry

end max_value_of_expression_l125_125337


namespace ratio_removing_middle_digit_l125_125604

theorem ratio_removing_middle_digit 
  (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9)
  (h1 : 10 * b + c = 8 * a) 
  (h2 : 10 * a + b = 8 * c) : 
  (10 * a + c) / b = 17 :=
by sorry

end ratio_removing_middle_digit_l125_125604


namespace subset_of_inter_eq_self_l125_125187

variable {α : Type*}
variables (M N : Set α)

theorem subset_of_inter_eq_self (h : M ∩ N = M) : M ⊆ N :=
sorry

end subset_of_inter_eq_self_l125_125187


namespace emily_weight_l125_125037

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end emily_weight_l125_125037


namespace largest_unattainable_sum_l125_125683

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end largest_unattainable_sum_l125_125683


namespace positive_difference_of_squares_l125_125407

theorem positive_difference_of_squares {x y : ℕ} (hx : x > y) (hxy_sum : x + y = 70) (hxy_diff : x - y = 20) :
  x^2 - y^2 = 1400 :=
by
  sorry

end positive_difference_of_squares_l125_125407


namespace zeros_not_adjacent_probability_l125_125353

-- Definitions based on the conditions
def total_arrangements : ℕ := Nat.choose 6 2
def non_adjacent_zero_arrangements : ℕ := Nat.choose 5 2

-- The probability that the 2 zeros are not adjacent
def probability_non_adjacent_zero : ℚ :=
  (non_adjacent_zero_arrangements : ℚ) / (total_arrangements : ℚ)

-- The theorem statement
theorem zeros_not_adjacent_probability :
  probability_non_adjacent_zero = 2 / 3 :=
by
  -- The proof would go here
  sorry

end zeros_not_adjacent_probability_l125_125353


namespace cone_height_l125_125300

theorem cone_height (R : ℝ) (r h l : ℝ)
  (volume_sphere : ∀ R,  V_sphere = (4 / 3) * π * R^3)
  (volume_cone : ∀ r h,  V_cone = (1 / 3) * π * r^2 * h)
  (lateral_surface_area : ∀ r l, A_lateral = π * r * l)
  (area_base : ∀ r, A_base = π * r^2)
  (vol_eq : (1/3) * π * r^2 * h = (4/3) * π * R^3)
  (lat_eq : π * r * l = 3 * π * r^2) 
  (pyth_rel : l^2 = r^2 + h^2) :
  h = 4 * R * Real.sqrt 2 := 
sorry

end cone_height_l125_125300


namespace min_value_a4b3c2_l125_125218

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l125_125218


namespace correct_operation_l125_125287

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end correct_operation_l125_125287


namespace missing_side_length_of_pan_l125_125424

-- Definition of the given problem's conditions
def pan_side_length := 29
def total_fudge_pieces := 522
def fudge_piece_area := 1

-- Proof statement in Lean 4
theorem missing_side_length_of_pan : 
  (total_fudge_pieces * fudge_piece_area) = (pan_side_length * 18) :=
by
  sorry

end missing_side_length_of_pan_l125_125424


namespace largest_unattainable_sum_l125_125682

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end largest_unattainable_sum_l125_125682


namespace distinct_arrangements_balloon_l125_125827

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l125_125827


namespace distinct_arrangements_balloon_l125_125812

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125812


namespace number_of_panes_l125_125444

theorem number_of_panes (length width total_area : ℕ) (h_length : length = 12) (h_width : width = 8) (h_total_area : total_area = 768) :
  total_area / (length * width) = 8 :=
by
  sorry

end number_of_panes_l125_125444


namespace ravi_jump_height_l125_125079

theorem ravi_jump_height (j1 j2 j3 : ℕ) (average : ℕ) (ravi_jump_height : ℕ) (h : j1 = 23 ∧ j2 = 27 ∧ j3 = 28) 
  (ha : average = (j1 + j2 + j3) / 3) (hr : ravi_jump_height = 3 * average / 2) : ravi_jump_height = 39 :=
by
  sorry

end ravi_jump_height_l125_125079


namespace minimum_phi_l125_125801

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the condition for g overlapping with f after shifting by φ
noncomputable def shifted_g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ)

theorem minimum_phi (φ : ℝ) (h : φ > 0) :
  (∃ (x : ℝ), shifted_g x φ = f x) ↔ (∃ k : ℕ, φ = Real.pi / 6 + k * Real.pi) :=
sorry

end minimum_phi_l125_125801


namespace wave_propagation_l125_125323

def accum (s : String) : String :=
  String.join (List.intersperse "-" (s.data.enum.map (λ (i : Nat × Char) =>
    String.mk [i.2.toUpper] ++ String.mk (List.replicate i.1 i.2.toLower))))

theorem wave_propagation (s : String) :
  s = "dremCaheя" → accum s = "D-Rr-Eee-Mmmm-Ccccc-Aaaaaa-Hhhhhhh-Eeeeeeee-Яяяяяяяяя" :=
  by
  intro h
  rw [h]
  sorry

end wave_propagation_l125_125323


namespace calculate_result_l125_125780

def binary_op (x y : ℝ) : ℝ := x^2 + y^2

theorem calculate_result (h : ℝ) : binary_op (binary_op h h) (binary_op h h) = 8 * h^4 :=
by
  sorry

end calculate_result_l125_125780


namespace singer_arrangements_l125_125281

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end singer_arrangements_l125_125281


namespace volume_larger_of_cube_cut_plane_l125_125617

/-- Define the vertices and the midpoints -/
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def R : Point := ⟨0, 0, 0⟩
def X : Point := ⟨1, 2, 0⟩
def Y : Point := ⟨2, 0, 1⟩

/-- Equation of the plane passing through R, X and Y -/
def plane_eq (p : Point) : Prop :=
p.x - 2 * p.y - 2 * p.z = 0

/-- The volume of the larger of the two solids formed by cutting the cube with the plane -/
noncomputable def volume_larger_solid : ℝ :=
8 - (4/3 - (1/6))

/-- The statement for the given math problem -/
theorem volume_larger_of_cube_cut_plane :
  volume_larger_solid = 41/6 :=
by
  sorry

end volume_larger_of_cube_cut_plane_l125_125617


namespace candy_ratio_l125_125741

theorem candy_ratio
  (red_candies : ℕ)
  (yellow_candies : ℕ)
  (blue_candies : ℕ)
  (total_candies : ℕ)
  (remaining_candies : ℕ)
  (h1 : red_candies = 40)
  (h2 : yellow_candies = 3 * red_candies - 20)
  (h3 : remaining_candies = 90)
  (h4 : total_candies = remaining_candies + yellow_candies)
  (h5 : blue_candies = total_candies - red_candies - yellow_candies) :
  blue_candies / yellow_candies = 1 / 2 :=
sorry

end candy_ratio_l125_125741


namespace crayons_per_row_correct_l125_125472

-- Declare the given conditions
def total_crayons : ℕ := 210
def num_rows : ℕ := 7

-- Define the expected number of crayons per row
def crayons_per_row : ℕ := 30

-- The desired proof statement: Prove that dividing total crayons by the number of rows yields the expected crayons per row.
theorem crayons_per_row_correct : total_crayons / num_rows = crayons_per_row :=
by sorry

end crayons_per_row_correct_l125_125472


namespace faster_speed_l125_125304

-- Definitions based on conditions:
variable (v : ℝ) -- define the faster speed

-- Conditions:
def initial_speed := 10 -- initial speed in km/hr
def additional_distance := 20 -- additional distance in km
def actual_distance := 50 -- actual distance traveled in km

-- The problem statement:
theorem faster_speed : v = 14 :=
by
  have actual_time : ℝ := actual_distance / initial_speed
  have faster_distance : ℝ := actual_distance + additional_distance
  have equation : actual_time = faster_distance / v
  sorry

end faster_speed_l125_125304


namespace average_temperature_is_95_l125_125260

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l125_125260


namespace balloon_arrangements_l125_125831

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l125_125831


namespace johns_age_l125_125887

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125887


namespace sequence_sum_after_operations_l125_125416

-- Define the initial sequence length
def initial_sequence := [1, 9, 8, 8]

-- Define the sum of initial sequence
def initial_sum := initial_sequence.sum

-- Define the number of operations
def ops := 100

-- Define the increase per operation
def increase_per_op := 7

-- Define the final sum after operations
def final_sum := initial_sum + (increase_per_op * ops)

-- Prove the final sum is 726 after 100 operations
theorem sequence_sum_after_operations : final_sum = 726 := by
  -- Proof omitted as per instructions
  sorry

end sequence_sum_after_operations_l125_125416


namespace smallest_value_of_expression_l125_125913

noncomputable def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem smallest_value_of_expression :
  ∀ z : Fin 4 → ℝ, (∀ i, f (z i) = 0) → 
  ∃ (a b c d : Fin 4), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ d ≠ b ∧ a ≠ c ∧ 
  |(z a * z b) + (z c * z d)| = 8 :=
by
  sorry

end smallest_value_of_expression_l125_125913


namespace baron_munchausen_claim_l125_125449

-- Given conditions and question:
def weight_partition_problem (weights : Finset ℕ) (h_card : weights.card = 50) (h_distinct : ∀ w ∈ weights,  1 ≤ w ∧ w ≤ 100) (h_sum_even : weights.sum id % 2 = 0) : Prop :=
  ¬(∃ (s1 s2 : Finset ℕ), s1 ∪ s2 = weights ∧ s1 ∩ s2 = ∅ ∧ s1.sum id = s2.sum id)

-- We need to prove that the above statement is true.
theorem baron_munchausen_claim :
  ∀ (weights : Finset ℕ), weights.card = 50 ∧ (∀ w ∈ weights, 1 ≤ w ∧ w ≤ 100) ∧ weights.sum id % 2 = 0 → weight_partition_problem weights (by sorry) (by sorry) (by sorry) :=
sorry

end baron_munchausen_claim_l125_125449


namespace johns_age_l125_125891

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l125_125891


namespace convince_the_king_l125_125928

/-- Define the types of inhabitants -/
inductive Inhabitant
| Knight
| Liar
| Normal

/-- Define the king's preference -/
def K (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- All knights tell the truth -/
def tells_truth (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => True
  | Inhabitant.Liar => False
  | Inhabitant.Normal => False

/-- All liars always lie -/
def tells_lie (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => True
  | Inhabitant.Normal => False

/-- Normal persons can tell both truths and lies -/
def can_tell_both (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- Prove there exists a true statement and a false statement to convince the king -/
theorem convince_the_king (p : Inhabitant) :
  (∃ S : Prop, (S ↔ tells_truth p) ∧ K p) ∧ (∃ S' : Prop, (¬ S' ↔ tells_lie p) ∧ K p) :=
by
  sorry

end convince_the_king_l125_125928


namespace part1_part2_l125_125798

-- Definition of the quadratic equation and its real roots condition
def quadratic_has_real_roots (k : ℝ) : Prop :=
  let Δ := (2 * k - 1)^2 - 4 * (k^2 - 1)
  Δ ≥ 0

-- Proving part (1): The range of real number k
theorem part1 (k : ℝ) (hk : quadratic_has_real_roots k) : k ≤ 5 / 4 := 
  sorry

-- Definition using the given condition in part (2)
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 16 + x₁ * x₂

-- Sum and product of roots of the quadratic equation
theorem part2 (k : ℝ) (h : quadratic_has_real_roots k) 
  (hx_sum : ∃ x₁ x₂ : ℝ, x₁ + x₂ = 1 - 2 * k ∧ x₁ * x₂ = k^2 - 1 ∧ roots_condition x₁ x₂) : k = -2 :=
  sorry

end part1_part2_l125_125798


namespace maximize_A_plus_C_l125_125389

theorem maximize_A_plus_C (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
 (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (hB : B = 2) (h7 : (A + C) % (B + D) = 0) 
 (h8 : A < 10) (h9 : B < 10) (h10 : C < 10) (h11 : D < 10) : 
 A + C ≤ 15 :=
sorry

end maximize_A_plus_C_l125_125389


namespace ratio_of_divisor_to_quotient_l125_125674

noncomputable def r : ℕ := 5
noncomputable def n : ℕ := 113

-- Assuming existence of k and quotient Q
axiom h1 : ∃ (k Q : ℕ), (3 * r + 3 = k * Q) ∧ (n = (3 * r + 3) * Q + r)

theorem ratio_of_divisor_to_quotient : ∃ (D Q : ℕ), (D = 3 * r + 3) ∧ (n = D * Q + r) ∧ (D / Q = 3) :=
  by sorry

end ratio_of_divisor_to_quotient_l125_125674


namespace min_value_f_l125_125172

noncomputable def f (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x - 1

theorem min_value_f (a : ℝ) : 
  (∀ x ∈ (Set.Icc (-1 : ℝ) 1), f a x ≥ 
    if a < -1 then 2 * a 
    else if -1 ≤ a ∧ a ≤ 1 then -1 - a ^ 2 
    else -2 * a) := 
by
  sorry

end min_value_f_l125_125172


namespace fraction_inequality_fraction_inequality_equality_case_l125_125019

variables {α β a b : ℝ}

theorem fraction_inequality 
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b) ≤ (β / α + α / β) :=
sorry

-- Additional equality statement
theorem fraction_inequality_equality_case
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b = β / α + α / β) ↔ (a = α ∧ b = β ∨ a = β ∧ b = α) :=
sorry

end fraction_inequality_fraction_inequality_equality_case_l125_125019


namespace smallest_white_marbles_l125_125147

/-
Let n be the total number of Peter's marbles.
Half of the marbles are orange.
One fifth of the marbles are purple.
Peter has 8 silver marbles.
-/
def total_marbles (n : ℕ) : ℕ :=
  n

def orange_marbles (n : ℕ) : ℕ :=
  n / 2

def purple_marbles (n : ℕ) : ℕ :=
  n / 5

def silver_marbles : ℕ :=
  8

def white_marbles (n : ℕ) : ℕ :=
  n - (orange_marbles n + purple_marbles n + silver_marbles)

-- Prove that the smallest number of white marbles Peter could have is 1.
theorem smallest_white_marbles : ∃ n : ℕ, n % 10 = 0 ∧ white_marbles n = 1 :=
sorry

end smallest_white_marbles_l125_125147


namespace sara_no_ingredients_pies_l125_125395

theorem sara_no_ingredients_pies:
  ∀ (total_pies : ℕ) (berries_pies : ℕ) (cream_pies : ℕ) (nuts_pies : ℕ) (coconut_pies : ℕ),
  total_pies = 60 →
  berries_pies = 1/3 * total_pies →
  cream_pies = 1/2 * total_pies →
  nuts_pies = 3/5 * total_pies →
  coconut_pies = 1/5 * total_pies →
  (total_pies - nuts_pies) = 24 :=
by
  intros total_pies berries_pies cream_pies nuts_pies coconut_pies ht hb hc hn hcoc
  sorry

end sara_no_ingredients_pies_l125_125395


namespace lcm_condition_l125_125644

theorem lcm_condition (m : ℕ) (h_m_pos : m > 0) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 :=
by
  sorry

end lcm_condition_l125_125644


namespace Jolene_total_raised_l125_125510

theorem Jolene_total_raised :
  let babysitting_earnings := 4 * 30
  let car_washing_earnings := 5 * 12
  babysitting_earnings + car_washing_earnings = 180 :=
by
  let babysitting_earnings := 4 * 30
  let car_washing_earnings := 5 * 12
  calc
    babysitting_earnings + car_washing_earnings = 120 + 60 : by rfl
    ... = 180 : by rfl

end Jolene_total_raised_l125_125510


namespace part_a_part_b_l125_125289

-- Part (a)
theorem part_a (S : ℕ) (coins : Fin 6 → ℕ)
  (H1 : ∀ (i j : Fin 6), i ≠ j → (coins i + coins j) % 2 = 0)
  (H2 : ∀ (i j k : Fin 6), i ≠ j ∧ i ≠ k ∧ j ≠ k → (coins i + coins j + coins k) % 3 = 0) :
  S % 6 = 0 :=
sorry

-- Part (b)
theorem part_b (S : ℕ) (coins : Fin 8 → ℕ)
  (H1 : ∀ (i j : Fin 8), i ≠ j → (coins i + coins j) % 2 = 0)
  (H2 : ∀ (i j k : Fin 8), i ≠ j ∧ i ≠ k ∧ j ≠ k → (coins i + coins j + coins k) % 3 = 0)
  (H3 : ∀ (i j k l : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l, 
        (coins i + coins j + coins k + coins l) % 4 = 0)
  (H4 : ∀ (i j k l m : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
        k ≠ l ∧ k ≠ m ∧ l ≠ m, (coins i + coins j + coins k + coins l + coins m) % 5 = 0)
  (H5 : ∀ (i j k l m n : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ i ≠ n ∧ 
        j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ j ≠ n ∧ k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n, 
        (coins i + coins j + coins k + coins l + coins m + coins n) % 6 = 0)
  (H6 : ∀ (i j k l m n o : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ i ≠ n ∧ i ≠ o ∧
        j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ j ≠ n ∧ j ≠ o ∧ k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ k ≠ o ∧
        l ≠ m ∧ l ≠ n ∧ l ≠ o ∧ m ≠ n ∧ m ≠ o ∧ n ≠ o, 
        (coins i + coins j + coins k + coins l + coins m + coins n + coins o) % 7 = 0) :
  false :=
sorry

end part_a_part_b_l125_125289


namespace unique_four_digit_number_l125_125116

theorem unique_four_digit_number (a b c d : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) (hd : d ≤ 9)
  (h1 : a + b = c + d)
  (h2 : b + d = 2 * (a + c))
  (h3 : a + d = c)
  (h4 : b + c - a = 3 * d) :
  a = 1 ∧ b = 8 ∧ c = 5 ∧ d = 4 :=
by
  sorry

end unique_four_digit_number_l125_125116


namespace correct_fill_l125_125748

/- Define the conditions and the statement in Lean 4 -/
def sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

/- The condition is that the phrase without the number has 21 characters -/
def initial_length : ℕ := 21

/- Define the term "тридцать две" as the correct number to fill the blank -/
def correct_number := "тридцать две"

/- The target phrase with the correct number filled in -/
def target_sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ " ++ correct_number ++ " БУКВЫ"

/- Prove that the correct number fills the blank correctly -/
theorem correct_fill :
  (String.length target_sentence = 38) :=
by
  /- Convert everything to string length and verify -/
  sorry

end correct_fill_l125_125748


namespace value_that_number_exceeds_l125_125430

theorem value_that_number_exceeds (V : ℤ) (h : 69 = V + 3 * (86 - 69)) : V = 18 :=
by
  sorry

end value_that_number_exceeds_l125_125430


namespace probability_sum_three_dice_3_l125_125963

-- Definition of a fair six-sided die
def fair_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of probability of an event
def probability (s : Set ℕ) (event : ℕ → Prop) : ℚ :=
  if h : finite s then (s.filter event).to_finset.card / s.to_finset.card else 0

theorem probability_sum_three_dice_3 :
  let dice := List.repeat fair_six_sided_die 3 in
  let event := λ result : List ℕ => result.sum = 3 in
  probability ({(r1, r2, r3) | r1 ∈ fair_six_sided_die ∧ r2 ∈ fair_six_sided_die ∧ r3 ∈ fair_six_sided_die }) (λ (r1, r2, r3) => r1 + r2 + r3 = 3) = 1 / 216 :=
by
  sorry

end probability_sum_three_dice_3_l125_125963


namespace cubic_identity_l125_125854

variable {a b c : ℝ}

theorem cubic_identity (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 := 
by 
  sorry

end cubic_identity_l125_125854


namespace positive_integer_is_48_l125_125669

theorem positive_integer_is_48 (n p : ℕ) (h_prime : Prime p) (h_eq : n = 24 * p) (h_min : n ≥ 48) : n = 48 :=
by
  sorry

end positive_integer_is_48_l125_125669


namespace arctan_sum_in_right_triangle_l125_125063

theorem arctan_sum_in_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  (Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4) :=
sorry

end arctan_sum_in_right_triangle_l125_125063


namespace find_P2_l125_125221

def P1 : ℕ := 64
def total_pigs : ℕ := 86

theorem find_P2 : ∃ (P2 : ℕ), P1 + P2 = total_pigs ∧ P2 = 22 :=
by 
  sorry

end find_P2_l125_125221


namespace probability_not_B_given_A_l125_125016

noncomputable def PA : ℚ := 1/3
noncomputable def PB : ℚ := 1/4
noncomputable def P_A_given_B : ℚ := 3/4

def PnotB_given_A : ℚ := 1 - (P_A_given_B * PB / PA)

theorem probability_not_B_given_A (PA PB P_A_given_B : ℚ) 
  (hPA : PA = 1/3) (hPB : PB = 1/4) (hP_A_given_B : P_A_given_B = 3/4) : 
  PnotB_given_A = 7/16 :=
by
  rw [hPA, hPB, hP_A_given_B]
  simp [PnotB_given_A, PA, PB, P_A_given_B]
  sorry

end probability_not_B_given_A_l125_125016


namespace range_of_a_l125_125699

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Define the conditions given:
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def monotonic_increasing_on_nonnegative_reals (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2) → (f x1 < f x2)

def inequality_in_interval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, (1 / 2 ≤ x) → (x ≤ 1) → (f (a * x + 1) ≤ f (x - 2))

-- The theorem we want to prove
theorem range_of_a (h1 : even_function f)
                   (h2 : monotonic_increasing_on_nonnegative_reals f)
                   (h3 : inequality_in_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l125_125699


namespace incorrect_weight_conclusion_l125_125796

theorem incorrect_weight_conclusion (x y : ℝ) (h1 : y = 0.85 * x - 85.71) :
  ¬ (x = 160 → y = 50.29) :=
sorry

end incorrect_weight_conclusion_l125_125796


namespace summer_sales_is_2_million_l125_125759

def spring_sales : ℝ := 4.8
def autumn_sales : ℝ := 7
def winter_sales : ℝ := 2.2
def spring_percentage : ℝ := 0.3

theorem summer_sales_is_2_million :
  ∃ (total_sales : ℝ), total_sales = (spring_sales / spring_percentage) ∧
  ∃ summer_sales : ℝ, total_sales = spring_sales + summer_sales + autumn_sales + winter_sales ∧
  summer_sales = 2 :=
by
  sorry

end summer_sales_is_2_million_l125_125759


namespace prob_even_heads_40_l125_125425

noncomputable def probability_even_heads (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (1/2) * (1 + (2/5) ^ n)

theorem prob_even_heads_40 :
  probability_even_heads 40 = 1/2 * (1 + (2/5) ^ 40) :=
by {
  sorry
}

end prob_even_heads_40_l125_125425


namespace problem_statement_l125_125750

theorem problem_statement :
  let pct := 208 / 100
  let initial_value := 1265
  let step1 := pct * initial_value
  let step2 := step1 ^ 2
  let answer := step2 / 12
  answer = 576857.87 := 
by 
  sorry

end problem_statement_l125_125750


namespace correct_factorization_l125_125148

-- Define the expressions involved in the options
def option_A (x a b : ℝ) : Prop := x * (a - b) = a * x - b * x
def option_B (x y : ℝ) : Prop := x^2 - 1 + y^2 = (x - 1) * (x + 1) + y^2
def option_C (x : ℝ) : Prop := x^2 - 1 = (x + 1) * (x - 1)
def option_D (x a b c : ℝ) : Prop := a * x + b * x + c = x * (a + b) + c

-- Theorem stating that option C represents true factorization
theorem correct_factorization (x : ℝ) : option_C x := by
  sorry

end correct_factorization_l125_125148


namespace largest_circle_radius_l125_125158

noncomputable def largest_inscribed_circle_radius (AB BC CD DA : ℝ) : ℝ :=
  let s := (AB + BC + CD + DA) / 2
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA))
  A / s

theorem largest_circle_radius {AB BC CD DA : ℝ} (hAB : AB = 10) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 13)
  : largest_inscribed_circle_radius AB BC CD DA = 3 * Real.sqrt 245 / 10 :=
by
  simp [largest_inscribed_circle_radius, hAB, hBC, hCD, hDA]
  sorry

end largest_circle_radius_l125_125158


namespace no_month_5_mondays_and_5_thursdays_l125_125877

theorem no_month_5_mondays_and_5_thursdays (n : ℕ) (h : n = 28 ∨ n = 29 ∨ n = 30 ∨ n = 31) :
  ¬ (∃ (m : ℕ) (t : ℕ), m = 5 ∧ t = 5 ∧ 5 * (m + t) ≤ n) := by sorry

end no_month_5_mondays_and_5_thursdays_l125_125877


namespace find_m_value_l125_125639

-- Define the points P and Q and the condition of perpendicularity
def points_PQ (m : ℝ) : Prop := 
  let P := (-2, m)
  let Q := (m, 4)
  let slope_PQ := (m - 4) / (-2 - m)
  slope_PQ * (-1) = -1

-- Problem statement: Find the value of m such that the above condition holds
theorem find_m_value : ∃ (m : ℝ), points_PQ m ∧ m = 1 :=
by sorry

end find_m_value_l125_125639


namespace find_a_b_l125_125624

theorem find_a_b (a b : ℤ) : (∀ (s : ℂ), s^2 + s - 1 = 0 → a * s^18 + b * s^17 + 1 = 0) → (a = 987 ∧ b = -1597) :=
by
  sorry

end find_a_b_l125_125624


namespace emily_weight_l125_125038

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end emily_weight_l125_125038


namespace probability_of_even_adjacent_is_0_25_l125_125115

open Finset

def even_digit_five_digit_numbers : Finset (Finset ℕ) :=
  (finset.range 5).powerset.filter (λ s, s.card = 5 ∧
    (s.filter (λ n, n % 2 = 0)).nonempty ∧
    (s.contains 1 → s.contains 2))

noncomputable def total_five_digit_numbers :=
  (finset.range 5).powerset.filter (λ s, s.card = 5)

noncomputable def probability_even_adjacent :=
  (even_digit_five_digit_numbers.card : ℝ) / (total_five_digit_numbers.card : ℝ)

theorem probability_of_even_adjacent_is_0_25 :
  probability_even_adjacent = 0.25 :=
begin
  sorry
end

end probability_of_even_adjacent_is_0_25_l125_125115


namespace find_fraction_l125_125916

theorem find_fraction (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4)
  (h3 : b = 1) : a / b = (17 + Real.sqrt 269) / 10 :=
by
  sorry

end find_fraction_l125_125916


namespace vinegar_evaporation_rate_l125_125994

def percentage_vinegar_evaporates_each_year (x : ℕ) : Prop :=
  let initial_vinegar : ℕ := 100
  let vinegar_left_after_first_year : ℕ := initial_vinegar - x
  let vinegar_left_after_two_years : ℕ := vinegar_left_after_first_year * (100 - x) / 100
  vinegar_left_after_two_years = 64

theorem vinegar_evaporation_rate :
  ∃ x : ℕ, percentage_vinegar_evaporates_each_year x ∧ x = 20 :=
by
  sorry

end vinegar_evaporation_rate_l125_125994


namespace balloon_arrangements_l125_125847

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l125_125847


namespace ace_first_king_second_prob_l125_125267

def cards : Type := { x : ℕ // x < 52 }

def ace (c : cards) : Prop := 
  c.1 = 0 ∨ c.1 = 1 ∨ c.1 = 2 ∨ c.1 = 3

def king (c : cards) : Prop := 
  c.1 = 4 ∨ c.1 = 5 ∨ c.1 = 6 ∨ c.1 = 7

def prob_ace_first_king_second : ℚ := 4 / 52 * 4 / 51

theorem ace_first_king_second_prob :
  prob_ace_first_king_second = 4 / 663 := by
  sorry

end ace_first_king_second_prob_l125_125267


namespace handshakes_at_meetup_l125_125409

theorem handshakes_at_meetup :
  let gremlins := 25
  let imps := 20
  let sprites := 10
  ∃ (total_handshakes : ℕ), total_handshakes = 1095 :=
by
  sorry

end handshakes_at_meetup_l125_125409


namespace pick_theorem_l125_125099

def lattice_polygon (vertices : List (ℤ × ℤ)) : Prop :=
  ∀ v ∈ vertices, ∃ i j : ℤ, v = (i, j)

variables {n m : ℕ}
variables {A : ℤ}
variables {vertices : List (ℤ × ℤ)}

def lattice_point_count_inside (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count inside points
  sorry

def lattice_point_count_boundary (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count boundary points
  sorry

theorem pick_theorem (h : lattice_polygon vertices) :
  lattice_point_count_inside vertices = n → 
  lattice_point_count_boundary vertices = m → 
  A = n + m / 2 - 1 :=
sorry

end pick_theorem_l125_125099


namespace sum_of_digits_correct_l125_125983

theorem sum_of_digits_correct :
  ∃ a b c : ℕ,
    (1 + 7 + 3 + a) % 9 = 0 ∧
    (1 + 3 - (7 + b)) % 11 = 0 ∧
    (c % 2 = 0) ∧
    ((1 + 7 + 3 + c) % 3 = 0) ∧
    (a + b + c = 19) :=
sorry

end sum_of_digits_correct_l125_125983


namespace convert_kmph_to_mps_l125_125417

theorem convert_kmph_to_mps (speed_kmph : ℕ) (one_kilometer_in_meters : ℕ) (one_hour_in_seconds : ℕ) :
  speed_kmph = 108 →
  one_kilometer_in_meters = 1000 →
  one_hour_in_seconds = 3600 →
  (speed_kmph * one_kilometer_in_meters) / one_hour_in_seconds = 30 := by
  intros h1 h2 h3
  sorry

end convert_kmph_to_mps_l125_125417


namespace parabola_centroid_locus_l125_125912

/-- Let P_0 be a parabola defined by the equation y = m * x^2. 
    Let A and B be points on P_0 such that the tangents at A and B are perpendicular. 
    Let G be the centroid of the triangle formed by A, B, and the vertex of P_0.
    Let P_n be the nth derived parabola.
    Prove that the equation of P_n is y = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n). -/
theorem parabola_centroid_locus (n : ℕ) (m : ℝ) 
  (h_pos_m : 0 < m) :
  ∃ P_n : ℝ → ℝ, 
    ∀ x : ℝ, P_n x = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n) :=
sorry

end parabola_centroid_locus_l125_125912


namespace phi_range_l125_125649

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem phi_range (φ : ℝ) : 
  (|φ| ≤ Real.pi / 2) ∧ 
  (∀ x ∈ Set.Ioo (Real.pi / 24) (Real.pi / 3), f x φ > 2) →
  (Real.pi / 12 ≤ φ ∧ φ ≤ Real.pi / 6) :=
by
  sorry

end phi_range_l125_125649


namespace triangle_no_real_solution_l125_125383

theorem triangle_no_real_solution (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (habc : a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬ (∀ x, x^2 - 2 * b * x + 2 * a * c = 0 ∧
       x^2 - 2 * c * x + 2 * a * b = 0 ∧
       x^2 - 2 * a * x + 2 * b * c = 0) :=
by
  intro H
  sorry

end triangle_no_real_solution_l125_125383


namespace quadratic_solution_pair_l125_125954

open Real

noncomputable def solution_pair : ℝ × ℝ :=
  ((45 - 15 * sqrt 5) / 2, (45 + 15 * sqrt 5) / 2)

theorem quadratic_solution_pair (a c : ℝ) 
  (h1 : (∃ x : ℝ, a * x^2 + 30 * x + c = 0 ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 30 * y + c ≠ 0))
  (h2 : a + c = 45)
  (h3 : a < c) :
  (a, c) = solution_pair :=
sorry

end quadratic_solution_pair_l125_125954


namespace z_is_46_percent_less_than_y_l125_125366

variable (w e y z : ℝ)

-- Conditions
def w_is_60_percent_of_e := w = 0.60 * e
def e_is_60_percent_of_y := e = 0.60 * y
def z_is_150_percent_of_w := z = w * 1.5000000000000002

-- Proof Statement
theorem z_is_46_percent_less_than_y (h1 : w_is_60_percent_of_e w e)
                                    (h2 : e_is_60_percent_of_y e y)
                                    (h3 : z_is_150_percent_of_w z w) :
                                    100 - (z / y * 100) = 46 :=
by
  sorry

end z_is_46_percent_less_than_y_l125_125366


namespace inequality_abc_l125_125482

theorem inequality_abc (a b c : ℝ) 
  (habc : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ ab + bc + ca = 1) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2) :=
sorry

end inequality_abc_l125_125482


namespace problem_stated_l125_125749

-- Definitions of constants based on conditions
def a : ℕ := 5
def b : ℕ := 4
def c : ℕ := 3
def d : ℕ := 400
def x : ℕ := 401

-- Mathematical theorem stating the question == answer given conditions
theorem problem_stated : a * x + b * x + c * x + d = 5212 := 
by 
  sorry

end problem_stated_l125_125749


namespace domain_of_f_l125_125240

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (-x^2 + 9 * x + 10)) / Real.log (x - 1)

theorem domain_of_f :
  {x : ℝ | -x^2 + 9 * x + 10 ≥ 0 ∧ x - 1 > 0 ∧ Real.log (x - 1) ≠ 0} =
  {x : ℝ | (1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 10)} :=
by
  sorry

end domain_of_f_l125_125240


namespace abs_diff_of_solutions_l125_125920

theorem abs_diff_of_solutions (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
sorry

end abs_diff_of_solutions_l125_125920


namespace least_number_conditioned_l125_125626

theorem least_number_conditioned (n : ℕ) :
  n % 56 = 3 ∧ n % 78 = 3 ∧ n % 9 = 0 ↔ n = 2187 := 
sorry

end least_number_conditioned_l125_125626


namespace sum_of_eight_l125_125524

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l125_125524


namespace problem_statement_l125_125697

theorem problem_statement
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
                 (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + 2*b3*x + c3)) :
  b1 * c1 + b2 * c2 + 2 * b3 * c3 = 0 := 
sorry

end problem_statement_l125_125697


namespace penguins_remaining_to_get_fish_l125_125579

def total_penguins : Nat := 36
def fed_penguins : Nat := 19

theorem penguins_remaining_to_get_fish : (total_penguins - fed_penguins = 17) :=
by
  sorry

end penguins_remaining_to_get_fish_l125_125579


namespace ratio_paperback_fiction_to_nonfiction_l125_125399

-- Definitions
def total_books := 160
def hardcover_nonfiction := 25
def paperback_nonfiction := hardcover_nonfiction + 20
def paperback_fiction := total_books - hardcover_nonfiction - paperback_nonfiction

-- Theorem statement
theorem ratio_paperback_fiction_to_nonfiction : paperback_fiction / paperback_nonfiction = 2 :=
by
  -- proof details would go here
  sorry

end ratio_paperback_fiction_to_nonfiction_l125_125399


namespace johns_age_l125_125885

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125885


namespace smallest_altitude_le_3_l125_125248

theorem smallest_altitude_le_3 (a b c h_a h_b h_c : ℝ) (r : ℝ) (h_r : r = 1)
    (h_a_ge_b : a ≥ b) (h_b_ge_c : b ≥ c) 
    (area_eq1 : (a + b + c) / 2 * r = (a * h_a) / 2) 
    (area_eq2 : (a + b + c) / 2 * r = (b * h_b) / 2) 
    (area_eq3 : (a + b + c) / 2 * r = (c * h_c) / 2) : 
    min h_a (min h_b h_c) ≤ 3 := 
by
  sorry

end smallest_altitude_le_3_l125_125248


namespace solve_system_of_equations_l125_125397

theorem solve_system_of_equations:
  ∃ (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

end solve_system_of_equations_l125_125397


namespace sum_of_interior_angles_of_regular_hexagon_l125_125957

theorem sum_of_interior_angles_of_regular_hexagon : 
  ∑ (i : Fin 6), 180 = 720 := 
sorry

end sum_of_interior_angles_of_regular_hexagon_l125_125957


namespace mod_21_solution_l125_125734

theorem mod_21_solution (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n < 21) (h₂ : 47635 ≡ n [MOD 21]) : n = 19 :=
by
  sorry

end mod_21_solution_l125_125734


namespace least_element_in_T_l125_125538

variable (S : Finset ℕ)
variable (T : Finset ℕ)
variable (hS : S = Finset.range 16 \ {0})
variable (hT : T.card = 5)
variable (hTsubS : T ⊆ S)
variable (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0))

theorem least_element_in_T (S T : Finset ℕ) (hT : T.card = 5) (hTsubS : T ⊆ S)
  (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) : 
  ∃ m ∈ T, m = 5 :=
by
  sorry

end least_element_in_T_l125_125538


namespace arithmetic_sequence_length_l125_125653

theorem arithmetic_sequence_length : 
  ∀ {a d l : ℤ}, a = 6 → d = 4 → l = 206 → 
  ∃ n : ℤ, l = a + (n-1) * d ∧ n = 51 := 
by 
  intros a d l ha hd hl
  use (51 : ℤ)
  rw [ha, hd, hl]
  split
  { calc
      206 = 6 + (51 - 1) * 4 : by norm_num }
  { norm_num }

end arithmetic_sequence_length_l125_125653


namespace regular_polygon_sides_l125_125443

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l125_125443


namespace balloon_arrangements_l125_125830

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l125_125830


namespace distinct_arrangements_balloon_l125_125834

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125834


namespace systems_on_second_street_l125_125227

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end systems_on_second_street_l125_125227


namespace problem_solution_l125_125858

def f (x : ℤ) : ℤ := 3 * x + 1
def g (x : ℤ) : ℤ := 4 * x - 3

theorem problem_solution :
  (f (g (f 3))) / (g (f (g 3))) = 112 / 109 := by
sorry

end problem_solution_l125_125858


namespace polynomial_divisibility_l125_125065

noncomputable def polynomial_with_positive_int_coeffs : Type :=
{ f : ℕ → ℕ // ∀ m n : ℕ, f m < f n ↔ m < n }

theorem polynomial_divisibility
  (f : polynomial_with_positive_int_coeffs)
  (n : ℕ) (hn : n > 0) :
  f.1 n ∣ f.1 (f.1 n + 1) ↔ n = 1 :=
sorry

end polynomial_divisibility_l125_125065


namespace triangle_side_s_l125_125251

/-- The sides of a triangle have lengths 8, 13, and s where s is a whole number.
    What is the smallest possible value of s?
    We need to show that the minimum possible value of s such that 8 + s > 13,
    s < 21, and 13 + s > 8 is s = 6. -/
theorem triangle_side_s (s : ℕ) : 
  (8 + s > 13) ∧ (8 + 13 > s) ∧ (13 + s > 8) → s = 6 :=
by
  sorry

end triangle_side_s_l125_125251


namespace triangle_area_l125_125722

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 180 :=
by
  sorry

end triangle_area_l125_125722


namespace age_of_person_l125_125074

/-- Given that Noah's age is twice someone's age and Noah will be 22 years old after 10 years, 
    this theorem states that the age of the person whose age is half of Noah's age is 6 years old. -/
theorem age_of_person (N : ℕ) (P : ℕ) (h1 : P = N / 2) (h2 : N + 10 = 22) : P = 6 := by
  sorry

end age_of_person_l125_125074


namespace range_of_m_l125_125336

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem range_of_m (m : ℝ) : (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) ↔ m ≤ -1/3 :=
by
  sorry

end range_of_m_l125_125336


namespace convert_to_scientific_notation_l125_125570

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l125_125570


namespace cone_base_radius_l125_125460

noncomputable def sector_radius : ℝ := 9
noncomputable def central_angle_deg : ℝ := 240

theorem cone_base_radius :
  let arc_length := (central_angle_deg * Real.pi * sector_radius) / 180
  let base_circumference := arc_length
  let base_radius := base_circumference / (2 * Real.pi)
  base_radius = 6 :=
by
  sorry

end cone_base_radius_l125_125460


namespace find_abc_sum_l125_125717

variables {x : ℤ}

def poly1 (a b : ℤ) : Polynomial ℤ := Polynomial.C b + Polynomial.X * (Polynomial.C a + Polynomial.X)
def poly2 (b c : ℤ) : Polynomial ℤ := Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)

-- Conditions
axiom gcd_condition (a b c : ℤ) : Polynomial.gcd (poly1 a b) (poly2 b c) = Polynomial.C 1 * (Polynomial.X + Polynomial.C 1)
axiom lcm_condition (a b c : ℤ) : Polynomial.lcm (poly1 a b) (poly2 b c) = Polynomial.X^3 - Polynomial.C 2 * Polynomial.X^2 - Polynomial.C 7 * Polynomial.X - Polynomial.C 6

-- Theorem statement
theorem find_abc_sum (a b c : ℤ) : a + b + c = 6 :=
by
  sorry

end find_abc_sum_l125_125717


namespace shares_difference_l125_125993

-- conditions: the ratio is 3:7:12, and the difference between q and r's share is Rs. 3000
theorem shares_difference (x : ℕ) (h : 12 * x - 7 * x = 3000) : 7 * x - 3 * x = 2400 :=
by
  -- simply skip the proof since it's not required in the prompt
  sorry

end shares_difference_l125_125993


namespace quadratic_has_real_solution_l125_125028

theorem quadratic_has_real_solution (a b c : ℝ) : 
  ∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0 ∨ 
           x^2 + (b - c) * x + (c - a) = 0 ∨ 
           x^2 + (c - a) * x + (a - b) = 0 :=
  sorry

end quadratic_has_real_solution_l125_125028


namespace time_difference_l125_125121

theorem time_difference (speed_Xanthia speed_Molly book_pages : ℕ) (minutes_in_hour : ℕ) :
  speed_Xanthia = 120 ∧ speed_Molly = 40 ∧ book_pages = 360 ∧ minutes_in_hour = 60 →
  (book_pages / speed_Molly - book_pages / speed_Xanthia) * minutes_in_hour = 360 := by
  sorry

end time_difference_l125_125121


namespace arithmetic_geometric_progression_inequality_l125_125212

theorem arithmetic_geometric_progression_inequality
  {a b c d e f D g : ℝ}
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (e_pos : 0 < e) (f_pos : 0 < f)
  (h1 : b = a + D)
  (h2 : c = a + 2 * D)
  (h3 : e = a * g)
  (h4 : f = a * g^2)
  (h5 : d = a + 3 * D)
  (h6 : d = a * g^3) : 
  b * c ≥ e * f :=
by sorry

end arithmetic_geometric_progression_inequality_l125_125212


namespace distinct_arrangements_balloon_l125_125838

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125838


namespace parking_lot_cars_l125_125726

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end parking_lot_cars_l125_125726


namespace jeremy_age_l125_125766

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end jeremy_age_l125_125766


namespace fraction_problem_l125_125082

theorem fraction_problem 
  (x : ℚ)
  (h : x = 45 / (8 - (3 / 7))) : 
  x = 315 / 53 := 
sorry

end fraction_problem_l125_125082


namespace geometric_seq_sum_l125_125526

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l125_125526


namespace custom_dollar_five_neg3_l125_125461

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end custom_dollar_five_neg3_l125_125461


namespace area_inside_Z_outside_X_l125_125156

structure Circle :=
  (center : Real × Real)
  (radius : ℝ)

def tangent (A B : Circle) : Prop :=
  dist A.center B.center = A.radius + B.radius

theorem area_inside_Z_outside_X (X Y Z : Circle)
  (hX : X.radius = 1) 
  (hY : Y.radius = 1) 
  (hZ : Z.radius = 1)
  (tangent_XY : tangent X Y)
  (tangent_XZ : tangent X Z)
  (non_intersect_YZ : dist Z.center Y.center > Z.radius + Y.radius) :
  π - 1/2 * π = 1/2 * π := 
by
  sorry

end area_inside_Z_outside_X_l125_125156


namespace sum_of_eight_l125_125525

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l125_125525


namespace correct_student_mark_l125_125400

theorem correct_student_mark
  (avg_wrong : ℕ) (num_students : ℕ) (wrong_mark : ℕ) (avg_correct : ℕ)
  (h1 : num_students = 10) (h2 : avg_wrong = 100) (h3 : wrong_mark = 90) (h4 : avg_correct = 92) :
  ∃ (x : ℕ), x = 10 :=
by
  sorry

end correct_student_mark_l125_125400


namespace find_c_d_l125_125494

theorem find_c_d (y c d : ℕ) (H1 : y = c + Real.sqrt d) (H2 : y^2 + 4 * y + 4 / y + 1 / (y^2) = 30) :
  c + d = 5 :=
sorry

end find_c_d_l125_125494


namespace johns_age_l125_125895

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l125_125895


namespace tangent_line_to_C1_and_C2_is_correct_l125_125331

def C1 (x : ℝ) : ℝ := x ^ 2
def C2 (x : ℝ) : ℝ := -(x - 2) ^ 2
def l (x : ℝ) : ℝ := -2 * x + 3

theorem tangent_line_to_C1_and_C2_is_correct :
  (∃ x1 : ℝ, C1 x1 = l x1 ∧ deriv C1 x1 = deriv l x1) ∧
  (∃ x2 : ℝ, C2 x2 = l x2 ∧ deriv C2 x2 = deriv l x2) :=
sorry

end tangent_line_to_C1_and_C2_is_correct_l125_125331


namespace min_value_condition_l125_125539

theorem min_value_condition
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 36) :
  ∃ x : ℝ, x = (ae)^2 + (bf)^2 + (cg)^2 + (dh)^2 ∧ x ≥ 576 := sorry

end min_value_condition_l125_125539


namespace parallel_lines_l125_125032

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end parallel_lines_l125_125032


namespace quadratic_inequality_solution_l125_125938

theorem quadratic_inequality_solution :
  ∀ x : ℝ, x ∈ Ioo ((4 - Real.sqrt 19) / 3) ((4 + Real.sqrt 19) / 3) → (-3 * x^2 + 8 * x + 1 < 0) :=
by
  intro x hx
  have h1 : x ∈ Ioo ((4 - Real.sqrt 19) / 3) ((4 + Real.sqrt 19) / 3) := hx
  -- Further proof would go here
  sorry

end quadratic_inequality_solution_l125_125938


namespace range_of_a_l125_125542

-- Define the set M
def M : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

-- Define the set N
def N (a : ℝ) : Set ℝ := { x | x ≤ a }

-- The theorem to be proved
theorem range_of_a (a : ℝ) (h : (M ∩ N a).Nonempty) : a ≥ -1 := sorry

end range_of_a_l125_125542


namespace regular_polygon_sides_l125_125436

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l125_125436


namespace maximum_gold_coins_l125_125122

theorem maximum_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n ≤ 146 :=
by
  sorry

end maximum_gold_coins_l125_125122


namespace third_side_correct_length_longest_side_feasibility_l125_125154

-- Definitions for part (a)
def adjacent_side_length : ℕ := 40
def total_fencing_length : ℕ := 140

-- Define third side given the conditions
def third_side_length : ℕ :=
  total_fencing_length - (2 * adjacent_side_length)

-- Problem (a)
theorem third_side_correct_length (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  third_side_length = 60 :=
sorry

-- Definitions for part (b)
def longest_side_possible1 : ℕ := 85
def longest_side_possible2 : ℕ := 65

-- Problem (b)
theorem longest_side_feasibility (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  ¬ (longest_side_possible1 = 85 ∧ longest_side_possible2 = 65) :=
sorry

end third_side_correct_length_longest_side_feasibility_l125_125154


namespace solve_p_plus_q_l125_125695

noncomputable def a : ℚ := p / q

def rx_condition (x : ℝ) : Prop :=
  let w := Real.floor x in
  let f := x - w in
  w * f = a * (x^2 + x)

def valid_sum (xs : Set ℝ) : Prop :=
  (∃ S, S = (xs.filter rx_condition).sum ∧ S = 666)

theorem solve_p_plus_q (p q : ℕ) (h_coprime : Nat.gcd p q = 1) (h_posp : 0 < p) (h_posq : 0 < q) :
  valid_sum {x : ℝ | true} → p + q = 4 :=
by
  sorry

end solve_p_plus_q_l125_125695


namespace number_of_integers_as_difference_of_squares_l125_125046

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l125_125046


namespace find_initial_amount_l125_125788

-- Definitions for conditions
def final_amount : ℝ := 5565
def rate_year1 : ℝ := 0.05
def rate_year2 : ℝ := 0.06

-- Theorem statement to prove the initial amount
theorem find_initial_amount (P : ℝ) 
  (H : final_amount = (P * (1 + rate_year1)) * (1 + rate_year2)) :
  P = 5000 := 
sorry

end find_initial_amount_l125_125788


namespace smallest_number_among_options_l125_125762

noncomputable def binary_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 111111 => 63
  | _ => 0

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 210 => 2 * 6^2 + 1 * 6
  | _ => 0

noncomputable def base_nine_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 85 => 8 * 9 + 5
  | _ => 0

theorem smallest_number_among_options :
  min 75 (min (binary_to_decimal 111111) (min (base_six_to_decimal 210) (base_nine_to_decimal 85))) = binary_to_decimal 111111 :=
by 
  sorry

end smallest_number_among_options_l125_125762


namespace algebraic_expression_transformation_l125_125496

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 + 4 * x + 3 = (x - 1)^2 + a * (x - 1) + b) → (a + b = 14) :=
by
  intros h
  sorry

end algebraic_expression_transformation_l125_125496


namespace geometric_sequence_sum_eight_l125_125514

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l125_125514


namespace travel_time_difference_l125_125076

variable (x : ℝ)

theorem travel_time_difference 
  (distance : ℝ) 
  (speed_diff : ℝ)
  (time_diff_minutes : ℝ)
  (personB_speed : ℝ) 
  (personA_speed := personB_speed - speed_diff) 
  (time_diff_hours := time_diff_minutes / 60) :
  distance = 30 ∧ speed_diff = 3 ∧ time_diff_minutes = 40 ∧ personB_speed = x → 
    (30 / (x - 3)) - (30 / x) = 40 / 60 := 
by 
  sorry

end travel_time_difference_l125_125076


namespace students_in_classroom_l125_125596

theorem students_in_classroom (n : ℕ) :
  n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 → n = 21 ∨ n = 45 :=
by
  sorry

end students_in_classroom_l125_125596


namespace maximum_food_per_guest_l125_125090

theorem maximum_food_per_guest (total_food : ℕ) (min_guests : ℕ) (total_food_eq : total_food = 337) (min_guests_eq : min_guests = 169) :
  ∃ max_food_per_guest, max_food_per_guest = total_food / min_guests ∧ max_food_per_guest = 2 := 
by
  sorry

end maximum_food_per_guest_l125_125090


namespace trisect_54_degree_angle_l125_125183

theorem trisect_54_degree_angle :
  ∃ (a1 a2 : ℝ), a1 = 18 ∧ a2 = 36 ∧ a1 + a2 + a2 = 54 :=
by sorry

end trisect_54_degree_angle_l125_125183


namespace custom_op_value_l125_125463

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_value_l125_125463


namespace radio_price_position_l125_125310

theorem radio_price_position (n : ℕ) (h₁ : n = 42)
  (h₂ : ∃ m : ℕ, m = 18 ∧ 
    (∀ k : ℕ, k < m → (∃ x : ℕ, x > k))) : 
    ∃ m : ℕ, m = 24 :=
by
  sorry

end radio_price_position_l125_125310


namespace range_of_m_l125_125640

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (¬ (proposition_p m) ∧ proposition_q m) ∨ (proposition_p m ∧ ¬ (proposition_q m)) →
  (1/3 <= m ∧ m < 15) :=
sorry

end range_of_m_l125_125640


namespace distinct_arrangements_balloon_l125_125805

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l125_125805


namespace spencer_walk_distance_l125_125210

theorem spencer_walk_distance :
  let distance_house_library := 0.3
  let distance_library_post_office := 0.1
  let total_distance := 0.8
  (total_distance - (distance_house_library + distance_library_post_office)) = 0.4 :=
by
  sorry

end spencer_walk_distance_l125_125210


namespace sum_of_four_digits_l125_125490

theorem sum_of_four_digits (EH OY AY OH : ℕ) (h1 : EH = 4 * OY) (h2 : AY = 4 * OH) : EH + OY + AY + OH = 150 :=
sorry

end sum_of_four_digits_l125_125490


namespace correct_option_division_l125_125415

theorem correct_option_division (x : ℝ) : 
  (-6 * x^3) / (-2 * x^2) = 3 * x :=
by 
  sorry

end correct_option_division_l125_125415


namespace polynomial_approx_eq_l125_125367

theorem polynomial_approx_eq (x : ℝ) (h : x^4 - 4*x^3 + 4*x^2 + 4 = 4.999999999999999) : x = 1 :=
sorry

end polynomial_approx_eq_l125_125367


namespace faye_science_problems_l125_125623

variable (total_problems math_problems science_problems : Nat)
variable (finished_at_school left_for_homework : Nat)

theorem faye_science_problems :
  finished_at_school = 40 ∧ left_for_homework = 15 ∧ math_problems = 46 →
  total_problems = finished_at_school + left_for_homework →
  science_problems = total_problems - math_problems →
  science_problems = 9 :=
by
  sorry

end faye_science_problems_l125_125623


namespace problem_inequality_l125_125338

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem problem_inequality (a : ℝ) (m n : ℝ) 
  (h1 : m ∈ Set.Icc 0 2) (h2 : n ∈ Set.Icc 0 2) 
  (h3 : |m - n| ≥ 1) 
  (h4 : f m a / f n a = 1) : 
  1 ≤ a / (Real.exp 1 - 1) ∧ a / (Real.exp 1 - 1) ≤ Real.exp 1 :=
by sorry

end problem_inequality_l125_125338


namespace shape_of_triangle_l125_125368

noncomputable theory

open Real

variable {A B C a b : ℝ}

-- Define the theorem corresponding to the problem statement
theorem shape_of_triangle (h1 : a * cos A = b * cos B) (A_angle : 0 < A) (A_angle_lt_pi : A < π)
  (B_angle : 0 < B) (B_angle_lt_pi : B < π) (C_angle : 0 < C) (C_angle_lt_pi : C < π)
  (angle_sum : A + B + C = π) : 
  (A = B ∨ A + B = π / 2) := 
sorry

end shape_of_triangle_l125_125368


namespace distinct_arrangements_balloon_l125_125813

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125813


namespace cubics_of_sum_and_product_l125_125855

theorem cubics_of_sum_and_product (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 11) : 
  x^3 + y^3 = 670 :=
by
  sorry

end cubics_of_sum_and_product_l125_125855


namespace jack_change_l125_125879

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end jack_change_l125_125879


namespace find_a9_l125_125249

variable {a_n : ℕ → ℝ}

-- Definition of arithmetic progression
def is_arithmetic_progression (a : ℕ → ℝ) (a1 d : ℝ) := ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Conditions
variables (a1 d : ℝ)
variable (h1 : a1 + (a1 + d)^2 = -3)
variable (h2 : ((a1 + a1 + 4 * d) * 5 / 2) = 10)

-- Question, needing the final statement
theorem find_a9 (a : ℕ → ℝ) (ha : is_arithmetic_progression a a1 d) : a 9 = 20 :=
by
    -- Since the theorem requires solving the statements, we use sorry to skip the proof.
    sorry

end find_a9_l125_125249


namespace balloon_permutations_l125_125819

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l125_125819


namespace triangle_side_c_l125_125030

theorem triangle_side_c
  (a b c : ℝ)
  (A B C : ℝ)
  (h_bc : b = 3)
  (h_sinC : Real.sin C = 56 / 65)
  (h_sinB : Real.sin B = 12 / 13)
  (h_Angles : A + B + C = π)
  (h_valid_triangle : ∀ {x y z : ℝ}, x + y > z ∧ x + z > y ∧ y + z > x):
  c = 14 / 5 :=
sorry

end triangle_side_c_l125_125030


namespace sphere_radius_l125_125205

theorem sphere_radius (r : ℝ) (π : ℝ)
    (h1 : Volume = (4 / 3) * π * r^3)
    (h2 : SurfaceArea = 4 * π * r^2)
    (h3 : Volume = SurfaceArea) :
    r = 3 :=
by
  -- Here starts the proof, but we use 'sorry' to skip it as per the instructions.
  sorry

end sphere_radius_l125_125205


namespace cube_faces_sum_l125_125598

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : a = 12) (h2 : b = 13) (h3 : c = 14)
  (h4 : d = 15) (h5 : e = 16) (h6 : f = 17)
  (h_pairs : a + f = b + e ∧ b + e = c + d) :
  a + b + c + d + e + f = 87 := by
  sorry

end cube_faces_sum_l125_125598


namespace initial_alcohol_solution_percentage_l125_125754

noncomputable def initial_percentage_of_alcohol (P : ℝ) :=
  let initial_volume := 6 -- initial volume of solution in liters
  let added_alcohol := 1.2 -- added volume of pure alcohol in liters
  let final_volume := initial_volume + added_alcohol -- final volume in liters
  let final_percentage := 0.5 -- final percentage of alcohol
  ∃ P, (initial_volume * (P / 100) + added_alcohol) / final_volume = final_percentage

theorem initial_alcohol_solution_percentage : initial_percentage_of_alcohol 40 :=
by 
  -- Prove that initial percentage P is 40
  have hs : initial_percentage_of_alcohol 40 := by sorry
  exact hs

end initial_alcohol_solution_percentage_l125_125754


namespace total_flag_distance_moved_l125_125768

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l125_125768


namespace complex_inequality_l125_125487

open Complex

noncomputable def problem_statement : Prop :=
∀ (a b c : ℂ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (a / Complex.abs a ≠ b / Complex.abs b) ⟹
    max (Complex.abs (a * c + b)) (Complex.abs (b * c + a)) ≥
    (1 / 2) * Complex.abs (a + b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b))

theorem complex_inequality : problem_statement :=
by
  sorry

end complex_inequality_l125_125487


namespace not_symmetric_about_point_l125_125025

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (4 - x)

theorem not_symmetric_about_point : ¬ (∀ h : ℝ, f (1 + h) = f (1 - h)) :=
by
  sorry

end not_symmetric_about_point_l125_125025


namespace mike_needs_more_money_l125_125071

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end mike_needs_more_money_l125_125071


namespace balloon_arrangements_l125_125841

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l125_125841


namespace average_temperature_is_95_l125_125262

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l125_125262


namespace identify_base_7_l125_125224

theorem identify_base_7 :
  ∃ b : ℕ, (b > 1) ∧ 
  (2 * b^4 + 3 * b^3 + 4 * b^2 + 5 * b^1 + 1 * b^0) +
  (1 * b^4 + 5 * b^3 + 6 * b^2 + 4 * b^1 + 2 * b^0) =
  (4 * b^4 + 2 * b^3 + 4 * b^2 + 2 * b^1 + 3 * b^0) ∧
  b = 7 :=
by
  sorry

end identify_base_7_l125_125224


namespace coin_flip_probability_l125_125084

noncomputable def probability_successful_outcomes : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 3
  successful_outcomes / total_outcomes

theorem coin_flip_probability :
  probability_successful_outcomes = 3 / 32 :=
by
  sorry

end coin_flip_probability_l125_125084


namespace eight_term_sum_l125_125521

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l125_125521


namespace min_value_expression_l125_125917

/-- 
Given real numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p such that 
abcd = 16, efgh = 16, ijkl = 16, and mnop = 16, prove that the minimum value of 
(aeim)^2 + (bfjn)^2 + (cgko)^2 + (dhlp)^2 is 1024. 
-/
theorem min_value_expression (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 16) 
  (h3 : i * j * k * l = 16) 
  (h4 : m * n * o * p = 16) : 
  (a * e * i * m) ^ 2 + (b * f * j * n) ^ 2 + (c * g * k * o) ^ 2 + (d * h * l * p) ^ 2 ≥ 1024 :=
by 
  sorry


end min_value_expression_l125_125917


namespace cost_per_adult_is_3_l125_125767

-- Define the number of people in the group
def total_people : ℕ := 12

-- Define the number of kids in the group
def kids : ℕ := 7

-- Define the total cost for the group
def total_cost : ℕ := 15

-- Define the number of adults, which is the total number of people minus the number of kids
def adults : ℕ := total_people - kids

-- Define the cost per adult meal, which is the total cost divided by the number of adults
noncomputable def cost_per_adult : ℕ := total_cost / adults

-- The theorem stating the cost per adult meal is $3
theorem cost_per_adult_is_3 : cost_per_adult = 3 :=
by
  -- The proof is skipped
  sorry

end cost_per_adult_is_3_l125_125767


namespace visible_steps_on_escalator_l125_125990

variable (steps_visible : ℕ) -- The number of steps visible on the escalator
variable (al_steps : ℕ := 150) -- Al walks down 150 steps
variable (bob_steps : ℕ := 75) -- Bob walks up 75 steps
variable (al_speed : ℕ := 3) -- Al's walking speed
variable (bob_speed : ℕ := 1) -- Bob's walking speed
variable (escalator_speed : ℚ) -- The speed of the escalator

theorem visible_steps_on_escalator : steps_visible = 120 :=
by
  -- Define times taken by Al and Bob
  let al_time := al_steps / al_speed
  let bob_time := bob_steps / bob_speed

  -- Define effective speeds considering escalator speed 'escalator_speed'
  let al_effective_speed := al_speed - escalator_speed
  let bob_effective_speed := bob_speed + escalator_speed

  -- Calculate the total steps walked if the escalator was stopped (same total steps)
  have al_total_steps := al_effective_speed * al_time
  have bob_total_steps := bob_effective_speed * bob_time

  -- Set up the equation
  have eq := al_total_steps = bob_total_steps

  -- Substitute and solve for escalator_speed
  sorry

end visible_steps_on_escalator_l125_125990


namespace sum_is_272_l125_125365

-- Define the constant number x
def x : ℕ := 16

-- Define the sum of the number and its square
def sum_of_number_and_its_square (n : ℕ) : ℕ := n + n^2

-- State the theorem that the sum of the number and its square is 272 when the number is 16
theorem sum_is_272 : sum_of_number_and_its_square x = 272 :=
by
  sorry

end sum_is_272_l125_125365


namespace johns_age_l125_125897

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l125_125897


namespace geometric_sequence_sum_eight_l125_125516

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l125_125516


namespace faye_age_l125_125470

variable (C D E F : ℕ)

-- Conditions
axiom h1 : D = 16
axiom h2 : D = E - 4
axiom h3 : E = C + 5
axiom h4 : F = C + 2

-- Goal: Prove that F = 17
theorem faye_age : F = 17 :=
by
  sorry

end faye_age_l125_125470


namespace sandyPhoneBill_is_340_l125_125551

namespace SandyPhoneBill

variable (sandyAgeNow : ℕ) (kimAgeNow : ℕ) (sandyPhoneBill : ℕ)

-- Conditions
def kimCurrentAge := kimAgeNow = 10
def sandyFutureAge := sandyAgeNow + 2 = 3 * (kimAgeNow + 2)
def sandyPhoneBillDefinition := sandyPhoneBill = 10 * sandyAgeNow

-- Target proof
theorem sandyPhoneBill_is_340 
  (h1 : kimCurrentAge)
  (h2 : sandyFutureAge)
  (h3 : sandyPhoneBillDefinition) :
  sandyPhoneBill = 340 :=
sorry

end SandyPhoneBill

end sandyPhoneBill_is_340_l125_125551


namespace john_spent_at_candy_store_l125_125197

-- Definition of the conditions
def allowance : ℚ := 1.50
def arcade_spent : ℚ := (3 / 5) * allowance
def remaining_after_arcade : ℚ := allowance - arcade_spent
def toy_store_spent : ℚ := (1 / 3) * remaining_after_arcade

-- Statement and Proof of the Problem
theorem john_spent_at_candy_store : (remaining_after_arcade - toy_store_spent) = 0.40 :=
by
  -- Proof is left as an exercise
  sorry

end john_spent_at_candy_store_l125_125197


namespace sequence_tends_to_zero_l125_125233

noncomputable def p (n : ℕ) : ℝ := (1 / (4 * n * Real.sqrt 3)) * Real.exp (Real.sqrt (2 * n / 3))

theorem sequence_tends_to_zero (r : ℝ) (hr : r > 1) :
  Filter.Tendsto (λ n : ℕ, p n / (r ^ n)) Filter.atTop (nhds 0) :=
begin
  sorry
end

end sequence_tends_to_zero_l125_125233


namespace exists_polynomial_f_divides_f_x2_sub_1_l125_125394

open Polynomial

theorem exists_polynomial_f_divides_f_x2_sub_1 (n : ℕ) :
    ∃ f : Polynomial ℝ, degree f = n ∧ f ∣ (f.comp (X ^ 2 - 1)) :=
by {
  sorry
}

end exists_polynomial_f_divides_f_x2_sub_1_l125_125394


namespace Julie_money_left_after_purchase_l125_125380

noncomputable def saved_money : ℝ := 1500
noncomputable def number_lawns : ℕ := 20
noncomputable def money_per_lawn : ℝ := 20
noncomputable def number_newspapers : ℕ := 600
noncomputable def money_per_newspaper : ℝ := 0.4
noncomputable def number_dogs : ℕ := 24
noncomputable def money_per_dog : ℝ := 15
noncomputable def cost_bike : ℝ := 2345

theorem Julie_money_left_after_purchase :
  let total_earnings := (number_lawns * money_per_lawn
                       + number_newspapers * money_per_newspaper
                       + number_dogs * money_per_dog)
  in let total_money := saved_money + total_earnings
  in let money_left := total_money - cost_bike
  in money_left = 155 := by
  sorry

end Julie_money_left_after_purchase_l125_125380


namespace total_flag_distance_moved_l125_125769

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l125_125769


namespace tetrahedron_volume_correct_l125_125113

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2))

theorem tetrahedron_volume_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 = c^2) :
  tetrahedron_volume a b c = (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2)) :=
by
  sorry

end tetrahedron_volume_correct_l125_125113


namespace units_digit_of_modifiedLucas_L20_eq_d_l125_125615

def modifiedLucas : ℕ → ℕ
| 0 => 3
| 1 => 2
| n + 2 => 2 * modifiedLucas (n + 1) + modifiedLucas n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_modifiedLucas_L20_eq_d :
  ∃ d, units_digit (modifiedLucas (modifiedLucas 20)) = d :=
by
  sorry

end units_digit_of_modifiedLucas_L20_eq_d_l125_125615


namespace jeremy_age_l125_125765

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end jeremy_age_l125_125765


namespace total_fish_l125_125129

-- Conditions
def initial_fish : ℕ := 22
def given_fish : ℕ := 47

-- Question: Total fish Mrs. Sheridan has now
theorem total_fish : initial_fish + given_fish = 69 := by
  sorry

end total_fish_l125_125129


namespace jason_home_distance_l125_125209

theorem jason_home_distance :
  let v1 := 60 -- speed in miles per hour
  let t1 := 0.5 -- time in hours
  let d1 := v1 * t1 -- distance covered in first part of the journey
  let v2 := 90 -- speed in miles per hour for the second part
  let t2 := 1.0 -- remaining time in hours
  let d2 := v2 * t2 -- distance covered in second part of the journey
  let total_distance := d1 + d2 -- total distance to Jason's home
  total_distance = 120 := 
by
  simp only
  sorry

end jason_home_distance_l125_125209


namespace tan_alpha_plus_pi_div_4_l125_125196

noncomputable def tan_plus_pi_div_4 (α : ℝ) : ℝ := Real.tan (α + Real.pi / 4)

theorem tan_alpha_plus_pi_div_4 (α : ℝ) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : (Real.cos α, Real.sin α) • (Real.cos α ^ 2, Real.sin α - 1) = 1 / 5)
  : tan_plus_pi_div_4 α = -1 / 7 := sorry

end tan_alpha_plus_pi_div_4_l125_125196


namespace inequality_solution_set_l125_125628

theorem inequality_solution_set (x : ℝ) :
  (x - 3)^2 - 2 * Real.sqrt ((x - 3)^2) - 3 < 0 ↔ 0 < x ∧ x < 6 :=
by
  sorry

end inequality_solution_set_l125_125628


namespace general_term_an_l125_125633

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 2
noncomputable def S_n (n : ℕ) : ℕ := n^2 + 3 * n

theorem general_term_an (n : ℕ) (h : 1 ≤ n) : a_n n = (S_n n) - (S_n (n-1)) :=
by sorry

end general_term_an_l125_125633


namespace find_prime_and_int_solutions_l125_125785

-- Define the conditions
def is_solution (p x : ℕ) : Prop :=
  x^(p-1) ∣ (p-1)^x + 1

-- Define the statement to be proven
theorem find_prime_and_int_solutions :
  ∀ p x : ℕ, Prime p → (1 ≤ x ∧ x ≤ 2 * p) →
  (is_solution p x ↔ 
    (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ 
    (p = 3 ∧ (x = 1 ∨ x = 3)) ∨
    (x = 1))
:=
by sorry

end find_prime_and_int_solutions_l125_125785


namespace solution_set_quadratic_l125_125414

theorem solution_set_quadratic (a x : ℝ) (h : a < 0) : 
  (x^2 - 2 * a * x - 3 * a^2 < 0) ↔ (3 * a < x ∧ x < -a) := 
by
  sorry

end solution_set_quadratic_l125_125414


namespace simplify_expression_l125_125554

theorem simplify_expression : ( (2^8 + 4^5) * (2^3 - (-2)^3) ^ 8 ) = 0 := 
by sorry

end simplify_expression_l125_125554


namespace function_b_is_even_and_monotonically_increasing_l125_125992

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ s → b ∈ s → a < b → f a ≤ f b

def f (x : ℝ) : ℝ := abs x + 1

theorem function_b_is_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on f (Set.Ioi 0) :=
by
  sorry

end function_b_is_even_and_monotonically_increasing_l125_125992


namespace population_control_l125_125064

   noncomputable def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
   initial_population * (1 + growth_rate / 100) ^ years

   theorem population_control {initial_population : ℝ} {threshold_population : ℝ} {growth_rate : ℝ} {years : ℕ} :
     initial_population = 1.3 ∧ threshold_population = 1.4 ∧ growth_rate = 0.74 ∧ years = 10 →
     population_growth initial_population growth_rate years < threshold_population :=
   by
     intros
     sorry
   
end population_control_l125_125064


namespace max_writers_and_editors_l125_125294

theorem max_writers_and_editors (total people writers editors y x : ℕ) 
  (h1 : total = 110) 
  (h2 : writers = 45) 
  (h3 : editors = 38 + y) 
  (h4 : y > 0) 
  (h5 : 45 + editors + 2 * x = 110) : 
  x = 13 := 
sorry

end max_writers_and_editors_l125_125294


namespace regular_polygon_sides_l125_125474

theorem regular_polygon_sides (n : ℕ) (h : 2 ≤ n) (h_angle : 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l125_125474


namespace cylinder_volume_ratio_l125_125422

noncomputable def volume_ratio (h1 h2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  let r1 := c1 / (2 * Real.pi)
  let r2 := c2 / (2 * Real.pi)
  let V1 := Real.pi * r1^2 * h1
  let V2 := Real.pi * r2^2 * h2
  if V1 > V2 then V1 / V2 else V2 / V1

theorem cylinder_volume_ratio :
  volume_ratio 7 6 6 7 = 7 / 4 :=
by
  sorry

end cylinder_volume_ratio_l125_125422


namespace probability_at_least_one_girl_l125_125299

theorem probability_at_least_one_girl (boys girls : ℕ) (total : ℕ) (choose_two : ℕ) : 
  boys = 3 → girls = 2 → total = boys + girls → choose_two = 2 → 
  1 - (Nat.choose boys choose_two) / (Nat.choose total choose_two) = 7 / 10 :=
by
  sorry

end probability_at_least_one_girl_l125_125299


namespace danivan_drugstore_end_of_week_inventory_l125_125318

-- Define the initial conditions in Lean
def initial_inventory := 4500
def sold_monday := 2445
def sold_tuesday := 900
def sold_wednesday_to_sunday := 50 * 5
def supplier_delivery := 650

-- Define the statement of the proof problem
theorem danivan_drugstore_end_of_week_inventory :
  initial_inventory - (sold_monday + sold_tuesday + sold_wednesday_to_sunday) + supplier_delivery = 1555 :=
by
  sorry

end danivan_drugstore_end_of_week_inventory_l125_125318


namespace complex_number_identity_l125_125497

theorem complex_number_identity : |-i| + i^2018 = 0 := by
  sorry

end complex_number_identity_l125_125497


namespace twenty_first_term_is_4641_l125_125580

def nthGroupStart (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

def sumGroup (start n : ℕ) : ℕ :=
  (n * (start + (start + n - 1))) / 2

theorem twenty_first_term_is_4641 : sumGroup (nthGroupStart 21) 21 = 4641 := by
  sorry

end twenty_first_term_is_4641_l125_125580


namespace cost_of_tea_l125_125504

theorem cost_of_tea (x : ℕ) (h1 : 9 * x < 1000) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end cost_of_tea_l125_125504


namespace john_age_proof_l125_125898

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l125_125898


namespace regular_polygon_sides_l125_125435

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l125_125435


namespace necessary_not_sufficient_l125_125719

theorem necessary_not_sufficient (a b c : ℝ) : (a < b) → (ac^2 < b * c^2) ∧ ∀a b c : ℝ, (ac^2 < b * c^2) → (a < b) :=
sorry

end necessary_not_sufficient_l125_125719


namespace least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l125_125006

noncomputable def sum_of_cubes (n : ℕ) : ℕ :=
  (n * (n + 1)/2)^2

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem least_m_for_sum_of_cubes_is_perfect_cube 
  (h : ∃ m : ℕ, ∀ (a : ℕ), (sum_of_cubes (2*m+1) = a^3) → a = 6):
  m = 1 := sorry

theorem least_k_for_sum_of_squares_is_perfect_square 
  (h : ∃ k : ℕ, ∀ (b : ℕ), (sum_of_squares (2*k+1) = b^2) → b = 77):
  k = 5 := sorry

end least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l125_125006


namespace trapezium_area_l125_125473

theorem trapezium_area (a b area h : ℝ) (h1 : a = 20) (h2 : b = 15) (h3 : area = 245) :
  area = 1 / 2 * (a + b) * h → h = 14 :=
by
  sorry

end trapezium_area_l125_125473


namespace sum_first_n_arithmetic_sequence_l125_125641

theorem sum_first_n_arithmetic_sequence (a1 d : ℝ) (S : ℕ → ℝ) :
  (S 3 + S 6 = 18) → 
  S 3 = 3 * a1 + 3 * d → 
  S 6 = 6 * a1 + 15 * d → 
  S 5 = 10 :=
by
  sorry

end sum_first_n_arithmetic_sequence_l125_125641


namespace cards_left_l125_125382

def number_of_initial_cards : ℕ := 67
def number_of_cards_taken : ℕ := 9

theorem cards_left (l : ℕ) (d : ℕ) (hl : l = number_of_initial_cards) (hd : d = number_of_cards_taken) : l - d = 58 :=
by
  sorry

end cards_left_l125_125382


namespace common_difference_of_arithmetic_sequence_l125_125373

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) (d a1 : ℝ) (h1 : a 3 = a1 + 2 * d) (h2 : a 5 = a1 + 4 * d)
  (h3 : a 7 = a1 + 6 * d) (h4 : a 10 = a1 + 9 * d) (h5 : a 13 = a1 + 12 * d) (h6 : (a 3) + (a 5) = 2) (h7 : (a 7) + (a 10) + (a 13) = 9) :
  d = (1 / 3) := by
  sorry

end common_difference_of_arithmetic_sequence_l125_125373


namespace height_of_box_l125_125974

def base_area : ℕ := 20 * 20
def cost_per_box : ℝ := 1.30
def total_volume : ℕ := 3060000
def amount_spent : ℝ := 663

theorem height_of_box : ∃ h : ℕ, 400 * h = total_volume / (amount_spent / cost_per_box) := sorry

end height_of_box_l125_125974


namespace parallel_lines_l125_125033

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end parallel_lines_l125_125033


namespace infinitely_many_solutions_l125_125781

theorem infinitely_many_solutions (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := sorry

end infinitely_many_solutions_l125_125781


namespace remainder_when_divided_by_29_l125_125743

theorem remainder_when_divided_by_29 (N : ℤ) (h : N % 899 = 63) : N % 29 = 10 :=
sorry

end remainder_when_divided_by_29_l125_125743


namespace custom_op_value_l125_125464

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_value_l125_125464


namespace mike_needs_more_money_l125_125070

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end mike_needs_more_money_l125_125070


namespace gcd_m_n_l125_125410

def m := 122^2 + 234^2 + 345^2 + 10
def n := 123^2 + 233^2 + 347^2 + 10

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l125_125410


namespace partition_contains_all_distances_l125_125709

open Set
open Real

theorem partition_contains_all_distances (P1 P2 P3 : set ℝ^3) (hP1P2P3 : ∀ x, x ∈ P1 ∨ x ∈ P2 ∨ x ∈ P3)
  (hDisjoint : ∀ x, ¬(x ∈ P1 ∧ x ∈ P2 ∧ x ∈ P3)) :
  ∃ (i : {1, 2, 3}), ∀ a ∈ ℝ, ∃ (M N : ℝ^3), M ∈ [if i = 1 then P1 else if i = 2 then P2 else P3] ∧ N ∈ [if i = 1 then P1 else if i = 2 then P2 else P3] ∧ dist M N = a :=
sorry

end partition_contains_all_distances_l125_125709


namespace probability_equals_two_thirds_l125_125347

-- Definitions for total arrangements and favorable arrangements
def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 2

-- Probability that 2 zeros are not adjacent
def probability_not_adjacent : ℚ := favorable_arrangements / total_arrangements

theorem probability_equals_two_thirds : probability_not_adjacent = 2 / 3 := 
by 
  let total_arrangements := 15
  let favorable_arrangements := 10
  have h1 : probability_not_adjacent = (10 : ℚ) / (15 : ℚ) := rfl
  have h2 : (10 : ℚ) / (15 : ℚ) = 2 / 3 := by norm_num
  exact Eq.trans h1 h2 

end probability_equals_two_thirds_l125_125347


namespace sequence_solution_l125_125459

/-
Let {s : ℕ → ℚ} be a sequence defined by:
1. s 1 = 2
2. ∀ n > 1, if n % 3 = 0 then s n = 2 + s (n / 3)
3. ∀ n > 1, if n % 3 ≠ 0 then s n = 2 / s (n - 1)

Prove that if s n = 13 / 29 then n = 154305.
-/

noncomputable def s : ℕ → ℚ
| 1       => 2
| (n + 1) => if (n + 1) % 3 = 0 then 2 + s ((n + 1) / 3) else 2 / s n

theorem sequence_solution (n : ℕ) (h : s n = 13 / 29) : n = 154305 :=
by
  sorry

end sequence_solution_l125_125459


namespace johns_age_l125_125896

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l125_125896


namespace superhero_speed_l125_125761

def convert_speed (speed_mph : ℕ) (mile_to_km : ℚ) : ℚ :=
  let speed_kmh := (speed_mph : ℚ) * (1 / mile_to_km)
  speed_kmh / 60

theorem superhero_speed :
  convert_speed 36000 (6 / 10) = 1000 :=
by sorry

end superhero_speed_l125_125761


namespace regular_polygon_sides_l125_125441

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l125_125441


namespace painted_surface_area_is_33_l125_125301

/-- 
Problem conditions:
    1. We have 14 unit cubes each with side length 1 meter.
    2. The cubes are arranged in a rectangular formation with dimensions 3x3x1.
The question:
    Prove that the total painted surface area is 33 square meters.
-/
def total_painted_surface_area (cubes : ℕ) (dim_x dim_y dim_z : ℕ) : ℕ :=
  let top_area := dim_x * dim_y
  let side_area := 2 * (dim_x * dim_z + dim_y * dim_z + (dim_z - 1) * dim_x)
  top_area + side_area

theorem painted_surface_area_is_33 :
  total_painted_surface_area 14 3 3 1 = 33 :=
by
  -- Proof would go here
  sorry

end painted_surface_area_is_33_l125_125301


namespace total_votes_l125_125676

theorem total_votes (bob_votes total_votes : ℕ) (h1 : bob_votes = 48) (h2 : (2 : ℝ) / 5 * total_votes = bob_votes) :
  total_votes = 120 :=
by
  sorry

end total_votes_l125_125676


namespace fraction_identity_l125_125330

-- Definitions for conditions
variables (a b : ℚ)

-- The main statement to prove
theorem fraction_identity (h : a/b = 2/5) : (a + b) / b = 7 / 5 :=
by
  sorry

end fraction_identity_l125_125330


namespace rowing_speed_still_water_l125_125139

theorem rowing_speed_still_water (v r : ℕ) (h1 : r = 18) (h2 : 1 / (v - r) = 3 * (1 / (v + r))) : v = 36 :=
by sorry

end rowing_speed_still_water_l125_125139


namespace solve_nat_pairs_l125_125979

theorem solve_nat_pairs (n m : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end solve_nat_pairs_l125_125979


namespace medians_perpendicular_area_l125_125544

variable {ABC : Type*} [real_inner_product_space ℝ ABC]

noncomputable def area_of_triangle (A B C : ABC) : ℝ :=
  1 / 2 * ∥(B - A) × (C - A)∥

theorem medians_perpendicular_area {A B C D E : ABC}
  (hD : is_median A B C D) (hE : is_median B A C E)
  (h_perp : ⟪D - A, E - B⟫ = 0)
  (h_AD : ∥D - A∥ = 15)
  (h_BE : ∥E - B∥ = 20) :
  area_of_triangle A B C = 200 := by
  sorry

end medians_perpendicular_area_l125_125544


namespace chapters_in_first_book_l125_125313

theorem chapters_in_first_book (x : ℕ) (h1 : 2 * 15 = 30) (h2 : (x + 30) / 2 + x + 30 = 75) : x = 20 :=
sorry

end chapters_in_first_book_l125_125313


namespace calories_per_serving_l125_125105

theorem calories_per_serving (x : ℕ) (total_calories bread_calories servings : ℕ)
    (h1: total_calories = 500) (h2: bread_calories = 100) (h3: servings = 2)
    (h4: total_calories = bread_calories + (servings * x)) :
    x = 200 :=
by
  sorry

end calories_per_serving_l125_125105


namespace donna_additional_flyers_l125_125069

theorem donna_additional_flyers (m d a : ℕ) (h1 : m = 33) (h2 : d = 2 * m + a) (h3 : d = 71) : a = 5 :=
by
  have m_val : m = 33 := h1
  rw [m_val] at h2
  linarith [h3, h2]

end donna_additional_flyers_l125_125069


namespace john_age_proof_l125_125899

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l125_125899


namespace balloon_arrangements_l125_125844

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l125_125844


namespace sequence_result_l125_125939

theorem sequence_result (initial_value : ℕ) (total_steps : ℕ) 
    (net_effect_one_cycle : ℕ) (steps_per_cycle : ℕ) : 
    initial_value = 100 ∧ total_steps = 26 ∧ 
    net_effect_one_cycle = (15 - 12 + 3) ∧ steps_per_cycle = 3 
    → 
    ∀ (resulting_value : ℕ), resulting_value = 151 :=
by
  sorry

end sequence_result_l125_125939


namespace probability_interval_l125_125480

open MeasureTheory ProbabilityTheory

theorem probability_interval (X : ℝ → ℝ) (μ σ : ℝ) :
  (∀ x, X x ∈ Normal μ (σ^2)) →
  (P {x | μ - 2 * σ < X x ∧ X x < μ + 2 * σ} = 0.9544) →
  (P {x | μ - σ < X x ∧ X x < μ + σ} = 0.6826) →
  μ = 4 →
  σ = 1 →
  P {x | 5 < X x ∧ X x < 6} = 0.1359 :=
by
  intros
  sorry

end probability_interval_l125_125480


namespace regular_polygon_sides_l125_125439

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l125_125439


namespace sum_of_x_y_l125_125859

theorem sum_of_x_y (x y : ℕ) (h1 : 10 * x + y = 75) (h2 : 10 * y + x = 57) : x + y = 12 :=
sorry

end sum_of_x_y_l125_125859


namespace sin_5pi_over_6_l125_125163

theorem sin_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 :=
by
  -- According to the cofunction identity for sine,
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by
    rw [Real.sin_sub_pi]
  -- Considering the identity sin(π - x) = sin(x),
  rw [Real.sin_of_real, Real.sin_pi_div_six]
  sorry

end sin_5pi_over_6_l125_125163


namespace cost_of_adult_ticket_l125_125298

theorem cost_of_adult_ticket (A : ℝ) (H1 : ∀ (cost_child : ℝ), cost_child = 7) 
                             (H2 : ∀ (num_adults : ℝ), num_adults = 2) 
                             (H3 : ∀ (num_children : ℝ), num_children = 2) 
                             (H4 : ∀ (total_cost : ℝ), total_cost = 58) :
    A = 22 :=
by
  -- You can assume variables for children's cost, number of adults, and number of children
  let cost_child := 7
  let num_adults := 2
  let num_children := 2
  let total_cost := 58
  
  -- Formalize the conditions given
  have H_children_cost : num_children * cost_child = 14 := by simp [cost_child, num_children]
  
  -- Establish the total cost equation
  have H_total_equation : num_adults * A + num_children * cost_child = total_cost := 
    by sorry  -- (Total_equation_proof)
  
  -- Solve for A
  sorry  -- Proof step

end cost_of_adult_ticket_l125_125298


namespace fraction_relevant_quarters_l125_125926

-- Define the total number of quarters and the number of relevant quarters
def total_quarters : ℕ := 50
def relevant_quarters : ℕ := 10

-- Define the theorem that states the fraction of relevant quarters is 1/5
theorem fraction_relevant_quarters : (relevant_quarters : ℚ) / total_quarters = 1 / 5 := by
  sorry

end fraction_relevant_quarters_l125_125926


namespace probability_non_adjacent_l125_125351

def total_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n m 

def non_adjacent_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n (m - 1)

def probability_zeros_non_adjacent (n m : ℕ) : ℚ :=
  (non_adjacent_arrangements n m : ℚ) / (total_arrangements n m : ℚ)

theorem probability_non_adjacent (a b : ℕ) (h₁ : a = 4) (h₂ : b = 2) :
  probability_zeros_non_adjacent 5 2 = 2 / 3 := 
by 
  rw [probability_zeros_non_adjacent]
  rw [non_adjacent_arrangements, total_arrangements]
  sorry

end probability_non_adjacent_l125_125351


namespace johnny_marbles_l125_125379

theorem johnny_marbles : (nat.choose 10 4) = 210 := sorry

end johnny_marbles_l125_125379


namespace find_b_l125_125923

noncomputable def p (x : ℕ) := 3 * x + 5
noncomputable def q (x : ℕ) (b : ℕ) := 4 * x - b

theorem find_b : ∃ (b : ℕ), p (q 3 b) = 29 ∧ b = 4 := sorry

end find_b_l125_125923


namespace digit_for_multiple_of_9_l125_125011

theorem digit_for_multiple_of_9 (d : ℕ) : (23450 + d) % 9 = 0 ↔ d = 4 := by
  sorry

end digit_for_multiple_of_9_l125_125011


namespace find_s_base_10_l125_125620

-- Defining the conditions of the problem
def s_in_base_b_equals_42 (b : ℕ) : Prop :=
  let factor_1 := b + 3
  let factor_2 := b + 4
  let factor_3 := b + 5
  let produced_number := factor_1 * factor_2 * factor_3
  produced_number = 2 * b^3 + 3 * b^2 + 2 * b + 5

-- The proof problem as a Lean 4 statement
theorem find_s_base_10 :
  (∃ b : ℕ, s_in_base_b_equals_42 b) →
  13 + 14 + 15 = 42 :=
sorry

end find_s_base_10_l125_125620


namespace distinct_arrangements_balloon_l125_125822

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l125_125822


namespace polygon_interior_exterior_relation_l125_125253

theorem polygon_interior_exterior_relation (n : ℕ) (h1 : (n-2) * 180 = 2 * 360) : n = 6 :=
by sorry

end polygon_interior_exterior_relation_l125_125253


namespace find_speed_l125_125549

noncomputable def distance : ℝ := 600
noncomputable def speed1 : ℝ := 50
noncomputable def meeting_distance : ℝ := distance / 2
noncomputable def departure_time1 : ℝ := 7
noncomputable def departure_time2 : ℝ := 8
noncomputable def meeting_time : ℝ := 13

theorem find_speed (x : ℝ) : 
  (meeting_distance / speed1 = meeting_time - departure_time1) ∧
  (meeting_distance / x = meeting_time - departure_time2) → 
  x = 60 :=
by
  sorry

end find_speed_l125_125549


namespace pairs_satisfying_equation_l125_125375

theorem pairs_satisfying_equation :
  ∀ x y : ℝ, (x ^ 4 + 1) * (y ^ 4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  intros x y
  sorry

end pairs_satisfying_equation_l125_125375


namespace polygon_interior_angles_sum_l125_125252

theorem polygon_interior_angles_sum (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := 
by sorry

end polygon_interior_angles_sum_l125_125252


namespace florida_north_dakota_license_plate_difference_l125_125068

theorem florida_north_dakota_license_plate_difference :
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  florida_license_plates = north_dakota_license_plates :=
by
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  show florida_license_plates = north_dakota_license_plates
  sorry

end florida_north_dakota_license_plate_difference_l125_125068


namespace count_three_digit_concave_numbers_l125_125432

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

theorem count_three_digit_concave_numbers : 
  (∃! n : ℕ, n = 240) := by
  sorry

end count_three_digit_concave_numbers_l125_125432


namespace balloon_arrangements_l125_125851

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l125_125851


namespace correct_operation_l125_125738
variable (a x y: ℝ)

theorem correct_operation : 
  ¬ (5 * a - 2 * a = 3) ∧
  ¬ ((x + 2 * y)^2 = x^2 + 4 * y^2) ∧
  ¬ (x^8 / x^4 = x^2) ∧
  ((2 * a)^3 = 8 * a^3) :=
by
  sorry

end correct_operation_l125_125738


namespace scientific_notation_of_small_number_l125_125309

theorem scientific_notation_of_small_number : (0.0000003 : ℝ) = 3 * 10 ^ (-7) := 
by
  sorry

end scientific_notation_of_small_number_l125_125309


namespace company_members_and_days_l125_125426

theorem company_members_and_days {t n : ℕ} (h : t = 6) :
    n = (t * (t - 1)) / 2 → n = 15 :=
by
  intro hn
  rw [h] at hn
  simp at hn
  exact hn

end company_members_and_days_l125_125426


namespace find_three_fifths_of_neg_twelve_sevenths_l125_125475

def a : ℚ := -12 / 7
def b : ℚ := 3 / 5
def c : ℚ := -36 / 35

theorem find_three_fifths_of_neg_twelve_sevenths : b * a = c := by 
  -- sorry is a placeholder for the actual proof
  sorry

end find_three_fifths_of_neg_twelve_sevenths_l125_125475


namespace inequality_abc_l125_125696

variables {a b c : ℝ}

theorem inequality_abc 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2) ∧ 
    (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := 
by
  sorry

end inequality_abc_l125_125696


namespace convert_to_scientific_notation_l125_125569

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l125_125569


namespace minimum_ab_l125_125188

theorem minimum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ab = a + 4 * b + 5) : ab ≥ 25 :=
sorry

end minimum_ab_l125_125188


namespace ratio_of_books_l125_125701

theorem ratio_of_books (books_last_week : ℕ) (pages_per_book : ℕ) (pages_this_week : ℕ)
  (h_books_last_week : books_last_week = 5)
  (h_pages_per_book : pages_per_book = 300)
  (h_pages_this_week : pages_this_week = 4500) :
  (pages_this_week / pages_per_book) / books_last_week = 3 := by
  sorry

end ratio_of_books_l125_125701


namespace grilled_cheese_sandwiches_l125_125881

theorem grilled_cheese_sandwiches (h g : ℕ) (c_ham c_grilled total_cheese : ℕ)
  (h_count : h = 10)
  (ham_cheese : c_ham = 2)
  (grilled_cheese : c_grilled = 3)
  (cheese_used : total_cheese = 50)
  (sandwich_eq : total_cheese = h * c_ham + g * c_grilled) :
  g = 10 :=
by
  sorry

end grilled_cheese_sandwiches_l125_125881


namespace food_price_before_tax_and_tip_l125_125134

theorem food_price_before_tax_and_tip (total_paid : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (P : ℝ) (h1 : total_paid = 198) (h2 : tax_rate = 0.10) (h3 : tip_rate = 0.20) : 
  P = 150 :=
by
  -- Given that total_paid = 198, tax_rate = 0.10, tip_rate = 0.20,
  -- we should show that the actual price of the food before tax
  -- and tip is $150.
  sorry

end food_price_before_tax_and_tip_l125_125134


namespace ordered_notebooks_amount_l125_125606

def initial_notebooks : ℕ := 10
def ordered_notebooks (x : ℕ) : ℕ := x
def lost_notebooks : ℕ := 2
def current_notebooks : ℕ := 14

theorem ordered_notebooks_amount (x : ℕ) (h : initial_notebooks + ordered_notebooks x - lost_notebooks = current_notebooks) : x = 6 :=
by
  sorry

end ordered_notebooks_amount_l125_125606


namespace regular_hexagon_interior_angles_l125_125956

theorem regular_hexagon_interior_angles (n : ℕ) (h : n = 6) :
  (n - 2) * 180 = 720 :=
by
  subst h
  rfl

end regular_hexagon_interior_angles_l125_125956


namespace problem_statement_l125_125486

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 2)
    (h3 : 0 < b) (h4 : b < 2) (h5 : 0 < c) (h6 : c < 2) :
    ¬ ((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) :=
by
  sorry

end problem_statement_l125_125486


namespace solution_set_of_inequality_l125_125094

theorem solution_set_of_inequality (x : ℝ) : (x^2 - |x| > 0) ↔ (x < -1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l125_125094


namespace right_triangle_area_l125_125276

theorem right_triangle_area (a b c : ℝ) (h₁ : a = 24) (h₂ : c = 26) (h₃ : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 120 :=
by
  sorry

end right_triangle_area_l125_125276


namespace first_number_is_nine_l125_125075

theorem first_number_is_nine (x : ℤ) (h : 11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) : x = 9 :=
by {
  sorry
}

end first_number_is_nine_l125_125075


namespace parallel_lines_coefficient_l125_125362

theorem parallel_lines_coefficient (a : ℝ) :
  (x + 2*a*y - 1 = 0) → (3*a - 1)*x - a*y - 1 = 0 → (a = 0 ∨ a = 1/6) :=
by
  sorry

end parallel_lines_coefficient_l125_125362


namespace number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l125_125513

def S := { x : ℝ // x ≠ 0 }

def f (x : S) : S := sorry

lemma functional_equation (x y : S) (h : (x.val + y.val) ≠ 0) :
  (f x).val + (f y).val = (f ⟨(x.val * y.val) / (x.val + y.val) * (f ⟨x.val + y.val, sorry⟩).val, sorry⟩).val := sorry

-- Prove that the number of possible values of f(3) is 1

theorem number_of_values_f3 : ∃ n : ℕ, n = 1 := sorry

-- Prove that the sum of all possible values of f(3) is 1/3

theorem sum_of_values_f3 : ∃ s : ℚ, s = 1/3 := sorry

-- Prove that n * s = 1/3

theorem product_of_n_and_s (n : ℕ) (s : ℚ) (hn : n = 1) (hs : s = 1/3) : n * s = 1/3 := by
  rw [hn, hs]
  norm_num

end number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l125_125513


namespace product_of_variables_l125_125123

variables (a b c d : ℚ)

theorem product_of_variables :
  4 * a + 5 * b + 7 * c + 9 * d = 82 →
  d + c = 2 * b →
  2 * b + 2 * c = 3 * a →
  c - 2 = d →
  a * b * c * d = 276264960 / 14747943 := by
  sorry

end product_of_variables_l125_125123


namespace julie_money_left_l125_125381

def cost_of_bike : ℕ := 2345
def initial_savings : ℕ := 1500

def mowing_rate : ℕ := 20
def mowing_jobs : ℕ := 20

def paper_rate : ℚ := 0.40
def paper_jobs : ℕ := 600

def dog_rate : ℕ := 15
def dog_jobs : ℕ := 24

def earnings_from_mowing : ℕ := mowing_rate * mowing_jobs
def earnings_from_papers : ℚ := paper_rate * paper_jobs
def earnings_from_dogs : ℕ := dog_rate * dog_jobs

def total_earnings : ℚ := earnings_from_mowing + earnings_from_papers + earnings_from_dogs
def total_money_available : ℚ := initial_savings + total_earnings

def money_left_after_purchase : ℚ := total_money_available - cost_of_bike

theorem julie_money_left : money_left_after_purchase = 155 := sorry

end julie_money_left_l125_125381


namespace find_ticket_cost_l125_125107

-- Define the initial amount Tony had
def initial_amount : ℕ := 20

-- Define the amount Tony paid for a hot dog
def hot_dog_cost : ℕ := 3

-- Define the amount Tony had after buying the ticket and the hot dog
def remaining_amount : ℕ := 9

-- Define the function to find the baseball ticket cost
def ticket_cost (t : ℕ) : Prop := initial_amount - t - hot_dog_cost = remaining_amount

-- The statement to prove
theorem find_ticket_cost : ∃ t : ℕ, ticket_cost t ∧ t = 8 := 
by 
  existsi 8
  unfold ticket_cost
  simp
  exact sorry

end find_ticket_cost_l125_125107


namespace race_outcomes_l125_125236

open Fintype

-- Define the number of participants
def participants : Finset String := {"Abe", "Bobby", "Charles", "Devin", "Edwin", "Fred"}

noncomputable def count_outcomes (s : Finset String) : ℕ :=
  let charles_top3 : ℕ := 3
  let charles_not_in_top3 : Finset String := s.erase "Charles"
  let remaining_two_selected : ℕ := choose (charles_not_in_top3.card) 2
  let remaining_two_arranged : ℕ := 2.factorial
  charles_top3 * remaining_two_selected * remaining_two_arranged

theorem race_outcomes : count_outcomes participants = 60 := by
  sorry

end race_outcomes_l125_125236


namespace prove_ax5_by5_l125_125492

variables {a b x y : ℝ}

theorem prove_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 30)
                      (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 :=
sorry

end prove_ax5_by5_l125_125492


namespace jeremy_age_l125_125764

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end jeremy_age_l125_125764


namespace min_value_a4b3c2_l125_125216

noncomputable def a (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def b (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def c (x : ℝ) : ℝ := if x > 0 then x else 0

theorem min_value_a4b3c2 (a b c : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 :=
by sorry

example : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (1/a + 1/b + 1/c = 9) ∧ a^4 * b^3 * c^2 = 1/1152 :=
by 
  use [1/4, 1/3, 1/2]
  split
  norm_num -- 0 < 1/4
  split
  norm_num -- 0 < 1/3
  split
  norm_num -- 0 < 1/2
  split
  norm_num -- 1/(1/4) + 1/(1/3) + 1/(1/2) = 9
  norm_num -- (1/4)^4 * (1/3)^3 * (1/2)^2 = 1/1152

end min_value_a4b3c2_l125_125216


namespace parabola_and_hyperbola_equation_l125_125646

theorem parabola_and_hyperbola_equation (a b c : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (hp_eq : c = 2)
    (intersect : (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | p.2^2 = 4 * c * p.1}
                ∧ (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1}) :
    (∀ x y : ℝ, y^2 = 4*x ↔ c = 1)
    ∧ (∃ a', a' = 1 / 2 ∧ ∀ x y : ℝ, 4 * x^2 - (4 * y^2) / 3 = 1 ↔ a = a') := 
by 
  -- Proof will be here
  sorry

end parabola_and_hyperbola_equation_l125_125646


namespace stickers_distribution_l125_125803

theorem stickers_distribution : 
  (10 + 5 - 1).choose (5 - 1) = 1001 := 
by
  sorry

end stickers_distribution_l125_125803


namespace find_d_l125_125607

theorem find_d
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd4 : 4 = a * Real.sin 0 + d)
  (hdm2 : -2 = a * Real.sin (π) + d) :
  d = 1 := by
  sorry

end find_d_l125_125607


namespace probability_non_adjacent_l125_125352

def total_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n m 

def non_adjacent_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n (m - 1)

def probability_zeros_non_adjacent (n m : ℕ) : ℚ :=
  (non_adjacent_arrangements n m : ℚ) / (total_arrangements n m : ℚ)

theorem probability_non_adjacent (a b : ℕ) (h₁ : a = 4) (h₂ : b = 2) :
  probability_zeros_non_adjacent 5 2 = 2 / 3 := 
by 
  rw [probability_zeros_non_adjacent]
  rw [non_adjacent_arrangements, total_arrangements]
  sorry

end probability_non_adjacent_l125_125352


namespace hiking_hours_l125_125272

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end hiking_hours_l125_125272


namespace danivan_drugstore_end_of_week_inventory_l125_125319

-- Define the initial conditions in Lean
def initial_inventory := 4500
def sold_monday := 2445
def sold_tuesday := 900
def sold_wednesday_to_sunday := 50 * 5
def supplier_delivery := 650

-- Define the statement of the proof problem
theorem danivan_drugstore_end_of_week_inventory :
  initial_inventory - (sold_monday + sold_tuesday + sold_wednesday_to_sunday) + supplier_delivery = 1555 :=
by
  sorry

end danivan_drugstore_end_of_week_inventory_l125_125319


namespace triangle_angle_bisector_sum_l125_125947

theorem triangle_angle_bisector_sum (P Q R : ℝ × ℝ)
  (hP : P = (-8, 5)) (hQ : Q = (-15, -19)) (hR : R = (1, -7)) 
  (a b c : ℕ) (h : a + c = 89) 
  (gcd_abc : Int.gcd (Int.gcd a b) c = 1) :
  a + c = 89 :=
by
  sorry

end triangle_angle_bisector_sum_l125_125947


namespace percentage_increase_l125_125955

theorem percentage_increase (S P : ℝ) (h1 : (S * (1 + P / 100)) * 0.8 = 1.04 * S) : P = 30 :=
by 
  sorry

end percentage_increase_l125_125955


namespace valid_three_digit_numbers_count_l125_125853

noncomputable def count_valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let excluded_numbers := 81 + 72
  total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 747 :=
by
  sorry

end valid_three_digit_numbers_count_l125_125853


namespace buffy_breath_holding_time_l125_125910

theorem buffy_breath_holding_time (k : ℕ) (b : ℕ) : 
  k = 3 * 60 ∧ b = k - 20 → b - 40 = 120 := 
by
  intros h
  cases h with hk hb
  rw [hk, hb]
  norm_num
  sorry  -- This "sorry" is here to skip the proof

end buffy_breath_holding_time_l125_125910


namespace mutually_exclusive_not_necessarily_complementary_l125_125662

noncomputable theory
open_locale classical

variables {Ω : Type*} [probability_space Ω]

theorem mutually_exclusive_not_necessarily_complementary (A B : set Ω) 
  [measurable_set A] [measurable_set B] (h : P(A ∪ B) = P(A) + P(B) = 1) :
  (A ∩ B = ∅) ∧ ¬(A = Ω \ B ∧ B = Ω \ A) :=
by
  sorry

end mutually_exclusive_not_necessarily_complementary_l125_125662


namespace giselle_initial_doves_l125_125791

theorem giselle_initial_doves (F : ℕ) (h1 : ∀ F, F > 0) (h2 : 3 * F * 3 / 4 + F = 65) : F = 20 :=
sorry

end giselle_initial_doves_l125_125791


namespace number_of_girls_calculation_l125_125102

theorem number_of_girls_calculation : 
  ∀ (number_of_boys number_of_girls total_children : ℕ), 
  number_of_boys = 27 → total_children = 62 → number_of_girls = total_children - number_of_boys → number_of_girls = 35 :=
by
  intros number_of_boys number_of_girls total_children 
  intros h_boys h_total h_calc
  rw [h_boys, h_total] at h_calc
  simp at h_calc
  exact h_calc

end number_of_girls_calculation_l125_125102


namespace cosine_eq_one_fifth_l125_125792

theorem cosine_eq_one_fifth {α : ℝ} 
  (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := 
sorry

end cosine_eq_one_fifth_l125_125792


namespace income_is_10000_l125_125950

-- Define the necessary variables: income, expenditure, and savings
variables (income expenditure : ℕ) (x : ℕ)

-- Define the conditions given in the problem
def ratio_condition : Prop := income = 10 * x ∧ expenditure = 7 * x
def savings_condition : Prop := income - expenditure = 3000

-- State the theorem that needs to be proved
theorem income_is_10000 (h_ratio : ratio_condition income expenditure x) (h_savings : savings_condition income expenditure) : income = 10000 :=
sorry

end income_is_10000_l125_125950


namespace geometric_sequence_sum_l125_125531

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l125_125531


namespace cost_of_natural_seedless_raisins_l125_125491

theorem cost_of_natural_seedless_raisins
  (cost_golden: ℝ) (n_golden: ℕ) (n_natural: ℕ) (cost_mixture: ℝ) (cost_per_natural: ℝ) :
  cost_golden = 2.55 ∧ n_golden = 20 ∧ n_natural = 20 ∧ cost_mixture = 3
  → cost_per_natural = 3.45 :=
by
  sorry

end cost_of_natural_seedless_raisins_l125_125491


namespace find_k_l125_125175

theorem find_k 
  (h : ∀ x, 2 * x ^ 2 + 14 * x + k = 0 → x = ((-14 + Real.sqrt 10) / 4) ∨ x = ((-14 - Real.sqrt 10) / 4)) :
  k = 93 / 4 :=
sorry

end find_k_l125_125175


namespace x_value_l125_125119

theorem x_value (x : ℤ) (h : x = (2009^2 - 2009) / 2009) : x = 2008 := by
  sorry

end x_value_l125_125119


namespace problem_statement_l125_125867

noncomputable def C_points_count (A B : (ℝ × ℝ)) : ℕ :=
  if A = (0, 0) ∧ B = (12, 0) then 4 else 0

theorem problem_statement :
  let A := (0, 0)
  let B := (12, 0)
  C_points_count A B = 4 :=
by
  sorry

end problem_statement_l125_125867


namespace colorings_10x10_board_l125_125345

def colorings_count (n : ℕ) : ℕ := 2^11 - 2

theorem colorings_10x10_board : colorings_count 10 = 2046 :=
by
  sorry

end colorings_10x10_board_l125_125345


namespace diff_of_squares_count_l125_125044

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l125_125044


namespace second_player_wins_for_n_11_l125_125211

theorem second_player_wins_for_n_11 (N : ℕ) (h1 : N = 11) :
  ∃ (list : List ℕ), (∀ x ∈ list, x > 0 ∧ x ≤ 25) ∧
     list.sum ≥ 200 ∧
     (∃ sublist : List ℕ, sublist.sum ≥ 200 - N ∧ sublist.sum ≤ 200 + N) :=
by
  let N := 11
  sorry

end second_player_wins_for_n_11_l125_125211


namespace smaller_cone_volume_ratio_l125_125880

theorem smaller_cone_volume_ratio :
  let r := 12
  let theta1 := 120
  let theta2 := 240
  let arc_length_small := (theta1 / 360) * (2 * Real.pi * r)
  let arc_length_large := (theta2 / 360) * (2 * Real.pi * r)
  let r1 := arc_length_small / (2 * Real.pi)
  let r2 := arc_length_large / (2 * Real.pi)
  let l := r
  let h1 := Real.sqrt (l^2 - r1^2)
  let h2 := Real.sqrt (l^2 - r2^2)
  let V1 := (1 / 3) * Real.pi * r1^2 * h1
  let V2 := (1 / 3) * Real.pi * r2^2 * h2
  V1 / V2 = Real.sqrt 10 / 10 := sorry

end smaller_cone_volume_ratio_l125_125880


namespace roots_of_unity_probability_l125_125921

open Complex

noncomputable def prob_condition (v w : ℂ) (h : v ≠ w ∧ v^2023 = 1 ∧ w^2023 = 1) : ℚ :=
if (sqrt (2 + sqrt 3)) ≤ abs (v + w) then 1 else 0

theorem roots_of_unity_probability :
  (∑ v w in (finset.range 2023).image (λ k, exp (2 * real.pi * I * k / 2023)), 
    if v ≠ w then prob_condition v w ⟨ne_of_ne_zero _,⟨_,_⟩⟩ else 0) / (2023 * 2022) = 337 / 2022 :=
sorry

end roots_of_unity_probability_l125_125921


namespace pipe_A_fills_tank_in_16_hours_l125_125706

theorem pipe_A_fills_tank_in_16_hours
  (A : ℝ)
  (h1 : ∀ t : ℝ, t = 12.000000000000002 → (1/A + 1/24) * t = 5/4) :
  A = 16 :=
by sorry

end pipe_A_fills_tank_in_16_hours_l125_125706


namespace factorize_expression_l125_125003

variable {R : Type} [CommRing R] (m a : R)

theorem factorize_expression : m * a^2 - m = m * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expression_l125_125003


namespace team_structure_ways_l125_125601

open Nat

noncomputable def combinatorial_structure (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem team_structure_ways :
  let total_members := 13
  let team_lead_choices := total_members
  let remaining_after_lead := total_members - 1
  let project_manager_choices := combinatorial_structure remaining_after_lead 3
  let remaining_after_pm1 := remaining_after_lead - 3
  let subordinate_choices_pm1 := combinatorial_structure remaining_after_pm1 3
  let remaining_after_pm2 := remaining_after_pm1 - 3
  let subordinate_choices_pm2 := combinatorial_structure remaining_after_pm2 3
  let remaining_after_pm3 := remaining_after_pm2 - 3
  let subordinate_choices_pm3 := combinatorial_structure remaining_after_pm3 3
  let total_ways := team_lead_choices * project_manager_choices * subordinate_choices_pm1 * subordinate_choices_pm2 * subordinate_choices_pm3
  total_ways = 4804800 :=
by
  sorry

end team_structure_ways_l125_125601


namespace max_value_of_x_l125_125278

theorem max_value_of_x : ∃ x : ℝ, 
  ( (4*x - 16) / (3*x - 4) )^2 + ( (4*x - 16) / (3*x - 4) ) = 18 
  ∧ x = (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
sorry

end max_value_of_x_l125_125278


namespace product_of_digits_l125_125670

theorem product_of_digits (A B : ℕ) (h1 : A + B = 13) (h2 : (10 * A + B) % 4 = 0) : A * B = 42 :=
by
  sorry

end product_of_digits_l125_125670


namespace album_pages_l125_125145

variable (x y : ℕ)

theorem album_pages :
  (20 * x < y) ∧
  (23 * x > y) ∧
  (21 * x + y = 500) →
  x = 12 := by
  sorry

end album_pages_l125_125145


namespace polynomial_solution_l125_125169

noncomputable def q (x : ℝ) : ℝ :=
  -20 / 93 * x^3 - 110 / 93 * x^2 - 372 / 93 * x - 525 / 93

theorem polynomial_solution :
  (q 1 = -11) ∧
  (q 2 = -15) ∧
  (q 3 = -25) ∧
  (q 5 = -65) :=
by
  sorry

end polynomial_solution_l125_125169


namespace no_four_consecutive_lucky_numbers_l125_125924

def is_lucky (n : ℕ) : Prop :=
  let digits := n.digits 10
  n > 999999 ∧ n < 10000000 ∧ (∀ d ∈ digits, d ≠ 0) ∧ 
  n % (digits.foldl (λ x y => x * y) 1) = 0

theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end no_four_consecutive_lucky_numbers_l125_125924


namespace driver_net_hourly_rate_l125_125297

theorem driver_net_hourly_rate
  (hours : ℝ) (speed : ℝ) (efficiency : ℝ) (cost_per_gallon : ℝ) (compensation_rate : ℝ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : efficiency = 25)
  (h4 : cost_per_gallon = 2.50)
  (h5 : compensation_rate = 0.60)
  :
  ((compensation_rate * (speed * hours) - (cost_per_gallon * (speed * hours / efficiency))) / hours) = 25 :=
sorry

end driver_net_hourly_rate_l125_125297


namespace hyperbolas_same_asymptotes_l125_125777

-- Define the given hyperbolas
def hyperbola1 (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
def hyperbola2 (x y M : ℝ) : Prop := (y^2 / 25) - (x^2 / M) = 1

-- The main theorem statement
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, hyperbola1 x y → hyperbola2 x y M) ↔ M = 225/16 :=
by
  sorry

end hyperbolas_same_asymptotes_l125_125777


namespace danivan_drugstore_end_of_week_inventory_l125_125320

-- Define the initial conditions in Lean
def initial_inventory := 4500
def sold_monday := 2445
def sold_tuesday := 900
def sold_wednesday_to_sunday := 50 * 5
def supplier_delivery := 650

-- Define the statement of the proof problem
theorem danivan_drugstore_end_of_week_inventory :
  initial_inventory - (sold_monday + sold_tuesday + sold_wednesday_to_sunday) + supplier_delivery = 1555 :=
by
  sorry

end danivan_drugstore_end_of_week_inventory_l125_125320


namespace domain_of_w_l125_125277

def w (x : ℝ) : ℝ := real.sqrt (x - 2) + real.cbrt (x + 1)

theorem domain_of_w : {x : ℝ | ∃ y : ℝ, w x = y} = Ici 2 :=
by
  sorry

end domain_of_w_l125_125277


namespace intersection_P_Q_l125_125195

def P : Set ℝ := {x | Real.log x / Real.log 2 < -1}
def Q : Set ℝ := {x | abs x < 1}

theorem intersection_P_Q : P ∩ Q = {x | 0 < x ∧ x < 1 / 2} := by
  sorry

end intersection_P_Q_l125_125195


namespace john_total_replacement_cost_l125_125377

def cost_to_replace_all_doors
  (num_bedroom_doors : ℕ)
  (num_outside_doors : ℕ)
  (cost_outside_door : ℕ)
  (cost_bedroom_door : ℕ) : ℕ :=
  let total_cost_outside_doors := num_outside_doors * cost_outside_door
  let total_cost_bedroom_doors := num_bedroom_doors * cost_bedroom_door
  total_cost_outside_doors + total_cost_bedroom_doors

theorem john_total_replacement_cost :
  let num_bedroom_doors := 3
  let num_outside_doors := 2
  let cost_outside_door := 20
  let cost_bedroom_door := cost_outside_door / 2
  cost_to_replace_all_doors num_bedroom_doors num_outside_doors cost_outside_door cost_bedroom_door = 70 := by
  sorry

end john_total_replacement_cost_l125_125377


namespace problem_one_problem_two_l125_125698

-- Define p and q
def p (a x : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := |x - 3| < 1

-- Problem (1)
theorem problem_one (a : ℝ) (h_a : a = 1) (h_pq : p a x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem_two (a : ℝ) (h_a_pos : a > 0) (suff : ¬ p a x → ¬ q x) (not_necess : ¬ (¬ q x → ¬ p a x)) : 
  (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

end problem_one_problem_two_l125_125698


namespace find_digit_A_l125_125061

open Nat

theorem find_digit_A :
  let n := 52
  let k := 13
  let number_of_hands := choose n k
  number_of_hands = 635013587600 → 0 = 0 := by
  suffices h: 635013587600 = 635013587600 by
    simp [h]
  sorry

end find_digit_A_l125_125061


namespace percent_of_x_is_65_l125_125200

variable (z y x : ℝ)

theorem percent_of_x_is_65 :
  (0.45 * z = 0.39 * y) → (y = 0.75 * x) → (z / x = 0.65) := by
  sorry

end percent_of_x_is_65_l125_125200


namespace polynomial_factorization_l125_125056

theorem polynomial_factorization (m n : ℤ) (h₁ : (x^2 + m * x + 6 : ℤ) = (x - 2) * (x + n)) : m = -5 := by
  sorry

end polynomial_factorization_l125_125056


namespace simplify_expr_l125_125715

noncomputable def expr1 : ℝ := 3 * Real.sqrt 8 / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7)
noncomputable def expr2 : ℝ := -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7)

theorem simplify_expr : expr1 = expr2 := by
  sorry

end simplify_expr_l125_125715


namespace probability_of_two_white_balls_l125_125132

-- Define the total number of balls
def total_balls : ℕ := 11

-- Define the number of white balls
def white_balls : ℕ := 5

-- Define the number of ways to choose 2 out of n (combinations)
def choose (n r : ℕ) : ℕ := n.choose r

-- Define the total combinations of drawing 2 balls out of 11
def total_combinations : ℕ := choose total_balls 2

-- Define the combinations of drawing 2 white balls out of 5
def white_combinations : ℕ := choose white_balls 2

-- Define the probability of drawing 2 white balls
noncomputable def probability_white : ℚ := (white_combinations : ℚ) / (total_combinations : ℚ)

-- Now, state the theorem that states the desired result
theorem probability_of_two_white_balls : probability_white = 2 / 11 := sorry

end probability_of_two_white_balls_l125_125132


namespace square_completion_l125_125665

theorem square_completion (a : ℝ) (h : a^2 + 2 * a - 2 = 0) : (a + 1)^2 = 3 := 
by 
  sorry

end square_completion_l125_125665


namespace no_real_solution_l125_125797

noncomputable def augmented_matrix (m : ℝ) : Matrix (Fin 2) (Fin 3) ℝ :=
  ![![m, 4, m+2], ![1, m, m]]

theorem no_real_solution (m : ℝ) :
  (∀ (a b : ℝ), ¬ ∃ (x y : ℝ), a * x + b * y = m ∧ a * x + b * y = 4 ∧ a * x + b * y = m + 2) ↔ m = 2 :=
by
sorry

end no_real_solution_l125_125797


namespace Mary_current_age_l125_125130

theorem Mary_current_age
  (M J : ℕ) 
  (h1 : J - 5 = (M - 5) + 7) 
  (h2 : J + 5 = 2 * (M + 5)) : 
  M = 2 :=
by
  /- We need to show that the current age of Mary (M) is 2
     given the conditions h1 and h2.-/
  sorry

end Mary_current_age_l125_125130


namespace sin2theta_cos2theta_sum_l125_125334

theorem sin2theta_cos2theta_sum (θ : ℝ) (h1 : Real.sin θ = 2 * Real.cos θ) (h2 : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_cos2theta_sum_l125_125334


namespace bowling_ball_weight_l125_125703

variables (b c k : ℝ)

def condition1 : Prop := 9 * b = 6 * c
def condition2 : Prop := c + k = 42
def condition3 : Prop := 3 * k = 2 * c

theorem bowling_ball_weight
  (h1 : condition1 b c)
  (h2 : condition2 c k)
  (h3 : condition3 c k) :
  b = 16.8 :=
sorry

end bowling_ball_weight_l125_125703


namespace smallest_m_satisfying_condition_l125_125631

def D (n : ℕ) : Finset ℕ := (n.divisors : Finset ℕ)

def F (n i : ℕ) : Finset ℕ :=
  (D n).filter (λ a => a % 4 = i)

def f (n i : ℕ) : ℕ :=
  (F n i).card

theorem smallest_m_satisfying_condition :
  ∃ m : ℕ, f m 0 + f m 1 - f m 2 - f m 3 = 2017 ∧
           m = 2^34 * 3^6 * 7^2 * 11^2 :=
by
  sorry

end smallest_m_satisfying_condition_l125_125631


namespace sum_of_consecutive_even_integers_is_24_l125_125953

theorem sum_of_consecutive_even_integers_is_24 (x : ℕ) (h_pos : x > 0)
    (h_eq : (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2))) :
    (x - 2) + x + (x + 2) = 24 :=
sorry

end sum_of_consecutive_even_integers_is_24_l125_125953


namespace distinct_arrangements_balloon_l125_125839

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125839


namespace no_such_n_exists_l125_125165

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, 0 < n ∧
  (∃ a : ℕ, 2 * n^2 + 1 = a^2) ∧
  (∃ b : ℕ, 3 * n^2 + 1 = b^2) ∧
  (∃ c : ℕ, 6 * n^2 + 1 = c^2) :=
sorry

end no_such_n_exists_l125_125165


namespace average_temperature_is_95_l125_125261

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l125_125261


namespace time_fraction_l125_125691

variable (t₅ t₁₅ : ℝ)

def total_distance (t₅ t₁₅ : ℝ) : ℝ :=
  5 * t₅ + 15 * t₁₅

def total_time (t₅ t₁₅ : ℝ) : ℝ :=
  t₅ + t₁₅

def average_speed_eq (t₅ t₁₅ : ℝ) : Prop :=
  10 * (t₅ + t₁₅) = 5 * t₅ + 15 * t₁₅

theorem time_fraction (t₅ t₁₅ : ℝ) (h : average_speed_eq t₅ t₁₅) :
  (t₁₅ / (t₅ + t₁₅)) = 1 / 2 := by
  sorry

end time_fraction_l125_125691


namespace singer_arrangements_l125_125284

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end singer_arrangements_l125_125284


namespace cylinder_volume_ratio_l125_125778

theorem cylinder_volume_ratio
  (h : ℝ)     -- height of cylinder B (radius of cylinder A)
  (r : ℝ)     -- radius of cylinder B (height of cylinder A)
  (VA : ℝ)    -- volume of cylinder A
  (VB : ℝ)    -- volume of cylinder B
  (cond1 : r = h / 3)
  (cond2 : VB = 3 * VA)
  (cond3 : VB = N * Real.pi * h^3) :
  N = 1 / 3 := 
sorry

end cylinder_volume_ratio_l125_125778


namespace number_of_trees_in_garden_l125_125371

def total_yard_length : ℕ := 600
def distance_between_trees : ℕ := 24
def tree_at_each_end : ℕ := 1

theorem number_of_trees_in_garden : (total_yard_length / distance_between_trees) + tree_at_each_end = 26 := by
  sorry

end number_of_trees_in_garden_l125_125371


namespace solve_system_of_equations_l125_125235

theorem solve_system_of_equations (a b c x y z : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (a * y + b * x = c) ∧ (c * x + a * z = b) ∧ (b * z + c * y = a) →
  (x = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
  (y = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
  (z = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by
  sorry

end solve_system_of_equations_l125_125235


namespace min_third_side_triangle_l125_125250

theorem min_third_side_triangle (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_distinct_1 : 42 * a ≠ 72 * b) (h_distinct_2 : 42 * a ≠ c) (h_distinct_3 : 72 * b ≠ c) :
    (42 * a + 72 * b > c) ∧ (42 * a + c > 72 * b) ∧ (72 * b + c > 42 * a) → c ≥ 7 :=
sorry

end min_third_side_triangle_l125_125250


namespace problem_statement_l125_125658

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end problem_statement_l125_125658


namespace gcd_459_357_l125_125114

-- Define the numbers involved
def num1 := 459
def num2 := 357

-- State the proof problem
theorem gcd_459_357 : Int.gcd num1 num2 = 51 := by
  sorry

end gcd_459_357_l125_125114


namespace min_sum_of_dimensions_l125_125263

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a * b * c = 1645) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  a + b + c ≥ 129 :=
sorry

end min_sum_of_dimensions_l125_125263


namespace sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l125_125290

-- Define the first proof problem
theorem sqrt_1001_1003_plus_1_eq_1002 : Real.sqrt (1001 * 1003 + 1) = 1002 := 
by sorry

-- Define the second proof problem to verify the identity
theorem verify_identity (n : ℤ) : (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
by sorry

-- Define the third proof problem
theorem sqrt_2014_2017_plus_1_eq_2014_2017 : Real.sqrt (2014 * 2015 * 2016 * 2017 + 1) = 2014 * 2017 :=
by sorry

end sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l125_125290


namespace problem_equiv_none_of_these_l125_125666

variable {x y : ℝ}

theorem problem_equiv_none_of_these (hx : x ≠ 0) (hx3 : x ≠ 3) (hy : y ≠ 0) (hy5 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) →
  ¬(3 * x + 2 * y = x * y) ∧
  ¬(y = 3 * x / (5 - y)) ∧
  ¬(x / 3 + y / 2 = 3) ∧
  ¬(3 * y / (y - 5) = x) :=
sorry

end problem_equiv_none_of_these_l125_125666


namespace pq_eqv_l125_125981

theorem pq_eqv (p q : Prop) : 
  ((¬ p ∧ ¬ q) ∧ (p ∨ q)) ↔ ((p ∧ ¬ q) ∨ (¬ p ∧ q)) :=
by
  sorry

end pq_eqv_l125_125981


namespace eight_term_sum_l125_125522

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l125_125522


namespace trig_expr_value_l125_125630

theorem trig_expr_value :
  (Real.cos (7 * Real.pi / 24)) ^ 4 +
  (Real.sin (11 * Real.pi / 24)) ^ 4 +
  (Real.sin (17 * Real.pi / 24)) ^ 4 +
  (Real.cos (13 * Real.pi / 24)) ^ 4 = 3 / 2 :=
by
  sorry

end trig_expr_value_l125_125630


namespace john_age_proof_l125_125901

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l125_125901


namespace gcd_of_powers_l125_125581

theorem gcd_of_powers (a b : ℕ) (h1 : a = 2^300 - 1) (h2 : b = 2^315 - 1) :
  gcd a b = 32767 :=
by
  sorry

end gcd_of_powers_l125_125581


namespace find_a_of_binomial_square_l125_125359

theorem find_a_of_binomial_square (a : ℚ) :
  (∃ b : ℚ, (3 * (x : ℚ) + b)^2 = 9 * x^2 + 21 * x + a) ↔ a = 49 / 4 :=
by
  sorry

end find_a_of_binomial_square_l125_125359


namespace percent_increase_of_income_l125_125198

theorem percent_increase_of_income (original_income new_income : ℝ) 
  (h1 : original_income = 120) (h2 : new_income = 180) :
  ((new_income - original_income) / original_income) * 100 = 50 := 
by 
  rw [h1, h2]
  norm_num

end percent_increase_of_income_l125_125198


namespace same_terminal_side_l125_125120

theorem same_terminal_side : 
  let θ1 := 23 * Real.pi / 3
  let θ2 := 5 * Real.pi / 3
  (∃ k : ℤ, θ1 - 2 * k * Real.pi = θ2) :=
sorry

end same_terminal_side_l125_125120


namespace blanket_cost_l125_125978

theorem blanket_cost (x : ℝ) 
    (h₁ : 200 + 750 + 2 * x = 1350) 
    (h₂ : 2 + 5 + 2 = 9) 
    (h₃ : (200 + 750 + 2 * x) / 9 = 150) : 
    x = 200 :=
by
    have h_total : 200 + 750 + 2 * x = 1350 := h₁
    have h_avg : (200 + 750 + 2 * x) / 9 = 150 := h₃
    sorry

end blanket_cost_l125_125978


namespace initial_roses_l125_125730

theorem initial_roses {x : ℕ} (h : x + 11 = 14) : x = 3 := by
  sorry

end initial_roses_l125_125730


namespace distinct_arrangements_balloon_l125_125804

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l125_125804


namespace parallel_lines_l125_125035

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end parallel_lines_l125_125035


namespace square_side_length_l125_125144

theorem square_side_length (s : ℝ) (h : s^2 = 1 / 9) : s = 1 / 3 :=
sorry

end square_side_length_l125_125144


namespace trains_crossing_time_l125_125291

noncomputable def timeToCross (L1 L2 : ℕ) (v1 v2 : ℕ) : ℝ :=
  let total_distance := (L1 + L2 : ℝ)
  let relative_speed := ((v1 + v2) * 1000 / 3600 : ℝ) -- converting km/hr to m/s
  total_distance / relative_speed

theorem trains_crossing_time :
  timeToCross 140 160 60 40 = 10.8 := 
  by 
    sorry

end trains_crossing_time_l125_125291


namespace find_consecutive_numbers_l125_125786

theorem find_consecutive_numbers :
  ∃ (a b c d : ℕ),
      a % 11 = 0 ∧
      b % 7 = 0 ∧
      c % 5 = 0 ∧
      d % 4 = 0 ∧
      b = a + 1 ∧
      c = a + 2 ∧
      d = a + 3 ∧
      (a % 10) = 3 ∧
      (b % 10) = 4 ∧
      (c % 10) = 5 ∧
      (d % 10) = 6 :=
sorry

end find_consecutive_numbers_l125_125786


namespace geometric_sequence_S8_l125_125532

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l125_125532


namespace cycle_selling_price_l125_125757

noncomputable def selling_price (cost_price : ℝ) (gain_percent : ℝ) : ℝ :=
  let gain_amount := (gain_percent / 100) * cost_price
  cost_price + gain_amount

theorem cycle_selling_price :
  selling_price 450 15.56 = 520.02 :=
by
  sorry

end cycle_selling_price_l125_125757


namespace find_m_l125_125652

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def symmetric_about_line (x1 y1 x2 y2 m : ℝ) : Prop := (y1 - y2) / (x1 - x2) = -1
def product_y (y1 y2 : ℝ) : Prop := y1 * y2 = -1 / 2

-- Theorem to be proven
theorem find_m 
  (x1 y1 x2 y2 m : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : symmetric_about_line x1 y1 x2 y2 m)
  (h4 : product_y y1 y2) :
  m = 9 / 4 :=
sorry

end find_m_l125_125652


namespace more_grandsons_or_granddaughters_probability_l125_125702

theorem more_grandsons_or_granddaughters_probability :
  let total_outcomes := 2 ^ 12
  let equal_count_outcomes := Nat.choose 12 6
  let equal_probability := equal_count_outcomes / total_outcomes
  let result_probability := 1 - equal_probability
  result_probability = 793 / 1024 :=
by
  let total_outcomes := (2 : ℚ) ^ 12
  let equal_count_outcomes := Nat.choose 12 6
  let equal_probability := equal_count_outcomes / total_outcomes
  let result_probability := 1 - equal_probability
  have h1 : total_outcomes = 4096 := by norm_num
  have h2 : equal_count_outcomes = 924 := by norm_num
  have h3 : equal_probability = 231 / 1024 := by {
    rw [h2, h1],
    norm_num,
  }
  have h4 : result_probability = 793 / 1024 := by {
    rw [h3],
    norm_num,
  }
  exact h4

end more_grandsons_or_granddaughters_probability_l125_125702


namespace length_of_chord_EF_l125_125679

theorem length_of_chord_EF 
  (rO rN rP : ℝ)
  (AB BC CD : ℝ)
  (AG_EF_intersec_E AG_EF_intersec_F : ℝ)
  (EF : ℝ)
  (cond1 : rO = 10)
  (cond2 : rN = 20)
  (cond3 : rP = 30)
  (cond4 : AB = 2 * rO)
  (cond5 : BC = 2 * rN)
  (cond6 : CD = 2 * rP)
  (cond7 : EF = 6 * Real.sqrt (24 + 2/3)) :
  EF = 6 * Real.sqrt 24.6666 := sorry

end length_of_chord_EF_l125_125679


namespace compute_div_square_of_negatives_l125_125314

theorem compute_div_square_of_negatives : (-128)^2 / (-64)^2 = 4 := by
  sorry

end compute_div_square_of_negatives_l125_125314


namespace sum_of_eight_l125_125523

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l125_125523


namespace correct_operation_l125_125288

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end correct_operation_l125_125288


namespace binary_op_property_l125_125794

variable (X : Type)
variable (star : X → X → X)
variable (h : ∀ x y : X, star (star x y) x = y)

theorem binary_op_property (x y : X) : star x (star y x) = y := 
by 
  sorry

end binary_op_property_l125_125794


namespace geometric_sequence_S8_l125_125533

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l125_125533


namespace equation_of_plane_l125_125625

/--
The equation of the plane passing through the points (2, -2, 2) and (0, 0, 2),
and which is perpendicular to the plane 2x - y + 4z = 8, is given by:
Ax + By + Cz + D = 0 where A, B, C, D are integers such that A > 0 and gcd(|A|,|B|,|C|,|D|) = 1.
-/
theorem equation_of_plane :
  ∃ (A B C D : ℤ),
    A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
    (∀ x y z : ℤ, A * x + B * y + C * z + D = 0 ↔ x + y = 0) :=
sorry

end equation_of_plane_l125_125625


namespace tan_sum_identity_l125_125179

theorem tan_sum_identity (α β : ℝ)
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l125_125179


namespace range_of_a_l125_125632

theorem range_of_a (a : ℝ) : (∃ x > 0, (2 * x - a) / (x + 1) = 1) ↔ a > -1 :=
by {
    sorry
}

end range_of_a_l125_125632


namespace product_divisible_by_3_or_5_l125_125566

theorem product_divisible_by_3_or_5 {a b c d : ℕ} (h : Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d) :
  (a * b * c * d) % 3 = 0 ∨ (a * b * c * d) % 5 = 0 :=
by
  sorry

end product_divisible_by_3_or_5_l125_125566


namespace balloon_arrangements_l125_125843

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l125_125843


namespace fourth_guard_ran_150_meters_l125_125305

def rectangle_width : ℕ := 200
def rectangle_length : ℕ := 300
def total_perimeter : ℕ := 2 * (rectangle_width + rectangle_length)
def three_guards_total_distance : ℕ := 850

def fourth_guard_distance : ℕ := total_perimeter - three_guards_total_distance

theorem fourth_guard_ran_150_meters :
  fourth_guard_distance = 150 :=
by
  -- calculation skipped here
  -- proving fourth_guard_distance as derived being 150 meters
  sorry

end fourth_guard_ran_150_meters_l125_125305


namespace prob_sum_to_3_three_dice_correct_l125_125962

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l125_125962


namespace perpendicular_k_value_parallel_k_value_l125_125636

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

end perpendicular_k_value_parallel_k_value_l125_125636


namespace coconut_grove_produce_trees_l125_125866

theorem coconut_grove_produce_trees (x : ℕ)
  (h1 : 60 * (x + 3) + 120 * x + 180 * (x - 3) = 100 * 3 * x)
  : x = 6 := sorry

end coconut_grove_produce_trees_l125_125866


namespace Rachelle_GPA_Probability_correct_l125_125225

noncomputable def Rachelle_GPA_Probability : ℚ :=
  let P_A_English := 1 / 7 in
  let P_B_English := 1 / 5 in
  let P_C_English := 1 / 3 in
  let P_D_English := 1 - (P_A_English + P_B_English + P_C_English) in
  let P_A_History := 1 / 5 in
  let P_B_History := 1 / 4 in
  let P_C_History := 1 / 2 in
  let P_D_History := 1 - (P_A_History + P_B_History + P_C_History) in

  -- Ensure no D grades
  let P_NotD_English := 1 - P_D_English in
  let P_NotD_History := 1 - P_D_History in

  -- Calculate probabilities for achieving GPA >= 3.5 (Total Points ≥ 14)
  let P_A_A := P_A_English * P_A_History in
  let P_A_B := P_A_English * P_B_History in
  let P_B_A := P_B_English * P_A_History in
  let P_B_B := P_B_English * P_B_History in

  -- Total probability of Rachelle achieving GPA ≥ 3.5 excluding D grades
  let Total_Probability := P_A_A + P_A_B + P_B_A + P_B_B in

  -- Return answer
  Total_Probability

theorem Rachelle_GPA_Probability_correct : Rachelle_GPA_Probability = 27 / 175 :=
by admit -- sorry to skip the detailed proof steps.

end Rachelle_GPA_Probability_correct_l125_125225


namespace sandy_phone_bill_expense_l125_125552

def sandy_age_now (kim_age : ℕ) : ℕ := 3 * (kim_age + 2) - 2

def sandy_phone_bill (sandy_age : ℕ) : ℕ := 10 * sandy_age

theorem sandy_phone_bill_expense
  (kim_age : ℕ)
  (kim_age_condition : kim_age = 10)
  : sandy_phone_bill (sandy_age_now kim_age) = 340 := by
  sorry

end sandy_phone_bill_expense_l125_125552


namespace max_number_of_eligible_ages_l125_125944

-- Definitions based on the problem conditions
def average_age : ℝ := 31
def std_dev : ℝ := 5
def acceptable_age_range (a : ℝ) : Prop := 26 ≤ a ∧ a ≤ 36
def has_masters_degree : Prop := 24 ≤ 26  -- simplified for context indicated in problem
def has_work_experience : Prop := 26 ≥ 26

-- Define the maximum number of different ages of the eligible applicants
noncomputable def max_diff_ages : ℕ := 36 - 26 + 1  -- This matches the solution step directly

-- The theorem stating the result
theorem max_number_of_eligible_ages :
  max_diff_ages = 11 :=
by {
  sorry
}

end max_number_of_eligible_ages_l125_125944


namespace regular_polygon_sides_l125_125437

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l125_125437


namespace derivative_at_neg_one_l125_125055

theorem derivative_at_neg_one (a b c : ℝ) (h : (4*a*(1:ℝ)^3 + 2*b*(1:ℝ)) = 2) :
  (4*a*(-1:ℝ)^3 + 2*b*(-1:ℝ)) = -2 :=
by
  sorry

end derivative_at_neg_one_l125_125055


namespace flag_movement_distance_l125_125772

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l125_125772


namespace distinct_arrangements_balloon_l125_125825

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l125_125825


namespace num_valid_four_digit_numbers_l125_125852

theorem num_valid_four_digit_numbers :
  let N (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d
  ∃ (a b c d : ℕ), 5000 ≤ N a b c d ∧ N a b c d < 7000 ∧ (N a b c d % 5 = 0) ∧ (2 ≤ b ∧ b < c ∧ c ≤ 7) ∧
                   (60 = (if a = 5 ∨ a = 6 then (if d = 0 ∨ d = 5 then 15 else 0) else 0)) :=
sorry

end num_valid_four_digit_numbers_l125_125852


namespace solve_for_x_l125_125556

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end solve_for_x_l125_125556


namespace max_sum_of_arithmetic_sequence_l125_125020

theorem max_sum_of_arithmetic_sequence 
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hS18 : S 18 > 0)
  (hS19 : S 19 < 0)
  (hSn_def : ∀ n, S n = n / 2 * (a 1 + a n))
  : S 9 = max (S n) :=
by {
  sorry
}

end max_sum_of_arithmetic_sequence_l125_125020


namespace walking_speed_l125_125429

-- Define the constants and variables
def speed_there := 25 -- speed from village to post-office in kmph
def total_time := 5.8 -- total round trip time in hours
def distance := 20.0 -- distance to the post-office in km
 
-- Define the theorem that needs to be proved
theorem walking_speed :
  ∃ (speed_back : ℝ), speed_back = 4 := 
by
  sorry

end walking_speed_l125_125429


namespace find_g_at_75_l125_125242

noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y^2
axiom g_at_50 : g 50 = 25

-- The main result to be proved
theorem find_g_at_75 : g 75 = 100 / 9 :=
by
  sorry

end find_g_at_75_l125_125242


namespace geometric_seq_sum_l125_125527

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l125_125527


namespace man_speed_down_l125_125141

variable (d : ℝ) (v : ℝ)

theorem man_speed_down (h1 : 32 > 0) (h2 : 38.4 > 0) (h3 : d > 0) (h4 : v > 0) 
  (avg_speed : 38.4 = (2 * d) / ((d / 32) + (d / v))) : v = 48 :=
sorry

end man_speed_down_l125_125141


namespace area_of_YZW_l125_125870

-- Definitions from conditions
def area_of_triangle_XYZ := 36
def base_XY := 8
def base_YW := 32

-- The theorem to prove
theorem area_of_YZW : 1/2 * base_YW * (2 * area_of_triangle_XYZ / base_XY) = 144 := 
by
  -- Placeholder for the proof  
  sorry

end area_of_YZW_l125_125870


namespace most_stable_performance_l125_125477

theorem most_stable_performance 
    (S_A S_B S_C S_D : ℝ)
    (h_A : S_A = 0.54) 
    (h_B : S_B = 0.61) 
    (h_C : S_C = 0.7) 
    (h_D : S_D = 0.63) :
    S_A <= S_B ∧ S_A <= S_C ∧ S_A <= S_D :=
by {
  sorry
}

end most_stable_performance_l125_125477


namespace balloon_permutations_l125_125817

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l125_125817


namespace seating_arrangement_correct_l125_125391

-- Define the number of seating arrangements based on the given conditions

def seatingArrangements : Nat := 
  2 * 4 * 6

theorem seating_arrangement_correct :
  seatingArrangements = 48 := by
  sorry

end seating_arrangement_correct_l125_125391


namespace arithmetic_mean_of_1_and_4_l125_125086

theorem arithmetic_mean_of_1_and_4 : 
  (1 + 4) / 2 = 5 / 2 := by
  sorry

end arithmetic_mean_of_1_and_4_l125_125086


namespace total_cans_collected_l125_125393

def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8
def total_cans : ℕ := 72

theorem total_cans_collected :
  (bags_on_saturday + bags_on_sunday) * cans_per_bag = total_cans :=
by
  sorry

end total_cans_collected_l125_125393


namespace find_two_digit_number_l125_125989

def tens_digit (n: ℕ) := n / 10
def unit_digit (n: ℕ) := n % 10
def is_required_number (n: ℕ) : Prop :=
  tens_digit n + 2 = unit_digit n ∧ n < 30 ∧ 10 ≤ n

theorem find_two_digit_number (n : ℕ) :
  is_required_number n → n = 13 ∨ n = 24 :=
by
  -- Proof placeholder
  sorry

end find_two_digit_number_l125_125989


namespace largest_of_five_consecutive_odd_integers_with_product_93555_l125_125010

theorem largest_of_five_consecutive_odd_integers_with_product_93555 : 
  ∃ n, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) ∧ (n + 8 = 19) :=
sorry

end largest_of_five_consecutive_odd_integers_with_product_93555_l125_125010


namespace find_value_of_x_l125_125745

theorem find_value_of_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := 
sorry

end find_value_of_x_l125_125745


namespace find_x_l125_125872

noncomputable def angle_sum_triangle (A B C: ℝ) : Prop :=
  A + B + C = 180

noncomputable def vertical_angles_equal (A B: ℝ) : Prop :=
  A = B

noncomputable def right_angle_sum (D E: ℝ) : Prop :=
  D + E = 90

theorem find_x 
  (angle_ABC angle_BAC angle_DCE : ℝ) 
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : angle_sum_triangle angle_ABC angle_BAC angle_DCE)
  (h4 : vertical_angles_equal angle_DCE angle_DCE)
  (h5 : right_angle_sum angle_DCE 30) :
  angle_DCE = 60 :=
by
  sorry

end find_x_l125_125872


namespace cubic_roots_relations_l125_125078

theorem cubic_roots_relations 
    (a b c d : ℚ) 
    (x1 x2 x3 : ℚ) 
    (h : a ≠ 0)
    (hroots : a * x1^3 + b * x1^2 + c * x1 + d = 0 
      ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 
      ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
    :
    (x1 + x2 + x3 = -b / a) 
    ∧ (x1 * x2 + x1 * x3 + x2 * x3 = c / a) 
    ∧ (x1 * x2 * x3 = -d / a) := 
sorry

end cubic_roots_relations_l125_125078


namespace total_marks_calculation_l125_125865

def average (total_marks : ℕ) (num_candidates : ℕ) : ℕ := total_marks / num_candidates
def total_marks (average : ℕ) (num_candidates : ℕ) : ℕ := average * num_candidates

theorem total_marks_calculation
  (num_candidates : ℕ)
  (average_marks : ℕ)
  (range_min : ℕ)
  (range_max : ℕ)
  (h1 : num_candidates = 250)
  (h2 : average_marks = 42)
  (h3 : range_min = 10)
  (h4 : range_max = 80) :
  total_marks average_marks num_candidates = 10500 :=
by 
  sorry

end total_marks_calculation_l125_125865


namespace correct_proposition_l125_125484

open Real

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > 1

-- Define f and its derivative
def f (x : ℝ) := x - sin x
def f_prime (x : ℝ) := 1 - cos x

-- Define proposition q
def q : Prop := ∀ x : ℝ, 0 ≤ f_prime x

-- The statement of the proof problem
theorem correct_proposition : (¬ p) ∧ q := by
  -- Proof goes here
  sorry

end correct_proposition_l125_125484


namespace average_temperature_l125_125255

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l125_125255


namespace distinct_arrangements_balloon_l125_125836

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125836


namespace least_number_to_add_l125_125117

theorem least_number_to_add (n : ℕ) (h : n = 28523) : 
  ∃ x, x + n = 29560 ∧ 3 ∣ (x + n) ∧ 5 ∣ (x + n) ∧ 7 ∣ (x + n) ∧ 8 ∣ (x + n) :=
by 
  sorry

end least_number_to_add_l125_125117


namespace shift_right_symmetric_l125_125651

open Real

/-- Given the function y = sin(2x + π/3), after shifting the graph of the function right
    by φ (0 < φ < π/2) units, the resulting graph is symmetric about the y-axis.
    Prove that the value of φ is 5π/12.
-/
theorem shift_right_symmetric (φ : ℝ) (hφ₁ : 0 < φ) (hφ₂ : φ < π / 2)
  (h_sym : ∃ k : ℤ, -2 * φ + π / 3 = k * π + π / 2) : φ = 5 * π / 12 :=
sorry

end shift_right_symmetric_l125_125651


namespace least_xy_l125_125483

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : x * y = 108 :=
by
  sorry

end least_xy_l125_125483


namespace output_in_scientific_notation_l125_125576

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l125_125576


namespace half_angle_quadrant_l125_125050

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 / 2 * Real.pi)
  : (k % 2 = 0 → k * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi + 3 / 4 * Real.pi) ∨
    (k % 2 = 1 → (k + 1) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k + 1) * Real.pi + 3 / 4 * Real.pi) :=
by
  sorry

end half_angle_quadrant_l125_125050


namespace claire_earning_l125_125613

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end claire_earning_l125_125613


namespace johns_age_l125_125884

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125884


namespace faye_homework_problems_l125_125004

----- Definitions based on the conditions given -----

def total_math_problems : ℕ := 46
def total_science_problems : ℕ := 9
def problems_finished_at_school : ℕ := 40

----- Theorem statement -----

theorem faye_homework_problems : total_math_problems + total_science_problems - problems_finished_at_school = 15 := by
  -- Sorry is used here to skip the proof
  sorry

end faye_homework_problems_l125_125004


namespace max_value_cos_sin_expr_l125_125171

theorem max_value_cos_sin_expr :
  ∀ θ1 θ2 θ3 θ4 θ5 θ6 θ7 : ℝ,
    (cos θ1 * sin θ2 + cos θ2 * sin θ3 + cos θ3 * sin θ4 +
     cos θ4 * sin θ5 + cos θ5 * sin θ6 + cos θ6 * sin θ7 +
     cos θ7 * sin θ1) ≤ 7 / 2 :=
begin
  sorry
end

end max_value_cos_sin_expr_l125_125171


namespace find_k_value_l125_125181

variables {R : Type*} [Field R] {a b x k : R}

-- Definitions for the conditions in the problem
def f (x : R) (a b : R) : R := (b * x + 1) / (2 * x + a)

-- Statement of the problem
theorem find_k_value (h_ab : a * b ≠ 2)
  (h_k : ∀ (x : R), x ≠ 0 → f x a b * f (x⁻¹) a b = k) :
  k = (1 : R) / 4 :=
by
  sorry

end find_k_value_l125_125181


namespace quadratic_transformed_correct_l125_125627

noncomputable def quadratic_transformed (a b c : ℝ) (r s : ℝ) (h1 : a ≠ 0) 
  (h_roots : r + s = -b / a ∧ r * s = c / a) : Polynomial ℝ :=
Polynomial.C (a * b * c) + Polynomial.C ((-(a + b) * b)) * Polynomial.X + Polynomial.X^2

-- The theorem statement
theorem quadratic_transformed_correct (a b c r s : ℝ) (h1 : a ≠ 0)
  (h_roots : r + s = -b / a ∧ r * s = c / a) :
  (quadratic_transformed a b c r s h1 h_roots).roots = {a * (r + b), a * (s + b)} :=
sorry

end quadratic_transformed_correct_l125_125627


namespace flag_movement_distance_l125_125771

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l125_125771


namespace halfway_between_one_eighth_and_one_third_l125_125091

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 + 1 / 3) / 2 = 11 / 48 :=
by
  -- Skipping the proof here
  sorry

end halfway_between_one_eighth_and_one_third_l125_125091


namespace johns_age_l125_125905

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125905


namespace soda_cans_ratio_l125_125732

theorem soda_cans_ratio
  (initial_cans : ℕ := 22)
  (cans_taken : ℕ := 6)
  (final_cans : ℕ := 24)
  (x : ℚ := 1 / 2)
  (cans_left : ℕ := 16)
  (cans_bought : ℕ := 16 * 1 / 2) :
  (cans_bought / cans_left : ℚ) = 1 / 2 :=
sorry

end soda_cans_ratio_l125_125732


namespace find_total_salary_l125_125428

noncomputable def total_salary (salary_left : ℕ) : ℚ :=
  salary_left * (120 / 19)

theorem find_total_salary
  (food : ℚ) (house_rent : ℚ) (clothes : ℚ) (transport : ℚ) (remaining : ℕ) :
  food = 1 / 4 →
  house_rent = 1 / 8 →
  clothes = 3 / 10 →
  transport = 1 / 6 →
  remaining = 35000 →
  total_salary remaining = 210552.63 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_total_salary_l125_125428


namespace find_x_from_conditions_l125_125789

theorem find_x_from_conditions 
  (x y : ℕ) 
  (h1 : 1 ≤ x)
  (h2 : x ≤ 100)
  (h3 : 1 ≤ y)
  (h4 : y ≤ 100)
  (h5 : y > x)
  (h6 : (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x) 
  : x = 16 := 
sorry

end find_x_from_conditions_l125_125789


namespace sales_on_second_street_l125_125230

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end sales_on_second_street_l125_125230


namespace regular_polygon_sides_l125_125438

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l125_125438


namespace output_in_scientific_notation_l125_125575

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l125_125575


namespace output_in_scientific_notation_l125_125574

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l125_125574


namespace balloon_arrangements_l125_125846

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l125_125846


namespace isosceles_BMC_l125_125239

open EuclideanGeometry

variable {A B C D M O : Point ℝ}

-- Given conditions in the problem
variables (h1 : Trapezoid A B C D)
          (h2 : ∃ (O : Point ℝ), IsDiagonalIntersection h1 A C B D O)
          (h3 : ∃ (circ1 circ2 : Circle ℝ), 
                 CircumscribedTriangle circ1 A O B ∧
                 CircumscribedTriangle circ2 C O D ∧
                 IntersectsAtPoint circ1 circ2 M ∧
                 M ∈ Line.segment A D)

-- The goal to prove
theorem isosceles_BMC : BM = CM :=
by
  cases h2 with O hO,
  cases h3 with circ1 hcirc1,
  cases hcirc1 with circ2 hcirc2,
  sorry

end isosceles_BMC_l125_125239


namespace complex_div_l125_125592

open Complex

theorem complex_div (i : ℂ) (hi : i = Complex.I) : 
  (6 + 7 * i) / (1 + 2 * i) = 4 - i := 
by 
  sorry

end complex_div_l125_125592


namespace difference_shares_l125_125447

-- Given conditions in the problem
variable (V : ℕ) (F R : ℕ)
variable (hV : V = 1500)
variable (hRatioF : F = 3 * (V / 5))
variable (hRatioR : R = 11 * (V / 5))

-- The statement we need to prove
theorem difference_shares : R - F = 2400 :=
by
  -- Using the conditions to derive the result.
  sorry

end difference_shares_l125_125447


namespace smallest_integer_k_condition_l125_125280

theorem smallest_integer_k_condition :
  ∃ k : ℤ, k > 1 ∧ k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1 ∧ k = 61 :=
by
  sorry

end smallest_integer_k_condition_l125_125280


namespace actual_height_of_boy_l125_125237

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end actual_height_of_boy_l125_125237


namespace population_increase_rate_l125_125247

theorem population_increase_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) 
  (h1 : persons = 240) 
  (h2 : minutes = 60) 
  (h3 : seconds_per_person = (minutes * 60) / persons) 
  : seconds_per_person = 15 :=
by 
  sorry

end population_increase_rate_l125_125247


namespace smallest_class_size_l125_125369

theorem smallest_class_size (N : ℕ) (G : ℕ) (h1: 0.25 < (G : ℝ) / N) (h2: (G : ℝ) / N < 0.30) : N = 7 := 
sorry

end smallest_class_size_l125_125369


namespace cars_in_parking_lot_l125_125729

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cars_in_parking_lot_l125_125729


namespace balloon_arrangements_l125_125845

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l125_125845


namespace actual_distance_traveled_l125_125361

theorem actual_distance_traveled
  (t : ℕ)
  (H1 : 6 * t = 3 * t + 15) :
  3 * t = 15 :=
by
  exact sorry

end actual_distance_traveled_l125_125361


namespace scientific_notation_of_448000_l125_125572

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l125_125572


namespace find_difference_l125_125450

noncomputable def g : ℝ → ℝ := sorry    -- Definition of the function g (since it's graph-based and specific)

-- Given conditions
variables (c d : ℝ)
axiom h1 : Function.Injective g          -- g is an invertible function (injective functions have inverses)
axiom h2 : g c = d
axiom h3 : g d = 6

-- Theorem to prove
theorem find_difference : c - d = -2 :=
by {
  -- sorry is needed since the exact proof steps are not provided
  sorry
}

end find_difference_l125_125450


namespace different_genre_pairs_count_l125_125048

theorem different_genre_pairs_count 
  (mystery_books : Finset ℕ)
  (fantasy_books : Finset ℕ)
  (biographies : Finset ℕ)
  (h1 : mystery_books.card = 4)
  (h2 : fantasy_books.card = 4)
  (h3 : biographies.card = 4) :
  (mystery_books.product (fantasy_books ∪ biographies)).card +
  (fantasy_books.product (mystery_books ∪ biographies)).card +
  (biographies.product (mystery_books ∪ fantasy_books)).card = 48 := 
sorry

end different_genre_pairs_count_l125_125048


namespace area_of_right_triangle_ABC_l125_125503

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def area_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) (hAB : AB = 30) (hBC : BC = 40) : ℝ :=
  1 / 2 * AB * BC

theorem area_of_right_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) 
  (hAB : AB = 30) (hBC : BC = 40) : 
  area_triangle_ABC AB BC angleB hangle hAB hBC = 600 :=
by
  sorry

end area_of_right_triangle_ABC_l125_125503


namespace count_of_squares_difference_l125_125042

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l125_125042


namespace recommended_apps_l125_125176

namespace RogerPhone

-- Let's define the conditions.
def optimalApps : ℕ := 50
def currentApps (R : ℕ) : ℕ := 2 * R
def appsToDelete : ℕ := 20

-- Defining the problem as a theorem.
theorem recommended_apps (R : ℕ) (h1 : 2 * R = optimalApps + appsToDelete) : R = 35 := by
  sorry

end RogerPhone

end recommended_apps_l125_125176


namespace friends_gcd_l125_125752

theorem friends_gcd {a b : ℤ} (h : ∃ n : ℤ, a * b = n * n) : 
  ∃ m : ℤ, a * Int.gcd a b = m * m :=
sorry

end friends_gcd_l125_125752


namespace parking_lot_cars_l125_125727

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end parking_lot_cars_l125_125727


namespace boy_running_time_l125_125040

theorem boy_running_time (s : ℝ) (v : ℝ) (h1 : s = 35) (h2 : v = 9) : 
  (4 * s) / (v * 1000 / 3600) = 56 := by
  sorry

end boy_running_time_l125_125040


namespace diff_of_squares_1500_l125_125043

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l125_125043


namespace total_distance_covered_l125_125970

noncomputable def speed_train_a : ℚ := 80          -- Speed of Train A in kmph
noncomputable def speed_train_b : ℚ := 110         -- Speed of Train B in kmph
noncomputable def duration : ℚ := 15               -- Duration in minutes
noncomputable def conversion_factor : ℚ := 60      -- Conversion factor from hours to minutes

theorem total_distance_covered : 
    (speed_train_a / conversion_factor) * duration + 
    (speed_train_b / conversion_factor) * duration = 47.5 :=
by
  sorry

end total_distance_covered_l125_125970


namespace planting_ways_l125_125136

namespace FarmerField

/-- The types of crops available for planting -/
inductive Crop
| corn | wheat | soybeans | potatoes | rice

/-- A 3x3 grid represented as a 9-element set -/
def Field := Fin 3 × Fin 3

structure PlantingConfiguration :=
(sections : Field → Crop)
(no_adjacent_corn_soybeans : ∀ {i j : Field}, i ≠ j → sections i = Crop.corn → sections j ≠ Crop.soybeans)
(no_adjacent_wheat_potatoes : ∀ {i j : Field}, i ≠ j → sections i = Crop.wheat → sections j ≠ Crop.potatoes)

noncomputable def count_valid_configurations : ℕ :=
sorry

theorem planting_ways : count_valid_configurations = 2045 :=
by
  -- We skip the proof here
  sorry

end FarmerField

end planting_ways_l125_125136


namespace shaded_area_correct_l125_125376

noncomputable def shaded_area (s r_small : ℝ) : ℝ :=
  let hex_area := (3 * Real.sqrt 3 / 2) * s^2
  let semi_area := 6 * (1/2 * Real.pi * (s/2)^2)
  let small_circle_area := 6 * (Real.pi * (r_small)^2)
  hex_area - (semi_area + small_circle_area)

theorem shaded_area_correct : shaded_area 4 0.5 = 24 * Real.sqrt 3 - (27 * Real.pi / 2) := by
  sorry

end shaded_area_correct_l125_125376


namespace gcd_176_88_l125_125949

theorem gcd_176_88 : Nat.gcd 176 88 = 88 :=
by
  sorry

end gcd_176_88_l125_125949


namespace possible_second_game_scores_count_l125_125269

theorem possible_second_game_scores_count :
  ∃ (A1 A3 B2 : ℕ),
  (A1 + A3 = 22) ∧ (B2 = 11) ∧ (A1 < 11) ∧ (A3 < 11) ∧ ((B2 - A2 = 2) ∨ (B2 >= A2 + 2)) ∧ (A1 + B1 + A2 + B2 + A3 + B3 = 62) :=
  sorry

end possible_second_game_scores_count_l125_125269


namespace average_temperature_l125_125257

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l125_125257


namespace find_c_of_triangle_area_l125_125688

-- Define the problem in Lean 4 statement.
theorem find_c_of_triangle_area (A : ℝ) (b c : ℝ) (area : ℝ)
  (hA : A = 60 * Real.pi / 180)  -- Converting degrees to radians
  (hb : b = 1)
  (hArea : area = Real.sqrt 3) :
  c = 4 :=
by 
  -- Lean proof goes here (we include sorry to skip)
  sorry

end find_c_of_triangle_area_l125_125688


namespace max_capacity_per_car_l125_125958

-- Conditions
def num_cars : ℕ := 2
def num_vans : ℕ := 3
def people_per_car : ℕ := 5
def people_per_van : ℕ := 3
def max_people_per_van : ℕ := 8
def additional_people : ℕ := 17

-- Theorem to prove maximum capacity of each car is 6 people
theorem max_capacity_per_car (num_cars num_vans people_per_car people_per_van max_people_per_van additional_people : ℕ) : 
  (num_cars = 2 ∧ num_vans = 3 ∧ people_per_car = 5 ∧ people_per_van = 3 ∧ max_people_per_van = 8 ∧ additional_people = 17) →
  ∃ max_people_per_car, max_people_per_car = 6 :=
by
  sorry

end max_capacity_per_car_l125_125958


namespace number_of_sodas_l125_125971

theorem number_of_sodas (cost_sandwich : ℝ) (num_sandwiches : ℕ) (cost_soda : ℝ) (total_cost : ℝ):
  cost_sandwich = 2.45 → 
  num_sandwiches = 2 → 
  cost_soda = 0.87 → 
  total_cost = 8.38 → 
  (total_cost - num_sandwiches * cost_sandwich) / cost_soda = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end number_of_sodas_l125_125971


namespace average_temperature_l125_125256

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l125_125256


namespace balloon_arrangements_l125_125828

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l125_125828


namespace remainder_of_76_pow_k_mod_7_is_6_l125_125279

theorem remainder_of_76_pow_k_mod_7_is_6 (k : ℕ) (hk : k % 2 = 1) : (76 ^ k) % 7 = 6 :=
sorry

end remainder_of_76_pow_k_mod_7_is_6_l125_125279


namespace dart_not_land_in_circle_probability_l125_125296

theorem dart_not_land_in_circle_probability :
  let side_length := 1
  let radius := side_length / 2
  let area_square := side_length * side_length
  let area_circle := π * radius * radius
  let prob_inside_circle := area_circle / area_square
  let prob_outside_circle := 1 - prob_inside_circle
  prob_outside_circle = 1 - (π / 4) :=
by
  sorry

end dart_not_land_in_circle_probability_l125_125296


namespace barber_loss_is_25_l125_125742

-- Definition of conditions
structure BarberScenario where
  haircut_cost : ℕ
  counterfeit_bill : ℕ
  real_change : ℕ
  change_given : ℕ
  real_bill_given : ℕ

def barberScenario_example : BarberScenario :=
  { haircut_cost := 15,
    counterfeit_bill := 20,
    real_change := 20,
    change_given := 5,
    real_bill_given := 20 }

-- Lean 4 problem statement
theorem barber_loss_is_25 (b : BarberScenario) : 
  b.haircut_cost = 15 ∧
  b.counterfeit_bill = 20 ∧
  b.real_change = 20 ∧
  b.change_given = 5 ∧
  b.real_bill_given = 20 → (15 + 5 + 20 - 20 + 5 = 25) :=
by
  intro h
  cases' h with h1 h23
  sorry

end barber_loss_is_25_l125_125742


namespace triangle_inequality_l125_125931

theorem triangle_inequality
  (R r p : ℝ) (a b c : ℝ)
  (h1 : a * b + b * c + c * a = r^2 + p^2 + 4 * R * r)
  (h2 : 16 * R * r - 5 * r^2 ≤ p^2)
  (h3 : p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2):
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := 
  by
    sorry

end triangle_inequality_l125_125931


namespace Ravi_jumps_39_inches_l125_125080

-- Define the heights of the next three highest jumpers
def h₁ : ℝ := 23
def h₂ : ℝ := 27
def h₃ : ℝ := 28

-- Define the average height of the three jumpers
def average_height : ℝ := (h₁ + h₂ + h₃) / 3

-- Define Ravi's jump height
def Ravi_jump_height : ℝ := 1.5 * average_height

-- The theorem to prove
theorem Ravi_jumps_39_inches : Ravi_jump_height = 39 := by
  sorry
 
end Ravi_jumps_39_inches_l125_125080


namespace polygon_is_decagon_l125_125488

-- Definitions based on conditions
def exterior_angles_sum (x : ℕ) : ℝ := 360

def interior_angles_sum (x : ℕ) : ℝ := 4 * exterior_angles_sum x

def interior_sum_formula (n : ℕ) : ℝ := (n - 2) * 180

-- Mathematically equivalent proof problem
theorem polygon_is_decagon (n : ℕ) (h1 : exterior_angles_sum n = 360)
  (h2 : interior_angles_sum n = 4 * exterior_angles_sum n)
  (h3 : interior_sum_formula n = interior_angles_sum n) : n = 10 :=
sorry

end polygon_is_decagon_l125_125488


namespace sum_due_is_l125_125089

-- Definitions and conditions from the problem
def BD : ℤ := 288
def TD : ℤ := 240
def face_value (FV : ℤ) : Prop := BD = TD + (TD * TD) / FV

-- Proof statement
theorem sum_due_is (FV : ℤ) (h : face_value FV) : FV = 1200 :=
sorry

end sum_due_is_l125_125089


namespace toms_total_cost_l125_125966

theorem toms_total_cost :
  let costA := 4 * 15
  let costB := 3 * 12
  let discountB := 0.20 * costB
  let costBDiscounted := costB - discountB
  let costC := 2 * 18
  costA + costBDiscounted + costC = 124.80 := 
by
  sorry

end toms_total_cost_l125_125966


namespace distance_between_circle_centers_l125_125031

theorem distance_between_circle_centers
  (R r d : ℝ)
  (h1 : R = 7)
  (h2 : r = 4)
  (h3 : d = 5 + 1)
  (h_total_diameter : 5 + 8 + 1 = 14)
  (h_radius_R : R = 14 / 2)
  (h_radius_r : r = 8 / 2) : d = 6 := 
by sorry

end distance_between_circle_centers_l125_125031


namespace lineup_combinations_l125_125705

open Finset

-- Define the given conditions
def soccer_team : Finset ℕ := range 16
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the proof statement
theorem lineup_combinations : 
  (∑ (k : ℕ) in range 3,
    (if k = 2 then 6 * (soccer_team \ quadruplets).choose (5)
     else if k = 1 then 4 * (soccer_team \ quadruplets).choose (6)
     else (soccer_team \ quadruplets).choose (7))) = 9240 :=
by
  -- Sum the combinations for up to 2 quadruplets in the lineup
  sorry

end lineup_combinations_l125_125705


namespace hiking_hours_l125_125273

-- Define the given conditions
def water_needed_violet_per_hour : ℕ := 800
def water_needed_dog_per_hour : ℕ := 400
def total_water_carry_capacity_liters : ℕ := 4.8 * 1000 -- converted to ml

-- Define the statement to prove
theorem hiking_hours : (total_water_carry_capacity_liters / (water_needed_violet_per_hour + water_needed_dog_per_hour)) = 4 := by
  sorry

end hiking_hours_l125_125273


namespace distinct_arrangements_balloon_l125_125835

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125835


namespace johns_age_l125_125903

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125903


namespace combined_area_difference_l125_125656

theorem combined_area_difference :
  let area_11x11 := 2 * (11 * 11)
  let area_5_5x11 := 2 * (5.5 * 11)
  area_11x11 - area_5_5x11 = 121 :=
by
  sorry

end combined_area_difference_l125_125656


namespace division_result_l125_125737

theorem division_result (x : ℕ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end division_result_l125_125737


namespace prob_all_one_l125_125975

-- Define the probability function for a single die landing on a specific number
def prob_of_die_landing_on (n : ℕ) : ℚ :=
  if n ∈ {1, 2, 3, 4, 5, 6} then 1 / 6 else 0

-- Define the event that four dice landing on specific numbers
def prob_of_fours_dice_landing_on (a b c d : ℕ) : ℚ :=
  prob_of_die_landing_on a * prob_of_die_landing_on b * prob_of_die_landing_on c * prob_of_die_landing_on d

-- Define the theorem we want to prove
theorem prob_all_one : prob_of_fours_dice_landing_on 1 1 1 1 = 1 / 1296 :=
by sorry

end prob_all_one_l125_125975


namespace find_fraction_squares_l125_125660

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end find_fraction_squares_l125_125660


namespace balloon_arrangements_l125_125842

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l125_125842


namespace night_crew_worker_fraction_l125_125995

noncomputable def box_fraction_day : ℝ := 5/7

theorem night_crew_worker_fraction
  (D N : ℝ) -- Number of workers in day and night crew
  (B : ℝ)  -- Number of boxes each worker in the day crew loads
  (H1 : ∀ day_boxes_loaded : ℝ, day_boxes_loaded = D * B)
  (H2 : ∀ night_boxes_loaded : ℝ, night_boxes_loaded = N * (B / 2))
  (H3 : (D * B) / ((D * B) + (N * (B / 2))) = box_fraction_day) :
  N / D = 4/5 := 
sorry

end night_crew_worker_fraction_l125_125995


namespace johns_age_l125_125889

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l125_125889


namespace prime_1002_n_count_l125_125476

theorem prime_1002_n_count :
  ∃! n : ℕ, n ≥ 2 ∧ Prime (n^3 + 2) :=
by
  sorry

end prime_1002_n_count_l125_125476


namespace count_diff_of_squares_l125_125047

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l125_125047


namespace consecutive_odd_product_l125_125704

theorem consecutive_odd_product (n : ℤ) :
  (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by sorry

end consecutive_odd_product_l125_125704


namespace total_flag_distance_moved_l125_125770

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l125_125770


namespace distinct_arrangements_balloon_l125_125811

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125811


namespace cubic_root_sum_eq_constant_term_divided_l125_125458

theorem cubic_root_sum_eq_constant_term_divided 
  (a b c : ℝ) 
  (h_roots : (24 * a^3 - 36 * a^2 + 14 * a - 1 = 0) 
           ∧ (24 * b^3 - 36 * b^2 + 14 * b - 1 = 0) 
           ∧ (24 * c^3 - 36 * c^2 + 14 * c - 1 = 0))
  (h_bounds : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) 
  : (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = (158 / 73) := 
sorry

end cubic_root_sum_eq_constant_term_divided_l125_125458


namespace trigonometric_inequalities_l125_125643

noncomputable def a : ℝ := Real.sin (21 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (72 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (23 * Real.pi / 180)

-- The proof statement
theorem trigonometric_inequalities : c > a ∧ a > b :=
by
  sorry

end trigonometric_inequalities_l125_125643


namespace simplify_fraction_subtraction_l125_125081

theorem simplify_fraction_subtraction : (7 / 3) - (5 / 6) = 3 / 2 := by
  sorry

end simplify_fraction_subtraction_l125_125081


namespace floor_sqrt_12_squared_l125_125166

theorem floor_sqrt_12_squared : (Int.floor (Real.sqrt 12))^2 = 9 := by
  sorry

end floor_sqrt_12_squared_l125_125166


namespace positive_divisors_8_fact_l125_125654

-- Factorial function definition
def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Function to compute the number of divisors from prime factors
def numDivisors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

-- Known prime factorization of 8!
noncomputable def factors_8_fact : List (Nat × Nat) :=
  [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Theorem statement
theorem positive_divisors_8_fact : numDivisors factors_8_fact = 96 :=
  sorry

end positive_divisors_8_fact_l125_125654


namespace solve_system_eq_l125_125936

theorem solve_system_eq (x y : ℝ) :
  x^2 + y^2 + 6 * x * y = 68 ∧ 2 * x^2 + 2 * y^2 - 3 * x * y = 16 ↔
  (x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end solve_system_eq_l125_125936


namespace remainder_when_7645_divided_by_9_l125_125735

/--
  Prove that the remainder when 7645 is divided by 9 is 4,
  given that a number is congruent to the sum of its digits modulo 9.
-/
theorem remainder_when_7645_divided_by_9 :
  7645 % 9 = 4 :=
by
  -- Main proof should go here
  sorry

end remainder_when_7645_divided_by_9_l125_125735


namespace frustum_lateral_surface_area_l125_125137

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) (hh : h = 5) :
  let d := r1 - r2
  let s := Real.sqrt (h^2 + d^2)
  let A := Real.pi * s * (r1 + r2)
  A = 12 * Real.pi * Real.sqrt 41 :=
by
  -- hr1 and hr2 imply that r1 and r2 are constants, therefore d = 8 - 4 = 4
  -- h = 5 and d = 4 imply s = sqrt (5^2 + 4^2) = sqrt 41
  -- The area A is then pi * sqrt 41 * (8 + 4) = 12 * pi * sqrt 41
  sorry

end frustum_lateral_surface_area_l125_125137


namespace coloring_ways_10x10_board_l125_125346

-- Define the \(10 \times 10\) board size
def size : ℕ := 10

-- Define colors as an inductive type
inductive color
| blue
| green

-- Assume h1: each 2x2 square has 2 blue and 2 green cells
def each_2x2_square_valid (board : ℕ × ℕ → color) : Prop :=
∀ i j, i < size - 1 → j < size - 1 →
  (∃ (c1 c2 c3 c4 : color),
    board (i, j) = c1 ∧
    board (i+1, j) = c2 ∧
    board (i, j+1) = c3 ∧
    board (i+1, j+1) = c4 ∧
    [c1, c2, c3, c4].count (λ x, x = color.blue) = 2 ∧
    [c1, c2, c3, c4].count (λ x, x = color.green) = 2)

-- The theorem we want to prove
theorem coloring_ways_10x10_board :
  ∃ (board : ℕ × ℕ → color), each_2x2_square_valid board ∧ (∃ n : ℕ, n = 2046) :=
sorry

end coloring_ways_10x10_board_l125_125346


namespace fraction_of_shaded_area_is_11_by_12_l125_125708

noncomputable def shaded_fraction_of_square : ℚ :=
  let s : ℚ := 1 -- Assume the side length of the square is 1 for simplicity.
  let P := (0, s / 2)
  let Q := (s / 3, s)
  let V := (0, s)
  let base := s / 2
  let height := s / 3
  let triangle_area := (1 / 2) * base * height
  let square_area := s * s
  let shaded_area := square_area - triangle_area
  shaded_area / square_area

theorem fraction_of_shaded_area_is_11_by_12 : shaded_fraction_of_square = 11 / 12 :=
  sorry

end fraction_of_shaded_area_is_11_by_12_l125_125708


namespace geometric_sequence_first_term_l125_125405

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 18) 
  (h2 : a * r^4 = 1458) : 
  a = 6 := 
by 
  sorry

end geometric_sequence_first_term_l125_125405


namespace probability_equals_two_thirds_l125_125349

-- Definitions for total arrangements and favorable arrangements
def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 2

-- Probability that 2 zeros are not adjacent
def probability_not_adjacent : ℚ := favorable_arrangements / total_arrangements

theorem probability_equals_two_thirds : probability_not_adjacent = 2 / 3 := 
by 
  let total_arrangements := 15
  let favorable_arrangements := 10
  have h1 : probability_not_adjacent = (10 : ℚ) / (15 : ℚ) := rfl
  have h2 : (10 : ℚ) / (15 : ℚ) = 2 / 3 := by norm_num
  exact Eq.trans h1 h2 

end probability_equals_two_thirds_l125_125349


namespace hiking_hours_l125_125271

def violet_water_per_hour : ℕ := 800 -- Violet's water need per hour in ml
def dog_water_per_hour : ℕ := 400    -- Dog's water need per hour in ml
def total_water_capacity : ℚ := 4.8  -- Total water capacity Violet can carry in L

theorem hiking_hours :
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000 in
  total_water_capacity / total_water_per_hour = 4 :=
by
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000
  have h1 : violet_water_per_hour = 800 := rfl
  have h2 : dog_water_per_hour = 400 := rfl
  have h3 : total_water_capacity = 4.8 := rfl
  have h4 : total_water_per_hour = 1.2 := by simp [violet_water_per_hour, dog_water_per_hour]
  have h5 : total_water_capacity / total_water_per_hour = 4 := by simp [total_water_capacity, total_water_per_hour]
  exact h5

end hiking_hours_l125_125271


namespace angle_A_is_60_degrees_triangle_area_l125_125023

-- Define the basic setup for the triangle and its angles
variables (a b c : ℝ) -- internal angles of the triangle ABC
variables (B C : ℝ) -- sides opposite to angles b and c respectively

-- Given conditions
axiom equation_1 : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a
axiom perimeter_condition : a + b + c = 8
axiom circumradius_condition : ∃ R : ℝ, R = Real.sqrt 3

-- Question 1: Prove the measure of angle A is 60 degrees
theorem angle_A_is_60_degrees (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a) : 
  a = 60 :=
sorry

-- Question 2: Prove the area of triangle ABC
theorem triangle_area (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a)
(h_perimeter : a + b + c = 8) (h_circumradius : ∃ R : ℝ, R = Real.sqrt 3) :
  ∃ S : ℝ, S = 4 * Real.sqrt 3 / 3 :=
sorry

end angle_A_is_60_degrees_triangle_area_l125_125023


namespace dyed_pink_correct_l125_125965

def silk_dyed_green := 61921
def total_yards_dyed := 111421
def yards_dyed_pink := total_yards_dyed - silk_dyed_green

theorem dyed_pink_correct : yards_dyed_pink = 49500 := by 
  sorry

end dyed_pink_correct_l125_125965


namespace remainder_of_sum_of_squares_mod_8_l125_125411

theorem remainder_of_sum_of_squares_mod_8 :
  let a := 445876
  let b := 985420
  let c := 215546
  let d := 656452
  let e := 387295
  a % 8 = 4 → b % 8 = 4 → c % 8 = 6 → d % 8 = 4 → e % 8 = 7 →
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remainder_of_sum_of_squares_mod_8_l125_125411


namespace art_class_students_not_in_science_l125_125448

theorem art_class_students_not_in_science (n S A S_inter_A_only_A : ℕ) 
  (h_n : n = 120) 
  (h_S : S = 85) 
  (h_A : A = 65) 
  (h_union: n = S + A - S_inter_A_only_A) : 
  S_inter_A_only_A = 30 → 
  A - S_inter_A_only_A = 35 :=
by
  intros h
  rw [h]
  sorry

end art_class_students_not_in_science_l125_125448


namespace find_c_l125_125650

noncomputable def f (x a b c : ℤ) := x^3 + a * x^2 + b * x + c

theorem find_c (a b c : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : f a a b c = a^3) (h₄ : f b a b c = b^3) : c = 16 :=
sorry

end find_c_l125_125650


namespace problem_solution_l125_125564

theorem problem_solution (a b : ℕ) (x : ℝ) (h1 : x^2 + 14 * x = 24) (h2 : x = Real.sqrt a - b) (h3 : a > 0) (h4 : b > 0) :
  a + b = 80 := 
sorry

end problem_solution_l125_125564


namespace final_inventory_is_correct_l125_125317

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct_l125_125317


namespace vertex_of_parabola_l125_125562

theorem vertex_of_parabola (x : ℝ) : 
  ∀ x y : ℝ, (y = x^2 - 6 * x + 1) → (∃ h k : ℝ, y = (x - h)^2 + k ∧ h = 3 ∧ k = -8) :=
by
  -- This is to state that given the parabola equation x^2 - 6x + 1, its vertex coordinates are (3, -8).
  sorry

end vertex_of_parabola_l125_125562


namespace final_inventory_is_correct_l125_125316

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end final_inventory_is_correct_l125_125316


namespace value_of_x_l125_125557

theorem value_of_x : 
  let x := (sqrt (7^2 + 24^2)) / (sqrt (49 + 16)) 
  in x = 25 * sqrt 65 / 65 
  := 
  sorry

end value_of_x_l125_125557


namespace balloon_arrangements_l125_125849

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l125_125849


namespace sequence_value_2023_l125_125481

theorem sequence_value_2023 (a : ℕ → ℕ) (h₁ : a 1 = 3)
  (h₂ : ∀ m n : ℕ, a (m + n) = a m + a n) : a 2023 = 6069 := by
  sorry

end sequence_value_2023_l125_125481


namespace problem_statement_l125_125659

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end problem_statement_l125_125659


namespace part_1_part_2_l125_125478

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (1 + 2^x)

theorem part_1 (a : ℝ) (h1 : f a 1 + f a (-1) = 0) : a = 1 / 2 :=
by sorry

theorem part_2 : ∃ a : ℝ, ∀ x : ℝ, f a (-x) + f a x = 0 :=
by sorry

end part_1_part_2_l125_125478


namespace scientific_notation_of_448000_l125_125571

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l125_125571


namespace percentage_of_first_to_second_l125_125860

theorem percentage_of_first_to_second (X : ℝ) (h1 : first = (7/100) * X) (h2 : second = (14/100) * X) : (first / second) * 100 = 50 := 
by
  sorry

end percentage_of_first_to_second_l125_125860


namespace probability_of_choosing_two_yellow_apples_l125_125223

theorem probability_of_choosing_two_yellow_apples :
  let total_apples := 10
  let red_apples := 6
  let yellow_apples := 4
  let total_ways_to_choose_2 := (total_apples.choose 2)
  let ways_to_choose_2_yellow := (yellow_apples.choose 2)
  (ways_to_choose_2_yellow : ℚ) / total_ways_to_choose_2 = 2 / 15 := by
sorry

end probability_of_choosing_two_yellow_apples_l125_125223


namespace strawberries_taken_out_l125_125455

theorem strawberries_taken_out : 
  ∀ (initial_total_strawberries buckets strawberries_left_per_bucket : ℕ),
  initial_total_strawberries = 300 → 
  buckets = 5 → 
  strawberries_left_per_bucket = 40 → 
  (initial_total_strawberries / buckets - strawberries_left_per_bucket = 20) :=
by
  intros initial_total_strawberries buckets strawberries_left_per_bucket h1 h2 h3
  sorry

end strawberries_taken_out_l125_125455


namespace odd_function_inequality_solution_l125_125180

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x - 2 else -(x - 2)

theorem odd_function_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by
  -- A placeholder for the actual proof
  sorry

end odd_function_inequality_solution_l125_125180


namespace pastries_sold_is_correct_l125_125996

-- Definitions of the conditions
def initial_pastries : ℕ := 56
def remaining_pastries : ℕ := 27

-- Statement of the theorem
theorem pastries_sold_is_correct : initial_pastries - remaining_pastries = 29 :=
by
  sorry

end pastries_sold_is_correct_l125_125996


namespace original_number_of_students_l125_125747

theorem original_number_of_students (x : ℕ)
  (h1: 40 * x / x = 40)
  (h2: 12 * 34 = 408)
  (h3: (40 * x + 408) / (x + 12) = 36) : x = 6 :=
by
  sorry

end original_number_of_students_l125_125747


namespace matching_charge_and_minutes_l125_125911

def charge_at_time (x : ℕ) : ℕ :=
  100 - x / 6

def minutes_past_midnight (x : ℕ) : ℕ :=
  x % 60

theorem matching_charge_and_minutes :
  ∃ x, (x = 292 ∨ x = 343 ∨ x = 395 ∨ x = 446 ∨ x = 549) ∧ 
       charge_at_time x = minutes_past_midnight x :=
by {
  sorry
}

end matching_charge_and_minutes_l125_125911


namespace gcd_24_36_54_l125_125244

-- Define the numbers and the gcd function
def num1 : ℕ := 24
def num2 : ℕ := 36
def num3 : ℕ := 54

-- The Lean statement to prove that the gcd of num1, num2, and num3 is 6
theorem gcd_24_36_54 : Nat.gcd (Nat.gcd num1 num2) num3 = 6 := by
  sorry

end gcd_24_36_54_l125_125244


namespace distinct_arrangements_balloon_l125_125814

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125814


namespace managers_in_sample_l125_125597

-- Definitions based on the conditions
def total_employees : ℕ := 160
def number_salespeople : ℕ := 104
def number_managers : ℕ := 32
def number_logistics : ℕ := 24
def sample_size : ℕ := 20

-- Theorem statement
theorem managers_in_sample : (number_managers * sample_size) / total_employees = 4 := by
  -- Proof omitted, as per the instructions
  sorry

end managers_in_sample_l125_125597


namespace intersection_at_y_axis_l125_125502

theorem intersection_at_y_axis : ∃ y, (y = 5 * 0 + 1) ∧ (0, y) = (0, 1) :=
begin
  use 1,
  split,
  { norm_num, },
  { refl, },
end

end intersection_at_y_axis_l125_125502


namespace min_value_a4b3c2_l125_125215

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l125_125215


namespace prime_pattern_l125_125512

theorem prime_pattern (n x : ℕ) (h1 : x = (10^n - 1) / 9) (h2 : Prime x) : Prime n :=
sorry

end prime_pattern_l125_125512


namespace max_sum_unit_hexagons_l125_125067

theorem max_sum_unit_hexagons (k : ℕ) (hk : k ≥ 3) : 
  ∃ S, S = 6 + (3 * k - 9) * k * (k + 1) / 2 + (3 * (k^2 - 2)) * (k * (k + 1) * (2 * k + 1) / 6) / 6 ∧
       S = 3 * (k * k - 14 * k + 33 * k - 28) / 2 :=
by
  sorry

end max_sum_unit_hexagons_l125_125067


namespace regular_polygon_sides_l125_125442

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l125_125442


namespace emily_weight_l125_125036

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end emily_weight_l125_125036


namespace average_attendance_percentage_l125_125725

theorem average_attendance_percentage :
  let total_laborers := 300
  let day1_present := 150
  let day2_present := 225
  let day3_present := 180
  let day1_percentage := (day1_present / total_laborers) * 100
  let day2_percentage := (day2_present / total_laborers) * 100
  let day3_percentage := (day3_present / total_laborers) * 100
  let average_percentage := (day1_percentage + day2_percentage + day3_percentage) / 3
  average_percentage = 61.7 := by
  sorry

end average_attendance_percentage_l125_125725


namespace quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l125_125802

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := 2*k - 1
  let c := -k - 1
  discriminant a b c > 0 := by
  sorry

theorem determine_k_from_roots_relation (x1 x2 k : ℝ) 
  (h1 : x1 + x2 = -(2*k - 1))
  (h2 : x1 * x2 = -k - 1)
  (h3 : x1 + x2 - 4*(x1 * x2) = 2) :
  k = -3/2 := by
  sorry

end quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l125_125802


namespace cheenu_time_difference_l125_125155

def cheenu_bike_time_per_mile (distance_bike : ℕ) (time_bike : ℕ) : ℕ := time_bike / distance_bike
def cheenu_walk_time_per_mile (distance_walk : ℕ) (time_walk : ℕ) : ℕ := time_walk / distance_walk
def time_difference (time1 : ℕ) (time2 : ℕ) : ℕ := time2 - time1

theorem cheenu_time_difference 
  (distance_bike : ℕ) (time_bike : ℕ) 
  (distance_walk : ℕ) (time_walk : ℕ) 
  (H_bike : distance_bike = 20) (H_time_bike : time_bike = 80) 
  (H_walk : distance_walk = 8) (H_time_walk : time_walk = 160) :
  time_difference (cheenu_bike_time_per_mile distance_bike time_bike) (cheenu_walk_time_per_mile distance_walk time_walk) = 16 := 
by
  sorry

end cheenu_time_difference_l125_125155


namespace abs_ab_eq_2_sqrt_65_l125_125948

theorem abs_ab_eq_2_sqrt_65
  (a b : ℝ)
  (h1 : b^2 - a^2 = 16)
  (h2 : a^2 + b^2 = 36) :
  |a * b| = 2 * Real.sqrt 65 := 
sorry

end abs_ab_eq_2_sqrt_65_l125_125948


namespace number_of_sets_X_l125_125951

noncomputable def finite_set_problem (M A B : Finset ℕ) : Prop :=
  (M.card = 10) ∧ 
  (A ⊆ M) ∧ 
  (B ⊆ M) ∧ 
  (A ∩ B = ∅) ∧ 
  (A.card = 2) ∧ 
  (B.card = 3) ∧ 
  (∃ (X : Finset ℕ), X ⊆ M ∧ ¬(A ⊆ X) ∧ ¬(B ⊆ X))

theorem number_of_sets_X (M A B : Finset ℕ) (h : finite_set_problem M A B) : 
  ∃ n : ℕ, n = 672 := 
sorry

end number_of_sets_X_l125_125951


namespace johns_age_l125_125888

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l125_125888


namespace problem_statement_l125_125160

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end problem_statement_l125_125160


namespace odd_function_behavior_l125_125673

theorem odd_function_behavior (f : ℝ → ℝ)
  (h_odd: ∀ x, f (-x) = -f x)
  (h_increasing: ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_max: ∀ x, 3 ≤ x → x ≤ 7 → f x ≤ 5) :
  (∀ x, -7 ≤ x → x ≤ -3 → f x ≥ -5) ∧ (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) :=
sorry

end odd_function_behavior_l125_125673


namespace chessboard_property_exists_l125_125457

theorem chessboard_property_exists (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ i j, x i j = t i - t j := 
sorry

end chessboard_property_exists_l125_125457


namespace max_two_digit_times_max_one_digit_is_three_digit_l125_125720

def max_two_digit : ℕ := 99
def max_one_digit : ℕ := 9
def product := max_two_digit * max_one_digit

theorem max_two_digit_times_max_one_digit_is_three_digit :
  100 ≤ product ∧ product < 1000 :=
by
  -- Prove that the product is a three-digit number
  sorry

end max_two_digit_times_max_one_digit_is_three_digit_l125_125720


namespace expression_meaningful_if_not_three_l125_125054

-- Definition of meaningful expression
def meaningful_expr (x : ℝ) : Prop := (x ≠ 3)

theorem expression_meaningful_if_not_three (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ meaningful_expr x := by
  sorry

end expression_meaningful_if_not_three_l125_125054


namespace largest_number_4597_l125_125980

def swap_adjacent_digits (n : ℕ) : ℕ :=
  sorry

def max_number_after_two_swaps_subtract_100 (n : ℕ) : ℕ :=
  -- logic to perform up to two adjacent digit swaps and subtract 100
  sorry

theorem largest_number_4597 : max_number_after_two_swaps_subtract_100 4597 = 4659 :=
  sorry

end largest_number_4597_l125_125980


namespace least_positive_n_l125_125642

theorem least_positive_n : ∃ n : ℕ, (1 / (n : ℝ) - 1 / (n + 1 : ℝ) < 1 / 12) ∧ (∀ m : ℕ, (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 12) → n ≤ m) :=
by {
  sorry
}

end least_positive_n_l125_125642


namespace range_of_k_l125_125190

noncomputable def point_satisfies_curve (a k : ℝ) : Prop :=
(-a)^2 - a * (-a) + 2 * a + k = 0

theorem range_of_k (a k : ℝ) (h : point_satisfies_curve a k) : k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l125_125190


namespace conclusion_1_conclusion_2_conclusion_3_main_theorem_l125_125976

-- First, define the necessary conditions
variables {V : Type*} [inner_product_space ℝ V]

-- Definition of parallel between lines
def parallel (l m : submodule ℝ V) : Prop := ∃ (v : V), l = submodule.span ℝ {v} ∧ m = submodule.span ℝ {v}

-- Definition of perpendicular between lines
def perpendicular (l m : submodule ℝ V) : Prop := ∃ (u v : V), l = submodule.span ℝ {u} ∧ m = submodule.span ℝ {v} ∧ inner_product_space.is_orthogonal V u v

-- Formulating the statements
theorem conclusion_1 {l m n : submodule ℝ V} (h1 : parallel l m) (h2 : parallel m n) : parallel l n := sorry
theorem conclusion_2 {l m n : submodule ℝ V} (h1 : perpendicular l m) (h2 : parallel m n) : perpendicular l n := sorry
theorem conclusion_3 {l m n : submodule ℝ V} (h1 : nonempty (l ⊓ m)) (h2 : parallel m n) : ¬ nonempty (l ⊓ n) := sorry

-- Main theorem combining conclusions
theorem main_theorem {l m n : submodule ℝ V} :
  (∀ (l m n : submodule ℝ V), parallel l m → parallel m n → parallel l n) ∧
  (∀ (l m n : submodule ℝ V), perpendicular l m → parallel m n → perpendicular l n) ∧
  (∀ (l m n : submodule ℝ V), nonempty (l ⊓ m) → parallel m n → ¬ nonempty (l ⊓ n)) := 
begin
  split,
  { intros l m n h1 h2,
    exact conclusion_1 h1 h2,
  },
  split,
  { intros l m n h1 h2,
    exact conclusion_2 h1 h2,
  },
  { intros l m n h1 h2,
    exact conclusion_3 h1 h2
  }
end

end conclusion_1_conclusion_2_conclusion_3_main_theorem_l125_125976


namespace domain_of_function_l125_125402

/-- The domain of the function \( y = \lg (12 + x - x^2) \) is the interval \(-3 < x < 4\). -/
theorem domain_of_function :
  {x : ℝ | 12 + x - x^2 > 0} = {x : ℝ | -3 < x ∧ x < 4} :=
sorry

end domain_of_function_l125_125402


namespace Jasper_height_in_10_minutes_l125_125206

noncomputable def OmarRate : ℕ := 240 / 12
noncomputable def JasperRate : ℕ := 3 * OmarRate
noncomputable def JasperHeight (time: ℕ) : ℕ := JasperRate * time

theorem Jasper_height_in_10_minutes :
  JasperHeight 10 = 600 :=
by
  sorry

end Jasper_height_in_10_minutes_l125_125206


namespace johns_age_l125_125893

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l125_125893


namespace simplify_expression_l125_125935

theorem simplify_expression (x y : ℝ) (h : x = -3) : 
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6 * x + 9) + 5 * x^3 * y^2 / (x^2 * y^2) = -66 :=
by
  sorry

end simplify_expression_l125_125935


namespace line_parallel_to_plane_line_perpendicular_to_plane_l125_125648

theorem line_parallel_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  A * m + B * n + C * p = 0 ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

theorem line_perpendicular_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  (A / m = B / n ∧ B / n = C / p) ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

end line_parallel_to_plane_line_perpendicular_to_plane_l125_125648


namespace income_is_10000_l125_125245

theorem income_is_10000 (x : ℝ) (h : 10 * x = 8 * x + 2000) : 10 * x = 10000 := by
  have h1 : 2 * x = 2000 := by
    linarith
  have h2 : x = 1000 := by
    linarith
  linarith

end income_is_10000_l125_125245


namespace sum_reciprocals_transformed_roots_l125_125213

theorem sum_reciprocals_transformed_roots (a b c : ℝ) (h : ∀ x, (x^3 - 2 * x - 5 = 0) → (x = a) ∨ (x = b) ∨ (x = c)) : 
  (1 / (a - 2)) + (1 / (b - 2)) + (1 / (c - 2)) = 10 := 
by sorry

end sum_reciprocals_transformed_roots_l125_125213


namespace union_of_P_and_Q_l125_125029

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem union_of_P_and_Q : (P ∪ Q) = {x | -1 < x ∧ x < 3} := by
  -- skipping the proof
  sorry

end union_of_P_and_Q_l125_125029


namespace map_at_three_l125_125919

variable (A B : Type)
variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (h_map : ∀ x : ℝ, f x = a * x - 1)
variable (h_cond : f 2 = 3)

theorem map_at_three : f 3 = 5 := by
  sorry

end map_at_three_l125_125919


namespace johns_age_l125_125883

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125883


namespace accum_correct_l125_125324

def accum (s : String) : String :=
  '-'.intercalate (List.map (fun (i : Nat) => (s.get! i).toUpper.toString ++ (s.get! i).toLower.toString * i) (List.range s.length))

theorem accum_correct (s : String) : accum s = 
  '-'.intercalate (List.map (fun (i : Nat) => (s.get! i).toUpper.toString ++ (s.get! i).toLower.toString * i) (List.range s.length)) :=
  sorry

end accum_correct_l125_125324


namespace multiply_powers_l125_125999

theorem multiply_powers (a : ℝ) : (a^3) * (a^3) = a^6 := by
  sorry

end multiply_powers_l125_125999


namespace express_y_in_terms_of_x_l125_125489

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := 
by
  sorry

end express_y_in_terms_of_x_l125_125489


namespace charlie_share_l125_125753

theorem charlie_share (A B C : ℕ) 
  (h1 : (A - 10) * 18 = (B - 20) * 11)
  (h2 : (A - 10) * 24 = (C - 15) * 11)
  (h3 : A + B + C = 1105) : 
  C = 495 := 
by
  sorry

end charlie_share_l125_125753


namespace new_ticket_price_l125_125578

theorem new_ticket_price (a : ℕ) (x : ℝ) (initial_price : ℝ) (revenue_increase : ℝ) (spectator_increase : ℝ)
  (h₀ : initial_price = 25)
  (h₁ : spectator_increase = 1.5)
  (h₂ : revenue_increase = 1.14)
  (h₃ : x = 0.76):
  initial_price * x = 19 :=
by
  sorry

end new_ticket_price_l125_125578


namespace problem_l125_125201

theorem problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 533.33 := by
  sorry

end problem_l125_125201


namespace balloon_arrangements_l125_125850

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l125_125850


namespace g_of_2_l125_125243

noncomputable def g : ℝ → ℝ := sorry

axiom cond1 (x y : ℝ) : x * g y = y * g x
axiom cond2 : g 10 = 30

theorem g_of_2 : g 2 = 6 := by
  sorry

end g_of_2_l125_125243


namespace at_least_three_points_in_circle_l125_125005

noncomputable def point_in_circle (p : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
(dist p c) ≤ r

theorem at_least_three_points_in_circle (points : Fin 51 → (ℝ × ℝ)) (side_length : ℝ) (circle_radius : ℝ)
  (h_side_length : side_length = 1) (h_circle_radius : circle_radius = 1 / 7) : 
  ∃ (c : ℝ × ℝ), ∃ (p1 p2 p3 : Fin 51), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    point_in_circle (points p1) c circle_radius ∧ 
    point_in_circle (points p2) c circle_radius ∧ 
    point_in_circle (points p3) c circle_radius :=
sorry

end at_least_three_points_in_circle_l125_125005


namespace trigonometric_expression_value_l125_125454

theorem trigonometric_expression_value :
  4 * Real.cos (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) -
  Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 3 / 4 := sorry

end trigonometric_expression_value_l125_125454


namespace johns_age_l125_125890

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l125_125890


namespace third_term_of_arithmetic_sequence_is_negative_22_l125_125677

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem third_term_of_arithmetic_sequence_is_negative_22
  (a d : ℤ)
  (H1 : arithmetic_sequence a d 14 = 14)
  (H2 : arithmetic_sequence a d 15 = 17) :
  arithmetic_sequence a d 2 = -22 :=
sorry

end third_term_of_arithmetic_sequence_is_negative_22_l125_125677


namespace ram_leela_piggy_bank_l125_125550

theorem ram_leela_piggy_bank (final_amount future_deposits weeks: ℕ) 
  (initial_deposit common_diff: ℕ) (total_deposits : ℕ) 
  (h_total : total_deposits = (weeks * (initial_deposit + (initial_deposit + (weeks - 1) * common_diff)) / 2)) 
  (h_final : final_amount = 1478) 
  (h_weeks : weeks = 52) 
  (h_future_deposits : future_deposits = total_deposits) 
  (h_initial_deposit : initial_deposit = 1) 
  (h_common_diff : common_diff = 1) 
  : final_amount - future_deposits = 100 :=
sorry

end ram_leela_piggy_bank_l125_125550


namespace balloon_arrangements_l125_125832

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l125_125832


namespace gcd_of_power_of_two_plus_one_l125_125150

theorem gcd_of_power_of_two_plus_one (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := 
sorry

end gcd_of_power_of_two_plus_one_l125_125150


namespace S₈_proof_l125_125535

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l125_125535


namespace number_of_dissimilar_terms_l125_125453

theorem number_of_dissimilar_terms :
  let n := 7;
  let k := 4;
  let number_of_terms := Nat.choose (n + k - 1) (k - 1);
  number_of_terms = 120 :=
by
  sorry

end number_of_dissimilar_terms_l125_125453


namespace balloon_arrangements_l125_125833

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l125_125833


namespace employed_population_percentage_l125_125687

theorem employed_population_percentage
  (P : ℝ) -- Total population
  (E : ℝ) -- Fraction of population that is employed
  (employed_males : ℝ) -- Fraction of population that is employed males
  (employed_females_fraction : ℝ)
  (h1 : employed_males = 0.8 * P)
  (h2 : employed_females_fraction = 1 / 3) :
  E = 0.6 :=
by
  -- We don't need the proof here.
  sorry

end employed_population_percentage_l125_125687


namespace price_jemma_sells_each_frame_is_5_l125_125619

noncomputable def jemma_price_per_frame : ℝ :=
  let num_frames_jemma := 400
  let num_frames_dorothy := num_frames_jemma / 2
  let total_income := 2500
  let P_jemma := total_income / (num_frames_jemma + num_frames_dorothy / 2)
  P_jemma

theorem price_jemma_sells_each_frame_is_5 :
  jemma_price_per_frame = 5 := by
  sorry

end price_jemma_sells_each_frame_is_5_l125_125619


namespace arnold_danny_age_l125_125744

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 9 → x = 4 :=
by
  intro h
  sorry

end arnold_danny_age_l125_125744


namespace intersection_with_y_axis_is_correct_l125_125501

theorem intersection_with_y_axis_is_correct (x y : ℝ) (h : y = 5 * x + 1) (hx : x = 0) : y = 1 :=
by
  sorry

end intersection_with_y_axis_is_correct_l125_125501


namespace singer_arrangements_l125_125282

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end singer_arrangements_l125_125282


namespace smallest_n_square_area_l125_125445

theorem smallest_n_square_area (n : ℕ) (n_positive : 0 < n) : ∃ k : ℕ, 14 * n = k^2 ↔ n = 14 := 
sorry

end smallest_n_square_area_l125_125445


namespace geometric_sequence_sum_eight_l125_125515

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l125_125515


namespace diff_squares_count_l125_125045

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l125_125045


namespace arithmetic_seq_S10_l125_125062

open BigOperators

variables (a : ℕ → ℚ) (d : ℚ)

-- Definitions based on the conditions
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) := ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 5 = 1
axiom h2 : a 1 + a 7 + a 10 = a 4 + a 6

-- We aim to prove the sum of the first 10 terms
def S (n : ℕ) :=
  ∑ i in Finset.range n, a (i + 1)

theorem arithmetic_seq_S10 : arithmetic_seq a d → S a 10 = 25 / 3 :=
by
  sorry

end arithmetic_seq_S10_l125_125062


namespace nesbitts_inequality_l125_125479

variable (a b c : ℝ)

theorem nesbitts_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) >= 3 / 2 := 
sorry

end nesbitts_inequality_l125_125479


namespace range_of_m_l125_125194

open Set

variable {m : ℝ}

def A : Set ℝ := { x | x^2 < 16 }
def B (m : ℝ) : Set ℝ := { x | x < m }

theorem range_of_m (h : A ∩ B m = A) : 4 ≤ m :=
by
  sorry

end range_of_m_l125_125194


namespace find_angle_B_l125_125182

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1: 0 < A ∧ A < π / 2)
  (h2: 0 < B ∧ B < π / 2) (h3: 0 < C ∧ C < π / 2)
  (h4: a * real.cos C + c * real.lcos A = 2 * b * real.cos B)
  (h5: A + B + C = π) :
  B = π / 3 :=
by
  sorry

end find_angle_B_l125_125182


namespace max_red_dominated_rows_plus_blue_dominated_columns_l125_125511

-- Definitions of the problem conditions and statement
theorem max_red_dominated_rows_plus_blue_dominated_columns (m n : ℕ)
  (h1 : Odd m) (h2 : Odd n) (h3 : 0 < m ∧ 0 < n) :
  ∃ A : Finset (Fin m) × Finset (Fin n),
  (A.1.card + A.2.card = m + n - 2) :=
sorry

end max_red_dominated_rows_plus_blue_dominated_columns_l125_125511


namespace brianna_initial_marbles_l125_125609

-- Defining the variables and constants
def initial_marbles : Nat := 24
def marbles_lost : Nat := 4
def marbles_given : Nat := 2 * marbles_lost
def marbles_ate : Nat := marbles_lost / 2
def marbles_remaining : Nat := 10

-- The main statement to prove
theorem brianna_initial_marbles :
  marbles_remaining + marbles_ate + marbles_given + marbles_lost = initial_marbles :=
by
  sorry

end brianna_initial_marbles_l125_125609


namespace john_age_proof_l125_125900

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l125_125900


namespace rowing_speed_in_still_water_l125_125600

theorem rowing_speed_in_still_water (v c : ℝ) (t : ℝ) (h1 : c = 1.1) (h2 : (v + c) * t = (v - c) * 2 * t) : v = 3.3 :=
sorry

end rowing_speed_in_still_water_l125_125600


namespace correct_number_of_statements_l125_125567

noncomputable def number_of_correct_statements := 1

def statement_1 : Prop := false -- Equal angles are not preserved
def statement_2 : Prop := false -- Equal lengths are not preserved
def statement_3 : Prop := false -- The longest segment feature is not preserved
def statement_4 : Prop := true  -- The midpoint feature is preserved

theorem correct_number_of_statements :
  (statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4) = true →
  number_of_correct_statements = 1 :=
by
  sorry

end correct_number_of_statements_l125_125567


namespace no_integer_n_exists_l125_125782

theorem no_integer_n_exists : ∀ (n : ℤ), n ^ 2022 - 2 * n ^ 2021 + 3 * n ^ 2019 ≠ 2020 :=
by sorry

end no_integer_n_exists_l125_125782


namespace divisor_of_form_4k_minus_1_l125_125915

theorem divisor_of_form_4k_minus_1
  (n : ℕ) (hn1 : Odd n) (hn_pos : 0 < n)
  (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (1 / (x : ℚ) + 1 / (y : ℚ) = 4 / n)) :
  ∃ k : ℕ, ∃ d, d ∣ n ∧ d = 4 * k - 1 ∧ k ∈ Set.Ici 1 :=
sorry

end divisor_of_form_4k_minus_1_l125_125915


namespace mod_5_pow_1000_div_29_l125_125856

theorem mod_5_pow_1000_div_29 : 5^1000 % 29 = 21 := 
by 
  -- The proof will go here.
  sorry

end mod_5_pow_1000_div_29_l125_125856


namespace new_person_weight_is_75_l125_125125

noncomputable def new_person_weight (previous_person_weight: ℝ) (average_increase: ℝ) (total_people: ℕ): ℝ :=
  previous_person_weight + total_people * average_increase

theorem new_person_weight_is_75 :
  new_person_weight 55 2.5 8 = 75 := 
by
  sorry

end new_person_weight_is_75_l125_125125


namespace balloon_permutations_l125_125816

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l125_125816


namespace croissants_for_breakfast_l125_125693

def total_items (C : ℕ) : Prop :=
  C + 18 + 30 = 110

theorem croissants_for_breakfast (C : ℕ) (h : total_items C) : C = 62 :=
by {
  -- The proof might be here, but since it's not required:
  sorry
}

end croissants_for_breakfast_l125_125693


namespace lower_limit_total_people_l125_125622

/-- 
  Given:
    1. Exactly 3/7 of the people in the room are under the age of 21.
    2. Exactly 5/10 of the people in the room are over the age of 65.
    3. There are 30 people in the room under the age of 21.
  Prove: The lower limit of the total number of people in the room is 70.
-/
theorem lower_limit_total_people (T : ℕ) (h1 : (3 / 7) * T = 30) : T = 70 := by
  sorry

end lower_limit_total_people_l125_125622


namespace max_profit_at_l125_125083

variables (k x : ℝ) (hk : k > 0)

-- Define the quantities based on problem conditions
def profit (k x : ℝ) : ℝ :=
  0.072 * k * x ^ 2 - k * x ^ 3

-- State the theorem
theorem max_profit_at (k : ℝ) (hk : k > 0) : 
  ∃ x, profit k x = 0.072 * k * x ^ 2 - k * x ^ 3 ∧ x = 0.048 :=
sorry

end max_profit_at_l125_125083


namespace right_triangle_area_l125_125275

theorem right_triangle_area (a b c : ℝ) (ht : a^2 + b^2 = c^2) (h1 : a = 24) (h2 : c = 26) : 
    (1/2) * a * b = 120 :=
begin
  sorry
end

end right_triangle_area_l125_125275


namespace age_difference_l125_125098

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end age_difference_l125_125098


namespace laptop_full_price_l125_125553

theorem laptop_full_price (p : ℝ) (deposit : ℝ) (h1 : deposit = 0.25 * p) (h2 : deposit = 400) : p = 1600 :=
by
  sorry

end laptop_full_price_l125_125553


namespace students_neither_correct_l125_125546

-- Define the total number of students and the numbers for chemistry, biology, and both
def total_students := 75
def chemistry_students := 42
def biology_students := 33
def both_subject_students := 18

-- Define a function to calculate the number of students taking neither chemistry nor biology
def students_neither : ℕ :=
  total_students - ((chemistry_students - both_subject_students) 
                    + (biology_students - both_subject_students) 
                    + both_subject_students)

-- Theorem stating that the number of students taking neither chemistry nor biology is as expected
theorem students_neither_correct : students_neither = 18 :=
  sorry

end students_neither_correct_l125_125546


namespace volume_of_prism_l125_125561

theorem volume_of_prism (a b c : ℝ)
  (h_ab : a * b = 36)
  (h_ac : a * c = 54)
  (h_bc : b * c = 72) :
  a * b * c = 648 :=
by
  sorry

end volume_of_prism_l125_125561


namespace probability_4_students_same_vehicle_l125_125295

-- Define the number of vehicles
def num_vehicles : ℕ := 3

-- Define the probability that 4 students choose the same vehicle
def probability_same_vehicle (n : ℕ) : ℚ :=
  3 / (3^(n : ℤ))

-- Prove that the probability for 4 students is 1/27
theorem probability_4_students_same_vehicle : probability_same_vehicle 4 = 1 / 27 := 
  sorry

end probability_4_students_same_vehicle_l125_125295


namespace portraits_count_l125_125151

theorem portraits_count (P S : ℕ) (h1 : S = 6 * P) (h2 : P + S = 200) : P = 28 := 
by
  -- The proof will be here.
  sorry

end portraits_count_l125_125151


namespace S₈_proof_l125_125537

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l125_125537


namespace num_divisors_of_8_factorial_l125_125655

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the number of positive divisors function which uses prime factorization
noncomputable def num_divisors (n : ℕ) : ℕ :=
let factors := unique_factorization_monoid.factors n in
factors.to_finset.prod (λ p, factors.count p + 1)

-- Mathematical statement to prove
theorem num_divisors_of_8_factorial : num_divisors (factorial 8) = 96 := 
sorry

end num_divisors_of_8_factorial_l125_125655


namespace coefficient_of_x3_in_expansion_l125_125871

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x3_in_expansion :
  (∑ k in Finset.range (41), binom_coeff 40 k * (1 : ℤ)^(40 - k) * (2 : ℤ)^k) = 79040 :=
begin
  have h₁ : binom_coeff 40 3 = 9880,
  { simp [binom_coeff, Nat.choose],
    norm_num },

  have h₂ : 1^(40 - 3) * 2^3 = 8,
  { norm_num },

  have h₃ : binom_coeff 40 3 * 8 = 79040,
  { norm_num,
    linarith },

  exact h₃,
end

end coefficient_of_x3_in_expansion_l125_125871


namespace students_more_than_guinea_pigs_l125_125471

-- Definitions based on the problem's conditions
def students_per_classroom : Nat := 22
def guinea_pigs_per_classroom : Nat := 3
def classrooms : Nat := 5

-- The proof statement
theorem students_more_than_guinea_pigs :
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 95 :=
by
  sorry

end students_more_than_guinea_pigs_l125_125471


namespace color_10x10_board_l125_125344

theorem color_10x10_board : 
  ∃ (ways : ℕ), ways = 2046 ∧ 
    ∀ (board : ℕ × ℕ → bool), 
    (∀ x y, 0 ≤ x ∧ x < 9 → 0 ≤ y ∧ y < 9 → 
      (board (x, y) + board (x + 1, y) + board (x, y + 1) + board (x + 1, y + 1) = 2)) 
    → (count_valid_colorings board = ways) := 
by 
  sorry  -- Proof is not provided, as per instructions.

end color_10x10_board_l125_125344


namespace car_transport_distance_l125_125959

theorem car_transport_distance
  (d_birdhouse : ℕ) 
  (d_lawnchair : ℕ) 
  (d_car : ℕ)
  (h1 : d_birdhouse = 1200)
  (h2 : d_birdhouse = 3 * d_lawnchair)
  (h3 : d_lawnchair = 2 * d_car) :
  d_car = 200 := 
by
  sorry

end car_transport_distance_l125_125959


namespace poly_div_factor_l125_125322

theorem poly_div_factor (c : ℚ) : 2 * x + 7 ∣ 8 * x^4 + 27 * x^3 + 6 * x^2 + c * x - 49 ↔
  c = 47.25 :=
  sorry

end poly_div_factor_l125_125322


namespace trail_mix_total_weight_l125_125608

theorem trail_mix_total_weight :
  let peanuts := 0.16666666666666666
  let chocolate_chips := 0.16666666666666666
  let raisins := 0.08333333333333333
  let almonds := 0.14583333333333331
  let cashews := (1 / 8 : Real)
  let dried_cranberries := (3 / 32 : Real)
  (peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries) = 0.78125 :=
by
  sorry

end trail_mix_total_weight_l125_125608


namespace Eiffel_Tower_model_scale_l125_125942

theorem Eiffel_Tower_model_scale
  (h_tower : ℝ := 324)
  (h_model_cm : ℝ := 18) :
  (h_tower / (h_model_cm / 100)) / 100 = 18 :=
by
  sorry

end Eiffel_Tower_model_scale_l125_125942


namespace solve_system_of_equations_l125_125558

theorem solve_system_of_equations (x y : ℝ) :
  (1 / 2 * x - 3 / 2 * y = -1) ∧ (2 * x + y = 3) → 
  (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l125_125558


namespace card_draw_probability_l125_125960

theorem card_draw_probability :
  (13 / 52) * (13 / 51) * (13 / 50) = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l125_125960


namespace arithmetic_sequence_sum_l125_125694

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := 
sorry

end arithmetic_sequence_sum_l125_125694


namespace inscribed_square_length_l125_125713

-- Define the right triangle PQR with given sides
variables (PQ QR PR : ℕ)
variables (h s : ℚ)

-- Given conditions
def right_triangle_PQR : Prop := PQ = 5 ∧ QR = 12 ∧ PR = 13
def altitude_Q_to_PR : Prop := h = (PQ * QR) / PR
def side_length_of_square : Prop := s = h * (1 - h / PR)

theorem inscribed_square_length (PQ QR PR h s : ℚ) 
    (right_triangle_PQR : PQ = 5 ∧ QR = 12 ∧ PR = 13)
    (altitude_Q_to_PR : h = (PQ * QR) / PR) 
    (side_length_of_square : s = h * (1 - h / PR)) 
    : s = 6540 / 2207 := by
  -- we skip the proof here as requested
  sorry

end inscribed_square_length_l125_125713


namespace value_of_x_l125_125014

theorem value_of_x (x y : ℝ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end value_of_x_l125_125014


namespace convert_to_scientific_notation_l125_125568

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l125_125568


namespace only_exprC_cannot_be_calculated_with_square_of_binomial_l125_125583

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end only_exprC_cannot_be_calculated_with_square_of_binomial_l125_125583


namespace num_days_c_worked_l125_125588

theorem num_days_c_worked (d : ℕ) :
  let daily_wage_c := 100
  let daily_wage_b := (4 * 20)
  let daily_wage_a := (3 * 20)
  let total_earning := 1480
  let earning_a := 6 * daily_wage_a
  let earning_b := 9 * daily_wage_b
  let earning_c := d * daily_wage_c
  total_earning = earning_a + earning_b + earning_c →
  d = 4 :=
by {
  sorry
}

end num_days_c_worked_l125_125588


namespace probability_non_adjacent_l125_125350

def total_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n m 

def non_adjacent_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n (m - 1)

def probability_zeros_non_adjacent (n m : ℕ) : ℚ :=
  (non_adjacent_arrangements n m : ℚ) / (total_arrangements n m : ℚ)

theorem probability_non_adjacent (a b : ℕ) (h₁ : a = 4) (h₂ : b = 2) :
  probability_zeros_non_adjacent 5 2 = 2 / 3 := 
by 
  rw [probability_zeros_non_adjacent]
  rw [non_adjacent_arrangements, total_arrangements]
  sorry

end probability_non_adjacent_l125_125350


namespace smallest_number_div_by_225_with_digits_0_1_l125_125118

theorem smallest_number_div_by_225_with_digits_0_1 :
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ n ∧ (∀ m : ℕ, (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ m → n ≤ m) ∧ n = 11111111100 :=
sorry

end smallest_number_div_by_225_with_digits_0_1_l125_125118


namespace sam_spent_136_96_l125_125231

def glove_original : Real := 35
def glove_discount : Real := 0.20
def baseball_price : Real := 15
def bat_original : Real := 50
def bat_discount : Real := 0.10
def cleats_price : Real := 30
def cap_price : Real := 10
def tax_rate : Real := 0.07

def total_spent (glove_original : Real) (glove_discount : Real) (baseball_price : Real) (bat_original : Real) (bat_discount : Real) (cleats_price : Real) (cap_price : Real) (tax_rate : Real) : Real :=
  let glove_price := glove_original - (glove_discount * glove_original)
  let bat_price := bat_original - (bat_discount * bat_original)
  let total_before_tax := glove_price + baseball_price + bat_price + cleats_price + cap_price
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

theorem sam_spent_136_96 :
  total_spent glove_original glove_discount baseball_price bat_original bat_discount cleats_price cap_price tax_rate = 136.96 :=
sorry

end sam_spent_136_96_l125_125231


namespace square_side_length_l125_125943

theorem square_side_length (s : ℝ) (h : s^2 = 3 * 4 * s) : s = 12 :=
by
  sorry

end square_side_length_l125_125943


namespace min_value_a4b3c2_l125_125214

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l125_125214


namespace fish_count_l125_125390

theorem fish_count (initial_fish : ℝ) (bought_fish : ℝ) (total_fish : ℝ) 
  (h1 : initial_fish = 212.0) 
  (h2 : bought_fish = 280.0) 
  (h3 : total_fish = initial_fish + bought_fish) : 
  total_fish = 492.0 := 
by 
  sorry

end fish_count_l125_125390


namespace distance_between_neg5_and_neg1_l125_125563

theorem distance_between_neg5_and_neg1 : 
  dist (-5 : ℝ) (-1) = 4 := by
sorry

end distance_between_neg5_and_neg1_l125_125563


namespace justin_home_time_l125_125908

noncomputable def dinner_duration : ℕ := 45
noncomputable def homework_duration : ℕ := 30
noncomputable def cleaning_room_duration : ℕ := 30
noncomputable def taking_out_trash_duration : ℕ := 5
noncomputable def emptying_dishwasher_duration : ℕ := 10

noncomputable def total_time_required : ℕ :=
  dinner_duration + homework_duration + cleaning_room_duration + taking_out_trash_duration + emptying_dishwasher_duration

noncomputable def latest_start_time_hour : ℕ := 18 -- 6 pm in 24-hour format
noncomputable def total_time_required_hours : ℕ := 2
noncomputable def movie_time_hour : ℕ := 20 -- 8 pm in 24-hour format

theorem justin_home_time : latest_start_time_hour - total_time_required_hours = 16 := -- 4 pm in 24-hour format
by
  sorry

end justin_home_time_l125_125908


namespace perimeter_of_triangle_ABC_l125_125922

noncomputable def triangle_perimeter (r1 r2 r3 : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let x3 := r3 * Real.cos θ3
  let y3 := r3 * Real.sin θ3
  let d12 := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d23 := Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)
  let d31 := Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)
  d12 + d23 + d31

--prove

theorem perimeter_of_triangle_ABC (θ1 θ2 θ3: ℝ)
  (h1: θ1 - θ2 = Real.pi / 3)
  (h2: θ2 - θ3 = Real.pi / 3) :
  triangle_perimeter 4 5 7 θ1 θ2 θ3 = sorry := 
sorry

end perimeter_of_triangle_ABC_l125_125922


namespace hungarian_license_plates_l125_125863

/-- 
In Hungarian license plates, digits can be identical. Based on observations, 
someone claimed that on average, approximately 3 out of every 10 vehicles 
have such license plates. Is this statement true?
-/
theorem hungarian_license_plates : 
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  abs (probability - 0.3) < 0.05 :=
by {
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  sorry
}

end hungarian_license_plates_l125_125863


namespace operation_4_3_is_5_l125_125124

def custom_operation (m n : ℕ) : ℕ := n ^ 2 - m

theorem operation_4_3_is_5 : custom_operation 4 3 = 5 :=
by
  -- Proof goes here
  sorry

end operation_4_3_is_5_l125_125124


namespace sum_of_values_satisfying_l125_125582

theorem sum_of_values_satisfying (x : ℝ) (h : Real.sqrt ((x - 2) ^ 2) = 8) :
  ∃ x1 x2 : ℝ, (Real.sqrt ((x1 - 2) ^ 2) = 8) ∧ (Real.sqrt ((x2 - 2) ^ 2) = 8) ∧ x1 + x2 = 4 := 
by
  sorry

end sum_of_values_satisfying_l125_125582


namespace x_finishes_remaining_work_in_14_days_l125_125127

-- Define the work rates of X and Y
def work_rate_X : ℚ := 1 / 21
def work_rate_Y : ℚ := 1 / 15

-- Define the amount of work Y completed in 5 days
def work_done_by_Y_in_5_days : ℚ := 5 * work_rate_Y

-- Define the remaining work after Y left
def remaining_work : ℚ := 1 - work_done_by_Y_in_5_days

-- Define the number of days needed for X to finish the remaining work
def x_days_remaining : ℚ := remaining_work / work_rate_X

-- Statement to prove
theorem x_finishes_remaining_work_in_14_days : x_days_remaining = 14 := by
  sorry

end x_finishes_remaining_work_in_14_days_l125_125127


namespace probability_equals_two_thirds_l125_125348

-- Definitions for total arrangements and favorable arrangements
def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 2

-- Probability that 2 zeros are not adjacent
def probability_not_adjacent : ℚ := favorable_arrangements / total_arrangements

theorem probability_equals_two_thirds : probability_not_adjacent = 2 / 3 := 
by 
  let total_arrangements := 15
  let favorable_arrangements := 10
  have h1 : probability_not_adjacent = (10 : ℚ) / (15 : ℚ) := rfl
  have h2 : (10 : ℚ) / (15 : ℚ) = 2 / 3 := by norm_num
  exact Eq.trans h1 h2 

end probability_equals_two_thirds_l125_125348


namespace jane_sandwich_count_l125_125985

noncomputable def total_sandwiches : ℕ := 5 * 7 * 4

noncomputable def turkey_swiss_reduction : ℕ := 5 * 1 * 1

noncomputable def salami_bread_reduction : ℕ := 5 * 1 * 4

noncomputable def correct_sandwich_count : ℕ := 115

theorem jane_sandwich_count : total_sandwiches - turkey_swiss_reduction - salami_bread_reduction = correct_sandwich_count :=
by
  sorry

end jane_sandwich_count_l125_125985


namespace triangle_side_AC_value_l125_125499

theorem triangle_side_AC_value
  (AB BC : ℝ) (AC : ℕ)
  (hAB : AB = 1)
  (hBC : BC = 2007)
  (hAC_int : ∃ (n : ℕ), AC = n) :
  AC = 2007 :=
by
  sorry

end triangle_side_AC_value_l125_125499


namespace unique_symmetric_matrix_pair_l125_125321

theorem unique_symmetric_matrix_pair (a b : ℝ) :
  (∃! M : Matrix (Fin 2) (Fin 2) ℝ, M = M.transpose ∧ Matrix.trace M = a ∧ Matrix.det M = b)
  ↔ (∃ t : ℝ, a = 2 * t ∧ b = t^2) :=
by
  sorry

end unique_symmetric_matrix_pair_l125_125321


namespace ratio_of_pens_to_notebooks_is_5_to_4_l125_125092

theorem ratio_of_pens_to_notebooks_is_5_to_4 (P N : ℕ) (hP : P = 50) (hN : N = 40) :
  (P / Nat.gcd P N) = 5 ∧ (N / Nat.gcd P N) = 4 :=
by
  -- Proof goes here
  sorry

end ratio_of_pens_to_notebooks_is_5_to_4_l125_125092


namespace profit_percent_l125_125586

theorem profit_percent (marked_price : ℝ) (num_bought : ℝ) (num_payed_price : ℝ) (discount_percent : ℝ) : 
  num_bought = 56 → 
  num_payed_price = 46 → 
  discount_percent = 0.01 →
  marked_price = 1 →
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 20.52 :=
by 
  intro hnum_bought hnum_payed_price hdiscount_percent hmarked_price 
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  sorry

end profit_percent_l125_125586


namespace cylinder_volume_ratio_l125_125131

theorem cylinder_volume_ratio (a b : ℕ) (h_dim : (a, b) = (9, 12)) :
  let r₁ := (a : ℝ) / (2 * Real.pi)
  let h₁ := (↑b : ℝ)
  let V₁ := (Real.pi * r₁^2 * h₁)
  let r₂ := (b : ℝ) / (2 * Real.pi)
  let h₂ := (↑a : ℝ)
  let V₂ := (Real.pi * r₂^2 * h₂)
  (if V₂ > V₁ then V₂ / V₁ else V₁ / V₂) = (16 / 3) :=
by {
  sorry
}

end cylinder_volume_ratio_l125_125131


namespace no_integer_solutions_m2n_eq_2mn_minus_3_l125_125467

theorem no_integer_solutions_m2n_eq_2mn_minus_3 :
  ∀ (m n : ℤ), m + 2 * n ≠ 2 * m * n - 3 := 
sorry

end no_integer_solutions_m2n_eq_2mn_minus_3_l125_125467


namespace incorrect_statements_l125_125799

noncomputable def f : ℝ → ℝ := λ x, Real.cos (x + Real.pi / 3)

theorem incorrect_statements :
  (¬ (∃ k : ℤ, f (x + k * (2 * Real.pi)) = f x ∧ k ≠ 0 → f x = f (x - 2 * Real.pi))) ∨
  (¬ (∀ x : ℝ, f (x + (8 * Real.pi / 3)) = f x)) ∨
  (¬ (∃ x : ℝ, f (x + Real.pi) = 0 → x = Real.pi / 6)) ∨
  (¬ (∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + Real.pi / 3)))
:= by
  sorry

end incorrect_statements_l125_125799


namespace number_of_cut_red_orchids_l125_125104

variable (initial_red_orchids added_red_orchids final_red_orchids : ℕ)

-- Conditions
def initial_red_orchids_in_vase (initial_red_orchids : ℕ) : Prop :=
  initial_red_orchids = 9

def final_red_orchids_in_vase (final_red_orchids : ℕ) : Prop :=
  final_red_orchids = 15

-- Proof statement
theorem number_of_cut_red_orchids (initial_red_orchids added_red_orchids final_red_orchids : ℕ)
  (h1 : initial_red_orchids_in_vase initial_red_orchids) 
  (h2 : final_red_orchids_in_vase final_red_orchids) :
  final_red_orchids = initial_red_orchids + added_red_orchids → added_red_orchids = 6 := by
  simp [initial_red_orchids_in_vase, final_red_orchids_in_vase] at *
  sorry

end number_of_cut_red_orchids_l125_125104


namespace percent_problem_l125_125360

theorem percent_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end percent_problem_l125_125360


namespace total_invested_expression_l125_125057

variables (x y T : ℝ)

axiom annual_income_exceed_65 : 0.10 * x - 0.08 * y = 65
axiom total_invested_is_T : x + y = T

theorem total_invested_expression :
  T = 1.8 * y + 650 :=
sorry

end total_invested_expression_l125_125057


namespace radius_larger_circle_l125_125404

theorem radius_larger_circle (r : ℝ) (AC BC : ℝ) (h1 : 5 * r = AC / 2) (h2 : 15 = BC) : 
  5 * r = 18.75 :=
by
  sorry

end radius_larger_circle_l125_125404


namespace copper_sheet_area_l125_125329

noncomputable def area_of_copper_sheet (l w h : ℝ) (thickness_mm : ℝ) : ℝ :=
  let volume := l * w * h
  let thickness_cm := thickness_mm / 10
  (volume / thickness_cm) / 10000

theorem copper_sheet_area :
  ∀ (l w h thickness_mm : ℝ), 
  l = 80 → w = 20 → h = 5 → thickness_mm = 1 → 
  area_of_copper_sheet l w h thickness_mm = 8 := 
by
  intros l w h thickness_mm hl hw hh hthickness_mm
  rw [hl, hw, hh, hthickness_mm]
  simp [area_of_copper_sheet]
  sorry

end copper_sheet_area_l125_125329


namespace cosine_triangle_ABC_l125_125876

noncomputable def triangle_cosine_proof (a b : ℝ) (A : ℝ) (cosB : ℝ) : Prop :=
  let sinA := Real.sin A
  let sinB := b * sinA / a
  let cosB_expr := Real.sqrt (1 - sinB^2)
  cosB = cosB_expr

theorem cosine_triangle_ABC : triangle_cosine_proof (Real.sqrt 7) 2 (Real.pi / 4) (Real.sqrt 35 / 7) :=
by
  sorry

end cosine_triangle_ABC_l125_125876


namespace fraction_exponentiation_l125_125998

theorem fraction_exponentiation : (3/4 : ℚ)^3 = 27/64 := by
  sorry

end fraction_exponentiation_l125_125998


namespace baker_initial_cakes_l125_125774

theorem baker_initial_cakes (sold : ℕ) (left : ℕ) (initial : ℕ) 
  (h_sold : sold = 41) (h_left : left = 13) : 
  sold + left = initial → initial = 54 :=
by
  intros
  exact sorry

end baker_initial_cakes_l125_125774


namespace slope_of_line_l125_125672

theorem slope_of_line {m : ℝ} (h1: 2 * 0 - m * (1/4) + 1 = 0) :
  m = 4 ∧ ((2 : ℝ) / m = (1 / 2)) :=
by
  sorry

end slope_of_line_l125_125672


namespace ratio_of_earnings_l125_125506

theorem ratio_of_earnings (jacob_hourly: ℕ) (jake_total: ℕ) (days: ℕ) (hours_per_day: ℕ) (jake_hourly: ℕ) (ratio: ℕ) 
  (h_jacob: jacob_hourly = 6)
  (h_jake_total: jake_total = 720)
  (h_days: days = 5)
  (h_hours_per_day: hours_per_day = 8)
  (h_jake_hourly: jake_hourly = jake_total / (days * hours_per_day))
  (h_ratio: ratio = jake_hourly / jacob_hourly) :
  ratio = 3 := 
sorry

end ratio_of_earnings_l125_125506


namespace perimeter_of_garden_l125_125560

-- Define the area of the square garden
def area_square_garden : ℕ := 49

-- Define the relationship between q and p
def q_equals_p_plus_21 (q p : ℕ) : Prop := q = p + 21

-- Define the length of the side of the square garden
def side_length (area : ℕ) : ℕ := Nat.sqrt area

-- Define the perimeter of the square garden
def perimeter (side_length : ℕ) : ℕ := 4 * side_length

-- Define the perimeter of the square garden as a specific perimeter
def specific_perimeter (side_length : ℕ) : ℕ := perimeter side_length

-- Statement of the theorem
theorem perimeter_of_garden (q p : ℕ) (h1 : q = 49) (h2 : q_equals_p_plus_21 q p) : 
  specific_perimeter (side_length 49) = 28 := by
  sorry

end perimeter_of_garden_l125_125560


namespace mass_percentage_O_in_Al2_CO3_3_correct_l125_125007

noncomputable def mass_percentage_O_in_Al2_CO3_3 : ℚ := 
  let mass_O := 9 * 16.00
  let molar_mass_Al2_CO3_3 := (2 * 26.98) + (3 * 12.01) + (9 * 16.00)
  (mass_O / molar_mass_Al2_CO3_3) * 100

theorem mass_percentage_O_in_Al2_CO3_3_correct :
  mass_percentage_O_in_Al2_CO3_3 = 61.54 :=
by
  unfold mass_percentage_O_in_Al2_CO3_3
  sorry

end mass_percentage_O_in_Al2_CO3_3_correct_l125_125007


namespace minimize_expression_l125_125328

theorem minimize_expression (x : ℝ) : x = -1 → ∀ y : ℝ, 3 * y * y + 6 * y - 2 ≥ 3 * (-1) * (-1) + 6 * (-1) - 2 :=
by {
  sorry
}

end minimize_expression_l125_125328


namespace rowing_time_one_hour_l125_125986

noncomputable def total_time_to_travel (Vm Vr distance : ℝ) : ℝ :=
  let upstream_speed := Vm - Vr
  let downstream_speed := Vm + Vr
  let one_way_distance := distance / 2
  let time_upstream := one_way_distance / upstream_speed
  let time_downstream := one_way_distance / downstream_speed
  time_upstream + time_downstream

theorem rowing_time_one_hour : 
  total_time_to_travel 8 1.8 7.595 = 1 := 
sorry

end rowing_time_one_hour_l125_125986


namespace sandy_worked_days_l125_125232

-- Definitions based on the conditions
def total_hours_worked : ℕ := 45
def hours_per_day : ℕ := 9

-- The theorem that we need to prove
theorem sandy_worked_days : total_hours_worked / hours_per_day = 5 :=
by sorry

end sandy_worked_days_l125_125232


namespace original_number_without_10s_digit_l125_125142

theorem original_number_without_10s_digit (h : ℕ) (n : ℕ) 
  (h_eq_1 : h = 1) 
  (n_eq : n = 2 * 1000 + h * 100 + 84) 
  (div_by_6: n % 6 = 0) : n = 2184 → 284 = 284 :=
by
  sorry

end original_number_without_10s_digit_l125_125142


namespace arithmetic_geometric_mean_inequality_l125_125637

variable {a b : ℝ}

noncomputable def A (a b : ℝ) := (a + b) / 2
noncomputable def B (a b : ℝ) := Real.sqrt (a * b)

theorem arithmetic_geometric_mean_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ b) : A a b > B a b := 
by
  sorry

end arithmetic_geometric_mean_inequality_l125_125637


namespace general_term_a_general_term_b_l125_125184

def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
∀ n, a_n n = n ∧ S_n n = (n^2 + n) / 2

def sequence_b (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :=
  (b_n 1 = 1/2) ∧
  (∀ n, b_n (n+1) = (n+1) / n * b_n n) ∧ 
  (∀ n, b_n n = n / 2) ∧ 
  (∀ n, T_n n = (n^2 + n) / 4) ∧ 
  (∀ m, m = 1 → T_n m = 1/2)

-- Arithmetic sequence {a_n}
theorem general_term_a (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 2 = 2) (h2 : S 5 = 15) :
  arithmetic_sequence a S := sorry

-- Sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (T : ℕ → ℝ) (h1 : b 1 = 1/2) (h2 : ∀ n, b (n+1) = (n+1) / n * b n) :
  sequence_b b T := sorry

end general_term_a_general_term_b_l125_125184


namespace money_needed_l125_125073

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end money_needed_l125_125073


namespace shingle_area_l125_125434

-- Definitions from conditions
def length := 10 -- uncut side length in inches
def width := 7   -- uncut side width in inches
def trapezoid_base1 := 6 -- base of the trapezoid in inches
def trapezoid_height := 2 -- height of the trapezoid in inches

-- Definition derived from conditions
def trapezoid_base2 := length - trapezoid_base1 -- the second base of the trapezoid

-- Required proof in Lean
theorem shingle_area : (length * width - (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height)) = 60 := 
by
  sorry

end shingle_area_l125_125434


namespace compute_abc_l125_125664

theorem compute_abc (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30) 
  (h2 : (1 / a + 1 / b + 1 / c + 420 / (a * b * c) = 1)) : 
  a * b * c = 450 := 
sorry

end compute_abc_l125_125664


namespace maximum_sphere_radius_squared_l125_125110

def cone_base_radius : ℝ := 4
def cone_height : ℝ := 10
def axes_intersection_distance_from_base : ℝ := 4

theorem maximum_sphere_radius_squared :
  let m : ℕ := 144
  let n : ℕ := 29
  m + n = 173 :=
by
  sorry

end maximum_sphere_radius_squared_l125_125110


namespace age_difference_l125_125746

variables (P M Mo : ℕ)

def patrick_michael_ratio (P M : ℕ) : Prop := (P * 5 = M * 3)
def michael_monica_ratio (M Mo : ℕ) : Prop := (M * 4 = Mo * 3)
def sum_of_ages (P M Mo : ℕ) : Prop := (P + M + Mo = 88)

theorem age_difference (P M Mo : ℕ) : 
  patrick_michael_ratio P M → 
  michael_monica_ratio M Mo → 
  sum_of_ages P M Mo → 
  (Mo - P = 22) :=
by
  sorry

end age_difference_l125_125746


namespace set_intersection_complement_l125_125421

open Set

theorem set_intersection_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  (U \ A) ∩ B = {3} :=
by
  sorry

end set_intersection_complement_l125_125421


namespace diagram_is_knowledge_structure_l125_125401

inductive DiagramType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

axiom given_diagram : DiagramType
axiom diagram_is_one_of_them : 
  given_diagram = DiagramType.ProgramFlowchart ∨ 
  given_diagram = DiagramType.ProcessFlowchart ∨ 
  given_diagram = DiagramType.KnowledgeStructureDiagram ∨ 
  given_diagram = DiagramType.OrganizationalStructureDiagram

theorem diagram_is_knowledge_structure :
  given_diagram = DiagramType.KnowledgeStructureDiagram :=
sorry

end diagram_is_knowledge_structure_l125_125401


namespace actual_height_of_boy_l125_125238

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end actual_height_of_boy_l125_125238


namespace chris_packed_percentage_l125_125909

theorem chris_packed_percentage (K C : ℕ) (h : K / (C : ℝ) = 2 / 3) :
  (C / (K + C : ℝ)) * 100 = 60 :=
by
  sorry

end chris_packed_percentage_l125_125909


namespace zero_in_interval_l125_125332

theorem zero_in_interval (a b : ℝ) (ha : 1 < a) (hb : 0 < b ∧ b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ (a^x + x - b = 0) :=
by {
  sorry
}

end zero_in_interval_l125_125332


namespace closest_point_exists_l125_125008

def closest_point_on_line_to_point (x : ℝ) (y : ℝ) : Prop :=
  ∃(p : ℝ × ℝ), p = (3, 1) ∧ ∀(q : ℝ × ℝ), q.2 = (q.1 + 3) / 3 → dist p (3, 2) ≤ dist q (3, 2)

theorem closest_point_exists :
  closest_point_on_line_to_point 3 2 :=
sorry

end closest_point_exists_l125_125008


namespace percentage_answered_second_question_correctly_l125_125495

open Finset

-- Define our universal set of total students
variable (U : Type) [Fintype U]

-- Define sets A and B
variable (A B : Finset U)

-- Define the conditions
axiom condition1 : (A.card : ℝ) / (card U) = 0.75
axiom condition2 : (A ∩ B).card / (card U) = 0.25
axiom condition3 : (U \ (A ∪ B)).card / (card U) = 0.20

-- Define the theorem to prove the percentage who answered the second question correctly
theorem percentage_answered_second_question_correctly : 
  (B.card : ℝ) / (card U) = 0.30 :=
sorry

end percentage_answered_second_question_correctly_l125_125495


namespace perimeter_C_is_74_l125_125168

/-- Definitions of side lengths based on given perimeters -/
def side_length_A (p_A : ℕ) : ℕ :=
  p_A / 4

def side_length_B (p_B : ℕ) : ℕ :=
  p_B / 4

/-- Definition of side length of C in terms of side lengths of A and B -/
def side_length_C (s_A s_B : ℕ) : ℚ :=
  (s_A : ℚ) / 2 + 2 * (s_B : ℚ)

/-- Definition of perimeter in terms of side length -/
def perimeter (s : ℚ) : ℚ :=
  4 * s

/-- Theorem statement: the perimeter of square C is 74 -/
theorem perimeter_C_is_74 (p_A p_B : ℕ) (h₁ : p_A = 20) (h₂ : p_B = 32) :
  perimeter (side_length_C (side_length_A p_A) (side_length_B p_B)) = 74 := by
  sorry

end perimeter_C_is_74_l125_125168


namespace probability_sum_3_is_1_over_216_l125_125964

-- Let E be the event that three fair dice sum to 3
def event_sum_3 (d1 d2 d3 : ℕ) : Prop := d1 + d2 + d3 = 3

-- Probabilities of rolling a particular outcome on a single die
noncomputable def P_roll_1 (n : ℕ) := if n = 1 then 1/6 else 0

-- Define the probability of the event E occurring
noncomputable def P_event_sum_3 := 
  ∑ d1 in {1, 2, 3, 4, 5, 6}, 
  ∑ d2 in {1, 2, 3, 4, 5, 6}, 
  ∑ d3 in {1, 2, 3, 4, 5, 6}, 
  if event_sum_3 d1 d2 d3 then P_roll_1 d1 * P_roll_1 d2 * P_roll_1 d3 else 0

-- The main theorem to prove the desired probability
theorem probability_sum_3_is_1_over_216 : P_event_sum_3 = 1/216 := by 
  sorry

end probability_sum_3_is_1_over_216_l125_125964


namespace total_number_of_students_l125_125577

theorem total_number_of_students (T G : ℕ) (h1 : 50 + G = T) (h2 : G = 50 * T / 100) : T = 100 :=
  sorry

end total_number_of_students_l125_125577


namespace jenny_kenny_see_each_other_l125_125507

-- Definitions of conditions
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def paths_distance : ℝ := 300
def radius_building : ℝ := 75
def start_distance : ℝ := 300

-- Theorem statement
theorem jenny_kenny_see_each_other : ∃ t : ℝ, (t = 120) :=
by
  sorry

end jenny_kenny_see_each_other_l125_125507


namespace lowest_fraction_done_in_an_hour_by_two_people_l125_125418

def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 5
def c_rate : ℚ := 1 / 6

theorem lowest_fraction_done_in_an_hour_by_two_people : 
  min (min (a_rate + b_rate) (a_rate + c_rate)) (b_rate + c_rate) = 11 / 30 := 
by
  sorry

end lowest_fraction_done_in_an_hour_by_two_people_l125_125418


namespace johns_age_l125_125907

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125907


namespace john_yasmin_child_ratio_l125_125692

theorem john_yasmin_child_ratio
  (gabriel_grandkids : ℕ)
  (yasmin_children : ℕ)
  (john_children : ℕ)
  (h1 : gabriel_grandkids = 6)
  (h2 : yasmin_children = 2)
  (h3 : john_children + yasmin_children = gabriel_grandkids) :
  john_children / yasmin_children = 2 :=
by 
  sorry

end john_yasmin_child_ratio_l125_125692


namespace ratio_of_speeds_is_2_l125_125700

-- Definitions based on conditions
def rate_of_machine_B : ℕ := 100 / 40 -- Rate of Machine B (parts per minute)
def rate_of_machine_A : ℕ := 50 / 10 -- Rate of Machine A (parts per minute)
def ratio_of_speeds (rate_A rate_B : ℕ) : ℕ := rate_A / rate_B -- Ratio of speeds

-- Proof statement
theorem ratio_of_speeds_is_2 : ratio_of_speeds rate_of_machine_A rate_of_machine_B = 2 := by
  sorry

end ratio_of_speeds_is_2_l125_125700


namespace max_value_of_function_cos_sin_l125_125565

noncomputable def max_value_function (x : ℝ) : ℝ := 
  (Real.cos x)^3 + (Real.sin x)^2 - Real.cos x

theorem max_value_of_function_cos_sin : 
  ∃ x ∈ (Set.univ : Set ℝ), max_value_function x = (32 / 27) := 
sorry

end max_value_of_function_cos_sin_l125_125565


namespace symmetric_about_y_axis_l125_125189

-- Condition: f is an odd function defined on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given that f is odd and F is defined as specified
theorem symmetric_about_y_axis (f : ℝ → ℝ)
  (hf : odd_function f) :
  ∀ x : ℝ, |f x| + f (|x|) = |f (-x)| + f (|x|) := 
by
  sorry

end symmetric_about_y_axis_l125_125189


namespace division_result_l125_125736

-- Definitions for the values used in the problem
def numerator := 0.0048 * 3.5
def denominator := 0.05 * 0.1 * 0.004

-- Theorem statement
theorem division_result : numerator / denominator = 840 := by 
  sorry

end division_result_l125_125736


namespace exists_ellipse_l125_125335

theorem exists_ellipse (a : ℝ) : ∃ a : ℝ, ∀ x y : ℝ, (x^2 + y^2 / a = 1) → a > 0 ∧ a ≠ 1 := 
by 
  sorry

end exists_ellipse_l125_125335


namespace cost_of_other_disc_l125_125988

theorem cost_of_other_disc (x : ℝ) (total_spent : ℝ) (num_discs : ℕ) (num_850_discs : ℕ) (price_850 : ℝ) 
    (total_cost : total_spent = 93) (num_bought : num_discs = 10) (num_850 : num_850_discs = 6) (price_per_850 : price_850 = 8.50) 
    (total_cost_850 : num_850_discs * price_850 = 51) (remaining_discs_cost : total_spent - 51 = 42) (remaining_discs : num_discs - num_850_discs = 4) :
    total_spent = num_850_discs * price_850 + (num_discs - num_850_discs) * x → x = 10.50 :=
by
  sorry

end cost_of_other_disc_l125_125988


namespace find_C_coordinates_l125_125647

open Real

noncomputable def coordC (A B : ℝ × ℝ) : ℝ × ℝ :=
  let n := A.1
  let m := B.1
  let coord_n_y : ℝ := n
  let coord_m_y : ℝ := m
  let y_value (x : ℝ) : ℝ := sqrt 3 / x
  (sqrt 3 / 2, 2)

theorem find_C_coordinates :
  ∃ C : ℝ × ℝ, 
  (∃ A B : ℝ × ℝ, 
   A.2 = sqrt 3 / A.1 ∧
   B.2 = sqrt 3 / B.1 + 6 ∧
   A.2 + 6 = B.2 ∧
   B.2 > A.2 ∧ 
   (sqrt 3 / 2, 2) = coordC A B) ∧
   (sqrt 3 / 2, 2) = (C.1, C.2) :=
by
  sorry

end find_C_coordinates_l125_125647


namespace train_length_is_120_l125_125333

-- Definitions based on conditions
def bridge_length : ℕ := 600
def total_time : ℕ := 30
def on_bridge_time : ℕ := 20

-- Proof statement
theorem train_length_is_120 (x : ℕ) (speed1 speed2 : ℕ) :
  (speed1 = (bridge_length + x) / total_time) ∧
  (speed2 = bridge_length / on_bridge_time) ∧
  (speed1 = speed2) →
  x = 120 :=
by
  sorry

end train_length_is_120_l125_125333


namespace find_first_number_l125_125668

theorem find_first_number (x : ℝ) (h1 : 2994 / x = 175) (h2 : 29.94 / 1.45 = 17.5) : x = 17.1 :=
by
  sorry

end find_first_number_l125_125668


namespace inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l125_125485

theorem inequality_8xyz_leq_1 (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_cases_8xyz_eq_1 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ 
  (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨ 
  (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l125_125485


namespace largest_square_area_correct_l125_125689

noncomputable def area_of_largest_square (x y z : ℝ) : Prop := 
  ∃ (area : ℝ), (z^2 = area) ∧ 
                 (x^2 + y^2 = z^2) ∧ 
                 (x^2 + y^2 + 2*z^2 = 722) ∧ 
                 (area = 722 / 3)

theorem largest_square_area_correct (x y z : ℝ) :
  area_of_largest_square x y z :=
  sorry

end largest_square_area_correct_l125_125689


namespace geometric_sequence_sum_S8_l125_125517

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l125_125517


namespace math_problem_l125_125412

theorem math_problem
  (numerator : ℕ := (Nat.factorial 10))
  (denominator : ℕ := (10 * 11 / 2)) :
  (numerator / denominator : ℚ) = 66069 + 1 / 11 := by
  sorry

end math_problem_l125_125412


namespace find_common_difference_find_possible_a1_l125_125384

structure ArithSeq :=
  (a : ℕ → ℤ) -- defining the sequence
  
noncomputable def S (n : ℕ) (a : ArithSeq) : ℤ :=
  (n * (2 * a.a 0 + (n - 1) * (a.a 1 - a.a 0))) / 2

axiom a4 (a : ArithSeq) : a.a 3 = 10

axiom S20 (a : ArithSeq) : S 20 a = 590

theorem find_common_difference (a : ArithSeq) (d : ℤ) : 
  (a.a 1 - a.a 0 = d) →
  d = 3 :=
sorry

theorem find_possible_a1 (a : ArithSeq) : 
  (∃a1: ℤ, a1 ∈ Set.range a.a) →
  (∀n : ℕ, S n a ≤ S 7 a) →
  Set.range a.a ∩ {n | 18 ≤ n ∧ n ≤ 20} = {18, 19, 20} :=
sorry

end find_common_difference_find_possible_a1_l125_125384


namespace passengers_on_ship_l125_125051

theorem passengers_on_ship :
  (∀ P : ℕ, 
    (P / 12) + (P / 8) + (P / 3) + (P / 6) + 35 = P) → P = 120 :=
by 
  sorry

end passengers_on_ship_l125_125051


namespace combination_multiplication_and_addition_l125_125456

theorem combination_multiplication_and_addition :
  (Nat.choose 10 3) * (Nat.choose 8 3) + (Nat.choose 5 2) = 6730 :=
by
  sorry

end combination_multiplication_and_addition_l125_125456


namespace factorization_correct_l125_125167

theorem factorization_correct (x y : ℝ) : 
  x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := 
by
  sorry

end factorization_correct_l125_125167


namespace coeff_x3_product_l125_125326

open Polynomial

noncomputable def poly1 := (C 3 * X ^ 3) + (C 2 * X ^ 2) + (C 4 * X) + (C 5)
noncomputable def poly2 := (C 4 * X ^ 3) + (C 6 * X ^ 2) + (C 5 * X) + (C 2)

theorem coeff_x3_product : coeff (poly1 * poly2) 3 = 10 := by
  sorry

end coeff_x3_product_l125_125326


namespace find_added_number_l125_125595

variable (x : ℝ) -- We define the variable x as a real number
-- We define the given conditions

def added_number (y : ℝ) : Prop :=
  (2 * (62.5 + y) / 5) - 5 = 22

theorem find_added_number : added_number x → x = 5 := by
  sorry

end find_added_number_l125_125595


namespace lana_total_spending_l125_125128

theorem lana_total_spending (ticket_price : ℕ) (tickets_friends : ℕ) (tickets_extra : ℕ)
  (H1 : ticket_price = 6)
  (H2 : tickets_friends = 8)
  (H3 : tickets_extra = 2) :
  ticket_price * (tickets_friends + tickets_extra) = 60 :=
by
  sorry

end lana_total_spending_l125_125128


namespace hexagon_perimeter_l125_125398

-- Define the length of one side of the hexagon
def side_length : ℕ := 5

-- Define the number of sides of a hexagon
def num_sides : ℕ := 6

-- Problem statement: Prove the perimeter of a regular hexagon with the given side length
theorem hexagon_perimeter (s : ℕ) (n : ℕ) : s = side_length ∧ n = num_sides → n * s = 30 :=
by sorry

end hexagon_perimeter_l125_125398


namespace distinct_arrangements_balloon_l125_125824

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l125_125824


namespace total_cost_of_shirt_and_coat_l125_125784

-- Definition of the conditions
def shirt_cost : ℕ := 150
def one_third_of_coat (coat_cost: ℕ) : Prop := shirt_cost = coat_cost / 3

-- Theorem stating the problem to prove
theorem total_cost_of_shirt_and_coat (coat_cost : ℕ) (h : one_third_of_coat coat_cost) : shirt_cost + coat_cost = 600 :=
by 
  -- Proof goes here, using sorry as placeholder
  sorry

end total_cost_of_shirt_and_coat_l125_125784


namespace sum_reciprocals_of_roots_l125_125710

theorem sum_reciprocals_of_roots (p q x₁ x₂ : ℝ) (h₀ : x₁ + x₂ = -p) (h₁ : x₁ * x₂ = q) :
  (1 / x₁ + 1 / x₂) = -p / q :=
by 
  sorry

end sum_reciprocals_of_roots_l125_125710


namespace An_nonempty_finite_l125_125162

def An (n : ℕ) : Set (ℕ × ℕ) :=
  { p : ℕ × ℕ | ∃ (k : ℕ), ∃ (a : ℕ), ∃ (b : ℕ), a = Nat.sqrt (p.1^2 + p.2 + n) ∧ b = Nat.sqrt (p.2^2 + p.1 + n) ∧ k = a + b }

theorem An_nonempty_finite (n : ℕ) (h : n ≥ 1) : Set.Nonempty (An n) ∧ Set.Finite (An n) :=
by
  sorry -- The proof goes here

end An_nonempty_finite_l125_125162


namespace moles_of_ammonia_combined_l125_125173

theorem moles_of_ammonia_combined (n_CO2 n_Urea n_NH3 : ℕ) (h1 : n_CO2 = 1) (h2 : n_Urea = 1) (h3 : n_Urea = n_CO2)
  (h4 : n_Urea = 2 * n_NH3): n_NH3 = 2 := 
by
  sorry

end moles_of_ammonia_combined_l125_125173


namespace cars_in_parking_lot_l125_125728

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cars_in_parking_lot_l125_125728


namespace probability_fail_then_succeed_l125_125431

theorem probability_fail_then_succeed
  (P_fail_first : ℚ := 9 / 10)
  (P_succeed_second : ℚ := 1 / 9) :
  P_fail_first * P_succeed_second = 1 / 10 :=
by
  sorry

end probability_fail_then_succeed_l125_125431


namespace find_m_plus_n_l125_125178

variable (x n m : ℝ)

def condition : Prop := (x + 5) * (x + n) = x^2 + m * x - 5

theorem find_m_plus_n (hnm : condition x n m) : m + n = 3 := 
sorry

end find_m_plus_n_l125_125178


namespace factor_x4_plus_81_l125_125241

theorem factor_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) :=
by 
  -- The proof is omitted.
  sorry

end factor_x4_plus_81_l125_125241


namespace business_ownership_l125_125140

variable (x : ℝ) (total_value : ℝ)
variable (fraction_sold : ℝ)
variable (sale_amount : ℝ)

-- Conditions
axiom total_value_condition : total_value = 10000
axiom fraction_sold_condition : fraction_sold = 3 / 5
axiom sale_amount_condition : sale_amount = 2000
axiom equation_condition : (fraction_sold * x * total_value = sale_amount)

theorem business_ownership : x = 1 / 3 := by 
  have hv := total_value_condition
  have hf := fraction_sold_condition
  have hs := sale_amount_condition
  have he := equation_condition
  sorry

end business_ownership_l125_125140


namespace clowns_attended_l125_125621

-- Definition of the problem's conditions
def num_children : ℕ := 30
def initial_candies : ℕ := 700
def candies_sold_per_person : ℕ := 20
def remaining_candies : ℕ := 20

-- Main theorem stating that 4 clowns attended the carousel
theorem clowns_attended (num_clowns : ℕ) (candies_left: num_clowns * candies_sold_per_person + num_children * candies_sold_per_person = initial_candies - remaining_candies) : num_clowns = 4 := by
  sorry

end clowns_attended_l125_125621


namespace mass_percentage_of_H_in_H2O_is_11_19_l125_125327

def mass_of_hydrogen : Float := 1.008
def mass_of_oxygen : Float := 16.00
def mass_of_H2O : Float := 2 * mass_of_hydrogen + mass_of_oxygen
def mass_percentage_hydrogen : Float :=
  (2 * mass_of_hydrogen / mass_of_H2O) * 100

theorem mass_percentage_of_H_in_H2O_is_11_19 :
  mass_percentage_hydrogen = 11.19 :=
  sorry

end mass_percentage_of_H_in_H2O_is_11_19_l125_125327


namespace faster_speed_l125_125303

theorem faster_speed (x : ℝ) (h1 : 10 ≠ 0) (h2 : 5 * 10 = 50) (h3 : 50 + 20 = 70) (h4 : 5 = 70 / x) : x = 14 :=
by
  -- proof steps go here
  sorry

end faster_speed_l125_125303


namespace solve_for_angle_a_l125_125874

theorem solve_for_angle_a (a b c d e : ℝ) (h1 : a + b + c + d = 360) (h2 : e = 360 - (a + d)) : a = 360 - e - b - c :=
by
  sorry

end solve_for_angle_a_l125_125874


namespace greg_rolls_more_ones_than_fives_l125_125052

def probability_more_ones_than_fives (n : ℕ) : ℚ :=
  if n = 6 then 695 / 1944 else 0

theorem greg_rolls_more_ones_than_fives :
  probability_more_ones_than_fives 6 = 695 / 1944 :=
by sorry

end greg_rolls_more_ones_than_fives_l125_125052


namespace geometric_sequence_S8_l125_125534

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l125_125534


namespace highest_possible_average_l125_125392

theorem highest_possible_average (average_score : ℕ) (total_tests : ℕ) (lowest_score : ℕ) 
  (total_marks : ℕ := total_tests * average_score)
  (new_total_tests : ℕ := total_tests - 1)
  (resulting_average : ℚ := (total_marks - lowest_score) / new_total_tests) :
  average_score = 68 ∧ total_tests = 9 ∧ lowest_score = 0 → resulting_average = 76.5 := sorry

end highest_possible_average_l125_125392


namespace arc_length_TQ_l125_125498

-- Definitions from the conditions
def center (O : Type) : Prop := true
def inscribedAngle (T I Q : Type) (angle : ℝ) := angle = 45
def radius (T : Type) (len : ℝ) := len = 12

-- Theorem to be proved
theorem arc_length_TQ (O T I Q : Type) (r : ℝ) (angle : ℝ) 
  (h_center : center O) 
  (h_angle : inscribedAngle T I Q angle)
  (h_radius : radius T r) :
  ∃ l : ℝ, l = 6 * Real.pi := 
sorry

end arc_length_TQ_l125_125498


namespace volume_of_Q_3_l125_125388

noncomputable def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 2       -- 1 + 1
  | 2 => 2 + 3 / 16
  | 3 => (2 + 3 / 16) + 3 / 64
  | _ => sorry -- This handles cases n >= 4, which we don't need.

theorem volume_of_Q_3 : Q 3 = 143 / 64 := by
  sorry

end volume_of_Q_3_l125_125388


namespace behavior_on_neg_interval_l125_125363

variable (f : ℝ → ℝ)

-- condition 1: f is an odd function
def odd_function : Prop :=
  ∀ x, f (-x) = -f x

-- condition 2: f is increasing on [3, 7]
def increasing_3_7 : Prop :=
  ∀ x y, (3 ≤ x ∧ x < y ∧ y ≤ 7) → f x < f y

-- condition 3: minimum value of f on [3, 7] is 5
def minimum_3_7 : Prop :=
  ∃ a, 3 ≤ a ∧ a ≤ 7 ∧ f a = 5

-- Use the above conditions to prove the required property on [-7, -3].
theorem behavior_on_neg_interval 
  (h1 : odd_function f) 
  (h2 : increasing_3_7 f) 
  (h3 : minimum_3_7 f) : 
  (∀ x y, (-7 ≤ x ∧ x < y ∧ y ≤ -3) → f x < f y) 
  ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ -5 :=
sorry

end behavior_on_neg_interval_l125_125363


namespace coloring_ways_l125_125343

theorem coloring_ways : 
  let colorings (n : ℕ) := {f : fin n → fin n → bool // ∀ x y, f x y ≠ f (x + 1) y ∧ f x y ≠ f x (y + 1)} in
  let valid (f : fin 10 → fin 10 → bool) :=
    ∀ i j, (f i j = f (i + 1) (j + 1)) ∧ (f i (j + 1) ≠ f (i + 1) j) in
  lift₂ (λ (coloring : colorings 10) (_ : valid coloring),
    (card colorings 10) - 2) = 2046 :=
by sorry

end coloring_ways_l125_125343


namespace f_neg_9_over_2_f_in_7_8_l125_125645

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (x + 1) else sorry

theorem f_neg_9_over_2 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  f (-9 / 2) = -1 / 3 :=
by
  have h_period : f (-9 / 2) = f (-1 / 2) := by
    sorry  -- Using periodicity property
  have h_odd1 : f (-1 / 2) = -f (1 / 2) := by
    sorry  -- Using odd function property
  have h_def : f (1 / 2) = 1 / 3 := by
    sorry  -- Using the definition of f(x) for x in [0, 1)
  rw [h_period, h_odd1, h_def]
  norm_num

theorem f_in_7_8 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  ∀ x : ℝ, (7 < x ∧ x ≤ 8) → f x = - (x - 8) / (x - 9) :=
by
  intro x hx
  have h_period : f x = f (x - 8) := by
    sorry  -- Using periodicity property
  sorry  -- Apply the negative intervals and substitution to achieve the final form

end f_neg_9_over_2_f_in_7_8_l125_125645


namespace johns_age_l125_125886

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l125_125886


namespace probability_wheel_l125_125146

theorem probability_wheel (P : ℕ → ℚ) 
  (hA : P 0 = 1/4) 
  (hB : P 1 = 1/3) 
  (hC : P 2 = 1/6) 
  (hSum : P 0 + P 1 + P 2 + P 3 = 1) : 
  P 3 = 1/4 := 
by 
  -- Proof here
  sorry

end probability_wheel_l125_125146


namespace max_students_for_distribution_l125_125126

theorem max_students_for_distribution : 
  ∃ (n : Nat), (∀ k, k ∣ 1048 ∧ k ∣ 828 → k ≤ n) ∧ 
               (n ∣ 1048 ∧ n ∣ 828) ∧ 
               n = 4 :=
by
  sorry

end max_students_for_distribution_l125_125126


namespace manufacturing_department_percentage_l125_125590

theorem manufacturing_department_percentage (total_degrees mfg_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : mfg_degrees = 162) : (mfg_degrees / total_degrees) * 100 = 45 :=
by 
  sorry

end manufacturing_department_percentage_l125_125590


namespace rent_percentage_l125_125589

variable (E : ℝ)
variable (last_year_rent : ℝ := 0.20 * E)
variable (this_year_earnings : ℝ := 1.20 * E)
variable (this_year_rent : ℝ := 0.30 * this_year_earnings)

theorem rent_percentage (E : ℝ) (h_last_year_rent : last_year_rent = 0.20 * E)
  (h_this_year_earnings : this_year_earnings = 1.20 * E)
  (h_this_year_rent : this_year_rent = 0.30 * this_year_earnings) : 
  this_year_rent / last_year_rent * 100 = 180 := by
  sorry

end rent_percentage_l125_125589


namespace avg_cost_is_12_cents_l125_125987

noncomputable def avg_cost_per_pencil 
    (price_per_package : ℝ)
    (num_pencils : ℕ)
    (shipping_cost : ℝ)
    (discount_rate : ℝ) : ℝ :=
  let price_after_discount := price_per_package - (discount_rate * price_per_package)
  let total_cost := price_after_discount + shipping_cost
  let total_cost_cents := total_cost * 100
  total_cost_cents / num_pencils

theorem avg_cost_is_12_cents :
  avg_cost_per_pencil 29.70 300 8.50 0.10 = 12 := 
by {
  sorry
}

end avg_cost_is_12_cents_l125_125987


namespace susan_age_in_5_years_l125_125984

variable (J N S X : ℕ)

-- Conditions
axiom h1 : J - 8 = 2 * (N - 8)
axiom h2 : J + X = 37
axiom h3 : S = N - 3

-- Theorem statement
theorem susan_age_in_5_years : S + 5 = N + 2 :=
by sorry

end susan_age_in_5_years_l125_125984


namespace longer_side_length_l125_125312

theorem longer_side_length (total_rope_length shorter_side_length longer_side_length : ℝ) 
  (h1 : total_rope_length = 100) 
  (h2 : shorter_side_length = 22) 
  : 2 * shorter_side_length + 2 * longer_side_length = total_rope_length -> longer_side_length = 28 :=
by sorry

end longer_side_length_l125_125312


namespace geom_prog_common_ratio_l125_125671

variable {α : Type*} [Field α]

theorem geom_prog_common_ratio (x y z r : α) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (h1 : x * (y + z) = a) (h2 : y * (z + x) = a * r) (h3 : z * (x + y) = a * r^2) :
  r^2 + r + 1 = 0 :=
by
  sorry

end geom_prog_common_ratio_l125_125671


namespace quadratic_inequality_l125_125027

theorem quadratic_inequality (t x₁ x₂ : ℝ) (α β : ℝ)
  (ht : (2 * x₁^2 - t * x₁ - 2 = 0) ∧ (2 * x₂^2 - t * x₂ - 2 = 0))
  (hx : α ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ β)
  (hαβ : α < β)
  (roots : α + β = t / 2 ∧ α * β = -1) :
  4*x₁*x₂ - t*(x₁ + x₂) - 4 < 0 := 
sorry

end quadratic_inequality_l125_125027


namespace tensor_op_correct_l125_125779

-- Define the operation ⊗
def tensor_op (x y : ℝ) : ℝ := x^2 + y

-- Goal: Prove h ⊗ (h ⊗ h) = 2h^2 + h for some h in ℝ
theorem tensor_op_correct (h : ℝ) : tensor_op h (tensor_op h h) = 2 * h^2 + h :=
by
  sorry

end tensor_op_correct_l125_125779


namespace abc_cubed_sum_l125_125918

theorem abc_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
    a^3 + b^3 + c^3 = -36 :=
by sorry

end abc_cubed_sum_l125_125918


namespace jack_change_l125_125878

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end jack_change_l125_125878


namespace area_outside_squares_inside_triangle_l125_125157

noncomputable def side_length_large_square : ℝ := 6
noncomputable def side_length_small_square1 : ℝ := 2
noncomputable def side_length_small_square2 : ℝ := 3
noncomputable def area_large_square := side_length_large_square ^ 2
noncomputable def area_small_square1 := side_length_small_square1 ^ 2
noncomputable def area_small_square2 := side_length_small_square2 ^ 2
noncomputable def area_triangle_EFG := area_large_square / 2
noncomputable def total_area_small_squares := area_small_square1 + area_small_square2

theorem area_outside_squares_inside_triangle :
  (area_triangle_EFG - total_area_small_squares) = 5 :=
by
  sorry

end area_outside_squares_inside_triangle_l125_125157


namespace part_1_part_2_l125_125541

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_1 (x : ℝ) : f x ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

theorem part_2 (b : ℝ) (h₁ : b ≠ 0) (x : ℝ) (h₂ : f x ≥ (|2 * b + 1| + |1 - b|) / |b|) : x ≤ -1.5 :=
by sorry

end part_1_part_2_l125_125541


namespace trapezoid_proof_l125_125548

variables {Point : Type} [MetricSpace Point]

-- Definitions of the points and segments as given conditions.
variables (A B C D E : Point)

-- Definitions representing the trapezoid and point E's property.
def is_trapezoid (ABCD : (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)) : Prop :=
  (A ≠ B) ∧ (C ≠ D)

def on_segment (E : Point) (A D : Point) : Prop :=
  -- This definition will encompass the fact that E is on segment AD.
  -- Representing the notion that E lies between A and D.
  dist A E + dist E D = dist A D

def equal_perimeters (E : Point) (A B C D : Point) : Prop :=
  let p1 := (dist A B + dist B E + dist E A)
  let p2 := (dist B C + dist C E + dist E B)
  let p3 := (dist C D + dist D E + dist E C)
  p1 = p2 ∧ p2 = p3

-- The theorem we need to prove.
theorem trapezoid_proof (ABCD : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) (onSeg : on_segment E A D) (eqPerim : equal_perimeters E A B C D) : 
  dist B C = dist A D / 2 :=
sorry

end trapezoid_proof_l125_125548


namespace dogs_not_liking_any_food_l125_125059

-- Declare variables
variable (n w s ws c cs : ℕ)

-- Define problem conditions
def total_dogs := n
def dogs_like_watermelon := w
def dogs_like_salmon := s
def dogs_like_watermelon_and_salmon := ws
def dogs_like_chicken := c
def dogs_like_chicken_and_salmon_but_not_watermelon := cs

-- Define the statement proving the number of dogs that do not like any of the three foods
theorem dogs_not_liking_any_food : 
  n = 75 → 
  w = 15 → 
  s = 54 → 
  ws = 12 → 
  c = 20 → 
  cs = 7 → 
  (75 - ((w - ws) + (s - ws - cs) + (c - cs) + ws + cs) = 5) :=
by
  intros _ _ _ _ _ _
  sorry

end dogs_not_liking_any_food_l125_125059


namespace num_4_digit_odd_distinct_l125_125545

theorem num_4_digit_odd_distinct : 
  ∃ n : ℕ, n = 5 * 4 * 3 * 2 :=
sorry

end num_4_digit_odd_distinct_l125_125545


namespace sum_of_squares_of_sums_l125_125220

axiom roots_of_polynomial (p q r : ℝ) : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0

theorem sum_of_squares_of_sums (p q r : ℝ)
  (h_roots : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := 
sorry

end sum_of_squares_of_sums_l125_125220


namespace circle_radius_increase_l125_125559

-- Defining the problem conditions and the resulting proof
theorem circle_radius_increase (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = n * (Real.sqrt 3 - 1) / 2 :=
sorry  -- Proof is left as an exercise

end circle_radius_increase_l125_125559


namespace distinct_arrangements_balloon_l125_125807

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l125_125807


namespace no_solution_5x_plus_2_eq_17y_l125_125933

theorem no_solution_5x_plus_2_eq_17y :
  ¬∃ (x y : ℕ), 5^x + 2 = 17^y :=
sorry

end no_solution_5x_plus_2_eq_17y_l125_125933


namespace f_is_odd_and_increasing_l125_125191

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_is_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_is_odd_and_increasing_l125_125191


namespace scientific_notation_of_448000_l125_125573

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l125_125573


namespace prism_diagonal_and_surface_area_l125_125433

/-- 
  A rectangular prism has dimensions of 12 inches, 16 inches, and 21 inches.
  Prove that the length of the diagonal is 29 inches, 
  and the total surface area of the prism is 1560 square inches.
-/
theorem prism_diagonal_and_surface_area :
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  d = 29 ∧ S = 1560 := by
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  sorry

end prism_diagonal_and_surface_area_l125_125433


namespace integer_diff_of_squares_1_to_2000_l125_125041

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l125_125041


namespace diamond_4_3_l125_125618

def diamond (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem diamond_4_3 : diamond 4 3 = 1 :=
by
  -- The proof will go here.
  sorry

end diamond_4_3_l125_125618


namespace tom_fractions_l125_125266

theorem tom_fractions (packages : ℕ) (cars_per_package : ℕ) (cars_left : ℕ) (nephews : ℕ) :
  packages = 10 → 
  cars_per_package = 5 → 
  cars_left = 30 → 
  nephews = 2 → 
  ∃ fraction_given : ℚ, fraction_given = 1/5 :=
by
  intros
  sorry

end tom_fractions_l125_125266


namespace claire_earnings_l125_125612

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end claire_earnings_l125_125612


namespace exists_prime_among_15_numbers_l125_125982

theorem exists_prime_among_15_numbers 
    (integers : Fin 15 → ℕ)
    (h1 : ∀ i, 1 < integers i)
    (h2 : ∀ i, integers i < 1998)
    (h3 : ∀ i j, i ≠ j → Nat.gcd (integers i) (integers j) = 1) :
    ∃ i, Nat.Prime (integers i) :=
by
  sorry

end exists_prime_among_15_numbers_l125_125982


namespace particle_jumps_distinct_sequences_l125_125302

/-- Given a particle that starts at the origin and makes 5 jumps along the x-axis,
where each jump is either +1 or -1 units. The particle ends up at point (3,0).
Prove that there are 5 distinct sequences of such jumps. -/
theorem particle_jumps_distinct_sequences :
  set.count { seq | (∃ (jumps : ℕ) (positive: Finset ℤ), positive.card = 4 ∧ jumps - 4 = 3) } = 5 :=
sorry

end particle_jumps_distinct_sequences_l125_125302


namespace percent_eighth_graders_combined_l125_125603

theorem percent_eighth_graders_combined (p_students : ℕ) (m_students : ℕ)
  (p_grade8_percent : ℚ) (m_grade8_percent : ℚ) :
  p_students = 160 → m_students = 250 →
  p_grade8_percent = 18 / 100 → m_grade8_percent = 22 / 100 →
  100 * (p_grade8_percent * p_students + m_grade8_percent * m_students) / (p_students + m_students) = 20 := 
by
  intros h1 h2 h3 h4
  sorry

end percent_eighth_graders_combined_l125_125603


namespace minimum_knights_l125_125265

/-!
Problem: There are 1001 people seated around a round table. Each person is either a knight (always tells the truth) or a liar (always lies). Next to each knight, there is exactly one liar, and next to each liar, there is exactly one knight. Prove that the minimum number of knights is 502.
-/

def person := Type
def is_knight (p : person) : Prop := sorry
def is_liar (p : person) : Prop := sorry

axiom round_table (persons : list person) : (∀ (p : person),
  (is_knight p → (∃! q : person, is_liar q ∧ (q = list.nth_le persons ((list.index_of p persons + 1) % 1001) sorry ∨ q = list.nth_le persons ((list.index_of p persons - 1 + 1001) % 1001) sorry))) ∧
  (is_liar p → (∃! k : person, is_knight k ∧ (k = list.nth_le persons ((list.index_of p persons + 1) % 1001) sorry ∨ k = list.nth_le persons ((list.index_of p persons - 1 + 1001) % 1001) sorry))))

theorem minimum_knights (persons : list person) (h : persons.length = 1001) : 
  (∃ (knights : list person), (∀ k ∈ knights, is_knight k) ∧ (∀ l ∉ knights, is_liar l) ∧ knights.length = 502) :=
sorry

end minimum_knights_l125_125265


namespace least_k_divisible_by_240_l125_125202

theorem least_k_divisible_by_240 : ∃ (k : ℕ), k^2 % 240 = 0 ∧ k = 60 :=
by
  sorry

end least_k_divisible_by_240_l125_125202


namespace coefficient_x2y3_in_expansion_l125_125787

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem coefficient_x2y3_in_expansion (x y : ℝ) : 
  binomial 5 3 * (2 : ℝ) ^ 2 * (-1 : ℝ) ^ 3 = -40 := by
sorry

end coefficient_x2y3_in_expansion_l125_125787


namespace average_is_12_or_15_l125_125616

variable {N : ℝ} (h : 12 < N ∧ N < 22)

theorem average_is_12_or_15 : (∃ x ∈ ({12, 15} : Set ℝ), x = (9 + 15 + N) / 3) :=
by
  have h1 : 12 < (24 + N) / 3 := by sorry
  have h2 : (24 + N) / 3 < 15.3333 := by sorry
  sorry

end average_is_12_or_15_l125_125616


namespace sport_tournament_attendance_l125_125868

theorem sport_tournament_attendance :
  let total_attendance := 500
  let team_A_supporters := 0.35 * total_attendance
  let team_B_supporters := 0.25 * total_attendance
  let team_C_supporters := 0.20 * total_attendance
  let team_D_supporters := 0.15 * total_attendance
  let AB_overlap := 0.10 * team_A_supporters
  let BC_overlap := 0.05 * team_B_supporters
  let CD_overlap := 0.07 * team_C_supporters
  let atmosphere_attendees := 30
  let total_supporters := team_A_supporters + team_B_supporters + team_C_supporters + team_D_supporters
                         - (AB_overlap + BC_overlap + CD_overlap)
  let unsupported_people := total_attendance - total_supporters - atmosphere_attendees
  unsupported_people = 26 :=
by
  sorry

end sport_tournament_attendance_l125_125868


namespace new_mean_rent_l125_125088

theorem new_mean_rent (avg_rent : ℕ) (num_friends : ℕ) (rent_increase_pct : ℕ) (initial_rent : ℕ) :
  avg_rent = 800 →
  num_friends = 4 →
  rent_increase_pct = 25 →
  initial_rent = 800 →
  (avg_rent * num_friends + initial_rent * rent_increase_pct / 100) / num_friends = 850 :=
by
  intros h_avg h_num h_pct h_init
  sorry

end new_mean_rent_l125_125088


namespace find_value_in_box_l125_125973

theorem find_value_in_box (x : ℕ) :
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * x ↔ x = 50 := by
  sorry

end find_value_in_box_l125_125973


namespace y_coordinate_equidistant_l125_125274

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ P : ℝ × ℝ, P = (0, y) → dist (3, 0) P = dist (2, 5) P) ∧ y = 2 := 
by
  sorry

end y_coordinate_equidistant_l125_125274


namespace first_player_winning_strategy_l125_125270

theorem first_player_winning_strategy (num_chips : ℕ) : 
  (num_chips = 110) → 
  ∃ (moves : ℕ → ℕ × ℕ), (∀ n, 1 ≤ (moves n).1 ∧ (moves n).1 ≤ 9) ∧ 
  (∀ n, (moves n).1 ≠ (moves (n-1)).1) →
  (∃ move_sequence : ℕ → ℕ, ∀ k, move_sequence k ≤ num_chips ∧ 
  ((move_sequence (k+1) < move_sequence k) ∨ (move_sequence (k+1) = 0 ∧ move_sequence k = 1)) ∧ 
  (move_sequence k > 0) ∧ (move_sequence 0 = num_chips) →
  num_chips ≡ 14 [MOD 32]) :=
by 
  sorry

end first_player_winning_strategy_l125_125270


namespace product_of_numbers_l125_125969

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 30 * k) : 
  x * y = 400 / 7 := 
sorry

end product_of_numbers_l125_125969


namespace extremum_implies_a_max_min_values_in_interval_l125_125024

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x

-- Problem statements
theorem extremum_implies_a (a : ℝ) (h : ∃ x, x = 1 ∧ (f' x = 0)) : a = 3 := by
  sorry

theorem max_min_values_in_interval (a : ℝ) (h : a = 3) : 
  ∃ (M m : ℝ), 
    (∀ x ∈ Icc (0:ℝ) (2:ℝ), f x 3 ≤ M) ∧ 
    (∀ x ∈ Icc (0:ℝ) (2:ℝ), f x 3 ≥ m) ∧
    M = 2 ∧ m = -2 := by
  sorry

end extremum_implies_a_max_min_values_in_interval_l125_125024


namespace find_numbers_l125_125009

theorem find_numbers (x y : ℝ) (h₁ : x + y = x * y) (h₂ : x * y = x / y) :
  (x = 1 / 2) ∧ (y = -1) := by
  sorry

end find_numbers_l125_125009


namespace distinct_arrangements_balloon_l125_125809

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l125_125809


namespace determine_time_Toronto_l125_125408

noncomputable def timeDifferenceBeijingToronto: ℤ := -12

def timeBeijing: ℕ × ℕ := (1, 8) -- (day, hour) format for simplicity: October 1st, 8:00

def timeToronto: ℕ × ℕ := (30, 20) -- Expected result in (day, hour): September 30th, 20:00

theorem determine_time_Toronto :
  timeDifferenceBeijingToronto = -12 →
  timeBeijing = (1, 8) →
  timeToronto = (30, 20) :=
by
  -- proof to be written 
  sorry

end determine_time_Toronto_l125_125408


namespace find_weight_b_l125_125946

theorem find_weight_b (A B C : ℕ) 
  (h1 : A + B + C = 90)
  (h2 : A + B = 50)
  (h3 : B + C = 56) : 
  B = 16 :=
sorry

end find_weight_b_l125_125946


namespace age_difference_problem_l125_125403

theorem age_difference_problem 
    (minimum_age : ℕ := 25)
    (current_age_Jane : ℕ := 28)
    (years_ahead : ℕ := 6)
    (Dara_age_in_6_years : ℕ := (current_age_Jane + years_ahead) / 2):
    minimum_age - (Dara_age_in_6_years - years_ahead) = 14 :=
by
  -- all definition parts: minimum_age, current_age_Jane, years_ahead,
  -- Dara_age_in_6_years are present
  sorry

end age_difference_problem_l125_125403


namespace total_canoes_built_l125_125451

-- Definition of the conditions as suggested by the problem
def num_canoes_in_february : Nat := 5
def growth_rate : Nat := 3
def number_of_months : Nat := 5

-- Final statement to prove
theorem total_canoes_built : (num_canoes_in_february * (growth_rate^number_of_months - 1)) / (growth_rate - 1) = 605 := 
by sorry

end total_canoes_built_l125_125451


namespace average_salary_of_officers_l125_125060

-- Define the given conditions
def avg_salary_total := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 480

-- Define the expected result
def avg_salary_officers := 440

-- Define the problem and statement to be proved in Lean
theorem average_salary_of_officers :
  (num_officers + num_non_officers) * avg_salary_total - num_non_officers * avg_salary_non_officers = num_officers * avg_salary_officers := 
by
  sorry

end average_salary_of_officers_l125_125060


namespace money_needed_l125_125072

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end money_needed_l125_125072


namespace certain_number_plus_two_l125_125716

theorem certain_number_plus_two (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end certain_number_plus_two_l125_125716


namespace balloon_arrangements_l125_125840

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l125_125840


namespace sphere_center_ratio_l125_125386

/-
Let O be the origin and let (a, b, c) be a fixed point.
A plane with the equation x + 2y + 3z = 6 passes through (a, b, c)
and intersects the x-axis, y-axis, and z-axis at A, B, and C, respectively, all distinct from O.
Let (p, q, r) be the center of the sphere passing through A, B, C, and O.
Prove: a / p + b / q + c / r = 2
-/
theorem sphere_center_ratio (a b c : ℝ) (p q r : ℝ)
  (h_plane : a + 2 * b + 3 * c = 6) 
  (h_p : p = 3)
  (h_q : q = 1.5)
  (h_r : r = 1) :
  a / p + b / q + c / r = 2 :=
by
  sorry

end sphere_center_ratio_l125_125386


namespace scientific_notation_of_0_0000003_l125_125308

theorem scientific_notation_of_0_0000003 : 0.0000003 = 3 * 10^(-7) := by
  sorry

end scientific_notation_of_0_0000003_l125_125308


namespace determine_values_of_a_and_b_l125_125186

namespace MathProofProblem

variables (a b : ℤ)

theorem determine_values_of_a_and_b :
  (b + 1 = 2) ∧ (a - 1 ≠ -3) ∧ (a - 1 = -3) ∧ (b + 1 ≠ 2) ∧ (a - 1 = 2) ∧ (b + 1 = -3) →
  a = 3 ∧ b = -4 := by
  sorry

end MathProofProblem

end determine_values_of_a_and_b_l125_125186


namespace math_problem_l125_125663

variables (a b c d m : ℤ)

theorem math_problem (h1 : a = -b) (h2 : c * d = 1) (h3 : m = -1) : c * d - a - b + m^2022 = 2 :=
by
  sorry

end math_problem_l125_125663


namespace proof_part1_proof_part2_l125_125153

-- Proof problem for the first part (1)
theorem proof_part1 (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := 
by
  sorry

-- Proof problem for the second part (2)
theorem proof_part2 (a : ℝ) : a * (a - 2) - 2 * a * (1 - 3 * a) = 7 * a^2 - 4 * a := 
by
  sorry

end proof_part1_proof_part2_l125_125153


namespace grid_rows_l125_125222

theorem grid_rows (R : ℕ) :
  let squares_per_row := 15
  let red_squares := 4 * 6
  let blue_squares := 4 * squares_per_row
  let green_squares := 66
  let total_squares := red_squares + blue_squares + green_squares 
  total_squares = squares_per_row * R →
  R = 10 :=
by
  intros
  sorry

end grid_rows_l125_125222


namespace find_a4_l125_125680

noncomputable def a (n : ℕ) : ℕ := sorry -- Define the arithmetic sequence
def S (n : ℕ) : ℕ := sorry -- Define the sum function for the sequence

theorem find_a4 (h1 : S 5 = 25) (h2 : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l125_125680


namespace necessary_without_sufficient_for_parallel_lines_l125_125185

noncomputable def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 2 = 0
noncomputable def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 1 = 0

theorem necessary_without_sufficient_for_parallel_lines :
  (∀ (a : ℝ), a = 2 → (∀ (x y : ℝ), line1 a x y → line2 a x y)) ∧ 
  ¬ (∀ (a : ℝ), (∀ (x y : ℝ), line1 a x y → line2 a x y) → a = 2) :=
sorry

end necessary_without_sufficient_for_parallel_lines_l125_125185


namespace sqrt_7_irrational_l125_125740

theorem sqrt_7_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a: ℝ) / b = Real.sqrt 7 := by
  sorry

end sqrt_7_irrational_l125_125740


namespace ratio_of_larger_to_smaller_l125_125096

theorem ratio_of_larger_to_smaller
  (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h3 : x + y = 6 * (x - y)) :
  x / y = 7 / 5 :=
by sorry

end ratio_of_larger_to_smaller_l125_125096


namespace probability_of_edge_endpoints_in_icosahedron_l125_125268

theorem probability_of_edge_endpoints_in_icosahedron :
  let vertices := 12
  let edges := 30
  let connections_per_vertex := 5
  (5 / (vertices - 1)) = (5 / 11) := by
  sorry

end probability_of_edge_endpoints_in_icosahedron_l125_125268


namespace sum_of_squares_l125_125972

theorem sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 :=
by
  sorry

end sum_of_squares_l125_125972


namespace distinct_arrangements_balloon_l125_125810

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125810


namespace balloon_arrangements_l125_125829

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l125_125829


namespace ducks_in_marsh_l125_125103

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) : total_birds - geese = 37 := by
  sorry

end ducks_in_marsh_l125_125103


namespace slope_range_of_tangent_line_l125_125174

theorem slope_range_of_tangent_line (x : ℝ) (h : x ≠ 0) : (1 - 1/(x^2)) < 1 :=
by
  calc 
    1 - 1/(x^2) < 1 := sorry

end slope_range_of_tangent_line_l125_125174


namespace large_rect_area_is_294_l125_125634

-- Define the dimensions of the smaller rectangles
def shorter_side : ℕ := 7
def longer_side : ℕ := 2 * shorter_side

-- Condition 1: Each smaller rectangle has a shorter side measuring 7 feet
axiom smaller_rect_shorter_side : ∀ (r : ℕ), r = shorter_side → r = 7

-- Condition 4: The longer side of each smaller rectangle is twice the shorter side
axiom smaller_rect_longer_side : ∀ (r : ℕ), r = longer_side → r = 2 * shorter_side

-- Condition 2: Three rectangles are aligned vertically
def vertical_height : ℕ := 3 * shorter_side

-- Condition 3: One rectangle is aligned horizontally adjoining them
def horizontal_length : ℕ := longer_side

-- The dimensions of the larger rectangle EFGH
def large_rect_width : ℕ := vertical_height
def large_rect_length : ℕ := horizontal_length

-- Calculate the area of the larger rectangle EFGH
def large_rect_area : ℕ := large_rect_width * large_rect_length

-- Prove that the area of the large rectangle is 294 square feet
theorem large_rect_area_is_294 : large_rect_area = 294 := by
  sorry

end large_rect_area_is_294_l125_125634


namespace cost_of_pencils_and_notebooks_l125_125635

variable (P N : ℝ)

theorem cost_of_pencils_and_notebooks
  (h1 : 4 * P + 3 * N = 9600)
  (h2 : 2 * P + 2 * N = 5400) :
  8 * P + 7 * N = 20400 := by
  sorry

end cost_of_pencils_and_notebooks_l125_125635


namespace no_solution_xyz_l125_125170

theorem no_solution_xyz : ∀ (x y z : Nat), (1 ≤ x) → (x ≤ 9) → (0 ≤ y) → (y ≤ 9) → (0 ≤ z) → (z ≤ 9) →
    100 * x + 10 * y + z ≠ 10 * x * y + x * z :=
by
  intros x y z hx1 hx9 hy1 hy9 hz1 hz9
  sorry

end no_solution_xyz_l125_125170


namespace distinct_arrangements_balloon_l125_125837

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l125_125837


namespace tom_total_distance_l125_125967

/-- Tom swims for 1.5 hours at 2.5 miles per hour. 
    Tom runs for 0.75 hours at 6.5 miles per hour. 
    Tom bikes for 3 hours at 12 miles per hour. 
    The total distance Tom covered is 44.625 miles.
-/
theorem tom_total_distance
  (swim_time : ℝ := 1.5) (swim_speed : ℝ := 2.5)
  (run_time : ℝ := 0.75) (run_speed : ℝ := 6.5)
  (bike_time : ℝ := 3) (bike_speed : ℝ := 12) :
  swim_time * swim_speed + run_time * run_speed + bike_time * bike_speed = 44.625 :=
by
  sorry

end tom_total_distance_l125_125967


namespace fraction_of_yard_occupied_l125_125143

/-
Proof Problem: Given a rectangular yard that measures 30 meters by 8 meters and contains
an isosceles trapezoid-shaped flower bed with parallel sides measuring 14 meters and 24 meters,
and a height of 6 meters, prove that the fraction of the yard occupied by the flower bed is 19/40.
-/

theorem fraction_of_yard_occupied (length_yard width_yard b1 b2 h area_trapezoid area_yard : ℝ) 
  (h_length_yard : length_yard = 30) 
  (h_width_yard : width_yard = 8) 
  (h_b1 : b1 = 14) 
  (h_b2 : b2 = 24) 
  (h_height_trapezoid : h = 6) 
  (h_area_trapezoid : area_trapezoid = (1/2) * (b1 + b2) * h) 
  (h_area_yard : area_yard = length_yard * width_yard) : 
  area_trapezoid / area_yard = 19 / 40 := 
by {
  -- Follow-up steps to prove the statement would go here
  sorry
}

end fraction_of_yard_occupied_l125_125143


namespace factor_expression_eq_l125_125775

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

end factor_expression_eq_l125_125775


namespace balloon_permutations_l125_125820

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l125_125820


namespace problem_statement_l125_125161

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end problem_statement_l125_125161


namespace min_filtration_cycles_l125_125869

theorem min_filtration_cycles {c₀ : ℝ} (initial_concentration : c₀ = 225)
  (max_concentration : ℝ := 7.5) (reduction_factor : ℝ := 1 / 3)
  (log2 : ℝ := 0.3010) (log3 : ℝ := 0.4771) :
  ∃ n : ℕ, (c₀ * (reduction_factor ^ n) ≤ max_concentration ∧ n ≥ 9) :=
sorry

end min_filtration_cycles_l125_125869


namespace required_C6H6_for_C6H5CH3_and_H2_l125_125325

-- Define the necessary molecular structures and stoichiometry
def C6H6 : Type := ℕ -- Benzene
def CH4 : Type := ℕ -- Methane
def C6H5CH3 : Type := ℕ -- Toluene
def H2 : Type := ℕ -- Hydrogen

-- Balanced equation condition
def balanced_reaction (x : C6H6) (y : CH4) (z : C6H5CH3) (w : H2) : Prop :=
  x = y ∧ x = z ∧ x = w

-- Given conditions
def condition (m : ℕ) : Prop :=
  balanced_reaction m m m m

theorem required_C6H6_for_C6H5CH3_and_H2 :
  ∀ (n : ℕ), condition n → n = 3 → n = 3 :=
by
  intros n h hn
  exact hn

end required_C6H6_for_C6H5CH3_and_H2_l125_125325


namespace charlie_contribution_l125_125731

theorem charlie_contribution (a b c : ℝ) (h₁ : a + b + c = 72) (h₂ : a = 1/4 * (b + c)) (h₃ : b = 1/5 * (a + c)) :
  c = 49 :=
by sorry

end charlie_contribution_l125_125731


namespace find_number_of_students_l125_125977

theorem find_number_of_students
    (S N : ℕ) 
    (h₁ : 4 * S + 3 = N)
    (h₂ : 5 * S = N + 6) : 
  S = 9 :=
by
  sorry

end find_number_of_students_l125_125977


namespace boys_in_class_l125_125864

theorem boys_in_class (g b : ℕ) 
  (h_ratio : 4 * g = 3 * b) (h_total : g + b = 28) : b = 16 :=
by
  sorry

end boys_in_class_l125_125864


namespace jeremy_age_l125_125763

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end jeremy_age_l125_125763


namespace complement_of_M_wrt_U_l125_125543

-- Definitions of the sets U and M as given in the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

-- The goal is to show the complement of M w.r.t. U is {2, 4, 6}
theorem complement_of_M_wrt_U :
  (U \ M) = {2, 4, 6} := 
by
  sorry

end complement_of_M_wrt_U_l125_125543


namespace exists_indices_l125_125793

open Nat List

theorem exists_indices (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (h1 : ∀ i : Fin m, a i ≤ n) (h2 : ∀ i j : Fin m, i ≤ j → a i ≤ a j)
  (h3 : ∀ j : Fin n, b j ≤ m) (h4 : ∀ i j : Fin n, i ≤ j → b i ≤ b j) :
  ∃ i : Fin m, ∃ j : Fin n, a i + i.val + 1 = b j + j.val + 1 := by
  sorry

end exists_indices_l125_125793


namespace S₈_proof_l125_125536

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l125_125536


namespace monotonic_increasing_range_l125_125192

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, (3*x^2 + 2*x - a) ≥ 0) ↔ (a ≤ -1/3) :=
by
  sorry

end monotonic_increasing_range_l125_125192


namespace ab_cd_is_1_or_minus_1_l125_125505

theorem ab_cd_is_1_or_minus_1 (a b c d : ℤ) (h1 : ∃ k₁ : ℤ, a = k₁ * (a * b - c * d))
  (h2 : ∃ k₂ : ℤ, b = k₂ * (a * b - c * d)) (h3 : ∃ k₃ : ℤ, c = k₃ * (a * b - c * d))
  (h4 : ∃ k₄ : ℤ, d = k₄ * (a * b - c * d)) :
  a * b - c * d = 1 ∨ a * b - c * d = -1 := 
sorry

end ab_cd_is_1_or_minus_1_l125_125505


namespace negation_of_existential_l125_125721

theorem negation_of_existential (x : ℝ) : ¬(∃ x : ℝ, x^2 - 2 * x + 3 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 3 ≤ 0 := 
by
  sorry

end negation_of_existential_l125_125721


namespace find_subtracted_value_l125_125760

theorem find_subtracted_value (x y : ℕ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 :=
by
  sorry

end find_subtracted_value_l125_125760


namespace sampling_methods_correct_l125_125133

def first_method_sampling : String :=
  "Simple random sampling"

def second_method_sampling : String :=
  "Systematic sampling"

theorem sampling_methods_correct :
  first_method_sampling = "Simple random sampling" ∧ second_method_sampling = "Systematic sampling" :=
by
  sorry

end sampling_methods_correct_l125_125133


namespace quadratic_has_distinct_real_roots_l125_125093

-- Definitions of the coefficients
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := -2

-- Definition of the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem stating the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots :
  discriminant a b c > 0 :=
by
  -- Coefficients specific to the problem
  unfold a b c
  -- Calculate the discriminant
  unfold discriminant
  -- Substitute the values and compute
  sorry -- Skipping the actual proof as per instructions

end quadratic_has_distinct_real_roots_l125_125093


namespace find_rs_l125_125930

-- Define a structure to hold the conditions
structure Conditions (r s : ℝ) : Prop :=
  (positive_r : 0 < r)
  (positive_s : 0 < s)
  (eq1 : r^3 + s^3 = 1)
  (eq2 : r^6 + s^6 = (15 / 16))

-- State the theorem
theorem find_rs (r s : ℝ) (h : Conditions r s) : rs = 1 / (48 : ℝ)^(1/3) :=
by
  sorry

end find_rs_l125_125930


namespace parametric_equation_correct_max_min_x_plus_y_l125_125686

noncomputable def parametric_equation (φ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ)

theorem parametric_equation_correct (ρ θ : ℝ) (h : ρ^2 - 4 * Real.sqrt 2 * Real.cos (θ - π/4) + 6 = 0) :
  ∃ (φ : ℝ), parametric_equation φ = ( 2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ) := 
sorry

theorem max_min_x_plus_y (P : ℝ × ℝ) (hP : ∃ (φ : ℝ), P = parametric_equation φ) :
  ∃ f : ℝ, (P.fst + P.snd) = f ∧ (f = 6 ∨ f = 2) :=
sorry

end parametric_equation_correct_max_min_x_plus_y_l125_125686


namespace johns_age_l125_125894

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l125_125894


namespace min_value_proof_l125_125203

open Real

theorem min_value_proof (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * sqrt 2 :=
by
  sorry

end min_value_proof_l125_125203


namespace sum_of_digits_l125_125013

noncomputable def digits_divisibility (C F : ℕ) : Prop :=
  (C >= 0 ∧ C <= 9 ∧ F >= 0 ∧ F <= 9) ∧
  (C + 8 + 5 + 4 + F + 7 + 2) % 9 = 0 ∧
  (100 * 4 + 10 * F + 72) % 8 = 0

theorem sum_of_digits (C F : ℕ) (h : digits_divisibility C F) : C + F = 10 :=
sorry

end sum_of_digits_l125_125013


namespace correct_operation_l125_125285

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l125_125285


namespace bijective_bounded_dist_l125_125914

open Int

theorem bijective_bounded_dist {k : ℕ} (f : ℤ → ℤ) 
    (hf_bijective : Function.Bijective f)
    (hf_property : ∀ i j : ℤ, |i - j| ≤ k → |f i - (f j)| ≤ k) :
    ∀ i j : ℤ, |f i - (f j)| = |i - j| := 
sorry

end bijective_bounded_dist_l125_125914


namespace flag_movement_distance_l125_125773

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l125_125773


namespace systems_on_second_street_l125_125228

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end systems_on_second_street_l125_125228


namespace joan_missed_games_l125_125508

-- Define the number of total games and games attended as constants
def total_games : ℕ := 864
def games_attended : ℕ := 395

-- The theorem statement: the number of missed games is equal to 469
theorem joan_missed_games : total_games - games_attended = 469 :=
by
  -- Proof goes here
  sorry

end joan_missed_games_l125_125508


namespace option_C_is_different_l125_125149

def cause_and_effect_relationship (description: String) : Prop :=
  description = "A: Great teachers produce outstanding students" ∨
  description = "B: When the water level rises, the boat goes up" ∨
  description = "D: The higher you climb, the farther you see"

def not_cause_and_effect_relationship (description: String) : Prop :=
  description = "C: The brighter the moon, the fewer the stars"

theorem option_C_is_different :
  ∀ (description: String),
  (not_cause_and_effect_relationship description) →
  ¬ cause_and_effect_relationship description :=
by intros description h1 h2; sorry

end option_C_is_different_l125_125149


namespace geometric_sequence_fraction_l125_125015

open Classical

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_fraction {a : ℕ → ℝ} {q : ℝ}
  (h₀ : ∀ n, 0 < a n)
  (h₁ : geometric_seq a q)
  (h₂ : 2 * (1 / 2 * a 2) = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_fraction_l125_125015


namespace parallel_lines_l125_125861

theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (3 * a - 1) * x - 4 * a * y - 1 = 0 → False) → 
  (a = 0 ∨ a = -1/3) :=
sorry

end parallel_lines_l125_125861
