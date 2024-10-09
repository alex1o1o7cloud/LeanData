import Mathlib

namespace max_rooks_in_cube_l1260_126073

def non_attacking_rooks (n : ℕ) (cube : ℕ × ℕ × ℕ) : ℕ :=
  if cube = (8, 8, 8) then 64 else 0

theorem max_rooks_in_cube:
  non_attacking_rooks 64 (8, 8, 8) = 64 :=
by
  -- proof by logical steps matching the provided solution, if necessary, start with sorry for placeholder
  sorry

end max_rooks_in_cube_l1260_126073


namespace cost_of_slices_eaten_by_dog_is_correct_l1260_126040

noncomputable def total_cost_before_tax : ℝ :=
  2 * 3 + 1 * 2 + 1 * 5 + 3 * 0.5 + 0.25 + 1.5 + 1.25

noncomputable def sales_tax_rate : ℝ := 0.06

noncomputable def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

noncomputable def slices : ℝ := 8

noncomputable def cost_per_slice : ℝ := total_cost_after_tax / slices

noncomputable def slices_eaten_by_dog : ℝ := 8 - 3

noncomputable def cost_of_slices_eaten_by_dog : ℝ := cost_per_slice * slices_eaten_by_dog

theorem cost_of_slices_eaten_by_dog_is_correct : 
  cost_of_slices_eaten_by_dog = 11.59 := by
    sorry

end cost_of_slices_eaten_by_dog_is_correct_l1260_126040


namespace relationship_between_a_b_c_l1260_126031

noncomputable def a : ℝ := 81 ^ 31
noncomputable def b : ℝ := 27 ^ 41
noncomputable def c : ℝ := 9 ^ 61

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_between_a_b_c_l1260_126031


namespace evaluate_binom_mul_factorial_l1260_126028

theorem evaluate_binom_mul_factorial (n : ℕ) (h : n > 0) :
  (Nat.choose (n + 2) n) * n! = ((n + 2) * (n + 1) * n!) / 2 := by
  sorry

end evaluate_binom_mul_factorial_l1260_126028


namespace find_pages_revised_twice_l1260_126077

def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_time : ℕ) (cost_revised_once : ℕ) (cost_revised_twice : ℕ) (total_cost : ℕ) :=
  ∃ (x : ℕ), 
    (total_pages - pages_revised_once - x) * cost_first_time
    + pages_revised_once * (cost_first_time + cost_revised_once)
    + x * (cost_first_time + cost_revised_once + cost_revised_once) = total_cost 

theorem find_pages_revised_twice :
  pages_revised_twice 100 35 6 4 4 860 ↔ ∃ x, x = 15 :=
by
  sorry

end find_pages_revised_twice_l1260_126077


namespace mike_needs_percentage_to_pass_l1260_126076

theorem mike_needs_percentage_to_pass :
  ∀ (mike_score marks_short max_marks : ℕ),
  mike_score = 212 → marks_short = 22 → max_marks = 780 →
  ((mike_score + marks_short : ℕ) / (max_marks : ℕ) : ℚ) * 100 = 30 :=
by
  intros mike_score marks_short max_marks Hmike Hshort Hmax
  rw [Hmike, Hshort, Hmax]
  -- Proof will be filled out here
  sorry

end mike_needs_percentage_to_pass_l1260_126076


namespace fraction_evaluation_l1260_126038

theorem fraction_evaluation : (1 / 2) + (1 / 2 * 1 / 2) = 3 / 4 := by
  sorry

end fraction_evaluation_l1260_126038


namespace second_solution_salt_percent_l1260_126006

theorem second_solution_salt_percent (S : ℝ) (x : ℝ) 
  (h1 : 0.14 * S - 0.14 * (S / 4) + (x / 100) * (S / 4) = 0.16 * S) : 
  x = 22 :=
by 
  -- Proof omitted
  sorry

end second_solution_salt_percent_l1260_126006


namespace shirt_cost_l1260_126088

variables (S : ℝ)

theorem shirt_cost (h : 2 * S + (S + 3) + (1/2) * (2 * S + S + 3) = 36) : S = 7.88 :=
sorry

end shirt_cost_l1260_126088


namespace find_new_length_l1260_126022

def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 24
def final_length_cm : ℕ := 76

theorem find_new_length : initial_length_cm - erased_length_cm = final_length_cm := by
  sorry

end find_new_length_l1260_126022


namespace total_pieces_l1260_126001

-- Define the given conditions
def pieces_eaten_per_person : ℕ := 4
def num_people : ℕ := 3

-- Theorem stating the result
theorem total_pieces (h : num_people > 0) : (num_people * pieces_eaten_per_person) = 12 := 
by
  sorry

end total_pieces_l1260_126001


namespace senior_employee_bonus_l1260_126056

theorem senior_employee_bonus (J S : ℝ) 
  (h1 : S = J + 1200)
  (h2 : J + S = 5000) : 
  S = 3100 :=
sorry

end senior_employee_bonus_l1260_126056


namespace comparison_abc_l1260_126082

variable (f : Real → Real)
variable (a b c : Real)
variable (x : Real)
variable (h_even : ∀ x, f (-x + 1) = f (x + 1))
variable (h_periodic : ∀ x, f (x + 2) = f x)
variable (h_mono : ∀ x y, 0 < x ∧ y < 1 ∧ x < y → f x < f y)
variable (h_f0 : f 0 = 0)
variable (a_def : a = f (Real.log 2))
variable (b_def : b = f (Real.log 3))
variable (c_def : c = f 0.5)

theorem comparison_abc : b > a ∧ a > c :=
sorry

end comparison_abc_l1260_126082


namespace sum_of_roots_l1260_126083

theorem sum_of_roots :
  let a := (6 : ℝ) + 3 * Real.sqrt 3
  let b := (3 : ℝ) + Real.sqrt 3
  let c := -(3 : ℝ)
  let root_sum := -b / a
  root_sum = -1 + Real.sqrt 3 / 3 := sorry

end sum_of_roots_l1260_126083


namespace part_I_part_II_l1260_126004

noncomputable def f (x : ℝ) (m : ℝ) := m - |x - 2|

theorem part_I (m : ℝ) : (∀ x, f (x + 1) m >= 0 → 0 <= x ∧ x <= 2) ↔ m = 1 := by
  sorry

theorem part_II (a b c : ℝ) (m : ℝ) : (1 / a + 1 / (2 * b) + 1 / (3 * c) = m) → (m = 1) → (a + 2 * b + 3 * c >= 9) := by
  sorry

end part_I_part_II_l1260_126004


namespace jordan_rectangle_width_l1260_126099

noncomputable def carol_length : ℝ := 4.5
noncomputable def carol_width : ℝ := 19.25
noncomputable def jordan_length : ℝ := 3.75

noncomputable def carol_area : ℝ := carol_length * carol_width
noncomputable def jordan_width : ℝ := carol_area / jordan_length

theorem jordan_rectangle_width : jordan_width = 23.1 := by
  -- proof will go here
  sorry

end jordan_rectangle_width_l1260_126099


namespace small_triangle_perimeter_l1260_126079

theorem small_triangle_perimeter (P : ℕ) (P₁ : ℕ) (P₂ : ℕ) (P₃ : ℕ)
  (h₁ : P = 11) (h₂ : P₁ = 5) (h₃ : P₂ = 7) (h₄ : P₃ = 9) :
  (P₁ + P₂ + P₃) - P = 10 :=
by
  sorry

end small_triangle_perimeter_l1260_126079


namespace functional_expression_result_l1260_126049

theorem functional_expression_result {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y) :
  ∀ t : ℝ, (f (4 * t) - f t) / (f (3 * t) - f (2 * t)) = 3 :=
sorry

end functional_expression_result_l1260_126049


namespace jessica_not_work_days_l1260_126019

theorem jessica_not_work_days:
  ∃ (x y z : ℕ), 
    (x + y + z = 30) ∧
    (80 * x - 40 * y + 40 * z = 1600) ∧
    (z = 5) ∧
    (y = 5) :=
by
  sorry

end jessica_not_work_days_l1260_126019


namespace greatest_number_of_roses_l1260_126070

noncomputable def individual_rose_price: ℝ := 2.30
noncomputable def dozen_rose_price: ℝ := 36
noncomputable def two_dozen_rose_price: ℝ := 50
noncomputable def budget: ℝ := 680

theorem greatest_number_of_roses (P: ℝ → ℝ → ℝ → ℝ → ℕ) :
  P individual_rose_price dozen_rose_price two_dozen_rose_price budget = 325 :=
sorry

end greatest_number_of_roses_l1260_126070


namespace islanders_liars_count_l1260_126009

def number_of_liars (N : ℕ) : ℕ :=
  if N = 30 then 28 else 0

theorem islanders_liars_count : number_of_liars 30 = 28 :=
  sorry

end islanders_liars_count_l1260_126009


namespace geom_progr_sum_eq_l1260_126000

variable (a b q : ℝ) (n p : ℕ)

theorem geom_progr_sum_eq (h : a * (1 - q ^ (n * p)) / (1 - q) = b * (1 - q ^ (n * p)) / (1 - q ^ p)) :
  b = a * (1 - q ^ p) / (1 - q) :=
by
  sorry

end geom_progr_sum_eq_l1260_126000


namespace total_distance_traveled_l1260_126037

theorem total_distance_traveled (d : ℝ) (h1 : d/3 + d/4 + d/5 = 47/60) : 3 * d = 3 :=
by
  sorry

end total_distance_traveled_l1260_126037


namespace identify_functions_l1260_126005

-- Define the first expression
def expr1 (x : ℝ) : ℝ := x - (x - 3)

-- Define the second expression
noncomputable def expr2 (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (1 - x)

-- Define the third expression
noncomputable def expr3 (x : ℝ) : ℝ :=
if x < 0 then x - 1 else x + 1

-- Define the fourth expression
noncomputable def expr4 (x : ℝ) : ℝ :=
if x ∈ Set.Ioo (-1) 1 then 0 else 1

-- Proof statement
theorem identify_functions :
  (∀ x, ∃! y, expr1 x = y) ∧ (∀ x, ∃! y, expr3 x = y) ∧
  (¬ ∃ x, ∃! y, expr2 x = y) ∧ (¬ ∀ x, ∃! y, expr4 x = y) := by
    sorry

end identify_functions_l1260_126005


namespace sin_alpha_pi_over_3_plus_sin_alpha_l1260_126094

-- Defining the problem with the given conditions
variable (α : ℝ)
variable (hcos : Real.cos (α + (2 / 3) * Real.pi) = 4 / 5)
variable (hα : -Real.pi / 2 < α ∧ α < 0)

-- Statement to prove
theorem sin_alpha_pi_over_3_plus_sin_alpha :
  Real.sin (α + Real.pi / 3) + Real.sin α = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_alpha_pi_over_3_plus_sin_alpha_l1260_126094


namespace stones_in_pile_l1260_126024

theorem stones_in_pile (initial_stones : ℕ) (final_stones_A : ℕ) (final_stones_B_min final_stones_B_max final_stones_B : ℕ) (operations : ℕ) :
  initial_stones = 2006 ∧ final_stones_A = 1990 ∧ final_stones_B_min = 2080 ∧ final_stones_B_max = 2100 ∧ operations < 20 ∧ (final_stones_B_min ≤ final_stones_B ∧ final_stones_B ≤ final_stones_B_max) 
  → final_stones_B = 2090 :=
by
  sorry

end stones_in_pile_l1260_126024


namespace orthocenter_of_triangle_l1260_126069

theorem orthocenter_of_triangle :
  ∀ (A B C H : ℝ × ℝ × ℝ),
    A = (2, 3, 4) → 
    B = (6, 4, 2) → 
    C = (4, 5, 6) → 
    H = (17/53, 152/53, 725/53) → 
    true :=
by sorry

end orthocenter_of_triangle_l1260_126069


namespace daniel_video_games_l1260_126030

/--
Daniel has a collection of some video games. 80 of them, Daniel bought for $12 each.
Of the rest, 50% were bought for $7. All others had a price of $3 each.
Daniel spent $2290 on all the games in his collection.
Prove that the total number of video games in Daniel's collection is 346.
-/
theorem daniel_video_games (n : ℕ) (r : ℕ)
    (h₀ : 80 * 12 = 960)
    (h₁ : 2290 - 960 = 1330)
    (h₂ : r / 2 * 7 + r / 2 * 3 = 1330):
    n = 80 + r → n = 346 :=
by
  intro h_total
  have r_eq : r = 266 := by sorry
  rw [r_eq] at h_total
  exact h_total

end daniel_video_games_l1260_126030


namespace sum_of_solutions_eq_zero_l1260_126044

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero : 
  (∃ x : ℝ, f x = 20) ∧ (∃ y : ℝ, f y = 20 ∧ x = -y) → 
  x + y = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l1260_126044


namespace compare_squares_l1260_126093

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2 * a := by
  -- the proof would go here, but we skip it according to the instruction
  sorry

end compare_squares_l1260_126093


namespace accident_rate_is_100_million_l1260_126066

theorem accident_rate_is_100_million (X : ℕ) (h1 : 96 * 3000000000 = 2880 * X) : X = 100000000 :=
by
  sorry

end accident_rate_is_100_million_l1260_126066


namespace max_value_of_f_f_lt_x3_minus_2x2_l1260_126003

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + Real.log x + b

theorem max_value_of_f (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) :
  f a b (Real.sqrt 2 / 2) = - (3 + 2 * Real.log 2) / 4 := by
  sorry

theorem f_lt_x3_minus_2x2 (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) (x : ℝ) (hx : 0 < x) :
  f a b x < x^3 - 2 * x^2 := by
  sorry

end max_value_of_f_f_lt_x3_minus_2x2_l1260_126003


namespace expression_evaluate_l1260_126039

theorem expression_evaluate (a b c : ℤ) (h1 : b = a + 2) (h2 : c = b - 10) (ha : a = 4)
(h3 : a ≠ -1) (h4 : b ≠ 2) (h5 : b ≠ -4) (h6 : c ≠ -6) : (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 :=
by
  sorry

end expression_evaluate_l1260_126039


namespace initial_people_count_l1260_126072

-- Definitions from conditions
def initial_people (W : ℕ) : ℕ := W
def net_increase : ℕ := 5 - 2
def current_people : ℕ := 19

-- Theorem to prove: initial_people == 16 given conditions
theorem initial_people_count (W : ℕ) (h1 : W + net_increase = current_people) : initial_people W = 16 :=
by
  sorry

end initial_people_count_l1260_126072


namespace city_population_divided_l1260_126078

theorem city_population_divided (total_population : ℕ) (parts : ℕ) (male_parts : ℕ) 
  (h1 : total_population = 1000) (h2 : parts = 5) (h3 : male_parts = 2) : 
  ∃ males : ℕ, males = 400 :=
by
  sorry

end city_population_divided_l1260_126078


namespace partnership_profit_l1260_126036

noncomputable def totalProfit (P Q R : ℕ) (unit_value_per_share : ℕ) : ℕ :=
  let profit_p := 36 * 2 + 18 * 10
  let profit_q := 24 * 12
  let profit_r := 36 * 12
  (profit_p + profit_q + profit_r) * unit_value_per_share

theorem partnership_profit (P Q R : ℕ) (unit_value_per_share : ℕ) :
  (P / Q = 3 / 2) → (Q / R = 4 / 3) → 
  (unit_value_per_share = 144 / 288) → 
  totalProfit P Q R (unit_value_per_share * 1) = 486 := 
by
  intros h1 h2 h3
  sorry

end partnership_profit_l1260_126036


namespace geostationary_orbit_distance_l1260_126090

noncomputable def distance_between_stations (earth_radius : ℝ) (orbit_altitude : ℝ) (num_stations : ℕ) : ℝ :=
  let θ : ℝ := 360 / num_stations
  let R : ℝ := earth_radius + orbit_altitude
  let sin_18 := (Real.sqrt 5 - 1) / 4
  2 * R * sin_18

theorem geostationary_orbit_distance :
  distance_between_stations 3960 22236 10 = -13098 + 13098 * Real.sqrt 5 :=
by
  sorry

end geostationary_orbit_distance_l1260_126090


namespace sqrt_sum_eq_nine_l1260_126060

theorem sqrt_sum_eq_nine (x : ℝ) (h : Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) :
  (7 + x) * (28 - x) = 529 :=
sorry

end sqrt_sum_eq_nine_l1260_126060


namespace subtract_complex_eq_l1260_126084

noncomputable def subtract_complex (a b : ℂ) : ℂ := a - b

theorem subtract_complex_eq (i : ℂ) (h_i : i^2 = -1) :
  subtract_complex (5 - 3 * i) (7 - 7 * i) = -2 + 4 * i :=
by
  sorry

end subtract_complex_eq_l1260_126084


namespace a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l1260_126016

theorem a_sq_greater_than_b_sq_neither_sufficient_nor_necessary 
  (a b : ℝ) : ¬ ((a^2 > b^2) → (a > b)) ∧  ¬ ((a > b) → (a^2 > b^2)) := sorry

end a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l1260_126016


namespace exists_zero_in_interval_l1260_126034

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

theorem exists_zero_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- This is just the Lean statement, no proof is provided
  sorry

end exists_zero_in_interval_l1260_126034


namespace anna_should_plant_8_lettuce_plants_l1260_126025

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end anna_should_plant_8_lettuce_plants_l1260_126025


namespace RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l1260_126021

-- Definitions representing the conditions
def TotalCrates : ℕ := 50
def PriceGrapes : ℕ := 15
def PriceMangoes : ℕ := 20
def PricePassionFruits : ℕ := 25
def CratesGrapes : ℕ := 13
def CratesMangoes : ℕ := 20
def CratesPassionFruits : ℕ := TotalCrates - CratesGrapes - CratesMangoes

def RevenueWithoutDiscounts : ℕ :=
  (CratesGrapes * PriceGrapes) +
  (CratesMangoes * PriceMangoes) +
  (CratesPassionFruits * PricePassionFruits)

def DiscountGrapes : Float := if CratesGrapes > 10 then 0.10 else 0.0
def DiscountMangoes : Float := if CratesMangoes > 15 then 0.15 else 0.0
def DiscountPassionFruits : Float := if CratesPassionFruits > 5 then 0.20 else 0.0

def DiscountedPrice (price : ℕ) (discount : Float) : Float := 
  price.toFloat * (1.0 - discount)

def RevenueWithDiscounts : Float :=
  (CratesGrapes.toFloat * DiscountedPrice PriceGrapes DiscountGrapes) +
  (CratesMangoes.toFloat * DiscountedPrice PriceMangoes DiscountMangoes) +
  (CratesPassionFruits.toFloat * DiscountedPrice PricePassionFruits DiscountPassionFruits)

-- Proof problems
theorem RevenueWithoutDiscounts_is_1020 : RevenueWithoutDiscounts = 1020 := sorry
theorem RevenueWithDiscounts_is_855_5 : RevenueWithDiscounts = 855.5 := sorry
theorem Difference_is_164_5 : (RevenueWithoutDiscounts.toFloat - RevenueWithDiscounts) = 164.5 := sorry

end RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l1260_126021


namespace problem1_xy_xplusy_l1260_126091

theorem problem1_xy_xplusy (x y: ℝ) (h1: x * y = 5) (h2: x + y = 6) : x - y = 4 ∨ x - y = -4 := 
sorry

end problem1_xy_xplusy_l1260_126091


namespace largest_value_l1260_126014

theorem largest_value :
  max (max (max (max (4^2) (4 * 2)) (4 - 2)) (4 / 2)) (4 + 2) = 4^2 :=
by sorry

end largest_value_l1260_126014


namespace infinite_non_prime_numbers_l1260_126046

theorem infinite_non_prime_numbers : ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ (¬(Nat.Prime (2 ^ (2 ^ m) + 1) ∨ ¬Nat.Prime (2018 ^ (2 ^ m) + 1))) := sorry

end infinite_non_prime_numbers_l1260_126046


namespace find_A_minus_B_l1260_126075

def A : ℕ := (55 * 100) + (19 * 10)
def B : ℕ := 173 + (5 * 224)

theorem find_A_minus_B : A - B = 4397 := by
  sorry

end find_A_minus_B_l1260_126075


namespace stacy_berries_multiple_l1260_126053

theorem stacy_berries_multiple (Skylar_berries : ℕ) (Stacy_berries : ℕ) (Steve_berries : ℕ) (m : ℕ)
  (h1 : Skylar_berries = 20)
  (h2 : Steve_berries = Skylar_berries / 2)
  (h3 : Stacy_berries = m * Steve_berries + 2)
  (h4 : Stacy_berries = 32) :
  m = 3 :=
by
  sorry

end stacy_berries_multiple_l1260_126053


namespace mowing_lawn_each_week_l1260_126074

-- Definitions based on the conditions
def riding_speed : ℝ := 2 -- acres per hour with riding mower
def push_speed : ℝ := 1 -- acre per hour with push mower
def total_hours : ℝ := 5 -- total hours

-- The problem we want to prove
theorem mowing_lawn_each_week (A : ℝ) :
  (3 / 4) * A / riding_speed + (1 / 4) * A / push_speed = total_hours → 
  A = 15 :=
by
  sorry

end mowing_lawn_each_week_l1260_126074


namespace preferred_order_for_boy_l1260_126017

variable (p q : ℝ)
variable (h : p < q)

theorem preferred_order_for_boy (p q : ℝ) (h : p < q) : 
  (2 * p * q - p^2 * q) > (2 * p * q - p * q^2) := 
sorry

end preferred_order_for_boy_l1260_126017


namespace discriminant_negative_of_positive_parabola_l1260_126051

variable (a b c : ℝ)

theorem discriminant_negative_of_positive_parabola (h1 : ∀ x : ℝ, a * x^2 + b * x + c > 0) (h2 : a > 0) : b^2 - 4*a*c < 0 := 
sorry

end discriminant_negative_of_positive_parabola_l1260_126051


namespace find_radius_l1260_126032

theorem find_radius (QP QO r : ℝ) (hQP : QP = 420) (hQO : QO = 427) : r = 77 :=
by
  -- Given QP^2 + r^2 = QO^2
  have h : (QP ^ 2) + (r ^ 2) = (QO ^ 2) := sorry
  -- Calculate the squares
  have h1 : (420 ^ 2) = 176400 := sorry
  have h2 : (427 ^ 2) = 182329 := sorry
  -- r^2 = 182329 - 176400
  have h3 : r ^ 2 = 5929 := sorry
  -- Therefore, r = 77
  exact sorry

end find_radius_l1260_126032


namespace cars_meet_in_two_hours_l1260_126023

theorem cars_meet_in_two_hours (t : ℝ) (d : ℝ) (v1 v2 : ℝ) (h1 : d = 60) (h2 : v1 = 13) (h3 : v2 = 17) (h4 : v1 * t + v2 * t = d) : t = 2 := 
by
  sorry

end cars_meet_in_two_hours_l1260_126023


namespace investment_rate_l1260_126052

theorem investment_rate (P_total P_7000 P_15000 I_total : ℝ)
  (h_investment : P_total = 22000)
  (h_investment_7000 : P_7000 = 7000)
  (h_investment_15000 : P_15000 = P_total - P_7000)
  (R_7000 : ℝ)
  (h_rate_7000 : R_7000 = 0.18)
  (I_7000 : ℝ)
  (h_interest_7000 : I_7000 = P_7000 * R_7000)
  (h_total_interest : I_total = 3360) :
  ∃ (R_15000 : ℝ), (I_total - I_7000) = P_15000 * R_15000 ∧ R_15000 = 0.14 := 
by
  sorry

end investment_rate_l1260_126052


namespace number_drawn_from_first_group_l1260_126089

theorem number_drawn_from_first_group (n: ℕ) (groups: ℕ) (interval: ℕ) (fourth_group_number: ℕ) (total_bags: ℕ) 
    (h1: total_bags = 50) (h2: groups = 5) (h3: interval = total_bags / groups)
    (h4: interval = 10) (h5: fourth_group_number = 36) : n = 6 :=
by
  sorry

end number_drawn_from_first_group_l1260_126089


namespace mark_fewer_than_susan_l1260_126057

variable (apples_total : ℕ) (greg_apples : ℕ) (susan_apples : ℕ) (mark_apples : ℕ) (mom_apples : ℕ)

def evenly_split (total : ℕ) : ℕ := total / 2

theorem mark_fewer_than_susan
    (h1 : apples_total = 18)
    (h2 : greg_apples = evenly_split apples_total)
    (h3 : susan_apples = 2 * greg_apples)
    (h4 : mom_apples = 40 + 9)
    (h5 : mark_apples = mom_apples - susan_apples) :
    susan_apples - mark_apples = 13 := 
sorry

end mark_fewer_than_susan_l1260_126057


namespace isosceles_triangle_base_length_l1260_126065

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l1260_126065


namespace ceil_evaluation_l1260_126035

theorem ceil_evaluation : 
  (Int.ceil (4 * (8 - 1 / 3 : ℚ))) = 31 :=
by
  sorry

end ceil_evaluation_l1260_126035


namespace correct_total_l1260_126055

-- Define the conditions in Lean
variables (y : ℕ) -- y is a natural number (non-negative integer)

-- Define the values of the different coins in cents
def value_of_quarter := 25
def value_of_dollar := 100
def value_of_nickel := 5
def value_of_dime := 10

-- Define the errors in terms of y
def error_due_to_quarters := y * (value_of_dollar - value_of_quarter) -- 75y
def error_due_to_nickels := y * (value_of_dime - value_of_nickel) -- 5y

-- Net error calculation
def net_error := error_due_to_quarters - error_due_to_nickels -- 70y

-- Math proof problem statement
theorem correct_total (h : error_due_to_quarters = 75 * y ∧ error_due_to_nickels = 5 * y) :
  net_error = 70 * y :=
by sorry

end correct_total_l1260_126055


namespace initial_balance_before_check_deposit_l1260_126061

theorem initial_balance_before_check_deposit (new_balance : ℝ) (initial_balance : ℝ) : 
  (50 = 1 / 4 * new_balance) → (initial_balance = new_balance - 50) → initial_balance = 150 :=
by
  sorry

end initial_balance_before_check_deposit_l1260_126061


namespace median_mean_l1260_126058

theorem median_mean (n : ℕ) (h : n + 4 = 8) : (4 + 6 + 8 + 14 + 16) / 5 = 9.6 := by
  sorry

end median_mean_l1260_126058


namespace geometric_sequence_sum_l1260_126095

theorem geometric_sequence_sum (a : ℕ → ℤ)
  (h1 : a 0 = 1)
  (h_q : ∀ n, a (n + 1) = a n * -2) :
  a 0 + |a 1| + a 2 + |a 3| = 15 := by
  sorry

end geometric_sequence_sum_l1260_126095


namespace expression_bounds_l1260_126064

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ∧
  (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ≤ 4 :=
by
  sorry

end expression_bounds_l1260_126064


namespace total_distance_12_hours_l1260_126010

-- Define the initial conditions for the speed and distance calculation
def speed_increase : ℕ → ℕ
  | 0 => 50
  | n + 1 => speed_increase n + 2

def distance_in_hour (n : ℕ) : ℕ := speed_increase n

-- Define the total distance traveled in 12 hours
def total_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => total_distance n + distance_in_hour n

theorem total_distance_12_hours :
  total_distance 12 = 732 := by
  sorry

end total_distance_12_hours_l1260_126010


namespace value_of_f_g_6_squared_l1260_126002

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem value_of_f_g_6_squared : (f (g 6))^2 = 26569 :=
by
  -- Place your proof here
  sorry

end value_of_f_g_6_squared_l1260_126002


namespace initial_percentage_of_water_l1260_126047

theorem initial_percentage_of_water (P : ℕ) : 
  (P / 100) * 120 + 54 = (3 / 4) * 120 → P = 30 :=
by 
  intro h
  sorry

end initial_percentage_of_water_l1260_126047


namespace circle_equation_midpoint_trajectory_l1260_126041

-- Definition for the circle equation proof
theorem circle_equation (x y : ℝ) (h : (x - 3)^2 + (y - 2)^2 = 13)
  (hx : x = 3) (hy : y = 2) : 
  (x - 3)^2 + (y - 2)^2 = 13 := by
  sorry -- Placeholder for proof

-- Definition for the midpoint trajectory proof
theorem midpoint_trajectory (x y : ℝ) (hx : x = (2 * x - 11) / 2)
  (hy : y = (2 * y - 2) / 2) (h : (2 * x - 11)^2 + (2 * y - 2)^2 = 13) :
  (x - 11 / 2)^2 + (y - 1)^2 = 13 / 4 := by
  sorry -- Placeholder for proof

end circle_equation_midpoint_trajectory_l1260_126041


namespace toms_weekly_earnings_l1260_126027

variable (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def total_money_per_week (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_per_week

theorem toms_weekly_earnings :
  total_money_per_week 8 12 5 7 = 3360 :=
by
  sorry

end toms_weekly_earnings_l1260_126027


namespace cora_reading_ratio_l1260_126096

variable (P : Nat) 
variable (M T W Th F : Nat)

-- Conditions
def conditions (P M T W Th F : Nat) : Prop := 
  P = 158 ∧ 
  M = 23 ∧ 
  T = 38 ∧ 
  W = 61 ∧ 
  Th = 12 ∧ 
  F = Th

-- The theorem statement
theorem cora_reading_ratio (h : conditions P M T W Th F) : F / Th = 1 / 1 :=
by
  -- We use the conditions to apply the proof
  obtain ⟨hp, hm, ht, hw, hth, hf⟩ := h
  rw [hf]
  norm_num
  sorry

end cora_reading_ratio_l1260_126096


namespace solve_y_l1260_126026

theorem solve_y 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (remainder_condition : x = (96.12 * y))
  (division_condition : x = (96.0624 * y + 5.76)) : 
  y = 100 := 
 sorry

end solve_y_l1260_126026


namespace min_value_of_b1_plus_b2_l1260_126092

theorem min_value_of_b1_plus_b2 (b : ℕ → ℕ) (h1 : ∀ n ≥ 1, b (n + 2) = (b n + 4030) / (1 + b (n + 1)))
  (h2 : ∀ n, b n > 0) : ∃ b1 b2, b1 * b2 = 4030 ∧ b1 + b2 = 127 :=
by {
  sorry
}

end min_value_of_b1_plus_b2_l1260_126092


namespace additional_grassy_ground_l1260_126067

theorem additional_grassy_ground (r1 r2 : ℝ) (h1: r1 = 16) (h2: r2 = 23) :
  (π * r2 ^ 2) - (π * r1 ^ 2) = 273 * π :=
by
  sorry

end additional_grassy_ground_l1260_126067


namespace mother_reaches_timothy_l1260_126062

/--
Timothy leaves home for school, riding his bicycle at a rate of 6 miles per hour.
Fifteen minutes after he leaves, his mother sees Timothy's math homework lying on his bed and immediately leaves home to bring it to him.
If his mother drives at 36 miles per hour, prove that she must drive 1.8 miles to reach Timothy.
-/
theorem mother_reaches_timothy
  (timothy_speed : ℕ)
  (mother_speed : ℕ)
  (delay_minutes : ℕ)
  (distance_must_drive : ℕ)
  (h_speed_t : timothy_speed = 6)
  (h_speed_m : mother_speed = 36)
  (h_delay : delay_minutes = 15)
  (h_distance : distance_must_drive = 18 / 10 ) :
  ∃ t : ℚ, (timothy_speed * (delay_minutes / 60) + timothy_speed * t) = (mother_speed * t) := sorry

end mother_reaches_timothy_l1260_126062


namespace total_students_l1260_126081

-- Defining the conditions
variable (H : ℕ) -- Number of students who ordered hot dogs
variable (students_ordered_burgers : ℕ) -- Number of students who ordered burgers

-- Given conditions
def burger_condition := students_ordered_burgers = 30
def hotdog_condition := students_ordered_burgers = 2 * H

-- Theorem to prove the total number of students
theorem total_students (H : ℕ) (students_ordered_burgers : ℕ) 
  (h1 : burger_condition students_ordered_burgers) 
  (h2 : hotdog_condition students_ordered_burgers H) : 
  students_ordered_burgers + H = 45 := 
by
  sorry

end total_students_l1260_126081


namespace arithmetic_seq_a4_l1260_126048

-- Definition of an arithmetic sequence with the first three terms given.
def arithmetic_seq (a : ℕ → ℕ) :=
  a 0 = 2 ∧ a 1 = 4 ∧ a 2 = 6 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

-- The actual proof goal.
theorem arithmetic_seq_a4 : ∃ a : ℕ → ℕ, arithmetic_seq a ∧ a 3 = 8 :=
by
  sorry

end arithmetic_seq_a4_l1260_126048


namespace estimate_points_in_interval_l1260_126068

-- Define the conditions
def total_data_points : ℕ := 1000
def frequency_interval : ℝ := 0.16
def interval_estimation : ℝ := total_data_points * frequency_interval

-- Lean theorem statement
theorem estimate_points_in_interval : interval_estimation = 160 :=
by
  sorry

end estimate_points_in_interval_l1260_126068


namespace cost_price_of_book_l1260_126015

-- Define the variables and conditions
variable (C : ℝ)
variable (P : ℝ)
variable (S : ℝ)

-- State the conditions given in the problem
def conditions := S = 260 ∧ P = 0.20 * C ∧ S = C + P

-- State the theorem
theorem cost_price_of_book (h : conditions C P S) : C = 216.67 :=
sorry

end cost_price_of_book_l1260_126015


namespace pages_in_first_chapter_l1260_126018

theorem pages_in_first_chapter
  (total_pages : ℕ)
  (second_chapter_pages : ℕ)
  (first_chapter_pages : ℕ)
  (h1 : total_pages = 81)
  (h2 : second_chapter_pages = 68) :
  first_chapter_pages = 81 - 68 :=
sorry

end pages_in_first_chapter_l1260_126018


namespace fruit_count_correct_l1260_126029

def george_oranges := 45
def amelia_oranges := george_oranges - 18
def amelia_apples := 15
def george_apples := amelia_apples + 5

def olivia_orange_rate := 3
def olivia_apple_rate := 2
def olivia_minutes := 30
def olivia_cycle_minutes := 5
def olivia_cycles := olivia_minutes / olivia_cycle_minutes
def olivia_oranges := olivia_orange_rate * olivia_cycles
def olivia_apples := olivia_apple_rate * olivia_cycles

def total_oranges := george_oranges + amelia_oranges + olivia_oranges
def total_apples := george_apples + amelia_apples + olivia_apples
def total_fruits := total_oranges + total_apples

theorem fruit_count_correct : total_fruits = 137 := by
  sorry

end fruit_count_correct_l1260_126029


namespace correct_operation_l1260_126085

theorem correct_operation (a b : ℝ) : 
  (-a^3 * b)^2 = a^6 * b^2 :=
by
  sorry

end correct_operation_l1260_126085


namespace common_difference_l1260_126097

theorem common_difference (a1 d : ℕ) (S3 : ℕ) (h1 : S3 = 6) (h2 : a1 = 1)
  (h3 : S3 = 3 * (2 * a1 + 2 * d) / 2) : d = 1 :=
by
  sorry

end common_difference_l1260_126097


namespace solve_system_l1260_126011

theorem solve_system :
  ∃! (x y : ℝ), (2 * x + y + 8 ≤ 0) ∧ (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) ∧ (x = -3 ∧ y = -2) := 
  by
  sorry

end solve_system_l1260_126011


namespace smallest_possible_S_l1260_126042

/-- Define the maximum possible sum for n dice --/
def max_sum (n : ℕ) : ℕ := 6 * n

/-- Define the transformation of the dice sum when each result is transformed to 7 - d_i --/
def transformed_sum (n R : ℕ) : ℕ := 7 * n - R

/-- Determine the smallest possible S under given conditions --/
theorem smallest_possible_S :
  ∃ n : ℕ, max_sum n ≥ 2001 ∧ transformed_sum n 2001 = 337 :=
by
  -- TODO: Complete the proof
  sorry

end smallest_possible_S_l1260_126042


namespace correct_transformation_of_95_sq_l1260_126059

theorem correct_transformation_of_95_sq : 95^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end correct_transformation_of_95_sq_l1260_126059


namespace distinct_digit_numbers_count_l1260_126043

def numDistinctDigitNumbers : Nat := 
  let first_digit_choices := 10
  let second_digit_choices := 9
  let third_digit_choices := 8
  let fourth_digit_choices := 7
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem distinct_digit_numbers_count : numDistinctDigitNumbers = 5040 :=
by
  sorry

end distinct_digit_numbers_count_l1260_126043


namespace simplify_expression_l1260_126007

theorem simplify_expression (x : ℝ) : 3 * x + 4 - x + 8 = 2 * x + 12 :=
by
  sorry

end simplify_expression_l1260_126007


namespace ellipse_focal_length_l1260_126080

theorem ellipse_focal_length :
  ∀ a b c : ℝ, (a^2 = 11) → (b^2 = 3) → (c^2 = a^2 - b^2) → (2 * c = 4 * Real.sqrt 2) :=
by
  sorry

end ellipse_focal_length_l1260_126080


namespace actual_total_area_in_acres_l1260_126054

-- Define the conditions
def base_cm : ℝ := 20
def height_cm : ℝ := 12
def rect_length_cm : ℝ := 20
def rect_width_cm : ℝ := 5
def scale_cm_to_miles : ℝ := 3
def sq_mile_to_acres : ℝ := 640

-- Define the total area in acres calculation
def total_area_cm_squared : ℝ := 120 + 100
def total_area_miles_squared : ℝ := total_area_cm_squared * (scale_cm_to_miles ^ 2)
def total_area_acres : ℝ := total_area_miles_squared * sq_mile_to_acres

-- The theorem statement
theorem actual_total_area_in_acres : total_area_acres = 1267200 :=
by
  sorry

end actual_total_area_in_acres_l1260_126054


namespace slope_and_y_intercept_l1260_126087

def line_equation (x y : ℝ) : Prop := 4 * y = 6 * x - 12

theorem slope_and_y_intercept (x y : ℝ) (h : line_equation x y) : 
  ∃ m b : ℝ, (m = 3/2) ∧ (b = -3) ∧ (y = m * x + b) :=
  sorry

end slope_and_y_intercept_l1260_126087


namespace find_analytical_expression_of_f_l1260_126098

-- Define the function f and the condition it needs to satisfy
variable (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1))

-- State the objective to prove
theorem find_analytical_expression_of_f : 
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (1 + x) := by
  sorry

end find_analytical_expression_of_f_l1260_126098


namespace solve_squares_and_circles_l1260_126063

theorem solve_squares_and_circles (x y : ℝ) :
  (5 * x + 2 * y = 39) ∧ (3 * x + 3 * y = 27) → (x = 7) ∧ (y = 2) :=
by
  intro h
  sorry

end solve_squares_and_circles_l1260_126063


namespace minimally_intersecting_triples_modulo_1000_eq_344_l1260_126020

def minimally_intersecting_triples_count_modulo : ℕ :=
  let total_count := 57344
  total_count % 1000

theorem minimally_intersecting_triples_modulo_1000_eq_344 :
  minimally_intersecting_triples_count_modulo = 344 := by
  sorry

end minimally_intersecting_triples_modulo_1000_eq_344_l1260_126020


namespace intercepts_of_line_l1260_126045

theorem intercepts_of_line (x y : ℝ) : 
  (x + 6 * y + 2 = 0) → (x = -2) ∧ (y = -1 / 3) :=
by
  sorry

end intercepts_of_line_l1260_126045


namespace total_height_increase_l1260_126012

def height_increase_per_decade : ℕ := 90
def decades_in_two_centuries : ℕ := (2 * 100) / 10

theorem total_height_increase :
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end total_height_increase_l1260_126012


namespace faye_books_l1260_126071

theorem faye_books (initial_books given_away final_books books_bought: ℕ) 
  (h1 : initial_books = 34) 
  (h2 : given_away = 3) 
  (h3 : final_books = 79) 
  (h4 : final_books = initial_books - given_away + books_bought) : 
  books_bought = 48 := 
by 
  sorry

end faye_books_l1260_126071


namespace ryan_chinese_learning_hours_l1260_126013

theorem ryan_chinese_learning_hours : 
    ∀ (h_english : ℕ) (diff : ℕ), 
    h_english = 7 → 
    h_english = 2 + (h_english - diff) → 
    diff = 5 := by
  intros h_english diff h_english_eq h_english_diff_eq
  sorry

end ryan_chinese_learning_hours_l1260_126013


namespace pipe_r_fill_time_l1260_126050

theorem pipe_r_fill_time (x : ℝ) : 
  (1 / 3 + 1 / 9 + 1 / x = 1 / 2) → 
  x = 18 :=
by 
  sorry

end pipe_r_fill_time_l1260_126050


namespace sqrt_of_neg_five_squared_l1260_126086

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 ∨ Real.sqrt ((-5 : ℝ)^2) = -5 :=
by
  sorry

end sqrt_of_neg_five_squared_l1260_126086


namespace mistaken_fraction_l1260_126033

theorem mistaken_fraction (n correct_result student_result : ℕ) (h1 : n = 384)
  (h2 : correct_result = (5 * n) / 16) (h3 : student_result = correct_result + 200) : 
  (student_result / n : ℚ) = 5 / 6 :=
by
  sorry

end mistaken_fraction_l1260_126033


namespace percentage_error_in_area_l1260_126008

theorem percentage_error_in_area (S : ℝ) (h : S > 0) :
  let S' := S * 1.06
  let A := S^2
  let A' := (S')^2
  (A' - A) / A * 100 = 12.36 := by
  sorry

end percentage_error_in_area_l1260_126008
