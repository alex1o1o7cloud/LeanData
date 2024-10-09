import Mathlib

namespace sum_of_averages_is_six_l805_80542

variable (a b c d e : ℕ)

def average_teacher : ℚ :=
  (5 * a + 4 * b + 3 * c + 2 * d + e) / (a + b + c + d + e)

def average_kati : ℚ :=
  (5 * e + 4 * d + 3 * c + 2 * b + a) / (a + b + c + d + e)

theorem sum_of_averages_is_six (a b c d e : ℕ) : 
    average_teacher a b c d e + average_kati a b c d e = 6 := by
  sorry

end sum_of_averages_is_six_l805_80542


namespace original_square_perimeter_l805_80566

theorem original_square_perimeter (x : ℝ) 
  (h1 : ∀ r, r = x ∨ r = 4 * x) 
  (h2 : 28 * x = 56) : 
  4 * (4 * x) = 32 :=
by
  -- We don't need to consider the proof as per instructions.
  sorry

end original_square_perimeter_l805_80566


namespace contradiction_example_l805_80580

theorem contradiction_example (a b c d : ℝ) 
(h1 : a + b = 1) 
(h2 : c + d = 1) 
(h3 : ac + bd > 1) : 
¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_example_l805_80580


namespace common_point_sufficient_condition_l805_80563

theorem common_point_sufficient_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3) → k ≤ -2 * Real.sqrt 2 :=
by
  -- Proof will go here
  sorry

end common_point_sufficient_condition_l805_80563


namespace melanie_sale_revenue_correct_l805_80513

noncomputable def melanie_revenue : ℝ :=
let red_cost := 0.08
let green_cost := 0.10
let yellow_cost := 0.12
let red_gumballs := 15
let green_gumballs := 18
let yellow_gumballs := 22
let total_gumballs := red_gumballs + green_gumballs + yellow_gumballs
let total_cost := (red_cost * red_gumballs) + (green_cost * green_gumballs) + (yellow_cost * yellow_gumballs)
let discount := if total_gumballs >= 20 then 0.30 else if total_gumballs >= 10 then 0.20 else 0
let final_cost := total_cost * (1 - discount)
final_cost

theorem melanie_sale_revenue_correct : melanie_revenue = 3.95 :=
by
  -- All calculations and proofs omitted for brevity, as per instructions above
  sorry

end melanie_sale_revenue_correct_l805_80513


namespace first_term_of_geometric_sequence_l805_80507

theorem first_term_of_geometric_sequence (a r : ℚ) (h1 : a * r^2 = 12) (h2 : a * r^3 = 16) : a = 27 / 4 :=
by {
  sorry
}

end first_term_of_geometric_sequence_l805_80507


namespace part1_l805_80502

def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

theorem part1 (a : ℝ) (h : a = 1) :
  (Set.compl B ∪ A a) = {x | x ≤ 1 ∨ x ≥ 2} :=
by
  sorry

end part1_l805_80502


namespace sector_area_120_deg_radius_3_l805_80511

theorem sector_area_120_deg_radius_3 (r : ℝ) (theta_deg : ℝ) (theta_rad : ℝ) (A : ℝ)
  (h1 : r = 3)
  (h2 : theta_deg = 120)
  (h3 : theta_rad = (2 * Real.pi / 3))
  (h4 : A = (1 / 2) * theta_rad * r^2) :
  A = 3 * Real.pi :=
  sorry

end sector_area_120_deg_radius_3_l805_80511


namespace find_blue_chips_l805_80582

def num_chips_satisfies (n m : ℕ) : Prop :=
  (n > m) ∧ (n + m > 2) ∧ (n + m < 50) ∧
  (n * (n - 1) + m * (m - 1)) = 2 * n * m

theorem find_blue_chips (n : ℕ) :
  (∃ m : ℕ, num_chips_satisfies n m) → 
  n = 3 ∨ n = 6 ∨ n = 10 ∨ n = 15 ∨ n = 21 ∨ n = 28 :=
by
  sorry

end find_blue_chips_l805_80582


namespace makes_at_least_one_shot_l805_80594
noncomputable section

/-- The probability of making the free throw. -/
def free_throw_make_prob : ℚ := 4/5

/-- The probability of making the high school 3-pointer. -/
def high_school_make_prob : ℚ := 1/2

/-- The probability of making the professional 3-pointer. -/
def pro_make_prob : ℚ := 1/3

/-- The probability of making at least one of the three shots. -/
theorem makes_at_least_one_shot :
  (1 - ((1 - free_throw_make_prob) * (1 - high_school_make_prob) * (1 - pro_make_prob))) = 14 / 15 :=
by
  sorry

end makes_at_least_one_shot_l805_80594


namespace find_certain_number_l805_80526

theorem find_certain_number (x : ℕ) (h1 : 172 = 4 * 43) (h2 : 43 - 172 / x = 28) (h3 : 172 % x = 7) : x = 11 := by
  sorry

end find_certain_number_l805_80526


namespace sin_675_eq_neg_sqrt2_div_2_l805_80512

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end sin_675_eq_neg_sqrt2_div_2_l805_80512


namespace range_of_a_l805_80550

noncomputable def f (x a : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 3 < a :=
by
  sorry

end range_of_a_l805_80550


namespace triangle_perimeter_l805_80584

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_perimeter {a b c : ℕ} (h : is_triangle 15 11 19) : 15 + 11 + 19 = 45 := by
  sorry

end triangle_perimeter_l805_80584


namespace mints_ratio_l805_80562

theorem mints_ratio (n : ℕ) (green_mints red_mints : ℕ) (h1 : green_mints + red_mints = n) (h2 : green_mints = 3 * (n / 4)) : green_mints / red_mints = 3 :=
by
  sorry

end mints_ratio_l805_80562


namespace smallest_sum_of_two_squares_l805_80591

theorem smallest_sum_of_two_squares :
  ∃ n : ℕ, (∀ m : ℕ, m < n → (¬ (∃ a b c d e f : ℕ, m = a^2 + b^2 ∧  m = c^2 + d^2 ∧ m = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))))) ∧
          (∃ a b c d e f : ℕ, n = a^2 + b^2 ∧  n = c^2 + d^2 ∧ n = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))) :=
sorry

end smallest_sum_of_two_squares_l805_80591


namespace non_congruent_rectangles_count_l805_80564

theorem non_congruent_rectangles_count :
  (∃ (l w : ℕ), l + w = 50 ∧ l ≠ w) ∧
  (∀ (l w : ℕ), l + w = 50 ∧ l ≠ w → l > w) →
  (∃ (n : ℕ), n = 24) :=
by
  sorry

end non_congruent_rectangles_count_l805_80564


namespace part_a_l805_80573

theorem part_a (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, a (n + 2) = a (n + 1) * a n + 1) :
  ∀ n, ¬ (4 ∣ a n) :=
by
  sorry

end part_a_l805_80573


namespace club_membership_l805_80570

theorem club_membership (n : ℕ) 
  (h1 : n % 10 = 6)
  (h2 : n % 11 = 6)
  (h3 : 150 ≤ n)
  (h4 : n ≤ 300) : 
  n = 226 := 
sorry

end club_membership_l805_80570


namespace max_square_plots_l805_80572

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end max_square_plots_l805_80572


namespace martin_spends_30_dollars_on_berries_l805_80549

def cost_per_package : ℝ := 2.0
def cups_per_package : ℝ := 1.0
def cups_per_day : ℝ := 0.5
def days : ℝ := 30

theorem martin_spends_30_dollars_on_berries :
  (days / (cups_per_package / cups_per_day)) * cost_per_package = 30 :=
by
  sorry

end martin_spends_30_dollars_on_berries_l805_80549


namespace statements_correct_l805_80515

theorem statements_correct :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (∀ x : ℝ, (∀ x, x^2 + x + 1 ≠ 0) ↔ (∃ x, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) ↔ p ∧ q) ∧
  (∀ x : ℝ, (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬ (x^2 - 3*x + 2 > 0) → x ≤ 2)) :=
by
  sorry

end statements_correct_l805_80515


namespace last_two_digits_of_floor_l805_80569

def last_two_digits (n : Nat) : Nat :=
  n % 100

theorem last_two_digits_of_floor :
  let x := 10^93
  let y := 10^31
  last_two_digits (Nat.floor (x / (y + 3))) = 8 :=
by
  sorry

end last_two_digits_of_floor_l805_80569


namespace sufficient_not_necessary_l805_80516

variable (a : ℝ)

theorem sufficient_not_necessary :
  (a > 1 → a^2 > a) ∧ (¬(a > 1) ∧ a^2 > a → a < 0) :=
by
  sorry

end sufficient_not_necessary_l805_80516


namespace probability_at_least_one_each_color_in_bag_l805_80505

open BigOperators

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_at_least_one_each_color : ℚ :=
  let total_ways := num_combinations 9 5
  let favorable_ways := 27 + 27 + 27 -- 3 scenarios (2R+1B+2G, 2B+1R+2G, 2G+1R+2B)
  favorable_ways / total_ways

theorem probability_at_least_one_each_color_in_bag :
  prob_at_least_one_each_color = 9 / 14 :=
by
  sorry

end probability_at_least_one_each_color_in_bag_l805_80505


namespace johns_mistake_l805_80585

theorem johns_mistake (a b : ℕ) (h1 : 10000 * a + b = 11 * a * b)
  (h2 : 100 ≤ a ∧ a ≤ 999) (h3 : 1000 ≤ b ∧ b ≤ 9999) : a + b = 1093 :=
sorry

end johns_mistake_l805_80585


namespace shirts_sold_correct_l805_80597

-- Define the conditions
def shoes_sold := 6
def cost_per_shoe := 3
def earnings_per_person := 27
def total_earnings := 2 * earnings_per_person
def earnings_from_shoes := shoes_sold * cost_per_shoe
def cost_per_shirt := 2
def earnings_from_shirts := total_earnings - earnings_from_shoes

-- Define the total number of shirts sold and the target value to prove
def shirts_sold : Nat := earnings_from_shirts / cost_per_shirt

-- Prove that shirts_sold is 18
theorem shirts_sold_correct : shirts_sold = 18 := by
  sorry

end shirts_sold_correct_l805_80597


namespace jason_work_hours_l805_80565

variable (x y : ℕ)

def working_hours : Prop :=
  (4 * x + 6 * y = 88) ∧
  (x + y = 18)

theorem jason_work_hours (h : working_hours x y) : y = 8 :=
  by
    sorry

end jason_work_hours_l805_80565


namespace third_term_of_sequence_l805_80541

theorem third_term_of_sequence (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = (1 / 2) * a n + (1 / (2 * n))) : a 3 = 3 / 4 := by
  sorry

end third_term_of_sequence_l805_80541


namespace expression_value_as_fraction_l805_80578

theorem expression_value_as_fraction :
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 :=
by
  sorry

end expression_value_as_fraction_l805_80578


namespace book_purchasing_methods_l805_80592

theorem book_purchasing_methods :
  ∃ (A B C D : ℕ),
  A + B + C + D = 10 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
  3 * A + 5 * B + 7 * C + 11 * D = 70 ∧
  (∃ N : ℕ, N = 4) :=
by sorry

end book_purchasing_methods_l805_80592


namespace correct_operation_l805_80588

variables (a b : ℝ)

theorem correct_operation : 5 * a * b - 3 * a * b = 2 * a * b :=
by sorry

end correct_operation_l805_80588


namespace average_gas_mileage_round_trip_l805_80509

-- necessary definitions related to the problem conditions
def total_distance_one_way := 150
def fuel_efficiency_going := 35
def fuel_efficiency_return := 30
def round_trip_distance := total_distance_one_way + total_distance_one_way

-- calculation of gasoline used for each trip and total usage
def gasoline_used_going := total_distance_one_way / fuel_efficiency_going
def gasoline_used_return := total_distance_one_way / fuel_efficiency_return
def total_gasoline_used := gasoline_used_going + gasoline_used_return

-- calculation of average gas mileage
def average_gas_mileage := round_trip_distance / total_gasoline_used

-- the final theorem to prove the average gas mileage for the round trip 
theorem average_gas_mileage_round_trip : average_gas_mileage = 32 := 
by
  sorry

end average_gas_mileage_round_trip_l805_80509


namespace surface_area_of_rectangular_prism_l805_80519

def SurfaceArea (length : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  2 * ((length * width) + (width * height) + (height * length))

theorem surface_area_of_rectangular_prism 
  (l w h : ℕ) 
  (hl : l = 1) 
  (hw : w = 2) 
  (hh : h = 2) : 
  SurfaceArea l w h = 16 := by
  sorry

end surface_area_of_rectangular_prism_l805_80519


namespace combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l805_80536

theorem combined_sum_of_interior_numbers_of_eighth_and_ninth_rows :
  (2 ^ (8 - 1) - 2) + (2 ^ (9 - 1) - 2) = 380 :=
by
  -- The steps of the proof would go here, but for the purpose of this task:
  sorry

end combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l805_80536


namespace max_elements_of_valid_set_l805_80517

def valid_set (M : Finset ℤ) : Prop :=
  ∀ (a b c : ℤ), a ∈ M → b ∈ M → c ∈ M → (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (a + b ∈ M ∨ a + c ∈ M ∨ b + c ∈ M)

theorem max_elements_of_valid_set (M : Finset ℤ) (h : valid_set M) : M.card ≤ 7 :=
sorry

end max_elements_of_valid_set_l805_80517


namespace volleyball_club_lineups_l805_80598
-- Import the required Lean library

-- Define the main problem
theorem volleyball_club_lineups :
  let total_players := 18
  let quadruplets := 4
  let starters := 6
  let eligible_lineups := Nat.choose 18 6 - Nat.choose 14 2 - Nat.choose 14 6
  eligible_lineups = 15470 :=
by
  sorry

end volleyball_club_lineups_l805_80598


namespace equation_no_solution_B_l805_80590

theorem equation_no_solution_B :
  ¬(∃ x : ℝ, |-3 * x| + 5 = 0) :=
sorry

end equation_no_solution_B_l805_80590


namespace brad_reads_26_pages_per_day_l805_80558

-- Define conditions
def greg_daily_reading : ℕ := 18
def brad_extra_pages : ℕ := 8

-- Define Brad's daily reading
def brad_daily_reading : ℕ := greg_daily_reading + brad_extra_pages

-- The theorem to be proven
theorem brad_reads_26_pages_per_day : brad_daily_reading = 26 := by
  sorry

end brad_reads_26_pages_per_day_l805_80558


namespace sum_of_interior_numbers_l805_80500

def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

theorem sum_of_interior_numbers :
  sum_interior 8 + sum_interior 9 + sum_interior 10 = 890 :=
by
  sorry

end sum_of_interior_numbers_l805_80500


namespace product_of_a_values_has_three_solutions_eq_20_l805_80557

noncomputable def f (x : ℝ) : ℝ := abs ((x^2 - 10 * x + 25) / (x - 5) - (x^2 - 3 * x) / (3 - x))

def has_three_solutions (a : ℝ) : Prop :=
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ abs (abs (f x1) - 5) = a ∧ abs (abs (f x2) - 5) = a ∧ abs (abs (f x3) - 5) = a)

theorem product_of_a_values_has_three_solutions_eq_20 :
  ∃ a1 a2 : ℝ, has_three_solutions a1 ∧ has_three_solutions a2 ∧ a1 * a2 = 20 :=
sorry

end product_of_a_values_has_three_solutions_eq_20_l805_80557


namespace simplify_fraction_l805_80581

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 :=
by
  sorry

end simplify_fraction_l805_80581


namespace equilateral_triangle_sum_perimeters_l805_80571

theorem equilateral_triangle_sum_perimeters (s : ℝ) (h : ∑' n, 3 * s / 2 ^ n = 360) : 
  s = 60 := 
by 
  sorry

end equilateral_triangle_sum_perimeters_l805_80571


namespace compound_interest_rate_l805_80524

theorem compound_interest_rate
  (P : ℝ) (t : ℕ) (A : ℝ) (interest : ℝ)
  (hP : P = 6000)
  (ht : t = 2)
  (hA : A = 7260)
  (hInterest : interest = 1260.000000000001)
  (hA_eq : A = P + interest) :
  ∃ r : ℝ, (1 + r)^(t : ℝ) = A / P ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l805_80524


namespace electric_car_travel_distance_l805_80593

theorem electric_car_travel_distance {d_electric d_diesel : ℕ} 
  (h1 : d_diesel = 120) 
  (h2 : d_electric = d_diesel + 50 * d_diesel / 100) : 
  d_electric = 180 := 
by 
  sorry

end electric_car_travel_distance_l805_80593


namespace simplify_336_to_fraction_l805_80586

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l805_80586


namespace Doug_age_l805_80520

theorem Doug_age
  (B : ℕ) (D : ℕ) (N : ℕ)
  (h1 : 2 * B = N)
  (h2 : B + D = 90)
  (h3 : 20 * N = 2000) : 
  D = 40 := sorry

end Doug_age_l805_80520


namespace judy_hits_percentage_l805_80533

theorem judy_hits_percentage 
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (single_hits_percentage : ℚ) :
  total_hits = 35 →
  home_runs = 1 →
  triples = 1 →
  doubles = 5 →
  single_hits_percentage = (total_hits - (home_runs + triples + doubles)) / total_hits * 100 →
  single_hits_percentage = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end judy_hits_percentage_l805_80533


namespace log_identity_l805_80545

theorem log_identity (c b : ℝ) (h1 : c = Real.log 81 / Real.log 4) (h2 : b = Real.log 3 / Real.log 2) : c = 2 * b := by
  sorry

end log_identity_l805_80545


namespace income_increase_l805_80525

-- Definitions based on conditions
def original_price := 1.0
def original_items := 100.0
def discount := 0.10
def increased_sales := 0.15

-- Calculations for new values
def new_price := original_price * (1 - discount)
def new_items := original_items * (1 + increased_sales)
def original_income := original_price * original_items
def new_income := new_price * new_items

-- The percentage increase in income
def percentage_increase := ((new_income - original_income) / original_income) * 100

-- The theorem to prove that the percentage increase in gross income is 3.5%
theorem income_increase : percentage_increase = 3.5 := 
by
  -- This is where the proof would go
  sorry

end income_increase_l805_80525


namespace probability_B_does_not_lose_l805_80555

def prob_A_wins : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Theorem: the probability that B does not lose is 70%.
theorem probability_B_does_not_lose : prob_A_wins + prob_draw ≤ 1 → 1 - prob_A_wins - (1 - prob_draw - prob_A_wins) = 0.7 := by
  sorry

end probability_B_does_not_lose_l805_80555


namespace sum_of_ages_is_24_l805_80553

def age_problem :=
  ∃ (x y z : ℕ), 2 * x^2 + y^2 + z^2 = 194 ∧ (x + x + y + z = 24)

theorem sum_of_ages_is_24 : age_problem :=
by
  sorry

end sum_of_ages_is_24_l805_80553


namespace find_a_l805_80529

theorem find_a (z a : ℂ) (h1 : ‖z‖ = 2) (h2 : (z - a)^2 = a) : a = 2 :=
sorry

end find_a_l805_80529


namespace man_walking_speed_l805_80543

theorem man_walking_speed (length_of_bridge : ℝ) (time_to_cross : ℝ) 
  (h1 : length_of_bridge = 1250) (h2 : time_to_cross = 15) : 
  (length_of_bridge / time_to_cross) * (60 / 1000) = 5 := 
sorry

end man_walking_speed_l805_80543


namespace find_b_for_continuity_at_2_l805_80556

noncomputable def f (x : ℝ) (b : ℝ) :=
if x ≤ 2 then 3 * x^2 + 1 else b * x - 6

theorem find_b_for_continuity_at_2
  (b : ℝ) 
  (h_cont : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) :
  b = 19 / 2 := by sorry

end find_b_for_continuity_at_2_l805_80556


namespace min_sum_abc_l805_80596

theorem min_sum_abc (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c + b * c + c = 2014) : a + b + c = 40 :=
sorry

end min_sum_abc_l805_80596


namespace problem1_problem2_l805_80576

-- Definition and conditions
def i := Complex.I

-- Problem 1
theorem problem1 : (2 + 2 * i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i)) ^ 2010 = -1 := 
by
  sorry

-- Problem 2
theorem problem2 : (4 - i^5) * (6 + 2 * i^7) + (7 + i^11) * (4 - 3 * i) = 47 - 39 * i := 
by
  sorry

end problem1_problem2_l805_80576


namespace ratio_trumpet_to_flute_l805_80567

-- Given conditions
def flute_players : ℕ := 5
def trumpet_players (T : ℕ) : ℕ := T
def trombone_players (T : ℕ) : ℕ := T - 8
def drummers (T : ℕ) : ℕ := T - 8 + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players (T : ℕ) : ℕ := T - 8 + 3
def total_seats_needed (T : ℕ) : ℕ := 
  flute_players + trumpet_players T + trombone_players T + drummers T + clarinet_players + french_horn_players T

-- Proof statement
theorem ratio_trumpet_to_flute 
  (T : ℕ) (h : total_seats_needed T = 65) : trumpet_players T / flute_players = 3 :=
sorry

end ratio_trumpet_to_flute_l805_80567


namespace final_sum_l805_80528

-- Assuming an initial condition for the values on the calculators
def initial_values : List Int := [2, 1, -1]

-- Defining the operations to be applied on the calculators
def operations (vals : List Int) : List Int :=
  match vals with
  | [a, b, c] => [a * a, b * b * b, -c]
  | _ => vals  -- This case handles unexpected input formats

-- Applying the operations for 43 participants
def final_values (vals : List Int) (n : Nat) : List Int :=
  if n = 0 then vals
  else final_values (operations vals) (n - 1)

-- Prove that the final sum of the values on the calculators equals 2 ^ 2 ^ 43
theorem final_sum : 
  final_values initial_values 43 = [2 ^ 2 ^ 43, 1, -1] → 
  List.sum (final_values initial_values 43) = 2 ^ 2 ^ 43 :=
by
  intro h -- This introduces the hypothesis that the final values list equals the expected values
  sorry   -- Provide an ultimate proof for the statement.

end final_sum_l805_80528


namespace minimum_y_value_inequality_proof_l805_80506
-- Import necessary Lean library

-- Define a > 0, b > 0, and a + b = 1
variables {a b : ℝ}
variables (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 1)

-- Statement for Part (I): Prove the minimum value of y is 25/4
theorem minimum_y_value :
  (a + 1/a) * (b + 1/b) = 25/4 :=
sorry

-- Statement for Part (II): Prove the inequality
theorem inequality_proof :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 :=
sorry

end minimum_y_value_inequality_proof_l805_80506


namespace tree_height_at_two_years_l805_80568

variable (h : ℕ → ℕ)

-- Given conditions
def condition1 := h 4 = 81
def condition2 := ∀ t : ℕ, h (t + 1) = 3 * h t

theorem tree_height_at_two_years
  (h_tripled : ∀ t : ℕ, h (t + 1) = 3 * h t)
  (h_at_four : h 4 = 81) :
  h 2 = 9 :=
by
  -- Formal proof will be provided here
  sorry

end tree_height_at_two_years_l805_80568


namespace pieces_of_gum_per_nickel_l805_80504

-- Definitions based on the given conditions
def initial_nickels : ℕ := 5
def remaining_nickels : ℕ := 2
def total_gum_pieces : ℕ := 6

-- We need to prove that Quentavious gets 2 pieces of gum per nickel.
theorem pieces_of_gum_per_nickel 
  (initial_nickels remaining_nickels total_gum_pieces : ℕ)
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum_pieces = 6) :
  total_gum_pieces / (initial_nickels - remaining_nickels) = 2 :=
by {
  sorry
}

end pieces_of_gum_per_nickel_l805_80504


namespace work_complete_in_15_days_l805_80501

theorem work_complete_in_15_days :
  let A_rate := (1 : ℚ) / 20
  let B_rate := (1 : ℚ) / 30
  let C_rate := (1 : ℚ) / 10
  let all_together_rate := A_rate + B_rate + C_rate
  let work_2_days := 2 * all_together_rate
  let B_C_rate := B_rate + C_rate
  let work_next_2_days := 2 * B_C_rate
  let total_work_4_days := work_2_days + work_next_2_days
  let remaining_work := 1 - total_work_4_days
  let B_time := remaining_work / B_rate

  2 + 2 + B_time = 15 :=
by
  sorry

end work_complete_in_15_days_l805_80501


namespace tangent_lines_to_circle_passing_through_point_l805_80539

theorem tangent_lines_to_circle_passing_through_point :
  ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 1 → ((x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = -1)) :=
by
  sorry

end tangent_lines_to_circle_passing_through_point_l805_80539


namespace pizza_cost_l805_80503

theorem pizza_cost
  (initial_money_frank : ℕ)
  (initial_money_bill : ℕ)
  (final_money_bill : ℕ)
  (pizza_cost : ℕ)
  (number_of_pizzas : ℕ)
  (money_given_to_bill : ℕ) :
  initial_money_frank = 42 ∧
  initial_money_bill = 30 ∧
  final_money_bill = 39 ∧
  number_of_pizzas = 3 ∧
  money_given_to_bill = final_money_bill - initial_money_bill →
  3 * pizza_cost + money_given_to_bill = initial_money_frank →
  pizza_cost = 11 :=
by
  sorry

end pizza_cost_l805_80503


namespace problem_statement_l805_80546

theorem problem_statement 
  (h1 : 17 ≡ 3 [MOD 7])
  (h2 : 3^1 ≡ 3 [MOD 7])
  (h3 : 3^2 ≡ 2 [MOD 7])
  (h4 : 3^3 ≡ 6 [MOD 7])
  (h5 : 3^4 ≡ 4 [MOD 7])
  (h6 : 3^5 ≡ 5 [MOD 7])
  (h7 : 3^6 ≡ 1 [MOD 7])
  (h8 : 3^100 ≡ 4 [MOD 7]) :
  17^100 ≡ 4 [MOD 7] :=
by sorry

end problem_statement_l805_80546


namespace correct_option_c_l805_80548

theorem correct_option_c (a : ℝ) : (-2 * a) ^ 3 = -8 * a ^ 3 :=
sorry

end correct_option_c_l805_80548


namespace minimum_n_for_obtuse_triangle_l805_80587

def α₀ : ℝ := 60 
def β₀ : ℝ := 59.999
def γ₀ : ℝ := 60.001

def α (n : ℕ) : ℝ := (-2)^n * (α₀ - 60) + 60
def β (n : ℕ) : ℝ := (-2)^n * (β₀ - 60) + 60
def γ (n : ℕ) : ℝ := (-2)^n * (γ₀ - 60) + 60

theorem minimum_n_for_obtuse_triangle : ∃ n : ℕ, β n > 90 ∧ ∀ m : ℕ, m < n → β m ≤ 90 :=
by sorry

end minimum_n_for_obtuse_triangle_l805_80587


namespace jason_fish_count_ninth_day_l805_80599

def fish_growth_day1 := 8 * 3
def fish_growth_day2 := fish_growth_day1 * 3
def fish_growth_day3 := fish_growth_day2 * 3
def fish_day4_removed := 2 / 5 * fish_growth_day3
def fish_after_day4 := fish_growth_day3 - fish_day4_removed
def fish_growth_day5 := fish_after_day4 * 3
def fish_growth_day6 := fish_growth_day5 * 3
def fish_day6_removed := 3 / 7 * fish_growth_day6
def fish_after_day6 := fish_growth_day6 - fish_day6_removed
def fish_growth_day7 := fish_after_day6 * 3
def fish_growth_day8 := fish_growth_day7 * 3
def fish_growth_day9 := fish_growth_day8 * 3
def fish_final := fish_growth_day9 + 20

theorem jason_fish_count_ninth_day : fish_final = 18083 :=
by
  -- proof steps will go here
  sorry

end jason_fish_count_ninth_day_l805_80599


namespace S_of_1_eq_8_l805_80551

variable (x : ℝ)

-- Definition of original polynomial R(x)
def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

-- Definition of new polynomial S(x) created by adding 2 to each coefficient of R(x)
def S (x : ℝ) : ℝ := 5 * x^3 - 3 * x + 6

-- The theorem we want to prove
theorem S_of_1_eq_8 : S 1 = 8 := by
  sorry

end S_of_1_eq_8_l805_80551


namespace second_set_number_l805_80527

theorem second_set_number (x : ℕ) (sum1 : ℕ) (avg2 : ℕ) (total_avg : ℕ)
  (h1 : sum1 = 98) (h2 : avg2 = 11) (h3 : total_avg = 8)
  (h4 : 16 + x ≠ 0) :
  (98 + avg2 * x = total_avg * (x + 16)) → x = 10 :=
by
  sorry

end second_set_number_l805_80527


namespace parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l805_80531

section vector

variables {k : ℝ}
def a : ℝ × ℝ := (6, 2)
def b : ℝ × ℝ := (-2, k)

-- Parallel condition
theorem parallel_vectors : 
  (∀ c : ℝ, (6, 2) = -2 * (c * k, c)) → k = -2 / 3 :=
by 
  sorry

-- Perpendicular condition
theorem perpendicular_vectors : 
  6 * (-2) + 2 * k = 0 → k = 6 :=
by 
  sorry

-- Obtuse angle condition
theorem obtuse_angle_vectors : 
  6 * (-2) + 2 * k < 0 ∧ k ≠ -2 / 3 → k < 6 ∧ k ≠ -2 / 3 :=
by 
  sorry

end vector

end parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l805_80531


namespace find_x_l805_80537

theorem find_x (x : ℝ) (h : 0.35 * 400 = 0.20 * x): x = 700 :=
sorry

end find_x_l805_80537


namespace least_pos_int_x_l805_80583

theorem least_pos_int_x (x : ℕ) (h1 : ∃ k : ℤ, (3 * x + 43) = 53 * k) 
  : x = 21 :=
sorry

end least_pos_int_x_l805_80583


namespace balloon_ratio_l805_80589

/-- Janice has 6 water balloons. --/
def Janice_balloons : Nat := 6

/-- Randy has half as many water balloons as Janice. --/
def Randy_balloons : Nat := Janice_balloons / 2

/-- Cynthia has 12 water balloons. --/
def Cynthia_balloons : Nat := 12

/-- The ratio of Cynthia's water balloons to Randy's water balloons is 4:1. --/
theorem balloon_ratio : Cynthia_balloons / Randy_balloons = 4 := by
  sorry

end balloon_ratio_l805_80589


namespace calculate_value_l805_80518

theorem calculate_value :
  let number := 1.375
  let coef := 0.6667
  let increment := 0.75
  coef * number + increment = 1.666675 :=
by
  sorry

end calculate_value_l805_80518


namespace student_competition_distribution_l805_80534

theorem student_competition_distribution :
  ∃ f : Fin 4 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ x : Fin 4, f x = i ∧ ∃ y : Fin 4, f y = j) ∧ 
  (Finset.univ.image f).card = 3 := 
sorry

end student_competition_distribution_l805_80534


namespace total_points_other_team_members_l805_80552

variable (x y : ℕ)

theorem total_points_other_team_members :
  (1 / 3 * x + 3 / 8 * x + 18 + y = x) ∧ (y ≤ 24) → y = 17 :=
by
  intro h
  have h1 : 1 / 3 * x + 3 / 8 * x + 18 + y = x := h.1
  have h2 : y ≤ 24 := h.2
  sorry

end total_points_other_team_members_l805_80552


namespace no_negative_roots_of_P_l805_80559

def P (x : ℝ) : ℝ := x^4 - 5 * x^3 + 3 * x^2 - 7 * x + 1

theorem no_negative_roots_of_P : ∀ x : ℝ, P x = 0 → x ≥ 0 := 
by 
    sorry

end no_negative_roots_of_P_l805_80559


namespace common_points_circle_ellipse_l805_80554

theorem common_points_circle_ellipse :
    (∃ (p1 p2: ℝ × ℝ),
        p1 ≠ p2 ∧
        (p1, p2).fst.1 ^ 2 + (p1, p2).fst.2 ^ 2 = 4 ∧
        9 * (p1, p2).fst.1 ^ 2 + 4 * (p1, p2).fst.2 ^ 2 = 36 ∧
        (p1, p2).snd.1 ^ 2 + (p1, p2).snd.2 ^ 2 = 4 ∧
        9 * (p1, p2).snd.1 ^ 2 + 4 * (p1, p2).snd.2 ^ 2 = 36) :=
sorry

end common_points_circle_ellipse_l805_80554


namespace percent_increase_from_may_to_june_l805_80561

noncomputable def profit_increase_from_march_to_april (P : ℝ) : ℝ := 1.30 * P
noncomputable def profit_decrease_from_april_to_may (P : ℝ) : ℝ := 1.04 * P
noncomputable def profit_increase_from_march_to_june (P : ℝ) : ℝ := 1.56 * P

theorem percent_increase_from_may_to_june (P : ℝ) :
  (1.04 * P * (1 + 0.50)) = 1.56 * P :=
by
  sorry

end percent_increase_from_may_to_june_l805_80561


namespace apple_cost_price_l805_80535

theorem apple_cost_price (SP : ℝ) (loss_frac : ℝ) (CP : ℝ) (h_SP : SP = 19) (h_loss_frac : loss_frac = 1 / 6) (h_loss : SP = CP - loss_frac * CP) : CP = 22.8 :=
by
  sorry

end apple_cost_price_l805_80535


namespace max_ab_of_tangent_circles_l805_80577

theorem max_ab_of_tangent_circles (a b : ℝ) 
  (hC1 : ∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4)
  (hC2 : ∀ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1)
  (h_tangent : a + b = 3) :
  ab ≤ 9 / 4 :=
by
  sorry

end max_ab_of_tangent_circles_l805_80577


namespace find_number_l805_80522

theorem find_number (N : ℝ) 
  (h1 : (5 / 6) * N = (5 / 16) * N + 200) : 
  N = 384 :=
sorry

end find_number_l805_80522


namespace B_profit_percentage_l805_80532

theorem B_profit_percentage (cost_price_A : ℝ) (profit_A : ℝ) (selling_price_C : ℝ) 
  (h1 : cost_price_A = 154) 
  (h2 : profit_A = 0.20) 
  (h3 : selling_price_C = 231) : 
  (selling_price_C - (cost_price_A * (1 + profit_A))) / (cost_price_A * (1 + profit_A)) * 100 = 25 :=
by
  sorry

end B_profit_percentage_l805_80532


namespace third_vertex_y_coordinate_correct_l805_80575

noncomputable def third_vertex_y_coordinate (x1 y1 x2 y2 : ℝ) (h : y1 = y2) (h_dist : |x1 - x2| = 10) : ℝ :=
  y1 + 5 * Real.sqrt 3

theorem third_vertex_y_coordinate_correct : 
  third_vertex_y_coordinate 3 4 13 4 rfl (by norm_num) = 4 + 5 * Real.sqrt 3 :=
by
  sorry

end third_vertex_y_coordinate_correct_l805_80575


namespace polynomial_solution_l805_80508

theorem polynomial_solution (P : ℝ → ℝ) :
  (∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))) →
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x ^ 4 + β * x ^ 2 :=
by
  sorry

end polynomial_solution_l805_80508


namespace total_walnut_trees_l805_80560

-- Define the conditions
def current_walnut_trees := 4
def new_walnut_trees := 6

-- State the lean proof problem
theorem total_walnut_trees : current_walnut_trees + new_walnut_trees = 10 := by
  sorry

end total_walnut_trees_l805_80560


namespace no_solution_for_equation_l805_80595

theorem no_solution_for_equation :
  ¬ ∃ x : ℝ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  sorry

end no_solution_for_equation_l805_80595


namespace bobby_toy_cars_l805_80547

theorem bobby_toy_cars (initial_cars : ℕ) (increase_rate : ℕ → ℕ) (n : ℕ) :
  initial_cars = 16 →
  increase_rate 1 = initial_cars + (initial_cars / 2) →
  increase_rate 2 = increase_rate 1 + (increase_rate 1 / 2) →
  increase_rate 3 = increase_rate 2 + (increase_rate 2 / 2) →
  n = 3 →
  increase_rate n = 54 :=
by
  intros
  sorry

end bobby_toy_cars_l805_80547


namespace sum_of_numbers_l805_80579

variable (x y S : ℝ)
variable (H1 : x + y = S)
variable (H2 : x * y = 375)
variable (H3 : (1 / x) + (1 / y) = 0.10666666666666667)

theorem sum_of_numbers (H1 : x + y = S) (H2 : x * y = 375) (H3 : (1 / x) + (1 / y) = 0.10666666666666667) : S = 40 :=
by {
  sorry
}

end sum_of_numbers_l805_80579


namespace largest_apartment_size_l805_80540

theorem largest_apartment_size (rent_per_sqft : ℝ) (budget : ℝ) (s : ℝ) :
  rent_per_sqft = 0.9 →
  budget = 630 →
  s = budget / rent_per_sqft →
  s = 700 :=
by
  sorry

end largest_apartment_size_l805_80540


namespace history_paper_pages_l805_80544

theorem history_paper_pages (days: ℕ) (pages_per_day: ℕ) (h₁: days = 3) (h₂: pages_per_day = 27) : days * pages_per_day = 81 := 
by
  sorry

end history_paper_pages_l805_80544


namespace no_power_of_two_divides_3n_plus_1_l805_80574

theorem no_power_of_two_divides_3n_plus_1 (n : ℕ) (hn : n > 1) : ¬ (2^n ∣ 3^n + 1) := sorry

end no_power_of_two_divides_3n_plus_1_l805_80574


namespace distance_from_home_to_school_l805_80538

theorem distance_from_home_to_school
  (x y : ℝ)
  (h1 : x = y / 3)
  (h2 : x = (y + 18) / 5) : x = 9 := 
by
  sorry

end distance_from_home_to_school_l805_80538


namespace find_sum_of_coefficients_l805_80530

theorem find_sum_of_coefficients
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + x^3 - 2 * x^2 + 17 * x - 5) :
  a + b + c + d = 5 :=
by
  sorry

end find_sum_of_coefficients_l805_80530


namespace findPerpendicularLine_l805_80521

-- Defining the condition: the line passes through point (-1, 2)
def pointOnLine (x y : ℝ) (a b : ℝ) (c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Defining the condition: the line is perpendicular to 2x - 3y + 4 = 0
def isPerpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

-- The original line equation: 2x - 3y + 4 = 0
def originalLine (x y : ℝ) : Prop :=
  2 * x - 3 * y + 4 = 0

-- The target equation of the line: 3x + 2y - 1 = 0
def targetLine (x y : ℝ) : Prop :=
  3 * x + 2 * y - 1 = 0

theorem findPerpendicularLine :
  (pointOnLine (-1) 2 3 2 (-1)) ∧
  (isPerpendicular 3 2 2 (-3)) →
  (∀ x y, targetLine x y ↔ 3 * x + 2 * y - 1 = 0) :=
by
  sorry

end findPerpendicularLine_l805_80521


namespace fraction_checked_by_worker_y_l805_80523

variables (P X Y : ℕ)
variables (defective_rate_x defective_rate_y total_defective_rate : ℚ)
variables (h1 : X + Y = P)
variables (h2 : defective_rate_x = 0.005)
variables (h3 : defective_rate_y = 0.008)
variables (h4 : total_defective_rate = 0.007)
variables (defective_x : ℚ := 0.005 * X)
variables (defective_y : ℚ := 0.008 * Y)
variables (total_defective_products : ℚ := 0.007 * P)
variables (h5 : defective_x + defective_y = total_defective_products)

theorem fraction_checked_by_worker_y : Y / P = 2 / 3 :=
by sorry

end fraction_checked_by_worker_y_l805_80523


namespace range_of_f_ge_1_l805_80510

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (x + 1) ^ 2 else 4 - Real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end range_of_f_ge_1_l805_80510


namespace remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l805_80514

theorem remainder_7_mul_12_pow_24_add_2_pow_24_mod_13 :
  (7 * 12^24 + 2^24) % 13 = 8 := by
  sorry

end remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l805_80514
