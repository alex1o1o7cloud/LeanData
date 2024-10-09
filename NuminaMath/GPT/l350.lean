import Mathlib

namespace worker_schedule_l350_35047

open Nat

theorem worker_schedule (x : ℕ) :
  24 * 3 + (15 - 3) * x > 408 :=
by
  sorry

end worker_schedule_l350_35047


namespace find_a_l350_35063

noncomputable section

def f (x a : ℝ) : ℝ := Real.sqrt (1 + a * 4^x)

theorem find_a (a : ℝ) : 
  (∀ (x : ℝ), x ≤ -1 → 1 + a * 4^x ≥ 0) → a = -4 :=
sorry

end find_a_l350_35063


namespace polynomial_division_properties_l350_35076

open Polynomial

noncomputable def g : Polynomial ℝ := 3 * X^4 + 9 * X^3 - 7 * X^2 + 2 * X + 5
noncomputable def e : Polynomial ℝ := X^2 + 2 * X - 3

theorem polynomial_division_properties (s t : Polynomial ℝ) (h : g = s * e + t) (h_deg : t.degree < e.degree) :
  s.eval 1 + t.eval (-1) = -22 :=
sorry

end polynomial_division_properties_l350_35076


namespace three_sleep_simultaneously_l350_35097

noncomputable def professors := Finset.range 5

def sleeping_times (p: professors) : Finset ℕ 
-- definition to be filled in, stating that p falls asleep twice.
:= sorry 

def moment_two_asleep (p q: professors) : ℕ 
-- definition to be filled in, stating that p and q are asleep together once.
:= sorry

theorem three_sleep_simultaneously :
  ∃ t : ℕ, ∃ p1 p2 p3 : professors, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  (t ∈ sleeping_times p1) ∧
  (t ∈ sleeping_times p2) ∧
  (t ∈ sleeping_times p3) := by
  sorry

end three_sleep_simultaneously_l350_35097


namespace radishes_difference_l350_35001

theorem radishes_difference 
    (total_radishes : ℕ)
    (groups : ℕ)
    (first_basket : ℕ)
    (second_basket : ℕ)
    (total_radishes_eq : total_radishes = 88)
    (groups_eq : groups = 4)
    (first_basket_eq : first_basket = 37)
    (second_basket_eq : second_basket = total_radishes - first_basket)
  : second_basket - first_basket = 14 :=
by
  sorry

end radishes_difference_l350_35001


namespace geometric_series_sum_l350_35055

/-- The first term of the geometric series. -/
def a : ℚ := 3

/-- The common ratio of the geometric series. -/
def r : ℚ := -3 / 4

/-- The sum of the geometric series is equal to 12/7. -/
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 12 / 7 := 
by
  /- The Sum function and its properties for the geometric series will be used here. -/
  sorry

end geometric_series_sum_l350_35055


namespace lemons_needed_for_3_dozen_is_9_l350_35084

-- Define the conditions
def lemon_tbs : ℕ := 4
def juice_needed_per_dozen : ℕ := 12
def dozens_needed : ℕ := 3
def total_juice_needed : ℕ := juice_needed_per_dozen * dozens_needed

-- The number of lemons needed to make 3 dozen cupcakes
def lemons_needed (total_juice : ℕ) (lemon_juice : ℕ) : ℕ :=
  total_juice / lemon_juice

-- Prove the number of lemons needed == 9
theorem lemons_needed_for_3_dozen_is_9 : lemons_needed total_juice_needed lemon_tbs = 9 :=
  by sorry

end lemons_needed_for_3_dozen_is_9_l350_35084


namespace net_effect_on_sale_l350_35005

variable (P Q : ℝ) -- Price and Quantity

theorem net_effect_on_sale :
  let reduced_price := 0.40 * P
  let increased_quantity := 2.50 * Q
  let price_after_tax := 0.44 * P
  let price_after_discount := 0.418 * P
  let final_revenue := price_after_discount * increased_quantity 
  let original_revenue := P * Q
  final_revenue / original_revenue = 1.045 :=
by
  sorry

end net_effect_on_sale_l350_35005


namespace seth_spent_more_on_ice_cream_l350_35093

-- Definitions based on the conditions
def cartons_ice_cream := 20
def cartons_yogurt := 2
def cost_per_carton_ice_cream := 6
def cost_per_carton_yogurt := 1

-- Theorem statement
theorem seth_spent_more_on_ice_cream :
  (cartons_ice_cream * cost_per_carton_ice_cream) - (cartons_yogurt * cost_per_carton_yogurt) = 118 :=
by
  sorry

end seth_spent_more_on_ice_cream_l350_35093


namespace remainder_of_number_of_minimally_intersecting_triples_l350_35039

noncomputable def number_of_minimally_intersecting_triples : Nat :=
  let n := (8 * 7 * 6) * (4 ^ 5)
  n % 1000

theorem remainder_of_number_of_minimally_intersecting_triples :
  number_of_minimally_intersecting_triples = 64 := by
  sorry

end remainder_of_number_of_minimally_intersecting_triples_l350_35039


namespace unique_intersection_l350_35028

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end unique_intersection_l350_35028


namespace friendly_sequences_exist_l350_35072

theorem friendly_sequences_exist :
  ∃ (a b : ℕ → ℕ), 
    (∀ n, a n = 2^(n-1)) ∧ 
    (∀ n, b n = 2*n - 1) ∧ 
    (∀ k : ℕ, ∃ (i j : ℕ), k = a i * b j) :=
by
  sorry

end friendly_sequences_exist_l350_35072


namespace ratio_equivalence_l350_35079

theorem ratio_equivalence (x : ℝ) :
  ((20 / 10) * 100 = (25 / x) * 100) → x = 12.5 :=
by
  intro h
  sorry

end ratio_equivalence_l350_35079


namespace rate_of_interest_l350_35081

theorem rate_of_interest (P A T SI : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 2)
  (h4 : SI = A - P) (h5 : SI = (P * R * T) / 100) : R = 10 :=
by
  sorry

end rate_of_interest_l350_35081


namespace find_f_x_l350_35075

theorem find_f_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x - 1) : 
  ∀ x : ℤ, f x = 2 * x - 3 :=
sorry

end find_f_x_l350_35075


namespace largest_among_options_l350_35004

theorem largest_among_options (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > (1/2) ∧ b > a^2 + b^2 ∧ b > 2*a*b := 
by
  sorry

end largest_among_options_l350_35004


namespace convert_decimal_to_fraction_l350_35021

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end convert_decimal_to_fraction_l350_35021


namespace probability_point_in_square_l350_35031

theorem probability_point_in_square (r : ℝ) (hr : 0 < r) :
  (∃ p : ℝ, p = 2 / Real.pi) :=
by
  sorry

end probability_point_in_square_l350_35031


namespace negative_half_less_than_negative_third_l350_35051

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l350_35051


namespace find_range_of_a_l350_35085

def quadratic_function (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 1) * x + 2

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def is_monotonic_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  is_decreasing_on f I ∨ is_increasing_on f I

theorem find_range_of_a (a : ℝ) :
  is_monotonic_on (quadratic_function a) (Set.Icc (-4) 4) ↔ (a ≤ -3 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l350_35085


namespace set_intersection_eq_l350_35038

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- The proof statement
theorem set_intersection_eq :
  A ∩ B = A :=
sorry

end set_intersection_eq_l350_35038


namespace arithmetic_progression_of_squares_l350_35048

theorem arithmetic_progression_of_squares 
  (a b c : ℝ)
  (h : 1 / (a + b) - 1 / (a + c) = 1 / (b + c) - 1 / (a + c)) :
  2 * b^2 = a^2 + c^2 :=
by
  sorry

end arithmetic_progression_of_squares_l350_35048


namespace new_price_of_computer_l350_35008

theorem new_price_of_computer (d : ℝ) (h : 2 * d = 520) : d * 1.3 = 338 := 
sorry

end new_price_of_computer_l350_35008


namespace proof_sets_l350_35083

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}
def complement (s : Set ℕ) : Set ℕ := {x | x ∈ I ∧ x ∉ s}

theorem proof_sets :
  M ∩ (complement N) = {4, 5} ∧ {2, 7, 8} = complement (M ∪ N) :=
by
  sorry

end proof_sets_l350_35083


namespace find_first_number_l350_35074

theorem find_first_number (x : ℝ) : (x + 16 + 8 + 22) / 4 = 13 ↔ x = 6 :=
by 
  sorry

end find_first_number_l350_35074


namespace star_neg5_4_star_neg3_neg6_l350_35010

-- Definition of the new operation
def star (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- The first proof problem
theorem star_neg5_4 : star (-5) 4 = -42 := by sorry

-- The second proof problem
theorem star_neg3_neg6 : star (-3) (-6) = 39 := by sorry

end star_neg5_4_star_neg3_neg6_l350_35010


namespace exists_composite_l350_35064

theorem exists_composite (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, x^(2^n) + y^(2^n) = k * (k + 1) :=
by {
  sorry -- proof goes here
}

end exists_composite_l350_35064


namespace gre_exam_month_l350_35012

def months_of_year := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def start_month := "June"
def preparation_duration := 5

theorem gre_exam_month :
  months_of_year[(months_of_year.indexOf start_month + preparation_duration) % 12] = "November" := by
  sorry

end gre_exam_month_l350_35012


namespace joan_balloon_gain_l350_35036

theorem joan_balloon_gain
  (initial_balloons : ℕ)
  (final_balloons : ℕ)
  (h_initial : initial_balloons = 9)
  (h_final : final_balloons = 11) :
  final_balloons - initial_balloons = 2 :=
by {
  sorry
}

end joan_balloon_gain_l350_35036


namespace number_of_correct_conclusions_l350_35067

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def f (x : ℝ) : ℝ := x - floor x

theorem number_of_correct_conclusions : 
  ∃ n, n = 3 ∧ 
  (0 ≤ f 0) ∧ 
  (∀ x : ℝ, 0 ≤ f x) ∧ 
  (∀ x : ℝ, f x < 1) ∧ 
  (∀ x : ℝ, f (x + 1) = f x) ∧ 
  ¬ (∀ x : ℝ, f (-x) = f x) := 
sorry

end number_of_correct_conclusions_l350_35067


namespace euler_totient_divisibility_l350_35000

theorem euler_totient_divisibility (n : ℕ) (h : n > 0) : n ∣ Nat.totient (2^n - 1) := by
  sorry

end euler_totient_divisibility_l350_35000


namespace number_difference_l350_35015

theorem number_difference (a b : ℕ) (h1 : a + b = 44) (h2 : 8 * a = 3 * b) : b - a = 20 := by
  sorry

end number_difference_l350_35015


namespace chadSavingsIsCorrect_l350_35069

noncomputable def chadSavingsAfterTaxAndConversion : ℝ :=
  let euroToUsd := 1.20
  let poundToUsd := 1.40
  let euroIncome := 600 * euroToUsd
  let poundIncome := 250 * poundToUsd
  let dollarIncome := 150 + 150
  let totalIncome := euroIncome + poundIncome + dollarIncome
  let taxRate := 0.10
  let taxedIncome := totalIncome * (1 - taxRate)
  let savingsRate := if taxedIncome ≤ 1000 then 0.20
                     else if taxedIncome ≤ 2000 then 0.30
                     else if taxedIncome ≤ 3000 then 0.40
                     else 0.50
  let savings := taxedIncome * savingsRate
  savings

theorem chadSavingsIsCorrect : chadSavingsAfterTaxAndConversion = 369.90 := by
  sorry

end chadSavingsIsCorrect_l350_35069


namespace digit_68th_is_1_l350_35086

noncomputable def largest_n : ℕ :=
  (10^100 - 1) / 14

def digit_at_68th_place (n : ℕ) : ℕ :=
  (n / 10^(68 - 1)) % 10

theorem digit_68th_is_1 : digit_at_68th_place largest_n = 1 :=
sorry

end digit_68th_is_1_l350_35086


namespace average_age_remains_l350_35090

theorem average_age_remains (total_age : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) (initial_people_avg : ℕ) 
                            (total_age_eq : total_age = initial_people_avg * 8) 
                            (new_total_age : ℕ := total_age - leaving_age)
                            (new_avg : ℝ := new_total_age / remaining_people) :
  (initial_people_avg = 25) ∧ (leaving_age = 20) ∧ (remaining_people = 7) → new_avg = 180 / 7 := 
by
  sorry

end average_age_remains_l350_35090


namespace clown_blew_more_balloons_l350_35096

theorem clown_blew_more_balloons :
  ∀ (initial_balloons final_balloons additional_balloons : ℕ),
    initial_balloons = 47 →
    final_balloons = 60 →
    additional_balloons = final_balloons - initial_balloons →
    additional_balloons = 13 :=
by
  intros initial_balloons final_balloons additional_balloons h1 h2 h3
  sorry

end clown_blew_more_balloons_l350_35096


namespace gardener_payment_l350_35040

theorem gardener_payment (total_cost : ℕ) (rect_area : ℕ) (rect_side1 : ℕ) (rect_side2 : ℕ)
                         (square1_area : ℕ) (square2_area : ℕ) (cost_per_are : ℕ) :
  total_cost = 570 →
  rect_area = 600 → rect_side1 = 20 → rect_side2 = 30 →
  square1_area = 400 → square2_area = 900 →
  cost_per_are * (rect_area + square1_area + square2_area) / 100 = total_cost →
  cost_per_are = 30 →
  ∃ (rect_payment : ℕ) (square1_payment : ℕ) (square2_payment : ℕ),
    rect_payment = 6 * cost_per_are ∧
    square1_payment = 4 * cost_per_are ∧
    square2_payment = 9 * cost_per_are ∧
    rect_payment + square1_payment + square2_payment = total_cost :=
by
  intros
  sorry

end gardener_payment_l350_35040


namespace cylindrical_tank_volume_increase_l350_35033

theorem cylindrical_tank_volume_increase (k : ℝ) (H R : ℝ) 
  (hR : R = 10) (hH : H = 5)
  (condition : (π * (10 * k)^2 * 5 - π * 10^2 * 5) = (π * 10^2 * (5 + k) - π * 10^2 * 5)) :
  k = (1 + Real.sqrt 101) / 10 :=
by
  sorry

end cylindrical_tank_volume_increase_l350_35033


namespace permutations_count_l350_35027

-- Define the conditions
variable (n : ℕ)
variable (a : Fin n → ℕ)

-- Define the main proposition
theorem permutations_count (hn : 2 ≤ n) (h_perm : ∀ k : Fin n, a k ≥ k.val - 2) :
  ∃! L, L = 2 * 3 ^ (n - 2) :=
by
  sorry

end permutations_count_l350_35027


namespace neighborhood_has_exactly_one_item_l350_35099

noncomputable def neighborhood_conditions : Prop :=
  let total_households := 120
  let households_no_items := 15
  let households_car_and_bike := 28
  let households_car := 52
  let households_bike := 32
  let households_scooter := 18
  let households_skateboard := 8
  let households_at_least_one_item := total_households - households_no_items
  let households_car_only := households_car - households_car_and_bike
  let households_bike_only := households_bike - households_car_and_bike
  let households_exactly_one_item := households_car_only + households_bike_only + households_scooter + households_skateboard
  households_at_least_one_item = 105 ∧ households_exactly_one_item = 54

theorem neighborhood_has_exactly_one_item :
  neighborhood_conditions :=
by
  -- Proof goes here
  sorry

end neighborhood_has_exactly_one_item_l350_35099


namespace find_n_l350_35053

def Point : Type := ℝ × ℝ

def A : Point := (5, -8)
def B : Point := (9, -30)
def C (n : ℝ) : Point := (n, n)

def collinear (p1 p2 p3 : Point) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_n (n : ℝ) (h : collinear A B (C n)) : n = 3 := 
by
  sorry

end find_n_l350_35053


namespace dima_story_retelling_count_l350_35032

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end dima_story_retelling_count_l350_35032


namespace medium_supermarkets_in_sample_l350_35073

-- Definitions of the conditions
def total_supermarkets : ℕ := 200 + 400 + 1400
def prop_medium_supermarkets : ℚ := 400 / total_supermarkets
def sample_size : ℕ := 100

-- Problem: Prove that the number of medium-sized supermarkets in the sample is 20.
theorem medium_supermarkets_in_sample : 
  (sample_size * prop_medium_supermarkets) = 20 :=
by
  sorry

end medium_supermarkets_in_sample_l350_35073


namespace distance_light_travels_500_years_l350_35052

def distance_light_travels_one_year : ℝ := 5.87e12
def years : ℕ := 500

theorem distance_light_travels_500_years :
  distance_light_travels_one_year * years = 2.935e15 := 
sorry

end distance_light_travels_500_years_l350_35052


namespace overall_gain_is_correct_l350_35070

noncomputable def overall_gain_percentage : ℝ :=
  let CP_A := 100
  let SP_A := 120 / (1 - 0.20)
  let gain_A := SP_A - CP_A

  let CP_B := 200
  let SP_B := 240 / (1 + 0.10)
  let gain_B := SP_B - CP_B

  let CP_C := 150
  let SP_C := (165 / (1 + 0.05)) / (1 - 0.10)
  let gain_C := SP_C - CP_C

  let CP_D := 300
  let SP_D := (345 / (1 - 0.05)) / (1 + 0.15)
  let gain_D := SP_D - CP_D

  let total_gain := gain_A + gain_B + gain_C + gain_D
  let total_CP := CP_A + CP_B + CP_C + CP_D
  (total_gain / total_CP) * 100

theorem overall_gain_is_correct : abs (overall_gain_percentage - 14.48) < 0.01 := by
  sorry

end overall_gain_is_correct_l350_35070


namespace toothpicks_in_20th_stage_l350_35018

theorem toothpicks_in_20th_stage :
  (3 + 3 * (20 - 1) = 60) :=
by
  sorry

end toothpicks_in_20th_stage_l350_35018


namespace find_function_l350_35034

def satisfies_functional_eqn (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem find_function (f : ℝ → ℝ) :
  satisfies_functional_eqn f → (∀ y : ℝ, f y = y^2 - 1) :=
by
  intro h
  sorry

end find_function_l350_35034


namespace ratio_mn_eq_x_plus_one_over_two_x_plus_one_l350_35035

theorem ratio_mn_eq_x_plus_one_over_two_x_plus_one (x : ℝ) (m n : ℝ) 
  (hx : x > 0) 
  (hmn : m * n ≠ 0) 
  (hineq : m * x > n * x + n) : 
  m / (m + n) = (x + 1) / (2 * x + 1) := 
by 
  sorry

end ratio_mn_eq_x_plus_one_over_two_x_plus_one_l350_35035


namespace square_field_area_l350_35043

def square_area (side_length : ℝ) : ℝ :=
  side_length * side_length

theorem square_field_area :
  square_area 20 = 400 := by
  sorry

end square_field_area_l350_35043


namespace max_possible_value_of_y_l350_35046

theorem max_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 :=
sorry

end max_possible_value_of_y_l350_35046


namespace stampsLeftover_l350_35006

-- Define the number of stamps each person has
def oliviaStamps : ℕ := 52
def parkerStamps : ℕ := 66
def quinnStamps : ℕ := 23

-- Define the album's capacity in stamps
def albumCapacity : ℕ := 15

-- Define the total number of leftovers
def totalLeftover : ℕ := (oliviaStamps + parkerStamps + quinnStamps) % albumCapacity

-- Define the theorem we want to prove
theorem stampsLeftover : totalLeftover = 6 := by
  sorry

end stampsLeftover_l350_35006


namespace sequence_fifth_term_l350_35078

theorem sequence_fifth_term (a : ℤ) (d : ℤ) (n : ℕ) (a_n : ℤ) :
  a_n = 89 ∧ d = 11 ∧ n = 5 → a + (n-1) * -d = 45 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  exact sorry

end sequence_fifth_term_l350_35078


namespace sum_of_triangle_angles_sin_halves_leq_one_l350_35056

theorem sum_of_triangle_angles_sin_halves_leq_one (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC : A + B + C = Real.pi) : 
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := 
sorry 

end sum_of_triangle_angles_sin_halves_leq_one_l350_35056


namespace right_triangle_hypotenuse_l350_35044

theorem right_triangle_hypotenuse (a b : ℝ) (m_a m_b : ℝ)
    (h1 : m_a = Real.sqrt (b^2 + (a / 2)^2))
    (h2 : m_b = Real.sqrt (a^2 + (b / 2)^2))
    (h3 : m_a = Real.sqrt 30)
    (h4 : m_b = 6) :
  Real.sqrt (4 * (a^2 + b^2)) = 2 * Real.sqrt 52.8 :=
by
  sorry

end right_triangle_hypotenuse_l350_35044


namespace cost_per_pizza_is_12_l350_35017

def numberOfPeople := 15
def peoplePerPizza := 3
def earningsPerNight := 4
def nightsBabysitting := 15

-- We aim to prove that the cost per pizza is $12
theorem cost_per_pizza_is_12 : 
  (earningsPerNight * nightsBabysitting) / (numberOfPeople / peoplePerPizza) = 12 := 
by 
  sorry

end cost_per_pizza_is_12_l350_35017


namespace geom_seq_sum_l350_35059

theorem geom_seq_sum (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 3 + a 5 = 21)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 3 + a 5 + a 7 = 42 :=
sorry

end geom_seq_sum_l350_35059


namespace evaluate_fraction_l350_35037

-- Let's restate the problem in Lean
theorem evaluate_fraction :
  (∃ q, (2024 / 2023 - 2023 / 2024) = 4047 / q) :=
by
  -- Substitute a = 2023
  let a := 2023
  -- Provide the value we expect for q to hold in the reduced fraction.
  use (a * (a + 1)) -- The expected denominator
  -- The proof for the theorem is omitted here
  sorry

end evaluate_fraction_l350_35037


namespace how_many_roses_cut_l350_35025

theorem how_many_roses_cut :
  ∀ (r_i r_f r_c : ℕ), r_i = 6 → r_f = 16 → r_c = r_f - r_i → r_c = 10 :=
by
  intros r_i r_f r_c hri hrf heq
  rw [hri, hrf] at heq
  exact heq

end how_many_roses_cut_l350_35025


namespace negate_p_l350_35009

theorem negate_p (p : Prop) :
  (∃ x : ℝ, 0 < x ∧ 3^x < x^3) ↔ (¬ (∀ x : ℝ, 0 < x → 3^x ≥ x^3)) :=
by sorry

end negate_p_l350_35009


namespace nat_pair_solution_l350_35011

theorem nat_pair_solution (x y : ℕ) : 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by
  sorry

end nat_pair_solution_l350_35011


namespace tourists_number_l350_35020

theorem tourists_number (m : ℕ) (k l : ℤ) (n : ℕ) (hn : n = 23) (hm1 : 2 * m ≡ 1 [MOD n]) (hm2 : 3 * m ≡ 13 [MOD n]) (hn_gt_13 : n > 13) : n = 23 := 
by
  sorry

end tourists_number_l350_35020


namespace range_of_m_l350_35024

theorem range_of_m {m : ℝ} (h1 : ∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1)))
                   (h2 : ∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))
                   (h3 : ¬(∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1))) ∧
                           (∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))) :
  m > 1 :=
by
  sorry

end range_of_m_l350_35024


namespace min_sum_of_factors_of_72_l350_35087

theorem min_sum_of_factors_of_72 (a b: ℤ) (h: a * b = 72) : a + b = -73 :=
sorry

end min_sum_of_factors_of_72_l350_35087


namespace total_pieces_of_mail_l350_35049

-- Definitions based on given conditions
def pieces_each_friend_delivers : ℕ := 41
def pieces_johann_delivers : ℕ := 98
def number_of_friends : ℕ := 2

-- Theorem statement to prove the total number of pieces of mail delivered
theorem total_pieces_of_mail :
  (number_of_friends * pieces_each_friend_delivers) + pieces_johann_delivers = 180 := 
by
  -- proof would go here
  sorry

end total_pieces_of_mail_l350_35049


namespace paint_after_third_day_l350_35061

def initial_paint := 2
def paint_used_first_day (x : ℕ) := (1 / 2) * x
def remaining_after_first_day (x : ℕ) := x - paint_used_first_day x
def paint_used_second_day (y : ℕ) := (1 / 4) * y
def remaining_after_second_day (y : ℕ) := y - paint_used_second_day y
def paint_used_third_day (z : ℕ) := (1 / 3) * z
def remaining_after_third_day (z : ℕ) := z - paint_used_third_day z

theorem paint_after_third_day :
  remaining_after_third_day 
    (remaining_after_second_day 
      (remaining_after_first_day initial_paint)) = initial_paint / 2 := 
  by
  sorry

end paint_after_third_day_l350_35061


namespace solve_arrangement_equation_l350_35066

def arrangement_numeral (x : ℕ) : ℕ :=
  x * (x - 1) * (x - 2)

theorem solve_arrangement_equation (x : ℕ) (h : 3 * (arrangement_numeral x)^3 = 2 * (arrangement_numeral (x + 1))^2 + 6 * (arrangement_numeral x)^2) : x = 5 := 
sorry

end solve_arrangement_equation_l350_35066


namespace qualifying_rate_l350_35042

theorem qualifying_rate (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = 1 - a - b + a * b :=
by sorry

end qualifying_rate_l350_35042


namespace height_of_flagpole_l350_35092

theorem height_of_flagpole 
  (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) 
  (flagpole_shadow : ℝ) (house_height : ℝ)
  (h1 : house_shadow = 70)
  (h2 : tree_height = 28)
  (h3 : tree_shadow = 40)
  (h4 : flagpole_shadow = 25)
  (h5 : house_height = (tree_height * house_shadow) / tree_shadow) :
  round ((house_height * flagpole_shadow / house_shadow) : ℝ) = 18 := 
by
  sorry

end height_of_flagpole_l350_35092


namespace eccentricity_hyperbola_l350_35014

variables (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variables (H : c = Real.sqrt (a^2 + b^2))
variables (L1 : ∀ x y : ℝ, x = c → (x^2/a^2 - y^2/b^2 = 1))
variables (L2 : ∀ (B C : ℝ × ℝ), (B.1 = c ∧ C.1 = c) ∧ (B.2 = -C.2) ∧ (B.2 = b^2/a))

theorem eccentricity_hyperbola : ∃ e, e = 2 :=
sorry

end eccentricity_hyperbola_l350_35014


namespace price_increase_solution_l350_35071

variable (x : ℕ)

def initial_profit := 10
def initial_sales := 500
def price_increase_effect := 20
def desired_profit := 6000

theorem price_increase_solution :
  ((initial_sales - price_increase_effect * x) * (initial_profit + x) = desired_profit) → (x = 5) :=
by
  sorry

end price_increase_solution_l350_35071


namespace M_diff_N_eq_l350_35077

noncomputable def A_diff_B (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

noncomputable def M : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

noncomputable def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem M_diff_N_eq : A_diff_B M N = { x | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end M_diff_N_eq_l350_35077


namespace smallest_prime_with_digits_sum_22_l350_35041

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end smallest_prime_with_digits_sum_22_l350_35041


namespace cheryl_mms_l350_35023

/-- Cheryl's m&m problem -/
theorem cheryl_mms (c l g d : ℕ) (h1 : c = 25) (h2 : l = 7) (h3 : g = 13) :
  (c - l - g) = d → d = 5 :=
by
  sorry

end cheryl_mms_l350_35023


namespace part_a_part_b_part_c_l350_35094

-- Part a
def can_ratings_increase_after_first_migration (QA_before : ℚ) (QB_before : ℚ) (QA_after : ℚ) (QB_after : ℚ) : Prop :=
  QA_before < QA_after ∧ QB_before < QB_after

-- Part b
def can_ratings_increase_after_second_migration (QA_after_first : ℚ) (QB_after_first : ℚ) (QA_after_second : ℚ) (QB_after_second : ℚ) : Prop :=
  QA_after_second ≤ QA_after_first ∨ QB_after_second ≤ QB_after_first

-- Part c
def can_all_ratings_increase_after_reversed_migration (QA_before : ℚ) (QB_before : ℚ) (QC_before : ℚ) (QA_after_first : ℚ) (QB_after_first : ℚ) (QC_after_first : ℚ)
  (QA_after_second : ℚ) (QB_after_second : ℚ) (QC_after_second : ℚ) : Prop :=
  QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧
  QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second ∧ QC_after_first <= QC_after_second


-- Lean statements
theorem part_a (QA_before QA_after QB_before QB_after : ℚ) (Q_moved : ℚ) 
  (h : QA_before < QA_after ∧ QA_after < Q_moved ∧ QB_before < QB_after ∧ QB_after < Q_moved) : 
  can_ratings_increase_after_first_migration QA_before QB_before QA_after QB_after := 
by sorry

theorem part_b (QA_after_first QB_after_first QA_after_second QB_after_second : ℚ):
  ¬ can_ratings_increase_after_second_migration QA_after_first QB_after_first QA_after_second QB_after_second := 
by sorry

theorem part_c (QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first
  QA_after_second QB_after_second QC_after_second: ℚ)
  (h: QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧ 
      QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second) :
   can_all_ratings_increase_after_reversed_migration QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first QA_after_second QB_after_second QC_after_second :=
by sorry

end part_a_part_b_part_c_l350_35094


namespace g_at_six_l350_35089

def g (x : ℝ) : ℝ := 2 * x^4 - 19 * x^3 + 30 * x^2 - 12 * x - 72

theorem g_at_six : g 6 = 288 :=
by
  sorry

end g_at_six_l350_35089


namespace middle_income_sample_count_l350_35054

def total_households : ℕ := 600
def high_income_families : ℕ := 150
def middle_income_families : ℕ := 360
def low_income_families : ℕ := 90
def sample_size : ℕ := 80

theorem middle_income_sample_count : 
  (middle_income_families / total_households) * sample_size = 48 := 
by
  sorry

end middle_income_sample_count_l350_35054


namespace simplify_expression_l350_35082

variables (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b)

theorem simplify_expression :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) :=
by
  sorry

end simplify_expression_l350_35082


namespace inequality_correct_l350_35026

theorem inequality_correct (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1) : (1 - a) ^ a > (1 - b) ^ b :=
sorry

end inequality_correct_l350_35026


namespace broken_seashells_l350_35050

-- Define the total number of seashells Tom found
def total_seashells : ℕ := 7

-- Define the number of unbroken seashells
def unbroken_seashells : ℕ := 3

-- Prove that the number of broken seashells equals 4
theorem broken_seashells : total_seashells - unbroken_seashells = 4 := by
  sorry

end broken_seashells_l350_35050


namespace inequality_proof_l350_35030

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  abc ≥ (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ∧
  (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end inequality_proof_l350_35030


namespace find_image_point_l350_35095

noncomputable def lens_equation (t f k : ℝ) : Prop :=
  (1 / k) + (1 / t) = (1 / f)

theorem find_image_point
  (O F T T_star K_star K : ℝ)
  (OT OTw OTw_star FK : ℝ)
  (OT_eq : OT = OTw)
  (OTw_star_eq : OTw_star = OT)
  (similarity_condition : ∀ (CTw_star OF : ℝ), CTw_star / OF = (CTw_star + OK) / OK)
  : lens_equation OTw FK K :=
sorry

end find_image_point_l350_35095


namespace math_problem_l350_35060

theorem math_problem : 
  ∃ (n m k : ℕ), 
    (∀ d : ℕ, d ∣ n → d > 0) ∧ 
    (n = m * 6^k) ∧
    (∀ d : ℕ, d ∣ m → 6 ∣ d → False) ∧
    (m + k = 60466182) ∧ 
    (n.factors.count 1 = 2023) :=
sorry

end math_problem_l350_35060


namespace lesser_fraction_exists_l350_35057

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l350_35057


namespace sum_divisible_by_seventeen_l350_35068

theorem sum_divisible_by_seventeen :
  (90 + 91 + 92 + 93 + 94 + 95 + 96 + 97) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_seventeen_l350_35068


namespace exists_increasing_or_decreasing_subsequence_l350_35091

theorem exists_increasing_or_decreasing_subsequence (n : ℕ) (a : Fin (n^2 + 1) → ℝ) :
  ∃ (b : Fin (n + 1) → ℝ), (StrictMono b ∨ StrictAnti b) :=
sorry

end exists_increasing_or_decreasing_subsequence_l350_35091


namespace max_gcd_lcm_l350_35016

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l350_35016


namespace exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l350_35062

open Real EuclideanGeometry

def is_isosceles_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def is_isosceles_triangle_3D (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def five_points_isosceles (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 5, is_isosceles_triangle (pts i) (pts j) (pts k)

def six_points_isosceles (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 6, is_isosceles_triangle (pts i) (pts j) (pts k)

def seven_points_isosceles_3D (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∀ i j k : Fin 7, is_isosceles_triangle_3D (pts i) (pts j) (pts k)

theorem exists_five_points_isosceles : ∃ (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)), five_points_isosceles pts :=
sorry

theorem exists_six_points_isosceles : ∃ (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)), six_points_isosceles pts :=
sorry

theorem exists_seven_points_isosceles_3D : ∃ (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)), seven_points_isosceles_3D pts :=
sorry

end exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l350_35062


namespace number_of_possible_scenarios_l350_35003

-- Definitions based on conditions
def num_companies : Nat := 5
def reps_company_A : Nat := 2
def reps_other_companies : Nat := 1
def total_speakers : Nat := 3

-- Problem statement
theorem number_of_possible_scenarios : 
  ∃ (scenarios : Nat), scenarios = 16 ∧ 
  (scenarios = 
    (Nat.choose reps_company_A 1 * Nat.choose 4 2) + 
    Nat.choose 4 3) :=
by
  sorry

end number_of_possible_scenarios_l350_35003


namespace no_solution_for_m_l350_35013

theorem no_solution_for_m (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (m : ℕ) (h3 : (ab)^2015 = (a^2 + b^2)^m) : false := 
sorry

end no_solution_for_m_l350_35013


namespace f_monotonic_f_odd_find_a_k_range_l350_35019
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

-- (1) Prove the monotonicity of the function f
theorem f_monotonic (a : ℝ) : ∀ {x y : ℝ}, x < y → f a x < f a y := sorry

-- (2) If f is an odd function, find the value of the real number a
theorem f_odd_find_a : ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) → a = -1/2 := sorry

-- (3) Under the condition in (2), if the inequality holds for all x ∈ ℝ, find the range of values for k
theorem k_range (k : ℝ) :
  (∀ x : ℝ, f (-1/2) (x^2 - 2*x) + f (-1/2) (2*x^2 - k) > 0) → k < -1/3 := sorry

end f_monotonic_f_odd_find_a_k_range_l350_35019


namespace contradiction_prop_l350_35007

theorem contradiction_prop (p : Prop) : 
  (∃ x : ℝ, x < -1 ∧ x^2 - x + 1 < 0) → (∀ x : ℝ, x < -1 → x^2 - x + 1 ≥ 0) :=
sorry

end contradiction_prop_l350_35007


namespace find_x_l350_35058

-- Definitions based on conditions
def parabola_eq (y x p : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola (p : ℝ) : Prop := ∃ y x, parabola_eq y x p ∧ (x = 1) ∧ (y = 2)
def valid_p (p : ℝ) : Prop := p > 0
def dist_to_focus (x : ℝ) : ℝ := 1
def dist_to_line (x : ℝ) : ℝ := abs (x + 1)

-- Main statement to be proven
theorem find_x (p : ℝ) (h1 : point_on_parabola p) (h2 : valid_p p) :
  ∃ x, dist_to_focus x = dist_to_line x ∧ x = 1 :=
sorry

end find_x_l350_35058


namespace min_value_of_expression_is_6_l350_35098

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a

theorem min_value_of_expression_is_6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : min_value_of_expression a b c = 6 :=
by
  sorry

end min_value_of_expression_is_6_l350_35098


namespace choir_average_age_l350_35029

-- Each condition as a definition in Lean 4
def avg_age_females := 28
def num_females := 12
def avg_age_males := 32
def num_males := 18
def total_people := num_females + num_males

-- The total sum of ages calculated from the given conditions
def sum_ages_females := avg_age_females * num_females
def sum_ages_males := avg_age_males * num_males
def total_sum_ages := sum_ages_females + sum_ages_males

-- The final proof statement to be proved
theorem choir_average_age : 
  (total_sum_ages : ℝ) / (total_people : ℝ) = 30.4 := by
  sorry

end choir_average_age_l350_35029


namespace gamma_donuts_received_l350_35045

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end gamma_donuts_received_l350_35045


namespace boat_travel_distance_upstream_l350_35002

noncomputable def upstream_distance (v : ℝ) : ℝ :=
  let d := 2.5191640969412834 * (v + 3)
  d

theorem boat_travel_distance_upstream :
  ∀ v : ℝ, 
  (∀ D : ℝ, D / (v + 3) = 2.5191640969412834 → D / (v - 3) = D / (v + 3) + 0.5) → 
  upstream_distance 33.2299691632954 = 91.25 :=
by
  sorry

end boat_travel_distance_upstream_l350_35002


namespace intersection_eq_l350_35065

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end intersection_eq_l350_35065


namespace areas_of_isosceles_triangles_l350_35088

theorem areas_of_isosceles_triangles (A B C : ℝ) (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13)
  (hA : A = 1/2 * a * a) (hB : B = 1/2 * b * b) (hC : C = 1/2 * c * c) :
  A + B = C :=
by
  sorry

end areas_of_isosceles_triangles_l350_35088


namespace square_distance_from_B_to_center_l350_35080

noncomputable def distance_squared (a b : ℝ) : ℝ := a^2 + b^2

theorem square_distance_from_B_to_center :
  ∀ (a b : ℝ),
    (a^2 + (b + 8)^2 = 75) →
    ((a + 2)^2 + b^2 = 75) →
    distance_squared a b = 122 :=
by
  intros a b h1 h2
  sorry

end square_distance_from_B_to_center_l350_35080


namespace sufficient_and_not_necessary_condition_l350_35022

theorem sufficient_and_not_necessary_condition (a b : ℝ) (hb: a < 0 ∧ b < 0) : a + b < 0 :=
by
  sorry

end sufficient_and_not_necessary_condition_l350_35022
