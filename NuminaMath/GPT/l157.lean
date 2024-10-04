import Mathlib

namespace find_marks_in_mathematics_l157_157629

theorem find_marks_in_mathematics
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (subjects : ℕ)
  (marks_math : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  subjects = 5 →
  (average * subjects = english + marks_math + physics + chemistry + biology) →
  marks_math = 95 :=
  by
    intros h_eng h_phy h_chem h_bio h_avg h_sub h_eq
    rw [h_eng, h_phy, h_chem, h_bio, h_avg, h_sub] at h_eq
    sorry

end find_marks_in_mathematics_l157_157629


namespace negative_fraction_comparison_l157_157305

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l157_157305


namespace evaluate_polynomial_at_two_l157_157149

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem evaluate_polynomial_at_two : f 2 = 41 := by
  sorry

end evaluate_polynomial_at_two_l157_157149


namespace radius_ratio_in_right_triangle_l157_157053

theorem radius_ratio_in_right_triangle (PQ QR PR PS SR : ℝ)
  (h₁ : PQ = 5) (h₂ : QR = 12) (h₃ : PR = 13)
  (h₄ : PS + SR = PR) (h₅ : PS / SR = 5 / 8)
  (r_p r_q : ℝ)
  (hr_p : r_p = (1 / 2 * PQ * PS / 3) / ((PQ + PS / 3 + PS) / 3))
  (hr_q : r_q = (1 / 2 * QR * SR) / ((PS / 3 + QR + SR) / 3)) :
  r_p / r_q = 175 / 576 :=
sorry

end radius_ratio_in_right_triangle_l157_157053


namespace quadratic_has_two_real_roots_l157_157682

theorem quadratic_has_two_real_roots (a b c : ℝ) (h : a * c < 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * (x1^2) + b * x1 + c = 0 ∧ a * (x2^2) + b * x2 + c = 0) :=
by
  sorry

end quadratic_has_two_real_roots_l157_157682


namespace total_baseball_cards_is_100_l157_157407

-- Define the initial number of baseball cards Mike has
def initial_baseball_cards : ℕ := 87

-- Define the number of baseball cards Sam gave to Mike
def given_baseball_cards : ℕ := 13

-- Define the total number of baseball cards Mike has now
def total_baseball_cards : ℕ := initial_baseball_cards + given_baseball_cards

-- State the theorem that the total number of baseball cards is 100
theorem total_baseball_cards_is_100 : total_baseball_cards = 100 := by
  sorry

end total_baseball_cards_is_100_l157_157407


namespace value_of_a3_a5_l157_157963

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

theorem value_of_a3_a5 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 :=
  sorry

end value_of_a3_a5_l157_157963


namespace problem_l157_157660

open Set

-- Definitions for set A and set B
def setA : Set ℝ := { x | x^2 + 2 * x - 3 < 0 }
def setB : Set ℤ := { k : ℤ | true }
def evenIntegers : Set ℝ := { x : ℝ | ∃ k : ℤ, x = 2 * k }

-- The intersection of set A and even integers over ℝ
def A_cap_B : Set ℝ := setA ∩ evenIntegers

-- The Proposition that A_cap_B equals {-2, 0}
theorem problem : A_cap_B = ({-2, 0} : Set ℝ) :=
by 
  sorry

end problem_l157_157660


namespace bus_fare_one_way_cost_l157_157543

-- Define the conditions
def zoo_entry (dollars : ℕ) : ℕ := dollars -- Zoo entry cost is $5 per person
def initial_money : ℕ := 40 -- They bring $40 with them
def money_left : ℕ := 24 -- They have $24 left after spending on zoo entry and bus fare

-- Given values
def noah_ava : ℕ := 2 -- Number of persons, Noah and Ava
def zoo_entry_cost : ℕ := 5 -- $5 per person for zoo entry
def total_money_spent := initial_money - money_left -- Money spent on zoo entry and bus fare

-- Function to calculate the total cost based on bus fare x
def total_cost (x : ℕ) : ℕ := noah_ava * zoo_entry_cost + 2 * noah_ava * x

-- Assertion to be proved
theorem bus_fare_one_way_cost : 
  ∃ (x : ℕ), total_cost x = total_money_spent ∧ x = 150 / 100 := sorry

end bus_fare_one_way_cost_l157_157543


namespace Tim_change_l157_157111

theorem Tim_change (initial_amount paid_amount : ℕ) (h₀ : initial_amount = 50) (h₁ : paid_amount = 45) : initial_amount - paid_amount = 5 :=
by
  sorry

end Tim_change_l157_157111


namespace total_rainfall_2007_correct_l157_157208

noncomputable def rainfall_2005 : ℝ := 40.5
noncomputable def rainfall_2006 : ℝ := rainfall_2005 + 3
noncomputable def rainfall_2007 : ℝ := rainfall_2006 + 4
noncomputable def total_rainfall_2007 : ℝ := 12 * rainfall_2007

theorem total_rainfall_2007_correct : total_rainfall_2007 = 570 := 
sorry

end total_rainfall_2007_correct_l157_157208


namespace leila_cakes_monday_l157_157997

def number_of_cakes_monday (m : ℕ) : Prop :=
  let cakes_friday := 9
  let cakes_saturday := 3 * m
  let total_cakes := m + cakes_friday + cakes_saturday
  total_cakes = 33

theorem leila_cakes_monday : ∃ m : ℕ, number_of_cakes_monday m ∧ m = 6 :=
by 
  -- We propose that the number of cakes she ate on Monday, denoted as m, is 6.
  -- We need to prove that this satisfies the given conditions.
  -- This line is a placeholder for the proof.
  sorry

end leila_cakes_monday_l157_157997


namespace solution_l157_157646

namespace ProofProblem

variables (a b : ℝ)

def five_times_a_minus_b_eq_60 := 5 * a - b = 60
def six_times_a_plus_b_lt_90 := 6 * a + b < 90

theorem solution (h1 : five_times_a_minus_b_eq_60 a b) (h2 : six_times_a_plus_b_lt_90 a b) :
  a < 150 / 11 ∧ b < 8.18 :=
sorry

end ProofProblem

end solution_l157_157646


namespace area_of_trapezium_l157_157873

variables (x : ℝ) (h : x > 0)

def shorter_base := 2 * x
def altitude := 2 * x
def longer_base := 6 * x

theorem area_of_trapezium (hx : x > 0) :
  (1 / 2) * (shorter_base x + longer_base x) * altitude x = 8 * x^2 := 
sorry

end area_of_trapezium_l157_157873


namespace aimee_poll_l157_157296

theorem aimee_poll (W P : ℕ) (h1 : 0.35 * W = 21) (h2 : 2 * W = P) : P = 120 :=
by
  -- proof in Lean is omitted, placeholder
  sorry

end aimee_poll_l157_157296


namespace shelves_needed_l157_157137

def books_in_stock : Nat := 27
def books_sold : Nat := 6
def books_per_shelf : Nat := 7

theorem shelves_needed :
  let remaining_books := books_in_stock - books_sold
  let shelves := remaining_books / books_per_shelf
  shelves = 3 :=
by
  sorry

end shelves_needed_l157_157137


namespace cornelia_travel_countries_l157_157950

theorem cornelia_travel_countries (europe south_america asia half_remaining : ℕ) 
  (h1 : europe = 20)
  (h2 : south_america = 10)
  (h3 : asia = 6)
  (h4 : asia = half_remaining / 2) : 
  europe + south_america + half_remaining = 42 :=
by
  sorry

end cornelia_travel_countries_l157_157950


namespace value_of_a_l157_157374

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l157_157374


namespace uncovered_area_l157_157237

def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side : ℕ := 4

theorem uncovered_area (height width side : ℕ) (h : height = shoebox_height) (w : width = shoebox_width) (s : side = block_side) :
  (width * height) - (side * side) = 8 :=
by
  rw [h, w, s]
  -- Area of shoebox bottom = width * height
  -- Area of square block = side * side
  -- Uncovered area = (width * height) - (side * side)
  -- Therefore, (6 * 4) - (4 * 4) = 24 - 16 = 8
  sorry

end uncovered_area_l157_157237


namespace range_of_f_lt_f2_l157_157831

-- Definitions for the given conditions
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (S : Set ℝ) := ∀ ⦃a b : ℝ⦄, a ∈ S → b ∈ S → a < b → f a < f b

-- Lean 4 statement for the proof problem
theorem range_of_f_lt_f2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on f {x | x ≤ 0}) : 
  ∀ x : ℝ, f x < f 2 → x > 2 ∨ x < -2 :=
by
  sorry

end range_of_f_lt_f2_l157_157831


namespace cylinder_problem_l157_157961

theorem cylinder_problem (r h : ℝ) (h1 : π * r^2 * h = 2) (h2 : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 :=
sorry

end cylinder_problem_l157_157961


namespace six_times_number_eq_132_l157_157005

theorem six_times_number_eq_132 (x : ℕ) (h : x / 11 = 2) : 6 * x = 132 :=
sorry

end six_times_number_eq_132_l157_157005


namespace john_payment_l157_157839

def john_buys := 20
def dave_pays := 6
def cost_per_candy := 1.50

theorem john_payment : (john_buys - dave_pays) * cost_per_candy = 21 := by
  sorry

end john_payment_l157_157839


namespace value_range_a_l157_157090

theorem value_range_a (a : ℝ) :
  (∀ (x : ℝ), |x + 2| * |x - 3| ≥ 4 / (a - 1)) ↔ (a < 1 ∨ a = 3) :=
by
  sorry

end value_range_a_l157_157090


namespace mark_current_trees_l157_157539

theorem mark_current_trees (x : ℕ) (h : x + 12 = 25) : x = 13 :=
by {
  -- proof omitted
  sorry
}

end mark_current_trees_l157_157539


namespace AC_amount_l157_157773

variable (A B C : ℝ)

theorem AC_amount
  (h1 : A + B + C = 400)
  (h2 : B + C = 150)
  (h3 : C = 50) :
  A + C = 300 := by
  sorry

end AC_amount_l157_157773


namespace smallest_n_l157_157700

theorem smallest_n (n : ℕ) (h1 : n ≡ 1 [MOD 3]) (h2 : n ≡ 4 [MOD 5]) (h3 : n > 20) : n = 34 := 
sorry

end smallest_n_l157_157700


namespace vampires_after_two_nights_l157_157094

-- Define the initial conditions and calculations
def initial_vampires : ℕ := 2
def transformation_rate : ℕ := 5
def first_night_vampires : ℕ := initial_vampires * transformation_rate + initial_vampires
def second_night_vampires : ℕ := first_night_vampires * transformation_rate + first_night_vampires

-- Prove that the number of vampires after two nights is 72
theorem vampires_after_two_nights : second_night_vampires = 72 :=
by sorry

end vampires_after_two_nights_l157_157094


namespace maximize_h_at_1_l157_157562

-- Definitions and conditions
def f (x : ℝ) : ℝ := -2 * x + 2
def g (x : ℝ) : ℝ := -3 * x + 6
def h (x : ℝ) : ℝ := f x * g x

-- The theorem to prove
theorem maximize_h_at_1 : (∀ x : ℝ, h x <= h 1) :=
sorry

end maximize_h_at_1_l157_157562


namespace find_a_value_l157_157367

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l157_157367


namespace right_triangle_side_length_l157_157518

theorem right_triangle_side_length
  (c : ℕ) (a : ℕ) (h_c : c = 13) (h_a : a = 12) :
  ∃ b : ℕ, b = 5 ∧ c^2 = a^2 + b^2 :=
by
  -- Definitions from conditions
  have h_c_square : c^2 = 169 := by rw [h_c]; norm_num
  have h_a_square : a^2 = 144 := by rw [h_a]; norm_num
  -- Prove the final result
  sorry

end right_triangle_side_length_l157_157518


namespace min_value_of_f_l157_157243

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 16) + Real.sqrt ((x + 1)^2 + 9))

theorem min_value_of_f :
  ∃ (x : ℝ), f x = 5 * Real.sqrt 2 := sorry

end min_value_of_f_l157_157243


namespace quadrilateral_side_length_l157_157454

-- Definitions
def inscribed_quadrilateral (a b c d r : ℝ) : Prop :=
  ∃ (O : ℝ) (A B C D : ℝ), 
    O = r ∧ 
    A = a ∧ B = b ∧ C = c ∧ 
    (r^2 + r^2 = (a^2 + b^2) / 2) ∧
    (r^2 + r^2 = (b^2 + c^2) / 2) ∧
    (r^2 + r^2 = (c^2 + d^2) / 2)

-- Theorem statement
theorem quadrilateral_side_length :
  inscribed_quadrilateral 250 250 100 200 250 :=
sorry

end quadrilateral_side_length_l157_157454


namespace hyperbola_condition_l157_157669

noncomputable def a_b_sum (a b : ℝ) : ℝ :=
  a + b

theorem hyperbola_condition
  (a b : ℝ)
  (h1 : a^2 - b^2 = 1)
  (h2 : abs (a - b) = 2)
  (h3 : a > b) :
  a_b_sum a b = 1/2 :=
sorry

end hyperbola_condition_l157_157669


namespace andy_time_difference_l157_157463

def time_dawn : ℕ := 20
def time_andy : ℕ := 46
def double_time_dawn : ℕ := 2 * time_dawn

theorem andy_time_difference :
  time_andy - double_time_dawn = 6 := by
  sorry

end andy_time_difference_l157_157463


namespace taxi_ride_cost_l157_157457

theorem taxi_ride_cost (base_fare : ℝ) (rate_per_mile : ℝ) (additional_charge : ℝ) (distance : ℕ) (cost : ℝ) :
  base_fare = 2 ∧ rate_per_mile = 0.30 ∧ additional_charge = 5 ∧ distance = 12 ∧ 
  cost = base_fare + (rate_per_mile * distance) + additional_charge → cost = 10.60 :=
by
  intros
  sorry

end taxi_ride_cost_l157_157457


namespace function_properties_l157_157394

noncomputable def f (x : ℝ) : ℝ := Real.sin (x * Real.cos x)

theorem function_properties :
  (f x = -f (-x)) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → 0 < f x) ∧
  ¬(∃ T, ∀ x, f (x + T) = f x) ∧
  (∀ n : ℤ, f (n * Real.pi) = 0) := 
by
  sorry

end function_properties_l157_157394


namespace increasing_interval_l157_157495

noncomputable def f (x : ℝ) := Real.log x / Real.log (1 / 2)

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def h (x : ℝ) : ℝ := x^2 + x - 2

theorem increasing_interval :
  is_monotonically_increasing (f ∘ h) {x : ℝ | x < -2} :=
sorry

end increasing_interval_l157_157495


namespace subset_eq_possible_sets_of_B_l157_157825

theorem subset_eq_possible_sets_of_B (B : Set ℕ) 
  (h1 : {1, 2} ⊆ B)
  (h2 : B ⊆ {1, 2, 3, 4}) :
  B = {1, 2} ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end subset_eq_possible_sets_of_B_l157_157825


namespace num_people_in_group_l157_157619

-- Given conditions as definitions
def cost_per_adult_meal : ℤ := 3
def num_kids : ℤ := 7
def total_cost : ℤ := 15

-- Statement to prove
theorem num_people_in_group : 
  ∃ (num_adults : ℤ), 
    total_cost = num_adults * cost_per_adult_meal ∧ 
    (num_adults + num_kids) = 12 :=
by
  sorry

end num_people_in_group_l157_157619


namespace total_eyes_insects_l157_157697

-- Defining the conditions given in the problem
def numSpiders : Nat := 3
def numAnts : Nat := 50
def eyesPerSpider : Nat := 8
def eyesPerAnt : Nat := 2

-- Statement to prove: the total number of eyes among Nina's pet insects is 124
theorem total_eyes_insects : (numSpiders * eyesPerSpider + numAnts * eyesPerAnt) = 124 := by
  sorry

end total_eyes_insects_l157_157697


namespace construction_costs_l157_157940

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end construction_costs_l157_157940


namespace four_digit_number_2010_l157_157591

theorem four_digit_number_2010 (a b c d : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
        1000 * a + 100 * b + 10 * c + d < 10000)
  (h_eq : a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2 * b^6 + 3 * c^6 + 4 * d^6)
          = 1000 * a + 100 * b + 10 * c + d)
  : 1000 * a + 100 * b + 10 * c + d = 2010 :=
sorry

end four_digit_number_2010_l157_157591


namespace barbeck_steve_guitar_ratio_l157_157466

theorem barbeck_steve_guitar_ratio (b s d : ℕ) 
  (h1 : b = s) 
  (h2 : d = 3 * b) 
  (h3 : b + s + d = 27) 
  (h4 : d = 18) : 
  b / s = 2 / 1 := 
by 
  sorry

end barbeck_steve_guitar_ratio_l157_157466


namespace required_large_loans_l157_157903

-- We start by introducing the concepts of the number of small, medium, and large loans
def small_loans : Type := ℕ
def medium_loans : Type := ℕ
def large_loans : Type := ℕ

-- Definition of the conditions as two scenarios
def Scenario1 (m s b : ℕ) : Prop := (m = 9 ∧ s = 6 ∧ b = 1)
def Scenario2 (m s b : ℕ) : Prop := (m = 3 ∧ s = 2 ∧ b = 3)

-- Definition of the problem
theorem required_large_loans (m s b : ℕ) (H1 : Scenario1 m s b) (H2 : Scenario2 m s b) :
  b = 4 :=
sorry

end required_large_loans_l157_157903


namespace num_ordered_pairs_eq_seven_l157_157783

theorem num_ordered_pairs_eq_seven : ∃ n, n = 7 ∧ ∀ (x y : ℕ), (x * y = 64) → (x > 0 ∧ y > 0) → n = 7 :=
by
  sorry

end num_ordered_pairs_eq_seven_l157_157783


namespace q_true_given_not_p_and_p_or_q_l157_157379

theorem q_true_given_not_p_and_p_or_q (p q : Prop) (hnp : ¬p) (hpq : p ∨ q) : q :=
by
  sorry

end q_true_given_not_p_and_p_or_q_l157_157379


namespace ratio_of_Patrick_to_Joseph_l157_157270

def countries_traveled_by_George : Nat := 6
def countries_traveled_by_Joseph : Nat := countries_traveled_by_George / 2
def countries_traveled_by_Zack : Nat := 18
def countries_traveled_by_Patrick : Nat := countries_traveled_by_Zack / 2

theorem ratio_of_Patrick_to_Joseph : countries_traveled_by_Patrick / countries_traveled_by_Joseph = 3 :=
by
  -- The definition conditions have already been integrated above
  sorry

end ratio_of_Patrick_to_Joseph_l157_157270


namespace nathaniel_wins_probability_is_5_over_11_l157_157846

open ProbabilityTheory

noncomputable def nathaniel_wins_probability : ℝ :=
  if ∃ n : ℕ, (∑ k in finset.range (n + 1), k % 7) = 0 then
    5 / 11
  else
    sorry

theorem nathaniel_wins_probability_is_5_over_11 :
  nathaniel_wins_probability = 5 / 11 :=
sorry

end nathaniel_wins_probability_is_5_over_11_l157_157846


namespace find_x_equals_4_l157_157583

noncomputable def repeatingExpr (x : ℝ) : ℝ :=
2 + 4 / (1 + 4 / (2 + 4 / (1 + 4 / x)))

theorem find_x_equals_4 :
  ∃ x : ℝ, x = repeatingExpr x ∧ x = 4 :=
by
  use 4
  sorry

end find_x_equals_4_l157_157583


namespace area_of_segment_l157_157132

theorem area_of_segment (R : ℝ) (hR : R > 0) (h_perimeter : 4 * R = 2 * R + 2 * R) :
  (1 - (1 / 2) * Real.sin 2) * R^2 = (fun R => (1 - (1 / 2) * Real.sin 2) * R^2) R :=
by
  sorry

end area_of_segment_l157_157132


namespace find_positive_square_root_l157_157983

theorem find_positive_square_root (x : ℝ) (h_pos : x > 0) (h_eq : x^2 = 625) : x = 25 :=
sorry

end find_positive_square_root_l157_157983


namespace find_largest_integer_solution_l157_157638

theorem find_largest_integer_solution:
  ∃ x: ℤ, (1/4 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < (7/9 : ℝ) ∧ (x = 4) := by
  sorry

end find_largest_integer_solution_l157_157638


namespace exam_total_students_l157_157524
-- Import the necessary Lean libraries

-- Define the problem conditions and the proof goal
theorem exam_total_students (T : ℕ) (h1 : 27 * T / 100 ≤ T) (h2 : 54 * T / 100 ≤ T) (h3 : 57 = 19 * T / 100) :
  T = 300 :=
  sorry  -- Proof is omitted here.

end exam_total_students_l157_157524


namespace find_missing_number_l157_157639

theorem find_missing_number (x : ℤ) (h : 10010 - 12 * x * 2 = 9938) : x = 3 :=
by
  sorry

end find_missing_number_l157_157639


namespace negation_of_exists_implies_forall_l157_157246

theorem negation_of_exists_implies_forall :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry

end negation_of_exists_implies_forall_l157_157246


namespace find_x_eq_3_plus_sqrt7_l157_157390

variable (x y : ℝ)
variable (h1 : x > y)
variable (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40)
variable (h3 : x * y + x + y = 8)

theorem find_x_eq_3_plus_sqrt7 (h1 : x > y) (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40) (h3 : x * y + x + y = 8) : 
  x = 3 + Real.sqrt 7 :=
sorry

end find_x_eq_3_plus_sqrt7_l157_157390


namespace average_test_score_l157_157516

theorem average_test_score (x : ℝ) :
  (0.45 * 95 + 0.50 * x + 0.05 * 60 = 84.75) → x = 78 :=
by
  sorry

end average_test_score_l157_157516


namespace surface_area_of_sphere_containing_prism_l157_157183

-- Assume the necessary geometric context and definitions are available.
def rightSquarePrism (a h : ℝ) (V : ℝ) := 
  a^2 * h = V

theorem surface_area_of_sphere_containing_prism 
  (a h V : ℝ) (S : ℝ) (π := Real.pi)
  (prism_on_sphere : ∀ (prism : rightSquarePrism a h V), True)
  (height_eq_4 : h = 4) 
  (volume_eq_16 : V = 16) :
  S = 4 * π * 24 :=
by
  -- proof steps would go here
  sorry

end surface_area_of_sphere_containing_prism_l157_157183


namespace find_solutions_l157_157216

-- Define the conditions
variable (n : ℕ)
noncomputable def valid_solution (a b c d : ℕ) : Prop := 
  a^2 + b^2 + c^2 + d^2 = 7 * 4^n

-- Define each possible solution
def sol1 : ℕ × ℕ × ℕ × ℕ := (5 * 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1))
def sol2 : ℕ × ℕ × ℕ × ℕ := (2 ^ (n + 1), 2 ^ n, 2 ^ n, 2 ^ n)
def sol3 : ℕ × ℕ × ℕ × ℕ := (3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 2 ^ (n - 1))

-- State the theorem
theorem find_solutions (a b c d : ℕ) (n : ℕ) :
  valid_solution n a b c d →
  (a, b, c, d) = sol1 n ∨
  (a, b, c, d) = sol2 n ∨
  (a, b, c, d) = sol3 n :=
sorry

end find_solutions_l157_157216


namespace total_dolls_48_l157_157508

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l157_157508


namespace solve_system_eq_l157_157872

theorem solve_system_eq (x y z : ℝ) :
    (x^2 - y^2 + z = 64 / (x * y)) ∧
    (y^2 - z^2 + x = 64 / (y * z)) ∧
    (z^2 - x^2 + y = 64 / (x * z)) ↔ 
    (x = 4 ∧ y = 4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = -4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = 4 ∧ z = -4) ∨ 
    (x = 4 ∧ y = -4 ∧ z = -4) := by
  sorry

end solve_system_eq_l157_157872


namespace ninety_eight_times_ninety_eight_l157_157156

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 :=
by
  sorry

end ninety_eight_times_ninety_eight_l157_157156


namespace merchant_profit_percentage_l157_157926

-- Given
def initial_cost_price : ℝ := 100
def marked_price : ℝ := initial_cost_price + 0.50 * initial_cost_price
def discount_percentage : ℝ := 0.20
def discount : ℝ := discount_percentage * marked_price
def selling_price : ℝ := marked_price - discount

-- Prove
theorem merchant_profit_percentage :
  ((selling_price - initial_cost_price) / initial_cost_price) * 100 = 20 :=
by
  sorry

end merchant_profit_percentage_l157_157926


namespace least_candies_to_remove_for_equal_distribution_l157_157627

theorem least_candies_to_remove_for_equal_distribution :
  ∃ k : ℕ, k = 4 ∧ ∀ n : ℕ, 24 - k = 5 * n :=
sorry

end least_candies_to_remove_for_equal_distribution_l157_157627


namespace find_a_value_l157_157372

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l157_157372


namespace patio_length_four_times_width_l157_157791

theorem patio_length_four_times_width (w l : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 :=
by
  sorry

end patio_length_four_times_width_l157_157791


namespace subcommittee_ways_l157_157756

theorem subcommittee_ways : 
  let R := 10 in
  let D := 8 in
  let kR := 4 in
  let kD := 3 in
  (Nat.choose R kR) * (Nat.choose D kD) = 11760 :=
by
  sorry

end subcommittee_ways_l157_157756


namespace range_of_a_l157_157672

def f (a x : ℝ) : ℝ := a * x^2 - 2 * x - |x^2 - a * x + 1|

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, x ≠ x₁ ∧ x ≠ x₂ → f a x ≠ 0

theorem range_of_a :
  { a : ℝ | has_exactly_two_zeros a } =
  { a : ℝ | (a < 0) ∨ (0 < a ∧ a < 1) ∨ (1 < a) } :=
sorry

end range_of_a_l157_157672


namespace rational_number_25_units_away_l157_157086

theorem rational_number_25_units_away (x : ℚ) (h : |x| = 2.5) : x = 2.5 ∨ x = -2.5 := 
by
  sorry

end rational_number_25_units_away_l157_157086


namespace average_of_remaining_numbers_l157_157418

theorem average_of_remaining_numbers (sum : ℕ) (average : ℕ) (remaining_sum : ℕ) (remaining_average : ℚ) :
  (average = 90) →
  (sum = 1080) →
  (remaining_sum = sum - 72 - 84) →
  (remaining_average = remaining_sum / 10) →
  remaining_average = 92.4 :=
by
  sorry

end average_of_remaining_numbers_l157_157418


namespace tetrahedron_volume_l157_157050

noncomputable def volume_of_tetrahedron (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABC_ABD : ℝ) : ℝ :=
  (1/3) * area_ABC * area_ABD * (Real.sin angle_ABC_ABD) * (AB / (Real.sqrt 2))

theorem tetrahedron_volume :
  let AB := 5 -- edge AB length in cm
  let area_ABC := 18 -- area of face ABC in cm^2
  let area_ABD := 24 -- area of face ABD in cm^2
  let angle_ABC_ABD := Real.pi / 4 -- 45 degrees in radians
  volume_of_tetrahedron AB area_ABC area_ABD angle_ABC_ABD = 43.2 :=
by
  sorry

end tetrahedron_volume_l157_157050


namespace simplify_polynomial_l157_157867

def P (x : ℝ) : ℝ := 3*x^3 + 4*x^2 - 5*x + 8
def Q (x : ℝ) : ℝ := 2*x^3 + x^2 + 3*x - 15

theorem simplify_polynomial (x : ℝ) : P x - Q x = x^3 + 3*x^2 - 8*x + 23 := 
by 
  -- proof goes here
  sorry

end simplify_polynomial_l157_157867


namespace line_through_points_on_parabola_l157_157811

theorem line_through_points_on_parabola 
  (x1 y1 x2 y2 : ℝ)
  (h_parabola_A : y1^2 = 4 * x1)
  (h_parabola_B : y2^2 = 4 * x2)
  (h_midpoint : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  ∃ (m b : ℝ), m = 1 ∧ b = 2 ∧ (∀ x y : ℝ, y = m * x + b ↔ x - y = 0) :=
sorry

end line_through_points_on_parabola_l157_157811


namespace remaining_budget_for_public_spaces_l157_157728

noncomputable def total_budget : ℝ := 32
noncomputable def policing_budget : ℝ := total_budget / 2
noncomputable def education_budget : ℝ := 12
noncomputable def remaining_budget : ℝ := total_budget - (policing_budget + education_budget)

theorem remaining_budget_for_public_spaces : remaining_budget = 4 :=
by
  -- Proof is skipped
  sorry

end remaining_budget_for_public_spaces_l157_157728


namespace proposition_B_proposition_D_l157_157485

open Real

variable (a b : ℝ)

theorem proposition_B (h : a^2 ≠ b^2) : a ≠ b := 
sorry

theorem proposition_D (h : a > abs b) : a^2 > b^2 :=
sorry

end proposition_B_proposition_D_l157_157485


namespace solve_equation_l157_157709

theorem solve_equation (x y : ℝ) : 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end solve_equation_l157_157709


namespace paint_walls_l157_157665

theorem paint_walls (d h e : ℕ) : 
  ∃ (x : ℕ), (d * d * e = 2 * h * h * x) ↔ x = (d^2 * e) / (2 * h^2) := by
  sorry

end paint_walls_l157_157665


namespace eval_inverse_l157_157711

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h₁ : g 4 = 6)
variable (h₂ : g 7 = 2)
variable (h₃ : g 3 = 7)
variable (h_inv₁ : g_inv 6 = 4)
variable (h_inv₂ : g_inv 7 = 3)

theorem eval_inverse (g : ℕ → ℕ)
(g_inv : ℕ → ℕ)
(h₁ : g 4 = 6)
(h₂ : g 7 = 2)
(h₃ : g 3 = 7)
(h_inv₁ : g_inv 6 = 4)
(h_inv₂ : g_inv 7 = 3) :
g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end eval_inverse_l157_157711


namespace proportion_solution_l157_157273

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := by
  sorry

end proportion_solution_l157_157273


namespace incircle_excircle_relation_l157_157199

variables {α : Type*} [LinearOrderedField α]

-- Defining the area expressions and radii
def area_inradius (a b c r : α) : α := (a + b + c) * r / 2
def area_exradius1 (a b c r1 : α) : α := (b + c - a) * r1 / 2
def area_exradius2 (a b c r2 : α) : α := (a + c - b) * r2 / 2
def area_exradius3 (a b c r3 : α) : α := (a + b - c) * r3 / 2

theorem incircle_excircle_relation (a b c r r1 r2 r3 Q : α) 
  (h₁ : Q = area_inradius a b c r)
  (h₂ : Q = area_exradius1 a b c r1)
  (h₃ : Q = area_exradius2 a b c r2)
  (h₄ : Q = area_exradius3 a b c r3) :
  1 / r = 1 / r1 + 1 / r2 + 1 / r3 :=
by 
  sorry

end incircle_excircle_relation_l157_157199


namespace first_route_red_lights_longer_l157_157128

-- Conditions
def first_route_base_time : ℕ := 10
def red_light_time : ℕ := 3
def num_stoplights : ℕ := 3
def second_route_time : ℕ := 14

-- Question to Answer
theorem first_route_red_lights_longer : (first_route_base_time + num_stoplights * red_light_time - second_route_time) = 5 := by
  sorry

end first_route_red_lights_longer_l157_157128


namespace quadratic_inequality_solution_l157_157254

theorem quadratic_inequality_solution
  (x : ℝ) 
  (h1 : ∀ x, x^2 + 2 * x - 3 > 0 ↔ x < -3 ∨ x > 1) :
  (2 * x^2 - 3 * x - 2 < 0) ↔ (-1 / 2 < x ∧ x < 2) :=
by {
  sorry
}

end quadratic_inequality_solution_l157_157254


namespace find_number_l157_157828

theorem find_number :
  let f_add (a b : ℝ) : ℝ := a * b
  let f_sub (a b : ℝ) : ℝ := a + b
  let f_mul (a b : ℝ) : ℝ := a / b
  let f_div (a b : ℝ) : ℝ := a - b
  (f_div 9 8) * (f_mul 7 some_number) + (f_sub some_number 10) = 13.285714285714286 :=
  let some_number := 5
  sorry

end find_number_l157_157828


namespace solution_set_l157_157894

-- Define the two conditions as hypotheses
variables (x : ℝ)

def condition1 : Prop := x + 6 ≤ 8
def condition2 : Prop := x - 7 < 2 * (x - 3)

-- The statement to prove
theorem solution_set (h1 : condition1 x) (h2 : condition2 x) : -1 < x ∧ x ≤ 2 :=
by
  sorry

end solution_set_l157_157894


namespace crow_eating_time_l157_157924

theorem crow_eating_time (n : ℕ) (h : ∀ t : ℕ, t = (n / 5) → t = 4) : (4 + (4 / 5) = 4.8) :=
by
  sorry

end crow_eating_time_l157_157924


namespace choir_members_correct_l157_157245

noncomputable def choir_membership : ℕ :=
  let n := 226
  n

theorem choir_members_correct (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
by
  sorry

end choir_members_correct_l157_157245


namespace find_m_l157_157451

theorem find_m (m : ℕ) (h₁ : 0 < m) : 
  144^5 + 91^5 + 56^5 + 19^5 = m^5 → m = 147 := by
  -- Mathematically, we know the sum of powers equals a fifth power of 147
  -- 144^5 = 61917364224
  -- 91^5 = 6240321451
  -- 56^5 = 550731776
  -- 19^5 = 2476099
  -- => 61917364224 + 6240321451 + 550731776 + 2476099 = 68897423550
  -- Find the nearest  m such that m^5 = 68897423550
  sorry

end find_m_l157_157451


namespace mira_jogging_distance_l157_157409

def jogging_speed : ℝ := 5 -- speed in miles per hour
def jogging_hours_per_day : ℝ := 2 -- hours per day
def days_count : ℕ := 5 -- number of days

theorem mira_jogging_distance :
  (jogging_speed * jogging_hours_per_day * days_count : ℝ) = 50 :=
by
  sorry

end mira_jogging_distance_l157_157409


namespace greatest_number_of_rented_trucks_l157_157274

-- Define the conditions
def total_trucks_on_monday : ℕ := 24
def trucks_returned_percentage : ℕ := 50
def trucks_on_lot_saturday (R : ℕ) (P : ℕ) : ℕ := (R * P) / 100
def min_trucks_on_lot_saturday : ℕ := 12

-- Define the theorem
theorem greatest_number_of_rented_trucks : ∃ R, R = total_trucks_on_monday ∧ trucks_returned_percentage = 50 ∧ min_trucks_on_lot_saturday = 12 → R = 24 :=
by
  sorry

end greatest_number_of_rented_trucks_l157_157274


namespace max_balls_l157_157855

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l157_157855


namespace favorite_number_l157_157074

theorem favorite_number (S₁ S₂ S₃ : ℕ) (total_sum : ℕ) (adjacent_sum : ℕ) 
  (h₁ : S₁ = 8) (h₂ : S₂ = 14) (h₃ : S₃ = 12) 
  (h_total_sum : total_sum = 17) 
  (h_adjacent_sum : adjacent_sum = 12) : 
  ∃ x : ℕ, x = 5 := 
by 
  sorry

end favorite_number_l157_157074


namespace black_ants_employed_l157_157261

theorem black_ants_employed (total_ants : ℕ) (red_ants : ℕ) 
  (h1 : total_ants = 900) (h2 : red_ants = 413) :
    total_ants - red_ants = 487 :=
by
  -- The proof is given below.
  sorry

end black_ants_employed_l157_157261


namespace compare_negative_fractions_l157_157321

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l157_157321


namespace first_valve_fill_time_l157_157103

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end first_valve_fill_time_l157_157103


namespace intersection_ac_bf_mz_l157_157221

/-- Lean 4 statement for the proof problem -/
theorem intersection_ac_bf_mz 
  (A B C M D E F Z X Y : MyPoint)
  (h_triangle_ABC : Triangle A B C)
  (h_M_mid : Midpoint M B C)
  (h_D_on_AB : D ∈ LineSegment A B)
  (h_B_between_A_D : B ∈ LineSegment A D)
  (h_EDC_ACB : ∠ E D C = ∠ A C B)
  (h_DCE_BAC : ∠ D C E = ∠ B A C)
  (h_F_inter : Inter (LineThrough E C) (ParallelLineThrough D E A) F)
  (h_Z_inter : Inter (LineThrough A E) (LineThrough D F) Z)
  : Intersect (LineThrough A C) (LineThrough B F) (LineThrough M Z) :=
sorry

end intersection_ac_bf_mz_l157_157221


namespace right_triangle_exists_and_r_inscribed_circle_l157_157437

theorem right_triangle_exists_and_r_inscribed_circle (d : ℝ) (hd : d > 0) :
  ∃ (a b c : ℝ), 
    a < b ∧ 
    a^2 + b^2 = c^2 ∧
    b = a + d ∧ 
    c = b + d ∧ 
    (a + b - c) / 2 = d :=
by
  sorry

end right_triangle_exists_and_r_inscribed_circle_l157_157437


namespace dividend_calculation_l157_157095

theorem dividend_calculation (D : ℝ) (Q : ℝ) (R : ℝ) (Dividend : ℝ) (h1 : D = 47.5) (h2 : Q = 24.3) (h3 : R = 32.4)  :
  Dividend = D * Q + R := by
  rw [h1, h2, h3]
  sorry -- This skips the actual computation proof

end dividend_calculation_l157_157095


namespace compare_fractions_l157_157312

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l157_157312


namespace area_correct_l157_157885

noncomputable def area_of_30_60_90_triangle (hypotenuse : ℝ) (angle : ℝ) : ℝ :=
if hypotenuse = 10 ∧ angle = 30 then 25 * Real.sqrt 3 / 2 else 0

theorem area_correct {hypotenuse angle : ℝ} (h1 : hypotenuse = 10) (h2 : angle = 30) :
  area_of_30_60_90_triangle hypotenuse angle = 25 * Real.sqrt 3 / 2 :=
by
  sorry

end area_correct_l157_157885


namespace complete_the_square_l157_157718

theorem complete_the_square (z : ℤ) : 
    z^2 - 6*z + 17 = (z - 3)^2 + 8 :=
sorry

end complete_the_square_l157_157718


namespace ratio_of_lengths_l157_157601

theorem ratio_of_lengths (l1 l2 l3 : ℝ)
    (h1 : l2 = (1/2) * (l1 + l3))
    (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
    l1 / l3 = 7 / 5 := by
  sorry

end ratio_of_lengths_l157_157601


namespace lcm_of_15_18_20_is_180_l157_157643

theorem lcm_of_15_18_20_is_180 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end lcm_of_15_18_20_is_180_l157_157643


namespace rectangle_dimensions_l157_157568

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l157_157568


namespace exists_similarity_point_l157_157195

variable {Point : Type} [MetricSpace Point]

noncomputable def similar_triangles (A B A' B' : Point) (O : Point) : Prop :=
  dist A O / dist A' O = dist A B / dist A' B' ∧ dist B O / dist B' O = dist A B / dist A' B'

theorem exists_similarity_point (A B A' B' : Point) (h1 : dist A B ≠ 0) (h2: dist A' B' ≠ 0) :
  ∃ O : Point, similar_triangles A B A' B' O :=
  sorry

end exists_similarity_point_l157_157195


namespace gcd_of_78_and_104_l157_157737

theorem gcd_of_78_and_104 : Int.gcd 78 104 = 26 := by
  sorry

end gcd_of_78_and_104_l157_157737


namespace area_ratio_l157_157733

variables {A B C D: Type} [LinearOrderedField A]
variables {AB AD AR AE : A}

-- Conditions
axiom cond1 : AR = (2 / 3) * AB
axiom cond2 : AE = (1 / 3) * AD

theorem area_ratio (h : A) (h1 : A) (S_ABCD : A) (S_ARE : A)
  (h_eq : S_ABCD = AD * h)
  (h1_eq : S_ARE = (1 / 2) * AE * h1)
  (ratio_heights : h / h1 = 3 / 2) :
  S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_l157_157733


namespace proof_standard_deviation_l157_157715

noncomputable def standard_deviation (average_age : ℝ) (max_diff_ages : ℕ) : ℝ := sorry

theorem proof_standard_deviation :
  let average_age := 31
  let max_diff_ages := 19
  standard_deviation average_age max_diff_ages = 9 := 
by
  sorry

end proof_standard_deviation_l157_157715


namespace determine_n_l157_157630

theorem determine_n (n : ℕ) (h : 3^n = 27 * 81^3 / 9^4) : n = 7 := by
  sorry

end determine_n_l157_157630


namespace martians_cannot_hold_hands_l157_157636

-- Define the number of hands each Martian possesses
def hands_per_martian := 3

-- Define the number of Martians
def number_of_martians := 7

-- Define the total number of hands
def total_hands := hands_per_martian * number_of_martians

-- Prove that it is not possible for the seven Martians to hold hands with each other
theorem martians_cannot_hold_hands :
  ¬ ∃ (pairs : ℕ), 2 * pairs = total_hands :=
by
  sorry

end martians_cannot_hold_hands_l157_157636


namespace sqrt_fraction_addition_l157_157946

theorem sqrt_fraction_addition :
  (Real.sqrt ((25 : ℝ) / 36 + 16 / 9)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_addition_l157_157946


namespace sufficient_but_not_necessary_not_necessary_condition_l157_157483

theorem sufficient_but_not_necessary 
  (α : ℝ) (h : Real.sin α = Real.cos α) :
  Real.cos (2 * α) = 0 :=
by sorry

theorem not_necessary_condition 
  (α : ℝ) (h : Real.cos (2 * α) = 0) :
  ∃ β : ℝ, Real.sin β ≠ Real.cos β :=
by sorry

end sufficient_but_not_necessary_not_necessary_condition_l157_157483


namespace proof_b_lt_a_lt_c_l157_157487

noncomputable def a : ℝ := 2^(4/5)
noncomputable def b : ℝ := 4^(2/7)
noncomputable def c : ℝ := 25^(1/5)

theorem proof_b_lt_a_lt_c : b < a ∧ a < c := by
  sorry

end proof_b_lt_a_lt_c_l157_157487


namespace quadratic_roots_solution_l157_157233

theorem quadratic_roots_solution (x : ℝ) (h : x > 0) (h_roots : 7 * x^2 - 8 * x - 6 = 0) : (x = 6 / 7) ∨ (x = 1) :=
sorry

end quadratic_roots_solution_l157_157233


namespace books_read_last_month_l157_157434

namespace BookReading

variable (W : ℕ) -- Number of books William read last month.

-- Conditions
axiom cond1 : ∃ B : ℕ, B = 3 * W -- Brad read thrice as many books as William did last month.
axiom cond2 : W = 2 * 8 -- This month, William read twice as much as Brad, who read 8 books.
axiom cond3 : ∃ (B_prev : ℕ) (B_curr : ℕ), B_prev = 3 * W ∧ B_curr = 8 ∧ W + 16 = B_prev + B_curr + 4 -- Total books equation

theorem books_read_last_month : W = 2 := by
  sorry

end BookReading

end books_read_last_month_l157_157434


namespace divisibility_of_difference_by_9_l157_157113

theorem divisibility_of_difference_by_9 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) :
  9 ∣ ((10 * a + b) - (10 * b + a)) :=
by {
  -- The problem statement
  sorry
}

end divisibility_of_difference_by_9_l157_157113


namespace max_balls_drawn_l157_157860

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l157_157860


namespace nathaniel_wins_probability_l157_157847

/-- 
  Nathaniel and Obediah play a game where they take turns rolling a fair six-sided die 
  and keep a running tally. A player wins if the tally is a multiple of 7.
  If Nathaniel goes first, the probability that he wins is 5/11.
-/
theorem nathaniel_wins_probability :
  ∀ (die : ℕ → ℕ) (tally : ℕ → ℕ)
  (turn : ℕ → ℕ) (current_player : ℕ)
  (win_condition : ℕ → Prop),
  (∀ i, die i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ i, tally (i + 1) = tally i + die (i % 6)) →
  (win_condition n ↔ tally n % 7 = 0) →
  current_player 0 = 0 →  -- Nathaniel starts
  (turn i = if i % 2 = 0 then 0 else 1) →
  P(current_player wins) = 5/11 :=
by
  sorry

end nathaniel_wins_probability_l157_157847


namespace fraction_of_passengers_from_Africa_l157_157515

theorem fraction_of_passengers_from_Africa :
  (1/4 + 1/8 + 1/6 + A + 36/96 = 1) → (96 - 36) = (11/24 * 96) → 
  A = 1/12 :=
by
  sorry

end fraction_of_passengers_from_Africa_l157_157515


namespace car_mpg_in_city_l157_157932

theorem car_mpg_in_city 
    (miles_per_tank_highway : Real)
    (miles_per_tank_city : Real)
    (mpg_difference : Real)
    : True := by
  let H := 21.05
  let T := 720 / H
  let C := H - 10
  have h1 : 720 = H * T := by
    sorry
  have h2 : 378 = C * T := by
    sorry
  exact True.intro

end car_mpg_in_city_l157_157932


namespace vertical_asymptote_at_neg_two_over_three_l157_157958

theorem vertical_asymptote_at_neg_two_over_three : 
  ∃ x : ℝ, 6 * x + 4 = 0 ∧ x = -2 / 3 := 
by
  use -2 / 3
  sorry

end vertical_asymptote_at_neg_two_over_three_l157_157958


namespace smallest_digit_divisible_by_11_l157_157167

theorem smallest_digit_divisible_by_11 : ∃ d : ℕ, (0 ≤ d ∧ d ≤ 9) ∧ d = 6 ∧ (d + 7 - (4 + 3 + 6)) % 11 = 0 := by
  sorry

end smallest_digit_divisible_by_11_l157_157167


namespace rectangle_dimensions_l157_157569

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l157_157569


namespace standard_product_probability_l157_157256

noncomputable def num_standard_product_range (n : ℕ) (p : ℝ) : set ℕ :=
  {m | m ∈ set.Icc 793 827}

theorem standard_product_probability (n : ℕ) (p : ℝ) :
  n = 900 → p = 0.9 → 
  (P : real) → P = 0.95 →
  let X := binomial pmf.mk p n in
  P (num_standard_product_range n p) = 0.95 := sorry

end standard_product_probability_l157_157256


namespace solve_equation_l157_157869

theorem solve_equation (x : ℝ) : x * (x + 5)^3 * (5 - x) = 0 ↔ x = 0 ∨ x = -5 ∨ x = 5 := by
  sorry

end solve_equation_l157_157869


namespace five_aliens_have_more_limbs_than_five_martians_l157_157462

-- Definitions based on problem conditions

def number_of_alien_arms : ℕ := 3
def number_of_alien_legs : ℕ := 8

-- Martians have twice as many arms as Aliens and half as many legs
def number_of_martian_arms : ℕ := 2 * number_of_alien_arms
def number_of_martian_legs : ℕ := number_of_alien_legs / 2

-- Total limbs for five aliens and five martians
def total_limbs_for_aliens (n : ℕ) : ℕ := n * (number_of_alien_arms + number_of_alien_legs)
def total_limbs_for_martians (n : ℕ) : ℕ := n * (number_of_martian_arms + number_of_martian_legs)

-- The theorem to prove
theorem five_aliens_have_more_limbs_than_five_martians :
  total_limbs_for_aliens 5 - total_limbs_for_martians 5 = 5 :=
sorry

end five_aliens_have_more_limbs_than_five_martians_l157_157462


namespace horner_v3_value_correct_l157_157960

def f (x : ℕ) : ℕ :=
  x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℕ) : ℕ :=
  ((((x + 0) * x + 2) * x + 3) * x + 1) * x + 1

theorem horner_v3_value_correct :
  horner_eval 3 = 36 :=
sorry

end horner_v3_value_correct_l157_157960


namespace geometric_sequence_common_ratio_l157_157689

theorem geometric_sequence_common_ratio (a_1 q : ℝ) (hne1 : q ≠ 1)
  (h : (a_1 * (1 - q^4) / (1 - q)) = 5 * (a_1 * (1 - q^2) / (1 - q))) :
  q = -1 ∨ q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l157_157689


namespace average_after_15th_inning_l157_157282

theorem average_after_15th_inning (A : ℝ) 
    (h_avg_increase : (14 * A + 75) = 15 * (A + 3)) : 
    A + 3 = 33 :=
by {
  sorry
}

end average_after_15th_inning_l157_157282


namespace expression_always_integer_l157_157458

theorem expression_always_integer (m : ℕ) : 
  ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = (k : ℚ) := 
sorry

end expression_always_integer_l157_157458


namespace floor_add_frac_eq_154_l157_157010

theorem floor_add_frac_eq_154 (r : ℝ) (h : ⌊r⌋ + r = 15.4) : r = 7.4 := 
sorry

end floor_add_frac_eq_154_l157_157010


namespace determine_base_l157_157383

theorem determine_base (r : ℕ) (a b x : ℕ) (h₁ : r ≤ 100) 
  (h₂ : x = a * r + a) (h₃ : a < r) (h₄ : a > 0) 
  (h₅ : x^2 = b * r^3 + b) : r = 2 ∨ r = 23 :=
by
  sorry

end determine_base_l157_157383


namespace distance_on_dirt_road_l157_157441

theorem distance_on_dirt_road :
  ∀ (initial_gap distance_gap_on_city dirt_road_distance : ℝ),
  initial_gap = 2 → 
  distance_gap_on_city = initial_gap - ((initial_gap - (40 * (1 / 30)))) → 
  dirt_road_distance = distance_gap_on_city * (40 / 60) * (70 / 40) * (30 / 70) →
  dirt_road_distance = 1 :=
by
  intros initial_gap distance_gap_on_city dirt_road_distance h1 h2 h3
  -- The proof would go here
  sorry

end distance_on_dirt_road_l157_157441


namespace problem_1_problem_2_l157_157967

def f (a : ℝ) (x : ℝ) : ℝ := abs (a * x + 1)

def g (a : ℝ) (x : ℝ) : ℝ := f a x - abs (x + 1)

theorem problem_1 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 ↔ f a x ≤ 3) → a = 2 := by
  intro h
  sorry

theorem problem_2 (a : ℝ) : a = 2 → (∃ x : ℝ, ∀ y : ℝ, g a y ≥ g a x ∧ g a x = -1/2) := by
  intro ha2
  use -1/2
  sorry

end problem_1_problem_2_l157_157967


namespace units_sold_at_original_price_l157_157792

-- Define the necessary parameters and assumptions
variables (a x y : ℝ)
variables (total_units sold_original sold_discount sold_offseason : ℝ)
variables (purchase_price sell_price discount_price clearance_price : ℝ)

-- Define specific conditions
def purchase_units := total_units = 1000
def selling_price := sell_price = 1.25 * a
def discount_cond := discount_price = 1.25 * 0.9 * a
def clearance_cond := clearance_price = 1.25 * 0.60 * a
def holiday_limit := y ≤ 100
def profitability_condition := 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a

-- The theorem asserting at least 426 units sold at the original price ensures profitability
theorem units_sold_at_original_price (h1 : total_units = 1000)
  (h2 : sell_price = 1.25 * a) (h3 : discount_price = 1.25 * 0.9 * a)
  (h4 : clearance_price = 1.25 * 0.60 * a) (h5 : y ≤ 100)
  (h6 : 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a) :
  x ≥ 426 :=
by
  sorry

end units_sold_at_original_price_l157_157792


namespace remaining_fruit_count_l157_157633

theorem remaining_fruit_count (trees : ℕ) (fruits_per_tree : ℕ) (picked_fraction : ℚ) 
  (trees_eq : trees = 8) (fruits_per_tree_eq : fruits_per_tree = 200) (picked_fraction_eq : picked_fraction = 2/5) :
  let total_fruits := trees * fruits_per_tree
  let picked_fruits := picked_fraction * fruits_per_tree * trees
  let remaining_fruits := total_fruits - picked_fruits
  remaining_fruits = 960 := 
by 
  sorry

end remaining_fruit_count_l157_157633


namespace construction_costs_correct_l157_157938

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end construction_costs_correct_l157_157938


namespace grasshoppers_after_transformations_l157_157571

-- Define initial conditions and transformation rules
def initial_crickets : ℕ := 30
def initial_grasshoppers : ℕ := 30

-- Define the transformations
def red_haired_transforms (g : ℕ) (c : ℕ) : ℕ × ℕ :=
  (g - 4, c + 1)

def green_haired_transforms (c : ℕ) (g : ℕ) : ℕ × ℕ :=
  (c - 5, g + 2)

-- Define the total number of transformations and the resulting condition
def total_transformations : ℕ := 18
def final_crickets : ℕ := 0

-- The proof goal
theorem grasshoppers_after_transformations : 
  initial_grasshoppers = 30 → 
  initial_crickets = 30 → 
  (∀ t, t = total_transformations → 
          ∀ g c, 
          (g, c) = (0, 6) → 
          (∃ m n, (m + n = t ∧ final_crickets = c))) →
  final_grasshoppers = 6 :=
by
  sorry

end grasshoppers_after_transformations_l157_157571


namespace geometric_sequence_sum_l157_157391

noncomputable def sum_of_first_n_terms (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_a1 : a_1 = 1) (h_a5 : a_5 = 16) :
  sum_of_first_n_terms 1 q 7 = 127 :=
by
  sorry

end geometric_sequence_sum_l157_157391


namespace avg_one_fourth_class_l157_157423

variable (N : ℕ) -- Total number of students

-- Define the average grade for the entire class
def avg_entire_class : ℝ := 84

-- Define the average grade of three fourths of the class
def avg_three_fourths_class : ℝ := 80

-- Statement to prove
theorem avg_one_fourth_class (A : ℝ) (h1 : 1/4 * A + 3/4 * avg_three_fourths_class = avg_entire_class) : 
  A = 96 := 
sorry

end avg_one_fourth_class_l157_157423


namespace geometric_sequence_a7_l157_157992

-- Define the geometric sequence
def geometic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Conditions
def a1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2a4 (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 16

-- The statement to prove
theorem geometric_sequence_a7 (a : ℕ → ℝ) (h1 : a1 a) (h2 : a2a4 a) (gs : geometic_sequence a) :
  a 7 = 64 :=
by
  sorry

end geometric_sequence_a7_l157_157992


namespace toys_produced_in_week_l157_157933

-- Define the number of working days in a week
def working_days_in_week : ℕ := 4

-- Define the number of toys produced per day
def toys_produced_per_day : ℕ := 1375

-- The statement to be proved
theorem toys_produced_in_week :
  working_days_in_week * toys_produced_per_day = 5500 :=
by
  sorry

end toys_produced_in_week_l157_157933


namespace ellipse_condition_l157_157355

theorem ellipse_condition (k : ℝ) :
  (4 < k ∧ k < 9) ↔ (9 - k > 0 ∧ k - 4 > 0 ∧ 9 - k ≠ k - 4) :=
by sorry

end ellipse_condition_l157_157355


namespace total_doll_count_l157_157507

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l157_157507


namespace arithmetic_seq_of_equal_roots_l157_157024

theorem arithmetic_seq_of_equal_roots (a b c : ℝ) (h : b ≠ 0) 
    (h_eq_roots : ∃ x, b*x^2 - 4*b*x + 2*(a + c) = 0 ∧ (∀ y, b*y^2 - 4*b*y + 2*(a + c) = 0 → x = y)) : 
    b - a = c - b := 
by 
  -- placeholder for proof body
  sorry

end arithmetic_seq_of_equal_roots_l157_157024


namespace tiles_needed_l157_157121

theorem tiles_needed (A_classroom : ℝ) (side_length_tile : ℝ) (H_classroom : A_classroom = 56) (H_side_length : side_length_tile = 0.4) :
  A_classroom / (side_length_tile * side_length_tile) = 350 :=
by
  sorry

end tiles_needed_l157_157121


namespace find_a_value_l157_157373

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l157_157373


namespace g_of_minus_1_eq_9_l157_157060

-- defining f(x) and g(f(x)), and stating the objective to prove g(-1)=9
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 5

theorem g_of_minus_1_eq_9 : g (-1) = 9 :=
  sorry

end g_of_minus_1_eq_9_l157_157060


namespace max_distance_covered_l157_157588

theorem max_distance_covered 
  (D : ℝ)
  (h1 : (D / 2) / 5 + (D / 2) / 4 = 6) : 
  D = 40 / 3 :=
by
  sorry

end max_distance_covered_l157_157588


namespace family_ages_l157_157382

-- Define the conditions
variables (D M S F : ℕ)

-- Condition 1: In the year 2000, the mother was 4 times the daughter's age.
axiom mother_age : M = 4 * D

-- Condition 2: In the year 2000, the father was 6 times the son's age.
axiom father_age : F = 6 * S

-- Condition 3: The son is 1.5 times the age of the daughter.
axiom son_age_ratio : S = 3 * D / 2

-- Condition 4: In the year 2010, the father became twice the mother's age.
axiom father_mother_2010 : F + 10 = 2 * (M + 10)

-- Condition 5: The age gap between the mother and father has always been the same.
axiom age_gap_constant : F - M = (F + 10) - (M + 10)

-- Define the theorem
theorem family_ages :
  D = 10 ∧ S = 15 ∧ M = 40 ∧ F = 90 ∧ (F - M = 50) := sorry

end family_ages_l157_157382


namespace set_equality_l157_157028

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 4})
variable (hB : B = {3, 4})

theorem set_equality : ({2, 5} : Set ℕ) = U \ (A ∪ B) :=
by
  sorry

end set_equality_l157_157028


namespace box_volume_in_cubic_yards_l157_157770

theorem box_volume_in_cubic_yards (v_feet : ℕ) (conv_factor : ℕ) (v_yards : ℕ)
  (h1 : v_feet = 216) (h2 : conv_factor = 3) (h3 : 27 = conv_factor ^ 3) : 
  v_yards = 8 :=
by
  sorry

end box_volume_in_cubic_yards_l157_157770


namespace simplify_expression_l157_157009

theorem simplify_expression :
  let a := 7
  let b := 11
  let c := 19
  (49 * (1 / 11 - 1 / 19) + 121 * (1 / 19 - 1 / 7) + 361 * (1 / 7 - 1 / 11)) /
  (7 * (1 / 11 - 1 / 19) + 11 * (1 / 19 - 1 / 7) + 19 * (1 / 7 - 1 / 11)) = 37 := by
  sorry

end simplify_expression_l157_157009


namespace points_per_member_l157_157942

theorem points_per_member
    (total_members : ℕ)
    (absent_members : ℕ)
    (total_points : ℕ)
    (present_members : ℕ)
    (points_per_member : ℕ)
    (h1 : total_members = 5)
    (h2 : absent_members = 2)
    (h3 : total_points = 18)
    (h4 : present_members = total_members - absent_members)
    (h5 : points_per_member = total_points / present_members) :
  points_per_member = 6 :=
by
  sorry

end points_per_member_l157_157942


namespace combined_spots_l157_157978

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end combined_spots_l157_157978


namespace Matt_jumped_for_10_minutes_l157_157694

def Matt_skips_per_second : ℕ := 3

def total_skips : ℕ := 1800

def minutes_jumped (m : ℕ) : Prop :=
  m * (Matt_skips_per_second * 60) = total_skips

theorem Matt_jumped_for_10_minutes : minutes_jumped 10 :=
by
  sorry

end Matt_jumped_for_10_minutes_l157_157694


namespace probability_at_least_one_needs_device_l157_157184

theorem probability_at_least_one_needs_device :
  let pA := 0.4 in
  let pB := 0.5 in
  let pC := 0.7 in
  (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.91 := by
    sorry

end probability_at_least_one_needs_device_l157_157184


namespace at_least_half_team_B_can_serve_l157_157911

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l157_157911


namespace tangent_circle_OM_perp_KL_l157_157120

open EuclideanGeometry

variables {A B C M O P Q K L : Point}

-- Definitions based on conditions
def Circle_Tangent (A B C O : Point) : Prop := 
Circle O B ∧ Circle O C ∧ tangent A B ∧ tangent A C

def Angle_Vertex (A O : Point) (B C : Point) : Prop :=
Angle A O B = Angle A O C ∧ B ≠ C

def Points_On_Arc (B C M : Point) (O : Point) : Prop :=
M ∈ arc B C ∧ M ≠ B ∧ M ≠ C ∧ ¬ collinear A O M

def Line_Intersections (B M C O A P Q : Point) : Prop :=
intersection (line B M) (line A O) = {P} ∧ intersection (line C M) (line A O) = {Q}

def Perpendicular_Foot (P K AC : Point) (Q L AB : Point) : Prop :=
foot P AC = K ∧ foot Q AB = L

-- The theorem we want to prove
theorem tangent_circle_OM_perp_KL 
    (h1 : Circle_Tangent A B C O)
    (h2 : Angle_Vertex A O B C)
    (h3 : Points_On_Arc B C M O)
    (h4 : Line_Intersections B M C O A P Q)
    (h5 : Perpendicular_Foot P K (line A C) Q L (line A B)) : 
    Perpendicular (line O M) (line K L) :=
sorry

end tangent_circle_OM_perp_KL_l157_157120


namespace cookie_count_per_box_l157_157774

theorem cookie_count_per_box (A B C T: ℝ) (H1: A = 2) (H2: B = 0.75) (H3: C = 3) (H4: T = 276) :
  T / (A + B + C) = 48 :=
by
  sorry

end cookie_count_per_box_l157_157774


namespace alice_two_turns_probability_l157_157613

def alice_to_alice_first_turn : ℚ := 2 / 3
def alice_to_bob_first_turn : ℚ := 1 / 3
def bob_to_alice_second_turn : ℚ := 1 / 4
def bob_keeps_second_turn : ℚ := 3 / 4
def alice_keeps_second_turn : ℚ := 2 / 3

def probability_alice_keeps_twice : ℚ := alice_to_alice_first_turn * alice_keeps_second_turn
def probability_alice_bob_alice : ℚ := alice_to_bob_first_turn * bob_to_alice_second_turn

theorem alice_two_turns_probability : 
  probability_alice_keeps_twice + probability_alice_bob_alice = 37 / 108 := 
by
  sorry

end alice_two_turns_probability_l157_157613


namespace valid_square_numbers_l157_157952

noncomputable def is_valid_number (N P Q : ℕ) (q : ℕ) : Prop :=
  N = P * 10^q + Q ∧ N = 2 * P * Q

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem valid_square_numbers : 
  ∀ (N : ℕ), (∃ (P Q : ℕ) (q : ℕ), is_valid_number N P Q q) → is_perfect_square N :=
sorry

end valid_square_numbers_l157_157952


namespace tutors_meet_in_360_days_l157_157614

noncomputable def lcm_four_days : ℕ := Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9)

theorem tutors_meet_in_360_days :
  lcm_four_days = 360 := 
by
  -- The proof steps are omitted.
  sorry

end tutors_meet_in_360_days_l157_157614


namespace smallest_sum_of_factors_of_12_factorial_l157_157884

theorem smallest_sum_of_factors_of_12_factorial :
  ∃ (x y z w : Nat), x * y * z * w = Nat.factorial 12 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w = 147 :=
by
  sorry

end smallest_sum_of_factors_of_12_factorial_l157_157884


namespace tan_identity_15_eq_sqrt3_l157_157615

theorem tan_identity_15_eq_sqrt3 :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end tan_identity_15_eq_sqrt3_l157_157615


namespace closest_point_on_ellipse_to_line_l157_157160

theorem closest_point_on_ellipse_to_line :
  ∃ (x y : ℝ), 
    7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0 ∧ (x, y) = (3 / 2, -7 / 4) :=
by
  sorry

end closest_point_on_ellipse_to_line_l157_157160


namespace no_pieces_left_impossible_l157_157579

/-- Starting with 100 pieces and 1 pile, and given the ability to either:
1. Remove one piece from a pile of at least 3 pieces and divide the remaining pile into two non-empty piles,
2. Eliminate a pile containing a single piece,
prove that it is impossible to reach a situation with no pieces left. -/
theorem no_pieces_left_impossible :
  ∀ (p t : ℕ), p = 100 → t = 1 →
  (∀ (p' t' : ℕ),
    (p' = p - 1 ∧ t' = t + 1 ∧ 3 ≤ p) ∨
    (p' = p - 1 ∧ t' = t - 1 ∧ ∃ k, k = 1 ∧ t ≠ 0) →
    false) :=
by
  intros
  sorry

end no_pieces_left_impossible_l157_157579


namespace find_x_l157_157484

theorem find_x (x y z p q r: ℝ) 
  (h1 : (x * y) / (x + y) = p)
  (h2 : (x * z) / (x + z) = q)
  (h3 : (y * z) / (y + z) = r)
  (hp_nonzero : p ≠ 0)
  (hq_nonzero : q ≠ 0)
  (hr_nonzero : r ≠ 0)
  (hxy : x ≠ -y)
  (hxz : x ≠ -z)
  (hyz : y ≠ -z)
  (hpq : p = 3 * q)
  (hpr : p = 2 * r) : x = 3 * p / 2 := 
sorry

end find_x_l157_157484


namespace greatest_percentage_l157_157150

theorem greatest_percentage (pA : ℝ) (pB : ℝ) (wA : ℝ) (wB : ℝ) (sA : ℝ) (sB : ℝ) :
  pA = 0.4 → pB = 0.6 → wA = 0.8 → wB = 0.1 → sA = 0.9 → sB = 0.5 →
  pA * min wA sA + pB * min wB sB = 0.38 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Here you would continue with the proof by leveraging the conditions
  sorry

end greatest_percentage_l157_157150


namespace range_of_a_l157_157832

theorem range_of_a (a b : ℝ) (h : a - 4 * Real.sqrt b = 2 * Real.sqrt (a - b)) : 
  a ∈ {x | 0 ≤ x} ∧ ((a = 0) ∨ (4 ≤ a ∧ a ≤ 20)) :=
by
  sorry

end range_of_a_l157_157832


namespace larger_square_area_total_smaller_squares_area_l157_157135
noncomputable def largerSquareSideLengthFromCircleRadius (r : ℝ) : ℝ :=
  2 * (2 * r)

noncomputable def squareArea (side : ℝ) : ℝ :=
  side * side

theorem larger_square_area (r : ℝ) (h : r = 3) :
  squareArea (largerSquareSideLengthFromCircleRadius r) = 144 :=
by
  sorry

theorem total_smaller_squares_area (r : ℝ) (h : r = 3) :
  4 * squareArea (2 * r) = 144 :=
by
  sorry

end larger_square_area_total_smaller_squares_area_l157_157135


namespace cost_per_crayon_l157_157994

-- Definitions for conditions
def half_dozen := 6
def total_crayons := 4 * half_dozen
def total_cost := 48

-- Problem statement
theorem cost_per_crayon :
  (total_cost / total_crayons) = 2 := 
  by
    sorry

end cost_per_crayon_l157_157994


namespace problem_y_equals_x_squared_plus_x_minus_6_l157_157648

theorem problem_y_equals_x_squared_plus_x_minus_6 (x y : ℝ) :
  (y = x^2 + x - 6 ∧ x = 0 → y = -6) ∧ 
  (y = 0 → x = -3 ∨ x = 2) :=
by
  sorry

end problem_y_equals_x_squared_plus_x_minus_6_l157_157648


namespace solve_nested_function_l157_157969

def f (x : ℝ) : ℝ := x^2 + 12 * x + 30

theorem solve_nested_function :
  ∃ x : ℝ, f (f (f (f (f x)))) = 0 ↔ (x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32)) :=
by sorry

end solve_nested_function_l157_157969


namespace total_trees_on_farm_l157_157146

/-
We need to prove that the total number of trees on the farm now is 88 given the conditions.
-/
theorem total_trees_on_farm 
    (initial_mahogany : ℕ)
    (initial_narra : ℕ)
    (total_fallen : ℕ)
    (more_mahogany_fell_than_narra : ℕ)
    (replanted_narra_factor : ℕ)
    (replanted_mahogany_factor : ℕ) :
    initial_mahogany = 50 →
    initial_narra = 30 →
    total_fallen = 5 →
    more_mahogany_fell_than_narra = 1 →
    replanted_narra_factor = 2 →
    replanted_mahogany_factor = 3 →
    let N := (total_fallen - more_mahogany_fell_than_narra) / 2 in
    let M := N + more_mahogany_fell_than_narra in
    let remaining_mahogany := initial_mahogany - M in
    let remaining_narra := initial_narra - N in
    let planted_narra := replanted_narra_factor * N in
    let planted_mahogany := replanted_mahogany_factor * M in
    remaining_mahogany + planted_mahogany + remaining_narra + planted_narra = 88 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  let N : ℕ := (total_fallen - more_mahogany_fell_than_narra) / 2,
  let M : ℕ := N + more_mahogany_fell_than_narra,
  let remaining_mahogany : ℕ := initial_mahogany - M,
  let remaining_narra : ℕ := initial_narra - N,
  let planted_narra : ℕ := replanted_narra_factor * N,
  let planted_mahogany : ℕ := replanted_mahogany_factor * M,
  have hN : N = 2, {
    sorry,
  },
  have hM : M = 3, {
    sorry,
  },
  suffices : remaining_mahogany + planted_mahogany + remaining_narra + planted_narra = 88, {
    exact this,
  },
  sorry,
}

end total_trees_on_farm_l157_157146


namespace visitors_on_that_day_l157_157775

theorem visitors_on_that_day 
  (prev_visitors : ℕ) 
  (additional_visitors : ℕ) 
  (h1 : prev_visitors = 100)
  (h2 : additional_visitors = 566)
  : prev_visitors + additional_visitors = 666 := by
  sorry

end visitors_on_that_day_l157_157775


namespace gcd_lcm_sum_correct_l157_157919

def gcd_lcm_sum : ℕ :=
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  gcd_40_60 + 2 * lcm_20_15

theorem gcd_lcm_sum_correct : gcd_lcm_sum = 140 := by
  -- Definitions based on conditions
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  
  -- sorry to skip the proof
  sorry

end gcd_lcm_sum_correct_l157_157919


namespace hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l157_157736

theorem hundredth_odd_positive_integer_equals_199 : (2 * 100 - 1 = 199) :=
by {
  sorry
}

theorem even_integer_following_199_equals_200 : (199 + 1 = 200) :=
by {
  sorry
}

end hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l157_157736


namespace points_opposite_sides_of_line_l157_157204

theorem points_opposite_sides_of_line (a : ℝ) :
  (1 + 1 - a) * (2 - 1 - a) < 0 ↔ 1 < a ∧ a < 2 :=
by sorry

end points_opposite_sides_of_line_l157_157204


namespace second_machine_copies_per_minute_l157_157593

-- Definitions based on conditions
def copies_per_minute_first := 35
def total_copies_half_hour := 3300
def time_minutes := 30

-- Theorem statement
theorem second_machine_copies_per_minute : 
  ∃ (x : ℕ), (copies_per_minute_first * time_minutes + x * time_minutes = total_copies_half_hour) ∧ (x = 75) := by
  sorry

end second_machine_copies_per_minute_l157_157593


namespace older_brother_stamps_l157_157251

variable (y o : ℕ)

def condition1 : Prop := o = 2 * y + 1
def condition2 : Prop := o + y = 25

theorem older_brother_stamps (h1 : condition1 y o) (h2 : condition2 y o) : o = 17 :=
by
  sorry

end older_brother_stamps_l157_157251


namespace can_encode_number_l157_157459

theorem can_encode_number : ∃ (m n : ℕ), (0.07 = 1 / (m : ℝ) + 1 / (n : ℝ)) :=
by
  -- Proof omitted
  sorry

end can_encode_number_l157_157459


namespace line_equation_minimized_area_l157_157011

theorem line_equation_minimized_area :
  ∀ (l_1 l_2 l_3 : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop),
    (∀ x y : ℝ, l_1 (x, y) ↔ 3 * x + 2 * y - 1 = 0) ∧
    (∀ x y : ℝ, l_2 (x, y) ↔ 5 * x + 2 * y + 1 = 0) ∧
    (∀ x y : ℝ, l_3 (x, y) ↔ 3 * x - 5 * y + 6 = 0) →
    (∃ c : ℝ, ∀ x y : ℝ, l (x, y) ↔ 3 * x - 5 * y + c = 0) →
    (∃ x y : ℝ, l_1 (x, y) ∧ l_2 (x, y) ∧ l (x, y)) →
    (∀ a : ℝ, ∀ x y : ℝ, l (x, y) ↔ x + y = a) →
    (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, l (x, y) ↔ 2 * x - y + 4 = 0) → 
    sorry :=
sorry

end line_equation_minimized_area_l157_157011


namespace gcd_of_7854_and_15246_is_6_six_is_not_prime_l157_157164

theorem gcd_of_7854_and_15246_is_6 : gcd 7854 15246 = 6 := sorry

theorem six_is_not_prime : ¬ Prime 6 := sorry

end gcd_of_7854_and_15246_is_6_six_is_not_prime_l157_157164


namespace cruzs_marbles_l157_157902

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l157_157902


namespace degenerate_ellipse_value_c_l157_157335

theorem degenerate_ellipse_value_c (c : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0) ∧
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0 → (x+1)^2 + (y-7)^2 = 0) ↔ c = 52 :=
by
  sorry

end degenerate_ellipse_value_c_l157_157335


namespace mrs_oaklyn_profit_is_correct_l157_157410

def cost_of_buying_rugs (n : ℕ) (cost_per_rug : ℕ) : ℕ :=
  n * cost_per_rug

def transportation_fee (n : ℕ) (fee_per_rug : ℕ) : ℕ :=
  n * fee_per_rug

def selling_price_before_tax (n : ℕ) (price_per_rug : ℕ) : ℕ :=
  n * price_per_rug

def total_tax (price_before_tax : ℕ) (tax_rate : ℕ) : ℕ :=
  price_before_tax * tax_rate / 100

def total_selling_price_after_tax (price_before_tax : ℕ) (tax_amount : ℕ) : ℕ :=
  price_before_tax + tax_amount

def profit (selling_price_after_tax : ℕ) (cost_of_buying : ℕ) (transport_fee : ℕ) : ℕ :=
  selling_price_after_tax - (cost_of_buying + transport_fee)

def rugs := 20
def cost_per_rug := 40
def transport_fee_per_rug := 5
def price_per_rug := 60
def tax_rate := 10

theorem mrs_oaklyn_profit_is_correct : 
  profit 
    (total_selling_price_after_tax 
      (selling_price_before_tax rugs price_per_rug) 
      (total_tax (selling_price_before_tax rugs price_per_rug) tax_rate)
    )
    (cost_of_buying_rugs rugs cost_per_rug) 
    (transportation_fee rugs transport_fee_per_rug) 
  = 420 :=
by sorry

end mrs_oaklyn_profit_is_correct_l157_157410


namespace sequence_sixth_term_l157_157833

theorem sequence_sixth_term :
  ∃ (a : ℕ → ℕ),
    a 1 = 3 ∧
    a 5 = 43 ∧
    (∀ n, a (n + 1) = (1/4) * (a n + a (n + 2))) →
    a 6 = 129 :=
sorry

end sequence_sixth_term_l157_157833


namespace sum_of_squares_l157_157723

/-- 
Given two real numbers x and y, if their product is 120 and their sum is 23, 
then the sum of their squares is 289.
-/
theorem sum_of_squares (x y : ℝ) (h₁ : x * y = 120) (h₂ : x + y = 23) :
  x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l157_157723


namespace max_balls_l157_157854

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l157_157854


namespace compare_rat_neg_l157_157329

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l157_157329


namespace max_divisor_of_expression_l157_157592

theorem max_divisor_of_expression 
  (n : ℕ) (hn : n > 0) : ∃ k, k = 8 ∧ 8 ∣ (5^n + 2 * 3^(n-1) + 1) :=
by
  sorry

end max_divisor_of_expression_l157_157592


namespace lobster_distribution_l157_157887

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

end lobster_distribution_l157_157887


namespace find_2theta_plus_phi_l157_157356

variable (θ φ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (hφ : 0 < φ ∧ φ < π / 2)
variable (tan_hθ : Real.tan θ = 2 / 5)
variable (cos_hφ : Real.cos φ = 1 / 2)

theorem find_2theta_plus_phi : 2 * θ + φ = π / 4 := by
  sorry

end find_2theta_plus_phi_l157_157356


namespace large_square_area_l157_157092

theorem large_square_area (a b c : ℕ) (h1 : 4 * a < b) (h2 : c^2 = a^2 + b^2 + 10) : c^2 = 36 :=
  sorry

end large_square_area_l157_157092


namespace triangle_perimeter_l157_157738

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p1 p3 + distance p2 p3

theorem triangle_perimeter :
  let p1 := (1, 4)
  let p2 := (-7, 0)
  let p3 := (1, 0)
  perimeter p1 p2 p3 = 4 * Real.sqrt 5 + 12 :=
by
  sorry

end triangle_perimeter_l157_157738


namespace construction_costs_l157_157939

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end construction_costs_l157_157939


namespace num_quadricycles_l157_157449

theorem num_quadricycles (b t q : ℕ) (h1 : b + t + q = 10) (h2 : 2 * b + 3 * t + 4 * q = 30) : q = 2 :=
by sorry

end num_quadricycles_l157_157449


namespace sin_cos_eq_l157_157181

theorem sin_cos_eq (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := sorry

end sin_cos_eq_l157_157181


namespace go_piece_arrangement_l157_157744

theorem go_piece_arrangement (w b : ℕ) (pieces : List ℕ) 
    (h_w : w = 180) (h_b : b = 181)
    (h_pieces : pieces.length = w + b) 
    (h_black_count : pieces.count 1 = b) 
    (h_white_count : pieces.count 0 = w) :
    ∃ (i j : ℕ), i < j ∧ j < pieces.length ∧ 
    ((j - i - 1 = 178) ∨ (j - i - 1 = 181)) ∧ 
    (pieces.get ⟨i, sorry⟩ = 1) ∧ 
    (pieces.get ⟨j, sorry⟩ = 1) := 
sorry

end go_piece_arrangement_l157_157744


namespace closest_point_on_ellipse_l157_157162

theorem closest_point_on_ellipse : 
  ∃ (x y : ℝ), (7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0) ∧ 
  (∀ (x' y' : ℝ), 7 * x'^2 + 4 * y'^2 = 28 → dist (x, y) (0, 0) ≤ dist (x', y') (0, 0)) :=
sorry

end closest_point_on_ellipse_l157_157162


namespace gain_percent_40_l157_157271

theorem gain_percent_40 (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1260) :
  ((selling_price - cost_price) / cost_price) * 100 = 40 :=
by
  sorry

end gain_percent_40_l157_157271


namespace max_marks_mike_l157_157540

theorem max_marks_mike (pass_percentage : ℝ) (scored_marks : ℝ) (shortfall : ℝ) : 
  pass_percentage = 0.30 → 
  scored_marks = 212 → 
  shortfall = 28 → 
  (scored_marks + shortfall) = 240 → 
  (scored_marks + shortfall) = pass_percentage * (max_marks : ℝ) → 
  max_marks = 800 := 
by 
  intros hp hs hsh hps heq 
  sorry

end max_marks_mike_l157_157540


namespace cruzs_marbles_l157_157900

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l157_157900


namespace parabola_above_line_l157_157652

variable {a b c : ℝ}

theorem parabola_above_line
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (H : (b - c) ^ 2 - 4 * a * c < 0) :
  (b + c) ^ 2 - 4 * c * (a + b) < 0 := 
sorry

end parabola_above_line_l157_157652


namespace find_m_value_l157_157026

open Nat

theorem find_m_value {m : ℕ} (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 24 :=
  sorry

end find_m_value_l157_157026


namespace buttons_ratio_l157_157405

theorem buttons_ratio
  (initial_buttons : ℕ)
  (shane_multiplier : ℕ)
  (final_buttons : ℕ)
  (total_buttons_after_shane : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  total_buttons_after_shane = initial_buttons + shane_multiplier * initial_buttons →
  (total_buttons_after_shane - final_buttons) / total_buttons_after_shane = 1 / 2 :=
by
  intros
  sorry

end buttons_ratio_l157_157405


namespace pascal_28_25_eq_2925_l157_157918

-- Define the Pascal's triangle nth-row function
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the theorem to prove that the 25th element in the 28 element row is 2925
theorem pascal_28_25_eq_2925 :
  pascal 27 24 = 2925 :=
by
  sorry

end pascal_28_25_eq_2925_l157_157918


namespace positional_relationship_of_circles_l157_157504

theorem positional_relationship_of_circles 
  (m n : ℝ)
  (h1 : ∃ (x y : ℝ), x^2 - 10 * x + n = 0 ∧ y^2 - 10 * y + n = 0 ∧ x = 2 ∧ y = m) :
  n = 2 * m ∧ m = 8 → 16 > 2 + 8 :=
by
  sorry

end positional_relationship_of_circles_l157_157504


namespace eliza_height_l157_157341

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end eliza_height_l157_157341


namespace remainder_when_divided_by_x_minus_2_l157_157348

def p (x : ℕ) : ℕ := x^5 - 2 * x^3 + 4 * x + 5

theorem remainder_when_divided_by_x_minus_2 : p 2 = 29 := 
by {
  sorry
}

end remainder_when_divided_by_x_minus_2_l157_157348


namespace find_y_l157_157015

theorem find_y : (12 : ℝ)^3 * (2 : ℝ)^4 / 432 = 5184 → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end find_y_l157_157015


namespace no_zero_terms_in_arithmetic_progression_l157_157802

theorem no_zero_terms_in_arithmetic_progression (a d : ℤ) (h : ∃ (n : ℕ), 2 * a + (2 * n - 1) * d = ((3 * n - 1) * (2 * a + (3 * n - 2) * d)) / 2) :
  ∀ (m : ℕ), a + (m - 1) * d ≠ 0 :=
by
  sorry

end no_zero_terms_in_arithmetic_progression_l157_157802


namespace local_minimum_at_two_l157_157656

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_at_two : ∃ a : ℝ, a = 2 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - a| ∧ |x - a| < δ) → f x > f a :=
by sorry

end local_minimum_at_two_l157_157656


namespace John_pays_amount_l157_157838

/-- Prove the amount John pays given the conditions -/
theorem John_pays_amount
  (total_candies : ℕ)
  (candies_paid_by_dave : ℕ)
  (cost_per_candy : ℚ)
  (candies_paid_by_john := total_candies - candies_paid_by_dave)
  (total_cost_paid_by_john := candies_paid_by_john * cost_per_candy) :
  total_candies = 20 →
  candies_paid_by_dave = 6 →
  cost_per_candy = 1.5 →
  total_cost_paid_by_john = 21 := 
by
  intros h1 h2 h3
  -- Proof is skipped
  sorry

end John_pays_amount_l157_157838


namespace subcommittee_count_l157_157747

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose republicans subcommittee_republicans * choose democrats subcommittee_democrats = 11760 :=
by
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end subcommittee_count_l157_157747


namespace find_original_speed_l157_157607

theorem find_original_speed (r : ℝ) (t : ℝ)
  (h_circumference : r * t = 15 / 5280)
  (h_increase : (r + 8) * (t - 1/10800) = 15 / 5280) :
  r = 7.5 :=
sorry

end find_original_speed_l157_157607


namespace solve_for_b_l157_157690

noncomputable def g (a b : ℝ) (x : ℝ) := 1 / (2 * a * x + 3 * b)

theorem solve_for_b (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (g a b (2) = 1 / (4 * a + 3 * b)) → (4 * a + 3 * b = 1 / 2) → b = (1 - 4 * a) / 3 :=
by
  sorry

end solve_for_b_l157_157690


namespace gcd_6Pn_n_minus_2_l157_157168

-- Auxiliary definition to calculate the nth pentagonal number
def pentagonal (n : ℕ) : ℕ := n ^ 2

-- Statement of the theorem
theorem gcd_6Pn_n_minus_2 (n : ℕ) (hn : 0 < n) : 
  ∃ d, d = Int.gcd (6 * pentagonal n) (n - 2) ∧ d ≤ 24 ∧ (∀ k, Int.gcd (6 * pentagonal k) (k - 2) ≤ 24) :=
sorry

end gcd_6Pn_n_minus_2_l157_157168


namespace sum_of_roots_l157_157364

theorem sum_of_roots (a b : ℝ) (h1 : a^2 - 4*a - 2023 = 0) (h2 : b^2 - 4*b - 2023 = 0) : a + b = 4 :=
sorry

end sum_of_roots_l157_157364


namespace union_of_A_and_B_l157_157538

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end union_of_A_and_B_l157_157538


namespace team_B_eligible_l157_157912

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l157_157912


namespace find_b_l157_157207

variable (a b c : ℝ)
variable (sin cos : ℝ → ℝ)

-- Assumptions or Conditions
variables (h1 : a^2 - c^2 = 2 * b) 
variables (h2 : sin (b) = 4 * cos (a) * sin (c))

theorem find_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin (b) = 4 * cos (a) * sin (c)) : b = 4 := 
by
  sorry

end find_b_l157_157207


namespace math_players_count_l157_157616

-- Define the conditions given in the problem.
def total_players : ℕ := 25
def physics_players : ℕ := 9
def both_subjects_players : ℕ := 5

-- Statement to be proven
theorem math_players_count :
  total_players = physics_players + both_subjects_players + (total_players - physics_players - both_subjects_players) → 
  total_players - physics_players + both_subjects_players = 21 := 
sorry

end math_players_count_l157_157616


namespace weighted_average_correct_l157_157268

noncomputable def weightedAverage := 
  (5 * (3/5 : ℝ) + 3 * (4/9 : ℝ) + 8 * 0.45 + 4 * 0.067) / (5 + 3 + 8 + 4)

theorem weighted_average_correct :
  weightedAverage = 0.41 :=
by
  sorry

end weighted_average_correct_l157_157268


namespace condition_not_right_triangle_l157_157920

theorem condition_not_right_triangle 
  (AB BC AC : ℕ) (angleA angleB angleC : ℕ)
  (h_A : AB = 3 ∧ BC = 4 ∧ AC = 5)
  (h_B : AB / BC = 3 / 4 ∧ BC / AC = 4 / 5 ∧ AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB)
  (h_C : angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 ∧ angleA + angleB + angleC = 180)
  (h_D : angleA = 40 ∧ angleB = 50 ∧ angleA + angleB + angleC = 180) :
  angleA = 45 ∧ angleB = 60 ∧ angleC = 75 ∧ (¬ (angleA = 90 ∨ angleB = 90 ∨ angleC = 90)) :=
sorry

end condition_not_right_triangle_l157_157920


namespace men_left_hostel_l157_157934

-- Definitions based on the conditions given
def initialMen : ℕ := 250
def initialDays : ℕ := 28
def remainingDays : ℕ := 35

-- The theorem we need to prove
theorem men_left_hostel (x : ℕ) (h : initialMen * initialDays = (initialMen - x) * remainingDays) : x = 50 :=
by
  sorry

end men_left_hostel_l157_157934


namespace range_of_x_in_function_l157_157681

theorem range_of_x_in_function (x : ℝ) :
  (x - 1 ≥ 0) ∧ (x - 2 ≠ 0) → (x ≥ 1 ∧ x ≠ 2) :=
by
  intro h
  sorry

end range_of_x_in_function_l157_157681


namespace sport_formulation_water_l157_157110

theorem sport_formulation_water
  (f : ℝ) (c : ℝ) (w : ℝ) 
  (f_s : ℝ) (c_s : ℝ) (w_s : ℝ)
  (standard_ratio : f / c = 1 / 12 ∧ f / w = 1 / 30)
  (sport_ratio_corn_syrup : f_s / c_s = 3 * (f / c))
  (sport_ratio_water : f_s / w_s = (1 / 2) * (f / w))
  (corn_syrup_amount : c_s = 3) :
  w_s = 45 :=
by
  sorry

end sport_formulation_water_l157_157110


namespace point_exists_if_square_or_rhombus_l157_157279

-- Definitions to state the problem
structure Point (α : Type*) := (x : α) (y : α)
structure Rectangle (α : Type*) := (A B C D : Point α)

-- Definition of equidistant property
def isEquidistant (α : Type*) [LinearOrderedField α] (P : Point α) (R : Rectangle α) : Prop :=
  let d1 := abs (P.y - R.A.y)
  let d2 := abs (P.y - R.C.y)
  let d3 := abs (P.x - R.A.x)
  let d4 := abs (P.x - R.B.x)
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4

-- Theorem stating the problem
theorem point_exists_if_square_or_rhombus {α : Type*} [LinearOrderedField α]
  (R : Rectangle α) : 
  (∃ P : Point α, isEquidistant α P R) ↔ 
  (∃ (a b : α), (a ≠ b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b) ∨ 
                (a = b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b)) :=
sorry

end point_exists_if_square_or_rhombus_l157_157279


namespace smallest_k_power_l157_157582

theorem smallest_k_power (k : ℕ) (hk : ∀ m : ℕ, m < 14 → 7^m ≤ 4^19) : 7^14 > 4^19 :=
sorry

end smallest_k_power_l157_157582


namespace lolita_milk_per_week_l157_157402

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end lolita_milk_per_week_l157_157402


namespace dandelion_seed_production_l157_157411

theorem dandelion_seed_production :
  (one_seed : ℕ) (produced_seeds : ℕ)
  (germinated_fraction : ℚ)
  (new_seedlings_count : ℕ)
  (seed_count_after_two_months : ℕ) :
  one_seed = 1 →
  produced_seeds = 50 →
  germinated_fraction = 1/2 →
  new_seedlings_count = produced_seeds * germinated_fraction.numerator / germinated_fraction.denominator →
  seed_count_after_two_months = new_seedlings_count * produced_seeds →
  seed_count_after_two_months = 1250 :=
by
  intros
  sorry

end dandelion_seed_production_l157_157411


namespace balls_in_rightmost_box_l157_157259

theorem balls_in_rightmost_box (a : ℕ → ℕ)
  (h₀ : a 1 = 7)
  (h₁ : ∀ i, 1 ≤ i ∧ i ≤ 1990 → a i + a (i + 1) + a (i + 2) + a (i + 3) = 30) :
  a 1993 = 7 :=
sorry

end balls_in_rightmost_box_l157_157259


namespace value_of_x_yplusz_l157_157200

theorem value_of_x_yplusz (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 :=
by
  sorry

end value_of_x_yplusz_l157_157200


namespace max_balls_drawn_l157_157861

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l157_157861


namespace Heather_total_distance_walked_l157_157033

theorem Heather_total_distance_walked :
  let d1 := 0.645
  let d2 := 1.235
  let d3 := 0.875
  let d4 := 1.537
  let d5 := 0.932
  (d1 + d2 + d3 + d4 + d5) = 5.224 := 
by
  sorry -- Proof goes here

end Heather_total_distance_walked_l157_157033


namespace initial_apples_9_l157_157226

def initial_apple_count (picked : ℕ) (remaining : ℕ) : ℕ :=
  picked + remaining

theorem initial_apples_9 (picked : ℕ) (remaining : ℕ) :
  picked = 2 → remaining = 7 → initial_apple_count picked remaining = 9 := by
sorry

end initial_apples_9_l157_157226


namespace study_group_number_l157_157232

theorem study_group_number (b : ℤ) :
  (¬ (b % 2 = 0) ∧ (b + b^3 < 8000) ∧ ¬ (∃ r : ℚ, r^2 = 13) ∧ (b % 7 = 0)
  ∧ (∃ r : ℚ, r = b) ∧ ¬ (b % 14 = 0)) →
  b = 7 :=
by
  sorry

end study_group_number_l157_157232


namespace max_balls_count_l157_157850

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l157_157850


namespace total_candies_needed_l157_157789

def candies_per_box : ℕ := 156
def number_of_children : ℕ := 20

theorem total_candies_needed : candies_per_box * number_of_children = 3120 := by
  sorry

end total_candies_needed_l157_157789


namespace max_tan_beta_l157_157018

theorem max_tan_beta (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2) 
  (h : α + β ≠ π / 2) (h_sin_cos : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β ≤ Real.sqrt 3 / 3 :=
sorry

end max_tan_beta_l157_157018


namespace patients_before_doubling_l157_157596

theorem patients_before_doubling (C P : ℕ) 
    (h1 : (1 / 4) * C = 13) 
    (h2 : C = 2 * P) : 
    P = 26 := 
sorry

end patients_before_doubling_l157_157596


namespace tom_age_ratio_l157_157093

theorem tom_age_ratio (T N : ℕ) (h1 : T = 2 * (T / 2)) (h2 : T - N = 3 * ((T / 2) - 3 * N)) : T / N = 16 :=
  sorry

end tom_age_ratio_l157_157093


namespace total_balls_in_box_l157_157513

theorem total_balls_in_box (red blue yellow total : ℕ) 
  (h1 : 2 * blue = 3 * red)
  (h2 : 3 * yellow = 4 * red) 
  (h3 : yellow = 40)
  (h4 : red + blue + yellow = total) : total = 90 :=
sorry

end total_balls_in_box_l157_157513


namespace tigers_in_zoo_l157_157990

-- Given definitions
def ratio_lions_tigers := 3 / 4
def number_of_lions := 21
def number_of_tigers := 28

-- Problem statement
theorem tigers_in_zoo : (number_of_lions : ℚ) / 3 * 4 = number_of_tigers := by
  sorry

end tigers_in_zoo_l157_157990


namespace david_total_hours_on_course_l157_157002

def hours_per_week_class := 2 * 3 + 4 -- hours per week in class
def hours_per_week_homework := 4 -- hours per week in homework
def total_hours_per_week := hours_per_week_class + hours_per_week_homework -- total hours per week

theorem david_total_hours_on_course :
  let total_weeks := 24
  in total_weeks * total_hours_per_week = 336 := by
  sorry

end david_total_hours_on_course_l157_157002


namespace cut_into_four_and_reassemble_l157_157782

-- Definitions as per conditions in the problem
def figureArea : ℕ := 36
def nParts : ℕ := 4
def squareArea (s : ℕ) : ℕ := s * s

-- Property to be proved
theorem cut_into_four_and_reassemble :
  ∃ (s : ℕ), squareArea s = figureArea / nParts ∧ s * s = figureArea :=
by
  sorry

end cut_into_four_and_reassemble_l157_157782


namespace construction_costs_correct_l157_157937

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end construction_costs_correct_l157_157937


namespace cruzs_marbles_l157_157901

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l157_157901


namespace plane_eq_l157_157798

def gcd4 (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd (Int.gcd (abs a) (abs b)) (abs c)) (abs d)

theorem plane_eq (A B C D : ℤ) (A_pos : A > 0) 
  (gcd_1 : gcd4 A B C D = 1) 
  (H_parallel : (A, B, C) = (3, 2, -4)) 
  (H_point : A * 2 + B * 3 + C * (-1) + D = 0) : 
  A = 3 ∧ B = 2 ∧ C = -4 ∧ D = -16 := 
sorry

end plane_eq_l157_157798


namespace compare_rat_neg_l157_157334

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l157_157334


namespace a_squared_plus_b_squared_plus_c_squared_geq_sqrt_3_l157_157247

theorem a_squared_plus_b_squared_plus_c_squared_geq_sqrt_3 (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) : a^2 + b^2 + c^2 ≥ real.sqrt 3 := 
by
  sorry

end a_squared_plus_b_squared_plus_c_squared_geq_sqrt_3_l157_157247


namespace inequality_proof_l157_157427

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 :=
by
  sorry

end inequality_proof_l157_157427


namespace repeating_decimal_division_l157_157476

def repeating_decimal_081_as_fraction : ℚ := 9 / 11
def repeating_decimal_272_as_fraction : ℚ := 30 / 11

theorem repeating_decimal_division : 
  (repeating_decimal_081_as_fraction / repeating_decimal_272_as_fraction) = (3 / 10) := 
by 
  sorry

end repeating_decimal_division_l157_157476


namespace contractor_fired_people_l157_157123

theorem contractor_fired_people :
  ∀ (total_days : ℕ) (initial_people : ℕ) (partial_days : ℕ) 
    (partial_work_fraction : ℚ) (remaining_days : ℕ) 
    (fired_people : ℕ),
  total_days = 100 →
  initial_people = 10 →
  partial_days = 20 →
  partial_work_fraction = 1 / 4 →
  remaining_days = 75 →
  (initial_people - fired_people) * remaining_days * (1 - partial_work_fraction) / partial_days = initial_people * total_days →
  fired_people = 2 :=
by
  intros total_days initial_people partial_days partial_work_fraction remaining_days fired_people
  intro h1 h2 h3 h4 h5 h6
  sorry

end contractor_fired_people_l157_157123


namespace total_hovering_time_is_24_hours_l157_157143

-- Define the initial conditions
def mountain_time_day1 : ℕ := 3
def central_time_day1 : ℕ := 4
def eastern_time_day1 : ℕ := 2

-- Define the additional time hovered in each zone on the second day
def additional_time_per_zone_day2 : ℕ := 2

-- Calculate the total time spent on each day
def total_time_day1 : ℕ := mountain_time_day1 + central_time_day1 + eastern_time_day1
def total_additional_time_day2 : ℕ := 3 * additional_time_per_zone_day2 -- there are three zones
def total_time_day2 : ℕ := total_time_day1 + total_additional_time_day2

-- Calculate the total time over the two days
def total_time_two_days : ℕ := total_time_day1 + total_time_day2

-- Prove that the total time over the two days is 24 hours
theorem total_hovering_time_is_24_hours : total_time_two_days = 24 := by
  sorry

end total_hovering_time_is_24_hours_l157_157143


namespace p_of_neg3_equals_14_l157_157220

-- Functions definitions
def u (x : ℝ) : ℝ := 4 * x + 5
def p (y : ℝ) : ℝ := y^2 - 2 * y + 6

-- Theorem statement
theorem p_of_neg3_equals_14 : p (-3) = 14 := by
  sorry

end p_of_neg3_equals_14_l157_157220


namespace parabola_above_line_l157_157651

variable {a b c : ℝ}

theorem parabola_above_line
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (H : (b - c) ^ 2 - 4 * a * c < 0) :
  (b + c) ^ 2 - 4 * c * (a + b) < 0 := 
sorry

end parabola_above_line_l157_157651


namespace quadratic_inequality_solution_l157_157174

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) :=
by sorry

end quadratic_inequality_solution_l157_157174


namespace even_diagonal_moves_l157_157446

def King_Moves (ND D : ℕ) :=
  ND + D = 63 ∧ ND % 2 = 0

theorem even_diagonal_moves (ND D : ℕ) (traverse_board : King_Moves ND D) : D % 2 = 0 :=
by
  sorry

end even_diagonal_moves_l157_157446


namespace find_a_l157_157177

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem find_a (a : ℝ) (h : f a (f a 1) = 2) : a = -2 := by
  sorry

end find_a_l157_157177


namespace oreos_total_l157_157531

variable (Jordan : ℕ)
variable (James : ℕ := 4 * Jordan + 7)

theorem oreos_total (h : James = 43) : 43 + Jordan = 52 :=
sorry

end oreos_total_l157_157531


namespace percentage_is_36_point_4_l157_157280

def part : ℝ := 318.65
def whole : ℝ := 875.3

theorem percentage_is_36_point_4 : (part / whole) * 100 = 36.4 := 
by sorry

end percentage_is_36_point_4_l157_157280


namespace highway_extension_l157_157767

theorem highway_extension 
  (current_length : ℕ) 
  (desired_length : ℕ) 
  (first_day_miles : ℕ) 
  (miles_needed : ℕ) 
  (second_day_miles : ℕ) 
  (h1 : current_length = 200) 
  (h2 : desired_length = 650) 
  (h3 : first_day_miles = 50) 
  (h4 : miles_needed = 250) 
  (h5 : second_day_miles = desired_length - current_length - miles_needed - first_day_miles) :
  second_day_miles / first_day_miles = 3 := 
sorry

end highway_extension_l157_157767


namespace nursing_home_milk_l157_157340

theorem nursing_home_milk :
  ∃ x y : ℕ, (2 * x + 16 = y) ∧ (4 * x - 12 = y) ∧ (x = 14) ∧ (y = 44) :=
by
  sorry

end nursing_home_milk_l157_157340


namespace cone_volume_l157_157986

theorem cone_volume (l h : ℝ) (l_eq : l = 5) (h_eq : h = 4) : 
  (1 / 3) * Real.pi * ((l^2 - h^2).sqrt)^2 * h = 12 * Real.pi := 
by 
  sorry

end cone_volume_l157_157986


namespace penny_frogs_count_l157_157293

theorem penny_frogs_count :
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  tree_frogs + poison_frogs + wood_frogs = 78 :=
by
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  show tree_frogs + poison_frogs + wood_frogs = 78
  sorry

end penny_frogs_count_l157_157293


namespace percentage_customers_not_pay_tax_l157_157763

theorem percentage_customers_not_pay_tax
  (daily_shoppers : ℕ)
  (weekly_tax_payers : ℕ)
  (h1 : daily_shoppers = 1000)
  (h2 : weekly_tax_payers = 6580)
  : ((7000 - weekly_tax_payers) / 7000) * 100 = 6 := 
by sorry

end percentage_customers_not_pay_tax_l157_157763


namespace find_P_l157_157611

theorem find_P (P : ℕ) (h : 4 * (P + 4 + 8 + 20) = 252) : P = 31 :=
by
  -- Assume this proof is nontrivial and required steps
  sorry

end find_P_l157_157611


namespace train_length_l157_157606

theorem train_length (L : ℝ) (h1 : ∀ t1 : ℝ, t1 = 15 → ∀ p1 : ℝ, p1 = 180 → (L + p1) / t1 = v)
(h2 : ∀ t2 : ℝ, t2 = 20 → ∀ p2 : ℝ, p2 = 250 → (L + p2) / t2 = v) : 
L = 30 :=
by
  have h1 := h1 15 rfl 180 rfl
  have h2 := h2 20 rfl 250 rfl
  sorry

end train_length_l157_157606


namespace number_of_girls_l157_157560

theorem number_of_girls (n : ℕ) (A : ℝ) 
    (h1 : A = (n * (A + 1) + 55 - 80) / n) : n = 25 :=
by 
  sorry

end number_of_girls_l157_157560


namespace express_B_using_roster_l157_157176

open Set

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem express_B_using_roster :
  B = {4, 9, 16} := by
  sorry

end express_B_using_roster_l157_157176


namespace p_sufficient_not_necessary_for_q_l157_157361

-- Define the propositions p and q based on the given conditions
def p (α : ℝ) : Prop := α = Real.pi / 4
def q (α : ℝ) : Prop := Real.sin α = Real.cos α

-- Theorem that states p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (α : ℝ) : p α → (q α) ∧ ¬(q α → p α) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l157_157361


namespace profits_ratio_l157_157431

-- Definitions
def investment_ratio (p q : ℕ) := 7 * p = 5 * q
def investment_period_p := 10
def investment_period_q := 20

-- Prove the ratio of profits
theorem profits_ratio (p q : ℕ) (h1 : investment_ratio p q) :
  (7 * p * investment_period_p / (5 * q * investment_period_q)) = 7 / 10 :=
sorry

end profits_ratio_l157_157431


namespace surveyed_parents_women_l157_157211

theorem surveyed_parents_women (W : ℝ) :
  (5/6 : ℝ) * W + (3/4 : ℝ) * (1 - W) = 0.8 → W = 0.6 :=
by
  intro h
  have hw : W * (1/6) + (1 - W) * (1/4) = 0.2 := sorry
  have : W = 0.6 := sorry
  exact this

end surveyed_parents_women_l157_157211


namespace ratios_of_PQR_and_XYZ_l157_157914

-- Define triangle sides
def sides_PQR : ℕ × ℕ × ℕ := (7, 24, 25)
def sides_XYZ : ℕ × ℕ × ℕ := (9, 40, 41)

-- Perimeter calculation functions
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Area calculation functions for right triangles
def area (a b : ℕ) : ℕ := (a * b) / 2

-- Required proof statement
theorem ratios_of_PQR_and_XYZ :
  let (a₁, b₁, c₁) := sides_PQR
  let (a₂, b₂, c₂) := sides_XYZ
  area a₁ b₁ * 15 = 7 * area a₂ b₂ ∧ perimeter a₁ b₁ c₁ * 45 = 28 * perimeter a₂ b₂ c₂ :=
sorry

end ratios_of_PQR_and_XYZ_l157_157914


namespace clarinet_cost_correct_l157_157951

noncomputable def total_spent : ℝ := 141.54
noncomputable def song_book_cost : ℝ := 11.24
noncomputable def clarinet_cost : ℝ := total_spent - song_book_cost

theorem clarinet_cost_correct : clarinet_cost = 130.30 :=
by
  sorry

end clarinet_cost_correct_l157_157951


namespace no_solution_iff_a_leq_8_l157_157044

theorem no_solution_iff_a_leq_8 (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_iff_a_leq_8_l157_157044


namespace find_line_equation_l157_157803

-- Definition of a line passing through a point
def passes_through (l : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := l p.1 p.2

-- Definition of intercepts being opposite
def opposite_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ l a 0 ∧ l 0 (-a)

-- The line passing through the point (7, 1)
def line_exists (l : ℝ → ℝ → Prop) : Prop :=
  passes_through l (7, 1) ∧ opposite_intercepts l

-- Main theorem to prove the equation of the line
theorem find_line_equation (l : ℝ → ℝ → Prop) :
  line_exists l ↔ (∀ x y, l x y ↔ x - 7 * y = 0) ∨ (∀ x y, l x y ↔ x - y - 6 = 0) :=
sorry

end find_line_equation_l157_157803


namespace basket_weight_l157_157930

def weight_of_basket_alone (n_pears : ℕ) (weight_per_pear total_weight : ℚ) : ℚ :=
  total_weight - (n_pears * weight_per_pear)

theorem basket_weight :
  weight_of_basket_alone 30 0.36 11.26 = 0.46 := by
  sorry

end basket_weight_l157_157930


namespace manager_salary_l157_157716

theorem manager_salary 
  (a : ℝ) (n : ℕ) (m_total : ℝ) (new_avg : ℝ) (m_avg_inc : ℝ)
  (h1 : n = 20) 
  (h2 : a = 1600) 
  (h3 : m_avg_inc = 100) 
  (h4 : new_avg = a + m_avg_inc)
  (h5 : m_total = n * a)
  (h6 : new_avg = (m_total + M) / (n + 1)) : 
  M = 3700 :=
by
  sorry

end manager_salary_l157_157716


namespace pool_filling_time_l157_157570

theorem pool_filling_time :
  (∀ t : ℕ, t >= 6 → ∃ v : ℝ, v = (2^(t-6)) * 0.25) →
  ∃ t : ℕ, t = 8 :=
by
  intros h
  existsi 8
  sorry

end pool_filling_time_l157_157570


namespace find_a_value_l157_157368

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l157_157368


namespace find_c_l157_157966

theorem find_c (c : ℝ)
  (h1 : ∃ y : ℝ, y = (-2)^2 - (-2) + c)
  (h2 : ∃ m : ℝ, m = 2 * (-2) - 1)
  (h3 : ∃ x y, y - (4 + c) = -5 * (x + 2) ∧ x = 0 ∧ y = 0) :
  c = 4 :=
sorry

end find_c_l157_157966


namespace tangent_lines_parabola_through_point_l157_157971

theorem tangent_lines_parabola_through_point :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = x ^ 2 + 1 → (y - 0) = m * (x - 0)) 
     ∧ ((m = 2 ∧ y = 2 * x) ∨ (m = -2 ∧ y = -2 * x)) :=
sorry

end tangent_lines_parabola_through_point_l157_157971


namespace max_silver_coins_l157_157923

theorem max_silver_coins (n : ℕ) : (n < 150) ∧ (n % 15 = 3) → n = 138 :=
by
  sorry

end max_silver_coins_l157_157923


namespace cubic_roots_inequalities_l157_157688

theorem cubic_roots_inequalities 
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ z : ℂ, (a * z^3 + b * z^2 + c * z + d = 0) → z.re < 0) :
  a * b > 0 ∧ b * c - a * d > 0 ∧ a * d > 0 :=
by
  sorry

end cubic_roots_inequalities_l157_157688


namespace volume_ratio_of_cube_cut_l157_157421

/-
  The cube ABCDEFGH has its side length assumed to be 1.
  The points K, L, M divide the vertical edges AA', BB', CC'
  respectively, in the ratios 1:2, 1:3, 1:4. 
  We need to prove that the plane KLM cuts the cube into
  two parts such that the volume ratio of the two parts is 4:11.
-/
theorem volume_ratio_of_cube_cut (s : ℝ) (K L M : ℝ) :
  ∃ (Vbelow Vabove : ℝ), 
    s = 1 → 
    K = 1/3 → 
    L = 1/4 → 
    M = 1/5 → 
    Vbelow / Vabove = 4 / 11 :=
sorry

end volume_ratio_of_cube_cut_l157_157421


namespace transformed_average_l157_157021

theorem transformed_average (n : ℕ) (original_average factor : ℝ) 
  (h1 : n = 15) (h2 : original_average = 21.5) (h3 : factor = 7) :
  (original_average * factor) = 150.5 :=
by
  sorry

end transformed_average_l157_157021


namespace dandelion_seed_production_l157_157412

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end dandelion_seed_production_l157_157412


namespace emails_left_are_correct_l157_157076

-- Define the initial conditions for the problem
def initial_emails : ℕ := 400
def trash_emails : ℕ := initial_emails / 2
def remaining_after_trash : ℕ := initial_emails - trash_emails
def work_emails : ℕ := (remaining_after_trash * 40) / 100

-- Define the final number of emails left in the inbox
def emails_left_in_inbox : ℕ := remaining_after_trash - work_emails

-- The proof goal
theorem emails_left_are_correct : emails_left_in_inbox = 120 :=
by 
    -- The computations are correct based on the conditions provided
    have h_trash : trash_emails = 200 := by rfl
    have h_remaining : remaining_after_trash = 200 := by rw [← h_trash, Nat.sub_eq_iff_eq_add (Nat.le_refl 200)]
    have h_work : work_emails = 80 := by 
        rw [← h_remaining, Nat.mul_div_cancel (Nat.le_refl 8000) (Nat.lt_of_sub_one_eq_zero (by refl), 4000)]
    show emails_left_in_inbox = 120 := by
        rw [emails_left_in_inbox, h_remaining, h_work, Nat.sub_eq_iff_eq_add (Nat.le_refl 80)]
        exact rfl

end emails_left_are_correct_l157_157076


namespace region_in_plane_l157_157351

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem region_in_plane (x y : ℝ) :
  (f x + f y ≤ 0) ∧ (f x - f y ≥ 0) ↔
  ((x - 3)^2 + (y - 3)^2 ≤ 8) ∧ ((x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6)) :=
by
  sorry

end region_in_plane_l157_157351


namespace compound_interest_years_l157_157866

-- Define the parameters
def principal : ℝ := 7500
def future_value : ℝ := 8112
def annual_rate : ℝ := 0.04
def compounding_periods : ℕ := 1

-- Define the proof statement
theorem compound_interest_years :
  ∃ t : ℕ, future_value = principal * (1 + annual_rate / compounding_periods) ^ t ∧ t = 2 :=
by
  sorry

end compound_interest_years_l157_157866


namespace fraction_addition_simplest_form_l157_157778

theorem fraction_addition_simplest_form :
  (7 / 12) + (3 / 8) = 23 / 24 :=
by
  -- Adding a sorry to skip the proof
  sorry

end fraction_addition_simplest_form_l157_157778


namespace sum_squares_l157_157724

theorem sum_squares (w x y z : ℝ) (h1 : w + x + y + z = 0) (h2 : w^2 + x^2 + y^2 + z^2 = 1) :
  -1 ≤ w * x + x * y + y * z + z * w ∧ w * x + x * y + y * z + z * w ≤ 0 := 
by 
  sorry

end sum_squares_l157_157724


namespace smallest_divisible_by_15_18_20_is_180_l157_157642

theorem smallest_divisible_by_15_18_20_is_180 :
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (20 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (20 ∣ m)) → n ≤ m ∧ n = 180 := by
  sorry

end smallest_divisible_by_15_18_20_is_180_l157_157642


namespace range_of_a_l157_157805

noncomputable def e := Real.exp 1

theorem range_of_a (a : Real) 
  (h : ∀ x : Real, 1 ≤ x ∧ x ≤ 2 → Real.exp x - a ≥ 0) : 
  a ≤ e :=
by
  sorry

end range_of_a_l157_157805


namespace median_and_mode_l157_157771

theorem median_and_mode (data : List ℝ) (h : data = [6, 7, 4, 7, 5, 2]) :
  ∃ median mode, median = 5.5 ∧ mode = 7 := 
by {
  sorry
}

end median_and_mode_l157_157771


namespace range_of_a_l157_157057

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : (A ∩ B a).Nonempty) : a > 1 :=
sorry

end range_of_a_l157_157057


namespace neg_cos_ge_a_l157_157503

theorem neg_cos_ge_a (a : ℝ) : (¬ ∃ x : ℝ, Real.cos x ≥ a) ↔ a = 2 := 
sorry

end neg_cos_ge_a_l157_157503


namespace tea_blend_ratio_l157_157453

theorem tea_blend_ratio (x y : ℝ)
  (h1 : 18 * x + 20 * y = (21 * (x + y)) / 1.12)
  (h2 : x + y ≠ 0) :
  x / y = 5 / 3 :=
by
  -- proof will go here
  sorry

end tea_blend_ratio_l157_157453


namespace geo_seq_4th_term_l157_157564

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end geo_seq_4th_term_l157_157564


namespace dean_ordered_two_pizzas_l157_157152

variable (P : ℕ)

-- Each large pizza is cut into 12 slices
def slices_per_pizza := 12

-- Dean ate half of the Hawaiian pizza
def dean_slices := slices_per_pizza / 2

-- Frank ate 3 slices of Hawaiian pizza
def frank_slices := 3

-- Sammy ate a third of the cheese pizza
def sammy_slices := slices_per_pizza / 3

-- Total slices eaten plus slices left over equals total slices from pizzas
def total_slices_eaten := dean_slices + frank_slices + sammy_slices
def slices_left_over := 11
def total_pizza_slices := total_slices_eaten + slices_left_over

-- Total pizzas ordered is the total slices divided by slices per pizza
def pizzas_ordered := total_pizza_slices / slices_per_pizza

-- Prove that Dean ordered 2 large pizzas
theorem dean_ordered_two_pizzas : pizzas_ordered = 2 := by
  -- Proof omitted, add your proof here
  sorry

end dean_ordered_two_pizzas_l157_157152


namespace compare_negative_fractions_l157_157320

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l157_157320


namespace functions_same_function_C_functions_same_function_D_l157_157100

theorem functions_same_function_C (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by sorry

theorem functions_same_function_D (x : ℝ) : x = (x^3)^(1/3) :=
by sorry

end functions_same_function_C_functions_same_function_D_l157_157100


namespace subcommittee_count_l157_157751

theorem subcommittee_count :
  (Nat.choose 10 4) * (Nat.choose 8 3) = 11760 :=
by
  sorry

end subcommittee_count_l157_157751


namespace number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l157_157283

-- Definitions based on the conditions
def peopleO : ℕ := 28
def peopleA : ℕ := 7
def peopleB : ℕ := 9
def peopleAB : ℕ := 3

-- Proof for Question 1
theorem number_of_ways_to_select_one_person : peopleO + peopleA + peopleB + peopleAB = 47 := by
  sorry

-- Proof for Question 2
theorem number_of_ways_to_select_one_person_each_type : peopleO * peopleA * peopleB * peopleAB = 5292 := by
  sorry

end number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l157_157283


namespace smallest_disk_cover_count_l157_157740

theorem smallest_disk_cover_count (D : ℝ) (r : ℝ) (n : ℕ) 
  (hD : D = 1) (hr : r = 1 / 2) : n = 7 :=
by
  sorry

end smallest_disk_cover_count_l157_157740


namespace largest_angle_in_triangle_l157_157989

theorem largest_angle_in_triangle
    (a b c : ℝ)
    (h_sum_two_angles : a + b = (7 / 5) * 90)
    (h_angle_difference : b = a + 40) :
    max a (max b c) = 83 :=
by
  sorry

end largest_angle_in_triangle_l157_157989


namespace quadratic_sum_l157_157891

theorem quadratic_sum (a b c : ℝ) (h : ∀ x : ℝ, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) :
  a + b + c = -88 := by
  sorry

end quadratic_sum_l157_157891


namespace solve_for_x_l157_157202

theorem solve_for_x (x : ℝ) (h : 3 * (x - 5) = 3 * (18 - 5)) : x = 18 :=
by
  sorry

end solve_for_x_l157_157202


namespace intersection_of_A_and_B_l157_157972

def A := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := { x : ℝ | -1 < x ∧ x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

end intersection_of_A_and_B_l157_157972


namespace compare_neg_fractions_l157_157327

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l157_157327


namespace expr_C_always_positive_l157_157776

-- Define the expressions as Lean definitions
def expr_A (x : ℝ) : ℝ := x^2
def expr_B (x : ℝ) : ℝ := abs (-x + 1)
def expr_C (x : ℝ) : ℝ := (-x)^2 + 2
def expr_D (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem expr_C_always_positive : ∀ (x : ℝ), expr_C x > 0 :=
by
  sorry

end expr_C_always_positive_l157_157776


namespace degree_to_radian_conversion_l157_157151

theorem degree_to_radian_conversion : (-330 : ℝ) * (π / 180) = -(11 * π / 6) :=
by 
  sorry

end degree_to_radian_conversion_l157_157151


namespace parallel_lines_perpendicular_lines_l157_157030

-- Define the lines
def l₁ (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- The first proof statement: lines l₁ and l₂ are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → (a * (a - 1) - 2 = 0)) → (a = 2 ∨ a = -1) :=
by
  sorry

-- The second proof statement: lines l₁ and l₂ are perpendicular
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ((a - 1) * 1 + 2 * a = 0)) → (a = -1 / 3) :=
by
  sorry

end parallel_lines_perpendicular_lines_l157_157030


namespace geometric_seq_fourth_term_l157_157566

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end geometric_seq_fourth_term_l157_157566


namespace solution_set_of_inequality_l157_157253

theorem solution_set_of_inequality (x : ℝ) : (x * (2 - x) ≤ 0) ↔ (x ≤ 0 ∨ x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l157_157253


namespace expression_is_perfect_cube_l157_157705

theorem expression_is_perfect_cube {x y z : ℝ} (h : x + y + z = 0) :
  ∃ m : ℝ, 
    (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) * 
    (x^3 * y * z + x * y^3 * z + x * y * z^3) *
    (x^3 * y^2 * z + x^3 * y * z^2 + x^2 * y^3 * z + x * y^3 * z^2 + x^2 * y * z^3 + x * y^2 * z^3) =
    m ^ 3 := 
by 
  sorry

end expression_is_perfect_cube_l157_157705


namespace interest_difference_l157_157252

noncomputable def principal : ℝ := 6200
noncomputable def rate : ℝ := 5 / 100
noncomputable def time : ℝ := 10

noncomputable def interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem interest_difference :
  (principal - interest principal rate time) = 3100 := 
by
  sorry

end interest_difference_l157_157252


namespace reflection_across_x_axis_l157_157921

theorem reflection_across_x_axis (x y : ℝ) : (x, -y) = (-2, 3) ↔ (x, y) = (-2, -3) :=
by sorry

end reflection_across_x_axis_l157_157921


namespace find_higher_interest_rate_l157_157227

-- Definitions and conditions based on the problem
def total_investment : ℕ := 4725
def higher_rate_investment : ℕ := 1925
def lower_rate_investment : ℕ := total_investment - higher_rate_investment
def lower_rate : ℝ := 0.08
def higher_to_lower_interest_ratio : ℝ := 2

-- The main theorem to prove the higher interest rate
theorem find_higher_interest_rate (r : ℝ) (h1 : higher_rate_investment = 1925) (h2 : lower_rate_investment = 2800) :
  1925 * r = 2 * (2800 * 0.08) → r = 448 / 1925 :=
sorry

end find_higher_interest_rate_l157_157227


namespace chromium_percentage_new_alloy_l157_157381

theorem chromium_percentage_new_alloy :
  let wA := 15
  let pA := 0.12
  let wB := 30
  let pB := 0.08
  let wC := 20
  let pC := 0.20
  let wD := 35
  let pD := 0.05
  let total_weight := wA + wB + wC + wD
  let total_chromium := (wA * pA) + (wB * pB) + (wC * pC) + (wD * pD)
  total_weight = 100 ∧ total_chromium = 9.95 → total_chromium / total_weight * 100 = 9.95 :=
by
  sorry

end chromium_percentage_new_alloy_l157_157381


namespace area_of_second_side_l157_157876

theorem area_of_second_side 
  (L W H : ℝ) 
  (h1 : L * H = 120) 
  (h2 : L * W = 60) 
  (h3 : L * W * H = 720) : 
  W * H = 72 :=
sorry

end area_of_second_side_l157_157876


namespace exists_multiple_digits_0_1_l157_157551

theorem exists_multiple_digits_0_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k ≤ n) ∧ (∃ m : ℕ, m * n = k) ∧ (∀ d : ℕ, ∃ i : ℕ, i ≤ n ∧ d = 0 ∨ d = 1) :=
sorry

end exists_multiple_digits_0_1_l157_157551


namespace arithmetic_sequence_terms_l157_157987

theorem arithmetic_sequence_terms (a : ℕ → ℕ) (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 34)
  (h2 : a n + a (n - 1) + a (n - 2) = 146)
  (h3 : n * (a 1 + a n) = 780) : n = 13 :=
sorry

end arithmetic_sequence_terms_l157_157987


namespace uncovered_area_is_8_l157_157240

-- Conditions
def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side_length : ℕ := 4

-- Theorem to prove
theorem uncovered_area_is_8
  (sh_height : ℕ := shoebox_height)
  (sh_width : ℕ := shoebox_width)
  (bl_length : ℕ := block_side_length)
  : sh_height * sh_width - bl_length * bl_length = 8 :=
by {
  -- Placeholder for proof; we are not proving it as per instructions.
  sorry
}

end uncovered_area_is_8_l157_157240


namespace sequences_converge_and_find_limits_l157_157998

theorem sequences_converge_and_find_limits (x y : ℕ → ℝ)
  (h1 : x 1 = 1)
  (h2 : y 1 = Real.sqrt 3)
  (h3 : ∀ n : ℕ, x (n + 1) * y (n + 1) = x n)
  (h4 : ∀ n : ℕ, x (n + 1)^2 + y n = 2) :
  ∃ (Lx Ly : ℝ), (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |x n - Lx| < ε) ∧ 
                  (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |y n - Ly| < ε) ∧ 
                  Lx = 0 ∧ 
                  Ly = 2 := 
sorry

end sequences_converge_and_find_limits_l157_157998


namespace distribute_positions_l157_157631

structure DistributionProblem :=
  (volunteer_positions : ℕ)
  (schools : ℕ)
  (min_positions : ℕ)
  (distinct_allocations : ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c)

noncomputable def count_ways (p : DistributionProblem) : ℕ :=
  if p.volunteer_positions = 7 ∧ p.schools = 3 ∧ p.min_positions = 1 then 6 else 0

theorem distribute_positions (p : DistributionProblem) :
  count_ways p = 6 :=
by
  sorry

end distribute_positions_l157_157631


namespace find_ratio_l157_157500

variable {R : Type} [LinearOrderedField R]

def f (x a b : R) : R := x^3 + a*x^2 + b*x - a^2 - 7*a

def condition1 (a b : R) : Prop := f 1 a b = 10

def condition2 (a b : R) : Prop :=
  let f' := fun x => 3*x^2 + 2*a*x + b
  f' 1 = 0

theorem find_ratio (a b : R) (h1 : condition1 a b) (h2 : condition2 a b) :
  a / b = -2 / 3 :=
  sorry

end find_ratio_l157_157500


namespace subcommittee_count_l157_157759

-- Define the conditions: number of Republicans and Democrats in the Senate committee
def numRepublicans : ℕ := 10
def numDemocrats : ℕ := 8
def chooseRepublicans : ℕ := 4
def chooseDemocrats : ℕ := 3

-- Define the main proof problem based on the conditions and the correct answer
theorem subcommittee_count :
  (Nat.choose numRepublicans chooseRepublicans) * (Nat.choose numDemocrats chooseDemocrats) = 11760 := by
  sorry

end subcommittee_count_l157_157759


namespace max_marks_l157_157436

theorem max_marks (M : ℝ) (h1 : 0.40 * M = 200) : M = 500 := by
  sorry

end max_marks_l157_157436


namespace nathaniel_wins_probability_l157_157845

def fair_six_sided_die : Type := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def probability_nathaniel_wins : ℚ :=
  have fair_die : fair_six_sided_die := sorry,
  have nathaniel_first : Prop := sorry,
  have win_condition (sum : ℕ) : Prop := sum % 7 = 0,

  if nathaniel_first ∧ ∀ sum. win_condition sum
  then 5 / 11
  else 0

theorem nathaniel_wins_probability :
  probability_nathaniel_wins = 5 / 11 :=
sorry

end nathaniel_wins_probability_l157_157845


namespace divisor_is_three_l157_157452

noncomputable def find_divisor (n : ℕ) (reduction : ℕ) (result : ℕ) : ℕ :=
  n / result

theorem divisor_is_three (x : ℝ) : 
  (original : ℝ) → (reduction : ℝ) → (new_result : ℝ) → 
  original = 45 → new_result = 45 - 30 → (original / x = new_result) → 
  x = 3 := by 
  intros original reduction new_result h1 h2 h3
  sorry

end divisor_is_three_l157_157452


namespace math_problem_common_factors_and_multiples_l157_157818

-- Definitions
def a : ℕ := 180
def b : ℕ := 300

-- The Lean statement to be proved
theorem math_problem_common_factors_and_multiples :
    Nat.lcm a b = 900 ∧
    Nat.gcd a b = 60 ∧
    {d | d ∣ a ∧ d ∣ b} = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} :=
by
  sorry

end math_problem_common_factors_and_multiples_l157_157818


namespace max_ab_value_l157_157182

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, exp x ≥ a * (x - 1) + b) : ab ≤ 1/2 * exp 3 :=
sorry

end max_ab_value_l157_157182


namespace Margarita_vs_Ricciana_l157_157228

-- Definitions based on the conditions.
def Ricciana_run : ℕ := 20
def Ricciana_jump : ℕ := 4
def Ricciana_total : ℕ := Ricciana_run + Ricciana_jump

def Margarita_run : ℕ := 18
def Margarita_jump : ℕ := 2 * Ricciana_jump - 1
def Margarita_total : ℕ := Margarita_run + Margarita_jump

-- The statement to be proved.
theorem Margarita_vs_Ricciana : (Margarita_total - Ricciana_total = 1) :=
by
  sorry

end Margarita_vs_Ricciana_l157_157228


namespace quadratic_inequality_solution_l157_157172

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) := sorry

end quadratic_inequality_solution_l157_157172


namespace at_least_half_team_B_can_serve_on_submarine_l157_157905

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l157_157905


namespace product_of_first_two_numbers_l157_157896

theorem product_of_first_two_numbers (A B C : ℕ) (h_coprime: Nat.gcd A B = 1 ∧ Nat.gcd B C = 1 ∧ Nat.gcd A C = 1)
  (h_product: B * C = 1073) (h_sum: A + B + C = 85) : A * B = 703 :=
sorry

end product_of_first_two_numbers_l157_157896


namespace smallest_divisible_by_15_18_20_is_180_l157_157641

theorem smallest_divisible_by_15_18_20_is_180 :
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (20 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (20 ∣ m)) → n ≤ m ∧ n = 180 := by
  sorry

end smallest_divisible_by_15_18_20_is_180_l157_157641


namespace first_valve_time_l157_157106

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end first_valve_time_l157_157106


namespace value_of_a_minus_n_plus_k_l157_157443

theorem value_of_a_minus_n_plus_k :
  ∃ (a k n : ℤ), 
    (∀ x : ℤ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) ∧ 
    (a - n + k = 3) :=
sorry

end value_of_a_minus_n_plus_k_l157_157443


namespace pentagonal_pyramid_faces_l157_157433

-- Definition of a pentagonal pyramid
structure PentagonalPyramid where
  base_sides : Nat := 5
  triangular_faces : Nat := 5

-- The goal is to prove that the total number of faces is 6
theorem pentagonal_pyramid_faces (P : PentagonalPyramid) : P.base_sides + 1 = 6 :=
  sorry

end pentagonal_pyramid_faces_l157_157433


namespace problem_solution_l157_157957

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * x^(x^2) = 59052 :=
by
  rw [h]
  -- The condition is now x = 3
  let t := 3 + 3 * 3^(3^2)
  have : t = 59052 := sorry
  exact this

end problem_solution_l157_157957


namespace expression_for_x_expression_for_y_l157_157534

variables {A B C : ℝ}

-- Conditions: A, B, and C are positive numbers with A > B > C > 0
axiom h1 : A > 0
axiom h2 : B > 0
axiom h3 : C > 0
axiom h4 : A > B
axiom h5 : B > C

-- A is x% greater than B
variables {x : ℝ}
axiom h6 : A = (1 + x / 100) * B

-- A is y% greater than C
variables {y : ℝ}
axiom h7 : A = (1 + y / 100) * C

-- Proving the expressions for x and y
theorem expression_for_x : x = 100 * ((A - B) / B) :=
sorry

theorem expression_for_y : y = 100 * ((A - C) / C) :=
sorry

end expression_for_x_expression_for_y_l157_157534


namespace find_a_value_l157_157370

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l157_157370


namespace remaining_soup_feeds_20_adults_l157_157931

theorem remaining_soup_feeds_20_adults (cans_of_soup : ℕ) (feed_4_adults : ℕ) (feed_7_children : ℕ) (initial_cans : ℕ) (children_fed : ℕ)
    (h1 : feed_4_adults = 4)
    (h2 : feed_7_children = 7)
    (h3 : initial_cans = 8)
    (h4 : children_fed = 21) : 
    (initial_cans - (children_fed / feed_7_children)) * feed_4_adults = 20 :=
by
  sorry

end remaining_soup_feeds_20_adults_l157_157931


namespace problem_l157_157180

theorem problem (x : ℕ) (h : 2^x + 2^x + 2^x = 256) : x * (x + 1) = 72 :=
sorry

end problem_l157_157180


namespace rain_probability_l157_157428

-- Define the probability of rain on any given day, number of trials, and specific number of successful outcomes.
def prob_rain_each_day : ℚ := 1/5
def num_days : ℕ := 10
def num_rainy_days : ℕ := 3

-- Define the binomial probability mass function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Statement to prove
theorem rain_probability : binomial_prob num_days num_rainy_days prob_rain_each_day = 1966080 / 9765625 :=
by
  sorry

end rain_probability_l157_157428


namespace compare_neg_fractions_l157_157325

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l157_157325


namespace coconuts_for_crab_l157_157695

theorem coconuts_for_crab (C : ℕ) (H1 : 6 * C * 19 = 342) : C = 3 :=
sorry

end coconuts_for_crab_l157_157695


namespace sequence_induction_l157_157527

theorem sequence_induction (a b : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : b 1 = 4)
  (h₃ : ∀ n : ℕ, 0 < n → 2 * b n = a n + a (n + 1))
  (h₄ : ∀ n : ℕ, 0 < n → (a (n + 1))^2 = b n * b (n + 1)) :
  (∀ n : ℕ, 0 < n → a n = n * (n + 1)) ∧ (∀ n : ℕ, 0 < n → b n = (n + 1)^2) :=
by
  sorry

end sequence_induction_l157_157527


namespace train_length_is_300_l157_157927

noncomputable def speed_kmph : Float := 90
noncomputable def speed_mps : Float := (speed_kmph * 1000) / 3600
noncomputable def time_sec : Float := 12
noncomputable def length_of_train : Float := speed_mps * time_sec

theorem train_length_is_300 : length_of_train = 300 := by
  sorry

end train_length_is_300_l157_157927


namespace compare_negative_fractions_l157_157322

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l157_157322


namespace team_B_elibility_l157_157907

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l157_157907


namespace pencil_pen_costs_l157_157420

noncomputable def cost_of_items (p q : ℝ) : ℝ := 4 * p + 4 * q

theorem pencil_pen_costs (p q : ℝ) (h1 : 6 * p + 3 * q = 5.40) (h2 : 3 * p + 5 * q = 4.80) : cost_of_items p q = 4.80 :=
by
  sorry

end pencil_pen_costs_l157_157420


namespace common_tangent_y_intercept_l157_157623

theorem common_tangent_y_intercept
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (m b : ℝ)
  (h_c1 : c1 = (5, -2))
  (h_c2 : c2 = (20, 6))
  (h_r1 : r1 = 5)
  (h_r2 : r2 = 12)
  (h_tangent : ∃m > 0, ∃b, (∀ x y, y = m * x + b → (x - 5)^2 + (y + 2)^2 > 25 ∧ (x - 20)^2 + (y - 6)^2 > 144)) :
  b = -2100 / 161 :=
by
  sorry

end common_tangent_y_intercept_l157_157623


namespace carol_total_points_l157_157622

/-- Conditions -/
def first_round_points : ℤ := 17
def second_round_points : ℤ := 6
def last_round_points : ℤ := -16

/-- Proof problem statement -/
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l157_157622


namespace number_of_dvds_remaining_l157_157153

def initial_dvds : ℕ := 850

def week1_rented : ℕ := (initial_dvds * 25) / 100
def week1_sold : ℕ := 15
def remaining_after_week1 : ℕ := initial_dvds - week1_rented - week1_sold

def week2_rented : ℕ := (remaining_after_week1 * 35) / 100
def week2_sold : ℕ := 25
def remaining_after_week2 : ℕ := remaining_after_week1 - week2_rented - week2_sold

def week3_rented : ℕ := (remaining_after_week2 * 50) / 100
def week3_sold : ℕ := (remaining_after_week2 - week3_rented) * 5 / 100
def remaining_after_week3 : ℕ := remaining_after_week2 - week3_rented - week3_sold

theorem number_of_dvds_remaining : remaining_after_week3 = 181 :=
by
  -- proof goes here
  sorry

end number_of_dvds_remaining_l157_157153


namespace compare_fractions_l157_157311

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l157_157311


namespace negative_fraction_comparison_l157_157308

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l157_157308


namespace tan_difference_identity_l157_157197

theorem tan_difference_identity {α : ℝ} (h : Real.tan α = 4 * Real.sin (7 * Real.pi / 3)) :
  Real.tan (α - Real.pi / 3) = Real.sqrt 3 / 7 := 
sorry

end tan_difference_identity_l157_157197


namespace arithmetic_sequence_a10_l157_157526

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 7 = 9) (h2 : a 13 = -3) 
  (ha : ∀ n, a n = a1 + (n - 1) * d) :
  a 10 = 3 :=
by sorry

end arithmetic_sequence_a10_l157_157526


namespace sum_arithmetic_series_l157_157014

theorem sum_arithmetic_series :
  let a := -42
  let d := 2
  let l := 0
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = -462 := by
sorry

end sum_arithmetic_series_l157_157014


namespace total_students_l157_157618

theorem total_students
  (T : ℝ) 
  (h1 : 0.20 * T = 168)
  (h2 : 0.30 * T = 252) : T = 840 :=
sorry

end total_students_l157_157618


namespace poll_total_l157_157298

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l157_157298


namespace find_focus_parabola_l157_157799

theorem find_focus_parabola
  (x y : ℝ) 
  (h₁ : y = 9 * x^2 + 6 * x - 4) :
  ∃ (h k p : ℝ), (x + 1/3)^2 = 1/3 * (y + 5) ∧ 4 * p = 1/3 ∧ h = -1/3 ∧ k = -5 ∧ (h, k + p) = (-1/3, -59/12) :=
sorry

end find_focus_parabola_l157_157799


namespace magnitude_diff_l157_157488

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition_1 : ‖a‖ = 2 := sorry
def condition_2 : ‖b‖ = 2 := sorry
def condition_3 : ‖a + b‖ = Real.sqrt 7 := sorry

-- Proof statement
theorem magnitude_diff (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a + b‖ = Real.sqrt 7) : 
  ‖a - b‖ = 3 :=
sorry

end magnitude_diff_l157_157488


namespace find_a_plus_c_l157_157536

def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_a_plus_c (a b c d : ℝ) 
  (h_vertex_f : -a / 2 = v) (h_vertex_g : -c / 2 = w)
  (h_root_v_g : g v c d = 0) (h_root_w_f : f w a b = 0)
  (h_intersect : f 50 a b = -200 ∧ g 50 c d = -200)
  (h_min_value_f : ∀ x, f (-a / 2) a b ≤ f x a b)
  (h_min_value_g : ∀ x, g (-c / 2) c d ≤ g x c d)
  (h_min_difference : f (-a / 2) a b = g (-c / 2) c d - 50) :
  a + c = sorry :=
sorry

end find_a_plus_c_l157_157536


namespace major_axis_length_l157_157386

-- Define the problem setup
structure Cylinder :=
  (base_radius : ℝ)
  (height : ℝ)

structure Sphere :=
  (radius : ℝ)

-- Define the conditions
def cylinder : Cylinder :=
  { base_radius := 6, height := 0 }  -- height isn't significant for this problem

def sphere1 : Sphere :=
  { radius := 6 }

def sphere2 : Sphere :=
  { radius := 6 }

def distance_between_centers : ℝ :=
  13

-- Statement of the problem in Lean 4
theorem major_axis_length : 
  cylinder.base_radius = 6 →
  sphere1.radius = 6 →
  sphere2.radius = 6 →
  distance_between_centers = 13 →
  ∃ major_axis_length : ℝ, major_axis_length = 13 :=
by
  intros h1 h2 h3 h4
  existsi 13
  sorry

end major_axis_length_l157_157386


namespace man_older_than_son_by_46_l157_157935

-- Given conditions about the ages
def sonAge : ℕ := 44

def manAge_in_two_years (M : ℕ) : Prop := M + 2 = 2 * (sonAge + 2)

-- The problem to verify
theorem man_older_than_son_by_46 (M : ℕ) (h : manAge_in_two_years M) : M - sonAge = 46 :=
by
  sorry

end man_older_than_son_by_46_l157_157935


namespace ice_cream_flavors_l157_157036

-- We have four basic flavors and want to combine four scoops from these flavors.
def ice_cream_combinations : ℕ :=
  Nat.choose 7 3

theorem ice_cream_flavors : ice_cream_combinations = 35 :=
by
  sorry

end ice_cream_flavors_l157_157036


namespace triangle_find_C_angle_triangle_find_perimeter_l157_157350

variable (A B C a b c : ℝ)

theorem triangle_find_C_angle
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c) :
  C = π / 3 :=
sorry

theorem triangle_find_perimeter
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h2 : c = Real.sqrt 7)
  (h3 : a * b = 6) :
  a + b + c = 5 + Real.sqrt 7 :=
sorry

end triangle_find_C_angle_triangle_find_perimeter_l157_157350


namespace tangent_line_equation_l157_157883

theorem tangent_line_equation :
  ∀ (x : ℝ) (y : ℝ), y = 4 * x - x^3 → 
  (x = -1) → (y = -3) →
  (∀ (m : ℝ), m = 4 - 3 * (-1)^2) →
  ∃ (line_eq : ℝ → ℝ), (∀ x, line_eq x = x - 2) :=
by
  sorry

end tangent_line_equation_l157_157883


namespace work_days_of_b_l157_157272

theorem work_days_of_b (d : ℕ) 
  (A B C : ℕ)
  (h_ratioA : A = (3 * 115) / 5)
  (h_ratioB : B = (4 * 115) / 5)
  (h_C : C = 115)
  (h_total_wages : 1702 = (A * 6) + (B * d) + (C * 4)) :
  d = 9 := 
sorry

end work_days_of_b_l157_157272


namespace problem_l157_157385

noncomputable def a_seq (n : ℕ) : ℚ := sorry

def is_geometric_sequence (seq : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = q * seq n

theorem problem (h_positive : ∀ n : ℕ, 0 < a_seq n)
                (h_ratio : ∀ n : ℕ, 2 * a_seq n = 3 * a_seq (n + 1))
                (h_product : a_seq 1 * a_seq 4 = 8 / 27) :
  is_geometric_sequence a_seq (2 / 3) ∧ 
  (∃ n : ℕ, a_seq n = 16 / 81 ∧ n = 6) :=
by
  sorry

end problem_l157_157385


namespace total_boxes_l157_157731

theorem total_boxes (initial_empty_boxes : ℕ) (boxes_added_per_operation : ℕ) (total_operations : ℕ) (final_non_empty_boxes : ℕ):
  initial_empty_boxes = 2013 →
  boxes_added_per_operation = 13 →
  final_non_empty_boxes = 2013 →
  total_operations = final_non_empty_boxes →
  initial_empty_boxes + boxes_added_per_operation * total_operations = 28182 :=
by
  intros h_initial h_boxes_added h_final_non_empty h_total_operations
  rw [h_initial, h_boxes_added, h_final_non_empty, h_total_operations]
  calc
    2013 + 13 * 2013 = 2013 * (1 + 13) : by ring
    ... = 2013 * 14 : by norm_num
    ... = 28182 : by norm_num

end total_boxes_l157_157731


namespace absolute_slope_of_dividing_line_l157_157732
   
noncomputable def circles := 
  [(10, 90), (15, 70), (20, 80)]

def radius := 4

def is_equally_divided_by_line (L : ℝ → ℝ) (C : list (ℝ × ℝ)) (r : ℝ) : Prop :=
  -- Define a condition that the line L splits the total area of circles 
  -- C into equal parts
  sorry

def line (m : ℝ) (x : ℝ): ℝ := 
  m * x -- A placeholder for line equation definition

theorem absolute_slope_of_dividing_line :
  ∃ m : ℝ, is_equally_divided_by_line (line m) circles radius ∧ |m| = 1 :=
begin
  sorry
end

end absolute_slope_of_dividing_line_l157_157732


namespace graph_intersect_x_axis_exactly_once_l157_157674

theorem graph_intersect_x_axis_exactly_once (a : ℝ) :
    (∀ x : ℝ, (a-1) * x^2 - 4 * x + 2 * a = 0 → x = -(1/2)) ∨ -- Quadratic condition with one real root giving unique intersection
    ((a-1) = 0 ∧ ∃ x : ℝ, -4 * x + 2 * a = 0) -- Linear condition giving unique intersection
    ↔ a = -1 ∨ a = 2 ∨ a = 1 :=
by
    sorry

end graph_intersect_x_axis_exactly_once_l157_157674


namespace ratio_of_lengths_l157_157604

variables (l1 l2 l3 : ℝ)

theorem ratio_of_lengths (h1 : l2 = (1/2) * (l1 + l3))
                         (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
  (l1 / l3) = (7 / 5) :=
by
  sorry

end ratio_of_lengths_l157_157604


namespace correct_sum_of_integers_l157_157542

theorem correct_sum_of_integers
  (x y : ℕ)
  (h1 : x - y = 5)
  (h2 : x * y = 84) :
  x + y = 19 :=
sorry

end correct_sum_of_integers_l157_157542


namespace average_minutes_per_day_l157_157620

theorem average_minutes_per_day
  (f : ℕ) -- Number of fifth graders
  (third_grade_minutes : ℕ := 10)
  (fourth_grade_minutes : ℕ := 18)
  (fifth_grade_minutes : ℕ := 12)
  (third_grade_students : ℕ := 3 * f)
  (fourth_grade_students : ℕ := (3 / 2) * f) -- Assumed to work with integer or rational numbers
  (fifth_grade_students : ℕ := f)
  (total_minutes_third_grade : ℕ := third_grade_minutes * third_grade_students)
  (total_minutes_fourth_grade : ℕ := fourth_grade_minutes * fourth_grade_students)
  (total_minutes_fifth_grade : ℕ := fifth_grade_minutes * fifth_grade_students)
  (total_minutes : ℕ := total_minutes_third_grade + total_minutes_fourth_grade + total_minutes_fifth_grade)
  (total_students : ℕ := third_grade_students + fourth_grade_students + fifth_grade_students) :
  (total_minutes / total_students : ℝ) = 12.55 :=
by
  sorry

end average_minutes_per_day_l157_157620


namespace sum_of_integers_is_34_l157_157048

theorem sum_of_integers_is_34 (a b : ℕ) (h1 : a - b = 6) (h2 : a * b = 272) (h3a : a > 0) (h3b : b > 0) : a + b = 34 :=
  sorry

end sum_of_integers_is_34_l157_157048


namespace natural_numbers_satisfying_conditions_l157_157338

variable (a b : ℕ)

theorem natural_numbers_satisfying_conditions :
  (90 < a + b ∧ a + b < 100) ∧ (0.9 < (a : ℝ) / b ∧ (a : ℝ) / b < 0.91) ↔ (a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52) := by
  sorry

end natural_numbers_satisfying_conditions_l157_157338


namespace positive_difference_of_squares_l157_157089

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 70) (h2 : a - b = 20) : a^2 - b^2 = 1400 :=
by
sorry

end positive_difference_of_squares_l157_157089


namespace julias_preferred_number_l157_157686

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem julias_preferred_number : ∃ n : ℕ, n > 100 ∧ n < 200 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 104 :=
by
  sorry

end julias_preferred_number_l157_157686


namespace max_balls_drawn_l157_157856

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l157_157856


namespace modulus_of_z_l157_157496

-- Define the complex number z
def z : ℂ := -5 + 12 * Complex.I

-- Define a theorem stating the modulus of z is 13
theorem modulus_of_z : Complex.abs z = 13 :=
by
  -- This will be the place to provide proof steps
  sorry

end modulus_of_z_l157_157496


namespace max_value_f_l157_157081

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_f :
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ 1 :=
sorry

end max_value_f_l157_157081


namespace maximum_and_minimum_values_l157_157188

noncomputable def f (p q x : ℝ) : ℝ := x^3 - p * x^2 - q * x

theorem maximum_and_minimum_values
  (p q : ℝ)
  (h1 : f p q 1 = 0)
  (h2 : (deriv (f p q)) 1 = 0) :
  ∃ (max_val min_val : ℝ), max_val = 4 / 27 ∧ min_val = 0 := 
by {
  sorry
}

end maximum_and_minimum_values_l157_157188


namespace probability_odd_and_multiple_of_5_l157_157576

/-- Given three distinct integers selected at random between 1 and 2000, inclusive, the probability that the product of the three integers is odd and a multiple of 5 is between 0.01 and 0.05. -/
theorem probability_odd_and_multiple_of_5 :
  ∃ p : ℚ, (0.01 < p ∧ p < 0.05) :=
sorry

end probability_odd_and_multiple_of_5_l157_157576


namespace range_of_a_l157_157502

noncomputable def f (x : ℝ) : ℝ := x + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - a / x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem range_of_a (e : ℝ) (a : ℝ) (H : ∀ x ∈ Set.Icc 1 e, f x ≥ g x a) :
  -2 ≤ a ∧ a ≤ (2 * e) / (e - 1) :=
by
  sorry

end range_of_a_l157_157502


namespace billy_has_62_crayons_l157_157148

noncomputable def billy_crayons (total_crayons : ℝ) (jane_crayons : ℝ) : ℝ :=
  total_crayons - jane_crayons

theorem billy_has_62_crayons : billy_crayons 114 52.0 = 62 := by
  sorry

end billy_has_62_crayons_l157_157148


namespace compare_fractions_l157_157315

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l157_157315


namespace sum_of_terms_l157_157525

theorem sum_of_terms (a d : ℕ) (h1 : a + d < a + 2 * d)
  (h2 : (a + d) * (a + 20) = (a + 2 * d) ^ 2)
  (h3 : a + 20 - a = 20) :
  a + (a + d) + (a + 2 * d) + (a + 20) = 46 :=
by
  sorry

end sum_of_terms_l157_157525


namespace closest_point_on_ellipse_l157_157161

theorem closest_point_on_ellipse : 
  ∃ (x y : ℝ), (7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0) ∧ 
  (∀ (x' y' : ℝ), 7 * x'^2 + 4 * y'^2 = 28 → dist (x, y) (0, 0) ≤ dist (x', y') (0, 0)) :=
sorry

end closest_point_on_ellipse_l157_157161


namespace find_c_l157_157812

-- Define the quadratic polynomial with given conditions
def quadratic (b c x y : ℝ) : Prop :=
  y = x^2 + b * x + c

-- Define the condition that the polynomial passes through two particular points
def passes_through_points (b c : ℝ) : Prop :=
  (quadratic b c 1 4) ∧ (quadratic b c 5 4)

-- The theorem stating c is 9 given the conditions
theorem find_c (b c : ℝ) (h : passes_through_points b c) : c = 9 :=
by {
  sorry
}

end find_c_l157_157812


namespace inequality_l157_157250

-- Definition of the given condition
def condition (a b c : ℝ) : Prop :=
  a^2 * b * c + a * b^2 * c + a * b * c^2 = 1

-- Theorem to prove the inequality
theorem inequality (a b c : ℝ) (h : condition a b c) : a^2 + b^2 + c^2 ≥ real.sqrt 3 :=
sorry

end inequality_l157_157250


namespace floor_eq_l157_157796

theorem floor_eq (r : ℝ) (h : ⌊r⌋ + r = 12.4) : r = 6.4 := by
  sorry

end floor_eq_l157_157796


namespace num_integer_solutions_abs_eq_3_l157_157425

theorem num_integer_solutions_abs_eq_3 :
  (∀ (x y : ℤ), (|x| + |y| = 3) → 
  ∃ (s : Finset (ℤ × ℤ)), s.card = 12 ∧ (∀ (a b : ℤ), (a, b) ∈ s ↔ (|a| + |b| = 3))) :=
by
  sorry

end num_integer_solutions_abs_eq_3_l157_157425


namespace a0_a2_a4_sum_l157_157980

theorem a0_a2_a4_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 5 = a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5) →
  a0 + a2 + a4 = -121 :=
by
  intros h
  sorry

end a0_a2_a4_sum_l157_157980


namespace complex_roots_eqn_l157_157874

open Complex

theorem complex_roots_eqn (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) 
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I := 
sorry

end complex_roots_eqn_l157_157874


namespace find_b_l157_157129

theorem find_b (a b : ℝ) (h1 : 2 * a + b = 6) (h2 : -2 * a + b = 2) : b = 4 :=
sorry

end find_b_l157_157129


namespace standard_deviation_is_two_l157_157864

def weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum / l.length)

noncomputable def variance (l : List ℝ) : ℝ :=
  ((l.map (λ x => (x - mean l)^2)).sum / l.length)

noncomputable def standard_deviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_two : standard_deviation weights = 2 := 
by
  sorry

end standard_deviation_is_two_l157_157864


namespace correct_calculation_result_l157_157936

theorem correct_calculation_result :
  ∃ x : ℕ, 6 * x = 42 ∧ 3 * x = 21 :=
by
  sorry

end correct_calculation_result_l157_157936


namespace nice_subset_unique_l157_157482

-- Define the problem statement in Lean 4
theorem nice_subset_unique (n : ℕ) (S : set (fin n)) :
  (∀ k ∈ S, ∀ (distribution : finset ((fin n) × (fin n))),
    (∀ t ∈ distribution, ∃ (group : finset (fin n)) (t ≥ group.card),
        group.subset distribution.image.2) →
      (∀ (k : ℕ), k ∈ finset.range n → ∀ (kids : finset (fin n)),
        kids.card = k → ∃ (group : finset (fin n)), group.card ≥ k ∧ group ⊆ kids)) ↔
  S = set.univ := sorry

end nice_subset_unique_l157_157482


namespace unique_solution_a_l157_157192

theorem unique_solution_a (a : ℚ) : 
  (∃ x : ℚ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0 ∧ 
  ∀ y : ℚ, (y ≠ x → (a^2 - 1) * y^2 + (a + 1) * y + 1 ≠ 0)) ↔ a = 1 ∨ a = 5/3 := 
sorry

end unique_solution_a_l157_157192


namespace sarah_bought_new_shirts_l157_157072

-- Define the given conditions
def original_shirts : ℕ := 9
def total_shirts : ℕ := 17

-- The proof statement: Prove that the number of new shirts is 8
theorem sarah_bought_new_shirts : total_shirts - original_shirts = 8 := by
  sorry

end sarah_bought_new_shirts_l157_157072


namespace distribute_objects_l157_157347

theorem distribute_objects (n r : ℕ) (h : n ≤ r) :
  ∃ ways : ℕ, ways = Nat.choose (r - 1) (n - 1) ∧ ways = ways :=
by
  sorry

end distribute_objects_l157_157347


namespace subcommittee_count_l157_157749

theorem subcommittee_count :
  let nR := 10 in 
  let nD := 8 in 
  let kR := 4 in 
  let kD := 3 in 
  (nat.choose nR kR) * (nat.choose nD kD) = 11760 := 
by 
  let nR := 10
  let nD := 8
  let kR := 4
  let kD := 3
  -- sorry replaces the actual proof steps
  sorry

end subcommittee_count_l157_157749


namespace equal_lengths_l157_157004

noncomputable def F (x y z : ℝ) := (x+y+z) * (x+y-z) * (y+z-x) * (x+z-y)

variables {a b c d e f : ℝ}

axiom acute_angled_triangle (x y z : ℝ) : Prop

axiom altitudes_sum_greater (x y z : ℝ) : Prop

axiom cond1 : acute_angled_triangle a b c
axiom cond2 : acute_angled_triangle b d f
axiom cond3 : acute_angled_triangle a e f
axiom cond4 : acute_angled_triangle e c d

axiom cond5 : altitudes_sum_greater a b c
axiom cond6 : altitudes_sum_greater b d f
axiom cond7 : altitudes_sum_greater a e f
axiom cond8 : altitudes_sum_greater e c d

axiom cond9 : F a b c = F b d f
axiom cond10 : F a e f = F e c d

theorem equal_lengths : a = d ∧ b = e ∧ c = f := by
  sorry -- Proof not required.

end equal_lengths_l157_157004


namespace correct_answer_B_l157_157102

def point_slope_form (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x - 2)

def proposition_2 (k : ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℝ, @point_slope_form k x y

def proposition_3 (k : ℝ) : Prop := point_slope_form k 2 (-1)

def proposition_4 (k : ℝ) : Prop := k ≠ 0

theorem correct_answer_B : 
  (∃ k : ℝ, @point_slope_form k 2 (-1)) ∧ 
  (∀ k : ℝ, @point_slope_form k 2 (-1)) ∧
  (∀ k : ℝ, k ≠ 0) → true := 
by
  intro h
  sorry

end correct_answer_B_l157_157102


namespace sin_cos_of_theta_l157_157965

open Real

theorem sin_cos_of_theta (θ : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4))
  (hxθ : ∃ r, r > 0 ∧ P = (r * cos θ, r * sin θ)) :
  sin θ + cos θ = 1 / 5 := 
by
  sorry

end sin_cos_of_theta_l157_157965


namespace solve_cyclic_quadrilateral_area_l157_157020

noncomputable def cyclic_quadrilateral_area (AB BC AD CD : ℝ) (cyclic : Bool) : ℝ :=
  if cyclic ∧ AB = 2 ∧ BC = 6 ∧ AD = 4 ∧ CD = 4 then 8 * Real.sqrt 3 else 0

theorem solve_cyclic_quadrilateral_area :
  cyclic_quadrilateral_area 2 6 4 4 true = 8 * Real.sqrt 3 :=
by
  sorry

end solve_cyclic_quadrilateral_area_l157_157020


namespace compute_fraction_sum_l157_157393

-- Define the equation whose roots are a, b, c
def cubic_eq (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x = 12

-- State the main theorem
theorem compute_fraction_sum 
  (a b c : ℝ) 
  (ha : cubic_eq a) 
  (hb : cubic_eq b) 
  (hc : cubic_eq c) :
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  ∃ (r : ℝ), r = -23/12 ∧ (ab/c + bc/a + ca/b) = r := 
  sorry

end compute_fraction_sum_l157_157393


namespace probability_at_least_one_even_l157_157435

theorem probability_at_least_one_even :
  let usable_digits := {0, 3, 5, 7, 8, 9}
  let even_digits := {0, 8}
  let code_length := 4
  let num_usable_digits := 6
  let num_usable_odd_digits := 4
  (1 - ((num_usable_odd_digits ^ code_length) / (num_usable_digits ^ code_length))) = (65 / 81) := by
  sorry

end probability_at_least_one_even_l157_157435


namespace decrement_value_is_15_l157_157082

noncomputable def decrement_value (n : ℕ) (original_mean updated_mean : ℕ) : ℕ :=
  (n * original_mean - n * updated_mean) / n

theorem decrement_value_is_15 : decrement_value 50 200 185 = 15 :=
by
  sorry

end decrement_value_is_15_l157_157082


namespace find_circle_parameter_l157_157170

theorem find_circle_parameter (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y + c = 0 ∧ ((x + 4)^2 + (y - 1)^2 = 25)) → c = -8 :=
by
  sorry

end find_circle_parameter_l157_157170


namespace lens_discount_l157_157068

theorem lens_discount :
  ∃ (P : ℚ), ∀ (D : ℚ),
    (300 - D = 240) →
    (P = (D / 300) * 100) →
    P = 20 :=
by
  sorry

end lens_discount_l157_157068


namespace min_f_on_interval_l157_157165

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_f_on_interval : 
  ∀ x, 0 < x ∧ x < π / 2 → f x ≥ 3 + 2 * sqrt 2 :=
sorry

end min_f_on_interval_l157_157165


namespace max_balls_drawn_l157_157857

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l157_157857


namespace unique_abc_solution_l157_157477

theorem unique_abc_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
    (h4 : a^4 + b^2 * c^2 = 16 * a) (h5 : b^4 + c^2 * a^2 = 16 * b) (h6 : c^4 + a^2 * b^2 = 16 * c) : 
    (a, b, c) = (2, 2, 2) :=
  by
    sorry

end unique_abc_solution_l157_157477


namespace student_correct_answers_l157_157108

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 70) : C = 90 :=
sorry

end student_correct_answers_l157_157108


namespace five_more_limbs_l157_157461

-- Definition of the number of limbs an alien has
def alien_limbs : ℕ := 3 + 8

-- Definition of the number of limbs a Martian has
def martian_limbs : ℕ := (8 / 2) + (3 * 2)

-- The main statement that we need to prove
theorem five_more_limbs : 5 * alien_limbs - 5 * martian_limbs = 5 := by
  have h1 : alien_limbs = 11 := rfl
  have h2 : martian_limbs = 10 := rfl
  calc
    5 * alien_limbs - 5 * martian_limbs
        = 5 * 11 - 5 * 10 := by rw [h1, h2]
    ... = 55 - 50     := by rfl
    ... = 5           := by rfl

end five_more_limbs_l157_157461


namespace range_of_a_l157_157671

def f (a x : ℝ) : ℝ := a * x^2 - 2 * x - |x^2 - a * x + 1|

theorem range_of_a (a : ℝ) :
  (f a 0 = 0 → (a ∈ (-∞, 0) ∪ (0, 1) ∪ (1, +∞))) :=
sorry

end range_of_a_l157_157671


namespace solve_system_equations_l157_157871

noncomputable def system_equations : Prop :=
  ∃ x y : ℝ,
    (8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0) ∧
    (8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0) ∧
    ((x = 0 ∧ y = 4) ∨ (x = -7.5 ∧ y = 1) ∨ (x = -4.5 ∧ y = 0))

theorem solve_system_equations : system_equations := 
by
  sorry

end solve_system_equations_l157_157871


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l157_157029

variable (U A B : Set ℝ)
variable (x : ℝ)

def universal_set := { x | x ≤ 4 }
def set_A := { x | -2 < x ∧ x < 3 }
def set_B := { x | -3 < x ∧ x ≤ 3 }

theorem complement_U_A : (U \ A) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem intersection_A_B : (A ∩ B) = { x | -2 < x ∧ x < 3 } := sorry

theorem complement_U_intersection_A_B : (U \ (A ∩ B)) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem complement_U_A_intersection_B : ((U \ A) ∩ B) = { x | -3 < x ∧ x ≤ -2 ∨ x = 3 } := sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l157_157029


namespace ratio_of_lengths_l157_157600

theorem ratio_of_lengths (l1 l2 l3 : ℝ)
    (h1 : l2 = (1/2) * (l1 + l3))
    (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
    l1 / l3 = 7 / 5 := by
  sorry

end ratio_of_lengths_l157_157600


namespace parabola_position_l157_157781

-- Define the two parabolas as functions
def parabola1 (x : ℝ) : ℝ := x^2 - 2 * x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + 2 * x + 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (1, parabola1 1) -- (1, 2)
def vertex2 : ℝ × ℝ := (-1, parabola2 (-1)) -- (-1, 0)

-- Define the proof problem where we show relative positions
theorem parabola_position :
  (vertex1.1 > vertex2.1) ∧ (vertex1.2 > vertex2.2) :=
by
  sorry

end parabola_position_l157_157781


namespace lanies_salary_l157_157687

variables (hours_worked_per_week : ℚ) (hourly_rate : ℚ)

namespace Lanie
def salary (fraction_of_weekly_hours : ℚ) : ℚ :=
  (fraction_of_weekly_hours * hours_worked_per_week) * hourly_rate

theorem lanies_salary : 
  hours_worked_per_week = 40 ∧
  hourly_rate = 15 ∧
  fraction_of_weekly_hours = 4 / 5 →
  salary fraction_of_weekly_hours = 480 :=
by
  -- Proof steps go here
  sorry
end Lanie

end lanies_salary_l157_157687


namespace balls_in_boxes_l157_157662

theorem balls_in_boxes : (3^4 = 81) :=
by
  sorry

end balls_in_boxes_l157_157662


namespace sequence_a_n_l157_157354

theorem sequence_a_n (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = 3 + 2^n) →
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) ↔ 
  (∀ n : ℕ, a n = if n = 1 then 5 else 2^(n-1)) :=
by
  sorry

end sequence_a_n_l157_157354


namespace find_a_value_l157_157366

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l157_157366


namespace mean_of_combined_sets_l157_157888

theorem mean_of_combined_sets (mean_set1 mean_set2 : ℝ) (n1 n2 : ℕ) 
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 20) (h3 : n1 = 5) (h4 : n2 = 8) :
  (n1 * mean_set1 + n2 * mean_set2) / (n1 + n2) = 235 / 13 :=
by
  sorry

end mean_of_combined_sets_l157_157888


namespace range_of_a_l157_157653

variable (a : ℝ)
def proposition_p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def proposition_q := ∃ x₀ : ℝ, x₀^2 - x₀ + a = 0

theorem range_of_a (h1 : proposition_p a ∨ proposition_q a)
    (h2 : ¬ (proposition_p a ∧ proposition_q a)) :
    a < 0 ∨ (1 / 4) < a ∧ a < 4 :=
  sorry

end range_of_a_l157_157653


namespace cos_difference_of_angles_l157_157480

theorem cos_difference_of_angles (α β : ℝ) 
    (h1 : Real.cos (α + β) = 1 / 5) 
    (h2 : Real.tan α * Real.tan β = 1 / 2) : 
    Real.cos (α - β) = 3 / 5 := 
sorry

end cos_difference_of_angles_l157_157480


namespace complex_number_eq_l157_157654

theorem complex_number_eq (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 :=
sorry

end complex_number_eq_l157_157654


namespace exponentiation_distributes_over_multiplication_l157_157474

theorem exponentiation_distributes_over_multiplication (a b c : ℝ) : (a * b) ^ c = a ^ c * b ^ c := 
sorry

end exponentiation_distributes_over_multiplication_l157_157474


namespace division_of_people_l157_157114

theorem division_of_people (people : Fin 6) :
  ∃ (ways : ℕ), ways = 50 ∧ 
  ((∃ A B : Finset (Fin 6), A ∪ B = Finset.univ ∧ A ∩ B = ∅ ∧ 4 ≤ A.card ∧ 2 ≤ B.card) 
   ∨ (∃ A B : Finset (Fin 6), B ∪ A = Finset.univ ∧ B ∩ A = ∅ ∧ 2 ≤ A.card ∧ 4 ≤ B.card)) :=
sorry

end division_of_people_l157_157114


namespace word_identification_l157_157574

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end word_identification_l157_157574


namespace simplify_and_evaluate_expr_l157_157415

theorem simplify_and_evaluate_expr :
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 :=
by
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  sorry

end simplify_and_evaluate_expr_l157_157415


namespace fifth_graders_buy_more_l157_157676

-- Define the total payments made by eighth graders and fifth graders
def eighth_graders_payment : ℕ := 210
def fifth_graders_payment : ℕ := 240
def number_of_fifth_graders : ℕ := 25

-- The price per notebook in whole cents
def price_per_notebook (p : ℕ) : Prop :=
  ∃ k1 k2 : ℕ, k1 * p = eighth_graders_payment ∧ k2 * p = fifth_graders_payment

-- The difference in the number of notebooks bought by the fifth graders and the eighth graders
def notebook_difference (p : ℕ) : ℕ :=
  let eighth_graders_notebooks := eighth_graders_payment / p
  let fifth_graders_notebooks := fifth_graders_payment / p
  fifth_graders_notebooks - eighth_graders_notebooks

-- Theorem stating the difference in the number of notebooks equals 2
theorem fifth_graders_buy_more (p : ℕ) (h : price_per_notebook p) : notebook_difference p = 2 :=
  sorry

end fifth_graders_buy_more_l157_157676


namespace total_animals_l157_157572

theorem total_animals (B : ℕ) (h1 : 4 * B + 8 = 44) : B + 4 = 13 := by
  sorry

end total_animals_l157_157572


namespace least_candies_to_remove_for_equal_distribution_l157_157628

theorem least_candies_to_remove_for_equal_distribution :
  ∃ k : ℕ, k = 4 ∧ ∀ n : ℕ, 24 - k = 5 * n :=
sorry

end least_candies_to_remove_for_equal_distribution_l157_157628


namespace compare_fractions_l157_157314

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l157_157314


namespace subcommittee_count_l157_157758

-- Define the conditions: number of Republicans and Democrats in the Senate committee
def numRepublicans : ℕ := 10
def numDemocrats : ℕ := 8
def chooseRepublicans : ℕ := 4
def chooseDemocrats : ℕ := 3

-- Define the main proof problem based on the conditions and the correct answer
theorem subcommittee_count :
  (Nat.choose numRepublicans chooseRepublicans) * (Nat.choose numDemocrats chooseDemocrats) = 11760 := by
  sorry

end subcommittee_count_l157_157758


namespace range_of_k_l157_157353

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Define the condition that the function does not pass through the third quadrant
def does_not_pass_third_quadrant (k : ℝ) : Prop :=
  ∀ x : ℝ, (x < 0 ∧ linear_function k x < 0) → false

-- Theorem statement proving the range of k
theorem range_of_k (k : ℝ) : does_not_pass_third_quadrant k ↔ (0 ≤ k ∧ k < 2) :=
by
  sorry

end range_of_k_l157_157353


namespace find_point_C_l157_157278

noncomputable def point_on_z_axis (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def point_A : ℝ × ℝ × ℝ := (1, 0, 2)
def point_B : ℝ × ℝ × ℝ := (1, 1, 1)

theorem find_point_C :
  ∃ C : ℝ × ℝ × ℝ, (C = point_on_z_axis 1) ∧ (dist C point_A = dist C point_B) :=
by
  sorry

end find_point_C_l157_157278


namespace lolita_milk_per_week_l157_157401

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end lolita_milk_per_week_l157_157401


namespace aimee_poll_l157_157295

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l157_157295


namespace expected_value_is_0_point_25_l157_157941

-- Define the probabilities and earnings
def prob_roll_1 := 1/4
def earning_1 := 4
def prob_roll_2 := 1/4
def earning_2 := -3
def prob_roll_3_to_6 := 1/8
def earning_3_to_6 := 0

-- Define the expected value calculation
noncomputable def expected_value : ℝ := 
  (prob_roll_1 * earning_1) + 
  (prob_roll_2 * earning_2) + 
  (prob_roll_3_to_6 * earning_3_to_6) * 4  -- For 3, 4, 5, and 6

-- The theorem to be proved
theorem expected_value_is_0_point_25 : expected_value = 0.25 := by
  sorry

end expected_value_is_0_point_25_l157_157941


namespace nonagon_area_l157_157456

noncomputable def area_of_nonagon (r : ℝ) : ℝ :=
  (9 / 2) * r^2 * Real.sin (Real.pi * 40 / 180)

theorem nonagon_area (r : ℝ) : 
  area_of_nonagon r = 2.891 * r^2 :=
by
  sorry

end nonagon_area_l157_157456


namespace unique_intersection_x_axis_l157_157673

theorem unique_intersection_x_axis (a : ℝ) :
  (∀ (x : ℝ), (a - 1) * x^2 - 4 * x + 2 * a = 0) → a = 1 :=
begin
  sorry
end

end unique_intersection_x_axis_l157_157673


namespace tower_building_l157_157764

theorem tower_building :
  let red := 3
  let blue := 2
  let green := 4
  let height := 7
  let total_cubes := red + blue + green
  let choose_7_from_9 := @nat.choose total_cubes height
  let permutation : ℕ := nat.factorial height / (nat.factorial red * nat.factorial blue * nat.factorial (green - 1))
in choose_7_from_9 * permutation = 15120 :=
by
  let red := 3
  let blue := 2
  let green := 4
  let height := 7
  let total_cubes := red + blue + green
  let choose_7_from_9 := @nat.choose total_cubes height
  let permutation : ℕ := nat.factorial height / (nat.factorial red * nat.factorial blue * nat.factorial (green - 1))
  sorry

end tower_building_l157_157764


namespace polynomial_identity_l157_157666

theorem polynomial_identity (a0 a1 a2 a3 a4 a5 : ℤ) (x : ℤ) :
  (1 + 3 * x) ^ 5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  a0 - a1 + a2 - a3 + a4 - a5 = -32 :=
by
  sorry

end polynomial_identity_l157_157666


namespace arithmetic_sequence_sum_l157_157023

theorem arithmetic_sequence_sum (a d : ℚ) (a1 : a = 1 / 2) 
(S : ℕ → ℚ) (Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) 
(S2_eq_a3 : S 2 = a + 2 * d) :
  ∀ n, S n = (1 / 4 : ℚ) * n^2 + (1 / 4 : ℚ) * n :=
by
  intros n
  sorry

end arithmetic_sequence_sum_l157_157023


namespace prime_gt3_43_divides_expression_l157_157710

theorem prime_gt3_43_divides_expression {p : ℕ} (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (7^p - 6^p - 1) % 43 = 0 := 
  sorry

end prime_gt3_43_divides_expression_l157_157710


namespace q_domain_range_l157_157691

open Set

-- Given the function h with the specified domain and range
variable (h : ℝ → ℝ) (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3 → h x ∈ Icc 0 2)

def q (x : ℝ) : ℝ := 2 - h (x - 2)

theorem q_domain_range :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → (q h x) ∈ Icc 0 2) ∧
  (∀ y, q h y ∈ Icc 0 2 ↔ y ∈ Icc 1 5) :=
by
  sorry

end q_domain_range_l157_157691


namespace penguins_count_l157_157608

variable (P B : ℕ)

theorem penguins_count (h1 : B = 2 * P) (h2 : P + B = 63) : P = 21 :=
by
  sorry

end penguins_count_l157_157608


namespace total_doll_count_l157_157506

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l157_157506


namespace problem_equivalence_of_angles_l157_157552

noncomputable def ctg (x : ℝ) : ℝ := 1 / (Real.tan x)

theorem problem_equivalence_of_angles
  (a b c t S ω : ℝ)
  (hS : S = Real.sqrt ((a^2 + b^2 + c^2)^2 + (4 * t)^2))
  (h1 : ctg ω = (a^2 + b^2 + c^2) / (4 * t))
  (h2 : Real.cos ω = (a^2 + b^2 + c^2) / S)
  (h3 : Real.sin ω = (4 * t) / S) :
  True :=
sorry

end problem_equivalence_of_angles_l157_157552


namespace not_periodic_fraction_l157_157585

theorem not_periodic_fraction :
  ¬ ∃ (n k : ℕ), ∀ m ≥ n + k, ∃ l, 10^m + l = 10^(m+n) + l ∧ ((0.1234567891011121314 : ℝ) = (0.1234567891011121314 + l / (10^(m+n)))) :=
sorry

end not_periodic_fraction_l157_157585


namespace initial_avg_mark_l157_157417

variable (A : ℝ) -- The initial average mark

-- Conditions
def num_students : ℕ := 33
def avg_excluded_students : ℝ := 40
def num_excluded_students : ℕ := 3
def avg_remaining_students : ℝ := 95

-- Equation derived from the problem conditions
def initial_avg :=
  A * num_students - avg_excluded_students * num_excluded_students = avg_remaining_students * (num_students - num_excluded_students)

theorem initial_avg_mark :
  initial_avg A →
  A = 90 :=
by
  intro h
  sorry

end initial_avg_mark_l157_157417


namespace people_receiving_roses_l157_157865

-- Defining the conditions.
def initial_roses : Nat := 40
def stolen_roses : Nat := 4
def roses_per_person : Nat := 4

-- Stating the theorem.
theorem people_receiving_roses : 
  (initial_roses - stolen_roses) / roses_per_person = 9 :=
by sorry

end people_receiving_roses_l157_157865


namespace inequality_solution_l157_157234

noncomputable def solution_set : Set ℝ :=
  {x | -4 < x ∧ x < (17 - Real.sqrt 201) / 4} ∪ {x | (17 + Real.sqrt 201) / 4 < x ∧ x < 2 / 3}

theorem inequality_solution (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 2 / 3) :
  (2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2) ↔ x ∈ solution_set := by
  sorry

end inequality_solution_l157_157234


namespace raisin_cookies_difference_l157_157034

-- Definitions based on conditions:
def raisin_cookies_baked_yesterday : ℕ := 300
def raisin_cookies_baked_today : ℕ := 280

-- Proof statement:
theorem raisin_cookies_difference : raisin_cookies_baked_yesterday - raisin_cookies_baked_today = 20 := 
by
  sorry

end raisin_cookies_difference_l157_157034


namespace hyperbola_asymptote_eqn_l157_157190

theorem hyperbola_asymptote_eqn :
  ∀ (x y : ℝ),
  (y ^ 2 / 4 - x ^ 2 = 1) → (y = 2 * x ∨ y = -2 * x) := by
sorry

end hyperbola_asymptote_eqn_l157_157190


namespace weekly_milk_consumption_l157_157404

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end weekly_milk_consumption_l157_157404


namespace fraction_value_l157_157038

theorem fraction_value (m n : ℤ) (h : (m - 8) * (m - 8) + abs (n + 6) = 0) : n / m = -(3 / 4) :=
by sorry

end fraction_value_l157_157038


namespace mikes_lower_rate_l157_157541

theorem mikes_lower_rate (x : ℕ) (high_rate : ℕ) (total_paid : ℕ) (lower_payments : ℕ) (higher_payments : ℕ)
  (h1 : high_rate = 310)
  (h2 : total_paid = 3615)
  (h3 : lower_payments = 5)
  (h4 : higher_payments = 7)
  (h5 : lower_payments * x + higher_payments * high_rate = total_paid) :
  x = 289 :=
sorry

end mikes_lower_rate_l157_157541


namespace knights_probability_l157_157898

theorem knights_probability :
  let knights : Nat := 30
  let chosen : Nat := 4
  let probability (n k : Nat) := 1 - (((n - k + 1) * (n - k - 1) * (n - k - 3) * (n - k - 5)) / 
                                      ((n - 0) * (n - 1) * (n - 2) * (n - 3)))
  probability knights chosen = (389 / 437) := sorry

end knights_probability_l157_157898


namespace participants_count_l157_157426

theorem participants_count (x y : ℕ) 
    (h1 : y = x + 41)
    (h2 : y = 3 * x - 35) : 
    x = 38 ∧ y = 79 :=
by
  sorry

end participants_count_l157_157426


namespace complement_intersection_l157_157399

-- Defining the universal set U and subsets A and B
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 3, 4}
def B : Finset ℕ := {3, 4, 5}

-- Proving the complement of the intersection of A and B in U
theorem complement_intersection : (U \ (A ∩ B)) = {1, 2, 5} :=
by sorry

end complement_intersection_l157_157399


namespace parabola_above_line_l157_157650

variable (a b c : ℝ) (h : (b - c)^2 - 4 * a * c < 0)

theorem parabola_above_line : (b - c)^2 - 4 * a * c < 0 → (b - c)^2 - 4 * c * (a + b) < 0 :=
by sorry

end parabola_above_line_l157_157650


namespace total_trees_now_l157_157147

-- Definitions from conditions
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_fallen_trees : ℕ := 5

-- Additional definitions capturing relations
def fell_narra_trees (x : ℕ) : Prop := x + (x + 1) = total_fallen_trees
def new_narra_trees_planted (x : ℕ) : ℕ := 2 * x
def new_mahogany_trees_planted (x : ℕ) : ℕ := 3 * (x + 1)

-- Final goal
theorem total_trees_now (x : ℕ) (h : fell_narra_trees x) :
  initial_mahogany_trees + initial_narra_trees
  - total_fallen_trees
  + new_narra_trees_planted x
  + new_mahogany_trees_planted x = 88 := by
  sorry

end total_trees_now_l157_157147


namespace no_real_roots_l157_157109

-- Define the polynomial P(X) = X^5
def P (X : ℝ) : ℝ := X^5

-- Prove that for every α ∈ ℝ*, the polynomial P(X + α) - P(X) has no real roots
theorem no_real_roots (α : ℝ) (hα : α ≠ 0) : ∀ (X : ℝ), P (X + α) ≠ P X :=
by sorry

end no_real_roots_l157_157109


namespace solve_for_k_l157_157824

theorem solve_for_k (x y : ℤ) (h₁ : x = 1) (h₂ : y = k) (h₃ : 2 * x + y = 6) : k = 4 :=
by 
  sorry

end solve_for_k_l157_157824


namespace no_three_collinear_l157_157397

theorem no_three_collinear (p : ℕ) (hp : p.prime) (hp_odd : p % 2 = 1) :
  ∃ (points : Fin p → Fin p × Fin p), 
    (∀ i : Fin p, points i = (i, ⟨i.val^2 % p, sorry⟩)) ∧ 
    (∀ i j k : Fin p, i ≠ j → j ≠ k → i ≠ k → 
      ¬ collinear ℝ 
        {points i, points j, points k}) :=
sorry

end no_three_collinear_l157_157397


namespace total_boxes_count_l157_157730

theorem total_boxes_count
  (initial_boxes : ℕ := 2013)
  (boxes_per_operation : ℕ := 13)
  (operations : ℕ := 2013)
  (non_empty_boxes : ℕ := 2013)
  (total_boxes : ℕ := initial_boxes + boxes_per_operation * operations) :
  non_empty_boxes = operations → total_boxes = 28182 :=
by
  sorry

end total_boxes_count_l157_157730


namespace Joan_paid_158_l157_157438

theorem Joan_paid_158 (J K : ℝ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end Joan_paid_158_l157_157438


namespace chairs_per_row_l157_157895

/-- There are 10 rows of chairs, with the first row for awardees, the second and third rows for
    administrators and teachers, the last two rows for parents, and the remaining five rows for students.
    Given that 4/5 of the student seats are occupied, and there are 15 vacant seats among the students,
    proves that the number of chairs per row is 15. --/
theorem chairs_per_row (x : ℕ) (h1 : 10 = 1 + 1 + 1 + 5 + 2)
  (h2 : 4 / 5 * (5 * x) + 1 / 5 * (5 * x) = 5 * x)
  (h3 : 1 / 5 * (5 * x) = 15) : x = 15 :=
sorry

end chairs_per_row_l157_157895


namespace find_k_l157_157061

def f (a b c x : ℤ) : ℤ := a * x * x + b * x + c

theorem find_k : 
  ∃ k : ℤ, 
    ∃ a b c : ℤ, 
      f a b c 1 = 0 ∧
      60 < f a b c 6 ∧ f a b c 6 < 70 ∧
      120 < f a b c 9 ∧ f a b c 9 < 130 ∧
      10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)
      ∧ k = 4 :=
by
  sorry

end find_k_l157_157061


namespace minimum_value_l157_157066

theorem minimum_value (a b : ℝ) (h1 : 2 * a + 3 * b = 5) (h2 : a > 0) (h3 : b > 0) : 
  (1 / a) + (1 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l157_157066


namespace compare_negative_fractions_l157_157317

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l157_157317


namespace proof_x_squared_minus_y_squared_l157_157365

theorem proof_x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 9 / 14) (h2 : x - y = 3 / 14) :
  x^2 - y^2 = 27 / 196 := by
  sorry

end proof_x_squared_minus_y_squared_l157_157365


namespace range_of_f_when_a_eq_2_max_value_implies_a_l157_157499

-- first part
theorem range_of_f_when_a_eq_2 (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 3) :
  (∀ y, (y = x^2 + 3*x - 3) → (y ≥ -21/4 ∧ y ≤ 15)) :=
by sorry

-- second part
theorem max_value_implies_a (a : ℝ) (hx : ∀ x, -1 ≤ x ∧ x ≤ 3 → x^2 + (2*a - 1)*x - 3 ≤ 1) :
  a = -1 ∨ a = -1 / 3 :=
by sorry

end range_of_f_when_a_eq_2_max_value_implies_a_l157_157499


namespace lychee_harvest_l157_157996

theorem lychee_harvest : 
  let last_year_red := 350
  let last_year_yellow := 490
  let this_year_red := 500
  let this_year_yellow := 700
  let sold_red := 2/3 * this_year_red
  let sold_yellow := 3/7 * this_year_yellow
  let remaining_red_after_sale := this_year_red - sold_red
  let remaining_yellow_after_sale := this_year_yellow - sold_yellow
  let family_ate_red := 3/5 * remaining_red_after_sale
  let family_ate_yellow := 4/9 * remaining_yellow_after_sale
  let remaining_red := remaining_red_after_sale - family_ate_red
  let remaining_yellow := remaining_yellow_after_sale - family_ate_yellow
  (this_year_red - last_year_red) / last_year_red * 100 = 42.86
  ∧ (this_year_yellow - last_year_yellow) / last_year_yellow * 100 = 42.86
  ∧ remaining_red = 67
  ∧ remaining_yellow = 223 :=
by
    intros
    sorry

end lychee_harvest_l157_157996


namespace age_of_50th_student_l157_157079

theorem age_of_50th_student (avg_50_students : ℝ) (total_students : ℕ)
                           (avg_15_students : ℝ) (group_1_count : ℕ)
                           (avg_15_students_2 : ℝ) (group_2_count : ℕ)
                           (avg_10_students : ℝ) (group_3_count : ℕ)
                           (avg_9_students : ℝ) (group_4_count : ℕ) :
                           avg_50_students = 20 → total_students = 50 →
                           avg_15_students = 18 → group_1_count = 15 →
                           avg_15_students_2 = 22 → group_2_count = 15 →
                           avg_10_students = 25 → group_3_count = 10 →
                           avg_9_students = 24 → group_4_count = 9 →
                           ∃ (age_50th_student : ℝ), age_50th_student = 66 := by
                           sorry

end age_of_50th_student_l157_157079


namespace compare_rat_neg_l157_157331

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l157_157331


namespace team_B_at_least_half_can_serve_l157_157909

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l157_157909


namespace power_of_2_multiplication_l157_157277

theorem power_of_2_multiplication : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end power_of_2_multiplication_l157_157277


namespace compare_negative_fractions_l157_157318

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l157_157318


namespace brenda_age_l157_157943

variable (A B J : ℕ)

theorem brenda_age :
  (A = 3 * B) →
  (J = B + 6) →
  (A = J) →
  (B = 3) :=
by
  intros h1 h2 h3
  -- condition: A = 3 * B
  -- condition: J = B + 6
  -- condition: A = J
  -- prove B = 3
  sorry

end brenda_age_l157_157943


namespace total_dolls_48_l157_157510

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l157_157510


namespace non_divisible_by_twenty_l157_157470

theorem non_divisible_by_twenty (k : ℤ) (h : ∃ m : ℤ, k * (k + 1) * (k + 2) = 5 * m) :
  ¬ (∃ l : ℤ, k * (k + 1) * (k + 2) = 20 * l) := sorry

end non_divisible_by_twenty_l157_157470


namespace granger_cisco_combined_spots_l157_157976

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end granger_cisco_combined_spots_l157_157976


namespace find_difference_square_l157_157982

theorem find_difference_square (x y c b : ℝ) (h1 : x * y = c^2) (h2 : (1 / x^2) + (1 / y^2) = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := 
by sorry

end find_difference_square_l157_157982


namespace probability_of_three_different_colors_draw_l157_157760

open ProbabilityTheory

def number_of_blue_chips : ℕ := 4
def number_of_green_chips : ℕ := 5
def number_of_red_chips : ℕ := 6
def number_of_yellow_chips : ℕ := 3
def total_number_of_chips : ℕ := 18

def P_B : ℚ := number_of_blue_chips / total_number_of_chips
def P_G : ℚ := number_of_green_chips / total_number_of_chips
def P_R : ℚ := number_of_red_chips / total_number_of_chips
def P_Y : ℚ := number_of_yellow_chips / total_number_of_chips

def P_different_colors : ℚ := 2 * ((P_B * P_G + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G + P_R * P_Y) +
                                    (P_B * P_R + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G))

theorem probability_of_three_different_colors_draw :
  P_different_colors = 141 / 162 :=
by
  -- Placeholder for the actual proof.
  sorry

end probability_of_three_different_colors_draw_l157_157760


namespace div_by_37_l157_157549

theorem div_by_37 : (333^555 + 555^333) % 37 = 0 :=
by sorry

end div_by_37_l157_157549


namespace cos_30_deg_plus_2a_l157_157664

theorem cos_30_deg_plus_2a (a : ℝ) (h : Real.cos (Real.pi * (75 / 180) - a) = 1 / 3) : 
  Real.cos (Real.pi * (30 / 180) + 2 * a) = 7 / 9 := 
by 
  sorry

end cos_30_deg_plus_2a_l157_157664


namespace polynomial_factors_l157_157045

theorem polynomial_factors (h k : ℤ)
  (h1 : 3 * (-2)^4 - 2 * h * (-2)^2 + h * (-2) + k = 0)
  (h2 : 3 * 1^4 - 2 * h * 1^2 + h * 1 + k = 0)
  (h3 : 3 * (-3)^4 - 2 * h * (-3)^2 + h * (-3) + k = 0) :
  |3 * h - 2 * k| = 11 :=
by
  sorry

end polynomial_factors_l157_157045


namespace find_x_l157_157647

theorem find_x (x : ℝ) (h1 : 3 * Real.sin (2 * x) = 2 * Real.sin x) (h2 : 0 < x ∧ x < Real.pi) :
  x = Real.arccos (1 / 3) :=
by
  sorry

end find_x_l157_157647


namespace cos_transformation_l157_157655

variable {θ a : ℝ}

theorem cos_transformation (h : Real.sin (θ + π / 12) = a) :
  Real.cos (θ + 7 * π / 12) = -a := 
sorry

end cos_transformation_l157_157655


namespace comparison_b_a_c_l157_157486

noncomputable def a : ℝ := Real.sqrt 1.2
noncomputable def b : ℝ := Real.exp 0.1
noncomputable def c : ℝ := 1 + Real.log 1.1

theorem comparison_b_a_c : b > a ∧ a > c :=
by
  unfold a b c
  sorry

end comparison_b_a_c_l157_157486


namespace find_sum_2017_l157_157678

-- Define the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Given conditions
variables (a : ℕ → ℤ)
axiom h1 : is_arithmetic_sequence a
axiom h2 : sum_first_n_terms a 2011 = -2011
axiom h3 : a 1012 = 3

-- Theorem to be proven
theorem find_sum_2017 : sum_first_n_terms a 2017 = 2017 :=
by sorry

end find_sum_2017_l157_157678


namespace shaggy_seeds_l157_157801

theorem shaggy_seeds {N : ℕ} (h1 : 50 < N) (h2 : N < 65) (h3 : N = 60) : 
  ∃ L : ℕ, L = 54 := by
  let L := 54
  sorry

end shaggy_seeds_l157_157801


namespace rectangular_prism_volume_l157_157131

theorem rectangular_prism_volume
  (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66) :
  a * b * c = 150 :=
by sorry

end rectangular_prism_volume_l157_157131


namespace height_difference_l157_157945

theorem height_difference :
  let janet_height := 3.6666666666666665
  let sister_height := 2.3333333333333335
  janet_height - sister_height = 1.333333333333333 :=
by
  sorry

end height_difference_l157_157945


namespace bob_deli_total_cost_l157_157944

-- Definitions based on the problem's conditions
def sandwich_cost : ℕ := 5
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount_threshold : ℕ := 50
def discount_amount : ℕ := 10

-- The total initial cost without discount
def initial_total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The final cost after applying discount if applicable
def final_cost : ℕ :=
  if initial_total_cost > discount_threshold then
    initial_total_cost - discount_amount
  else
    initial_total_cost

-- Statement to prove
theorem bob_deli_total_cost : final_cost = 55 := by
  sorry

end bob_deli_total_cost_l157_157944


namespace det_A_is_2_l157_157217

-- Define the matrix A
def A (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 2], ![-3, d]]

-- Define the inverse of matrix A 
noncomputable def A_inv (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / (a * d + 6)) • ![![d, -2], ![3, a]]

-- Condition: A + A_inv = 0
def condition (a d : ℝ) : Prop := A a d + A_inv a d = 0

-- Main theorem: determinant of A under the given condition
theorem det_A_is_2 (a d : ℝ) (h : condition a d) : Matrix.det (A a d) = 2 :=
by sorry

end det_A_is_2_l157_157217


namespace no_integer_solutions_l157_157071

theorem no_integer_solutions (x y : ℤ) : 2 * x^2 - 5 * y^2 ≠ 7 :=
  sorry

end no_integer_solutions_l157_157071


namespace value_of_a_minus_b_l157_157827

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) : a - b = -10 ∨ a - b = 10 :=
sorry

end value_of_a_minus_b_l157_157827


namespace trig_identity_example_l157_157807

theorem trig_identity_example (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 / 3 :=
by
  sorry

end trig_identity_example_l157_157807


namespace subcommittee_ways_l157_157754

theorem subcommittee_ways : 
  let R := 10 in
  let D := 8 in
  let kR := 4 in
  let kD := 3 in
  (Nat.choose R kR) * (Nat.choose D kD) = 11760 :=
by
  sorry

end subcommittee_ways_l157_157754


namespace max_value_expression_l157_157785

theorem max_value_expression : 
  ∃ x_max : ℝ, 
    (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ -3 * x_max^2 + 15 * x_max + 9) ∧
    (-3 * x_max^2 + 15 * x_max + 9 = 111 / 4) :=
by
  sorry

end max_value_expression_l157_157785


namespace middle_school_students_count_l157_157991

variable (M H m h : ℕ)
variable (total_students : ℕ := 36)
variable (percentage_middle : ℕ := 20)
variable (percentage_high : ℕ := 25)

theorem middle_school_students_count :
  total_students = 36 ∧ (m = h) →
  (percentage_middle / 100 * M = m) ∧
  (percentage_high / 100 * H = h) →
  M + H = total_students →
  M = 16 :=
by sorry

end middle_school_students_count_l157_157991


namespace min_value_2_div_a_1_div_b_l157_157489

theorem min_value_2_div_a_1_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (h_perpendicular : ((a - 1) ≠ 0) ∧ (1-a) * (-1/(2 * b)) = -1) : 
    (2 / a + 1 / b) ≥ 8 :=
sorry

end min_value_2_div_a_1_div_b_l157_157489


namespace triangle_base_length_l157_157384

theorem triangle_base_length (x : ℝ) :
  (∃ s : ℝ, 4 * s = 64 ∧ s * s = 256) ∧ (32 * x / 2 = 256) → x = 16 := by
  sorry

end triangle_base_length_l157_157384


namespace number_of_boys_in_school_l157_157210

theorem number_of_boys_in_school (B : ℕ) (girls : ℕ) (difference : ℕ) 
    (h1 : girls = 697) (h2 : girls = B + 228) : B = 469 := 
by
  sorry

end number_of_boys_in_school_l157_157210


namespace midpoint_P_AB_l157_157679

structure Point := (x : ℝ) (y : ℝ)

def segment_midpoint (P A B : Point) : Prop := P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

variables {A D C E P B : Point}
variables (h1 : A.x = D.x ∧ A.y = D.y)
variables (h2 : D.x = C.x ∧ D.y = C.y)
variables (h3 : D.x = P.x ∧ D.y = P.y ∧ P.x = E.x ∧ P.y = E.y)
variables (h4 : B.x = E.x ∧ B.y = E.y)
variables (h5 : A.x = C.x ∧ A.y = C.y)
variables (angle_ADC : ∀ x y : ℝ, (x - A.x)^2 + (y - A.y)^2 = (x - D.x)^2 + (y - D.y)^2 → (x - C.x)^2 + (y - C.y)^2 = (x - D.x)^2 + (y - D.y)^2)
variables (angle_DPE : ∀ x y : ℝ, (x - D.x)^2 + (y - P.y)^2 = (x - P.x)^2 + (y - E.y)^2 → (x - E.x)^2 + (y - E.y)^2 = (x - P.x)^2 + (y - E.y)^2)
variables (angle_BEC : ∀ x y : ℝ, (x - B.x)^2 + (y - E.y)^2 = (x - E.x)^2 + (y - C.y)^2 → (x - B.x)^2 + (y - C.y)^2 = (x - E.x)^2 + (y - C.y)^2)

theorem midpoint_P_AB : segment_midpoint P A B := 
sorry

end midpoint_P_AB_l157_157679


namespace arbitrary_large_sum_of_digits_l157_157064

noncomputable def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem arbitrary_large_sum_of_digits (a : Nat) (h1 : 2 ≤ a) (h2 : ¬ (2 ∣ a)) (h3 : ¬ (5 ∣ a)) :
  ∃ m : Nat, sum_of_digits (a^m) > m :=
by
  sorry

end arbitrary_large_sum_of_digits_l157_157064


namespace square_side_length_l157_157626

variable (s d k : ℝ)

theorem square_side_length {s d k : ℝ} (h1 : s + d = k) (h2 : d = s * Real.sqrt 2) : 
  s = k / (1 + Real.sqrt 2) :=
sorry

end square_side_length_l157_157626


namespace find_f_2006_l157_157494

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x
def g_def (f : ℝ → ℝ) (g : ℝ → ℝ) := ∀ x : ℝ, g x = f (x - 1)
def f_at_2 (f : ℝ → ℝ) := f 2 = 2

-- The theorem to prove
theorem find_f_2006 (f g : ℝ → ℝ) 
  (even_f : is_even f) 
  (odd_g : is_odd g) 
  (g_eq_f_shift : g_def f g) 
  (f_eq_2 : f_at_2 f) : 
  f 2006 = 2 := 
sorry

end find_f_2006_l157_157494


namespace fifth_student_gold_stickers_l157_157255

theorem fifth_student_gold_stickers :
  ∀ s1 s2 s3 s4 s5 s6 : ℕ,
  s1 = 29 →
  s2 = 35 →
  s3 = 41 →
  s4 = 47 →
  s6 = 59 →
  (s2 - s1 = 6) →
  (s3 - s2 = 6) →
  (s4 - s3 = 6) →
  (s6 - s4 = 12) →
  s5 = s4 + (s2 - s1) →
  s5 = 53 := by
  intros s1 s2 s3 s4 s5 s6 hs1 hs2 hs3 hs4 hs6 hd1 hd2 hd3 hd6 heq
  subst_vars
  sorry

end fifth_student_gold_stickers_l157_157255


namespace inequality_system_solution_l157_157359

theorem inequality_system_solution (a b x : ℝ) 
  (h1 : x - a > 2)
  (h2 : x + 1 < b)
  (h3 : -1 < x)
  (h4 : x < 1) :
  (a + b) ^ 2023 = -1 :=
by 
  sorry

end inequality_system_solution_l157_157359


namespace find_b_squared_l157_157126

-- Assume a and b are real numbers and positive
variables (a b : ℝ)
-- Given conditions
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom magnitude : a^2 + b^2 = 100
axiom equidistant : 2 * a - 4 * b = 7

-- Main proof statement
theorem find_b_squared : b^2 = 287 / 17 := sorry

end find_b_squared_l157_157126


namespace lcm_of_15_18_20_is_180_l157_157644

theorem lcm_of_15_18_20_is_180 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end lcm_of_15_18_20_is_180_l157_157644


namespace fiona_hoodies_l157_157645

theorem fiona_hoodies (F C : ℕ) (h1 : F + C = 8) (h2 : C = F + 2) : F = 3 :=
by
  sorry

end fiona_hoodies_l157_157645


namespace geometry_problem_l157_157973

-- Definitions for geometric relationships: parallel and perpendicular
variables {Line Plane : Type}
variable (a b : Line)
variable (α β γ : Plane)

-- Relation Definitions
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p q : Plane) : Prop := sorry
def plane_perpendicular (p q : Plane) : Prop := sorry

-- Given conditions
axiom h1 : plane_perpendicular α γ
axiom h2 : plane_parallel β γ

-- Prove the statement
theorem geometry_problem : plane_perpendicular α β := 
by 
  sorry

end geometry_problem_l157_157973


namespace compare_rat_neg_l157_157332

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l157_157332


namespace part1_part2_l157_157806

def A (x : ℝ) (a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def B (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

theorem part1 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∧ B x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

theorem part2 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∨ B x) ↔ (1 < x ∧ x ≤ 3) :=
sorry

end part1_part2_l157_157806


namespace vector_magnitude_proof_l157_157196

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_proof
  (a b c : ℝ × ℝ)
  (h_a : a = (-2, 1))
  (h_b : b = (-2, 3))
  (h_c : ∃ m : ℝ, c = (m, -1) ∧ (m * b.1 + (-1) * b.2 = 0)) :
  vector_magnitude (a.1 - c.1, a.2 - c.2) = Real.sqrt 17 / 2 :=
by
  sorry

end vector_magnitude_proof_l157_157196


namespace max_value_of_ab_expression_l157_157218

noncomputable def max_ab_expression : ℝ :=
  let a := 4
  let b := 20 / 3
  a * b * (60 - 5 * a - 3 * b)

theorem max_value_of_ab_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 3 * b < 60 →
  ab * (60 - 5 * a - 3 * b) ≤ max_ab_expression :=
sorry

end max_value_of_ab_expression_l157_157218


namespace john_pays_total_cost_l157_157840

def number_of_candy_bars_John_buys : ℕ := 20
def number_of_candy_bars_Dave_pays_for : ℕ := 6
def cost_per_candy_bar : ℚ := 1.50

theorem john_pays_total_cost :
  number_of_candy_bars_John_buys - number_of_candy_bars_Dave_pays_for = 14 →
  14 * cost_per_candy_bar = 21 :=
  by
  intros h
  linarith
  sorry

end john_pays_total_cost_l157_157840


namespace largest_y_coordinate_l157_157625

theorem largest_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y = 2 := 
by 
  -- Proof will be provided here
  sorry

end largest_y_coordinate_l157_157625


namespace find_coordinates_of_Q_l157_157051

theorem find_coordinates_of_Q (x y : ℝ) (P : ℝ × ℝ) (hP : P = (1, 2))
    (perp : x + 2 * y = 0) (length : x^2 + y^2 = 5) :
    (x, y) = (-2, 1) :=
by
  -- Proof should go here
  sorry

end find_coordinates_of_Q_l157_157051


namespace negative_fraction_comparison_l157_157310

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l157_157310


namespace place_sweet_hexagons_l157_157455

def sweetHexagon (h : ℝ) : Prop := h = 1
def convexPolygon (A : ℝ) : Prop := A ≥ 1900000
def hexagonPlacementPossible (N : ℕ) : Prop := N ≤ 2000000

theorem place_sweet_hexagons:
  (∀ h, sweetHexagon h) →
  (∃ A, convexPolygon A) →
  (∃ N, hexagonPlacementPossible N) →
  True :=
by
  intros _ _ _ 
  exact True.intro

end place_sweet_hexagons_l157_157455


namespace possible_k_values_l157_157557

variables (p q r s k : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
          (h5 : p * q = r * s)
          (h6 : p * k ^ 3 + q * k ^ 2 + r * k + s = 0)
          (h7 : q * k ^ 3 + r * k ^ 2 + s * k + p = 0)

noncomputable def roots_of_unity := {k : ℂ | k ^ 4 = 1}

theorem possible_k_values : k ∈ roots_of_unity :=
by {
  sorry
}

end possible_k_values_l157_157557


namespace book_pages_total_l157_157663

-- Define the conditions as hypotheses
def total_pages (P : ℕ) : Prop :=
  let read_first_day := P / 2
  let read_second_day := P / 4
  let read_third_day := P / 6
  let read_total := read_first_day + read_second_day + read_third_day
  let remaining_pages := P - read_total
  remaining_pages = 20

-- The proof statement
theorem book_pages_total (P : ℕ) (h : total_pages P) : P = 240 := sorry

end book_pages_total_l157_157663


namespace complex_identity_l157_157822

theorem complex_identity (a b : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - 2 * i) * i = a + b * i) : a * b = 2 :=
by
  sorry

end complex_identity_l157_157822


namespace richmond_tickets_l157_157875

theorem richmond_tickets (total_tickets : ℕ) (second_half_tickets : ℕ) (first_half_tickets : ℕ) :
  total_tickets = 9570 →
  second_half_tickets = 5703 →
  first_half_tickets = total_tickets - second_half_tickets →
  first_half_tickets = 3867 := by
  sorry

end richmond_tickets_l157_157875


namespace kayak_rental_cost_l157_157263

theorem kayak_rental_cost
    (canoe_cost_per_day : ℕ := 14)
    (total_revenue : ℕ := 288)
    (canoe_kayak_ratio : ℕ × ℕ := (3, 2))
    (canoe_kayak_difference : ℕ := 4)
    (number_of_kayaks : ℕ := 8)
    (number_of_canoes : ℕ := number_of_kayaks + canoe_kayak_difference)
    (canoe_revenue : ℕ := number_of_canoes * canoe_cost_per_day) :
    number_of_kayaks * kayak_cost_per_day = total_revenue - canoe_revenue →
    kayak_cost_per_day = 15 := 
by
  sorry

end kayak_rental_cost_l157_157263


namespace Charley_total_beads_pulled_l157_157303

-- Definitions and conditions
def initial_white_beads := 105
def initial_black_beads := 210
def initial_blue_beads := 60

def first_round_black_pulled := (2 / 7) * initial_black_beads
def first_round_white_pulled := (3 / 7) * initial_white_beads
def first_round_blue_pulled := (1 / 4) * initial_blue_beads

def first_round_total_pulled := first_round_black_pulled + first_round_white_pulled + first_round_blue_pulled

def remaining_black_beads := initial_black_beads - first_round_black_pulled
def remaining_white_beads := initial_white_beads - first_round_white_pulled
def remaining_blue_beads := initial_blue_beads - first_round_blue_pulled

def added_white_beads := 45
def added_black_beads := 80

def total_black_beads := remaining_black_beads + added_black_beads
def total_white_beads := remaining_white_beads + added_white_beads

def second_round_black_pulled := (3 / 8) * total_black_beads
def second_round_white_pulled := (1 / 3) * added_white_beads

def second_round_total_pulled := second_round_black_pulled + second_round_white_pulled

def total_beads_pulled := first_round_total_pulled + second_round_total_pulled 

-- Theorem statement
theorem Charley_total_beads_pulled : total_beads_pulled = 221 := 
by
  -- we can ignore the proof step and leave it to be filled
  sorry

end Charley_total_beads_pulled_l157_157303


namespace find_phi_l157_157244

theorem find_phi (φ : ℝ) (h₁ : 0 ≤ φ ∧ φ ≤ π) 
  (h₂ : ∀ x : ℝ, sin (x + φ) = sin (-x + φ)) : 
  φ = π / 2 := 
  sorry

end find_phi_l157_157244


namespace prob_chocolate_milk_4_of_5_days_l157_157702

noncomputable def prob_chocolate : ℚ := 2 / 3
noncomputable def prob_regular : ℚ := 1 / 3

-- We want to prove that prob_combined (5 choose 4) P(C)⁴ P(R)¹ = 80 / 243
theorem prob_chocolate_milk_4_of_5_days :
  let ways := Nat.choose 5 4,
      prob := prob_chocolate ^ 4 * prob_regular ^ 1
  in ways * prob = 80 / 243 :=
by
  have ways := Nat.choose 5 4
  have prob := prob_chocolate ^ 4 * prob_regular ^ 1
  sorry

end prob_chocolate_milk_4_of_5_days_l157_157702


namespace find_a_b_largest_x_l157_157360

def polynomial (a b x : ℤ) : ℤ := 2 * (a * x - 3) - 3 * (b * x + 5)

-- Given conditions
variables (a b : ℤ)
#check polynomial

-- Part 1: Prove the values of a and b
theorem find_a_b (h1 : polynomial a b 2 = -31) (h2 : a + b = 0) : a = -1 ∧ b = 1 :=
by sorry

-- Part 2: Given a and b found in Part 1, find the largest integer x such that P > 0
noncomputable def P (x : ℤ) : ℤ := -5 * x - 21

theorem largest_x {a b : ℤ} (ha : a = -1) (hb : b = 1) : ∃ x : ℤ, P x > 0 ∧ ∀ y : ℤ, (P y > 0 → y ≤ x) :=
by sorry

end find_a_b_largest_x_l157_157360


namespace boats_meet_time_l157_157262

theorem boats_meet_time (v_A v_C current distance : ℝ) : 
  v_A = 7 → 
  v_C = 3 → 
  current = 2 → 
  distance = 20 → 
  (distance / (v_A + current + v_C - current) = 2 ∨
   distance / (v_A + current - (v_C + current)) = 5) := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Apply simplifications or calculations as necessary
  sorry

end boats_meet_time_l157_157262


namespace calculate_truncated_cone_volume_l157_157139

noncomputable def volume_of_truncated_cone (R₁ R₂ h : ℝ) :
    ℝ := ((1 / 3) * Real.pi * h * (R₁ ^ 2 + R₁ * R₂ + R₂ ^ 2))

theorem calculate_truncated_cone_volume : 
    volume_of_truncated_cone 10 5 10 = (1750 / 3) * Real.pi := by
sorry

end calculate_truncated_cone_volume_l157_157139


namespace largest_number_proof_l157_157955

/-- 
The largest natural number that does not end in zero and decreases by an integer factor 
when one (not the first) digit is removed.
-/
def largest_number_decreasing_by_factor : ℕ := 
  let x := 8
  let a := 1
  let c := 625
  let n := 1
  let r := 5
  let number := 10^(n+1) * a + 10^n * x + c
  { number | number ∉ [0], number % 10 ≠ 0, (r=5), 2 ≤ r ≤ 19 }

theorem largest_number_proof :
  largest_number_decreasing_by_factor = 180625 :=
sorry

end largest_number_proof_l157_157955


namespace subcommittee_count_l157_157746

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose republicans subcommittee_republicans * choose democrats subcommittee_democrats = 11760 :=
by
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end subcommittee_count_l157_157746


namespace max_balls_count_l157_157851

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l157_157851


namespace granger_cisco_combined_spots_l157_157975

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end granger_cisco_combined_spots_l157_157975


namespace largest_room_width_l157_157567

theorem largest_room_width (w : ℕ) :
  (w * 30 - 15 * 8 = 1230) → (w = 45) :=
by
  intro h
  sorry

end largest_room_width_l157_157567


namespace directrix_of_parabola_l157_157163

theorem directrix_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 4 * x^2 - 6) : 
    ∃ d, (∀ x, y x = 4 * x^2 - 6) ∧ d = -97/16 ↔ (y (-6 - d)) = -10 := 
    sorry

end directrix_of_parabola_l157_157163


namespace geometric_sequence_sum_l157_157490

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) -- a_n is a sequence of real numbers
  (q : ℝ) -- q is the common ratio
  (h1 : a 1 + a 2 = 20) -- first condition
  (h2 : a 3 + a 4 = 80) -- second condition
  (h_geom : ∀ n, a (n + 1) = a n * q) -- property of geometric sequence
  : a 5 + a 6 = 320 := 
sorry

end geometric_sequence_sum_l157_157490


namespace john_brown_bags_l157_157532

theorem john_brown_bags :
  (∃ b : ℕ, 
     let total_macaroons := 12
     let weight_per_macaroon := 5
     let total_weight := total_macaroons * weight_per_macaroon
     let remaining_weight := 45
     let bag_weight := total_weight - remaining_weight
     let macaroons_per_bag := bag_weight / weight_per_macaroon
     total_macaroons / macaroons_per_bag = b
  ) → b = 4 :=
by
  sorry

end john_brown_bags_l157_157532


namespace billion_to_scientific_notation_l157_157144

theorem billion_to_scientific_notation : 
  (98.36 * 10^9) = 9.836 * 10^10 := 
by
  sorry

end billion_to_scientific_notation_l157_157144


namespace prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l157_157124

-- Definitions
def fair_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Question 1: Probability that a + b >= 9
theorem prob_sum_geq_9 (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  a + b ≥ 9 → (∃ (valid_outcomes : Finset (ℕ × ℕ)),
    valid_outcomes = {(3, 6), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4), (6, 5), (6, 6)} ∧
    valid_outcomes.card = 10 ∧
    10 / 36 = 5 / 18) :=
sorry

-- Question 2: Probability that the line ax + by + 5 = 0 is tangent to the circle x^2 + y^2 = 1
theorem prob_tangent_line (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (tangent_outcomes : Finset (ℕ × ℕ)),
    tangent_outcomes = {(3, 4), (4, 3)} ∧
    a^2 + b^2 = 25 ∧
    tangent_outcomes.card = 2 ∧
    2 / 36 = 1 / 18) :=
sorry

-- Question 3: Probability that the lengths a, b, and 5 form an isosceles triangle
theorem prob_isosceles_triangle (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (isosceles_outcomes : Finset (ℕ × ℕ)),
    isosceles_outcomes = {(1, 5), (2, 5), (3, 3), (3, 5), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6)} ∧
    isosceles_outcomes.card = 14 ∧
    14 / 36 = 7 / 18) :=
sorry

end prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l157_157124


namespace parabola_directrix_l157_157561

theorem parabola_directrix (x y : ℝ) (h_parabola : x^2 = (1/2) * y) : y = - (1/8) :=
sorry

end parabola_directrix_l157_157561


namespace find_x_l157_157528

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 152) : x = 16 := 
by 
  sorry

end find_x_l157_157528


namespace clara_climbs_stone_blocks_l157_157948

-- Define the number of steps per level
def steps_per_level : Nat := 8

-- Define the number of blocks per step
def blocks_per_step : Nat := 3

-- Define the number of levels in the tower
def levels : Nat := 4

-- Define a function to compute the total number of blocks given the constants
def total_blocks (steps_per_level blocks_per_step levels : Nat) : Nat :=
  steps_per_level * blocks_per_step * levels

-- Statement of the theorem
theorem clara_climbs_stone_blocks :
  total_blocks steps_per_level blocks_per_step levels = 96 :=
by
  -- Lean requires 'sorry' as a placeholder for the proof.
  sorry

end clara_climbs_stone_blocks_l157_157948


namespace part1_part2_l157_157358

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem part1 (x : ℝ) : 
  (∀ (x : ℝ), f x 1 ≥ 1 → x ≤ -3 / 2) :=
sorry

theorem part2 (x t : ℝ) (h : ∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) : 
  0 < m ∧ m < 3 / 4 :=
sorry

end part1_part2_l157_157358


namespace larger_jar_half_full_l157_157006

-- Defining the capacities of the jars
variables (S L W : ℚ)

-- Conditions
def equal_amount_water (S L W : ℚ) : Prop :=
  W = (1/5 : ℚ) * S ∧ W = (1/4 : ℚ) * L

-- Question: What fraction will the larger jar be filled if the water from the smaller jar is added to it?
theorem larger_jar_half_full (S L W : ℚ) (h : equal_amount_water S L W) :
  (2 * W) / L = (1 / 2 : ℚ) :=
sorry

end larger_jar_half_full_l157_157006


namespace ratio_of_lengths_l157_157602

variables (l1 l2 l3 : ℝ)

theorem ratio_of_lengths (h1 : l2 = (1/2) * (l1 + l3))
                         (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
  (l1 / l3) = (7 / 5) :=
by
  sorry

end ratio_of_lengths_l157_157602


namespace fraction_burritos_given_away_l157_157685

noncomputable def total_burritos_bought : Nat := 3 * 20
noncomputable def burritos_eaten : Nat := 3 * 10
noncomputable def burritos_left : Nat := 10
noncomputable def burritos_before_eating : Nat := burritos_eaten + burritos_left
noncomputable def burritos_given_away : Nat := total_burritos_bought - burritos_before_eating

theorem fraction_burritos_given_away : (burritos_given_away : ℚ) / total_burritos_bought = 1 / 3 := by
  sorry

end fraction_burritos_given_away_l157_157685


namespace inequality_solution_set_l157_157893

theorem inequality_solution_set (x : ℝ) : (x - 1 < 7) ∧ (3 * x + 1 ≥ -2) ↔ -1 ≤ x ∧ x < 8 :=
by
  sorry

end inequality_solution_set_l157_157893


namespace abs_gt_one_iff_square_inequality_l157_157419

theorem abs_gt_one_iff_square_inequality (x : ℝ) : |x| > 1 ↔ x^2 - 1 > 0 := 
sorry

end abs_gt_one_iff_square_inequality_l157_157419


namespace value_of_A_l157_157400

theorem value_of_A (h p a c k e : ℤ) 
  (H : h = 8)
  (PACK : p + a + c + k = 50)
  (PECK : p + e + c + k = 54)
  (CAKE : c + a + k + e = 40) : 
  a = 25 :=
by 
  sorry

end value_of_A_l157_157400


namespace triangle_area_l157_157058

variables (a b : ℝ × ℝ)

def reflect_x (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1, -v.2)

def parallelogram_area (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem triangle_area
  (h₁ : a = (4, -1))
  (h₂ : b = (3, 4)) :
  let a' := reflect_x a in
  0.5 * (parallelogram_area a' b) = 13 / 2 :=
by
  -- Since no proof steps are required, we just use sorry to skip the proof.
  sorry

end triangle_area_l157_157058


namespace intersection_P_Q_correct_l157_157492

-- Define sets P and Q based on given conditions
def is_in_P (x : ℝ) : Prop := x > 1
def is_in_Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the intersection P ∩ Q and the correct answer
def P_inter_Q (x : ℝ) : Prop := is_in_P x ∧ is_in_Q x
def correct_ans (x : ℝ) : Prop := 1 < x ∧ x ≤ 2

-- Prove that P ∩ Q = (1, 2]
theorem intersection_P_Q_correct : ∀ x : ℝ, P_inter_Q x ↔ correct_ans x :=
by sorry

end intersection_P_Q_correct_l157_157492


namespace area_of_square_BCFE_eq_2304_l157_157555

-- Definitions of points and side lengths as per the conditions given in the problem
variables (A B C D E F G : Type*) [euclidean_space A] [euclidean_space B] 
[euclidean_space C] [euclidean_space D] [euclidean_space E] 
[euclidean_space F] [euclidean_space G]

-- Definitions of side lengths
def AB : ℝ := 36
def CD : ℝ := 64

def side_length_of_square (x : ℝ) := x * x 

-- The goal is to prove that the area of square BCFE equals 2304
theorem area_of_square_BCFE_eq_2304 (x : ℝ) 
  (h1: similarity (triangle A B G) (triangle F D C))
  (h2: AB = 36)
  (h3: CD = 64)
  : side_length_of_square x = 2304 := sorry

end area_of_square_BCFE_eq_2304_l157_157555


namespace marble_draw_probability_l157_157117

def probability_first_white_second_red : ℚ :=
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let first_white_probability := white_marbles / total_marbles
  let remaining_marbles_after_first_draw := total_marbles - 1
  let second_red_probability := red_marbles / remaining_marbles_after_first_draw
  first_white_probability * second_red_probability

theorem marble_draw_probability :
  probability_first_white_second_red = 4 / 15 := by
  sorry

end marble_draw_probability_l157_157117


namespace number_of_sides_of_polygon_l157_157984

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 40) : 
  (360 / exterior_angle) = 9 :=
by
  sorry

end number_of_sides_of_polygon_l157_157984


namespace subcommittee_count_l157_157745

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose republicans subcommittee_republicans * choose democrats subcommittee_democrats = 11760 :=
by
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end subcommittee_count_l157_157745


namespace david_total_hours_on_course_l157_157001

def hours_per_week_class := 2 * 3 + 4 -- hours per week in class
def hours_per_week_homework := 4 -- hours per week in homework
def total_hours_per_week := hours_per_week_class + hours_per_week_homework -- total hours per week

theorem david_total_hours_on_course :
  let total_weeks := 24
  in total_weeks * total_hours_per_week = 336 := by
  sorry

end david_total_hours_on_course_l157_157001


namespace percent_increase_is_fifteen_l157_157635

noncomputable def percent_increase_from_sale_price_to_regular_price (P : ℝ) : ℝ :=
  ((P - (0.87 * P)) / (0.87 * P)) * 100

theorem percent_increase_is_fifteen (P : ℝ) (h : P > 0) :
  percent_increase_from_sale_price_to_regular_price P = 15 :=
by
  -- The proof is not required, so we use sorry.
  sorry

end percent_increase_is_fifteen_l157_157635


namespace rahul_savings_l157_157698

variable (NSC PPF total_savings : ℕ)

theorem rahul_savings (h1 : NSC / 3 = PPF / 2) (h2 : PPF = 72000) : total_savings = 180000 :=
by
  sorry

end rahul_savings_l157_157698


namespace a_squared_plus_b_squared_plus_c_squared_geq_sqrt_3_l157_157248

theorem a_squared_plus_b_squared_plus_c_squared_geq_sqrt_3 (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) : a^2 + b^2 + c^2 ≥ real.sqrt 3 := 
by
  sorry

end a_squared_plus_b_squared_plus_c_squared_geq_sqrt_3_l157_157248


namespace number_of_schools_l157_157610

theorem number_of_schools (cost_per_school : ℝ) (population : ℝ) (savings_per_day_per_person : ℝ) (days_in_year : ℕ) :
  cost_per_school = 5 * 10^5 →
  population = 1.3 * 10^9 →
  savings_per_day_per_person = 0.01 →
  days_in_year = 365 →
  (population * savings_per_day_per_person * days_in_year) / cost_per_school = 9.49 * 10^3 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_schools_l157_157610


namespace exponentiation_property_l157_157080

variable (a : ℝ)

theorem exponentiation_property : a^2 * a^3 = a^5 := by
  sorry

end exponentiation_property_l157_157080


namespace min_value_expression_l157_157395

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + 4 / y) ≥ 9 :=
sorry

end min_value_expression_l157_157395


namespace algebraic_expression_value_l157_157707

theorem algebraic_expression_value (x : ℝ) (h : x = 4 * Real.sin (Real.pi / 4) - 2) :
  (1 / (x - 1) / (x + 2) / (x ^ 2 - 2 * x + 1) - x / (x + 2)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end algebraic_expression_value_l157_157707


namespace eliza_height_is_68_l157_157344

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end eliza_height_is_68_l157_157344


namespace range_of_independent_variable_of_sqrt_l157_157430

theorem range_of_independent_variable_of_sqrt (x : ℝ) : (2 * x - 3 ≥ 0) ↔ (x ≥ 3 / 2) := sorry

end range_of_independent_variable_of_sqrt_l157_157430


namespace range_of_independent_variable_l157_157680

noncomputable def function : ℝ → ℝ := λ x, (Real.sqrt (x - 1)) / (x - 2)

theorem range_of_independent_variable (x : ℝ) :
  (1 ≤ x ∧ x ≠ 2) ↔ ∃ y, y = function x := by
  sorry

end range_of_independent_variable_l157_157680


namespace probability_third_smallest_five_l157_157713

theorem probability_third_smallest_five :
  let n := 15
  let m := 10
  let k := 5
  (k = 3) → 
  (k ≤ m) → 
  (m ≤ n) → 
  let successful_arrangements := Nat.choose 4 2 * Nat.choose 10 7
  let total_ways := Nat.choose 15 10
  (successful_arrangements / total_ways: Rat) = 240 / 1001 := 
sorry

end probability_third_smallest_five_l157_157713


namespace bah_to_yah_conversion_l157_157201

theorem bah_to_yah_conversion :
  (10 : ℝ) * (1500 * (3/5) * (10/16)) / 16 = 562.5 := by
sorry

end bah_to_yah_conversion_l157_157201


namespace find_other_number_l157_157720

-- Given: 
-- LCM of two numbers is 2310
-- GCD of two numbers is 55
-- One number is 605,
-- Prove: The other number is 210

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 2310) (h_gcd : Nat.gcd a b = 55) (h_b : b = 605) :
  a = 210 :=
sorry

end find_other_number_l157_157720


namespace inequality_range_of_a_l157_157886

theorem inequality_range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (1 ≤ y ∧ y ≤ 3) → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by
  intros h
  sorry

end inequality_range_of_a_l157_157886


namespace inequality_l157_157249

-- Definition of the given condition
def condition (a b c : ℝ) : Prop :=
  a^2 * b * c + a * b^2 * c + a * b * c^2 = 1

-- Theorem to prove the inequality
theorem inequality (a b c : ℝ) (h : condition a b c) : a^2 + b^2 + c^2 ≥ real.sqrt 3 :=
sorry

end inequality_l157_157249


namespace find_initial_children_l157_157683

variables (x y : ℕ)

-- Defining the conditions 
def initial_children_on_bus (x : ℕ) : Prop :=
  ∃ y : ℕ, x - 68 + y = 12 ∧ 68 - y = 24 + y

-- Theorem statement
theorem find_initial_children : initial_children_on_bus x → x = 58 :=
by
  -- Skipping the proof for now
  sorry

end find_initial_children_l157_157683


namespace largest_number_in_set_l157_157772

theorem largest_number_in_set (b : ℕ) (h₀ : 2 + 6 + b = 18) (h₁ : 2 ≤ 6 ∧ 6 ≤ b):
  b = 10 :=
sorry

end largest_number_in_set_l157_157772


namespace gamma_suff_not_nec_for_alpha_l157_157823

variable {α β γ : Prop}

theorem gamma_suff_not_nec_for_alpha
  (h1 : β → α)
  (h2 : γ ↔ β) :
  (γ → α) ∧ (¬(α → γ)) :=
by {
  sorry
}

end gamma_suff_not_nec_for_alpha_l157_157823


namespace subcommittee_count_l157_157748

theorem subcommittee_count :
  let nR := 10 in 
  let nD := 8 in 
  let kR := 4 in 
  let kD := 3 in 
  (nat.choose nR kR) * (nat.choose nD kD) = 11760 := 
by 
  let nR := 10
  let nD := 8
  let kR := 4
  let kD := 3
  -- sorry replaces the actual proof steps
  sorry

end subcommittee_count_l157_157748


namespace circle_symmetry_l157_157882

theorem circle_symmetry {a : ℝ} (h : a ≠ 0) :
  ∀ {x y : ℝ}, (x^2 + y^2 + 2*a*x - 2*a*y = 0) → (x + y = 0) :=
sorry

end circle_symmetry_l157_157882


namespace uncovered_area_is_8_l157_157239

-- Conditions
def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side_length : ℕ := 4

-- Theorem to prove
theorem uncovered_area_is_8
  (sh_height : ℕ := shoebox_height)
  (sh_width : ℕ := shoebox_width)
  (bl_length : ℕ := block_side_length)
  : sh_height * sh_width - bl_length * bl_length = 8 :=
by {
  -- Placeholder for proof; we are not proving it as per instructions.
  sorry
}

end uncovered_area_is_8_l157_157239


namespace unique_solution_exists_l157_157169

def f (x y z : ℕ) : ℕ := (x + y - 2) * (x + y - 1) / 2 - z

theorem unique_solution_exists :
  ∀ (a b c d : ℕ), f a b c = 1993 ∧ f c d a = 1993 → (a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42) :=
by
  intros a b c d h
  sorry

end unique_solution_exists_l157_157169


namespace first_valve_time_l157_157105

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end first_valve_time_l157_157105


namespace parabola_above_line_l157_157649

variable (a b c : ℝ) (h : (b - c)^2 - 4 * a * c < 0)

theorem parabola_above_line : (b - c)^2 - 4 * a * c < 0 → (b - c)^2 - 4 * c * (a + b) < 0 :=
by sorry

end parabola_above_line_l157_157649


namespace sequence_sum_l157_157954

-- Definitions representing the given conditions
variables (A H M O X : ℕ)

-- Assuming the conditions as hypotheses
theorem sequence_sum (h₁ : A + 9 + H = 19) (h₂ : 9 + H + M = 19) (h₃ : H + M + O = 19)
  (h₄ : M + O + X = 19) : A + H + M + O = 26 :=
sorry

end sequence_sum_l157_157954


namespace cube_properties_l157_157741

theorem cube_properties (s y : ℝ) (h1 : s^3 = 8 * y) (h2 : 6 * s^2 = 6 * y) : y = 64 := by
  sorry

end cube_properties_l157_157741


namespace mole_fractions_C4H8O2_l157_157472

/-- 
Given:
- The molecular formula of C4H8O2,
- 4 moles of carbon (C) atoms,
- 8 moles of hydrogen (H) atoms,
- 2 moles of oxygen (O) atoms.

Prove that:
The mole fractions of each element in C4H8O2 are:
- Carbon (C): 2/7
- Hydrogen (H): 4/7
- Oxygen (O): 1/7
--/
theorem mole_fractions_C4H8O2 :
  let m_C := 4
  let m_H := 8
  let m_O := 2
  let total_moles := m_C + m_H + m_O
  let mole_fraction_C := m_C / total_moles
  let mole_fraction_H := m_H / total_moles
  let mole_fraction_O := m_O / total_moles
  mole_fraction_C = 2 / 7 ∧ mole_fraction_H = 4 / 7 ∧ mole_fraction_O = 1 / 7 := by
  sorry

end mole_fractions_C4H8O2_l157_157472


namespace ellipse_eccentricity_l157_157809

theorem ellipse_eccentricity (a c : ℝ) (h : 2 * a = 2 * (2 * c)) : (c / a) = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l157_157809


namespace tan_product_pi_8_l157_157469

theorem tan_product_pi_8 :
  (Real.tan (π / 8)) * (Real.tan (3 * π / 8)) * (Real.tan (5 * π / 8)) * (Real.tan (7 * π / 8)) = 1 :=
sorry

end tan_product_pi_8_l157_157469


namespace find_x_l157_157667

theorem find_x (x : ℕ) (h1 : x % 6 = 0) (h2 : x^2 > 144) (h3 : x < 30) : x = 18 ∨ x = 24 :=
sorry

end find_x_l157_157667


namespace application_form_choices_l157_157291

theorem application_form_choices (majors : Finset ℕ) (A : ℕ) (h_majors : majors.card = 7) (h_A : A ∈ majors) :
  let available_majors := majors.erase A in
  let ways_first_two := Finset.choose available_majors 2 in
  let ways_next_three := Finset.choose majors.erase (Finset.erase available_majors) 3 in
  let total_ways := ways_first_two.card * ways_next_three.card in
  total_ways = 150 :=
by
  sorry

end application_form_choices_l157_157291


namespace quadratic_inequality_solution_l157_157173

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) :=
by sorry

end quadratic_inequality_solution_l157_157173


namespace compare_fractions_l157_157313

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l157_157313


namespace problem_statement_l157_157780

open Real

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sin θ)

theorem problem_statement (A B : ℝ × ℝ) 
  (θA θB : ℝ) 
  (hA : A = curve_C θA) 
  (hB : B = curve_C θB) 
  (h_perpendicular : θB = θA + π / 2) :
  (1 / (A.1 ^ 2 + A.2 ^ 2)) + (1 / (B.1 ^ 2 + B.2 ^ 2)) = 5 / 4 := by
  sorry

end problem_statement_l157_157780


namespace initial_bacteria_count_l157_157242

theorem initial_bacteria_count :
  ∀ (n : ℕ), (n * 5^8 = 1953125) → n = 5 :=
by
  intro n
  intro h
  sorry

end initial_bacteria_count_l157_157242


namespace probability_of_five_is_max_l157_157281

def bag : finset ℕ := {1, 2, 3, 4, 5, 6}

def cards_selected : finset (finset ℕ) :=
  finset.powersetLen 4 bag

def five_is_max (s : finset ℕ) : Prop :=
  ∃ k ∈ s, k = 5 ∧ ∀ m ∈ s, m ≤ 5

theorem probability_of_five_is_max :
  let total_ways := (cards_selected.card : ℚ),
      favorable_ways := (finset.filter five_is_max cards_selected).card
  in favorable_ways / total_ways = 4 / 15 :=
by
  sorry

end probability_of_five_is_max_l157_157281


namespace sub_neg_eq_add_pos_l157_157621

theorem sub_neg_eq_add_pos : 0 - (-2) = 2 := 
by
  sorry

end sub_neg_eq_add_pos_l157_157621


namespace compare_neg_fractions_l157_157323

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l157_157323


namespace common_ratio_of_geometric_series_l157_157300

theorem common_ratio_of_geometric_series (a S r : ℝ) (h1 : a = 500) (h2 : S = 2500) (h3 : a / (1 - r) = S) : r = 4 / 5 :=
by
  rw [h1, h2] at h3
  sorry

end common_ratio_of_geometric_series_l157_157300


namespace problem_1_problem_2_problem_3_l157_157193

open Set Real

def U : Set ℝ := univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | -a < x ∧ x ≤ a + 3 }

theorem problem_1 :
  (A ∪ B) = { x | 1 ≤ x ∧ x < 8 } :=
sorry

theorem problem_2 :
  (U \ A) ∩ B = { x | 5 ≤ x ∧ x < 8 } :=
sorry

theorem problem_3 (a : ℝ) (h : C a ∩ A = C a) :
  a ≤ -1 :=
sorry

end problem_1_problem_2_problem_3_l157_157193


namespace max_balls_drawn_l157_157852

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l157_157852


namespace eliza_height_l157_157342

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end eliza_height_l157_157342


namespace snacks_in_3h40m_l157_157848

def minutes_in_hours (hours : ℕ) : ℕ := hours * 60

def snacks_in_time (total_minutes : ℕ) (snack_interval : ℕ) : ℕ := total_minutes / snack_interval

theorem snacks_in_3h40m : snacks_in_time (minutes_in_hours 3 + 40) 20 = 11 :=
by
  sorry

end snacks_in_3h40m_l157_157848


namespace subcommittee_count_l157_157753

theorem subcommittee_count :
  (Nat.choose 10 4) * (Nat.choose 8 3) = 11760 :=
by
  sorry

end subcommittee_count_l157_157753


namespace provisions_remaining_days_l157_157286

-- Definitions based on the conditions
def initial_men : ℕ := 1000
def initial_provisions_days : ℕ := 60
def days_elapsed : ℕ := 15
def reinforcement_men : ℕ := 1250

-- Mathematical computation for Lean
def total_provisions : ℕ := initial_men * initial_provisions_days
def provisions_left : ℕ := initial_men * (initial_provisions_days - days_elapsed)
def total_men_after_reinforcement : ℕ := initial_men + reinforcement_men

-- Statement to prove
theorem provisions_remaining_days : provisions_left / total_men_after_reinforcement = 20 :=
by
  -- The proof steps will be filled here, but for now, we use sorry to skip them.
  sorry

end provisions_remaining_days_l157_157286


namespace marble_draw_probability_l157_157116

def probability_first_white_second_red : ℚ :=
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let first_white_probability := white_marbles / total_marbles
  let remaining_marbles_after_first_draw := total_marbles - 1
  let second_red_probability := red_marbles / remaining_marbles_after_first_draw
  first_white_probability * second_red_probability

theorem marble_draw_probability :
  probability_first_white_second_red = 4 / 15 := by
  sorry

end marble_draw_probability_l157_157116


namespace not_an_algorithm_option_B_l157_157101

def is_algorithm (description : String) : Prop :=
  description = "clear and finite steps to solve a problem producing correct results when executed by a computer"

def operation_to_string (option : Char) : String :=
  match option with
  | 'A' => "Calculating the area of a circle given its radius"
  | 'B' => "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
  | 'C' => "Finding the equation of a line given two points in the coordinate plane"
  | 'D' => "The rules of addition, subtraction, multiplication, and division"
  | _ => ""

noncomputable def categorize_operation (option : Char) : Prop :=
  option = 'B' ↔ ¬ is_algorithm (operation_to_string option)

theorem not_an_algorithm_option_B :
  categorize_operation 'B' :=
by
  sorry

end not_an_algorithm_option_B_l157_157101


namespace vectors_parallel_l157_157517

-- Let s and n be the direction vector and normal vector respectively
def s : ℝ × ℝ × ℝ := (2, 1, 1)
def n : ℝ × ℝ × ℝ := (-4, -2, -2)

-- Statement that vectors s and n are parallel
theorem vectors_parallel : ∃ (k : ℝ), n = (k • s) := by
  use -2
  simp [s, n]
  sorry

end vectors_parallel_l157_157517


namespace subcommittee_ways_l157_157755

theorem subcommittee_ways : 
  let R := 10 in
  let D := 8 in
  let kR := 4 in
  let kD := 3 in
  (Nat.choose R kR) * (Nat.choose D kD) = 11760 :=
by
  sorry

end subcommittee_ways_l157_157755


namespace team_B_at_least_half_can_serve_l157_157908

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l157_157908


namespace swapped_coefficients_have_roots_l157_157974

theorem swapped_coefficients_have_roots 
  (a b c p q r : ℝ)
  (h1 : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0))
  (h2 : ∀ x : ℝ, ¬ (p * x^2 + q * x + r = 0))
  (h3 : b^2 < 4 * p * c)
  (h4 : q^2 < 4 * a * r) :
  ∃ x : ℝ, a * x^2 + q * x + c = 0 ∧ ∃ y : ℝ, p * y^2 + b * y + r = 0 :=
by
  sorry

end swapped_coefficients_have_roots_l157_157974


namespace number_of_teachers_l157_157598

theorem number_of_teachers
    (number_of_students : ℕ)
    (classes_per_student : ℕ)
    (classes_per_teacher : ℕ)
    (students_per_class : ℕ)
    (total_teachers : ℕ)
    (h1 : number_of_students = 2400)
    (h2 : classes_per_student = 5)
    (h3 : classes_per_teacher = 4)
    (h4 : students_per_class = 30)
    (h5 : total_teachers * classes_per_teacher * students_per_class = number_of_students * classes_per_student) :
    total_teachers = 100 :=
by
  sorry

end number_of_teachers_l157_157598


namespace probability_of_hypotenuse_le_one_l157_157229

open MeasureTheory

noncomputable def probability_hypotenuse_le_one : ℝ :=
  measure (set_of (λ (p : ℝ × ℝ), 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1 ∧ p.1^2 + p.2^2 ≤ 1)) / 
  measure (set_of (λ (p : ℝ × ℝ), 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1))

theorem probability_of_hypotenuse_le_one : 
  probability_hypotenuse_le_one = π / 4 :=
sorry

end probability_of_hypotenuse_le_one_l157_157229


namespace large_box_total_chocolate_bars_l157_157450

def number_of_small_boxes : ℕ := 15
def chocolate_bars_per_small_box : ℕ := 20
def total_chocolate_bars (n : ℕ) (m : ℕ) : ℕ := n * m

theorem large_box_total_chocolate_bars :
  total_chocolate_bars number_of_small_boxes chocolate_bars_per_small_box = 300 :=
by
  sorry

end large_box_total_chocolate_bars_l157_157450


namespace David_min_max_rides_l157_157460

-- Definitions based on the conditions
variable (Alena_rides : ℕ := 11)
variable (Bara_rides : ℕ := 20)
variable (Cenek_rides : ℕ := 4)
variable (every_pair_rides_at_least_once : Prop := true)

-- Hypotheses for the problem
axiom Alena_has_ridden : Alena_rides = 11
axiom Bara_has_ridden : Bara_rides = 20
axiom Cenek_has_ridden : Cenek_rides = 4
axiom Pairs_have_ridden : every_pair_rides_at_least_once

-- Statement for the minimum and maximum rides of David
theorem David_min_max_rides (David_rides : ℕ) :
  (David_rides = 11) ∨ (David_rides = 29) :=
sorry

end David_min_max_rides_l157_157460


namespace sum_of_powers_l157_157830

theorem sum_of_powers (a b : ℝ) (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 72 := 
by
  sorry

end sum_of_powers_l157_157830


namespace parallel_vectors_eq_l157_157031

theorem parallel_vectors_eq (m : ℤ) (h : (m, 4) = (3 * k, -2 * k)) : m = -6 :=
by
  sorry

end parallel_vectors_eq_l157_157031


namespace find_x_l157_157037

theorem find_x (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 :=
sorry

end find_x_l157_157037


namespace problem_statement_l157_157970

noncomputable
def parabola := λ (x : ℝ), x^2

noncomputable
def circle (h : ℝ) := λ (x : ℝ y : ℝ), x^2 + (y - h)^2 = 1

theorem problem_statement : 
    let center : ℝ × ℝ := (0, 5 / 4)
    let area := (sqrt 3) - (π / 3)
    in
    (circle (5 / 4) = (λ (x : ℝ y : ℝ), x^2 + (y - (5 / 4))^2 = 1)) ∧
    (∀ x, parabola x = x^2) ∧
    ∃ a b, ∫ t within 0..a.length, ((5 / 4) - sqrt(1 - t^2) - t^2) = area :=
begin
    sorry
end

end problem_statement_l157_157970


namespace minute_hand_length_l157_157594

theorem minute_hand_length (r : ℝ) (h : 20 * (2 * Real.pi / 60) * r = Real.pi / 3) : r = 1 / 2 :=
by
  sorry

end minute_hand_length_l157_157594


namespace common_face_sum_is_9_l157_157083

noncomputable def common_sum (vertices : Fin 9 → ℕ) : ℕ :=
  let total_sum := (Finset.sum (Finset.univ : Finset (Fin 9)) vertices)
  let additional_sum := 9
  let total_with_addition := total_sum + additional_sum
  total_with_addition / 6

theorem common_face_sum_is_9 :
  ∀ (vertices : Fin 9 → ℕ), (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 9) →
  Finset.sum (Finset.univ : Finset (Fin 9)) vertices = 45 →
  common_sum vertices = 9 := 
by
  intros vertices h1 h_sum
  unfold common_sum
  sorry

end common_face_sum_is_9_l157_157083


namespace odd_function_value_sum_l157_157808

theorem odd_function_value_sum
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fneg1 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end odd_function_value_sum_l157_157808


namespace negative_fraction_comparison_l157_157309

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l157_157309


namespace probability_three_digit_multiple_5_remainder_3_div_7_l157_157166

theorem probability_three_digit_multiple_5_remainder_3_div_7 :
  (∃ (P : ℝ), P = (26 / 900)) := 
by sorry

end probability_three_digit_multiple_5_remainder_3_div_7_l157_157166


namespace sam_pennies_total_l157_157704

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gifted_pennies : ℕ := 250

theorem sam_pennies_total :
  initial_pennies + found_pennies - exchanged_pennies + gifted_pennies = 1435 := 
sorry

end sam_pennies_total_l157_157704


namespace emails_in_inbox_l157_157077

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end emails_in_inbox_l157_157077


namespace percent_in_range_70_to_79_is_correct_l157_157125

-- Define the total number of students.
def total_students : Nat := 8 + 12 + 11 + 5 + 7

-- Define the number of students within the $70\%-79\%$ range.
def students_70_to_79 : Nat := 11

-- Define the percentage of the students within the $70\%-79\%$ range.
def percent_70_to_79 : ℚ := (students_70_to_79 : ℚ) / (total_students : ℚ) * 100

theorem percent_in_range_70_to_79_is_correct : percent_70_to_79 = 25.58 := by
  sorry

end percent_in_range_70_to_79_is_correct_l157_157125


namespace units_digit_fraction_l157_157266

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10000 % 10 = 4 :=
by
  -- Placeholder for actual proof
  sorry

end units_digit_fraction_l157_157266


namespace find_m_correct_l157_157059

structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  XY_length : dist X Y = 80
  XZ_length : dist X Z = 100
  YZ_length : dist Y Z = 120

noncomputable def find_m (t : Triangle) : ℝ :=
  let s := (80 + 100 + 120) / 2
  let A := 1 / 2 * 80 * 100
  let r1 := A / s
  let r2 := r1 / 2
  let r3 := r1 / 4
  let O2 := ((40 / 3), 50 + (40 / 3))
  let O3 := (40 + (20 / 3), (20 / 3))
  let O2O3 := dist O2 O3
  let m := (O2O3^2) / 10
  m

theorem find_m_correct (t : Triangle) : find_m t = 610 := sorry

end find_m_correct_l157_157059


namespace negative_fraction_comparison_l157_157307

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l157_157307


namespace number_of_children_l157_157637

-- Definition of the conditions
def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 30

-- Theorem statement
theorem number_of_children (n : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  n = total_pencils / pencils_per_child :=
by
  have h : n = 30 / 2 := sorry
  exact h

end number_of_children_l157_157637


namespace compare_neg_fractions_l157_157324

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l157_157324


namespace cindy_correct_answer_l157_157304

theorem cindy_correct_answer (x : ℝ) (h : (x - 5) / 7 = 15) :
  (x - 7) / 5 = 20.6 :=
by
  sorry

end cindy_correct_answer_l157_157304


namespace Lisa_favorite_number_l157_157097

theorem Lisa_favorite_number (a b : ℕ) (h : 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b)^2 = (a + b)^3 → 10 * a + b = 27 := by
  intro h_eq
  sorry

end Lisa_favorite_number_l157_157097


namespace factorization_exists_l157_157837

-- Define the polynomial f(x)
def f (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 12

-- Definition for polynomial g(x)
def g (a : ℤ) (x : ℚ) : ℚ := x^2 + a*x + 3

-- Definition for polynomial h(x)
def h (b : ℤ) (x : ℚ) : ℚ := x^2 + b*x + 4

-- The main statement to prove
theorem factorization_exists :
  ∃ (a b : ℤ), (∀ x, f x = (g a x) * (h b x)) :=
by
  sorry

end factorization_exists_l157_157837


namespace seed_production_l157_157413

theorem seed_production :
  ∀ (initial_seeds : ℕ) (germination_rate : ℝ) (seed_count_per_plant : ℕ),
    initial_seeds = 1 →
    germination_rate = 0.5 →
    seed_count_per_plant = 50 →
    let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate in
    new_plants * seed_count_per_plant = 1250 :=
by
  intros initial_seeds germination_rate seed_count_per_plant h1 h2 h3
  let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate
  have : new_plants = 25, by {
    rw [h1, h3, h2],
    norm_num,
  }
  rw this
  norm_num

end seed_production_l157_157413


namespace lucky_larry_l157_157067

theorem lucky_larry (a b c d e k : ℤ) 
    (h1 : a = 2) 
    (h2 : b = 3) 
    (h3 : c = 4) 
    (h4 : d = 5)
    (h5 : a - b - c - d + e = 2 - (b - (c - (d + e)))) 
    (h6 : k * 2 = e) : 
    k = 2 := by
  sorry

end lucky_larry_l157_157067


namespace sales_in_fourth_month_l157_157766

theorem sales_in_fourth_month
  (sale1 : ℕ)
  (sale2 : ℕ)
  (sale3 : ℕ)
  (sale5 : ℕ)
  (sale6 : ℕ)
  (average : ℕ)
  (h_sale1 : sale1 = 2500)
  (h_sale2 : sale2 = 6500)
  (h_sale3 : sale3 = 9855)
  (h_sale5 : sale5 = 7000)
  (h_sale6 : sale6 = 11915)
  (h_average : average = 7500) :
  ∃ sale4 : ℕ, sale4 = 14230 := by
  sorry

end sales_in_fourth_month_l157_157766


namespace length_EF_eq_diameter_Γ_l157_157493

open_locale classical

noncomputable theory

variables {O1 O2 A B C D E F : EuclideanGeometry.Point ℝ} {Γ : EuclideanGeometry.Circle ℝ}

/-- Given two circles intersecting at two points, a line through the intersection point with tangents
intersecting on another circle, prove a specific segment length is the diameter of the circle. -/
theorem length_EF_eq_diameter_Γ
  (h1 : EuclideanGeometry.circle O1)
  (h2 : EuclideanGeometry.circle O2)
  (hA : EuclideanGeometry.point_on_circle A h1)
  (hB : EuclideanGeometry.point_on_circle A h2)
  (hC : EuclideanGeometry.line_through B intersects h1 at C)
  (hD : EuclideanGeometry.line_through B intersects h2 at D)
  (hE : EuclideanGeometry.tangent_to_circle C from h1 meets
        EuclideanGeometry.tangent_to_circle D from h2 at E)
  (hΓ : EuclideanGeometry.circumcircle_of_triangle A O1 O2 = Γ)
  (hF : EuclideanGeometry.line_through A E intersects Γ at F)
  : EuclideanGeometry.length_segment E F = EuclideanGeometry.diameter Γ :=
sorry

end length_EF_eq_diameter_Γ_l157_157493


namespace largest_band_members_l157_157769

theorem largest_band_members 
  (r x m : ℕ) 
  (h1 : (r * x + 3 = m)) 
  (h2 : ((r - 3) * (x + 1) = m))
  (h3 : m < 100) : 
  m = 75 :=
sorry

end largest_band_members_l157_157769


namespace distance_on_dirt_road_is_1_km_l157_157440

variable (initial_gap : ℝ) (highway_speed : ℝ) (city_speed : ℝ) (good_road_speed : ℝ) (dirt_road_speed : ℝ)

def distance_between_on_dirt_road (initial_gap : ℝ) (highway_speed : ℝ) (city_speed : ℝ) (good_road_speed : ℝ) (dirt_road_speed : ℝ) : ℝ :=
  initial_gap * (city_speed / highway_speed) * (good_road_speed / city_speed) * (dirt_road_speed / good_road_speed)

theorem distance_on_dirt_road_is_1_km :
  distance_between_on_dirt_road 2 60 40 70 30 = 1 :=
  by
    unfold distance_between_on_dirt_road
    sorry

end distance_on_dirt_road_is_1_km_l157_157440


namespace swan_populations_after_10_years_l157_157617

noncomputable def swan_population_rita (R : ℝ) : ℝ :=
  480 * (1 - R / 100) ^ 10

noncomputable def swan_population_sarah (S : ℝ) : ℝ :=
  640 * (1 - S / 100) ^ 10

noncomputable def swan_population_tom (T : ℝ) : ℝ :=
  800 * (1 - T / 100) ^ 10

theorem swan_populations_after_10_years 
  (R S T : ℝ) :
  swan_population_rita R = 480 * (1 - R / 100) ^ 10 ∧
  swan_population_sarah S = 640 * (1 - S / 100) ^ 10 ∧
  swan_population_tom T = 800 * (1 - T / 100) ^ 10 := 
by sorry

end swan_populations_after_10_years_l157_157617


namespace compare_neg_fractions_l157_157326

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l157_157326


namespace nth_equation_l157_157069

open Nat

theorem nth_equation (n : ℕ) (hn : 0 < n) :
  (n + 1)/((n + 1) * (n + 1) - 1) - (1/(n * (n + 1) * (n + 2))) = 1/(n + 1) := 
by
  sorry

end nth_equation_l157_157069


namespace scientific_notation_of_population_l157_157070

theorem scientific_notation_of_population : (85000000 : ℝ) = 8.5 * 10^7 := 
by
  sorry

end scientific_notation_of_population_l157_157070


namespace sum_eq_zero_l157_157804

variable {a b c : ℝ}

theorem sum_eq_zero (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
    (h4 : a ≠ b ∨ b ≠ c ∨ c ≠ a)
    (h5 : (a^2) / (2 * (a^2) + b * c) + (b^2) / (2 * (b^2) + c * a) + (c^2) / (2 * (c^2) + a * b) = 1) :
  a + b + c = 0 :=
sorry

end sum_eq_zero_l157_157804


namespace word_identification_l157_157575

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end word_identification_l157_157575


namespace cellini_inscription_l157_157735

noncomputable def famous_master_engravings (x: Type) : String :=
  "Эту шкатулку изготовил сын Челлини"

theorem cellini_inscription (x: Type) (created_by_cellini : x) :
  famous_master_engravings x = "Эту шкатулку изготовил сын Челлини" :=
by
  sorry

end cellini_inscription_l157_157735


namespace length_of_goods_train_l157_157925

-- Define the given data
def speed_kmph := 72
def platform_length_m := 250
def crossing_time_s := 36

-- Convert speed from kmph to m/s
def speed_mps := speed_kmph * (5 / 18)

-- Define the total distance covered while crossing the platform
def distance_covered_m := speed_mps * crossing_time_s

-- Define the length of the train
def train_length_m := distance_covered_m - platform_length_m

-- The theorem to be proven
theorem length_of_goods_train : train_length_m = 470 := by
  sorry

end length_of_goods_train_l157_157925


namespace average_of_remaining_numbers_l157_157559

theorem average_of_remaining_numbers (S : ℕ) 
  (h₁ : S = 85 * 10) 
  (S' : ℕ) 
  (h₂ : S' = S - 70 - 76) : 
  S' / 8 = 88 := 
sorry

end average_of_remaining_numbers_l157_157559


namespace abs_sum_zero_implies_diff_eq_five_l157_157040

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end abs_sum_zero_implies_diff_eq_five_l157_157040


namespace team_B_eligible_l157_157913

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l157_157913


namespace compare_rat_neg_l157_157330

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l157_157330


namespace mary_needs_more_cups_l157_157693

theorem mary_needs_more_cups (total_cups required_cups added_cups : ℕ) (h1 : required_cups = 8) (h2 : added_cups = 2) : total_cups = 6 :=
by
  sorry

end mary_needs_more_cups_l157_157693


namespace largest_number_l157_157558

def HCF (a b c d : ℕ) : Prop := d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ 
                                ∀ e, (e ∣ a ∧ e ∣ b ∧ e ∣ c) → e ≤ d
def LCM (a b c m : ℕ) : Prop := m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧ 
                                ∀ n, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem largest_number (a b c : ℕ)
  (hcf: HCF a b c 210)
  (lcm_has_factors: ∃ k1 k2 k3, k1 = 11 ∧ k2 = 17 ∧ k3 = 23 ∧
                                LCM a b c (210 * k1 * k2 * k3)) :
  max a (max b c) = 4830 := 
by
  sorry

end largest_number_l157_157558


namespace at_least_half_team_B_can_serve_on_submarine_l157_157904

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l157_157904


namespace valid_three_digit_numbers_count_l157_157820

def count_three_digit_numbers : ℕ := 900

def count_invalid_numbers : ℕ := (90 + 90 - 9)

def count_valid_three_digit_numbers : ℕ := 900 - (90 + 90 - 9)

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 729 :=
by
  show 900 - (90 + 90 - 9) = 729
  sorry

end valid_three_digit_numbers_count_l157_157820


namespace fraction_of_short_students_l157_157380

theorem fraction_of_short_students 
  (total_students tall_students average_students : ℕ) 
  (htotal : total_students = 400) 
  (htall : tall_students = 90) 
  (haverage : average_students = 150) : 
  (total_students - (tall_students + average_students)) / total_students = 2 / 5 :=
by
  sorry

end fraction_of_short_students_l157_157380


namespace find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l157_157928

theorem find_k_and_max_ck:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    ∃ (c_k : ℝ), c_k > 0 ∧ (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k) →
  (∀ (k : ℝ), 0 ≤ k ∧ k ≤ 2) :=
by
  sorry

theorem largest_ck_for_k0:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ 1) := 
by
  sorry

theorem largest_ck_for_k2:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ (8/9) * (x + y + z)^2) :=
by
  sorry

end find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l157_157928


namespace no_prime_solution_in_2_to_7_l157_157236

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_solution_in_2_to_7 : ∀ p : ℕ, is_prime p ∧ 2 ≤ p ∧ p ≤ 7 → (2 * p^3 - p^2 - 15 * p + 22) ≠ 0 :=
by
  intros p hp
  have h := hp.left
  sorry

end no_prime_solution_in_2_to_7_l157_157236


namespace find_M_l157_157088

variable (p q r M : ℝ)
variable (h1 : p + q + r = 100)
variable (h2 : p + 10 = M)
variable (h3 : q - 5 = M)
variable (h4 : r / 5 = M)

theorem find_M : M = 15 := by
  sorry

end find_M_l157_157088


namespace find_analytical_expression_l157_157352

open Real

-- Definitions for the function and the conditions
axiom f : ℝ → ℝ
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_frac : f (2 / 3) = 4

-- The target theorem to be proven
theorem find_analytical_expression : ∀ x, f x = 8 ^ x :=
by
  sorry

end find_analytical_expression_l157_157352


namespace domain_of_function_l157_157784

theorem domain_of_function (x : ℝ) : 4 - x ≥ 0 ∧ x ≠ 2 ↔ (x ≤ 4 ∧ x ≠ 2) :=
sorry

end domain_of_function_l157_157784


namespace exists_constant_not_geometric_l157_157889

-- Definitions for constant and geometric sequences
def is_constant_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, seq n = c

def is_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, seq (n + 1) = r * seq n

-- The negation problem statement
theorem exists_constant_not_geometric :
  ∃ seq : ℕ → ℝ, is_constant_sequence seq ∧ ¬is_geometric_sequence seq :=
sorry

end exists_constant_not_geometric_l157_157889


namespace sample_size_l157_157605

-- Define the given conditions
def number_of_male_athletes : Nat := 42
def number_of_female_athletes : Nat := 30
def sampled_female_athletes : Nat := 5

-- Define the target total sample size
def total_sample_size (male_athletes female_athletes sample_females : Nat) : Nat :=
  sample_females * male_athletes / female_athletes + sample_females

-- State the theorem to prove
theorem sample_size (h1: number_of_male_athletes = 42) 
                    (h2: number_of_female_athletes = 30)
                    (h3: sampled_female_athletes = 5) :
  total_sample_size number_of_male_athletes number_of_female_athletes sampled_female_athletes = 12 :=
by
  -- Proof is omitted
  sorry

end sample_size_l157_157605


namespace lottery_probability_theorem_l157_157729

noncomputable theory

def total_ways_to_draw : ℕ := 120
def ways_to_end_after_fourth_draw : ℕ := 36
def probability_event_ends_after_fourth_draw (total ways ways_to_end : ℕ) : ℚ :=
  ways_to_end / total

theorem lottery_probability_theorem :
  probability_event_ends_after_fourth_draw total_ways_to_draw ways_to_end_after_fourth_draw = 3 / 10 := 
  by 
  sorry

end lottery_probability_theorem_l157_157729


namespace triangle_inequality_l157_157699

variables {a b c : ℝ} {α : ℝ}

-- Assuming a, b, c are sides of a triangle
def triangle_sides (a b c : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Cosine rule definition
noncomputable def cos_alpha (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)

theorem triangle_inequality (h_sides: triangle_sides a b c) (h_cos : α = cos_alpha a b c) :
  (2 * b * c * (cos_alpha a b c)) / (b + c) < b + c - a
  ∧ b + c - a < 2 * b * c / a :=
by
  sorry

end triangle_inequality_l157_157699


namespace total_height_increase_in_4_centuries_l157_157979

def height_increase_per_decade : ℕ := 75
def years_per_century : ℕ := 100
def years_per_decade : ℕ := 10
def centuries : ℕ := 4

theorem total_height_increase_in_4_centuries :
  height_increase_per_decade * (centuries * years_per_century / years_per_decade) = 3000 := by
  sorry

end total_height_increase_in_4_centuries_l157_157979


namespace max_balls_possible_l157_157858

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l157_157858


namespace parabola_coefficients_sum_l157_157879

theorem parabola_coefficients_sum :
  ∃ a b c : ℝ, 
  (∀ y : ℝ, (7 = -(6 ^ 2) * a + b * 6 + c)) ∧
  (5 = a * (-4) ^ 2 + b * (-4) + c) ∧
  (a + b + c = -42) := 
sorry

end parabola_coefficients_sum_l157_157879


namespace determine_pairs_l157_157336

-- Definitions:
variable (c d : ℝ) (a : ℕ → ℝ)

-- Conditions:
def condition1 := ∀ n ≥ 1, a n > 0
def condition2 := ∀ n ≥ 1, a n ≥ c * a (n + 1) + d * (∑ j in Finset.range (n-1), a j)

-- Lean theorem statement:
theorem determine_pairs (h1 : condition1 a) (h2 : condition2 c d a) :
  (c ≤ 0) ∨ (d ≤ 0) ∨ (0 < c ∧ c < 1 ∧ d ≤ (c - 1)^2 / (4 * c)) :=
begin
  sorry  -- Proof goes here
end

end determine_pairs_l157_157336


namespace largest_non_zero_ending_factor_decreasing_number_l157_157956

theorem largest_non_zero_ending_factor_decreasing_number :
  ∃ n: ℕ, n = 180625 ∧ (n % 10 ≠ 0) ∧ (∃ m: ℕ, m < n ∧ (n % m = 0) ∧ (n / 10 ≤ m ∧ m * 10 > 0)) :=
by {
  sorry
}

end largest_non_zero_ending_factor_decreasing_number_l157_157956


namespace find_k_l157_157345

theorem find_k : ∃ k : ℕ, ∀ n : ℕ, n > 0 → (2^n + 11) % (2^k - 1) = 0 ↔ k = 4 :=
by
  sorry

end find_k_l157_157345


namespace new_average_amount_l157_157877

theorem new_average_amount (A : ℝ) (H : A = 14) (new_amount : ℝ) (H1 : new_amount = 56) : 
  ((7 * A + new_amount) / 8) = 19.25 :=
by
  rw [H, H1]
  norm_num

end new_average_amount_l157_157877


namespace max_blocks_fit_l157_157264

-- Define the dimensions of the block and the box
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the volumes calculation
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

-- Define the dimensions of the block and the box
def block : Dimensions := { length := 3, width := 1, height := 2 }
def box : Dimensions := { length := 4, width := 3, height := 6 }

-- Prove that the maximum number of blocks that can fit in the box is 12
theorem max_blocks_fit : (volume box) / (volume block) = 12 := by sorry

end max_blocks_fit_l157_157264


namespace death_rate_is_three_l157_157209

-- Let birth_rate be the average birth rate in people per two seconds
def birth_rate : ℕ := 6
-- Let net_population_increase be the net population increase per day
def net_population_increase : ℕ := 129600
-- Let seconds_per_day be the total number of seconds in a day
def seconds_per_day : ℕ := 86400

noncomputable def death_rate_per_two_seconds : ℕ :=
  let net_increase_per_second := net_population_increase / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  2 * (birth_rate_per_second - net_increase_per_second)

theorem death_rate_is_three :
  death_rate_per_two_seconds = 3 := by
  sorry

end death_rate_is_three_l157_157209


namespace julia_fourth_day_candies_l157_157841

-- Definitions based on conditions
def first_day (x : ℚ) := (1/5) * x
def second_day (x : ℚ) := (1/2) * (4/5) * x
def third_day (x : ℚ) := (1/2) * (2/5) * x
def fourth_day (x : ℚ) := (2/5) * x - (1/2) * (2/5) * x

-- The Lean statement to prove
theorem julia_fourth_day_candies (x : ℚ) (h : x ≠ 0): 
  fourth_day x / x = 1/5 :=
by
  -- insert proof here
  sorry

end julia_fourth_day_candies_l157_157841


namespace min_sum_intercepts_l157_157657

theorem min_sum_intercepts (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (1 : ℝ) * a + (1 : ℝ) * b = a * b) : a + b = 4 :=
by
  sorry

end min_sum_intercepts_l157_157657


namespace isosceles_triangle_count_l157_157471

noncomputable def valid_points : List (ℕ × ℕ) :=
  [(2, 5), (5, 5)]

theorem isosceles_triangle_count 
  (A B : ℕ × ℕ) 
  (H_A : A = (2, 2)) 
  (H_B : B = (5, 2)) : 
  valid_points.length = 2 :=
  sorry

end isosceles_triangle_count_l157_157471


namespace problem_solution_l157_157019

noncomputable def circles_intersect (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (A ∈ { p | p.1^2 + p.2^2 = 1 }) ∧ (B ∈ { p | p.1^2 + p.2^2 = 1 }) ∧
  (A ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ (B ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ 
  (dist A B = (4 * Real.sqrt 5) / 5)

theorem problem_solution (m : ℝ) : circles_intersect m ↔ (m = 1 ∨ m = -3) := by
  sorry

end problem_solution_l157_157019


namespace average_marks_physics_chemistry_l157_157290

theorem average_marks_physics_chemistry
  (P C M : ℕ)
  (h1 : (P + C + M) / 3 = 60)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 140) :
  (P + C) / 2 = 70 :=
by
  sorry

end average_marks_physics_chemistry_l157_157290


namespace mod_multiplication_example_l157_157777

theorem mod_multiplication_example :
  (98 % 75) * (202 % 75) % 75 = 71 :=
by
  have h1 : 98 % 75 = 23 := by sorry
  have h2 : 202 % 75 = 52 := by sorry
  have h3 : 1196 % 75 = 71 := by sorry
  exact h3

end mod_multiplication_example_l157_157777


namespace farm_entrance_fee_for_students_is_five_l157_157388

theorem farm_entrance_fee_for_students_is_five
  (students : ℕ) (adults : ℕ) (adult_fee : ℕ) (total_cost : ℕ) (student_fee : ℕ)
  (h_students : students = 35)
  (h_adults : adults = 4)
  (h_adult_fee : adult_fee = 6)
  (h_total_cost : total_cost = 199)
  (h_equation : students * student_fee + adults * adult_fee = total_cost) :
  student_fee = 5 :=
by
  sorry

end farm_entrance_fee_for_students_is_five_l157_157388


namespace factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l157_157793

variable {α : Type*} [CommRing α]

-- Problem 1
theorem factorize_2x2_minus_8 (x : α) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorize_ax2_minus_2ax_plus_a (a x : α) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
sorry

end factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l157_157793


namespace arun_speed_ratio_l157_157212

namespace SpeedRatio

variables (V_a V_n V_a' : ℝ)
variable (distance : ℝ := 30)
variable (original_speed_Arun : ℝ := 5)
variable (time_Arun time_Anil time_Arun_new_speed : ℝ)

-- Conditions
theorem arun_speed_ratio :
  V_a = original_speed_Arun →
  time_Arun = distance / V_a →
  time_Anil = distance / V_n →
  time_Arun = time_Anil + 2 →
  time_Arun_new_speed = distance / V_a' →
  time_Arun_new_speed = time_Anil - 1 →
  V_a' / V_a = 2 := 
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1] at *
  sorry

end SpeedRatio

end arun_speed_ratio_l157_157212


namespace part1_range_of_a_part2_range_of_a_l157_157968

section
variable (a : ℝ) (f : ℝ → ℝ)

def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / (x - 1) + a) * Real.log x

theorem part1_range_of_a (a : ℝ) : (∀ x, x > 0 ∧ x ≠ 1 → f a x > 0) → 0 ≤ a ∧ a ≤ 1 :=
begin
  sorry
end

theorem part2_range_of_a (a : ℝ) : (∃ x, 1 < x ∧ f a x = (1 / (x - 1) + a) * Real.log x ∧
  ∀ y, 1 < y → (f a y)' = 0) → 0 < a ∧ a < 0.5 :=
begin
  sorry
end

end

end part1_range_of_a_part2_range_of_a_l157_157968


namespace max_triangle_area_l157_157813

-- Define the parabola with y^2 = 2 * p * x for p > 0
def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  { point | ∃ y : ℝ, (y, y^2 / (2 * p)) = point }

-- Define the focus point
def focus (p : ℝ) (hp : p > 0) : ℝ × ℝ :=
  (p / 2, 0)

-- Define the line passing through the focus with inclination theta
def line_through_focus (p : ℝ) (theta : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  { point | ∃ x : ℝ, (x, Real.tan theta * (x - p / 2)) = point }

-- Define the points A and B as intersection points of the parabola and line
def points_intersection (p : ℝ) (theta : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  parabola p hp ∩ line_through_focus p theta hp 

-- Define the area of triangle AOB (O is origin, A and B are intersection points)
def triangle_area (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - A.2 * B.1)

-- The theorem to prove
theorem max_triangle_area (p : ℝ) (theta : ℝ) (hp : p > 0) :
  ∃ A B : ℝ × ℝ, A ∈ points_intersection p theta hp ∧ B ∈ points_intersection p theta hp ∧
  triangle_area A B = p^2 / 2 :=
sorry

end max_triangle_area_l157_157813


namespace shifted_parabola_l157_157521

theorem shifted_parabola (x : ℝ) : 
  let original := 5 * x^2 in
  let shifted_left := 5 * (x + 2)^2 in
  let shifted_up := shifted_left + 3 in
  shifted_up = 5 * (x + 2)^2 + 3 := 
by
  sorry

end shifted_parabola_l157_157521


namespace lamp_probability_l157_157703

theorem lamp_probability :
  (∃(red_lamps blue_lamps : ℕ), red_lamps = 4 ∧ blue_lamps = 4) ∧
  (∃(leftmost_color rightmost_color : string), leftmost_color = "blue" ∧ rightmost_color = "red") ∧
  (∃(leftmost_status rightmost_status : string), leftmost_status = "off" ∧ rightmost_status = "on") ∧
  (∃(total_arrangements favorable_outcomes : ℕ), total_arrangements = 70 * 70 ∧ favorable_outcomes = 15 * 20) →
  (∃(probability : ℚ), probability = 3 / 49) :=
sorry

end lamp_probability_l157_157703


namespace differential_system_solution_l157_157800

noncomputable def x (t : ℝ) := 1 - t - Real.exp (-6 * t) * Real.cos t
noncomputable def y (t : ℝ) := 1 - 7 * t + Real.exp (-6 * t) * Real.cos t + Real.exp (-6 * t) * Real.sin t

theorem differential_system_solution :
  (∀ t : ℝ, (deriv x t) = -7 * x t + y t + 5) ∧
  (∀ t : ℝ, (deriv y t) = -2 * x t - 5 * y t - 37 * t) ∧
  (x 0 = 0) ∧
  (y 0 = 0) :=
by 
  sorry

end differential_system_solution_l157_157800


namespace ellen_smoothie_total_l157_157475

theorem ellen_smoothie_total :
  0.2 + 0.1 + 0.2 + 0.15 + 0.05 = 0.7 :=
by sorry

end ellen_smoothie_total_l157_157475


namespace petya_winning_probability_l157_157224

noncomputable def petya_wins_probability : ℚ :=
  (1 / 4) ^ 4

-- The main theorem statement
theorem petya_winning_probability :
  petya_wins_probability = 1 / 256 :=
by sorry

end petya_winning_probability_l157_157224


namespace find_acute_angles_right_triangle_l157_157016

theorem find_acute_angles_right_triangle (α β : ℝ)
  (h₁ : α + β = π / 2)
  (h₂ : 0 < α ∧ α < π / 2)
  (h₃ : 0 < β ∧ β < π / 2)
  (h4 : Real.tan α + Real.tan β + Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan α ^ 3 + Real.tan β ^ 3 = 70) :
  (α = 75 * (π / 180) ∧ β = 15 * (π / 180)) 
  ∨ (α = 15 * (π / 180) ∧ β = 75 * (π / 180)) := 
sorry

end find_acute_angles_right_triangle_l157_157016


namespace fred_seashells_l157_157175

def seashells_given : ℕ := 25
def seashells_left : ℕ := 22
def seashells_found : ℕ := 47

theorem fred_seashells :
  seashells_found = seashells_given + seashells_left :=
  by sorry

end fred_seashells_l157_157175


namespace remaining_oranges_l157_157634

theorem remaining_oranges (num_trees : ℕ) (oranges_per_tree : ℕ) (fraction_picked : ℚ) (remaining_oranges : ℕ) :
  num_trees = 8 →
  oranges_per_tree = 200 →
  fraction_picked = 2 / 5 →
  remaining_oranges = num_trees * oranges_per_tree - num_trees * (fraction_picked * oranges_per_tree : ℚ).nat_abs →
  remaining_oranges = 960 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

#print remaining_oranges

end remaining_oranges_l157_157634


namespace factorize_difference_of_squares_l157_157794

-- We are proving that the factorization of m^2 - 9 is equal to (m+3)(m-3)
theorem factorize_difference_of_squares (m : ℝ) : m ^ 2 - 9 = (m + 3) * (m - 3) := 
by 
  sorry

end factorize_difference_of_squares_l157_157794


namespace simplify_expression_l157_157842

theorem simplify_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = 3 * (a + b)) :
  (a / b) + (b / a) - (3 / (a * b)) = 1 := 
sorry

end simplify_expression_l157_157842


namespace eval_expression_at_minus_3_l157_157007

theorem eval_expression_at_minus_3 :
  (5 + 2 * x * (x + 2) - 4^2) / (x - 4 + x^2) = -5 / 2 :=
by
  let x := -3
  sorry

end eval_expression_at_minus_3_l157_157007


namespace problem_solution_l157_157573

theorem problem_solution
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2007)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2007)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2007)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1003 := 
sorry

end problem_solution_l157_157573


namespace problem1_solution_set_problem2_a_range_l157_157501

section
variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a

-- Problem 1
theorem problem1_solution_set (h : a = 3) : {x | f x a ≤ 6} = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

def g (x : ℝ) := |2 * x - 3|

-- Problem 2
theorem problem2_a_range : ∀ a : ℝ, ∀ x : ℝ, f x a + g x ≥ 5 ↔ 4 ≤ a :=
by
  sorry
end

end problem1_solution_set_problem2_a_range_l157_157501


namespace sin_330_eq_neg_half_l157_157008

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by sorry

end sin_330_eq_neg_half_l157_157008


namespace solve_for_x_l157_157878

theorem solve_for_x (x : ℝ) :
    (1 / 3 * ((x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x - 10) → x = 12.5 :=
by
  intro h
  sorry

end solve_for_x_l157_157878


namespace cost_of_three_stamps_is_correct_l157_157546

-- Define the cost of one stamp
def cost_of_one_stamp : ℝ := 0.34

-- Define the number of stamps
def number_of_stamps : ℕ := 3

-- Define the expected total cost for three stamps
def expected_cost : ℝ := 1.02

-- Prove that the cost of three stamps is equal to the expected cost
theorem cost_of_three_stamps_is_correct : cost_of_one_stamp * number_of_stamps = expected_cost :=
by
  sorry

end cost_of_three_stamps_is_correct_l157_157546


namespace number_of_kids_stayed_home_is_668278_l157_157155

  def number_of_kids_who_stayed_home : Prop :=
    ∃ X : ℕ, X + 150780 = 819058 ∧ X = 668278

  theorem number_of_kids_stayed_home_is_668278 : number_of_kids_who_stayed_home :=
    sorry
  
end number_of_kids_stayed_home_is_668278_l157_157155


namespace ab_cardinals_l157_157063

open Set

/-- a|A| = b|B| given the conditions.
1. a and b are positive integers.
2. A and B are finite sets of integers such that:
   a. A and B are disjoint.
   b. If an integer i belongs to A or to B, then i + a ∈ A or i - b ∈ B.
-/
theorem ab_cardinals 
  (a b : ℕ) (A B : Finset ℤ) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (disjoint_AB : Disjoint A B)
  (condition_2 : ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := 
sorry

end ab_cardinals_l157_157063


namespace cost_price_l157_157589

theorem cost_price (SP : ℝ) (profit_percentage : ℝ) : SP = 600 ∧ profit_percentage = 60 → ∃ CP : ℝ, CP = 375 :=
by
  intro h
  sorry

end cost_price_l157_157589


namespace buses_in_parking_lot_l157_157897

def initial_buses : ℕ := 7
def additional_buses : ℕ := 6
def total_buses : ℕ := initial_buses + additional_buses

theorem buses_in_parking_lot : total_buses = 13 := by
  sorry

end buses_in_parking_lot_l157_157897


namespace range_of_independent_variable_x_in_sqrt_function_l157_157429

theorem range_of_independent_variable_x_in_sqrt_function :
  (∀ x : ℝ, ∃ y : ℝ, y = sqrt (2 * x - 3)) → x ≥ 3 / 2 :=
sorry

end range_of_independent_variable_x_in_sqrt_function_l157_157429


namespace prove_y_eq_x_l157_157203

theorem prove_y_eq_x
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y)
  (h2 : y = 2 + 1 / x) : y = x :=
sorry

end prove_y_eq_x_l157_157203


namespace sum_of_two_integers_l157_157881

noncomputable def sum_of_integers (a b : ℕ) : ℕ :=
a + b

theorem sum_of_two_integers (a b : ℕ) (h1 : a - b = 14) (h2 : a * b = 120) : sum_of_integers a b = 26 := 
by
  sorry

end sum_of_two_integers_l157_157881


namespace smallest_k_values_l157_157154

def cos_squared_eq_one (k : ℕ) : Prop :=
  ∃ n : ℕ, k^2 + 49 = 180 * n

theorem smallest_k_values :
  ∃ (k1 k2 : ℕ), (cos_squared_eq_one k1) ∧ (cos_squared_eq_one k2) ∧
  (∀ k < k1, ¬ cos_squared_eq_one k) ∧ (∀ k < k2, ¬ cos_squared_eq_one k) ∧ 
  k1 = 31 ∧ k2 = 37 :=
by
  sorry

end smallest_k_values_l157_157154


namespace numerator_of_fraction_l157_157206

theorem numerator_of_fraction (y x : ℝ) (hy : y > 0) (h : (9 * y) / 20 + x / y = 0.75 * y) : x = 3 :=
sorry

end numerator_of_fraction_l157_157206


namespace parabola_transformation_l157_157519

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the transformed parabola after shifting 2 units to the left and 3 units up
def transformed_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

-- State the theorem to prove the transformation
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 5 * (x + 2)^2 + 3 :=
begin
  sorry
end

end parabola_transformation_l157_157519


namespace second_player_can_form_palindrome_l157_157578

def is_palindrome (s : List Char) : Prop :=
  s = s.reverse

theorem second_player_can_form_palindrome :
  ∀ (moves : List Char), moves.length = 1999 →
  ∃ (sequence : List Char), sequence.length = 1999 ∧ is_palindrome sequence :=
by
  sorry

end second_player_can_form_palindrome_l157_157578


namespace prime_k_for_equiangular_polygons_l157_157194

-- Definitions for conditions in Lean 4
def is_equiangular_polygon (n : ℕ) (angle : ℕ) : Prop :=
  angle = 180 - 360 / n

def is_prime (k : ℕ) : Prop :=
  Nat.Prime k

def valid_angle (x : ℕ) (k : ℕ) : Prop :=
  x < 180 / k

-- The main statement
theorem prime_k_for_equiangular_polygons (n1 n2 x k : ℕ) :
  is_equiangular_polygon n1 x →
  is_equiangular_polygon n2 (k * x) →
  1 < k →
  is_prime k →
  k = 3 :=
by sorry -- proof is not required

end prime_k_for_equiangular_polygons_l157_157194


namespace shortest_distance_dasha_vasya_l157_157052

variables (dasha galia asya borya vasya : Type)
variables (dist : ∀ (a b : Type), ℕ)
variables (dist_dasha_galia : dist dasha galia = 15)
variables (dist_vasya_galia : dist vasya galia = 17)
variables (dist_asya_galia : dist asya galia = 12)
variables (dist_galia_borya : dist galia borya = 10)
variables (dist_asya_borya : dist asya borya = 8)

theorem shortest_distance_dasha_vasya : dist dasha vasya = 18 :=
by sorry

end shortest_distance_dasha_vasya_l157_157052


namespace geometric_seq_fourth_term_l157_157565

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end geometric_seq_fourth_term_l157_157565


namespace sum_of_series_eq_one_third_l157_157790

theorem sum_of_series_eq_one_third :
  ∑' k : ℕ, (2^k / (8^k - 1)) = 1 / 3 :=
sorry

end sum_of_series_eq_one_third_l157_157790


namespace value_of_a_l157_157375

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l157_157375


namespace bird_probability_l157_157712

def uniform_probability (segment_count bird_count : ℕ) : ℚ :=
  if bird_count = segment_count then
    1 / (segment_count ^ bird_count)
  else
    0

theorem bird_probability :
  let wire_length := 10
  let birds := 10
  let distance := 1
  let segments := wire_length / distance
  segments = birds ->
  uniform_probability segments birds = 1 / (10 ^ 10) := by
  intros
  sorry

end bird_probability_l157_157712


namespace painting_time_equation_l157_157953

theorem painting_time_equation (t : ℝ) :
  (1/6 + 1/8) * (t - 2) = 1 :=
sorry

end painting_time_equation_l157_157953


namespace at_least_half_team_B_can_serve_l157_157910

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l157_157910


namespace factorization_of_cubic_polynomial_l157_157230

theorem factorization_of_cubic_polynomial (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = (x + y + z) * (x^2 + y^2 + z^2 - x * y - y * z - z * x) := 
by sorry

end factorization_of_cubic_polynomial_l157_157230


namespace original_price_of_house_l157_157696

theorem original_price_of_house (P: ℝ) (sold_price: ℝ) (profit: ℝ) (commission: ℝ):
  sold_price = 100000 ∧ profit = 0.20 ∧ commission = 0.05 → P = 86956.52 :=
by
  sorry -- Proof not provided

end original_price_of_house_l157_157696


namespace binomial_divisible_by_prime_l157_157065

theorem binomial_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h_range : 1 ≤ k ∧ k ≤ p - 1) : p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_l157_157065


namespace books_not_sold_l157_157122

-- Definitions capturing the conditions
variable (B : ℕ)
variable (books_price : ℝ := 3.50)
variable (total_received : ℝ := 252)

-- Lean statement to capture the proof problem
theorem books_not_sold (h : (2 / 3 : ℝ) * B * books_price = total_received) :
  B / 3 = 36 :=
by
  sorry

end books_not_sold_l157_157122


namespace simplify_expression_l157_157868

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 2) * (5 * x ^ 12 - 3 * x ^ 11 + 2 * x ^ 9 - x ^ 6) =
  15 * x ^ 13 - 19 * x ^ 12 - 6 * x ^ 11 + 6 * x ^ 10 - 4 * x ^ 9 - 3 * x ^ 7 + 2 * x ^ 6 :=
by
  sorry

end simplify_expression_l157_157868


namespace repeating_decimal_division_l157_157917

-- Define x and y as the repeating decimals.
noncomputable def x : ℚ := 84 / 99
noncomputable def y : ℚ := 21 / 99

-- Proof statement of the equivalence.
theorem repeating_decimal_division : (x / y) = 4 := by
  sorry

end repeating_decimal_division_l157_157917


namespace log_product_l157_157467

open Real

theorem log_product : log 9 / log 2 * (log 5 / log 3) * (log 8 / log (sqrt 5)) = 12 :=
by
  sorry

end log_product_l157_157467


namespace baby_panda_daily_bamboo_intake_l157_157141

theorem baby_panda_daily_bamboo_intake :
  ∀ (adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week : ℕ),
    adult_bamboo_per_day = 138 →
    total_bamboo_per_week = 1316 →
    total_bamboo_per_week = 7 * adult_bamboo_per_day + 7 * baby_bamboo_per_day →
    baby_bamboo_per_day = 50 :=
by
  intros adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week h1 h2 h3
  sorry

end baby_panda_daily_bamboo_intake_l157_157141


namespace sampling_probability_equal_l157_157512

theorem sampling_probability_equal :
  let total_people := 2014
  let first_sample := 14
  let remaining_people := total_people - first_sample
  let sample_size := 50
  let probability := sample_size / total_people
  50 / 2014 = 25 / 1007 :=
by
  sorry

end sampling_probability_equal_l157_157512


namespace proof_problem_l157_157722

noncomputable def calc_a_star_b (a b : ℤ) : ℚ :=
1 / (a:ℚ) + 1 / (b:ℚ)

theorem proof_problem (a b : ℤ) (h1 : a + b = 10) (h2 : a * b = 24) :
  calc_a_star_b a b = 5 / 12 ∧ (a * b > a + b) := by
  sorry

end proof_problem_l157_157722


namespace abs_sum_zero_implies_diff_eq_five_l157_157041

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end abs_sum_zero_implies_diff_eq_five_l157_157041


namespace total_pens_bought_l157_157544

-- Define the problem conditions
def pens_given_to_friends : ℕ := 22
def pens_kept_for_herself : ℕ := 34

-- Theorem statement
theorem total_pens_bought : pens_given_to_friends + pens_kept_for_herself = 56 := by
  sorry

end total_pens_bought_l157_157544


namespace sum_of_tens_and_units_digit_of_8_pow_100_l157_157265

noncomputable def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
noncomputable def units_digit (n : ℕ) : ℕ := n % 10
noncomputable def sum_of_digits (n : ℕ) := tens_digit n + units_digit n

theorem sum_of_tens_and_units_digit_of_8_pow_100 : sum_of_digits (8 ^ 100) = 13 :=
by 
  sorry

end sum_of_tens_and_units_digit_of_8_pow_100_l157_157265


namespace hakeem_artichoke_dip_l157_157816

theorem hakeem_artichoke_dip 
(total_money : ℝ)
(cost_per_artichoke : ℝ)
(artichokes_per_dip : ℕ)
(dip_per_three_artichokes : ℕ)
(h : total_money = 15)
(h₁ : cost_per_artichoke = 1.25)
(h₂ : artichokes_per_dip = 3)
(h₃ : dip_per_three_artichokes = 5) : 
total_money / cost_per_artichoke * (dip_per_three_artichokes / artichokes_per_dip) = 20 := 
sorry

end hakeem_artichoke_dip_l157_157816


namespace packs_of_red_bouncy_balls_l157_157995

/-- Given the following conditions:
1. Kate bought 6 packs of yellow bouncy balls.
2. Each pack contained 18 bouncy balls.
3. Kate bought 18 more red bouncy balls than yellow bouncy balls.
Prove that the number of packs of red bouncy balls Kate bought is 7. -/
theorem packs_of_red_bouncy_balls (packs_yellow : ℕ) (balls_per_pack : ℕ) (extra_red_balls : ℕ)
  (h1 : packs_yellow = 6)
  (h2 : balls_per_pack = 18)
  (h3 : extra_red_balls = 18)
  : (packs_yellow * balls_per_pack + extra_red_balls) / balls_per_pack = 7 :=
by
  sorry

end packs_of_red_bouncy_balls_l157_157995


namespace sin4x_eq_sin2x_solution_set_l157_157640

noncomputable def solution_set (x : ℝ) : Prop :=
  0 < x ∧ x < (3 / 2) * Real.pi ∧ Real.sin (4 * x) = Real.sin (2 * x)

theorem sin4x_eq_sin2x_solution_set :
  { x : ℝ | solution_set x } =
  { (Real.pi / 6), (Real.pi / 2), Real.pi, (5 * Real.pi / 6), (7 * Real.pi / 6) } :=
by
  sorry

end sin4x_eq_sin2x_solution_set_l157_157640


namespace most_reasonable_sampling_method_l157_157612

-- Definitions based on the conditions in the problem:
def area_divided_into_200_plots : Prop := true
def plan_randomly_select_20_plots : Prop := true
def large_difference_in_plant_coverage : Prop := true
def goal_representative_sample_accurate_estimate : Prop := true

-- Main theorem statement
theorem most_reasonable_sampling_method
  (h1 : area_divided_into_200_plots)
  (h2 : plan_randomly_select_20_plots)
  (h3 : large_difference_in_plant_coverage)
  (h4 : goal_representative_sample_accurate_estimate) :
  Stratified_sampling := 
sorry

end most_reasonable_sampling_method_l157_157612


namespace relationship_between_y1_y2_l157_157138

variable (k b y1 y2 : ℝ)

-- Let A = (-3, y1) and B = (4, y2) be points on the line y = kx + b, with k < 0
axiom A_on_line : y1 = k * -3 + b
axiom B_on_line : y2 = k * 4 + b
axiom k_neg : k < 0

theorem relationship_between_y1_y2 : y1 > y2 :=
by sorry

end relationship_between_y1_y2_l157_157138


namespace compare_negative_fractions_l157_157319

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l157_157319


namespace boxed_meals_solution_count_l157_157140

theorem boxed_meals_solution_count :
  ∃ n : ℕ, n = 4 ∧ 
  ∃ x y z : ℕ, 
      x + y + z = 22 ∧ 
      10 * x + 8 * y + 5 * z = 183 ∧ 
      x > 0 ∧ y > 0 ∧ z > 0 :=
sorry

end boxed_meals_solution_count_l157_157140


namespace distance_from_point_to_line_condition_l157_157112

theorem distance_from_point_to_line_condition (a : ℝ) : (|a - 2| = 3) ↔ (a = 5 ∨ a = -1) :=
by
  sorry

end distance_from_point_to_line_condition_l157_157112


namespace sequence_A_decreases_sequence_G_decreases_l157_157396

noncomputable def A : ℕ → ℝ := λ n : ℕ,
  if n = 0 then (x + y) / 2 else
  (A (n - 1) + G (n - 1)) / 2

noncomputable def G : ℕ → ℝ := λ n : ℕ,
  if n = 0 then real.sqrt (x * y) else
  real.sqrt (A (n - 1) * G (n - 1))

theorem sequence_A_decreases (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  ∀ n : ℕ, A x y (n + 1) < A x y n :=
by
  sorry

theorem sequence_G_decreases (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  ∀ n : ℕ, G x y (n + 1) < G x y n :=
by
  sorry

end sequence_A_decreases_sequence_G_decreases_l157_157396


namespace solution_of_inequality_l157_157087

theorem solution_of_inequality (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := 
sorry

end solution_of_inequality_l157_157087


namespace remaining_area_is_correct_l157_157235

-- Define the given conditions:
def original_length : ℕ := 25
def original_width : ℕ := 35
def square_side : ℕ := 7

-- Define a function to calculate the area of the original cardboard:
def area_original : ℕ := original_length * original_width

-- Define a function to calculate the area of one square corner:
def area_corner : ℕ := square_side * square_side

-- Define a function to calculate the total area removed:
def total_area_removed : ℕ := 4 * area_corner

-- Define a function to calculate the remaining area:
def area_remaining : ℕ := area_original - total_area_removed

-- The theorem we want to prove:
theorem remaining_area_is_correct : area_remaining = 679 := by
  -- Here, we would provide the proof if required, but we use sorry for now.
  sorry

end remaining_area_is_correct_l157_157235


namespace cows_count_l157_157929

theorem cows_count (D C : ℕ) (h_legs : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end cows_count_l157_157929


namespace imaginary_unit_multiplication_l157_157442

theorem imaginary_unit_multiplication (i : ℂ) (h1 : i * i = -1) : i * (1 + i) = i - 1 :=
by
  sorry

end imaginary_unit_multiplication_l157_157442


namespace poly_not_33_l157_157548

theorem poly_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by sorry

end poly_not_33_l157_157548


namespace mutually_exclusive_event_l157_157099

def Event := String  -- define a simple type for events

/-- Define the events -/
def at_most_one_hit : Event := "at most one hit"
def two_hits : Event := "two hits"

/-- Define a function to check mutual exclusiveness -/
def mutually_exclusive (e1 e2 : Event) : Prop := 
  e1 ≠ e2

theorem mutually_exclusive_event :
  mutually_exclusive at_most_one_hit two_hits :=
by
  sorry

end mutually_exclusive_event_l157_157099


namespace decaf_percentage_correct_l157_157587

def initial_stock : ℝ := 400
def initial_decaf_percent : ℝ := 0.20
def additional_stock : ℝ := 100
def additional_decaf_percent : ℝ := 0.70

theorem decaf_percentage_correct :
  ((initial_decaf_percent * initial_stock + additional_decaf_percent * additional_stock) / (initial_stock + additional_stock)) * 100 = 30 :=
by
  sorry

end decaf_percentage_correct_l157_157587


namespace sum_of_squares_gt_five_l157_157085

theorem sum_of_squares_gt_five (a b c : ℝ) (h : a + b + c = 4) : a^2 + b^2 + c^2 > 5 :=
sorry

end sum_of_squares_gt_five_l157_157085


namespace eliza_height_is_68_l157_157343

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end eliza_height_is_68_l157_157343


namespace plane_point_to_center_ratio_l157_157692

variable (a b c p q r : ℝ)

theorem plane_point_to_center_ratio :
  (a / p) + (b / q) + (c / r) = 2 ↔ 
  (∀ (α β γ : ℝ), α = 2 * p ∧ β = 2 * q ∧ γ = 2 * r ∧ (α, 0, 0) = (a, b, c) → 
  (a / (2 * p)) + (b / (2 * q)) + (c / (2 * r)) = 1) :=
by {
  sorry
}

end plane_point_to_center_ratio_l157_157692


namespace displacement_during_interval_l157_157445

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem displacement_during_interval :
  (∫ t in (0 : ℝ)..3, velocity t) = 36 :=
by
  sorry

end displacement_during_interval_l157_157445


namespace max_tickets_l157_157481


theorem max_tickets (cost_regular : ℕ) (cost_discounted : ℕ) (threshold : ℕ) (total_money : ℕ) 
  (h1 : cost_regular = 15) 
  (h2 : cost_discounted = 12) 
  (h3 : threshold = 5)
  (h4 : total_money = 150) 
  : (total_money / cost_regular ≤ 10) ∧ 
    ((total_money - threshold * cost_regular) / cost_discounted + threshold = 11) :=
by
  sorry

end max_tickets_l157_157481


namespace compare_fractions_l157_157316

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l157_157316


namespace team_B_elibility_l157_157906

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l157_157906


namespace average_minutes_run_per_day_l157_157301

theorem average_minutes_run_per_day (f : ℕ) (h_nonzero : f ≠ 0)
  (third_avg fourth_avg fifth_avg : ℕ)
  (third_avg_eq : third_avg = 14)
  (fourth_avg_eq : fourth_avg = 18)
  (fifth_avg_eq : fifth_avg = 8)
  (third_count fourth_count fifth_count : ℕ)
  (third_count_eq : third_count = 3 * fourth_count)
  (fourth_count_eq : fourth_count = f / 2)
  (fifth_count_eq : fifth_count = f) :
  (third_avg * third_count + fourth_avg * fourth_count + fifth_avg * fifth_count) / (third_count + fourth_count + fifth_count) = 38 / 3 :=
by
  sorry

end average_minutes_run_per_day_l157_157301


namespace bucket_full_weight_l157_157098

theorem bucket_full_weight (x y c d : ℝ) 
  (h1 : x + (3/4) * y = c)
  (h2 : x + (3/5) * y = d) :
  x + y = (5/3) * c - (5/3) * d :=
by
  sorry

end bucket_full_weight_l157_157098


namespace expected_red_balls_in_B_l157_157091

-- Define the initial conditions
def initial_red_A := 4
def initial_white_A := 3
def initial_red_B := 3
def initial_white_B := 4

-- Define the total number of balls in each box initially
def total_A := initial_red_A + initial_white_A
def total_B := initial_red_B + initial_white_B

-- Define the probability of drawing each type of ball from each box
def prob_red_A := initial_red_A.toRational / total_A.toRational
def prob_white_A := initial_white_A.toRational / total_A.toRational
def prob_red_B := initial_red_B.toRational / total_B.toRational
def prob_white_B := initial_white_B.toRational / total_B.toRational

-- Define the probabilities of each scenario after transferring the balls
def prob_xi_2 := prob_white_A * (initial_red_B.toRational / (total_B + 1).toRational)
def prob_xi_4 := prob_red_A * (initial_white_B.toRational / (total_B + 1).toRational)
def prob_xi_3 := 1.toRational - prob_xi_2 - prob_xi_4

-- Define the expected value computation
def E_xi := 2.toRational * prob_xi_2 + 3.toRational * prob_xi_3 + 4.toRational * prob_xi_4

theorem expected_red_balls_in_B : E_xi = 25.toRational / 8.toRational :=
sorry

end expected_red_balls_in_B_l157_157091


namespace grasshoppers_positions_swap_l157_157899

theorem grasshoppers_positions_swap :
  ∃ (A B C: ℤ), A = -1 ∧ B = 0 ∧ C = 1 ∧
  (∀ m n p : ℤ, (A, B, C) = (m, n, p) → n = 0 → 
  (m^2 - n^2 + p^2 = 0) → A = 1 ∧ B = 0 ∧ C = -1) :=
begin
  -- Adding assumptions
  let x₁ := -1 : ℤ,
  let x₂ := 0 : ℤ,
  let x₃ := 1 : ℤ,
  existsi x₁, existsi x₂, existsi x₃,
  split, refl,
  split, refl,
  split, refl,
  intros m n p hperm hnze hyp,
  sorry -- the detailed proof will go here
end

end grasshoppers_positions_swap_l157_157899


namespace find_weight_of_second_square_l157_157292

-- Define the initial conditions
def uniform_density_thickness (density : ℝ) (thickness : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ), l₁ = l₂ → density = thickness

-- Define the first square properties
def first_square (side_length₁ weight₁ : ℝ) : Prop :=
  side_length₁ = 4 ∧ weight₁ = 16

-- Define the second square properties
def second_square (side_length₂ : ℝ) : Prop :=
  side_length₂ = 6

-- Define the proportional relationship between the area and weight
def proportional_weight (side_length₁ weight₁ side_length₂ weight₂ : ℝ) : Prop :=
  (side_length₁^2 / weight₁) = (side_length₂^2 / weight₂)

-- Lean statement to prove the weight of the second square
theorem find_weight_of_second_square (density thickness side_length₁ weight₁ side_length₂ weight₂ : ℝ)
  (h_density_thickness : uniform_density_thickness density thickness)
  (h_first_square : first_square side_length₁ weight₁)
  (h_second_square : second_square side_length₂)
  (h_proportional_weight : proportional_weight side_length₁ weight₁ side_length₂ weight₂) : 
  weight₂ = 36 :=
by 
  sorry

end find_weight_of_second_square_l157_157292


namespace point_on_graph_l157_157584

def f (x : ℝ) : ℝ := -2 * x + 3

theorem point_on_graph (x y : ℝ) : 
  ( (x = 1 ∧ y = 1) ↔ y = f x ) :=
by 
  sorry

end point_on_graph_l157_157584


namespace cost_of_one_basketball_deck_l157_157406

theorem cost_of_one_basketball_deck (total_money_spent : ℕ) 
  (mary_sunglasses_cost : ℕ) (mary_jeans_cost : ℕ) 
  (rose_shoes_cost : ℕ) (rose_decks_count : ℕ) 
  (mary_total_cost : total_money_spent = 2 * mary_sunglasses_cost + mary_jeans_cost)
  (rose_total_cost : total_money_spent = rose_shoes_cost + 2 * (total_money_spent - rose_shoes_cost) / rose_decks_count) :
  (total_money_spent - rose_shoes_cost) / rose_decks_count = 25 := 
by 
  sorry

end cost_of_one_basketball_deck_l157_157406


namespace original_price_l157_157145

variable (P : ℝ)
variable (S : ℝ := 140)
variable (discount : ℝ := 0.60)

theorem original_price :
  (S = P * (1 - discount)) → (P = 350) :=
by
  sorry

end original_price_l157_157145


namespace cost_per_square_meter_l157_157597

noncomputable def costPerSquareMeter 
  (length : ℝ) (breadth : ℝ) (width : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / ((length * width) + (breadth * width) - (width * width))

theorem cost_per_square_meter (H1 : length = 110)
                              (H2 : breadth = 60)
                              (H3 : width = 10)
                              (H4 : total_cost = 4800) : 
  costPerSquareMeter length breadth width total_cost = 3 := 
by
  sorry

end cost_per_square_meter_l157_157597


namespace selling_price_before_clearance_l157_157133

-- Define the cost price (CP)
def CP : ℝ := 100

-- Define the gain percent before the clearance sale
def gain_percent_before : ℝ := 0.35

-- Define the discount percent during the clearance sale
def discount_percent : ℝ := 0.10

-- Define the gain percent during the clearance sale
def gain_percent_sale : ℝ := 0.215

-- Calculate the selling price before the clearance sale (SP_before)
def SP_before : ℝ := CP * (1 + gain_percent_before)

-- Calculate the selling price during the clearance sale (SP_sale)
def SP_sale : ℝ := SP_before * (1 - discount_percent)

-- Proof statement in Lean 4
theorem selling_price_before_clearance : SP_before = 135 :=
by
  -- Place to fill in the proof later
  sorry

end selling_price_before_clearance_l157_157133


namespace value_of_ab_l157_157042

theorem value_of_ab (a b c : ℝ) (C : ℝ) (h1 : (a + b) ^ 2 - c ^ 2 = 4) (h2 : C = Real.pi / 3) : 
  a * b = 4 / 3 :=
by
  sorry

end value_of_ab_l157_157042


namespace math_proof_problem_l157_157017

-- The given conditions
def condition1 {α : ℝ} : Prop := 2 * sin α * tan α = 3
def condition2 {α : ℝ} : Prop := 0 < α ∧ α < π

-- The propositions to be proved
def proposition1 : Prop := ∃ α : ℝ, condition1 α ∧ condition2 α ∧ α = π / 3
def proposition2 : Prop :=
  ∀ (α : ℝ) (hα : α = π / 3), ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 4) → 
  -1 ≤ 4 * sin x * sin (x - α) ∧ 4 * sin x * sin (x - α) ≤ 0

-- The proof problem statement
theorem math_proof_problem : proposition1 ∧ proposition2 := by
  sorry

end math_proof_problem_l157_157017


namespace aimee_poll_l157_157294

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l157_157294


namespace digital_earth_concept_wrong_l157_157609

theorem digital_earth_concept_wrong :
  ∀ (A C D : Prop),
  (A → true) →
  (C → true) →
  (D → true) →
  ¬(B → true) :=
by
  sorry

end digital_earth_concept_wrong_l157_157609


namespace compute_2a_minus_b_l157_157535

noncomputable def conditions (a b : ℝ) : Prop :=
  a^3 - 12 * a^2 + 47 * a - 60 = 0 ∧
  -b^3 + 12 * b^2 - 47 * b + 180 = 0

theorem compute_2a_minus_b (a b : ℝ) (h : conditions a b) : 2 * a - b = 2 := 
  sorry

end compute_2a_minus_b_l157_157535


namespace transformed_cube_edges_l157_157448

-- Let's define the problem statement
theorem transformed_cube_edges : 
  let original_edges := 12 
  let new_edges_per_edge := 2 
  let additional_edges_per_pyramid := 1 
  let total_edges := original_edges + (original_edges * new_edges_per_edge) + (original_edges * additional_edges_per_pyramid) 
  total_edges = 48 :=
by sorry

end transformed_cube_edges_l157_157448


namespace midpoint_sum_of_coordinates_l157_157863

theorem midpoint_sum_of_coordinates
  (M : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hmx : (C.1 + D.1) / 2 = M.1)
  (hmy : (C.2 + D.2) / 2 = M.2)
  (hM : M = (3, 5))
  (hC : C = (5, 3)) :
  D.1 + D.2 = 8 :=
by
  sorry

end midpoint_sum_of_coordinates_l157_157863


namespace probability_first_white_second_red_l157_157119

noncomputable def marble_probability (total_marbles first_white second_red : ℚ) : ℚ :=
  first_white * second_red

theorem probability_first_white_second_red :
  let total_marbles := 10 in
  let first_white := 6 / total_marbles in
  let second_red_given_white := 4 / (total_marbles - 1) in
  marble_probability total_marbles first_white second_red_given_white = 4 / 15 :=
by
  sorry

end probability_first_white_second_red_l157_157119


namespace inbox_emails_after_movements_l157_157075

def initial_emails := 400
def trash_emails := initial_emails / 2
def remaining_emails := initial_emails - trash_emails
def work_emails := 0.4 * remaining_emails
def final_inbox_emails := remaining_emails - work_emails

theorem inbox_emails_after_movements : final_inbox_emails = 120 :=
by
  sorry

end inbox_emails_after_movements_l157_157075


namespace negation_of_exists_l157_157179

theorem negation_of_exists:
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := sorry

end negation_of_exists_l157_157179


namespace exists_non_regular_triangle_with_similar_medians_as_sides_l157_157387

theorem exists_non_regular_triangle_with_similar_medians_as_sides 
  (a b c : ℝ) 
  (s_a s_b s_c : ℝ)
  (h1 : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h2 : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h3 : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (similarity_cond : (2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∃ (s_a s_b s_c : ℝ), 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2 ∧ 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2 ∧ 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2) ∧
  ((2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :=
sorry

end exists_non_regular_triangle_with_similar_medians_as_sides_l157_157387


namespace sum_of_decimals_l157_157779

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 :=
by
  sorry

end sum_of_decimals_l157_157779


namespace product_or_double_is_perfect_square_l157_157389

variable {a b c : ℤ}

-- Conditions
def sides_of_triangle (a b c : ℤ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def no_common_divisor (a b c : ℤ) : Prop := gcd (gcd a b) c = 1

def all_fractions_are_integers (a b c : ℤ) : Prop :=
  (a + b - c) ≠ 0 ∧ (b + c - a) ≠ 0 ∧ (c + a - b) ≠ 0 ∧
  ((a^2 + b^2 - c^2) % (a + b - c) = 0) ∧ 
  ((b^2 + c^2 - a^2) % (b + c - a) = 0) ∧ 
  ((c^2 + a^2 - b^2) % (c + a - b) = 0)

-- Mathematical proof problem statement in Lean 4
theorem product_or_double_is_perfect_square (a b c : ℤ) 
  (h1 : sides_of_triangle a b c)
  (h2 : no_common_divisor a b c)
  (h3 : all_fractions_are_integers a b c) :
  ∃ k : ℤ, k^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
           k^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := sorry

end product_or_double_is_perfect_square_l157_157389


namespace max_value_of_quadratic_l157_157787

theorem max_value_of_quadratic : 
  ∃ x : ℝ, (∃ M : ℝ, ∀ y : ℝ, (-3 * y^2 + 15 * y + 9 <= M)) ∧ M = 111 / 4 :=
by
  sorry

end max_value_of_quadratic_l157_157787


namespace symmetric_function_value_l157_157043

noncomputable def f (x a : ℝ) := (|x - 2| + a) / (Real.sqrt (4 - x^2))

theorem symmetric_function_value :
  ∃ a : ℝ, (∀ x : ℝ, f x a = (|x - 2| + a) / (Real.sqrt (4 - x^2)) ∧ f x a = -f (-x) a) →
  f (a / 2) a = (Real.sqrt 3) / 3 :=
by
  sorry

end symmetric_function_value_l157_157043


namespace nine_chapters_problem_l157_157834

def cond1 (x y : ℕ) : Prop := y = 6 * x - 6
def cond2 (x y : ℕ) : Prop := y = 5 * x + 5

theorem nine_chapters_problem (x y : ℕ) :
  (cond1 x y ∧ cond2 x y) ↔ (y = 6 * x - 6 ∧ y = 5 * x + 5) :=
by
  sorry

end nine_chapters_problem_l157_157834


namespace quadratic_inequality_solution_l157_157349

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 4 * x > 45 ↔ x < -9 ∨ x > 5 := 
  sorry

end quadratic_inequality_solution_l157_157349


namespace max_value_expression_l157_157786

theorem max_value_expression : 
  ∃ x_max : ℝ, 
    (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ -3 * x_max^2 + 15 * x_max + 9) ∧
    (-3 * x_max^2 + 15 * x_max + 9 = 111 / 4) :=
by
  sorry

end max_value_expression_l157_157786


namespace subcommittee_count_l157_157752

theorem subcommittee_count :
  (Nat.choose 10 4) * (Nat.choose 8 3) = 11760 :=
by
  sorry

end subcommittee_count_l157_157752


namespace area_triangle_AMC_l157_157213

noncomputable def area_of_triangle_AMC (AB AD AM : ℝ) : ℝ :=
  if AB = 10 ∧ AD = 12 ∧ AM = 9 then
    (1 / 2) * AM * AB
  else 0

theorem area_triangle_AMC :
  ∀ (AB AD AM : ℝ), AB = 10 → AD = 12 → AM = 9 → area_of_triangle_AMC AB AD AM = 45 := by
  intros AB AD AM hAB hAD hAM
  simp [area_of_triangle_AMC, hAB, hAD, hAM]
  sorry

end area_triangle_AMC_l157_157213


namespace parabola_distance_focus_P_l157_157964

noncomputable def distance_PF : ℝ := sorry

theorem parabola_distance_focus_P : ∀ (P : ℝ × ℝ) (F : ℝ × ℝ),
  P.2^2 = 4 * P.1 ∧ F = (1, 0) ∧ P.1 = 4 → distance_PF = 5 :=
by
  intros P F h
  sorry

end parabola_distance_focus_P_l157_157964


namespace geo_seq_4th_term_l157_157563

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end geo_seq_4th_term_l157_157563


namespace intersection_of_M_and_N_l157_157659

-- Define sets M and N
def M : Set ℕ := {x | x < 6}
def N : Set ℝ := {x | (x-2) * (x-9) < 0}

-- Define a proof statement with the appropriate claim
theorem intersection_of_M_and_N : M ∩ {x: ℕ | (x : ℝ ∈ N)} = {3, 4, 5} := sorry

end intersection_of_M_and_N_l157_157659


namespace quadratic_inequality_solution_l157_157171

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) := sorry

end quadratic_inequality_solution_l157_157171


namespace simplify_fraction_addition_l157_157073

theorem simplify_fraction_addition (a b : ℚ) (h1 : a = 4 / 252) (h2 : b = 17 / 36) :
  a + b = 41 / 84 := 
by
  sorry

end simplify_fraction_addition_l157_157073


namespace product_of_935421_and_625_l157_157580

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 :=
by
  sorry

end product_of_935421_and_625_l157_157580


namespace purely_imaginary_m_complex_division_a_plus_b_l157_157186

-- Problem 1: Prove that m=-2 for z to be purely imaginary
theorem purely_imaginary_m (m : ℝ) (h : ∀ z : ℂ, z = (m - 1) * (m + 2) + (m - 1) * I → z.im = z.im) : m = -2 :=
sorry

-- Problem 2: Prove a+b = 13/10 with given conditions
theorem complex_division_a_plus_b (a b : ℝ) (m : ℝ) (h_m : m = 2) 
  (h_z : z = 4 + I) (h_eq : (z + I) / (z - I) = a + b * I) : a + b = 13 / 10 :=
sorry

end purely_imaginary_m_complex_division_a_plus_b_l157_157186


namespace cyclic_sum_inequality_l157_157222

theorem cyclic_sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∑ cyc, (a * (a^2 + b * c)) / (b + c) ≥ ∑ cyc, (a * b) :=
  sorry

end cyclic_sum_inequality_l157_157222


namespace afternoon_more_than_evening_l157_157444

def campers_in_morning : Nat := 33
def campers_in_afternoon : Nat := 34
def campers_in_evening : Nat := 10

theorem afternoon_more_than_evening : campers_in_afternoon - campers_in_evening = 24 := by
  sorry

end afternoon_more_than_evening_l157_157444


namespace area_of_rhombus_is_375_l157_157439

-- define the given diagonals
def diagonal1 := 25
def diagonal2 := 30

-- define the formula for the area of a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

-- state the theorem
theorem area_of_rhombus_is_375 : area_of_rhombus diagonal1 diagonal2 = 375 := 
by 
  -- The proof is omitted as per the requirement
  sorry

end area_of_rhombus_is_375_l157_157439


namespace sum_divisible_by_4_l157_157198

theorem sum_divisible_by_4 (a b c d x : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9) : 4 ∣ (a + b + c + d) :=
by
  sorry

end sum_divisible_by_4_l157_157198


namespace ratio_of_lengths_l157_157599

theorem ratio_of_lengths (l1 l2 l3 : ℝ)
    (h1 : l2 = (1/2) * (l1 + l3))
    (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
    l1 / l3 = 7 / 5 := by
  sorry

end ratio_of_lengths_l157_157599


namespace parallel_line_slope_l157_157581

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1 / 2 ∧ (∀ x1 y1 : ℝ, 3 * x1 - 6 * y1 = 12 → 
    ∃ k : ℝ, y1 = m * x1 + k) :=
by
  sorry

end parallel_line_slope_l157_157581


namespace root_and_value_of_a_equation_has_real_roots_l157_157027

theorem root_and_value_of_a (a : ℝ) (other_root : ℝ) :
  (∃ x : ℝ, x^2 + a * x + a - 1 = 0 ∧ x = 2) → a = -1 ∧ other_root = -1 :=
by sorry

theorem equation_has_real_roots (a : ℝ) :
  ∃ x : ℝ, x^2 + a * x + a - 1 = 0 :=
by sorry

end root_and_value_of_a_equation_has_real_roots_l157_157027


namespace polynomial_factorization_l157_157357

noncomputable def polynomial_expr (a b c : ℝ) :=
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2)

noncomputable def factored_form (a b c : ℝ) :=
  (a - b) * (b - c) * (c - a) * (b^2 + c^2 + a^2)

theorem polynomial_factorization (a b c : ℝ) :
  polynomial_expr a b c = factored_form a b c :=
by {
  sorry
}

end polynomial_factorization_l157_157357


namespace fruit_store_problem_l157_157762

-- Define the conditions
def total_weight : Nat := 140
def total_cost : Nat := 1000

def purchase_price_A : Nat := 5
def purchase_price_B : Nat := 9

def selling_price_A : Nat := 8
def selling_price_B : Nat := 13

-- Define the total purchase price equation
def purchase_cost (x : Nat) : Nat := purchase_price_A * x + purchase_price_B * (total_weight - x)

-- Define the profit calculation
def profit (x : Nat) (y : Nat) : Nat := (selling_price_A - purchase_price_A) * x + (selling_price_B - purchase_price_B) * y

-- State the problem
theorem fruit_store_problem :
  ∃ x y : Nat, x + y = total_weight ∧ purchase_cost x = total_cost ∧ profit x y = 495 :=
by
  sorry

end fruit_store_problem_l157_157762


namespace brick_width_l157_157447

theorem brick_width (L W : ℕ) (l : ℕ) (b : ℕ) (n : ℕ) (A B : ℕ) 
    (courtyard_area_eq : A = L * W * 10000)
    (brick_area_eq : B = l * b)
    (total_bricks_eq : A = n * B)
    (courtyard_dims : L = 30 ∧ W = 16)
    (brick_len : l = 20)
    (num_bricks : n = 24000) :
    b = 10 := by
  sorry

end brick_width_l157_157447


namespace smallest_n_divisibility_problem_l157_157739

theorem smallest_n_divisibility_problem :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → ¬(n^2 + n) % k = 0)) ∧ n = 4 :=
by
  sorry

end smallest_n_divisibility_problem_l157_157739


namespace geometric_sequence_product_l157_157835

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n : ℕ, a (n + 1) = r * a n)
variable (h_condition : a 5 * a 14 = 5)

theorem geometric_sequence_product :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_product_l157_157835


namespace find_m_l157_157025
open Nat

theorem find_m (m : ℕ) (hm : m > 0) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 24 := 
by
  sorry

end find_m_l157_157025


namespace largest_c_l157_157013

theorem largest_c (c : ℝ) : (∃ x : ℝ, x^2 + 4 * x + c = -3) → c ≤ 1 :=
by
  sorry

end largest_c_l157_157013


namespace series_fraction_simplify_l157_157624

theorem series_fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by 
  sorry

end series_fraction_simplify_l157_157624


namespace total_drink_volume_l157_157055

theorem total_drink_volume (coke_parts sprite_parts mtndew_parts : ℕ) (coke_volume : ℕ) :
  coke_parts = 2 → sprite_parts = 1 → mtndew_parts = 3 → coke_volume = 6 →
  (coke_volume / coke_parts) * (coke_parts + sprite_parts + mtndew_parts) = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end total_drink_volume_l157_157055


namespace total_profit_calculation_l157_157107

variables {I_B T_B : ℝ}

-- Conditions as definitions
def investment_A (I_B : ℝ) : ℝ := 3 * I_B
def period_A (T_B : ℝ) : ℝ := 2 * T_B
def profit_B (I_B T_B : ℝ) : ℝ := I_B * T_B
def total_profit (I_B T_B : ℝ) : ℝ := 7 * I_B * T_B

-- To prove
theorem total_profit_calculation
  (h1 : investment_A I_B = 3 * I_B)
  (h2 : period_A T_B = 2 * T_B)
  (h3 : profit_B I_B T_B = 4000)
  : total_profit I_B T_B = 28000 := by
  sorry

end total_profit_calculation_l157_157107


namespace weekly_milk_consumption_l157_157403

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end weekly_milk_consumption_l157_157403


namespace find_a_value_l157_157371

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l157_157371


namespace david_total_course_hours_l157_157000

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l157_157000


namespace plane_hovering_time_l157_157142

theorem plane_hovering_time :
  let mt_day1 := 3
  let ct_day1 := 4
  let et_day1 := 2
  let add_hours := 2
  let mt_day2 := mt_day1 + add_hours
  let ct_day2 := ct_day1 + add_hours
  let et_day2 := et_day1 + add_hours
  let total_mt := mt_day1 + mt_day2
  let total_ct := ct_day1 + ct_day2
  let total_et := et_day1 + et_day2
  let total_hovering_time := total_mt + total_ct + total_et
  total_hovering_time = 24 :=
by
  simp [mt_day1, ct_day1, et_day1, add_hours, mt_day2, ct_day2, et_day2, total_mt, total_ct, total_et, total_hovering_time, Nat.add]
  exact sorry

end plane_hovering_time_l157_157142


namespace faster_speed_l157_157378

theorem faster_speed (S : ℝ) (actual_speed : ℝ := 10) (extra_distance : ℝ := 20) (actual_distance : ℝ := 20) :
  actual_distance / actual_speed = (actual_distance + extra_distance) / S → S = 20 :=
by
  sorry

end faster_speed_l157_157378


namespace gcd_lcm_45_75_l157_157916

theorem gcd_lcm_45_75 : gcd 45 75 = 15 ∧ lcm 45 75 = 1125 :=
by sorry

end gcd_lcm_45_75_l157_157916


namespace exists_n_such_that_not_square_l157_157632

theorem exists_n_such_that_not_square : ∃ n : ℕ, n > 1 ∧ ¬(∃ k : ℕ, k ^ 2 = 2 ^ (2 ^ n - 1) - 7) := 
sorry

end exists_n_such_that_not_square_l157_157632


namespace total_doll_count_l157_157505

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l157_157505


namespace value_of_a_l157_157376

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l157_157376


namespace chord_length_l157_157039

theorem chord_length
  (a b c A B C : ℝ)
  (h₁ : c * Real.sin C = 3 * a * Real.sin A + 3 * b * Real.sin B)
  (O : ℝ → ℝ → Prop)
  (hO : ∀ x y, O x y ↔ x^2 + y^2 = 12)
  (l : ℝ → ℝ → Prop)
  (hl : ∀ x y, l x y ↔ a * x - b * y + c = 0) :
  (2 * Real.sqrt ( (2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 )) = 6 :=
by
  sorry

end chord_length_l157_157039


namespace four_digit_numbers_count_l157_157817

open Nat

def is_valid_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def four_diff_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def leading_digit_not_zero (a : ℕ) : Prop :=
  a ≠ 0

def largest_digit_seven (a b c d : ℕ) : Prop :=
  a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7

theorem four_digit_numbers_count :
  ∃ n, n = 45 ∧
  ∀ (a b c d : ℕ),
    four_diff_digits a b c d ∧
    leading_digit_not_zero a ∧
    is_multiple_of_5 (a * 1000 + b * 100 + c * 10 + d) ∧
    is_multiple_of_3 (a * 1000 + b * 100 + c * 10 + d) ∧
    largest_digit_seven a b c d →
    n = 45 :=
sorry

end four_digit_numbers_count_l157_157817


namespace polygon_with_three_times_exterior_angle_sum_is_octagon_l157_157289

theorem polygon_with_three_times_exterior_angle_sum_is_octagon
  (n : ℕ)
  (h : (n - 2) * 180 = 3 * 360) : n = 8 := by
  sorry

end polygon_with_three_times_exterior_angle_sum_is_octagon_l157_157289


namespace monotonous_count_between_1_and_9999_l157_157003

def is_monotonous (n : Nat) : Prop :=
  if n < 10 then True
  else 
    let digits := List.digits n
    List.strict_sorted (· < ·) digits ∨
    List.strict_sorted (· > ·) digits ∨
    List.all_eq digits

def monotonous_count := Finset.card (Finset.filter is_monotonous (Finset.range 10000))

theorem monotonous_count_between_1_and_9999 : monotonous_count = 556 := by
  sorry

end monotonous_count_between_1_and_9999_l157_157003


namespace allocation_schemes_l157_157714

theorem allocation_schemes (V D : ℕ) (hV : V = 5) (hD : D = 3) :
  ∃ (f : fin V → fin D), (∀ d : fin D, ∃ v : fin V, f v = d) ∧
  (finset.univ.card : ℕ) = 150 :=
by
  sorry

end allocation_schemes_l157_157714


namespace avg_age_difference_l157_157717

noncomputable def team_size : ℕ := 11
noncomputable def avg_age_team : ℝ := 26
noncomputable def wicket_keeper_extra_age : ℝ := 3
noncomputable def num_remaining_players : ℕ := 9
noncomputable def avg_age_remaining_players : ℝ := 23

theorem avg_age_difference :
  avg_age_team - avg_age_remaining_players = 0.33 := 
by
  sorry

end avg_age_difference_l157_157717


namespace max_value_of_quadratic_l157_157788

theorem max_value_of_quadratic : 
  ∃ x : ℝ, (∃ M : ℝ, ∀ y : ℝ, (-3 * y^2 + 15 * y + 9 <= M)) ∧ M = 111 / 4 :=
by
  sorry

end max_value_of_quadratic_l157_157788


namespace age_of_third_boy_l157_157726

theorem age_of_third_boy (a b c : ℕ) (h1 : a = 9) (h2 : b = 9) (h_sum : a + b + c = 29) : c = 11 :=
by
  sorry

end age_of_third_boy_l157_157726


namespace benches_required_l157_157284

theorem benches_required (students_base5 : ℕ := 312) (base_student_seating : ℕ := 5) (seats_per_bench : ℕ := 3) : ℕ :=
  let chairs := 3 * base_student_seating^2 + 1 * base_student_seating^1 + 2 * base_student_seating^0
  let benches := (chairs / seats_per_bench) + if (chairs % seats_per_bench > 0) then 1 else 0
  benches

example : benches_required = 28 :=
by sorry

end benches_required_l157_157284


namespace scientific_notation_of_area_l157_157078

theorem scientific_notation_of_area : 2720000 = 2.72 * 10^6 :=
by
  sorry

end scientific_notation_of_area_l157_157078


namespace ratio_of_larger_to_smaller_l157_157727

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 2 := 
by
  sorry

end ratio_of_larger_to_smaller_l157_157727


namespace road_trip_ratio_l157_157701

theorem road_trip_ratio (D R: ℝ) (h1 : 1 / 2 * D = 40) (h2 : 2 * (D + R * D + 40) = 560 - (D + R * D + 40)) :
  R = 5 / 6 := by
  sorry

end road_trip_ratio_l157_157701


namespace gas_cost_is_4_l157_157684

theorem gas_cost_is_4
    (mileage_rate : ℝ)
    (truck_efficiency : ℝ)
    (profit : ℝ)
    (trip_distance : ℝ)
    (trip_cost : ℝ)
    (gallons_used : ℝ)
    (cost_per_gallon : ℝ) :
  mileage_rate = 0.5 →
  truck_efficiency = 20 →
  profit = 180 →
  trip_distance = 600 →
  trip_cost = mileage_rate * trip_distance - profit →
  gallons_used = trip_distance / truck_efficiency →
  cost_per_gallon = trip_cost / gallons_used →
  cost_per_gallon = 4 :=
by
  sorry

end gas_cost_is_4_l157_157684


namespace determine_abcd_l157_157056

theorem determine_abcd (a b c d : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) 
    (h₂ : 0 ≤ c ∧ c ≤ 9) (h₃ : 0 ≤ d ∧ d ≤ 9) 
    (h₄ : (10 * a + b) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 27 / 37) :
    1000 * a + 100 * b + 10 * c + d = 3644 :=
by
  sorry

end determine_abcd_l157_157056


namespace students_not_enrolled_in_either_course_l157_157675

theorem students_not_enrolled_in_either_course 
  (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h_total : total = 87) (h_french : french = 41) (h_german : german = 22) (h_both : both = 9) : 
  ∃ (not_enrolled : ℕ), not_enrolled = (total - (french + german - both)) ∧ not_enrolled = 33 := by
  have h_french_or_german : ℕ := french + german - both
  have h_not_enrolled : ℕ := total - h_french_or_german
  use h_not_enrolled
  sorry

end students_not_enrolled_in_either_course_l157_157675


namespace poll_total_l157_157299

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l157_157299


namespace quadratic_as_sum_of_two_with_zero_discriminants_l157_157547

theorem quadratic_as_sum_of_two_with_zero_discriminants (c : ℝ) :
  ∃ (p q : Polynomial ℝ), p.degree = 2 ∧ q.degree = 2 ∧
  p.Coeff 2 = 0 ∧ q.Coeff 2 = 0 ∧
  (Polynomial.ofCoeff 2 (2 : ℝ) + Polynomial.ofCoeff 0 c) = p + q :=
sorry

end quadratic_as_sum_of_two_with_zero_discriminants_l157_157547


namespace marks_lost_per_wrong_answer_l157_157677

theorem marks_lost_per_wrong_answer (x : ℝ) : 
  (score_per_correct = 4) ∧ 
  (num_questions = 60) ∧ 
  (total_marks = 120) ∧ 
  (correct_answers = 36) ∧ 
  (wrong_answers = num_questions - correct_answers) ∧
  (wrong_answers = 24) ∧
  (total_score_from_correct = score_per_correct * correct_answers) ∧ 
  (total_marks_lost = total_score_from_correct - total_marks) ∧ 
  (total_marks_lost = wrong_answers * x) → 
  x = 1 := 
by 
  sorry

end marks_lost_per_wrong_answer_l157_157677


namespace problem_part1_problem_part2_l157_157398

open Real

-- Part (1)
theorem problem_part1 : ∀ x > 0, log x ≤ x - 1 := 
by 
  sorry -- proof goes here


-- Part (2)
theorem problem_part2 : (∀ x > 0, log x ≤ a * x + (a - 1) / x - 1) → 1 ≤ a := 
by 
  sorry -- proof goes here

end problem_part1_problem_part2_l157_157398


namespace total_dolls_48_l157_157509

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l157_157509


namespace triangle_area_CO_B_l157_157862

-- Define the conditions as given in the problem
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def Q : Point := ⟨0, 15⟩

variable (p : ℝ)
def C : Point := ⟨0, p⟩
def B : Point := ⟨15, 0⟩

-- Prove the area of triangle COB is 15p / 2
theorem triangle_area_CO_B :
  p ≥ 0 → p ≤ 15 → 
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  area = (15 * p) / 2 := 
by
  intros hp0 hp15
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  have : area = (15 * p) / 2 := sorry
  exact this

end triangle_area_CO_B_l157_157862


namespace knights_and_liars_solution_l157_157708

-- Definitions of each person's statement as predicates
def person1_statement (liar : ℕ → Prop) : Prop := liar 2 ∧ liar 3 ∧ liar 4 ∧ liar 5 ∧ liar 6
def person2_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ ∀ i, i ≠ 1 → ¬ liar i
def person3_statement (liar : ℕ → Prop) : Prop := liar 4 ∧ liar 5 ∧ liar 6 ∧ ¬ liar 3 ∧ ¬ liar 2 ∧ ¬ liar 1
def person4_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ liar 2 ∧ liar 3 ∧ ∀ i, i > 3 → ¬ liar i
def person5_statement (liar : ℕ → Prop) : Prop := liar 6 ∧ ∀ i, i ≠ 6 → ¬ liar i
def person6_statement (liar : ℕ → Prop) : Prop := liar 5 ∧ ∀ i, i ≠ 5 → ¬ liar i

-- Definition of a knight and a liar
def is_knight (statement : Prop) : Prop := statement
def is_liar (statement : Prop) : Prop := ¬ statement

-- Defining the theorem
theorem knights_and_liars_solution (knight liar : ℕ → Prop) : 
  is_liar (person1_statement liar) ∧ 
  is_knight (person2_statement liar) ∧ 
  is_liar (person3_statement liar) ∧ 
  is_liar (person4_statement liar) ∧ 
  is_knight (person5_statement liar) ∧ 
  is_liar (person6_statement liar) :=
by
  sorry

end knights_and_liars_solution_l157_157708


namespace integer_pairs_count_l157_157819

theorem integer_pairs_count : ∃ (pairs : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), (x ≥ y ∧ (x, y) ∈ pairs → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 211))
  ∧ pairs.card = 3 :=
by
  sorry

end integer_pairs_count_l157_157819


namespace lillian_candies_total_l157_157223

variable (initial_candies : ℕ)
variable (candies_given_by_father : ℕ)

theorem lillian_candies_total (initial_candies : ℕ) (candies_given_by_father : ℕ) :
  initial_candies = 88 →
  candies_given_by_father = 5 →
  initial_candies + candies_given_by_father = 93 :=
by
  intros
  sorry

end lillian_candies_total_l157_157223


namespace cannot_form_right_triangle_l157_157742

theorem cannot_form_right_triangle (a b c : ℕ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) : 
  a^2 + b^2 ≠ c^2 :=
by 
  rw [h_a, h_b, h_c]
  sorry

end cannot_form_right_triangle_l157_157742


namespace max_balls_possible_l157_157859

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l157_157859


namespace solve_for_t_l157_157187

variable (S₁ S₂ u t : ℝ)

theorem solve_for_t 
  (h₀ : u ≠ 0) 
  (h₁ : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
by
  sorry

end solve_for_t_l157_157187


namespace max_area_225_l157_157084

noncomputable def max_area_rect_perim60 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) : ℝ :=
max (x * y) (30 - x)

theorem max_area_225 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) :
  max_area_rect_perim60 x y h1 h2 = 225 :=
sorry

end max_area_225_l157_157084


namespace find_PO_l157_157658

variables {P : ℝ × ℝ} {O F : ℝ × ℝ}

def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def isosceles_triangle (O P F : ℝ × ℝ) : Prop :=
  dist O P = dist O F ∨ dist O P = dist P F

theorem find_PO
  (P : ℝ × ℝ) (O : ℝ × ℝ) (F : ℝ × ℝ)
  (hO : origin O) (hF : focus F) (hP : on_parabola P) (h_iso : isosceles_triangle O P F) :
  dist O P = 1 ∨ dist O P = 3 / 2 :=
sorry

end find_PO_l157_157658


namespace hyperbola_asymptote_eqn_l157_157191

theorem hyperbola_asymptote_eqn :
  ∀ (x y : ℝ),
  (y ^ 2 / 4 - x ^ 2 = 1) → (y = 2 * x ∨ y = -2 * x) := by
sorry

end hyperbola_asymptote_eqn_l157_157191


namespace relationship_between_p_and_q_l157_157670

theorem relationship_between_p_and_q (p q : ℝ) 
  (h : ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (2*x)^2 + p*(2*x) + q = 0) :
  2 * p^2 = 9 * q :=
sorry

end relationship_between_p_and_q_l157_157670


namespace min_value_fraction_l157_157478

theorem min_value_fraction (a b : ℝ) (h : x^2 - 3*x + a*b < 0 ∧ 1 < x ∧ x < 2) (h1 : a > b) : 
  (∃ minValue : ℝ, minValue = 4 ∧ ∀ a b : ℝ, a > b → minValue ≤ (a^2 + b^2) / (a - b)) := 
sorry

end min_value_fraction_l157_157478


namespace stan_needs_more_minutes_l157_157416

/-- Stan has 10 songs each of 3 minutes and 15 songs each of 2 minutes. His run takes 100 minutes.
    Prove that he needs 40 more minutes of songs in his playlist. -/
theorem stan_needs_more_minutes 
    (num_3min_songs : ℕ) 
    (num_2min_songs : ℕ) 
    (time_per_3min_song : ℕ) 
    (time_per_2min_song : ℕ) 
    (total_run_time : ℕ) 
    (given_minutes_3min_songs : num_3min_songs = 10)
    (given_minutes_2min_songs : num_2min_songs = 15)
    (given_time_per_3min_song : time_per_3min_song = 3)
    (given_time_per_2min_song : time_per_2min_song = 2)
    (given_total_run_time : total_run_time = 100)
    : num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song = 60 →
      total_run_time - (num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song) = 40 := 
by
    sorry

end stan_needs_more_minutes_l157_157416


namespace spring_length_at_9kg_l157_157922

theorem spring_length_at_9kg :
  (∃ (k b : ℝ), (∀ x : ℝ, y = k * x + b) ∧ 
                 (y = 10 ∧ x = 0) ∧ 
                 (y = 10.5 ∧ x = 1)) → 
  (∀ x : ℝ, x = 9 → y = 14.5) :=
sorry

end spring_length_at_9kg_l157_157922


namespace fodder_lasting_days_l157_157743

theorem fodder_lasting_days (buffalo_fodder_rate cow_fodder_rate ox_fodder_rate : ℕ)
  (initial_buffaloes initial_cows initial_oxen added_buffaloes added_cows initial_days : ℕ)
  (h1 : 3 * buffalo_fodder_rate = 4 * cow_fodder_rate)
  (h2 : 3 * buffalo_fodder_rate = 2 * ox_fodder_rate)
  (h3 : initial_days * (initial_buffaloes * buffalo_fodder_rate + initial_cows * cow_fodder_rate + initial_oxen * ox_fodder_rate) = 4320) :
  (4320 / ((initial_buffaloes + added_buffaloes) * buffalo_fodder_rate + (initial_cows + added_cows) * cow_fodder_rate + initial_oxen * ox_fodder_rate)) = 9 :=
by 
  sorry

end fodder_lasting_days_l157_157743


namespace relationship_among_abc_l157_157959

noncomputable def a : ℝ := 2^(1.2)
noncomputable def b : ℝ := (1 / 2)^(-0.8)
noncomputable def c : ℝ := 2 * log 5 2

theorem relationship_among_abc : c < b ∧ b < a :=
by
  -- we'll just leave sorry for the actual proof
  sorry

end relationship_among_abc_l157_157959


namespace average_of_multiples_of_9_l157_157797

-- Define the problem in Lean
theorem average_of_multiples_of_9 :
  let pos_multiples := [9, 18, 27, 36, 45]
  let neg_multiples := [-9, -18, -27, -36, -45]
  (pos_multiples.sum + neg_multiples.sum) / 2 = 0 :=
by
  sorry

end average_of_multiples_of_9_l157_157797


namespace focus_of_parabola_l157_157012

noncomputable def parabola_focus (a h k : ℝ) : ℝ × ℝ :=
  (h, k + 1 / (4 * a))

theorem focus_of_parabola :
  parabola_focus 9 (-1/3) (-3) = (-1/3, -107/36) := 
  sorry

end focus_of_parabola_l157_157012


namespace mb_less_than_neg_one_point_five_l157_157424

theorem mb_less_than_neg_one_point_five (m b : ℚ) (h1 : m = 3/4) (h2 : b = -2) : m * b < -1.5 :=
by {
  -- sorry skips the proof
  sorry
}

end mb_less_than_neg_one_point_five_l157_157424


namespace solve_system_of_equations_l157_157814

theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : 2 * x + y = 21) : x + y = 9 := by
  sorry

end solve_system_of_equations_l157_157814


namespace first_tap_fill_time_l157_157765

theorem first_tap_fill_time (T : ℚ) :
  (∀ (second_tap_empty_time : ℚ), second_tap_empty_time = 8) →
  (∀ (combined_fill_time : ℚ), combined_fill_time = 40 / 3) →
  (1/T - 1/8 = 3/40) →
  T = 5 :=
by
  intros h1 h2 h3
  sorry

end first_tap_fill_time_l157_157765


namespace inscribed_square_area_l157_157554

noncomputable def area_inscribed_square (AB CD : ℕ) (BCFE : ℕ) : Prop :=
  AB = 36 ∧ CD = 64 ∧ BCFE = (AB * CD)

theorem inscribed_square_area :
  ∀ (AB CD : ℕ),
  area_inscribed_square AB CD 2304 :=
by
  intros
  sorry

end inscribed_square_area_l157_157554


namespace count_whole_numbers_in_interval_l157_157821

open Real

theorem count_whole_numbers_in_interval : 
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℕ, (sqrt 7 < x ∧ x < exp 2) ↔ (3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end count_whole_numbers_in_interval_l157_157821


namespace total_morning_afternoon_emails_l157_157529

-- Define the conditions
def morning_emails : ℕ := 5
def afternoon_emails : ℕ := 8
def evening_emails : ℕ := 72

-- State the proof problem
theorem total_morning_afternoon_emails : 
  morning_emails + afternoon_emails = 13 := by
  sorry

end total_morning_afternoon_emails_l157_157529


namespace shifted_parabola_eq_l157_157520

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end shifted_parabola_eq_l157_157520


namespace krishan_money_l157_157892

theorem krishan_money
  (R G K : ℕ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 637) : 
  K = 3774 := 
by
  sorry

end krishan_money_l157_157892


namespace aimee_poll_l157_157297

theorem aimee_poll (W P : ℕ) (h1 : 0.35 * W = 21) (h2 : 2 * W = P) : P = 120 :=
by
  -- proof in Lean is omitted, placeholder
  sorry

end aimee_poll_l157_157297


namespace range_of_a_l157_157985

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 2 * x > x^2 + a) → a < -8 :=
by
  intro h
  -- Complete the proof by showing that 2x - x^2 has a minimum value of -8 on [-2, 3] and hence proving a < -8.
  sorry

end range_of_a_l157_157985


namespace brownie_pan_dimensions_l157_157032

def brownie_dimensions (m n : ℕ) : Prop :=
  let numSectionsLength := m - 1
  let numSectionsWidth := n - 1
  let totalPieces := (numSectionsLength + 1) * (numSectionsWidth + 1)
  let interiorPieces := (numSectionsLength - 1) * (numSectionsWidth - 1)
  let perimeterPieces := totalPieces - interiorPieces
  (numSectionsLength = 3) ∧ (numSectionsWidth = 5) ∧ (interiorPieces = 2 * perimeterPieces)

theorem brownie_pan_dimensions :
  ∃ (m n : ℕ), brownie_dimensions m n ∧ m = 6 ∧ n = 12 :=
by
  existsi 6
  existsi 12
  unfold brownie_dimensions
  simp
  exact sorry

end brownie_pan_dimensions_l157_157032


namespace rectangle_area_l157_157130

variable {x : ℝ} (h : x > 0)

theorem rectangle_area (W : ℝ) (L : ℝ) (hL : L = 3 * W) (h_diag : W^2 + L^2 = x^2) :
  (W * L) = (3 / 10) * x^2 := by
  sorry

end rectangle_area_l157_157130


namespace min_d_value_l157_157497

noncomputable def minChordLength (a : ℝ) : ℝ :=
  let P1 := (Real.arcsin a, Real.arcsin a)
  let P2 := (Real.arccos a, -Real.arccos a)
  let d_sq := 2 * ((Real.arcsin a)^2 + (Real.arccos a)^2)
  Real.sqrt d_sq

theorem min_d_value {a : ℝ} (h₁ : a ∈ Set.Icc (-1) 1) : 
  ∃ d : ℝ, d = minChordLength a ∧ d ≥ (π / 2) :=
sorry

end min_d_value_l157_157497


namespace multiplication_result_l157_157870

theorem multiplication_result :
  121 * 54 = 6534 := by
  sorry

end multiplication_result_l157_157870


namespace ratio_of_lengths_l157_157603

variables (l1 l2 l3 : ℝ)

theorem ratio_of_lengths (h1 : l2 = (1/2) * (l1 + l3))
                         (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
  (l1 / l3) = (7 / 5) :=
by
  sorry

end ratio_of_lengths_l157_157603


namespace trig_triple_angle_l157_157668

theorem trig_triple_angle (θ : ℝ) (h : Real.tan θ = 5) :
  Real.tan (3 * θ) = 55 / 37 ∧
  Real.sin (3 * θ) = 55 * Real.sqrt 1369 / (37 * Real.sqrt 4394) ∨ Real.sin (3 * θ) = -(55 * Real.sqrt 1369 / (37 * Real.sqrt 4394)) ∧
  Real.cos (3 * θ) = Real.sqrt (1369 / 4394) ∨ Real.cos (3 * θ) = -Real.sqrt (1369 / 4394) :=
by
  sorry

end trig_triple_angle_l157_157668


namespace correct_conclusions_l157_157661

-- Define the initial polynomials 
def p1 := (x : ℝ) => x
def p2 := (x y : ℝ) => 2 * x + y

-- Define conditions
def sqrt_condition (x y : ℝ) := ∃ (x y : ℝ), sqrt (x - 1) + abs (y - 2) = 0
def linear_condition (x y : ℝ) := 3 * x + y = 1

-- Define calculation of sums and verifying value of n
def M_n (n : ℕ) (x y : ℝ) := (1 / 2 + 2 ^ (n - 1)) * (3 * x + y)
def delta (n : ℕ) (x y : ℝ) := M_n n x y - M_n (n - 2) x y

theorem correct_conclusions :
  (sqrt_condition 1 2 → M_n 4 1 2 = 42.5) ∧
  ¬(∃ (c : ℝ), c = 33 / 32) ∧
  (linear_condition 2 1 → delta 13 1 (-2 / 3) = 3072) :=
by
  sorry

end correct_conclusions_l157_157661


namespace negation_of_proposition_l157_157890

variable (f : ℕ+ → ℕ)

theorem negation_of_proposition :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) :=
by sorry

end negation_of_proposition_l157_157890


namespace negation_of_p_is_neg_p_l157_157719

-- Define the original proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Define what it means for the negation of p to be satisfied
def neg_p := ∀ n : ℕ, 2^n ≤ 100

-- Statement to prove the logical equivalence between the negation of p and neg_p
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l157_157719


namespace total_journey_distance_l157_157288

variable (D : ℝ) (T : ℝ) (v₁ : ℝ) (v₂ : ℝ)

theorem total_journey_distance :
  T = 10 → 
  v₁ = 21 → 
  v₂ = 24 → 
  (T = (D / (2 * v₁)) + (D / (2 * v₂))) → 
  D = 224 :=
by
  intros hT hv₁ hv₂ hDistance
  -- Proof goes here
  sorry

end total_journey_distance_l157_157288


namespace uncovered_area_l157_157238

def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side : ℕ := 4

theorem uncovered_area (height width side : ℕ) (h : height = shoebox_height) (w : width = shoebox_width) (s : side = block_side) :
  (width * height) - (side * side) = 8 :=
by
  rw [h, w, s]
  -- Area of shoebox bottom = width * height
  -- Area of square block = side * side
  -- Uncovered area = (width * height) - (side * side)
  -- Therefore, (6 * 4) - (4 * 4) = 24 - 16 = 8
  sorry

end uncovered_area_l157_157238


namespace initial_bottles_of_water_l157_157054

theorem initial_bottles_of_water {B : ℕ} (h1 : 100 - (6 * B + 5) = 71) : B = 4 :=
by
  sorry

end initial_bottles_of_water_l157_157054


namespace problem_statement_l157_157363

theorem problem_statement (A B : ℝ) (hA : A = 10 * π / 180) (hB : B = 35 * π / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
  1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) + Real.tan A * (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) :=
by
  sorry

end problem_statement_l157_157363


namespace max_balls_drawn_l157_157853

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l157_157853


namespace original_price_of_radio_l157_157768

theorem original_price_of_radio (P : ℝ) (h : 0.95 * P = 465.5) : P = 490 :=
sorry

end original_price_of_radio_l157_157768


namespace set_equality_x_plus_y_l157_157915

theorem set_equality_x_plus_y (x y : ℝ) (A B : Set ℝ) (hA : A = {0, |x|, y}) (hB : B = {x, x * y, Real.sqrt (x - y)}) (h : A = B) : x + y = -2 :=
by
  sorry

end set_equality_x_plus_y_l157_157915


namespace isosceles_triangle_perimeter_l157_157962

theorem isosceles_triangle_perimeter (m : ℝ) (a b : ℝ) 
  (h1 : 3 = a ∨ 3 = b)
  (h2 : a ≠ b)
  (h3 : a^2 - (m+1)*a + 2*m = 0)
  (h4 : b^2 - (m+1)*b + 2*m = 0) :
  (a + b + a = 11) ∨ (a + a + b = 10) := 
sorry

end isosceles_triangle_perimeter_l157_157962


namespace find_a_l157_157836

-- Definitions matching the conditions
def seq (a b c d : ℤ) := [a, b, c, d, 0, 1, 1, 2, 3, 5, 8]

-- Conditions provided in the problem
def fib_property (a b c d : ℤ) : Prop :=
    d + 0 = 1 ∧ 
    c + 1 = 0 ∧ 
    b + (-1) = 1 ∧ 
    a + 2 = -1

-- Theorem statement to prove
theorem find_a (a b c d : ℤ) (h : fib_property a b c d) : a = -3 :=
by
  sorry

end find_a_l157_157836


namespace determine_students_and_benches_l157_157829

theorem determine_students_and_benches (a b s : ℕ) :
  (s = a * b + 5) ∧ (s = 8 * b - 4) →
  ((a = 7 ∧ b = 9 ∧ s = 68) ∨ (a = 5 ∧ b = 3 ∧ s = 20)) :=
by
  sorry

end determine_students_and_benches_l157_157829


namespace count_solutions_l157_157511

theorem count_solutions :
  ∃ (n : ℕ), (∀ (x y z : ℕ), x * y * z + x * y + y * z + z * x + x + y + z = 2012 ↔ n = 27) :=
sorry

end count_solutions_l157_157511


namespace simplify_expression_l157_157706

theorem simplify_expression (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 :=
sorry

end simplify_expression_l157_157706


namespace compound_interest_rate_l157_157158

theorem compound_interest_rate
  (P A t n r : ℝ)
  (P_condition : P = 50000)
  (t_condition : t = 2)
  (n_condition : n = 2)
  (A_condition : A = 54121.608) :
  (1 + r / n)^(n * t) = A / P → r ≈ 0.0398 := by
sorry

end compound_interest_rate_l157_157158


namespace stratified_sampling_grade11_l157_157127

noncomputable def g10 : ℕ := 500
noncomputable def total_students : ℕ := 1350
noncomputable def g10_sample : ℕ := 120
noncomputable def ratio : ℚ := g10_sample / g10
noncomputable def g11 : ℕ := 450
noncomputable def g12 : ℕ := g11 - 50

theorem stratified_sampling_grade11 :
  g10 + g11 + g12 = total_students →
  (g10_sample / g10) = ratio →
  sample_g11 = g11 * ratio →
  sample_g11 = 108 :=
by
  sorry

end stratified_sampling_grade11_l157_157127


namespace find_a_l157_157022

-- Define the real numbers x, y, and a
variables (x y a : ℝ)

-- Define the conditions as premises
axiom cond1 : x + 3 * y + 5 ≥ 0
axiom cond2 : x + y - 1 ≤ 0
axiom cond3 : x + a ≥ 0

-- Define z as x + 2y and state its minimum value is -4
def z : ℝ := x + 2 * y
axiom min_z : z = -4

-- The theorem to prove the value of a given the above conditions
theorem find_a : a = 2 :=
sorry

end find_a_l157_157022


namespace compare_exponents_and_logs_l157_157826

theorem compare_exponents_and_logs :
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  a > b ∧ b > c :=
by
  -- Definitions from the conditions
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  -- Proof here (omitted)
  sorry

end compare_exponents_and_logs_l157_157826


namespace range_of_a_l157_157225

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∨ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) ∧ ¬ (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∧ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) →
  a ≤ -2 ∨ 1 ≤ a ∧ a < 2 :=
by {
  sorry
}

end range_of_a_l157_157225


namespace percent_psychology_majors_l157_157464

theorem percent_psychology_majors
  (total_students : ℝ)
  (pct_freshmen : ℝ)
  (pct_freshmen_liberal_arts : ℝ)
  (pct_freshmen_psychology_majors : ℝ)
  (h1 : pct_freshmen = 0.6)
  (h2 : pct_freshmen_liberal_arts = 0.4)
  (h3 : pct_freshmen_psychology_majors = 0.048)
  :
  (pct_freshmen_psychology_majors / (pct_freshmen * pct_freshmen_liberal_arts)) * 100 = 20 := 
by
  sorry

end percent_psychology_majors_l157_157464


namespace combined_spots_l157_157977

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end combined_spots_l157_157977


namespace subcommittee_count_l157_157750

theorem subcommittee_count :
  let nR := 10 in 
  let nD := 8 in 
  let kR := 4 in 
  let kD := 3 in 
  (nat.choose nR kR) * (nat.choose nD kD) = 11760 := 
by 
  let nR := 10
  let nD := 8
  let kR := 4
  let kD := 3
  -- sorry replaces the actual proof steps
  sorry

end subcommittee_count_l157_157750


namespace sum_of_squares_l157_157556

def positive_integers (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0

def sum_of_values (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧ Int.gcd x y + Int.gcd y z + Int.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h1 : positive_integers x y z) (h2 : sum_of_values x y z) :
  x^2 + y^2 + z^2 = 296 :=
by sorry

end sum_of_squares_l157_157556


namespace arithmetic_sequence_formula_geometric_sequence_sum_l157_157491

variables {a_n S_n b_n T_n : ℕ → ℚ} {a_3 S_3 a_5 b_3 T_3 : ℚ} {q : ℚ}

def is_arithmetic_sequence (a_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, a_n n = a_1 + (n - 1) * d

def sum_first_n_arithmetic (S_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

def is_geometric_sequence (b_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, b_n n = b_1 * q^(n-1)

def sum_first_n_geometric (T_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, T_n n = if q = 1 then n * b_1 else b_1 * (1 - q^n) / (1 - q)

theorem arithmetic_sequence_formula {a_1 d : ℚ} (h_arith : is_arithmetic_sequence a_n a_1 d)
    (h_sum : sum_first_n_arithmetic S_n a_1 d) (h1 : a_n 3 = 5) (h2 : S_n 3 = 9) :
    ∀ n, a_n n = 2 * n - 1 := sorry

theorem geometric_sequence_sum {b_1 : ℚ} (h_geom : is_geometric_sequence b_n b_1 q)
    (h_sum : sum_first_n_geometric T_n b_1 q) (h3 : q > 0) (h4 : b_n 3 = a_n 5) (h5 : T_n 3 = 13) :
    ∀ n, T_n n = (3^n - 1) / 2 := sorry

end arithmetic_sequence_formula_geometric_sequence_sum_l157_157491


namespace find_a_2016_l157_157725

theorem find_a_2016 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 1, a (n + 1) = 3 * S n)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)):
  a 2016 = 3 * 4 ^ 2014 := 
by 
  sorry

end find_a_2016_l157_157725


namespace smallest_prime_angle_l157_157049

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_angle :
  ∃ (x : ℕ), is_prime x ∧ is_prime (2 * x) ∧ x + 2 * x = 90 ∧ x = 29 :=
by sorry

end smallest_prime_angle_l157_157049


namespace cost_per_yellow_ink_l157_157414

def initial_amount : ℕ := 50
def cost_per_black_ink : ℕ := 11
def num_black_inks : ℕ := 2
def cost_per_red_ink : ℕ := 15
def num_red_inks : ℕ := 3
def additional_amount_needed : ℕ := 43
def num_yellow_inks : ℕ := 2

theorem cost_per_yellow_ink :
  let total_cost_needed := initial_amount + additional_amount_needed
  let total_black_ink_cost := cost_per_black_ink * num_black_inks
  let total_red_ink_cost := cost_per_red_ink * num_red_inks
  let total_non_yellow_cost := total_black_ink_cost + total_red_ink_cost
  let total_yellow_ink_cost := total_cost_needed - total_non_yellow_cost
  let cost_per_yellow_ink := total_yellow_ink_cost / num_yellow_inks
  cost_per_yellow_ink = 13 :=
by
  sorry

end cost_per_yellow_ink_l157_157414


namespace find_common_divisor_l157_157721

open Int

theorem find_common_divisor (n : ℕ) (h1 : 2287 % n = 2028 % n)
  (h2 : 2028 % n = 1806 % n) : n = Int.gcd (Int.gcd 259 222) 481 := by
  sorry -- Proof goes here

end find_common_divisor_l157_157721


namespace eight_distinct_solutions_l157_157537

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

theorem eight_distinct_solutions : 
  ∃ S : Finset ℝ, S.card = 8 ∧ ∀ x ∈ S, f (f (f x)) = x :=
sorry

end eight_distinct_solutions_l157_157537


namespace total_legs_arms_proof_l157_157465

/-
There are 4 birds, each with 2 legs.
There are 6 dogs, each with 4 legs.
There are 5 snakes, each with no legs.
There are 2 spiders, each with 8 legs.
There are 3 horses, each with 4 legs.
There are 7 rabbits, each with 4 legs.
There are 2 octopuses, each with 8 arms.
There are 8 ants, each with 6 legs.
There is 1 unique creature with 12 legs.
We need to prove that the total number of legs and arms is 164.
-/

def total_legs_arms : Nat := 
  (4 * 2) + (6 * 4) + (5 * 0) + (2 * 8) + (3 * 4) + (7 * 4) + (2 * 8) + (8 * 6) + (1 * 12)

theorem total_legs_arms_proof : total_legs_arms = 164 := by
  sorry

end total_legs_arms_proof_l157_157465


namespace probability_three_red_balls_l157_157046

open scoped BigOperators

noncomputable def hypergeometric_prob (r : ℕ) (b : ℕ) (k : ℕ) (d : ℕ) : ℝ :=
  (Nat.choose r d * Nat.choose b (k - d) : ℝ) / Nat.choose (r + b) k

theorem probability_three_red_balls :
  hypergeometric_prob 10 5 5 3 = 1200 / 3003 :=
by sorry

end probability_three_red_balls_l157_157046


namespace proof_problem_l157_157115

def work_problem :=
  ∃ (B : ℝ),
  (1 / 6) + (1 / B) + (1 / 24) = (1 / 3) ∧ B = 8

theorem proof_problem : work_problem :=
by
  sorry

end proof_problem_l157_157115


namespace value_of_a_l157_157377

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l157_157377


namespace quadratic_is_perfect_square_l157_157231

theorem quadratic_is_perfect_square (a b c x : ℝ) (h : b^2 - 4 * a * c = 0) :
  a * x^2 + b * x + c = 0 ↔ (2 * a * x + b)^2 = 0 := 
by
  sorry

end quadratic_is_perfect_square_l157_157231


namespace equivalent_expression_l157_157337

theorem equivalent_expression :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) =
  5^128 - 2^128 := by
  sorry

end equivalent_expression_l157_157337


namespace sock_pairing_l157_157047

def sockPicker : Prop :=
  let white_socks := 5
  let brown_socks := 5
  let blue_socks := 2
  let total_socks := 12
  let choose (n k : ℕ) := Nat.choose n k
  (choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 21) ∧
  (choose (white_socks + brown_socks) 2 = 45) ∧
  (45 = 45)

theorem sock_pairing :
  sockPicker :=
by sorry

end sock_pairing_l157_157047


namespace area_of_quadrilateral_PQRS_l157_157577

noncomputable def calculate_area_of_quadrilateral_PQRS (PQ PR : ℝ) (PS_corrected : ℝ) : ℝ :=
  let area_ΔPQR := (1/2) * PQ * PR
  let RS := Real.sqrt (PR^2 - PQ^2)
  let area_ΔPRS := (1/2) * PR * RS
  area_ΔPQR + area_ΔPRS

theorem area_of_quadrilateral_PQRS :
  let PQ := 8
  let PR := 10
  let PS_corrected := Real.sqrt (PQ^2 + PR^2)
  calculate_area_of_quadrilateral_PQRS PQ PR PS_corrected = 70 := 
by
  sorry

end area_of_quadrilateral_PQRS_l157_157577


namespace fixed_point_of_tangent_line_l157_157947

theorem fixed_point_of_tangent_line (x y : ℝ) (h1 : x = 3) 
  (h2 : ∃ m : ℝ, (3 - m)^2 + (y - 2)^2 = 4) :
  ∃ (k l : ℝ), k = 4 / 3 ∧ l = 2 :=
by
  sorry

end fixed_point_of_tangent_line_l157_157947


namespace rectangular_prism_volume_l157_157339

theorem rectangular_prism_volume (a b c V : ℝ) (h1 : a * b = 20) (h2 : b * c = 12) (h3 : a * c = 15) (hb : b = 5) : V = 75 :=
  sorry

end rectangular_prism_volume_l157_157339


namespace symmetric_point_exists_l157_157479

-- Define the point M
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point M
def M : Point3D := { x := 3, y := 3, z := 3 }

-- Define the parametric form of the line
def line (t : ℝ) : Point3D := { x := 1 - t, y := 1.5, z := 3 + t }

-- Define the point M' that we want to prove is symmetrical to M with respect to the line
def symmPoint : Point3D := { x := 1, y := 0, z := 1 }

-- The theorem that we need to prove, ensuring M' is symmetrical to M with respect to the given line
theorem symmetric_point_exists : ∃ t, line t = symmPoint ∧ 
  (∀ M_0 : Point3D, M_0.x = (M.x + symmPoint.x) / 2 ∧ M_0.y = (M.y + symmPoint.y) / 2 ∧ M_0.z = (M.z + symmPoint.z) / 2)
  → line t = M_0
  → M_0 = { x := 2, y := 1.5, z := 2 } := 
by
  sorry

end symmetric_point_exists_l157_157479


namespace tangent_eq_inequality_not_monotonic_l157_157498

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / (x + a)

theorem tangent_eq (a : ℝ) (h : 0 < a) : 
  ∃ k : ℝ, (k, f 1 a) ∈ {
    p : ℝ × ℝ | p.1 - (a + 1) * p.2 - 1 = 0 
  } :=
  sorry

theorem inequality (x : ℝ) (h : 1 ≤ x) : f x 1 ≤ (x - 1) / 2 := 
  sorry

theorem not_monotonic (a : ℝ) (h : 0 < a) : 
  ¬(∀ x y : ℝ, x < y → f x a ≤ f y a ∨ x < y → f x a ≥ f y a) := 
  sorry

end tangent_eq_inequality_not_monotonic_l157_157498


namespace distance_between_first_and_last_tree_l157_157241

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 100) 
  (h3 : d / 5 = 20) :
  (20 * 9 = 180) :=
by
  sorry

end distance_between_first_and_last_tree_l157_157241


namespace circumcircle_of_right_triangle_l157_157522

theorem circumcircle_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  ∃ (x y : ℝ), (x - 0)^2 + (y - 0)^2 = 25 :=
by
  sorry

end circumcircle_of_right_triangle_l157_157522


namespace range_of_a_l157_157362

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 + a * x + 1
noncomputable def quadratic_eq (x₀ a : ℝ) : Prop := x₀^2 - x₀ + a = 0

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, quadratic a x > 0) (q : ∃ x₀ : ℝ, quadratic_eq x₀ a) : 0 ≤ a ∧ a ≤ 1/4 :=
  sorry

end range_of_a_l157_157362


namespace probability_first_white_second_red_l157_157118

noncomputable def marble_probability (total_marbles first_white second_red : ℚ) : ℚ :=
  first_white * second_red

theorem probability_first_white_second_red :
  let total_marbles := 10 in
  let first_white := 6 / total_marbles in
  let second_red_given_white := 4 / (total_marbles - 1) in
  marble_probability total_marbles first_white second_red_given_white = 4 / 15 :=
by
  sorry

end probability_first_white_second_red_l157_157118


namespace cos_of_angle_in_third_quadrant_l157_157185

theorem cos_of_angle_in_third_quadrant 
  (α : ℝ)
  (h1 : π < α ∧ α < (3 * π) / 2)
  (h2 : tan α = 1 / 2) :
  cos α = - (2 * Real.sqrt 5 / 5) :=
sorry

end cos_of_angle_in_third_quadrant_l157_157185


namespace find_coeff_a9_l157_157810

theorem find_coeff_a9 (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (x^3 + x^10 = a + a1 * (x + 1) + a2 * (x + 1)^2 + 
  a3 * (x + 1)^3 + a4 * (x + 1)^4 + a5 * (x + 1)^5 + 
  a6 * (x + 1)^6 + a7 * (x + 1)^7 + a8 * (x + 1)^8 + 
  a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a9 = -10 :=
sorry

end find_coeff_a9_l157_157810


namespace compare_neg_fractions_l157_157328

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l157_157328


namespace xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l157_157795

theorem xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔ 
  (∃ a : ℕ, 0 < a ∧ x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by
  sorry

end xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l157_157795


namespace jacqueline_has_29_percent_more_soda_than_liliane_l157_157215

variable (A : ℝ) -- A is the amount of soda Alice has

-- Define the amount of soda Jacqueline has
def J (A : ℝ) : ℝ := 1.80 * A

-- Define the amount of soda Liliane has
def L (A : ℝ) : ℝ := 1.40 * A

-- The statement that needs to be proven
theorem jacqueline_has_29_percent_more_soda_than_liliane (A : ℝ) (hA : A > 0) : 
  ((J A - L A) / L A) * 100 = 29 :=
by
  sorry

end jacqueline_has_29_percent_more_soda_than_liliane_l157_157215


namespace raft_drift_time_l157_157136

theorem raft_drift_time (s : ℝ) (v_down v_up v_c : ℝ) 
  (h1 : v_down = s / 3) 
  (h2 : v_up = s / 4) 
  (h3 : v_down = v_c + v_c)
  (h4 : v_up = v_c - v_c) :
  v_c = s / 24 → (s / v_c) = 24 := 
by
  sorry

end raft_drift_time_l157_157136


namespace number_of_triangles_fitting_in_square_l157_157035

-- Define the conditions for the right triangle and the square
def right_triangle_height := 2
def right_triangle_width := 2
def square_side := 2

-- Define the areas
def area_triangle := (1 / 2) * right_triangle_height * right_triangle_width
def area_square := square_side * square_side

-- Define the proof statement to show the number of right triangles fitting in the square is 2
theorem number_of_triangles_fitting_in_square : (area_square / area_triangle) = 2 := by
  sorry

end number_of_triangles_fitting_in_square_l157_157035


namespace lead_points_l157_157432

-- Define final scores
def final_score_team : ℕ := 68
def final_score_green : ℕ := 39

-- Prove the lead
theorem lead_points : final_score_team - final_score_green = 29 :=
by
  sorry

end lead_points_l157_157432


namespace min_people_liking_both_l157_157545

theorem min_people_liking_both {A B U : Finset ℕ} (hU : U.card = 150) (hA : A.card = 130) (hB : B.card = 120) :
  (A ∩ B).card ≥ 100 :=
by
  -- Proof to be filled later
  sorry

end min_people_liking_both_l157_157545


namespace problem1_problem2_l157_157468

-- Theorem for problem 1
theorem problem1 (a b : ℤ) : (a^3 * b^4) ^ 2 / (a * b^2) ^ 3 = a^3 * b^2 := 
by sorry

-- Theorem for problem 2
theorem problem2 (a : ℤ) : (-a^2) ^ 3 * a^2 + a^8 = 0 := 
by sorry

end problem1_problem2_l157_157468


namespace average_marks_l157_157258

-- Define the conditions
variables (M P C : ℕ)
axiom condition1 : M + P = 30
axiom condition2 : C = P + 20

-- Define the target statement
theorem average_marks : (M + C) / 2 = 25 :=
by
  sorry

end average_marks_l157_157258


namespace squares_with_center_25_60_l157_157849

theorem squares_with_center_25_60 :
  let center_x := 25
  let center_y := 60
  let non_neg_int_coords (x : ℤ) (y : ℤ) := x ≥ 0 ∧ y ≥ 0
  let is_center (x : ℤ) (y : ℤ) := x = center_x ∧ y = center_y
  let num_squares := 650
  ∃ n : ℤ, (n = num_squares) ∧ ∀ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ), 
    non_neg_int_coords x₁ y₁ ∧ non_neg_int_coords x₂ y₂ ∧ 
    non_neg_int_coords x₃ y₃ ∧ non_neg_int_coords x₄ y₄ ∧ 
    is_center ((x₁ + x₂ + x₃ + x₄) / 4) ((y₁ + y₂ + y₃ + y₄) / 4) → 
    ∃ (k : ℤ), n = 650 :=
sorry

end squares_with_center_25_60_l157_157849


namespace at_least_one_gt_one_l157_157392

variable (a b : ℝ)

theorem at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l157_157392


namespace negative_fraction_comparison_l157_157306

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l157_157306


namespace man_speed_with_stream_l157_157595

variable (V_m V_as : ℝ)
variable (V_s V_ws : ℝ)

theorem man_speed_with_stream
  (cond1 : V_m = 5)
  (cond2 : V_as = 8)
  (cond3 : V_as = V_m - V_s)
  (cond4 : V_ws = V_m + V_s) :
  V_ws = 8 := 
by
  sorry

end man_speed_with_stream_l157_157595


namespace oldest_son_cookies_l157_157530

def youngest_son_cookies : Nat := 2
def total_cookies : Nat := 54
def days : Nat := 9

theorem oldest_son_cookies : ∃ x : Nat, 9 * (x + youngest_son_cookies) = total_cookies ∧ x = 4 := by
  sorry

end oldest_son_cookies_l157_157530


namespace problem_statement_l157_157981

variable {x : Real}
variable {m : Int}
variable {n : Int}

theorem problem_statement (h1 : x^m = 5) (h2 : x^n = 10) : x^(2 * m - n) = 5 / 2 :=
by
  sorry

end problem_statement_l157_157981


namespace lengths_available_total_cost_l157_157761

def available_lengths := [1, 2, 3, 4, 5, 6]
def pipe_prices := [10, 15, 20, 25, 30, 35]

-- Given conditions
def purchased_pipes := [2, 5]
def target_perimeter_is_even := True

-- Prove: 
theorem lengths_available (x : ℕ) (hx : x ∈ available_lengths) : 
  3 < x ∧ x < 7 → x = 4 ∨ x = 5 ∨ x = 6 := by
  sorry

-- Prove: 
theorem total_cost (p : ℕ) (h : target_perimeter_is_even) : 
  p = 75 := by
  sorry

end lengths_available_total_cost_l157_157761


namespace power_of_two_divides_sub_one_l157_157590

theorem power_of_two_divides_sub_one (k : ℕ) (h_odd : k % 2 = 1) : ∀ n ≥ 1, 2^(n+2) ∣ k^(2^n) - 1 :=
by
  sorry

end power_of_two_divides_sub_one_l157_157590


namespace exists_prime_divisor_in_sequence_l157_157999

theorem exists_prime_divisor_in_sequence
  (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d)
  (a : ℕ → ℕ)
  (h0 : a 1 = c)
  (hs : ∀ n, a (n+1) = a n ^ d + c) :
  ∀ (n : ℕ), 2 ≤ n →
  ∃ (p : ℕ), Prime p ∧ p ∣ a n ∧ ∀ i, 1 ≤ i ∧ i < n → ¬ p ∣ a i := sorry

end exists_prime_divisor_in_sequence_l157_157999


namespace total_pears_picked_l157_157408

theorem total_pears_picked :
  let mike_pears := 8
  let jason_pears := 7
  let fred_apples := 6
  -- The total number of pears picked is the sum of Mike's and Jason's pears.
  mike_pears + jason_pears = 15 :=
by {
  sorry
}

end total_pears_picked_l157_157408


namespace z_value_l157_157267

theorem z_value (z : ℝ) (h : |z + 2| = |z - 3|) : z = 1 / 2 := 
sorry

end z_value_l157_157267


namespace marked_elements_duplicate_l157_157993

open Nat

def table : Matrix (Fin 4) (Fin 10) ℕ := ![
  ![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
  ![9, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
  ![8, 9, 0, 1, 2, 3, 4, 5, 6, 7], 
  ![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
]

theorem marked_elements_duplicate 
  (marked : Fin 4 → Fin 10) 
  (h_marked_unique_row : ∀ i1 i2, i1 ≠ i2 → marked i1 ≠ marked i2)
  (h_marked_unique_col : ∀ j, ∃ i, marked i = j) :
  ∃ i1 i2, i1 ≠ i2 ∧ table i1 (marked i1) = table i2 (marked i2) := sorry

end marked_elements_duplicate_l157_157993


namespace subcommittee_count_l157_157757

-- Define the conditions: number of Republicans and Democrats in the Senate committee
def numRepublicans : ℕ := 10
def numDemocrats : ℕ := 8
def chooseRepublicans : ℕ := 4
def chooseDemocrats : ℕ := 3

-- Define the main proof problem based on the conditions and the correct answer
theorem subcommittee_count :
  (Nat.choose numRepublicans chooseRepublicans) * (Nat.choose numDemocrats chooseDemocrats) = 11760 := by
  sorry

end subcommittee_count_l157_157757


namespace zigzag_lines_divide_regions_l157_157205

-- Define the number of regions created by n zigzag lines
def regions (n : ℕ) : ℕ := (2 * n * (2 * n + 1)) / 2 + 1 - 2 * n

-- Main theorem
theorem zigzag_lines_divide_regions (n : ℕ) : ∃ k : ℕ, k = regions n := by
  sorry

end zigzag_lines_divide_regions_l157_157205


namespace tangent_parallel_line_coordinates_l157_157988

theorem tangent_parallel_line_coordinates :
  ∃ (m n : ℝ), 
    (∀ x : ℝ, (deriv (λ x => x^4 + x) x = 4 * x^3 + 1)) ∧ 
    (deriv (λ x => x^4 + x) m = -3) ∧ 
    (n = m^4 + m) ∧ 
    (m, n) = (-1, 0) :=
by
  sorry

end tangent_parallel_line_coordinates_l157_157988


namespace extracurricular_hours_l157_157533

theorem extracurricular_hours :
  let soccer_hours_per_day := 2
  let soccer_days := 3
  let band_hours_per_day := 1.5
  let band_days := 2
  let total_soccer_hours := soccer_hours_per_day * soccer_days
  let total_band_hours := band_hours_per_day * band_days
  total_soccer_hours + total_band_hours = 9 := 
by
  -- The proof steps go here.
  sorry

end extracurricular_hours_l157_157533


namespace problem_l157_157473

theorem problem 
  (x : ℝ) 
  (h1 : x ∈ Set.Icc (-3 : ℝ) 3) 
  (h2 : x ≠ -5/3) : 
  (4 * x ^ 2 + 2) / (5 + 3 * x) ≥ 1 ↔ x ∈ (Set.Icc (-3) (-3/4) ∪ Set.Icc 1 3) :=
sorry

end problem_l157_157473


namespace pieces_per_package_l157_157550

-- Definitions from conditions
def total_pieces_of_gum : ℕ := 486
def number_of_packages : ℕ := 27

-- Mathematical statement to prove
theorem pieces_per_package : total_pieces_of_gum / number_of_packages = 18 := sorry

end pieces_per_package_l157_157550


namespace ninety_eight_times_ninety_eight_l157_157157

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 :=
by
  sorry

end ninety_eight_times_ninety_eight_l157_157157


namespace first_valve_fill_time_l157_157104

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end first_valve_fill_time_l157_157104


namespace closest_point_on_ellipse_to_line_l157_157159

theorem closest_point_on_ellipse_to_line :
  ∃ (x y : ℝ), 
    7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0 ∧ (x, y) = (3 / 2, -7 / 4) :=
by
  sorry

end closest_point_on_ellipse_to_line_l157_157159


namespace compare_rat_neg_l157_157333

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l157_157333


namespace no_integer_solutions_l157_157214

theorem no_integer_solutions (x y : ℤ) :
  ¬ (x^2 + 3 * x * y - 2 * y^2 = 122) :=
sorry

end no_integer_solutions_l157_157214


namespace find_prime_factors_l157_157302

-- Define n and the prime numbers p and q
def n : ℕ := 400000001
def p : ℕ := 20201
def q : ℕ := 19801

-- Main theorem statement
theorem find_prime_factors (hn : n = p * q) 
  (hp : Prime p) 
  (hq : Prime q) : 
  n = 400000001 ∧ p = 20201 ∧ q = 19801 := 
by {
  sorry
}

end find_prime_factors_l157_157302


namespace problem_f_2019_l157_157189

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f1 : f 1 = 1/4
axiom f2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2019 : f 2019 = -1/2 :=
by
  sorry

end problem_f_2019_l157_157189


namespace general_formula_l157_157178

def a (n : ℕ) : ℕ :=
match n with
| 0 => 1
| k+1 => 2 * a k + 4

theorem general_formula (n : ℕ) : a (n+1) = 5 * 2^n - 4 :=
by
  sorry

end general_formula_l157_157178


namespace symmetric_point_origin_l157_157880

theorem symmetric_point_origin (x y : Int) (hx : x = -(-4)) (hy : y = -(3)) :
    (x, y) = (4, -3) := by
  sorry

end symmetric_point_origin_l157_157880


namespace simplify_expression_l157_157553

theorem simplify_expression (x : ℝ) : 3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + x^3) = -x^3 + 9 * x^2 + 6 * x - 3 :=
by
  sorry -- Proof is omitted.

end simplify_expression_l157_157553


namespace part_a_part_b_part_c_l157_157276

/-- (a) Given that p = 33 and q = 216, show that the equation f(x) = 0 has 
three distinct integer solutions and the equation g(x) = 0 has two distinct integer solutions.
-/
theorem part_a (p q : ℕ) (h_p : p = 33) (h_q : q = 216) :
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = 216 ∧ x1 + x2 + x3 = 33 ∧ x1 = 0))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = 216 ∧ y1 + y1 = 22)) := sorry

/-- (b) Suppose that the equation f(x) = 0 has three distinct integer solutions 
and the equation g(x) = 0 has two distinct integer solutions. Prove the necessary conditions 
for p and q.
-/
theorem part_b (p q : ℕ) 
  (h_f : ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  (h_g : ∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p)) :
  (∃ k : ℕ, p = 3 * k) ∧ (∃ l : ℕ, q = 9 * l) ∧ (∃ m n : ℕ, p^2 - 3 * q = m^2 ∧ p^2 - 4 * q = n^2) := sorry

/-- (c) Prove that there are infinitely many pairs of positive integers (p, q) for which:
1. The equation f(x) = 0 has three distinct integer solutions.
2. The equation g(x) = 0 has two distinct integer solutions.
3. The greatest common divisor of p and q is 3.
-/
theorem part_c :
  ∃ (p q : ℕ) (infinitely_many : ℕ → Prop),
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p))
  ∧ ∃ k : ℕ, gcd p q = 3 ∧ infinitely_many k := sorry

end part_a_part_b_part_c_l157_157276


namespace garden_roller_area_l157_157285

theorem garden_roller_area (length : ℝ) (area_5rev : ℝ) (d1 d2 : ℝ) (π : ℝ) :
  length = 4 ∧ area_5rev = 88 ∧ π = 22 / 7 ∧ d2 = 1.4 →
  let circumference := π * d2
  let area_rev := circumference * length
  let new_area_5rev := 5 * area_rev
  new_area_5rev = 88 :=
by
  sorry

end garden_roller_area_l157_157285


namespace sum_of_interior_angles_hexagon_l157_157096

theorem sum_of_interior_angles_hexagon : 
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_hexagon_l157_157096


namespace BowlingAlleyTotalPeople_l157_157260

/--
There are 31 groups of people at the bowling alley.
Each group has about 6 people.
Prove that the total number of people at the bowling alley is 186.
-/
theorem BowlingAlleyTotalPeople : 
  let groups := 31
  let people_per_group := 6
  groups * people_per_group = 186 :=
by
  sorry

end BowlingAlleyTotalPeople_l157_157260


namespace damage_in_dollars_l157_157287

noncomputable def euros_to_dollars (euros : ℝ) : ℝ := euros * (1 / 0.9)

theorem damage_in_dollars :
  euros_to_dollars 45000000 = 49995000 :=
by
  -- This is where the proof would go
  sorry

end damage_in_dollars_l157_157287


namespace base_of_second_fraction_l157_157514

theorem base_of_second_fraction (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 :=
by
  sorry

end base_of_second_fraction_l157_157514


namespace triangle_side_range_a_l157_157257

theorem triangle_side_range_a {a : ℝ} : 2 < a ∧ a < 5 ↔
  3 + (2 * a + 1) > 8 ∧ 
  8 - 3 < 2 * a + 1 ∧ 
  8 - (2 * a + 1) < 3 :=
by
  sorry

end triangle_side_range_a_l157_157257


namespace find_a_value_l157_157369

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l157_157369


namespace intersection_line_canonical_equation_l157_157586

def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0
def canonical_equation (x y z : ℝ) : Prop := 
  (x - 1) / 35 = (y - 4 / 7) / 23 ∧ (y - 4 / 7) / 23 = z / 49

theorem intersection_line_canonical_equation (x y z : ℝ) :
  plane1 x y z → plane2 x y z → canonical_equation x y z :=
by
  intros h1 h2
  unfold plane1 at h1
  unfold plane2 at h2
  unfold canonical_equation
  sorry

end intersection_line_canonical_equation_l157_157586


namespace min_value_of_x3y2z_l157_157062

noncomputable def min_value_of_polynomial (x y z : ℝ) : ℝ :=
  x^3 * y^2 * z

theorem min_value_of_x3y2z
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1 / x + 1 / y + 1 / z = 9) :
  min_value_of_polynomial x y z = 1 / 46656 :=
sorry

end min_value_of_x3y2z_l157_157062


namespace hakeem_can_make_20_ounces_l157_157815

def artichokeDipNumberOfOunces (total_dollars: ℝ) (cost_per_artichoke: ℝ) (a_per_dip: ℝ) (o_per_dip: ℝ) : ℝ :=
  let artichoke_count := total_dollars / cost_per_artichoke
  let ounces_per_artichoke := o_per_dip / a_per_dip
  artichoke_count * ounces_per_artichoke

theorem hakeem_can_make_20_ounces:
  artichokeDipNumberOfOunces 15 1.25 3 5 = 20 :=
by
  sorry

end hakeem_can_make_20_ounces_l157_157815


namespace four_times_sum_of_squares_gt_sum_squared_l157_157219

open Real

theorem four_times_sum_of_squares_gt_sum_squared
  {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 :=
sorry

end four_times_sum_of_squares_gt_sum_squared_l157_157219


namespace find_x_l157_157844

variable {a b x : ℝ}
variable (h₁ : b ≠ 0)
variable (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b)

theorem find_x (h₁ : b ≠ 0) (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a :=
by
  sorry

end find_x_l157_157844


namespace amount_of_water_in_first_tank_l157_157422

theorem amount_of_water_in_first_tank 
  (C : ℝ)
  (H1 : 0 < C)
  (H2 : 0.45 * C = 450)
  (water_in_first_tank : ℝ)
  (water_in_second_tank : ℝ := 450)
  (additional_water_needed : ℝ := 1250)
  (total_capacity : ℝ := 2 * C)
  (total_water_needed : ℝ := 2000) : 
  water_in_first_tank = 300 :=
by 
  sorry

end amount_of_water_in_first_tank_l157_157422


namespace number_of_outfits_l157_157269

theorem number_of_outfits (num_shirts : ℕ) (num_pants : ℕ) (num_shoe_types : ℕ) (shoe_styles_per_type : ℕ) (h_shirts : num_shirts = 4) (h_pants : num_pants = 4) (h_shoes : num_shoe_types = 2) (h_styles : shoe_styles_per_type = 2) :
  num_shirts * num_pants * (num_shoe_types * shoe_styles_per_type) = 64 :=
by {
  sorry
}

end number_of_outfits_l157_157269


namespace largest_y_coordinate_of_degenerate_ellipse_l157_157949

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 := by
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l157_157949


namespace machine_transportation_l157_157734

theorem machine_transportation (x y : ℕ) 
  (h1 : x + 6 - y = 10) 
  (h2 : 400 * x + 800 * (20 - x) + 300 * (6 - y) + 500 * y = 16000) : 
  x = 5 ∧ y = 1 := 
sorry

end machine_transportation_l157_157734


namespace percentage_decrease_revenue_l157_157275

theorem percentage_decrease_revenue (old_revenue new_revenue : Float) (h_old : old_revenue = 69.0) (h_new : new_revenue = 42.0) : 
  (old_revenue - new_revenue) / old_revenue * 100 = 39.13 := by
  rw [h_old, h_new]
  norm_num
  sorry

end percentage_decrease_revenue_l157_157275


namespace monotonically_increasing_interval_l157_157346

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1 / x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / 2 → (∀ y : ℝ, y < x → f y < f x) :=
by
  intro x h
  intro y hy
  sorry

end monotonically_increasing_interval_l157_157346


namespace is_possible_to_finish_7th_l157_157523

theorem is_possible_to_finish_7th 
  (num_teams : ℕ)
  (wins_ASTC : ℕ)
  (losses_ASTC : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ) 
  (total_points : ℕ)
  (rank_ASTC : ℕ)
  (points_ASTC : ℕ)
  (points_needed_by_top_6 : ℕ → ℕ)
  (points_8th_and_9th : ℕ) :
  num_teams = 9 ∧ wins_ASTC = 5 ∧ losses_ASTC = 3 ∧ points_per_win = 3 ∧ points_per_draw = 1 ∧ 
  total_points = 108 ∧ rank_ASTC = 7 ∧ points_ASTC = 15 ∧ points_needed_by_top_6 7 = 105 ∧ points_8th_and_9th ≤ 3 →
  ∃ (top_7_points : ℕ), 
  top_7_points = 105 ∧ (top_7_points + points_8th_and_9th) = total_points := 
sorry

end is_possible_to_finish_7th_l157_157523


namespace radius_intersection_xy_plane_l157_157134

noncomputable def center_sphere : ℝ × ℝ × ℝ := (3, 3, 3)

def radius_xz_circle : ℝ := 2

def xz_center : ℝ × ℝ × ℝ := (3, 0, 3)

def xy_center : ℝ × ℝ × ℝ := (3, 3, 0)

theorem radius_intersection_xy_plane (r : ℝ) (s : ℝ) 
(h_center : center_sphere = (3, 3, 3)) 
(h_xz : xz_center = (3, 0, 3))
(h_r_xz : radius_xz_circle = 2)
(h_xy : xy_center = (3, 3, 0)):
s = 3 := 
sorry

end radius_intersection_xy_plane_l157_157134


namespace solution_interval_l157_157843

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem solution_interval :
  ∃ x_0, f x_0 = 0 ∧ 2 < x_0 ∧ x_0 < 3 :=
by
  sorry

end solution_interval_l157_157843
