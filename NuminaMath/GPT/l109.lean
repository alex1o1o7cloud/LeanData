import Mathlib

namespace value_of_n_l109_10981

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l109_10981


namespace james_earnings_l109_10939

-- Define the conditions
def rain_gallons_per_inch : ℕ := 15
def rain_monday : ℕ := 4
def rain_tuesday : ℕ := 3
def price_per_gallon : ℝ := 1.2

-- State the theorem to be proved
theorem james_earnings : (rain_monday * rain_gallons_per_inch + rain_tuesday * rain_gallons_per_inch) * price_per_gallon = 126 :=
by
  sorry

end james_earnings_l109_10939


namespace max_capacity_per_car_l109_10953

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

end max_capacity_per_car_l109_10953


namespace Sierra_Crest_Trail_Length_l109_10954

theorem Sierra_Crest_Trail_Length (a b c d e : ℕ) 
(h1 : a + b + c = 36) 
(h2 : b + d = 30) 
(h3 : d + e = 38) 
(h4 : a + d = 32) : 
a + b + c + d + e = 74 := by
  sorry

end Sierra_Crest_Trail_Length_l109_10954


namespace intersection_A_B_l109_10932

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_A_B : A ∩ B = {70} := by
  sorry

end intersection_A_B_l109_10932


namespace books_into_bags_l109_10900

def books := Finset.range 5
def bags := Finset.range 4

noncomputable def arrangement_count : ℕ :=
  -- definition of arrangement_count can be derived from the solution logic
  sorry

theorem books_into_bags : arrangement_count = 51 := 
  sorry

end books_into_bags_l109_10900


namespace walmart_pot_stacking_l109_10963

theorem walmart_pot_stacking :
  ∀ (total_pots pots_per_set shelves : ℕ),
    total_pots = 60 →
    pots_per_set = 5 →
    shelves = 4 →
    (total_pots / pots_per_set / shelves) = 3 :=
by 
  intros total_pots pots_per_set shelves h1 h2 h3
  sorry

end walmart_pot_stacking_l109_10963


namespace tan_alpha_neg_four_over_three_l109_10949

theorem tan_alpha_neg_four_over_three (α : ℝ) (h_cos : Real.cos α = -3/5) (h_alpha_range : α ∈ Set.Ioo (-π) 0) : Real.tan α = -4/3 :=
  sorry

end tan_alpha_neg_four_over_three_l109_10949


namespace locus_of_center_of_circle_l109_10982

theorem locus_of_center_of_circle (x y a : ℝ)
  (hC : x^2 + y^2 - (2 * a^2 - 4) * x - 4 * a^2 * y + 5 * a^4 - 4 = 0) :
  2 * x - y + 4 = 0 ∧ -2 ≤ x ∧ x < 0 :=
sorry

end locus_of_center_of_circle_l109_10982


namespace tangent_properties_l109_10999

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom func_eq : ∀ x, f (x - 2) = f (-x)
axiom tangent_eq_at_1 : ∀ x, (x = 1 → f x = 2 * x + 1)

-- Prove the required results
theorem tangent_properties :
  (deriv f 1 = 2) ∧ (∃ B C, (∀ x, (x = -3) → f x = B -2 * (x + 3)) ∧ (B = 3) ∧ (C = -3)) :=
by
  sorry

end tangent_properties_l109_10999


namespace gear_revolutions_l109_10959

variable (r_p : ℝ) 

theorem gear_revolutions (h1 : 40 * (1 / 6) = r_p * (1 / 6) + 5) : r_p = 10 := 
by
  sorry

end gear_revolutions_l109_10959


namespace base_conversion_is_248_l109_10928

theorem base_conversion_is_248 (a b c n : ℕ) 
  (h1 : n = 49 * a + 7 * b + c) 
  (h2 : n = 81 * c + 9 * b + a) 
  (h3 : 0 ≤ a ∧ a ≤ 6) 
  (h4 : 0 ≤ b ∧ b ≤ 6) 
  (h5 : 0 ≤ c ∧ c ≤ 6)
  (h6 : 0 ≤ a ∧ a ≤ 8) 
  (h7 : 0 ≤ b ∧ b ≤ 8) 
  (h8 : 0 ≤ c ∧ c ≤ 8) 
  : n = 248 :=
by 
  sorry

end base_conversion_is_248_l109_10928


namespace smallest_base_for_100_l109_10995

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end smallest_base_for_100_l109_10995


namespace solution_inequality_l109_10935

-- Define the condition as a predicate
def inequality_condition (x : ℝ) : Prop :=
  (x - 1) * (x + 1) < 0

-- State the theorem that we need to prove
theorem solution_inequality : ∀ x : ℝ, inequality_condition x → (-1 < x ∧ x < 1) :=
by
  intro x hx
  sorry

end solution_inequality_l109_10935


namespace placing_pencils_l109_10972

theorem placing_pencils (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
    (h1 : total_pencils = 6) (h2 : num_rows = 2) : pencils_per_row = 3 :=
by
  sorry

end placing_pencils_l109_10972


namespace parts_in_batch_l109_10980

theorem parts_in_batch (a : ℕ) (h₁ : 20 * (a / 20) + 13 = a) (h₂ : 27 * (a / 27) + 20 = a) 
  (h₃ : 500 ≤ a) (h₄ : a ≤ 600) : a = 533 :=
by sorry

end parts_in_batch_l109_10980


namespace intersection_sets_l109_10944

theorem intersection_sets :
  let A := { x : ℝ | x^2 - 1 ≥ 0 }
  let B := { x : ℝ | 1 ≤ x ∧ x < 3 }
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_sets_l109_10944


namespace final_position_correct_total_distance_correct_l109_10994

def movements : List Int := [15, -25, 20, -35]

-- Final Position: 
def final_position (moves : List Int) : Int := moves.sum

-- Total Distance Traveled calculated by taking the absolutes and summing:
def total_distance (moves : List Int) : Nat :=
  moves.map (λ x => Int.natAbs x) |>.sum

theorem final_position_correct : final_position movements = -25 :=
by
  sorry

theorem total_distance_correct : total_distance movements = 95 :=
by
  sorry

end final_position_correct_total_distance_correct_l109_10994


namespace focus_of_parabola_eq_l109_10950

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := -5 * x^2 + 10 * x - 2

-- Statement of the theorem to find the focus of the given parabola
theorem focus_of_parabola_eq (x : ℝ) : 
  let vertex_x := 1
  let vertex_y := 3
  let a := -5
  ∃ focus_x focus_y, 
    focus_x = vertex_x ∧ 
    focus_y = vertex_y - (1 / (4 * a)) ∧
    focus_x = 1 ∧
    focus_y = 59 / 20 := 
  sorry

end focus_of_parabola_eq_l109_10950


namespace exists_two_digit_pair_product_l109_10979

theorem exists_two_digit_pair_product (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (hprod : a * b = 8670) : a * b = 8670 :=
by
  exact hprod

end exists_two_digit_pair_product_l109_10979


namespace value_of_expression_l109_10901

theorem value_of_expression :
  4 * 5 + 5 * 4 = 40 :=
sorry

end value_of_expression_l109_10901


namespace cost_of_first_ring_is_10000_l109_10989

theorem cost_of_first_ring_is_10000 (x : ℝ) (h₁ : x + 2*x - x/2 = 25000) : x = 10000 :=
sorry

end cost_of_first_ring_is_10000_l109_10989


namespace angle_D_calculation_l109_10902

theorem angle_D_calculation (A B E C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50)
  (h4 : E = 60)
  (h5 : A + B + E = 180)
  (h6 : B + C + D = 180) :
  D = 55 :=
by
  sorry

end angle_D_calculation_l109_10902


namespace peaches_left_at_stand_l109_10926

def initial_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def spoiled_peaches : ℝ := 12.0
def sold_peaches : ℝ := 27.0

theorem peaches_left_at_stand :
  initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 81.0 :=
by
  -- initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 84.0
  sorry

end peaches_left_at_stand_l109_10926


namespace pages_per_day_read_l109_10936

theorem pages_per_day_read (start_date : ℕ) (end_date : ℕ) (total_pages : ℕ) (fraction_covered : ℚ) (pages_read : ℕ) (days : ℕ) :
  start_date = 1 →
  end_date = 12 →
  total_pages = 144 →
  fraction_covered = 2/3 →
  pages_read = fraction_covered * total_pages →
  days = end_date - start_date + 1 →
  pages_read / days = 8 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pages_per_day_read_l109_10936


namespace exists_x_f_lt_g_l109_10915

noncomputable def f (x : ℝ) := (2 / Real.exp 1) ^ x

noncomputable def g (x : ℝ) := (Real.exp 1 / 3) ^ x

theorem exists_x_f_lt_g : ∃ x : ℝ, f x < g x := by
  sorry

end exists_x_f_lt_g_l109_10915


namespace Zhang_Hai_average_daily_delivery_is_37_l109_10908

theorem Zhang_Hai_average_daily_delivery_is_37
  (d1_packages : ℕ) (d1_count : ℕ)
  (d2_packages : ℕ) (d2_count : ℕ)
  (d3_packages : ℕ) (d3_count : ℕ)
  (total_days : ℕ) 
  (h1 : d1_packages = 41) (h2 : d1_count = 1)
  (h3 : d2_packages = 35) (h4 : d2_count = 2)
  (h5 : d3_packages = 37) (h6 : d3_count = 4)
  (h7 : total_days = 7) :
  (d1_count * d1_packages + d2_count * d2_packages + d3_count * d3_packages) / total_days = 37 := 
by sorry

end Zhang_Hai_average_daily_delivery_is_37_l109_10908


namespace infinite_non_congruent_integers_l109_10943

theorem infinite_non_congruent_integers (a : ℕ → ℤ) (m : ℕ → ℤ) (k : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → 2 ≤ m i)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < k → 2 * m i ≤ m (i + 1)) :
  ∃ (x : ℕ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ¬ (x % (m i) = a i % (m i)) :=
sorry

end infinite_non_congruent_integers_l109_10943


namespace sara_payment_equivalence_l109_10969

variable (cost_book1 cost_book2 change final_amount : ℝ)

theorem sara_payment_equivalence
  (h1 : cost_book1 = 5.5)
  (h2 : cost_book2 = 6.5)
  (h3 : change = 8)
  (h4 : final_amount = cost_book1 + cost_book2 + change) :
  final_amount = 20 := by
  sorry

end sara_payment_equivalence_l109_10969


namespace polynomial_factorization_l109_10905

theorem polynomial_factorization (m n : ℤ) (h₁ : (x^2 + m * x + 6 : ℤ) = (x - 2) * (x + n)) : m = -5 := by
  sorry

end polynomial_factorization_l109_10905


namespace find_a5_l109_10990

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

theorem find_a5 (a : ℕ → ℤ) (d : ℤ)
  (h_seq : arithmetic_sequence a d)
  (h1 : a 1 + a 5 = 8)
  (h4 : a 4 = 7) : 
  a 5 = 10 := sorry

end find_a5_l109_10990


namespace distance_preserving_l109_10964

variables {Point : Type} {d : Point → Point → ℕ} {f : Point → Point}

axiom distance_one (A B : Point) : d A B = 1 → d (f A) (f B) = 1

theorem distance_preserving :
  ∀ (A B : Point) (n : ℕ), n > 0 → d A B = n → d (f A) (f B) = n :=
by
  sorry

end distance_preserving_l109_10964


namespace part_a_part_b_part_c_l109_10934

theorem part_a (p q : ℝ) : q < p^2 → ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = 2 * p) ∧ (r1 * r2 = q) :=
by
  sorry

theorem part_b (p q : ℝ) : q = 4 * p - 4 → (2^2 - 2 * p * 2 + q = 0) :=
by
  sorry

theorem part_c (p q : ℝ) : q = p^2 ∧ q = 4 * p - 4 → (p = 2 ∧ q = 4) :=
by
  sorry

end part_a_part_b_part_c_l109_10934


namespace prob_sum_divisible_by_4_is_1_4_l109_10913

/-- 
  Given two wheels each with numbers from 1 to 8, 
  the probability that the sum of two selected numbers from the wheels is divisible by 4.
-/
noncomputable def prob_sum_divisible_by_4 : ℚ :=
  let outcomes : ℕ := 8 * 8
  let favorable_outcomes : ℕ := 16
  favorable_outcomes / outcomes

theorem prob_sum_divisible_by_4_is_1_4 : prob_sum_divisible_by_4 = 1 / 4 := 
  by
    -- Statement is left as sorry as the proof steps are not required.
    sorry

end prob_sum_divisible_by_4_is_1_4_l109_10913


namespace inequality_example_l109_10938

theorem inequality_example (a b c : ℝ) : a^2 + 4 * b^2 + 9 * c^2 ≥ 2 * a * b + 3 * a * c + 6 * b * c :=
by
  sorry

end inequality_example_l109_10938


namespace multiples_of_eleven_ending_in_seven_l109_10907

theorem multiples_of_eleven_ending_in_seven (n : ℕ) : 
  (∀ k : ℕ, n > 0 ∧ n < 2000 ∧ (∃ m : ℕ, n = 11 * m) ∧ n % 10 = 7) → ∃ c : ℕ, c = 18 := 
by
  sorry

end multiples_of_eleven_ending_in_seven_l109_10907


namespace unique_double_digit_in_range_l109_10922

theorem unique_double_digit_in_range (a b : ℕ) (h₁ : a = 10) (h₂ : b = 40) : 
  ∃! n : ℕ, (10 ≤ n ∧ n ≤ 40) ∧ (n % 10 = n / 10) ∧ (n % 10 = 3) :=
by {
  sorry
}

end unique_double_digit_in_range_l109_10922


namespace find_five_dollar_bills_l109_10948

-- Define the number of bills
def total_bills (x y : ℕ) : Prop := x + y = 126

-- Define the total value of the bills
def total_value (x y : ℕ) : Prop := 5 * x + 10 * y = 840

-- Now we state the theorem
theorem find_five_dollar_bills (x y : ℕ) (h1 : total_bills x y) (h2 : total_value x y) : x = 84 :=
by sorry

end find_five_dollar_bills_l109_10948


namespace die_vanishing_probability_and_floor_value_l109_10937

/-
Given conditions:
1. The die has four faces labeled 0, 1, 2, 3.
2. When the die lands on a face labeled:
   - 0: the die vanishes.
   - 1: nothing happens (one die remains).
   - 2: the die replicates into 2 dice.
   - 3: the die replicates into 3 dice.
3. All dice (original and replicas) will continuously be rolled.
Prove:
  The value of ⌊10/p⌋ is 24, where p is the probability that all dice will eventually disappear.
-/

theorem die_vanishing_probability_and_floor_value : 
  ∃ (p : ℝ), 
  (p^3 + p^2 - 3 * p + 1 = 0 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p = Real.sqrt 2 - 1) 
  ∧ ⌊10 / p⌋ = 24 := 
    sorry

end die_vanishing_probability_and_floor_value_l109_10937


namespace product_wavelengths_eq_n_cbrt_mn2_l109_10970

variable (m n : ℝ)

noncomputable def common_ratio (m n : ℝ) := (n / m)^(1/3)

noncomputable def wavelength_jiazhong (m n : ℝ) := (m^2 * n)^(1/3)
noncomputable def wavelength_nanlu (m n : ℝ) := (n^4 / m)^(1/3)

theorem product_wavelengths_eq_n_cbrt_mn2
  (h : n = m * (common_ratio m n)^3) :
  (wavelength_jiazhong m n) * (wavelength_nanlu m n) = n * (m * n^2)^(1/3) :=
by
  sorry

end product_wavelengths_eq_n_cbrt_mn2_l109_10970


namespace identify_fraction_l109_10956

variable {a b : ℚ}

def is_fraction (x : ℚ) (y : ℚ) := ∃ (n : ℚ), x = n / y

theorem identify_fraction :
  is_fraction 2 a ∧ ¬ is_fraction (2 * a) 3 ∧ ¬ is_fraction (-b) 2 ∧ ¬ is_fraction (3 * a + 1) 2 :=
by
  sorry

end identify_fraction_l109_10956


namespace logs_quadratic_sum_l109_10933

theorem logs_quadratic_sum (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_roots : ∀ x, 2 * x^2 + 4 * x + 1 = 0 → (x = Real.log a) ∨ (x = Real.log b)) :
  (Real.log a)^2 + Real.log (a^2) + a * b = 1 / Real.exp 2 - 1 / 2 :=
by
  sorry

end logs_quadratic_sum_l109_10933


namespace work_rate_l109_10984

theorem work_rate (x : ℝ) (h : (1 / x + 1 / 15 = 1 / 6)) : x = 10 :=
sorry

end work_rate_l109_10984


namespace find_ab_l109_10968

theorem find_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end find_ab_l109_10968


namespace weighted_average_correct_l109_10931

-- Define the marks
def english_marks : ℝ := 76
def mathematics_marks : ℝ := 65
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 85

-- Define the weightages
def english_weightage : ℝ := 0.20
def mathematics_weightage : ℝ := 0.25
def physics_weightage : ℝ := 0.25
def chemistry_weightage : ℝ := 0.15
def biology_weightage : ℝ := 0.15

-- Define the weighted sum calculation
def weighted_sum : ℝ :=
  english_marks * english_weightage + 
  mathematics_marks * mathematics_weightage + 
  physics_marks * physics_weightage + 
  chemistry_marks * chemistry_weightage + 
  biology_marks * biology_weightage

-- Define the theorem statement: the weighted average marks
theorem weighted_average_correct : weighted_sum = 74.75 :=
by
  sorry

end weighted_average_correct_l109_10931


namespace intersection_A_B_l109_10967

def A := { x : ℝ | -5 < x ∧ x < 2 }
def B := { x : ℝ | x^2 - 9 < 0 }
def AB := { x : ℝ | -3 < x ∧ x < 2 }

theorem intersection_A_B : A ∩ B = AB := by
  sorry

end intersection_A_B_l109_10967


namespace descent_property_l109_10976

def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem descent_property (x : ℝ) (h : x < 3) : (quadratic_function (x + 1) < quadratic_function x) :=
sorry

end descent_property_l109_10976


namespace Sarah_skateboard_speed_2160_mph_l109_10919

-- Definitions based on the conditions
def miles_to_inches (miles : ℕ) : ℕ := miles * 63360
def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

/-- Pete walks backwards 3 times faster than Susan walks forwards --/
def Susan_walks_forwards_speed (pete_walks_hands_speed : ℕ) : ℕ := pete_walks_hands_speed / 3

/-- Tracy does cartwheels twice as fast as Susan walks forwards --/
def Tracy_cartwheels_speed (susan_walks_forwards_speed : ℕ) : ℕ := susan_walks_forwards_speed * 2

/-- Mike swims 8 times faster than Tracy does cartwheels --/
def Mike_swims_speed (tracy_cartwheels_speed : ℕ) : ℕ := tracy_cartwheels_speed * 8

/-- Pete can walk on his hands at 1/4 the speed Tracy can do cartwheels --/
def Pete_walks_hands_speed : ℕ := 2

/-- Pete rides his bike 5 times faster than Mike swims --/
def Pete_rides_bike_speed (mike_swims_speed : ℕ) : ℕ := mike_swims_speed * 5

/-- Patty can row 3 times faster than Pete walks backwards (in feet per hour) --/
def Patty_rows_speed (pete_walks_backwards_speed : ℕ) : ℕ := pete_walks_backwards_speed * 3

/-- Sarah can skateboard 6 times faster than Patty rows (in miles per minute) --/
def Sarah_skateboards_speed (patty_rows_speed_ft_per_hr : ℕ) : ℕ := (patty_rows_speed_ft_per_hr * 6 * 60) * 63360 * 60

theorem Sarah_skateboard_speed_2160_mph : Sarah_skateboards_speed (Patty_rows_speed (Pete_walks_hands_speed * 3)) = 2160 * 63360 * 60 :=
by
  sorry

end Sarah_skateboard_speed_2160_mph_l109_10919


namespace arithmetic_mean_of_fractions_l109_10960

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5) + (4 / 7)) = 17 / 35 :=
by
  sorry

end arithmetic_mean_of_fractions_l109_10960


namespace sum_of_x_y_l109_10987

theorem sum_of_x_y (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 48) : x + y = 2 :=
sorry

end sum_of_x_y_l109_10987


namespace paid_amount_divisible_by_11_l109_10930

-- Define the original bill amount and the increased bill amount
def original_bill (x : ℕ) : ℕ := x
def paid_amount (x : ℕ) : ℕ := (11 * x) / 10

-- Theorem: The paid amount is divisible by 11
theorem paid_amount_divisible_by_11 (x : ℕ) (h : x % 10 = 0) : paid_amount x % 11 = 0 :=
by
  sorry

end paid_amount_divisible_by_11_l109_10930


namespace hyperbola_equation_l109_10923

open Real

theorem hyperbola_equation (e e' : ℝ) (h₁ : 2 * x^2 + y^2 = 2) (h₂ : e * e' = 1) :
  y^2 - x^2 = 2 :=
sorry

end hyperbola_equation_l109_10923


namespace find_divisor_l109_10911

theorem find_divisor (n : ℕ) (d : ℕ) (h1 : n = 105829) (h2 : d = 10) (h3 : ∃ k, n - d = k * d) : d = 3 :=
by
  sorry

end find_divisor_l109_10911


namespace particle_probability_at_2_3_after_5_moves_l109_10903

theorem particle_probability_at_2_3_after_5_moves:
  ∃ (C : ℕ), C = Nat.choose 5 2 ∧
  (1/2 ^ 5 * C) = (Nat.choose 5 2) * ((1/2: ℝ) ^ 5) := by
sorry

end particle_probability_at_2_3_after_5_moves_l109_10903


namespace min_val_expression_l109_10920

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end min_val_expression_l109_10920


namespace domain_of_f_l109_10942

noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / Real.sqrt (3*x - 9)

theorem domain_of_f : ∀ x : ℝ, (3 < x) ↔ (∃ y : ℝ, f y ≠ y) :=
by
  sorry

end domain_of_f_l109_10942


namespace correct_calculation_l109_10971

theorem correct_calculation :
  -4^2 / (-2)^3 * (-1 / 8) = -1 / 4 := by
  sorry

end correct_calculation_l109_10971


namespace middle_admitted_is_correct_l109_10977

-- Define the total number of admitted people.
def total_admitted := 100

-- Define the proportions of South, North, and Middle volumes.
def south_ratio := 11
def north_ratio := 7
def middle_ratio := 2

-- Calculating the total ratio.
def total_ratio := south_ratio + north_ratio + middle_ratio

-- Hypothesis that we are dealing with the correct ratio and total.
def middle_admitted (total_admitted : ℕ) (total_ratio : ℕ) (middle_ratio : ℕ) : ℕ :=
  total_admitted * middle_ratio / total_ratio

-- Proof statement
theorem middle_admitted_is_correct :
  middle_admitted total_admitted total_ratio middle_ratio = 10 :=
by
  -- This line would usually contain the detailed proof steps, which are omitted here.
  sorry

end middle_admitted_is_correct_l109_10977


namespace cost_price_of_watch_l109_10916

theorem cost_price_of_watch (C SP1 SP2 : ℝ)
    (h1 : SP1 = 0.90 * C)
    (h2 : SP2 = 1.02 * C)
    (h3 : SP2 = SP1 + 140) :
    C = 1166.67 :=
by
  sorry

end cost_price_of_watch_l109_10916


namespace not_detecting_spy_probability_l109_10951

-- Definitions based on conditions
def forest_size : ℝ := 10
def detection_radius : ℝ := 10

-- Inoperative detector - assuming NE corner
def detector_NE_inoperative : Prop := true

-- Probability calculation result
def probability_not_detected : ℝ := 0.087

-- Theorem to prove
theorem not_detecting_spy_probability :
  (forest_size = 10) ∧ (detection_radius = 10) ∧ detector_NE_inoperative →
  probability_not_detected = 0.087 :=
by
  sorry

end not_detecting_spy_probability_l109_10951


namespace average_pushups_is_correct_l109_10996

theorem average_pushups_is_correct :
  ∀ (David Zachary Emily : ℕ),
    David = 510 →
    Zachary = David - 210 →
    Emily = David - 132 →
    (David + Zachary + Emily) / 3 = 396 :=
by
  intro David Zachary Emily hDavid hZachary hEmily
  -- All calculations and proofs will go here, but we'll leave them as sorry for now.
  sorry

end average_pushups_is_correct_l109_10996


namespace number_of_beavers_in_second_group_l109_10962

-- Define the number of beavers and the time for the first group
def numBeavers1 := 20
def time1 := 3

-- Define the time for the second group
def time2 := 5

-- Define the total work done (which is constant)
def work := numBeavers1 * time1

-- Define the number of beavers in the second group
def numBeavers2 := 12

-- Theorem stating the mathematical equivalence
theorem number_of_beavers_in_second_group : numBeavers2 * time2 = work :=
by
  -- remaining proof steps would go here
  sorry

end number_of_beavers_in_second_group_l109_10962


namespace surface_area_ratio_l109_10909

-- Defining conditions
variable (V_E V_J : ℝ) (A_E A_J : ℝ)
variable (volume_ratio : V_J = 30 * (Real.sqrt 30) * V_E)

-- Statement to prove
theorem surface_area_ratio (h : V_J = 30 * (Real.sqrt 30) * V_E) :
  A_J = 30 * A_E :=
by
  sorry

end surface_area_ratio_l109_10909


namespace greatest_possible_value_of_q_minus_r_l109_10906

noncomputable def max_difference (q r : ℕ) : ℕ :=
  if q < r then r - q else q - r

theorem greatest_possible_value_of_q_minus_r (q r : ℕ) (x y : ℕ) (hq : q = 10 * x + y) (hr : r = 10 * y + x) (cond : q ≠ r) (hqr : max_difference q r < 20) : q - r = 18 :=
  sorry

end greatest_possible_value_of_q_minus_r_l109_10906


namespace binom_coeff_div_prime_l109_10947

open Nat

theorem binom_coeff_div_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p :=
by
  sorry

end binom_coeff_div_prime_l109_10947


namespace problem_statement_l109_10986

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end problem_statement_l109_10986


namespace simplified_expression_evaluate_at_zero_l109_10917

noncomputable def simplify_expr (x : ℝ) : ℝ :=
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2 * x + 1))

theorem simplified_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  simplify_expr x = 1 / (x - 1) :=
by sorry

theorem evaluate_at_zero (h₁ : (0 : ℝ) ≠ -1) (h₂ : (0 : ℝ) ≠ 1) : 
  simplify_expr 0 = -1 :=
by sorry

end simplified_expression_evaluate_at_zero_l109_10917


namespace chloe_total_books_l109_10993

noncomputable def total_books (average_books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (science_fiction_shelves : ℕ) (history_shelves : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + science_fiction_shelves + history_shelves) * average_books_per_shelf

theorem chloe_total_books : 
  total_books 85 7 5 3 2 = 14500 / 100 :=
  by
  sorry

end chloe_total_books_l109_10993


namespace range_of_a_l109_10914

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), (1 ≤ x ∧ x ≤ 2) ∧ (2 ≤ y ∧ y ≤ 3) → (x * y ≤ a * x^2 + 2 * y^2)) →
  a ≥ -1 :=
by {
  sorry
}

end range_of_a_l109_10914


namespace deepak_present_age_l109_10958

theorem deepak_present_age (x : ℕ) (h : 4 * x + 6 = 26) : 3 * x = 15 := 
by 
  sorry

end deepak_present_age_l109_10958


namespace exists_integers_abcd_l109_10985

theorem exists_integers_abcd (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end exists_integers_abcd_l109_10985


namespace negation_of_exists_l109_10955

theorem negation_of_exists (x : ℕ) : (¬ ∃ x : ℕ, x^2 ≤ x) := 
by 
  sorry

end negation_of_exists_l109_10955


namespace current_rate_l109_10924

variable (c : ℝ)

def still_water_speed : ℝ := 3.6

axiom rowing_time_ratio (c : ℝ) : (2 : ℝ) * (still_water_speed - c) = still_water_speed + c

theorem current_rate : c = 1.2 :=
by
  sorry

end current_rate_l109_10924


namespace intersection_S_T_l109_10966

open Set

def S : Set ℝ := { x | x ≥ 1 }
def T : Set ℝ := { -2, -1, 0, 1, 2 }

theorem intersection_S_T : S ∩ T = { 1, 2 } := by
  sorry

end intersection_S_T_l109_10966


namespace shadow_projection_height_l109_10992

theorem shadow_projection_height :
  ∃ (x : ℝ), (∃ (shadow_area : ℝ), shadow_area = 192) ∧ 1000 * x = 25780 :=
by
  sorry

end shadow_projection_height_l109_10992


namespace diminished_value_l109_10965

theorem diminished_value (x y : ℝ) (h1 : x = 160)
  (h2 : x / 5 + 4 = x / 4 - y) : y = 4 :=
by
  sorry

end diminished_value_l109_10965


namespace pipe_fill_without_hole_l109_10983

theorem pipe_fill_without_hole :
  ∀ (T : ℝ), 
  (1 / T - 1 / 60 = 1 / 20) → 
  T = 15 := 
by
  intros T h
  sorry

end pipe_fill_without_hole_l109_10983


namespace cos_triple_angle_l109_10988

theorem cos_triple_angle
  (θ : ℝ)
  (h : Real.cos θ = 1/3) :
  Real.cos (3 * θ) = -23 / 27 :=
by
  sorry

end cos_triple_angle_l109_10988


namespace focus_of_parabola_l109_10927

noncomputable def parabola_focus (a h k : ℝ) : ℝ × ℝ :=
  (h, k + 1 / (4 * a))

theorem focus_of_parabola :
  parabola_focus 9 (-1/3) (-3) = (-1/3, -107/36) := 
  sorry

end focus_of_parabola_l109_10927


namespace complement_of_set_M_l109_10945

open Set

def universal_set : Set ℝ := univ

def set_M : Set ℝ := {x | x^2 < 2 * x}

def complement_M : Set ℝ := compl set_M

theorem complement_of_set_M :
  complement_M = {x | x ≤ 0 ∨ x ≥ 2} :=
sorry

end complement_of_set_M_l109_10945


namespace blue_pill_cost_l109_10925

theorem blue_pill_cost
  (days : Int := 10)
  (total_expenditure : Int := 430)
  (daily_cost : Int := total_expenditure / days) :
  ∃ (y : Int), y + (y - 3) = daily_cost ∧ y = 23 := by
  sorry

end blue_pill_cost_l109_10925


namespace rajan_income_l109_10918

theorem rajan_income : 
  ∀ (x y : ℕ), 
  7 * x - 6 * y = 1000 → 
  6 * x - 5 * y = 1000 → 
  7 * x = 7000 := 
by 
  intros x y h1 h2
  sorry

end rajan_income_l109_10918


namespace geometric_sequence_sum_l109_10997

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (S_n : ℝ) (S_3n : ℝ) (S_4n : ℝ)
    (h1 : S_n = 2) 
    (h2 : S_3n = 14) 
    (h3 : ∀ m : ℕ, S_m = a_n 1 * (1 - q^m) / (1 - q)) :
    S_4n = 30 :=
by
  sorry

end geometric_sequence_sum_l109_10997


namespace each_group_has_145_bananas_l109_10946

theorem each_group_has_145_bananas (total_bananas : ℕ) (groups_bananas : ℕ) : 
  total_bananas = 290 ∧ groups_bananas = 2 → total_bananas / groups_bananas = 145 := 
by 
  sorry

end each_group_has_145_bananas_l109_10946


namespace quadratic_difference_l109_10929

theorem quadratic_difference (f : ℝ → ℝ) (hpoly : ∃ c d e : ℤ, ∀ x, f x = c*x^2 + d*x + e) 
(h : f (Real.sqrt 3) - f (Real.sqrt 2) = 4) : 
f (Real.sqrt 10) - f (Real.sqrt 7) = 12 := sorry

end quadratic_difference_l109_10929


namespace lara_puts_flowers_in_vase_l109_10973

theorem lara_puts_flowers_in_vase : 
  ∀ (total_flowers mom_flowers flowers_given_more : ℕ), 
    total_flowers = 52 →
    mom_flowers = 15 →
    flowers_given_more = 6 →
  (total_flowers - (mom_flowers + (mom_flowers + flowers_given_more))) = 16 :=
by
  intros total_flowers mom_flowers flowers_given_more h1 h2 h3
  sorry

end lara_puts_flowers_in_vase_l109_10973


namespace jack_emails_morning_l109_10961

-- Definitions from conditions
def emails_evening : ℕ := 7
def additional_emails_morning : ℕ := 2
def emails_morning : ℕ := emails_evening + additional_emails_morning

-- The proof problem
theorem jack_emails_morning : emails_morning = 9 := by
  -- proof goes here
  sorry

end jack_emails_morning_l109_10961


namespace least_number_subtracted_l109_10998

/-- The least number that must be subtracted from 50248 so that the 
remaining number is divisible by both 20 and 37 is 668. -/
theorem least_number_subtracted (n : ℕ) (x : ℕ ) (y : ℕ ) (a : ℕ) (b : ℕ) :
  n = 50248 → x = 20 → y = 37 → (a = 20 * 37) →
  (50248 - b) % a = 0 → 50248 - b < a → b = 668 :=
by
  sorry

end least_number_subtracted_l109_10998


namespace no_purchase_count_l109_10912

def total_people : ℕ := 15
def people_bought_tvs : ℕ := 9
def people_bought_computers : ℕ := 7
def people_bought_both : ℕ := 3

theorem no_purchase_count : total_people - (people_bought_tvs - people_bought_both) - (people_bought_computers - people_bought_both) - people_bought_both = 2 := by
  sorry

end no_purchase_count_l109_10912


namespace similar_triangle_shortest_side_l109_10941

theorem similar_triangle_shortest_side (a b c : ℕ) (p : ℕ) (h : a = 8 ∧ b = 10 ∧ c = 12 ∧ p = 150) :
  ∃ x : ℕ, (x = p / (a + b + c) ∧ 8 * x = 40) :=
by
  sorry

end similar_triangle_shortest_side_l109_10941


namespace recurring_decimal_36_exceeds_decimal_35_l109_10904

-- Definition of recurring decimal 0.36...
def recurring_decimal_36 : ℚ := 36 / 99

-- Definition of 0.35 as fraction
def decimal_35 : ℚ := 7 / 20

-- Statement of the math proof problem
theorem recurring_decimal_36_exceeds_decimal_35 :
  recurring_decimal_36 - decimal_35 = 3 / 220 := by
  sorry

end recurring_decimal_36_exceeds_decimal_35_l109_10904


namespace jack_bill_age_difference_l109_10978

def jack_bill_ages_and_difference (a b : ℕ) :=
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  (a + b = 2) ∧ (7 * a - 29 * b = 14) → jack_age - bill_age = 18

theorem jack_bill_age_difference (a b : ℕ) (h₀ : a + b = 2) (h₁ : 7 * a - 29 * b = 14) : 
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  jack_age - bill_age = 18 :=
by {
  sorry
}

end jack_bill_age_difference_l109_10978


namespace total_bins_sum_l109_10974

def total_bins_soup : ℝ := 0.2
def total_bins_vegetables : ℝ := 0.35
def total_bins_fruits : ℝ := 0.15
def total_bins_pasta : ℝ := 0.55
def total_bins_canned_meats : ℝ := 0.275
def total_bins_beans : ℝ := 0.175

theorem total_bins_sum :
  total_bins_soup + total_bins_vegetables + total_bins_fruits + total_bins_pasta + total_bins_canned_meats + total_bins_beans = 1.7 :=
by
  sorry

end total_bins_sum_l109_10974


namespace max_area_of_garden_l109_10975

theorem max_area_of_garden
  (w : ℕ) (l : ℕ)
  (h1 : l = 2 * w)
  (h2 : l + 2 * w = 480) : l * w = 28800 :=
sorry

end max_area_of_garden_l109_10975


namespace intersection_of_A_and_B_l109_10991

-- Definitions for the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l109_10991


namespace dunkers_lineup_count_l109_10957

theorem dunkers_lineup_count (players : Finset ℕ) (h_players : players.card = 15) (alice zen : ℕ) 
  (h_alice : alice ∈ players) (h_zen : zen ∈ players) (h_distinct : alice ≠ zen) :
  (∃ (S : Finset (Finset ℕ)), S.card = 2717 ∧ ∀ s ∈ S, s.card = 5 ∧ ¬ (alice ∈ s ∧ zen ∈ s)) :=
by
  sorry

end dunkers_lineup_count_l109_10957


namespace cylinder_increase_l109_10952

theorem cylinder_increase (x : ℝ) (r h : ℝ) (π : ℝ) 
  (h₁ : r = 5) (h₂ : h = 10) 
  (h₃ : π > 0) 
  (h_equal_volumes : π * (r + x) ^ 2 * h = π * r ^ 2 * (h + x)) :
  x = 5 / 2 :=
by
  -- Proof is omitted
  sorry

end cylinder_increase_l109_10952


namespace six_rational_right_triangles_same_perimeter_l109_10940

theorem six_rational_right_triangles_same_perimeter :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ a₄ b₄ c₄ a₅ b₅ c₅ a₆ b₆ c₆ : ℕ),
    a₁^2 + b₁^2 = c₁^2 ∧ a₂^2 + b₂^2 = c₂^2 ∧ a₃^2 + b₃^2 = c₃^2 ∧
    a₄^2 + b₄^2 = c₄^2 ∧ a₅^2 + b₅^2 = c₅^2 ∧ a₆^2 + b₆^2 = c₆^2 ∧
    a₁ + b₁ + c₁ = 720 ∧ a₂ + b₂ + c₂ = 720 ∧ a₃ + b₃ + c₃ = 720 ∧
    a₄ + b₄ + c₄ = 720 ∧ a₅ + b₅ + c₅ = 720 ∧ a₆ + b₆ + c₆ = 720 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧ (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₄, b₄, c₄) ∧ (a₁, b₁, c₁) ≠ (a₅, b₅, c₅) ∧
    (a₁, b₁, c₁) ≠ (a₆, b₆, c₆) ∧ (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₂, b₂, c₂) ≠ (a₄, b₄, c₄) ∧ (a₂, b₂, c₂) ≠ (a₅, b₅, c₅) ∧
    (a₂, b₂, c₂) ≠ (a₆, b₆, c₆) ∧ (a₃, b₃, c₃) ≠ (a₄, b₄, c₄) ∧
    (a₃, b₃, c₃) ≠ (a₅, b₅, c₅) ∧ (a₃, b₃, c₃) ≠ (a₆, b₆, c₆) ∧
    (a₄, b₄, c₄) ≠ (a₅, b₅, c₅) ∧ (a₄, b₄, c₄) ≠ (a₆, b₆, c₆) ∧
    (a₅, b₅, c₅) ≠ (a₆, b₆, c₆) :=
sorry

end six_rational_right_triangles_same_perimeter_l109_10940


namespace remainder_when_dividing_P_by_DDD_l109_10921

variables (P D D' D'' Q Q' Q'' R R' R'' : ℕ)

-- Define the conditions
def condition1 : Prop := P = Q * D + R
def condition2 : Prop := Q = Q' * D' + R'
def condition3 : Prop := Q' = Q'' * D'' + R''

-- Theorem statement asserting the given conclusion
theorem remainder_when_dividing_P_by_DDD' 
  (H1 : condition1 P D Q R)
  (H2 : condition2 Q D' Q' R')
  (H3 : condition3 Q' D'' Q'' R'') : 
  P % (D * D' * D') = R'' * D * D' + R * D' + R := 
sorry

end remainder_when_dividing_P_by_DDD_l109_10921


namespace cornbread_pieces_count_l109_10910

-- Define the dimensions of the pan and the pieces of cornbread
def pan_length := 24
def pan_width := 20
def piece_length := 3
def piece_width := 2
def margin := 1

-- Define the effective width after considering the margin
def effective_width := pan_width - margin

-- Prove the number of pieces of cornbread is 72
theorem cornbread_pieces_count :
  (pan_length / piece_length) * (effective_width / piece_width) = 72 :=
by
  sorry

end cornbread_pieces_count_l109_10910
