import Mathlib

namespace part1_part2_l504_50471

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end part1_part2_l504_50471


namespace primes_div_conditions_unique_l504_50401

theorem primes_div_conditions_unique (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p ∣ q + 6) ∧ (q ∣ p + 7) → (p = 19 ∧ q = 13) :=
sorry

end primes_div_conditions_unique_l504_50401


namespace infinite_sum_eq_3_over_8_l504_50424

theorem infinite_sum_eq_3_over_8 :
  ∑' n : ℕ, (n : ℝ) / (n^4 + 4) = 3 / 8 :=
sorry

end infinite_sum_eq_3_over_8_l504_50424


namespace new_person_weight_l504_50460

-- Define the given conditions as Lean definitions
def weight_increase_per_person : ℝ := 2.5
def num_people : ℕ := 8
def replaced_person_weight : ℝ := 65

-- State the theorem using the given conditions and the correct answer
theorem new_person_weight :
  (weight_increase_per_person * num_people) + replaced_person_weight = 85 :=
sorry

end new_person_weight_l504_50460


namespace functional_equation_solution_l504_50499

theorem functional_equation_solution (f : ℕ → ℕ) 
  (H : ∀ a b : ℕ, f (f a + f b) = a + b) : 
  ∀ n : ℕ, f n = n := 
by
  sorry

end functional_equation_solution_l504_50499


namespace soda_cost_90_cents_l504_50496

theorem soda_cost_90_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s = 360)
  (h2 : 2 * b + 4 * s = 480) :
  s = 90 :=
by
  sorry

end soda_cost_90_cents_l504_50496


namespace ellipse_slope_ratio_l504_50409

theorem ellipse_slope_ratio (a b x1 y1 x2 y2 c k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c = a / 2) (h4 : a = 2) (h5 : c = 1) (h6 : b = Real.sqrt 3) 
  (h7 : 3 * x1 ^ 2 + 4 * y1 ^ 2 = 12 * c ^ 2) 
  (h8 : 3 * x2 ^ 2 + 4 * y2 ^ 2 = 12 * c ^ 2) 
  (h9 : x1 = y1 - c) (h10 : x2 = y2 - c)
  (h11 : y1^2 = 9 / 4)
  (h12 : y1 = -3 / 2 ∨ y1 = 3 / 2) 
  (h13 : k1 = -3 / 2) 
  (h14 : k2 = -1 / 2) :
  k1 / k2 = 3 := 
  sorry

end ellipse_slope_ratio_l504_50409


namespace sector_area_is_2pi_l504_50497

/-- Problem Statement: Prove that the area of a sector of a circle with radius 4 and central
    angle 45° (or π/4 radians) is 2π. -/
theorem sector_area_is_2pi (r : ℝ) (θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_is_2pi_l504_50497


namespace customer_paid_l504_50453

theorem customer_paid (cost_price : ℕ) (markup_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 6672 → markup_percent = 25 → selling_price = cost_price + (markup_percent * cost_price / 100) → selling_price = 8340 :=
by
  intros h_cost_price h_markup_percent h_selling_price
  rw [h_cost_price, h_markup_percent] at h_selling_price
  exact h_selling_price

end customer_paid_l504_50453


namespace positive_solution_sqrt_eq_l504_50431

theorem positive_solution_sqrt_eq (y : ℝ) (hy_pos : 0 < y) : 
    (∃ a, a = y ∧ a^2 = y * a) ∧ (∃ b, b = y ∧ b^2 = y + b) ∧ y = 2 :=
by 
  sorry

end positive_solution_sqrt_eq_l504_50431


namespace tan_alpha_l504_50429

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 1 / 3) (h2 : Real.sin (2 * α) > 0) : 
  Real.tan α = Real.sqrt 2 / 4 :=
by 
  sorry

end tan_alpha_l504_50429


namespace cylinder_surface_area_l504_50455

noncomputable def surface_area_of_cylinder (r l : ℝ) : ℝ :=
  2 * Real.pi * r * (r + l)

theorem cylinder_surface_area (r : ℝ) (h_radius : r = 1) (l : ℝ) (h_length : l = 2 * r) :
  surface_area_of_cylinder r l = 6 * Real.pi := by
  -- Using the given conditions and definition, we need to prove the surface area is 6π
  sorry

end cylinder_surface_area_l504_50455


namespace complement_union_example_l504_50457

open Set

variable (I : Set ℕ) (A : Set ℕ) (B : Set ℕ)

noncomputable def complement (U : Set ℕ) (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem complement_union_example
    (hI : I = {0, 1, 2, 3, 4})
    (hA : A = {0, 1, 2, 3})
    (hB : B = {2, 3, 4}) :
    (complement I A) ∪ (complement I B) = {0, 1, 4} := by
  sorry

end complement_union_example_l504_50457


namespace coefficient_of_expression_l504_50486

theorem coefficient_of_expression :
  ∀ (a b : ℝ), (∃ (c : ℝ), - (2/3) * (a * b) = c * (a * b)) :=
by
  intros a b
  use (-2/3)
  sorry

end coefficient_of_expression_l504_50486


namespace value_of_y_l504_50408

variable (y : ℚ)

def first_boy_marbles : ℚ := 4 * y + 2
def second_boy_marbles : ℚ := 2 * y
def third_boy_marbles : ℚ := y + 3
def total_marbles : ℚ := 31

theorem value_of_y (h : first_boy_marbles y + second_boy_marbles y + third_boy_marbles y = total_marbles) :
  y = 26 / 7 :=
by
  sorry

end value_of_y_l504_50408


namespace num_bicycles_eq_20_l504_50439

-- Definitions based on conditions
def num_cars : ℕ := 10
def num_motorcycles : ℕ := 5
def total_wheels : ℕ := 90
def wheels_per_bicycle : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_motorcycle : ℕ := 2

-- Statement to prove
theorem num_bicycles_eq_20 (B : ℕ) 
  (h_wheels_from_bicycles : wheels_per_bicycle * B = 2 * B)
  (h_wheels_from_cars : num_cars * wheels_per_car = 40)
  (h_wheels_from_motorcycles : num_motorcycles * wheels_per_motorcycle = 10)
  (h_total_wheels : wheels_per_bicycle * B + 40 + 10 = total_wheels) :
  B = 20 :=
sorry

end num_bicycles_eq_20_l504_50439


namespace bananas_to_oranges_cost_l504_50428

noncomputable def cost_equivalence (bananas apples oranges : ℕ) : Prop :=
  (5 * bananas = 3 * apples) ∧
  (8 * apples = 5 * oranges)

theorem bananas_to_oranges_cost (bananas apples oranges : ℕ) 
  (h : cost_equivalence bananas apples oranges) :
  oranges = 9 :=
by sorry

end bananas_to_oranges_cost_l504_50428


namespace desired_annual_profit_is_30500000_l504_50400

noncomputable def annual_fixed_costs : ℝ := 50200000
noncomputable def average_cost_per_car : ℝ := 5000
noncomputable def number_of_cars : ℕ := 20000
noncomputable def selling_price_per_car : ℝ := 9035

noncomputable def total_revenue : ℝ :=
  selling_price_per_car * number_of_cars

noncomputable def total_variable_costs : ℝ :=
  average_cost_per_car * number_of_cars

noncomputable def total_costs : ℝ :=
  annual_fixed_costs + total_variable_costs

noncomputable def desired_annual_profit : ℝ :=
  total_revenue - total_costs

theorem desired_annual_profit_is_30500000:
  desired_annual_profit = 30500000 := by
  sorry

end desired_annual_profit_is_30500000_l504_50400


namespace geom_seq_inverse_sum_l504_50472

theorem geom_seq_inverse_sum 
  (a_2 a_3 a_4 a_5 : ℚ) 
  (h1 : a_2 * a_5 = -3 / 4) 
  (h2 : a_2 + a_3 + a_4 + a_5 = 5 / 4) :
  1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5 = -4 / 3 :=
sorry

end geom_seq_inverse_sum_l504_50472


namespace simple_interest_fraction_l504_50417

theorem simple_interest_fraction (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (F : ℝ)
  (h1 : R = 5)
  (h2 : T = 4)
  (h3 : SI = (P * R * T) / 100)
  (h4 : SI = F * P) :
  F = 1/5 :=
by
  sorry

end simple_interest_fraction_l504_50417


namespace number_of_pies_is_correct_l504_50430

def weight_of_apples : ℕ := 120
def weight_for_applesauce (w : ℕ) : ℕ := w / 2
def weight_for_pies (w wholly_app : ℕ) : ℕ := w - wholly_app
def pies (weight_per_pie total_weight : ℕ) : ℕ := total_weight / weight_per_pie

theorem number_of_pies_is_correct :
  pies 4 (weight_for_pies weight_of_apples (weight_for_applesauce weight_of_apples)) = 15 :=
by
  sorry

end number_of_pies_is_correct_l504_50430


namespace pizzaCostPerSlice_l504_50481

/-- Define the constants and parameters for the problem --/
def largePizzaCost : ℝ := 10.00
def numberOfSlices : ℕ := 8
def firstToppingCost : ℝ := 2.00
def secondThirdToppingCost : ℝ := 1.00
def otherToppingCost : ℝ := 0.50
def toppings : List String := ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"]

/-- Calculate the total number of toppings --/
def numberOfToppings : ℕ := toppings.length

/-- Calculate the total cost of the pizza including all toppings --/
noncomputable def totalPizzaCost : ℝ :=
  largePizzaCost + 
  firstToppingCost + 
  2 * secondThirdToppingCost + 
  (numberOfToppings - 3) * otherToppingCost

/-- Calculate the cost per slice --/
noncomputable def costPerSlice : ℝ := totalPizzaCost / numberOfSlices

/-- Proof statement: The cost per slice is $2.00 --/
theorem pizzaCostPerSlice : costPerSlice = 2 := by
  sorry

end pizzaCostPerSlice_l504_50481


namespace area_of_square_with_given_diagonal_l504_50487

theorem area_of_square_with_given_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : ∃ (A : ℝ), A = 64 :=
by
  use (8 * 8)
  sorry

end area_of_square_with_given_diagonal_l504_50487


namespace toppings_combination_l504_50446

-- Define the combination function
def combination (n k : ℕ) : ℕ := n.choose k

theorem toppings_combination :
  combination 9 3 = 84 := by
  sorry

end toppings_combination_l504_50446


namespace ryan_hours_difference_l504_50454

theorem ryan_hours_difference :
  let hours_english := 6
  let hours_chinese := 7
  hours_chinese - hours_english = 1 := 
by
  -- this is where the proof steps would go
  sorry

end ryan_hours_difference_l504_50454


namespace a_101_mod_49_l504_50491

def a (n : ℕ) : ℕ := 5 ^ n + 9 ^ n

theorem a_101_mod_49 : (a 101) % 49 = 0 :=
by
  -- proof to be filled here
  sorry

end a_101_mod_49_l504_50491


namespace range_of_fx_l504_50479

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_fx (k : ℝ) (x : ℝ) (h1 : k < -1) (h2 : x ∈ Set.Ici (0.5)) :
  Set.Icc (0 : ℝ) 2 = {y | ∃ x, f x k = y ∧ x ∈ Set.Ici 0.5} :=
sorry

end range_of_fx_l504_50479


namespace quadratic_has_real_root_iff_b_in_interval_l504_50463

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l504_50463


namespace total_cats_l504_50406

def initial_siamese_cats : Float := 13.0
def initial_house_cats : Float := 5.0
def added_cats : Float := 10.0

theorem total_cats : initial_siamese_cats + initial_house_cats + added_cats = 28.0 := by
  sorry

end total_cats_l504_50406


namespace count_arithmetic_progressions_22_1000_l504_50443

def num_increasing_arithmetic_progressions (n k max_val : ℕ) : ℕ :=
  -- This is a stub for the arithmetic sequence counting function.
  sorry

theorem count_arithmetic_progressions_22_1000 :
  num_increasing_arithmetic_progressions 22 22 1000 = 23312 :=
sorry

end count_arithmetic_progressions_22_1000_l504_50443


namespace initial_books_gathered_l504_50402

-- Conditions
def total_books_now : Nat := 59
def books_found : Nat := 26

-- Proof problem
theorem initial_books_gathered : total_books_now - books_found = 33 :=
by
  sorry -- Proof to be provided later

end initial_books_gathered_l504_50402


namespace oranges_per_child_l504_50411

theorem oranges_per_child (children oranges : ℕ) (h1 : children = 4) (h2 : oranges = 12) : oranges / children = 3 := by
  sorry

end oranges_per_child_l504_50411


namespace average_speed_of_car_l504_50407

theorem average_speed_of_car 
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (total_time : ℕ)
  (h1 : speed_first_hour = 90)
  (h2 : speed_second_hour = 40)
  (h3 : total_time = 2) : 
  (speed_first_hour + speed_second_hour) / total_time = 65 := 
by
  sorry

end average_speed_of_car_l504_50407


namespace exists_almost_square_divides_2010_l504_50414

noncomputable def almost_square (a b : ℕ) : Prop :=
  (a = b + 1 ∨ b = a + 1) ∧ a * b = 2010

theorem exists_almost_square_divides_2010 :
  ∃ (a b : ℕ), almost_square a b :=
sorry

end exists_almost_square_divides_2010_l504_50414


namespace area_of_square_field_l504_50469

theorem area_of_square_field (s : ℕ) (area : ℕ) (cost_per_meter : ℕ) (total_cost : ℕ) (gate_width : ℕ) :
  (cost_per_meter = 3) →
  (total_cost = 1998) →
  (gate_width = 1) →
  (total_cost = cost_per_meter * (4 * s - 2 * gate_width)) →
  (area = s^2) →
  area = 27889 :=
by
  intros h_cost_per_meter h_total_cost h_gate_width h_cost_eq h_area_eq
  sorry

end area_of_square_field_l504_50469


namespace polynomial_identity_l504_50461

theorem polynomial_identity (x : ℝ) : 
  (2 * x^2 + 5 * x + 8) * (x + 1) - (x + 1) * (x^2 - 2 * x + 50) 
  + (3 * x - 7) * (x + 1) * (x - 2) = 4 * x^3 - 2 * x^2 - 34 * x - 28 := 
by 
  sorry

end polynomial_identity_l504_50461


namespace solve_for_y_l504_50474

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l504_50474


namespace find_c_plus_d_l504_50451

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
if x < 3 then c * x + d else 10 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x, g c d (g c d x) = x) : c + d = 4.5 :=
sorry

end find_c_plus_d_l504_50451


namespace no_such_b_exists_l504_50462

theorem no_such_b_exists (k n : ℕ) (a : ℕ) 
  (hk : Odd k) (hn : Odd n)
  (hk_gt_one : k > 1) (hn_gt_one : n > 1) 
  (hka : k ∣ 2^a + 1) (hna : n ∣ 2^a - 1) : 
  ¬ ∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 :=
sorry

end no_such_b_exists_l504_50462


namespace smallest_nonnegative_a_l504_50412

open Real

theorem smallest_nonnegative_a (a b : ℝ) (h_b : b = π / 4)
(sin_eq : ∀ (x : ℤ), sin (a * x + b) = sin (17 * x)) : 
a = 17 - π / 4 := by 
  sorry

end smallest_nonnegative_a_l504_50412


namespace cos_value_given_sin_condition_l504_50444

open Real

theorem cos_value_given_sin_condition (x : ℝ) (h : sin (x + π / 12) = -1/4) : 
  cos (5 * π / 6 - 2 * x) = -7 / 8 :=
sorry -- Proof steps are omitted.

end cos_value_given_sin_condition_l504_50444


namespace no_three_even_segments_with_odd_intersections_l504_50489

open Set

def is_even_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 0

def is_odd_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 1

theorem no_three_even_segments_with_odd_intersections :
  ¬ ∃ (S1 S2 S3 : Set ℝ),
    (is_even_length S1) ∧
    (is_even_length S2) ∧
    (is_even_length S3) ∧
    (is_odd_length (S1 ∩ S2)) ∧
    (is_odd_length (S1 ∩ S3)) ∧
    (is_odd_length (S2 ∩ S3)) :=
by
  -- Proof here
  sorry

end no_three_even_segments_with_odd_intersections_l504_50489


namespace number_of_bought_bottle_caps_l504_50465

/-- Define the initial number of bottle caps and the final number of bottle caps --/
def initial_bottle_caps : ℕ := 40
def final_bottle_caps : ℕ := 47

/-- Proof that the number of bottle caps Joshua bought is equal to 7 --/
theorem number_of_bought_bottle_caps : final_bottle_caps - initial_bottle_caps = 7 :=
by
  sorry

end number_of_bought_bottle_caps_l504_50465


namespace factor_polynomial_l504_50450

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end factor_polynomial_l504_50450


namespace add_multiply_round_l504_50442

theorem add_multiply_round :
  let a := 73.5891
  let b := 24.376
  let c := (a + b) * 2
  (Float.round (c * 100) / 100) = 195.93 :=
by
  sorry

end add_multiply_round_l504_50442


namespace product_not_divisible_by_770_l504_50468

theorem product_not_divisible_by_770 (a b : ℕ) (h : a + b = 770) : ¬ (a * b) % 770 = 0 :=
sorry

end product_not_divisible_by_770_l504_50468


namespace george_run_speed_last_half_mile_l504_50482

theorem george_run_speed_last_half_mile :
  ∀ (distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance)
    (today_total_time normal_total_time remaining_time : ℝ),
    distance_school = 2 →
    normal_speed = 4 →
    first_segment_distance = 3 / 4 →
    first_segment_speed = 3 →
    second_segment_distance = 3 / 4 →
    second_segment_speed = 4 →
    remaining_distance = 1 / 2 →
    normal_total_time = distance_school / normal_speed →
    today_total_time = (first_segment_distance / first_segment_speed) + (second_segment_distance / second_segment_speed) →
    normal_total_time = today_total_time + remaining_time →
    (remaining_distance / remaining_time) = 8 :=
by
  intros distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance today_total_time normal_total_time remaining_time h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end george_run_speed_last_half_mile_l504_50482


namespace functional_equation_solution_form_l504_50437

noncomputable def functional_equation_problem (f : ℝ → ℝ) :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem functional_equation_solution_form :
  (∀ f : ℝ → ℝ, (functional_equation_problem f) → (∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ 2 + b * x)) :=
by 
  sorry

end functional_equation_solution_form_l504_50437


namespace initial_money_eq_l504_50470

-- Definitions for the problem conditions
def spent_on_sweets : ℝ := 1.25
def spent_on_friends : ℝ := 2 * 1.20
def money_left : ℝ :=  4.85

-- Statement of the problem to prove
theorem initial_money_eq :
  spent_on_sweets + spent_on_friends + money_left = 8.50 := 
sorry

end initial_money_eq_l504_50470


namespace permutation_sum_inequality_l504_50456

noncomputable def permutations (n : ℕ) : List (List ℚ) :=
  List.permutations ((List.range (n+1)).map (fun i => if i = 0 then (1 : ℚ) else (1 : ℚ) / i))

theorem permutation_sum_inequality (n : ℕ) (a b : Fin n → ℚ)
  (ha : ∃ p : List ℚ, p ∈ permutations n ∧ ∀ i, a i = p.get? i) 
  (hb : ∃ q : List ℚ, q ∈ permutations n ∧ ∀ i, b i = q.get? i)
  (h_sum : ∀ i j : Fin n, i ≤ j → a i + b i ≥ a j + b j) 
  (m : Fin n) :
  a m + b m ≤ 4 / (m + 1) :=
sorry

end permutation_sum_inequality_l504_50456


namespace solution_set_fraction_inequality_l504_50473

theorem solution_set_fraction_inequality (x : ℝ) : 
  (x + 1) / (x - 1) ≤ 0 ↔ -1 ≤ x ∧ x < 1 :=
sorry

end solution_set_fraction_inequality_l504_50473


namespace difference_in_distances_l504_50436

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (revolutions : ℕ) : ℝ :=
  circumference r * revolutions

theorem difference_in_distances :
  let r1 := 22.4
  let r2 := 34.2
  let revolutions := 400
  let D1 := distance_covered r1 revolutions
  let D2 := distance_covered r2 revolutions
  D2 - D1 = 29628 :=
by
  sorry

end difference_in_distances_l504_50436


namespace rectangle_perimeter_l504_50410

theorem rectangle_perimeter (s : ℕ) (ABCD_area : 4 * s * s = 400) :
  2 * (2 * s + 2 * s) = 80 :=
by
  -- Skipping the proof
  sorry

end rectangle_perimeter_l504_50410


namespace prob_score_at_most_7_l504_50441

-- Definitions based on the conditions
def prob_10_ring : ℝ := 0.15
def prob_9_ring : ℝ := 0.35
def prob_8_ring : ℝ := 0.2
def prob_7_ring : ℝ := 0.1

-- Define the event of scoring no more than 7
def score_at_most_7 := prob_7_ring

-- Theorem statement
theorem prob_score_at_most_7 : score_at_most_7 = 0.1 := by 
  -- proof goes here
  sorry

end prob_score_at_most_7_l504_50441


namespace simplest_common_denominator_l504_50485

theorem simplest_common_denominator (x y : ℕ) (h1 : 2 * x ≠ 0) (h2 : 4 * y^2 ≠ 0) (h3 : 5 * x * y ≠ 0) :
  ∃ d : ℕ, d = 20 * x * y^2 :=
by {
  sorry
}

end simplest_common_denominator_l504_50485


namespace simplify_fraction_l504_50422

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) : 
  (2 * a / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4)) := 
by 
  sorry 

end simplify_fraction_l504_50422


namespace fraction_simplifies_to_two_l504_50433

theorem fraction_simplifies_to_two :
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 2 := by
  sorry

end fraction_simplifies_to_two_l504_50433


namespace sea_creatures_lost_l504_50415

theorem sea_creatures_lost (sea_stars : ℕ) (seashells : ℕ) (snails : ℕ) (items_left : ℕ)
  (h1 : sea_stars = 34) (h2 : seashells = 21) (h3 : snails = 29) (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l504_50415


namespace min_num_of_teams_l504_50488

theorem min_num_of_teams (num_athletes : ℕ) (max_team_size : ℕ) (h1 : num_athletes = 30) (h2 : max_team_size = 9) :
  ∃ (min_teams : ℕ), min_teams = 5 ∧ (∀ nal : ℕ, (nal > 0 ∧ num_athletes % nal = 0 ∧ nal ≤ max_team_size) → num_athletes / nal ≥ min_teams) :=
by
  sorry

end min_num_of_teams_l504_50488


namespace walking_time_12_hours_l504_50405

theorem walking_time_12_hours :
  ∀ t : ℝ, 
  (∀ (v1 v2 : ℝ), 
  v1 = 7 ∧ v2 = 3 →
  120 = (v1 + v2) * t) →
  t = 12 := 
by
  intros t h
  specialize h 7 3 ⟨rfl, rfl⟩
  sorry

end walking_time_12_hours_l504_50405


namespace remainder_x_101_div_x2_plus1_x_plus1_l504_50452

theorem remainder_x_101_div_x2_plus1_x_plus1 : 
  (x^101) % ((x^2 + 1) * (x + 1)) = x :=
by
  sorry

end remainder_x_101_div_x2_plus1_x_plus1_l504_50452


namespace playground_area_l504_50426

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end playground_area_l504_50426


namespace highway_length_proof_l504_50478

variable (L : ℝ) (v1 v2 : ℝ) (t : ℝ)

def highway_length : Prop :=
  v1 = 55 ∧ v2 = 35 ∧ t = 1 / 15 ∧ (L / v2 - L / v1 = t) ∧ L = 6.42

theorem highway_length_proof : highway_length L 55 35 (1 / 15) := by
  sorry

end highway_length_proof_l504_50478


namespace num_valid_n_l504_50420

theorem num_valid_n (n q r : ℤ) (h₁ : 10000 ≤ n) (h₂ : n ≤ 99999)
  (h₃ : n = 50 * q + r) (h₄ : 200 ≤ q) (h₅ : q ≤ 1999)
  (h₆ : 0 ≤ r) (h₇ : r < 50) :
  (∃ (count : ℤ), count = 14400) := by
  sorry

end num_valid_n_l504_50420


namespace find_set_C_l504_50419

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def C : Set ℝ := {a | B a ⊆ A}

theorem find_set_C : C = {0, 1, 2} :=
by
  sorry

end find_set_C_l504_50419


namespace least_possible_value_of_a_plus_b_l504_50404

theorem least_possible_value_of_a_plus_b : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  Nat.gcd (a + b) 330 = 1 ∧
  b ∣ a^a ∧ 
  ∀ k : ℕ, b^3 ∣ a^a → (k ∣ a → k = 1) ∧
  a + b = 392 :=
by
  sorry

end least_possible_value_of_a_plus_b_l504_50404


namespace rhombus_area_l504_50464

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 30) (h2 : d2 = 16) : (d1 * d2) / 2 = 240 := by
  sorry

end rhombus_area_l504_50464


namespace calc_expression_l504_50480

theorem calc_expression : 
  (abs (Real.sqrt 2 - Real.sqrt 3) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3) :=
by
  -- Given that sqrt(3) > sqrt(2)
  have h1 : Real.sqrt 3 > Real.sqrt 2 := by sorry
  -- And cos(45°) = sqrt(2)/2
  have h2 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  -- Now prove the expression equivalency
  sorry

end calc_expression_l504_50480


namespace john_has_25_roommates_l504_50475

def roommates_of_bob := 10
def roommates_of_john := 2 * roommates_of_bob + 5

theorem john_has_25_roommates : roommates_of_john = 25 := 
by
  sorry

end john_has_25_roommates_l504_50475


namespace binomial_theorem_fifth_term_l504_50493
-- Import the necessary library

-- Define the theorem as per the given conditions and required proof
theorem binomial_theorem_fifth_term
  (a x : ℝ) 
  (hx : x ≠ 0) 
  (ha : a ≠ 0) : 
  (Nat.choose 8 4 * (a / x)^4 * (x / a^3)^4 = 70 / a^8) :=
by
  -- Applying the binomial theorem and simplifying the expression
  rw [Nat.choose]
  sorry

end binomial_theorem_fifth_term_l504_50493


namespace inequality_proof_l504_50458

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (x^2 / (y - 1) + y^2 / (x - 1) ≥ 8) :=
  sorry

end inequality_proof_l504_50458


namespace trains_same_distance_at_meeting_l504_50427

theorem trains_same_distance_at_meeting
  (d v : ℝ) (h_d : 0 < d) (h_v : 0 < v) :
  ∃ t : ℝ, v * t + v * (t - 1) = d ∧ 
  v * t = (d + v) / 2 ∧ 
  d - (v * (t - 1)) = (d + v) / 2 :=
by
  sorry

end trains_same_distance_at_meeting_l504_50427


namespace license_plate_count_l504_50477

theorem license_plate_count :
  let digits := 10
  let letters := 26
  let positions := 6
  positions * digits^5 * letters^3 = 105456000 := by
  sorry

end license_plate_count_l504_50477


namespace total_weight_of_fish_l504_50440

theorem total_weight_of_fish (fry : ℕ) (survival_rate : ℚ) 
  (first_catch : ℕ) (first_avg_weight : ℚ) 
  (second_catch : ℕ) (second_avg_weight : ℚ)
  (third_catch : ℕ) (third_avg_weight : ℚ)
  (total_weight : ℚ) :
  fry = 100000 ∧ 
  survival_rate = 0.95 ∧ 
  first_catch = 40 ∧ 
  first_avg_weight = 2.5 ∧ 
  second_catch = 25 ∧ 
  second_avg_weight = 2.2 ∧ 
  third_catch = 35 ∧ 
  third_avg_weight = 2.8 ∧ 
  total_weight = fry * survival_rate * 
    ((first_catch * first_avg_weight + 
      second_catch * second_avg_weight + 
      third_catch * third_avg_weight) / 100) / 10000 →
  total_weight = 24 :=
by
  sorry

end total_weight_of_fish_l504_50440


namespace polynomial_expansion_correct_l504_50492

def polynomial1 (z : ℤ) : ℤ := 3 * z^3 + 4 * z^2 - 5
def polynomial2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2
def expandedPolynomial (z : ℤ) : ℤ := 12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10

theorem polynomial_expansion_correct (z : ℤ) :
  (polynomial1 z) * (polynomial2 z) = expandedPolynomial z :=
by sorry

end polynomial_expansion_correct_l504_50492


namespace chair_capacity_l504_50483

theorem chair_capacity
  (total_chairs : ℕ)
  (total_board_members : ℕ)
  (not_occupied_fraction : ℚ)
  (occupied_people_per_chair : ℕ)
  (attending_board_members : ℕ)
  (total_chairs_eq : total_chairs = 40)
  (not_occupied_fraction_eq : not_occupied_fraction = 2/5)
  (occupied_people_per_chair_eq : occupied_people_per_chair = 2)
  (attending_board_members_eq : attending_board_members = 48)
  : total_board_members = 48 := 
by
  sorry

end chair_capacity_l504_50483


namespace correct_sum_l504_50432

theorem correct_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 4) (h3 : x * y = 98) : x + y = 18 := 
by
  sorry

end correct_sum_l504_50432


namespace sin_pi_over_4_plus_alpha_l504_50416

open Real

theorem sin_pi_over_4_plus_alpha
  (α : ℝ)
  (hα : 0 < α ∧ α < π)
  (h_tan : tan (α - π / 4) = 1 / 3) :
  sin (π / 4 + α) = 3 * sqrt 10 / 10 :=
sorry

end sin_pi_over_4_plus_alpha_l504_50416


namespace total_ticket_cost_l504_50403

theorem total_ticket_cost 
  (young_discount : ℝ := 0.55) 
  (old_discount : ℝ := 0.30) 
  (full_price : ℝ := 10)
  (num_young : ℕ := 2) 
  (num_middle : ℕ := 2) 
  (num_old : ℕ := 2) 
  (grandma_ticket_cost : ℝ := 7) :
  2 * (full_price * young_discount) + 2 * full_price + 2 * grandma_ticket_cost = 43 :=
by 
  sorry

end total_ticket_cost_l504_50403


namespace math_proof_problem_l504_50494

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2 * Real.sqrt 3

noncomputable def conditions (e : ℝ) (vertex : ℝ × ℝ) (p q : ℝ × ℝ) : Prop :=
  e = 1 / 2
  ∧ vertex = (0, 2 * Real.sqrt 3)  -- focus of the parabola
  ∧ p = (-2, -3)
  ∧ q = (-2, 3)

noncomputable def max_area_quadrilateral (area : ℝ) : Prop :=
  area = 12 * Real.sqrt 3

theorem math_proof_problem : 
  ∃ a b p q area, ellipse_standard_eq a b ∧ conditions (1/2) (0, 2 * Real.sqrt 3) p q 
  ∧ p = (-2, -3) ∧ q = (-2, 3) → max_area_quadrilateral area := 
  sorry

end math_proof_problem_l504_50494


namespace find_TU_square_l504_50425

-- Definitions
variables (P Q R S T U : ℝ × ℝ)
variable (side : ℝ)
variable (QT RU PT SU PQ : ℝ)

-- Setting the conditions
variables (side_eq_10 : side = 10)
variables (QT_eq_7 : QT = 7)
variables (RU_eq_7 : RU = 7)
variables (PT_eq_24 : PT = 24)
variables (SU_eq_24 : SU = 24)
variables (PQ_eq_10 : PQ = 10)

-- The theorem statement
theorem find_TU_square : TU^2 = 1150 :=
by
  -- Proof to be done here.
  sorry

end find_TU_square_l504_50425


namespace part_I_part_II_l504_50490

def f (x a : ℝ) := |2 * x - a| + 5 * x

theorem part_I (x : ℝ) : f x 3 ≥ 5 * x + 1 ↔ (x ≤ 1 ∨ x ≥ 2) := sorry

theorem part_II (a x : ℝ) (h : (∀ x, f x a ≤ 0 ↔ x ≤ -1)) : a = 3 := sorry

end part_I_part_II_l504_50490


namespace trigonometric_identity_l504_50447

noncomputable def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem trigonometric_identity :
  special_operation (Real.sin (Real.pi / 12)) (Real.cos (Real.pi / 12))
  = - (1 + 2 * Real.sqrt 3) / 4 :=
by
  sorry

end trigonometric_identity_l504_50447


namespace max_min_values_on_circle_l504_50466

def on_circle (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 - 4 * x - 4 * y + 7 = 0

theorem max_min_values_on_circle (x y : ℝ) (h : on_circle x y) :
  16 ≤ (x + 1) ^ 2 + (y + 2) ^ 2 ∧ (x + 1) ^ 2 + (y + 2) ^ 2 ≤ 36 :=
  sorry

end max_min_values_on_circle_l504_50466


namespace sums_of_coordinates_of_A_l504_50498

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l504_50498


namespace bag_cost_is_10_l504_50438

def timothy_initial_money : ℝ := 50
def tshirt_cost : ℝ := 8
def keychain_cost : ℝ := 2
def keychains_per_set : ℝ := 3
def number_of_tshirts : ℝ := 2
def number_of_bags : ℝ := 2
def number_of_keychains : ℝ := 21

noncomputable def cost_of_each_bag : ℝ :=
  let cost_of_tshirts := number_of_tshirts * tshirt_cost
  let remaining_money_after_tshirts := timothy_initial_money - cost_of_tshirts
  let cost_of_keychains := (number_of_keychains / keychains_per_set) * keychain_cost
  let remaining_money_after_keychains := remaining_money_after_tshirts - cost_of_keychains
  remaining_money_after_keychains / number_of_bags

theorem bag_cost_is_10 :
  cost_of_each_bag = 10 := by
  sorry

end bag_cost_is_10_l504_50438


namespace arithmetic_sequence_sum_l504_50445

theorem arithmetic_sequence_sum :
  let first_term := 1
  let common_diff := 2
  let last_term := 33
  let n := (last_term + 1) / common_diff
  (n * (first_term + last_term)) / 2 = 289 :=
by
  sorry

end arithmetic_sequence_sum_l504_50445


namespace factorize1_factorize2_l504_50449

-- Part 1: Prove the factorization of xy - 1 - x + y
theorem factorize1 (x y : ℝ) : (x * y - 1 - x + y) = (y - 1) * (x + 1) :=
  sorry

-- Part 2: Prove the factorization of (a^2 + b^2)^2 - 4a^2b^2
theorem factorize2 (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
  sorry

end factorize1_factorize2_l504_50449


namespace correct_statement_l504_50421

theorem correct_statement (a b : ℚ) :
  (|a| = b → a = b) ∧ (|a| > |b| → a > b) ∧ (|a| > b → |a| > |b|) ∧ (|a| = b → a^2 = (-b)^2) ↔ 
  (true ∧ false ∧ false ∧ true) :=
by
  sorry

end correct_statement_l504_50421


namespace line_through_point_l504_50495

theorem line_through_point (k : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, (x = 3) ∧ (y = -2) → (2 - 3 * k * x = -4 * y)) → k = -2/3 :=
by
  sorry

end line_through_point_l504_50495


namespace expand_expression_l504_50484

variable (x : ℝ)

theorem expand_expression : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 :=
by
  sorry

end expand_expression_l504_50484


namespace quadratic_increasing_for_x_geq_3_l504_50435

theorem quadratic_increasing_for_x_geq_3 (x : ℝ) : 
  x ≥ 3 → y = 2 * (x - 3)^2 - 1 → ∃ d > 0, ∀ p ≥ x, y ≤ 2 * (p - 3)^2 - 1 := sorry

end quadratic_increasing_for_x_geq_3_l504_50435


namespace maximum_value_of_k_minus_b_l504_50467

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b

theorem maximum_value_of_k_minus_b (b : ℝ) (k : ℝ) (x : ℝ) 
  (h₀ : 0 ≤ b ∧ b ≤ 2) 
  (h₁ : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h₂ : ∀ x ∈ Set.Icc 1 (Real.exp 1), f x 1 b ≥ (k * x - x * Real.log x - 1)) :
  k - b ≤ 0 :=
sorry

end maximum_value_of_k_minus_b_l504_50467


namespace speed_of_stream_l504_50476

-- Definitions of the problem's conditions
def downstream_distance := 72
def upstream_distance := 30
def downstream_time := 3
def upstream_time := 3

-- The unknowns
variables (b s : ℝ)

-- The effective speed equations based on the problem conditions
def effective_speed_downstream := b + s
def effective_speed_upstream := b - s

-- The core conditions of the problem
def condition1 : Prop := downstream_distance = effective_speed_downstream * downstream_time
def condition2 : Prop := upstream_distance = effective_speed_upstream * upstream_time

-- The problem statement transformed into a Lean theorem
theorem speed_of_stream (h1 : condition1) (h2 : condition2) : s = 7 := 
sorry

end speed_of_stream_l504_50476


namespace minimize_y_l504_50418

theorem minimize_y (a b : ℝ) : 
  ∃ x, x = (a + b) / 2 ∧ ∀ x', ((x' - a)^3 + (x' - b)^3) ≥ ((x - a)^3 + (x - b)^3) :=
sorry

end minimize_y_l504_50418


namespace hyperbola_asymptotes_l504_50413

def hyperbola (x y : ℝ) : Prop := (x^2 / 8) - (y^2 / 2) = 1

theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola x y → (y = (1/2) * x ∨ y = - (1/2) * x) :=
by
  sorry

end hyperbola_asymptotes_l504_50413


namespace intersecting_lines_fixed_point_l504_50459

variable (p a b : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : b ≠ 0)
variable (h3 : b^2 ≠ 2 * p * a)

def parabola (M : ℝ × ℝ) : Prop := M.2^2 = 2 * p * M.1

def fixed_points (A B : ℝ × ℝ) : Prop :=
  A = (a, b) ∧ B = (-a, 0)

def intersect_parabola (M1 M2 M : ℝ × ℝ) : Prop :=
  parabola p M ∧ parabola p M1 ∧ parabola p M2 ∧ M ≠ M1 ∧ M ≠ M2

theorem intersecting_lines_fixed_point (M M1 M2 : ℝ × ℝ)
  (hP : parabola p M) 
  (hA : (a, b) ≠ M) 
  (hB : (-a, 0) ≠ M) 
  (h_intersect : intersect_parabola p M1 M2 M) :
  ∃ C : ℝ × ℝ, C = (a, 2 * p * a / b) :=
sorry

end intersecting_lines_fixed_point_l504_50459


namespace vector_perpendicular_to_plane_l504_50434

theorem vector_perpendicular_to_plane
  (a b c d : ℝ)
  (x1 y1 z1 x2 y2 z2 : ℝ)
  (h1 : a * x1 + b * y1 + c * z1 + d = 0)
  (h2 : a * x2 + b * y2 + c * z2 + d = 0) :
  a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2) = 0 :=
sorry

end vector_perpendicular_to_plane_l504_50434


namespace total_points_correct_l504_50423

-- Define the number of teams
def num_teams : ℕ := 16

-- Define the number of draws
def num_draws : ℕ := 30

-- Define the scoring system
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def loss_deduction_threshold : ℕ := 3
def points_deduction_per_threshold : ℕ := 1

-- Define the total number of games
def total_games : ℕ := num_teams * (num_teams - 1) / 2

-- Define the number of wins (non-draw games)
def num_wins : ℕ := total_games - num_draws

-- Define the total points from wins
def total_points_from_wins : ℕ := num_wins * points_for_win

-- Define the total points from draws
def total_points_from_draws : ℕ := num_draws * points_for_draw * 2

-- Define the total points (as no team lost more than twice, no deductions apply)
def total_points : ℕ := total_points_from_wins + total_points_from_draws

theorem total_points_correct :
  total_points = 330 := by
  sorry

end total_points_correct_l504_50423


namespace general_term_formula_l504_50448

variable (a S : ℕ → ℚ)

-- Condition 1: The sum of the first n terms of the sequence {a_n} is S_n
def sum_first_n_terms (n : ℕ) : ℚ := S n

-- Condition 2: a_n = 3S_n - 2
def a_n (n : ℕ) : Prop := a n = 3 * S n - 2

theorem general_term_formula (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ k, k ≥ 2 → a (k) = - (1/2) * a (k - 1) ) : 
  a n = (-1/2)^(n-1) :=
sorry

end general_term_formula_l504_50448
