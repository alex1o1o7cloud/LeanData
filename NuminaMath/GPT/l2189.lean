import Mathlib

namespace NUMINAMATH_GPT_exists_n_consecutive_composites_l2189_218963

theorem exists_n_consecutive_composites (n : ℕ) (h : n ≥ 1) (a r : ℕ) :
  ∃ K : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (a + (K + i) * r)) := 
sorry

end NUMINAMATH_GPT_exists_n_consecutive_composites_l2189_218963


namespace NUMINAMATH_GPT_find_aa_l2189_218955

-- Given conditions
def m : ℕ := 7

-- Definition for checking if a number's tens place is 1
def tens_place_one (n : ℕ) : Prop :=
  (n / 10) % 10 = 1

-- The main statement to prove
theorem find_aa : ∃ x : ℕ, x < 10 ∧ tens_place_one (m * x^3) ∧ x = 6 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_aa_l2189_218955


namespace NUMINAMATH_GPT_fruit_cost_l2189_218917

theorem fruit_cost:
  let strawberry_cost := 2.20
  let cherry_cost := 6 * strawberry_cost
  let blueberry_cost := cherry_cost / 2
  let strawberries_count := 3
  let cherries_count := 4.5
  let blueberries_count := 6.2
  let total_cost := (strawberries_count * strawberry_cost) + (cherries_count * cherry_cost) + (blueberries_count * blueberry_cost)
  total_cost = 106.92 :=
by
  sorry

end NUMINAMATH_GPT_fruit_cost_l2189_218917


namespace NUMINAMATH_GPT_eq_margin_l2189_218939

variables (C S n : ℝ) (M : ℝ)

theorem eq_margin (h : M = 1 / n * (2 * C - S)) : M = S / (n + 2) :=
sorry

end NUMINAMATH_GPT_eq_margin_l2189_218939


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2189_218945

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) : a > 0 ∧ b < 0 :=
by 
  have hb : b < 0 := sorry
  exact ⟨h1, hb⟩

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2189_218945


namespace NUMINAMATH_GPT_problem_x_l2189_218975

theorem problem_x (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 :=
sorry

end NUMINAMATH_GPT_problem_x_l2189_218975


namespace NUMINAMATH_GPT_common_area_of_triangles_is_25_l2189_218966

-- Define basic properties and conditions of an isosceles right triangle with hypotenuse = 10 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = 10^2
def is_isosceles_right_triangle (a b : ℝ) : Prop := a = b ∧ hypotenuse a b

-- Definitions representing the triangls
noncomputable def triangle1 := ∃ a b : ℝ, is_isosceles_right_triangle a b
noncomputable def triangle2 := ∃ a b : ℝ, is_isosceles_right_triangle a b

-- The area common to both triangles is the focus
theorem common_area_of_triangles_is_25 : 
  triangle1 ∧ triangle2 → 
  ∃ area : ℝ, area = 25 
  := 
sorry

end NUMINAMATH_GPT_common_area_of_triangles_is_25_l2189_218966


namespace NUMINAMATH_GPT_complement_intersection_eq_l2189_218944

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 2, 5}) (hB : B = {1, 3, 4})

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_intersection_eq_l2189_218944


namespace NUMINAMATH_GPT_greatest_integer_l2189_218957

-- Define the conditions for the problem
def isMultiple4 (n : ℕ) : Prop := n % 4 = 0
def notMultiple8 (n : ℕ) : Prop := n % 8 ≠ 0
def notMultiple12 (n : ℕ) : Prop := n % 12 ≠ 0
def gcf4 (n : ℕ) : Prop := Nat.gcd n 24 = 4
def lessThan200 (n : ℕ) : Prop := n < 200

-- State the main theorem
theorem greatest_integer : ∃ n : ℕ, lessThan200 n ∧ gcf4 n ∧ n = 196 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_l2189_218957


namespace NUMINAMATH_GPT_james_tv_watching_time_l2189_218932

theorem james_tv_watching_time
  (ep_jeopardy : ℕ := 20) -- Each episode of Jeopardy is 20 minutes long
  (n_jeopardy : ℕ := 2) -- James watched 2 episodes of Jeopardy
  (n_wheel : ℕ := 2) -- James watched 2 episodes of Wheel of Fortune
  (wheel_factor : ℕ := 2) -- Wheel of Fortune episodes are twice as long as Jeopardy episodes
  : (ep_jeopardy * n_jeopardy + ep_jeopardy * wheel_factor * n_wheel) / 60 = 2 :=
by
  sorry

end NUMINAMATH_GPT_james_tv_watching_time_l2189_218932


namespace NUMINAMATH_GPT_number_of_connections_l2189_218911

theorem number_of_connections (n : ℕ) (d : ℕ) (h₀ : n = 40) (h₁ : d = 4) : 
  (n * d) / 2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_connections_l2189_218911


namespace NUMINAMATH_GPT_ring_toss_total_amount_l2189_218974

-- Defining the amounts made in the two periods
def amount_first_period : Nat := 382
def amount_second_period : Nat := 374

-- The total amount made
def total_amount : Nat := amount_first_period + amount_second_period

-- Statement that the total amount calculated is equal to the given answer
theorem ring_toss_total_amount :
  total_amount = 756 := by
  sorry

end NUMINAMATH_GPT_ring_toss_total_amount_l2189_218974


namespace NUMINAMATH_GPT_incorrect_transformation_D_l2189_218986

theorem incorrect_transformation_D (x y m : ℝ) (hxy: x = y) : m = 0 → ¬ (x / m = y / m) :=
by
  intro hm
  simp [hm]
  -- Lean's simp tactic simplifies known equalities
  -- The simp tactic will handle the contradiction case directly when m = 0.
  sorry

end NUMINAMATH_GPT_incorrect_transformation_D_l2189_218986


namespace NUMINAMATH_GPT_wire_goes_around_field_l2189_218959

theorem wire_goes_around_field :
  (7348 / (4 * Real.sqrt 27889)) = 11 :=
by
  sorry

end NUMINAMATH_GPT_wire_goes_around_field_l2189_218959


namespace NUMINAMATH_GPT_men_left_bus_l2189_218940

theorem men_left_bus (M W : ℕ) (initial_passengers : M + W = 72) 
  (women_half_men : W = M / 2) 
  (equal_men_women_after_changes : ∃ men_left : ℕ, ∀ W_new, W_new = W + 8 → M - men_left = W_new → M - men_left = 32) :
  ∃ men_left : ℕ, men_left = 16 :=
  sorry

end NUMINAMATH_GPT_men_left_bus_l2189_218940


namespace NUMINAMATH_GPT_unique_root_conditions_l2189_218971

theorem unique_root_conditions (m : ℝ) (x y : ℝ) :
  (x^2 = 2 * abs x ∧ abs x - y - m = 1 - y^2) ↔ m = 3 / 4 := sorry

end NUMINAMATH_GPT_unique_root_conditions_l2189_218971


namespace NUMINAMATH_GPT_ratio_children_to_adults_l2189_218942

variable (male_adults : ℕ) (female_adults : ℕ) (total_people : ℕ)
variable (total_adults : ℕ) (children : ℕ)

theorem ratio_children_to_adults :
  male_adults = 100 →
  female_adults = male_adults + 50 →
  total_people = 750 →
  total_adults = male_adults + female_adults →
  children = total_people - total_adults →
  children / total_adults = 2 :=
by
  intros h_male h_female h_total h_adults h_children
  sorry

end NUMINAMATH_GPT_ratio_children_to_adults_l2189_218942


namespace NUMINAMATH_GPT_trainer_voice_radius_l2189_218976

noncomputable def area_of_heard_voice (r : ℝ) : ℝ := (1/4) * Real.pi * r^2

theorem trainer_voice_radius :
  ∃ r : ℝ, abs (r - 140) < 1 ∧ area_of_heard_voice r = 15393.804002589986 :=
by
  sorry

end NUMINAMATH_GPT_trainer_voice_radius_l2189_218976


namespace NUMINAMATH_GPT_find_integer_n_l2189_218923

theorem find_integer_n (n : ℤ) (h : (⌊(n^2 : ℤ)/4⌋ - (⌊n/2⌋)^2 = 2)) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_integer_n_l2189_218923


namespace NUMINAMATH_GPT_fixed_point_l2189_218983

noncomputable def fixed_point_function (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : (ℝ × ℝ) :=
  (1, a^(1 - (1 : ℝ)) + 5)

theorem fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : fixed_point_function a h₀ h₁ = (1, 6) :=
by 
  sorry

end NUMINAMATH_GPT_fixed_point_l2189_218983


namespace NUMINAMATH_GPT_sqrt_of_S_l2189_218903

def initial_time := 16 * 3600 + 11 * 60 + 22
def initial_date := 16
def total_seconds_in_a_day := 86400
def total_seconds_in_an_hour := 3600

theorem sqrt_of_S (S : ℕ) (hS : S = total_seconds_in_a_day + total_seconds_in_an_hour) : 
  Real.sqrt S = 300 := 
sorry

end NUMINAMATH_GPT_sqrt_of_S_l2189_218903


namespace NUMINAMATH_GPT_sugar_water_inequality_one_sugar_water_inequality_two_l2189_218909

variable (a b m : ℝ)

-- Condition constraints
variable (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m)

-- Sugar Water Experiment One Inequality
theorem sugar_water_inequality_one : a / b > a / (b + m) := 
by
  sorry

-- Sugar Water Experiment Two Inequality
theorem sugar_water_inequality_two : a / b < (a + m) / b := 
by
  sorry

end NUMINAMATH_GPT_sugar_water_inequality_one_sugar_water_inequality_two_l2189_218909


namespace NUMINAMATH_GPT_power_eq_l2189_218948

theorem power_eq (a b c : ℝ) (h₁ : a = 81) (h₂ : b = 4 / 3) : (a ^ b) = 243 * (3 ^ (1 / 3)) := by
  sorry

end NUMINAMATH_GPT_power_eq_l2189_218948


namespace NUMINAMATH_GPT_jill_travels_less_than_john_l2189_218949

theorem jill_travels_less_than_john :
  ∀ (John Jill Jim : ℕ), 
  John = 15 → 
  Jim = 2 → 
  (Jim = (20 / 100) * Jill) → 
  (John - Jill) = 5 := 
by
  intros John Jill Jim HJohn HJim HJimJill
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_jill_travels_less_than_john_l2189_218949


namespace NUMINAMATH_GPT_carina_coffee_l2189_218929

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end NUMINAMATH_GPT_carina_coffee_l2189_218929


namespace NUMINAMATH_GPT_right_triangle_area_l2189_218937

/-- Given a right triangle where one leg is 18 cm and the hypotenuse is 30 cm,
    prove that the area of the triangle is 216 square centimeters. -/
theorem right_triangle_area (a b c : ℝ) 
    (ha : a = 18) 
    (hc : c = 30) 
    (h_right : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 216 :=
by
  -- Substitute the values given and solve the area.
  sorry

end NUMINAMATH_GPT_right_triangle_area_l2189_218937


namespace NUMINAMATH_GPT_quadratic_no_real_roots_iff_m_gt_one_l2189_218919

theorem quadratic_no_real_roots_iff_m_gt_one (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 :=
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_iff_m_gt_one_l2189_218919


namespace NUMINAMATH_GPT_sum_of_squares_l2189_218970

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2189_218970


namespace NUMINAMATH_GPT_initial_sodium_chloride_percentage_l2189_218969

theorem initial_sodium_chloride_percentage :
  ∀ (P : ℝ),
  (∃ (C : ℝ), C = 24) → -- Tank capacity
  (∃ (E_rate : ℝ), E_rate = 0.4) → -- Evaporation rate per hour
  (∃ (time : ℝ), time = 6) → -- Time in hours
  (1 / 4 * C = 6) → -- Volume of mixture
  (6 * P / 100 + (6 - 6 * P / 100 - E_rate * time) = 3.6) → -- Concentration condition
  P = 30 :=
by
  intros P hC hE_rate htime hvolume hconcentration
  rcases hC with ⟨C, hC⟩
  rcases hE_rate with ⟨E_rate, hE_rate⟩
  rcases htime with ⟨time, htime⟩
  rw [hC, hE_rate, htime] at *
  sorry

end NUMINAMATH_GPT_initial_sodium_chloride_percentage_l2189_218969


namespace NUMINAMATH_GPT_first_route_red_lights_longer_l2189_218988

-- Conditions
def first_route_base_time : ℕ := 10
def red_light_time : ℕ := 3
def num_stoplights : ℕ := 3
def second_route_time : ℕ := 14

-- Question to Answer
theorem first_route_red_lights_longer : (first_route_base_time + num_stoplights * red_light_time - second_route_time) = 5 := by
  sorry

end NUMINAMATH_GPT_first_route_red_lights_longer_l2189_218988


namespace NUMINAMATH_GPT_calculate_expression_l2189_218950

theorem calculate_expression : abs (-2) - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2189_218950


namespace NUMINAMATH_GPT_minimum_value_is_one_l2189_218906

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  (1 / (3 * a + 2)) + (1 / (3 * b + 2)) + (1 / (3 * c + 2))

theorem minimum_value_is_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  minimum_value a b c = 1 := by
  sorry

end NUMINAMATH_GPT_minimum_value_is_one_l2189_218906


namespace NUMINAMATH_GPT_minimum_value_l2189_218936

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃z, z = (x^2 + y^2) / (x + y)^2 ∧ z ≥ 1/2 := 
sorry

end NUMINAMATH_GPT_minimum_value_l2189_218936


namespace NUMINAMATH_GPT_at_most_2n_div_3_good_triangles_l2189_218951

-- Definitions based on problem conditions
universe u

structure Polygon (α : Type u) :=
(vertices : List α)
(convex : True)  -- Placeholder for convexity condition

-- Definition for a good triangle
structure Triangle (α : Type u) :=
(vertices : Fin 3 → α)
(unit_length : (Fin 3) → (Fin 3) → Bool)  -- Placeholder for unit length side condition

noncomputable def count_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) : Nat := sorry

theorem at_most_2n_div_3_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) :
  count_good_triangles P ≤ P.vertices.length * 2 / 3 := 
sorry

end NUMINAMATH_GPT_at_most_2n_div_3_good_triangles_l2189_218951


namespace NUMINAMATH_GPT_sophia_book_length_l2189_218999

variables {P : ℕ}

def total_pages (P : ℕ) : Prop :=
  (2 / 3 : ℝ) * P = (1 / 3 : ℝ) * P + 90

theorem sophia_book_length 
  (h1 : total_pages P) :
  P = 270 :=
sorry

end NUMINAMATH_GPT_sophia_book_length_l2189_218999


namespace NUMINAMATH_GPT_aunt_gemma_dog_food_l2189_218918

theorem aunt_gemma_dog_food :
  ∀ (dogs : ℕ) (grams_per_meal : ℕ) (meals_per_day : ℕ) (sack_kg : ℕ) (days : ℕ), 
    dogs = 4 →
    grams_per_meal = 250 →
    meals_per_day = 2 →
    sack_kg = 50 →
    days = 50 →
    (dogs * meals_per_day * grams_per_meal * days) / (1000 * sack_kg) = 2 :=
by
  intros dogs grams_per_meal meals_per_day sack_kg days
  intros h_dogs h_grams_per_meal h_meals_per_day h_sack_kg h_days
  sorry

end NUMINAMATH_GPT_aunt_gemma_dog_food_l2189_218918


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2189_218954

open scoped Nat

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n) :
  ∃ r : ℝ, (∀ n : ℕ, a n = a 0 * r ^ n) ∧ (r = 4) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2189_218954


namespace NUMINAMATH_GPT_total_books_proof_l2189_218980

def initial_books : ℝ := 41.0
def added_books_first : ℝ := 33.0
def added_books_next : ℝ := 2.0

theorem total_books_proof : initial_books + added_books_first + added_books_next = 76.0 :=
by
  sorry

end NUMINAMATH_GPT_total_books_proof_l2189_218980


namespace NUMINAMATH_GPT_quadratic_always_positive_l2189_218990

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 :=
sorry

end NUMINAMATH_GPT_quadratic_always_positive_l2189_218990


namespace NUMINAMATH_GPT_lcm_9_12_15_l2189_218978

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end NUMINAMATH_GPT_lcm_9_12_15_l2189_218978


namespace NUMINAMATH_GPT_gcd_2025_2070_l2189_218922

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end NUMINAMATH_GPT_gcd_2025_2070_l2189_218922


namespace NUMINAMATH_GPT_derek_dogs_count_l2189_218962

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end NUMINAMATH_GPT_derek_dogs_count_l2189_218962


namespace NUMINAMATH_GPT_cars_needed_to_double_march_earnings_l2189_218961

-- Definition of given conditions
def base_salary : Nat := 1000
def commission_per_car : Nat := 200
def march_earnings : Nat := 2000

-- Question to prove
theorem cars_needed_to_double_march_earnings : 
  (2 * march_earnings - base_salary) / commission_per_car = 15 := 
by sorry

end NUMINAMATH_GPT_cars_needed_to_double_march_earnings_l2189_218961


namespace NUMINAMATH_GPT_least_weight_of_oranges_l2189_218933

theorem least_weight_of_oranges :
  ∀ (a o : ℝ), (a ≥ 8 + 3 * o) → (a ≤ 4 * o) → (o ≥ 8) :=
by
  intros a o h1 h2
  sorry

end NUMINAMATH_GPT_least_weight_of_oranges_l2189_218933


namespace NUMINAMATH_GPT_angle_between_vectors_acute_l2189_218993

def isAcuteAngle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 > 0

def notCollinear (a b : ℝ × ℝ) : Prop :=
  ¬ ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem angle_between_vectors_acute (m : ℝ) :
  let a := (-1, 1)
  let b := (2 * m, m + 3)
  isAcuteAngle a b ∧ notCollinear a b ↔ m < 3 ∧ m ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_vectors_acute_l2189_218993


namespace NUMINAMATH_GPT_find_a2016_l2189_218947

theorem find_a2016 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : ∀ n : ℕ, a (n + 2) = a (n + 1) - a n) : a 2016 = -2 := 
by sorry

end NUMINAMATH_GPT_find_a2016_l2189_218947


namespace NUMINAMATH_GPT_question_a_gt_b_neither_sufficient_nor_necessary_l2189_218926

theorem question_a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by
  sorry

end NUMINAMATH_GPT_question_a_gt_b_neither_sufficient_nor_necessary_l2189_218926


namespace NUMINAMATH_GPT_Vincent_sells_8_literature_books_per_day_l2189_218995

theorem Vincent_sells_8_literature_books_per_day
  (fantasy_book_cost : ℕ)
  (literature_book_cost : ℕ)
  (fantasy_books_sold_per_day : ℕ)
  (total_earnings_5_days : ℕ)
  (H_fantasy_book_cost : fantasy_book_cost = 4)
  (H_literature_book_cost : literature_book_cost = 2)
  (H_fantasy_books_sold_per_day : fantasy_books_sold_per_day = 5)
  (H_total_earnings_5_days : total_earnings_5_days = 180) :
  ∃ L : ℕ, L = 8 :=
by
  sorry

end NUMINAMATH_GPT_Vincent_sells_8_literature_books_per_day_l2189_218995


namespace NUMINAMATH_GPT_percent_of_number_l2189_218924

theorem percent_of_number (x : ℝ) (hx : (120 / x) = (75 / 100)) : x = 160 := 
sorry

end NUMINAMATH_GPT_percent_of_number_l2189_218924


namespace NUMINAMATH_GPT_pigeonhole_principle_f_m_l2189_218905

theorem pigeonhole_principle_f_m :
  ∀ (n : ℕ) (f : ℕ × ℕ → Fin (n + 1)), n ≤ 44 →
    ∃ (i j l k p m : ℕ),
      1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
      1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p ∧
      f (i, j) = f (i, k) ∧ f (i, k) = f (l, j) ∧ f (l, j) = f (l, k) :=
by {
  sorry
}

end NUMINAMATH_GPT_pigeonhole_principle_f_m_l2189_218905


namespace NUMINAMATH_GPT_Wendy_runs_farther_l2189_218991

-- Define the distances Wendy ran and walked
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- Define the difference in distances
def difference : ℝ := distance_ran - distance_walked

-- The theorem to prove
theorem Wendy_runs_farther : difference = 10.66 := by
  sorry

end NUMINAMATH_GPT_Wendy_runs_farther_l2189_218991


namespace NUMINAMATH_GPT_total_frogs_in_both_ponds_l2189_218908

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end NUMINAMATH_GPT_total_frogs_in_both_ponds_l2189_218908


namespace NUMINAMATH_GPT_min_bound_of_gcd_condition_l2189_218943

theorem min_bound_of_gcd_condition :
  ∃ c > 0, ∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n ∧
  (∀ i j : ℕ, i ≤ n ∧ j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n) ^ (n / 2) :=
sorry

end NUMINAMATH_GPT_min_bound_of_gcd_condition_l2189_218943


namespace NUMINAMATH_GPT_fraction_solution_l2189_218985

theorem fraction_solution (N : ℝ) (h : N = 12.0) : (0.6667 * N + 1) = (3/4) * N := by 
  sorry

end NUMINAMATH_GPT_fraction_solution_l2189_218985


namespace NUMINAMATH_GPT_find_z_coordinate_of_point_on_line_passing_through_l2189_218965

theorem find_z_coordinate_of_point_on_line_passing_through
  (p1 p2 : ℝ × ℝ × ℝ)
  (x_value : ℝ)
  (z_value : ℝ)
  (h1 : p1 = (1, 3, 2))
  (h2 : p2 = (4, 2, -1))
  (h3 : x_value = 3)
  (param : ℝ)
  (h4 : x_value = (1 + 3 * param))
  (h5 : z_value = (2 - 3 * param)) :
  z_value = 0 := by
  sorry

end NUMINAMATH_GPT_find_z_coordinate_of_point_on_line_passing_through_l2189_218965


namespace NUMINAMATH_GPT_arcade_game_monster_perimeter_l2189_218972

theorem arcade_game_monster_perimeter :
  let r := 1 -- radius of the circle in cm
  let theta := 60 -- central angle of the missing sector in degrees
  let circumference := 2 * Real.pi * r -- circumference of the full circle
  let arc_fraction := (360 - theta) / 360 -- fraction of the circle forming the arc
  let arc_length := arc_fraction * circumference -- length of the arc
  let perimeter := arc_length + 2 * r -- total perimeter (arc + two radii)
  perimeter = (5 / 3) * Real.pi + 2 :=
by
  sorry

end NUMINAMATH_GPT_arcade_game_monster_perimeter_l2189_218972


namespace NUMINAMATH_GPT_math_problem_l2189_218979

theorem math_problem : 2357 + 3572 + 5723 + 2 * 7235 = 26122 :=
  by sorry

end NUMINAMATH_GPT_math_problem_l2189_218979


namespace NUMINAMATH_GPT_equilateral_triangle_in_ellipse_l2189_218964

-- Given
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def altitude_on_y_axis (v : ℝ × ℝ := (0, 1)) : Prop := 
  v.1 = 0 ∧ v.2 = 1

-- The problem statement translated into a Lean proof goal
theorem equilateral_triangle_in_ellipse :
  ∃ (m n : ℕ), 
    (∀ (x y : ℝ), ellipse x y) →
    altitude_on_y_axis (0,1) →
    m.gcd n = 1 ∧ m + n = 937 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_in_ellipse_l2189_218964


namespace NUMINAMATH_GPT_uniquely_determine_T_l2189_218992

theorem uniquely_determine_T'_n (b e : ℤ) (S' T' : ℕ → ℤ)
  (hb : ∀ n, S' n = n * (2 * b + (n - 1) * e) / 2)
  (ht : ∀ n, T' n = n * (n + 1) * (3 * b + (n - 1) * e) / 6)
  (h3028 : S' 3028 = 3028 * (b + 1514 * e)) :
  T' 4543 = (4543 * (4543 + 1) * (3 * b + 4542 * e)) / 6 :=
by
  sorry

end NUMINAMATH_GPT_uniquely_determine_T_l2189_218992


namespace NUMINAMATH_GPT_greatest_possible_mean_BC_l2189_218973

-- Mean weights for piles A, B
def mean_weight_A : ℝ := 60
def mean_weight_B : ℝ := 70

-- Combined mean weight for piles A and B
def mean_weight_AB : ℝ := 64

-- Combined mean weight for piles A and C
def mean_weight_AC : ℝ := 66

-- Prove that the greatest possible integer value for the mean weight of
-- the rocks in the combined piles B and C
theorem greatest_possible_mean_BC : ∃ (w : ℝ), (⌊w⌋ = 75) :=
by
  -- Definitions and assumptions based on problem conditions
  have h1 : mean_weight_A = 60 := rfl
  have h2 : mean_weight_B = 70 := rfl
  have h3 : mean_weight_AB = 64 := rfl
  have h4 : mean_weight_AC = 66 := rfl
  sorry

end NUMINAMATH_GPT_greatest_possible_mean_BC_l2189_218973


namespace NUMINAMATH_GPT_quadratic_eq_standard_form_coefficients_l2189_218935

-- Define initial quadratic equation
def initial_eq (x : ℝ) : Prop := (x + 5) * (x + 3) = 2 * x^2

-- Define the quadratic equation in standard form
def standard_form (x : ℝ) : Prop := x^2 - 8 * x - 15 = 0

-- Prove that given the initial equation, it can be converted to its standard form
theorem quadratic_eq_standard_form (x : ℝ) :
  initial_eq x → standard_form x := 
sorry

-- Verify the coefficients of the quadratic term, linear term, and constant term
theorem coefficients (x : ℝ) :
  initial_eq x → 
  (∀ a b c : ℝ, (a = 1) ∧ (b = -8) ∧ (c = -15) → standard_form x) :=
sorry

end NUMINAMATH_GPT_quadratic_eq_standard_form_coefficients_l2189_218935


namespace NUMINAMATH_GPT_ratio_trumpet_to_running_l2189_218941

def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 40

theorem ratio_trumpet_to_running : (trumpet_hours : ℚ) / running_hours = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_trumpet_to_running_l2189_218941


namespace NUMINAMATH_GPT_first_term_geometric_series_l2189_218927

theorem first_term_geometric_series (a1 q : ℝ) (h1 : a1 / (1 - q) = 1)
  (h2 : |a1| / (1 - |q|) = 2) (h3 : -1 < q) (h4 : q < 1) (h5 : q ≠ 0) :
  a1 = 4 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_term_geometric_series_l2189_218927


namespace NUMINAMATH_GPT_sqrt_floor_19992000_l2189_218968

theorem sqrt_floor_19992000 : (Int.floor (Real.sqrt 19992000)) = 4471 := by
  sorry

end NUMINAMATH_GPT_sqrt_floor_19992000_l2189_218968


namespace NUMINAMATH_GPT_computer_multiplications_in_30_minutes_l2189_218901

def multiplications_per_second : ℕ := 20000
def seconds_per_minute : ℕ := 60
def minutes : ℕ := 30
def total_seconds : ℕ := minutes * seconds_per_minute
def expected_multiplications : ℕ := 36000000

theorem computer_multiplications_in_30_minutes :
  multiplications_per_second * total_seconds = expected_multiplications :=
by
  sorry

end NUMINAMATH_GPT_computer_multiplications_in_30_minutes_l2189_218901


namespace NUMINAMATH_GPT_average_speed_is_80_l2189_218902

def distance : ℕ := 100

def time : ℚ := 5 / 4  -- 1.25 hours expressed as a rational number

noncomputable def average_speed : ℚ := distance / time

theorem average_speed_is_80 : average_speed = 80 := by
  sorry

end NUMINAMATH_GPT_average_speed_is_80_l2189_218902


namespace NUMINAMATH_GPT_min_points_to_win_l2189_218914

theorem min_points_to_win : ∀ (points : ℕ), (∀ (race_results : ℕ → ℕ), 
  (points = race_results 1 * 4 + race_results 2 * 2 + race_results 3 * 1) 
  ∧ (∀ i, 1 ≤ race_results i ∧ race_results i ≤ 4) 
  ∧ (∀ i j, i ≠ j → race_results i ≠ race_results j) 
  ∧ (race_results 1 + race_results 2 + race_results 3 = 4)) → (15 ≤ points) :=
by
  sorry

end NUMINAMATH_GPT_min_points_to_win_l2189_218914


namespace NUMINAMATH_GPT_equal_students_initially_l2189_218915

theorem equal_students_initially (B G : ℕ) (h1 : B = G) (h2 : B = 2 * (G - 8)) : B + G = 32 :=
by
  sorry

end NUMINAMATH_GPT_equal_students_initially_l2189_218915


namespace NUMINAMATH_GPT_max_value_at_2_l2189_218938

noncomputable def f (x : ℝ) : ℝ := -x^3 + 12 * x

theorem max_value_at_2 : ∃ a : ℝ, (∀ x : ℝ, f x ≤ f a) ∧ a = 2 := 
by
  sorry

end NUMINAMATH_GPT_max_value_at_2_l2189_218938


namespace NUMINAMATH_GPT_pages_read_tonight_l2189_218930

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem pages_read_tonight :
  let pages_3_nights_ago := 20
  let pages_2_nights_ago := 20^2 + 5
  let pages_last_night := sum_of_digits pages_2_nights_ago * 3
  let total_pages := 500
  total_pages - (pages_3_nights_ago + pages_2_nights_ago + pages_last_night) = 48 :=
by
  sorry

end NUMINAMATH_GPT_pages_read_tonight_l2189_218930


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2189_218967

theorem problem_part1 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  2 * ((1 / x) + (1 / y) + (1 / z)) ≤ (1 / p) + (1 / q) + (1 / r) :=
sorry

theorem problem_part2 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  x * y + y * z + z * x ≥ 2 * (p * x + q * y + r * z) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2189_218967


namespace NUMINAMATH_GPT_sum_of_integers_l2189_218916

theorem sum_of_integers (s : Finset ℕ) (h₀ : ∀ a ∈ s, 0 ≤ a ∧ a ≤ 124)
  (h₁ : ∀ a ∈ s, a^3 % 125 = 2) : s.sum id = 265 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l2189_218916


namespace NUMINAMATH_GPT_perfect_square_n_l2189_218953

theorem perfect_square_n (m : ℤ) :
  ∃ (n : ℤ), (n = 7 * m^2 + 6 * m + 1 ∨ n = 7 * m^2 - 6 * m + 1) ∧ ∃ (k : ℤ), 7 * n + 2 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_n_l2189_218953


namespace NUMINAMATH_GPT_problem1_problem2_l2189_218920

-- Definitions of the sets A and B based on the given conditions
def A : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - a) * (x - 3 * a) < 0 }

-- Proof statement for problem (1)
theorem problem1 (a : ℝ) : (∀ x, x ∈ A → x ∈ (B a)) ↔ (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

-- Proof statement for problem (2)
theorem problem2 (a : ℝ) : (∀ x, (x ∈ A ∧ x ∈ (B a)) ↔ (3 < x ∧ x < 4)) ↔ (a = 3) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2189_218920


namespace NUMINAMATH_GPT_find_C_coordinates_l2189_218998

open Real

noncomputable def pointC_coordinates (A B : ℝ × ℝ) (hA : A = (-1, 0)) (hB : B = (3, 8)) (hdist : dist A C = 2 * dist C B) : ℝ × ℝ :=
  (⟨7 / 3, 20 / 3⟩)

theorem find_C_coordinates :
  ∀ (A B C : ℝ × ℝ), 
  A = (-1, 0) → B = (3, 8) → dist A C = 2 * dist C B →
  C = (7 / 3, 20 / 3) :=
by 
  intros A B C hA hB hdist
  -- We will use the given conditions and definitions to find the coordinates of C
  sorry

end NUMINAMATH_GPT_find_C_coordinates_l2189_218998


namespace NUMINAMATH_GPT_find_central_angle_of_sector_l2189_218952

variables (r θ : ℝ)

def sector_arc_length (r θ : ℝ) := r * θ
def sector_area (r θ : ℝ) := 0.5 * r^2 * θ

theorem find_central_angle_of_sector
  (l : ℝ)
  (A : ℝ)
  (hl : l = sector_arc_length r θ)
  (hA : A = sector_area r θ)
  (hl_val : l = 4)
  (hA_val : A = 2) :
  θ = 4 :=
sorry

end NUMINAMATH_GPT_find_central_angle_of_sector_l2189_218952


namespace NUMINAMATH_GPT_customer_paid_correct_amount_l2189_218981

theorem customer_paid_correct_amount (cost_price : ℕ) (markup_percentage : ℕ) (total_price : ℕ) :
  cost_price = 6500 → 
  markup_percentage = 30 → 
  total_price = cost_price + (cost_price * markup_percentage / 100) → 
  total_price = 8450 :=
by
  intros h_cost_price h_markup_percentage h_total_price
  sorry

end NUMINAMATH_GPT_customer_paid_correct_amount_l2189_218981


namespace NUMINAMATH_GPT_probability_not_grade_5_l2189_218921

theorem probability_not_grade_5 :
  let A1 := 0.3
  let A2 := 0.4
  let A3 := 0.2
  let A4 := 0.1
  (A1 + A2 + A3 + A4 = 1) → (1 - A1 = 0.7) := by
  intros A1_def A2_def A3_def A4_def h
  sorry

end NUMINAMATH_GPT_probability_not_grade_5_l2189_218921


namespace NUMINAMATH_GPT_sum_infinite_geometric_series_l2189_218934

theorem sum_infinite_geometric_series :
  let a := 1
  let r := (1 : ℝ) / 3
  ∑' (n : ℕ), a * r ^ n = (3 : ℝ) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_infinite_geometric_series_l2189_218934


namespace NUMINAMATH_GPT_find_rate_per_kg_mangoes_l2189_218997

noncomputable def rate_per_kg_mangoes
  (cost_grapes_rate : ℕ)
  (quantity_grapes : ℕ)
  (quantity_mangoes : ℕ)
  (total_paid : ℕ)
  (rate_grapes : ℕ)
  (rate_mangoes : ℕ) :=
  total_paid = (rate_grapes * quantity_grapes) + (rate_mangoes * quantity_mangoes)

theorem find_rate_per_kg_mangoes :
  rate_per_kg_mangoes 70 8 11 1165 70 55 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_per_kg_mangoes_l2189_218997


namespace NUMINAMATH_GPT_graduation_photo_arrangement_l2189_218904

theorem graduation_photo_arrangement (teachers middle_positions other_students : Finset ℕ) (A B : ℕ) :
  teachers.card = 2 ∧ middle_positions.card = 2 ∧ 
  (other_students ∪ {A, B}).card = 4 ∧ ∀ t ∈ teachers, t ∈ middle_positions →
  ∃ arrangements : ℕ, arrangements = 8 :=
by
  sorry

end NUMINAMATH_GPT_graduation_photo_arrangement_l2189_218904


namespace NUMINAMATH_GPT_consecutive_numbers_equation_l2189_218913

theorem consecutive_numbers_equation (x y z : ℤ) (h1 : z = 3) (h2 : y = z + 1) (h3 : x = y + 1) 
(h4 : 2 * x + 3 * y + 3 * z = 5 * y + n) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_numbers_equation_l2189_218913


namespace NUMINAMATH_GPT_probability_diamond_first_and_ace_or_king_second_l2189_218982

-- Define the condition of the combined deck consisting of two standard decks (104 cards total)
def two_standard_decks := 104

-- Define the number of diamonds, aces, and kings in the combined deck
def number_of_diamonds := 26
def number_of_aces := 8
def number_of_kings := 8

-- Define the events for drawing cards
def first_card_is_diamond := (number_of_diamonds : ℕ) / (two_standard_decks : ℕ)
def second_card_is_ace_or_king_if_first_is_not_ace_or_king :=
  (16 / 103 : ℚ) -- 16 = 8 (aces) + 8 (kings)
def second_card_is_ace_or_king_if_first_is_ace_or_king :=
  (15 / 103 : ℚ) -- 15 = 7 (remaining aces) + 7 (remaining kings) + 1 (remaining ace or king of the same suit)

-- Define the probabilities of the combined event
def probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king :=
  (22 / 104) * (16 / 103)
def probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king :=
  (4 / 104) * (15 / 103)

-- Define the total probability combining both events
noncomputable def total_probability :=
  probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king +
  probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king

-- Theorem stating the desired probability result
theorem probability_diamond_first_and_ace_or_king_second :
  total_probability = (103 / 2678 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_diamond_first_and_ace_or_king_second_l2189_218982


namespace NUMINAMATH_GPT_percentage_students_went_on_trip_l2189_218907

theorem percentage_students_went_on_trip
  (total_students : ℕ)
  (students_march : ℕ)
  (students_march_more_than_100 : ℕ)
  (students_june : ℕ)
  (students_june_more_than_100 : ℕ)
  (total_more_than_100_either_trip : ℕ) :
  total_students = 100 → students_march = 20 → students_march_more_than_100 = 7 →
  students_june = 15 → students_june_more_than_100 = 6 →
  70 * total_more_than_100_either_trip = 7 * 100 →
  (students_march + students_june) * 100 / total_students = 35 :=
by
  intros h_total h_march h_march_100 h_june h_june_100 h_total_100
  sorry

end NUMINAMATH_GPT_percentage_students_went_on_trip_l2189_218907


namespace NUMINAMATH_GPT_total_difference_is_18_l2189_218931

-- Define variables for Mike, Joe, and Anna's bills
variables (m j a : ℝ)

-- Define the conditions given in the problem
def MikeTipped := (0.15 * m = 3)
def JoeTipped := (0.25 * j = 3)
def AnnaTipped := (0.10 * a = 3)

-- Prove the total amount of money that was different between the highest and lowest bill is 18
theorem total_difference_is_18 (MikeTipped : 0.15 * m = 3) (JoeTipped : 0.25 * j = 3) (AnnaTipped : 0.10 * a = 3) :
  |a - j| = 18 := 
sorry

end NUMINAMATH_GPT_total_difference_is_18_l2189_218931


namespace NUMINAMATH_GPT_river_depth_mid_may_l2189_218960

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_river_depth_mid_may_l2189_218960


namespace NUMINAMATH_GPT_total_games_is_272_l2189_218912

-- Define the number of players
def n : ℕ := 17

-- Define the formula for the number of games played
def total_games (n : ℕ) : ℕ := n * (n - 1)

-- Define a theorem stating that the total games played is 272
theorem total_games_is_272 : total_games n = 272 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_games_is_272_l2189_218912


namespace NUMINAMATH_GPT_students_left_in_final_year_l2189_218987

variable (s10 s_next s_final x : Nat)

-- Conditions
def initial_students : Prop := s10 = 150
def students_after_joining : Prop := s_next = s10 + 30
def students_final_year : Prop := s_final = s_next - x
def final_year_students : Prop := s_final = 165

-- Theorem to prove
theorem students_left_in_final_year (h1 : initial_students s10)
                                     (h2 : students_after_joining s10 s_next)
                                     (h3 : students_final_year s_next s_final x)
                                     (h4 : final_year_students s_final) :
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_students_left_in_final_year_l2189_218987


namespace NUMINAMATH_GPT_total_food_needed_l2189_218946

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end NUMINAMATH_GPT_total_food_needed_l2189_218946


namespace NUMINAMATH_GPT_max_x_satisfies_inequality_l2189_218956

theorem max_x_satisfies_inequality (k : ℝ) :
    (∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) → k = 8 :=
by
  intros h
  /- The proof goes here. -/
  sorry

end NUMINAMATH_GPT_max_x_satisfies_inequality_l2189_218956


namespace NUMINAMATH_GPT_bob_clean_time_l2189_218984

-- Definitions for the problem conditions
def alice_time : ℕ := 30
def bob_time := (1 / 3 : ℚ) * alice_time

-- The proof problem statement (only) in Lean 4
theorem bob_clean_time : bob_time = 10 := by
  sorry

end NUMINAMATH_GPT_bob_clean_time_l2189_218984


namespace NUMINAMATH_GPT_range_of_m_l2189_218958

theorem range_of_m (m : ℝ) : 
  ((∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ (m < -1)) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2189_218958


namespace NUMINAMATH_GPT_line_contains_diameter_of_circle_l2189_218994

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 8 = 0

noncomputable def equation_of_line (x y : ℝ) : Prop :=
  2*x - y - 1 = 0

theorem line_contains_diameter_of_circle :
  (∃ x y : ℝ, equation_of_circle x y ∧ equation_of_line x y) :=
sorry

end NUMINAMATH_GPT_line_contains_diameter_of_circle_l2189_218994


namespace NUMINAMATH_GPT_y_equals_4_if_abs_diff_eq_l2189_218977

theorem y_equals_4_if_abs_diff_eq (y : ℝ) (h : |y - 3| = |y - 5|) : y = 4 :=
sorry

end NUMINAMATH_GPT_y_equals_4_if_abs_diff_eq_l2189_218977


namespace NUMINAMATH_GPT_students_walk_fraction_l2189_218996

theorem students_walk_fraction (h1 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/3))
                               (h2 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/5))
                               (h3 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/8))
                               (h4 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/10)) :
  ∃ (students : ℕ), (students - num1 - num2 - num3 - num4) / students = 29 / 120 :=
by
  sorry

end NUMINAMATH_GPT_students_walk_fraction_l2189_218996


namespace NUMINAMATH_GPT_complex_mul_l2189_218928

theorem complex_mul (i : ℂ) (h : i^2 = -1) :
    (1 - i) * (1 + 2 * i) = 3 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_mul_l2189_218928


namespace NUMINAMATH_GPT_fraction_is_terminating_decimal_l2189_218900

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_terminating_decimal_l2189_218900


namespace NUMINAMATH_GPT_max_members_in_band_l2189_218925

theorem max_members_in_band (m : ℤ) (h1 : 30 * m % 31 = 6) (h2 : 30 * m < 1200) : 30 * m = 360 :=
by {
  sorry -- Proof steps are not required according to the procedure
}

end NUMINAMATH_GPT_max_members_in_band_l2189_218925


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l2189_218910

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m, m < n → (∃ k1 k2 : ℕ, (m + 150 = 2^k1 * 5^k2 ∧ m > 0) → false)) ∧ (∃ k1 k2 : ℕ, (n + 150 = 2^k1 * 5^k2) ∧ n > 0) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l2189_218910


namespace NUMINAMATH_GPT_total_cases_after_three_weeks_l2189_218989

-- Definitions and conditions directly from the problem
def week1_cases : ℕ := 5000
def week2_cases : ℕ := week1_cases / 2
def week3_cases : ℕ := week2_cases + 2000
def total_cases : ℕ := week1_cases + week2_cases + week3_cases

-- The theorem to prove
theorem total_cases_after_three_weeks :
  total_cases = 12000 := 
by
  -- Sorry allows us to skip the actual proof
  sorry

end NUMINAMATH_GPT_total_cases_after_three_weeks_l2189_218989
