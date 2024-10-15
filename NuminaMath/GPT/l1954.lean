import Mathlib

namespace NUMINAMATH_GPT_largest_class_students_l1954_195432

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 115) : x = 27 := 
by 
  sorry

end NUMINAMATH_GPT_largest_class_students_l1954_195432


namespace NUMINAMATH_GPT_polynomial_exists_l1954_195489

open Polynomial

noncomputable def exists_polynomial_2013 : Prop :=
  ∃ (f : Polynomial ℤ), (∀ (n : ℕ), n ≤ f.natDegree → (coeff f n = 1 ∨ coeff f n = -1))
                         ∧ ((X - 1) ^ 2013 ∣ f)

theorem polynomial_exists : exists_polynomial_2013 :=
  sorry

end NUMINAMATH_GPT_polynomial_exists_l1954_195489


namespace NUMINAMATH_GPT_arrange_books_correct_l1954_195480

def math_books : Nat := 4
def history_books : Nat := 4

def arrangements (m h : Nat) : Nat := sorry

theorem arrange_books_correct :
  arrangements math_books history_books = 576 := sorry

end NUMINAMATH_GPT_arrange_books_correct_l1954_195480


namespace NUMINAMATH_GPT_pizza_store_total_sales_l1954_195435

theorem pizza_store_total_sales (pepperoni bacon cheese : ℕ) (h1 : pepperoni = 2) (h2 : bacon = 6) (h3 : cheese = 6) :
  pepperoni + bacon + cheese = 14 :=
by sorry

end NUMINAMATH_GPT_pizza_store_total_sales_l1954_195435


namespace NUMINAMATH_GPT_vertex_y_coord_of_h_l1954_195495

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 1
def h (x : ℝ) : ℝ := f x - g x

theorem vertex_y_coord_of_h : h (-1 / 10) = 79 / 20 := by
  sorry

end NUMINAMATH_GPT_vertex_y_coord_of_h_l1954_195495


namespace NUMINAMATH_GPT_smallest_prime_perimeter_l1954_195488

-- Define a function that checks if a number is an odd prime
def is_odd_prime (n : ℕ) : Prop :=
  n > 2 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)) ∧ (n % 2 = 1)

-- Define a function that checks if three numbers are consecutive odd primes
def consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  b = a + 2 ∧ c = b + 2

-- Define a function that checks if three numbers form a scalene triangle and satisfy the triangle inequality
def scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem to prove
theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), consecutive_odd_primes a b c ∧ scalene_triangle a b c ∧ (a + b + c = 23) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_perimeter_l1954_195488


namespace NUMINAMATH_GPT_clea_escalator_time_l1954_195455

theorem clea_escalator_time (x y k : ℕ) (h1 : 90 * x = y) (h2 : 30 * (x + k) = y) :
  (y / k) = 45 := by
  sorry

end NUMINAMATH_GPT_clea_escalator_time_l1954_195455


namespace NUMINAMATH_GPT_isosceles_triangle_l1954_195447

theorem isosceles_triangle
  (α β γ : ℝ)
  (triangle_sum : α + β + γ = Real.pi)
  (second_triangle_angle1 : α + β < Real.pi)
  (second_triangle_angle2 : α + γ < Real.pi) :
  β = γ := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_l1954_195447


namespace NUMINAMATH_GPT_latest_start_time_l1954_195424

-- Define the weights of the turkeys
def turkey_weights : List ℕ := [16, 18, 20, 22]

-- Define the roasting time per pound
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time in 24-hour format
def dinner_time : ℕ := 18 * 60 -- 18:00 in minutes

-- Calculate the total roasting time
def total_roasting_time (weights : List ℕ) (time_per_pound : ℕ) : ℕ :=
  weights.foldr (λ weight acc => weight * time_per_pound + acc) 0

-- Calculate the latest start time
def latest_roasting_start_time (total_time : ℕ) (dinner_time : ℕ) : ℕ :=
  let start_time := dinner_time - total_time
  if start_time < 0 then start_time + 24 * 60 else start_time

-- Convert minutes to hours:minutes format
def time_in_hours_minutes (time : ℕ) : String :=
  let hours := time / 60
  let minutes := time % 60
  toString hours ++ ":" ++ toString minutes

theorem latest_start_time : 
  time_in_hours_minutes (latest_roasting_start_time (total_roasting_time turkey_weights roasting_time_per_pound) dinner_time) = "23:00" := by
  sorry

end NUMINAMATH_GPT_latest_start_time_l1954_195424


namespace NUMINAMATH_GPT_tangent_line_of_circle_l1954_195450

theorem tangent_line_of_circle (x y : ℝ)
    (C_def : (x - 2)^2 + (y - 3)^2 = 25)
    (P : (ℝ × ℝ)) (P_def : P = (-1, 7)) :
    (3 * x - 4 * y + 31 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_of_circle_l1954_195450


namespace NUMINAMATH_GPT_find_a5_l1954_195454

-- Sequence definition
def a : ℕ → ℤ
| 0     => 1
| (n+1) => 2 * a n + 3

-- Theorem to prove
theorem find_a5 : a 4 = 61 := sorry

end NUMINAMATH_GPT_find_a5_l1954_195454


namespace NUMINAMATH_GPT_perpendicular_vectors_l1954_195479

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n : ℝ × ℝ := (-3, 2)

-- Define the conditions to be checked
def km_plus_n (k : ℝ) : ℝ × ℝ := (k * m.1 + n.1, k * m.2 + n.2)
def m_minus_3n : ℝ × ℝ := (m.1 - 3 * n.1, m.2 - 3 * n.2)

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that for k = 19, the two vectors are perpendicular
theorem perpendicular_vectors (k : ℝ) (h : k = 19) : dot_product (km_plus_n k) (m_minus_3n) = 0 := by
  rw [h]
  simp [km_plus_n, m_minus_3n, dot_product]
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l1954_195479


namespace NUMINAMATH_GPT_number_of_valid_3_digit_numbers_l1954_195411

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_3_digit_numbers_count : ℕ :=
  let digits := [(4, 8), (8, 4), (6, 6)]
  digits.length * 9

theorem number_of_valid_3_digit_numbers : valid_3_digit_numbers_count = 27 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_3_digit_numbers_l1954_195411


namespace NUMINAMATH_GPT_sum_of_solutions_l1954_195449

  theorem sum_of_solutions :
    (∃ x : ℝ, x = abs (2 * x - abs (50 - 2 * x)) ∧ ∃ y : ℝ, y = abs (2 * y - abs (50 - 2 * y)) ∧ ∃ z : ℝ, z = abs (2 * z - abs (50 - 2 * z)) ∧ (x + y + z = 170 / 3)) :=
  sorry
  
end NUMINAMATH_GPT_sum_of_solutions_l1954_195449


namespace NUMINAMATH_GPT_tan_alpha_is_three_halves_l1954_195417

theorem tan_alpha_is_three_halves (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) : 
  Real.tan α = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_is_three_halves_l1954_195417


namespace NUMINAMATH_GPT_greatest_value_is_B_l1954_195456

def x : Int := -6

def A : Int := 2 + x
def B : Int := 2 - x
def C : Int := x - 1
def D : Int := x
def E : Int := x / 2

theorem greatest_value_is_B :
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_is_B_l1954_195456


namespace NUMINAMATH_GPT_equation_is_linear_l1954_195419

-- Define the conditions and the proof statement
theorem equation_is_linear (m n : ℕ) : 3 * x ^ (2 * m + 1) - 2 * y ^ (n - 1) = 7 → (2 * m + 1 = 1) ∧ (n - 1 = 1) → m = 0 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_equation_is_linear_l1954_195419


namespace NUMINAMATH_GPT_units_digit_of_large_power_l1954_195422

theorem units_digit_of_large_power
  (units_147_1997_pow2999: ℕ) 
  (h1 : units_147_1997_pow2999 = (147 ^ 1997) % 10)
  (h2 : ∀ k, (7 ^ (k * 4 + 1)) % 10 = 7)
  (h3 : ∀ m, (7 ^ (m * 4 + 3)) % 10 = 3)
  : units_147_1997_pow2999 % 10 = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_of_large_power_l1954_195422


namespace NUMINAMATH_GPT_num_five_dollar_coins_l1954_195437

theorem num_five_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_num_five_dollar_coins_l1954_195437


namespace NUMINAMATH_GPT_toys_per_week_production_l1954_195433

-- Define the necessary conditions
def days_per_week : Nat := 4
def toys_per_day : Nat := 1500

-- Define the theorem to prove the total number of toys produced per week
theorem toys_per_week_production : 
  ∀ (days_per_week toys_per_day : Nat), 
    (days_per_week = 4) →
    (toys_per_day = 1500) →
    (days_per_week * toys_per_day = 6000) := 
by
  intros
  sorry

end NUMINAMATH_GPT_toys_per_week_production_l1954_195433


namespace NUMINAMATH_GPT_area_of_largest_square_l1954_195487

theorem area_of_largest_square (a b c : ℕ) (h_triangle : c^2 = a^2 + b^2) (h_sum_areas : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_area_of_largest_square_l1954_195487


namespace NUMINAMATH_GPT_range_of_a_l1954_195418

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
by
  sorry -- The proof is omitted as per the instructions.

end NUMINAMATH_GPT_range_of_a_l1954_195418


namespace NUMINAMATH_GPT_range_of_a_l1954_195481

noncomputable def A : Set ℝ := {x | x^2 ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : a ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1954_195481


namespace NUMINAMATH_GPT_intersection_A_B_l1954_195443

open Set

-- Given definitions of sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 2 * x ≥ 0}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {-1, 0, 2} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1954_195443


namespace NUMINAMATH_GPT_calculate_expression_l1954_195498

theorem calculate_expression :
  107 * 107 + 93 * 93 = 20098 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1954_195498


namespace NUMINAMATH_GPT_constant_value_AP_AQ_l1954_195478

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def circle_O (x y : ℝ) : Prop :=
  (x^2 + y^2) = 12 / 7

theorem constant_value_AP_AQ (x y : ℝ) (h : circle_O x y) :
  ∃ (P Q : ℝ × ℝ), ellipse_trajectory (P.1) (P.2) ∧ ellipse_trajectory (Q.1) (Q.2) ∧ 
  ((P.1 - x) * (Q.1 - x) + (P.2 - y) * (Q.2 - y)) = - (12 / 7) :=
sorry

end NUMINAMATH_GPT_constant_value_AP_AQ_l1954_195478


namespace NUMINAMATH_GPT_log_base_243_l1954_195445

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_base_243 : log_base 3 243 = 5 := by
  -- this is the statement, proof is omitted
  sorry

end NUMINAMATH_GPT_log_base_243_l1954_195445


namespace NUMINAMATH_GPT_trip_first_part_length_l1954_195438

theorem trip_first_part_length
  (total_distance : ℝ := 50)
  (first_speed : ℝ := 66)
  (second_speed : ℝ := 33)
  (average_speed : ℝ := 44) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ total_distance) ∧ 44 = total_distance / (x / first_speed + (total_distance - x) / second_speed) ∧ x = 25 :=
by
  sorry

end NUMINAMATH_GPT_trip_first_part_length_l1954_195438


namespace NUMINAMATH_GPT_inverse_function_of_13_l1954_195458

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def f_inv (y : ℝ) : ℝ := (y - 4) / 3

theorem inverse_function_of_13 : f_inv (f_inv 13) = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_inverse_function_of_13_l1954_195458


namespace NUMINAMATH_GPT_rabbit_probability_l1954_195444

def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def moves : ℕ := 11
def paths_after_11_moves : ℕ := 3 ^ moves
def favorable_paths : ℕ := 24

theorem rabbit_probability :
  (favorable_paths : ℚ) / paths_after_11_moves = 24 / 177147 := by
  sorry

end NUMINAMATH_GPT_rabbit_probability_l1954_195444


namespace NUMINAMATH_GPT_circle_with_all_three_colors_l1954_195404

-- Define color type using an inductive type with three colors
inductive Color
| red
| green
| blue

-- Define a function that assigns a color to each point in the plane
def color_function (point : ℝ × ℝ) : Color := sorry

-- Define the main theorem stating that for any coloring, there exists a circle that contains points of all three colors
theorem circle_with_all_three_colors (color_func : ℝ × ℝ → Color) (exists_red : ∃ p : ℝ × ℝ, color_func p = Color.red)
                                      (exists_green : ∃ p : ℝ × ℝ, color_func p = Color.green) 
                                      (exists_blue : ∃ p : ℝ × ℝ, color_func p = Color.blue) :
    ∃ (c : ℝ × ℝ) (r : ℝ), ∃ p1 p2 p3 : ℝ × ℝ, 
             color_func p1 = Color.red ∧ color_func p2 = Color.green ∧ color_func p3 = Color.blue ∧ 
             (dist p1 c = r) ∧ (dist p2 c = r) ∧ (dist p3 c = r) :=
by 
  sorry

end NUMINAMATH_GPT_circle_with_all_three_colors_l1954_195404


namespace NUMINAMATH_GPT_recent_quarter_revenue_l1954_195413

theorem recent_quarter_revenue :
  let revenue_year_ago : Float := 69.0
  let percentage_decrease : Float := 30.434782608695656
  let decrease_in_revenue : Float := revenue_year_ago * (percentage_decrease / 100)
  let recent_quarter_revenue := revenue_year_ago - decrease_in_revenue
  recent_quarter_revenue = 48.0 := by
  sorry

end NUMINAMATH_GPT_recent_quarter_revenue_l1954_195413


namespace NUMINAMATH_GPT_fish_market_customers_l1954_195471

theorem fish_market_customers :
  let num_tuna := 10
  let weight_per_tuna := 200
  let weight_per_customer := 25
  let num_customers_no_fish := 20
  let total_tuna_weight := num_tuna * weight_per_tuna
  let num_customers_served := total_tuna_weight / weight_per_customer
  num_customers_served + num_customers_no_fish = 100 := 
by
  sorry

end NUMINAMATH_GPT_fish_market_customers_l1954_195471


namespace NUMINAMATH_GPT_lizette_stamps_count_l1954_195408

-- Conditions
def lizette_more : ℕ := 125
def minerva_stamps : ℕ := 688

-- Proof of Lizette's stamps count
theorem lizette_stamps_count : (minerva_stamps + lizette_more = 813) :=
by 
  sorry

end NUMINAMATH_GPT_lizette_stamps_count_l1954_195408


namespace NUMINAMATH_GPT_vacation_cost_in_usd_l1954_195427

theorem vacation_cost_in_usd :
  let n := 7
  let rent_per_person_eur := 65
  let transport_per_person_usd := 25
  let food_per_person_gbp := 50
  let activities_per_person_jpy := 2750
  let eur_to_usd := 1.20
  let gbp_to_usd := 1.40
  let jpy_to_usd := 0.009
  let total_rent_usd := n * rent_per_person_eur * eur_to_usd
  let total_transport_usd := n * transport_per_person_usd
  let total_food_usd := n * food_per_person_gbp * gbp_to_usd
  let total_activities_usd := n * activities_per_person_jpy * jpy_to_usd
  let total_cost_usd := total_rent_usd + total_transport_usd + total_food_usd + total_activities_usd
  total_cost_usd = 1384.25 := by
    sorry

end NUMINAMATH_GPT_vacation_cost_in_usd_l1954_195427


namespace NUMINAMATH_GPT_total_time_spent_l1954_195420

-- Definitions based on the conditions
def number_of_chairs := 2
def number_of_tables := 2
def minutes_per_piece := 8
def total_pieces := number_of_chairs + number_of_tables

-- The statement we want to prove
theorem total_time_spent : total_pieces * minutes_per_piece = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_time_spent_l1954_195420


namespace NUMINAMATH_GPT_least_possible_multiple_l1954_195463

theorem least_possible_multiple (x y z k : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 1 ≤ k)
  (h1 : 3 * x = k * z) (h2 : 4 * y = k * z) (h3 : x - y + z = 19) : 3 * x = 12 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_multiple_l1954_195463


namespace NUMINAMATH_GPT_no_real_solutions_l1954_195477

theorem no_real_solutions :
  ∀ z : ℝ, ¬ ((-6 * z + 27) ^ 2 + 4 = -2 * |z|) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1954_195477


namespace NUMINAMATH_GPT_least_area_of_figure_l1954_195442

theorem least_area_of_figure (c : ℝ) (hc : c > 1) : 
  ∃ A : ℝ, A = (4 / 3) * (c - 1)^(3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_least_area_of_figure_l1954_195442


namespace NUMINAMATH_GPT_root_relation_l1954_195483

theorem root_relation (a b x y : ℝ)
  (h1 : x + y = a)
  (h2 : (1 / x) + (1 / y) = 1 / b)
  (h3 : x = 3 * y)
  (h4 : y = a / 4) :
  b = 3 * a / 16 :=
by
  sorry

end NUMINAMATH_GPT_root_relation_l1954_195483


namespace NUMINAMATH_GPT_problem_1110_1111_1112_1113_l1954_195414

theorem problem_1110_1111_1112_1113 (r : ℕ) (hr : r > 5) : 
  (r^3 + r^2 + r) * (r^3 + r^2 + r + 1) * (r^3 + r^2 + r + 2) * (r^3 + r^2 + r + 3) = (r^6 + 2 * r^5 + 3 * r^4 + 5 * r^3 + 4 * r^2 + 3 * r + 1)^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1110_1111_1112_1113_l1954_195414


namespace NUMINAMATH_GPT_find_polynomial_l1954_195472

-- Define the polynomial function and the constant
variables {F : Type*} [Field F]

-- The main condition of the problem
def satisfies_condition (p : F → F) (c : F) :=
  ∀ x : F, p (p x) = x * p x + c * x^2

-- Prove the correct answers
theorem find_polynomial (p : F → F) (c : F) : 
  (c = 0 → ∀ x, p x = x) ∧ (c = -2 → ∀ x, p x = -x) :=
by
  sorry

end NUMINAMATH_GPT_find_polynomial_l1954_195472


namespace NUMINAMATH_GPT_cost_of_song_book_l1954_195468

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_song_book_l1954_195468


namespace NUMINAMATH_GPT_max_superior_squares_l1954_195448

theorem max_superior_squares (n : ℕ) (h : n > 2004) :
  ∃ superior_squares_count : ℕ, superior_squares_count = n * (n - 2004) := 
sorry

end NUMINAMATH_GPT_max_superior_squares_l1954_195448


namespace NUMINAMATH_GPT_gcd_apb_ab_eq1_gcd_aplusb_aminsb_l1954_195493

theorem gcd_apb_ab_eq1 (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a * b) = 1 ∧ Int.gcd (a - b) (a * b) = 1 := by
  sorry

theorem gcd_aplusb_aminsb (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a - b) = 1 ∨ Int.gcd (a + b) (a - b) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_apb_ab_eq1_gcd_aplusb_aminsb_l1954_195493


namespace NUMINAMATH_GPT_basis_service_B_l1954_195462

def vector := ℤ × ℤ

def not_collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 ≠ v1.2 * v2.1

def A : vector × vector := ((0, 0), (2, 3))
def B : vector × vector := ((-1, 3), (5, -2))
def C : vector × vector := ((3, 4), (6, 8))
def D : vector × vector := ((2, -3), (-2, 3))

theorem basis_service_B : not_collinear B.1 B.2 := by
  sorry

end NUMINAMATH_GPT_basis_service_B_l1954_195462


namespace NUMINAMATH_GPT_fenced_area_with_cutout_l1954_195415

theorem fenced_area_with_cutout :
  let rectangle_length : ℕ := 20
  let rectangle_width : ℕ := 16
  let cutout_length : ℕ := 4
  let cutout_width : ℕ := 4
  rectangle_length * rectangle_width - cutout_length * cutout_width = 304 := by
  sorry

end NUMINAMATH_GPT_fenced_area_with_cutout_l1954_195415


namespace NUMINAMATH_GPT_problem_solution_l1954_195406

theorem problem_solution (x : ℝ) (h : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1954_195406


namespace NUMINAMATH_GPT_common_ratio_of_geometric_seq_l1954_195486

variable {α : Type} [LinearOrderedField α] 
variables (a d : α) (h₁ : d ≠ 0) (h₂ : (a + 2 * d) / (a + d) = (a + 5 * d) / (a + 2 * d))

theorem common_ratio_of_geometric_seq : (a + 2 * d) / (a + d) = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_seq_l1954_195486


namespace NUMINAMATH_GPT_walter_age_in_2005_l1954_195457

theorem walter_age_in_2005 
  (y : ℕ) (gy : ℕ)
  (h1 : gy = 3 * y)
  (h2 : (2000 - y) + (2000 - gy) = 3896) : y + 5 = 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_walter_age_in_2005_l1954_195457


namespace NUMINAMATH_GPT_john_payment_l1954_195426

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_john_payment_l1954_195426


namespace NUMINAMATH_GPT_complement_of_A_cap_B_l1954_195402

def set_A (x : ℝ) : Prop := x ≤ -4 ∨ x ≥ 2
def set_B (x : ℝ) : Prop := |x - 1| ≤ 3

def A_cap_B (x : ℝ) : Prop := set_A x ∧ set_B x

def complement_A_cap_B (x : ℝ) : Prop := ¬A_cap_B x

theorem complement_of_A_cap_B :
  {x : ℝ | complement_A_cap_B x} = {x : ℝ | x < 2 ∨ x > 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_cap_B_l1954_195402


namespace NUMINAMATH_GPT_range_of_ab_c2_l1954_195484

theorem range_of_ab_c2
  (a b c : ℝ)
  (h₁: -3 < b)
  (h₂: b < a)
  (h₃: a < -1)
  (h₄: -2 < c)
  (h₅: c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_ab_c2_l1954_195484


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_l1954_195464

theorem problem1_part1 : (3 - Real.pi)^0 - 2 * Real.cos (Real.pi / 6) + abs (1 - Real.sqrt 3) + (1 / 2)⁻¹ = 2 := by
  sorry

theorem problem1_part2 {x : ℝ} : x^2 - 2 * x - 9 = 0 -> (x = 1 + Real.sqrt 10 ∨ x = 1 - Real.sqrt 10) := by
  sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_l1954_195464


namespace NUMINAMATH_GPT_cylinder_new_volume_l1954_195492

-- Definitions based on conditions
def original_volume_r_h (π R H : ℝ) : ℝ := π * R^2 * H

def new_volume (π R H : ℝ) : ℝ := π * (3 * R)^2 * (2 * H)

theorem cylinder_new_volume (π R H : ℝ) (h_original_volume : original_volume_r_h π R H = 15) :
  new_volume π R H = 270 :=
by sorry

end NUMINAMATH_GPT_cylinder_new_volume_l1954_195492


namespace NUMINAMATH_GPT_tan_15_pi_over_4_l1954_195428

theorem tan_15_pi_over_4 : Real.tan (15 * Real.pi / 4) = -1 :=
by
-- The proof is omitted.
sorry

end NUMINAMATH_GPT_tan_15_pi_over_4_l1954_195428


namespace NUMINAMATH_GPT_least_number_of_teams_l1954_195466

/-- A coach has 30 players in a team. If he wants to form teams of at most 7 players each for a tournament, we aim to prove that the least number of teams that he needs is 5. -/
theorem least_number_of_teams (players teams : ℕ) 
  (h_players : players = 30) 
  (h_teams : ∀ t, t ≤ 7 → t ∣ players) : teams = 5 := by
  sorry

end NUMINAMATH_GPT_least_number_of_teams_l1954_195466


namespace NUMINAMATH_GPT_product_of_p_r_s_l1954_195425

-- Definition of conditions
def eq1 (p : ℕ) : Prop := 4^p + 4^3 = 320
def eq2 (r : ℕ) : Prop := 3^r + 27 = 108
def eq3 (s : ℕ) : Prop := 2^s + 7^4 = 2617

-- Main statement
theorem product_of_p_r_s (p r s : ℕ) (h1 : eq1 p) (h2 : eq2 r) (h3 : eq3 s) : p * r * s = 112 :=
by sorry

end NUMINAMATH_GPT_product_of_p_r_s_l1954_195425


namespace NUMINAMATH_GPT_matrix_power_problem_l1954_195434

def B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![4, 1], ![0, 2]]

theorem matrix_power_problem : B^15 - 3 * B^14 = ![![4, 3], ![0, -2]] :=
  by sorry

end NUMINAMATH_GPT_matrix_power_problem_l1954_195434


namespace NUMINAMATH_GPT_width_of_lawn_is_60_l1954_195421

-- Define the problem conditions in Lean
def length_of_lawn : ℕ := 70
def road_width : ℕ := 10
def total_road_cost : ℕ := 3600
def cost_per_sq_meter : ℕ := 3

-- Define the proof problem
theorem width_of_lawn_is_60 (W : ℕ) 
  (h1 : (road_width * W) + (road_width * length_of_lawn) - (road_width * road_width) 
        = total_road_cost / cost_per_sq_meter) : 
  W = 60 := 
by 
  sorry

end NUMINAMATH_GPT_width_of_lawn_is_60_l1954_195421


namespace NUMINAMATH_GPT_inequality_proof_l1954_195465

theorem inequality_proof
  (x y : ℝ)
  (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1954_195465


namespace NUMINAMATH_GPT_inclination_angle_of_line_l1954_195485

-- Lean definition for the line equation and inclination angle problem
theorem inclination_angle_of_line : 
  ∃ θ : ℝ, (θ ∈ Set.Ico 0 Real.pi) ∧ (∀ x y: ℝ, x + y - 1 = 0 → Real.tan θ = -1) ∧ θ = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l1954_195485


namespace NUMINAMATH_GPT_john_spent_on_candy_l1954_195459

theorem john_spent_on_candy (M : ℝ) 
  (h1 : M = 29.999999999999996)
  (h2 : 1/5 + 1/3 + 1/10 = 19/30) :
  (11 / 30) * M = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_john_spent_on_candy_l1954_195459


namespace NUMINAMATH_GPT_monotonic_intervals_max_min_values_l1954_195436

def f (x : ℝ) := x^3 - 3*x
def f_prime (x : ℝ) := 3*(x-1)*(x+1)

theorem monotonic_intervals :
  (∀ x : ℝ, x < -1 → 0 < f_prime x) ∧ (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0) ∧ (∀ x : ℝ, x > 1 → 0 < f_prime x) :=
  by
  sorry

theorem max_min_values :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 18 ∧ f x ≥ -2 ∧ 
  (f 1 = -2) ∧
  (f 3 = 18) :=
  by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_max_min_values_l1954_195436


namespace NUMINAMATH_GPT_correct_order_of_numbers_l1954_195474

theorem correct_order_of_numbers :
  let a := (4 / 5 : ℝ)
  let b := (81 / 100 : ℝ)
  let c := 0.801
  (a ≤ c ∧ c ≤ b) :=
by
  sorry

end NUMINAMATH_GPT_correct_order_of_numbers_l1954_195474


namespace NUMINAMATH_GPT_triangle_shortest_side_l1954_195475

theorem triangle_shortest_side (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) 
    (r : ℝ) (h3 : r = 5) 
    (h4 : a = 4) (h5 : b = 10)
    (circumcircle_tangent_property : 2 * (4 + 10) * r = 30) :
  min a (min b c) = 30 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_shortest_side_l1954_195475


namespace NUMINAMATH_GPT_linda_original_savings_l1954_195429

theorem linda_original_savings (S : ℝ) (h1 : 3 / 4 * S = 300 + 300) :
  S = 1200 :=
by
  sorry -- The proof is not required.

end NUMINAMATH_GPT_linda_original_savings_l1954_195429


namespace NUMINAMATH_GPT_select_people_english_japanese_l1954_195410

-- Definitions based on conditions
def total_people : ℕ := 9
def english_speakers : ℕ := 7
def japanese_speakers : ℕ := 3

-- Theorem statement
theorem select_people_english_japanese (h1 : total_people = 9) 
                                      (h2 : english_speakers = 7) 
                                      (h3 : japanese_speakers = 3) :
  ∃ n, n = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_select_people_english_japanese_l1954_195410


namespace NUMINAMATH_GPT_product_of_reciprocals_l1954_195470

theorem product_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 :=
by
  sorry

end NUMINAMATH_GPT_product_of_reciprocals_l1954_195470


namespace NUMINAMATH_GPT_min_gx1_gx2_l1954_195440

noncomputable def f (x a : ℝ) : ℝ := x - (1 / x) - a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := x - (a / 2) * Real.log x

theorem min_gx1_gx2 (x1 x2 a : ℝ) (h1 : 0 < x1 ∧ x1 < Real.exp 1) (h2 : 0 < x2) (hx1x2: x1 * x2 = 1) (ha : a > 0) :
  f x1 a = 0 ∧ f x2 a = 0 →
  g x1 a - g x2 a = -2 / Real.exp 1 :=
by sorry

end NUMINAMATH_GPT_min_gx1_gx2_l1954_195440


namespace NUMINAMATH_GPT_maximum_obtuse_vectors_l1954_195499

-- Definition: A vector in 3D space
structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition: Dot product of two vectors
def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Condition: Two vectors form an obtuse angle if their dot product is negative
def obtuse_angle (v1 v2 : Vector3D) : Prop :=
  dot_product v1 v2 < 0

-- Main statement incorporating the conditions and the conclusion
theorem maximum_obtuse_vectors :
  ∀ (v1 v2 v3 v4 : Vector3D),
  (obtuse_angle v1 v2) →
  (obtuse_angle v1 v3) →
  (obtuse_angle v1 v4) →
  (obtuse_angle v2 v3) →
  (obtuse_angle v2 v4) →
  (obtuse_angle v3 v4) →
  -- Conclusion: At most 4 vectors can be pairwise obtuse
  ∃ (v5 : Vector3D),
  ¬ (obtuse_angle v1 v5 ∧ obtuse_angle v2 v5 ∧ obtuse_angle v3 v5 ∧ obtuse_angle v4 v5) :=
sorry

end NUMINAMATH_GPT_maximum_obtuse_vectors_l1954_195499


namespace NUMINAMATH_GPT_snake_body_length_l1954_195441

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end NUMINAMATH_GPT_snake_body_length_l1954_195441


namespace NUMINAMATH_GPT_inhabitable_land_fraction_l1954_195416

theorem inhabitable_land_fraction (total_surface not_water_covered initially_inhabitable tech_advancement_viable : ℝ)
  (h1 : not_water_covered = 1 / 3 * total_surface)
  (h2 : initially_inhabitable = 1 / 3 * not_water_covered)
  (h3 : tech_advancement_viable = 1 / 2 * (not_water_covered - initially_inhabitable)) :
  (initially_inhabitable + tech_advancement_viable) / total_surface = 2 / 9 := 
sorry

end NUMINAMATH_GPT_inhabitable_land_fraction_l1954_195416


namespace NUMINAMATH_GPT_square_perimeter_l1954_195423

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end NUMINAMATH_GPT_square_perimeter_l1954_195423


namespace NUMINAMATH_GPT_selling_price_l1954_195494

def initial_cost : ℕ := 600
def food_cost_per_day : ℕ := 20
def number_of_days : ℕ := 40
def vaccination_and_deworming_cost : ℕ := 500
def profit : ℕ := 600

theorem selling_price (S : ℕ) :
  S = initial_cost + (food_cost_per_day * number_of_days) + vaccination_and_deworming_cost + profit :=
by
  sorry

end NUMINAMATH_GPT_selling_price_l1954_195494


namespace NUMINAMATH_GPT_avg_of_arithmetic_series_is_25_l1954_195491

noncomputable def arithmetic_series_avg : ℝ :=
  let a₁ := 15
  let d := 1 / 4
  let aₙ := 35
  let n := (aₙ - a₁) / d + 1
  let S := n * (a₁ + aₙ) / 2
  S / n

theorem avg_of_arithmetic_series_is_25 : arithmetic_series_avg = 25 := 
by
  -- Sorry, proof omitted due to instruction.
  sorry

end NUMINAMATH_GPT_avg_of_arithmetic_series_is_25_l1954_195491


namespace NUMINAMATH_GPT_students_not_taking_french_or_spanish_l1954_195476

theorem students_not_taking_french_or_spanish 
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (both_languages_students : ℕ) 
  (h_total_students : total_students = 28)
  (h_french_students : french_students = 5)
  (h_spanish_students : spanish_students = 10)
  (h_both_languages_students : both_languages_students = 4) :
  total_students - (french_students + spanish_students - both_languages_students) = 17 := 
by {
  -- Correct answer can be verified with the given conditions
  -- The proof itself is omitted (as instructed)
  sorry
}

end NUMINAMATH_GPT_students_not_taking_french_or_spanish_l1954_195476


namespace NUMINAMATH_GPT_minimum_value_condition_l1954_195496

theorem minimum_value_condition (a b : ℝ) (h : 16 * a ^ 2 + 2 * a + 8 * a * b + b ^ 2 - 1 = 0) : 
  ∃ m : ℝ, m = 3 * a + b ∧ m ≥ -1 :=
sorry

end NUMINAMATH_GPT_minimum_value_condition_l1954_195496


namespace NUMINAMATH_GPT_cone_prism_volume_ratio_l1954_195473

/--
Given:
- The base of the prism is a rectangle with side lengths 2r and 3r.
- The height of the prism is h.
- The base of the cone is a circle with radius r and height h.

Prove:
- The ratio of the volume of the cone to the volume of the prism is (π / 18).
-/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * Real.pi * r^2 * h) / (6 * r^2 * h) = Real.pi / 18 := by
  sorry

end NUMINAMATH_GPT_cone_prism_volume_ratio_l1954_195473


namespace NUMINAMATH_GPT_min_value_is_correct_l1954_195467

noncomputable def min_value (P : ℝ × ℝ) (A B C : ℝ × ℝ) : ℝ := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 +
  PB.1 * PC.1 + PB.2 * PC.2 +
  PC.1 * PA.1 + PC.2 * PA.2

theorem min_value_is_correct :
  ∃ P : ℝ × ℝ, P = (5/3, 1/3) ∧
  min_value P (1, 4) (4, 1) (0, -4) = -62/3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_is_correct_l1954_195467


namespace NUMINAMATH_GPT_probability_same_color_plates_l1954_195460

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_plates_l1954_195460


namespace NUMINAMATH_GPT_sin_cos_sum_eq_l1954_195400

theorem sin_cos_sum_eq (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan (θ + π / 4) = 1 / 2): 
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := 
  sorry

end NUMINAMATH_GPT_sin_cos_sum_eq_l1954_195400


namespace NUMINAMATH_GPT_mutually_exclusive_event_l1954_195482

def shooting_twice : Type := 
  { hit_first : Bool // hit_first = true ∨ hit_first = false }

def hitting_at_least_once (shoots : shooting_twice) : Prop :=
  shoots.1 ∨ (¬shoots.1 ∧ true)

def missing_both_times (shoots : shooting_twice) : Prop :=
  ¬shoots.1 ∧ (¬true ∨ true)

def mutually_exclusive (A : Prop) (B : Prop) : Prop :=
  A ∨ B → ¬ (A ∧ B)

theorem mutually_exclusive_event :
  ∀ shoots : shooting_twice, 
  mutually_exclusive (hitting_at_least_once shoots) (missing_both_times shoots) :=
by
  intro shoots
  unfold mutually_exclusive
  sorry

end NUMINAMATH_GPT_mutually_exclusive_event_l1954_195482


namespace NUMINAMATH_GPT_find_a_l1954_195490

-- Define the curve y = x^2 + x
def curve (x : ℝ) : ℝ := x^2 + x

-- Line equation ax - y + 1 = 0
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, line a x y → y = x^2 + x) ∧
  (deriv curve 1 = 2 * 1 + 1) →
  (2 * 1 + 1 = -1 / a) →
  a = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1954_195490


namespace NUMINAMATH_GPT_sum_a_b_l1954_195431

theorem sum_a_b (a b : ℚ) (h1 : a + 3 * b = 27) (h2 : 5 * a + 2 * b = 40) : a + b = 161 / 13 :=
  sorry

end NUMINAMATH_GPT_sum_a_b_l1954_195431


namespace NUMINAMATH_GPT_nine_point_circle_equation_l1954_195439

theorem nine_point_circle_equation 
  (α β γ : ℝ) 
  (x y z : ℝ) :
  (x^2 * (Real.sin α) * (Real.cos α) + y^2 * (Real.sin β) * (Real.cos β) + z^2 * (Real.sin γ) * (Real.cos γ) = 
  y * z * (Real.sin α) + x * z * (Real.sin β) + x * y * (Real.sin γ))
:= sorry

end NUMINAMATH_GPT_nine_point_circle_equation_l1954_195439


namespace NUMINAMATH_GPT_trig_eq_solutions_l1954_195469

open Real

theorem trig_eq_solutions (x : ℝ) :
  2 * sin x ^ 3 + 2 * sin x ^ 2 * cos x - sin x * cos x ^ 2 - cos x ^ 3 = 0 ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨ (∃ k : ℤ, x = arctan (sqrt 2 / 2) + k * π) ∨ (∃ m : ℤ, x = -arctan (sqrt 2 / 2) + m * π) :=
by
  sorry

end NUMINAMATH_GPT_trig_eq_solutions_l1954_195469


namespace NUMINAMATH_GPT_correct_train_process_l1954_195412

-- Define each step involved in the train process
inductive Step
| buy_ticket
| wait_for_train
| check_ticket
| board_train
| repair_train

open Step

-- Define each condition as a list of steps
def process_a : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]
def process_b : List Step := [wait_for_train, buy_ticket, board_train, check_ticket]
def process_c : List Step := [buy_ticket, wait_for_train, board_train, check_ticket]
def process_d : List Step := [repair_train, buy_ticket, check_ticket, board_train]

-- Define the correct process
def correct_process : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]

-- The theorem to prove that process A is the correct representation
theorem correct_train_process : process_a = correct_process :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_train_process_l1954_195412


namespace NUMINAMATH_GPT_simplify_fraction_l1954_195405

theorem simplify_fraction (x y z : ℝ) (hx : x = 5) (hz : z = 2) : (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1954_195405


namespace NUMINAMATH_GPT_lindas_initial_candies_l1954_195409

theorem lindas_initial_candies (candies_given : ℝ) (candies_left : ℝ) (initial_candies : ℝ) : 
  candies_given = 28 ∧ candies_left = 6 → initial_candies = candies_given + candies_left → initial_candies = 34 := 
by 
  sorry

end NUMINAMATH_GPT_lindas_initial_candies_l1954_195409


namespace NUMINAMATH_GPT_number_of_feasible_networks_10_l1954_195407

-- Definitions based on conditions
def feasible_networks (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 ^ (n - 1)

-- The proof problem statement
theorem number_of_feasible_networks_10 : feasible_networks 10 = 512 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_feasible_networks_10_l1954_195407


namespace NUMINAMATH_GPT_math_problem_l1954_195453

variable (x y : ℝ)

theorem math_problem (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by sorry

end NUMINAMATH_GPT_math_problem_l1954_195453


namespace NUMINAMATH_GPT_complex_series_sum_eq_zero_l1954_195446

open Complex

theorem complex_series_sum_eq_zero {ω : ℂ} (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_series_sum_eq_zero_l1954_195446


namespace NUMINAMATH_GPT_percentage_of_people_win_a_prize_l1954_195403

-- Define the constants used in the problem
def totalMinnows : Nat := 600
def minnowsPerPrize : Nat := 3
def totalPlayers : Nat := 800
def minnowsLeft : Nat := 240

-- Calculate the number of minnows given away as prizes
def minnowsGivenAway : Nat := totalMinnows - minnowsLeft

-- Calculate the number of prizes given away
def prizesGivenAway : Nat := minnowsGivenAway / minnowsPerPrize

-- Calculate the percentage of people winning a prize
def percentageWinners : Nat := (prizesGivenAway * 100) / totalPlayers

-- Theorem to prove the percentage of winners
theorem percentage_of_people_win_a_prize : 
    percentageWinners = 15 := 
sorry

end NUMINAMATH_GPT_percentage_of_people_win_a_prize_l1954_195403


namespace NUMINAMATH_GPT_find_angle_D_l1954_195461

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 40) (h4 : B + C = 130) : D = 40 := by
  sorry

end NUMINAMATH_GPT_find_angle_D_l1954_195461


namespace NUMINAMATH_GPT_functional_eq_solution_l1954_195401

noncomputable def f : ℚ → ℚ := sorry

theorem functional_eq_solution (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1):
  ∀ x : ℚ, f x = x + 1 :=
sorry

end NUMINAMATH_GPT_functional_eq_solution_l1954_195401


namespace NUMINAMATH_GPT_object_speed_l1954_195451

namespace problem

noncomputable def speed_in_miles_per_hour (distance_in_feet : ℕ) (time_in_seconds : ℕ) : ℝ :=
  let distance_in_miles := distance_in_feet / 5280
  let time_in_hours := time_in_seconds / 3600
  distance_in_miles / time_in_hours

theorem object_speed 
  (distance_in_feet : ℕ)
  (time_in_seconds : ℕ)
  (h : distance_in_feet = 80 ∧ time_in_seconds = 2) :
  speed_in_miles_per_hour distance_in_feet time_in_seconds = 27.27 :=
by
  sorry

end problem

end NUMINAMATH_GPT_object_speed_l1954_195451


namespace NUMINAMATH_GPT_product_remainder_mod_7_l1954_195497

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_product_remainder_mod_7_l1954_195497


namespace NUMINAMATH_GPT_cost_of_toys_l1954_195430

theorem cost_of_toys (x y : ℝ) (h1 : x + y = 40) (h2 : 90 / x = 150 / y) :
  x = 15 ∧ y = 25 :=
sorry

end NUMINAMATH_GPT_cost_of_toys_l1954_195430


namespace NUMINAMATH_GPT_red_shoes_drawn_l1954_195452

-- Define the main conditions
def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4
def probability_red : ℝ := 0.21428571428571427

-- Problem statement in Lean
theorem red_shoes_drawn (x : ℕ) (hx : ↑x / total_shoes = probability_red) : x = 2 := by
  sorry

end NUMINAMATH_GPT_red_shoes_drawn_l1954_195452
