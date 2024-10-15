import Mathlib

namespace NUMINAMATH_GPT_full_price_tickets_revenue_l849_84925

theorem full_price_tickets_revenue (f h d p : ℕ) 
  (h1 : f + h + d = 200) 
  (h2 : f * p + h * (p / 2) + d * (2 * p) = 5000) 
  (h3 : p = 50) : 
  f * p = 4500 :=
by
  sorry

end NUMINAMATH_GPT_full_price_tickets_revenue_l849_84925


namespace NUMINAMATH_GPT_remainder_division_l849_84938

theorem remainder_division (L S R : ℕ) (h1 : L - S = 1325) (h2 : L = 1650) (h3 : L = 5 * S + R) : 
  R = 25 :=
sorry

end NUMINAMATH_GPT_remainder_division_l849_84938


namespace NUMINAMATH_GPT_total_cost_shorts_tshirt_boots_shinguards_l849_84981

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end NUMINAMATH_GPT_total_cost_shorts_tshirt_boots_shinguards_l849_84981


namespace NUMINAMATH_GPT_lab_tech_ratio_l849_84941

theorem lab_tech_ratio (U T C : ℕ) (hU : U = 12) (hC : C = 6 * U) (hT : T = (C + U) / 14) :
  (T : ℚ) / U = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_lab_tech_ratio_l849_84941


namespace NUMINAMATH_GPT_shirt_price_after_discount_l849_84961

/-- Given a shirt with an initial cost price of $20 and a profit margin of 30%, 
    and a sale discount of 50%, prove that the final sale price of the shirt is $13. -/
theorem shirt_price_after_discount
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (discount : ℝ)
  (selling_price : ℝ)
  (final_price : ℝ)
  (h_cost : cost_price = 20)
  (h_profit_margin : profit_margin = 0.30)
  (h_discount : discount = 0.50)
  (h_selling_price : selling_price = cost_price + profit_margin * cost_price)
  (h_final_price : final_price = selling_price - discount * selling_price) :
  final_price = 13 := 
  sorry

end NUMINAMATH_GPT_shirt_price_after_discount_l849_84961


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l849_84989

-- Define the equation in terms of |x|
def equation (x : ℝ) : ℝ := |x|^3 - |x|^2 - 6 * |x| + 8

-- Lean statement to prove the sum and product of the roots
theorem sum_and_product_of_roots :
  (∀ x, equation x = 0 → (∃ L : List ℝ, L.sum = 0 ∧ L.prod = 16 ∧ ∀ y ∈ L, equation y = 0)) := 
sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l849_84989


namespace NUMINAMATH_GPT_farmer_land_area_l849_84942

theorem farmer_land_area
  (A : ℝ)
  (h1 : A / 3 + A / 4 + A / 5 + 26 = A) : A = 120 :=
sorry

end NUMINAMATH_GPT_farmer_land_area_l849_84942


namespace NUMINAMATH_GPT_proof_problem_l849_84930

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∀ x, (a * x^2 + b * x + 2 > 0) ↔ (x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ))) 

theorem proof_problem (a b : ℝ) (h : problem_statement a b) : a + b = -14 :=
sorry

end NUMINAMATH_GPT_proof_problem_l849_84930


namespace NUMINAMATH_GPT_elena_novel_pages_l849_84973

theorem elena_novel_pages
  (days_vacation : ℕ)
  (pages_first_two_days : ℕ)
  (pages_next_three_days : ℕ)
  (pages_last_day : ℕ)
  (h1 : days_vacation = 6)
  (h2 : pages_first_two_days = 2 * 42)
  (h3 : pages_next_three_days = 3 * 35)
  (h4 : pages_last_day = 15) :
  pages_first_two_days + pages_next_three_days + pages_last_day = 204 := by
  sorry

end NUMINAMATH_GPT_elena_novel_pages_l849_84973


namespace NUMINAMATH_GPT_animal_sighting_ratio_l849_84933

theorem animal_sighting_ratio
  (jan_sightings : ℕ)
  (feb_sightings : ℕ)
  (march_sightings : ℕ)
  (total_sightings : ℕ)
  (h1 : jan_sightings = 26)
  (h2 : feb_sightings = 3 * jan_sightings)
  (h3 : total_sightings = jan_sightings + feb_sightings + march_sightings)
  (h4 : total_sightings = 143) :
  (march_sightings : ℚ) / (feb_sightings : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_animal_sighting_ratio_l849_84933


namespace NUMINAMATH_GPT_total_money_raised_l849_84944

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end NUMINAMATH_GPT_total_money_raised_l849_84944


namespace NUMINAMATH_GPT_total_balloons_l849_84945

-- Define the number of balloons Alyssa, Sandy, and Sally have.
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Theorem stating that the total number of balloons is 104.
theorem total_balloons : alyssa_balloons + sandy_balloons + sally_balloons = 104 :=
by
  -- Proof is omitted for the purpose of this task.
  sorry

end NUMINAMATH_GPT_total_balloons_l849_84945


namespace NUMINAMATH_GPT_max_value_x_y_squared_l849_84962

theorem max_value_x_y_squared (x y : ℝ) (h : 3 * (x^3 + y^3) = x + y^2) : x + y^2 ≤ 1/3 :=
sorry

end NUMINAMATH_GPT_max_value_x_y_squared_l849_84962


namespace NUMINAMATH_GPT_problem_arith_seq_l849_84998

variables {a : ℕ → ℝ} (d : ℝ)
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arith_seq (h_arith : is_arithmetic_sequence a) 
  (h_condition : a 1 + a 6 + a 11 = 3) 
  : a 3 + a 9 = 2 :=
sorry

end NUMINAMATH_GPT_problem_arith_seq_l849_84998


namespace NUMINAMATH_GPT_initial_men_in_hostel_l849_84959

theorem initial_men_in_hostel (x : ℕ) (h1 : 36 * x = 45 * (x - 50)) : x = 250 := 
  sorry

end NUMINAMATH_GPT_initial_men_in_hostel_l849_84959


namespace NUMINAMATH_GPT_solve_equation_l849_84965

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x^2 + x + 1) / (x + 2) = x + 1 → x = -1 / 2 := 
by
  intro h1
  sorry

end NUMINAMATH_GPT_solve_equation_l849_84965


namespace NUMINAMATH_GPT_perpendicular_vectors_X_value_l849_84939

open Real

-- Define vectors a and b, and their perpendicularity condition
def vector_a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def vector_b : ℝ × ℝ := (1, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem statement
theorem perpendicular_vectors_X_value (x : ℝ) 
  (h : dot_product (vector_a x) vector_b = 0) : 
    x = -2 / 3 :=
by sorry

end NUMINAMATH_GPT_perpendicular_vectors_X_value_l849_84939


namespace NUMINAMATH_GPT_freds_change_l849_84946

theorem freds_change (ticket_cost : ℝ) (num_tickets : ℕ) (borrowed_movie_cost : ℝ) (total_paid : ℝ) 
  (h_ticket_cost : ticket_cost = 5.92) 
  (h_num_tickets : num_tickets = 2) 
  (h_borrowed_movie_cost : borrowed_movie_cost = 6.79) 
  (h_total_paid : total_paid = 20) : 
  total_paid - (num_tickets * ticket_cost + borrowed_movie_cost) = 1.37 := 
by 
  sorry

end NUMINAMATH_GPT_freds_change_l849_84946


namespace NUMINAMATH_GPT_cos_double_angle_l849_84913

theorem cos_double_angle (y0 : ℝ) (h : (1 / 3)^2 + y0^2 = 1) : 
  Real.cos (2 * Real.arccos (1 / 3)) = -7 / 9 := 
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l849_84913


namespace NUMINAMATH_GPT_size_relationship_l849_84983

noncomputable def a : ℝ := 1 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := 4

theorem size_relationship : a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_size_relationship_l849_84983


namespace NUMINAMATH_GPT_intersection_P_Q_l849_84928

def P : Set ℤ := { x | -4 ≤ x ∧ x ≤ 2 }

def Q : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem intersection_P_Q : P ∩ Q = {-2, -1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l849_84928


namespace NUMINAMATH_GPT_sum_eq_expected_l849_84915

noncomputable def complex_sum : Complex :=
  12 * Complex.exp (Complex.I * 3 * Real.pi / 13) + 12 * Complex.exp (Complex.I * 6 * Real.pi / 13)

noncomputable def expected_value : Complex :=
  24 * Real.cos (Real.pi / 13) * Complex.exp (Complex.I * 9 * Real.pi / 26)

theorem sum_eq_expected :
  complex_sum = expected_value :=
by
  sorry

end NUMINAMATH_GPT_sum_eq_expected_l849_84915


namespace NUMINAMATH_GPT_shaded_area_of_square_with_quarter_circles_l849_84929

theorem shaded_area_of_square_with_quarter_circles :
  let side_len : ℝ := 12
  let square_area := side_len * side_len
  let radius := side_len / 2
  let total_circle_area := 4 * (π * radius^2 / 4)
  let shaded_area := square_area - total_circle_area
  shaded_area = 144 - 36 * π := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_square_with_quarter_circles_l849_84929


namespace NUMINAMATH_GPT_determine_time_l849_84948

variable (g a V_0 V S t : ℝ)

def velocity_eq : Prop := V = (g + a) * t + V_0
def displacement_eq : Prop := S = 1 / 2 * (g + a) * t^2 + V_0 * t

theorem determine_time (h1 : velocity_eq g a V_0 V t) (h2 : displacement_eq g a V_0 S t) :
  t = 2 * S / (V + V_0) := 
sorry

end NUMINAMATH_GPT_determine_time_l849_84948


namespace NUMINAMATH_GPT_min_value_quadratic_l849_84977

theorem min_value_quadratic :
  ∃ (x y : ℝ), (∀ (a b : ℝ), (3*a^2 + 4*a*b + 2*b^2 - 6*a - 8*b + 6 ≥ 0)) ∧ 
  (3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6 = 0) := 
sorry

end NUMINAMATH_GPT_min_value_quadratic_l849_84977


namespace NUMINAMATH_GPT_definite_integral_l849_84910

open Real

theorem definite_integral : ∫ x in (0 : ℝ)..(π / 2), (x + sin x) = π^2 / 8 + 1 :=
by
  sorry

end NUMINAMATH_GPT_definite_integral_l849_84910


namespace NUMINAMATH_GPT_total_animals_sighted_l849_84990

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end NUMINAMATH_GPT_total_animals_sighted_l849_84990


namespace NUMINAMATH_GPT_squirrels_acorns_l849_84934

theorem squirrels_acorns (squirrels : ℕ) (total_collected : ℕ) (acorns_needed_per_squirrel : ℕ) (total_needed : ℕ) (acorns_still_needed : ℕ) : 
  squirrels = 5 → 
  total_collected = 575 → 
  acorns_needed_per_squirrel = 130 → 
  total_needed = squirrels * acorns_needed_per_squirrel →
  acorns_still_needed = total_needed - total_collected →
  acorns_still_needed / squirrels = 15 :=
by
  sorry

end NUMINAMATH_GPT_squirrels_acorns_l849_84934


namespace NUMINAMATH_GPT_vector_arithmetic_l849_84993

theorem vector_arithmetic (a b : ℝ × ℝ)
    (h₀ : a = (3, 5))
    (h₁ : b = (-2, 1)) :
    a - (2 : ℝ) • b = (7, 3) :=
sorry

end NUMINAMATH_GPT_vector_arithmetic_l849_84993


namespace NUMINAMATH_GPT_shampoo_duration_l849_84923

theorem shampoo_duration
  (rose_shampoo : ℚ := 1/3)
  (jasmine_shampoo : ℚ := 1/4)
  (daily_usage : ℚ := 1/12) :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := 
by
  sorry

end NUMINAMATH_GPT_shampoo_duration_l849_84923


namespace NUMINAMATH_GPT_apple_count_l849_84987

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end NUMINAMATH_GPT_apple_count_l849_84987


namespace NUMINAMATH_GPT_intersection_setA_setB_l849_84912

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 1) < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | (x - 2) / (x + 4) < 0 }

theorem intersection_setA_setB : 
  (setA ∩ setB) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_setA_setB_l849_84912


namespace NUMINAMATH_GPT_complex_quadrant_l849_84964

open Complex

theorem complex_quadrant 
  (z : ℂ) 
  (h : (1 - I) ^ 2 / z = 1 + I) :
  z = -1 - I :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l849_84964


namespace NUMINAMATH_GPT_compute_g3_l849_84991

def g (x : ℤ) : ℤ := 7 * x - 3

theorem compute_g3: g (g (g 3)) = 858 :=
by
  sorry

end NUMINAMATH_GPT_compute_g3_l849_84991


namespace NUMINAMATH_GPT_share_difference_l849_84952

theorem share_difference 
  (S : ℝ) -- Total sum of money
  (A B C D : ℝ) -- Shares of a, b, c, d respectively
  (h_proportion : A = 5 / 14 * S)
  (h_proportion : B = 2 / 14 * S)
  (h_proportion : C = 4 / 14 * S)
  (h_proportion : D = 3 / 14 * S)
  (h_d_share : D = 1500) :
  C - D = 500 :=
sorry

end NUMINAMATH_GPT_share_difference_l849_84952


namespace NUMINAMATH_GPT_sin_beta_value_l849_84904

theorem sin_beta_value (alpha beta : ℝ) (h1 : 0 < alpha) (h2 : alpha < beta) (h3 : beta < π / 2)
  (h4 : Real.sin alpha = 3 / 5) (h5 : Real.cos (alpha - beta) = 12 / 13) : Real.sin beta = 56 / 65 := by
  sorry

end NUMINAMATH_GPT_sin_beta_value_l849_84904


namespace NUMINAMATH_GPT_perfume_price_reduction_l849_84960

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end NUMINAMATH_GPT_perfume_price_reduction_l849_84960


namespace NUMINAMATH_GPT_rental_property_key_count_l849_84992

def number_of_keys (complexes apartments_per_complex keys_per_lock locks_per_apartment : ℕ) : ℕ :=
  complexes * apartments_per_complex * keys_per_lock * locks_per_apartment

theorem rental_property_key_count : 
  number_of_keys 2 12 3 1 = 72 := by
  sorry

end NUMINAMATH_GPT_rental_property_key_count_l849_84992


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l849_84919

-- Define a sequence as a list of real numbers
def seq : List ℚ := [8, -20, 50, -125]

-- Define the common ratio of a geometric sequence
def common_ratio (l : List ℚ) : ℚ := l.head! / l.tail!.head!

-- The theorem to prove the common ratio is -5/2
theorem geometric_sequence_common_ratio :
  common_ratio seq = -5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l849_84919


namespace NUMINAMATH_GPT_solution_of_abs_eq_l849_84916

theorem solution_of_abs_eq (x : ℝ) : |x - 5| = 3 * x + 6 → x = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_abs_eq_l849_84916


namespace NUMINAMATH_GPT_central_angle_of_sector_l849_84907

theorem central_angle_of_sector (r l θ : ℝ) 
  (h1 : 2 * r + l = 8) 
  (h2 : (1 / 2) * l * r = 4) 
  (h3 : θ = l / r) : θ = 2 := 
sorry

end NUMINAMATH_GPT_central_angle_of_sector_l849_84907


namespace NUMINAMATH_GPT_number_of_terms_before_4_appears_l849_84963

-- Define the parameters of the arithmetic sequence
def first_term : ℤ := 100
def common_difference : ℤ := -4
def nth_term (n : ℕ) : ℤ := first_term + common_difference * (n - 1)

-- Problem: Prove that the number of terms before the number 4 appears in this sequence is 24.
theorem number_of_terms_before_4_appears :
  ∃ n : ℕ, nth_term n = 4 ∧ n - 1 = 24 := 
by
  sorry

end NUMINAMATH_GPT_number_of_terms_before_4_appears_l849_84963


namespace NUMINAMATH_GPT_smallest_integer_condition_l849_84901

def is_not_prime (n : Nat) : Prop := ¬ Nat.Prime n

def is_not_square (n : Nat) : Prop :=
  ∀ m : Nat, m * m ≠ n

def has_no_prime_factor_less_than (n k : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p < k → ¬ (p ∣ n)

theorem smallest_integer_condition :
  ∃ n : Nat, n > 0 ∧ is_not_prime n ∧ is_not_square n ∧ has_no_prime_factor_less_than n 70 ∧ n = 5183 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_condition_l849_84901


namespace NUMINAMATH_GPT_kittens_count_l849_84927

def initial_kittens : ℕ := 8
def additional_kittens : ℕ := 2
def total_kittens : ℕ := 10

theorem kittens_count : initial_kittens + additional_kittens = total_kittens := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_kittens_count_l849_84927


namespace NUMINAMATH_GPT_number_square_roots_l849_84971

theorem number_square_roots (a x : ℤ) (h1 : x = (2 * a + 3) ^ 2) (h2 : x = (a - 18) ^ 2) : x = 169 :=
by 
  sorry

end NUMINAMATH_GPT_number_square_roots_l849_84971


namespace NUMINAMATH_GPT_sphere_volume_l849_84957

theorem sphere_volume (r : ℝ) (h1 : 4 * π * r^2 = 256 * π) : 
  (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l849_84957


namespace NUMINAMATH_GPT_value_of_x2_plus_9y2_l849_84996

theorem value_of_x2_plus_9y2 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -15) : x^2 + 9 * y^2 = 171 :=
sorry

end NUMINAMATH_GPT_value_of_x2_plus_9y2_l849_84996


namespace NUMINAMATH_GPT_find_floors_l849_84986

theorem find_floors (a b : ℕ) 
  (h1 : 3 * a + 4 * b = 25)
  (h2 : 2 * a + 3 * b = 18) : 
  a = 3 ∧ b = 4 := 
sorry

end NUMINAMATH_GPT_find_floors_l849_84986


namespace NUMINAMATH_GPT_solve_n_m_equation_l849_84924

theorem solve_n_m_equation : 
  ∃ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ∧ ((n, m) = (3, 5) ∨ (n, m) = (3, -5) ∨ (n, m) = (-3, 5) ∨ (n, m) = (-3, -5)) :=
by { sorry }

end NUMINAMATH_GPT_solve_n_m_equation_l849_84924


namespace NUMINAMATH_GPT_leah_total_coin_value_l849_84970

variable (p n : ℕ) -- Let p be the number of pennies and n be the number of nickels

-- Leah has 15 coins consisting of pennies and nickels
axiom coin_count : p + n = 15

-- If she had three more nickels, she would have twice as many pennies as nickels
axiom conditional_equation : p = 2 * (n + 3)

-- We want to prove that the total value of Leah's coins in cents is 27
theorem leah_total_coin_value : 5 * n + p = 27 := by
  sorry

end NUMINAMATH_GPT_leah_total_coin_value_l849_84970


namespace NUMINAMATH_GPT_spending_ratio_l849_84911

theorem spending_ratio 
  (lisa_tshirts : Real)
  (lisa_jeans : Real)
  (lisa_coats : Real)
  (carly_tshirts : Real)
  (carly_jeans : Real)
  (carly_coats : Real)
  (total_spent : Real)
  (hl1 : lisa_tshirts = 40)
  (hl2 : lisa_jeans = lisa_tshirts / 2)
  (hl3 : lisa_coats = 2 * lisa_tshirts)
  (hc1 : carly_tshirts = lisa_tshirts / 4)
  (hc2 : carly_coats = lisa_coats / 4)
  (htotal : total_spent = lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats)
  (h_total_spent_val : total_spent = 230) :
  carly_jeans = 3 * lisa_jeans :=
by
  -- Placeholder for theorem's proof
  sorry

end NUMINAMATH_GPT_spending_ratio_l849_84911


namespace NUMINAMATH_GPT_chocolate_bars_in_large_box_l849_84997

theorem chocolate_bars_in_large_box : 
  let small_boxes := 19 
  let bars_per_small_box := 25 
  let total_bars := small_boxes * bars_per_small_box 
  total_bars = 475 := by 
  -- declarations and assumptions
  let small_boxes : ℕ := 19 
  let bars_per_small_box : ℕ := 25 
  let total_bars : ℕ := small_boxes * bars_per_small_box 
  sorry

end NUMINAMATH_GPT_chocolate_bars_in_large_box_l849_84997


namespace NUMINAMATH_GPT_smallest_lcm_of_4digit_multiples_of_5_l849_84975

theorem smallest_lcm_of_4digit_multiples_of_5 :
  ∃ m n : ℕ, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (1000 ≤ n) ∧ (n ≤ 9999) ∧ (Nat.gcd m n = 5) ∧ (Nat.lcm m n = 201000) := 
sorry

end NUMINAMATH_GPT_smallest_lcm_of_4digit_multiples_of_5_l849_84975


namespace NUMINAMATH_GPT_range_of_a_l849_84921

-- Define the problem statement in Lean 4
theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((x^2 - (a-1)*x + 1) > 0)) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_range_of_a_l849_84921


namespace NUMINAMATH_GPT_negation_cube_of_every_odd_is_odd_l849_84905

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def cube (n : ℤ) : ℤ := n * n * n

def cube_of_odd_is_odd (n : ℤ) : Prop := odd n → odd (cube n)

theorem negation_cube_of_every_odd_is_odd :
  ¬ (∀ n : ℤ, odd n → odd (cube n)) ↔ ∃ n : ℤ, odd n ∧ ¬ odd (cube n) :=
sorry

end NUMINAMATH_GPT_negation_cube_of_every_odd_is_odd_l849_84905


namespace NUMINAMATH_GPT_seashells_at_end_of_month_l849_84954

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end NUMINAMATH_GPT_seashells_at_end_of_month_l849_84954


namespace NUMINAMATH_GPT_range_of_m_l849_84914

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Define the interval
def interval (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 2

-- Prove the range of m
theorem range_of_m (m : ℝ) : (∀ x : ℝ, interval x → f x > 2 * x + m) ↔ m < - 5 / 4 :=
by
  -- This is the theorem statement, hence the proof starts here
  sorry

end NUMINAMATH_GPT_range_of_m_l849_84914


namespace NUMINAMATH_GPT_solution_system_equations_l849_84974

theorem solution_system_equations :
  ∀ (x y : ℝ) (k n : ℤ),
    (4 * (Real.cos x) ^ 2 - 4 * Real.cos x * (Real.cos (6 * x)) ^ 2 + (Real.cos (6 * x)) ^ 2 = 0) ∧
    (Real.sin x = Real.cos y) →
    (∃ k n : ℤ, (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = (Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = -(Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = (5 * Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = -(5 * Real.pi / 6) + 2 * Real.pi * n)) :=
by
  sorry

end NUMINAMATH_GPT_solution_system_equations_l849_84974


namespace NUMINAMATH_GPT_correct_answer_l849_84958

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem correct_answer : P ∩ Q ⊆ P := by
  sorry

end NUMINAMATH_GPT_correct_answer_l849_84958


namespace NUMINAMATH_GPT_athletes_camp_duration_l849_84917

theorem athletes_camp_duration
  (h : ℕ)
  (initial_athletes : ℕ := 300)
  (rate_leaving : ℕ := 28)
  (rate_entering : ℕ := 15)
  (hours_entering : ℕ := 7)
  (difference : ℕ := 7) :
  300 - 28 * h + 15 * 7 = 300 + 7 → h = 4 :=
by
  sorry

end NUMINAMATH_GPT_athletes_camp_duration_l849_84917


namespace NUMINAMATH_GPT_words_per_page_l849_84940

theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p % 221 = 207) : p = 100 :=
sorry

end NUMINAMATH_GPT_words_per_page_l849_84940


namespace NUMINAMATH_GPT_original_polynomial_l849_84937

theorem original_polynomial {x y : ℝ} (P : ℝ) :
  P - (-x^2 * y) = 3 * x^2 * y - 2 * x * y - 1 → P = 2 * x^2 * y - 2 * x * y - 1 :=
sorry

end NUMINAMATH_GPT_original_polynomial_l849_84937


namespace NUMINAMATH_GPT_sum_of_two_numbers_l849_84972

theorem sum_of_two_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y = 12) (h2 : 1 / x = 3 * (1 / y)) : x + y = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l849_84972


namespace NUMINAMATH_GPT_smallest_integer_a_l849_84968

theorem smallest_integer_a (a : ℤ) (b : ℤ) (h1 : a < 21) (h2 : 20 ≤ b) (h3 : b < 31) (h4 : (a : ℝ) / b < 2 / 3) : 13 < a :=
sorry

end NUMINAMATH_GPT_smallest_integer_a_l849_84968


namespace NUMINAMATH_GPT_michelle_gas_left_l849_84982

def gasLeft (initialGas: ℝ) (usedGas: ℝ) : ℝ :=
  initialGas - usedGas

theorem michelle_gas_left :
  gasLeft 0.5 0.3333333333333333 = 0.1666666666666667 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_michelle_gas_left_l849_84982


namespace NUMINAMATH_GPT_trajectory_eq_of_moving_point_Q_l849_84900

-- Define the conditions and the correct answer
theorem trajectory_eq_of_moving_point_Q 
(a b : ℝ) (h : a > b) (h_pos : b > 0)
(P Q : ℝ × ℝ)
(h_ellipse : (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1)
(h_Q : Q = (P.1 * 2, P.2 * 2)) :
  (Q.1^2) / (4 * a^2) + (Q.2^2) / (4 * b^2) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_trajectory_eq_of_moving_point_Q_l849_84900


namespace NUMINAMATH_GPT_find_angle_A_find_perimeter_l849_84926

-- Given problem conditions as Lean definitions
def triangle_sides (a b c : ℝ) : Prop :=
  ∃ B : ℝ, c = a * (Real.cos B + Real.sqrt 3 * Real.sin B)

def triangle_area (S a : ℝ) : Prop :=
  S = Real.sqrt 3 / 4 ∧ a = 1

-- Prove angle A
theorem find_angle_A (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ A : ℝ, A = Real.pi / 6 := 
sorry

-- Prove perimeter
theorem find_perimeter (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ P : ℝ, P = Real.sqrt 3 + 2 := 
sorry

end NUMINAMATH_GPT_find_angle_A_find_perimeter_l849_84926


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l849_84931

theorem largest_angle_in_triangle (A B C : ℝ) 
  (h_sum : A + B = 126) 
  (h_diff : B = A + 40) 
  (h_triangle : A + B + C = 180) : max A (max B C) = 83 := 
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l849_84931


namespace NUMINAMATH_GPT_find_smallest_angle_l849_84966

theorem find_smallest_angle (x : ℝ) (h1 : Real.tan (2 * x) + Real.tan (3 * x) = 1) :
  x = 9 * Real.pi / 180 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_angle_l849_84966


namespace NUMINAMATH_GPT_car_X_travel_distance_l849_84935

def car_distance_problem (speed_X speed_Y : ℝ) (delay : ℝ) : ℝ :=
  let t := 7 -- duration in hours computed in the provided solution
  speed_X * t

theorem car_X_travel_distance
  (speed_X speed_Y : ℝ) (delay : ℝ)
  (h_speed_X : speed_X = 35) (h_speed_Y : speed_Y = 39) (h_delay : delay = 48 / 60) :
  car_distance_problem speed_X speed_Y delay = 245 :=
by
  rw [h_speed_X, h_speed_Y, h_delay]
  -- compute the given car distance problem using the values provided
  sorry

end NUMINAMATH_GPT_car_X_travel_distance_l849_84935


namespace NUMINAMATH_GPT_exists_triangle_free_not_4_colorable_l849_84908

/-- Define a graph as a structure with vertices and edges. -/
structure Graph (V : Type*) :=
  (adj : V → V → Prop)
  (symm : ∀ x y, adj x y → adj y x)
  (irreflexive : ∀ x, ¬adj x x)

/-- A definition of triangle-free graph. -/
def triangle_free {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c : V), G.adj a b → G.adj b c → G.adj c a → false

/-- A definition that a graph cannot be k-colored. -/
def not_k_colorable {V : Type*} (G : Graph V) (k : ℕ) : Prop :=
  ¬∃ (f : V → ℕ), (∀ (v : V), f v < k) ∧ (∀ (v w : V), G.adj v w → f v ≠ f w)

/-- There exists a triangle-free graph that is not 4-colorable. -/
theorem exists_triangle_free_not_4_colorable : ∃ (V : Type*) (G : Graph V), triangle_free G ∧ not_k_colorable G 4 := 
sorry

end NUMINAMATH_GPT_exists_triangle_free_not_4_colorable_l849_84908


namespace NUMINAMATH_GPT_find_a_l849_84949

noncomputable def quadratic_inequality_solution (a b : ℝ) : Prop :=
  a * ((-1/2) * (1/3)) * 20 = 20 ∧
  a < 0 ∧
  (-b / (2 * a)) = (-1 / 2 + 1 / 3)

theorem find_a (a b : ℝ) (h : quadratic_inequality_solution a b) : a = -12 :=
  sorry

end NUMINAMATH_GPT_find_a_l849_84949


namespace NUMINAMATH_GPT_fraction_bounds_l849_84984

theorem fraction_bounds (n : ℕ) (h : 0 < n) : (1 : ℚ) / 2 ≤ n / (n + 1 : ℚ) ∧ n / (n + 1 : ℚ) < 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_bounds_l849_84984


namespace NUMINAMATH_GPT_determinant_in_terms_of_roots_l849_84950

noncomputable def determinant_3x3 (a b c : ℝ) : ℝ :=
  (1 + a) * ((1 + b) * (1 + c) - 1) - 1 * (1 + c) + (1 + b) * 1

theorem determinant_in_terms_of_roots (a b c s p q : ℝ)
  (h1 : a + b + c = -s)
  (h2 : a * b + a * c + b * c = p)
  (h3 : a * b * c = -q) :
  determinant_3x3 a b c = -q + p - s :=
by
  sorry

end NUMINAMATH_GPT_determinant_in_terms_of_roots_l849_84950


namespace NUMINAMATH_GPT_find_prices_maximize_profit_l849_84922

-- Definition of conditions
def sales_eq1 (m n : ℝ) : Prop := 150 * m + 100 * n = 1450
def sales_eq2 (m n : ℝ) : Prop := 200 * m + 50 * n = 1100

def profit_function (x : ℕ) : ℝ := -2 * x + 1500
def range_x (x : ℕ) : Prop := 375 ≤ x ∧ x ≤ 500

-- Theorem to prove the unit prices
theorem find_prices : ∃ m n : ℝ, sales_eq1 m n ∧ sales_eq2 m n ∧ m = 3 ∧ n = 10 := 
sorry

-- Theorem to prove the profit function and maximum profit
theorem maximize_profit : ∃ (x : ℕ) (W : ℝ), range_x x ∧ W = profit_function x ∧ W = 750 :=
sorry

end NUMINAMATH_GPT_find_prices_maximize_profit_l849_84922


namespace NUMINAMATH_GPT_share_of_B_l849_84953

noncomputable def B_share (B_investment A_investment C_investment D_investment total_profit : ℝ) : ℝ :=
  (B_investment / (A_investment + B_investment + C_investment + D_investment)) * total_profit

theorem share_of_B (B_investment total_profit : ℝ) (hA : A_investment = 3 * B_investment) 
  (hC : C_investment = (3 / 2) * B_investment) 
  (hD : D_investment = (3 / 2) * B_investment) 
  (h_profit : total_profit = 19900) :
  B_share B_investment A_investment C_investment D_investment total_profit = 2842.86 :=
by
  rw [B_share, hA, hC, hD, h_profit]
  sorry

end NUMINAMATH_GPT_share_of_B_l849_84953


namespace NUMINAMATH_GPT_least_not_lucky_multiple_of_6_l849_84985

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end NUMINAMATH_GPT_least_not_lucky_multiple_of_6_l849_84985


namespace NUMINAMATH_GPT_minimum_of_a_plus_b_l849_84932

theorem minimum_of_a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) : a + b ≥ 9 :=
by sorry

end NUMINAMATH_GPT_minimum_of_a_plus_b_l849_84932


namespace NUMINAMATH_GPT_max_playground_area_l849_84994

/-- Mara is setting up a fence around a rectangular playground with given constraints.
    We aim to prove that the maximum area the fence can enclose is 10000 square feet. --/
theorem max_playground_area (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 400) 
  (h2 : l ≥ 100) 
  (h3 : w ≥ 50) : 
  l * w ≤ 10000 :=
sorry

end NUMINAMATH_GPT_max_playground_area_l849_84994


namespace NUMINAMATH_GPT_condition_I_condition_II_l849_84909

noncomputable def f (x a : ℝ) : ℝ := |x - a|

-- Condition (I) proof problem
theorem condition_I (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a ≥ 4 - |x - 1| ↔ (x ≤ -1 ∨ x ≥ 3) :=
by sorry

-- Condition (II) proof problem
theorem condition_II (a : ℝ) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_f : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2)
    (h_eq : 1/m + 1/(2*n) = a) : mn ≥ 2 :=
by sorry

end NUMINAMATH_GPT_condition_I_condition_II_l849_84909


namespace NUMINAMATH_GPT_expand_polynomial_expression_l849_84988

theorem expand_polynomial_expression (x : ℝ) : 
  (x + 6) * (x + 8) * (x - 3) = x^3 + 11 * x^2 + 6 * x - 144 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_expression_l849_84988


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l849_84969

theorem neither_sufficient_nor_necessary_condition
  (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0)
  (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) ↔
  ¬(∀ x, a1 * x^2 + b1 * x + c1 > 0 ↔ a2 * x^2 + b2 * x + c2 > 0) :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l849_84969


namespace NUMINAMATH_GPT_fifth_term_arithmetic_seq_l849_84920

theorem fifth_term_arithmetic_seq (a d : ℤ) 
  (h10th : a + 9 * d = 23) 
  (h11th : a + 10 * d = 26) 
  : a + 4 * d = 8 :=
sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_seq_l849_84920


namespace NUMINAMATH_GPT_inequality_proof_l849_84936

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  (1 / (a^2 + 1)) + (1 / (b^2 + 1)) + (1 / (c^2 + 1)) ≤ 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l849_84936


namespace NUMINAMATH_GPT_house_assignment_l849_84918

theorem house_assignment (n : ℕ) (assign : Fin n → Fin n) (pref : Fin n → Fin n → Fin n → Prop) :
  (∀ (p : Fin n), ∃ (better_assign : Fin n → Fin n),
    (∃ q, pref p (assign p) (better_assign p) ∧ pref q (assign q) (better_assign p) ∧ better_assign q ≠ assign q)
  ) → (∃ p, pref p (assign p) (assign p))
:= sorry

end NUMINAMATH_GPT_house_assignment_l849_84918


namespace NUMINAMATH_GPT_smallest_positive_integer_exists_l849_84947

theorem smallest_positive_integer_exists :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k m : ℕ), n = 5 * k + 3 ∧ n = 12 * m) ∧ n = 48 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_exists_l849_84947


namespace NUMINAMATH_GPT_original_square_side_length_l849_84967

theorem original_square_side_length (a : ℕ) (initial_thickness final_thickness : ℕ) (side_length_reduction_factor thickness_doubling_factor : ℕ) (s : ℕ) :
  a = 3 →
  final_thickness = 16 →
  initial_thickness = 1 →
  side_length_reduction_factor = 16 →
  thickness_doubling_factor = 16 →
  s * s = side_length_reduction_factor * a * a →
  s = 12 :=
by
  intros ha hfinal_thickness hin_initial_thickness hside_length_reduction_factor hthickness_doubling_factor h_area_equiv
  sorry

end NUMINAMATH_GPT_original_square_side_length_l849_84967


namespace NUMINAMATH_GPT_intersection_points_l849_84976

noncomputable def y1 := 2*((7 + Real.sqrt 61)/2)^2 - 3*((7 + Real.sqrt 61)/2) + 1
noncomputable def y2 := 2*((7 - Real.sqrt 61)/2)^2 - 3*((7 - Real.sqrt 61)/2) + 1

theorem intersection_points :
  ∃ (x y : ℝ), (y = 2*x^2 - 3*x + 1) ∧ (y = x^2 + 4*x + 4) ∧
                ((x = (7 + Real.sqrt 61)/2 ∧ y = y1) ∨
                 (x = (7 - Real.sqrt 61)/2 ∧ y = y2)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l849_84976


namespace NUMINAMATH_GPT_cost_of_second_batch_l849_84995

theorem cost_of_second_batch
  (C_1 C_2 : ℕ)
  (quantity_ratio cost_increase: ℕ) 
  (H1 : C_1 = 3000) 
  (H2 : C_2 = 9600) 
  (H3 : quantity_ratio = 3) 
  (H4 : cost_increase = 1)
  : (∃ x : ℕ, C_1 / x = C_2 / (x + cost_increase) / quantity_ratio) ∧ 
    (C_2 / (C_1 / 15 + cost_increase) / 3 = 16) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_second_batch_l849_84995


namespace NUMINAMATH_GPT_no_intersection_points_l849_84902

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

-- State the theorem
theorem no_intersection_points : ∀ x y : ℝ, f1 x = y ∧ f2 x = y → false := by
  sorry

end NUMINAMATH_GPT_no_intersection_points_l849_84902


namespace NUMINAMATH_GPT_least_non_lucky_multiple_of_7_correct_l849_84951

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

def least_non_lucky_multiple_of_7 : ℕ :=
  14

theorem least_non_lucky_multiple_of_7_correct : 
  ¬ is_lucky 14 ∧ ∀ m, m < 14 → m % 7 = 0 → ¬ ¬ is_lucky m :=
by
  sorry

end NUMINAMATH_GPT_least_non_lucky_multiple_of_7_correct_l849_84951


namespace NUMINAMATH_GPT_abc_sum_l849_84979

def f (x : Int) (a b c : Nat) : Int :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem abc_sum :
  ∃ a b c : Nat, 
  f 3 a b c = 7 ∧ 
  f 0 a b c = 6 ∧ 
  f (-3) a b c = -15 ∧ 
  a + b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_l849_84979


namespace NUMINAMATH_GPT_triangle_leg_length_l849_84980

theorem triangle_leg_length (perimeter_square : ℝ)
                            (base_triangle : ℝ)
                            (area_equality : ∃ (side_square : ℝ) (height_triangle : ℝ),
                                4 * side_square = perimeter_square ∧
                                side_square * side_square = (1/2) * base_triangle * height_triangle)
                            : ∃ (y : ℝ), y = 22.5 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_triangle_leg_length_l849_84980


namespace NUMINAMATH_GPT_problem_1_problem_2_l849_84956

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem problem_1 (x : ℝ) : (g x ≥ abs (x - 1)) ↔ (x ≥ 2/3) :=
by
  sorry

theorem problem_2 (c : ℝ) : (∀ x, abs (g x) - c ≥ abs (x - 1)) → (c ≤ -1/2) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l849_84956


namespace NUMINAMATH_GPT_percent_decrease_l849_84978

theorem percent_decrease (original_price sale_price : ℝ) 
  (h_original: original_price = 100) 
  (h_sale: sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_l849_84978


namespace NUMINAMATH_GPT_expected_worth_of_coin_flip_l849_84943

theorem expected_worth_of_coin_flip :
  let p_heads := 2 / 3
  let p_tails := 1 / 3
  let gain_heads := 5
  let loss_tails := -9
  (p_heads * gain_heads) + (p_tails * loss_tails) = 1 / 3 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_expected_worth_of_coin_flip_l849_84943


namespace NUMINAMATH_GPT_combined_payment_is_correct_l849_84906

-- Define the conditions for discounts
def discount_scheme (amount : ℕ) : ℕ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then (amount * 90) / 100
  else (amount * 80) / 100

-- Given conditions for Wang Bo's purchases
def first_purchase := 80
def second_purchase_with_discount_applied := 252

-- Two possible original amounts for the second purchase
def possible_second_purchases : Set ℕ :=
  { x | discount_scheme x = second_purchase_with_discount_applied }

-- Total amount to be considered for combined buys with discounts
def total_amount_paid := {x + first_purchase | x ∈ possible_second_purchases}

-- discount applied on the combined amount
def discount_applied_amount (combined : ℕ) : ℕ :=
  discount_scheme combined

-- Prove the combined amount is either 288 or 316
theorem combined_payment_is_correct :
  ∃ combined ∈ total_amount_paid, discount_applied_amount combined = 288 ∨ discount_applied_amount combined = 316 :=
sorry

end NUMINAMATH_GPT_combined_payment_is_correct_l849_84906


namespace NUMINAMATH_GPT_digits_right_of_decimal_l849_84999

theorem digits_right_of_decimal : 
  ∃ n : ℕ, (3^6 : ℚ) / ((6^4 : ℚ) * 625) = 9 * 10^(-4 : ℤ) ∧ n = 4 := 
by 
  sorry

end NUMINAMATH_GPT_digits_right_of_decimal_l849_84999


namespace NUMINAMATH_GPT_geometric_sequence_sum_l849_84903

theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : q ≠ 1)
    (hS2 : (a₁ * (1 - q^2)) / (1 - q) = 1)
    (hS4 : (a₁ * (1 - q^4)) / (1 - q) = 3) :
    (a₁ * (1 - q^8)) / (1 - q) = 15 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l849_84903


namespace NUMINAMATH_GPT_tim_income_percentage_less_than_juan_l849_84955

variables (M T J : ℝ)

theorem tim_income_percentage_less_than_juan 
  (h1 : M = 1.60 * T)
  (h2 : M = 0.80 * J) : 
  100 - 100 * (T / J) = 50 :=
by
  sorry

end NUMINAMATH_GPT_tim_income_percentage_less_than_juan_l849_84955
