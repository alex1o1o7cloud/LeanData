import Mathlib

namespace chips_recoloring_impossible_l864_86401

theorem chips_recoloring_impossible :
  (∀ a b c : ℕ, a = 2008 ∧ b = 2009 ∧ c = 2010 →
   ¬(∃ k : ℕ, a + b + c = k ∧ (a = k ∨ b = k ∨ c = k))) :=
by sorry

end chips_recoloring_impossible_l864_86401


namespace grain_demand_l864_86433

variable (F : ℝ)
def S0 : ℝ := 1800000 -- base supply value

theorem grain_demand : ∃ D : ℝ, S = 0.75 * D ∧ S = S0 * (1 + F) ∧ D = (1800000 * (1 + F) / 0.75) :=
by
  sorry

end grain_demand_l864_86433


namespace ammeter_sum_l864_86457

variable (A1 A2 A3 A4 A5 : ℝ)
variable (I2 : ℝ)
variable (h1 : I2 = 4)
variable (h2 : A1 = I2)
variable (h3 : A3 = 2 * A1)
variable (h4 : A5 = A3 + A1)
variable (h5 : A4 = (5 / 3) * A5)

theorem ammeter_sum (A1 A2 A3 A4 A5 I2 : ℝ) (h1 : I2 = 4) (h2 : A1 = I2) (h3 : A3 = 2 * A1)
                   (h4 : A5 = A3 + A1) (h5 : A4 = (5 / 3) * A5) :
  A1 + I2 + A3 + A4 + A5 = 48 := 
sorry

end ammeter_sum_l864_86457


namespace toms_age_l864_86422

variable (T J : ℕ)

theorem toms_age :
  (J - 6 = 3 * (T - 6)) ∧ (J + 4 = 2 * (T + 4)) → T = 16 :=
by
  intros h
  sorry

end toms_age_l864_86422


namespace lamp_probability_l864_86474

theorem lamp_probability (rope_length : ℝ) (pole_distance : ℝ) (h_pole_distance : pole_distance = 8) :
  let lamp_range := 2
  let favorable_segment_length := 4
  let total_rope_length := rope_length
  let probability := (favorable_segment_length / total_rope_length)
  rope_length = 8 → probability = 1 / 2 :=
by
  intros
  sorry

end lamp_probability_l864_86474


namespace original_triangle_area_l864_86490

theorem original_triangle_area (new_area : ℝ) (scaling_factor : ℝ) (area_ratio : ℝ) : 
  new_area = 32 → scaling_factor = 2 → 
  area_ratio = scaling_factor ^ 2 → 
  new_area / area_ratio = 8 := 
by
  intros
  -- insert your proof logic here
  sorry

end original_triangle_area_l864_86490


namespace xy_max_value_l864_86404

theorem xy_max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 12) :
  xy <= 9 := by
  sorry

end xy_max_value_l864_86404


namespace scale_reading_l864_86423

theorem scale_reading (a b c : ℝ) (h₁ : 10.15 < a ∧ a < 10.4) (h₂ : 10.275 = (10.15 + 10.4) / 2) : a = 10.3 := 
by 
  sorry

end scale_reading_l864_86423


namespace intersect_at_one_point_l864_86486

-- Define the equations as given in the conditions
def equation1 (b : ℝ) (x : ℝ) : ℝ := b * x ^ 2 + 2 * x + 2
def equation2 (x : ℝ) : ℝ := -2 * x - 2

-- Statement of the theorem
theorem intersect_at_one_point (b : ℝ) :
  (∀ x : ℝ, equation1 b x = equation2 x → x = 1) ↔ b = 1 := sorry

end intersect_at_one_point_l864_86486


namespace candidate_fails_by_50_marks_l864_86483

theorem candidate_fails_by_50_marks (T : ℝ) (pass_mark : ℝ) (h1 : pass_mark = 199.99999999999997)
    (h2 : 0.45 * T - 25 = 199.99999999999997) :
    199.99999999999997 - 0.30 * T = 50 :=
by
  sorry

end candidate_fails_by_50_marks_l864_86483


namespace range_of_m_l864_86410

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ (m ≤ 2) :=
by
  sorry

end range_of_m_l864_86410


namespace find_n_l864_86418

def binomial_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + b) ^ n

def expanded_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + 3 * b) ^ n

theorem find_n (n : ℕ) :
  (expanded_coefficient_sum n 1 1) / (binomial_coefficient_sum n 1 1) = 64 → n = 6 :=
by 
  sorry

end find_n_l864_86418


namespace total_possible_arrangements_l864_86461

-- Define the subjects
inductive Subject : Type
| PoliticalScience
| Chinese
| Mathematics
| English
| PhysicalEducation
| Physics

open Subject

-- Define the condition that the first period cannot be Chinese
def first_period_cannot_be_chinese (schedule : Fin 6 → Subject) : Prop :=
  schedule 0 ≠ Chinese

-- Define the condition that the fifth period cannot be English
def fifth_period_cannot_be_english (schedule : Fin 6 → Subject) : Prop :=
  schedule 4 ≠ English

-- Define the schedule includes six unique subjects
def schedule_includes_all_subjects (schedule : Fin 6 → Subject) : Prop :=
  ∀ s : Subject, ∃ i : Fin 6, schedule i = s

-- Define the main theorem to prove the total number of possible arrangements
theorem total_possible_arrangements : 
  ∃ (schedules : List (Fin 6 → Subject)), 
  (∀ schedule, schedule ∈ schedules → 
    first_period_cannot_be_chinese schedule ∧ 
    fifth_period_cannot_be_english schedule ∧ 
    schedule_includes_all_subjects schedule) ∧ 
  schedules.length = 600 :=
sorry

end total_possible_arrangements_l864_86461


namespace James_future_age_when_Thomas_reaches_James_current_age_l864_86468

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end James_future_age_when_Thomas_reaches_James_current_age_l864_86468


namespace number_of_terms_in_arithmetic_sequence_l864_86438

theorem number_of_terms_in_arithmetic_sequence : 
  ∀ (a d l : ℕ), a = 20 → d = 5 → l = 150 → 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 27 :=
by
  intros a d l ha hd hl
  use 27
  rw [ha, hd, hl]
  sorry

end number_of_terms_in_arithmetic_sequence_l864_86438


namespace right_triangle_angle_ratio_l864_86463

theorem right_triangle_angle_ratio
  (a b : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) 
  (h : a / b = 5 / 4)
  (h3 : a + b = 90) :
  (a = 50) ∧ (b = 40) :=
by
  sorry

end right_triangle_angle_ratio_l864_86463


namespace second_customer_payment_l864_86478

def price_of_headphones : ℕ := 30
def total_cost_first_customer (P H : ℕ) : ℕ := 5 * P + 8 * H
def total_cost_second_customer (P H : ℕ) : ℕ := 3 * P + 4 * H

theorem second_customer_payment
  (P : ℕ)
  (H_eq : H = price_of_headphones)
  (first_customer_eq : total_cost_first_customer P H = 840) :
  total_cost_second_customer P H = 480 :=
by
  -- Proof to be filled in later
  sorry

end second_customer_payment_l864_86478


namespace find_values_of_a_and_b_l864_86466

theorem find_values_of_a_and_b
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (x : ℝ) (hx : x > 1)
  (h : 9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17)
  (h2 : (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2) :
  a = 10 ^ Real.sqrt 2 ∧ b = 10 := by
sorry

end find_values_of_a_and_b_l864_86466


namespace find_k_inverse_proportion_l864_86411

theorem find_k_inverse_proportion :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) → (y = k / x)) ∧ k = 3 :=
by
  sorry

end find_k_inverse_proportion_l864_86411


namespace sin_eq_cos_example_l864_86403

theorem sin_eq_cos_example 
  (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180)
  (h_eq : Real.sin (n * Real.pi / 180) = Real.cos (682 * Real.pi / 180)) :
  n = 128 :=
sorry

end sin_eq_cos_example_l864_86403


namespace fruit_seller_price_l864_86496

theorem fruit_seller_price 
  (CP SP SP_profit : ℝ)
  (h1 : SP = CP * 0.88)
  (h2 : SP_profit = CP * 1.20)
  (h3 : SP_profit = 21.818181818181817) :
  SP = 16 := 
by 
  sorry

end fruit_seller_price_l864_86496


namespace cylindrical_pipe_height_l864_86462

theorem cylindrical_pipe_height (r_outer r_inner : ℝ) (SA : ℝ) (h : ℝ) 
  (h_outer : r_outer = 5)
  (h_inner : r_inner = 3)
  (h_SA : SA = 50 * Real.pi)
  (surface_area_eq: SA = 2 * Real.pi * (r_outer + r_inner) * h) : 
  h = 25 / 8 := 
by
  {
    sorry
  }

end cylindrical_pipe_height_l864_86462


namespace arithmetic_geometric_seq_l864_86435

theorem arithmetic_geometric_seq (a : ℕ → ℝ) (d a_1 : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0) (h_geom : (a 0, a 1, a 4) = (a_1, a_1 + d, a_1 + 4 * d) ∧ (a 1)^2 = a 0 * a 4)
  (h_sum : a 0 + a 1 + a 4 > 13) : a_1 > 1 :=
by sorry

end arithmetic_geometric_seq_l864_86435


namespace num_play_both_l864_86479

-- Definitions based on the conditions
def total_members : ℕ := 30
def play_badminton : ℕ := 17
def play_tennis : ℕ := 19
def play_neither : ℕ := 2

-- The statement we want to prove
theorem num_play_both :
  play_badminton + play_tennis - 8 = total_members - play_neither := by
  -- Omitted proof
  sorry

end num_play_both_l864_86479


namespace sum_c_d_eq_neg11_l864_86445

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x + 6) / (x^2 + c * x + d)

theorem sum_c_d_eq_neg11 (c d : ℝ) 
    (h₀ : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = 3 ∨ x = -4)) :
    c + d = -11 := 
sorry

end sum_c_d_eq_neg11_l864_86445


namespace negation_of_existential_proposition_l864_86459

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_of_existential_proposition_l864_86459


namespace min_ω_value_l864_86446

def min_ω (ω : Real) : Prop :=
  ω > 0 ∧ (∃ k : Int, ω = 2 * k + 2 / 3)

theorem min_ω_value : ∃ ω : Real, min_ω ω ∧ ω = 2 / 3 := by
  sorry

end min_ω_value_l864_86446


namespace total_ticket_sales_l864_86416

-- Define the parameters and the theorem to be proven.
theorem total_ticket_sales (total_people : ℕ) (kids : ℕ) (adult_ticket_price : ℕ) (kid_ticket_price : ℕ) 
  (adult_tickets := total_people - kids) 
  (adult_ticket_sales := adult_tickets * adult_ticket_price) 
  (kid_ticket_sales := kids * kid_ticket_price) : 
  total_people = 254 → kids = 203 → adult_ticket_price = 28 → kid_ticket_price = 12 → 
  adult_ticket_sales + kid_ticket_sales = 3864 := 
by
  intros h1 h2 h3 h4
  sorry

end total_ticket_sales_l864_86416


namespace angle_between_line_and_plane_l864_86448

noncomputable def vector_angle (m n : ℝ) : ℝ := 120

theorem angle_between_line_and_plane (m n : ℝ) : 
  (vector_angle m n = 120) → (90 - (vector_angle m n - 90) = 30) :=
by sorry

end angle_between_line_and_plane_l864_86448


namespace part1_solution_l864_86417

def f (x m : ℝ) := |x + m| + |2 * x + 1|

theorem part1_solution (x : ℝ) : f x (-1) ≤ 3 → -1 ≤ x ∧ x ≤ 1 := 
sorry

end part1_solution_l864_86417


namespace arithmetic_seq_a6_l864_86442

open Real

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (a (n + m) = a n + a m - a 0)

-- Given conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
  a 2 = 4

def condition_2 (a : ℕ → ℝ) : Prop :=
  a 4 = 2

-- Mathematical statement
theorem arithmetic_seq_a6 
  (a : ℕ → ℝ)
  (h_seq: arithmetic_sequence a)
  (h_cond1 : condition_1 a)
  (h_cond2 : condition_2 a) : 
  a 6 = 0 := 
sorry

end arithmetic_seq_a6_l864_86442


namespace work_done_by_first_group_l864_86494

theorem work_done_by_first_group :
  (6 * 8 * 5 : ℝ) / W = (4 * 3 * 8 : ℝ) / 30 →
  W = 75 :=
by
  sorry

end work_done_by_first_group_l864_86494


namespace book_distribution_l864_86488

theorem book_distribution (x : ℕ) (h1 : 9 * x + 7 < 11 * x) : 
  9 * x + 7 = totalBooks - 9 * x ∧ totalBooks - 9 * x = 7 :=
by
  sorry

end book_distribution_l864_86488


namespace water_usage_correct_l864_86469

variable (y : ℝ) (C₁ : ℝ) (C₂ : ℝ) (x : ℝ)

noncomputable def water_bill : ℝ :=
  if x ≤ 4 then C₁ * x else 4 * C₁ + C₂ * (x - 4)

theorem water_usage_correct (h1 : y = 12.8) (h2 : C₁ = 1.2) (h3 : C₂ = 1.6) : x = 9 :=
by
  have h4 : x > 4 := sorry
  sorry

end water_usage_correct_l864_86469


namespace range_of_a_minus_abs_b_l864_86402

theorem range_of_a_minus_abs_b (a b : ℝ) (h₁ : 1 < a ∧ a < 3) (h₂ : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l864_86402


namespace four_digit_arithmetic_sequence_l864_86458

theorem four_digit_arithmetic_sequence :
  ∃ (a b c d : ℕ), 1000 * a + 100 * b + 10 * c + d = 5555 ∨ 1000 * a + 100 * b + 10 * c + d = 2468 ∧
  (a + d = 10) ∧ (b + c = 10) ∧ (2 * b = a + c) ∧ (c - b = b - a) ∧ (d - c = c - b) ∧
  (1000 * d + 100 * c + 10 * b + a + 1000 * a + 100 * b + 10 * c + d = 11110) :=
sorry

end four_digit_arithmetic_sequence_l864_86458


namespace solution_correct_l864_86499

-- Define the conditions
def abs_inequality (x : ℝ) : Prop := abs (x - 3) + abs (x + 4) < 8
def quadratic_eq (x : ℝ) : Prop := x^2 - x - 12 = 0

-- Define the main statement to prove
theorem solution_correct : ∃ (x : ℝ), abs_inequality x ∧ quadratic_eq x ∧ x = -3 := sorry

end solution_correct_l864_86499


namespace similar_triangles_perimeter_l864_86431

open Real

-- Defining the similar triangles and their associated conditions
noncomputable def triangle1 := (4, 6, 8)
noncomputable def side2 := 2

-- Define the possible perimeters of the other triangle
theorem similar_triangles_perimeter (h : True) :
  (∃ x, x = 4.5 ∨ x = 6 ∨ x = 9) :=
sorry

end similar_triangles_perimeter_l864_86431


namespace correct_operation_B_l864_86464

theorem correct_operation_B (a b : ℝ) : - (a - b) = -a + b := 
by sorry

end correct_operation_B_l864_86464


namespace abs_add_three_eq_two_l864_86476

theorem abs_add_three_eq_two (a : ℝ) (h : a = -1) : |a + 3| = 2 :=
by
  rw [h]
  sorry

end abs_add_three_eq_two_l864_86476


namespace speed_conversion_l864_86450

theorem speed_conversion (s : ℝ) (h1 : s = 1 / 3) : s * 3.6 = 1.2 := by
  -- Proof follows from the conditions given
  sorry

end speed_conversion_l864_86450


namespace series_proof_l864_86414

theorem series_proof (a b : ℝ) (h : (∑' n : ℕ, (-1)^n * a / b^(n+1)) = 6) : 
  (∑' n : ℕ, (-1)^n * a / (a - b)^(n+1)) = 6 / 7 := 
sorry

end series_proof_l864_86414


namespace cost_per_meal_is_8_l864_86444

-- Define the conditions
def number_of_adults := 2
def number_of_children := 5
def total_bill := 56
def total_people := number_of_adults + number_of_children

-- Define the cost per meal
def cost_per_meal := total_bill / total_people

-- State the theorem we want to prove
theorem cost_per_meal_is_8 : cost_per_meal = 8 := 
by
  -- The proof would go here, but we'll use sorry to skip it
  sorry

end cost_per_meal_is_8_l864_86444


namespace objective_function_range_l864_86425

theorem objective_function_range:
  (∃ x y : ℝ, x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) ∧
  (∀ x y : ℝ, (x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) →
  (3*x + y ≥ (19:ℝ) / 9 ∧ 3*x + y ≤ 6)) :=
sorry

-- We have defined the conditions, the objective function, and the assertion in Lean 4.

end objective_function_range_l864_86425


namespace total_miles_Wednesday_l864_86493

-- The pilot flew 1134 miles on Tuesday and 1475 miles on Thursday.
def miles_flown_Tuesday : ℕ := 1134
def miles_flown_Thursday : ℕ := 1475

-- The miles flown on Wednesday is denoted as "x".
variable (x : ℕ)

-- The period is 4 weeks.
def weeks : ℕ := 4

-- We need to prove that the total miles flown on Wednesdays during this 4-week period is 4 * x.
theorem total_miles_Wednesday : 4 * x = 4 * x := by sorry

end total_miles_Wednesday_l864_86493


namespace jasmine_coffee_beans_purchase_l864_86420

theorem jasmine_coffee_beans_purchase (x : ℝ) (coffee_cost per_pound milk_cost per_gallon total_cost : ℝ)
  (h1 : coffee_cost = 2.50)
  (h2 : milk_cost = 3.50)
  (h3 : total_cost = 17)
  (h4 : milk_purchased = 2)
  (h_equation : coffee_cost * x + milk_cost * milk_purchased = total_cost) :
  x = 4 :=
by
  sorry

end jasmine_coffee_beans_purchase_l864_86420


namespace sum_of_abc_eq_11_l864_86471

theorem sum_of_abc_eq_11 (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_order : a < b ∧ b < c)
  (h_inv_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : a + b + c = 11 :=
  sorry

end sum_of_abc_eq_11_l864_86471


namespace lemonade_percentage_l864_86449

theorem lemonade_percentage (V : ℝ) (L : ℝ) :
  (0.80 * 0.40 * V + (100 - L) / 100 * 0.60 * V = 0.65 * V) →
  L = 99.45 :=
by
  intro h
  -- The proof would go here
  sorry

end lemonade_percentage_l864_86449


namespace Rohit_is_to_the_east_of_starting_point_l864_86455

-- Define the conditions and the problem statement.
def Rohit's_movements_proof
  (distance_south : ℕ) (distance_first_left : ℕ) (distance_second_left : ℕ) (distance_right : ℕ)
  (final_distance : ℕ) : Prop :=
  distance_south = 25 ∧
  distance_first_left = 20 ∧
  distance_second_left = 25 ∧
  distance_right = 15 ∧
  final_distance = 35 →
  (direction : String) → (distance : ℕ) →
  direction = "east" ∧ distance = final_distance

-- We can now state the theorem
theorem Rohit_is_to_the_east_of_starting_point :
  Rohit's_movements_proof 25 20 25 15 35 :=
by
  sorry

end Rohit_is_to_the_east_of_starting_point_l864_86455


namespace regular_polygon_sides_l864_86481

theorem regular_polygon_sides (n : ℕ) (h : ∀ n, (n > 2) → (360 / n = 20)) : n = 18 := sorry

end regular_polygon_sides_l864_86481


namespace eggs_per_snake_l864_86454

-- Define the conditions
def num_snakes : ℕ := 3
def price_regular : ℕ := 250
def price_super_rare : ℕ := 1000
def total_revenue : ℕ := 2250

-- Prove for the number of eggs each snake lays
theorem eggs_per_snake (E : ℕ) 
  (h1 : E * (num_snakes - 1) * price_regular + E * price_super_rare = total_revenue) : 
  E = 2 :=
sorry

end eggs_per_snake_l864_86454


namespace star_5_3_eq_31_l864_86452

def star (a b : ℤ) : ℤ := a^2 + a * b - b^2

theorem star_5_3_eq_31 : star 5 3 = 31 :=
by
  sorry

end star_5_3_eq_31_l864_86452


namespace fraction_arithmetic_l864_86436

theorem fraction_arithmetic : 
  (2 / 5 + 3 / 7) / (4 / 9 * 1 / 8) = 522 / 35 := by
  sorry

end fraction_arithmetic_l864_86436


namespace sum_of_three_squares_power_l864_86485

theorem sum_of_three_squares_power (n a b c k : ℕ) (h : n = a^2 + b^2 + c^2) (h_pos : n > 0) (k_pos : k > 0) :
  ∃ A B C : ℕ, n^(2*k) = A^2 + B^2 + C^2 :=
by
  sorry

end sum_of_three_squares_power_l864_86485


namespace calculate_speed_of_stream_l864_86491

noncomputable def speed_of_stream (boat_speed : ℕ) (downstream_distance : ℕ) (upstream_distance : ℕ) : ℕ :=
  let x := (downstream_distance * boat_speed - boat_speed * upstream_distance) / (downstream_distance + upstream_distance)
  x

theorem calculate_speed_of_stream :
  speed_of_stream 20 26 14 = 6 := by
  sorry

end calculate_speed_of_stream_l864_86491


namespace yvettes_final_bill_l864_86475

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end yvettes_final_bill_l864_86475


namespace range_of_a_l864_86473

-- Definitions of the sets U and A
def U := {x : ℝ | 0 < x ∧ x < 9}
def A (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) (H_non_empty : A a ≠ ∅) (H_not_subset : ¬ ∀ x, x ∈ A a → x ∈ U) : 
  1 < a ∧ a ≤ 9 :=
sorry

end range_of_a_l864_86473


namespace simplify_tangent_sum_l864_86467

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l864_86467


namespace largest_number_in_ratio_l864_86432

theorem largest_number_in_ratio (x : ℕ) (h : ((4 * x + 5 * x + 6 * x) / 3 : ℝ) = 20) : 6 * x = 24 := 
by 
  sorry

end largest_number_in_ratio_l864_86432


namespace fraction_eq_four_l864_86482

theorem fraction_eq_four (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) : 
  (2 * a + b) / b = 4 := 
by 
  sorry

end fraction_eq_four_l864_86482


namespace ordered_triples_count_l864_86495

theorem ordered_triples_count : 
  let b := 3003
  let side_length_squared := b * b
  let num_divisors := (2 + 1) * (2 + 1) * (2 + 1) * (2 + 1)
  let half_divisors := num_divisors / 2
  half_divisors = 40 := by
  sorry

end ordered_triples_count_l864_86495


namespace rightmost_three_digits_seven_pow_1983_add_123_l864_86427

theorem rightmost_three_digits_seven_pow_1983_add_123 :
  (7 ^ 1983 + 123) % 1000 = 466 := 
by 
  -- Proof steps are omitted
  sorry 

end rightmost_three_digits_seven_pow_1983_add_123_l864_86427


namespace intersection_complement_l864_86440

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement (U A : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The main theorem to be proved
theorem intersection_complement :
  B ∩ (complement U A) = {3, 4} := by
  sorry

end intersection_complement_l864_86440


namespace exists_root_in_interval_l864_86498

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

-- Conditions given in the problem
variables {a b c : ℝ}
variable  (h_a_nonzero : a ≠ 0)
variable  (h_neg_value : quadratic a b c 3.24 = -0.02)
variable  (h_pos_value : quadratic a b c 3.25 = 0.01)

-- Problem statement to be proved
theorem exists_root_in_interval : ∃ x : ℝ, 3.24 < x ∧ x < 3.25 ∧ quadratic a b c x = 0 :=
sorry

end exists_root_in_interval_l864_86498


namespace find_third_polygon_sides_l864_86424

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

theorem find_third_polygon_sides :
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  ∃ (m : ℕ), interior_angle m = third_polygon_angle ∧ m = 20 :=
by
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  use 20
  sorry

end find_third_polygon_sides_l864_86424


namespace A_50_correct_l864_86415

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![3, 2], 
    ![-8, -5]]

-- The theorem to prove
theorem A_50_correct : A^50 = ![![(-199 : ℤ), -100], 
                                 ![400, 201]] := 
by
  sorry

end A_50_correct_l864_86415


namespace whales_last_year_eq_4000_l864_86497

variable (W : ℕ) (last_year this_year next_year : ℕ)

theorem whales_last_year_eq_4000
    (h1 : this_year = 2 * last_year)
    (h2 : next_year = this_year + 800)
    (h3 : next_year = 8800) :
    last_year = 4000 := by
  sorry

end whales_last_year_eq_4000_l864_86497


namespace time_b_started_walking_l864_86460

/-- A's speed is 7 kmph, B's speed is 7.555555555555555 kmph, and B overtakes A after 1.8 hours. -/
theorem time_b_started_walking (t : ℝ) (A_speed : ℝ) (B_speed : ℝ) (overtake_time : ℝ)
    (hA : A_speed = 7) (hB : B_speed = 7.555555555555555) (hOvertake : overtake_time = 1.8) 
    (distance_A : ℝ) (distance_B : ℝ)
    (hDistanceA : distance_A = (t + overtake_time) * A_speed)
    (hDistanceB : distance_B = B_speed * overtake_time) :
  t = 8.57 / 60 := by
  sorry

end time_b_started_walking_l864_86460


namespace plane_split_four_regions_l864_86441

theorem plane_split_four_regions :
  (∀ x y : ℝ, y = 3 * x ∨ x = 3 * y) → (exists regions : ℕ, regions = 4) :=
by
  sorry

end plane_split_four_regions_l864_86441


namespace ratio_of_dolls_l864_86477

-- Definitions used in Lean 4 statement directly appear in the conditions
variable (I : ℕ) -- the number of dolls Ivy has
variable (Dina_dolls : ℕ := 60) -- Dina has 60 dolls
variable (Ivy_collectors : ℕ := 20) -- Ivy has 20 collector edition dolls

-- Condition based on given problem
axiom Ivy_collectors_condition : (2 / 3 : ℚ) * I = 20

-- Lean 4 statement for the proof problem
theorem ratio_of_dolls (h : 3 * Ivy_collectors = 2 * I) : Dina_dolls / I = 2 := by
  sorry

end ratio_of_dolls_l864_86477


namespace sampling_methods_correct_l864_86484

def first_method_sampling : String :=
  "Simple random sampling"

def second_method_sampling : String :=
  "Systematic sampling"

theorem sampling_methods_correct :
  first_method_sampling = "Simple random sampling" ∧ second_method_sampling = "Systematic sampling" :=
by
  sorry

end sampling_methods_correct_l864_86484


namespace marble_count_l864_86434

theorem marble_count (p y v : ℝ) (h1 : y + v = 10) (h2 : p + v = 12) (h3 : p + y = 5) :
  p + y + v = 13.5 :=
sorry

end marble_count_l864_86434


namespace monotonicity_and_zeros_l864_86456

open Real

noncomputable def f (x k : ℝ) : ℝ := exp x - k * x + k

theorem monotonicity_and_zeros
  (k : ℝ)
  (h₁ : k > exp 2)
  (x₁ x₂ : ℝ)
  (h₂ : f x₁ k = 0)
  (h₃ : f x₂ k = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := 
sorry

end monotonicity_and_zeros_l864_86456


namespace scientific_notation_example_l864_86470

theorem scientific_notation_example :
  (0.000000007: ℝ) = 7 * 10^(-9 : ℝ) :=
sorry

end scientific_notation_example_l864_86470


namespace machines_produce_x_units_l864_86487

variable (x : ℕ) (d : ℕ)

-- Define the conditions
def four_machines_produce_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  4 * (x / d) = x / d

def twelve_machines_produce_three_x_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  12 * (x / d) = 3 * (x / d)

-- Given the conditions, prove the number of days for 4 machines to produce x units
theorem machines_produce_x_units (x : ℕ) (d : ℕ) 
  (H1 : four_machines_produce_in_d_days x d)
  (H2 : twelve_machines_produce_three_x_in_d_days x d) : 
  x / d = x / d := 
by 
  sorry

end machines_produce_x_units_l864_86487


namespace boxes_in_carton_of_pencils_l864_86453

def cost_per_box_pencil : ℕ := 2
def cost_per_box_marker : ℕ := 4
def boxes_per_carton_marker : ℕ := 5
def cartons_of_pencils : ℕ := 20
def cartons_of_markers : ℕ := 10
def total_spent : ℕ := 600

theorem boxes_in_carton_of_pencils : ∃ x : ℕ, 20 * (2 * x) + 10 * (5 * 4) = 600 :=
by
  sorry

end boxes_in_carton_of_pencils_l864_86453


namespace total_value_of_assets_l864_86400

variable (value_expensive_stock : ℕ)
variable (shares_expensive_stock : ℕ)
variable (shares_other_stock : ℕ)
variable (value_other_stock : ℕ)

theorem total_value_of_assets
    (h1: value_expensive_stock = 78)
    (h2: shares_expensive_stock = 14)
    (h3: shares_other_stock = 26)
    (h4: value_other_stock = value_expensive_stock / 2) :
    shares_expensive_stock * value_expensive_stock + shares_other_stock * value_other_stock = 2106 := by
    sorry

end total_value_of_assets_l864_86400


namespace fare_collected_from_I_class_l864_86406

theorem fare_collected_from_I_class (x y : ℝ) 
  (h1 : ∀i, i = x → ∀ii, ii = 4 * x)
  (h2 : ∀f1, f1 = 3 * y)
  (h3 : ∀f2, f2 = y)
  (h4 : x * 3 * y + 4 * x * y = 224000) : 
  x * 3 * y = 96000 :=
by
  sorry

end fare_collected_from_I_class_l864_86406


namespace infinite_div_by_100_l864_86472

theorem infinite_div_by_100 : ∀ k : ℕ, ∃ n : ℕ, n > 0 ∧ (2 ^ n + n ^ 2) % 100 = 0 :=
by
  sorry

end infinite_div_by_100_l864_86472


namespace final_weight_of_box_l864_86451

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end final_weight_of_box_l864_86451


namespace transform_roots_to_quadratic_l864_86428

noncomputable def quadratic_formula (p q : ℝ) (x : ℝ) : ℝ :=
  x^2 + p * x + q

theorem transform_roots_to_quadratic (x₁ x₂ y₁ y₂ p q : ℝ)
  (h₁ : quadratic_formula p q x₁ = 0)
  (h₂ : quadratic_formula p q x₂ = 0)
  (h₃ : x₁ ≠ 1)
  (h₄ : x₂ ≠ 1)
  (hy₁ : y₁ = (x₁ + 1) / (x₁ - 1))
  (hy₂ : y₂ = (x₂ + 1) / (x₂ - 1)) :
  (1 + p + q) * y₁^2 + 2 * (1 - q) * y₁ + (1 - p + q) = 0 ∧
  (1 + p + q) * y₂^2 + 2 * (1 - q) * y₂ + (1 - p + q) = 0 := 
sorry

end transform_roots_to_quadratic_l864_86428


namespace cucumber_to_tomato_ratio_l864_86437

variable (total_rows : ℕ) (space_per_row_tomato : ℕ) (tomatoes_per_plant : ℕ) (total_tomatoes : ℕ)

/-- Aubrey's Garden -/
theorem cucumber_to_tomato_ratio (total_rows_eq : total_rows = 15)
  (space_per_row_tomato_eq : space_per_row_tomato = 8)
  (tomatoes_per_plant_eq : tomatoes_per_plant = 3)
  (total_tomatoes_eq : total_tomatoes = 120) :
  let total_tomato_plants := total_tomatoes / tomatoes_per_plant
  let rows_tomato := total_tomato_plants / space_per_row_tomato
  let rows_cucumber := total_rows - rows_tomato
  (2 * rows_tomato = rows_cucumber)
:=
by
  sorry

end cucumber_to_tomato_ratio_l864_86437


namespace equation_of_parallel_line_l864_86489

theorem equation_of_parallel_line (x y : ℝ) :
  (∀ b : ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → b = 0) →
  (∀ x y b: ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → 2 * x + y = 0) :=
by
  sorry

end equation_of_parallel_line_l864_86489


namespace syllogism_example_l864_86409

-- Definitions based on the conditions
def is_even (n : ℕ) := n % 2 = 0
def is_divisible_by_2 (n : ℕ) := n % 2 = 0

-- Given conditions:
axiom even_implies_divisible_by_2 : ∀ n : ℕ, is_even n → is_divisible_by_2 n
axiom h2012_is_even : is_even 2012

-- Proving the conclusion and the syllogism pattern
theorem syllogism_example : is_divisible_by_2 2012 :=
by
  apply even_implies_divisible_by_2
  apply h2012_is_even

end syllogism_example_l864_86409


namespace ben_is_10_l864_86419

-- Define the ages of the cousins
def ages : List ℕ := [6, 8, 10, 12, 14]

-- Define the conditions
def wentToPark (x y : ℕ) : Prop := x + y = 18
def wentToLibrary (x y : ℕ) : Prop := x + y < 20
def stayedHome (ben young : ℕ) : Prop := young = 6 ∧ ben ∈ ages ∧ ben ≠ 6 ∧ ben ≠ 12

-- The main theorem stating Ben's age
theorem ben_is_10 : ∃ ben, stayedHome ben 6 ∧ 
  (∃ x y, wentToPark x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) ∧
  (∃ x y, wentToLibrary x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) :=
by
  use 10
  -- Proof steps would go here
  sorry

end ben_is_10_l864_86419


namespace area_of_unpainted_section_l864_86492

-- Define the conditions
def board1_width : ℝ := 5
def board2_width : ℝ := 7
def cross_angle : ℝ := 45
def negligible_holes : Prop := true

-- The main statement
theorem area_of_unpainted_section (h1 : board1_width = 5) (h2 : board2_width = 7) (h3 : cross_angle = 45) (h4 : negligible_holes) : 
  ∃ (area : ℝ), area = 35 := 
sorry

end area_of_unpainted_section_l864_86492


namespace prove_value_of_expression_l864_86430

theorem prove_value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 :=
by 
  sorry

end prove_value_of_expression_l864_86430


namespace jeremy_school_distance_l864_86412

theorem jeremy_school_distance (d : ℝ) (v : ℝ) :
  (d = v * 0.5) ∧
  (d = (v + 15) * 0.3) ∧
  (d = (v - 10) * (2 / 3)) →
  d = 15 :=
by 
  sorry

end jeremy_school_distance_l864_86412


namespace ray_inequality_l864_86465

theorem ray_inequality (a : ℝ) :
  (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ 1)
  ∨ (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ -1) :=
sorry

end ray_inequality_l864_86465


namespace line_equation_with_equal_intercepts_l864_86407

theorem line_equation_with_equal_intercepts 
  (a : ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h : ∀ x y, l x y ↔ (a+1)*x + y + 2 - a = 0) 
  (intercept_condition : ∀ x y, l x 0 = l 0 y) : 
  (∀ x y, l x y ↔ x + y + 2 = 0) ∨ (∀ x y, l x y ↔ 3*x + y = 0) :=
sorry

end line_equation_with_equal_intercepts_l864_86407


namespace cylinder_height_l864_86421

   theorem cylinder_height (r h : ℝ) (SA : ℝ) (π : ℝ) :
     r = 3 → SA = 30 * π → SA = 2 * π * r^2 + 2 * π * r * h → h = 2 :=
   by
     intros hr hSA hSA_formula
     rw [hr] at hSA_formula
     rw [hSA] at hSA_formula
     sorry
   
end cylinder_height_l864_86421


namespace ruby_shares_with_9_friends_l864_86443

theorem ruby_shares_with_9_friends
    (total_candies : ℕ) (candies_per_friend : ℕ)
    (h1 : total_candies = 36) (h2 : candies_per_friend = 4) :
    total_candies / candies_per_friend = 9 := by
  sorry

end ruby_shares_with_9_friends_l864_86443


namespace jane_last_day_vases_l864_86426

def vasesPerDay : Nat := 16
def totalVases : Nat := 248

theorem jane_last_day_vases : totalVases % vasesPerDay = 8 := by
  sorry

end jane_last_day_vases_l864_86426


namespace theater_ticket_cost_l864_86447

theorem theater_ticket_cost
  (O B : ℕ)
  (h1 : O + B = 370)
  (h2 : B = O + 190) 
  : 12 * O + 8 * B = 3320 :=
by
  sorry

end theater_ticket_cost_l864_86447


namespace roots_expression_eval_l864_86439

theorem roots_expression_eval (p q r : ℝ) 
  (h1 : p + q + r = 2)
  (h2 : p * q + q * r + r * p = -1)
  (h3 : p * q * r = -2)
  (hp : p^3 - 2 * p^2 - p + 2 = 0)
  (hq : q^3 - 2 * q^2 - q + 2 = 0)
  (hr : r^3 - 2 * r^2 - r + 2 = 0) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = 16 :=
sorry

end roots_expression_eval_l864_86439


namespace intersection_of_A_and_B_l864_86429

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  sorry

end intersection_of_A_and_B_l864_86429


namespace time_to_pass_tree_l864_86480

-- Define the conditions given in the problem
def train_length : ℕ := 1200
def platform_length : ℕ := 700
def time_to_pass_platform : ℕ := 190

-- Calculate the total distance covered while passing the platform
def distance_passed_platform : ℕ := train_length + platform_length

-- The main theorem we need to prove
theorem time_to_pass_tree : (distance_passed_platform / time_to_pass_platform) * train_length = 120 := 
by
  sorry

end time_to_pass_tree_l864_86480


namespace Rockham_Soccer_League_members_l864_86405

theorem Rockham_Soccer_League_members (sock_cost tshirt_cost cap_cost total_cost members : ℕ) (h1 : sock_cost = 6) (h2 : tshirt_cost = sock_cost + 10) (h3 : cap_cost = 3) (h4 : total_cost = 4620) (h5 : total_cost = 50 * members) : members = 92 :=
by
  sorry

end Rockham_Soccer_League_members_l864_86405


namespace average_of_four_digits_l864_86413

theorem average_of_four_digits (sum9 : ℤ) (avg9 : ℤ) (avg5 : ℤ) (sum4 : ℤ) (n : ℤ) :
  avg9 = 18 →
  n = 9 →
  sum9 = avg9 * n →
  avg5 = 26 →
  sum4 = sum9 - (avg5 * 5) →
  avg4 = sum4 / 4 →
  avg4 = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_of_four_digits_l864_86413


namespace melting_point_of_ice_in_Celsius_l864_86408

theorem melting_point_of_ice_in_Celsius :
  ∀ (boiling_point_F boiling_point_C melting_point_F temperature_C temperature_F : ℤ),
    (boiling_point_F = 212) →
    (boiling_point_C = 100) →
    (melting_point_F = 32) →
    (temperature_C = 60) →
    (temperature_F = 140) →
    (5 * melting_point_F = 9 * 0 + 160) →         -- Using the given equation F = (9/5)C + 32 and C = 0
    melting_point_F = 32 ∧ 0 = 0 :=
by
  intros
  sorry

end melting_point_of_ice_in_Celsius_l864_86408
