import Mathlib

namespace bookstore_discount_l44_44785

theorem bookstore_discount (P MP price_paid : ℝ) (h1 : MP = 0.80 * P) (h2 : price_paid = 0.60 * MP) :
  price_paid / P = 0.48 :=
by
  sorry

end bookstore_discount_l44_44785


namespace a2_value_for_cubic_expansion_l44_44337

theorem a2_value_for_cubic_expansion (x a0 a1 a2 a3 : ℝ) : 
  (x ^ 3 = a0 + a1 * (x - 2) + a2 * (x - 2) ^ 2 + a3 * (x - 2) ^ 3) → a2 = 6 := by
  sorry

end a2_value_for_cubic_expansion_l44_44337


namespace greatest_perimeter_triangle_l44_44366

theorem greatest_perimeter_triangle :
  ∃ (x : ℕ), (x > (16 / 5)) ∧ (x < (16 / 3)) ∧ ((x = 4 ∨ x = 5) → 4 * x + x + 16 = 41) :=
by
  sorry

end greatest_perimeter_triangle_l44_44366


namespace angle_bisector_5cm_l44_44705

noncomputable def angle_bisector_length (a b c : ℝ) : ℝ :=
  real.sqrt (a * b * (1 - (c^2 / (a + b)^2)))

theorem angle_bisector_5cm
  (A B C : Type) [plane_angle A] [plane_angle C] [plane_angle B]
  (α β γ : ℝ) (a b c : ℝ)
  (hA : α = 20) (hC : γ = 40)
  (h_difference : AC - AB = 5) :
  angle_bisector_length a b c = 5 := sorry

end angle_bisector_5cm_l44_44705


namespace intersection_of_N_and_not_R_M_l44_44684

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def Not_R_M : Set ℝ := {x | x ≤ 2}

theorem intersection_of_N_and_not_R_M : 
  N ∩ Not_R_M = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_N_and_not_R_M_l44_44684


namespace geometric_sequence_sum_l44_44380

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end geometric_sequence_sum_l44_44380


namespace smallest_x_for_multiple_l44_44266

theorem smallest_x_for_multiple (x : ℕ) (h₁: 450 = 2 * 3^2 * 5^2) (h₂: 800 = 2^6 * 5^2) : 
  ((450 * x) % 800 = 0) ↔ x ≥ 32 :=
by
  sorry

end smallest_x_for_multiple_l44_44266


namespace cori_age_proof_l44_44481

theorem cori_age_proof:
  ∃ (x : ℕ), (3 + x = (1 / 3) * (19 + x)) ∧ x = 5 :=
by
  sorry

end cori_age_proof_l44_44481


namespace math_problem_l44_44302

theorem math_problem :
  (Int.ceil ((16 / 5 : ℚ) * (-34 / 4 : ℚ)) - Int.floor ((16 / 5 : ℚ) * Int.floor (-34 / 4 : ℚ))) = 2 :=
by
  sorry

end math_problem_l44_44302


namespace total_volume_of_mixed_solutions_l44_44453

theorem total_volume_of_mixed_solutions :
  let v1 := 3.6
  let v2 := 1.4
  v1 + v2 = 5.0 := by
  sorry

end total_volume_of_mixed_solutions_l44_44453


namespace rowing_time_from_A_to_B_and_back_l44_44372

-- Define the problem parameters and conditions
def rowing_speed_still_water : ℝ := 5
def distance_AB : ℝ := 12
def stream_speed : ℝ := 1

-- Define the problem to prove
theorem rowing_time_from_A_to_B_and_back :
  let downstream_speed := rowing_speed_still_water + stream_speed
  let upstream_speed := rowing_speed_still_water - stream_speed
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := distance_AB / upstream_speed
  let total_time := time_downstream + time_upstream
  total_time = 5 :=
by
  sorry

end rowing_time_from_A_to_B_and_back_l44_44372


namespace eval_f_at_3_l44_44687

def f (x : ℝ) : ℝ := 3 * x + 1

theorem eval_f_at_3 : f 3 = 10 :=
by
  -- computation of f at x = 3
  sorry

end eval_f_at_3_l44_44687


namespace solution_set_range_ineq_l44_44002

theorem solution_set_range_ineq (m : ℝ) :
  ∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0 ↔ (-5: ℝ)⁻¹ < m ∧ m ≤ 3 :=
by
  sorry

end solution_set_range_ineq_l44_44002


namespace amount_paid_l44_44227

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l44_44227


namespace sum_of_geometric_sequence_l44_44334

theorem sum_of_geometric_sequence :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 3280 / 6561 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  sorry

end sum_of_geometric_sequence_l44_44334


namespace A_profit_share_l44_44296

variables (profit : ℚ) (A_share B_share C_share D_share : ℚ)

-- Given conditions
def conditions : Prop :=
  A_share = 1/3 ∧
  B_share = 1/4 ∧
  C_share = 1/5 ∧
  D_share = 1 - (A_share + B_share + C_share) ∧
  profit = 2490

-- The main theorem statement
theorem A_profit_share (h : conditions profit A_share B_share C_share D_share) :
  A_share * profit = 830 :=
sorry

end A_profit_share_l44_44296


namespace tangent_line_intersection_l44_44786

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l44_44786


namespace minimal_sum_of_squares_of_roots_l44_44664

open Real

theorem minimal_sum_of_squares_of_roots :
  ∀ a : ℝ,
  (let x1 := 3*a + 1;
   let x2 := 2*a^2 - 3*a - 2;
   (a^2 + 18*a + 9) ≥ 0 →
   (x1^2 - 2*x2) = (5*a^2 + 12*a + 5) →
   a = -9 + 6*sqrt 2) :=
by
  sorry

end minimal_sum_of_squares_of_roots_l44_44664


namespace ceiling_of_square_of_neg_7_over_4_is_4_l44_44509

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l44_44509


namespace tan_neg_1140_eq_neg_sqrt3_l44_44336

theorem tan_neg_1140_eq_neg_sqrt3 
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_periodicity : ∀ θ : ℝ, ∀ n : ℤ, Real.tan (θ + n * 180) = Real.tan θ)
  (tan_60 : Real.tan 60 = Real.sqrt 3) :
  Real.tan (-1140) = -Real.sqrt 3 := 
sorry

end tan_neg_1140_eq_neg_sqrt3_l44_44336


namespace no_primes_divisible_by_45_l44_44176

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define 45 and its prime factors
def is_divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem to prove the number of prime numbers divisible by 45 is 0
theorem no_primes_divisible_by_45 : ∀ n : ℕ, is_prime n → is_divisible_by_45 n → false :=
by
  intro n
  assume h_prime h_div_45
  sorry

end no_primes_divisible_by_45_l44_44176


namespace four_pow_four_mul_five_pow_four_l44_44322

theorem four_pow_four_mul_five_pow_four : (4 ^ 4) * (5 ^ 4) = 160000 := by
  sorry

end four_pow_four_mul_five_pow_four_l44_44322


namespace ratio_of_ages_l44_44716

variable (x : Nat) -- The multiple of Marie's age
variable (marco_age marie_age : Nat) -- Marco's and Marie's ages

-- Conditions from (a)
axiom h1 : marie_age = 12
axiom h2 : marco_age = (12 * x) + 1
axiom h3 : marco_age + marie_age = 37

-- Statement to be proved
theorem ratio_of_ages : (marco_age : Nat) / (marie_age : Nat) = (25 / 12) :=
by
  -- Proof steps here
  sorry

end ratio_of_ages_l44_44716


namespace cargo_per_truck_is_2_5_l44_44158

-- Define our instance conditions
variables (x : ℝ) (n : ℕ)

-- Conditions extracted from the problem
def truck_capacity_change : Prop :=
  55 ≤ x ∧ x ≤ 64 ∧
  (x = (x / n - 0.5) * (n + 4))

-- Objective based on these conditions
theorem cargo_per_truck_is_2_5 :
  truck_capacity_change x n → (x = 60) → (n + 4 = 24) → (x / 24 = 2.5) :=
by 
  sorry

end cargo_per_truck_is_2_5_l44_44158


namespace total_cost_in_dollars_l44_44066

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end total_cost_in_dollars_l44_44066


namespace find_ages_l44_44933

variables (H J A : ℕ)

def conditions := 
  H + J + A = 90 ∧ 
  H = 2 * J - 5 ∧ 
  H + J - 10 = A

theorem find_ages (h_cond : conditions H J A) : 
  H = 32 ∧ 
  J = 18 ∧ 
  A = 40 :=
sorry

end find_ages_l44_44933


namespace sum_of_first_2015_digits_l44_44407

noncomputable def repeating_decimal : List ℕ := [1, 4, 2, 8, 5, 7]

def sum_first_n_digits (digits : List ℕ) (n : ℕ) : ℕ :=
  let repeat_length := digits.length
  let full_cycles := n / repeat_length
  let remaining_digits := n % repeat_length
  full_cycles * (digits.sum) + (digits.take remaining_digits).sum

theorem sum_of_first_2015_digits :
  sum_first_n_digits repeating_decimal 2015 = 9065 :=
by
  sorry

end sum_of_first_2015_digits_l44_44407


namespace ticket_price_l44_44229

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l44_44229


namespace perimeter_of_square_is_32_l44_44849

-- Given conditions
def radius := 4
def diameter := 2 * radius
def side_length_of_square := diameter

-- Question: What is the perimeter of the square?
def perimeter_of_square := 4 * side_length_of_square

-- Proof statement
theorem perimeter_of_square_is_32 : perimeter_of_square = 32 :=
sorry

end perimeter_of_square_is_32_l44_44849


namespace simplify_fraction_expression_l44_44395

theorem simplify_fraction_expression :
  5 * (12 / 7) * (49 / (-60)) = -7 := 
sorry

end simplify_fraction_expression_l44_44395


namespace inhabitant_eq_resident_l44_44776

-- Definitions
def inhabitant : Type := String
def resident : Type := String

-- The equivalence theorem
theorem inhabitant_eq_resident :
  ∀ (x : inhabitant), x = "resident" :=
by
  sorry

end inhabitant_eq_resident_l44_44776


namespace quadratic_inequality_solution_l44_44853

theorem quadratic_inequality_solution (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c) * x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} :=
sorry

end quadratic_inequality_solution_l44_44853


namespace total_cleaning_time_l44_44912

def hose_time : ℕ := 10
def shampoos : ℕ := 3
def shampoo_time : ℕ := 15

theorem total_cleaning_time : hose_time + shampoos * shampoo_time = 55 := by
  sorry

end total_cleaning_time_l44_44912


namespace students_selected_from_grade_10_l44_44125

theorem students_selected_from_grade_10 (students_grade10 students_grade11 students_grade12 total_selected : ℕ)
  (h_grade10 : students_grade10 = 1200)
  (h_grade11 : students_grade11 = 1000)
  (h_grade12 : students_grade12 = 800)
  (h_total_selected : total_selected = 100) :
  students_grade10 * total_selected = 40 * (students_grade10 + students_grade11 + students_grade12) :=
by
  sorry

end students_selected_from_grade_10_l44_44125


namespace least_product_of_primes_over_30_l44_44088

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l44_44088


namespace num_impossible_events_l44_44298

def water_boils_at_90C := false
def iron_melts_at_room_temp := false
def coin_flip_results_heads := true
def abs_value_not_less_than_zero := true

theorem num_impossible_events :
  water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
  coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true →
  (if ¬water_boils_at_90C then 1 else 0) + (if ¬iron_melts_at_room_temp then 1 else 0) +
  (if ¬coin_flip_results_heads then 1 else 0) + (if ¬abs_value_not_less_than_zero then 1 else 0) = 2
:= by
  intro h
  have : 
    water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
    coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true := h
  sorry

end num_impossible_events_l44_44298


namespace find_d_l44_44919

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem find_d (d : ℝ) (h₁ : 0 ≤ d ∧ d ≤ 2) (h₂ : 6 - ((1 / 2) * (2 - d) * 2) = 2 * ((1 / 2) * (2 - d) * 2)) : 
  d = 0 :=
sorry

end find_d_l44_44919


namespace max_leap_years_l44_44814

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) :
  leap_interval = 5 ∧ total_years = 200 → (years = total_years / leap_interval) :=
by
  sorry

end max_leap_years_l44_44814


namespace ticket_price_l44_44231

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l44_44231


namespace smallest_positive_value_of_a_minus_b_l44_44596

theorem smallest_positive_value_of_a_minus_b :
  ∃ (a b : ℤ), 17 * a + 6 * b = 13 ∧ a - b = 17 :=
by
  sorry

end smallest_positive_value_of_a_minus_b_l44_44596


namespace annie_has_12_brownies_left_l44_44981

noncomputable def initial_brownies := 100
noncomputable def portion_for_admin := (3 / 5 : ℚ) * initial_brownies
noncomputable def leftover_after_admin := initial_brownies - portion_for_admin
noncomputable def portion_for_carl := (1 / 4 : ℚ) * leftover_after_admin
noncomputable def leftover_after_carl := leftover_after_admin - portion_for_carl
noncomputable def portion_for_simon := 3
noncomputable def leftover_after_simon := leftover_after_carl - portion_for_simon
noncomputable def portion_for_friends := (2 / 3 : ℚ) * leftover_after_simon
noncomputable def each_friend_get := portion_for_friends / 5
noncomputable def total_given_to_friends := each_friend_get * 5
noncomputable def final_brownies := leftover_after_simon - total_given_to_friends

theorem annie_has_12_brownies_left : final_brownies = 12 := by
  sorry

end annie_has_12_brownies_left_l44_44981


namespace basic_astrophysics_degrees_l44_44452

-- Define the given percentages
def microphotonics_percentage : ℝ := 14
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 10
def gmo_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def total_circle_degrees : ℝ := 360

-- Define a proof problem to show that basic astrophysics research occupies 54 degrees in the circle
theorem basic_astrophysics_degrees :
  total_circle_degrees - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage + gmo_percentage + industrial_lubricants_percentage) = 15 ∧
  0.15 * total_circle_degrees = 54 :=
by
  sorry

end basic_astrophysics_degrees_l44_44452


namespace sequence_a_n_2013_l44_44350

theorem sequence_a_n_2013 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2013 = 3 :=
sorry

end sequence_a_n_2013_l44_44350


namespace first_grade_frequency_is_correct_second_grade_frequency_is_correct_l44_44300

def total_items : ℕ := 400
def second_grade_items : ℕ := 20
def first_grade_items : ℕ := total_items - second_grade_items

def frequency_first_grade : ℚ := first_grade_items / total_items
def frequency_second_grade : ℚ := second_grade_items / total_items

theorem first_grade_frequency_is_correct : frequency_first_grade = 0.95 := 
 by
 sorry

theorem second_grade_frequency_is_correct : frequency_second_grade = 0.05 := 
 by 
 sorry

end first_grade_frequency_is_correct_second_grade_frequency_is_correct_l44_44300


namespace find_number_l44_44036

theorem find_number (x : ℝ) (h : (((18 + x) / 3 + 10) / 5 = 4)) : x = 12 :=
by
  sorry

end find_number_l44_44036


namespace symmetric_circle_equation_l44_44750

noncomputable def equation_of_symmetric_circle (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2 * x - 6 * y + 9 = 0) ∧ (2 * x + y + 5 = 0)

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), 
    equation_of_symmetric_circle x y → 
    ∃ a b : ℝ, ((x - a)^2 + (y - b)^2 = 1) ∧ (a + 7 = 0) ∧ (b + 1 = 0) :=
sorry

end symmetric_circle_equation_l44_44750


namespace common_ratio_is_4_l44_44689

theorem common_ratio_is_4 
  (a : ℕ → ℝ) -- The geometric sequence
  (r : ℝ) -- The common ratio
  (h_geo_seq : ∀ n, a (n + 1) = r * a n) -- Definition of geometric sequence
  (h_condition : ∀ n, a n * a (n + 1) = 16 ^ n) -- Given condition
  : r = 4 := 
  sorry

end common_ratio_is_4_l44_44689


namespace dog_total_bones_l44_44601

-- Define the number of original bones and dug up bones as constants
def original_bones : ℕ := 493
def dug_up_bones : ℕ := 367

-- Define the total bones the dog has now
def total_bones : ℕ := original_bones + dug_up_bones

-- State and prove the theorem
theorem dog_total_bones : total_bones = 860 := by
  -- placeholder for the proof
  sorry

end dog_total_bones_l44_44601


namespace ceil_square_of_neg_fraction_l44_44524

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l44_44524


namespace geometric_series_first_term_l44_44134

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 24)
  (h_sum : S = a / (1 - r)) : 
  a = 18 :=
by {
  -- valid proof body goes here
  sorry
}

end geometric_series_first_term_l44_44134


namespace final_answer_after_subtracting_l44_44128

theorem final_answer_after_subtracting (n : ℕ) (h : n = 990) : (n / 9) - 100 = 10 :=
by
  sorry

end final_answer_after_subtracting_l44_44128


namespace paving_stones_needed_l44_44645

def length_courtyard : ℝ := 60
def width_courtyard : ℝ := 14
def width_stone : ℝ := 2
def paving_stones_required : ℕ := 140

theorem paving_stones_needed (L : ℝ) 
  (h1 : length_courtyard * width_courtyard = 840) 
  (h2 : paving_stones_required = 140)
  (h3 : (140 * (L * 2)) = 840) : 
  (length_courtyard * width_courtyard) / (L * width_stone) = 140 := 
by sorry

end paving_stones_needed_l44_44645


namespace maximum_value_of_vectors_l44_44016

open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 3))

def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1

def given_conditions (a b c : EuclideanSpace ℝ (Fin 3)) : Prop :=
  unit_vector a ∧ unit_vector b ∧ ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖ ∧ ‖c‖ = 2

theorem maximum_value_of_vectors
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖)
  (hc : ‖c‖ = 2) :
  ‖a + b - c‖ ≤ sqrt 2 + 2 := 
by
  sorry

end maximum_value_of_vectors_l44_44016


namespace find_k_l44_44332

theorem find_k (x y z k : ℝ) (h1 : 8 / (x + y + 1) = k / (x + z + 2)) (h2 : k / (x + z + 2) = 12 / (z - y + 3)) : k = 20 := by
  sorry

end find_k_l44_44332


namespace sandy_painting_area_l44_44058

theorem sandy_painting_area :
  let wall_height := 10
  let wall_length := 15
  let painting_height := 3
  let painting_length := 5
  let wall_area := wall_height * wall_length
  let painting_area := painting_height * painting_length
  let area_to_paint := wall_area - painting_area
  area_to_paint = 135 := 
by 
  sorry

end sandy_painting_area_l44_44058


namespace Grace_pool_water_capacity_l44_44569

theorem Grace_pool_water_capacity :
  let rate1 := 50 -- gallons per hour of the first hose
  let rate2 := 70 -- gallons per hour of the second hose
  let hours1 := 3 -- hours the first hose was used alone
  let hours2 := 2 -- hours both hoses were used together
  let water1 := rate1 * hours1 -- water from the first hose in the first period
  let water2 := rate2 * hours2 -- water from the second hose in the second period
  let water3 := rate1 * hours2 -- water from the first hose in the second period
  let total_water := water1 + water2 + water3 -- total water in the pool
  total_water = 390 :=
by
  sorry

end Grace_pool_water_capacity_l44_44569


namespace rectangle_area_ratio_l44_44922

theorem rectangle_area_ratio (l b : ℕ) (h1 : l = b + 10) (h2 : b = 8) : (l * b) / b = 18 := by
  sorry

end rectangle_area_ratio_l44_44922


namespace minimum_fencing_l44_44717

variable (a b z : ℝ)

def area_condition : Prop := a * b = 50
def length_condition : Prop := a + 2 * b = z

theorem minimum_fencing (h1 : area_condition a b) (h2 : length_condition a b z) : z ≥ 20 := 
  sorry

end minimum_fencing_l44_44717


namespace three_digit_sum_permutations_l44_44388

theorem three_digit_sum_permutations (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 1 ≤ b) (h₄ : b ≤ 9) (h₅ : 1 ≤ c) (h₆ : c ≤ 9)
  (h₇ : n = 100 * a + 10 * b + c)
  (h₈ : 222 * (a + b + c) - n = 1990) :
  n = 452 :=
by
  sorry

end three_digit_sum_permutations_l44_44388


namespace range_of_m_for_function_l44_44170

noncomputable def isFunctionDefinedForAllReal (f : ℝ → ℝ) := ∀ x : ℝ, true

theorem range_of_m_for_function :
  (∀ x : ℝ, x^2 - 2 * m * x + m + 2 > 0) ↔ (-1 < m ∧ m < 2) :=
sorry

end range_of_m_for_function_l44_44170


namespace polygon_sides_l44_44878

-- Definitions based on the conditions provided
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

def sum_exterior_angles : ℝ := 360 

def condition (n : ℕ) : Prop :=
  sum_interior_angles n = 2 * sum_exterior_angles + 180

-- Main theorem based on the correct answer
theorem polygon_sides (n : ℕ) (h : condition n) : n = 7 :=
sorry

end polygon_sides_l44_44878


namespace factorize_expr_l44_44998

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l44_44998


namespace rectangle_perimeter_l44_44420

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l44_44420


namespace range_of_a_l44_44342

def p (a : ℝ) : Prop := a > -1
def q (a : ℝ) : Prop := ∀ m : ℝ, -2 ≤ m ∧ m ≤ 4 → a^2 - a ≥ 4 - m

theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ (-1 < a ∧ a < 3) ∨ a ≤ -2 := by
  sorry

end range_of_a_l44_44342


namespace sum_of_ages_l44_44243

variable (S M : ℝ)  -- Variables for Sarah's and Matt's ages

-- Conditions
def sarah_older := S = M + 8
def future_age_relationship := S + 10 = 3 * (M - 5)

-- Theorem: The sum of their current ages is 41
theorem sum_of_ages (h1 : sarah_older S M) (h2 : future_age_relationship S M) : S + M = 41 := by
  sorry

end sum_of_ages_l44_44243


namespace function_even_iff_a_eq_one_l44_44198

theorem function_even_iff_a_eq_one (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = a * (3^x) + 1/(3^x)) → 
  (∀ x : ℝ, f x = f (-x)) ↔ a = 1 :=
by
  sorry

end function_even_iff_a_eq_one_l44_44198


namespace ceil_of_neg_frac_squared_l44_44491

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l44_44491


namespace triangle_area_l44_44650

theorem triangle_area (a b c : ℕ) (h₁ : a = 7) (h₂ : b = 24) (h₃ : c = 25) (h₄ : a^2 + b^2 = c^2) : 
  ∃ A : ℕ, A = 84 ∧ A = (a * b) / 2 := by
  sorry

end triangle_area_l44_44650


namespace sixth_term_of_geometric_sequence_l44_44753

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ n

theorem sixth_term_of_geometric_sequence (a : ℝ) (r : ℝ)
  (h1 : a = 243) (h2 : geometric_sequence a r 7 = 32) :
  geometric_sequence a r 5 = 1 :=
by
  sorry

end sixth_term_of_geometric_sequence_l44_44753


namespace isosceles_triangle_x_sum_l44_44261

theorem isosceles_triangle_x_sum :
  ∀ (x : ℝ), (∃ (a b : ℝ), a + b + 60 = 180 ∧ (a = x ∨ b = x) ∧ (a = b ∨ a = 60 ∨ b = 60))
  → (60 + 60 + 60 = 180) :=
by
  intro x h
  sorry

end isosceles_triangle_x_sum_l44_44261


namespace find_k_solution_l44_44840

theorem find_k_solution :
    ∃ k : ℝ, (4 + ∑' n : ℕ, (4 + (n : ℝ)*k) / 5^(n + 1) = 10) ∧ k = 16 :=
begin
  use 16,
  sorry
end

end find_k_solution_l44_44840


namespace sum_of_coeffs_l44_44818

theorem sum_of_coeffs (x y : ℤ) : (x - 3 * y) ^ 20 = 2 ^ 20 := by
  sorry

end sum_of_coeffs_l44_44818


namespace abc_not_8_l44_44559

theorem abc_not_8 (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 :=
sorry

end abc_not_8_l44_44559


namespace total_shaded_area_l44_44987

theorem total_shaded_area (S T U : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 2)
  (h3 : T / U = 2) :
  1 * (S * S) + 4 * (T * T) + 8 * (U * U) = 22.5 := by
sorry

end total_shaded_area_l44_44987


namespace cost_price_is_50_l44_44801

-- Define the conditions
def selling_price : ℝ := 80
def profit_rate : ℝ := 0.6

-- The cost price should be proven to be 50
def cost_price (C : ℝ) : Prop :=
  selling_price = C + (C * profit_rate)

theorem cost_price_is_50 : ∃ C : ℝ, cost_price C ∧ C = 50 := by
  sorry

end cost_price_is_50_l44_44801


namespace log_value_between_integers_l44_44410

theorem log_value_between_integers : (1 : ℤ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < (2 : ℤ) → 1 + 2 = 3 :=
by
  sorry

end log_value_between_integers_l44_44410


namespace simplify_fraction_1_simplify_fraction_2_l44_44740

variables (a b c : ℝ)

theorem simplify_fraction_1 :
  (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c) :=
sorry

theorem simplify_fraction_2 :
  (a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c) :=
sorry

end simplify_fraction_1_simplify_fraction_2_l44_44740


namespace ceil_square_eq_four_l44_44518

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l44_44518


namespace remainder_of_large_number_l44_44635

theorem remainder_of_large_number :
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  last_four_digits % 16 = 9 := 
by
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  show last_four_digits % 16 = 9
  sorry

end remainder_of_large_number_l44_44635


namespace two_digit_sum_reverse_l44_44749

theorem two_digit_sum_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_reverse_l44_44749


namespace ceil_square_of_neg_fraction_l44_44521

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l44_44521


namespace no_possible_k_l44_44983
open Classical

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_possible_k : 
  ∀ (k : ℕ), 
    (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ (p + q = 74) ∧ (x^2 - 74*x + k = 0)) -> False :=
by sorry

end no_possible_k_l44_44983


namespace hyperbola_eccentricity_l44_44554

theorem hyperbola_eccentricity (a b : ℝ) (h : ∃ P : ℝ × ℝ, ∃ A : ℝ × ℝ, ∃ F : ℝ × ℝ, 
  (∃ c : ℝ, F = (c, 0) ∧ A = (-a, 0) ∧ P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1 ∧ 
  (F.fst - P.fst) ^ 2 + P.snd ^ 2 = (F.fst + a) ^ 2 ∧ (F.fst - A.fst) ^ 2 + (F.snd - A.snd) ^ 2 = (F.fst + a) ^ 2 ∧ 
  (P.snd = F.snd) ∧ (abs (F.fst - A.fst) = abs (F.fst - P.fst)))) : 
∃ e : ℝ, e = 2 :=
by
  sorry

end hyperbola_eccentricity_l44_44554


namespace find_rectangle_width_l44_44025

variable (length_square : ℕ) (length_rectangle : ℕ) (width_rectangle : ℕ)

-- Given conditions
def square_side_length := 700
def rectangle_length := 400
def square_perimeter := 4 * square_side_length
def rectangle_perimeter := square_perimeter / 2
def rectangle_perimeter_eq := 2 * length_rectangle + 2 * width_rectangle

-- Statement to prove
theorem find_rectangle_width :
  (square_perimeter = 2800) →
  (rectangle_perimeter = 1400) →
  (length_rectangle = 400) →
  (rectangle_perimeter_eq = 1400) →
  (width_rectangle = 300) :=
by
  intros
  sorry

end find_rectangle_width_l44_44025


namespace quadratic_min_value_l44_44952

theorem quadratic_min_value (k : ℝ) :
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → y = (1/2) * (x - 1) ^ 2 + k) ∧
  (∀ y : ℝ, 3 ≤ y ∧ y ≤ 5 → y ≥ 3) → k = 1 :=
sorry

end quadratic_min_value_l44_44952


namespace unique_real_solution_l44_44485

-- Define the variables
variables (x y : ℝ)

-- State the condition
def equation (x y : ℝ) : Prop :=
  (2^(4*x + 2)) * (4^(2*x + 3)) = (8^(3*x + 4)) * y

-- State the theorem
theorem unique_real_solution (y : ℝ) (h_y : 0 < y) : ∃! x : ℝ, equation x y :=
sorry

end unique_real_solution_l44_44485


namespace similar_triangles_l44_44292

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end similar_triangles_l44_44292


namespace martha_children_l44_44908

noncomputable def num_children (total_cakes : ℕ) (cakes_per_child : ℕ) : ℕ :=
  total_cakes / cakes_per_child

theorem martha_children : num_children 18 6 = 3 := by
  sorry

end martha_children_l44_44908


namespace scaled_polynomial_roots_l44_44221

noncomputable def polynomial_with_scaled_roots : Polynomial ℂ :=
  Polynomial.X^3 - 3*Polynomial.X^2 + 5

theorem scaled_polynomial_roots :
  (∃ r1 r2 r3 : ℂ, polynomial_with_scaled_roots.eval r1 = 0 ∧ polynomial_with_scaled_roots.eval r2 = 0 ∧ polynomial_with_scaled_roots.eval r3 = 0 ∧
  (∃ q : Polynomial ℂ, q = Polynomial.X^3 - 9*Polynomial.X^2 + 135 ∧
  ∀ y, (q.eval y = 0 ↔ (polynomial_with_scaled_roots.eval (y / 3) = 0)))) := sorry

end scaled_polynomial_roots_l44_44221


namespace factorize_x_cube_minus_4x_l44_44997

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l44_44997


namespace scouts_attended_l44_44782

def chocolate_bar_cost : ℝ := 1.50
def total_spent : ℝ := 15
def sections_per_bar : ℕ := 3
def smores_per_scout : ℕ := 2

theorem scouts_attended (bars : ℝ) (sections : ℕ) (smores : ℕ) (scouts : ℕ) :
  bars = total_spent / chocolate_bar_cost →
  sections = bars * sections_per_bar →
  smores = sections →
  scouts = smores / smores_per_scout →
  scouts = 15 :=
by
  intro h1 h2 h3 h4
  sorry

end scouts_attended_l44_44782


namespace intersection_complement_l44_44851

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement : A ∩ (U \ B) = {1, 3} :=
by {
  sorry
}

end intersection_complement_l44_44851


namespace initial_ratio_of_milk_to_water_l44_44362

-- Define the capacity of the can, the amount of milk added, and the ratio when full.
def capacity : ℕ := 72
def additionalMilk : ℕ := 8
def fullRatioNumerator : ℕ := 2
def fullRatioDenominator : ℕ := 1

-- Define the initial amounts of milk and water in the can.
variables (M W : ℕ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  M + W + additionalMilk = capacity ∧
  (M + additionalMilk) * fullRatioDenominator = fullRatioNumerator * W

-- Define the expected result, the initial ratio of milk to water in the can.
def expected_ratio : ℕ × ℕ :=
  (5, 3)

-- The theorem to prove the initial ratio of milk to water given the conditions.
theorem initial_ratio_of_milk_to_water (M W : ℕ) (h : conditions M W) :
  (M / Nat.gcd M W, W / Nat.gcd M W) = expected_ratio :=
sorry

end initial_ratio_of_milk_to_water_l44_44362


namespace rectangle_perimeter_is_30_l44_44424

noncomputable def triangle_DEF_sides := (9 : ℕ, 12 : ℕ, 15 : ℕ)
noncomputable def rectangle_width := (6 : ℕ)

theorem rectangle_perimeter_is_30 :
  let area_triangle_DEF := (triangle_DEF_sides.1 * triangle_DEF_sides.2) / 2
  let rectangle_length := area_triangle_DEF / rectangle_width
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  rectangle_perimeter = 30 := by
  sorry

end rectangle_perimeter_is_30_l44_44424


namespace total_students_went_to_concert_l44_44921

/-- There are 12 buses and each bus took 57 students. We want to find out the total number of students who went to the concert. -/
theorem total_students_went_to_concert (num_buses : ℕ) (students_per_bus : ℕ) (total_students : ℕ) 
  (h1 : num_buses = 12) (h2 : students_per_bus = 57) (h3 : total_students = num_buses * students_per_bus) : 
  total_students = 684 := 
by
  sorry

end total_students_went_to_concert_l44_44921


namespace initial_loss_percentage_l44_44465

theorem initial_loss_percentage 
  (C : ℝ) 
  (h1 : selling_price_one_pencil_20 = 1 / 20)
  (h2 : selling_price_one_pencil_10 = 1 / 10)
  (h3 : C = 1 / (10 * 1.30)) :
  (C - selling_price_one_pencil_20) / C * 100 = 35 :=
by
  sorry

end initial_loss_percentage_l44_44465


namespace least_product_of_distinct_primes_gt_30_l44_44096

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l44_44096


namespace total_cookies_l44_44935

-- Definitions of the conditions
def cookies_in_bag : ℕ := 21
def bags_in_box : ℕ := 4
def boxes : ℕ := 2

-- Theorem stating the total number of cookies
theorem total_cookies : cookies_in_bag * bags_in_box * boxes = 168 := by
  sorry

end total_cookies_l44_44935


namespace pigeon_distance_l44_44722

-- Define the conditions
def pigeon_trip (d : ℝ) (v : ℝ) (wind : ℝ) (time_nowind : ℝ) (time_wind : ℝ) :=
  (2 * d / v = time_nowind) ∧
  (d / (v + wind) + d / (v - wind) = time_wind)

-- Define the theorems to be proven
theorem pigeon_distance : ∃ (d : ℝ), pigeon_trip d 40 10 3.75 4 ∧ d = 75 :=
  by {
  sorry
}

end pigeon_distance_l44_44722


namespace least_product_of_primes_gt_30_l44_44089

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l44_44089


namespace solve_for_x_l44_44064

theorem solve_for_x (x : ℝ) (h : 7 - 2 * x = -3) : x = 5 := by
  sorry

end solve_for_x_l44_44064


namespace direct_proportion_function_l44_44955

-- Definitions of the functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 3 * x^2 + 2

-- Definition of a direct proportion function
def is_direct_proportion (f : ℝ → ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, f x = k * x)

-- Theorem statement
theorem direct_proportion_function : is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l44_44955


namespace lloyd_house_of_cards_l44_44907

theorem lloyd_house_of_cards 
  (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ)
  (h1 : decks = 24) (h2 : cards_per_deck = 78) (h3 : layers = 48) :
  ((decks * cards_per_deck) / layers) = 39 := 
  by
  sorry

end lloyd_house_of_cards_l44_44907


namespace unique_point_graph_eq_l44_44486

theorem unique_point_graph_eq (c : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → x = -1 ∧ y = 6) ↔ c = 39 :=
sorry

end unique_point_graph_eq_l44_44486


namespace total_cleaning_time_l44_44911

def hose_time : ℕ := 10
def shampoos : ℕ := 3
def shampoo_time : ℕ := 15

theorem total_cleaning_time : hose_time + shampoos * shampoo_time = 55 := by
  sorry

end total_cleaning_time_l44_44911


namespace tangent_line_intersects_x_axis_at_9_div_2_l44_44795

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l44_44795


namespace binomial_divisible_by_prime_l44_44063

-- Define the conditions: p is prime and 0 < k < p
variables (p k : ℕ)
variable (hp : Nat.Prime p)
variable (hk : 0 < k ∧ k < p)

-- State that the binomial coefficient \(\binom{p}{k}\) is divisible by \( p \)
theorem binomial_divisible_by_prime
  (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_l44_44063


namespace percentage_relationship_l44_44113

variable {x y z : ℝ}

theorem percentage_relationship (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z :=
by
  sorry

end percentage_relationship_l44_44113


namespace sum_first_six_terms_geometric_sequence_l44_44837

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end sum_first_six_terms_geometric_sequence_l44_44837


namespace greatest_coloring_integer_l44_44898

theorem greatest_coloring_integer (α β : ℝ) (h1 : 1 < α) (h2 : α < β) :
  ∃ r : ℕ, r = 2 ∧ ∀ (f : ℕ → ℕ), ∃ x y : ℕ, x ≠ y ∧ f x = f y ∧ α ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β := 
sorry

end greatest_coloring_integer_l44_44898


namespace quotient_division_l44_44773

noncomputable def poly_division_quotient : Polynomial ℚ :=
  Polynomial.div (9 * Polynomial.X ^ 4 + 8 * Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 - 7 * Polynomial.X + 4) (3 * Polynomial.X ^ 2 + 2 * Polynomial.X + 5)

theorem quotient_division :
  poly_division_quotient = (3 * Polynomial.X ^ 2 - 2 * Polynomial.X + 2) :=
sorry

end quotient_division_l44_44773


namespace fraction_representing_repeating_decimal_l44_44354

theorem fraction_representing_repeating_decimal (x a b : ℕ) (h : x = 35) (h1 : 100 * x - x = 35) 
(h2 : ∃ (a b : ℕ), x = a / b ∧ gcd a b = 1 ∧ a + b = 134) : a + b = 134 := 
sorry

end fraction_representing_repeating_decimal_l44_44354


namespace probability_same_unit_l44_44643

theorem probability_same_unit
  (units : ℕ) (people : ℕ) (same_unit_cases total_cases : ℕ)
  (h_units : units = 4)
  (h_people : people = 2)
  (h_total_cases : total_cases = units * units)
  (h_same_unit_cases : same_unit_cases = units) :
  (same_unit_cases :  ℝ) / total_cases = 1 / 4 :=
by sorry

end probability_same_unit_l44_44643


namespace find_a_parallel_find_a_perpendicular_l44_44568

open Real

def line_parallel (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 = k2

def line_perpendicular (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 * k2 = -1

theorem find_a_parallel (a : ℝ) :
  line_parallel (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 1 ∨ a = 6 :=
by sorry

theorem find_a_perpendicular (a : ℝ) :
  line_perpendicular (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 3 ∨ a = -4 :=
by sorry

end find_a_parallel_find_a_perpendicular_l44_44568


namespace factorize_x_cube_minus_4x_l44_44994

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l44_44994


namespace minimum_value_correct_l44_44712

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a + 3*b = 1 then 1/a + 1/b else 0

theorem minimum_value_correct : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ minimum_value a b = 4 + 4 * Real.sqrt 3 :=
by
  sorry

end minimum_value_correct_l44_44712


namespace cost_ratio_two_pastries_pies_l44_44728

theorem cost_ratio_two_pastries_pies (s p : ℝ) (h1 : 2 * s = 3 * (2 * p)) :
  (s + p) / (2 * p) = 2 :=
by
  sorry

end cost_ratio_two_pastries_pies_l44_44728


namespace range_of_m_l44_44677

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x < y → -3 < x ∧ y < 3 → f x < f y)
  (h2 : ∀ m : ℝ, f (2 * m) < f (m + 1)) : 
  -3/2 < m ∧ m < 1 :=
  sorry

end range_of_m_l44_44677


namespace least_number_to_divisible_sum_l44_44780

-- Define the conditions and variables
def initial_number : ℕ := 1100
def divisor : ℕ := 23
def least_number_to_add : ℕ := 4

-- Statement to prove
theorem least_number_to_divisible_sum :
  ∃ least_n, least_n + initial_number % divisor = divisor ∧ least_n = least_number_to_add :=
  by
    sorry

end least_number_to_divisible_sum_l44_44780


namespace raritet_meets_ferries_l44_44123

theorem raritet_meets_ferries :
  (∀ (n : ℕ), ∃ (ferry_departure : Nat), ferry_departure = n ∧ ferry_departure + 8 = 8) →
  (∀ (m : ℕ), ∃ (raritet_departure : Nat), raritet_departure = m ∧ raritet_departure + 8 = 8) →
  ∃ (total_meetings : Nat), total_meetings = 17 := 
by
  sorry

end raritet_meets_ferries_l44_44123


namespace total_cost_in_dollars_l44_44067

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end total_cost_in_dollars_l44_44067


namespace distance_difference_l44_44979

-- Definition of speeds and time
def speed_alberto : ℕ := 16
def speed_clara : ℕ := 12
def time_hours : ℕ := 5

-- Distance calculation functions
def distance (speed time : ℕ) : ℕ := speed * time

-- Main theorem statement
theorem distance_difference : 
  distance speed_alberto time_hours - distance speed_clara time_hours = 20 :=
by
  sorry

end distance_difference_l44_44979


namespace eq_of_fraction_eq_l44_44211

variable {R : Type*} [Field R]

theorem eq_of_fraction_eq (a b : R) (h : (1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b))) : a = b :=
sorry

end eq_of_fraction_eq_l44_44211


namespace ceil_of_neg_frac_squared_l44_44489

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l44_44489


namespace Clara_sells_third_type_boxes_l44_44469

variable (total_cookies boxes_first boxes_second boxes_third : ℕ)
variable (cookies_per_first cookies_per_second cookies_per_third : ℕ)

theorem Clara_sells_third_type_boxes (h1 : cookies_per_first = 12)
                                    (h2 : boxes_first = 50)
                                    (h3 : cookies_per_second = 20)
                                    (h4 : boxes_second = 80)
                                    (h5 : cookies_per_third = 16)
                                    (h6 : total_cookies = 3320) :
                                    boxes_third = 70 :=
by
  sorry

end Clara_sells_third_type_boxes_l44_44469


namespace Xiao_Ming_min_steps_l44_44964

-- Problem statement: Prove that the minimum number of steps Xiao Ming needs to move from point A to point B is 5,
-- given his movement pattern and the fact that he can reach eight different positions from point C.

def min_steps_from_A_to_B : ℕ :=
  5

theorem Xiao_Ming_min_steps (A B C : Type) (f : A → B → C) : 
  (min_steps_from_A_to_B = 5) :=
by
  sorry

end Xiao_Ming_min_steps_l44_44964


namespace train_cross_signal_pole_time_l44_44450

theorem train_cross_signal_pole_time :
  ∀ (l_t l_p t_p : ℕ), l_t = 450 → l_p = 525 → t_p = 39 → 
  (l_t * t_p) / (l_t + l_p) = 18 := by
  sorry

end train_cross_signal_pole_time_l44_44450


namespace no_integer_roots_l44_44730

theorem no_integer_roots (a b x : ℤ) : 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 ≠ 0 :=
sorry

end no_integer_roots_l44_44730


namespace fish_distribution_l44_44213

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end fish_distribution_l44_44213


namespace oblique_line_plane_angle_range_l44_44405

/-- 
An oblique line intersects the plane at an angle other than a right angle. 
The angle cannot be $0$ radians or $\frac{\pi}{2}$ radians.
-/
theorem oblique_line_plane_angle_range (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) : 
  0 < θ ∧ θ < π / 2 :=
by {
  exact ⟨h₀, h₁⟩
}

end oblique_line_plane_angle_range_l44_44405


namespace bus_stop_time_l44_44274

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℕ) 
(distance : ℕ) (time_without_stoppages time_with_stoppages : ℝ) :
  speed_without_stoppages = 80 ∧ speed_with_stoppages = 40 ∧ distance = 80 ∧
  time_without_stoppages = distance / speed_without_stoppages ∧
  time_with_stoppages = distance / speed_with_stoppages →
  (time_with_stoppages - time_without_stoppages) * 60 = 30 :=
by
  sorry

end bus_stop_time_l44_44274


namespace remaining_number_l44_44249

theorem remaining_number (S : Finset ℕ) (hS : S = Finset.range 51) :
  ∃ n ∈ S, n % 2 = 0 := 
sorry

end remaining_number_l44_44249


namespace cost_price_of_article_l44_44811

noncomputable def cost_price (M : ℝ) : ℝ := 98.68 / 1.25

theorem cost_price_of_article (M : ℝ)
    (h1 : 0.95 * M = 98.68)
    (h2 : 98.68 = 1.25 * cost_price M) :
    cost_price M = 78.944 :=
by sorry

end cost_price_of_article_l44_44811


namespace lowest_score_for_average_l44_44737

theorem lowest_score_for_average
  (score1 score2 score3 : ℕ)
  (h1 : score1 = 81)
  (h2 : score2 = 72)
  (h3 : score3 = 93)
  (max_score : ℕ := 100)
  (desired_average : ℕ := 86)
  (number_of_exams : ℕ := 5) :
  ∃ x y : ℕ, x ≤ 100 ∧ y ≤ 100 ∧ (score1 + score2 + score3 + x + y) / number_of_exams = desired_average ∧ min x y = 84 :=
by
  sorry

end lowest_score_for_average_l44_44737


namespace fourth_person_knight_l44_44781

-- Let P1, P2, P3, and P4 be the statements made by the four people respectively.
def P1 := ∀ x y z w : Prop, x = y ∧ y = z ∧ z = w ∧ w = ¬w
def P2 := ∃! x y z w : Prop, x = true
def P3 := ∀ x y z w : Prop, (x = true ∧ y = true ∧ z = false) ∨ (x = true ∧ y = false ∧ z = true) ∨ (x = false ∧ y = true ∧ z = true)
def P4 := ∀ x : Prop, x = true → x = true

-- Now let's express the requirement of proving that the fourth person is a knight
theorem fourth_person_knight : P4 := by
  sorry

end fourth_person_knight_l44_44781


namespace find_a_and_theta_find_max_min_g_l44_44020

noncomputable def f (x a θ : ℝ) : ℝ := (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

-- Provided conditions
variable (a : ℝ)
variable (θ : ℝ)
variable (is_odd : ∀ x, f x a θ = -f (-x) a θ)
variable (f_pi_over_4 : f ((Real.pi) / 4) a θ = 0)
variable (theta_in_range : 0 < θ ∧ θ < Real.pi)

-- To Prove
theorem find_a_and_theta :
  a = -1 ∧ θ = (Real.pi / 2) :=
sorry

-- Define g(x) and its domain
noncomputable def g (x : ℝ) : ℝ := f x (-1) (Real.pi / 2) + f (x + (Real.pi / 3)) (-1) (Real.pi / 2)

-- Provided domain condition
variable (x_in_domain : 0 ≤ x ∧ x ≤ (Real.pi / 4))

-- To Prove maximum and minimum value of g(x)
theorem find_max_min_g :
  (∀ x, x ∈ Set.Icc (0 : ℝ) (Real.pi / 4) → -((Real.sqrt 3) / 2) ≤ g x ∧ g x ≤ (Real.sqrt 3) / 2)
  ∧ ∃ x_min, g x_min = -((Real.sqrt 3) / 2) ∧ x_min = (Real.pi / 8)
  ∧ ∃ x_max, g x_max = ((Real.sqrt 3) / 2) ∧ x_max = (Real.pi / 4) :=
sorry

end find_a_and_theta_find_max_min_g_l44_44020


namespace one_statement_is_true_l44_44237

theorem one_statement_is_true :
  ∃ (S1 S2 S3 S4 S5 : Prop),
    ((S1 ↔ (¬S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5)) ∧
     (S2 ↔ (¬S1 ∧ ¬S2 ∧ S3 ∧ S4 ∧ ¬S5)) ∧
     (S3 ↔ (¬S1 ∧ S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S4 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S5 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4 ∧ ¬S5))) ∧
    (S2) ∧ (¬S1) ∧ (¬S3) ∧ (¬S4) ∧ (¬S5) :=
by
  -- Proof goes here
  sorry

end one_statement_is_true_l44_44237


namespace machine_a_production_rate_l44_44777

/-
Given:
1. Machine p and machine q are each used to manufacture 440 sprockets.
2. Machine q produces 10% more sprockets per hour than machine a.
3. It takes machine p 10 hours longer to produce 440 sprockets than machine q.

Prove that machine a produces 4 sprockets per hour.
-/

theorem machine_a_production_rate (T A : ℝ) (hq : 440 = T * (1.1 * A)) (hp : 440 = (T + 10) * A) : A = 4 := 
by
  sorry

end machine_a_production_rate_l44_44777


namespace tangent_parallel_l44_44693

noncomputable def f (x : ℝ) := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P = (1, 0)) :
  (∃ x y : ℝ, P = (x, y) ∧ (fderiv ℝ f x) 1 = 3 / 1) ↔ P = (1, 0) :=
by
  sorry

end tangent_parallel_l44_44693


namespace LineDoesNotIntersectParabola_sum_r_s_l44_44218

noncomputable def r : ℝ := -0.6
noncomputable def s : ℝ := 40.6
def Q : ℝ × ℝ := (10, -6)
def line_through_Q_with_slope (m : ℝ) (p : ℝ × ℝ) : ℝ := m * p.1 - 10 * m - 6
def parabola (x : ℝ) : ℝ := 2 * x^2

theorem LineDoesNotIntersectParabola (m : ℝ) :
  r < m ∧ m < s ↔ (m^2 - 4 * 2 * (10 * m + 6) < 0) :=
by sorry

theorem sum_r_s : r + s = 40 :=
by sorry

end LineDoesNotIntersectParabola_sum_r_s_l44_44218


namespace nublian_total_words_l44_44602

-- Define the problem's constants and conditions
def nublian_alphabet_size := 6
def word_length_one := nublian_alphabet_size
def word_length_two := nublian_alphabet_size * nublian_alphabet_size
def word_length_three := nublian_alphabet_size * nublian_alphabet_size * nublian_alphabet_size

-- Define the total number of words
def total_words := word_length_one + word_length_two + word_length_three

-- Main theorem statement
theorem nublian_total_words : total_words = 258 := by
  sorry

end nublian_total_words_l44_44602


namespace smallest_prime_sum_of_three_different_primes_is_19_l44_44434

theorem smallest_prime_sum_of_three_different_primes_is_19 :
  ∃ (p : ℕ), Prime p ∧ p = 19 ∧ (∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → Prime a → Prime b → Prime c → a + b + c = p → p ≥ 19) :=
by
  sorry

end smallest_prime_sum_of_three_different_primes_is_19_l44_44434


namespace area_PST_is_5_l44_44890

noncomputable def area_of_triangle_PST 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : ℝ := 
  5

theorem area_PST_is_5 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : area_of_triangle_PST P Q R S T PQ QR PR PS PT hPQ hQR hPR hPS hPT = 5 :=
sorry

end area_PST_is_5_l44_44890


namespace ceil_square_of_neg_fraction_l44_44523

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l44_44523


namespace can_reach_2021_and_2021_cannot_reach_2022_and_2022_l44_44389

-- Define the allowed action in Lean
def allowed_action (a b : ℕ) : (ℕ × ℕ) :=
  (a + (Nat.digitSum b), b + (Nat.digitSum a))

-- Define the starting and target states
def initial_state : ℕ × ℕ := (1, 2)
def target_state_a : ℕ × ℕ := (2021, 2021)
def target_state_b : ℕ × ℕ := (2022, 2022)

-- Function to check if a sequence of actions can reach the target state
def can_reach_target (initial target : ℕ × ℕ) : Prop :=
  ∃ steps : List (ℕ × ℕ), 
    steps.head? = some initial ∧
    steps.getLast? = some target ∧
    ∀ s t, (s, t) ∈ List.zip steps (List.tail steps) → t = allowed_action s.fst s.snd

-- Problem (a): Can we turn 1 and 2 into 2021 and 2021?
theorem can_reach_2021_and_2021 : can_reach_target initial_state target_state_a := sorry

-- Problem (b): Can we turn 1 and 2 into 2022 and 2022?
theorem cannot_reach_2022_and_2022 : ¬ can_reach_target initial_state target_state_b := sorry

end can_reach_2021_and_2021_cannot_reach_2022_and_2022_l44_44389


namespace minimum_value_correct_l44_44711

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a + 3*b = 1 then 1/a + 1/b else 0

theorem minimum_value_correct : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ minimum_value a b = 4 + 4 * Real.sqrt 3 :=
by
  sorry

end minimum_value_correct_l44_44711


namespace closed_fishing_season_purpose_sustainable_l44_44076

-- Defining the options for the purpose of the closed fishing season
inductive FishingPurpose
| sustainable_development : FishingPurpose
| inspect_fishing_vessels : FishingPurpose
| prevent_red_tides : FishingPurpose
| zoning_management : FishingPurpose

-- Defining rational utilization of resources involving fishing seasons
def rational_utilization (closed_fishing_season: Bool) : FishingPurpose := 
  if closed_fishing_season then FishingPurpose.sustainable_development 
  else FishingPurpose.inspect_fishing_vessels -- fallback for contradiction; shouldn't be used

-- The theorem we want to prove
theorem closed_fishing_season_purpose_sustainable :
  rational_utilization true = FishingPurpose.sustainable_development :=
sorry

end closed_fishing_season_purpose_sustainable_l44_44076


namespace min_value_of_inverse_sum_l44_44710

theorem min_value_of_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : 
  ∃ c : ℝ, c = 4 + 2 * Real.sqrt 3 ∧ ∀x : ℝ, (x = (1 / a + 1 / b)) → x ≥ c :=
by
  sorry

end min_value_of_inverse_sum_l44_44710


namespace ceil_square_eq_four_l44_44514

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l44_44514


namespace two_digit_sum_reverse_l44_44746

theorem two_digit_sum_reverse (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
    (h₃ : 0 ≤ b) (h₄ : b ≤ 9)
    (h₅ : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
    (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_sum_reverse_l44_44746


namespace angle_equality_iff_l44_44738

variables {A A' B B' C C' G : Point}

-- Define the angles as given in conditions
def angle_A'AC (A' A C : Point) : ℝ := sorry
def angle_ABB' (A B B' : Point) : ℝ := sorry
def angle_AC'C (A C C' : Point) : ℝ := sorry
def angle_AA'B (A A' B : Point) : ℝ := sorry

-- Main theorem statement
theorem angle_equality_iff :
  angle_A'AC A' A C = angle_ABB' A B B' ↔ angle_AC'C A C C' = angle_AA'B A A' B :=
sorry

end angle_equality_iff_l44_44738


namespace find_x_for_sin_minus_cos_eq_sqrt2_l44_44828

theorem find_x_for_sin_minus_cos_eq_sqrt2 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
by
  sorry

end find_x_for_sin_minus_cos_eq_sqrt2_l44_44828


namespace original_salary_condition_l44_44931

variable (S: ℝ)

theorem original_salary_condition (h: 1.10 * 1.08 * 0.95 * 0.93 * S = 6270) :
  S = 6270 / (1.10 * 1.08 * 0.95 * 0.93) :=
by
  sorry

end original_salary_condition_l44_44931


namespace problem_statement_l44_44673

noncomputable def f1 (x : ℝ) : ℝ := x ^ 2

noncomputable def f2 (x : ℝ) : ℝ := 8 / x

noncomputable def f (x : ℝ) : ℝ := f1 x + f2 x

theorem problem_statement (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, 
  (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
  (f x1 = f a ∧ f x2 = f a ∧ f x3 = f a) ∧ 
  (x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0) := 
sorry

end problem_statement_l44_44673


namespace outlier_attribute_l44_44312

/-- Define the given attributes of the Dragon -/
def one_eyed := "одноокий"
def two_eared := "двуухий"
def three_tailed := "треххвостый"
def four_legged := "четырехлапый"
def five_spiked := "пятиглый"

/-- Define a predicate to check if an attribute contains doubled letters -/
def has_doubled_letters (s : String) : Bool :=
  let chars := s.toList
  chars.any (fun ch => chars.count ch > 1)

/-- Prove that "четырехлапый" (four-legged) does not fit the pattern of containing doubled letters -/
theorem outlier_attribute : ¬ has_doubled_letters four_legged :=
by
  -- Proof would be inserted here
  sorry

end outlier_attribute_l44_44312


namespace solve_equation_l44_44832

noncomputable def equation (x : ℝ) : ℝ :=
(13 * x - x^2) / (x + 1) * (x + (13 - x) / (x + 1))

theorem solve_equation :
  equation 1 = 42 ∧ equation 6 = 42 ∧ equation (3 + Real.sqrt 2) = 42 ∧ equation (3 - Real.sqrt 2) = 42 :=
by
  sorry

end solve_equation_l44_44832


namespace fraction_of_orange_juice_in_large_container_l44_44631

def total_capacity := 800 -- mL for each pitcher
def orange_juice_first_pitcher := total_capacity / 2 -- 400 mL
def orange_juice_second_pitcher := total_capacity / 4 -- 200 mL
def total_orange_juice := orange_juice_first_pitcher + orange_juice_second_pitcher -- 600 mL
def total_volume := total_capacity + total_capacity -- 1600 mL

theorem fraction_of_orange_juice_in_large_container :
  (total_orange_juice / total_volume) = 3 / 8 :=
by
  sorry

end fraction_of_orange_juice_in_large_container_l44_44631


namespace cos_triple_angle_l44_44184

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l44_44184


namespace scalene_triangles_count_l44_44658

/-- Proving existence of exactly 3 scalene triangles with integer side lengths and perimeter < 13. -/
theorem scalene_triangles_count : 
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    triangles.card = 3 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ triangles → a < b ∧ b < c ∧ a + b + c < 13 :=
sorry

end scalene_triangles_count_l44_44658


namespace total_cost_of_pencils_l44_44069

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end total_cost_of_pencils_l44_44069


namespace point_P_coordinates_l44_44694

/-- The point P where the tangent line to the curve f(x) = x^4 - x
is parallel to the line 3x - y = 0 is (1, 0). -/
theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), 
    let f := λ x : ℝ, x^4 - x in
    -- The tangent at P must have a slope equal to 3, the slope of the line 3x - y = 0.
    let slope_at_P := (deriv f P.1) in
    slope_at_P = 3 ∧ P = (1, 0) :=
sorry

end point_P_coordinates_l44_44694


namespace single_line_points_l44_44648

theorem single_line_points (S : ℝ) (h1 : 6 * S + 4 * (8 * S) = 38000) : S = 1000 :=
by
  sorry

end single_line_points_l44_44648


namespace remainder_when_divided_by_x_minus_3_l44_44003

def p (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7

theorem remainder_when_divided_by_x_minus_3 : p 3 = 52 := 
by
  -- proof here
  sorry

end remainder_when_divided_by_x_minus_3_l44_44003


namespace sum_of_7_more_likely_than_sum_of_8_l44_44085

noncomputable def probability_sum_equals_seven : ℚ := 6 / 36
noncomputable def probability_sum_equals_eight : ℚ := 5 / 36

theorem sum_of_7_more_likely_than_sum_of_8 :
  probability_sum_equals_seven > probability_sum_equals_eight :=
by 
  sorry

end sum_of_7_more_likely_than_sum_of_8_l44_44085


namespace maximal_number_of_coins_l44_44958

noncomputable def largest_number_of_coins (n k : ℕ) : Prop :=
n < 100 ∧ n = 12 * k + 3

theorem maximal_number_of_coins (n k : ℕ) : largest_number_of_coins n k → n = 99 :=
by
  sorry

end maximal_number_of_coins_l44_44958


namespace farmer_profit_l44_44285

def piglet_cost_per_month : Int := 10
def pig_revenue : Int := 300
def num_piglets_sold_early : Int := 3
def num_piglets_sold_late : Int := 3
def early_sale_months : Int := 12
def late_sale_months : Int := 16

def total_profit (num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months : Int) 
  (piglet_cost_per_month pig_revenue : Int) : Int := 
  let early_cost := num_piglets_sold_early * piglet_cost_per_month * early_sale_months
  let late_cost := num_piglets_sold_late * piglet_cost_per_month * late_sale_months
  let total_cost := early_cost + late_cost
  let total_revenue := (num_piglets_sold_early + num_piglets_sold_late) * pig_revenue
  total_revenue - total_cost

theorem farmer_profit : total_profit num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months piglet_cost_per_month pig_revenue = 960 := by
  sorry

end farmer_profit_l44_44285


namespace ceil_square_neg_seven_over_four_l44_44504

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l44_44504


namespace compute_expression_l44_44470

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l44_44470


namespace compute_sum_bk_ck_l44_44707

theorem compute_sum_bk_ck 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 =
                (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + b3*x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -2 := 
sorry

end compute_sum_bk_ck_l44_44707


namespace smallest_solution_eq_l44_44005

noncomputable def smallest_solution := 4 - Real.sqrt 3

theorem smallest_solution_eq (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 3 / (x - 4)) → x = smallest_solution :=
sorry

end smallest_solution_eq_l44_44005


namespace pipeA_filling_time_l44_44807

noncomputable def pipeA_fill_time := 60

def pipeB_fill_time := 40
def total_fill_time := 30
def half_fill_time := total_fill_time / 2

theorem pipeA_filling_time :
  let t := pipeA_fill_time in
  let rateB := 1 / pipeB_fill_time in
  let rateA := 1 / t in
  let rateA_B := rateA + rateB in
  rateB * half_fill_time + rateA_B * half_fill_time = 1 →
  t = 60 :=
by
  sorry

end pipeA_filling_time_l44_44807


namespace system_solutions_l44_44116

theorem system_solutions (x a : ℝ) (h1 : a = -3*x^2 + 5*x - 2) (h2 : (x + 2) * a = 4 * (x^2 - 1)) (hx : x ≠ -2) :
  (x = 0 ∧ a = -2) ∨ (x = 1 ∧ a = 0) ∨ (x = -8/3 ∧ a = -110/3) :=
  sorry

end system_solutions_l44_44116


namespace range_of_m_l44_44055

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0 
def neg_q_sufficient_for_neg_p (m : ℝ) : Prop :=
  ∀ x : ℝ, p x → q x m

theorem range_of_m (m : ℝ) : neg_q_sufficient_for_neg_p m → m ≥ 9 :=
by
  sorry

end range_of_m_l44_44055


namespace angle_bisector_slope_l44_44070

theorem angle_bisector_slope 
  (m1 m2 : ℝ) 
  (h1 : m1 = 2) 
  (h2 : m2 = 4) 
  : (∀ m : ℝ, y = m * x) → m = (sqrt 21 - 6) / 7 :=
by
  sorry

end angle_bisector_slope_l44_44070


namespace ratio_depth_to_height_l44_44463

noncomputable def height_ron : ℝ := 12
noncomputable def depth_water : ℝ := 60

theorem ratio_depth_to_height : depth_water / height_ron = 5 := by
  sorry

end ratio_depth_to_height_l44_44463


namespace sum_first_six_terms_geometric_seq_l44_44836

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end sum_first_six_terms_geometric_seq_l44_44836


namespace principal_amount_l44_44358

-- Define the conditions and required result
theorem principal_amount
  (P R T : ℝ)
  (hR : R = 0.5)
  (h_diff : (P * R * (T + 4) / 100) - (P * R * T / 100) = 40) :
  P = 2000 :=
  sorry

end principal_amount_l44_44358


namespace largest_4digit_div_by_35_l44_44641

theorem largest_4digit_div_by_35 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (35 ∣ n) ∧ (∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (35 ∣ m) → m ≤ n) ∧ n = 9985 :=
by
  sorry

end largest_4digit_div_by_35_l44_44641


namespace hexagon_coloring_l44_44928

def hex_colorings : ℕ := 2

theorem hexagon_coloring :
  ∃ c : ℕ, c = hex_colorings := by
  sorry

end hexagon_coloring_l44_44928


namespace count_positive_area_triangles_l44_44866

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l44_44866


namespace flagpole_break_height_l44_44289

theorem flagpole_break_height (h h_break distance : ℝ) (h_pos : 0 < h) (h_break_pos : 0 < h_break)
  (h_flagpole : h = 8) (d_distance : distance = 3) (h_relationship : (h_break ^ 2 + distance^2) = (h - h_break)^2) :
  h_break = Real.sqrt 3 :=
  sorry

end flagpole_break_height_l44_44289


namespace distance_last_pair_of_trees_l44_44880

theorem distance_last_pair_of_trees 
  (yard_length : ℝ := 1200)
  (num_trees : ℕ := 117)
  (initial_distance : ℝ := 5)
  (distance_increment : ℝ := 2) :
  let num_distances := num_trees - 1
  let last_distance := initial_distance + (num_distances - 1) * distance_increment
  last_distance = 235 := by 
  sorry

end distance_last_pair_of_trees_l44_44880


namespace greatest_integer_b_not_in_range_l44_44634

theorem greatest_integer_b_not_in_range :
  let f (x : ℝ) (b : ℝ) := x^2 + b*x + 20
  let g (x : ℝ) (b : ℝ) := x^2 + b*x + 24
  (¬ (∃ (x : ℝ), g x b = 0)) → (b = 9) :=
by
  sorry

end greatest_integer_b_not_in_range_l44_44634


namespace solve_fraction_eq_l44_44617

theorem solve_fraction_eq (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_eq_l44_44617


namespace Xiaobing_jumps_189_ropes_per_minute_l44_44488

-- Define conditions and variables
variable (x : ℕ) -- The number of ropes Xiaohan jumps per minute

-- Conditions:
-- 1. Xiaobing jumps x + 21 ropes per minute
-- 2. Time taken for Xiaobing to jump 135 ropes is the same as the time taken for Xiaohan to jump 120 ropes

theorem Xiaobing_jumps_189_ropes_per_minute (h : 135 * x = 120 * (x + 21)) :
    x + 21 = 189 :=
by
  sorry -- Proof is not required as per instructions

end Xiaobing_jumps_189_ropes_per_minute_l44_44488


namespace range_of_m_l44_44874

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by
  sorry

end range_of_m_l44_44874


namespace evaluate_expression_l44_44666

-- Define the conditions
def two_pow_nine : ℕ := 2 ^ 9
def neg_one_pow_eight : ℤ := (-1) ^ 8

-- Define the proof statement
theorem evaluate_expression : two_pow_nine + neg_one_pow_eight = 513 := 
by
  sorry

end evaluate_expression_l44_44666


namespace peter_drew_8_pictures_l44_44392

theorem peter_drew_8_pictures : 
  ∃ (P : ℕ), ∀ (Q R : ℕ), Q = P + 20 → R = 5 → R + P + Q = 41 → P = 8 :=
by
  sorry

end peter_drew_8_pictures_l44_44392


namespace find_first_number_l44_44620

theorem find_first_number (x : ℝ) : (10 + 70 + 28) / 3 = 36 →
  (x + 40 + 60) / 3 = 40 →
  x = 20 := 
by
  intros h_avg_old h_avg_new
  sorry

end find_first_number_l44_44620


namespace train_length_55_meters_l44_44948

noncomputable def V_f := 47 * 1000 / 3600 -- Speed of the faster train in m/s
noncomputable def V_s := 36 * 1000 / 3600 -- Speed of the slower train in m/s
noncomputable def t := 36 -- Time in seconds

theorem train_length_55_meters (L : ℝ) (Vf : ℝ := V_f) (Vs : ℝ := V_s) (time : ℝ := t) :
  (2 * L = (Vf - Vs) * time) → L = 55 :=
by
  sorry

end train_length_55_meters_l44_44948


namespace chi_squared_test_expectation_correct_distribution_table_correct_l44_44977

-- Given data for the contingency table
def male_good := 52
def male_poor := 8
def female_good := 28
def female_poor := 12
def total := 100

-- Define the $\chi^2$ calculation
def chi_squared_value : ℚ :=
  (total * (male_good * female_poor - male_poor * female_good)^2) / 
  ((male_good + male_poor) * (female_good + female_poor) * (male_good + female_good) * (male_poor + female_poor))

-- The $\chi^2$ value to compare against for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Prove that $\chi^2$ value is less than the critical value for 99% confidence
theorem chi_squared_test :
  chi_squared_value < critical_value_99 :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Probability data and expectations for successful shots
def prob_male_success : ℚ := 2 / 3
def prob_female_success : ℚ := 1 / 2

-- Probabilities of the number of successful shots
def prob_X_0 : ℚ := (1 - prob_male_success) ^ 2 * (1 - prob_female_success)
def prob_X_1 : ℚ := 2 * prob_male_success * (1 - prob_male_success) * (1 - prob_female_success) +
                    (1 - prob_male_success) ^ 2 * prob_female_success
def prob_X_2 : ℚ := prob_male_success ^ 2 * (1 - prob_female_success) +
                    2 * prob_male_success * (1 - prob_male_success) * prob_female_success
def prob_X_3 : ℚ := prob_male_success ^ 2 * prob_female_success

def expectation_X : ℚ :=
  0 * prob_X_0 + 
  1 * prob_X_1 + 
  2 * prob_X_2 + 
  3 * prob_X_3

-- The expected value of X
def expected_value_X : ℚ := 11 / 6

-- Prove the expected value is as calculated
theorem expectation_correct :
  expectation_X = expected_value_X :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Define the distribution table based on calculated probabilities
def distribution_table : List (ℚ × ℚ) :=
  [(0, prob_X_0), (1, prob_X_1), (2, prob_X_2), (3, prob_X_3)]

-- The correct distribution table
def correct_distribution_table : List (ℚ × ℚ) :=
  [(0, 1 / 18), (1, 5 / 18), (2, 4 / 9), (3, 2 / 9)]

-- Prove the distribution table is as calculated
theorem distribution_table_correct :
  distribution_table = correct_distribution_table :=
by
  -- Sorry to skip the proof as instructed
  sorry

end chi_squared_test_expectation_correct_distribution_table_correct_l44_44977


namespace line_does_not_pass_through_fourth_quadrant_l44_44756

-- Definitions of conditions
variables {a b c x y : ℝ}

-- The mathematical statement to be proven
theorem line_does_not_pass_through_fourth_quadrant
  (h1 : a * b < 0) (h2 : b * c < 0) :
  ¬ (∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_through_fourth_quadrant_l44_44756


namespace hit_target_at_least_once_l44_44104

variables (P : Set Bool → ℝ)
variables (A B : Set Bool)

def prob_A := P A = 0.6
def prob_B := P B = 0.5

theorem hit_target_at_least_once : P (A ∪ B) = 0.8 :=
by
  have complement_A : P (Set.univ \ A) = 0.4 := sorry
  have complement_B : P (Set.univ \ B) = 0.5 := sorry
  have neither_hit : P ((Set.univ \ A) ∩ (Set.univ \ B)) = 0.2 := sorry
  have target_at_least_once : P (A ∪ B) = 1 - 0.2 := sorry
  exact target_at_least_once

end hit_target_at_least_once_l44_44104


namespace shortest_wire_length_l44_44768

theorem shortest_wire_length (d1 d2 : ℝ) (r1 r2 : ℝ) (t : ℝ) :
  d1 = 8 ∧ d2 = 20 ∧ r1 = 4 ∧ r2 = 10 ∧ t = 8 * Real.sqrt 10 + 17.4 * Real.pi → 
  ∃ l : ℝ, l = t :=
by 
  sorry

end shortest_wire_length_l44_44768


namespace cos_2_alpha_plus_beta_eq_l44_44167

variable (α β : ℝ)

def tan_roots_of_quadratic (x : ℝ) : Prop := x^2 + 5 * x - 6 = 0

theorem cos_2_alpha_plus_beta_eq :
  ∀ α β : ℝ, tan_roots_of_quadratic (Real.tan α) ∧ tan_roots_of_quadratic (Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 :=
by
  intros
  sorry

end cos_2_alpha_plus_beta_eq_l44_44167


namespace range_of_a_l44_44347

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) is an odd and monotonically increasing function, to be defined later.

noncomputable def g (x a : ℝ) : ℝ :=
  f (x^2) + f (a - 2 * |x|)

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0 ∧ g x4 a = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l44_44347


namespace find_larger_number_l44_44639

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 :=
by
  -- proof to be filled
  sorry

end find_larger_number_l44_44639


namespace find_minimum_value_max_value_when_g_half_l44_44324

noncomputable def f (a x : ℝ) : ℝ := 1 - 2 * a - 2 * a * (Real.cos x) - 2 * (Real.sin x) ^ 2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a <= 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem find_minimum_value (a : ℝ) :
  ∃ g_val, g_val = g a :=
  sorry

theorem max_value_when_g_half : 
  g (-1) = 1 / 2 →
  ∃ max_val, max_val = (max (f (-1) π) (f (-1) 0)) :=
  sorry

end find_minimum_value_max_value_when_g_half_l44_44324


namespace age_problem_contradiction_l44_44970

theorem age_problem_contradiction (C1 C2 : ℕ) (k : ℕ)
  (h1 : 15 = k * (C1 + C2))
  (h2 : 20 = 2 * (C1 + 5 + C2 + 5)) : false :=
by
  sorry

end age_problem_contradiction_l44_44970


namespace correct_equation_l44_44940

variable (x : ℝ)
axiom area_eq_720 : x * (x - 6) = 720

theorem correct_equation : x * (x - 6) = 720 := by
  exact area_eq_720

end correct_equation_l44_44940


namespace wendy_total_sales_l44_44431

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end wendy_total_sales_l44_44431


namespace number_of_days_A_to_finish_remaining_work_l44_44119

theorem number_of_days_A_to_finish_remaining_work
  (A_days : ℕ) (B_days : ℕ) (B_work_days : ℕ) : 
  A_days = 9 → 
  B_days = 15 → 
  B_work_days = 10 → 
  ∃ d : ℕ, d = 3 :=
by 
  intros hA hB hBw
  sorry

end number_of_days_A_to_finish_remaining_work_l44_44119


namespace amount_paid_l44_44228

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l44_44228


namespace ceiling_of_square_of_neg_7_over_4_is_4_l44_44512

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l44_44512


namespace product_of_terms_eq_72_l44_44052

theorem product_of_terms_eq_72
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 12) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 72 :=
by
  sorry

end product_of_terms_eq_72_l44_44052


namespace max_g_value_on_interval_l44_44329

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end max_g_value_on_interval_l44_44329


namespace parabola_vertex_sum_l44_44072

theorem parabola_vertex_sum (p q r : ℝ) (h1 : ∀ x : ℝ, x = p * (x - 3)^2 + 2 → y) (h2 : p * (1 - 3)^2 + 2 = 6) :
  p + q + r = 6 :=
sorry

end parabola_vertex_sum_l44_44072


namespace yellow_area_is_1_5625_percent_l44_44804

def square_flag_area (s : ℝ) : ℝ := s ^ 2

def cross_yellow_occupies_25_percent (s : ℝ) (w : ℝ) : Prop :=
  4 * w * s - 4 * w ^ 2 = 0.25 * s ^ 2

def yellow_area (s w : ℝ) : ℝ := 4 * w ^ 2

def percent_of_flag_area_is_yellow (s w : ℝ) : Prop :=
  yellow_area s w = 0.015625 * s ^ 2

theorem yellow_area_is_1_5625_percent (s w : ℝ) (h1: cross_yellow_occupies_25_percent s w) : 
  percent_of_flag_area_is_yellow s w :=
by sorry

end yellow_area_is_1_5625_percent_l44_44804


namespace fran_travel_time_l44_44896

theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time joann_distance : ℝ) :
  joann_speed = 15 → joann_time = 4 → joann_distance = joann_speed * joann_time →
  fran_speed = 20 → fran_time = joann_distance / fran_speed →
  fran_time = 3 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end fran_travel_time_l44_44896


namespace problem_2_8_3_4_7_2_2_l44_44476

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l44_44476


namespace average_of_shifted_sample_l44_44352

theorem average_of_shifted_sample (x1 x2 x3 : ℝ) (hx_avg : (x1 + x2 + x3) / 3 = 40) (hx_var : ((x1 - 40) ^ 2 + (x2 - 40) ^ 2 + (x3 - 40) ^ 2) / 3 = 1) : 
  ((x1 + 40) + (x2 + 40) + (x3 + 40)) / 3 = 80 :=
sorry

end average_of_shifted_sample_l44_44352


namespace trajectory_passes_quadrants_l44_44586

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 4

-- Define the condition for a point to belong to the first quadrant
def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Define the condition for a point to belong to the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- State the theorem that the trajectory of point P passes through the first and second quadrants
theorem trajectory_passes_quadrants :
  (∃ x y : ℝ, circle_equation x y ∧ in_first_quadrant x y) ∧
  (∃ x y : ℝ, circle_equation x y ∧ in_second_quadrant x y) :=
sorry

end trajectory_passes_quadrants_l44_44586


namespace production_volume_l44_44282

/-- 
A certain school's factory produces 200 units of a certain product this year.
It is planned to increase the production volume by the same percentage \( x \)
over the next two years such that the total production volume over three years is 1400 units.
The goal is to prove that the correct equation for this scenario is:
200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400.
-/
theorem production_volume (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 := 
sorry

end production_volume_l44_44282


namespace johns_number_l44_44592

theorem johns_number (n : ℕ) :
  64 ∣ n ∧ 45 ∣ n ∧ 1000 < n ∧ n < 3000 -> n = 2880 :=
by
  sorry

end johns_number_l44_44592


namespace sum_lent_is_1100_l44_44802

variables (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)

-- Given conditions
def interest_formula := I = P * r * t
def interest_difference := I = P - 572

-- Values
def rate := r = 0.06
def time := t = 8

theorem sum_lent_is_1100 : P = 1100 :=
by
  -- Definitions and axioms
  sorry

end sum_lent_is_1100_l44_44802


namespace cylinder_surface_area_l44_44803

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area (h r : ℕ) (h_eq : h = 8) (r_eq : r = 3) :
  2 * Real.pi * r * h + 2 * Real.pi * r ^ 2 = 66 * Real.pi := by
  sorry

end cylinder_surface_area_l44_44803


namespace cost_of_coffee_A_per_kg_l44_44976

theorem cost_of_coffee_A_per_kg (x : ℝ) :
  (240 * x + 240 * 12 = 480 * 11) → x = 10 :=
by
  intros h
  sorry

end cost_of_coffee_A_per_kg_l44_44976


namespace cos_triple_angle_l44_44187

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l44_44187


namespace ceil_square_of_neg_fraction_l44_44522

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l44_44522


namespace least_number_with_remainders_l44_44445

theorem least_number_with_remainders :
  ∃ x, (x ≡ 4 [MOD 5]) ∧ (x ≡ 4 [MOD 6]) ∧ (x ≡ 4 [MOD 9]) ∧ (x ≡ 4 [MOD 18]) ∧ x = 94 := 
by 
  sorry

end least_number_with_remainders_l44_44445


namespace polynomial_evaluation_l44_44574

-- Given the value of y
def y : ℤ := 4

-- Our goal is to prove this mathematical statement
theorem polynomial_evaluation : (3 * (y ^ 2) + 4 * y + 2 = 66) := 
by 
    sorry

end polynomial_evaluation_l44_44574


namespace alice_spent_19_percent_l44_44464

variable (A : ℝ) (x : ℝ)
variable (h1 : ∃ (B : ℝ), B = 0.9 * A) -- Bob's initial amount in terms of Alice's initial amount
variable (h2 : A - x = 0.81 * A) -- Alice's remaining amount after spending x

theorem alice_spent_19_percent (h1 : ∃ (B : ℝ), B = 0.9 * A) (h2 : A - x = 0.81 * A) : (x / A) * 100 = 19 := by
  sorry

end alice_spent_19_percent_l44_44464


namespace remainder_sum_abc_mod5_l44_44193

theorem remainder_sum_abc_mod5 (a b c : ℕ) (h1 : a < 5) (h2 : b < 5) (h3 : c < 5)
  (h4 : a * b * c ≡ 1 [MOD 5])
  (h5 : 4 * c ≡ 3 [MOD 5])
  (h6 : 3 * b ≡ 2 + b [MOD 5]) :
  (a + b + c) % 5 = 1 :=
  sorry

end remainder_sum_abc_mod5_l44_44193


namespace factorize_expression_l44_44993

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l44_44993


namespace geometric_sequence_third_term_l44_44879

theorem geometric_sequence_third_term (a1 a5 a3 : ℕ) (r : ℝ) 
  (h1 : a1 = 4) 
  (h2 : a5 = 1296) 
  (h3 : a5 = a1 * r^4)
  (h4 : a3 = a1 * r^2) : 
  a3 = 36 := 
by 
  sorry

end geometric_sequence_third_term_l44_44879


namespace minimally_intersecting_remainder_l44_44483

noncomputable def count_minimally_intersecting_triples (s : Finset (Fin 7)) : Nat := sorry

theorem minimally_intersecting_remainder (s : Finset (Fin 7)) :
  let N := count_minimally_intersecting_triples s
  N % 1000 = 760 := sorry

end minimally_intersecting_remainder_l44_44483


namespace number_of_eggs_l44_44242

-- Define the conditions as assumptions
variables (marbles : ℕ) (eggs : ℕ)
variables (eggs_A eggs_B eggs_C : ℕ)
variables (marbles_A marbles_B marbles_C : ℕ)

-- Conditions from the problem
axiom eggs_total : marbles = 4
axiom marbles_total : eggs = 15
axiom eggs_groups : eggs_A ≠ eggs_B ∧ eggs_B ≠ eggs_C ∧ eggs_A ≠ eggs_C
axiom marbles_diff1 : marbles_B - marbles_A = eggs_B
axiom marbles_diff2 : marbles_C - marbles_B = eggs_C

-- Prove that the number of eggs in each group is as specified in the answer
theorem number_of_eggs :
  eggs_A = 12 ∧ eggs_B = 1 ∧ eggs_C = 2 :=
by {
  sorry
}

end number_of_eggs_l44_44242


namespace given_equation_roots_sum_cubes_l44_44598

theorem given_equation_roots_sum_cubes (r s t : ℝ) 
    (h1 : 6 * r ^ 3 + 1506 * r + 3009 = 0)
    (h2 : 6 * s ^ 3 + 1506 * s + 3009 = 0)
    (h3 : 6 * t ^ 3 + 1506 * t + 3009 = 0)
    (sum_roots : r + s + t = 0) :
    (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1504.5 := 
by 
  -- proof omitted
  sorry

end given_equation_roots_sum_cubes_l44_44598


namespace inclination_angle_range_l44_44021

theorem inclination_angle_range (k : ℝ) (α : ℝ) (h1 : -1 ≤ k) (h2 : k < 1)
  (h3 : k = Real.tan α) (h4 : 0 ≤ α) (h5 : α < 180) :
  (0 ≤ α ∧ α < 45) ∨ (135 ≤ α ∧ α < 180) :=
sorry

end inclination_angle_range_l44_44021


namespace ceiling_of_square_frac_l44_44495

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l44_44495


namespace wendy_total_sales_l44_44432

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end wendy_total_sales_l44_44432


namespace distance_james_rode_l44_44042

def speed : ℝ := 80.0
def time : ℝ := 16.0
def distance : ℝ := speed * time

theorem distance_james_rode :
  distance = 1280.0 :=
by
  -- to show the theorem is sane
  sorry

end distance_james_rode_l44_44042


namespace magnitude_of_complex_l44_44991

open Complex

theorem magnitude_of_complex : abs (Complex.mk (3/4) (-5/6)) = Real.sqrt (181) / 12 :=
by
  sorry

end magnitude_of_complex_l44_44991


namespace ceil_square_of_neg_seven_fourths_l44_44534

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l44_44534


namespace sqrt_simplification_l44_44027

theorem sqrt_simplification (m : ℝ) (h : m < 1) : real.sqrt (m^2 - 2*m + 1) = 1 - m := 
by sorry

end sqrt_simplification_l44_44027


namespace find_the_number_l44_44842

theorem find_the_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 8) : x = 32 := by
  sorry

end find_the_number_l44_44842


namespace largest_club_size_is_four_l44_44583

variable {Player : Type} -- Assume Player is a type

-- Definition of the lesson-taking relation
variable (takes_lessons_from : Player → Player → Prop)

-- Club conditions
def club_conditions (A B C : Player) : Prop :=
  (takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ takes_lessons_from C A)

theorem largest_club_size_is_four :
  ∀ (club : Finset Player),
  (∀ (A B C : Player), A ≠ B → B ≠ C → C ≠ A → A ∈ club → B ∈ club → C ∈ club → club_conditions takes_lessons_from A B C) →
  club.card ≤ 4 :=
sorry

end largest_club_size_is_four_l44_44583


namespace Mitch_hourly_rate_l44_44909

theorem Mitch_hourly_rate :
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  weekly_earnings / total_hours = 3 :=
by
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  sorry

end Mitch_hourly_rate_l44_44909


namespace probability_bons_wins_even_rolls_l44_44827
noncomputable def probability_of_Bons_winning (p6 : ℚ) (p_not6 : ℚ) : ℚ := 
  let r := p_not6^2
  let a := p_not6 * p6
  a / (1 - r)

theorem probability_bons_wins_even_rolls : 
  let p6 := (1 : ℚ) / 6
  let p_not6 := (5 : ℚ) / 6
  probability_of_Bons_winning p6 p_not6 = (5 : ℚ) / 11 := 
  sorry

end probability_bons_wins_even_rolls_l44_44827


namespace player2_wins_l44_44256

-- Definitions for the initial conditions and game rules
def initial_piles := [10, 15, 20]
def split_rule (piles : List ℕ) (move : ℕ → ℕ × ℕ) : List ℕ :=
  let (pile1, pile2) := move (piles.head!)
  (pile1 :: pile2 :: piles.tail!)

-- Winning condition proof
theorem player2_wins :
  ∀ piles : List ℕ, piles = [10, 15, 20] →
  (∀ move_count : ℕ, move_count = 42 →
  (move_count > 0 ∧ ¬ ∃ split : ℕ → ℕ × ℕ, move_count % 2 = 1)) :=
by
  intro piles hpiles
  intro move_count hmove_count
  sorry

end player2_wins_l44_44256


namespace not_perfect_square_l44_44956

theorem not_perfect_square (h1 : ∃ x : ℝ, x^2 = 1 ^ 2018) 
                           (h2 : ¬ ∃ x : ℝ, x^2 = 2 ^ 2019)
                           (h3 : ∃ x : ℝ, x^2 = 3 ^ 2020)
                           (h4 : ∃ x : ℝ, x^2 = 4 ^ 2021)
                           (h5 : ∃ x : ℝ, x^2 = 6 ^ 2022) : 
  2 ^ 2019 ≠ x^2 := 
sorry

end not_perfect_square_l44_44956


namespace each_person_tip_l44_44375

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l44_44375


namespace tangent_line_intersection_x_axis_l44_44792

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l44_44792


namespace least_product_of_distinct_primes_over_30_l44_44098

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l44_44098


namespace no_valid_two_digit_N_exists_l44_44850

def is_two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ (n : ℕ), n ^ 3 = x

def reverse_digits (N : ℕ) : ℕ :=
  match N / 10, N % 10 with
  | a, b => 10 * b + a

theorem no_valid_two_digit_N_exists : ∀ N : ℕ,
  is_two_digit_number N →
  (is_perfect_cube (N - reverse_digits N) ∧ (N - reverse_digits N) ≠ 27) → false :=
by sorry

end no_valid_two_digit_N_exists_l44_44850


namespace find_f_at_3_l44_44754

variable (f : ℝ → ℝ)

-- Conditions
-- 1. f is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
-- 2. f(-1) = 1/2
axiom f_neg_one : f (-1) = 1 / 2
-- 3. f(x+2) = f(x) + 2 for all x
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + 2

-- The target value to prove
theorem find_f_at_3 : f 3 = 3 / 2 := by
  sorry

end find_f_at_3_l44_44754


namespace ceil_of_neg_frac_squared_l44_44493

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l44_44493


namespace probability_of_shaded_triangle_l44_44363

theorem probability_of_shaded_triangle 
  (triangles : Finset ℝ) 
  (shaded_triangles : Finset ℝ)
  (h1 : triangles = {1, 2, 3, 4, 5})
  (h2 : shaded_triangles = {1, 4})
  : (shaded_triangles.card / triangles.card) = 2 / 5 := 
  by
  sorry

end probability_of_shaded_triangle_l44_44363


namespace inequality_proof_l44_44118

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by sorry

end inequality_proof_l44_44118


namespace tangent_circles_x_intersect_l44_44788

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l44_44788


namespace find_principal_amount_l44_44110

theorem find_principal_amount 
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ)
  (hA : A = 3087) (hr : r = 0.05) (hn : n = 1) (ht : t = 2)
  (hcomp : A = P * (1 + r / n)^(n * t)) :
  P = 2800 := 
  by sorry

end find_principal_amount_l44_44110


namespace OH_over_ON_eq_2_no_other_common_points_l44_44889

noncomputable def coordinates (t p : ℝ) : ℝ × ℝ :=
  (t^2 / (2 * p), t)

noncomputable def symmetric_point (M P : ℝ × ℝ) : ℝ × ℝ :=
  let (xM, yM) := M;
  let (xP, yP) := P;
  (2 * xP - xM, 2 * yP - yM)

noncomputable def line_ON (p t : ℝ) : ℝ → ℝ :=
  λ x => (p / t) * x

noncomputable def line_MH (t p : ℝ) : ℝ → ℝ :=
  λ x => (p / (2 * t)) * x + t

noncomputable def point_H (t p : ℝ) : ℝ × ℝ :=
  (2 * t^2 / p, 2 * t)

theorem OH_over_ON_eq_2
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  (H.snd) / (N.snd) = 2 := by
  sorry

theorem no_other_common_points
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  ∀ y, (y ≠ H.snd → ¬ ∃ x, line_MH t p x = y ∧ y^2 = 2 * p * x) := by 
  sorry

end OH_over_ON_eq_2_no_other_common_points_l44_44889


namespace no_prime_divisible_by_45_l44_44177

theorem no_prime_divisible_by_45 : ∀ (p : ℕ), Prime p → ¬ (45 ∣ p) :=
by {
  intros p h_prime h_div,
  have h_factors := Nat.factors_unique,
  sorry
}

end no_prime_divisible_by_45_l44_44177


namespace course_gender_relationship_expected_value_X_l44_44676

-- Define the data based on the problem statement
def total_students := 450
def total_boys := 250
def total_girls := 200
def boys_course_b := 150
def girls_course_a := 50
def boys_course_a := total_boys - boys_course_b -- 100
def girls_course_b := total_girls - girls_course_a -- 150

-- Test statistic for independence (calculated)
def chi_squared := 22.5
def critical_value := 10.828

-- Null hypothesis for independence
def H0 := "The choice of course is independent of gender"

-- part 1: proving independence rejection based on chi-squared value
theorem course_gender_relationship : chi_squared > critical_value :=
  by sorry

-- For part 2, stratified sampling and expected value
-- Define probabilities and expected value
def P_X_0 := 1/6
def P_X_1 := 1/2
def P_X_2 := 3/10
def P_X_3 := 1/30

def expected_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- part 2: proving expected value E(X) calculation
theorem expected_value_X : expected_X = 6/5 :=
  by sorry

end course_gender_relationship_expected_value_X_l44_44676


namespace jack_jill_total_difference_l44_44406

theorem jack_jill_total_difference :
  let original_price := 90.00
  let discount_rate := 0.20
  let tax_rate := 0.06

  -- Jack's calculation
  let jack_total :=
    let price_with_tax := original_price * (1 + tax_rate)
    price_with_tax * (1 - discount_rate)
  
  -- Jill's calculation
  let jill_total :=
    let discounted_price := original_price * (1 - discount_rate)
    discounted_price * (1 + tax_rate)

  -- Equality check
  jack_total = jill_total := 
by
  -- Place the proof here
  sorry

end jack_jill_total_difference_l44_44406


namespace complex_expression_ab_l44_44551

open Complex

theorem complex_expression_ab :
  ∀ (a b : ℝ), (2 + 3 * I) / I = a + b * I → a * b = 6 :=
by
  intros a b h
  sorry

end complex_expression_ab_l44_44551


namespace exists_abc_gcd_equation_l44_44729

theorem exists_abc_gcd_equation (n : ℕ) : ∃ a b c : ℤ, n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := sorry

end exists_abc_gcd_equation_l44_44729


namespace find_n_coordinates_l44_44855

variables {a b : ℝ}

def is_perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_n_coordinates (n : ℝ × ℝ) (h1 : is_perpendicular (a, b) n) (h2 : same_magnitude (a, b) n) :
  n = (b, -a) :=
sorry

end find_n_coordinates_l44_44855


namespace AM_GM_inequality_example_l44_44341

theorem AM_GM_inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 6) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 1 / 2 :=
by
  sorry

end AM_GM_inequality_example_l44_44341


namespace negation_proof_l44_44075

theorem negation_proof :
  (¬ ∃ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∀ x : ℝ, x > 1 ∧ x^2 ≤ 4) :=
by
  sorry

end negation_proof_l44_44075


namespace workshop_total_workers_l44_44276

noncomputable def average_salary_of_all (W : ℕ) : ℝ := 8000
noncomputable def average_salary_of_technicians : ℝ := 12000
noncomputable def average_salary_of_non_technicians : ℝ := 6000

theorem workshop_total_workers
    (W : ℕ)
    (T : ℕ := 7)
    (N : ℕ := W - T)
    (h1 : (T + N) = W)
    (h2 : average_salary_of_all W = 8000)
    (h3 : average_salary_of_technicians = 12000)
    (h4 : average_salary_of_non_technicians = 6000)
    (h5 : (7 * 12000) + (N * 6000) = (7 + N) * 8000) :
  W = 21 :=
by
  sorry


end workshop_total_workers_l44_44276


namespace rectangle_perimeter_is_30_l44_44425

noncomputable def triangle_DEF_sides := (9 : ℕ, 12 : ℕ, 15 : ℕ)
noncomputable def rectangle_width := (6 : ℕ)

theorem rectangle_perimeter_is_30 :
  let area_triangle_DEF := (triangle_DEF_sides.1 * triangle_DEF_sides.2) / 2
  let rectangle_length := area_triangle_DEF / rectangle_width
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  rectangle_perimeter = 30 := by
  sorry

end rectangle_perimeter_is_30_l44_44425


namespace zack_initial_marbles_l44_44272

theorem zack_initial_marbles :
  let a1 := 20
  let a2 := 30
  let a3 := 35
  let a4 := 25
  let a5 := 28
  let a6 := 40
  let r := 7
  let T := a1 + a2 + a3 + a4 + a5 + a6 + r
  T = 185 :=
by
  sorry

end zack_initial_marbles_l44_44272


namespace factorize_x_cube_minus_4x_l44_44995

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l44_44995


namespace money_left_after_spending_l44_44938

def initial_money : ℕ := 24
def doris_spent : ℕ := 6
def martha_spent : ℕ := doris_spent / 2
def total_spent : ℕ := doris_spent + martha_spent
def money_left := initial_money - total_spent

theorem money_left_after_spending : money_left = 15 := by
  sorry

end money_left_after_spending_l44_44938


namespace sandy_marks_loss_l44_44610

theorem sandy_marks_loss (n m c p : ℕ) (h1 : n = 30) (h2 : m = 65) (h3 : c = 25) (h4 : p = 3) :
  ∃ x : ℕ, (c * p - m) / (n - c) = x ∧ x = 2 := by
  sorry

end sandy_marks_loss_l44_44610


namespace percentage_of_boys_is_60_percent_l44_44816

-- Definition of the problem conditions
def totalPlayers := 50
def juniorGirls := 10
def half (n : ℕ) := n / 2
def girls := 2 * juniorGirls
def boys := totalPlayers - girls
def percentage_of_boys := (boys * 100) / totalPlayers

-- The theorem stating the proof problem
theorem percentage_of_boys_is_60_percent : percentage_of_boys = 60 := 
by 
  -- Proof omitted
  sorry

end percentage_of_boys_is_60_percent_l44_44816


namespace rectangular_prism_width_l44_44479

theorem rectangular_prism_width 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ)
  (hl : l = 5) (hh : h = 7) (hd : d = 14) :
  d = Real.sqrt (l^2 + w^2 + h^2) → w = Real.sqrt 122 :=
by 
  sorry

end rectangular_prism_width_l44_44479


namespace angle_bisector_length_is_5_l44_44703

open Real

noncomputable def triangleAngleBisectorLength (a b c : ℝ) : ℝ :=
  sqrt (a * b * (1 - (c * c) / ((a + b) * (a + b))))

theorem angle_bisector_length_is_5 :
  ∀ (A B C : ℝ), A = 20 ∧ C = 40 ∧ (b - c = 5) →
  triangleAngleBisectorLength a (2 * a * cos (A * π / 180) + 5) (2 * a * cos (A * π / 180)) = 5 :=
  by
  -- you can skip this part with sorry
  sorry

end angle_bisector_length_is_5_l44_44703


namespace mul_neg_x_squared_cubed_l44_44140

theorem mul_neg_x_squared_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 :=
sorry

end mul_neg_x_squared_cubed_l44_44140


namespace min_value_fraction_l44_44048

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end min_value_fraction_l44_44048


namespace tangent_point_value_l44_44797

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l44_44797


namespace sequence_equals_identity_l44_44078

theorem sequence_equals_identity (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j) : 
  ∀ i : ℕ, a i = i := 
by 
  sorry

end sequence_equals_identity_l44_44078


namespace sum_x_coordinates_common_points_l44_44657

theorem sum_x_coordinates_common_points (x y : ℤ) (h1 : y ≡ 3 * x + 5 [ZMOD 13]) (h2 : y ≡ 9 * x + 1 [ZMOD 13]) : x ≡ 5 [ZMOD 13] :=
sorry

end sum_x_coordinates_common_points_l44_44657


namespace min_value_of_inverse_sum_l44_44709

theorem min_value_of_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : 
  ∃ c : ℝ, c = 4 + 2 * Real.sqrt 3 ∧ ∀x : ℝ, (x = (1 / a + 1 / b)) → x ≥ c :=
by
  sorry

end min_value_of_inverse_sum_l44_44709


namespace probability_correct_arrangement_l44_44250

-- Definitions for conditions
def characters := {c : String | c = "医" ∨ c = "国"}

def valid_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"], ["国", "医", "医"]}

def correct_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"]}

-- Theorem statement
theorem probability_correct_arrangement :
  (correct_arrangements.card : ℚ) / (valid_arrangements.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_correct_arrangement_l44_44250


namespace seq_20_l44_44588

noncomputable def seq (n : ℕ) : ℝ := 
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 1/2
  else sorry -- The actual function definition based on the recurrence relation is omitted for brevity

lemma seq_recurrence (n : ℕ) (hn : n ≥ 1) :
  2 / seq (n + 1) = (seq n + seq (n + 2)) / (seq n * seq (n + 2)) :=
sorry

theorem seq_20 : seq 20 = 1/20 :=
sorry

end seq_20_l44_44588


namespace volume_of_right_prism_correct_l44_44670

variables {α β l : ℝ}

noncomputable def volume_of_right_prism (α β l : ℝ) : ℝ :=
  (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α))

theorem volume_of_right_prism_correct
  (α β l : ℝ)
  (α_gt0 : 0 < α) (α_lt90 : α < Real.pi / 2)
  (l_pos : 0 < l)
  : volume_of_right_prism α β l = (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α)) :=
sorry

end volume_of_right_prism_correct_l44_44670


namespace max_members_in_band_l44_44626

theorem max_members_in_band (m : ℤ) (h1 : 30 * m % 31 = 6) (h2 : 30 * m < 1200) : 30 * m = 360 :=
by {
  sorry -- Proof steps are not required according to the procedure
}

end max_members_in_band_l44_44626


namespace bike_travel_distance_l44_44279

-- Declaring the conditions as definitions
def speed : ℝ := 50 -- Speed in meters per second
def time : ℝ := 7 -- Time in seconds

-- Declaring the question and expected answer
def expected_distance : ℝ := 350 -- Expected distance in meters

-- The proof statement that needs to be proved
theorem bike_travel_distance : (speed * time = expected_distance) :=
by
  sorry

end bike_travel_distance_l44_44279


namespace evaluate_expression_at_values_l44_44435

theorem evaluate_expression_at_values :
  let x := 2
  let y := -1
  let z := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
    sorry

end evaluate_expression_at_values_l44_44435


namespace angle_bisector_length_l44_44704

-- Define the given conditions
def triangle_has_given_angles_and_side_diff (A C : ℝ) (AC_minus_AB : ℝ) : Prop :=
  A = 20 ∧ C = 40 ∧ AC_minus_AB = 5

-- Define the main theorem with the conclusion that the length of the angle bisector is 5 cm
theorem angle_bisector_length (A B C AC AB : ℝ) (h : triangle_has_given_angles_and_side_diff A C (AC - AB)) :
  let AC_minus_AB := 5 in
  ∃ l_b : ℝ, l_b = 5 :=
begin
  sorry
end

end angle_bisector_length_l44_44704


namespace flight_duration_l44_44046

theorem flight_duration :
  ∀ (h m : ℕ),
  3 * 60 + 42 = 15 * 60 + 57 →
  0 < m ∧ m < 60 →
  h + m = 18 :=
by
  intros h m h_def hm_bound
  sorry

end flight_duration_l44_44046


namespace stacy_current_height_l44_44742

-- Conditions
def last_year_height_stacy : ℕ := 50
def brother_growth : ℕ := 1
def stacy_growth : ℕ := brother_growth + 6

-- Statement to prove
theorem stacy_current_height : last_year_height_stacy + stacy_growth = 57 :=
by
  sorry

end stacy_current_height_l44_44742


namespace ceil_square_of_neg_seven_fourths_l44_44532

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l44_44532


namespace max_value_of_expression_l44_44164

open Real

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  2 * x * y + y * z + 2 * z * x ≤ 4 / 7 := 
sorry

end max_value_of_expression_l44_44164


namespace problem_part1_problem_part2_l44_44029

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end problem_part1_problem_part2_l44_44029


namespace maria_paid_9_l44_44225

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l44_44225


namespace inequality_proof_l44_44732

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9 * y + 3 * z) * (x + 4 * y + 2 * z) * (2 * x + 12 * y + 9 * z) ≥ 1029 * x * y * z :=
by
  sorry

end inequality_proof_l44_44732


namespace unique_triple_primes_l44_44891

theorem unique_triple_primes (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) (h3 : (p^3 + q^3 + r^3) / (p + q + r) = 249) : r = 19 :=
sorry

end unique_triple_primes_l44_44891


namespace multiplication_integer_multiple_l44_44238

theorem multiplication_integer_multiple (a b n : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
(h_eq : 10000 * a + b = n * (a * b)) : n = 73 := 
sorry

end multiplication_integer_multiple_l44_44238


namespace Mina_stops_in_D_or_A_l44_44723

-- Define the relevant conditions and problem statement
def circumference := 60
def total_distance := 6000
def quarters := ["A", "B", "C", "D"]
def start_position := "S"
def stop_position := if (total_distance % circumference) == 0 then "S" else ""

theorem Mina_stops_in_D_or_A : stop_position = start_position → start_position = "D" ∨ start_position = "A" :=
by
  sorry

end Mina_stops_in_D_or_A_l44_44723


namespace find_number_divided_by_6_l44_44109

theorem find_number_divided_by_6 (x : ℤ) (h : (x + 17) / 5 = 25) : x / 6 = 18 :=
by
  sorry

end find_number_divided_by_6_l44_44109


namespace no_primes_divisible_by_45_l44_44174

theorem no_primes_divisible_by_45 : ∀ p : ℕ, prime p → ¬ (45 ∣ p) := 
begin
  sorry
end

end no_primes_divisible_by_45_l44_44174


namespace range_of_m_l44_44682

variable (x m : ℝ)

theorem range_of_m (h1 : ∀ x : ℝ, 2 * x^2 - 2 * m * x + m < 0) 
    (h2 : ∃ a b : ℤ, a ≠ b ∧ ∀ x : ℝ, (a < x ∧ x < b) → 2 * x^2 - 2 * m * x + m < 0): 
    -8 / 5 ≤ m ∧ m < -2 / 3 ∨ 8 / 3 < m ∧ m ≤ 18 / 5 :=
sorry

end range_of_m_l44_44682


namespace no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l44_44487

theorem no_20_digit_number_starting_with_11111111111_is_a_perfect_square :
  ¬ ∃ (n : ℤ), (10^19 ≤ n ∧ n < 10^20 ∧ (11111111111 * 10^9 ≤ n ∧ n < 11111111112 * 10^9) ∧ (∃ k : ℤ, n = k^2)) :=
by
  sorry

end no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l44_44487


namespace perpendicular_vectors_relation_l44_44947

theorem perpendicular_vectors_relation (a b : ℝ) (h : 3 * a - 7 * b = 0) : a = 7 * b / 3 :=
by
  sorry

end perpendicular_vectors_relation_l44_44947


namespace cos_squared_identity_l44_44573

theorem cos_squared_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * Real.cos (π / 6 + α / 2) ^ 2 + 1 = 7 / 3 := 
by
    sorry

end cos_squared_identity_l44_44573


namespace product_of_successive_numbers_l44_44929

-- Given conditions
def n : ℝ := 51.49757275833493

-- Proof statement
theorem product_of_successive_numbers : n * (n + 1) = 2703.0000000000005 :=
by
  -- Proof would be supplied here
  sorry

end product_of_successive_numbers_l44_44929


namespace linear_function_l44_44456

theorem linear_function (f : ℝ → ℝ)
  (h : ∀ x, f (f x) = 4 * x + 6) :
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) :=
sorry

end linear_function_l44_44456


namespace intersection_of_A_and_B_l44_44160

noncomputable def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
noncomputable def B : Set ℝ := { x | 0 ≤ x }

theorem intersection_of_A_and_B :
  { x | x ∈ A ∧ x ∈ B } = { x | 0 ≤ x ∧ x ≤ 3 } :=
  by sorry

end intersection_of_A_and_B_l44_44160


namespace distance_between_trees_l44_44885

-- Variables representing the total length of the yard and the number of trees.
variable (length_of_yard : ℕ) (number_of_trees : ℕ)

-- The given conditions
def yard_conditions (length_of_yard number_of_trees : ℕ) :=
  length_of_yard = 700 ∧ number_of_trees = 26

-- The proof statement: If the yard is 700 meters long and there are 26 trees, 
-- then the distance between two consecutive trees is 28 meters.
theorem distance_between_trees (length_of_yard : ℕ) (number_of_trees : ℕ)
  (h : yard_conditions length_of_yard number_of_trees) : 
  (length_of_yard / (number_of_trees - 1)) = 28 := 
by
  sorry

end distance_between_trees_l44_44885


namespace checkerboard_contains_5_black_squares_l44_44652

def is_checkerboard (x y : ℕ) : Prop := 
  x < 8 ∧ y < 8 ∧ (x + y) % 2 = 0

def contains_5_black_squares (x y n : ℕ) : Prop :=
  ∃ k l : ℕ, k ≤ n ∧ l ≤ n ∧ (x + k + y + l) % 2 = 0 ∧ k * l >= 5

theorem checkerboard_contains_5_black_squares :
  ∃ num, num = 73 ∧
  (∀ x y n, contains_5_black_squares x y n → num = 73) :=
by
  sorry

end checkerboard_contains_5_black_squares_l44_44652


namespace rectangle_sides_l44_44656

theorem rectangle_sides (k : ℝ) (μ : ℝ) (a b : ℝ) 
  (h₀ : k = 8) 
  (h₁ : μ = 3/10) 
  (h₂ : 2 * (a + b) = k) 
  (h₃ : a * b = μ * (a^2 + b^2)) : 
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) :=
sorry

end rectangle_sides_l44_44656


namespace ceil_square_of_neg_fraction_l44_44519

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l44_44519


namespace number_problem_l44_44033

theorem number_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 34) / 10 = 2 := by
  sorry

end number_problem_l44_44033


namespace distinct_ways_to_divide_books_l44_44628

theorem distinct_ways_to_divide_books : 
  ∃ (ways : ℕ), ways = 5 := sorry

end distinct_ways_to_divide_books_l44_44628


namespace minimum_value_of_expression_l44_44331

theorem minimum_value_of_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 := by
  sorry

end minimum_value_of_expression_l44_44331


namespace function_periodicity_l44_44823

variable {R : Type*} [Ring R]

def periodic_function (f : R → R) (k : R) : Prop :=
  ∀ x : R, f (x + 4*k) = f x

theorem function_periodicity {f : ℝ → ℝ} {k : ℝ} (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) (hk : k ≠ 0) : 
  periodic_function f k :=
sorry

end function_periodicity_l44_44823


namespace campers_difference_l44_44726

theorem campers_difference (a_morning : ℕ) (b_morning_afternoon : ℕ) (a_afternoon : ℕ) (a_afternoon_evening : ℕ) (c_evening_only : ℕ) :
  a_morning = 33 ∧ b_morning_afternoon = 11 ∧ a_afternoon = 34 ∧ a_afternoon_evening = 20 ∧ c_evening_only = 10 →
  a_afternoon - (a_afternoon_evening + c_evening_only) = 4 := 
by
  -- The actual proof would go here
  sorry

end campers_difference_l44_44726


namespace square_plot_area_l44_44437

theorem square_plot_area (s : ℕ) 
  (cost_per_foot : ℕ) 
  (total_cost : ℕ) 
  (H1 : cost_per_foot = 58) 
  (H2 : total_cost = 1624) 
  (H3 : total_cost = 232 * s) : 
  s * s = 49 := 
  by sorry

end square_plot_area_l44_44437


namespace seventh_observation_is_eight_l44_44779

theorem seventh_observation_is_eight
  (s₆ : ℕ)
  (a₆ : ℕ)
  (s₇ : ℕ)
  (a₇ : ℕ)
  (h₁ : s₆ = 6 * a₆)
  (h₂ : a₆ = 15)
  (h₃ : s₇ = 7 * a₇)
  (h₄ : a₇ = 14) :
  s₇ - s₆ = 8 :=
by
  -- Place proof here
  sorry

end seventh_observation_is_eight_l44_44779


namespace dan_picked_more_apples_l44_44653

-- Define the number of apples picked by Benny and Dan
def apples_picked_by_benny := 2
def apples_picked_by_dan := 9

-- Lean statement to prove the given condition
theorem dan_picked_more_apples :
  apples_picked_by_dan - apples_picked_by_benny = 7 := 
sorry

end dan_picked_more_apples_l44_44653


namespace max_value_HMMT_l44_44326

theorem max_value_HMMT :
  ∀ (H M T : ℤ), H * M ^ 2 * T = H + 2 * M + T → H * M ^ 2 * T ≤ 8 :=
by
  sorry

end max_value_HMMT_l44_44326


namespace area_of_polygon_ABHFGD_l44_44208

noncomputable def total_area_ABHFGD : ℝ :=
  let side_ABCD := 3
  let side_EFGD := 5
  let area_ABCD := side_ABCD * side_ABCD
  let area_EFGD := side_EFGD * side_EFGD
  let area_DBH := 0.5 * 3 * (3 / 2 : ℝ) -- Area of triangle DBH
  let area_DFH := 0.5 * 5 * (5 / 2 : ℝ) -- Area of triangle DFH
  area_ABCD + area_EFGD - (area_DBH + area_DFH)

theorem area_of_polygon_ABHFGD : total_area_ABHFGD = 25.5 := by
  sorry

end area_of_polygon_ABHFGD_l44_44208


namespace range_of_m_l44_44887

theorem range_of_m (m : ℝ) :
  (∀ P : ℝ × ℝ, P.2 = 2 * P.1 + m → (abs (P.1^2 + (P.2 - 1)^2) = (1/2) * abs (P.1^2 + (P.2 - 4)^2)) → (-2 * Real.sqrt 5) ≤ m ∧ m ≤ (2 * Real.sqrt 5)) :=
sorry

end range_of_m_l44_44887


namespace hyperbola_asymptotes_l44_44925

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptotes (a b : ℝ) (h : hyperbola_eccentricity a b = Real.sqrt 3) :
  (∀ x y : ℝ, (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x)) :=
sorry

end hyperbola_asymptotes_l44_44925


namespace minimum_value_l44_44344

theorem minimum_value (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 2) 
  (h4 : a + b = 1) : 
  ∃ L, L = (3 * a * c / b) + (c / (a * b)) + (6 / (c - 2)) ∧ L = 1 / (a * (1 - a)) := sorry

end minimum_value_l44_44344


namespace passengers_with_round_trip_tickets_l44_44444

theorem passengers_with_round_trip_tickets (P R : ℝ) : 
  (0.40 * R = 0.25 * P) → (R / P = 0.625) :=
by
  intro h
  sorry

end passengers_with_round_trip_tickets_l44_44444


namespace calculate_speed_l44_44642

-- Define the distance and time conditions
def distance : ℝ := 390
def time : ℝ := 4

-- Define the expected answer for speed
def expected_speed : ℝ := 97.5

-- Prove that speed equals expected_speed given the conditions
theorem calculate_speed : (distance / time) = expected_speed :=
by
  -- skipped proof steps
  sorry

end calculate_speed_l44_44642


namespace juice_drinks_costs_2_l44_44262

-- Define the conditions and the proof problem
theorem juice_drinks_costs_2 (given_amount : ℕ) (amount_returned : ℕ) 
                            (pizza_cost : ℕ) (number_of_pizzas : ℕ) 
                            (number_of_juice_packs : ℕ) 
                            (total_spent_on_juice : ℕ) (cost_per_pack : ℕ) 
                            (h1 : given_amount = 50) (h2 : amount_returned = 22)
                            (h3 : pizza_cost = 12) (h4 : number_of_pizzas = 2)
                            (h5 : number_of_juice_packs = 2) 
                            (h6 : given_amount - amount_returned - number_of_pizzas * pizza_cost = total_spent_on_juice) 
                            (h7 : total_spent_on_juice / number_of_juice_packs = cost_per_pack) : 
                            cost_per_pack = 2 := by
  sorry

end juice_drinks_costs_2_l44_44262


namespace incorrect_conclusion_l44_44685

theorem incorrect_conclusion :
  ∃ (a x y : ℝ), 
  (x + 3 * y = 4 - a ∧ x - y = 3 * a) ∧ 
  (∀ (xa ya : ℝ), (xa = 2) → (x = 2 * xa + 1) ∧ (y = 1 - xa) → ¬ (xa + ya = 4 - xa)) :=
sorry

end incorrect_conclusion_l44_44685


namespace total_weight_of_snacks_l44_44047

-- Definitions for conditions
def weight_peanuts := 0.1
def weight_raisins := 0.4
def weight_almonds := 0.3

-- Theorem statement
theorem total_weight_of_snacks : weight_peanuts + weight_raisins + weight_almonds = 0.8 := by
  sorry

end total_weight_of_snacks_l44_44047


namespace market_value_correct_l44_44441

noncomputable def face_value : ℝ := 100
noncomputable def dividend_per_share : ℝ := 0.14 * face_value
noncomputable def yield : ℝ := 0.08

theorem market_value_correct :
  (dividend_per_share / yield) * 100 = 175 := by
  sorry

end market_value_correct_l44_44441


namespace work_problem_l44_44644

theorem work_problem 
  (A_work_time : ℤ) 
  (B_work_time : ℤ) 
  (x : ℤ)
  (A_work_rate : ℚ := 1 / 15 )
  (work_left : ℚ := 0.18333333333333335)
  (worked_together_for : ℚ := 7)
  (work_done : ℚ := 1 - work_left) :
  (7 * (1 / 15 + 1 / x) = work_done) → x = 20 :=
by sorry

end work_problem_l44_44644


namespace travel_time_difference_in_minutes_l44_44966

/-
A bus travels at an average speed of 40 miles per hour.
We need to prove that the difference in travel time between a 360-mile trip and a 400-mile trip equals 60 minutes.
-/

theorem travel_time_difference_in_minutes 
  (speed : ℝ) (distance1 distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end travel_time_difference_in_minutes_l44_44966


namespace eggs_in_box_l44_44035

-- Given conditions as definitions in Lean 4
def initial_eggs : ℕ := 7
def additional_whole_eggs : ℕ := 3

-- The proof statement
theorem eggs_in_box : initial_eggs + additional_whole_eggs = 10 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end eggs_in_box_l44_44035


namespace percentage_of_y_l44_44195

theorem percentage_of_y (x y P : ℝ) (h1 : 0.10 * x = (P/100) * y) (h2 : x / y = 2) : P = 20 :=
sorry

end percentage_of_y_l44_44195


namespace sum_of_sequence_eq_six_seventeenth_l44_44216

noncomputable def cn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.cos (n * Real.arctan (2 / 3))
noncomputable def dn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.sin (n * Real.arctan (2 / 3))

theorem sum_of_sequence_eq_six_seventeenth : 
  (∑' n : ℕ, (cn n * dn n / 8^n)) = 6/17 := sorry

end sum_of_sequence_eq_six_seventeenth_l44_44216


namespace find_pq_l44_44783

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_pq (p q : ℕ) 
(hp : is_prime p) 
(hq : is_prime q) 
(h : is_prime (q^2 - p^2)) : 
  p * q = 6 :=
by sorry

end find_pq_l44_44783


namespace least_product_of_primes_gt_30_l44_44090

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l44_44090


namespace valid_program_combinations_l44_44805

theorem valid_program_combinations :
  let courses := ['English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Science]
  let math_courses := ['Algebra, 'Geometry]
  (comb_count : Nat → Nat → Nat) =
    (choose 5 3 - choose 3 3) → 9 := 
by
  sorry

end valid_program_combinations_l44_44805


namespace ceil_square_neg_fraction_l44_44528

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l44_44528


namespace no_prime_divisible_by_45_l44_44175

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l44_44175


namespace max_g_value_on_interval_l44_44330

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end max_g_value_on_interval_l44_44330


namespace farmer_profit_l44_44287

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end farmer_profit_l44_44287


namespace wam_gm_gt_hm_l44_44381

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end wam_gm_gt_hm_l44_44381


namespace find_k_l44_44839

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end find_k_l44_44839


namespace arithmetic_sequence_a3_value_l44_44675

theorem arithmetic_sequence_a3_value 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + 2) 
  (h2 : (a 1 + 2)^2 = a 1 * (a 1 + 8)) : 
  a 2 = 5 := 
by 
  sorry

end arithmetic_sequence_a3_value_l44_44675


namespace two_digit_sum_reverse_l44_44748

theorem two_digit_sum_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_reverse_l44_44748


namespace total_journey_time_l44_44205

theorem total_journey_time
  (river_speed : ℝ)
  (boat_speed_still_water : ℝ)
  (distance_upstream : ℝ)
  (total_journey_time : ℝ) :
  river_speed = 2 → 
  boat_speed_still_water = 6 → 
  distance_upstream = 48 → 
  total_journey_time = (distance_upstream / (boat_speed_still_water - river_speed) + distance_upstream / (boat_speed_still_water + river_speed)) → 
  total_journey_time = 18 := 
by
  intros h1 h2 h3 h4
  sorry

end total_journey_time_l44_44205


namespace least_product_of_primes_gt_30_l44_44091

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l44_44091


namespace tangent_circles_x_intersect_l44_44789

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l44_44789


namespace inequality_always_holds_l44_44053

noncomputable def range_for_inequality (k : ℝ) : Prop :=
  0 < k ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)

theorem inequality_always_holds (x y k : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y = k) :
  (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2 ↔ range_for_inequality k :=
sorry

end inequality_always_holds_l44_44053


namespace inequality_bound_l44_44194

theorem inequality_bound (a b c d e p q : ℝ) (hpq : 0 < p ∧ p ≤ q)
  (ha : p ≤ a ∧ a ≤ q) (hb : p ≤ b ∧ b ≤ q) (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
sorry

end inequality_bound_l44_44194


namespace nadine_total_cleaning_time_l44_44913

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_l44_44913


namespace parabola_focus_l44_44543

theorem parabola_focus : 
  ∀ x y : ℝ, y = - (1 / 16) * x^2 → ∃ f : ℝ × ℝ, f = (0, -4) := 
by
  sorry

end parabola_focus_l44_44543


namespace fifth_friend_paid_40_l44_44671

variable (x1 x2 x3 x4 x5 : ℝ)

def conditions : Prop :=
  (x1 = 1/3 * (x2 + x3 + x4 + x5)) ∧
  (x2 = 1/4 * (x1 + x3 + x4 + x5)) ∧
  (x3 = 1/5 * (x1 + x2 + x4 + x5)) ∧
  (x4 = 1/6 * (x1 + x2 + x3 + x5)) ∧
  (x1 + x2 + x3 + x4 + x5 = 120)

theorem fifth_friend_paid_40 (h : conditions x1 x2 x3 x4 x5) : x5 = 40 := by
  sorry

end fifth_friend_paid_40_l44_44671


namespace ceil_square_neg_seven_over_four_l44_44502

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l44_44502


namespace find_t2_l44_44607

variable {P A1 A2 t1 r t2 : ℝ}
def conditions (P A1 A2 t1 r t2 : ℝ) :=
  P = 650 ∧
  A1 = 815 ∧
  A2 = 870 ∧
  t1 = 3 ∧
  A1 = P + (P * r * t1) / 100 ∧
  A2 = P + (P * r * t2) / 100

theorem find_t2
  (P A1 A2 t1 r t2 : ℝ)
  (hc : conditions P A1 A2 t1 r t2) :
  t2 = 4 :=
by
  sorry

end find_t2_l44_44607


namespace max_value_of_a2b3c2_l44_44713

theorem max_value_of_a2b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81 / 262144 :=
sorry

end max_value_of_a2b3c2_l44_44713


namespace rectangle_perimeter_l44_44421

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l44_44421


namespace ceil_square_neg_fraction_l44_44525

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l44_44525


namespace find_b_l44_44541

theorem find_b (b : ℝ) (h_floor : b + ⌊b⌋ = 22.6) : b = 11.6 :=
sorry

end find_b_l44_44541


namespace problem_2_8_3_4_7_2_2_l44_44477

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l44_44477


namespace bird_counts_remaining_l44_44081

theorem bird_counts_remaining
  (peregrine_falcons pigeons crows sparrows : ℕ)
  (chicks_per_pigeon chicks_per_crow chicks_per_sparrow : ℕ)
  (peregrines_eat_pigeons_percent peregrines_eat_crows_percent peregrines_eat_sparrows_percent : ℝ)
  (initial_peregrine_falcons : peregrine_falcons = 12)
  (initial_pigeons : pigeons = 80)
  (initial_crows : crows = 25)
  (initial_sparrows : sparrows = 15)
  (chicks_per_pigeon_cond : chicks_per_pigeon = 8)
  (chicks_per_crow_cond : chicks_per_crow = 5)
  (chicks_per_sparrow_cond : chicks_per_sparrow = 3)
  (peregrines_eat_pigeons_percent_cond : peregrines_eat_pigeons_percent = 0.4)
  (peregrines_eat_crows_percent_cond : peregrines_eat_crows_percent = 0.25)
  (peregrines_eat_sparrows_percent_cond : peregrines_eat_sparrows_percent = 0.1)
  : 
  (peregrine_falcons = 12) ∧
  (pigeons = 48) ∧
  (crows = 19) ∧
  (sparrows = 14) :=
by
  sorry

end bird_counts_remaining_l44_44081


namespace boys_in_class_l44_44203

-- Define the conditions given in the problem
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 4 * (boys + girls) / 7 ∧ girls = 3 * (boys + girls) / 7
def total_students (boys girls : ℕ) : Prop := boys + girls = 49

-- Define the statement to be proved
theorem boys_in_class (boys girls : ℕ) (h1 : ratio_boys_to_girls boys girls) (h2 : total_students boys girls) : boys = 28 :=
by
  sorry

end boys_in_class_l44_44203


namespace circles_intersect_l44_44623

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0

theorem circles_intersect :
  ∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y := by
  sorry

end circles_intersect_l44_44623


namespace find_a4_plus_b4_l44_44679

theorem find_a4_plus_b4 (a b : ℝ)
  (h1 : (a^2 - b^2)^2 = 100)
  (h2 : a^3 * b^3 = 512) :
  a^4 + b^4 = 228 :=
by
  sorry

end find_a4_plus_b4_l44_44679


namespace problem_statement_l44_44028

variable (a b c d x : ℕ)

theorem problem_statement
  (h1 : a + b = x)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : a + d = 6) :
  x = 12 :=
by
  sorry

end problem_statement_l44_44028


namespace parabola_directrix_l44_44542

theorem parabola_directrix (x : ℝ) (y : ℝ) (h : y = -4 * x ^ 2 - 3) : y = - 49 / 16 := sorry

end parabola_directrix_l44_44542


namespace half_angle_in_quadrant_l44_44015

theorem half_angle_in_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 / 2) * Real.pi) :
  (π / 2 < α / 2 ∧ α / 2 < π) ∨ (3 * π / 2 < α / 2 ∧ α / 2 < 2 * π) :=
sorry

end half_angle_in_quadrant_l44_44015


namespace derivative_of_odd_is_even_l44_44236

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Assume f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Assume g is the derivative of f
axiom g_derivative : ∀ x, g x = deriv f x

-- Goal: Prove that g is an even function, i.e., g(-x) = g(x)
theorem derivative_of_odd_is_even : ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_is_even_l44_44236


namespace distance_between_trees_l44_44884

theorem distance_between_trees (n : ℕ) (len : ℝ) (d : ℝ) 
  (h1 : n = 26) 
  (h2 : len = 400) 
  (h3 : len / (n - 1) = d) : 
  d = 16 :=
by
  sorry

end distance_between_trees_l44_44884


namespace base6_sub_base9_to_base10_l44_44667

theorem base6_sub_base9_to_base10 :
  (3 * 6^2 + 2 * 6^1 + 5 * 6^0) - (2 * 9^2 + 1 * 9^1 + 5 * 9^0) = -51 :=
by
  sorry

end base6_sub_base9_to_base10_l44_44667


namespace range_of_f_l44_44930

noncomputable def f (x : ℝ) := Real.arcsin (x ^ 2 - x)

theorem range_of_f :
  Set.range f = Set.Icc (-Real.arcsin (1/4)) (Real.pi / 2) :=
sorry

end range_of_f_l44_44930


namespace length_of_common_internal_tangent_l44_44767

-- Define the conditions
def circles_centers_distance : ℝ := 50
def radius_smaller_circle : ℝ := 7
def radius_larger_circle : ℝ := 10

-- Define the statement to be proven
theorem length_of_common_internal_tangent :
  let d := circles_centers_distance
  let r₁ := radius_smaller_circle
  let r₂ := radius_larger_circle
  ∃ (length_tangent : ℝ), length_tangent = Real.sqrt (d^2 - (r₁ + r₂)^2) := by
  -- Provide the correct answer based on the conditions
  sorry

end length_of_common_internal_tangent_l44_44767


namespace number_square_l44_44772

-- Define conditions.
def valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d * d ≤ 9

-- Main statement.
theorem number_square (n : ℕ) (valid_digits : ∀ d, d ∈ [n / 100, (n / 10) % 10, n % 10] → valid_digit d) : 
  n = 233 :=
by
  -- Proof goes here
  sorry

end number_square_l44_44772


namespace carlos_cycles_more_than_diana_l44_44401

theorem carlos_cycles_more_than_diana :
  let slope_carlos := 1
  let slope_diana := 0.75
  let rate_carlos := slope_carlos * 20
  let rate_diana := slope_diana * 20
  let distance_carlos_after_3_hours := 3 * rate_carlos
  let distance_diana_after_3_hours := 3 * rate_diana
  distance_carlos_after_3_hours - distance_diana_after_3_hours = 15 :=
sorry

end carlos_cycles_more_than_diana_l44_44401


namespace prime_check_for_d1_prime_check_for_d2_l44_44458

-- Define d1 and d2
def d1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def d2 : ℕ := 9^4 - 9^2 + 1

-- Prime checking function
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Stating the conditions and proofs
theorem prime_check_for_d1 : ¬ is_prime d1 :=
by {
  -- condition: ten 8's in base nine is divisible by d1 (5905) is not used here directly
  sorry
}

theorem prime_check_for_d2 : is_prime d2 :=
by {
  -- condition: twelve 8's in base nine is divisible by d2 (6481) is not used here directly
  sorry
}

end prime_check_for_d1_prime_check_for_d2_l44_44458


namespace total_leaves_correct_l44_44136

-- Definitions based on conditions
def basil_pots := 3
def rosemary_pots := 9
def thyme_pots := 6

def basil_leaves_per_pot := 4
def rosemary_leaves_per_pot := 18
def thyme_leaves_per_pot := 30

-- Calculate the total number of leaves
def total_leaves : Nat :=
  (basil_pots * basil_leaves_per_pot) +
  (rosemary_pots * rosemary_leaves_per_pot) +
  (thyme_pots * thyme_leaves_per_pot)

-- The statement to prove
theorem total_leaves_correct : total_leaves = 354 := by
  sorry

end total_leaves_correct_l44_44136


namespace no_linear_factor_l44_44659

theorem no_linear_factor : ∀ x y z : ℤ,
  ¬ ∃ a b c : ℤ, a*x + b*y + c*z + (x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z) = 0 :=
by sorry

end no_linear_factor_l44_44659


namespace circle_rational_points_l44_44608

theorem circle_rational_points :
  ( ∃ B : ℚ × ℚ, ∀ k : ℚ, B ∈ {p | p.1 ^ 2 + 2 * p.1 + p.2 ^ 2 = 1992} ) ∧ 
  ( (42 : ℤ)^2 + 2 * 42 + 12^2 = 1992 ) :=
by
  sorry

end circle_rational_points_l44_44608


namespace line_equation_l44_44402

theorem line_equation (p : ℝ × ℝ) (a : ℝ × ℝ) :
  p = (4, -4) →
  a = (1, 2 / 7) →
  ∃ (m b : ℝ), m = 2 / 7 ∧ b = -36 / 7 ∧ ∀ x y : ℝ, y = m * x + b :=
by
  intros hp ha
  sorry

end line_equation_l44_44402


namespace denis_dartboard_score_l44_44148

theorem denis_dartboard_score :
  ∀ P1 P2 P3 P4 : ℕ,
  P1 = 30 → 
  P2 = 38 → 
  P3 = 41 → 
  P1 + P2 + P3 + P4 = 4 * ((P1 + P2 + P3 + P4) / 4) → 
  P4 = 34 :=
by
  intro P1 P2 P3 P4 hP1 hP2 hP3 hTotal
  have hSum := hP1.symm ▸ hP2.symm ▸ hP3.symm ▸ hTotal
  sorry

end denis_dartboard_score_l44_44148


namespace bridge_length_l44_44299

theorem bridge_length (length_train : ℝ) (speed_train : ℝ) (time : ℝ) (h1 : length_train = 15) (h2 : speed_train = 275) (h3 : time = 48) : 
    (speed_train / 100) * time - length_train = 117 := 
by
    -- these are the provided conditions, enabling us to skip actual proof steps with 'sorry'
    sorry

end bridge_length_l44_44299


namespace positive_area_triangles_correct_l44_44863

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l44_44863


namespace apples_not_ripe_l44_44826

theorem apples_not_ripe (total_apples good_apples : ℕ) (h1 : total_apples = 14) (h2 : good_apples = 8) : total_apples - good_apples = 6 :=
by {
  sorry
}

end apples_not_ripe_l44_44826


namespace intersection_of_M_and_N_l44_44351

def M : Set ℝ := { x | |x + 1| ≤ 1}

def N : Set ℝ := {-1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} :=
by
  sorry

end intersection_of_M_and_N_l44_44351


namespace a5_a6_less_than_a4_squared_l44_44348

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem a5_a6_less_than_a4_squared
  (h_geo : is_geometric_sequence a q)
  (h_cond : a 5 * a 6 < (a 4) ^ 2) :
  0 < q ∧ q < 1 :=
sorry

end a5_a6_less_than_a4_squared_l44_44348


namespace min_value_of_x_plus_y_l44_44169

open Real

theorem min_value_of_x_plus_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0)
  (a : ℝ × ℝ := (1 - x, 4)) (b : ℝ × ℝ := (x, -y))
  (h₃ : ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)) :
  x + y = 9 :=
by
  sorry

end min_value_of_x_plus_y_l44_44169


namespace math_problem_l44_44264

-- Statement of the theorem
theorem math_problem :
  (0.66)^3 - ((0.1)^3 / ((0.66)^2 + 0.066 + (0.1)^2)) = 0.3612 :=
by
  sorry -- Proof is not required

end math_problem_l44_44264


namespace min_value_fraction_l44_44049

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end min_value_fraction_l44_44049


namespace game_rounds_l44_44971

noncomputable def play_game (A B C D : ℕ) : ℕ := sorry

theorem game_rounds : play_game 16 15 14 13 = 49 :=
by
  sorry

end game_rounds_l44_44971


namespace greatest_integer_sum_of_integers_l44_44106

-- Definition of the quadratic function
def quadratic_expr (n : ℤ) : ℤ := n^2 - 15 * n + 56

-- The greatest integer n such that quadratic_expr n ≤ 0
theorem greatest_integer (n : ℤ) (h : quadratic_expr n ≤ 0) : n ≤ 8 := 
  sorry

-- All integers that satisfy the quadratic inequality
theorem sum_of_integers (sum_n : ℤ) (h : ∀ n : ℤ, 7 ≤ n ∧ n ≤ 8 → quadratic_expr n ≤ 0) 
  (sum_eq : sum_n = 7 + 8) : sum_n = 15 :=
  sorry

end greatest_integer_sum_of_integers_l44_44106


namespace maria_paid_9_l44_44223

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l44_44223


namespace least_product_of_distinct_primes_greater_than_30_l44_44094

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l44_44094


namespace percent_equivalence_l44_44269

theorem percent_equivalence (y : ℝ) : 0.30 * (0.60 * y) = 0.18 * y :=
by sorry

end percent_equivalence_l44_44269


namespace find_lower_rate_l44_44461

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end find_lower_rate_l44_44461


namespace two_digit_sum_reverse_l44_44747

theorem two_digit_sum_reverse (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
    (h₃ : 0 ≤ b) (h₄ : b ≤ 9)
    (h₅ : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
    (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_sum_reverse_l44_44747


namespace ceil_of_neg_frac_squared_l44_44492

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l44_44492


namespace total_trophies_after_five_years_l44_44590

theorem total_trophies_after_five_years (michael_current_trophies : ℕ) (michael_increase : ℕ) (jack_multiplier : ℕ) (h1 : michael_current_trophies = 50) (h2 : michael_increase = 150) (h3 : jack_multiplier = 15) :
  let michael_five_years : ℕ := michael_current_trophies + michael_increase
  let jack_five_years : ℕ := jack_multiplier * michael_current_trophies
  michael_five_years + jack_five_years = 950 :=
by
  sorry

end total_trophies_after_five_years_l44_44590


namespace solve_for_y_l44_44197

theorem solve_for_y (x y : ℝ) (h1 : x ^ (2 * y) = 16) (h2 : x = 2) : y = 2 :=
by {
  sorry
}

end solve_for_y_l44_44197


namespace number_of_valid_three_digit_numbers_l44_44813

theorem number_of_valid_three_digit_numbers : 
  (∃ A B C : ℕ, 
      (100 * A + 10 * B + C + 297 = 100 * C + 10 * B + A) ∧ 
      (0 ≤ A ∧ A ≤ 9) ∧ 
      (0 ≤ B ∧ B ≤ 9) ∧ 
      (0 ≤ C ∧ C ≤ 9)) 
    ∧ (number_of_such_valid_numbers = 70) :=
by
  sorry

def number_of_such_valid_numbers : ℕ := 
  sorry

end number_of_valid_three_digit_numbers_l44_44813


namespace clock_angle_at_3_45_l44_44433

theorem clock_angle_at_3_45 :
  let minute_angle_rate := 6.0 -- degrees per minute
  let hour_angle_rate := 0.5  -- degrees per minute
  let initial_angle := 90.0   -- degrees at 3:00
  let minutes_passed := 45.0  -- minutes since 3:00
  let angle_difference_rate := minute_angle_rate - hour_angle_rate
  let angle_change := angle_difference_rate * minutes_passed
  let final_angle := initial_angle - angle_change
  let smaller_angle := if final_angle < 0 then 360.0 + final_angle else final_angle
  smaller_angle = 157.5 :=
by
  sorry

end clock_angle_at_3_45_l44_44433


namespace simplify_sqrt_sum_l44_44614

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l44_44614


namespace last_three_digits_l44_44831

theorem last_three_digits (n : ℕ) : 7^106 % 1000 = 321 :=
by
  sorry

end last_three_digits_l44_44831


namespace stacy_current_height_l44_44743

theorem stacy_current_height:
  ∀ (stacy_previous_height brother_growth stacy_growth : ℕ),
  stacy_previous_height = 50 →
  brother_growth = 1 →
  stacy_growth = brother_growth + 6 →
  stacy_previous_height + stacy_growth = 57 :=
by
  intros stacy_previous_height brother_growth stacy_growth
  intros h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end stacy_current_height_l44_44743


namespace second_movie_duration_proof_l44_44603

-- initial duration for the first movie (in minutes)
def first_movie_duration_minutes : ℕ := 1 * 60 + 48

-- additional duration for the second movie (in minutes)
def additional_duration_minutes : ℕ := 25

-- total duration for the second movie (in minutes)
def second_movie_duration_minutes : ℕ := first_movie_duration_minutes + additional_duration_minutes

-- convert total minutes to hours and minutes
def duration_in_hours_and_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem second_movie_duration_proof :
  duration_in_hours_and_minutes second_movie_duration_minutes = (2, 13) :=
by
  -- proof would go here
  sorry

end second_movie_duration_proof_l44_44603


namespace wendys_sales_are_205_l44_44429

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end wendys_sales_are_205_l44_44429


namespace rectangle_perimeter_of_equal_area_l44_44423

theorem rectangle_perimeter_of_equal_area (a b c : ℕ) (area_triangle width length : ℕ) :
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 ∧ (2 * area_triangle = a * b) ∧
  (width = 6) ∧ (area_triangle = width * length) -> 
  2 * (length + width) = 30 :=
by
  intros h,
  sorry

end rectangle_perimeter_of_equal_area_l44_44423


namespace correct_options_l44_44013

variable (x : Fin 6 → ℝ)

def median_of_4 (a b c d : ℝ) : ℝ := (b + c) / 2

def median_of_6 (a b c d e f : ℝ) : ℝ := (c + d) / 2

theorem correct_options (x : Fin 6 → ℝ)
  (h1 : x 0 = min (min (x 0) (x 1)) (min (min (x 2) (x 3)) (min (x 4) (x 5)))) 
  (h6 : x 5 = max (max (x 0) (x 1)) (max (max (x 2) (x 3)) (max (x 4) (x 5)))) :
  (median_of_4 (x 1) (x 2) (x 3) (x 4) = median_of_6 (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) ∧
  (range (x 1) (x 2) (x 3) (x 4) ≤ range (x 0) (x 1) (x 2) (x 3) (x 4) (x 5)) :=
sorry

end correct_options_l44_44013


namespace product_of_3_point_6_and_0_point_25_l44_44001

theorem product_of_3_point_6_and_0_point_25 : 3.6 * 0.25 = 0.9 := 
by 
  sorry

end product_of_3_point_6_and_0_point_25_l44_44001


namespace product_of_numbers_l44_44408

variable (x y z : ℝ)

theorem product_of_numbers :
  x + y + z = 36 ∧ x = 3 * (y + z) ∧ y = 6 * z → x * y * z = 268 := 
by
  sorry

end product_of_numbers_l44_44408


namespace number_of_piles_l44_44234

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end number_of_piles_l44_44234


namespace geometric_sequence_common_ratio_l44_44339

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : S 2 = 2 * a 2 + 3)
  (h2 : S 3 = 2 * a 3 + 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) : q = 2 := 
by
  sorry

end geometric_sequence_common_ratio_l44_44339


namespace solve_inequality_l44_44151

theorem solve_inequality (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∨ x ∈ Set.Icc 2 4) :=
sorry

end solve_inequality_l44_44151


namespace factorize_polynomial_l44_44539

theorem factorize_polynomial (x y : ℝ) :
  3 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 3 * (x + y) ^ 2 :=
by
  sorry

end factorize_polynomial_l44_44539


namespace part1_solution_set_part2_range_of_a_l44_44902

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_solution_set (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x | x ≤ 0} ∪ {x | x ≥ 5} :=
by 
  -- proof goes here
  sorry

theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
by
  -- proof goes here
  sorry

end part1_solution_set_part2_range_of_a_l44_44902


namespace range_of_m_l44_44580

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, abs (x - 3) + abs (x - m) < 5) : -2 < m ∧ m < 8 :=
  sorry

end range_of_m_l44_44580


namespace countThreeDigitNumbersWithPerfectCubeDigitSums_l44_44571

def isThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digitSum (n : ℕ) : ℕ := 
  (n / 100) + ((n % 100) / 10) + (n % 10)

def isPerfectCube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k*k*k = n

theorem countThreeDigitNumbersWithPerfectCubeDigitSums : 
  (Finset.filter (λ n, isPerfectCube (digitSum n)) (Finset.range' 100 900)).card = 10 := 
  sorry

end countThreeDigitNumbersWithPerfectCubeDigitSums_l44_44571


namespace binary_arithmetic_l44_44004

theorem binary_arithmetic :
    let a := 0b1011101
    let b := 0b1101
    let c := 0b101010
    let d := 0b110
    ((a + b) * c) / d = 0b1110111100 :=
by
  sorry

end binary_arithmetic_l44_44004


namespace find_C_l44_44629

theorem find_C (C : ℤ) (h : 4 * C + 3 = 31) : C = 7 := by
  sorry

end find_C_l44_44629


namespace quadratic_root_product_l44_44051

theorem quadratic_root_product (a b : ℝ) (m p r : ℝ)
  (h1 : a * b = 3)
  (h2 : ∀ x, x^2 - mx + 3 = 0 → x = a ∨ x = b)
  (h3 : ∀ x, x^2 - px + r = 0 → x = a + 2 / b ∨ x = b + 2 / a) :
  r = 25 / 3 := by
  sorry

end quadratic_root_product_l44_44051


namespace product_a_b_l44_44360

variable (a b c : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_pos_c : c > 0)
variable (h_c : c = 3)
variable (h_a : a = b^2)
variable (h_bc : b + c = b * c)

theorem product_a_b : a * b = 27 / 8 :=
by
  -- We need to prove that given the above conditions, a * b = 27 / 8
  sorry

end product_a_b_l44_44360


namespace average_disk_space_per_hour_l44_44284

theorem average_disk_space_per_hour :
  let days : ℕ := 15
  let total_mb : ℕ := 20000
  let hours_per_day : ℕ := 24
  let total_hours := days * hours_per_day
  total_mb / total_hours = 56 :=
by
  let days := 15
  let total_mb := 20000
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  have h : total_mb / total_hours = 56 := sorry
  exact h

end average_disk_space_per_hour_l44_44284


namespace cos_triple_angle_l44_44190

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l44_44190


namespace number_of_valid_x_l44_44974

theorem number_of_valid_x (x : ℕ) : 
  ((x + 3) * (x - 3) * (x ^ 2 + 9) < 500) ∧ (x - 3 > 0) ↔ x = 4 :=
sorry

end number_of_valid_x_l44_44974


namespace vanaspati_percentage_l44_44369

theorem vanaspati_percentage (Q : ℝ) (h1 : 0.60 * Q > 0) (h2 : Q + 10 > 0) (h3 : Q = 10) :
    let total_ghee := Q + 10
    let pure_ghee := 0.60 * Q + 10
    let pure_ghee_fraction := pure_ghee / total_ghee
    pure_ghee_fraction = 0.80 → 
    let vanaspati_fraction := 1 - pure_ghee_fraction
    vanaspati_fraction * 100 = 40 :=
by
  intros
  sorry

end vanaspati_percentage_l44_44369


namespace identify_heaviest_and_lightest_coin_in_13_weighings_l44_44412

theorem identify_heaviest_and_lightest_coin_in_13_weighings :
  ∀ (coins : Finₓ 10 → ℝ) 
    (balance_weighing : ∀ (a b : Finₓ 10), Prop), 
    (∀ i j, coins i ≠ coins j) → 
    (∃ strategy : ℕ → (Finₓ 10 × Finₓ 10),
      ∃ h : ℕ,
        h ≤ 13 ∧
        (∃ heaviest lightest : Finₓ 10,
          (∀ i, coins heaviest ≥ coins i) ∧ (∀ j, coins lightest ≤ coins j))) :=
by
  sorry

end identify_heaviest_and_lightest_coin_in_13_weighings_l44_44412


namespace split_tip_evenly_l44_44378

noncomputable def total_cost (julie_order : ℝ) (letitia_order : ℝ) (anton_order : ℝ) : ℝ :=
  julie_order + letitia_order + anton_order

noncomputable def total_tip (meal_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  tip_rate * meal_cost

noncomputable def tip_per_person (total_tip : ℝ) (num_people : ℝ) : ℝ :=
  total_tip / num_people

theorem split_tip_evenly :
  let julie_order := 10 in
  let letitia_order := 20 in
  let anton_order := 30 in
  let tip_rate := 0.20 in
  let num_people := 3 in
  tip_per_person (total_tip (total_cost julie_order letitia_order anton_order) tip_rate) num_people = 4 :=
by
  sorry

end split_tip_evenly_l44_44378


namespace least_product_of_distinct_primes_gt_30_l44_44095

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l44_44095


namespace sum_int_values_l44_44267

theorem sum_int_values (sum : ℤ) : 
  (∀ n : ℤ, (20 % (2 * n - 1) = 0) → sum = 2) :=
by
  sorry

end sum_int_values_l44_44267


namespace least_product_of_distinct_primes_greater_than_30_l44_44093

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l44_44093


namespace least_prime_product_l44_44103
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l44_44103


namespace parking_lot_wheels_l44_44365

-- Define the conditions
def num_cars : Nat := 10
def num_bikes : Nat := 2
def wheels_per_car : Nat := 4
def wheels_per_bike : Nat := 2

-- Define the total number of wheels
def total_wheels : Nat := (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike)

-- State the theorem
theorem parking_lot_wheels : total_wheels = 44 :=
by
  sorry

end parking_lot_wheels_l44_44365


namespace validate_triangle_count_l44_44865

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l44_44865


namespace retailer_should_focus_on_mode_l44_44798

-- Define the conditions as options.
inductive ClothingModels
| Average
| Mode
| Median
| Smallest

-- Define a function to determine the best measure to focus on in the market share survey.
def bestMeasureForMarketShareSurvey (choice : ClothingModels) : Prop :=
  match choice with
  | ClothingModels.Average => False
  | ClothingModels.Mode => True
  | ClothingModels.Median => False
  | ClothingModels.Smallest => False

-- The theorem stating that the mode is the best measure to focus on.
theorem retailer_should_focus_on_mode : bestMeasureForMarketShareSurvey ClothingModels.Mode :=
by
  -- This proof is intentionally left blank.
  sorry

end retailer_should_focus_on_mode_l44_44798


namespace angle_bisector_of_B_in_triangule_ABC_l44_44701

noncomputable def angle_bisector_length {ABC : Type*} [triangle ABC]
  (angle_A : ℝ) (angle_C : ℝ) (AC minus AB : ℝ) 
  : ℝ :=
  5

theorem angle_bisector_of_B_in_triangule_ABC 
  (A B C : Type*) [is_triangle A B C] (angle_A : 𝕜) (angle_C : 𝕜) (AC AB : ℝ) 
  (hypothesis_A : angle_A = 20)
  (hypothesis_C : angle_C = 40)
  (length_condition : AC - AB = 5) :
  angle_bisector_length angle_A angle_C length_condition = 5 := 
sorry

end angle_bisector_of_B_in_triangule_ABC_l44_44701


namespace identify_heaviest_and_lightest_coin_within_13_weighings_l44_44416

-- Lean 4 statement to encapsulate the given problem
theorem identify_heaviest_and_lightest_coin_within_13_weighings :
  ∃ (weighings: list (ℕ × ℕ)) (heaviest lightest: ℕ),
    (length weighings ≤ 13) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → i ≠ j) ∧
    (∀ (comp: ℕ × ℕ), comp ∈ weighings → comp.1 ≠ comp.2 ∧ 1 ≤ comp.1 ∧ comp.1 ≤ 10 ∧ 1 ≤ comp.2 ∧ comp.2 ≤ 10) ∧
    heaviest ≠ lightest ∧
    (∀ (i: ℕ), 1 ≤ i ∧ i ≤ 10 → 
      (i = heaviest ∨ i = lightest))
: sorry

end identify_heaviest_and_lightest_coin_within_13_weighings_l44_44416


namespace relationship_abc_l44_44553

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 15 - Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 11 - Real.sqrt 3

theorem relationship_abc : a > c ∧ c > b := 
by
  unfold a b c
  sorry

end relationship_abc_l44_44553


namespace undefined_values_l44_44159

theorem undefined_values (b : ℝ) : (b^2 - 9 = 0) ↔ (b = -3 ∨ b = 3) := by
  sorry

end undefined_values_l44_44159


namespace ceil_of_neg_frac_squared_l44_44494

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l44_44494


namespace geom_sum_first_eight_terms_l44_44335

theorem geom_sum_first_eight_terms (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/3) :
    ∑ k in finset.range 8, a * r^k = 9840/19683 := by
  sorry

end geom_sum_first_eight_terms_l44_44335


namespace factorize_x_squared_plus_2x_l44_44318

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l44_44318


namespace angle_bisector_length_B_l44_44702

-- Define the angles and sides of the triangle.
variables {A B C : Type} [angle_A : has_angle A 20°] [angle_C : has_angle C 40°] 
{triangle_ABC : Type} [triangleABC : triangle A B C]
def length_of_angle_bisector_B := 5 -- cm 

theorem angle_bisector_length_B :
  ∃ l, l = 5 ∧
  (∀ (a b c : Type) [has_angle a 20°] [has_angle b 120°] [has_angle c 40°] 
      (AC AB : ℝ), 
    AC - AB = 5 → 
    l = (AC + AB - 5)) :=
sorry

end angle_bisector_length_B_l44_44702


namespace curve_is_line_segment_l44_44073

noncomputable def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = Real.cos θ ^ 2 ∧ p.2 = Real.sin θ ^ 2}

theorem curve_is_line_segment :
  (∀ p ∈ parametric_curve, p.1 + p.2 = 1 ∧ p.1 ∈ Set.Icc 0 1) :=
by
  sorry

end curve_is_line_segment_l44_44073


namespace evaluate_expression_l44_44007

variable {x y : ℝ}

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y ^ 2) :
  (x - 1 / x ^ 2) * (y + 2 / y) = 2 * x ^ (5 / 2) - 1 / x := 
by
  sorry

end evaluate_expression_l44_44007


namespace at_least_one_nonnegative_l44_44731

theorem at_least_one_nonnegative (x y z : ℝ) : 
  (x^2 + y + 1/4 ≥ 0) ∨ (y^2 + z + 1/4 ≥ 0) ∨ (z^2 + x + 1/4 ≥ 0) :=
sorry

end at_least_one_nonnegative_l44_44731


namespace buratino_solved_16_problems_l44_44636

-- Defining the conditions given in the problem
def total_kopeks_received : ℕ := 655 * 100 + 35

def geometric_sum (n : ℕ) : ℕ := 2^n - 1

-- The goal is to prove that Buratino solved 16 problems
theorem buratino_solved_16_problems (n : ℕ) (h : geometric_sum n = total_kopeks_received) : n = 16 := by
  sorry

end buratino_solved_16_problems_l44_44636


namespace not_directly_nor_inversely_proportional_l44_44311

theorem not_directly_nor_inversely_proportional :
  ∀ (x y : ℝ),
    ((2 * x + y = 5) ∨ (2 * x + 3 * y = 12)) ∧
    ((¬ (∃ k : ℝ, x = k * y)) ∧ (¬ (∃ k : ℝ, x * y = k))) := sorry

end not_directly_nor_inversely_proportional_l44_44311


namespace cindy_correct_answer_l44_44305

theorem cindy_correct_answer (x : ℤ) (h : (x - 7) / 5 = 37) : (x - 5) / 7 = 26 :=
sorry

end cindy_correct_answer_l44_44305


namespace least_product_of_distinct_primes_over_30_l44_44100

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l44_44100


namespace number_drawn_from_3rd_group_l44_44293

theorem number_drawn_from_3rd_group {n k : ℕ} (pop_size : ℕ) (sample_size : ℕ) 
  (drawn_from_group : ℕ → ℕ) (group_id : ℕ) (num_in_13th_group : ℕ) : 
  pop_size = 160 → 
  sample_size = 20 → 
  (∀ i, 1 ≤ i ∧ i ≤ sample_size → ∃ j, group_id = i ∧ 
    (j = (i - 1) * (pop_size / sample_size) + drawn_from_group 1)) → 
  num_in_13th_group = 101 → 
  drawn_from_group 3 = 21 := 
by
  intros hp hs hg h13
  sorry

end number_drawn_from_3rd_group_l44_44293


namespace sequence_sum_problem_l44_44139

theorem sequence_sum_problem :
  let seq := [72, 76, 80, 84, 88, 92, 96, 100, 104, 108]
  3 * (seq.sum) = 2700 :=
by
  sorry

end sequence_sum_problem_l44_44139


namespace greatest_visible_unit_cubes_from_corner_l44_44449

theorem greatest_visible_unit_cubes_from_corner
  (n : ℕ) (units : ℕ) 
  (cube_volume : ∀ x, x = 1000)
  (face_size : ∀ x, x = 10) :
  (units = 274) :=
by sorry

end greatest_visible_unit_cubes_from_corner_l44_44449


namespace train_speed_l44_44809

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) :
  distance_meters = 180 →
  time_seconds = 17.998560115190784 →
  ((distance_meters / 1000) / (time_seconds / 3600)) = 36.00360072014403 :=
by 
  intros h1 h2
  rw [h1, h2]
  sorry

end train_speed_l44_44809


namespace probability_colors_match_l44_44056

section ProbabilityJellyBeans

structure JellyBeans where
  green : ℕ
  blue : ℕ
  red : ℕ

def total_jellybeans (jb : JellyBeans) : ℕ :=
  jb.green + jb.blue + jb.red

-- Define the situation using structures
def lila_jellybeans : JellyBeans := { green := 1, blue := 1, red := 1 }
def max_jellybeans : JellyBeans := { green := 2, blue := 1, red := 3 }

-- Define probabilities
noncomputable def probability (count : ℕ) (total : ℕ) : ℚ :=
  if total = 0 then 0 else (count : ℚ) / (total : ℚ)

-- Main theorem
theorem probability_colors_match :
  probability lila_jellybeans.green (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.green (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.blue (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.blue (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.red (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.red (total_jellybeans max_jellybeans) = 1 / 3 :=
by sorry

end ProbabilityJellyBeans

end probability_colors_match_l44_44056


namespace find_m_value_l44_44822

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
    (hf : ∀ x, f x = 3 * x ^ 2 - 1 / x + 4)
    (hg : ∀ x, g x = x ^ 2 - m)
    (hfg : f 3 - g 3 = 5) :
    m = -50 / 3 :=
  sorry

end find_m_value_l44_44822


namespace probability_at_least_one_expired_l44_44133

theorem probability_at_least_one_expired (total_bottles expired_bottles selected_bottles : ℕ) : 
  total_bottles = 10 → expired_bottles = 3 → selected_bottles = 3 → 
  (∃ probability, probability = 17 / 24) :=
by
  sorry

end probability_at_least_one_expired_l44_44133


namespace sum_of_squares_l44_44259

theorem sum_of_squares (x y z : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (h_sum : x * 1 + y * 2 + z * 3 = 12) : x^2 + y^2 + z^2 = 56 :=
by
  sorry

end sum_of_squares_l44_44259


namespace ceiling_of_square_frac_l44_44498

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l44_44498


namespace tangent_line_intersection_l44_44787

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l44_44787


namespace students_in_each_group_l44_44624

theorem students_in_each_group (num_boys : ℕ) (num_girls : ℕ) (num_groups : ℕ) 
  (h_boys : num_boys = 26) (h_girls : num_girls = 46) (h_groups : num_groups = 8) : 
  (num_boys + num_girls) / num_groups = 9 := 
by 
  sorry

end students_in_each_group_l44_44624


namespace arithmetic_example_l44_44475

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l44_44475


namespace flower_position_after_50_beats_l44_44062

-- Define the number of students
def num_students : Nat := 7

-- Define the initial position of the flower
def initial_position : Nat := 1

-- Define the number of drum beats
def drum_beats : Nat := 50

-- Theorem stating that after 50 drum beats, the flower will be with the 2nd student
theorem flower_position_after_50_beats : 
  (initial_position + (drum_beats % num_students)) % num_students = 2 := by
  -- Start the proof (this part usually would contain the actual proof logic)
  sorry

end flower_position_after_50_beats_l44_44062


namespace find_line_eq_show_point_on_circle_l44_44162

noncomputable section

variables {x y x0 y0 : ℝ} (P Q : ℝ × ℝ) (h1 : y0 ≠ 0)
  (h2 : P = (x0, y0))
  (h3 : P.1^2/4 + P.2^2/3 = 1)
  (h4 : Q = (x0/4, y0/3))

theorem find_line_eq (M : ℝ × ℝ) (hM : ∀ (M : ℝ × ℝ), 
  ((P.1 - M.1) , (P.2 - M.2)) • (Q.1 , Q.2) = 0) :
  ∀ (x0 y0 : ℝ), y0 ≠ 0 → ∀ (x y : ℝ), 
  (x0 * x / 4 + y0 * y / 3 = 1) :=
by sorry
  
theorem show_point_on_circle (F S : ℝ × ℝ)
  (hF : F = (1, 0)) (hs : ∀ (x0 y0 : ℝ), y0 ≠ 0 → 
  S = (4, 0) ∧ ((S.1 - P.1) ^ 2 + (S.2 - P.2) ^ 2 = 36)) :
  ∀ (x y : ℝ), 
  (x - 1) ^ 2 + y ^ 2 = 36 := 
by sorry

end find_line_eq_show_point_on_circle_l44_44162


namespace farmer_profit_l44_44288

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end farmer_profit_l44_44288


namespace floor_width_l44_44609

theorem floor_width (tile_length tile_width floor_length max_tiles : ℕ) (h1 : tile_length = 25) (h2 : tile_width = 65) (h3 : floor_length = 150) (h4 : max_tiles = 36) :
  ∃ floor_width : ℕ, floor_width = 450 :=
by
  sorry

end floor_width_l44_44609


namespace true_proposition_l44_44166

-- Definitions based on the conditions
def p (x : ℝ) := x * (x - 1) ≠ 0 → x ≠ 0 ∧ x ≠ 1
def q (a b c : ℝ) := a > b → c > 0 → a * c > b * c

-- The theorem based on the question and the conditions
theorem true_proposition (x a b c : ℝ) (hp : p x) (hq_false : ¬ q a b c) : p x ∨ q a b c :=
by
  sorry

end true_proposition_l44_44166


namespace avg_choc_pieces_per_cookie_l44_44143

theorem avg_choc_pieces_per_cookie {cookies chips mms pieces : ℕ} 
  (h1 : cookies = 48) 
  (h2 : chips = 108) 
  (h3 : mms = chips / 3) 
  (h4 : pieces = chips + mms) : 
  pieces / cookies = 3 := 
by sorry

end avg_choc_pieces_per_cookie_l44_44143


namespace sequence_general_formula_l44_44171

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- because sequences in the solution are 1-indexed.
  | 1 => 2
  | k+2 => sequence (k+1) + 3 * (k+1)

theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : 
  sequence n = 2 + 3 * n * (n - 1) / 2 :=
by
  sorry

#eval sequence 1  -- should output 2
#eval sequence 2  -- should output 5
#eval sequence 3  -- should output 11
#eval sequence 4  -- should output 20
#eval sequence 5  -- should output 32
#eval sequence 6  -- should output 47

end sequence_general_formula_l44_44171


namespace problem_statement_l44_44426

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l44_44426


namespace sum_series_l44_44149

theorem sum_series : (List.sum [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56, -59]) = -30 :=
by
  sorry

end sum_series_l44_44149


namespace relationship_xy_l44_44875

variable (x y : ℝ)

theorem relationship_xy (h₁ : x - y > x + 2) (h₂ : x + y + 3 < y - 1) : x < -4 ∧ y < -2 := 
by sorry

end relationship_xy_l44_44875


namespace collinear_points_value_l44_44695

/-- 
If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, 
then the value of a + b is 7.
-/
theorem collinear_points_value (a b : ℝ) (h_collinear : ∃ l : ℝ → ℝ × ℝ × ℝ, 
  l 0 = (2, a, b) ∧ l 1 = (a, 3, b) ∧ l 2 = (a, b, 4) ∧ 
  ∀ t s : ℝ, l t = l s → t = s) :
  a + b = 7 :=
sorry

end collinear_points_value_l44_44695


namespace distribution_schemes_count_l44_44950

theorem distribution_schemes_count :
  let A := 2,
      B := 2,
      total_pieces := 7,
      remaining_pieces := total_pieces - A - B,
      communities := 5 in
  ∑ (x : Finset ℕ) in (Finset.range (remaining_pieces + 1)).powerset, 
      if x.card = 3 ∨ x.card = 2 ∨ x.card = 1 then 
        1 
      else 
        0 = 35 := sorry

end distribution_schemes_count_l44_44950


namespace ratio_of_A_to_B_l44_44280

-- Definitions of the conditions.
def amount_A : ℕ := 200
def total_amount : ℕ := 600
def amount_B : ℕ := total_amount - amount_A

-- The proof statement.
theorem ratio_of_A_to_B :
  amount_A / amount_B = 1 / 2 := 
sorry

end ratio_of_A_to_B_l44_44280


namespace calories_per_shake_l44_44591

theorem calories_per_shake (total_calories_per_day : ℕ) (breakfast_calories : ℕ)
  (lunch_percentage_increase : ℕ) (dinner_multiplier : ℕ) (number_of_shakes : ℕ)
  (daily_calories : ℕ) :
  total_calories_per_day = breakfast_calories +
                            (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100)) +
                            (2 * (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100))) →
  daily_calories = total_calories_per_day + number_of_shakes * (daily_calories - total_calories_per_day) / number_of_shakes →
  daily_calories = 3275 → breakfast_calories = 500 →
  lunch_percentage_increase = 25 →
  dinner_multiplier = 2 →
  number_of_shakes = 3 →
  (daily_calories - total_calories_per_day) / number_of_shakes = 300 := by 
  sorry

end calories_per_shake_l44_44591


namespace remainder_when_A_divided_by_9_l44_44547

theorem remainder_when_A_divided_by_9 (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := 
by {
  sorry
}

end remainder_when_A_divided_by_9_l44_44547


namespace count_positive_area_triangles_l44_44867

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l44_44867


namespace find_k_l44_44041

variable (m n p k : ℝ)

-- Conditions
def cond1 : Prop := m = 2 * n + 5
def cond2 : Prop := p = 3 * m - 4
def cond3 : Prop := m + 4 = 2 * (n + k) + 5
def cond4 : Prop := p + 3 = 3 * (m + 4) - 4

theorem find_k (h1 : cond1 m n)
               (h2 : cond2 m p)
               (h3 : cond3 m n k)
               (h4 : cond4 m p) :
               k = 2 :=
  sorry

end find_k_l44_44041


namespace mutually_exclusive_event_l44_44989

-- Define the events
def hits_first_shot : Prop := sorry  -- Placeholder for "hits the target on the first shot"
def hits_second_shot : Prop := sorry  -- Placeholder for "hits the target on the second shot"
def misses_first_shot : Prop := ¬ hits_first_shot
def misses_second_shot : Prop := ¬ hits_second_shot

-- Define the main events in the problem
def hitting_at_least_once : Prop := hits_first_shot ∨ hits_second_shot
def missing_both_times : Prop := misses_first_shot ∧ misses_second_shot

-- Statement of the theorem
theorem mutually_exclusive_event :
  missing_both_times ↔ ¬ hitting_at_least_once :=
by sorry

end mutually_exclusive_event_l44_44989


namespace toothpick_grid_l44_44147

theorem toothpick_grid (l w : ℕ) (h_l : l = 45) (h_w : w = 25) :
  let effective_vertical_lines := l + 1 - (l + 1) / 5
  let effective_horizontal_lines := w + 1 - (w + 1) / 5
  let vertical_toothpicks := effective_vertical_lines * w
  let horizontal_toothpicks := effective_horizontal_lines * l
  let total_toothpicks := vertical_toothpicks + horizontal_toothpicks
  total_toothpicks = 1722 :=
by {
  sorry
}

end toothpick_grid_l44_44147


namespace stephen_total_distance_l44_44920

def speed_first_segment := 16 -- miles per hour
def time_first_segment := 10 / 60 -- hours

def speed_second_segment := 12 -- miles per hour
def headwind := 2 -- miles per hour
def actual_speed_second_segment := speed_second_segment - headwind
def time_second_segment := 20 / 60 -- hours

def speed_third_segment := 20 -- miles per hour
def tailwind := 4 -- miles per hour
def actual_speed_third_segment := speed_third_segment + tailwind
def time_third_segment := 15 / 60 -- hours

def distance_first_segment := speed_first_segment * time_first_segment
def distance_second_segment := actual_speed_second_segment * time_second_segment
def distance_third_segment := actual_speed_third_segment * time_third_segment

theorem stephen_total_distance : distance_first_segment + distance_second_segment + distance_third_segment = 12 := by
  sorry

end stephen_total_distance_l44_44920


namespace integer_modulo_solution_l44_44540

theorem integer_modulo_solution (a : ℤ) : 
  (5 ∣ a^3 + 3 * a + 1) ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  exact sorry

end integer_modulo_solution_l44_44540


namespace simplify_expression_l44_44394

variable (x y : ℝ)

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 9 * y = 45 * x + 9 * y := 
by sorry

end simplify_expression_l44_44394


namespace range_of_a_l44_44014

-- Define the condition p
def p (x : ℝ) : Prop := (2 * x^2 - 3 * x + 1) ≤ 0

-- Define the condition q
def q (x a : ℝ) : Prop := (x^2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0

-- Lean statement for the problem
theorem range_of_a (a : ℝ) : (¬ (∃ x, p x) → ¬ (∃ x, q x a)) → ((0 : ℝ) ≤ a ∧ a ≤ (1 / 2 : ℝ)) :=
by 
  sorry

end range_of_a_l44_44014


namespace staircase_perimeter_l44_44209

theorem staircase_perimeter (area : ℝ) (side_length : ℝ) (num_sides : ℕ) (right_angles : Prop) :
  area = 85 ∧ side_length = 1 ∧ num_sides = 10 ∧ right_angles → 
  ∃ perimeter : ℝ, perimeter = 30.5 :=
by
  intro h
  sorry

end staircase_perimeter_l44_44209


namespace distance_midpoints_eq_2_5_l44_44918

theorem distance_midpoints_eq_2_5 (A B C : ℝ) (hAB : A < B) (hBC : B < C) (hAC_len : C - A = 5) :
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    (M2 - M1 = 2.5) :=
by
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    sorry

end distance_midpoints_eq_2_5_l44_44918


namespace sequence_a_n_definition_l44_44674

theorem sequence_a_n_definition (a : ℕ+ → ℝ) 
  (h₀ : ∀ n : ℕ+, a (n + 1) = 2016 * a n / (2014 * a n + 2016))
  (h₁ : a 1 = 1) : 
  a 2017 = 1008 / (1007 * 2017 + 1) :=
sorry

end sequence_a_n_definition_l44_44674


namespace gcd_65536_49152_l44_44669

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 :=
by
  sorry

end gcd_65536_49152_l44_44669


namespace obtuse_vertex_angle_is_135_l44_44206

-- Define the obtuse scalene triangle with the given properties
variables {a b c : ℝ} (triangle : Triangle ℝ)
variables (φ : ℝ) (h_obtuse : φ > 90 ∧ φ < 180) (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_side_relation : a^2 + b^2 = 2 * c^2) (h_sine_obtuse : Real.sin φ = Real.sqrt 2 / 2)

-- The measure of the obtuse vertex angle is 135 degrees
theorem obtuse_vertex_angle_is_135 :
  φ = 135 := by
  sorry

end obtuse_vertex_angle_is_135_l44_44206


namespace smallest_possible_average_l44_44769

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def proper_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8

theorem smallest_possible_average :
  ∃ n : ℕ, (n + 2) - n = 2 ∧ (sum_of_digits n + sum_of_digits (n + 2)) % 4 = 0 ∧ (∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8) ∧ ∀ (d : ℕ), d ∈ (n + 2).digits 10 → d = 0 ∨ d = 4 ∨ d = 8 
  ∧ (n + (n + 2)) / 2 = 249 :=
sorry

end smallest_possible_average_l44_44769


namespace probability_exactly_three_even_l44_44138

theorem probability_exactly_three_even (p : ℕ → ℚ) (n : ℕ) (k : ℕ) (h : p 20 = 1/2 ∧ n = 5 ∧ k = 3) :
  (∃ C : ℚ, (C = (Nat.choose n k : ℚ)) ∧ (p 20)^n = 1/32) → (C * 1/32 = 5/16) :=
by
  sorry

end probability_exactly_three_even_l44_44138


namespace remainder_when_1_stmt_l44_44333

-- Define the polynomial g(s)
def g (s : ℚ) : ℚ := s^15 + 1

-- Define the remainder theorem statement in the context of this problem
theorem remainder_when_1_stmt (s : ℚ) : g 1 = 2 :=
  sorry

end remainder_when_1_stmt_l44_44333


namespace number_of_valid_triangles_l44_44871

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l44_44871


namespace cos_alpha_minus_beta_l44_44552

theorem cos_alpha_minus_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_cos_add : Real.cos (α + β) = -5 / 13)
  (h_tan_sum : Real.tan α + Real.tan β = 3) :
  Real.cos (α - β) = 1 :=
by
  sorry

end cos_alpha_minus_beta_l44_44552


namespace count_valid_age_pairs_l44_44894

theorem count_valid_age_pairs :
  ∃ (d n : ℕ) (a b : ℕ), 10 * a + b ≥ 30 ∧
                       10 * b + a ≥ 35 ∧
                       b > a ∧
                       ∃ k : ℕ, k = 10 := 
sorry

end count_valid_age_pairs_l44_44894


namespace non_degenerate_triangles_l44_44869

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l44_44869


namespace total_revenue_4706_l44_44127

noncomputable def totalTicketRevenue (seats : ℕ) (show2pm : ℕ × ℕ) (show5pm : ℕ × ℕ) (show8pm : ℕ × ℕ) : ℕ :=
  let revenue2pm := show2pm.1 * 4 + (seats - show2pm.1) * 6
  let revenue5pm := show5pm.1 * 5 + (seats - show5pm.1) * 8
  let revenue8pm := show8pm.1 * 7 + (show8pm.2 - show8pm.1) * 10
  revenue2pm + revenue5pm + revenue8pm

theorem total_revenue_4706 :
  totalTicketRevenue 250 (135, 250) (160, 250) (98, 225) = 4706 :=
by
  unfold totalTicketRevenue
  -- We provide the proof steps here in a real proof scenario.
  -- We are focusing on the statement formulation only.
  sorry

end total_revenue_4706_l44_44127


namespace af2_plus_bfg_plus_cg2_geq_0_l44_44240

theorem af2_plus_bfg_plus_cg2_geq_0 (a b c : ℝ) (f g : ℝ) :
  (a * f^2 + b * f * g + c * g^2 ≥ 0) ↔ (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) := 
sorry

end af2_plus_bfg_plus_cg2_geq_0_l44_44240


namespace major_axis_length_l44_44403

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by
  sorry

end major_axis_length_l44_44403


namespace cindy_correct_answer_l44_44820

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end cindy_correct_answer_l44_44820


namespace ceil_square_eq_four_l44_44515

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l44_44515


namespace problem_l44_44556

theorem problem 
  (a : ℝ) 
  (h_a : ∀ x : ℝ, |x + 1| - |2 - x| ≤ a ∧ a ≤ |x + 1| + |2 - x|)
  {m n : ℝ} 
  (h_mn : m > n) 
  (h_n : n > 0)
  (h: a = 3) 
  : 2 * m + 1 / (m^2 - 2 * m * n + n^2) ≥ 2 * n + a :=
by
  sorry

end problem_l44_44556


namespace tangent_line_intersects_x_axis_at_9_div_2_l44_44794

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l44_44794


namespace factorize_x_squared_plus_2x_l44_44316

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l44_44316


namespace days_required_by_x_l44_44963

theorem days_required_by_x (x y : ℝ) 
  (h1 : (1 / x + 1 / y = 1 / 12)) 
  (h2 : (1 / y = 1 / 24)) : 
  x = 24 := 
by
  sorry

end days_required_by_x_l44_44963


namespace smallest_positive_integer_l44_44774

theorem smallest_positive_integer 
  (x : ℤ) (h1 : x % 6 = 3) (h2 : x % 8 = 2) : x = 33 :=
sorry

end smallest_positive_integer_l44_44774


namespace find_lengths_l44_44379

-- Given definitions and conditions
variables (A B C D P K : Point)
variable (α : ℝ)
variable (AB AC BC : ℝ)
variable [Nonempty ℝ]

-- Define that D is the foot of the altitude from B
def is_altitude_foot (B A C D : Point) : Prop :=
  altitude_from B = D

-- Define specific lengths and their properties
def is_midpoint (A C K : Point) : Prop := midpoint A C K
def is_incenter (BCD P : Triangle) : Prop := incenter B C D = P
def is_centroid (ABC P : Triangle) : Prop := centroid A B C = P

-- Main theorem statement
theorem find_lengths (h_altitude_foot : is_altitude_foot B A C D)
    (h_AB : AB = 1)
    (h_incenter_centroid : is_incenter B C D P ∧ is_centroid A B C P) :
    AC = sqrt (5 / 2) ∧ BC = sqrt (5 / 2) :=
begin
  sorry, -- The proof would go here
end

end find_lengths_l44_44379


namespace correct_average_and_variance_l44_44881

theorem correct_average_and_variance
  (n : ℕ) (avg incorrect_variance correct_variance : ℝ)
  (incorrect_score1 actual_score1 incorrect_score2 actual_score2 : ℝ)
  (H1 : n = 48)
  (H2 : avg = 70)
  (H3 : incorrect_variance = 75)
  (H4 : incorrect_score1 = 50)
  (H5 : actual_score1 = 80)
  (H6 : incorrect_score2 = 100)
  (H7 : actual_score2 = 70)
  (Havg : avg = (n * avg - incorrect_score1 - incorrect_score2 + actual_score1 + actual_score2) / n)
  (Hvar : correct_variance = incorrect_variance + (actual_score1 - avg) ^ 2 + (actual_score2 - avg) ^ 2
                     - (incorrect_score1 - avg) ^ 2 - (incorrect_score2 - avg) ^ 2 / n) :
  avg = 70 ∧ correct_variance = 50 :=
by {
  sorry
}

end correct_average_and_variance_l44_44881


namespace complement_of_union_in_U_l44_44222

-- Define the universal set U
def U : Set ℕ := {x | x < 6 ∧ x > 0}

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of A ∪ B in U
def complement_U_union_A_B : Set ℕ := {x | x ∈ U ∧ x ∉ (A ∪ B)}

theorem complement_of_union_in_U : complement_U_union_A_B = {2, 4} :=
by {
  -- Placeholder for the proof
  sorry
}

end complement_of_union_in_U_l44_44222


namespace complement_intersection_l44_44683

noncomputable def M : Set ℝ := {x | 2 / x < 1}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

theorem complement_intersection : 
  ((Set.univ \ M) ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_intersection_l44_44683


namespace ceiling_of_square_of_neg_7_over_4_is_4_l44_44507

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l44_44507


namespace cos_triplet_angle_l44_44179

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l44_44179


namespace parallel_resistors_l44_44368
noncomputable def resistance_R (x y z w : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z + 1/w)

theorem parallel_resistors :
  resistance_R 5 7 3 9 = 315 / 248 :=
by
  sorry

end parallel_resistors_l44_44368


namespace required_sticks_l44_44982

variables (x y : ℕ)
variables (h1 : 2 * x + 3 * y = 96)
variables (h2 : x + y = 40)

theorem required_sticks (x y : ℕ) (h1 : 2 * x + 3 * y = 96) (h2 : x + y = 40) : 
  x = 24 ∧ y = 16 ∧ (96 - (x * 2 + y * 3) / 2) = 116 :=
by
  sorry

end required_sticks_l44_44982


namespace each_person_tip_l44_44376

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l44_44376


namespace convert_3652_from_base7_to_base10_l44_44214

def base7ToBase10(n : ℕ) := 
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d0 * (7^0) + d1 * (7^1) + d2 * (7^2) + d3 * (7^3)

theorem convert_3652_from_base7_to_base10 : base7ToBase10 3652 = 1360 :=
by
  sorry

end convert_3652_from_base7_to_base10_l44_44214


namespace find_x_l44_44115

theorem find_x : ∃ (x : ℝ), x > 0 ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_x_l44_44115


namespace find_heaviest_and_lightest_l44_44413

-- Definition of the main problem conditions
def coins : ℕ := 10
def max_weighings : ℕ := 13
def distinct_weights (c : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → i < c → j < c → weight i ≠ weight j

-- Noncomputed property representing the weight of each coin
noncomputable def weight : ℕ → ℝ := sorry

-- The main theorem statement
theorem find_heaviest_and_lightest (c : ℕ) (mw : ℕ) (dw : distinct_weights c) : c = coins ∧ mw = max_weighings
  → ∃ (h l : ℕ), h < c ∧ l < c ∧ (∀ (i : ℕ), i < c → weight i ≤ weight h ∧ weight i ≥ weight l) :=
by
  sorry

end find_heaviest_and_lightest_l44_44413


namespace number_of_piles_l44_44235

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end number_of_piles_l44_44235


namespace transformed_stats_l44_44859

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt ((l.map (λ x => (x - mean l)^2)).sum / l.length)

theorem transformed_stats (l : List ℝ) 
  (hmean : mean l = 10)
  (hstddev : std_dev l = 2) :
  mean (l.map (λ x => 2 * x - 1)) = 19 ∧ std_dev (l.map (λ x => 2 * x - 1)) = 4 := by
  sorry

end transformed_stats_l44_44859


namespace correct_equation_l44_44941

variable (x : ℝ)
axiom area_eq_720 : x * (x - 6) = 720

theorem correct_equation : x * (x - 6) = 720 := by
  exact area_eq_720

end correct_equation_l44_44941


namespace meeting_time_final_time_statement_l44_44984

-- Define the speeds and distance as given conditions
def brodie_speed : ℝ := 50
def ryan_speed : ℝ := 40
def initial_distance : ℝ := 120

-- Define what we know about their meeting time and validate it mathematically
theorem meeting_time :
  (initial_distance / (brodie_speed + ryan_speed)) = 4 / 3 := sorry

-- Assert the time in minutes for completeness
noncomputable def time_in_minutes : ℝ := ((4 / 3) * 60)

-- Assert final statement matching the answer in hours and minutes
theorem final_time_statement :
  time_in_minutes = 80 := sorry

end meeting_time_final_time_statement_l44_44984


namespace total_cost_of_pencils_l44_44068

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end total_cost_of_pencils_l44_44068


namespace least_prime_product_l44_44102
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l44_44102


namespace tangent_point_value_l44_44796

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l44_44796


namespace at_least_one_f_nonnegative_l44_44428

theorem at_least_one_f_nonnegative 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m * n > 1) : 
  (m^2 - m ≥ 0) ∨ (n^2 - n ≥ 0) :=
by sorry

end at_least_one_f_nonnegative_l44_44428


namespace average_chocolate_pieces_per_cookie_l44_44144

-- Definitions from the conditions
def number_of_cookies := 48
def number_of_chocolate_chips := 108
def number_of_m_and_ms := (1 / 3 : ℝ) * number_of_chocolate_chips
def total_number_of_chocolate_pieces := number_of_chocolate_chips + number_of_m_and_ms

-- Statement to prove
theorem average_chocolate_pieces_per_cookie : 
  total_number_of_chocolate_pieces / number_of_cookies = 3 := by
  sorry

end average_chocolate_pieces_per_cookie_l44_44144


namespace prob_two_red_balls_consecutively_without_replacement_l44_44038

def numOfRedBalls : ℕ := 3
def totalNumOfBalls : ℕ := 8

theorem prob_two_red_balls_consecutively_without_replacement :
  (numOfRedBalls / totalNumOfBalls) * ((numOfRedBalls - 1) / (totalNumOfBalls - 1)) = 3 / 28 :=
by
  sorry

end prob_two_red_balls_consecutively_without_replacement_l44_44038


namespace subset_S_A_inter_B_nonempty_l44_44386

open Finset

-- Definitions of sets A and B
def A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def B : Finset ℕ := {4, 5, 6, 7, 8}

-- Definition of the subset S and its condition
def S : Finset ℕ := {5, 6}

-- The statement to be proved
theorem subset_S_A_inter_B_nonempty : S ⊆ A ∧ S ∩ B ≠ ∅ :=
by {
  sorry -- proof to be provided
}

end subset_S_A_inter_B_nonempty_l44_44386


namespace caffeine_per_energy_drink_l44_44404

variable (amount_of_caffeine_per_drink : ℕ)

def maximum_safe_caffeine_per_day := 500
def drinks_per_day := 4
def additional_safe_amount := 20

theorem caffeine_per_energy_drink :
  4 * amount_of_caffeine_per_drink + additional_safe_amount = maximum_safe_caffeine_per_day →
  amount_of_caffeine_per_drink = 120 :=
by
  sorry

end caffeine_per_energy_drink_l44_44404


namespace square_neg_2x_squared_l44_44446

theorem square_neg_2x_squared (x : ℝ) : (-2 * x ^ 2) ^ 2 = 4 * x ^ 4 :=
by
  sorry

end square_neg_2x_squared_l44_44446


namespace sum_minimum_values_l44_44599

def P (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem sum_minimum_values (a b c d e f : ℝ)
  (hPQ : ∀ x, P (Q x d e f) a b c = 0 → x = -4 ∨ x = -2 ∨ x = 0 ∨ x = 2 ∨ x = 4)
  (hQP : ∀ x, Q (P x a b c) d e f = 0 → x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 3) :
  P 0 a b c + Q 0 d e f = -20 := sorry

end sum_minimum_values_l44_44599


namespace factorization_correct_l44_44321

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l44_44321


namespace zachary_cans_first_day_l44_44440

theorem zachary_cans_first_day :
  ∃ (first_day_cans : ℕ),
    ∃ (second_day_cans : ℕ),
      ∃ (third_day_cans : ℕ),
        ∃ (seventh_day_cans : ℕ),
          second_day_cans = 9 ∧
          third_day_cans = 14 ∧
          (∀ (n : ℕ), 2 ≤ n ∧ n < 7 → third_day_cans = second_day_cans + 5) →
          seventh_day_cans = 34 ∧
          first_day_cans = second_day_cans - 5 ∧
          first_day_cans = 4 :=

by
  sorry

end zachary_cans_first_day_l44_44440


namespace ceil_square_neg_seven_over_four_l44_44505

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l44_44505


namespace ceil_square_neg_fraction_l44_44526

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l44_44526


namespace complement_A_union_B_range_of_m_l44_44343

def setA : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 5*x - 14) }
def setB : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (-x^2 - 7*x - 12) }
def setC (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem complement_A_union_B :
  (A ∪ B)ᶜ = Set.Ioo (-2 : ℝ) 7 :=
sorry

theorem range_of_m (m : ℝ) :
  (A ∪ setC m = A) → (m < 2 ∨ m ≥ 6) :=
sorry

end complement_A_union_B_range_of_m_l44_44343


namespace book_pages_read_l44_44618

theorem book_pages_read (pages_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) :
  (pages_per_day = 100) →
  (days_per_week = 3) →
  (weeks = 7) →
  total_pages = pages_per_day * days_per_week * weeks →
  total_pages = 2100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end book_pages_read_l44_44618


namespace product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l44_44201

theorem product_div_sum_eq_5 (x : ℤ) (h : (x^3 - x) / (3 * x) = 5) : x = 4 := by
  sorry

theorem quotient_integer_condition (x : ℤ) : ((∃ k : ℤ, x = 3 * k + 1) ∨ (∃ k : ℤ, x = 3 * k - 1)) ↔ ∃ q : ℤ, (x^3 - x) / (3 * x) = q := by
  sorry

theorem next_consecutive_set (x : ℤ) (h : x = 4) : x - 1 = 3 ∧ x = 4 ∧ x + 1 = 5 := by
  sorry

end product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l44_44201


namespace women_to_total_population_ratio_l44_44736

/-- original population of Salem -/
def original_population (pop_leesburg : ℕ) : ℕ := 15 * pop_leesburg

/-- new population after people moved out -/
def new_population (orig_pop : ℕ) (moved_out : ℕ) : ℕ := orig_pop - moved_out

/-- ratio of two numbers -/
def ratio (num : ℕ) (denom : ℕ) : ℚ := num / denom

/-- population data -/
structure PopulationData :=
  (pop_leesburg : ℕ)
  (moved_out : ℕ)
  (women : ℕ)

/-- prove ratio of women to the total population in Salem -/
theorem women_to_total_population_ratio (data : PopulationData)
  (pop_leesburg_eq : data.pop_leesburg = 58940)
  (moved_out_eq : data.moved_out = 130000)
  (women_eq : data.women = 377050) : 
  ratio data.women (new_population (original_population data.pop_leesburg) data.moved_out) = 377050 / 754100 :=
by
  sorry

end women_to_total_population_ratio_l44_44736


namespace cost_of_pencils_l44_44906

def cost_of_notebooks : ℝ := 3 * 1.2
def cost_of_pens : ℝ := 1.7
def total_spent : ℝ := 6.8

theorem cost_of_pencils :
  total_spent - (cost_of_notebooks + cost_of_pens) = 1.5 :=
by
  sorry

end cost_of_pencils_l44_44906


namespace exists_special_function_l44_44665

theorem exists_special_function : ∃ (s : ℚ → ℤ), (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) ∧ (∀ x : ℚ, s x = 1 ∨ s x = -1) :=
by
  sorry

end exists_special_function_l44_44665


namespace sin_810_eq_one_l44_44467

theorem sin_810_eq_one : Real.sin (810 * Real.pi / 180) = 1 :=
by
  -- You can add the proof here
  sorry

end sin_810_eq_one_l44_44467


namespace find_divisor_l44_44778

theorem find_divisor :
  ∃ D : ℝ, 527652 = (D * 392.57) + 48.25 ∧ D = 1344.25 :=
by
  sorry

end find_divisor_l44_44778


namespace ticket_price_l44_44230

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l44_44230


namespace other_root_of_quadratic_l44_44916

theorem other_root_of_quadratic (m : ℝ) (h : ∃ α : ℝ, α = 1 ∧ (3 * α^2 + m * α = 5)) :
  ∃ β : ℝ, β = -5 / 3 :=
by
  sorry

end other_root_of_quadratic_l44_44916


namespace farmer_profit_l44_44286

def piglet_cost_per_month : Int := 10
def pig_revenue : Int := 300
def num_piglets_sold_early : Int := 3
def num_piglets_sold_late : Int := 3
def early_sale_months : Int := 12
def late_sale_months : Int := 16

def total_profit (num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months : Int) 
  (piglet_cost_per_month pig_revenue : Int) : Int := 
  let early_cost := num_piglets_sold_early * piglet_cost_per_month * early_sale_months
  let late_cost := num_piglets_sold_late * piglet_cost_per_month * late_sale_months
  let total_cost := early_cost + late_cost
  let total_revenue := (num_piglets_sold_early + num_piglets_sold_late) * pig_revenue
  total_revenue - total_cost

theorem farmer_profit : total_profit num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months piglet_cost_per_month pig_revenue = 960 := by
  sorry

end farmer_profit_l44_44286


namespace x_coordinate_l44_44700

theorem x_coordinate (x : ℝ) (y : ℝ) :
  (∃ m : ℝ, m = (0 + 6) / (4 + 8) ∧
            y + 6 = m * (x + 8) ∧
            y = 3) →
  x = 10 :=
by
  sorry

end x_coordinate_l44_44700


namespace ceiling_of_square_of_neg_7_over_4_is_4_l44_44511

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l44_44511


namespace factorize_x_squared_plus_2x_l44_44315

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l44_44315


namespace ceil_square_neg_seven_over_four_l44_44503

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l44_44503


namespace other_endpoint_sum_l44_44725

def endpoint_sum (A B M : (ℝ × ℝ)) : ℝ := 
  let (Ax, Ay) := A
  let (Mx, My) := M
  let (Bx, By) := B
  Bx + By

theorem other_endpoint_sum (A M : (ℝ × ℝ)) (hA : A = (6, 1)) (hM : M = (5, 7)) :
  ∃ B : (ℝ × ℝ), endpoint_sum A B M = 17 :=
by
  use (4, 13)
  rw [endpoint_sum, hA, hM]
  simp
  sorry

end other_endpoint_sum_l44_44725


namespace median_eq_range_le_l44_44012

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end median_eq_range_le_l44_44012


namespace ceil_square_eq_four_l44_44517

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l44_44517


namespace minimum_distance_from_midpoint_to_y_axis_l44_44926

theorem minimum_distance_from_midpoint_to_y_axis (M N : ℝ × ℝ) (P : ℝ × ℝ)
  (hM : M.snd ^ 2 = M.fst) (hN : N.snd ^ 2 = N.fst)
  (hlength : (M.fst - N.fst)^2 + (M.snd - N.snd)^2 = 16)
  (hP : P = ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)) :
  abs P.fst = 7 / 4 :=
sorry

end minimum_distance_from_midpoint_to_y_axis_l44_44926


namespace company_total_parts_l44_44260

noncomputable def total_parts_made (planning_days : ℕ) (initial_rate : ℕ) (extra_rate : ℕ) (extra_parts : ℕ) (x_days : ℕ) : ℕ :=
  let initial_production := planning_days * initial_rate
  let increased_rate := initial_rate + extra_rate
  let actual_production := x_days * increased_rate
  initial_production + actual_production

def planned_production (planning_days : ℕ) (initial_rate : ℕ) (x_days : ℕ) : ℕ :=
  planning_days * initial_rate + x_days * initial_rate

theorem company_total_parts
  (planning_days : ℕ)
  (initial_rate : ℕ)
  (extra_rate : ℕ)
  (extra_parts : ℕ)
  (x_days : ℕ)
  (h1 : planning_days = 3)
  (h2 : initial_rate = 40)
  (h3 : extra_rate = 7)
  (h4 : extra_parts = 150)
  (h5 : x_days = 21)
  (h6 : 7 * x_days = extra_parts) :
  total_parts_made planning_days initial_rate extra_rate extra_parts x_days = 1107 := by
  sorry

end company_total_parts_l44_44260


namespace chord_length_l44_44681

theorem chord_length
  (l_eq : ∀ (rho theta : ℝ), rho * (Real.sin theta - Real.cos theta) = 1)
  (gamma_eq : ∀ (rho : ℝ) (theta : ℝ), rho = 1) :
  ∃ AB : ℝ, AB = Real.sqrt 2 :=
by
  sorry

end chord_length_l44_44681


namespace smallest_number_increased_by_seven_divisible_by_37_47_53_l44_44265

theorem smallest_number_increased_by_seven_divisible_by_37_47_53 : 
  ∃ n : ℕ, (n + 7) % 37 = 0 ∧ (n + 7) % 47 = 0 ∧ (n + 7) % 53 = 0 ∧ n = 92160 :=
by
  sorry

end smallest_number_increased_by_seven_divisible_by_37_47_53_l44_44265


namespace derivative_y_l44_44830

noncomputable def y (x : ℝ) : ℝ :=
  (1 / 4) * Real.log ((x - 1) / (x + 1)) - (1 / 2) * Real.arctan x

theorem derivative_y (x : ℝ) : deriv y x = 1 / (x^4 - 1) :=
  sorry

end derivative_y_l44_44830


namespace multiplicative_inverse_mod_l44_44306

-- We define our variables
def a := 154
def m := 257
def inv_a := 20

-- Our main theorem stating that inv_a is indeed the multiplicative inverse of a modulo m
theorem multiplicative_inverse_mod : (a * inv_a) % m = 1 := by
  sorry

end multiplicative_inverse_mod_l44_44306


namespace find_vector_b_l44_44860

def vector_collinear (a b : ℝ × ℝ) : Prop :=
    ∃ k : ℝ, (a.1 = k * b.1 ∧ a.2 = k * b.2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2

theorem find_vector_b (a b : ℝ × ℝ) (h_collinear : vector_collinear a b) (h_dot : dot_product a b = -10) : b = (-4, 2) :=
    by
        sorry

end find_vector_b_l44_44860


namespace ceiling_of_square_frac_l44_44499

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l44_44499


namespace linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l44_44346

theorem linear_function_passing_through_point_and_intersecting_another_line (
  k b : ℝ)
  (h1 : (∀ x y : ℝ, y = k * x + b → ((x = 3 ∧ y = -3) ∨ (x = 3/4 ∧ y = 0))))
  (h2 : (∀ x : ℝ, 0 = (4 * x - 3) → x = 3/4))
  : k = -4 / 3 ∧ b = 1 := 
sorry

theorem area_of_triangle (
  k b : ℝ)
  (h1 : k = -4 / 3 ∧ b = 1)
  : 1 / 2 * 3 / 4 * 1 = 3 / 8 := 
sorry

end linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l44_44346


namespace beats_per_week_l44_44893

def beats_per_minute : ℕ := 200
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7

theorem beats_per_week : beats_per_minute * minutes_per_hour * hours_per_day * days_per_week = 168000 := by
  sorry

end beats_per_week_l44_44893


namespace total_percentage_of_failed_candidates_l44_44698

-- Define the given conditions
def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_of_boys_passed : ℚ := 0.28
def percentage_of_girls_passed : ℚ := 0.32

-- Define the proof statement
theorem total_percentage_of_failed_candidates : 
  (total_candidates - (percentage_of_boys_passed * number_of_boys + percentage_of_girls_passed * number_of_girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end total_percentage_of_failed_candidates_l44_44698


namespace even_n_condition_l44_44668

theorem even_n_condition (x : ℝ) (n : ℕ) (h : ∀ x, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : n % 2 = 0 :=
sorry

end even_n_condition_l44_44668


namespace solution_set_inequality_l44_44856

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f (x - 1/2) + f (x + 1) = 0)
variable (h2 : e ^ 3 * f 2018 = 1)
variable (h3 : ∀ x, f x > f'' (-x))
variable (h4 : ∀ x, f x = f (-x))

theorem solution_set_inequality :
  ∀ x, f (x - 1) > 1 / (e ^ x) ↔ x > 3 :=
sorry

end solution_set_inequality_l44_44856


namespace find_initial_pomelos_l44_44295

theorem find_initial_pomelos (g w w' g' : ℕ) 
  (h1 : w = 3 * g)
  (h2 : w' = w - 90)
  (h3 : g' = g - 60)
  (h4 : w' = 4 * g' - 26) 
  : g = 176 :=
by
  sorry

end find_initial_pomelos_l44_44295


namespace max_principals_l44_44990

theorem max_principals (n_years term_length max_principals: ℕ) 
  (h1 : n_years = 12) 
  (h2 : term_length = 4)
  (h3 : max_principals = 4): 
  (∃ p : ℕ, p = max_principals) :=
by
  sorry

end max_principals_l44_44990


namespace validate_triangle_count_l44_44864

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l44_44864


namespace cos_triple_angle_l44_44183

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l44_44183


namespace parametric_equation_solution_l44_44615

noncomputable def solve_parametric_equation (a b : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) : ℝ :=
  (5 / (a - 2 * b))

theorem parametric_equation_solution (a b x : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) 
  (h : (a * x - 3) / (b * x + 1) = 2) : 
  x = solve_parametric_equation a b ha2b ha3b :=
sorry

end parametric_equation_solution_l44_44615


namespace rationalize_denom_l44_44733

-- Assume we know about rationalizing denominators and integer arithmetic
theorem rationalize_denom (a b c : ℚ) : 
  let x := 2 + real.sqrt 5, y := 2 - real.sqrt 5 in
  (a, b, c) = (-9, -4, 5) →
  ((x / y) * (2 + real.sqrt 5) = a + b * real.sqrt 5) → 
  (a * b * c = 180) :=
by intro a b c h₁ h₂; sorry

end rationalize_denom_l44_44733


namespace candy_from_sister_l44_44006

variable (total_neighbors : Nat) (pieces_per_day : Nat) (days : Nat) (total_pieces : Nat)
variable (pieces_per_day_eq : pieces_per_day = 9)
variable (days_eq : days = 9)
variable (total_neighbors_eq : total_neighbors = 66)
variable (total_pieces_eq : total_pieces = 81)

theorem candy_from_sister : 
  total_pieces = total_neighbors + 15 :=
by
  sorry

end candy_from_sister_l44_44006


namespace copies_per_person_l44_44059

-- Definitions derived from the conditions
def pages_per_contract : ℕ := 20
def total_pages_copied : ℕ := 360
def number_of_people : ℕ := 9

-- Theorem stating the result based on the conditions
theorem copies_per_person : (total_pages_copied / pages_per_contract) / number_of_people = 2 := by
  sorry

end copies_per_person_l44_44059


namespace fraction_of_yellow_balls_l44_44037

theorem fraction_of_yellow_balls
  (total_balls : ℕ)
  (fraction_green : ℚ)
  (fraction_blue : ℚ)
  (number_blue : ℕ)
  (number_white : ℕ)
  (total_balls_eq : total_balls = number_blue * (1 / fraction_blue))
  (fraction_green_eq : fraction_green = 1 / 4)
  (fraction_blue_eq : fraction_blue = 1 / 8)
  (number_white_eq : number_white = 26)
  (number_blue_eq : number_blue = 6) :
  (total_balls - (total_balls * fraction_green + number_blue + number_white)) / total_balls = 1 / 12 :=
by
  sorry

end fraction_of_yellow_balls_l44_44037


namespace total_amount_l44_44129

def shares (a b c : ℕ) : Prop :=
  b = 1800 ∧ 2 * b = 3 * a ∧ 3 * c = 4 * b

theorem total_amount (a b c : ℕ) (h : shares a b c) : a + b + c = 5400 :=
by
  have h₁ : 2 * b = 3 * a := h.2.1
  have h₂ : 3 * c = 4 * b := h.2.2
  have hb : b = 1800 := h.1
  sorry

end total_amount_l44_44129


namespace min_perimeter_l44_44130

theorem min_perimeter (a b : ℕ) (h1 : b = 3 * a) (h2 : 3 * a + 8 * a = 11) (h3 : 2 * a + 12 * a = 14)
  : 2 * (15 + 11) = 52 := 
sorry

end min_perimeter_l44_44130


namespace find_stu_l44_44382

open Complex

theorem find_stu (p q r s t u : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h1 : p = (q + r) / (s - 3))
  (h2 : q = (p + r) / (t - 3))
  (h3 : r = (p + q) / (u - 3))
  (h4 : s * t + s * u + t * u = 8)
  (h5 : s + t + u = 4) :
  s * t * u = 10 := 
sorry

end find_stu_l44_44382


namespace gcd_1151_3079_l44_44325

def a : ℕ := 1151
def b : ℕ := 3079

theorem gcd_1151_3079 : gcd a b = 1 := by
  sorry

end gcd_1151_3079_l44_44325


namespace octagon_perimeter_l44_44975

def side_length_meters : ℝ := 2.3
def number_of_sides : ℕ := 8
def meter_to_cm (meters : ℝ) : ℝ := meters * 100

def perimeter_cm (side_length_meters : ℝ) (number_of_sides : ℕ) : ℝ :=
  meter_to_cm side_length_meters * number_of_sides

theorem octagon_perimeter :
  perimeter_cm side_length_meters number_of_sides = 1840 :=
by
  sorry

end octagon_perimeter_l44_44975


namespace solve_problem_l44_44660

def bracket (a b c : ℕ) : ℕ := (a + b) / c

theorem solve_problem :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 :=
by
  sorry

end solve_problem_l44_44660


namespace temperature_difference_l44_44915

theorem temperature_difference : 
  let beijing_temp := -6
  let changtai_temp := 15
  changtai_temp - beijing_temp = 21 := 
by
  -- Let the given temperatures
  let beijing_temp := -6
  let changtai_temp := 15
  -- Perform the subtraction and define the expected equality
  show changtai_temp - beijing_temp = 21
  -- Preliminary proof placeholder
  sorry

end temperature_difference_l44_44915


namespace inequality_solution_range_of_a_l44_44563

noncomputable def f (x : ℝ) : ℝ := |1 - 2 * x| - |1 + x| 

theorem inequality_solution (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := 
by sorry

theorem range_of_a (a x : ℝ) (h : a^2 + 2 * a + |1 + x| < f x) : -3 < a ∧ a < 1 :=
by sorry

end inequality_solution_range_of_a_l44_44563


namespace c_seq_formula_l44_44625

def x_seq (n : ℕ) : ℕ := 2 * n - 1
def y_seq (n : ℕ) : ℕ := n ^ 2
def c_seq (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem c_seq_formula (n : ℕ) : ∀ k, (c_seq k) = (2 * k - 1) ^ 2 :=
by
  sorry

end c_seq_formula_l44_44625


namespace second_intersection_of_parabola_l44_44647

theorem second_intersection_of_parabola (x_vertex_Pi1 x_vertex_Pi2 : ℝ) : 
  (∀ x : ℝ, x = (10 + 13) / 2 → x_vertex_Pi1 = x) →
  (∀ y : ℝ, y = (x_vertex_Pi2 / 2) → x_vertex_Pi1 = y) →
  (x_vertex_Pi2 = 2 * x_vertex_Pi1) →
  (13 + 33) / 2 = x_vertex_Pi2 :=
by
  sorry

end second_intersection_of_parabola_l44_44647


namespace b_came_third_four_times_l44_44699

variable (a b c N : ℕ)

theorem b_came_third_four_times
    (a_pos : a > 0) 
    (b_pos : b > 0) 
    (c_pos : c > 0)
    (a_gt_b : a > b) 
    (b_gt_c : b > c) 
    (a_b_c_sum : a + b + c = 8)
    (score_A : 4 * a + b = 26) 
    (score_B : a + 4 * c = 11) 
    (score_C : 3 * b + 2 * c = 11) 
    (B_won_first_event : a + b + c = 8) : 
    4 * c = 4 := 
sorry

end b_came_third_four_times_l44_44699


namespace min_value_of_sum_l44_44017

theorem min_value_of_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1 / x + 1 / y + 1 / z = 1) :
  x + 4 * y + 9 * z ≥ 36 ∧ (x + 4 * y + 9 * z = 36 ↔ x = 6 ∧ y = 3 ∧ z = 2) := 
sorry

end min_value_of_sum_l44_44017


namespace no_positive_integer_solutions_l44_44927

theorem no_positive_integer_solutions (A : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) :
  ¬(∃ x : ℕ, x^2 - 2 * A * x + A0 = 0) :=
by sorry

end no_positive_integer_solutions_l44_44927


namespace pure_imaginary_number_a_l44_44349

theorem pure_imaginary_number_a (a : ℝ) 
  (h1 : a^2 + 2 * a - 3 = 0)
  (h2 : a^2 - 4 * a + 3 ≠ 0) : a = -3 :=
sorry

end pure_imaginary_number_a_l44_44349


namespace solve_for_x_l44_44852

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 6 * x + 3

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -1) : x = - 3 / 4 :=
by
  sorry

end solve_for_x_l44_44852


namespace correct_equation_l44_44945

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l44_44945


namespace probability_of_blue_ball_l44_44582

theorem probability_of_blue_ball 
(P_red P_yellow P_blue : ℝ) 
(h_red : P_red = 0.48)
(h_yellow : P_yellow = 0.35) 
(h_prob : P_red + P_yellow + P_blue = 1) 
: P_blue = 0.17 := 
sorry

end probability_of_blue_ball_l44_44582


namespace average_salary_of_employees_l44_44619

theorem average_salary_of_employees (A : ℝ)
  (h1 : 24 * A + 11500 = 25 * (A + 400)) :
  A = 1500 := 
by
  sorry

end average_salary_of_employees_l44_44619


namespace middle_tree_less_half_tallest_tree_l44_44251

theorem middle_tree_less_half_tallest_tree (T M S : ℝ)
  (hT : T = 108)
  (hS : S = 1/4 * M)
  (hS_12 : S = 12) :
  (1/2 * T) - M = 6 := 
sorry

end middle_tree_less_half_tallest_tree_l44_44251


namespace lecture_hall_rows_l44_44622

-- We define the total number of seats
def total_seats (n : ℕ) : ℕ := n * (n + 11)

-- We state the problem with the given conditions
theorem lecture_hall_rows : 
  (400 ≤ total_seats n) ∧ (total_seats n ≤ 440) → n = 16 :=
by
  sorry

end lecture_hall_rows_l44_44622


namespace production_volume_increase_l44_44281

theorem production_volume_increase (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 :=
sorry

end production_volume_increase_l44_44281


namespace parallelogram_area_l44_44071

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (base_condition : b = 8) (altitude_condition : h = 2 * b) : 
  A = 128 :=
by 
  sorry

end parallelogram_area_l44_44071


namespace combination_of_seven_choose_three_l44_44061

theorem combination_of_seven_choose_three : nat.choose 7 3 = 35 :=
by {
  sorry
}

end combination_of_seven_choose_three_l44_44061


namespace arithmetic_example_l44_44473

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l44_44473


namespace cos_triple_angle_l44_44182

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l44_44182


namespace problem_statement_l44_44383

def f (x : ℝ) : ℝ := x^5 - x^3 + 1
def g (x : ℝ) : ℝ := x^2 - 2

theorem problem_statement (x1 x2 x3 x4 x5 : ℝ) 
  (h_roots : ∀ x, f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) :
  g x1 * g x2 * g x3 * g x4 * g x5 = -7 := 
sorry

end problem_statement_l44_44383


namespace square_B_perimeter_l44_44741

theorem square_B_perimeter :
  ∀ (sideA sideB : ℝ), (4 * sideA = 24) → (sideB^2 = (sideA^2) / 4) → (4 * sideB = 12) :=
by
  sorry

end square_B_perimeter_l44_44741


namespace part1_part2_l44_44900

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end part1_part2_l44_44900


namespace gcd_54_180_l44_44107

theorem gcd_54_180 : Nat.gcd 54 180 = 18 := by
  sorry

end gcd_54_180_l44_44107


namespace not_possible_total_47_l44_44409

open Nat

theorem not_possible_total_47 (h c : ℕ) : ¬ (13 * h + 5 * c = 47) :=
  sorry

end not_possible_total_47_l44_44409


namespace problem_abcd_eq_14400_l44_44744

theorem problem_abcd_eq_14400
 (a b c d : ℝ)
 (h1 : a^2 + b^2 + c^2 + d^2 = 762)
 (h2 : a * b + c * d = 260)
 (h3 : a * c + b * d = 365)
 (h4 : a * d + b * c = 244) :
 a * b * c * d = 14400 := 
sorry

end problem_abcd_eq_14400_l44_44744


namespace ceiling_of_square_of_neg_7_over_4_is_4_l44_44510

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l44_44510


namespace age_of_youngest_child_l44_44114

/-- Given that the sum of ages of 5 children born at 3-year intervals is 70, prove the age of the youngest child is 8. -/
theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 := 
  sorry

end age_of_youngest_child_l44_44114


namespace scientific_notation_26_billion_l44_44462

theorem scientific_notation_26_billion :
  ∃ (a : ℝ) (n : ℤ), (26 * 10^8 : ℝ) = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.6 ∧ n = 9 :=
sorry

end scientific_notation_26_billion_l44_44462


namespace john_bathroom_uses_during_movie_and_intermissions_l44_44045

-- Define the conditions
def uses_bathroom_interval := 50   -- John uses the bathroom every 50 minutes
def walking_time := 5              -- It takes him an additional 5 minutes to walk to and from the bathroom
def movie_length := 150            -- The movie length in minutes (2.5 hours)
def intermission_length := 15      -- Each intermission length in minutes
def intermission_count := 2        -- The number of intermissions

-- Derived condition
def effective_interval := uses_bathroom_interval + walking_time

-- Total movie time including intermissions
def total_movie_time := movie_length + (intermission_length * intermission_count)

-- Define the theorem to be proved
theorem john_bathroom_uses_during_movie_and_intermissions : 
  ∃ n : ℕ, n = 3 + 2 ∧ total_movie_time = 180 ∧ effective_interval = 55 :=
by
  sorry

end john_bathroom_uses_during_movie_and_intermissions_l44_44045


namespace suitable_b_values_l44_44988

theorem suitable_b_values (b : ℤ) :
  (∃ (c d e f : ℤ), 35 * c * d + (c * f + d * e) * b + 35 = 0 ∧
    c * e = 35 ∧ d * f = 35) →
  (∃ (k : ℤ), b = 2 * k) :=
by
  intro h
  sorry

end suitable_b_values_l44_44988


namespace polygon_angle_arithmetic_progression_l44_44799

theorem polygon_angle_arithmetic_progression
  (h1 : ∀ {n : ℕ}, n ≥ 3)   -- The polygon is convex and n-sided
  (h2 : ∀ (angles : Fin n → ℝ), (∀ i j, i < j → angles i + 5 = angles j))   -- The interior angles form an arithmetic progression with a common difference of 5°
  (h3 : ∀ (angles : Fin n → ℝ), (∃ i, angles i = 160))  -- The largest angle is 160°
  : n = 9 := sorry

end polygon_angle_arithmetic_progression_l44_44799


namespace johns_raise_percent_increase_l44_44640

theorem johns_raise_percent_increase (original_earnings new_earnings : ℝ) 
  (h₀ : original_earnings = 60) (h₁ : new_earnings = 110) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 83.33 :=
by
  sorry

end johns_raise_percent_increase_l44_44640


namespace negation_of_forall_implies_exists_l44_44248

theorem negation_of_forall_implies_exists :
  (¬ ∀ x : ℝ, x^2 > 1) = (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_forall_implies_exists_l44_44248


namespace least_product_of_distinct_primes_over_30_l44_44099

def is_prime (n : ℕ) := Nat.Prime n

def primes_greater_than_30 : List ℕ :=
  [31, 37] -- The first two primes greater than 30 for simplicity

theorem least_product_of_distinct_primes_over_30 :
  ∃ p q ∈ primes_greater_than_30, p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_over_30_l44_44099


namespace campers_morning_count_l44_44065

theorem campers_morning_count (afternoon_count : ℕ) (additional_morning : ℕ) (h1 : afternoon_count = 39) (h2 : additional_morning = 5) :
  afternoon_count + additional_morning = 44 :=
by
  sorry

end campers_morning_count_l44_44065


namespace problem_statement_l44_44370

noncomputable theory

open_locale real

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def is_cube (A B C D A₁ B₁ C₁ D₁ : Point3D) : Prop :=
-- Definition of a cube goes here (omitted for brevity)
sorry

def angle_between (u v : Point3D) : real.angle :=
-- Definition to calculate the angle (omitted for brevity)
sorry

def not_angle_60_deg (angle : real.angle) : Prop :=
angle ≠ real.angle.of_real (60 * (real.pi / 180))

theorem problem_statement (A B C D A₁ B₁ C₁ D₁ : Point3D)
  (h_cube : is_cube A B C D A₁ B₁ C₁ D₁) :
  not_angle_60_deg (angle_between 
    ⟨D.x - A.x, D.y - A.y, D.z - A.z⟩ 
    ⟨B.x - C.x, B.y - C.y, B.z - C.z⟩) :=
sorry

end problem_statement_l44_44370


namespace difference_in_dimes_l44_44917

theorem difference_in_dimes (n d q h : ℕ) 
  (hc1 : n + d + q + h = 150)
  (hc2 : 5 * n + 10 * d + 25 * q + 50 * h = 2000) : 
  (max_d : ℕ) (min_d : ℕ) (dmax : max_d = 250) 
  (dmin : min_d = 7) : 
  max_d - min_d = 243 := 
sorry

end difference_in_dimes_l44_44917


namespace unit_digit_8_pow_1533_l44_44268

theorem unit_digit_8_pow_1533 : (8^1533 % 10) = 8 := by
  sorry

end unit_digit_8_pow_1533_l44_44268


namespace tangent_triangle_area_l44_44655

noncomputable def area_of_tangent_triangle : ℝ :=
  let f : ℝ → ℝ := fun x => Real.log x
  let f' : ℝ → ℝ := fun x => 1 / x
  let tangent_line : ℝ → ℝ := fun x => x - 1
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1
  let base := 1
  let height := 1
  (1 / 2) * base * height

theorem tangent_triangle_area :
  area_of_tangent_triangle = 1 / 2 :=
sorry

end tangent_triangle_area_l44_44655


namespace positional_relationship_l44_44010

theorem positional_relationship 
  (m n : ℝ) 
  (h_points_on_ellipse : (m^2 / 4) + (n^2 / 3) = 1)
  (h_relation : n^2 = 3 - (3/4) * m^2) : 
  (∃ x y : ℝ, (x^2 + y^2 = 1/3) ∧ (m * x + n * y + 1 = 0)) ∨ 
  (∀ x y : ℝ, (x^2 + y^2 = 1/3) → (m * x + n * y + 1 ≠ 0)) :=
sorry

end positional_relationship_l44_44010


namespace time_to_cross_trains_l44_44962

/-- Length of the first train in meters -/
def length_train1 : ℕ := 50

/-- Length of the second train in meters -/
def length_train2 : ℕ := 120

/-- Speed of the first train in km/hr -/
def speed_train1_kmh : ℕ := 60

/-- Speed of the second train in km/hr -/
def speed_train2_kmh : ℕ := 40

/-- Relative speed in km/hr as trains are moving in opposite directions -/
def relative_speed_kmh : ℕ := speed_train1_kmh + speed_train2_kmh

/-- Convert speed from km/hr to m/s -/
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

/-- Relative speed in m/s -/
def relative_speed_ms : ℚ := kmh_to_ms relative_speed_kmh

/-- Total distance to be covered in meters -/
def total_distance : ℕ := length_train1 + length_train2

/-- Time taken in seconds to cross each other -/
def time_to_cross : ℚ := total_distance / relative_speed_ms

theorem time_to_cross_trains :
  time_to_cross = 6.12 := 
sorry

end time_to_cross_trains_l44_44962


namespace rationalize_denominator_l44_44734

theorem rationalize_denominator (A B C : ℤ) (h : A + B * Real.sqrt C = -(9) - 4 * Real.sqrt 5) : A * B * C = 180 :=
by
  have hA : A = -9 := by sorry
  have hB : B = -4 := by sorry
  have hC : C = 5 := by sorry
  rw [hA, hB, hC]
  norm_num

end rationalize_denominator_l44_44734


namespace similar_polygons_perimeter_ratio_l44_44578

-- Define the main function to assert the proportional relationship
theorem similar_polygons_perimeter_ratio (x y : ℕ) (h1 : 9 * y^2 = 64 * x^2) : x * 8 = y * 3 :=
by sorry

-- noncomputable if needed (only necessary when computation is involved, otherwise omit)

end similar_polygons_perimeter_ratio_l44_44578


namespace cos_triple_angle_l44_44185

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l44_44185


namespace inverse_proposition_false_l44_44074

-- Definitions for the conditions
def congruent (A B C D E F: ℝ) : Prop := 
  A = D ∧ B = E ∧ C = F

def angles_equal (α β γ δ ε ζ: ℝ) : Prop := 
  α = δ ∧ β = ε ∧ γ = ζ

def original_proposition (A B C D E F α β γ : ℝ) : Prop :=
  congruent A B C D E F → angles_equal α β γ A B C

-- The inverse proposition
def inverse_proposition (α β γ δ ε ζ A B C D E F : ℝ) : Prop :=
  angles_equal α β γ δ ε ζ → congruent A B C D E F

-- The main theorem: the inverse proposition is false
theorem inverse_proposition_false (α β γ δ ε ζ A B C D E F : ℝ) :
  ¬(inverse_proposition α β γ δ ε ζ A B C D E F) := by
  sorry

end inverse_proposition_false_l44_44074


namespace estimate_shaded_area_l44_44766

theorem estimate_shaded_area 
  (side_length : ℝ)
  (points_total : ℕ)
  (points_shaded : ℕ)
  (area_shaded_estimation : ℝ) :
  side_length = 6 →
  points_total = 800 →
  points_shaded = 200 →
  area_shaded_estimation = (36 * (200 / 800)) →
  area_shaded_estimation = 9 :=
by
  intros h_side_length h_points_total h_points_shaded h_area_shaded_estimation
  rw [h_side_length, h_points_total, h_points_shaded] at *
  norm_num at h_area_shaded_estimation
  exact h_area_shaded_estimation

end estimate_shaded_area_l44_44766


namespace expected_teachers_with_masters_degree_l44_44604

theorem expected_teachers_with_masters_degree
  (prob: ℚ) (teachers: ℕ) (h_prob: prob = 1/4) (h_teachers: teachers = 320) :
  prob * teachers = 80 :=
by
  sorry

end expected_teachers_with_masters_degree_l44_44604


namespace value_of_a_l44_44387

-- Conditions
def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

-- Theorem statement asserting the condition and the correct answer
theorem value_of_a (a : ℝ) : (A a ∩ B a).Nonempty → a = -2 :=
by
  sorry

end value_of_a_l44_44387


namespace sum_of_numbers_less_than_2_l44_44255

theorem sum_of_numbers_less_than_2:
  ∀ (a b c : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 2.2 :=
by
  -- We are stating that if a = 0.8, b = 1/2, and c = 0.9, and all are less than 2, then their sum is 2.2
  sorry

end sum_of_numbers_less_than_2_l44_44255


namespace tangent_intersection_point_l44_44790

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l44_44790


namespace matt_paper_piles_l44_44233

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end matt_paper_piles_l44_44233


namespace rectangle_perimeter_of_equal_area_l44_44422

theorem rectangle_perimeter_of_equal_area (a b c : ℕ) (area_triangle width length : ℕ) :
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 ∧ (2 * area_triangle = a * b) ∧
  (width = 6) ∧ (area_triangle = width * length) -> 
  2 * (length + width) = 30 :=
by
  intros h,
  sorry

end rectangle_perimeter_of_equal_area_l44_44422


namespace product_of_real_roots_l44_44537

theorem product_of_real_roots (x : ℝ) (h : x^5 = 100) : x = 10^(2/5) := by
  sorry

end product_of_real_roots_l44_44537


namespace train_length_l44_44770

theorem train_length (L : ℝ) :
  (∀ t₁ t₂ : ℝ, t₁ = t₂ → L = t₁ / 2) →
  (∀ t : ℝ, t = (8 / 3600) * 36 → L * 2 = t) →
  44 - 36 = 8 →
  L = 40 :=
by
  sorry

end train_length_l44_44770


namespace least_product_of_distinct_primes_greater_than_30_l44_44092

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l44_44092


namespace technicians_count_l44_44246

-- Define the conditions
def avg_sal_all (total_workers : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 850

def avg_sal_technicians (teches : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 1000

def avg_sal_rest (others : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 780

-- The main theorem to prove
theorem technicians_count (total_workers : ℕ)
  (teches others : ℕ)
  (total_salary : ℕ) :
  total_workers = 22 →
  total_salary = 850 * 22 →
  avg_sal_all total_workers 850 →
  avg_sal_technicians teches 1000 →
  avg_sal_rest others 780 →
  teches + others = total_workers →
  1000 * teches + 780 * others = total_salary →
  teches = 7 :=
by
  intros
  sorry

end technicians_count_l44_44246


namespace volume_of_extended_parallelepiped_l44_44307

theorem volume_of_extended_parallelepiped :
  let main_box_volume := 3 * 3 * 6
  let external_boxes_volume := 2 * (3 * 3 * 1 + 3 * 6 * 1 + 3 * 6 * 1)
  let spheres_volume := 8 * (1 / 8) * (4 / 3) * Real.pi * (1 ^ 3)
  let cylinders_volume := 12 * (1 / 4) * Real.pi * 1^2 * 3 + 12 * (1 / 4) * Real.pi * 1^2 * 6
  main_box_volume + external_boxes_volume + spheres_volume + cylinders_volume = (432 + 52 * Real.pi) / 3 :=
by
  sorry

end volume_of_extended_parallelepiped_l44_44307


namespace multiply_abs_value_l44_44303

theorem multiply_abs_value : -2 * |(-3 : ℤ)| = -6 := by
  sorry

end multiply_abs_value_l44_44303


namespace simplify_expression_l44_44739

variable (x y : ℕ)

theorem simplify_expression :
  7 * x + 9 * y + 3 - x + 12 * y + 15 = 6 * x + 21 * y + 18 :=
by
  sorry

end simplify_expression_l44_44739


namespace find_lcm_of_two_numbers_l44_44275

theorem find_lcm_of_two_numbers (A B : ℕ) (hcf : ℕ) (prod : ℕ) 
  (h1 : hcf = 22) (h2 : prod = 62216) (h3 : A * B = prod) (h4 : Nat.gcd A B = hcf) :
  Nat.lcm A B = 2828 := 
by
  sorry

end find_lcm_of_two_numbers_l44_44275


namespace intersection_of_complements_l44_44715

theorem intersection_of_complements 
  (U : Set ℕ) (A B : Set ℕ)
  (hU : U = { x | x ≤ 5 }) 
  (hA : A = {1, 2, 3}) 
  (hB : B = {1, 4}) :
  ((U \ A) ∩ (U \ B)) = {0, 5} :=
by sorry

end intersection_of_complements_l44_44715


namespace set_operation_example_l44_44662

def set_operation (A B : Set ℝ) := {x | (x ∈ A ∪ B) ∧ (x ∉ A ∩ B)}

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | 1 < x ∧ x < 3}

theorem set_operation_example : set_operation M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} :=
by {
  sorry
}

end set_operation_example_l44_44662


namespace unique_solution_l44_44301

theorem unique_solution (x y z : ℝ) 
  (h : x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14) : 
  x = -1 ∧ y = -2 ∧ z = -3 :=
by
  -- entering main proof section
  sorry

end unique_solution_l44_44301


namespace initial_markup_percentage_l44_44972

theorem initial_markup_percentage
  (cost_price : ℝ := 100)
  (profit_percentage : ℝ := 14)
  (discount_percentage : ℝ := 5)
  (selling_price : ℝ := cost_price * (1 + profit_percentage / 100))
  (x : ℝ := 20) :
  (cost_price + cost_price * x / 100) * (1 - discount_percentage / 100) = selling_price := by
  sorry

end initial_markup_percentage_l44_44972


namespace quadratic_inequality_empty_solution_range_l44_44877

theorem quadratic_inequality_empty_solution_range (b : ℝ) :
  (∀ x : ℝ, ¬ (x^2 + b * x + 1 ≤ 0)) ↔ -2 < b ∧ b < 2 :=
by
  sorry

end quadratic_inequality_empty_solution_range_l44_44877


namespace find_values_l44_44355

open Real

noncomputable def positive_numbers (x y : ℝ) := x > 0 ∧ y > 0

noncomputable def given_condition (x y : ℝ) := (sqrt (12 * x) * sqrt (20 * x) * sqrt (4 * y) * sqrt (25 * y) = 50)

theorem find_values (x y : ℝ) 
  (h1: positive_numbers x y) 
  (h2: given_condition x y) : 
  x * y = sqrt (25 / 24) := 
sorry

end find_values_l44_44355


namespace split_tips_evenly_l44_44373

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l44_44373


namespace tangent_intersection_point_l44_44791

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l44_44791


namespace composite_solid_volume_l44_44567

theorem composite_solid_volume :
  let V_prism := 2 * 2 * 1
  let V_cylinder := Real.pi * 1^2 * 3
  let V_overlap := Real.pi / 2
  V_prism + V_cylinder - V_overlap = 4 + 5 * Real.pi / 2 :=
by
  sorry

end composite_solid_volume_l44_44567


namespace sqrt_ineq_l44_44241

open Real

theorem sqrt_ineq (α β : ℝ) (hα : 1 ≤ α) (hβ : 1 ≤ β) :
  Int.floor (sqrt α) + Int.floor (sqrt (α + β)) + Int.floor (sqrt β) ≥
    Int.floor (sqrt (2 * α)) + Int.floor (sqrt (2 * β)) := by sorry

end sqrt_ineq_l44_44241


namespace solve_for_y_l44_44400

def diamond (a b : ℕ) : ℕ := 2 * a + b

theorem solve_for_y (y : ℕ) (h : diamond 4 (diamond 3 y) = 17) : y = 3 :=
by sorry

end solve_for_y_l44_44400


namespace show_watching_days_l44_44897

def numberOfEpisodes := 20
def lengthOfEachEpisode := 30
def dailyWatchingTime := 2

theorem show_watching_days:
  (numberOfEpisodes * lengthOfEachEpisode) / 60 / dailyWatchingTime = 5 := 
by
  sorry

end show_watching_days_l44_44897


namespace scientific_notation_of_12000_l44_44621

theorem scientific_notation_of_12000 : 12000 = 1.2 * 10^4 := 
by sorry

end scientific_notation_of_12000_l44_44621


namespace bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l44_44697

def bernardo (x : ℕ) : ℕ := 2 * x
def silvia (x : ℕ) : ℕ := x + 30

theorem bernardo_winning_N_initial (N : ℕ) :
  (∃ k : ℕ, (bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia N) = k
  ∧ 950 ≤ k ∧ k ≤ 999)
  → 34 ≤ N ∧ N ≤ 35 :=
by
  sorry

theorem bernardo_smallest_N (N : ℕ) (h : 34 ≤ N ∧ N ≤ 35) :
  (N = 34) :=
by
  sorry

theorem sum_of_digits_34 :
  (3 + 4 = 7) :=
by
  sorry

end bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l44_44697


namespace students_left_in_final_year_l44_44817

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

end students_left_in_final_year_l44_44817


namespace relationship_between_x1_x2_x3_l44_44692

variable {x1 x2 x3 : ℝ}

theorem relationship_between_x1_x2_x3
  (A_on_curve : (6 : ℝ) = 6 / x1)
  (B_on_curve : (12 : ℝ) = 6 / x2)
  (C_on_curve : (-6 : ℝ) = 6 / x3) :
  x3 < x2 ∧ x2 < x1 := 
sorry

end relationship_between_x1_x2_x3_l44_44692


namespace median_equality_of_sorted_quartet_range_subsets_l44_44011

variable {α : Type*} [LinearOrder α] [Add α] [Div α]
variable (x1 x2 x3 x4 x5 x6 : α)

theorem median_equality_of_sorted_quartet :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x3 + x4) / 2 = (x3 + x4) / 2 :=
sorry

theorem range_subsets :
  x1 ≤ x2 → x2 ≤ x3 → x3 ≤ x4 → x4 ≤ x5 → x5 ≤ x6 →
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_of_sorted_quartet_range_subsets_l44_44011


namespace find_x_y_z_l44_44708

theorem find_x_y_z (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) (h2 : x * y * z = 10)
  (h3 : x ^ Real.log x * y ^ Real.log y * z ^ Real.log z = 10) :
  (x = 1 ∧ y = 1 ∧ z = 10) ∨ (x = 10 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 10 ∧ z = 1) :=
sorry

end find_x_y_z_l44_44708


namespace cos_triplet_angle_l44_44180

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l44_44180


namespace distance_dormitory_to_city_l44_44443

variable (D : ℝ)
variable (c : ℝ := 12)
variable (f := (1/5) * D)
variable (b := (2/3) * D)

theorem distance_dormitory_to_city (h : f + b + c = D) : D = 90 := by
  sorry

end distance_dormitory_to_city_l44_44443


namespace ceil_square_eq_four_l44_44516

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l44_44516


namespace magnitude_of_power_l44_44000

-- Given conditions
def z : ℂ := 3 + 2 * Complex.I
def n : ℕ := 6

-- Mathematical statement to prove
theorem magnitude_of_power :
  Complex.abs (z ^ n) = 2197 :=
by
  sorry

end magnitude_of_power_l44_44000


namespace line_circle_distance_converse_line_circle_distance_l44_44391

structure TangentLineCircle (circle: Circle) (line: Line): Prop :=
(non_intersect: ¬∃ p: Point, Line.contains line p ∧ Circle.contains circle p)
(points_tangent: ∀ (a b: Point), Line.contains line a → Line.contains line b → 
  ∃ c d: Point, TangentPoint c circle a line ∧ TangentPoint d circle b line
  ∧ |a - b| ≤ |a - c| + |b - d|)

theorem line_circle_distance
  {circle: Circle} {line: Line}
  (h: TangentLineCircle circle line):
  ∀ (a b: Point), Line.contains line a  → Line.contains line b →
  |a - b| ≤ |a - c| + |b - d| := sorry

theorem converse_line_circle_distance
  {circle: Circle} {line: Line}
  (h: TangentLineCircle circle line):
  ∀ (a b: Point), Line.contains line a  → Line.contains line b →
  |a - b| > |a - d| - |b - d| → ∃ p: Point, Line.contains line p ∧ Circle.contains circle p := sorry

end line_circle_distance_converse_line_circle_distance_l44_44391


namespace cos_triplet_angle_l44_44178

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l44_44178


namespace michael_total_weight_loss_l44_44719

def weight_loss_march := 3
def weight_loss_april := 4
def weight_loss_may := 3

theorem michael_total_weight_loss : weight_loss_march + weight_loss_april + weight_loss_may = 10 := by
  sorry

end michael_total_weight_loss_l44_44719


namespace problem_part1_problem_part2_l44_44030

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end problem_part1_problem_part2_l44_44030


namespace find_f_neg_one_l44_44844

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f (x^2 + y) = f x + f (y^2)

theorem find_f_neg_one : f (-1) = 0 := sorry

end find_f_neg_one_l44_44844


namespace imaginary_part_z_l44_44019

theorem imaginary_part_z : 
  ∀ (z : ℂ), z = (5 - I) / (1 - I) → z.im = 2 := 
by
  sorry

end imaginary_part_z_l44_44019


namespace grandmother_current_age_l44_44271

theorem grandmother_current_age (yoojung_age_current yoojung_age_future grandmother_age_future : ℕ)
    (h1 : yoojung_age_current = 5)
    (h2 : yoojung_age_future = 10)
    (h3 : grandmother_age_future = 60) :
    grandmother_age_future - (yoojung_age_future - yoojung_age_current) = 55 :=
by 
  sorry

end grandmother_current_age_l44_44271


namespace minimum_value_of_sum_2_l44_44165

noncomputable def minimum_value_of_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) : 
  Prop := 
  x + y = 2

theorem minimum_value_of_sum_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) :
  minimum_value_of_sum x y hx hy inequality := 
sorry

end minimum_value_of_sum_2_l44_44165


namespace find_floor_abs_S_l44_44480

-- Conditions
-- For integers from 1 to 1500, x_1 + 2 = x_2 + 4 = x_3 + 6 = ... = x_1500 + 3000 = ∑(n=1 to 1500) x_n + 3001
def condition (x : ℕ → ℤ) (S : ℤ) : Prop :=
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 1500 →
    x a + 2 * a = S + 3001

-- Problem statement
theorem find_floor_abs_S (x : ℕ → ℤ) (S : ℤ)
  (h : condition x S) :
  (⌊|S|⌋ : ℤ) = 1500 :=
sorry

end find_floor_abs_S_l44_44480


namespace dealer_profit_percentage_l44_44969

-- Define the conditions
def cost_price_kg : ℕ := 1000
def given_weight_kg : ℕ := 575

-- Define the weight saved by the dealer
def weight_saved : ℕ := cost_price_kg - given_weight_kg

-- Define the profit percentage formula
def profit_percentage : ℕ → ℕ → ℚ := λ saved total_weight => (saved : ℚ) / (total_weight : ℚ) * 100

-- The main theorem statement
theorem dealer_profit_percentage : profit_percentage weight_saved cost_price_kg = 42.5 :=
by
  sorry

end dealer_profit_percentage_l44_44969


namespace complement_A_l44_44338

open Set

variable (A : Set ℝ) (x : ℝ)
def A_def : Set ℝ := { x | x ≥ 1 }

theorem complement_A : Aᶜ = { y | y < 1 } :=
by
  sorry

end complement_A_l44_44338


namespace find_k_no_solution_l44_44544

-- Conditions
def vector1 : ℝ × ℝ := (1, 3)
def direction1 : ℝ × ℝ := (5, -8)
def vector2 : ℝ × ℝ := (0, -1)
def direction2 (k : ℝ) : ℝ × ℝ := (-2, k)

-- Statement
theorem find_k_no_solution (k : ℝ) : 
  (∀ t s : ℝ, vector1 + t • direction1 ≠ vector2 + s • direction2 k) ↔ k = 16 / 5 :=
sorry

end find_k_no_solution_l44_44544


namespace alternating_sum_cubes_eval_l44_44824

noncomputable def alternating_sum_cubes : ℕ → ℤ
| 0 => 0
| n + 1 => alternating_sum_cubes n + (-1)^(n / 4) * (n + 1)^3

theorem alternating_sum_cubes_eval :
  alternating_sum_cubes 99 = S :=
by
  sorry

end alternating_sum_cubes_eval_l44_44824


namespace calculation_correct_l44_44654

theorem calculation_correct :
  (-1 : ℝ)^51 + (2 : ℝ)^(4^2 + 5^2 - 7^2) = -(127 / 128) := 
by
  sorry

end calculation_correct_l44_44654


namespace cross_country_hours_l44_44630

-- Definitions based on the conditions from part a)
def total_hours_required : ℕ := 1500
def hours_day_flying : ℕ := 50
def hours_night_flying : ℕ := 9
def goal_months : ℕ := 6
def hours_per_month : ℕ := 220

-- Problem statement: prove she has already completed 1261 hours of cross-country flying
theorem cross_country_hours : 
  (goal_months * hours_per_month) - (hours_day_flying + hours_night_flying) = 1261 := 
by
  -- Proof omitted (using the solution steps)
  sorry

end cross_country_hours_l44_44630


namespace streetlight_comb_l44_44364

/-
  We have 12 streetlights, out of which 3 can be turned off with the following rules:
  1. The first and last lights cannot be turned off.
  2. No two adjacent lights can be turned off.

  We must prove the number of ways to turn off 3 lights is 56.
-/

theorem streetlight_comb: 
  ∃ (n k: ℕ), n = 8 ∧ k = 3 ∧ (n.choose k) = 56 :=
begin
  use [8, 3],
  split, { refl },
  split, { refl },
  apply nat.choose_eq,
  norm_num,
end

end streetlight_comb_l44_44364


namespace initial_amount_is_53_l44_44721

variable (X : ℕ) -- Initial amount of money Olivia had
variable (ATM_collect : ℕ := 91) -- Money collected from ATM
variable (supermarket_spent_diff : ℕ := 39) -- Spent 39 dollars more at the supermarket
variable (money_left : ℕ := 14) -- Money left after supermarket

-- Define the final amount Olivia had
def final_amount (X ATM_collect supermarket_spent_diff : ℕ) : ℕ :=
  X + ATM_collect - (ATM_collect + supermarket_spent_diff)

-- Theorem stating that the initial amount X was 53 dollars
theorem initial_amount_is_53 : final_amount X ATM_collect supermarket_spent_diff = money_left → X = 53 :=
by
  intros h
  sorry

end initial_amount_is_53_l44_44721


namespace athlete_B_more_stable_l44_44207

variable (average_scores_A average_scores_B : ℝ)
variable (s_A_squared s_B_squared : ℝ)

theorem athlete_B_more_stable
  (h_avg : average_scores_A = average_scores_B)
  (h_var_A : s_A_squared = 1.43)
  (h_var_B : s_B_squared = 0.82) :
  s_A_squared > s_B_squared :=
by 
  rw [h_var_A, h_var_B]
  sorry

end athlete_B_more_stable_l44_44207


namespace right_angled_triangle_lines_l44_44361

theorem right_angled_triangle_lines (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 4 = 0 → x - 2 * y + 5 = 0 → m * x - 3 * y + 12 = 0 → 
    (exists x₁ y₁ : ℝ, 2 * x₁ - 1 * y₁ + 4 = 0 ∧ (x₁ - 5) ^ 2 / 4 + y₁ / (4) = (2^(1/2))^2) ∨ 
    (exists x₂ y₂ : ℝ, 1/2 * x₂ * y₂ - y₂ / 3 + 1 / 6 = 0 ∧ (x₂ + 5) ^ 2 / 9 + y₂ / 4 = small)) → 
    (m = -3 / 2 ∨ m = -6) :=
sorry

end right_angled_triangle_lines_l44_44361


namespace dealer_profit_percentage_l44_44122

-- Definitions of conditions
def cost_price (C : ℝ) : ℝ := C
def list_price (C : ℝ) : ℝ := 1.5 * C
def discount_rate : ℝ := 0.1
def discounted_price (C : ℝ) : ℝ := (1 - discount_rate) * list_price C
def price_for_45_articles (C : ℝ) : ℝ := 45 * discounted_price C
def cost_for_40_articles (C : ℝ) : ℝ := 40 * cost_price C

-- Statement of the problem
theorem dealer_profit_percentage (C : ℝ) (h₀ : C > 0) :
  (price_for_45_articles C - cost_for_40_articles C) / cost_for_40_articles C * 100 = 35 :=  
sorry

end dealer_profit_percentage_l44_44122


namespace pq_over_ef_l44_44393

noncomputable def point := (ℝ × ℝ)
noncomputable def rectangle_abcd := (8 : ℝ) × (6 : ℝ) × (8 : ℝ) × (6 : ℝ)

noncomputable def segment_overline_ab := (6 : ℝ)
noncomputable def point_e := (6 : ℝ) × (6 : ℝ)
noncomputable def segment_overline_bc := (4 : ℝ)
noncomputable def point_g := (8 : ℝ) × (2 : ℝ)
noncomputable def segment_overline_cd := (3 : ℝ)
noncomputable def point_f := (3 : ℝ) × (0 : ℝ)

noncomputable def point_p := ((48 / 11) : ℝ) × ((30 / 11) : ℝ)
noncomputable def point_q := ((24 / 5) : ℝ) × ((18 / 5) : ℝ)

noncomputable def length_ef : ℝ :=
  real.sqrt ((6 - 3)^2  + (6 - 0)^2)

noncomputable def length_pq : ℝ := 
  real.sqrt (((48 / 11) - (24 / 5))^2 + ((30 / 11) - (18 / 5))^2)

theorem pq_over_ef : 
  (length_pq / length_ef) = (8 / 55) :=
sorry

end pq_over_ef_l44_44393


namespace jerry_more_votes_l44_44043

-- Definitions based on conditions
def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375
def john_votes : ℕ := total_votes - jerry_votes

-- Theorem to prove the number of more votes Jerry received than John Pavich
theorem jerry_more_votes : jerry_votes - john_votes = 20196 :=
by
  -- Definitions and proof can be filled out here as required.
  sorry

end jerry_more_votes_l44_44043


namespace cos_five_pi_over_four_l44_44150

theorem cos_five_pi_over_four : Real.cos (5 * Real.pi / 4) = -1 / Real.sqrt 2 := 
by
  sorry

end cos_five_pi_over_four_l44_44150


namespace nat_numbers_l44_44484

theorem nat_numbers (n : ℕ) (h1 : n ≥ 2) (h2 : ∃a b : ℕ, a * b = n ∧ ∀ c : ℕ, 1 < c ∧ c ∣ n → a ≤ c ∧ n = a^2 + b^2) : 
  n = 5 ∨ n = 8 ∨ n = 20 :=
by
  sorry

end nat_numbers_l44_44484


namespace number_of_ways_to_choose_roles_l44_44039

-- Define the problem setup
def friends := Fin 6
def cooks (maria : Fin 1) := {f : Fin 6 | f ≠ maria}
def cleaners (cooks : Fin 6 → Prop) := {f : Fin 6 | ¬cooks f}

-- The number of ways to select one additional cook from the remaining friends
def chooseSecondCook : ℕ := Nat.choose 5 1  -- 5 ways

-- The number of ways to select two cleaners from the remaining friends
def chooseCleaners : ℕ := Nat.choose 4 2  -- 6 ways

-- The final number of ways to choose roles
theorem number_of_ways_to_choose_roles (maria : Fin 1) : 
  let total_ways : ℕ := chooseSecondCook * chooseCleaners
  total_ways = 30 := sorry

end number_of_ways_to_choose_roles_l44_44039


namespace marta_total_spent_l44_44600

theorem marta_total_spent :
  let sale_book_cost := 5 * 10
  let online_book_cost := 40
  let bookstore_book_cost := 3 * online_book_cost
  let total_spent := sale_book_cost + online_book_cost + bookstore_book_cost
  total_spent = 210 := sorry

end marta_total_spent_l44_44600


namespace split_tip_evenly_l44_44377

noncomputable def total_cost (julie_order : ℝ) (letitia_order : ℝ) (anton_order : ℝ) : ℝ :=
  julie_order + letitia_order + anton_order

noncomputable def total_tip (meal_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  tip_rate * meal_cost

noncomputable def tip_per_person (total_tip : ℝ) (num_people : ℝ) : ℝ :=
  total_tip / num_people

theorem split_tip_evenly :
  let julie_order := 10 in
  let letitia_order := 20 in
  let anton_order := 30 in
  let tip_rate := 0.20 in
  let num_people := 3 in
  tip_per_person (total_tip (total_cost julie_order letitia_order anton_order) tip_rate) num_people = 4 :=
by
  sorry

end split_tip_evenly_l44_44377


namespace trees_died_due_to_typhoon_l44_44686

-- defining the initial number of trees
def initial_trees : ℕ := 9

-- defining the additional trees grown after the typhoon
def additional_trees : ℕ := 5

-- defining the final number of trees after all events
def final_trees : ℕ := 10

-- we introduce D as the number of trees that died due to the typhoon
def trees_died (D : ℕ) : Prop := initial_trees - D + additional_trees = final_trees

-- the theorem we need to prove is that 4 trees died
theorem trees_died_due_to_typhoon : trees_died 4 :=
by
  sorry

end trees_died_due_to_typhoon_l44_44686


namespace num_ways_distinct_letters_l44_44353

def letters : List String := ["A₁", "A₂", "A₃", "N₁", "N₂", "N₃", "B₁", "B₂"]

theorem num_ways_distinct_letters : (letters.permutations.length = 40320) := by
  sorry

end num_ways_distinct_letters_l44_44353


namespace diff_one_tenth_and_one_tenth_percent_of_6000_l44_44954

def one_tenth_of_6000 := 6000 / 10
def one_tenth_percent_of_6000 := (1 / 1000) * 6000

theorem diff_one_tenth_and_one_tenth_percent_of_6000 : 
  (one_tenth_of_6000 - one_tenth_percent_of_6000) = 594 :=
by
  sorry

end diff_one_tenth_and_one_tenth_percent_of_6000_l44_44954


namespace sum_of_coordinates_of_B_is_zero_l44_44040

structure Point where
  x : Int
  y : Int

def translation_to_right (P : Point) (n : Int) : Point :=
  { x := P.x + n, y := P.y }

def translation_down (P : Point) (n : Int) : Point :=
  { x := P.x, y := P.y - n }

def A : Point := { x := -1, y := 2 }

def B : Point := translation_down (translation_to_right A 1) 2

theorem sum_of_coordinates_of_B_is_zero :
  B.x + B.y = 0 := by
  sorry

end sum_of_coordinates_of_B_is_zero_l44_44040


namespace unique_seq_l44_44155

def seq (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j

theorem unique_seq (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n) : 
  seq a ↔ (∀ n, a n = n) := 
by
  intros
  sorry

end unique_seq_l44_44155


namespace factorize_expr_l44_44999

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l44_44999


namespace triangle_piece_probability_l44_44646

theorem triangle_piece_probability 
  (circle : Type) 
  (chords: list (circle × circle)) 
  (h_uniform: ∀ p ∈ chords, p.1 ≠ p.2) 
  (h_three_chords: chords.length = 3)
  : ∃ m n : ℕ, 
    m ≥ 1 ∧ 
    n > 1 ∧ 
    Nat.gcd m n = 1 ∧ 
    (1 / 15 : ℚ) = m / n ∧ 
    (100 * m + n = 115) :=
sorry

end triangle_piece_probability_l44_44646


namespace similar_triangles_l44_44291

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end similar_triangles_l44_44291


namespace total_amount_l44_44806

noncomputable def x_share : ℝ := 60
noncomputable def y_share : ℝ := 27
noncomputable def z_share : ℝ := 0.30 * x_share

theorem total_amount (hx : y_share = 0.45 * x_share) : x_share + y_share + z_share = 105 :=
by
  have hx_val : x_share = 27 / 0.45 := by
  { -- Proof that x_share is indeed 60 as per the given problem
    sorry }
  sorry

end total_amount_l44_44806


namespace find_n_l44_44008

theorem find_n (n : ℕ) (hnpos : 0 < n)
  (hsquare : ∃ k : ℕ, k^2 = n^4 + 2*n^3 + 5*n^2 + 12*n + 5) :
  n = 1 ∨ n = 2 := 
sorry

end find_n_l44_44008


namespace distance_between_trees_l44_44651

def yard_length : ℝ := 1530
def number_of_trees : ℝ := 37
def number_of_gaps := number_of_trees - 1

theorem distance_between_trees :
  number_of_gaps ≠ 0 →
  (yard_length / number_of_gaps) = 42.5 :=
by
  sorry

end distance_between_trees_l44_44651


namespace probability_line_intersects_circle_l44_44084

theorem probability_line_intersects_circle :
  (let favorable_pairs : ℕ := 15 in
   let total_pairs : ℕ := 36 in
   let probability := (favorable_pairs : ℝ) / (total_pairs : ℝ) in
   probability = (5 / 12)) :=
by
  -- Probability computation logic
  let favorable_pairs := 15
  let total_pairs := 36
  let probability := (favorable_pairs : ℝ) / (total_pairs : ℝ)
  exact Eq.refl (5 / 12) sorry

end probability_line_intersects_circle_l44_44084


namespace sum_of_rationals_l44_44385

-- Let a1, a2, a3, a4 be 4 rational numbers such that the set of products of distinct pairs is given.
def valid_products (a1 a2 a3 a4 : ℚ) : Prop :=
  {a1 * a2, a1 * a3, a1 * a4, a2 * a3, a2 * a4, a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}

-- Define the theorem which asserts the sum of these rational numbers is either 9/4 or -9/4.
theorem sum_of_rationals (a1 a2 a3 a4 : ℚ) (h : valid_products a1 a2 a3 a4) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_l44_44385


namespace caps_percentage_l44_44980

open Real

-- Define the conditions as given in part (a)
def total_caps : ℝ := 575
def red_caps : ℝ := 150
def green_caps : ℝ := 120
def blue_caps : ℝ := 175
def yellow_caps : ℝ := total_caps - (red_caps + green_caps + blue_caps)

-- Define the problem asking for the percentages of each color and proving the answer
theorem caps_percentage :
  (red_caps / total_caps) * 100 = 26.09 ∧
  (green_caps / total_caps) * 100 = 20.87 ∧
  (blue_caps / total_caps) * 100 = 30.43 ∧
  (yellow_caps / total_caps) * 100 = 22.61 :=
by
  -- proof steps would go here
  sorry

end caps_percentage_l44_44980


namespace cattle_area_correct_l44_44760

-- Definitions based on the problem conditions
def length_km := 3.6
def width_km := 2.5 * length_km
def total_area_km2 := length_km * width_km
def cattle_area_km2 := total_area_km2 / 2

-- Theorem statement
theorem cattle_area_correct : cattle_area_km2 = 16.2 := by
  sorry

end cattle_area_correct_l44_44760


namespace average_paychecks_l44_44594

def first_paychecks : Nat := 6
def remaining_paychecks : Nat := 20
def total_paychecks : Nat := 26
def amount_first : Nat := 750
def amount_remaining : Nat := 770

theorem average_paychecks : 
  (first_paychecks * amount_first + remaining_paychecks * amount_remaining) / total_paychecks = 765 :=
by
  sorry

end average_paychecks_l44_44594


namespace arithmetic_sequence_sum_l44_44678

theorem arithmetic_sequence_sum {S : ℕ → ℤ} (m : ℕ) (hm : 0 < m)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l44_44678


namespace ratio_of_areas_l44_44263

theorem ratio_of_areas (r s_3 s_2 : ℝ) (h1 : s_3^2 = r^2) (h2 : s_2^2 = 2 * r^2) :
  (s_3^2 / s_2^2) = 1 / 2 := by
  sorry

end ratio_of_areas_l44_44263


namespace simplify_fraction_l44_44396

theorem simplify_fraction :
  (5 : ℚ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
sorry

end simplify_fraction_l44_44396


namespace maintain_order_time_l44_44257

theorem maintain_order_time :
  ∀ (x : ℕ), 
  (let ppl_per_min_norm := 9
   let ppl_per_min_cong := 3
   let total_people := 36 
   let teacher_time_saved := 6

   let time_without_order := total_people / ppl_per_min_cong
   let time_with_order := time_without_order - teacher_time_saved

   let ppl_passed_while_order := ppl_per_min_cong * x
   let ppl_passed_norm_order := ppl_per_min_norm * (time_with_order - x)

   ppl_passed_while_order + ppl_passed_norm_order = total_people) → 
  x = 3 :=
sorry

end maintain_order_time_l44_44257


namespace coin_landing_heads_prob_l44_44967

theorem coin_landing_heads_prob (p : ℝ) (h : p^2 * (1 - p)^3 = 0.03125) : p = 0.5 :=
by
sorry

end coin_landing_heads_prob_l44_44967


namespace cone_volume_not_product_base_height_l44_44080

noncomputable def cone_volume (S h : ℝ) := (1/3) * S * h

theorem cone_volume_not_product_base_height (S h : ℝ) :
  cone_volume S h ≠ S * h :=
by sorry

end cone_volume_not_product_base_height_l44_44080


namespace find_positive_real_solutions_l44_44323

variable {x_1 x_2 x_3 x_4 x_5 : ℝ}

theorem find_positive_real_solutions
  (h1 : (x_1^2 - x_3 * x_5) * (x_2^2 - x_3 * x_5) ≤ 0)
  (h2 : (x_2^2 - x_4 * x_1) * (x_3^2 - x_4 * x_1) ≤ 0)
  (h3 : (x_3^2 - x_5 * x_2) * (x_4^2 - x_5 * x_2) ≤ 0)
  (h4 : (x_4^2 - x_1 * x_3) * (x_5^2 - x_1 * x_3) ≤ 0)
  (h5 : (x_5^2 - x_2 * x_4) * (x_1^2 - x_2 * x_4) ≤ 0)
  (hx1 : 0 < x_1)
  (hx2 : 0 < x_2)
  (hx3 : 0 < x_3)
  (hx4 : 0 < x_4)
  (hx5 : 0 < x_5) :
  x_1 = x_2 ∧ x_2 = x_3 ∧ x_3 = x_4 ∧ x_4 = x_5 :=
by
  sorry

end find_positive_real_solutions_l44_44323


namespace min_value_expression_l44_44054

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (c : ℝ), c = 216 ∧
    ∀ (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
      ( (a^2 + 3*a + 2) * (b^2 + 3*b + 2) * (c^2 + 3*c + 2) / (a * b * c) ) ≥ 216 := 
sorry

end min_value_expression_l44_44054


namespace quadratic_distinct_real_roots_l44_44848

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (m ≠ 0 ∧ m < 1 / 5) ↔ ∃ (x y : ℝ), x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0 :=
sorry

end quadratic_distinct_real_roots_l44_44848


namespace simplify_radicals_l44_44611

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l44_44611


namespace geometric_series_properties_l44_44663

noncomputable def first_term := (7 : ℚ) / 8
noncomputable def common_ratio := (-1 : ℚ) / 2

theorem geometric_series_properties : 
  common_ratio = -1 / 2 ∧ 
  (first_term * (1 - common_ratio^4) / (1 - common_ratio)) = 35 / 64 := 
by 
  sorry

end geometric_series_properties_l44_44663


namespace cindy_correct_answer_l44_44819

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end cindy_correct_answer_l44_44819


namespace common_sales_days_in_july_l44_44965

/-- Prove that the number of common sales days in July for both stores is 1. --/
theorem common_sales_days_in_july :
  (finset.range (31)).filter (λ d, (d + 1) % 7 = 0 ∧ (d - 3) % 5 = 0).card = 1 :=
begin
  -- This is where the proof would be constructed.
  sorry,
end

end common_sales_days_in_july_l44_44965


namespace least_product_of_primes_over_30_l44_44087

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l44_44087


namespace ways_to_append_digit_divisible_by_3_l44_44082

-- Define a function that takes a digit and checks if it can make the number divisible by 3
def is_divisible_by_3 (n : ℕ) (d : ℕ) : Bool :=
  (n * 10 + d) % 3 == 0

-- Theorem stating that there are 4 ways to append a digit to make the number divisible by 3
theorem ways_to_append_digit_divisible_by_3 
  (n : ℕ) 
  (divisible_by_9_conditions : (n * 10 + 0) % 9 = 0 ∧ (n * 10 + 9) % 9 = 0) : 
  ∃ (ds : Finset ℕ), ds.card = 4 ∧ ∀ d ∈ ds, is_divisible_by_3 n d :=
  sorry

end ways_to_append_digit_divisible_by_3_l44_44082


namespace tic_tac_toe_tie_probability_l44_44882

theorem tic_tac_toe_tie_probability (john_wins martha_wins : ℚ) 
  (hj : john_wins = 4 / 9) 
  (hm : martha_wins = 5 / 12) : 
  1 - (john_wins + martha_wins) = 5 / 36 := 
by {
  /- insert proof here -/
  sorry
}

end tic_tac_toe_tie_probability_l44_44882


namespace find_k_l44_44841

-- Define the series summation function
def series (k : ℝ) : ℝ := 4 + (∑ n, (4 + n * k) / 5^n)

-- State the theorem with the given condition and required proof
theorem find_k (h : series k = 10) : k = 16 := sorry

end find_k_l44_44841


namespace find_x_for_opposite_expressions_l44_44953

theorem find_x_for_opposite_expressions :
  ∃ x : ℝ, (x + 1) + (3 * x - 5) = 0 ↔ x = 1 :=
by
  sorry

end find_x_for_opposite_expressions_l44_44953


namespace decreasing_number_4312_max_decreasing_number_divisible_by_9_l44_44031

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end decreasing_number_4312_max_decreasing_number_divisible_by_9_l44_44031


namespace remainder_when_sum_divided_by_40_l44_44718

theorem remainder_when_sum_divided_by_40 (x y : ℤ) 
  (h1 : x % 80 = 75) 
  (h2 : y % 120 = 115) : 
  (x + y) % 40 = 30 := 
  sorry

end remainder_when_sum_divided_by_40_l44_44718


namespace find_c_l44_44009

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_c (c : ℝ) (h1 : f 1 = 1) (h2 : ∀ x y : ℝ, f (x + y) = f x + f y + 8 * x * y - c) (h3 : f 7 = 163) :
  c = 2 / 3 :=
sorry

end find_c_l44_44009


namespace trigonometric_identity_l44_44672

theorem trigonometric_identity 
  (θ : ℝ) 
  (h : Real.tan θ = 2) : 
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l44_44672


namespace problem_solution_l44_44714

noncomputable def arithmetic_sequences
    (a : ℕ → ℚ) (b : ℕ → ℚ)
    (Sn : ℕ → ℚ) (Tn : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, Sn n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) ∧
  (∀ n : ℕ, Tn n = n / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) ∧
  (∀ n : ℕ, Sn n / Tn n = (2 * n - 3) / (4 * n - 3))

theorem problem_solution
    (a : ℕ → ℚ) (b : ℕ → ℚ) (Sn : ℕ → ℚ) (Tn : ℕ → ℚ)
    (h_arith : arithmetic_sequences a b Sn Tn) :
    (a 9 / (b 5 + b 7)) + (a 3 / (b 8 + b 4)) = 19 / 41 :=
by
  sorry

end problem_solution_l44_44714


namespace find_b_l44_44688

theorem find_b (b x : ℝ) (h₁ : 5 * x + 3 = b * x - 22) (h₂ : x = 5) : b = 10 := 
by 
  sorry

end find_b_l44_44688


namespace assertion1_false_assertion2_true_assertion3_false_assertion4_false_l44_44137

section

-- Assertion 1: ∀ x ∈ ℝ, x ≥ 1 is false
theorem assertion1_false : ¬(∀ x : ℝ, x ≥ 1) := 
sorry

-- Assertion 2: ∃ x ∈ ℕ, x ∈ ℝ is true
theorem assertion2_true : ∃ x : ℕ, (x : ℝ) = x := 
sorry

-- Assertion 3: ∀ x ∈ ℝ, x > 2 → x ≥ 3 is false
theorem assertion3_false : ¬(∀ x : ℝ, x > 2 → x ≥ 3) := 
sorry

-- Assertion 4: ∃ n ∈ ℤ, ∀ x ∈ ℝ, n ≤ x < n + 1 is false
theorem assertion4_false : ¬(∃ n : ℤ, ∀ x : ℝ, n ≤ x ∧ x < n + 1) := 
sorry

end

end assertion1_false_assertion2_true_assertion3_false_assertion4_false_l44_44137


namespace boa_constrictor_is_70_inches_l44_44978

-- Definitions based on given problem conditions
def garden_snake_length : ℕ := 10
def boa_constrictor_length : ℕ := 7 * garden_snake_length

-- Statement to prove
theorem boa_constrictor_is_70_inches : boa_constrictor_length = 70 :=
by
  sorry

end boa_constrictor_is_70_inches_l44_44978


namespace tiling_impossible_l44_44112

theorem tiling_impossible (T2 T14 : ℕ) :
  let S_before := 2 * T2
  let S_after := 2 * (T2 - 1) + 1 
  S_after ≠ S_before :=
sorry

end tiling_impossible_l44_44112


namespace triangle_angles_median_bisector_altitude_l44_44371

theorem triangle_angles_median_bisector_altitude {α β γ : ℝ} 
  (h : α + β + γ = 180) 
  (median_angle_condition : α / 4 + β / 4 + γ / 4 = 45) -- Derived from 90/4 = 22.5
  (median_from_C : 4 * α = γ) -- Given condition that angle is divided into 4 equal parts
  (median_angle_C : γ = 90) -- Derived that angle @ C must be right angle (90°)
  (sum_angles_C : α + β = 90) : 
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end triangle_angles_median_bisector_altitude_l44_44371


namespace max_ants_collisions_l44_44597

theorem max_ants_collisions (n : ℕ) (hpos : 0 < n) :
  ∃ (ants : Fin n → ℝ) (speeds: Fin n → ℝ) (finite_collisions : Prop)
    (collisions_bound : ℕ),
  (∀ i : Fin n, speeds i ≠ 0) →
  finite_collisions →
  collisions_bound = (n * (n - 1)) / 2 :=
by
  sorry

end max_ants_collisions_l44_44597


namespace ratio_of_population_l44_44142

theorem ratio_of_population (Z : ℕ) :
  let Y := 2 * Z
  let X := 3 * Y
  let W := X + Y
  X / (Z + W) = 2 / 3 :=
by
  sorry

end ratio_of_population_l44_44142


namespace encoded_integer_one_less_l44_44968

theorem encoded_integer_one_less (BDF BEA BFB EAB : ℕ)
  (hBDF : BDF = 1 * 7^2 + 3 * 7 + 6)
  (hBEA : BEA = 1 * 7^2 + 5 * 7 + 0)
  (hBFB : BFB = 1 * 7^2 + 5 * 7 + 1)
  (hEAB : EAB = 5 * 7^2 + 0 * 7 + 1)
  : EAB - 1 = 245 :=
by
  sorry

end encoded_integer_one_less_l44_44968


namespace num_perfect_square_factors_of_360_l44_44023

theorem num_perfect_square_factors_of_360 : 
  (∃ f : ℕ, prime_factors f = {2, 3, 5} ∧ positive_factors f = 360 ∧ perfect_square f 4) :=
  sorry

end num_perfect_square_factors_of_360_l44_44023


namespace sally_mcqueen_cost_l44_44905

theorem sally_mcqueen_cost :
  let lightning_mcqueen_cost := 140000
      mater_cost := 0.1 * lightning_mcqueen_cost
      sally_mcqueen_cost := 3 * mater_cost
  in sally_mcqueen_cost = 42000 :=
by
  let lightning_mcqueen_cost := 140000
  let mater_cost := 0.1 * lightning_mcqueen_cost
  let sally_mcqueen_cost := 3 * mater_cost
  calc 
    sally_mcqueen_cost 
    = 3 * (0.1 * lightning_mcqueen_cost) : by rw [mater_cost]
    = 3 * 14000                       : by rw [lightning_mcqueen_cost * 0.1]
    = 42000                           : by norm_num

end sally_mcqueen_cost_l44_44905


namespace original_water_amount_in_mixture_l44_44126

-- Define heat calculations and conditions
def latentHeatOfFusionIce : ℕ := 80       -- Latent heat of fusion for ice in cal/g
def initialTempWaterAdded : ℕ := 20      -- Initial temperature of added water in °C
def finalTempMixture : ℕ := 5            -- Final temperature of the mixture in °C
def specificHeatWater : ℕ := 1           -- Specific heat of water in cal/g°C

-- Define the known parameters of the problem
def totalMass : ℕ := 250               -- Total mass of the initial mixture in grams
def addedMassWater : ℕ := 1000         -- Mass of added water in grams
def initialTempMixtureIceWater : ℕ := 0  -- Initial temperature of the ice-water mixture in °C

-- Define the equation that needs to be solved
theorem original_water_amount_in_mixture (x : ℝ) :
  (250 - x) * 80 + (250 - x) * 5 + x * 5 = 15000 →
  x = 90.625 :=
by
  intro h
  sorry

end original_water_amount_in_mixture_l44_44126


namespace cost_price_l44_44720

variables (SP DS CP : ℝ)
variables (discount_rate profit_rate : ℝ)
variables (H1 : SP = 24000)
variables (H2 : discount_rate = 0.10)
variables (H3 : profit_rate = 0.08)
variables (H4 : DS = SP - (discount_rate * SP))
variables (H5 : DS = CP + (profit_rate * CP))

theorem cost_price (H1 : SP = 24000) (H2 : discount_rate = 0.10) 
  (H3 : profit_rate = 0.08) (H4 : DS = SP - (discount_rate * SP)) 
  (H5 : DS = CP + (profit_rate * CP)) : 
  CP = 20000 := 
sorry

end cost_price_l44_44720


namespace find_smallest_n_l44_44455

noncomputable def smallest_n (c : ℕ) (n : ℕ) : Prop :=
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ c → n + 2 - 2*k ≥ 0) ∧ c * (n - c + 1) = 2009

theorem find_smallest_n : ∃ n c : ℕ, smallest_n c n ∧ n = 89 :=
sorry

end find_smallest_n_l44_44455


namespace range_of_a_l44_44219

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x > (2 / 3), (deriv (f a)) x > 0) → a > -(1 / 9) :=
by
  sorry

end range_of_a_l44_44219


namespace identify_heaviest_and_lightest_l44_44415

   def coin : Type := ℕ  -- let's represent coins as natural numbers for simplicity.

   def has_different_weights (coins : list coin) : Prop := 
     ∀ (c1 c2 : coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → weight(c1) ≠ weight(c2)

   def weight : coin → ℝ := -- assume a function that gives the weight corresponding to a coin.
     sorry 

   theorem identify_heaviest_and_lightest (coins : list coin) 
     (h₁ : length coins = 10)
     (h₂ : has_different_weights coins) : 
     ∃ (heaviest lightest : coin), 
       (heaviest ∈ coins) ∧ (lightest ∈ coins) ∧
       (∀ c ∈ coins, weight c ≤ weight heaviest) ∧
       (∀ c ∈ coins, weight c ≥ weight lightest) :=
   by 
     sorry
   
end identify_heaviest_and_lightest_l44_44415


namespace trees_probability_l44_44124

theorem trees_probability (num_maple num_oak num_birch total_slots total_trees : ℕ) 
                         (maple_count oak_count birch_count : Prop)
                         (prob_correct : Prop) :
  num_maple = 4 →
  num_oak = 5 →
  num_birch = 6 →
  total_trees = 15 →
  total_slots = 10 →
  maple_count → oak_count → birch_count →
  prob_correct →
  (m + n = 57) :=
by
  intros
  sorry

end trees_probability_l44_44124


namespace problem_statement_l44_44427

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l44_44427


namespace simplified_identity_l44_44951

theorem simplified_identity :
  (12 : ℚ) * ( (1/3 : ℚ) + (1/4) + (1/6) + (1/12) )⁻¹ = 72 / 5 :=
  sorry

end simplified_identity_l44_44951


namespace arcade_spending_fraction_l44_44649

theorem arcade_spending_fraction (allowance remaining_after_arcade remaining_after_toystore: ℝ) (f: ℝ) : 
  allowance = 3.75 ∧
  remaining_after_arcade = (1 - f) * allowance ∧
  remaining_after_toystore = remaining_after_arcade - (1 / 3) * remaining_after_arcade ∧
  remaining_after_toystore = 1 →
  f = 3 / 5 :=
by
  sorry

end arcade_spending_fraction_l44_44649


namespace jerry_original_butterflies_l44_44895

/-- Define the number of butterflies Jerry originally had -/
def original_butterflies (let_go : ℕ) (now_has : ℕ) : ℕ := let_go + now_has

/-- Given conditions -/
def let_go : ℕ := 11
def now_has : ℕ := 82

/-- Theorem to prove the number of butterflies Jerry originally had -/
theorem jerry_original_butterflies : original_butterflies let_go now_has = 93 :=
by
  sorry

end jerry_original_butterflies_l44_44895


namespace determine_colors_l44_44762

-- Define the colors
inductive Color
| white
| red
| blue

open Color

-- Define the friends
inductive Friend
| Tamara 
| Valya
| Lida

open Friend

-- Define a function from Friend to their dress color and shoes color
def Dress : Friend → Color := sorry
def Shoes : Friend → Color := sorry

-- The problem conditions
axiom cond1 : Dress Tamara = Shoes Tamara
axiom cond2 : Shoes Valya = white
axiom cond3 : Dress Lida ≠ red
axiom cond4 : Shoes Lida ≠ red

-- The proof goal
theorem determine_colors :
  Dress Tamara = red ∧ Shoes Tamara = red ∧
  Dress Valya = blue ∧ Shoes Valya = white ∧
  Dress Lida = white ∧ Shoes Lida = blue :=
sorry

end determine_colors_l44_44762


namespace right_triangle_ratio_l44_44555

theorem right_triangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : (x - y)^2 + x^2 = (x + y)^2) : x / y = 4 :=
by
  sorry

end right_triangle_ratio_l44_44555


namespace ram_gohul_work_days_l44_44959

theorem ram_gohul_work_days (ram_days gohul_days : ℕ) (H_ram: ram_days = 10) (H_gohul: gohul_days = 15): 
  (ram_days * gohul_days) / (ram_days + gohul_days) = 6 := 
by
  sorry

end ram_gohul_work_days_l44_44959


namespace initial_tax_rate_l44_44357

variable (R : ℝ)

theorem initial_tax_rate
  (income : ℝ := 48000)
  (new_rate : ℝ := 0.30)
  (savings : ℝ := 7200)
  (tax_savings : income * (R / 100) - income * new_rate = savings) :
  R = 45 := by
  sorry

end initial_tax_rate_l44_44357


namespace ceil_square_neg_seven_over_four_l44_44506

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l44_44506


namespace find_z_l44_44560

open Complex

theorem find_z (z : ℂ) (h : (1 + 2 * z) / (1 - z) = Complex.I) : 
  z = -1 / 5 + 3 / 5 * Complex.I := 
sorry

end find_z_l44_44560


namespace number_of_squares_factors_of_2000_is_6_l44_44411

theorem number_of_squares_factors_of_2000_is_6 : 
  (finset.filter (λ n, n^2 ∣ 2000) (finset.range 2001)).card = 6 := 
by
  sorry

end number_of_squares_factors_of_2000_is_6_l44_44411


namespace stacy_berries_l44_44399

theorem stacy_berries (total_berries : ℕ) 
  (sylar_berries : ℕ) (stacy_to_steve : ℕ → ℕ) (steve_to_sylar : ℕ → ℕ) :
  total_berries = 1100 ∧ stacy_to_steve (steve_to_sylar sylar_berries) = 8 * sylar_berries ∧ stacy_to_steve = (λ n, 4 * n) ∧ steve_to_sylar = (λ n, 2 * n) →
  stacy_to_steve (steve_to_sylar sylar_berries) = 800 :=
by
  sorry

end stacy_berries_l44_44399


namespace number_of_men_l44_44690

variable (W M : ℝ)
variable (N_women N_men : ℕ)

theorem number_of_men (h1 : M = 2 * W)
  (h2 : N_women * W * 30 = 21600) :
  (N_men * M * 20 = 14400) → N_men = N_women / 3 :=
by
  sorry

end number_of_men_l44_44690


namespace nada_house_size_l44_44060

variable (N : ℕ) -- N represents the size of Nada's house

theorem nada_house_size :
  (1000 = 2 * N + 100) → (N = 450) :=
by
  intro h
  sorry

end nada_house_size_l44_44060


namespace arithmetic_example_l44_44474

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l44_44474


namespace president_and_committee_combination_l44_44886

theorem president_and_committee_combination : 
  (∃ (n : ℕ), n = 10 * (Nat.choose 9 3)) := 
by
  use 840
  sorry

end president_and_committee_combination_l44_44886


namespace jenna_more_than_four_times_martha_l44_44812

noncomputable def problems : ℝ := 20
noncomputable def martha_problems : ℝ := 2
noncomputable def angela_problems : ℝ := 9
noncomputable def jenna_problems : ℝ := 6  -- We calculated J = 6 from the conditions
noncomputable def mark_problems : ℝ := jenna_problems / 2

theorem jenna_more_than_four_times_martha :
  (jenna_problems - 4 * martha_problems = 2) :=
by
  sorry

end jenna_more_than_four_times_martha_l44_44812


namespace solve_for_q_l44_44616

theorem solve_for_q :
  ∀ (k l q : ℚ),
    (3 / 4 = k / 108) →
    (3 / 4 = (l + k) / 126) →
    (3 / 4 = (q - l) / 180) →
    q = 148.5 :=
by
  intros k l q hk hl hq
  sorry

end solve_for_q_l44_44616


namespace greatest_possible_large_chips_l44_44937

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (p : ℕ) 
  (h1 : s + l = 72) 
  (h2 : s = l + p) 
  (h_prime : Prime p) : 
  l ≤ 35 :=
sorry

end greatest_possible_large_chips_l44_44937


namespace nadine_total_cleaning_time_l44_44914

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_l44_44914


namespace eq_correct_l44_44942

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l44_44942


namespace thomas_total_drawings_l44_44258

theorem thomas_total_drawings :
  let colored_pencil_drawings := 14
  let blending_marker_drawings := 7
  let charcoal_drawings := 4
  colored_pencil_drawings + blending_marker_drawings + charcoal_drawings = 25 := 
by
  sorry

end thomas_total_drawings_l44_44258


namespace no_single_x_for_doughnut_and_syrup_l44_44581

theorem no_single_x_for_doughnut_and_syrup :
  ¬ ∃ x : ℝ, (x^2 - 9 * x + 13 < 0) ∧ (x^2 + x - 5 < 0) :=
sorry

end no_single_x_for_doughnut_and_syrup_l44_44581


namespace compute_expression_l44_44472

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l44_44472


namespace more_green_than_yellow_l44_44253

-- Define constants
def red_peaches : ℕ := 2
def yellow_peaches : ℕ := 6
def green_peaches : ℕ := 14

-- Prove the statement
theorem more_green_than_yellow : green_peaches - yellow_peaches = 8 :=
by
  sorry

end more_green_than_yellow_l44_44253


namespace intersection_complement_is_singleton_l44_44173

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 2, 5}

theorem intersection_complement_is_singleton : (U \ M) ∩ N = {1} := by
  sorry

end intersection_complement_is_singleton_l44_44173


namespace ceil_square_neg_fraction_l44_44527

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l44_44527


namespace remaining_sugar_l44_44390

-- Conditions as definitions
def total_sugar : ℝ := 9.8
def spilled_sugar : ℝ := 5.2

-- Theorem to prove the remaining sugar
theorem remaining_sugar : total_sugar - spilled_sugar = 4.6 := by
  sorry

end remaining_sugar_l44_44390


namespace total_orchids_l44_44934

-- Conditions
def current_orchids : ℕ := 2
def additional_orchids : ℕ := 4

-- Proof statement
theorem total_orchids : current_orchids + additional_orchids = 6 :=
by
  sorry

end total_orchids_l44_44934


namespace simplify_radicals_l44_44612

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l44_44612


namespace find_five_digit_number_l44_44438

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  digits.reverse.to_nat

theorem find_five_digit_number :
  ∃ n : ℕ, (9999 < n ∧ n < 100000) ∧ (9 * n = reverse_number n) ∧ n = 10989 :=
by
  sorry

end find_five_digit_number_l44_44438


namespace permutation_five_out_of_seven_l44_44254

theorem permutation_five_out_of_seven : ∃ (k : ℕ), k = nat.fact 7 / nat.fact (7 - 5) ∧ k = 2520 :=
by 
  have h := nat.factorial_eq_prod_range_one_succ 7,
  have h_r := nat.factorial_eq_prod_range_one_succ (7 - 5),
  -- The required number of permutations P(7, 5) = 7! / (7-5)!
  sorry

end permutation_five_out_of_seven_l44_44254


namespace optimal_purchasing_plan_l44_44120

def price_carnation := 5
def price_lily := 10
def total_flowers := 300
def max_carnations (x : ℕ) : Prop := x ≤ 2 * (total_flowers - x)

theorem optimal_purchasing_plan :
  ∃ (x y : ℕ), (x + y = total_flowers) ∧ (x = 200) ∧ (y = 100) ∧ (max_carnations x) ∧ 
  ∀ (x' y' : ℕ), (x' + y' = total_flowers) → max_carnations x' →
    (price_carnation * x + price_lily * y ≤ price_carnation * x' + price_lily * y') :=
by
  sorry

end optimal_purchasing_plan_l44_44120


namespace intersection_A_B_l44_44575

def A : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}
def B : Set ℝ := {x | x * (x + 1) ≥ 0}

theorem intersection_A_B :
  (A ∩ B) = {x | (0 ≤ x ∧ x ≤ 1) ∨ x = -1} :=
  sorry

end intersection_A_B_l44_44575


namespace reflection_line_sum_l44_44755

theorem reflection_line_sum (m b : ℝ) :
  (∀ (x y x' y' : ℝ), (x, y) = (2, 5) → (x', y') = (6, 1) →
  y' = m * x' + b ∧ y = m * x + b) → 
  m + b = 0 :=
sorry

end reflection_line_sum_l44_44755


namespace harmonyNumbersWithFirstDigit2_l44_44771

def isHarmonyNumber (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.sum = 6

def startsWithDigit (d n : ℕ) : Prop :=
  n / 1000 = d

theorem harmonyNumbersWithFirstDigit2 :
  ∃ c : ℕ, c = 15 ∧ ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → isHarmonyNumber n → startsWithDigit 2 n → ∃ m : ℕ, m < c ∧ m = n :=
sorry

end harmonyNumbersWithFirstDigit2_l44_44771


namespace train_speed_l44_44111

-- Definitions to capture the conditions
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 300
def time_to_cross_bridge : ℝ := 36

-- The speed of the train calculated according to the condition
def total_distance : ℝ := length_of_train + length_of_bridge

theorem train_speed : total_distance / time_to_cross_bridge = 11.11 :=
by
  sorry

end train_speed_l44_44111


namespace least_product_of_primes_over_30_l44_44086

theorem least_product_of_primes_over_30 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ 30 < p ∧ 30 < q ∧ (∀ p' q', p' ≠ q' → nat.prime p' → nat.prime q' → 30 < p' → 30 < q' → p * q ≤ p' * q') :=
begin
  sorry
end

end least_product_of_primes_over_30_l44_44086


namespace split_terms_addition_l44_44735

theorem split_terms_addition : 
  (-2017 - (2/3)) + (2016 + (3/4)) + (-2015 - (5/6)) + (16 + (1/2)) = -2000 - (1/4) :=
by
  sorry

end split_terms_addition_l44_44735


namespace parabola_directrix_l44_44577

-- Defining the given condition
def given_parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

-- Defining the expected directrix equation for the parabola
def directrix_equation (y : ℝ) : Prop := y = -1 / 8

-- The theorem we aim to prove
theorem parabola_directrix :
  (∀ x y : ℝ, given_parabola_equation x y) → (directrix_equation (-1 / 8)) :=
by
  -- Using 'sorry' here since the proof is not required
  sorry

end parabola_directrix_l44_44577


namespace unique_three_digit_numbers_l44_44632

theorem unique_three_digit_numbers (d1 d2 d3 : ℕ) :
  (d1 = 3 ∧ d2 = 0 ∧ d3 = 8) →
  ∃ nums : Finset ℕ, 
  (∀ n ∈ nums, (∃ h t u : ℕ, n = 100 * h + 10 * t + u ∧ 
                h ≠ 0 ∧ (h = d1 ∨ h = d2 ∨ h = d3) ∧ 
                (t = d1 ∨ t = d2 ∨ t = d3) ∧ (u = d1 ∨ u = d2 ∨ u = d3) ∧ 
                h ≠ t ∧ t ≠ u ∧ u ≠ h)) ∧ nums.card = 4 :=
by
  sorry

end unique_three_digit_numbers_l44_44632


namespace max_single_student_books_l44_44584

-- Definitions and conditions
variable (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ)
variable (total_avg_books_per_student : ℕ)

-- Given data
def given_data : Prop :=
  total_students = 20 ∧ no_books = 2 ∧ one_book = 8 ∧
  two_books = 3 ∧ total_avg_books_per_student = 2

-- Maximum number of books any single student could borrow
theorem max_single_student_books (total_students no_books one_book two_books total_avg_books_per_student : ℕ) 
  (h : given_data total_students no_books one_book two_books total_avg_books_per_student) : 
  ∃ max_books_borrowed, max_books_borrowed = 8 :=
by
  sorry

end max_single_student_books_l44_44584


namespace complex_number_in_second_quadrant_l44_44576

theorem complex_number_in_second_quadrant 
  (a b : ℝ) 
  (h : ¬ (a ≥ 0 ∨ b ≤ 0)) : 
  (a < 0 ∧ b > 0) :=
sorry

end complex_number_in_second_quadrant_l44_44576


namespace parabola_max_value_l44_44157

theorem parabola_max_value 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x = - (x + 1)^2 + 3) : 
  ∃ x, y x = 3 ∧ ∀ x', y x' ≤ 3 :=
by
  sorry

end parabola_max_value_l44_44157


namespace task_completion_time_l44_44215

noncomputable def john_work_rate := (1: ℚ) / 20
noncomputable def jane_work_rate := (1: ℚ) / 12
noncomputable def combined_work_rate := john_work_rate + jane_work_rate
noncomputable def time_jane_disposed := 4

theorem task_completion_time :
  (∃ x : ℚ, (combined_work_rate * x + john_work_rate * time_jane_disposed = 1) ∧ (x + time_jane_disposed = 10)) :=
by
  use 6  
  sorry

end task_completion_time_l44_44215


namespace average_of_three_quantities_l44_44245

theorem average_of_three_quantities (a b c d e : ℝ) 
    (h1 : (a + b + c + d + e) / 5 = 8)
    (h2 : (d + e) / 2 = 14) :
    (a + b + c) / 3 = 4 := 
sorry

end average_of_three_quantities_l44_44245


namespace minimum_excellence_percentage_l44_44888

theorem minimum_excellence_percentage (n : ℕ) (h : n = 100)
    (m c b : ℕ) 
    (h_math : m = 70)
    (h_chinese : c = 75) 
    (h_min_both : b = c - (n - m))
    (h_percent : b = 45) :
    b = 45 :=
    sorry

end minimum_excellence_percentage_l44_44888


namespace find_number_l44_44270

theorem find_number (x : ℕ) :
  ((4 * x) / 8 = 6) ∧ ((4 * x) % 8 = 4) → x = 13 :=
by
  sorry

end find_number_l44_44270


namespace exists_rhombus_with_given_side_and_diag_sum_l44_44308

-- Define the context of the problem
variables (a s : ℝ)

-- Necessary definitions for a rhombus
structure Rhombus (side diag_sum : ℝ) :=
  (side_length : ℝ)
  (diag_sum : ℝ)
  (d1 d2 : ℝ)
  (side_length_eq : side_length = side)
  (diag_sum_eq : d1 + d2 = diag_sum)
  (a_squared : 2 * (side_length)^2 = d1^2 + d2^2)

-- The proof problem
theorem exists_rhombus_with_given_side_and_diag_sum (a s : ℝ) : 
  ∃ (r : Rhombus a (2*s)), r.side_length = a ∧ r.diag_sum = 2 * s :=
by
  sorry

end exists_rhombus_with_given_side_and_diag_sum_l44_44308


namespace expected_variance_l44_44278

section

variable (p : ℝ) (n : ℕ) (X : ℕ → ℝ) (Y : ℕ → ℝ)

# Conditions
def prob_success : ℝ := 0.6
def num_shots : ℕ := 5
def points_per_shot : ℕ := 10
def successful_shots (i : ℕ) : ℝ := binomial n p i

# Given that Y = 10X
def total_points (i : ℕ) : ℝ := points_per_shot * (successful_shots i)

# Lean Statement
theorem expected_variance :
  E (fun i => successful_shots i) = 3 ∧ D (fun i => total_points i) = 120 :=
sorry

end

end expected_variance_l44_44278


namespace radius_of_circle_eqn_zero_l44_44310

def circle_eqn (x y : ℝ) := x^2 + 8*x + y^2 - 4*y + 20 = 0

theorem radius_of_circle_eqn_zero :
  ∀ x y : ℝ, circle_eqn x y → ∃ r : ℝ, r = 0 :=
by
  intros x y h
  -- Sorry to skip the proof as per instructions
  sorry

end radius_of_circle_eqn_zero_l44_44310


namespace factorize_x_cube_minus_4x_l44_44996

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l44_44996


namespace total_marbles_l44_44761

variables (y : ℝ) 

def first_friend_marbles : ℝ := 2 * y + 2
def second_friend_marbles : ℝ := y
def third_friend_marbles : ℝ := 3 * y - 1

theorem total_marbles :
  (first_friend_marbles y) + (second_friend_marbles y) + (third_friend_marbles y) = 6 * y + 1 :=
by
  sorry

end total_marbles_l44_44761


namespace emily_did_not_sell_bars_l44_44825

-- Definitions based on conditions
def cost_per_bar : ℕ := 4
def total_bars : ℕ := 8
def total_earnings : ℕ := 20

-- The statement to be proved
theorem emily_did_not_sell_bars :
  (total_bars - (total_earnings / cost_per_bar)) = 3 :=
by
  sorry

end emily_did_not_sell_bars_l44_44825


namespace minimum_value_x_plus_3y_plus_6z_l44_44050

theorem minimum_value_x_plus_3y_plus_6z 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y * z = 18) : 
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end minimum_value_x_plus_3y_plus_6z_l44_44050


namespace max_gangsters_chicago_max_gangsters_l44_44627

/-- Define the basic conditions from the problem statement --/
structure GangsterProblem :=
  (gangs : ℕ)                                    -- Number of gangs
  (gangster_belongs : ℕ → ℕ → Prop)               -- Gangster i belongs to gang j
  (hostile : ℕ → ℕ → Prop)                        -- Gang i is hostile to gang j
  (no_two_same_gangs : ∀ (g1 g2 : ℕ), g1 ≠ g2 → 
    ∃ (j : ℕ), gangster_belongs g1 j ≠ gangster_belongs g2 j) -- No two gangsters belong to the same set of gangs

/-- Our goal is to compute the maximum number of distinct gangsters under given conditions --/
theorem max_gangsters (G : GangsterProblem) : ℕ :=
  sorry

def ChicagoGangsters : GangsterProblem :=
  { gangs := 36,
    gangster_belongs := λ i j, sorry,
    hostile := λ i j, sorry,
    no_two_same_gangs := λ _ _ _, sorry
  }

theorem chicago_max_gangsters : max_gangsters ChicagoGangsters = 531441 :=
  sorry

end max_gangsters_chicago_max_gangsters_l44_44627


namespace algebraic_expression_value_l44_44026

theorem algebraic_expression_value (x : ℝ) :
  let a := 2003 * x + 2001
  let b := 2003 * x + 2002
  let c := 2003 * x + 2003
  a^2 + b^2 + c^2 - a * b - a * c - b * c = 3 :=
by
  sorry

end algebraic_expression_value_l44_44026


namespace cookies_and_milk_l44_44765

theorem cookies_and_milk :
  (∀ (c q : ℕ), (c = 18 → q = 3 → ∀ (p : ℕ), p = q * 2 → ∀ (c' : ℕ), c' = 9 → (p' : ℕ), p' = (c' * p) / c = 3)) := 
    by
  intros c q hc hq p hp c' hc' p'
  have h1 : p = 6, by
    rw [hq, hp]
    norm_num
  have h2 : 18 * p' = 9 * p, by
    rw [hc, hc']
    norm_num
  have h3 : p' = 3, by
    rw [h1] at h2
    norm_num at h2
    exact eq_div_of_mul_eq h2.symm
  exact h3

end cookies_and_milk_l44_44765


namespace matt_paper_piles_l44_44232

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end matt_paper_piles_l44_44232


namespace average_weight_decrease_l44_44923

theorem average_weight_decrease 
  (num_persons : ℕ)
  (avg_weight_initial : ℕ)
  (new_person_weight : ℕ)
  (new_avg_weight : ℚ)
  (weight_decrease : ℚ)
  (h1 : num_persons = 20)
  (h2 : avg_weight_initial = 60)
  (h3 : new_person_weight = 45)
  (h4 : new_avg_weight = (1200 + 45) / 21) : 
  weight_decrease = avg_weight_initial - new_avg_weight :=
by
  sorry

end average_weight_decrease_l44_44923


namespace sin_2B_minus_5pi_over_6_area_of_triangle_l44_44696

-- Problem (I)
theorem sin_2B_minus_5pi_over_6 {A B C : ℝ} (a b c : ℝ)
  (h: 3 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1) :
  Real.sin (2 * B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 :=
sorry

-- Problem (II)
theorem area_of_triangle {A B C : ℝ} (a b c : ℝ)
  (h1: a + c = 3 * Real.sqrt 3 / 2) (h2: b = Real.sqrt 3) :
  Real.sqrt (a * c) * Real.sin B / 2 = 15 * Real.sqrt 2 / 32 :=
sorry

end sin_2B_minus_5pi_over_6_area_of_triangle_l44_44696


namespace expected_value_traffic_jam_commute_l44_44910

open ProbabilityTheory

noncomputable def traffic_jam_expected_value (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem expected_value_traffic_jam_commute :
  traffic_jam_expected_value 6 (1/6 : ℚ) = 1 :=
by
  -- The expected value of a binomial distribution B(6, 1/6)
  -- is calculated by multiplying the number of trials n by the probability of success p.
  -- Here, n = 6 and p = 1/6, so the expected value is E(ξ) = 6 * (1/6) = 1.
  sorry

end expected_value_traffic_jam_commute_l44_44910


namespace eq_correct_l44_44943

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l44_44943


namespace ceil_of_neg_frac_squared_l44_44490

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l44_44490


namespace num_perfect_square_factors_of_360_l44_44024

theorem num_perfect_square_factors_of_360 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d : ℕ, d ∣ 360 → (∀ p e, p^e ∣ d → (p = 2 ∨ p = 3 ∨ p = 5) ∧ e % 2 = 0) :=
by
  sorry

end num_perfect_square_factors_of_360_l44_44024


namespace min_packs_to_buy_120_cans_l44_44398

/-- Prove that the minimum number of packs needed to buy exactly 120 cans of soda,
with packs available in sizes of 8, 15, and 30 cans, is 4. -/
theorem min_packs_to_buy_120_cans : 
  ∃ n, n = 4 ∧ ∀ x y z: ℕ, 8 * x + 15 * y + 30 * z = 120 → x + y + z ≥ n :=
sorry

end min_packs_to_buy_120_cans_l44_44398


namespace cookies_milk_conversion_l44_44763

theorem cookies_milk_conversion :
  (18 : ℕ) / (3 * 2 : ℕ) / (18 : ℕ) * (9 : ℕ) = (3 : ℕ) :=
by
  sorry

end cookies_milk_conversion_l44_44763


namespace percent_of_amount_l44_44436

theorem percent_of_amount (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_amount_l44_44436


namespace sum_first_six_terms_geometric_seq_l44_44835

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end sum_first_six_terms_geometric_seq_l44_44835


namespace factorization_correct_l44_44320

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l44_44320


namespace complete_residue_system_l44_44163

theorem complete_residue_system {m n : ℕ} {a : ℕ → ℕ} {b : ℕ → ℕ}
  (h₁ : ∀ i j, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ n → (a i) * (b j) % (m * n) ≠ (a i) * (b j)) :
  (∀ i₁ i₂, 1 ≤ i₁ → i₁ ≤ m → 1 ≤ i₂ → i₂ ≤ m → i₁ ≠ i₂ → (a i₁ % m ≠ a i₂ % m)) ∧ 
  (∀ j₁ j₂, 1 ≤ j₁ → j₁ ≤ n → 1 ≤ j₂ → j₂ ≤ n → j₁ ≠ j₂ → (b j₁ % n ≠ b j₂ % n)) := sorry

end complete_residue_system_l44_44163


namespace sum_first_10_log_a_l44_44562

-- Given sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := 2^n - 1

-- Function to get general term log_2 a_n
def log_a (n : ℕ) : ℕ := n - 1

-- The statement to prove
theorem sum_first_10_log_a : (List.range 10).sum = 45 := by 
  sorry

end sum_first_10_log_a_l44_44562


namespace time_to_cross_bridge_l44_44457

theorem time_to_cross_bridge 
  (speed_kmhr : ℕ) 
  (bridge_length_m : ℕ) 
  (h1 : speed_kmhr = 10)
  (h2 : bridge_length_m = 2500) :
  (bridge_length_m / (speed_kmhr * 1000 / 60) = 15) :=
by
  sorry

end time_to_cross_bridge_l44_44457


namespace ceil_square_of_neg_seven_fourths_l44_44536

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l44_44536


namespace AD_parallel_BC_l44_44854

-- Definitions for given points and segments
variables {A B C D : Point}
variables {AB BC CD : Segment}
variables {angle_ABC angle_BCD : ℝ}

-- Assumptions based on conditions
def non_closed_broken_line (A B C D : Point) : Prop :=
  True   -- Just a placeholder to mark the points as forming a broken line

def same_side (A D : Point) (BC : Line) : Prop :=
  True   -- Placeholder to indicate A and D are on the same side of BC

axiom AB_eq_CD : ↥AB = ↥CD
axiom angle_ABC_eq_angle_BCD : angle_ABC = angle_BCD
axiom A_D_same_side_BC : same_side A D (line B C)

-- Statement to prove that AD is parallel to BC given the conditions
theorem AD_parallel_BC
  (A B C D : Point)
  (AB_eq_CD : ↥AB = ↥CD)
  (angle_ABC_eq_angle_BCD : angle_ABC = angle_BCD)
  (A_D_same_side_BC : same_side A D (line B C))
  : parallel (line A D) (line B C) :=
sorry

end AD_parallel_BC_l44_44854


namespace ceil_square_of_neg_seven_fourths_l44_44533

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l44_44533


namespace product_equals_16896_l44_44549

theorem product_equals_16896 (A B C D : ℕ) (h1 : A + B + C + D = 70)
  (h2 : A = 3 * C + 1) (h3 : B = 3 * C + 5) (h4 : C = C) (h5 : D = 3 * C^2) :
  A * B * C * D = 16896 :=
by
  sorry

end product_equals_16896_l44_44549


namespace ceil_square_eq_four_l44_44513

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l44_44513


namespace pipe_A_fill_time_l44_44808

theorem pipe_A_fill_time (t : ℕ) : 
  (∀ x : ℕ, x = 40 → (1 * x) = 40) ∧
  (∀ y : ℕ, y = 30 → (15/40) + ((1/t) + (1/40)) * 15 = 1) ∧ t = 60 :=
sorry

end pipe_A_fill_time_l44_44808


namespace cupboard_slots_l44_44297

theorem cupboard_slots (shelves_from_top shelves_from_bottom slots_from_left slots_from_right : ℕ)
  (h_top : shelves_from_top = 1)
  (h_bottom : shelves_from_bottom = 3)
  (h_left : slots_from_left = 0)
  (h_right : slots_from_right = 6) :
  (shelves_from_top + 1 + shelves_from_bottom) * (slots_from_left + 1 + slots_from_right) = 35 := by
  sorry

end cupboard_slots_l44_44297


namespace cos_triple_angle_l44_44188

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l44_44188


namespace net_change_in_price_l44_44960

-- Define the initial price of the TV
def initial_price (P : ℝ) := P

-- Define the price after a 20% decrease
def decreased_price (P : ℝ) := 0.80 * P

-- Define the final price after a 50% increase on the decreased price
def final_price (P : ℝ) := 1.20 * P

-- Prove that the net change is 20% of the original price
theorem net_change_in_price (P : ℝ) : final_price P - initial_price P = 0.20 * P := by
  sorry

end net_change_in_price_l44_44960


namespace tagged_fish_in_second_catch_l44_44204

theorem tagged_fish_in_second_catch :
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  (total_tagged / N) * total_caught = 5 :=
by
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  show (total_tagged / N) * total_caught = 5
  sorry

end tagged_fish_in_second_catch_l44_44204


namespace bruce_age_multiple_of_son_l44_44985

structure Person :=
  (age : ℕ)

def bruce := Person.mk 36
def son := Person.mk 8
def multiple := 3

theorem bruce_age_multiple_of_son :
  ∃ (x : ℕ), bruce.age + x = multiple * (son.age + x) ∧ x = 6 :=
by
  use 6
  sorry

end bruce_age_multiple_of_son_l44_44985


namespace num_triangles_with_positive_area_l44_44873

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l44_44873


namespace positive_area_triangles_correct_l44_44862

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l44_44862


namespace dark_squares_more_than_light_l44_44784

/--
A 9 × 9 board is composed of alternating dark and light squares, with the upper-left square being dark.
Prove that there is exactly 1 more dark square than light square.
-/
theorem dark_squares_more_than_light :
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  dark_squares - light_squares = 1 :=
by
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  show dark_squares - light_squares = 1
  sorry

end dark_squares_more_than_light_l44_44784


namespace school_robes_l44_44294

theorem school_robes (total_singers robes_needed : ℕ) (robe_cost total_spent existing_robes : ℕ) 
  (h1 : total_singers = 30)
  (h2 : robe_cost = 2)
  (h3 : total_spent = 36)
  (h4 : total_singers - total_spent / robe_cost = existing_robes) :
  existing_robes = 12 :=
by sorry

end school_robes_l44_44294


namespace money_spent_twice_as_much_l44_44727

variable (p s : ℕ) 

theorem money_spent_twice_as_much 
    (h1 : 2 * s = 3 * 2 * p)
    (h2 : s + p < 2 * s) 
    (h3 : s + p >= 1 * p + 1 * s) :
    (s + p) = 2 * (2 * p) → 2 :=
by
    sorry

end money_spent_twice_as_much_l44_44727


namespace caleb_trip_duration_l44_44468

-- Define the times when the clock hands meet
def startTime := 7 * 60 + 38 -- 7:38 a.m. in minutes from midnight
def endTime := 13 * 60 + 5 -- 1:05 p.m. in minutes from midnight

def duration := endTime - startTime

theorem caleb_trip_duration :
  duration = 5 * 60 + 27 := by
sorry

end caleb_trip_duration_l44_44468


namespace factorize_x_squared_plus_2x_l44_44314

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l44_44314


namespace strictly_increasing_interval_l44_44564

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

theorem strictly_increasing_interval :
  (∀ k : ℤ, ∀ x : ℝ, 
    (2 * k * Real.pi - 5 * Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6) 
    → (f x) < (f (x + 1))) :=
by 
  sorry

end strictly_increasing_interval_l44_44564


namespace set_contains_difference_of_elements_l44_44196

variable {A : Set Int}

axiom cond1 (a : Int) (ha : a ∈ A) : 2 * a ∈ A
axiom cond2 (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a + b ∈ A

theorem set_contains_difference_of_elements 
  (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a - b ∈ A := by
  sorry

end set_contains_difference_of_elements_l44_44196


namespace SallyMcQueenCostCorrect_l44_44904

def LightningMcQueenCost : ℕ := 140000
def MaterCost : ℕ := (140000 * 10) / 100
def SallyMcQueenCost : ℕ := 3 * MaterCost

theorem SallyMcQueenCostCorrect : SallyMcQueenCost = 42000 := by
  sorry

end SallyMcQueenCostCorrect_l44_44904


namespace chairs_made_after_tables_l44_44550

def pieces_of_wood : Nat := 672
def wood_per_table : Nat := 12
def wood_per_chair : Nat := 8
def number_of_tables : Nat := 24

theorem chairs_made_after_tables (pieces_of_wood wood_per_table wood_per_chair number_of_tables : Nat) :
  wood_per_table * number_of_tables <= pieces_of_wood ->
  (pieces_of_wood - wood_per_table * number_of_tables) / wood_per_chair = 48 :=
by
  sorry

end chairs_made_after_tables_l44_44550


namespace probability_second_marble_purple_correct_l44_44815

/-!
  Bag A has 5 red marbles and 5 green marbles.
  Bag B has 8 purple marbles and 2 orange marbles.
  Bag C has 3 purple marbles and 7 orange marbles.
  Bag D has 4 purple marbles and 6 orange marbles.
  A marble is drawn at random from Bag A.
  If it is red, a marble is drawn at random from Bag B;
  if it is green, a marble is drawn at random from Bag C;
  but if it is neither (an impossible scenario in this setup), a marble would be drawn from Bag D.
  Prove that the probability of the second marble drawn being purple is 11/20.
-/

noncomputable def probability_second_marble_purple : ℚ :=
  let p_red_A := 5 / 10
  let p_green_A := 5 / 10
  let p_purple_B := 8 / 10
  let p_purple_C := 3 / 10
  (p_red_A * p_purple_B) + (p_green_A * p_purple_C)

theorem probability_second_marble_purple_correct :
  probability_second_marble_purple = 11 / 20 := sorry

end probability_second_marble_purple_correct_l44_44815


namespace simplify_sqrt_sum_l44_44613

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l44_44613


namespace plane_equation_correct_l44_44957

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

def planeEquation (n : Point3D) (A : Point3D) (P : Point3D) : ℝ :=
  n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

theorem plane_equation_correct :
  let A := ⟨3, -3, -6⟩
  let B := ⟨1, 9, -5⟩
  let C := ⟨6, 6, -4⟩
  let n := vectorBC B C
  ∀ P, planeEquation n A P = 0 ↔ 5 * (P.x - A.x) - 3 * (P.y - A.y) + 1 * (P.z - A.z) - 18 = 0 :=
by
  sorry

end plane_equation_correct_l44_44957


namespace no_positive_integer_solutions_l44_44239

theorem no_positive_integer_solutions (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2 * n) * y * (y + 1) :=
by
  sorry

end no_positive_integer_solutions_l44_44239


namespace problem_l44_44546

theorem problem 
  {a1 a2 : ℝ}
  (h1 : 0 ≤ a1)
  (h2 : 0 ≤ a2)
  (h3 : a1 + a2 = 1) :
  ∃ (b1 b2 : ℝ), 0 ≤ b1 ∧ 0 ≤ b2 ∧ b1 + b2 = 1 ∧ ((5/4 - a1) * b1 + 3 * (5/4 - a2) * b2 > 1) :=
by
  sorry

end problem_l44_44546


namespace remainder_of_4n_squared_l44_44691

theorem remainder_of_4n_squared {n : ℤ} (h : n % 13 = 7) : (4 * n^2) % 13 = 1 :=
by
  sorry

end remainder_of_4n_squared_l44_44691


namespace Lenora_scored_30_points_l44_44883

variable (x y : ℕ)
variable (hx : x + y = 40)
variable (three_point_success_rate : ℚ := 25 / 100)
variable (free_throw_success_rate : ℚ := 50 / 100)
variable (points_three_point : ℚ := 3)
variable (points_free_throw : ℚ := 1)
variable (three_point_contribution : ℚ := three_point_success_rate * points_three_point * x)
variable (free_throw_contribution : ℚ := free_throw_success_rate * points_free_throw * y)
variable (total_points : ℚ := three_point_contribution + free_throw_contribution)

theorem Lenora_scored_30_points : total_points = 30 :=
by
  sorry

end Lenora_scored_30_points_l44_44883


namespace pints_needed_for_9_cookies_l44_44764

-- Definitions based on the given conditions
def quarts_per_18_cookies : ℕ := 3
def pints_per_quart : ℕ := 2
def cookies_baked_with_milk (cookies pints : ℕ) := quarts_per_18_cookies * pints_per_quart = 2 * pints ∧ cookies = 18

-- The main theorem
theorem pints_needed_for_9_cookies : ∀ pints : ℕ, 
  (cookies_baked_with_milk 18 6) → pints = 3 → ∃ cookies : ℕ, cookies = 9 ∧ (cookies_baked_with_milk cookies pints) :=
by
  intro pints
  intro H
  intro Hp
  use 9
  split
  { rfl }
  { sorry }

end pints_needed_for_9_cookies_l44_44764


namespace calculate_grand_total_profit_l44_44418

-- Definitions based on conditions
def cost_per_type_A : ℕ := 8 * 10
def sell_price_type_A : ℕ := 125
def cost_per_type_B : ℕ := 12 * 18
def sell_price_type_B : ℕ := 280
def cost_per_type_C : ℕ := 15 * 12
def sell_price_type_C : ℕ := 350

def num_sold_type_A : ℕ := 45
def num_sold_type_B : ℕ := 35
def num_sold_type_C : ℕ := 25

-- Definition of profit calculations
def profit_per_type_A : ℕ := sell_price_type_A - cost_per_type_A
def profit_per_type_B : ℕ := sell_price_type_B - cost_per_type_B
def profit_per_type_C : ℕ := sell_price_type_C - cost_per_type_C

def total_profit_type_A : ℕ := num_sold_type_A * profit_per_type_A
def total_profit_type_B : ℕ := num_sold_type_B * profit_per_type_B
def total_profit_type_C : ℕ := num_sold_type_C * profit_per_type_C

def grand_total_profit : ℕ := total_profit_type_A + total_profit_type_B + total_profit_type_C

-- Statement to be proved
theorem calculate_grand_total_profit : grand_total_profit = 8515 := by
  sorry

end calculate_grand_total_profit_l44_44418


namespace problem_r_of_3_eq_88_l44_44384

def q (x : ℤ) : ℤ := 2 * x - 5
def r (x : ℤ) : ℤ := x^3 + 2 * x^2 - x - 4

theorem problem_r_of_3_eq_88 : r 3 = 88 :=
by
  sorry

end problem_r_of_3_eq_88_l44_44384


namespace wendys_sales_are_205_l44_44430

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end wendys_sales_are_205_l44_44430


namespace translate_line_up_l44_44946

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := -2 * x

-- Define the transformed line equation as a function
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Prove that translating the original line upward by 1 unit gives the translated line
theorem translate_line_up (x : ℝ) :
  original_line x + 1 = translated_line x :=
by
  unfold original_line translated_line
  simp

end translate_line_up_l44_44946


namespace ceil_square_of_neg_fraction_l44_44520

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l44_44520


namespace maria_paid_9_l44_44224

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l44_44224


namespace max_value_g_l44_44327

noncomputable def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_g : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x ∧ g x = 3 :=
sorry

end max_value_g_l44_44327


namespace octal_sum_l44_44834

open Nat

def octal_to_decimal (oct : ℕ) : ℕ :=
  match oct with
  | 0 => 0
  | n => let d3 := (n / 100) % 10
         let d2 := (n / 10) % 10
         let d1 := n % 10
         d3 * 8^2 + d2 * 8^1 + d1 * 8^0

def decimal_to_octal (dec : ℕ) : ℕ :=
  let rec aux (n : ℕ) (mul : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 8) (mul * 10) (acc + (n % 8) * mul)
  aux dec 1 0

theorem octal_sum :
  let a := 451
  let b := 167
  octal_to_decimal 451 + octal_to_decimal 167 = octal_to_decimal 640 := sorry

end octal_sum_l44_44834


namespace arithmetic_square_root_l44_44168

noncomputable def cube_root (x : ℝ) : ℝ :=
  x^(1/3)

noncomputable def sqrt_int_part (x : ℝ) : ℤ :=
  ⌊Real.sqrt x⌋

theorem arithmetic_square_root 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (h1 : cube_root a = 2) 
  (h2 : b = sqrt_int_part 5) 
  (h3 : c = 4 ∨ c = -4) : 
  Real.sqrt (a + ↑b + c) = Real.sqrt 14 ∨ Real.sqrt (a + ↑b + c) = Real.sqrt 6 := 
sorry

end arithmetic_square_root_l44_44168


namespace profit_distribution_l44_44459

theorem profit_distribution (x : ℕ) (hx : 2 * x = 4000) :
  let A := 2 * x
  let B := 3 * x
  let C := 5 * x
  A + B + C = 20000 := by
  sorry

end profit_distribution_l44_44459


namespace each_gets_10_fish_l44_44212

-- Define the constants and conditions
constant Ittymangnark Kingnook Oomyapeck : Type
constant fish : Type
constant eyes_of_fish : fish → ℕ
constant oomyapeck_eats_eyes : ℕ := 22
constant oomyapeck_gives_dog : ℕ := 2
constant total_eyes_eaten_by_oomyapeck : ℕ := oomyapeck_eats_eyes - oomyapeck_gives_dog
constant number_of_fish_oomyapeck_eats : ℕ := total_eyes_eaten_by_oomyapeck / 2
constant total_fish_divided : ℕ := number_of_fish_oomyapeck_eats
constant fish_split_equally : ℕ := total_fish_divided

-- The theorem statement
theorem each_gets_10_fish (day : Type) (H : Ittymangnark ≠ Kingnook ∧ Kingnook ≠ Oomyapeck ∧ Ittymangnark ≠ Oomyapeck) : 
  (number_of_fish_oomyapeck_eats = 10) ∧ (fish_split_equally = 10) :=
by {
  sorry
}

end each_gets_10_fish_l44_44212


namespace first_machine_rate_l44_44283

theorem first_machine_rate (x : ℝ) (h : (x + 55) * 30 = 2400) : x = 25 :=
by
  sorry

end first_machine_rate_l44_44283


namespace f_of_2014_l44_44858

theorem f_of_2014 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 4) = -f x + 2 * Real.sqrt 2)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  : f 2014 = Real.sqrt 2 :=
sorry

end f_of_2014_l44_44858


namespace first_step_induction_l44_44949

theorem first_step_induction (n : ℕ) (h : 1 < n) : 1 + 1/2 + 1/3 < 2 :=
by
  sorry

end first_step_induction_l44_44949


namespace boys_in_class_l44_44745

theorem boys_in_class 
  (avg_weight_incorrect : ℝ)
  (misread_weight_diff : ℝ)
  (avg_weight_correct : ℝ) 
  (n : ℕ) 
  (h1 : avg_weight_incorrect = 58.4) 
  (h2 : misread_weight_diff = 4) 
  (h3 : avg_weight_correct = 58.6) 
  (h4 : n * avg_weight_incorrect + misread_weight_diff = n * avg_weight_correct) :
  n = 20 := 
sorry

end boys_in_class_l44_44745


namespace distance_between_A_and_C_l44_44973

theorem distance_between_A_and_C :
  ∀ (AB BC CD AD AC : ℝ),
  AB = 3 → BC = 2 → CD = 5 → AD = 6 → AC = 1 := 
by
  intros AB BC CD AD AC hAB hBC hCD hAD
  have h1 : AD = AB + BC + CD := by sorry
  have h2 : 6 = 3 + 2 + AC := by sorry
  have h3 : 6 = 5 + AC := by sorry
  have h4 : AC = 1 := by sorry
  exact h4

end distance_between_A_and_C_l44_44973


namespace find_denominator_l44_44202

theorem find_denominator (y x : ℝ) (hy : y > 0) (h : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end find_denominator_l44_44202


namespace factorize_x_squared_plus_2x_l44_44317

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l44_44317


namespace find_five_digit_number_l44_44439

theorem find_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∃ rev_n : ℕ, rev_n = (n % 10) * 10000 + (n / 10 % 10) * 1000 + (n / 100 % 10) * 100 + (n / 1000 % 10) * 10 + (n / 10000) ∧ 9 * n = rev_n) ∧ n = 10989 :=
  sorry

end find_five_digit_number_l44_44439


namespace expression_evaluation_l44_44992

theorem expression_evaluation : 
  (81 ^ (1 / 4 - 1 / (log 4 / log 9)) + 25 ^ (log 8 / log 125)) * 49 ^ (log 2 / log 7) = 19 := 
by
  sorry

end expression_evaluation_l44_44992


namespace part1_solution_set_part2_range_of_a_l44_44903

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_solution_set (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x | x ≤ 0} ∪ {x | x ≥ 5} :=
by 
  -- proof goes here
  sorry

theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
by
  -- proof goes here
  sorry

end part1_solution_set_part2_range_of_a_l44_44903


namespace exists_infinitely_many_N_l44_44595

open Set

-- Conditions: Definition of the initial set S_0 and recursive sets S_n
variable {S_0 : Set ℕ} (h0 : Set.Finite S_0) -- S_0 is a finite set of positive integers
variable (S : ℕ → Set ℕ) 
(has_S : ∀ n, ∀ a, a ∈ S (n+1) ↔ (a-1 ∈ S n ∧ a ∉ S n ∨ a-1 ∉ S n ∧ a ∈ S n))

-- Main theorem: Proving the existence of infinitely many integers N such that 
-- S_N = S_0 ∪ {N + a : a ∈ S_0}
theorem exists_infinitely_many_N : 
  ∃ᶠ N in at_top, S N = S_0 ∪ {n | ∃ a ∈ S_0, n = N + a} := 
sorry

end exists_infinitely_many_N_l44_44595


namespace extra_apples_correct_l44_44077

def num_red_apples : ℕ := 6
def num_green_apples : ℕ := 15
def num_students : ℕ := 5
def num_apples_ordered : ℕ := num_red_apples + num_green_apples
def num_apples_taken : ℕ := num_students
def num_extra_apples : ℕ := num_apples_ordered - num_apples_taken

theorem extra_apples_correct : num_extra_apples = 16 := by
  sorry

end extra_apples_correct_l44_44077


namespace set_intersection_l44_44172

theorem set_intersection (M N : Set ℝ) 
  (hM : M = {x | 2 * x - 3 < 1}) 
  (hN : N = {x | -1 < x ∧ x < 3}) : 
  (M ∩ N) = {x | -1 < x ∧ x < 2} := 
by 
  sorry

end set_intersection_l44_44172


namespace dealer_purchased_articles_l44_44800

/-
The dealer purchases some articles for Rs. 25 and sells 12 articles for Rs. 38. 
The dealer has a profit percentage of 90%. Prove that the number of articles 
purchased by the dealer is 14.
-/

theorem dealer_purchased_articles (x : ℕ) 
    (total_cost : ℝ) (group_selling_price : ℝ) (group_size : ℕ) (profit_percentage : ℝ) 
    (h1 : total_cost = 25)
    (h2 : group_selling_price = 38)
    (h3 : group_size = 12)
    (h4 : profit_percentage = 90 / 100) :
    x = 14 :=
by
  sorry

end dealer_purchased_articles_l44_44800


namespace y_value_l44_44661

def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem y_value (x y : ℤ) (h1 : star 5 0 2 (-2) = (3, -2)) (h2 : star x y 0 3 = (3, -2)) :
  y = -5 :=
sorry

end y_value_l44_44661


namespace jessies_weight_after_first_week_l44_44044

-- Definitions from the conditions
def initial_weight : ℕ := 92
def first_week_weight_loss : ℕ := 56

-- The theorem statement
theorem jessies_weight_after_first_week : initial_weight - first_week_weight_loss = 36 := by
  -- Skip the proof
  sorry

end jessies_weight_after_first_week_l44_44044


namespace range_of_n_l44_44876

theorem range_of_n (m n : ℝ) (h₁ : n = m^2 + 2 * m + 2) (h₂ : |m| < 2) : -1 ≤ n ∧ n < 10 :=
sorry

end range_of_n_l44_44876


namespace part1_part2_l44_44565

-- Definition of the function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 1

-- Theorem for part (1)
theorem part1 
  (m n : ℝ)
  (h1 : ∀ x : ℝ, f x m < 0 ↔ -2 < x ∧ x < n) : 
  m = 3 / 2 ∧ n = 1 / 2 :=
sorry

-- Theorem for part (2)
theorem part2 
  (m : ℝ)
  (h2 : ∀ x : ℝ, m ≤ x ∧ x ≤ m + 1 → f x m < 0) : 
  -Real.sqrt 2 / 2 < m ∧ m < 0 :=
sorry

end part1_part2_l44_44565


namespace smallest_value_of_diff_l44_44589

-- Definitions of the side lengths from the conditions
def XY (x : ℝ) := x + 6
def XZ (x : ℝ) := 4 * x - 1
def YZ (x : ℝ) := x + 10

-- Conditions derived from the problem
noncomputable def valid_x (x : ℝ) := x > 5 / 3 ∧ x < 11 / 3

-- The proof statement
theorem smallest_value_of_diff : 
  ∀ (x : ℝ), valid_x x → (YZ x - XY x) = 4 :=
by
  intros x hx
  -- Proof goes here
  sorry

end smallest_value_of_diff_l44_44589


namespace amount_left_in_wallet_l44_44083

theorem amount_left_in_wallet
  (initial_amount : ℝ)
  (spent_amount : ℝ)
  (h_initial : initial_amount = 94)
  (h_spent : spent_amount = 16) :
  initial_amount - spent_amount = 78 :=
by
  sorry

end amount_left_in_wallet_l44_44083


namespace ratio_of_groups_l44_44460

variable (x : ℚ)

-- The total number of people in the calligraphy group
def calligraphy_group (x : ℚ) := x + (2 / 7) * x

-- The total number of people in the recitation group
def recitation_group (x : ℚ) := x + (1 / 5) * x

theorem ratio_of_groups (x : ℚ) (hx : x ≠ 0) : 
    (calligraphy_group x) / (recitation_group x) = (3 : ℚ) / (4 : ℚ) := by
  sorry

end ratio_of_groups_l44_44460


namespace problem_equivalence_l44_44561

noncomputable def S (n : ℕ) : ℝ := 2 ^ n

noncomputable def a : ℕ → ℝ
| 1       := 2
| (n + 1) := 2 ^ n

noncomputable def b (n : ℕ) : ℝ := a n * Real.log2 (a n)

noncomputable def T (n : ℕ) : ℝ :=
  ∑ k in finset.range n, b (k + 1)

theorem problem_equivalence (n : ℕ) : T n = (n - 2) * 2 ^ n + 4 := sorry

end problem_equivalence_l44_44561


namespace outer_squares_equal_three_times_inner_squares_l44_44821

theorem outer_squares_equal_three_times_inner_squares
  (a b c m_a m_b m_c : ℝ) 
  (h : m_a^2 + m_b^2 + m_c^2 = 3 / 4 * (a^2 + b^2 + c^2)) :
  a^2 + b^2 + c^2 = 3 * (m_a^2 + m_b^2 + m_c^2) := 
by 
  sorry

end outer_squares_equal_three_times_inner_squares_l44_44821


namespace find_sum_l44_44277

theorem find_sum {x y : ℝ} (h1 : x = 13.0) (h2 : x + y = 24) : 7 * x + 5 * y = 146 := 
by
  sorry

end find_sum_l44_44277


namespace ceil_square_of_neg_seven_fourths_l44_44535

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l44_44535


namespace part1_part2_l44_44901

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end part1_part2_l44_44901


namespace gold_beads_cannot_be_determined_without_cost_per_bead_l44_44304

-- Carly's bead conditions
def purple_rows : ℕ := 50
def purple_beads_per_row : ℕ := 20
def blue_rows : ℕ := 40
def blue_beads_per_row : ℕ := 18
def total_cost : ℝ := 180

-- The calculation of total purple and blue beads
def purple_beads : ℕ := purple_rows * purple_beads_per_row
def blue_beads : ℕ := blue_rows * blue_beads_per_row
def total_beads_without_gold : ℕ := purple_beads + blue_beads

-- Given the lack of cost per bead, the number of gold beads cannot be determined
theorem gold_beads_cannot_be_determined_without_cost_per_bead :
  ¬ (∃ cost_per_bead : ℝ, ∃ gold_beads : ℕ, (purple_beads + blue_beads + gold_beads) * cost_per_bead = total_cost) :=
sorry

end gold_beads_cannot_be_determined_without_cost_per_bead_l44_44304


namespace vec_addition_l44_44466

namespace VectorCalculation

open Real

def v1 : ℤ × ℤ := (3, -8)
def v2 : ℤ × ℤ := (2, -6)
def scalar : ℤ := 5

def scaled_v2 : ℤ × ℤ := (scalar * v2.1, scalar * v2.2)
def result : ℤ × ℤ := (v1.1 + scaled_v2.1, v1.2 + scaled_v2.2)

theorem vec_addition : result = (13, -38) := by
  sorry

end VectorCalculation

end vec_addition_l44_44466


namespace max_value_h3_solve_for_h_l44_44340

-- Definition part for conditions
def quadratic_function (h : ℝ) (x : ℝ) : ℝ :=
  -(x - h) ^ 2

-- Part (1): When h = 3, proving the maximum value of the function within 2 ≤ x ≤ 5 is 0.
theorem max_value_h3 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function 3 x ≤ 0 :=
by
  sorry

-- Part (2): If the maximum value of the function is -1, then the value of h is 6 or 1.
theorem solve_for_h (h : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function h x ≤ -1) ↔ h = 6 ∨ h = 1 :=
by
  sorry

end max_value_h3_solve_for_h_l44_44340


namespace sum_of_tangent_points_l44_44145

noncomputable def f (x : ℝ) : ℝ := 
  max (max (-7 * x - 19) (3 * x - 1)) (5 * x + 3)

theorem sum_of_tangent_points :
  ∃ x4 x5 x6 : ℝ, 
  (∃ q : ℝ → ℝ, 
    (∀ x, q x = f x ∨ (q x - (-7 * x - 19)) = b * (x - x4)^2
    ∨ (q x - (3 * x - 1)) = b * (x - x5)^2 
    ∨ (q x - (5 * x + 3)) = b * (x - x6)^2)) ∧
  x4 + x5 + x6 = -3.2 :=
sorry

end sum_of_tangent_points_l44_44145


namespace jacket_initial_reduction_percent_l44_44758

theorem jacket_initial_reduction_percent (P : ℝ) (x : ℝ) (h : P * (1 - x / 100) * 0.70 * 1.5873 = P) : x = 10 :=
sorry

end jacket_initial_reduction_percent_l44_44758


namespace percentage_error_in_area_l44_44442

noncomputable def side_with_error (s : ℝ) : ℝ := 1.04 * s

noncomputable def actual_area (s : ℝ) : ℝ := s ^ 2

noncomputable def calculated_area (s : ℝ) : ℝ := (side_with_error s) ^ 2

noncomputable def percentage_error (actual : ℝ) (calculated : ℝ) : ℝ :=
  ((calculated - actual) / actual) * 100

theorem percentage_error_in_area (s : ℝ) :
  percentage_error (actual_area s) (calculated_area s) = 8.16 := by
  sorry

end percentage_error_in_area_l44_44442


namespace sum_factorial_formula_l44_44833

def S (n : ℕ) : ℕ := (finset.range (n+1)).sum (λ i, (i + 1) * (i + 1).fact)

theorem sum_factorial_formula (n : ℕ) (h : n ≥ 1) : 
  S n = (n + 1).fact - 1 :=
sorry

end sum_factorial_formula_l44_44833


namespace compute_expression_l44_44471

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l44_44471


namespace number_square_roots_l44_44579

theorem number_square_roots (a x : ℤ) (h1 : x = (2 * a + 3) ^ 2) (h2 : x = (a - 18) ^ 2) : x = 169 :=
by 
  sorry

end number_square_roots_l44_44579


namespace geometric_progression_coincides_arithmetic_l44_44131

variables (a d q : ℝ)
variables (ap : ℕ → ℝ) (gp : ℕ → ℝ)

-- Define the N-th term of the arithmetic progression
def nth_term_ap (n : ℕ) : ℝ := a + n * d

-- Define the N-th term of the geometric progression
def nth_term_gp (n : ℕ) : ℝ := a * q^n

theorem geometric_progression_coincides_arithmetic :
  gp 3 = ap 10 →
  gp 4 = ap 74 :=
by
  intro h
  sorry

end geometric_progression_coincides_arithmetic_l44_44131


namespace find_b2_a2_minus_a1_l44_44857

theorem find_b2_a2_minus_a1 
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (d r : ℝ)
  (h_arith_seq : a₁ = -9 + d ∧ a₂ = a₁ + d)
  (h_geo_seq : b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ (-9) * (-1) = b₁ * b₃)
  (h_d_val : a₂ - a₁ = d)
  (h_b2_val : b₂ = -1) : 
  b₂ * (a₂ - a₁) = -8 :=
sorry

end find_b2_a2_minus_a1_l44_44857


namespace tangent_line_intersection_x_axis_l44_44793

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l44_44793


namespace thomas_weekly_wage_l44_44417

theorem thomas_weekly_wage (monthly_wage : ℕ) (weeks_in_month : ℕ) (weekly_wage : ℕ) 
    (h1 : monthly_wage = 19500) (h2 : weeks_in_month = 4) :
    weekly_wage = 4875 :=
by
  have h3 : weekly_wage = monthly_wage / weeks_in_month := sorry
  rw [h1, h2] at h3
  exact h3

end thomas_weekly_wage_l44_44417


namespace jimmy_cards_left_l44_44706

theorem jimmy_cards_left :
  ∀ (initial_cards jimmy_cards bob_cards mary_cards : ℕ),
    initial_cards = 18 →
    bob_cards = 3 →
    mary_cards = 2 * bob_cards →
    jimmy_cards = initial_cards - bob_cards - mary_cards →
    jimmy_cards = 9 := 
by
  intros initial_cards jimmy_cards bob_cards mary_cards h1 h2 h3 h4
  sorry

end jimmy_cards_left_l44_44706


namespace find_fixed_point_l44_44846

theorem find_fixed_point (c d k : ℝ) 
(h : ∀ k : ℝ, d = 5 * c^2 + k * c - 3 * k) : (c, d) = (3, 45) :=
sorry

end find_fixed_point_l44_44846


namespace ceil_square_neg_seven_over_four_l44_44501

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l44_44501


namespace ceil_square_neg_fraction_l44_44529

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l44_44529


namespace ceil_square_of_neg_seven_fourths_l44_44531

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l44_44531


namespace plane_passing_through_A_perpendicular_to_BC_l44_44638

-- Define the points A, B, and C
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := { x := -3, y := 7, z := 2 }
def B : Point3D := { x := 3, y := 5, z := 1 }
def C : Point3D := { x := 4, y := 5, z := 3 }

-- Define the vector BC as the difference between points C and B
def vectorBC (B C : Point3D) : Point3D :=
{ x := C.x - B.x,
  y := C.y - B.y,
  z := C.z - B.z }

-- Define the equation of the plane passing through point A and 
-- perpendicular to vector BC
def plane_eq (A : Point3D) (n : Point3D) (x y z : ℝ) : Prop :=
n.x * (x - A.x) + n.y * (y - A.y) + n.z * (z - A.z) = 0

-- Define the proof problem
theorem plane_passing_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ), plane_eq A (vectorBC B C) x y z ↔ x + 2 * z - 1 = 0 :=
by
  -- the proof part
  sorry

end plane_passing_through_A_perpendicular_to_BC_l44_44638


namespace num_girls_in_school_l44_44121

noncomputable def total_students : ℕ := 1600
noncomputable def sample_students : ℕ := 200
noncomputable def girls_less_than_boys_in_sample : ℕ := 10

-- Equations from conditions
def boys_in_sample (B G : ℕ) : Prop := G = B - girls_less_than_boys_in_sample
def sample_size (B G : ℕ) : Prop := B + G = sample_students

-- Proportion condition
def proportional_condition (G G_total : ℕ) : Prop := G * total_students = G_total * sample_students

-- Total number of girls in the school
def total_girls_in_school (G_total : ℕ) : Prop := G_total = 760

theorem num_girls_in_school :
  ∃ B G G_total : ℕ, boys_in_sample B G ∧ sample_size B G ∧ proportional_condition G G_total ∧ total_girls_in_school G_total :=
sorry

end num_girls_in_school_l44_44121


namespace population_growth_l44_44757

theorem population_growth 
  (P₀ : ℝ) (P₂ : ℝ) (r : ℝ)
  (hP₀ : P₀ = 15540) 
  (hP₂ : P₂ = 25460.736)
  (h_growth : P₂ = P₀ * (1 + r)^2) :
  r = 0.28 :=
by 
  sorry

end population_growth_l44_44757


namespace sum_of_slopes_of_tangents_l44_44545

open Real

noncomputable def eccentricity (a b : ℝ) : ℝ := √(1 - (b^2 / a^2))

theorem sum_of_slopes_of_tangents 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) 
  (k : ℝ) (hE : eccentricity a b = sqrt 2 / 2)
  (hx1n : x1 ≠ 0) :
  let E := λ x y, (x^2 / a^2) + (y^2 / b^2) = 1 in
  let A := (0, -b) in
  let P := (x1, k * (x1 - 1) + 1) in
  let Q := (x2, k * (x2 - 1) + 1) in
  x1 ≠ 1 ∧ x2 ≠ 1 ∧
  sum_of_roots := (λ x1 x2 k, x1 + x2) in
  ∃ x1 x2, (E x1 (k * (x1 - 1) + 1)) ∧ (E x2 (k * (x2 - 1) + 1))
           ∧ x1 ≠ x2 
           ∧ (k + sum_of_roots x1 x2 k) = 2 :=
begin
  sorry
end

end sum_of_slopes_of_tangents_l44_44545


namespace sum_of_other_endpoint_l44_44724

theorem sum_of_other_endpoint (x y : ℕ) : 
  (6 + x = 10) ∧ (1 + y = 14) → x + y = 17 := 
by
  intro h
  cases h with h1 h2
  have hx := by linarith
  have hy := by linarith
  rw [hx, hy]
  exact rfl

end sum_of_other_endpoint_l44_44724


namespace sin_sum_triangle_inequality_l44_44220

theorem sin_sum_triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_inequality_l44_44220


namespace quadratic_no_real_roots_l44_44775

theorem quadratic_no_real_roots :
  ¬ (∃ x : ℝ, x^2 - 2 * x + 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0) ∧ (x2^2 - 3 * x2 = 0) ∧
  ∃ y : ℝ, y^2 - 2 * y + 1 = 0 :=
by
  sorry

end quadratic_no_real_roots_l44_44775


namespace simplify_expression_l44_44538

theorem simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 :=
by
  sorry

end simplify_expression_l44_44538


namespace proper_sets_exist_l44_44132

def proper_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, (1 ≤ w ∧ w ≤ 500) → ∃ (used_weights : List ℕ), (used_weights ⊆ weights) ∧ (used_weights.sum = w ∧ ∀ (alternative_weights : List ℕ), (alternative_weights ⊆ weights ∧ alternative_weights.sum = w) → used_weights = alternative_weights)

theorem proper_sets_exist (weights : List ℕ) :
  (weights.sum = 500) → 
  ∃ (sets : List (List ℕ)), sets.length = 3 ∧ (∀ s ∈ sets, proper_set s) :=
by
  sorry

end proper_sets_exist_l44_44132


namespace cos_triple_angle_l44_44186

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l44_44186


namespace number_of_cloth_bags_l44_44244

-- Definitions based on the conditions
def dozen := 12

def total_peaches : ℕ := 5 * dozen
def peaches_in_knapsack : ℕ := 12
def peaches_per_bag : ℕ := 2 * peaches_in_knapsack

-- The proof statement
theorem number_of_cloth_bags :
  (total_peaches - peaches_in_knapsack) / peaches_per_bag = 2 := by
  sorry

end number_of_cloth_bags_l44_44244


namespace sum_first_six_terms_geometric_sequence_l44_44838

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end sum_first_six_terms_geometric_sequence_l44_44838


namespace max_value_g_l44_44328

noncomputable def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_g : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x ∧ g x = 3 :=
sorry

end max_value_g_l44_44328


namespace necessary_but_not_sufficient_condition_l44_44117

theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 - 4 = 0 → x + 2 = 0 :=
by
  sorry

end necessary_but_not_sufficient_condition_l44_44117


namespace fraction_zero_condition_l44_44359

theorem fraction_zero_condition (x : ℝ) (h : (abs x - 2) / (2 - x) = 0) : x = -2 :=
by
  sorry

end fraction_zero_condition_l44_44359


namespace angle_b_is_sixty_max_area_triangle_l44_44558

variables (a b c A B C : Real) (A_pos B_pos C_pos : Prop)
variables (A_sum : A + B + C = π) -- angles of a triangle
variables (triangle_sides : a = 2 * sin(A) ∨ b = 2 * sin(B) ∨ c = 2 * sin(C)) -- Law of Sines
variables (condition : (a + c) / b = cos(C) + sqrt(3) * sin(C))

noncomputable theory

-- Part 1: Show B is 60 degrees
theorem angle_b_is_sixty (hA : A_pos A) (hB : B_pos B) (hC : C_pos C) :
  B = π / 3 :=
  sorry

-- Part 2: Given b = 2, show the maximum area of the triangle is sqrt(3)
theorem max_area_triangle (hb : b = 2) :
  (∃ a c, (a + c) / b = cos(C) + sqrt(3) * sin(C) ∧
          let S := 1 / 2 * a * c * sin(B) in
          ∀ a' c', (a' + c') / b = cos(C) + sqrt(3) * sin(C) → 
                   1 / 2 * a' * c' * sin(B) ≤ S ∧ S = sqrt(3)) :=
  sorry

end angle_b_is_sixty_max_area_triangle_l44_44558


namespace amount_paid_l44_44226

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l44_44226


namespace tan_theta_values_l44_44161

theorem tan_theta_values (θ : ℝ) (h₁ : 0 < θ ∧ θ < Real.pi / 2) (h₂ : 12 / Real.sin θ + 12 / Real.cos θ = 35) : 
  Real.tan θ = 4 / 3 ∨ Real.tan θ = 3 / 4 := 
by
  sorry

end tan_theta_values_l44_44161


namespace cos_triple_angle_l44_44189

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l44_44189


namespace probability_five_digit_palindrome_divisible_by_11_l44_44454

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  n % 100 = 100*a + 10*b + c

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_five_digit_palindrome_divisible_by_11 :
  let count_palindromes := 9 * 10 * 10
  let count_divisible_by_11 := 165
  (count_divisible_by_11 : ℚ) / count_palindromes = 11 / 60 :=
by
  sorry

end probability_five_digit_palindrome_divisible_by_11_l44_44454


namespace farmer_initial_apples_l44_44751

variable (initial_apples given_away_apples remaining_apples : ℕ)

def initial_apple_count (given_away_apples remaining_apples : ℕ) : ℕ :=
  given_away_apples + remaining_apples

theorem farmer_initial_apples : initial_apple_count 88 39 = 127 := by
  -- Given conditions
  let given_away_apples := 88
  let remaining_apples := 39

  -- Calculate the initial apples
  let initial_apples := initial_apple_count given_away_apples remaining_apples

  -- We are supposed to prove initial apples count is 127
  show initial_apples = 127
  sorry

end farmer_initial_apples_l44_44751


namespace greatest_integer_n_l44_44633

theorem greatest_integer_n (n : ℤ) : n^2 - 9 * n + 20 ≤ 0 → n ≤ 5 := sorry

end greatest_integer_n_l44_44633


namespace acute_triangle_incorrect_option_l44_44367

theorem acute_triangle_incorrect_option (A B C : ℝ) (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90)
  (angle_sum : A + B + C = 180) (h_order : A > B ∧ B > C) : ¬(B + C < 90) :=
sorry

end acute_triangle_incorrect_option_l44_44367


namespace split_tips_evenly_l44_44374

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l44_44374


namespace radius_of_circle_l44_44585

variable {O : Type*} [MetricSpace O]

def distance_near : ℝ := 1
def distance_far : ℝ := 7
def diameter : ℝ := distance_near + distance_far

theorem radius_of_circle (P : O) (r : ℝ) (h1 : distance_near = 1) (h2 : distance_far = 7) :
  r = diameter / 2 :=
by
  -- Proof would go here 
  sorry

end radius_of_circle_l44_44585


namespace number_of_valid_triangles_l44_44870

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l44_44870


namespace cos_triple_angle_l44_44192

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l44_44192


namespace smallest_number_l44_44810

theorem smallest_number (x1 x2 x3 x4 : ℝ) 
  (h1 : x1 = -real.pi)
  (h2 : x2 = -3)
  (h3 : x3 = -real.sqrt 2)
  (h4 : x4 = -5 / 2) : 
  x1 < x2 ∧ x1 < x3 ∧ x1 < x4 :=
by
  sorry

end smallest_number_l44_44810


namespace number_of_zeros_g_l44_44018

variable (f : ℝ → ℝ)
variable (hf_cont : continuous f)
variable (hf_diff : differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, x * (deriv f x) + f x > 0)

theorem number_of_zeros_g (hg : ∀ x : ℝ, x > 0 → x * f x + 1 = 0 → false) : 
    ∀ x : ℝ , x > 0 → ¬ (x * f x + 1 = 0) :=
by
  sorry

end number_of_zeros_g_l44_44018


namespace proof_problem_l44_44135

variable (pots_basil pots_rosemary pots_thyme : ℕ)
variable (leaves_per_basil leaves_per_rosemary leaves_per_thyme : ℕ)
variable (total_leaves : ℕ)

-- Define the given conditions
def conditions : Prop :=
  pots_basil = 3 ∧
  leaves_per_basil = 4 ∧
  pots_rosemary = 9 ∧
  leaves_per_rosemary = 18 ∧
  pots_thyme = 6 ∧
  leaves_per_thyme = 30

-- Define the question and the correct answer
def correct_answer : Prop :=
  total_leaves = 354

-- Translate to proof problem
theorem proof_problem : conditions → (total_leaves = pots_basil * leaves_per_basil + pots_rosemary * leaves_per_rosemary + pots_thyme * leaves_per_thyme) → correct_answer :=
by
  intro h1 h2
  exact h2
  sorry -- proof placeholder

end proof_problem_l44_44135


namespace union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l44_44217

section
  def A : Set ℝ := {x : ℝ | ∃ q : ℚ, x = q}
  def B : Set ℝ := {x : ℝ | ¬ ∃ q : ℚ, x = q}

  theorem union_rational_irrational_is_real : A ∪ B = Set.univ :=
  by
    sorry

  theorem intersection_rational_irrational_is_empty : A ∩ B = ∅ :=
  by
    sorry
end

end union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l44_44217


namespace ham_and_bread_percentage_l44_44939

-- Defining the different costs as constants
def cost_of_bread : ℝ := 50
def cost_of_ham : ℝ := 150
def cost_of_cake : ℝ := 200

-- Defining the total cost of the items
def total_cost : ℝ := cost_of_bread + cost_of_ham + cost_of_cake

-- Defining the combined cost of ham and bread
def combined_cost_ham_and_bread : ℝ := cost_of_bread + cost_of_ham

-- The theorem stating that the combined cost of ham and bread is 50% of the total cost
theorem ham_and_bread_percentage : (combined_cost_ham_and_bread / total_cost) * 100 = 50 := by
  sorry  -- Proof to be provided

end ham_and_bread_percentage_l44_44939


namespace sum_of_six_digits_l44_44397

open Finset

theorem sum_of_six_digits 
(vars_cols : Finset ℕ) (vars_rows : Finset ℕ) 
(h1 : vars_cols ⊆ {2, 4, 6, 8}) (h2 : vars_rows ⊆ {1, 3, 5, 7, 9})
(h3 : vars_cols.sum id = 22) (h4 : vars_rows.sum id = 14)
(h5 : (vars_cols ∪ vars_rows).card = 6):
  (vars_cols ∪ vars_rows).sum id = 30 := 
  sorry

end sum_of_six_digits_l44_44397


namespace exists_a_lt_0_l44_44482

noncomputable def f : ℝ → ℝ :=
sorry

theorem exists_a_lt_0 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (Real.sqrt (x * y)) = (f x + f y) / 2)
  (h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :
  ∃ a : ℝ, 0 < a ∧ f a < 0 :=
sorry

end exists_a_lt_0_l44_44482


namespace farmer_initial_apples_l44_44752

variable (initial_apples given_away_apples remaining_apples : ℕ)

def initial_apple_count (given_away_apples remaining_apples : ℕ) : ℕ :=
  given_away_apples + remaining_apples

theorem farmer_initial_apples : initial_apple_count 88 39 = 127 := by
  -- Given conditions
  let given_away_apples := 88
  let remaining_apples := 39

  -- Calculate the initial apples
  let initial_apples := initial_apple_count given_away_apples remaining_apples

  -- We are supposed to prove initial apples count is 127
  show initial_apples = 127
  sorry

end farmer_initial_apples_l44_44752


namespace total_pages_proof_l44_44593

/-
Conditions:
1. Johnny's essay has 150 words.
2. Madeline's essay is double the length of Johnny's essay.
3. Timothy's essay has 30 more words than Madeline's essay.
4. One page contains 260 words.

Question:
Prove that the total number of pages do Johnny, Madeline, and Timothy's essays fill is 5.
-/

def johnny_words : ℕ := 150
def words_per_page : ℕ := 260

def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 30

def pages (words : ℕ) : ℕ := (words + words_per_page - 1) / words_per_page  -- division rounding up

def johnny_pages : ℕ := pages johnny_words
def madeline_pages : ℕ := pages madeline_words
def timothy_pages : ℕ := pages timothy_words

def total_pages : ℕ := johnny_pages + madeline_pages + timothy_pages

theorem total_pages_proof : total_pages = 5 :=
by sorry

end total_pages_proof_l44_44593


namespace range_of_transformed_sine_function_l44_44154

theorem range_of_transformed_sine_function :
  (∀ y, ∃ x, (0 < x ∧ x < 2 * Real.pi / 3) ∧ y = 2 * Real.sin (x + Real.pi / 6) - 1) ↔ (0 < y ∧ y ≤ 1) :=
sorry

end range_of_transformed_sine_function_l44_44154


namespace ceiling_of_square_of_neg_7_over_4_is_4_l44_44508

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l44_44508


namespace inequality_condition_l44_44899

noncomputable def inequality_holds_for_all (a b c : ℝ) : Prop :=
  ∀ (x : ℝ), a * Real.sin x + b * Real.cos x + c > 0

theorem inequality_condition (a b c : ℝ) :
  inequality_holds_for_all a b c ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end inequality_condition_l44_44899


namespace no_real_roots_range_l44_44345

theorem no_real_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≠ 0) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end no_real_roots_range_l44_44345


namespace one_odd_one_even_l44_44356

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem one_odd_one_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prime : is_prime a) (h_eq : a^2 + b^2 = c^2) : 
(is_odd b ∧ is_even c) ∨ (is_even b ∧ is_odd c) :=
sorry

end one_odd_one_even_l44_44356


namespace g_at_6_l44_44247

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_at_3 : g 3 = 4

theorem g_at_6 : g 6 = 8 :=
by 
  sorry

end g_at_6_l44_44247


namespace least_product_of_distinct_primes_gt_30_l44_44097

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l44_44097


namespace sum_of_first_15_terms_l44_44961

variable (a d : ℕ)

def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_15_terms (h : nth_term 4 + nth_term 12 = 16) : sum_of_first_n_terms 15 = 120 :=
by
  sorry

end sum_of_first_15_terms_l44_44961


namespace remainder_division_l44_44847

theorem remainder_division
  (P E M S F N T : ℕ)
  (h1 : P = E * M + S)
  (h2 : M = N * F + T) :
  (∃ r, P = (EF + 1) * (P / (EF + 1)) + r ∧ r = ET + S - N) :=
sorry

end remainder_division_l44_44847


namespace squares_perimeter_and_rectangle_area_l44_44932

theorem squares_perimeter_and_rectangle_area (x y : ℝ) (hx : x^2 + y^2 = 145) (hy : x^2 - y^2 = 105) : 
  (4 * x + 4 * y = 28 * Real.sqrt 5) ∧ ((x + y) * x = 175) := 
by 
  sorry

end squares_perimeter_and_rectangle_area_l44_44932


namespace ratio_a_to_c_l44_44034

variable (a b c : ℚ)

theorem ratio_a_to_c (h1 : a / b = 7 / 3) (h2 : b / c = 1 / 5) : a / c = 7 / 15 := 
sorry

end ratio_a_to_c_l44_44034


namespace calculate_a3_b3_l44_44141

theorem calculate_a3_b3 (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by
  sorry

end calculate_a3_b3_l44_44141


namespace least_even_p_l44_44845

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem least_even_p 
  (p : ℕ) 
  (hp : 2 ∣ p) -- p is an even integer
  (h : is_square (300 * p)) -- 300 * p is the square of an integer
  : p = 3 := 
sorry

end least_even_p_l44_44845


namespace find_angleB_find_maxArea_l44_44557

noncomputable def angleB (a b c : ℝ) (C : ℝ) :=
  (a + c) / b = Real.cos C + Real.sqrt 3 * Real.sin C

noncomputable def maxArea (a b c : ℝ) (B : ℝ) :=
  b = 2

theorem find_angleB (a b c : ℝ) (C : ℝ) (h : angleB a b c C) : 
  ∃ B, B = 60 ∧ angleB a b c C :=
sorry

theorem find_maxArea (a b c : ℝ) (B : ℝ) (hB : B = 60) (hb : maxArea a b c B) :
  ∃ S, S = Real.sqrt 3 :=
sorry

end find_angleB_find_maxArea_l44_44557


namespace probability_product_multiple_of_4_l44_44606

-- Definitions based on conditions
def spinner_paco : ℕ → Prop := λ n, 1 ≤ n ∧ n ≤ 5
def spinner_manu : ℕ → Prop := λ n, 1 ≤ n ∧ n ≤ 8

def multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- The theorem statement
theorem probability_product_multiple_of_4 :
  (∑ p in finset.filter spinner_paco (finset.range 6), 
     ∑ m in finset.filter spinner_manu (finset.range 9), 
     if multiple_of_4 (p * m) then (1 / 5) * (1 / 8) else 0) = 2 / 5 :=
by {
 sorry
}

end probability_product_multiple_of_4_l44_44606


namespace least_prime_product_l44_44101
open_locale classical

theorem least_prime_product {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) (hp_distinct : p ≠ q) (hp_gt_30 : p > 30) (hq_gt_30 : q > 30) :
  p * q = 1147 :=
by sorry

end least_prime_product_l44_44101


namespace number_of_marbles_in_Ellen_box_l44_44843

-- Defining the conditions given in the problem
def Dan_box_volume : ℕ := 216
def Ellen_side_multiplier : ℕ := 3
def marble_size_consistent_between_boxes : Prop := True -- Placeholder for the consistency condition

-- Main theorem statement
theorem number_of_marbles_in_Ellen_box :
  ∃ number_of_marbles_in_Ellen_box : ℕ,
  (∀ s : ℕ, s^3 = Dan_box_volume → (Ellen_side_multiplier * s)^3 / s^3 = 27 → 
  number_of_marbles_in_Ellen_box = 27 * Dan_box_volume) :=
by
  sorry

end number_of_marbles_in_Ellen_box_l44_44843


namespace correct_equation_l44_44944

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l44_44944


namespace quadratic_roots_l44_44759

theorem quadratic_roots (x : ℝ) (h : x^2 - 1 = 3) : x = 2 ∨ x = -2 :=
by
  sorry

end quadratic_roots_l44_44759


namespace concyclic_H_E_N_N1_N2_l44_44210

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def nine_point_center (A B C : Point) : Point := sorry
noncomputable def altitude (A B C : Point) : Point := sorry
noncomputable def salmon_circle_center (A O O₁ O₂ : Point) : Point := sorry
noncomputable def foot_of_perpendicular (O' B C : Point) : Point := sorry
noncomputable def is_concyclic (points : List Point) : Prop := sorry

theorem concyclic_H_E_N_N1_N2 (A B C D : Point):
  let H := altitude A B C
  let O := circumcenter A B C
  let O₁ := circumcenter A B D
  let O₂ := circumcenter A C D
  let N := nine_point_center A B C
  let N₁ := nine_point_center A B D
  let N₂ := nine_point_center A C D
  let O' := salmon_circle_center A O O₁ O₂
  let E := foot_of_perpendicular O' B C
  is_concyclic [H, E, N, N₁, N₂] :=
sorry

end concyclic_H_E_N_N1_N2_l44_44210


namespace convert_15_deg_to_rad_l44_44146

theorem convert_15_deg_to_rad (deg_to_rad : ℝ := Real.pi / 180) : 
  15 * deg_to_rad = Real.pi / 12 :=
by sorry

end convert_15_deg_to_rad_l44_44146


namespace point_inside_circle_range_l44_44200

theorem point_inside_circle_range (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) :=
  by
  sorry

end point_inside_circle_range_l44_44200


namespace repeating_decimal_to_fraction_l44_44105

noncomputable def x : ℚ := 0.6 + 41 / 990  

theorem repeating_decimal_to_fraction (h : x = 0.6 + 41 / 990) : x = 127 / 198 :=
by sorry

end repeating_decimal_to_fraction_l44_44105


namespace solve_for_x_l44_44108

theorem solve_for_x (x : ℝ) : (1 + 2*x + 3*x^2) / (3 + 2*x + x^2) = 3 → x = -2 :=
by
  intro h
  sorry

end solve_for_x_l44_44108


namespace larger_group_men_count_l44_44448

-- Define the conditions
def total_man_days (men : ℕ) (days : ℕ) : ℕ := men * days

-- Define the total work for 36 men in 18 days
def work_by_36_men_in_18_days : ℕ := total_man_days 36 18

-- Define the number of days the larger group takes
def days_for_larger_group : ℕ := 8

-- Problem Statement: Prove that if 36 men take 18 days to complete the work, and a larger group takes 8 days, then the larger group consists of 81 men.
theorem larger_group_men_count : 
  ∃ (M : ℕ), total_man_days M days_for_larger_group = work_by_36_men_in_18_days ∧ M = 81 := 
by
  -- Here would go the proof steps
  sorry

end larger_group_men_count_l44_44448


namespace factorize_x_squared_plus_2x_l44_44313

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l44_44313


namespace valid_functions_l44_44152

theorem valid_functions (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) * g (x - y) = (g x + g y)^2 - 4 * x^2 * g y + 2 * y^2 * g x) :
  (∀ x, g x = 0) ∨ (∀ x, g x = x^2) :=
by sorry

end valid_functions_l44_44152


namespace num_triangles_with_positive_area_l44_44872

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l44_44872


namespace geometric_seq_sum_l44_44252

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q))
    (h2 : S 10 = 10) (h3 : S 30 = 70) (hq_pos : 0 < q) :
    S 40 = 150 := by
  sorry

end geometric_seq_sum_l44_44252


namespace price_per_pound_second_coffee_l44_44290

theorem price_per_pound_second_coffee
  (price_first : ℝ) (total_mix_weight : ℝ) (sell_price_per_pound : ℝ) (each_kind_weight : ℝ) 
  (total_sell_price : ℝ) (total_first_cost : ℝ) (total_second_cost : ℝ) (price_second : ℝ) :
  price_first = 2.15 →
  total_mix_weight = 18 →
  sell_price_per_pound = 2.30 →
  each_kind_weight = 9 →
  total_sell_price = total_mix_weight * sell_price_per_pound →
  total_first_cost = each_kind_weight * price_first →
  total_second_cost = total_sell_price - total_first_cost →
  price_second = total_second_cost / each_kind_weight →
  price_second = 2.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end price_per_pound_second_coffee_l44_44290


namespace distance_between_parallel_sides_l44_44829

theorem distance_between_parallel_sides (a b : ℝ) (h : ℝ) (A : ℝ) :
  a = 20 → b = 10 → A = 150 → (A = 1 / 2 * (a + b) * h) → h = 10 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end distance_between_parallel_sides_l44_44829


namespace digit_is_9_if_divisible_by_11_l44_44548

theorem digit_is_9_if_divisible_by_11 (d : ℕ) : 
  (678000 + 9000 + 800 + 90 + d) % 11 = 0 -> d = 9 := by
  sorry

end digit_is_9_if_divisible_by_11_l44_44548


namespace sum_largest_smallest_ABC_l44_44587

def hundreds (n : ℕ) : ℕ := n / 100
def units (n : ℕ) : ℕ := n % 10
def tens (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_largest_smallest_ABC : 
  (hundreds 297 = 2) ∧ (units 297 = 7) ∧ (hundreds 207 = 2) ∧ (units 207 = 7) →
  (297 + 207 = 504) :=
by
  sorry

end sum_largest_smallest_ABC_l44_44587


namespace probability_stopping_in_C_l44_44451

noncomputable def probability_C : ℚ :=
  let P_A := 1 / 5
  let P_B := 1 / 5
  let x := (1 - (P_A + P_B)) / 3
  x

theorem probability_stopping_in_C :
  probability_C = 1 / 5 :=
by
  unfold probability_C
  sorry

end probability_stopping_in_C_l44_44451


namespace number_of_pairs_l44_44022

theorem number_of_pairs (x y : ℤ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000) :
  (x^2 + y^2) % 7 = 0 → (∃ n : ℕ, n = 20164) :=
by {
  sorry
}

end number_of_pairs_l44_44022


namespace quadratic_no_real_roots_l44_44079

theorem quadratic_no_real_roots :
  ∀ x : ℝ, ¬(x^2 - 2 * x + 3 = 0) :=
by
  sorry

end quadratic_no_real_roots_l44_44079


namespace unripe_oranges_after_days_l44_44570

-- Definitions and Conditions
def sacks_per_day := 65
def days := 6

-- Statement to prove
theorem unripe_oranges_after_days : sacks_per_day * days = 390 := by
  sorry

end unripe_oranges_after_days_l44_44570


namespace factorization_correct_l44_44319

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l44_44319


namespace ceiling_of_square_frac_l44_44500

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l44_44500


namespace neither_sufficient_nor_necessary_l44_44572

theorem neither_sufficient_nor_necessary (α β : ℝ) :
  (α + β = 90) ↔ ¬((α + β = 90) ↔ (Real.sin α + Real.sin β > 1)) :=
sorry

end neither_sufficient_nor_necessary_l44_44572


namespace greatest_possible_large_chips_l44_44936

theorem greatest_possible_large_chips 
  (total : ℕ) (s l p : ℕ) (prime_p : Nat.Prime p) (chips_eq : s + l = total) (num_eq : s = l + p)
  (total_eq : total = 72) : 
  l ≤ 35 := by
  have H : 2 * l + p = total := sorry -- Derived from s + l = 72 and s = l + p
  have p_value : p = 2 := sorry -- Smallest prime
  have H1 : 2 * l + 2 = 72 := sorry -- Substituting p = 2
  have H2 : 2 * l = 70 := sorry -- Simplifying
  have H3 : l = 35 := sorry -- Solving for l
  l ≤ 35 := sorry

end greatest_possible_large_chips_l44_44936


namespace cos_triple_angle_l44_44191

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l44_44191


namespace nursery_school_students_l44_44605

theorem nursery_school_students (S : ℕ)
  (h1 : ∃ x, x = S / 10)
  (h2 : 20 + (S / 10) = 25) : S = 50 :=
by
  sorry

end nursery_school_students_l44_44605


namespace problem_2_8_3_4_7_2_2_l44_44478

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l44_44478


namespace ceiling_of_square_frac_l44_44497

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l44_44497


namespace percent_apple_juice_in_blend_l44_44892

noncomputable def juice_blend_apple_percentage : ℚ :=
  let apple_juice_per_apple := 9 / 2
  let plum_juice_per_plum := 12 / 3
  let total_apple_juice := 4 * apple_juice_per_apple
  let total_plum_juice := 6 * plum_juice_per_plum
  let total_juice := total_apple_juice + total_plum_juice
  (total_apple_juice / total_juice) * 100

theorem percent_apple_juice_in_blend :
  juice_blend_apple_percentage = 43 :=
by
  sorry

end percent_apple_juice_in_blend_l44_44892


namespace curves_intersect_exactly_three_points_l44_44637

theorem curves_intersect_exactly_three_points (a : ℝ) :
  (∃! (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = a ^ 2 ∧ p.2 = p.1 ^ 2 - a) ↔ a > (1 / 2) :=
by sorry

end curves_intersect_exactly_three_points_l44_44637


namespace find_b_l44_44680

-- Define the conditions as hypotheses
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + b*x - 3

theorem find_b (x₁ x₂ b : ℝ) (h₁ : x₁ ≠ x₂)
  (h₂ : 3 * x₁^2 + 4 * x₁ + b = 0)
  (h₃ : 3 * x₂^2 + 4 * x₂ + b = 0)
  (h₄ : x₁^2 + x₂^2 = 34 / 9) :
  b = -3 :=
by
  -- Proof will be inserted here
  sorry

end find_b_l44_44680


namespace quadratic_eq_c_has_equal_roots_l44_44861

theorem quadratic_eq_c_has_equal_roots (c : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + c = 0 ∧
                      ∀ y : ℝ, x^2 - 4 * x + c = 0 → y = x) : c = 4 := sorry

end quadratic_eq_c_has_equal_roots_l44_44861


namespace shorter_piece_length_l44_44273

theorem shorter_piece_length (total_len : ℝ) (h1 : total_len = 60)
                            (short_len long_len : ℝ) (h2 : long_len = (1 / 2) * short_len)
                            (h3 : short_len + long_len = total_len) :
  short_len = 40 := 
  sorry

end shorter_piece_length_l44_44273


namespace solve_system_of_equations_l44_44447

theorem solve_system_of_equations (m b : ℤ) 
  (h1 : 3 * m + b = 11)
  (h2 : -4 * m - b = 11) : 
  m = -22 ∧ b = 77 :=
  sorry

end solve_system_of_equations_l44_44447


namespace ceil_square_neg_fraction_l44_44530

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l44_44530


namespace other_x_intercept_vertex_symmetric_l44_44156

theorem other_x_intercept_vertex_symmetric (a b c : ℝ)
  (h_vertex : ∀ x y : ℝ, (4, 10) = (x, y) → y = a * x^2 + b * x + c)
  (h_intercept : ∀ x : ℝ, (-1, 0) = (x, 0) → a * x^2 + b * x + c = 0) :
  a * 9^2 + b * 9 + c = 0 :=
sorry

end other_x_intercept_vertex_symmetric_l44_44156


namespace rewrite_expression_l44_44057

theorem rewrite_expression : ∀ x : ℝ, x^2 + 4 * x + 1 = (x + 2)^2 - 3 :=
by
  intros
  sorry

end rewrite_expression_l44_44057


namespace length_of_courtyard_l44_44419

-- Define the dimensions and properties of the courtyard and paving stones
def width := 33 / 2
def numPavingStones := 132
def pavingStoneLength := 5 / 2
def pavingStoneWidth := 2

-- Total area covered by paving stones
def totalArea := numPavingStones * (pavingStoneLength * pavingStoneWidth)

-- To prove: Length of the courtyard
theorem length_of_courtyard : totalArea / width = 40 := by
  sorry

end length_of_courtyard_l44_44419


namespace moles_KOH_combined_l44_44153

-- Define the number of moles of KI produced
def moles_KI_produced : ℕ := 3

-- Define the molar ratio from the balanced chemical equation
def molar_ratio_KOH_NH4I_KI : ℕ := 1

-- The number of moles of KOH combined to produce the given moles of KI
theorem moles_KOH_combined (moles_KOH moles_NH4I : ℕ) (h : moles_NH4I = 3) 
  (h_produced : moles_KI_produced = 3) (ratio : molar_ratio_KOH_NH4I_KI = 1) :
  moles_KOH = 3 :=
by {
  -- Placeholder for proof, use sorry to skip proving
  sorry
}

end moles_KOH_combined_l44_44153


namespace ceiling_of_square_frac_l44_44496

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l44_44496


namespace complement_intersection_l44_44566

def setM : Set ℝ := { x | 2 / x < 1 }
def setN : Set ℝ := { y | ∃ x, y = Real.sqrt (x - 1) }

theorem complement_intersection 
  (R : Set ℝ) : ((R \ setM) ∩ setN = { y | 0 ≤ y ∧ y ≤ 2 }) :=
  sorry

end complement_intersection_l44_44566


namespace range_of_a_l44_44199

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 3 → deriv (f a) x ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end range_of_a_l44_44199


namespace new_person_weight_l44_44924

theorem new_person_weight (avg_increase : Real) (n : Nat) (old_weight : Real) (W_new : Real) :
  avg_increase = 2.5 → n = 8 → old_weight = 67 → W_new = old_weight + n * avg_increase → W_new = 87 :=
by
  intros avg_increase_eq n_eq old_weight_eq calc_eq
  sorry

end new_person_weight_l44_44924


namespace trip_is_400_miles_l44_44986

def fuel_per_mile_empty_plane := 20
def fuel_increase_per_person := 3
def fuel_increase_per_bag := 2
def number_of_passengers := 30
def number_of_crew := 5
def bags_per_person := 2
def total_fuel_needed := 106000

def fuel_consumption_per_mile :=
  fuel_per_mile_empty_plane +
  (number_of_passengers + number_of_crew) * fuel_increase_per_person +
  (number_of_passengers + number_of_crew) * bags_per_person * fuel_increase_per_bag

def trip_length := total_fuel_needed / fuel_consumption_per_mile

theorem trip_is_400_miles : trip_length = 400 := 
by sorry

end trip_is_400_miles_l44_44986


namespace decreasing_number_4312_max_decreasing_number_divisible_by_9_l44_44032

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end decreasing_number_4312_max_decreasing_number_divisible_by_9_l44_44032


namespace non_degenerate_triangles_l44_44868

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l44_44868


namespace heaviest_and_lightest_in_13_weighings_l44_44414

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l44_44414


namespace cos_triple_angle_l44_44181

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l44_44181


namespace pairs_satisfying_equation_l44_44309

theorem pairs_satisfying_equation (a b : ℝ) : 
  (∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ a = b ∨ ∃ k : ℤ, a = k ∧ b = k) := 
by
  sorry

end pairs_satisfying_equation_l44_44309
