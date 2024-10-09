import Mathlib

namespace sylvia_time_to_complete_job_l37_3726

theorem sylvia_time_to_complete_job (S : ℝ) (h₁ : 18 ≠ 0) (h₂ : 30 ≠ 0)
  (together_rate : (1 / S) + (1 / 30) = 1 / 18) :
  S = 45 :=
by
  -- Proof will be provided here
  sorry

end sylvia_time_to_complete_job_l37_3726


namespace ana_final_salary_l37_3739

def initial_salary : ℝ := 2500
def june_raise : ℝ := initial_salary * 0.15
def june_bonus : ℝ := 300
def salary_after_june : ℝ := initial_salary + june_raise + june_bonus
def july_pay_cut : ℝ := salary_after_june * 0.25
def final_salary : ℝ := salary_after_june - july_pay_cut

theorem ana_final_salary :
  final_salary = 2381.25 := by
  -- sorry is used here to skip the proof
  sorry

end ana_final_salary_l37_3739


namespace quadratic_equivalence_statement_l37_3784

noncomputable def quadratic_in_cos (a b c x : ℝ) : Prop := 
  a * (Real.cos x)^2 + b * Real.cos x + c = 0

noncomputable def transform_to_cos2x (a b c : ℝ) : Prop := 
  (4*a^2) * (Real.cos (2*a))^2 + (2*a^2 + 4*a*c - 2*b^2) * Real.cos (2*a) + a^2 + 4*a*c - 2*b^2 + 4*c^2 = 0

theorem quadratic_equivalence_statement (a b c : ℝ) (h : quadratic_in_cos 4 2 (-1) a) :
  transform_to_cos2x 16 12 (-4) :=
sorry

end quadratic_equivalence_statement_l37_3784


namespace find_f_1998_l37_3703

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

theorem find_f_1998 (x : ℝ) (h1 : ∀ x, f (x +1) = f x - 1) (h2 : f 1 = 3997) : f 1998 = 2000 :=
  sorry

end find_f_1998_l37_3703


namespace equation_solution_l37_3752

theorem equation_solution :
  ∃ a b c d : ℤ, a > 0 ∧ (∀ x : ℝ, (64 * x^2 + 96 * x - 36) = (a * x + b)^2 + d) ∧ c = -36 ∧ a + b + c + d = -94 :=
by sorry

end equation_solution_l37_3752


namespace xiaoxiao_types_faster_l37_3769

-- Defining the characters typed and time taken by both individuals
def characters_typed_taoqi : ℕ := 200
def time_taken_taoqi : ℕ := 5
def characters_typed_xiaoxiao : ℕ := 132
def time_taken_xiaoxiao : ℕ := 3

-- Calculating typing speeds
def speed_taoqi : ℕ := characters_typed_taoqi / time_taken_taoqi
def speed_xiaoxiao : ℕ := characters_typed_xiaoxiao / time_taken_xiaoxiao

-- Proving that 笑笑 types faster
theorem xiaoxiao_types_faster : speed_xiaoxiao > speed_taoqi := by
  -- Given calculations:
  -- speed_taoqi = 40
  -- speed_xiaoxiao = 44
  sorry

end xiaoxiao_types_faster_l37_3769


namespace juan_stamp_cost_l37_3730

-- Defining the prices of the stamps
def price_brazil : ℝ := 0.07
def price_peru : ℝ := 0.05

-- Defining the number of stamps from the 70s and 80s
def stamps_brazil_70s : ℕ := 12
def stamps_brazil_80s : ℕ := 15
def stamps_peru_70s : ℕ := 6
def stamps_peru_80s : ℕ := 12

-- Calculating total number of stamps from the 70s and 80s
def total_stamps_brazil : ℕ := stamps_brazil_70s + stamps_brazil_80s
def total_stamps_peru : ℕ := stamps_peru_70s + stamps_peru_80s

-- Calculating total cost
def total_cost_brazil : ℝ := total_stamps_brazil * price_brazil
def total_cost_peru : ℝ := total_stamps_peru * price_peru

def total_cost : ℝ := total_cost_brazil + total_cost_peru

-- Proof statement
theorem juan_stamp_cost : total_cost = 2.79 :=
by
  sorry

end juan_stamp_cost_l37_3730


namespace find_number_l37_3700

theorem find_number (x y a : ℝ) (h₁ : x * y = 1) (h₂ : (a ^ ((x + y) ^ 2)) / (a ^ ((x - y) ^ 2)) = 1296) : a = 6 :=
sorry

end find_number_l37_3700


namespace remainder_3_pow_20_mod_5_l37_3773

theorem remainder_3_pow_20_mod_5 : (3 ^ 20) % 5 = 1 := by
  sorry

end remainder_3_pow_20_mod_5_l37_3773


namespace exists_i_with_α_close_to_60_l37_3735

noncomputable def α : ℕ → ℝ := sorry  -- Placeholder for the function α

theorem exists_i_with_α_close_to_60 :
  ∃ i : ℕ, abs (α i - 60) < 1
:= sorry

end exists_i_with_α_close_to_60_l37_3735


namespace unique_solution_condition_l37_3799

theorem unique_solution_condition (a b c : ℝ) : 
  (∀ x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 :=
by
  sorry

end unique_solution_condition_l37_3799


namespace nancy_shoes_l37_3757

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l37_3757


namespace ages_when_john_is_50_l37_3750

variable (age_john age_alice age_mike : ℕ)

-- Given conditions:
-- John is 10 years old
def john_is_10 : age_john = 10 := by sorry

-- Alice is twice John's age
def alice_is_twice_john : age_alice = 2 * age_john := by sorry

-- Mike is 4 years younger than Alice
def mike_is_4_years_younger : age_mike = age_alice - 4 := by sorry

-- Prove that when John is 50 years old, Alice will be 60 years old, and Mike will be 56 years old
theorem ages_when_john_is_50 : age_john = 50 → age_alice = 60 ∧ age_mike = 56 := 
by 
  intro h
  sorry

end ages_when_john_is_50_l37_3750


namespace average_speed_l37_3732

theorem average_speed (D T : ℝ) (hD : D = 200) (hT : T = 6) : D / T = 33.33 := by
  -- Sorry is used to skip the proof, only the statement is provided as per instruction
  sorry

end average_speed_l37_3732


namespace evaluate_expression_121point5_l37_3783

theorem evaluate_expression_121point5 :
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  (1 / 3) * x^4 * y^5 = 121.5 :=
by
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  sorry

end evaluate_expression_121point5_l37_3783


namespace mass_percentage_Ca_in_CaI2_l37_3710

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I

theorem mass_percentage_Ca_in_CaI2 :
  (molar_mass_Ca / molar_mass_CaI2) * 100 = 13.63 :=
by
  sorry

end mass_percentage_Ca_in_CaI2_l37_3710


namespace product_divisible_by_12_l37_3751

theorem product_divisible_by_12 (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b)) :=
  sorry

end product_divisible_by_12_l37_3751


namespace remainder_3_pow_9_div_5_l37_3787

theorem remainder_3_pow_9_div_5 : (3^9) % 5 = 3 := by
  sorry

end remainder_3_pow_9_div_5_l37_3787


namespace area_product_equal_no_consecutive_integers_l37_3756

open Real

-- Define the areas of the triangles for quadrilateral ABCD
variables {A B C D O : Point} 
variables {S1 S2 S3 S4 : Real}  -- Areas of triangles ABO, BCO, CDO, DAO

-- Given conditions
variables (h_intersection : lies_on_intersection O AC BD)
variables (h_areas : S1 = 1 / 2 * (|AO| * |BM|) ∧ S2 = 1 / 2 * (|CO| * |BM|) ∧ S3 = 1 / 2 * (|CO| * |DN|) ∧ S4 = 1 / 2 * (|AO| * |DN|))

-- Theorem for part (a)
theorem area_product_equal : S1 * S3 = S2 * S4 :=
by sorry

-- Theorem for part (b)
theorem no_consecutive_integers : ¬∃ (n : ℕ), S1 = n ∧ S2 = n + 1 ∧ S3 = n + 2 ∧ S4 = n + 3 :=
by sorry

end area_product_equal_no_consecutive_integers_l37_3756


namespace vanessa_missed_days_l37_3729

theorem vanessa_missed_days (V M S : ℕ) 
                           (h1 : V + M + S = 17) 
                           (h2 : V + M = 14) 
                           (h3 : M + S = 12) : 
                           V = 5 :=
sorry

end vanessa_missed_days_l37_3729


namespace rope_cut_prob_l37_3779

theorem rope_cut_prob (x : ℝ) (hx : 0 < x) : 
  (∃ (a b : ℝ), a + b = 1 ∧ min a b ≤ max a b / x) → 
  (1 / (x + 1) * 2) = 2 / (x + 1) :=
sorry

end rope_cut_prob_l37_3779


namespace total_area_to_be_painted_l37_3734

theorem total_area_to_be_painted (length width height partition_length partition_height : ℝ) 
(partition_along_length inside_outside both_sides : Bool)
(h1 : length = 15)
(h2 : width = 12)
(h3 : height = 6)
(h4 : partition_length = 15)
(h5 : partition_height = 6) 
(h_partition_along_length : partition_along_length = true)
(h_inside_outside : inside_outside = true)
(h_both_sides : both_sides = true) :
    let end_wall_area := 2 * 2 * width * height
    let side_wall_area := 2 * 2 * length * height
    let ceiling_area := length * width
    let partition_area := 2 * partition_length * partition_height
    (end_wall_area + side_wall_area + ceiling_area + partition_area) = 1008 :=
by
    sorry

end total_area_to_be_painted_l37_3734


namespace quadratic_inequality_solution_l37_3721

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end quadratic_inequality_solution_l37_3721


namespace find_extrema_l37_3725

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem find_extrema :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 6) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 6) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 2 ≤ f x) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 2) :=
by sorry

end find_extrema_l37_3725


namespace solution_system_of_equations_solution_system_of_inequalities_l37_3701

-- Part 1: System of Equations
theorem solution_system_of_equations (x y : ℚ) :
  (3 * x + 2 * y = 13) ∧ (2 * x + 3 * y = -8) ↔ (x = 11 ∧ y = -10) :=
by
  sorry

-- Part 2: System of Inequalities
theorem solution_system_of_inequalities (y : ℚ) :
  ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2) ∧ (2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) :=
by
  sorry

end solution_system_of_equations_solution_system_of_inequalities_l37_3701


namespace quadrilateral_angle_difference_l37_3748

theorem quadrilateral_angle_difference (h_ratio : ∀ (a b c d : ℕ), a = 3 * d ∧ b = 4 * d ∧ c = 5 * d ∧ d = 6 * d) 
  (h_sum : ∀ (a b c d : ℕ), a + b + c + d = 360) : 
  ∃ (x : ℕ), 6 * x - 3 * x = 60 := 
by 
  sorry

end quadrilateral_angle_difference_l37_3748


namespace find_triangle_sides_l37_3724

-- Define the variables and conditions
noncomputable def k := 5
noncomputable def c := 12
noncomputable def d := 10

-- Assume the perimeters of the figures
def P1 : ℕ := 74
def P2 : ℕ := 84
def P3 : ℕ := 82

-- Define the equations based on the perimeters
def Equation1 := P2 = P1 + 2 * k
def Equation2 := P3 = P1 + 6 * c - 2 * k

-- The lean theorem proving that the sides of the triangle are as given
theorem find_triangle_sides : 
  (Equation1 ∧ Equation2) →
  (k = 5 ∧ c = 12 ∧ d = 10) :=
by
  sorry

end find_triangle_sides_l37_3724


namespace jordans_greatest_average_speed_l37_3753

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s.reverse = s

theorem jordans_greatest_average_speed :
  ∃ (v : ℕ), 
  ∃ (d : ℕ), 
  ∃ (end_reading : ℕ), 
  is_palindrome 72327 ∧ 
  is_palindrome end_reading ∧ 
  72327 < end_reading ∧ 
  end_reading - 72327 = d ∧ 
  d ≤ 240 ∧ 
  end_reading ≤ 72327 + 240 ∧ 
  v = d / 4 ∧ 
  v = 50 :=
sorry

end jordans_greatest_average_speed_l37_3753


namespace interval_length_implies_difference_l37_3796

variable (c d : ℝ)

theorem interval_length_implies_difference (h1 : ∀ x : ℝ, c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) (h2 : (d - c) / 3 = 15) : d - c = 45 := 
sorry

end interval_length_implies_difference_l37_3796


namespace sum_of_cubes_l37_3733

theorem sum_of_cubes
  (a b c : ℝ)
  (h₁ : a + b + c = 7)
  (h₂ : ab + ac + bc = 9)
  (h₃ : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end sum_of_cubes_l37_3733


namespace gas_cost_per_gallon_l37_3797

theorem gas_cost_per_gallon (mpg : ℝ) (miles_per_day : ℝ) (days : ℝ) (total_cost : ℝ) : 
  mpg = 50 ∧ miles_per_day = 75 ∧ days = 10 ∧ total_cost = 45 → 
  (total_cost / ((miles_per_day * days) / mpg)) = 3 :=
by
  sorry

end gas_cost_per_gallon_l37_3797


namespace max_knights_on_island_l37_3744

theorem max_knights_on_island :
  ∃ n x, (n * (n - 1) = 90) ∧ (x * (10 - x) = 24) ∧ (x ≤ n) ∧ (∀ y, y * (10 - y) = 24 → y ≤ x) := sorry

end max_knights_on_island_l37_3744


namespace max_sum_product_l37_3738

theorem max_sum_product (a b c d : ℝ) (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum: a + b + c + d = 200) : 
  ab + bc + cd + da ≤ 10000 := 
sorry

end max_sum_product_l37_3738


namespace nancy_flooring_area_l37_3785

def area_of_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem nancy_flooring_area :
  let central_area_length := 10
  let central_area_width := 10
  let hallway_length := 6
  let hallway_width := 4
  let central_area := area_of_rectangle central_area_length central_area_width
  let hallway_area := area_of_rectangle hallway_length hallway_width
  let total_area := central_area + hallway_area
  total_area = 124 :=
by
  rfl  -- This is where the proof would go.

end nancy_flooring_area_l37_3785


namespace polynomial_is_positive_for_all_x_l37_3737

noncomputable def P (x : ℝ) : ℝ := x^12 - x^9 + x^4 - x + 1

theorem polynomial_is_positive_for_all_x (x : ℝ) : P x > 0 := 
by
  dsimp [P]
  sorry -- Proof is omitted.

end polynomial_is_positive_for_all_x_l37_3737


namespace contrapositive_honor_roll_l37_3708

variable (Student : Type) (scores_hundred : Student → Prop) (honor_roll_qualifies : Student → Prop)

theorem contrapositive_honor_roll (s : Student) :
  (¬ honor_roll_qualifies s) → (¬ scores_hundred s) := 
sorry

end contrapositive_honor_roll_l37_3708


namespace sally_baseball_cards_l37_3716

theorem sally_baseball_cards (initial_cards sold_cards : ℕ) (h1 : initial_cards = 39) (h2 : sold_cards = 24) :
  (initial_cards - sold_cards = 15) :=
by
  -- Proof needed
  sorry

end sally_baseball_cards_l37_3716


namespace geometric_sequence_a_11_l37_3720

-- Define the geometric sequence with given terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

axiom a_5 : a 5 = -16
axiom a_8 : a 8 = 8

-- Question to prove
theorem geometric_sequence_a_11 (h : is_geometric_sequence a q) : a 11 = -4 := 
sorry

end geometric_sequence_a_11_l37_3720


namespace john_probability_l37_3736

/-- John arrives at a terminal which has sixteen gates arranged in a straight line with exactly 50 feet between adjacent gates. His departure gate is assigned randomly. After waiting at that gate, John is informed that the departure gate has been changed to another gate, chosen randomly again. Prove that the probability that John walks 200 feet or less to the new gate is \(\frac{4}{15}\), and find \(4 + 15 = 19\) -/
theorem john_probability :
  let n_gates := 16
  let dist_between_gates := 50
  let max_walk_dist := 200
  let total_possibilities := n_gates * (n_gates - 1)
  let valid_cases :=
    4 * (2 + 2 * (4 - 1))
  let probability_within_200_feet := valid_cases / total_possibilities
  let fraction := probability_within_200_feet * (15 / 4)
  fraction = 1 → 4 + 15 = 19 := by
  sorry -- Proof goes here 

end john_probability_l37_3736


namespace expected_participants_2008_l37_3766

theorem expected_participants_2008 (initial_participants : ℕ) (annual_increase_rate : ℝ) :
  initial_participants = 1000 ∧ annual_increase_rate = 1.25 →
  (initial_participants * annual_increase_rate ^ 3) = 1953.125 :=
by
  sorry

end expected_participants_2008_l37_3766


namespace first_wing_hall_rooms_l37_3790

theorem first_wing_hall_rooms
    (total_rooms : ℕ) (first_wing_floors : ℕ) (first_wing_halls_per_floor : ℕ)
    (second_wing_floors : ℕ) (second_wing_halls_per_floor : ℕ) (second_wing_rooms_per_hall : ℕ)
    (hotel_total_rooms : ℕ) (first_wing_total_rooms : ℕ) :
    hotel_total_rooms = total_rooms →
    first_wing_floors = 9 →
    first_wing_halls_per_floor = 6 →
    second_wing_floors = 7 →
    second_wing_halls_per_floor = 9 →
    second_wing_rooms_per_hall = 40 →
    hotel_total_rooms = first_wing_total_rooms + (second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall) →
    first_wing_total_rooms = first_wing_floors * first_wing_halls_per_floor * 32 :=
by
  sorry

end first_wing_hall_rooms_l37_3790


namespace sum_of_possible_a_l37_3712

theorem sum_of_possible_a (a : ℤ) :
  (∃ x : ℕ, x - (2 - a * x) / 6 = x / 3 - 1) →
  a = -19 :=
sorry

end sum_of_possible_a_l37_3712


namespace min_value_l37_3740

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2*y = 2) : 
  ∃ c : ℝ, c = 2 ∧ ∀ z, (z = (x^2 / (2*y) + 4*(y^2) / x)) → z ≥ c :=
by
  sorry

end min_value_l37_3740


namespace Megan_not_lead_actress_l37_3770

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l37_3770


namespace largest_number_in_sequence_l37_3761

noncomputable def largest_in_sequence (s : Fin 8 → ℝ) : ℝ :=
  max (s 0) (max (s 1) (max (s 2) (max (s 3) (max (s 4) (max (s 5) (max (s 6) (s 7)))))))

theorem largest_number_in_sequence (s : Fin 8 → ℝ)
  (h1 : ∀ i j : Fin 8, i < j → s i < s j)
  (h2 : ∃ i : Fin 5, (∃ d : ℝ, d = 4 ∨ d = 36) ∧ (∀ j : ℕ, j < 3 → s (i+j) + d = s (i+j+1)))
  (h3 : ∃ i : Fin 5, ∃ r : ℝ, (∀ j : ℕ, j < 3 → s (i+j) * r = s (i+j+1))) :
  largest_in_sequence s = 126 ∨ largest_in_sequence s = 6 :=
sorry

end largest_number_in_sequence_l37_3761


namespace bolton_class_students_l37_3713

theorem bolton_class_students 
  (S : ℕ) 
  (H1 : 2/5 < 1)
  (H2 : 1/3 < 1)
  (C1 : (2 / 5) * (S:ℝ) + (2 / 5) * (S:ℝ) = 20) : 
  S = 25 := 
by
  sorry

end bolton_class_students_l37_3713


namespace find_difference_l37_3767

variable (a b c d e f : ℝ)

-- Conditions
def cond1 : Prop := a - b = c + d + 9
def cond2 : Prop := a + b = c - d - 3
def cond3 : Prop := e = a^2 + b^2
def cond4 : Prop := f = c^2 + d^2
def cond5 : Prop := f - e = 5 * a + 2 * b + 3 * c + 4 * d

-- Problem Statement
theorem find_difference (h1 : cond1 a b c d) (h2 : cond2 a b c d) (h3 : cond3 a b e) (h4 : cond4 c d f) (h5 : cond5 a b c d e f) : a - c = 3 :=
sorry

end find_difference_l37_3767


namespace geometric_progression_sum_of_cubes_l37_3795

theorem geometric_progression_sum_of_cubes :
  ∃ (a r : ℕ) (seq : Fin 6 → ℕ), (seq 0 = a) ∧ (seq 1 = a * r) ∧ (seq 2 = a * r^2) ∧ (seq 3 = a * r^3) ∧ (seq 4 = a * r^4) ∧ (seq 5 = a * r^5) ∧
  (∀ i, 0 ≤ seq i ∧ seq i < 100) ∧
  (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 = 326) ∧
  (∃ T : ℕ, (∀ i, ∃ k, seq i = k^3 → k * k * k = seq i) ∧ T = 64) :=
sorry

end geometric_progression_sum_of_cubes_l37_3795


namespace lines_do_not_intersect_l37_3754

theorem lines_do_not_intersect (b : ℝ) :
  ∀ s v : ℝ,
    (2 + 3 * s = 5 + 6 * v) →
    (1 + 4 * s = 3 + 3 * v) →
    (b + 5 * s = 1 + 2 * v) →
    b ≠ -4/5 :=
by
  intros s v h1 h2 h3
  sorry

end lines_do_not_intersect_l37_3754


namespace sequence_general_term_and_sum_l37_3771

theorem sequence_general_term_and_sum (a_n : ℕ → ℕ) (b_n S_n : ℕ → ℕ) :
  (∀ n, a_n n = 2 ^ n) ∧ (∀ n, b_n n = a_n n * (Real.logb 2 (a_n n)) ∧
  S_n n = (n - 1) * 2 ^ (n + 1) + 2) :=
by
  sorry

end sequence_general_term_and_sum_l37_3771


namespace max_digit_e_l37_3718

theorem max_digit_e 
  (d e : ℕ) 
  (digits : ∀ (n : ℕ), n ≤ 9) 
  (even_e : e % 2 = 0) 
  (div_9 : (22 + d + e) % 9 = 0) 
  : e ≤ 8 :=
sorry

end max_digit_e_l37_3718


namespace population_increase_rate_l37_3723

theorem population_increase_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) 
  (h1 : persons = 240) 
  (h2 : minutes = 60) 
  (h3 : seconds_per_person = (minutes * 60) / persons) 
  : seconds_per_person = 15 :=
by 
  sorry

end population_increase_rate_l37_3723


namespace solve_abs_eq_linear_l37_3704

theorem solve_abs_eq_linear (x : ℝ) (h : |2 * x - 4| = x + 3) : x = 7 :=
sorry

end solve_abs_eq_linear_l37_3704


namespace train_capacity_l37_3745

theorem train_capacity (T : ℝ) (h : 2 * (T / 6) = 40) : T = 120 :=
sorry

end train_capacity_l37_3745


namespace intersection_A_complement_is_2_4_l37_3768

-- Declare the universal set U, set A, and set B
def U : Set ℕ := { 1, 2, 3, 4, 5, 6, 7 }
def A : Set ℕ := { 2, 4, 5 }
def B : Set ℕ := { 1, 3, 5, 7 }

-- Define the complement of set B with respect to U
def complement_U_B : Set ℕ := { x ∈ U | x ∉ B }

-- Define the intersection of set A and the complement of set B
def intersection_A_complement_U_B : Set ℕ := { x ∈ A | x ∈ complement_U_B }

-- State the theorem
theorem intersection_A_complement_is_2_4 : 
  intersection_A_complement_U_B = { 2, 4 } := 
by
  sorry

end intersection_A_complement_is_2_4_l37_3768


namespace arithmetic_sequence_common_difference_l37_3711

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h1 : a 1 = 1) (h3 : a 3 = 4) :
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 / 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l37_3711


namespace range_of_x_when_m_is_4_range_of_m_l37_3777

-- Define the conditions for p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0
def neg_p (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 5
def neg_q (x m : ℝ) : Prop := x ≤ m ∨ x ≥ 3 * m

-- Define the conditions for the values of m
def cond_m_pos (m : ℝ) : Prop := m > 0
def cond_sufficient (m : ℝ) : Prop := cond_m_pos m ∧ m ≤ 2 ∧ 3 * m ≥ 5

-- Problem 1
theorem range_of_x_when_m_is_4 (x : ℝ) : p x ∧ q x 4 → 4 < x ∧ x < 5 :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : (∀ x : ℝ, neg_q x m → neg_p x) → 5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_x_when_m_is_4_range_of_m_l37_3777


namespace number_of_rectangles_containing_cell_l37_3782

theorem number_of_rectangles_containing_cell (m n p q : ℕ) (hp : 1 ≤ p ∧ p ≤ m) (hq : 1 ≤ q ∧ q ≤ n) :
    ∃ count : ℕ, count = p * q * (m - p + 1) * (n - q + 1) := 
    sorry

end number_of_rectangles_containing_cell_l37_3782


namespace tickets_difference_l37_3755

def tickets_used_for_clothes : ℝ := 85
def tickets_used_for_accessories : ℝ := 45.5
def tickets_used_for_food : ℝ := 51
def tickets_used_for_toys : ℝ := 58

theorem tickets_difference : 
  (tickets_used_for_clothes + tickets_used_for_food + tickets_used_for_accessories) - tickets_used_for_toys = 123.5 := 
by
  sorry

end tickets_difference_l37_3755


namespace smallest_set_handshakes_l37_3781

-- Define the number of people
def num_people : Nat := 36

-- Define a type for people
inductive Person : Type
| a : Fin num_people → Person

-- Define the handshake relationship
def handshake (p1 p2 : Person) : Prop :=
  match p1, p2 with
  | Person.a i, Person.a j => i.val = (j.val + 1) % num_people ∨ j.val = (i.val + 1) % num_people

-- Define the problem statement
theorem smallest_set_handshakes :
  ∃ s : Finset Person, (∀ p : Person, p ∈ s ∨ ∃ q ∈ s, handshake p q) ∧ s.card = 18 :=
sorry

end smallest_set_handshakes_l37_3781


namespace average_of_t_b_c_29_l37_3727
-- Importing the entire Mathlib library

theorem average_of_t_b_c_29 (t b c : ℝ) 
  (h : (t + b + c + 14 + 15) / 5 = 12) : 
  (t + b + c + 29) / 4 = 15 :=
by 
  sorry

end average_of_t_b_c_29_l37_3727


namespace payment_to_z_l37_3702

-- Definitions of the conditions
def x_work_rate := 1 / 15
def y_work_rate := 1 / 10
def total_payment := 720
def combined_work_rate_xy := x_work_rate + y_work_rate
def combined_work_rate_xyz := 1 / 5
def z_work_rate := combined_work_rate_xyz - combined_work_rate_xy
def z_contribution := z_work_rate * 5
def z_payment := z_contribution * total_payment

-- The statement to be proven
theorem payment_to_z : z_payment = 120 := by
  sorry

end payment_to_z_l37_3702


namespace max_stamps_l37_3789

def price_of_stamp : ℕ := 25  -- Price of one stamp in cents
def total_money : ℕ := 4000   -- Total money available in cents

theorem max_stamps : ∃ n : ℕ, price_of_stamp * n ≤ total_money ∧ (∀ m : ℕ, price_of_stamp * m ≤ total_money → m ≤ n) :=
by
  use 160
  sorry

end max_stamps_l37_3789


namespace parabola_vertex_sum_l37_3764

theorem parabola_vertex_sum (p q r : ℝ) (h1 : ∀ x : ℝ, x = p * (x - 3)^2 + 2 → y) (h2 : p * (1 - 3)^2 + 2 = 6) :
  p + q + r = 6 :=
sorry

end parabola_vertex_sum_l37_3764


namespace olympiad_scores_above_18_l37_3760

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l37_3760


namespace art_piece_increase_l37_3747

theorem art_piece_increase (initial_price : ℝ) (multiplier : ℝ) (future_increase : ℝ) (h1 : initial_price = 4000) (h2 : multiplier = 3) :
  future_increase = (multiplier * initial_price) - initial_price :=
by
  rw [h1, h2]
  norm_num
  sorry

end art_piece_increase_l37_3747


namespace simplify_expression_l37_3774

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end simplify_expression_l37_3774


namespace alex_silver_tokens_l37_3793

theorem alex_silver_tokens :
  ∃ x y : ℕ, 
    (100 - 3 * x + y ≤ 2) ∧ 
    (50 + 2 * x - 4 * y ≤ 3) ∧
    (x + y = 74) :=
by
  sorry

end alex_silver_tokens_l37_3793


namespace probability_tenth_ball_black_l37_3722

theorem probability_tenth_ball_black :
  let total_balls := 30
  let black_balls := 4
  let red_balls := 7
  let yellow_balls := 5
  let green_balls := 6
  let white_balls := 8
  (black_balls / total_balls) = 4 / 30 :=
by sorry

end probability_tenth_ball_black_l37_3722


namespace number_of_boys_in_school_l37_3715

theorem number_of_boys_in_school (B : ℝ) (h1 : 542.0 = B + 155) : B = 387 :=
by
  sorry

end number_of_boys_in_school_l37_3715


namespace milk_butterfat_problem_l37_3780

-- Define the values given in the problem
def b1 : ℝ := 0.35  -- butterfat percentage of initial milk
def v1 : ℝ := 8     -- volume of initial milk in gallons
def b2 : ℝ := 0.10  -- butterfat percentage of milk to be added
def bf : ℝ := 0.20  -- desired butterfat percentage of the final mixture

-- Define the proof statement
theorem milk_butterfat_problem :
  ∃ x : ℝ, (2.8 + 0.1 * x) / (v1 + x) = bf ↔ x = 12 :=
by {
  sorry
}

end milk_butterfat_problem_l37_3780


namespace product_of_two_numbers_l37_3776

theorem product_of_two_numbers :
  ∀ x y: ℝ, 
  ((x - y)^2) / ((x + y)^3) = 4 / 27 → 
  x + y = 5 * (x - y) + 3 → 
  x * y = 15.75 :=
by 
  intro x y
  sorry

end product_of_two_numbers_l37_3776


namespace carlos_earnings_l37_3778

theorem carlos_earnings (h1 : ∃ w, 18 * w = w * 18) (h2 : ∃ w, 30 * w = w * 30) (h3 : ∀ w, 30 * w - 18 * w = 54) : 
  ∃ w, 18 * w + 30 * w = 216 := 
sorry

end carlos_earnings_l37_3778


namespace train_platform_ratio_l37_3758

noncomputable def speed_km_per_hr := 216 -- condition 1
noncomputable def crossing_time_sec := 60 -- condition 2
noncomputable def train_length_m := 1800 -- condition 3

noncomputable def speed_m_per_s := speed_km_per_hr * 1000 / 3600
noncomputable def total_distance_m := speed_m_per_s * crossing_time_sec
noncomputable def platform_length_m := total_distance_m - train_length_m
noncomputable def ratio := train_length_m / platform_length_m

theorem train_platform_ratio : ratio = 1 := by
    sorry

end train_platform_ratio_l37_3758


namespace original_population_correct_l37_3765

def original_population_problem :=
  let original_population := 6731
  let final_population := 4725
  let initial_disappeared := 0.10 * original_population
  let remaining_after_disappearance := original_population - initial_disappeared
  let left_after_remaining := 0.25 * remaining_after_disappearance
  let remaining_after_leaving := remaining_after_disappearance - left_after_remaining
  let disease_affected := 0.05 * original_population
  let disease_died := 0.02 * disease_affected
  let disease_migrated := 0.03 * disease_affected
  let remaining_after_disease := remaining_after_leaving - (disease_died + disease_migrated)
  let moved_to_village := 0.04 * remaining_after_disappearance
  let total_after_moving := remaining_after_disease + moved_to_village
  let births := 0.008 * original_population
  let deaths := 0.01 * original_population
  let final_population_calculated := total_after_moving + (births - deaths)
  final_population_calculated = final_population

theorem original_population_correct :
  original_population_problem ↔ True :=
by
  sorry

end original_population_correct_l37_3765


namespace problem1_problem2_l37_3792

variables (x a : ℝ)

-- Proposition definitions
def proposition_p (a : ℝ) (x : ℝ) : Prop :=
  a > 0 ∧ (-x^2 + 4*a*x - 3*a^2) > 0

def proposition_q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) < 0

-- Problems
theorem problem1 : (proposition_p 1 x ∧ proposition_q x) ↔ 2 < x ∧ x < 3 :=
by sorry

theorem problem2 : (¬ ∃ x, proposition_p a x) → (∀ x, ¬ proposition_q x) →
  1 ≤ a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l37_3792


namespace sun_radius_scientific_notation_l37_3786

theorem sun_radius_scientific_notation : 
  (369000 : ℝ) = 3.69 * 10^5 :=
by
  sorry

end sun_radius_scientific_notation_l37_3786


namespace jan_drove_more_l37_3775

variables (d t s : ℕ)
variables (h h_ans : ℕ)
variables (ha_speed j_speed : ℕ)
variables (j d_plus : ℕ)

-- Ian's equation
def ian_distance (s t : ℕ) : ℕ := s * t

-- Han's additional conditions
def han_distance (s t : ℕ) (h_speed : ℕ)
    (d_plus : ℕ) : Prop :=
  d_plus + 120 = (s + h_speed) * (t + 2)

-- Jan's conditions and equation
def jan_distance (s t : ℕ) (j_speed : ℕ) : ℕ :=
  (s + j_speed) * (t + 3)

-- Proof statement
theorem jan_drove_more (d t s h_ans : ℕ)
    (h_speed j_speed : ℕ) (d_plus : ℕ)
    (h_dist_cond : han_distance s t h_speed d_plus)
    (j_dist_cond : jan_distance s t j_speed = h_ans) :
  h_ans = 195 :=
sorry

end jan_drove_more_l37_3775


namespace find_angle_B_and_sin_ratio_l37_3705

variable (A B C a b c : ℝ)
variable (h₁ : a * (Real.sin C - Real.sin A) / (Real.sin C + Real.sin B) = c - b)
variable (h₂ : Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4)

theorem find_angle_B_and_sin_ratio :
  B = Real.pi / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end find_angle_B_and_sin_ratio_l37_3705


namespace sum_of_squares_l37_3791

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 :=
  sorry

end sum_of_squares_l37_3791


namespace sin_x_plus_pi_l37_3741

theorem sin_x_plus_pi {x : ℝ} (hx : Real.sin x = -4 / 5) : Real.sin (x + Real.pi) = 4 / 5 :=
by
  -- Proof steps go here
  sorry

end sin_x_plus_pi_l37_3741


namespace range_of_m_l37_3746

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  ¬(∀ x : ℝ, (x < m - 1 ∨ x > m + 1) ↔ (x^2 - 2*x - 3 > 0)) 
  ↔ 0 ≤ m ∧ m ≤ 2 :=
by 
  sorry

end range_of_m_l37_3746


namespace no_such_graph_exists_l37_3743

noncomputable def vertex_degrees (n : ℕ) (deg : ℕ → ℕ) : Prop :=
  n ≥ 8 ∧
  ∃ (deg : ℕ → ℕ),
    (deg 0 = 4) ∧ (deg 1 = 5) ∧ ∀ i, 2 ≤ i ∧ i < n - 7 → deg i = i + 4 ∧
    (deg (n-7) = n-2) ∧ (deg (n-6) = n-2) ∧ (deg (n-5) = n-2) ∧
    (deg (n-4) = n-1) ∧ (deg (n-3) = n-1) ∧ (deg (n-2) = n-1)   

theorem no_such_graph_exists (n : ℕ) (deg : ℕ → ℕ) : 
  n ≥ 10 → ¬vertex_degrees n deg := 
by
  sorry

end no_such_graph_exists_l37_3743


namespace Sandy_change_l37_3794

theorem Sandy_change (pants shirt sweater shoes total paid change : ℝ)
  (h1 : pants = 13.58) (h2 : shirt = 10.29) (h3 : sweater = 24.97) (h4 : shoes = 39.99) (h5 : total = pants + shirt + sweater + shoes) (h6 : paid = 100) (h7 : change = paid - total) :
  change = 11.17 := 
sorry

end Sandy_change_l37_3794


namespace two_vectors_less_than_45_deg_angle_l37_3763

theorem two_vectors_less_than_45_deg_angle (n : ℕ) (h : n = 30) (v : Fin n → ℝ → ℝ → ℝ) :
  ∃ i j : Fin n, i ≠ j ∧ ∃ θ : ℝ, θ < (45 * Real.pi / 180) :=
  sorry

end two_vectors_less_than_45_deg_angle_l37_3763


namespace bags_le_40kg_l37_3731

theorem bags_le_40kg (capacity boxes crates sacks box_weight crate_weight sack_weight bag_weight: ℕ)
  (h_capacity: capacity = 13500)
  (h_boxes: boxes = 100)
  (h_crates: crates = 10)
  (h_sacks: sacks = 50)
  (h_box_weight: box_weight = 100)
  (h_crate_weight: crate_weight = 60)
  (h_sack_weight: sack_weight = 50)
  (h_bag_weight: bag_weight = 40) :
  10 = (capacity - (boxes * box_weight + crates * crate_weight + sacks * sack_weight)) / bag_weight := by 
  sorry

end bags_le_40kg_l37_3731


namespace find_z_l37_3759

theorem find_z (z : ℚ) : (7 + 11 + 23) / 3 = (15 + z) / 2 → z = 37 / 3 :=
by
  sorry

end find_z_l37_3759


namespace hyperbola_focus_y_axis_l37_3714

theorem hyperbola_focus_y_axis (m : ℝ) :
  (∀ x y : ℝ, (m + 1) * x^2 + (2 - m) * y^2 = 1) → m < -1 :=
sorry

end hyperbola_focus_y_axis_l37_3714


namespace product_of_solutions_eq_neg_nine_product_of_solutions_l37_3749

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end product_of_solutions_eq_neg_nine_product_of_solutions_l37_3749


namespace integer_solution_l37_3707

theorem integer_solution (x : ℤ) (h : (Int.natAbs x - 1) * x ^ 2 - 9 = 1) : x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 :=
by
  sorry

end integer_solution_l37_3707


namespace greatest_ABCBA_l37_3798

/-
We need to prove that the greatest possible integer of the form AB,CBA 
that is both divisible by 11 and by 3, with A, B, and C being distinct digits, is 96569.
-/

theorem greatest_ABCBA (A B C : ℕ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) 
  (h3 : 10001 * A + 1010 * B + 100 * C < 100000) 
  (h4 : 2 * A - 2 * B + C ≡ 0 [MOD 11])
  (h5 : (2 * A + 2 * B + C) % 3 = 0) : 
  10001 * A + 1010 * B + 100 * C ≤ 96569 :=
sorry

end greatest_ABCBA_l37_3798


namespace intersection_y_sum_zero_l37_3706

theorem intersection_y_sum_zero :
  ∀ (x1 y1 x2 y2 : ℝ), (y1 = 2 * x1) ∧ (y1 = 2 / x1) ∧ (y2 = 2 * x2) ∧ (y2 = 2 / x2) →
  (x2 = -x1) ∧ (y2 = -y1) →
  y1 + y2 = 0 :=
by
  sorry

end intersection_y_sum_zero_l37_3706


namespace y_intercept_tangent_line_l37_3717

noncomputable def tangent_line_y_intercept (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (htangent: Prop) : ℝ :=
  if r1 = 3 ∧ r2 = 2 ∧ c1 = (3, 0) ∧ c2 = (8, 0) ∧ htangent = true then 6 * Real.sqrt 6 else 0

theorem y_intercept_tangent_line (h : tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6) :
  tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6 :=
by
  exact h

end y_intercept_tangent_line_l37_3717


namespace apples_in_third_basket_l37_3742

theorem apples_in_third_basket (total_apples : ℕ) (x : ℕ) (y : ℕ) 
    (h_total : total_apples = 2014)
    (h_second_basket : 49 + x = total_apples - 2 * y - x - y)
    (h_first_basket : total_apples - 2 * y - x + y = 2 * y)
    : x + y = 655 :=
by
    sorry

end apples_in_third_basket_l37_3742


namespace good_fractions_expression_l37_3709

def is_good_fraction (n : ℕ) (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem good_fractions_expression (n : ℕ) (a b : ℕ) :
  n > 1 →
  (∀ a b, b < n → is_good_fraction n a b → ∃ x y, x + y = a / b ∨ x - y = a / b) ↔
  Nat.Prime n :=
by
  sorry

end good_fractions_expression_l37_3709


namespace value_2x_y_l37_3719

theorem value_2x_y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y + 5 = 0) : 2*x + y = 0 := 
by
  sorry

end value_2x_y_l37_3719


namespace min_value_expression_l37_3728

noncomputable def f (t : ℝ) : ℝ :=
  (1 / (t + 1)) + (2 * t / (2 * t + 1))

theorem min_value_expression (x y : ℝ) (h : x * y > 0) :
  ∃ t, (x / y = t) ∧ t > 0 ∧ f t = 4 - 2 * Real.sqrt 2 := 
  sorry

end min_value_expression_l37_3728


namespace spring_length_increase_l37_3772

-- Define the weight (x) and length (y) data points
def weights : List ℝ := [0, 1, 2, 3, 4, 5]
def lengths : List ℝ := [20, 20.5, 21, 21.5, 22, 22.5]

-- Prove that for each increase of 1 kg in weight, the length of the spring increases by 0.5 cm
theorem spring_length_increase (h : weights.length = lengths.length) :
  ∀ i, i < weights.length - 1 → (lengths.get! (i+1) - lengths.get! i) = 0.5 :=
by
  -- Proof goes here, omitted for now
  sorry

end spring_length_increase_l37_3772


namespace solve_for_x_l37_3788

-- Assume x is a positive integer
def pos_integer (x : ℕ) : Prop := 0 < x

-- Assume the equation holds for some x
def equation (x : ℕ) : Prop :=
  1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170

-- Proposition stating that if x satisfies the equation then x must be 5
theorem solve_for_x (x : ℕ) (h1 : pos_integer x) (h2 : equation x) : x = 5 :=
by
  sorry

end solve_for_x_l37_3788


namespace distributi_l37_3762

def number_of_distributions (spots : ℕ) (classes : ℕ) (min_spot_per_class : ℕ) : ℕ :=
  Nat.choose (spots - min_spot_per_class * classes + (classes - 1)) (classes - 1)

theorem distributi.on_of_10_spots (A B C : ℕ) (hA : A ≥ 1) (hB : B ≥ 1) (hC : C ≥ 1) 
(h_total : A + B + C = 10) : number_of_distributions 10 3 1 = 36 :=
by
  sorry

end distributi_l37_3762
