import Mathlib

namespace min_value_fraction_l1849_184942

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ (∀ (x : ℝ) (hx : x = 1 / a + 1 / b), x ≥ m) := 
by
  sorry

end min_value_fraction_l1849_184942


namespace oranges_worth_as_much_as_bananas_l1849_184974

-- Define the given conditions
def worth_same_bananas_oranges (bananas oranges : ℕ) : Prop :=
  (3 / 4 * 12 : ℝ) = 9 ∧ 9 = 6

/-- Prove how many oranges are worth as much as (2 / 3) * 9 bananas,
    given that (3 / 4) * 12 bananas are worth 6 oranges. -/
theorem oranges_worth_as_much_as_bananas :
  worth_same_bananas_oranges 12 6 →
  (2 / 3 * 9 : ℝ) = 4 :=
by
  sorry

end oranges_worth_as_much_as_bananas_l1849_184974


namespace mystical_swamp_l1849_184961

/-- 
In a mystical swamp, there are two species of talking amphibians: toads, whose statements are always true, and frogs, whose statements are always false. 
Five amphibians: Adam, Ben, Cara, Dan, and Eva make the following statements:
Adam: "Eva and I are different species."
Ben: "Cara is a frog."
Cara: "Dan is a frog."
Dan: "Of the five of us, at least three are toads."
Eva: "Adam is a toad."
Given these statements, prove that the number of frogs is 3.
-/
theorem mystical_swamp :
  (∀ α β : Prop, α ∨ ¬β) ∧ -- Adam's statement: "Eva and I are different species."
  (Cara = "frog") ∧          -- Ben's statement: "Cara is a frog."
  (Dan = "frog") ∧         -- Cara's statement: "Dan is a frog."
  (∃ t, t = nat → t ≥ 3) ∧ -- Dan's statement: "Of the five of us, at least three are toads."
  (Adam = "toad")               -- Eva's statement: "Adam is a toad."
  → num_frogs = 3 := sorry       -- Number of frogs is 3.

end mystical_swamp_l1849_184961


namespace grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l1849_184949

-- Definition of the conditions
def grandfather_age (gm_age dad_age : ℕ) := gm_age = 2 * dad_age
def dad_age_eight_times_xiaoming (dad_age xm_age : ℕ) := dad_age = 8 * xm_age
def grandfather_age_61 (gm_age : ℕ) := gm_age = 61
def twenty_times_xiaoming (gm_age xm_age : ℕ) := gm_age = 20 * xm_age

-- Question 1: Proof that Grandpa is 57 years older than Xiaoming 
theorem grandfather_older_than_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : grandfather_age gm_age dad_age) (h2 : dad_age_eight_times_xiaoming dad_age xm_age)
  (h3 : grandfather_age_61 gm_age)
  : gm_age - xm_age = 57 := 
sorry

-- Question 2: Proof that Dad is 31 years old when Grandpa's age is twenty times Xiaoming's age
theorem dad_age_when_twenty_times_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : twenty_times_xiaoming gm_age xm_age)
  (hm : grandfather_age gm_age dad_age)
  : dad_age = 31 :=
sorry

end grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l1849_184949


namespace circle_through_ABC_l1849_184931

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (1, 4)

-- Define the circle equation components to be proved
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3*y - 3 = 0

-- The theorem statement that we need to prove
theorem circle_through_ABC : 
  ∃ (D E F : ℝ), (∀ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → x^2 + y^2 + D*x + E*y + F = 0) 
  → circle_eqn x y :=
sorry

end circle_through_ABC_l1849_184931


namespace increase_in_tire_radius_l1849_184989

theorem increase_in_tire_radius
  (r : ℝ)
  (d1 d2 : ℝ)
  (conv_factor : ℝ)
  (original_radius : r = 16)
  (odometer_reading_outbound : d1 = 500)
  (odometer_reading_return : d2 = 485)
  (conversion_factor : conv_factor = 63360) :
  ∃ Δr : ℝ, Δr = 0.33 :=
by
  sorry

end increase_in_tire_radius_l1849_184989


namespace total_money_raised_l1849_184998

def tickets_sold : ℕ := 25
def price_per_ticket : ℝ := 2.0
def donation_count : ℕ := 2
def donation_amount : ℝ := 15.0
def additional_donation : ℝ := 20.0

theorem total_money_raised :
  (tickets_sold * price_per_ticket) + (donation_count * donation_amount) + additional_donation = 100 :=
by
  sorry

end total_money_raised_l1849_184998


namespace salary_percentage_difference_l1849_184980

theorem salary_percentage_difference (A B : ℝ) (h : A = 0.8 * B) :
  (B - A) / A * 100 = 25 :=
sorry

end salary_percentage_difference_l1849_184980


namespace probability_of_other_girl_l1849_184922

theorem probability_of_other_girl (A B : Prop) (P : Prop → ℝ) 
    (hA : P A = 3 / 4) 
    (hAB : P (A ∧ B) = 1 / 4) : 
    P (B ∧ A) / P A = 1 / 3 := by 
  -- The proof is skipped using the sorry keyword.
  sorry

end probability_of_other_girl_l1849_184922


namespace find_kgs_of_apples_l1849_184975

def cost_of_apples_per_kg : ℝ := 2
def num_packs_of_sugar : ℝ := 3
def cost_of_sugar_per_pack : ℝ := cost_of_apples_per_kg - 1
def weight_walnuts_kg : ℝ := 0.5
def cost_of_walnuts_per_kg : ℝ := 6
def cost_of_walnuts : ℝ := cost_of_walnuts_per_kg * weight_walnuts_kg
def total_cost : ℝ := 16

theorem find_kgs_of_apples (A : ℝ) :
  2 * A + (num_packs_of_sugar * cost_of_sugar_per_pack) + cost_of_walnuts = total_cost →
  A = 5 :=
by
  sorry

end find_kgs_of_apples_l1849_184975


namespace oldest_child_age_l1849_184978

open Nat

def avg_age (a b c d : ℕ) := (a + b + c + d) / 4

theorem oldest_child_age 
  (h_avg : avg_age 5 8 11 x = 9) : x = 12 :=
by
  sorry

end oldest_child_age_l1849_184978


namespace stratified_sampling_BA3_count_l1849_184999

-- Defining the problem parameters
def num_Om_BA1 : ℕ := 60
def num_Om_BA2 : ℕ := 20
def num_Om_BA3 : ℕ := 40
def total_sample_size : ℕ := 30

-- Proving using stratified sampling
theorem stratified_sampling_BA3_count : 
  (total_sample_size * num_Om_BA3 / (num_Om_BA1 + num_Om_BA2 + num_Om_BA3)) = 10 :=
by
  -- Since Lean doesn't handle reals and integers simplistically,
  -- we need to translate the division and multiplication properly.
  sorry

end stratified_sampling_BA3_count_l1849_184999


namespace general_admission_tickets_l1849_184956

-- Define the number of student tickets and general admission tickets
variables {S G : ℕ}

-- Define the conditions
def tickets_sold (S G : ℕ) : Prop := S + G = 525
def amount_collected (S G : ℕ) : Prop := 4 * S + 6 * G = 2876

-- The theorem to prove that the number of general admission tickets is 388
theorem general_admission_tickets : 
  ∀ (S G : ℕ), tickets_sold S G → amount_collected S G → G = 388 :=
by
  sorry -- Proof to be provided

end general_admission_tickets_l1849_184956


namespace cos_double_angle_l1849_184945

theorem cos_double_angle (α : ℝ) (h : Real.sin α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = 1 / 3 :=
by
  sorry

end cos_double_angle_l1849_184945


namespace bike_owners_without_car_l1849_184919

variable (T B C : ℕ) (H1 : T = 500) (H2 : B = 450) (H3 : C = 200)

theorem bike_owners_without_car (total bike_owners car_owners : ℕ) 
  (h_total : total = 500) (h_bike_owners : bike_owners = 450) (h_car_owners : car_owners = 200) : 
  (bike_owners - (bike_owners + car_owners - total)) = 300 := by
  sorry

end bike_owners_without_car_l1849_184919


namespace calculate_tax_l1849_184912

noncomputable def cadastral_value : ℝ := 3000000 -- 3 million rubles
noncomputable def tax_rate : ℝ := 0.001        -- 0.1% converted to decimal
noncomputable def tax : ℝ := cadastral_value * tax_rate -- Tax formula

theorem calculate_tax : tax = 3000 := by
  sorry

end calculate_tax_l1849_184912


namespace binary_operation_l1849_184934

def b11001 := 25  -- binary 11001 is 25 in decimal
def b1101 := 13   -- binary 1101 is 13 in decimal
def b101 := 5     -- binary 101 is 5 in decimal
def b100111010 := 314 -- binary 100111010 is 314 in decimal

theorem binary_operation : (b11001 * b1101 - b101) = b100111010 := by
  -- provide implementation details to prove the theorem
  sorry

end binary_operation_l1849_184934


namespace mary_max_earnings_l1849_184908

def regular_rate : ℝ := 8
def max_hours : ℝ := 60
def regular_hours : ℝ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate
def overtime_hours : ℝ := max_hours - regular_hours
def earnings_regular : ℝ := regular_hours * regular_rate
def earnings_overtime : ℝ := overtime_hours * overtime_rate
def total_earnings : ℝ := earnings_regular + earnings_overtime

theorem mary_max_earnings : total_earnings = 560 := by
  sorry

end mary_max_earnings_l1849_184908


namespace probability_at_least_four_girls_l1849_184940

noncomputable def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_least_four_girls
  (n : ℕ)
  (p : ℝ)
  (q : ℝ)
  (h_pq : p + q = 1)
  (h_p : p = 0.55)
  (h_q : q = 0.45)
  (h_n : n = 7) :
  (binomial_probability n 4 p) + (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) = 0.59197745 :=
sorry

end probability_at_least_four_girls_l1849_184940


namespace area_inside_rectangle_outside_circles_is_4_l1849_184955

-- Specify the problem in Lean 4
theorem area_inside_rectangle_outside_circles_is_4 :
  let CD := 3
  let DA := 5
  let radius_A := 1
  let radius_B := 2
  let radius_C := 3
  let area_rectangle := CD * DA
  let area_circles := (radius_A^2 + radius_B^2 + radius_C^2) * Real.pi / 4
  abs (area_rectangle - area_circles - 4) < 1 :=
by
  repeat { sorry }

end area_inside_rectangle_outside_circles_is_4_l1849_184955


namespace tabby_swimming_speed_l1849_184914

theorem tabby_swimming_speed :
  ∃ (S : ℝ), S = 4.125 ∧ (∀ (D : ℝ), 6 = (2 * D) / ((D / S) + (D / 11))) :=
by {
 sorry
}

end tabby_swimming_speed_l1849_184914


namespace vectors_perpendicular_l1849_184983

open Real

def vector := ℝ × ℝ

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector) : Prop :=
  dot_product v w = 0

def vector_sub (v w : vector) : vector :=
  (v.1 - w.1, v.2 - w.2)

theorem vectors_perpendicular :
  let a : vector := (2, 0)
  let b : vector := (1, 1)
  perpendicular (vector_sub a b) b :=
by
  sorry

end vectors_perpendicular_l1849_184983


namespace Eva_arts_marks_difference_l1849_184923

noncomputable def marks_difference_in_arts : ℕ := 
  let M1 := 90
  let A2 := 90
  let S1 := 60
  let M2 := 80
  let A1 := A2 - 75
  let S2 := 90
  A2 - A1

theorem Eva_arts_marks_difference : marks_difference_in_arts = 75 := by
  sorry

end Eva_arts_marks_difference_l1849_184923


namespace vector_triangle_c_solution_l1849_184987

theorem vector_triangle_c_solution :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, 4)
  let c : ℝ × ℝ := (4, -6)
  (4 • a + (3 • b - 2 • a) + c = (0, 0)) →
  c = (4, -6) :=
by
  intro h
  sorry

end vector_triangle_c_solution_l1849_184987


namespace right_triangle_inequality_l1849_184907

variable (a b c : ℝ)

theorem right_triangle_inequality
  (h1 : b < a) -- shorter leg is less than longer leg
  (h2 : c = Real.sqrt (a^2 + b^2)) -- hypotenuse from Pythagorean theorem
  : a + b / 2 > c ∧ c > (8 / 9) * (a + b / 2) := 
sorry

end right_triangle_inequality_l1849_184907


namespace base9_to_decimal_unique_solution_l1849_184900

theorem base9_to_decimal_unique_solution :
  ∃ m : ℕ, 1 * 9^4 + 6 * 9^3 + m * 9^2 + 2 * 9^1 + 7 = 11203 ∧ m = 3 :=
by
  sorry

end base9_to_decimal_unique_solution_l1849_184900


namespace find_a_if_odd_function_l1849_184997

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + Real.sqrt (a + x^2))

theorem find_a_if_odd_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - f x a) → a = 1 :=
by
  sorry

end find_a_if_odd_function_l1849_184997


namespace total_hangers_l1849_184962

theorem total_hangers (pink green blue yellow orange purple red : ℕ) 
  (h_pink : pink = 7)
  (h_green : green = 4)
  (h_blue : blue = green - 1)
  (h_yellow : yellow = blue - 1)
  (h_orange : orange = 2 * pink)
  (h_purple : purple = yellow + 3)
  (h_red : red = purple / 2) :
  pink + green + blue + yellow + orange + purple + red = 37 :=
sorry

end total_hangers_l1849_184962


namespace total_carrots_l1849_184953

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end total_carrots_l1849_184953


namespace sum_and_ratio_implies_difference_l1849_184963

theorem sum_and_ratio_implies_difference (a b : ℚ) (h1 : a + b = 500) (h2 : a / b = 0.8) : b - a = 55.55555555555556 := by
  sorry

end sum_and_ratio_implies_difference_l1849_184963


namespace time_to_clear_l1849_184929

def length_train1 := 121 -- in meters
def length_train2 := 153 -- in meters
def speed_train1 := 80 * 1000 / 3600 -- converting km/h to meters/s
def speed_train2 := 65 * 1000 / 3600 -- converting km/h to meters/s

def total_distance := length_train1 + length_train2
def relative_speed := speed_train1 + speed_train2

theorem time_to_clear : 
  (total_distance / relative_speed : ℝ) = 6.80 :=
by
  sorry

end time_to_clear_l1849_184929


namespace fee_difference_l1849_184918

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end fee_difference_l1849_184918


namespace determine_base_l1849_184903

theorem determine_base (r : ℕ) (a b x : ℕ) (h₁ : r ≤ 100) 
  (h₂ : x = a * r + a) (h₃ : a < r) (h₄ : a > 0) 
  (h₅ : x^2 = b * r^3 + b) : r = 2 ∨ r = 23 :=
by
  sorry

end determine_base_l1849_184903


namespace valid_starting_day_count_l1849_184927

-- Defining the structure of the 30-day month and conditions
def days_in_month : Nat := 30

-- A function to determine the number of each weekday in a month which also checks if the given day is valid as per conditions
def valid_starting_days : List Nat :=
  [1] -- '1' represents Tuesday being the valid starting day corresponding to equal number of Tuesdays and Thursdays

-- The theorem we want to prove
-- The goal is to prove that there is only 1 valid starting day for the 30-day month to have equal number of Tuesdays and Thursdays
theorem valid_starting_day_count (days : Nat) (valid_days : List Nat) : 
  days = days_in_month → valid_days = valid_starting_days :=
by
  -- Sorry to skip full proof implementation
  sorry

end valid_starting_day_count_l1849_184927


namespace ratio_third_to_second_is_one_l1849_184981

variable (x y : ℕ)

-- The second throw skips 2 more times than the first throw
def second_throw := x + 2
-- The third throw skips y times
def third_throw := y
-- The fourth throw skips 3 fewer times than the third throw
def fourth_throw := y - 3
-- The fifth throw skips 1 more time than the fourth throw
def fifth_throw := (y - 3) + 1

-- The fifth throw skipped 8 times
axiom fifth_throw_condition : fifth_throw y = 8
-- The total number of skips between all throws is 33
axiom total_skips_condition : x + second_throw x + y + fourth_throw y + fifth_throw y = 33

-- Prove the ratio of skips in third throw to the second throw is 1:1
theorem ratio_third_to_second_is_one : (third_throw y) / (second_throw x) = 1 := sorry

end ratio_third_to_second_is_one_l1849_184981


namespace adult_tickets_sold_l1849_184916

theorem adult_tickets_sold (A S : ℕ) (h1 : S = 3 * A) (h2 : A + S = 600) : A = 150 :=
by
  sorry

end adult_tickets_sold_l1849_184916


namespace inequality_holds_if_and_only_if_c_lt_0_l1849_184973

theorem inequality_holds_if_and_only_if_c_lt_0 (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c < 0) :=
sorry

end inequality_holds_if_and_only_if_c_lt_0_l1849_184973


namespace g_675_eq_42_l1849_184967

noncomputable def g : ℕ → ℕ := sorry

axiom gxy : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g15 : g 15 = 18
axiom g45 : g 45 = 24

theorem g_675_eq_42 : g 675 = 42 :=
sorry

end g_675_eq_42_l1849_184967


namespace sum_of_repeating_decimals_correct_l1849_184905

/-- Convert repeating decimals to fractions -/
def rep_dec_1 : ℚ := 1 / 9
def rep_dec_2 : ℚ := 2 / 9
def rep_dec_3 : ℚ := 1 / 3
def rep_dec_4 : ℚ := 4 / 9
def rep_dec_5 : ℚ := 5 / 9
def rep_dec_6 : ℚ := 2 / 3
def rep_dec_7 : ℚ := 7 / 9
def rep_dec_8 : ℚ := 8 / 9

/-- Define the terms in the sum -/
def term_1 : ℚ := 8 + rep_dec_1
def term_2 : ℚ := 7 + 1 + rep_dec_2
def term_3 : ℚ := 6 + 2 + rep_dec_3
def term_4 : ℚ := 5 + 3 + rep_dec_4
def term_5 : ℚ := 4 + 4 + rep_dec_5
def term_6 : ℚ := 3 + 5 + rep_dec_6
def term_7 : ℚ := 2 + 6 + rep_dec_7
def term_8 : ℚ := 1 + 7 + rep_dec_8

/-- Define the sum of the terms -/
def total_sum : ℚ := term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

/-- Proof problem statement -/
theorem sum_of_repeating_decimals_correct : total_sum = 39.2 := 
sorry

end sum_of_repeating_decimals_correct_l1849_184905


namespace total_workers_construction_l1849_184991

def number_of_monkeys : Nat := 239
def number_of_termites : Nat := 622
def total_workers (m : Nat) (t : Nat) : Nat := m + t

theorem total_workers_construction : total_workers number_of_monkeys number_of_termites = 861 := by
  sorry

end total_workers_construction_l1849_184991


namespace sheep_ratio_l1849_184902

theorem sheep_ratio (S : ℕ) (h1 : 400 - S = 2 * 150) :
  S / 400 = 1 / 4 :=
by
  sorry

end sheep_ratio_l1849_184902


namespace value_of_x_squared_plus_reciprocal_squared_l1849_184994

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : 45 = x^4 + 1 / x^4) : 
  x^2 + 1 / x^2 = Real.sqrt 47 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l1849_184994


namespace solve_for_lambda_l1849_184982

def vector_dot_product : (ℤ × ℤ) → (ℤ × ℤ) → ℤ
| (x1, y1), (x2, y2) => x1 * x2 + y1 * y2

theorem solve_for_lambda
  (a : ℤ × ℤ) (b : ℤ × ℤ) (lambda : ℤ)
  (h1 : a = (3, -2))
  (h2 : b = (1, 2))
  (h3 : vector_dot_product (a.1 + lambda * b.1, a.2 + lambda * b.2) a = 0) :
  lambda = 13 :=
sorry

end solve_for_lambda_l1849_184982


namespace first_two_digits_of_52x_l1849_184986

-- Define the digit values that would make 52x divisible by 6.
def digit_values (x : Nat) : Prop :=
  x = 2 ∨ x = 5 ∨ x = 8

-- The main theorem to prove the first two digits are 52 given the conditions.
theorem first_two_digits_of_52x (x : Nat) (h : digit_values x) : (52 * 10 + x) / 10 = 52 :=
by sorry

end first_two_digits_of_52x_l1849_184986


namespace new_average_weight_l1849_184932

theorem new_average_weight (avg_weight_19_students : ℝ) (new_student_weight : ℝ) (num_students_initial : ℕ) : 
  avg_weight_19_students = 15 → new_student_weight = 7 → num_students_initial = 19 → 
  let total_weight_with_new_student := (avg_weight_19_students * num_students_initial + new_student_weight) 
  let new_num_students := num_students_initial + 1 
  let new_avg_weight := total_weight_with_new_student / new_num_students 
  new_avg_weight = 14.6 :=
by
  intros h1 h2 h3
  let total_weight := avg_weight_19_students * num_students_initial
  let total_weight_with_new_student := total_weight + new_student_weight
  let new_num_students := num_students_initial + 1
  let new_avg_weight := total_weight_with_new_student / new_num_students
  have h4 : total_weight = 285 := by sorry
  have h5 : total_weight_with_new_student = 292 := by sorry
  have h6 : new_num_students = 20 := by sorry
  have h7 : new_avg_weight = 292 / 20 := by sorry
  have h8 : new_avg_weight = 14.6 := by sorry
  exact h8

end new_average_weight_l1849_184932


namespace not_divisor_of_44_l1849_184984

theorem not_divisor_of_44 (m j : ℤ) (H1 : m = j * (j + 1) * (j + 2) * (j + 3))
  (H2 : 11 ∣ m) : ¬ (∀ j : ℤ, 44 ∣ j * (j + 1) * (j + 2) * (j + 3)) :=
by
  sorry

end not_divisor_of_44_l1849_184984


namespace inequality_one_solution_l1849_184941

theorem inequality_one_solution (a : ℝ) :
  (∀ x : ℝ, |x^2 + 2 * a * x + 4 * a| ≤ 4 → x = -a) ↔ a = 2 :=
by sorry

end inequality_one_solution_l1849_184941


namespace find_k_intersect_lines_l1849_184970

theorem find_k_intersect_lines :
  ∃ (k : ℚ), ∀ (x y : ℚ), 
  (2 * x + 3 * y + 8 = 0) → (x - y - 1 = 0) → (x + k * y = 0) → k = -1/2 :=
by sorry

end find_k_intersect_lines_l1849_184970


namespace angle_P_of_extended_sides_l1849_184993

noncomputable def regular_pentagon_angle_sum : ℕ := 540

noncomputable def internal_angle_regular_pentagon (n : ℕ) (h : 5 = n) : ℕ :=
  regular_pentagon_angle_sum / n

def interior_angle_pentagon : ℕ := 108

theorem angle_P_of_extended_sides (ABCDE : Prop) (h1 : interior_angle_pentagon = 108)
  (P : Prop) (h3 : 72 + 72 = 144) : 180 - 144 = 36 := by 
  sorry

end angle_P_of_extended_sides_l1849_184993


namespace photos_ratio_l1849_184971

theorem photos_ratio (L R C : ℕ) (h1 : R = L) (h2 : C = 12) (h3 : R = C + 24) :
  L / C = 3 :=
by 
  sorry

end photos_ratio_l1849_184971


namespace functional_equation_solution_l1849_184946

-- Define ℕ* (positive integers) as a subtype of ℕ
def Nat.star := {n : ℕ // n > 0}

-- Define the problem statement
theorem functional_equation_solution (f : Nat.star → Nat.star) :
  (∀ m n : Nat.star, m.val ^ 2 + (f n).val ∣ m.val * (f m).val + n.val) →
  (∀ n : Nat.star, f n = n) :=
by
  intro h
  sorry

end functional_equation_solution_l1849_184946


namespace binomial_expansion_of_110_minus_1_l1849_184958

theorem binomial_expansion_of_110_minus_1:
  110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 109^5 :=
by
  -- We will use the binomial theorem: (a - b)^n = ∑ (k in range(n+1)), C(n, k) * a^(n-k) * (-b)^k
  -- where C(n, k) are the binomial coefficients.
  sorry

end binomial_expansion_of_110_minus_1_l1849_184958


namespace direction_vector_b_l1849_184913

theorem direction_vector_b (b : ℝ) 
  (P Q : ℝ × ℝ) (hP : P = (-3, 1)) (hQ : Q = (1, 5))
  (hdir : 3 - (-3) = 3 ∧ 5 - 1 = b) : b = 3 := by
  sorry

end direction_vector_b_l1849_184913


namespace cricketer_total_matches_l1849_184910

theorem cricketer_total_matches (n : ℕ)
  (avg_total : ℝ) (avg_first_6 : ℝ) (avg_last_4 : ℝ)
  (total_runs_eq : 6 * avg_first_6 + 4 * avg_last_4 = n * avg_total) :
  avg_total = 38.9 ∧ avg_first_6 = 42 ∧ avg_last_4 = 34.25 → n = 10 :=
by
  sorry

end cricketer_total_matches_l1849_184910


namespace same_color_points_distance_2004_l1849_184921

noncomputable def exists_same_color_points_at_distance_2004 (color : ℝ × ℝ → ℕ) : Prop :=
  ∃ (p q : ℝ × ℝ), (p ≠ q) ∧ (color p = color q) ∧ (dist p q = 2004)

/-- The plane is colored in two colors. Prove that there exist two points of the same color at a distance of 2004 meters. -/
theorem same_color_points_distance_2004 {color : ℝ × ℝ → ℕ}
  (hcolor : ∀ p, color p = 1 ∨ color p = 2) :
  exists_same_color_points_at_distance_2004 color :=
sorry

end same_color_points_distance_2004_l1849_184921


namespace increasing_intervals_g_l1849_184906

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem increasing_intervals_g : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (0 : ℝ), x ≤ y → g x ≤ g y) ∧
  (∀ x ∈ Set.Ici (1 : ℝ), ∀ y ∈ Set.Ici (1 : ℝ), x ≤ y → g x ≤ g y) := 
sorry

end increasing_intervals_g_l1849_184906


namespace time_left_to_use_exerciser_l1849_184936

-- Definitions based on the conditions
def total_time : ℕ := 2 * 60  -- Total time in minutes (120 minutes)
def piano_time : ℕ := 30  -- Time spent on piano
def writing_music_time : ℕ := 25  -- Time spent on writing music
def history_time : ℕ := 38  -- Time spent on history

-- The theorem statement that Joan has 27 minutes left
theorem time_left_to_use_exerciser : 
  total_time - (piano_time + writing_music_time + history_time) = 27 :=
by {
  sorry
}

end time_left_to_use_exerciser_l1849_184936


namespace five_b_value_l1849_184979

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := 
by
  sorry

end five_b_value_l1849_184979


namespace longest_sequence_positive_integer_x_l1849_184959

theorem longest_sequence_positive_integer_x :
  ∃ x : ℤ, 0 < x ∧ 34 * x - 10500 > 0 ∧ 17000 - 55 * x > 0 ∧ x = 309 :=
by
  use 309
  sorry

end longest_sequence_positive_integer_x_l1849_184959


namespace base7_digit_sum_l1849_184904

theorem base7_digit_sum (A B C : ℕ) (hA : 1 ≤ A ∧ A < 7) (hB : 1 ≤ B ∧ B < 7) 
  (hC : 1 ≤ C ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eq : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) : 
  B + C = 6 := 
sorry

end base7_digit_sum_l1849_184904


namespace revenue_fraction_large_cups_l1849_184996

theorem revenue_fraction_large_cups (total_cups : ℕ) (price_small : ℚ) (price_large : ℚ)
  (h1 : price_large = (7 / 6) * price_small) 
  (h2 : (1 / 5 : ℚ) * total_cups = total_cups - (4 / 5 : ℚ) * total_cups) :
  ((4 / 5 : ℚ) * (7 / 6 * price_small) * total_cups) / 
  (((1 / 5 : ℚ) * price_small + (4 / 5 : ℚ) * (7 / 6 * price_small)) * total_cups) = (14 / 17 : ℚ) :=
by
  intros
  have h_total_small := (1 / 5 : ℚ) * total_cups
  have h_total_large := (4 / 5 : ℚ) * total_cups
  have revenue_small := h_total_small * price_small
  have revenue_large := h_total_large * price_large
  have total_revenue := revenue_small + revenue_large
  have revenue_large_frac := revenue_large / total_revenue
  have target_frac := (14 / 17 : ℚ)
  have target := revenue_large_frac = target_frac
  sorry

end revenue_fraction_large_cups_l1849_184996


namespace inequality_may_not_hold_l1849_184966

theorem inequality_may_not_hold (a b c : ℝ) (h : a > b) : (c < 0) → ¬ (a/c > b/c) := 
sorry

end inequality_may_not_hold_l1849_184966


namespace proof_problem_l1849_184990

-- Define the propositions p and q.
def p (a : ℝ) : Prop := a < -1/2 

def q (a b : ℝ) : Prop := a > b → (1 / (a + 1)) < (1 / (b + 1))

-- Define the final proof problem: proving that "p or q" is true.
theorem proof_problem (a b : ℝ) : (p a) ∨ (q a b) := by
  sorry

end proof_problem_l1849_184990


namespace units_digit_of_2_pow_20_minus_1_l1849_184909

theorem units_digit_of_2_pow_20_minus_1 : (2^20 - 1) % 10 = 5 := 
  sorry

end units_digit_of_2_pow_20_minus_1_l1849_184909


namespace sum_of_remainders_l1849_184960

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  -- sorry is here to skip the actual proof as per instructions
  sorry

end sum_of_remainders_l1849_184960


namespace problem_statement_l1849_184957

theorem problem_statement (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : m + 5 < n) 
    (h4 : (m + 3 + m + 7 + m + 13 + n + 4 + n + 5 + 2 * n + 3) / 6 = n + 3)
    (h5 : (↑((m + 13) + (n + 4)) / 2 : ℤ) = n + 3) : 
  m + n = 37 :=
by
  sorry

end problem_statement_l1849_184957


namespace sum_of_digits_of_greatest_prime_divisor_l1849_184933

-- Define the number 32767
def number : ℕ := 32767

-- Assert that 32767 is 2^15 - 1
lemma number_def : number = 2^15 - 1 := by
  sorry

-- State that 151 is the greatest prime divisor of 32767
lemma greatest_prime_divisor : Nat.Prime 151 ∧ ∀ p : ℕ, Nat.Prime p → p ∣ number → p ≤ 151 := by
  sorry

-- Calculate the sum of the digits of 151
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Conclude the sum of the digits of the greatest prime divisor is 7
theorem sum_of_digits_of_greatest_prime_divisor : sum_of_digits 151 = 7 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_l1849_184933


namespace natasha_dimes_l1849_184977

theorem natasha_dimes (n : ℕ) (h1 : 10 < n) (h2 : n < 100) (h3 : n % 3 = 1) (h4 : n % 4 = 1) (h5 : n % 5 = 1) : n = 61 :=
sorry

end natasha_dimes_l1849_184977


namespace trisect_angle_l1849_184948

noncomputable def can_trisect_with_ruler_and_compasses (n : ℕ) : Prop :=
  ¬(3 ∣ n) → ∃ a b : ℤ, 3 * a + n * b = 1

theorem trisect_angle (n : ℕ) (h : ¬(3 ∣ n)) :
  can_trisect_with_ruler_and_compasses n :=
sorry

end trisect_angle_l1849_184948


namespace value_of_f_at_3_l1849_184938

def f (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 - 3 * x + 7

theorem value_of_f_at_3 : f 3 = 196 := by
  sorry

end value_of_f_at_3_l1849_184938


namespace dragon_jewels_end_l1849_184935

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end dragon_jewels_end_l1849_184935


namespace find_x1_l1849_184965

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 := 
sorry

end find_x1_l1849_184965


namespace card_area_after_one_inch_shortening_l1849_184969

def initial_length := 5
def initial_width := 7
def new_area_shortened_side_two := 21
def shorter_side_reduction := 2
def longer_side_reduction := 1

theorem card_area_after_one_inch_shortening :
  (initial_length - shorter_side_reduction) * initial_width = new_area_shortened_side_two →
  initial_length * (initial_width - longer_side_reduction) = 30 :=
by
  intro h
  sorry

end card_area_after_one_inch_shortening_l1849_184969


namespace unique_triplet_satisfying_conditions_l1849_184943

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
                 (c ∣ a * b + 1) ∧
                 (b ∣ c * a + 1) ∧
                 (a ∣ b * c + 1) ∧
                 a = 2 ∧ b = 3 ∧ c = 7 :=
by
  sorry

end unique_triplet_satisfying_conditions_l1849_184943


namespace elder_age_is_33_l1849_184925

-- Define the conditions
variables (y e : ℕ)

def age_difference_condition : Prop :=
  e = y + 20

def age_reduced_condition : Prop :=
  e - 8 = 5 * (y - 8)

-- State the theorem to prove the age of the elder person
theorem elder_age_is_33 (h1 : age_difference_condition y e) (h2 : age_reduced_condition y e): e = 33 :=
  sorry

end elder_age_is_33_l1849_184925


namespace find_c_l1849_184988

theorem find_c (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : c = 0 :=
by
  sorry

end find_c_l1849_184988


namespace squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l1849_184954

theorem squares_have_consecutive_digits (n : ℕ) (h : ∃ j : ℕ, n = 33330 + j ∧ j < 10) :
    ∃ (a b : ℕ), n ^ 2 / 10 ^ a % 10 = n ^ 2 / 10 ^ (a + 1) % 10 :=
by
  sorry

theorem generalized_squares_have_many_consecutive_digits (k : ℕ) (n : ℕ)
  (h1 : k ≥ 4)
  (h2 : ∃ j : ℕ, n = 33333 * 10 ^ (k - 4) + j ∧ j < 10 ^ (k - 4)) :
    ∃ m, ∃ l : ℕ, ∀ i < m, n^2 / 10 ^ (l + i) % 10 = n^2 / 10 ^ l % 10 :=
by
  sorry

end squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l1849_184954


namespace find_line_equation_l1849_184924

theorem find_line_equation :
  ∃ (m : ℝ), ∃ (b : ℝ), (∀ x y : ℝ,
  (x + 3 * y - 2 = 0 → y = -1/3 * x + 2/3) ∧
  (x = 3 → y = 0) →
  y = m * x + b) ∧
  (m = 3 ∧ b = -9) :=
  sorry

end find_line_equation_l1849_184924


namespace S15_eq_l1849_184952

-- Definitions in terms of the geometric sequence and given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions given in the problem
axiom geom_seq (n : ℕ) : S n = (a 0) * (1 - (a 1) ^ n) / (1 - (a 1))
axiom S5_eq : S 5 = 10
axiom S10_eq : S 10 = 50

-- The problem statement to prove
theorem S15_eq : S 15 = 210 :=
by sorry

end S15_eq_l1849_184952


namespace stream_speed_zero_l1849_184901

theorem stream_speed_zero (v_c v_s : ℝ)
  (h1 : v_c - v_s - 2 = 9)
  (h2 : v_c + v_s + 1 = 12) :
  v_s = 0 := 
sorry

end stream_speed_zero_l1849_184901


namespace sequence_sum_l1849_184915

noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry
noncomputable def a₅ : ℝ := sorry
noncomputable def a₆ : ℝ := sorry
noncomputable def a₇ : ℝ := sorry
noncomputable def a₈ : ℝ := sorry
noncomputable def q : ℝ := sorry

axiom condition_1 : a₁ + a₂ + a₃ + a₄ = 1
axiom condition_2 : a₅ + a₆ + a₇ + a₈ = 2
axiom condition_3 : q^4 = 2

theorem sequence_sum : q = (2:ℝ)^(1/4) → a₁ + a₂ + a₃ + a₄ = 1 → 
  (a₁ * q^16 + a₂ * q^17 + a₃ * q^18 + a₄ * q^19) = 16 := 
by
  intros hq hsum_s4
  sorry

end sequence_sum_l1849_184915


namespace initially_working_machines_l1849_184976

theorem initially_working_machines (N R x : ℝ) 
  (h1 : N * R = x / 3) 
  (h2 : 45 * R = x / 2) : 
  N = 30 := by
  sorry

end initially_working_machines_l1849_184976


namespace fraction_of_male_fish_l1849_184972

def total_fish : ℕ := 45
def female_fish : ℕ := 15
def male_fish := total_fish - female_fish

theorem fraction_of_male_fish : (male_fish : ℚ) / total_fish = 2 / 3 := by
  sorry

end fraction_of_male_fish_l1849_184972


namespace f_at_1_is_neg7007_l1849_184995

variable (a b c : ℝ)

def g (x : ℝ) := x^3 + a * x^2 + x + 10
def f (x : ℝ) := x^4 + x^3 + b * x^2 + 100 * x + c

theorem f_at_1_is_neg7007
  (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ g a (r1) = 0 ∧ g a (r2) = 0 ∧ g a (r3) = 0)
  (h2 : ∀ x, f x = 0 → g x = 0) :
  f 1 = -7007 := 
sorry

end f_at_1_is_neg7007_l1849_184995


namespace relationship_between_vars_l1849_184917

variable {α : Type*} [LinearOrderedAddCommGroup α]

theorem relationship_between_vars (a b : α) 
  (h1 : a + b < 0) 
  (h2 : b > 0) : a < -b ∧ -b < b ∧ b < -a :=
by
  sorry

end relationship_between_vars_l1849_184917


namespace dogs_in_kennel_l1849_184911

theorem dogs_in_kennel (C D : ℕ) (h1 : C = D - 8) (h2 : C * 4 = 3 * D) : D = 32 :=
sorry

end dogs_in_kennel_l1849_184911


namespace profit_difference_l1849_184920

theorem profit_difference
  (p1 p2 : ℝ)
  (h1 : p1 > p2)
  (h2 : p1 + p2 = 3635000)
  (h3 : p2 = 442500) :
  p1 - p2 = 2750000 :=
by 
  sorry

end profit_difference_l1849_184920


namespace findAngleC_findPerimeter_l1849_184944

noncomputable def triangleCondition (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m := (b+c, Real.sin A)
  let n := (a+b, Real.sin C - Real.sin B)
  m.1 * n.2 = m.2 * n.1 -- m parallel to n

noncomputable def lawOfSines (a b c A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def areaOfTriangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C -- Area calculation by a, b, and angle between them

theorem findAngleC (a b c A B C : ℝ) : 
  triangleCondition a b c A B C ∧ lawOfSines a b c A B C → 
  Real.cos C = -1/2 :=
sorry

theorem findPerimeter (a b c A B C : ℝ) : 
  b = 4 ∧ areaOfTriangle a b c A B C = 4 * Real.sqrt 3 → 
  a = 4 ∧ b = 4 ∧ c = 4 * Real.sqrt 3 ∧ a + b + c = 8 + 4 * Real.sqrt 3 :=
sorry

end findAngleC_findPerimeter_l1849_184944


namespace factor_expression_l1849_184926

theorem factor_expression (z : ℤ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := 
by sorry

end factor_expression_l1849_184926


namespace number_of_three_digit_numbers_is_48_l1849_184985

-- Define the problem: the cards and their constraints
def card1 := (1, 2)
def card2 := (3, 4)
def card3 := (5, 6)

-- The condition given is that 6 cannot be used as 9

-- Define the function to compute the number of different three-digit numbers
def number_of_three_digit_numbers : Nat := 6 * 4 * 2

/- Prove that the number of different three-digit numbers that can be formed is 48 -/
theorem number_of_three_digit_numbers_is_48 : number_of_three_digit_numbers = 48 :=
by
  -- We skip the proof here
  sorry

end number_of_three_digit_numbers_is_48_l1849_184985


namespace non_adjacent_arrangement_l1849_184992

-- Define the number of people
def numPeople : ℕ := 8

-- Define the number of specific people who must not be adjacent
def numSpecialPeople : ℕ := 3

-- Define the number of general people who are not part of the specific group
def numGeneralPeople : ℕ := numPeople - numSpecialPeople

-- Permutations calculation for general people
def permuteGeneralPeople : ℕ := Nat.factorial numGeneralPeople

-- Number of gaps available after arranging general people
def numGaps : ℕ := numGeneralPeople + 1

-- Permutations calculation for special people placed in the gaps
def permuteSpecialPeople : ℕ := Nat.descFactorial numGaps numSpecialPeople

-- Total permutations
def totalPermutations : ℕ := permuteSpecialPeople * permuteGeneralPeople

theorem non_adjacent_arrangement :
  totalPermutations = Nat.descFactorial 6 3 * Nat.factorial 5 := by
  sorry

end non_adjacent_arrangement_l1849_184992


namespace range_of_a_l1849_184947

theorem range_of_a (a : ℝ) (h1 : 2 * a + 1 < 17) (h2 : 2 * a + 1 > 7) : 3 < a ∧ a < 8 := by
  sorry

end range_of_a_l1849_184947


namespace track_length_eq_900_l1849_184950

/-- 
Bruce and Bhishma are running on a circular track. 
The speed of Bruce is 30 m/s and that of Bhishma is 20 m/s.
They start from the same point at the same time in the same direction.
They meet again for the first time after 90 seconds. 
Prove that the length of the track is 900 meters.
-/
theorem track_length_eq_900 :
  let speed_bruce := 30 -- [m/s]
  let speed_bhishma := 20 -- [m/s]
  let time_meet := 90 -- [s]
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  track_length = 900 :=
by
  let speed_bruce := 30
  let speed_bhishma := 20
  let time_meet := 90
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  have : track_length = 900 := by
    sorry
  exact this

end track_length_eq_900_l1849_184950


namespace arithmetic_geometric_sequence_l1849_184951

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 * a 5 = (a 4) ^ 2)
  (h4 : d ≠ 0) : d = -1 / 5 :=
by
  sorry

end arithmetic_geometric_sequence_l1849_184951


namespace base_conversion_unique_b_l1849_184968

theorem base_conversion_unique_b (b : ℕ) (h_b_pos : 0 < b) :
  (1 * 5^2 + 3 * 5^1 + 2 * 5^0) = (2 * b^2 + b) → b = 4 :=
by
  sorry

end base_conversion_unique_b_l1849_184968


namespace part1_part2_l1849_184930

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem part1 (a : ℝ) (h : 0 < a) (hf'1 : (1 - 2 * a * 1 - 1) = -2) :
  a = 1 ∧ (∀ x y : ℝ, y = -2 * (x - 1) → 2 * x + y - 2 = 0) :=
by
  sorry

theorem part2 {a : ℝ} (ha : a ≥ 1 / 8) :
  ∀ x : ℝ, (1 - 2 * a * x - 1 / x) ≤ 0 :=
by
  sorry

end part1_part2_l1849_184930


namespace driver_schedule_l1849_184939

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l1849_184939


namespace marble_probability_l1849_184928

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end marble_probability_l1849_184928


namespace average_calculation_l1849_184964

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 2 0) (average_two 1 2) 1 = 23 / 18 :=
by sorry

end average_calculation_l1849_184964


namespace prize_distribution_l1849_184937

/--
In a best-of-five competition where two players of equal level meet in the final, 
with a score of 2:1 after the first three games and the total prize money being 12,000 yuan, 
the prize awarded to the player who has won 2 games should be 9,000 yuan.
-/
theorem prize_distribution (prize_money : ℝ) 
  (A_wins : ℕ) (B_wins : ℕ) (prob_A : ℝ) (prob_B : ℝ) (total_games : ℕ) : 
  total_games = 5 → 
  prize_money = 12000 → 
  A_wins = 2 → 
  B_wins = 1 → 
  prob_A = 1/2 → 
  prob_B = 1/2 → 
  ∃ prize_for_A : ℝ, prize_for_A = 9000 :=
by
  intros
  sorry

end prize_distribution_l1849_184937
