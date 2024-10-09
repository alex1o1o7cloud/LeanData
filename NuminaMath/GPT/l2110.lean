import Mathlib

namespace minimum_A2_minus_B2_l2110_211026

noncomputable def A (x y z : ℝ) : ℝ := 
  Real.sqrt (x + 6) + Real.sqrt (y + 7) + Real.sqrt (z + 12)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 2) + Real.sqrt (y + 3) + Real.sqrt (z + 5)

theorem minimum_A2_minus_B2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z)^2 - (B x y z)^2 = 49.25 := 
by 
  sorry 

end minimum_A2_minus_B2_l2110_211026


namespace smallest_common_multiple_5_6_l2110_211005

theorem smallest_common_multiple_5_6 (n : ℕ) 
  (h_pos : 0 < n) 
  (h_5 : 5 ∣ n) 
  (h_6 : 6 ∣ n) :
  n = 30 :=
sorry

end smallest_common_multiple_5_6_l2110_211005


namespace solution_set_x2_minus_x_lt_0_l2110_211039

theorem solution_set_x2_minus_x_lt_0 :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ x^2 - x < 0 := 
by
  sorry

end solution_set_x2_minus_x_lt_0_l2110_211039


namespace range_of_a_l2110_211081

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| > Real.logb 2 a) →
  0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l2110_211081


namespace find_g_2_l2110_211072

variable (g : ℝ → ℝ)

-- Function satisfying the given conditions
axiom g_functional : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom g_nonzero : ∀ (x : ℝ), g x ≠ 0

-- The proof statement
theorem find_g_2 : g 2 = 1 := by
  sorry

end find_g_2_l2110_211072


namespace geologists_probability_l2110_211050

theorem geologists_probability
  (n roads : ℕ) (speed_per_hour : ℕ) 
  (angle_between_neighbors : ℕ)
  (distance_limit : ℝ) : 
  n = 6 ∧ speed_per_hour = 4 ∧ angle_between_neighbors = 60 ∧ distance_limit = 6 → 
  prob_distance_at_least_6_km = 0.5 :=
by
  sorry

noncomputable def prob_distance_at_least_6_km : ℝ := 0.5  -- Placeholder definition

end geologists_probability_l2110_211050


namespace tire_circumference_is_one_meter_l2110_211098

-- Definitions for the given conditions
def car_speed : ℕ := 24 -- in km/h
def tire_rotations_per_minute : ℕ := 400

-- Conversion factors
def km_to_m : ℕ := 1000
def hour_to_min : ℕ := 60

-- The equivalent proof problem
theorem tire_circumference_is_one_meter 
  (hs : car_speed * km_to_m / hour_to_min = 400 * tire_rotations_per_minute)
  : 400 = 400 * 1 := 
by
  sorry

end tire_circumference_is_one_meter_l2110_211098


namespace original_kittens_count_l2110_211010

theorem original_kittens_count 
  (K : ℕ) 
  (h1 : K - 3 + 9 = 12) : 
  K = 6 := by
sorry

end original_kittens_count_l2110_211010


namespace locker_count_proof_l2110_211062

theorem locker_count_proof (cost_per_digit : ℕ := 3)
  (total_cost : ℚ := 224.91) :
  (N : ℕ) = 2151 :=
by
  sorry

end locker_count_proof_l2110_211062


namespace cos_identity_example_l2110_211001

theorem cos_identity_example (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 3 / 5) : Real.cos (Real.pi / 3 - α) = 3 / 5 := by
  sorry

end cos_identity_example_l2110_211001


namespace counterpositive_prop_l2110_211093

theorem counterpositive_prop (a b c : ℝ) (h : a^2 + b^2 + c^2 < 3) : a + b + c ≠ 3 := 
sorry

end counterpositive_prop_l2110_211093


namespace smallest_prime_p_l2110_211023

theorem smallest_prime_p 
  (p q r : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime q) 
  (h3 : r > 0) 
  (h4 : p + q = r) 
  (h5 : q < p) 
  (h6 : q = 2) 
  (h7 : Nat.Prime r)  
  : p = 3 := 
sorry

end smallest_prime_p_l2110_211023


namespace product_of_two_numbers_l2110_211088

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by 
  sorry

end product_of_two_numbers_l2110_211088


namespace max_volume_l2110_211051

variable (x y z : ℝ) (V : ℝ)
variable (k : ℝ)

-- Define the constraint
def constraint := x + 2 * y + 3 * z = 180

-- Define the volume
def volume := x * y * z

-- The goal is to show that under the constraint, the maximum possible volume is 36000 cubic cm.
theorem max_volume :
  (∀ (x y z : ℝ) (h : constraint x y z), volume x y z ≤ 36000) :=
  sorry

end max_volume_l2110_211051


namespace scallops_per_person_l2110_211075

theorem scallops_per_person 
    (scallops_per_pound : ℕ)
    (cost_per_pound : ℝ)
    (total_cost : ℝ)
    (people : ℕ)
    (total_pounds : ℝ)
    (total_scallops : ℕ)
    (scallops_per_person : ℕ)
    (h1 : scallops_per_pound = 8)
    (h2 : cost_per_pound = 24)
    (h3 : total_cost = 48)
    (h4 : people = 8)
    (h5 : total_pounds = total_cost / cost_per_pound)
    (h6 : total_scallops = scallops_per_pound * total_pounds)
    (h7 : scallops_per_person = total_scallops / people) : 
    scallops_per_person = 2 := 
by {
    sorry
}

end scallops_per_person_l2110_211075


namespace seashells_increase_l2110_211086

def initial_seashells : ℕ := 50
def final_seashells : ℕ := 130
def week_increment (x : ℕ) : ℕ := 4 * x + initial_seashells

theorem seashells_increase (x : ℕ) (h: final_seashells = week_increment x) : x = 8 :=
by {
  sorry
}

end seashells_increase_l2110_211086


namespace sqrt_12_minus_sqrt_27_l2110_211060

theorem sqrt_12_minus_sqrt_27 :
  (Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3) := by
  sorry

end sqrt_12_minus_sqrt_27_l2110_211060


namespace train_speed_km_per_hr_l2110_211053

theorem train_speed_km_per_hr
  (train_length : ℝ) 
  (platform_length : ℝ)
  (time_seconds : ℝ) 
  (h_train_length : train_length = 470) 
  (h_platform_length : platform_length = 520) 
  (h_time_seconds : time_seconds = 64.79481641468682) :
  (train_length + platform_length) / time_seconds * 3.6 = 54.975 := 
sorry

end train_speed_km_per_hr_l2110_211053


namespace problem_I4_1_l2110_211074

theorem problem_I4_1 
  (x y : ℝ)
  (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
sorry

end problem_I4_1_l2110_211074


namespace calvin_total_insects_l2110_211061

-- Definitions based on the conditions
def roaches := 12
def scorpions := 3
def crickets := roaches / 2
def caterpillars := scorpions * 2

-- Statement of the problem
theorem calvin_total_insects : 
  roaches + scorpions + crickets + caterpillars = 27 :=
  by
    sorry

end calvin_total_insects_l2110_211061


namespace largest_integer_solution_l2110_211082

theorem largest_integer_solution (x : ℤ) (h : 3 - 2 * x > 0) : x ≤ 1 :=
by sorry

end largest_integer_solution_l2110_211082


namespace same_quadratic_function_b_l2110_211095

theorem same_quadratic_function_b (a c b : ℝ) :
    (∀ x : ℝ, a * (x - 2)^2 + c = (2 * x - 5) * (x - b)) → b = 3 / 2 :=
by
  sorry

end same_quadratic_function_b_l2110_211095


namespace find_number_l2110_211002

def condition (x : ℤ) : Prop := 3 * (x + 8) = 36

theorem find_number (x : ℤ) (h : condition x) : x = 4 := by
  sorry

end find_number_l2110_211002


namespace arithmetic_geometric_inequality_l2110_211027

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a ≠ b) :
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  B < (a - b)^2 / (8 * (A - B)) ∧ (a - b)^2 / (8 * (A - B)) < A :=
by
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  sorry

end arithmetic_geometric_inequality_l2110_211027


namespace rectangle_perimeter_l2110_211057

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 :=
by
  -- Proving the theorem here
  sorry

end rectangle_perimeter_l2110_211057


namespace equivalent_expression_l2110_211069

theorem equivalent_expression :
  (5+3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * 
  (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := 
  sorry

end equivalent_expression_l2110_211069


namespace smallest_mul_seven_perfect_square_l2110_211055

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the problem statement
theorem smallest_mul_seven_perfect_square :
  ∀ x : ℕ, x > 0 → (is_perfect_square (7 * x) ↔ x = 7) := 
by {
  sorry
}

end smallest_mul_seven_perfect_square_l2110_211055


namespace sequence_b_l2110_211019

theorem sequence_b (b : ℕ → ℝ) (h₁ : b 1 = 1)
  (h₂ : ∀ n : ℕ, n ≥ 1 → (b (n + 1)) ^ 4 = 64 * (b n) ^ 4) :
  b 50 = 2 ^ 49 := by
  sorry

end sequence_b_l2110_211019


namespace william_napkins_l2110_211011

-- Define the given conditions
variables (O A C G W : ℕ)
variables (ho: O = 10)
variables (ha: A = 2 * O)
variables (hc: C = A / 2)
variables (hg: G = 3 * C)
variables (hw: W = 15)

-- Prove the total number of napkins William has now
theorem william_napkins (O A C G W : ℕ) (ho: O = 10) (ha: A = 2 * O)
  (hc: C = A / 2) (hg: G = 3 * C) (hw: W = 15) : W + (O + A + C + G) = 85 :=
by {
  sorry
}

end william_napkins_l2110_211011


namespace activity_participants_l2110_211096

variable (A B C D : Prop)

theorem activity_participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) : B ∧ C ∧ ¬A ∧ ¬D :=
by
  sorry

end activity_participants_l2110_211096


namespace Hoelder_l2110_211084

variable (A B p q : ℝ)

theorem Hoelder (hA : 0 < A) (hB : 0 < B) (hp : 0 < p) (hq : 0 < q) (h : 1 / p + 1 / q = 1) : 
  A^(1/p) * B^(1/q) ≤ A / p + B / q := 
sorry

end Hoelder_l2110_211084


namespace evaluate_g_expressions_l2110_211056

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g_expressions : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end evaluate_g_expressions_l2110_211056


namespace expression_negativity_l2110_211079

-- Given conditions: a, b, and c are lengths of the sides of a triangle
variables (a b c : ℝ)
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- To prove: (a - b)^2 - c^2 < 0
theorem expression_negativity (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a - b)^2 - c^2 < 0 :=
sorry

end expression_negativity_l2110_211079


namespace car_kilometers_per_gallon_l2110_211076

theorem car_kilometers_per_gallon :
  ∀ (distance gallon_used : ℝ), distance = 120 → gallon_used = 6 →
  distance / gallon_used = 20 :=
by
  intros distance gallon_used h_distance h_gallon_used
  sorry

end car_kilometers_per_gallon_l2110_211076


namespace min_value_of_quadratic_function_l2110_211041

def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2 * x - 5

theorem min_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 :=
by
  sorry

end min_value_of_quadratic_function_l2110_211041


namespace find_number_of_small_gardens_l2110_211073

-- Define the conditions
def seeds_total : Nat := 52
def seeds_big_garden : Nat := 28
def seeds_per_small_garden : Nat := 4

-- Define the target value
def num_small_gardens : Nat := 6

-- The statement of the proof problem
theorem find_number_of_small_gardens 
  (H1 : seeds_total = 52) 
  (H2 : seeds_big_garden = 28) 
  (H3 : seeds_per_small_garden = 4) 
  : seeds_total - seeds_big_garden = 24 ∧ (seeds_total - seeds_big_garden) / seeds_per_small_garden = num_small_gardens := 
sorry

end find_number_of_small_gardens_l2110_211073


namespace no_integer_valued_function_l2110_211016

theorem no_integer_valued_function (f : ℤ → ℤ) (h : ∀ (m n : ℤ), f (m + f n) = f m - n) : False :=
sorry

end no_integer_valued_function_l2110_211016


namespace smallest_n_l2110_211003

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ 2000 * n % 21 = 0 ∧ ∀ m : ℕ, m > 0 ∧ 2000 * m % 21 = 0 → n ≤ m :=
sorry

end smallest_n_l2110_211003


namespace n_is_square_l2110_211067

theorem n_is_square (n m : ℕ) (h1 : 3 ≤ n) (h2 : m = (n * (n - 1)) / 2) (h3 : ∃ (cards : Finset ℕ), 
  (cards.card = n) ∧ (∀ i ∈ cards, i ∈ Finset.range (m + 1)) ∧ 
  (∀ (i j : ℕ) (hi : i ∈ cards) (hj : j ∈ cards), i ≠ j → 
    ((i + j) % m) ≠ ((i + j) % m))) : 
  ∃ k : ℕ, n = k * k := 
sorry

end n_is_square_l2110_211067


namespace min_value_of_x_plus_2y_l2110_211024

theorem min_value_of_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 1 / x + 1 / y = 2) : 
  x + 2 * y ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_x_plus_2y_l2110_211024


namespace quadrilateral_area_l2110_211077

-- Define the angles in the quadrilateral ABCD
def ABD : ℝ := 20
def DBC : ℝ := 60
def ADB : ℝ := 30
def BDC : ℝ := 70

-- Define the side lengths
variables (AB CD AD BC AC BD : ℝ)

-- Prove that the area of the quadrilateral ABCD is half the product of its sides
theorem quadrilateral_area (h1 : ABD = 20) (h2 : DBC = 60) (h3 : ADB = 30) (h4 : BDC = 70)
  : (1 / 2) * (AB * CD + AD * BC) = (1 / 2) * (AB * CD + AD * BC) :=
by
  sorry

end quadrilateral_area_l2110_211077


namespace Glenn_total_expenditure_l2110_211047

-- Define initial costs and discounts
def ticket_cost_Monday : ℕ := 5
def ticket_cost_Wednesday : ℕ := 2 * ticket_cost_Monday
def ticket_cost_Saturday : ℕ := 5 * ticket_cost_Monday
def discount_Wednesday (cost : ℕ) : ℕ := cost * 90 / 100
def additional_expense_Saturday : ℕ := 7

-- Define number of attendees
def attendees_Wednesday : ℕ := 4
def attendees_Saturday : ℕ := 2

-- Calculate total costs
def total_cost_Wednesday : ℕ :=
  attendees_Wednesday * discount_Wednesday ticket_cost_Wednesday
def total_cost_Saturday : ℕ :=
  attendees_Saturday * ticket_cost_Saturday + additional_expense_Saturday

-- Calculate the total money spent by Glenn
def total_spent : ℕ :=
  total_cost_Wednesday + total_cost_Saturday

-- Combine all conditions and conclusions into proof statement
theorem Glenn_total_expenditure : total_spent = 93 := by
  sorry

end Glenn_total_expenditure_l2110_211047


namespace probability_diff_colors_l2110_211032

theorem probability_diff_colors :
  let total_marbles := 24
  let prob_diff_colors := 
    (4 / 24) * (5 / 23) + 
    (4 / 24) * (12 / 23) + 
    (4 / 24) * (3 / 23) + 
    (5 / 24) * (12 / 23) + 
    (5 / 24) * (3 / 23) + 
    (12 / 24) * (3 / 23)
  prob_diff_colors = 191 / 552 :=
by sorry

end probability_diff_colors_l2110_211032


namespace pencil_cost_l2110_211037

theorem pencil_cost (P : ℝ) : 
  (∀ pen_cost total : ℝ, pen_cost = 3.50 → total = 291 → 38 * P + 56 * pen_cost = total → P = 2.50) :=
by
  intros pen_cost total h1 h2 h3
  sorry

end pencil_cost_l2110_211037


namespace leopards_to_rabbits_ratio_l2110_211009

theorem leopards_to_rabbits_ratio :
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  leopards / rabbits = 1 / 2 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  sorry

end leopards_to_rabbits_ratio_l2110_211009


namespace sequence_period_2016_l2110_211063

theorem sequence_period_2016 : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 1 / (1 - a n)) → 
  a 1 = 1 / 2 → 
  a 2016 = -1 :=
by
  sorry

end sequence_period_2016_l2110_211063


namespace chocolate_more_expensive_l2110_211030

variables (C P : ℝ)
theorem chocolate_more_expensive (h : 7 * C > 8 * P) : 8 * C > 9 * P :=
sorry

end chocolate_more_expensive_l2110_211030


namespace remaining_students_average_l2110_211034

theorem remaining_students_average
  (N : ℕ) (A : ℕ) (M : ℕ) (B : ℕ) (E : ℕ)
  (h1 : N = 20)
  (h2 : A = 80)
  (h3 : M = 5)
  (h4 : B = 50)
  (h5 : E = (N - M))
  : (N * A - M * B) / E = 90 :=
by
  -- Using sorries to skip the proof
  sorry

end remaining_students_average_l2110_211034


namespace max_r1_minus_r2_l2110_211044

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 2 + y^2 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def P (x y : ℝ) : Prop :=
  ellipse x y ∧ x > 0 ∧ y > 0

def r1 (x y : ℝ) (Q2 : ℝ × ℝ) : ℝ := 
  -- Assume a function that calculates the inradius of triangle ΔPF1Q2
  sorry

def r2 (x y : ℝ) (Q1 : ℝ × ℝ) : ℝ :=
  -- Assume a function that calculates the inradius of triangle ΔPF2Q1
  sorry

theorem max_r1_minus_r2 :
  ∃ (x y : ℝ) (Q1 Q2 : ℝ × ℝ), P x y →
    r1 x y Q2 - r2 x y Q1 = 1/3 := 
sorry

end max_r1_minus_r2_l2110_211044


namespace gcf_factorial_5_6_l2110_211085

theorem gcf_factorial_5_6 : Nat.gcd (Nat.factorial 5) (Nat.factorial 6) = Nat.factorial 5 := by
  sorry

end gcf_factorial_5_6_l2110_211085


namespace amount_spent_on_petrol_l2110_211029

theorem amount_spent_on_petrol
    (rent milk groceries education miscellaneous savings salary petrol : ℝ)
    (h1 : rent = 5000)
    (h2 : milk = 1500)
    (h3 : groceries = 4500)
    (h4 : education = 2500)
    (h5 : miscellaneous = 2500)
    (h6 : savings = 0.10 * salary)
    (h7 : savings = 2000)
    (total_salary : salary = 20000) : petrol = 2000 := by
  sorry

end amount_spent_on_petrol_l2110_211029


namespace sarah_flour_total_l2110_211059

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end sarah_flour_total_l2110_211059


namespace height_of_platform_l2110_211052

variable (h l w : ℕ)

-- Define the conditions as hypotheses
def measured_length_first_configuration : Prop := l + h - w = 40
def measured_length_second_configuration : Prop := w + h - l = 34

-- The goal is to prove that the height is 37 inches
theorem height_of_platform
  (h l w : ℕ)
  (config1 : measured_length_first_configuration h l w)
  (config2 : measured_length_second_configuration h l w) : 
  h = 37 := 
sorry

end height_of_platform_l2110_211052


namespace sampling_method_is_systematic_l2110_211043

-- Define the conditions of the problem
def conveyor_belt_transport : Prop := true
def inspectors_sampling_every_ten_minutes : Prop := true

-- Define what needs to be proved
theorem sampling_method_is_systematic :
  conveyor_belt_transport ∧ inspectors_sampling_every_ten_minutes → is_systematic_sampling :=
by
  sorry

-- Example definition that could be used in the proof
def is_systematic_sampling : Prop := true

end sampling_method_is_systematic_l2110_211043


namespace negation_of_all_honest_l2110_211087

-- Define the needed predicates
variable {Man : Type} -- Type for men
variable (man : Man → Prop)
variable (age : Man → ℕ)
variable (honest : Man → Prop)

-- Define the conditions and the statement we want to prove
theorem negation_of_all_honest :
  (∀ x, man x → age x > 30 → honest x) →
  (∃ x, man x ∧ age x > 30 ∧ ¬ honest x) :=
sorry

end negation_of_all_honest_l2110_211087


namespace smallest_number_l2110_211097

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1/2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end smallest_number_l2110_211097


namespace first_term_arith_seq_l2110_211033

noncomputable def is_increasing (a b c : ℕ) (d : ℕ) : Prop := b = a + d ∧ c = a + 2 * d ∧ 0 < d

theorem first_term_arith_seq (a₁ a₂ a₃ : ℕ) (d: ℕ) :
  is_increasing a₁ a₂ a₃ d ∧ a₁ + a₂ + a₃ = 12 ∧ a₁ * a₂ * a₃ = 48 → a₁ = 2 := sorry

end first_term_arith_seq_l2110_211033


namespace simple_interest_rate_l2110_211036

theorem simple_interest_rate 
  (P A T : ℝ) 
  (hP : P = 900) 
  (hA : A = 950) 
  (hT : T = 5) 
  : (A - P) * 100 / (P * T) = 1.11 :=
by
  sorry

end simple_interest_rate_l2110_211036


namespace squirrel_travel_time_l2110_211006

theorem squirrel_travel_time :
  ∀ (speed distance : ℝ), speed = 5 → distance = 3 →
  (distance / speed) * 60 = 36 := by
  intros speed distance h_speed h_distance
  rw [h_speed, h_distance]
  norm_num

end squirrel_travel_time_l2110_211006


namespace smallest_integer_C_l2110_211028

-- Define the function f(n) = 6^n / n!
def f (n : ℕ) : ℚ := (6 ^ n) / (Nat.factorial n)

theorem smallest_integer_C (C : ℕ) (h : ∀ n : ℕ, n > 0 → f n ≤ C) : C = 65 :=
by
  sorry

end smallest_integer_C_l2110_211028


namespace binomial_7_2_eq_21_l2110_211054

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l2110_211054


namespace product_of_remaining_numbers_l2110_211078

theorem product_of_remaining_numbers {a b c d : ℕ} (h1 : a = 11) (h2 : b = 22) (h3 : c = 33) (h4 : d = 44) :
  ∃ (x y z : ℕ), 
  (∃ n: ℕ, (a + b + c + d) - n * 3 = 3 ∧ -- We removed n groups of 3 different numbers
             x + y + z = 2 * n + (a + b + c + d)) ∧ -- We added 2 * n numbers back
  x * y * z = 12 := 
sorry

end product_of_remaining_numbers_l2110_211078


namespace prime_sum_55_l2110_211021

theorem prime_sum_55 (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hpqrs : p < q ∧ q < r ∧ r < s) 
  (h_eqn : 1 - (1 : ℚ)/p - (1 : ℚ)/q - (1 : ℚ)/r - (1 : ℚ)/s = 1 / (p * q * r * s)) :
  p + q + r + s = 55 := 
sorry

end prime_sum_55_l2110_211021


namespace total_amount_to_pay_l2110_211017

theorem total_amount_to_pay (cost_earbuds cost_smartwatch : ℕ) (tax_rate_earbuds tax_rate_smartwatch : ℚ) 
  (h1 : cost_earbuds = 200) (h2 : cost_smartwatch = 300) 
  (h3 : tax_rate_earbuds = 0.15) (h4 : tax_rate_smartwatch = 0.12) : 
  (cost_earbuds + cost_earbuds * tax_rate_earbuds + cost_smartwatch + cost_smartwatch * tax_rate_smartwatch = 566) := 
by 
  sorry

end total_amount_to_pay_l2110_211017


namespace roden_total_fish_l2110_211080

def total_goldfish : Nat :=
  15 + 10 + 3 + 4

def total_blue_fish : Nat :=
  7 + 12 + 7 + 8

def total_green_fish : Nat :=
  5 + 9 + 6

def total_purple_fish : Nat :=
  2

def total_red_fish : Nat :=
  1

def total_fish : Nat :=
  total_goldfish + total_blue_fish + total_green_fish + total_purple_fish + total_red_fish

theorem roden_total_fish : total_fish = 89 :=
by
  unfold total_fish total_goldfish total_blue_fish total_green_fish total_purple_fish total_red_fish
  sorry

end roden_total_fish_l2110_211080


namespace double_neg_cancel_l2110_211048

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end double_neg_cancel_l2110_211048


namespace max_value_expr_l2110_211070

theorem max_value_expr (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
sorry

end max_value_expr_l2110_211070


namespace smallest_mn_sum_l2110_211040

theorem smallest_mn_sum {n m : ℕ} (h1 : n > m) (h2 : 1978 ^ n % 1000 = 1978 ^ m % 1000) (h3 : m ≥ 1) : m + n = 106 := 
sorry

end smallest_mn_sum_l2110_211040


namespace triangle_side_length_l2110_211046

theorem triangle_side_length (BC : ℝ) (A : ℝ) (B : ℝ) (AB : ℝ) :
  BC = 2 → A = π / 3 → B = π / 4 → AB = (3 * Real.sqrt 2 + Real.sqrt 6) / 3 :=
by
  sorry

end triangle_side_length_l2110_211046


namespace four_hash_two_equals_forty_l2110_211012

def hash_op (a b : ℕ) : ℤ := (a^2 + b^2) * (a - b)

theorem four_hash_two_equals_forty : hash_op 4 2 = 40 := 
by
  sorry

end four_hash_two_equals_forty_l2110_211012


namespace correct_exponent_operation_l2110_211022

theorem correct_exponent_operation (x : ℝ) : x ^ 3 * x ^ 2 = x ^ 5 :=
by sorry

end correct_exponent_operation_l2110_211022


namespace calculate_F_l2110_211064

def f(a : ℝ) : ℝ := a^2 - 5 * a + 6
def F(a b c : ℝ) : ℝ := b^2 + a * c + 1

theorem calculate_F : F 3 (f 3) (f 5) = 19 :=
by
  sorry

end calculate_F_l2110_211064


namespace johnson_and_carter_tie_in_september_l2110_211038

def monthly_home_runs_johnson : List ℕ := [3, 14, 18, 13, 10, 16, 14, 5]
def monthly_home_runs_carter : List ℕ := [5, 9, 22, 11, 15, 17, 9, 9]

def cumulative_home_runs (runs : List ℕ) (up_to : ℕ) : ℕ :=
  (runs.take up_to).sum

theorem johnson_and_carter_tie_in_september :
  cumulative_home_runs monthly_home_runs_johnson 7 = cumulative_home_runs monthly_home_runs_carter 7 :=
by
  sorry

end johnson_and_carter_tie_in_september_l2110_211038


namespace additional_hours_on_days_without_practice_l2110_211015

def total_weekday_homework_hours : ℕ := 2 + 3 + 4 + 3 + 1
def total_weekend_homework_hours : ℕ := 8
def total_homework_hours : ℕ := total_weekday_homework_hours + total_weekend_homework_hours
def total_chore_hours : ℕ := 1 + 1
def total_hours : ℕ := total_homework_hours + total_chore_hours

theorem additional_hours_on_days_without_practice : ∀ (practice_nights : ℕ), 
  (2 ≤ practice_nights ∧ practice_nights ≤ 3) →
  (∃ tuesday_wednesday_thursday_weekend_day_hours : ℕ,
    tuesday_wednesday_thursday_weekend_day_hours = 15) :=
by
  intros practice_nights practice_nights_bounds
  -- Define days without practice in the worst case scenario
  let tuesday_hours := 3
  let wednesday_homework_hours := 4
  let wednesday_chore_hours := 1
  let thursday_hours := 3
  let weekend_day_hours := 4
  let days_without_practice_hours := tuesday_hours + (wednesday_homework_hours + wednesday_chore_hours) + thursday_hours + weekend_day_hours
  use days_without_practice_hours
  -- In the worst case, the total additional hours on days without practice should be 15.
  sorry

end additional_hours_on_days_without_practice_l2110_211015


namespace distance_P_to_outer_circle_l2110_211099

theorem distance_P_to_outer_circle
  (r_large r_small : ℝ) 
  (h_tangent_inner : true) 
  (h_tangent_diameter : true) 
  (P : ℝ) 
  (O1P : ℝ)
  (O2P : ℝ := r_small)
  (O1O2 : ℝ := r_large - r_small)
  (h_O1O2_eq_680 : O1O2 = 680)
  (h_O2P_eq_320 : O2P = 320) :
  r_large - O1P = 400 :=
by
  sorry

end distance_P_to_outer_circle_l2110_211099


namespace mike_max_marks_l2110_211045

theorem mike_max_marks (m : ℕ) (h : 30 * m = 237 * 10) : m = 790 := by
  sorry

end mike_max_marks_l2110_211045


namespace complement_set_A_is_04_l2110_211013

theorem complement_set_A_is_04 :
  let U := {0, 1, 2, 4}
  let compA := {1, 2}
  ∃ (A : Set ℕ), A = {0, 4} ∧ U = {0, 1, 2, 4} ∧ (U \ A) = compA := 
by
  sorry

end complement_set_A_is_04_l2110_211013


namespace xiao_ming_percentile_l2110_211035

theorem xiao_ming_percentile (total_students : ℕ) (rank : ℕ) 
  (h1 : total_students = 48) (h2 : rank = 5) :
  ∃ p : ℕ, (p = 90 ∨ p = 91) ∧ (43 < (p * total_students) / 100) ∧ ((p * total_students) / 100 ≤ 44) :=
by
  sorry

end xiao_ming_percentile_l2110_211035


namespace pyramid_base_side_length_l2110_211089

theorem pyramid_base_side_length (area : ℕ) (slant_height : ℕ) (s : ℕ) 
  (h1 : area = 100) 
  (h2 : slant_height = 20) 
  (h3 : area = (1 / 2) * s * slant_height) :
  s = 10 := 
by 
  sorry

end pyramid_base_side_length_l2110_211089


namespace prove_unattainable_y_l2110_211042

noncomputable def unattainable_y : Prop :=
  ∀ (x y : ℝ), x ≠ -4 / 3 → y = (2 - x) / (3 * x + 4) → y ≠ -1 / 3

theorem prove_unattainable_y : unattainable_y :=
by
  intro x y h1 h2
  sorry

end prove_unattainable_y_l2110_211042


namespace problem_l2110_211083

theorem problem (x y : ℚ) (h1 : x + y = 10 / 21) (h2 : x - y = 1 / 63) : 
  x^2 - y^2 = 10 / 1323 := 
by 
  sorry

end problem_l2110_211083


namespace tea_price_l2110_211008

theorem tea_price 
  (x : ℝ)
  (total_cost_80kg_tea : ℝ := 80 * x)
  (total_cost_20kg_tea : ℝ := 20 * 20)
  (total_selling_price : ℝ := 1920)
  (profit_condition : 1.2 * (total_cost_80kg_tea + total_cost_20kg_tea) = total_selling_price) :
  x = 15 :=
by
  sorry

end tea_price_l2110_211008


namespace joan_total_cents_l2110_211018

-- Conditions
def quarters : ℕ := 12
def dimes : ℕ := 8
def nickels : ℕ := 15
def pennies : ℕ := 25

def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10
def value_of_nickel : ℕ := 5
def value_of_penny : ℕ := 1

-- The problem statement
theorem joan_total_cents : 
  (quarters * value_of_quarter + dimes * value_of_dime + nickels * value_of_nickel + pennies * value_of_penny) = 480 := 
  sorry

end joan_total_cents_l2110_211018


namespace ab_value_l2110_211000

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l2110_211000


namespace combinatorial_calculation_l2110_211025

-- Define the proof problem.
theorem combinatorial_calculation : (Nat.choose 20 6) = 2583 := sorry

end combinatorial_calculation_l2110_211025


namespace eq_root_condition_l2110_211031

theorem eq_root_condition (k : ℝ) 
    (h_discriminant : -4 * k + 5 ≥ 0)
    (h_roots : ∃ x1 x2 : ℝ, 
        (x1 + x2 = 1 - 2 * k) ∧ 
        (x1 * x2 = k^2 - 1) ∧ 
        (x1^2 + x2^2 = 16 + x1 * x2)) :
    k = -2 :=
sorry

end eq_root_condition_l2110_211031


namespace solve_digits_A_B_l2110_211004

theorem solve_digits_A_B :
    ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 
    (A * (10 * A + B) = 100 * B + 10 * A + A) ∧ A = 8 ∧ B = 6 :=
by
  sorry

end solve_digits_A_B_l2110_211004


namespace number_of_rows_with_exactly_7_students_l2110_211065

theorem number_of_rows_with_exactly_7_students 
  (total_students : ℕ) (rows_with_6_students rows_with_7_students : ℕ) 
  (total_students_eq : total_students = 53)
  (seats_condition : total_students = 6 * rows_with_6_students + 7 * rows_with_7_students) 
  (no_seat_unoccupied : rows_with_6_students + rows_with_7_students = rows_with_6_students + rows_with_7_students) :
  rows_with_7_students = 5 := by
  sorry

end number_of_rows_with_exactly_7_students_l2110_211065


namespace x_interval_l2110_211049

theorem x_interval (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) (h3 : 2 * x - 1 > 0) : x > 1 / 2 := 
sorry

end x_interval_l2110_211049


namespace rectangle_perimeter_l2110_211071

theorem rectangle_perimeter 
(area : ℝ) (width : ℝ) (h1 : area = 200) (h2 : width = 10) : 
    ∃ (perimeter : ℝ), perimeter = 60 :=
by
  sorry

end rectangle_perimeter_l2110_211071


namespace sum_largest_and_smallest_l2110_211068

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end sum_largest_and_smallest_l2110_211068


namespace ratio_of_couch_to_table_l2110_211020

theorem ratio_of_couch_to_table
    (C T X : ℝ)
    (h1 : T = 3 * C)
    (h2 : X = 300)
    (h3 : C + T + X = 380) :
  X / T = 5 := 
by 
  sorry

end ratio_of_couch_to_table_l2110_211020


namespace additional_interest_rate_l2110_211058

variable (P A1 A2 T SI1 SI2 R AR : ℝ)
variable (h_P : P = 9000)
variable (h_A1 : A1 = 10200)
variable (h_A2 : A2 = 10740)
variable (h_T : T = 3)
variable (h_SI1 : SI1 = A1 - P)
variable (h_SI2 : SI2 = A2 - A1)
variable (h_R : SI1 = P * R * T / 100)
variable (h_AR : SI2 = P * AR * T / 100)

theorem additional_interest_rate :
  AR = 2 := by
  sorry

end additional_interest_rate_l2110_211058


namespace BigDigMiningCopperOutput_l2110_211091

theorem BigDigMiningCopperOutput :
  (∀ (total_output : ℝ) (nickel_percentage : ℝ) (iron_percentage : ℝ) (amount_of_nickel : ℝ),
      nickel_percentage = 0.10 → 
      iron_percentage = 0.60 → 
      amount_of_nickel = 720 →
      total_output = amount_of_nickel / nickel_percentage →
      (1 - nickel_percentage - iron_percentage) * total_output = 2160) :=
sorry

end BigDigMiningCopperOutput_l2110_211091


namespace merchants_and_cost_l2110_211066

theorem merchants_and_cost (n C : ℕ) (h1 : 8 * n = C + 3) (h2 : 7 * n = C - 4) : n = 7 ∧ C = 53 := 
by 
  sorry

end merchants_and_cost_l2110_211066


namespace solve_for_x_l2110_211007

noncomputable def log_b (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (b x : ℝ) (hb : b > 1) (hx : x > 0) :
  (4 * x) ^ log_b b 4 - (5 * x) ^ log_b b 5 + x = 0 ↔ x = 1 :=
by
  -- Proof placeholder
  sorry

end solve_for_x_l2110_211007


namespace staff_discount_l2110_211092

theorem staff_discount (d : ℝ) (S : ℝ) (h1 : d > 0)
    (h2 : 0.455 * d = (1 - S / 100) * (0.65 * d)) : S = 30 := by
    sorry

end staff_discount_l2110_211092


namespace find_expression_value_find_m_value_find_roots_and_theta_l2110_211090

-- Define the conditions
variable (θ : ℝ) (m : ℝ)
variable (h1 : θ > 0) (h2 : θ < 2 * Real.pi)
variable (h3 : ∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) → (x = Real.sin θ ∨ x = Real.cos θ))

-- Theorem 1: Find the value of a given expression
theorem find_expression_value :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (Real.sqrt 3 + 1) / 2 :=
  sorry

-- Theorem 2: Find the value of m
theorem find_m_value :
  m = Real.sqrt 3 / 2 :=
  sorry

-- Theorem 3: Find the roots of the equation and the value of θ
theorem find_roots_and_theta :
  (∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + Real.sqrt 3 / 2 = 0) → (x = Real.sqrt 3 / 2 ∨ x = 1 / 2)) ∧
  (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
  sorry

end find_expression_value_find_m_value_find_roots_and_theta_l2110_211090


namespace unique_solution_k_l2110_211094

theorem unique_solution_k (k : ℝ) :
  (∀ x : ℝ, (x + 3) / (k * x + 2) = x) ↔ (k = -1 / 12) :=
  sorry

end unique_solution_k_l2110_211094


namespace smallest_next_divisor_l2110_211014

theorem smallest_next_divisor (n : ℕ) (h_even : n % 2 = 0) (h_4_digit : 1000 ≤ n ∧ n < 10000) (h_div_493 : 493 ∣ n) :
  ∃ d : ℕ, (d > 493 ∧ d ∣ n) ∧ ∀ e, (e > 493 ∧ e ∣ n) → d ≤ e ∧ d = 510 := by
  sorry

end smallest_next_divisor_l2110_211014
