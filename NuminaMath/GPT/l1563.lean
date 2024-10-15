import Mathlib

namespace NUMINAMATH_GPT_sandwiches_difference_l1563_156306

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end NUMINAMATH_GPT_sandwiches_difference_l1563_156306


namespace NUMINAMATH_GPT_binary_to_decimal_l1563_156394

theorem binary_to_decimal :
  1 * 2^8 + 0 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 379 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1563_156394


namespace NUMINAMATH_GPT_calculate_total_feet_in_garden_l1563_156327

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_feet_in_garden_l1563_156327


namespace NUMINAMATH_GPT_a_2n_perfect_square_l1563_156373

-- Define the sequence a_n following the described recurrence relation.
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n-1) + a (n-3) + a (n-4)

-- Define the main theorem to prove
theorem a_2n_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k := by
  sorry

end NUMINAMATH_GPT_a_2n_perfect_square_l1563_156373


namespace NUMINAMATH_GPT_right_triangle_area_l1563_156391

theorem right_triangle_area (a : ℝ) (r : ℝ) (area : ℝ) :
  a = 3 → r = 3 / 8 → area = 21 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1563_156391


namespace NUMINAMATH_GPT_sum_of_80_consecutive_integers_l1563_156322

-- Definition of the problem using the given conditions
theorem sum_of_80_consecutive_integers (n : ℤ) (h : (80 * (n + (n + 79))) / 2 = 40) : n = -39 := by
  sorry

end NUMINAMATH_GPT_sum_of_80_consecutive_integers_l1563_156322


namespace NUMINAMATH_GPT_circle_integer_solution_max_sum_l1563_156308

theorem circle_integer_solution_max_sum : ∀ (x y : ℤ), (x - 1)^2 + (y + 2)^2 = 16 → x + y ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_integer_solution_max_sum_l1563_156308


namespace NUMINAMATH_GPT_beds_with_fewer_beds_l1563_156307

theorem beds_with_fewer_beds:
  ∀ (total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x : ℕ),
    total_rooms = 13 →
    rooms_with_fewer_beds = 8 →
    rooms_with_three_beds = total_rooms - rooms_with_fewer_beds →
    total_beds = 31 →
    8 * x + 3 * (total_rooms - rooms_with_fewer_beds) = total_beds →
    x = 2 :=
by
  intros total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x
  intros ht_rooms hrwb hrwtb htb h_eq
  sorry

end NUMINAMATH_GPT_beds_with_fewer_beds_l1563_156307


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l1563_156323

theorem sqrt_meaningful_range (x : ℝ): x + 2 ≥ 0 ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l1563_156323


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1563_156330

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℕ), a 1 = 2 ∧ a 2 + a 3 = 13 → a 4 + a 5 + a 6 = 42 :=
by
  intro a
  intro h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1563_156330


namespace NUMINAMATH_GPT_tan_45_deg_l1563_156378

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_45_deg_l1563_156378


namespace NUMINAMATH_GPT_inequality_always_true_l1563_156372

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end NUMINAMATH_GPT_inequality_always_true_l1563_156372


namespace NUMINAMATH_GPT_polynomial_evaluation_l1563_156318

-- Given the value of y
def y : ℤ := 4

-- Our goal is to prove this mathematical statement
theorem polynomial_evaluation : (3 * (y ^ 2) + 4 * y + 2 = 66) := 
by 
    sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1563_156318


namespace NUMINAMATH_GPT_cuboid_third_face_area_l1563_156332

-- Problem statement in Lean
theorem cuboid_third_face_area (l w h : ℝ) (A₁ A₂ V : ℝ) 
  (hw1 : l * w = 120)
  (hw2 : w * h = 60)
  (hw3 : l * w * h = 720) : 
  l * h = 72 :=
sorry

end NUMINAMATH_GPT_cuboid_third_face_area_l1563_156332


namespace NUMINAMATH_GPT_evaluate_triangle_l1563_156336

def triangle_op (a b : Int) : Int :=
  a * b - a - b + 1

theorem evaluate_triangle :
  triangle_op (-3) 4 = -12 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_triangle_l1563_156336


namespace NUMINAMATH_GPT_first_number_percentage_of_second_l1563_156393

theorem first_number_percentage_of_second (X : ℝ) (h1 : First = 0.06 * X) (h2 : Second = 0.18 * X) : 
  (First / Second) * 100 = 33.33 := 
by 
  sorry

end NUMINAMATH_GPT_first_number_percentage_of_second_l1563_156393


namespace NUMINAMATH_GPT_gretel_hansel_salary_difference_l1563_156333

theorem gretel_hansel_salary_difference :
  let hansel_initial_salary := 30000
  let hansel_raise_percentage := 10
  let gretel_initial_salary := 30000
  let gretel_raise_percentage := 15
  let hansel_new_salary := hansel_initial_salary + (hansel_raise_percentage / 100 * hansel_initial_salary)
  let gretel_new_salary := gretel_initial_salary + (gretel_raise_percentage / 100 * gretel_initial_salary)
  gretel_new_salary - hansel_new_salary = 1500 := sorry

end NUMINAMATH_GPT_gretel_hansel_salary_difference_l1563_156333


namespace NUMINAMATH_GPT_each_person_gets_9_apples_l1563_156315

-- Define the initial number of apples and the number of apples given to Jack's father
def initial_apples : ℕ := 55
def apples_given_to_father : ℕ := 10

-- Define the remaining apples after giving to Jack's father
def remaining_apples : ℕ := initial_apples - apples_given_to_father

-- Define the number of people sharing the remaining apples
def number_of_people : ℕ := 1 + 4

-- Define the number of apples each person will get
def apples_per_person : ℕ := remaining_apples / number_of_people

-- Prove that each person gets 9 apples
theorem each_person_gets_9_apples (h₁ : initial_apples = 55) 
                                  (h₂ : apples_given_to_father = 10) 
                                  (h₃ : number_of_people = 5) 
                                  (h₄ : remaining_apples = initial_apples - apples_given_to_father) 
                                  (h₅ : apples_per_person = remaining_apples / number_of_people) : 
  apples_per_person = 9 :=
by sorry

end NUMINAMATH_GPT_each_person_gets_9_apples_l1563_156315


namespace NUMINAMATH_GPT_BothNormal_l1563_156316

variable (Normal : Type) (Person : Type) (MrA MrsA : Person)
variables (isNormal : Person → Prop)

-- Conditions given in the problem
axiom MrA_statement : ∀ p : Person, p = MrsA → isNormal MrA → isNormal MrsA
axiom MrsA_statement : ∀ p : Person, p = MrA → isNormal MrsA → isNormal MrA

-- Question (translated to proof problem): 
-- prove that Mr. A and Mrs. A are both normal persons
theorem BothNormal : isNormal MrA ∧ isNormal MrsA := 
  by 
    sorry -- proof is omitted

end NUMINAMATH_GPT_BothNormal_l1563_156316


namespace NUMINAMATH_GPT_polynomial_is_perfect_square_trinomial_l1563_156343

-- The definition of a perfect square trinomial
def isPerfectSquareTrinomial (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b = c ∧ 4 * a * a + m * b = 4 * a * b * b

-- The main theorem to prove that if the polynomial is a perfect square trinomial, then m = 20
theorem polynomial_is_perfect_square_trinomial (a b : ℝ) (h : isPerfectSquareTrinomial 2 1 5 25) :
  ∀ x, (4 * x * x + 20 * x + 25 = (2 * x + 5) * (2 * x + 5)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_is_perfect_square_trinomial_l1563_156343


namespace NUMINAMATH_GPT_tuesday_pairs_of_boots_l1563_156320

theorem tuesday_pairs_of_boots (S B : ℝ) (x : ℤ) 
  (h1 : 22 * S + 16 * B = 460)
  (h2 : 8 * S + x * B = 560)
  (h3 : B = S + 15) : 
  x = 24 :=
sorry

end NUMINAMATH_GPT_tuesday_pairs_of_boots_l1563_156320


namespace NUMINAMATH_GPT_max_value_of_f_l1563_156371

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l1563_156371


namespace NUMINAMATH_GPT_part1_part2_l1563_156326

-- Part (1)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < -1 / 2 → (ax - 1) * (x + 1) > 0) →
  a = -2 :=
sorry

-- Part (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ,
    ((a < -1 ∧ -1 < x ∧ x < 1/a) ∨
     (a = -1 ∧ ∀ x : ℝ, false) ∨
     (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
     (a = 0 ∧ x < -1) ∨
     (a > 0 ∧ (x < -1 ∨ x > 1/a))) →
    (ax - 1) * (x + 1) > 0) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1563_156326


namespace NUMINAMATH_GPT_min_AB_distance_l1563_156358

theorem min_AB_distance : 
  ∀ (A B : ℝ × ℝ), 
  A ≠ B → 
  ((∃ (m : ℝ), A.2 = m * (A.1 - 1) + 1 ∧ B.2 = m * (B.1 - 1) + 1) ∧ 
    ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ 
    ((B.1 - 2)^2 + (B.2 - 3)^2 = 9)) → 
  dist A B = 4 :=
sorry

end NUMINAMATH_GPT_min_AB_distance_l1563_156358


namespace NUMINAMATH_GPT_kelly_snacks_l1563_156304

theorem kelly_snacks (peanuts raisins : ℝ) (h_peanuts : peanuts = 0.1) (h_raisins : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_kelly_snacks_l1563_156304


namespace NUMINAMATH_GPT_m_plus_n_sum_l1563_156392

theorem m_plus_n_sum :
  let m := 271
  let n := 273
  m + n = 544 :=
by {
  -- sorry included to skip the proof steps
  sorry
}

end NUMINAMATH_GPT_m_plus_n_sum_l1563_156392


namespace NUMINAMATH_GPT_molecular_weight_correct_l1563_156300

def potassium_weight : ℝ := 39.10
def chromium_weight : ℝ := 51.996
def oxygen_weight : ℝ := 16.00

def num_potassium_atoms : ℕ := 2
def num_chromium_atoms : ℕ := 2
def num_oxygen_atoms : ℕ := 7

def molecular_weight_of_compound : ℝ :=
  (num_potassium_atoms * potassium_weight) +
  (num_chromium_atoms * chromium_weight) +
  (num_oxygen_atoms * oxygen_weight)

theorem molecular_weight_correct :
  molecular_weight_of_compound = 294.192 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l1563_156300


namespace NUMINAMATH_GPT_john_cards_l1563_156334

theorem john_cards (C : ℕ) (h1 : 15 * 2 + C * 2 = 70) : C = 20 :=
by
  sorry

end NUMINAMATH_GPT_john_cards_l1563_156334


namespace NUMINAMATH_GPT_distance_after_12_seconds_time_to_travel_380_meters_l1563_156353

def distance_travelled (t : ℝ) : ℝ := 9 * t + 0.5 * t^2

theorem distance_after_12_seconds : distance_travelled 12 = 180 :=
by 
  sorry

theorem time_to_travel_380_meters : ∃ t : ℝ, distance_travelled t = 380 ∧ t = 20 :=
by 
  sorry

end NUMINAMATH_GPT_distance_after_12_seconds_time_to_travel_380_meters_l1563_156353


namespace NUMINAMATH_GPT_rectangle_new_area_l1563_156384

theorem rectangle_new_area
  (L W : ℝ) (h1 : L * W = 600) :
  let L' := 0.8 * L
  let W' := 1.3 * W
  (L' * W' = 624) :=
by
  -- Let L' = 0.8 * L
  -- Let W' = 1.3 * W
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rectangle_new_area_l1563_156384


namespace NUMINAMATH_GPT_sum_of_four_digit_numbers_l1563_156368

theorem sum_of_four_digit_numbers :
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324 :=
by
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  show (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324
  sorry

end NUMINAMATH_GPT_sum_of_four_digit_numbers_l1563_156368


namespace NUMINAMATH_GPT_travel_distance_l1563_156319

-- Define the conditions
def distance_10_gallons := 300 -- 300 miles on 10 gallons of fuel
def gallons_10 := 10 -- 10 gallons

-- Given the distance per gallon, calculate the distance for 15 gallons
def distance_per_gallon := distance_10_gallons / gallons_10

def gallons_15 := 15 -- 15 gallons

def distance_15_gallons := distance_per_gallon * gallons_15

-- Proof statement
theorem travel_distance (d_10 : distance_10_gallons = 300)
                        (g_10 : gallons_10 = 10)
                        (g_15 : gallons_15 = 15) :
  distance_15_gallons = 450 :=
  by
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_travel_distance_l1563_156319


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1563_156357

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1563_156357


namespace NUMINAMATH_GPT_interval_for_systematic_sampling_l1563_156397

-- Define the total population size
def total_population : ℕ := 1203

-- Define the sample size
def sample_size : ℕ := 40

-- Define the interval for systematic sampling
def interval (n m : ℕ) : ℕ := (n - (n % m)) / m

-- The proof statement that the interval \( k \) for segmenting is 30
theorem interval_for_systematic_sampling : interval total_population sample_size = 30 :=
by
  show interval 1203 40 = 30
  sorry

end NUMINAMATH_GPT_interval_for_systematic_sampling_l1563_156397


namespace NUMINAMATH_GPT_ellipse_foci_on_y_axis_l1563_156341

theorem ellipse_foci_on_y_axis (k : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x y, x^2 + k * y^2 = 2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ b^2 > a^2)
  → (0 < k ∧ k < 1) :=
sorry

end NUMINAMATH_GPT_ellipse_foci_on_y_axis_l1563_156341


namespace NUMINAMATH_GPT_q_f_digit_div_36_l1563_156356

theorem q_f_digit_div_36 (q f : ℕ) (hq : q ≠ f) (hq_digit: q < 10) (hf_digit: f < 10) :
    (457 * 10000 + q * 1000 + 89 * 10 + f) % 36 = 0 → q + f = 6 :=
sorry

end NUMINAMATH_GPT_q_f_digit_div_36_l1563_156356


namespace NUMINAMATH_GPT_largest_of_three_l1563_156399

theorem largest_of_three (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : ab + ac + bc = -8) 
  (h3 : abc = -20) : 
  max a (max b c) = (1 + Real.sqrt 41) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_largest_of_three_l1563_156399


namespace NUMINAMATH_GPT_binomial_probability_p_l1563_156325

noncomputable def binomial_expected_value (n p : ℝ) := n * p
noncomputable def binomial_variance (n p : ℝ) := n * p * (1 - p)

theorem binomial_probability_p (n p : ℝ) (h1: binomial_expected_value n p = 2) (h2: binomial_variance n p = 1) : 
  p = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_binomial_probability_p_l1563_156325


namespace NUMINAMATH_GPT_bhanu_house_rent_l1563_156340

theorem bhanu_house_rent (I : ℝ) 
  (h1 : 0.30 * I = 300) 
  (h2 : 210 = 210) : 
  210 / (I - 300) = 0.30 := 
by 
  sorry

end NUMINAMATH_GPT_bhanu_house_rent_l1563_156340


namespace NUMINAMATH_GPT_ratio_of_areas_l1563_156390

structure Triangle :=
  (AB BC AC AD AE : ℝ)
  (AB_pos : 0 < AB)
  (BC_pos : 0 < BC)
  (AC_pos : 0 < AC)
  (AD_pos : 0 < AD)
  (AE_pos : 0 < AE)

theorem ratio_of_areas (t : Triangle)
  (hAB : t.AB = 30)
  (hBC : t.BC = 45)
  (hAC : t.AC = 54)
  (hAD : t.AD = 24)
  (hAE : t.AE = 18) :
  (t.AD / t.AB) * (t.AE / t.AC) / (1 - (t.AD / t.AB) * (t.AE / t.AC)) = 4 / 11 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1563_156390


namespace NUMINAMATH_GPT_yield_difference_correct_l1563_156328

noncomputable def tomato_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def corn_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def onion_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def carrot_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)

theorem yield_difference_correct :
  let tomato_initial := 2073
  let corn_initial := 4112
  let onion_initial := 985
  let carrot_initial := 6250
  let tomato_growth := 12
  let corn_growth := 15
  let onion_growth := 8
  let carrot_growth := 10
  let tomato_total := tomato_yield tomato_initial tomato_growth
  let corn_total := corn_yield corn_initial corn_growth
  let onion_total := onion_yield onion_initial onion_growth
  let carrot_total := carrot_yield carrot_initial carrot_growth
  let highest_yield := max (max tomato_total corn_total) (max onion_total carrot_total)
  let lowest_yield := min (min tomato_total corn_total) (min onion_total carrot_total)
  highest_yield - lowest_yield = 5811.2 := by
  sorry

end NUMINAMATH_GPT_yield_difference_correct_l1563_156328


namespace NUMINAMATH_GPT_find_f_l1563_156369

theorem find_f (q f : ℕ) (h_digit_q : q ≤ 9) (h_digit_f : f ≤ 9)
  (h_distinct : q ≠ f) 
  (h_div_by_36 : (457 * 1000 + q * 100 + 89 * 10 + f) % 36 = 0)
  (h_sum_3 : q + f = 3) :
  f = 2 :=
sorry

end NUMINAMATH_GPT_find_f_l1563_156369


namespace NUMINAMATH_GPT_mikails_age_l1563_156395

-- Define the conditions
def dollars_per_year_old : ℕ := 5
def total_dollars_given : ℕ := 45

-- Main theorem statement
theorem mikails_age (age : ℕ) : (age * dollars_per_year_old = total_dollars_given) → age = 9 :=
by
  sorry

end NUMINAMATH_GPT_mikails_age_l1563_156395


namespace NUMINAMATH_GPT_sum_eq_zero_l1563_156385

variable {a b c : ℝ}

theorem sum_eq_zero (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
    (h4 : a ≠ b ∨ b ≠ c ∨ c ≠ a)
    (h5 : (a^2) / (2 * (a^2) + b * c) + (b^2) / (2 * (b^2) + c * a) + (c^2) / (2 * (c^2) + a * b) = 1) :
  a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_sum_eq_zero_l1563_156385


namespace NUMINAMATH_GPT_cars_to_sell_l1563_156367

theorem cars_to_sell (clients : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) (total_clients : ℕ) (h1 : selections_per_client = 2) 
  (h2 : selections_per_car = 3) (h3 : total_clients = 24) : (total_clients * selections_per_client / selections_per_car = 16) :=
by
  sorry

end NUMINAMATH_GPT_cars_to_sell_l1563_156367


namespace NUMINAMATH_GPT_initial_violet_marbles_eq_l1563_156361

variable {initial_violet_marbles : Nat}
variable (red_marbles : Nat := 14)
variable (total_marbles : Nat := 78)

theorem initial_violet_marbles_eq :
  initial_violet_marbles = total_marbles - red_marbles := by
  sorry

end NUMINAMATH_GPT_initial_violet_marbles_eq_l1563_156361


namespace NUMINAMATH_GPT_sin_of_300_degrees_l1563_156387

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_sin_of_300_degrees_l1563_156387


namespace NUMINAMATH_GPT_decorations_cost_correct_l1563_156350

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end NUMINAMATH_GPT_decorations_cost_correct_l1563_156350


namespace NUMINAMATH_GPT_chores_minutes_proof_l1563_156337

-- Definitions based on conditions
def minutes_of_cartoon_per_hour := 60
def cartoon_watched_hours := 2
def cartoon_watched_minutes := cartoon_watched_hours * minutes_of_cartoon_per_hour
def ratio_of_cartoon_to_chores := 10 / 8

-- Definition based on the question
def chores_minutes (cartoon_minutes : ℕ) : ℕ := (8 * cartoon_minutes) / 10

theorem chores_minutes_proof : chores_minutes cartoon_watched_minutes = 96 := 
by sorry 

end NUMINAMATH_GPT_chores_minutes_proof_l1563_156337


namespace NUMINAMATH_GPT_no_500_good_trinomials_l1563_156311

def is_good_quadratic_trinomial (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b^2 - 4 * a * c) > 0

theorem no_500_good_trinomials (S : Finset ℤ) (hS: S.card = 10)
  (hs_pos: ∀ x ∈ S, x > 0) : ¬(∃ T : Finset (ℤ × ℤ × ℤ), 
  T.card = 500 ∧ (∀ (a b c : ℤ), (a, b, c) ∈ T → is_good_quadratic_trinomial a b c)) :=
by
  sorry

end NUMINAMATH_GPT_no_500_good_trinomials_l1563_156311


namespace NUMINAMATH_GPT_line_ellipse_tangent_l1563_156396

theorem line_ellipse_tangent (m : ℝ) : 
  (∀ x y : ℝ, (y = m * x + 2) → (x^2 + (y^2 / 4) = 1)) → m^2 = 0 :=
sorry

end NUMINAMATH_GPT_line_ellipse_tangent_l1563_156396


namespace NUMINAMATH_GPT_cos_angle_identity_l1563_156338

theorem cos_angle_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_angle_identity_l1563_156338


namespace NUMINAMATH_GPT_num_3_digit_multiples_l1563_156345

def is_3_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999
def multiple_of (k n : Nat) : Prop := ∃ m : Nat, n = m * k

theorem num_3_digit_multiples (count_35_not_70 : Nat) (h : count_35_not_70 = 13) :
  let count_multiples_35 := (980 / 35) - (105 / 35) + 1
  let count_multiples_70 := (980 / 70) - (140 / 70) + 1
  count_multiples_35 - count_multiples_70 = count_35_not_70 := sorry

end NUMINAMATH_GPT_num_3_digit_multiples_l1563_156345


namespace NUMINAMATH_GPT_solve_inequality_l1563_156375

theorem solve_inequality :
  { x : ℝ | (x - 5) / (x - 3)^2 < 0 } = { x : ℝ | x < 3 } ∪ { x : ℝ | 3 < x ∧ x < 5 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1563_156375


namespace NUMINAMATH_GPT_expression_undefined_at_9_l1563_156342

theorem expression_undefined_at_9 (x : ℝ) : (3 * x ^ 3 - 5) / (x ^ 2 - 18 * x + 81) = 0 → x = 9 :=
by sorry

end NUMINAMATH_GPT_expression_undefined_at_9_l1563_156342


namespace NUMINAMATH_GPT_georgia_coughs_5_times_per_minute_l1563_156321

-- Definitions
def georgia_coughs_per_minute (G : ℕ) := true
def robert_coughs_per_minute (G : ℕ) := 2 * G
def total_coughs (G : ℕ) := 20 * (G + 2 * G) = 300

-- Theorem to prove
theorem georgia_coughs_5_times_per_minute (G : ℕ) 
  (h1 : georgia_coughs_per_minute G) 
  (h2 : robert_coughs_per_minute G = 2 * G) 
  (h3 : total_coughs G) : G = 5 := 
sorry

end NUMINAMATH_GPT_georgia_coughs_5_times_per_minute_l1563_156321


namespace NUMINAMATH_GPT_sandwich_cost_proof_l1563_156359

/-- Definitions of ingredient costs and quantities. --/
def bread_cost : ℝ := 0.15
def ham_cost : ℝ := 0.25
def cheese_cost : ℝ := 0.35
def mayo_cost : ℝ := 0.10
def lettuce_cost : ℝ := 0.05
def tomato_cost : ℝ := 0.08

def num_bread_slices : ℕ := 2
def num_ham_slices : ℕ := 2
def num_cheese_slices : ℕ := 2
def num_mayo_tbsp : ℕ := 1
def num_lettuce_leaf : ℕ := 1
def num_tomato_slices : ℕ := 2

/-- Calculation of the total cost in dollars and conversion to cents. --/
def sandwich_cost_in_dollars : ℝ :=
  (num_bread_slices * bread_cost) + 
  (num_ham_slices * ham_cost) + 
  (num_cheese_slices * cheese_cost) + 
  (num_mayo_tbsp * mayo_cost) + 
  (num_lettuce_leaf * lettuce_cost) + 
  (num_tomato_slices * tomato_cost)

def sandwich_cost_in_cents : ℝ :=
  sandwich_cost_in_dollars * 100

/-- Prove that the cost of the sandwich in cents is 181. --/
theorem sandwich_cost_proof : sandwich_cost_in_cents = 181 := by
  sorry

end NUMINAMATH_GPT_sandwich_cost_proof_l1563_156359


namespace NUMINAMATH_GPT_distance_third_day_l1563_156313

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end NUMINAMATH_GPT_distance_third_day_l1563_156313


namespace NUMINAMATH_GPT_keith_attended_games_l1563_156301

-- Definitions based on the given conditions
def total_games : ℕ := 8
def missed_games : ℕ := 4

-- The proof goal: Keith's attendance
def attended_games : ℕ := total_games - missed_games

-- Main statement to prove the total games Keith attended
theorem keith_attended_games : attended_games = 4 := by
  -- Sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_keith_attended_games_l1563_156301


namespace NUMINAMATH_GPT_number_of_sandwiches_l1563_156379

-- Definitions based on conditions
def breads : Nat := 5
def meats : Nat := 7
def cheeses : Nat := 6
def total_sandwiches : Nat := breads * meats * cheeses
def turkey_mozzarella_exclusions : Nat := breads
def rye_beef_exclusions : Nat := cheeses

-- The proof problem statement
theorem number_of_sandwiches (total_sandwiches := 210) 
  (turkey_mozzarella_exclusions := 5) 
  (rye_beef_exclusions := 6) : 
  total_sandwiches - turkey_mozzarella_exclusions - rye_beef_exclusions = 199 := 
by sorry

end NUMINAMATH_GPT_number_of_sandwiches_l1563_156379


namespace NUMINAMATH_GPT_trig_expression_eq_zero_l1563_156305

theorem trig_expression_eq_zero (α : ℝ) (h1 : Real.sin α = -2 / Real.sqrt 5) (h2 : Real.cos α = 1 / Real.sqrt 5) :
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 0 := by
  sorry

end NUMINAMATH_GPT_trig_expression_eq_zero_l1563_156305


namespace NUMINAMATH_GPT_remainder_of_65_power_65_plus_65_mod_97_l1563_156360

theorem remainder_of_65_power_65_plus_65_mod_97 :
  (65^65 + 65) % 97 = 33 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_65_power_65_plus_65_mod_97_l1563_156360


namespace NUMINAMATH_GPT_faye_science_problems_l1563_156302

variable (total_problems math_problems science_problems : Nat)
variable (finished_at_school left_for_homework : Nat)

theorem faye_science_problems :
  finished_at_school = 40 ∧ left_for_homework = 15 ∧ math_problems = 46 →
  total_problems = finished_at_school + left_for_homework →
  science_problems = total_problems - math_problems →
  science_problems = 9 :=
by
  sorry

end NUMINAMATH_GPT_faye_science_problems_l1563_156302


namespace NUMINAMATH_GPT_problem_statement_l1563_156363

variable (a b c : ℝ)
variable (x : ℝ)

theorem problem_statement (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1 :=
by
  intros x hx
  let f := fun x => a * x^2 - b * x + c
  let g := fun x => (a + b) * x^2 + c
  have h1 : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |f x| < 1 := h
  sorry

end NUMINAMATH_GPT_problem_statement_l1563_156363


namespace NUMINAMATH_GPT_smallest_value_of_diff_l1563_156331

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

end NUMINAMATH_GPT_smallest_value_of_diff_l1563_156331


namespace NUMINAMATH_GPT_total_books_for_girls_l1563_156376

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end NUMINAMATH_GPT_total_books_for_girls_l1563_156376


namespace NUMINAMATH_GPT_inequality_solution_l1563_156377

theorem inequality_solution (x : ℝ) : |2 * x - 7| < 3 → 2 < x ∧ x < 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1563_156377


namespace NUMINAMATH_GPT_rahul_spends_10_percent_on_clothes_l1563_156398

theorem rahul_spends_10_percent_on_clothes 
    (salary : ℝ) (house_rent_percent : ℝ) (education_percent : ℝ) (remaining_after_expense : ℝ) (expenses : ℝ) (clothes_percent : ℝ) 
    (h_salary : salary = 2125) 
    (h_house_rent_percent : house_rent_percent = 0.20)
    (h_education_percent : education_percent = 0.10)
    (h_remaining_after_expense : remaining_after_expense = 1377)
    (h_expenses : expenses = salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)
    (h_clothes_expense : remaining_after_expense = salary - (salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)) :
    clothes_percent = 0.10 := 
by 
  sorry

end NUMINAMATH_GPT_rahul_spends_10_percent_on_clothes_l1563_156398


namespace NUMINAMATH_GPT_no_integer_roots_l1563_156370

theorem no_integer_roots (x : ℤ) : ¬ (x^3 - 5 * x^2 - 11 * x + 35 = 0) := 
sorry

end NUMINAMATH_GPT_no_integer_roots_l1563_156370


namespace NUMINAMATH_GPT_meeting_point_l1563_156365

/-- Along a straight alley with 400 streetlights placed at equal intervals, numbered consecutively from 1 to 400,
    Alla and Boris set out towards each other from opposite ends of the alley with different constant speeds.
    Alla starts at streetlight number 1 and Boris starts at streetlight number 400. When Alla is at the 55th streetlight,
    Boris is at the 321st streetlight. The goal is to prove that they will meet at the 163rd streetlight.
-/
theorem meeting_point (n : ℕ) (h1 : n = 400) (h2 : ∀ i j k l : ℕ, i = 55 → j = 321 → k = 1 → l = 400) : 
  ∃ m, m = 163 := 
by
  sorry

end NUMINAMATH_GPT_meeting_point_l1563_156365


namespace NUMINAMATH_GPT_find_r_l1563_156329

theorem find_r 
  (r s : ℝ)
  (h1 : 9 * (r * r) * s = -6)
  (h2 : r * r + 2 * r * s = -16 / 3)
  (h3 : 2 * r + s = 2 / 3)
  (polynomial_condition : ∀ x : ℝ, 9 * x^3 - 6 * x^2 - 48 * x + 54 = 9 * (x - r)^2 * (x - s)) 
: r = -2 / 3 :=
sorry

end NUMINAMATH_GPT_find_r_l1563_156329


namespace NUMINAMATH_GPT_range_of_m_l1563_156314

theorem range_of_m (a b m : ℝ) (h1 : 2 * b = 2 * a + b) (h2 : b * b = a * a * b) (h3 : 0 < Real.log b / Real.log m) (h4 : Real.log b / Real.log m < 1) : m > 8 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1563_156314


namespace NUMINAMATH_GPT_problem_statement_l1563_156364

-- Define the binary operation "*"
def custom_mul (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the problem with the conditions
theorem problem_statement : custom_mul 5 (-3) = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1563_156364


namespace NUMINAMATH_GPT_tan_2A_cos_pi3_minus_A_l1563_156352

variable (A : ℝ)

def line_equation (A : ℝ) : Prop :=
  (4 * Real.tan A = 3)

theorem tan_2A : line_equation A → Real.tan (2 * A) = -24 / 7 :=
by
  intro h 
  sorry

theorem cos_pi3_minus_A : (0 < A ∧ A < Real.pi) →
    Real.tan A = 4 / 3 →
    Real.cos (Real.pi / 3 - A) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_tan_2A_cos_pi3_minus_A_l1563_156352


namespace NUMINAMATH_GPT_meters_to_centimeters_l1563_156383

theorem meters_to_centimeters : (3.5 : ℝ) * 100 = 350 :=
by
  sorry

end NUMINAMATH_GPT_meters_to_centimeters_l1563_156383


namespace NUMINAMATH_GPT_lily_remaining_milk_l1563_156312

def initial_milk : ℚ := (11 / 2)
def given_away : ℚ := (17 / 4)
def remaining_milk : ℚ := initial_milk - given_away

theorem lily_remaining_milk : remaining_milk = 5 / 4 :=
by
  -- Here, we would provide the proof steps, but we can use sorry to skip it.
  exact sorry

end NUMINAMATH_GPT_lily_remaining_milk_l1563_156312


namespace NUMINAMATH_GPT_cost_of_math_books_l1563_156339

theorem cost_of_math_books (M : ℕ) : 
  (∃ (total_books math_books history_books total_cost : ℕ),
    total_books = 90 ∧
    math_books = 60 ∧
    history_books = total_books - math_books ∧
    history_books * 5 + math_books * M = total_cost ∧
    total_cost = 390) → 
  M = 4 :=
by
  -- We provide the assumed conditions
  intro h
  -- We will skip the proof with sorry
  sorry

end NUMINAMATH_GPT_cost_of_math_books_l1563_156339


namespace NUMINAMATH_GPT_zero_in_interval_l1563_156346

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := -2 * x + 3
noncomputable def h (x : ℝ) : ℝ := f x + 2 * x - 3

theorem zero_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ h x = 0 := 
sorry

end NUMINAMATH_GPT_zero_in_interval_l1563_156346


namespace NUMINAMATH_GPT_ratio_x_y_l1563_156347

theorem ratio_x_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l1563_156347


namespace NUMINAMATH_GPT_cougar_sleep_hours_l1563_156355

-- Definitions
def total_sleep_hours (C Z : Nat) : Prop :=
  C + Z = 70

def zebra_cougar_difference (C Z : Nat) : Prop :=
  Z = C + 2

-- Theorem statement
theorem cougar_sleep_hours :
  ∃ C : Nat, ∃ Z : Nat, zebra_cougar_difference C Z ∧ total_sleep_hours C Z ∧ C = 34 :=
sorry

end NUMINAMATH_GPT_cougar_sleep_hours_l1563_156355


namespace NUMINAMATH_GPT_problem_solution_l1563_156386

open Real

noncomputable def length_and_slope_MP 
    (length_MN : ℝ) 
    (slope_MN : ℝ) 
    (length_NP : ℝ) 
    (slope_NP : ℝ) 
    : (ℝ × ℝ) := sorry

theorem problem_solution :
  length_and_slope_MP 6 14 7 8 = (5.55, 25.9) :=
  sorry

end NUMINAMATH_GPT_problem_solution_l1563_156386


namespace NUMINAMATH_GPT_smallest_side_length_1008_l1563_156382

def smallest_side_length_original_square :=
  let n := Nat.lcm 7 8
  let n := Nat.lcm n 9
  let lcm := Nat.lcm n 10
  2 * lcm

theorem smallest_side_length_1008 :
  smallest_side_length_original_square = 1008 := by
  sorry

end NUMINAMATH_GPT_smallest_side_length_1008_l1563_156382


namespace NUMINAMATH_GPT_students_use_red_color_l1563_156324

theorem students_use_red_color
  (total_students : ℕ)
  (students_use_green : ℕ)
  (students_use_both : ℕ)
  (total_students_eq : total_students = 70)
  (students_use_green_eq : students_use_green = 52)
  (students_use_both_eq : students_use_both = 38) :
  ∃ (students_use_red : ℕ), students_use_red = 56 :=
by
  -- We will skip the proof part as specified
  sorry

end NUMINAMATH_GPT_students_use_red_color_l1563_156324


namespace NUMINAMATH_GPT_FirstCandidatePercentage_l1563_156354

noncomputable def percentage_of_first_candidate_marks (PassingMarks TotalMarks MarksFirstCandidate : ℝ) :=
  (MarksFirstCandidate / TotalMarks) * 100

theorem FirstCandidatePercentage 
  (PassingMarks TotalMarks MarksFirstCandidate : ℝ)
  (h1 : PassingMarks = 200)
  (h2 : 0.45 * TotalMarks = PassingMarks + 25)
  (h3 : MarksFirstCandidate = PassingMarks - 50)
  : percentage_of_first_candidate_marks PassingMarks TotalMarks MarksFirstCandidate = 30 :=
sorry

end NUMINAMATH_GPT_FirstCandidatePercentage_l1563_156354


namespace NUMINAMATH_GPT_max_visible_sum_l1563_156310

-- Definitions for the problem conditions

def numbers : List ℕ := [1, 3, 6, 12, 24, 48]

def num_faces (cubes : List ℕ) : Prop :=
  cubes.length = 18 -- since each of 3 cubes has 6 faces, we expect 18 numbers in total.

def is_valid_cube (cube : List ℕ) : Prop :=
  ∀ n ∈ cube, n ∈ numbers

def are_cubes (cubes : List (List ℕ)) : Prop :=
  cubes.length = 3 ∧ ∀ cube ∈ cubes, is_valid_cube cube ∧ cube.length = 6

-- The main theorem stating the maximum possible sum of the visible numbers
theorem max_visible_sum (cubes : List (List ℕ)) (h : are_cubes cubes) : ∃ s, s = 267 :=
by
  sorry

end NUMINAMATH_GPT_max_visible_sum_l1563_156310


namespace NUMINAMATH_GPT_number_of_girls_l1563_156349

theorem number_of_girls (n : ℕ) (A : ℝ) 
    (h1 : A = (n * (A + 1) + 55 - 80) / n) : n = 25 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_girls_l1563_156349


namespace NUMINAMATH_GPT_integral_value_l1563_156317

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end NUMINAMATH_GPT_integral_value_l1563_156317


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1563_156309

-- Definitions for lines and planes
def line : Type := ℝ × ℝ × ℝ
def plane : Type := ℝ × ℝ × ℝ × ℝ

-- Predicate for perpendicularity of a line to a plane
def perp_to_plane (l : line) (α : plane) : Prop := sorry

-- Predicate for parallelism of two planes
def parallel_planes (α β : plane) : Prop := sorry

-- Predicate for perpendicularity of two lines
def perp_lines (l m : line) : Prop := sorry

-- Predicate for a line being parallel to a plane
def parallel_to_plane (m : line) (β : plane) : Prop := sorry

-- Given conditions
variable (l : line)
variable (m : line)
variable (alpha : plane)
variable (beta : plane)
variable (H1 : perp_to_plane l alpha) -- l ⊥ α
variable (H2 : parallel_to_plane m beta) -- m ∥ β

-- Theorem statement
theorem sufficient_but_not_necessary :
  (parallel_planes alpha beta → perp_lines l m) ∧ ¬(perp_lines l m → parallel_planes alpha beta) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1563_156309


namespace NUMINAMATH_GPT_zoo_sea_lions_l1563_156348

variable (S P : ℕ)

theorem zoo_sea_lions (h1 : S / P = 4 / 11) (h2 : P = S + 84) : S = 48 := 
sorry

end NUMINAMATH_GPT_zoo_sea_lions_l1563_156348


namespace NUMINAMATH_GPT_peter_has_4_finches_l1563_156374

variable (parakeet_eats_per_day : ℕ) (parrot_eats_per_day : ℕ) (finch_eats_per_day : ℕ)
variable (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ)
variable (total_birdseed : ℕ)

theorem peter_has_4_finches
    (h1 : parakeet_eats_per_day = 2)
    (h2 : parrot_eats_per_day = 14)
    (h3 : finch_eats_per_day = 1)
    (h4 : num_parakeets = 3)
    (h5 : num_parrots = 2)
    (h6 : total_birdseed = 266)
    (h7 : total_birdseed = (num_parakeets * parakeet_eats_per_day + num_parrots * parrot_eats_per_day) * 7 + num_finches * finch_eats_per_day * 7) :
    num_finches = 4 :=
by
  sorry

end NUMINAMATH_GPT_peter_has_4_finches_l1563_156374


namespace NUMINAMATH_GPT_replace_90_percent_in_3_days_cannot_replace_all_banknotes_l1563_156381

-- Define constants and conditions
def total_old_banknotes : ℕ := 3628800
def daily_cost : ℕ := 90000
def major_repair_cost : ℕ := 700000
def max_daily_print_after_repair : ℕ := 1000000
def budget_limit : ℕ := 1000000

-- Define the day's print capability function (before repair)
def daily_print (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  if num_days = 1 then banknotes_remaining / 2
  else (banknotes_remaining / (num_days + 1))

-- Define the budget calculation before repair
def print_costs (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  daily_cost * num_days

-- Lean theorem to be stated proving that 90% of the banknotes can be replaced within 3 days
theorem replace_90_percent_in_3_days :
  ∃ (days : ℕ) (banknotes_replaced : ℕ), days = 3 ∧ banknotes_replaced = 3265920 ∧ print_costs days total_old_banknotes ≤ budget_limit :=
sorry

-- Lean theorem to be stated proving that not all banknotes can be replaced within the given budget
theorem cannot_replace_all_banknotes :
  ∀ banknotes_replaced cost : ℕ,
  banknotes_replaced < total_old_banknotes ∧ cost ≤ budget_limit →
  banknotes_replaced + (total_old_banknotes / (4 + 1)) < total_old_banknotes :=
sorry

end NUMINAMATH_GPT_replace_90_percent_in_3_days_cannot_replace_all_banknotes_l1563_156381


namespace NUMINAMATH_GPT_focal_distance_of_ellipse_l1563_156389

theorem focal_distance_of_ellipse :
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 9) = 1 → (2 * Real.sqrt 7) = 2 * Real.sqrt 7 :=
by
  intros x y hxy
  sorry

end NUMINAMATH_GPT_focal_distance_of_ellipse_l1563_156389


namespace NUMINAMATH_GPT_related_sequence_exists_l1563_156380

theorem related_sequence_exists :
  ∃ b : Fin 5 → ℕ, b = ![11, 10, 9, 8, 7] :=
by
  let a : Fin 5 → ℕ := ![1, 5, 9, 13, 17]
  let b : Fin 5 → ℕ := ![
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 0) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 1) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 2) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 3) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 4) / 4
  ]
  existsi b
  sorry

end NUMINAMATH_GPT_related_sequence_exists_l1563_156380


namespace NUMINAMATH_GPT_sum_of_coordinates_of_B_l1563_156388

-- Definitions
def Point := (ℝ × ℝ)
def isMidpoint (M A B : Point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Given conditions
def M : Point := (4, 8)
def A : Point := (10, 4)

-- Statement to prove
theorem sum_of_coordinates_of_B (B : Point) (h : isMidpoint M A B) :
  B.1 + B.2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_B_l1563_156388


namespace NUMINAMATH_GPT_cylinder_base_radius_l1563_156303

theorem cylinder_base_radius (l w : ℝ) (h_l : l = 6) (h_w : w = 4) (h_circ : l = 2 * Real.pi * r ∨ w = 2 * Real.pi * r) : 
    r = 3 / Real.pi ∨ r = 2 / Real.pi := by
  sorry

end NUMINAMATH_GPT_cylinder_base_radius_l1563_156303


namespace NUMINAMATH_GPT_probability_not_equal_genders_l1563_156351

noncomputable def probability_more_grandsons_or_more_granddaughters : ℚ :=
  let total_ways := 2 ^ 12
  let equal_distribution_ways := (Nat.choose 12 6)
  let probability_equal := (equal_distribution_ways : ℚ) / (total_ways : ℚ)
  1 - probability_equal

theorem probability_not_equal_genders (n : ℕ) (p : ℚ) (hp : p = 1 / 2) (hn : n = 12) :
  probability_more_grandsons_or_more_granddaughters = 793 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_equal_genders_l1563_156351


namespace NUMINAMATH_GPT_compute_expression_l1563_156366
-- Import the standard math library to avoid import errors.

-- Define the theorem statement based on the given conditions and the correct answer.
theorem compute_expression :
  (75 * 2424 + 25 * 2424) / 2 = 121200 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1563_156366


namespace NUMINAMATH_GPT_work_together_zero_days_l1563_156362

theorem work_together_zero_days (a b : ℝ) (ha : a = 1/18) (hb : b = 1/9) (x : ℝ) (hx : 1 - x * a = 2/3) : x = 6 →
  (a - a) * (b - b) = 0 := by
  sorry

end NUMINAMATH_GPT_work_together_zero_days_l1563_156362


namespace NUMINAMATH_GPT_actual_distance_between_city_centers_l1563_156335

-- Define the conditions
def map_distance_cm : ℝ := 45
def scale_cm_to_km : ℝ := 10

-- Define the proof statement
theorem actual_distance_between_city_centers
  (md : ℝ := map_distance_cm)
  (scale : ℝ := scale_cm_to_km) :
  md * scale = 450 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_between_city_centers_l1563_156335


namespace NUMINAMATH_GPT_correct_answer_is_C_l1563_156344

structure Point where
  x : ℤ
  y : ℤ

def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def A : Point := ⟨1, -1⟩
def B : Point := ⟨0, 2⟩
def C : Point := ⟨-3, 2⟩
def D : Point := ⟨4, 0⟩

theorem correct_answer_is_C : inSecondQuadrant C := sorry

end NUMINAMATH_GPT_correct_answer_is_C_l1563_156344
