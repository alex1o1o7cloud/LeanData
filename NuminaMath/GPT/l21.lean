import Mathlib

namespace larger_number_l21_21095

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end larger_number_l21_21095


namespace increased_contact_area_effect_l21_21726

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l21_21726


namespace polygon_sides_l21_21318

theorem polygon_sides (x : ℝ) (hx : 0 < x) (h : x + 5 * x = 180) : 12 = 360 / x :=
by {
  -- Steps explaining: x should be the exterior angle then proof follows.
  sorry
}

end polygon_sides_l21_21318


namespace find_m_l21_21465

theorem find_m (x m : ℝ) :
  (2 * x + m) * (x - 3) = 2 * x^2 - 3 * m ∧ 
  (∀ c : ℝ, c * x = 0 → c = 0) → 
  m = 6 :=
by sorry

end find_m_l21_21465


namespace seating_arrangement_l21_21009

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l21_21009


namespace cookie_revenue_l21_21771

theorem cookie_revenue :
  let robyn_day1_packs := 25
  let robyn_day1_price := 4.0
  let lucy_day1_packs := 17
  let lucy_day1_price := 5.0
  let robyn_day2_packs := 15
  let robyn_day2_price := 3.5
  let lucy_day2_packs := 9
  let lucy_day2_price := 4.5
  let robyn_day3_packs := 23
  let robyn_day3_price := 4.5
  let lucy_day3_packs := 20
  let lucy_day3_price := 3.5
  let robyn_day1_revenue := robyn_day1_packs * robyn_day1_price
  let lucy_day1_revenue := lucy_day1_packs * lucy_day1_price
  let robyn_day2_revenue := robyn_day2_packs * robyn_day2_price
  let lucy_day2_revenue := lucy_day2_packs * lucy_day2_price
  let robyn_day3_revenue := robyn_day3_packs * robyn_day3_price
  let lucy_day3_revenue := lucy_day3_packs * lucy_day3_price
  let robyn_total_revenue := robyn_day1_revenue + robyn_day2_revenue + robyn_day3_revenue
  let lucy_total_revenue := lucy_day1_revenue + lucy_day2_revenue + lucy_day3_revenue
  let total_revenue := robyn_total_revenue + lucy_total_revenue
  total_revenue = 451.5 := 
by
  sorry

end cookie_revenue_l21_21771


namespace Sara_spent_on_hotdog_l21_21082

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end Sara_spent_on_hotdog_l21_21082


namespace proof_average_l21_21088

def average_two (x y : ℚ) : ℚ := (x + y) / 2
def average_three (x y z : ℚ) : ℚ := (x + y + z) / 3

theorem proof_average :
  average_three (2 * average_three 3 2 0) (average_two 0 3) (1 * 3) = 47 / 18 :=
by
  sorry

end proof_average_l21_21088


namespace quadratic_increasing_for_x_geq_3_l21_21676

theorem quadratic_increasing_for_x_geq_3 (x : ℝ) : 
  x ≥ 3 → y = 2 * (x - 3)^2 - 1 → ∃ d > 0, ∀ p ≥ x, y ≤ 2 * (p - 3)^2 - 1 := sorry

end quadratic_increasing_for_x_geq_3_l21_21676


namespace sale_book_cost_l21_21637

variable (x : ℝ)

def fiveSaleBooksCost (x : ℝ) : ℝ :=
  5 * x

def onlineBooksCost : ℝ :=
  40

def bookstoreBooksCost : ℝ :=
  3 * 40

def totalCost (x : ℝ) : ℝ :=
  fiveSaleBooksCost x + onlineBooksCost + bookstoreBooksCost

theorem sale_book_cost :
  totalCost x = 210 → x = 10 := by
  sorry

end sale_book_cost_l21_21637


namespace dividend_in_terms_of_a_l21_21325

variable (a Q R D : ℕ)

-- Given conditions as hypotheses
def condition1 : Prop := D = 25 * Q
def condition2 : Prop := D = 7 * R
def condition3 : Prop := Q - R = 15
def condition4 : Prop := R = 3 * a

-- Prove that the dividend given these conditions equals the expected expression
theorem dividend_in_terms_of_a (a : ℕ) (Q : ℕ) (R : ℕ) (D : ℕ) :
  condition1 D Q → condition2 D R → condition3 Q R → condition4 R a →
  (D * Q + R) = 225 * a^2 + 1128 * a + 5625 :=
by
  intro h1 h2 h3 h4
  sorry

end dividend_in_terms_of_a_l21_21325


namespace B_subset_A_l21_21306

variable {α : Type*}
variable (A B : Set α)

def A_def : Set ℝ := { x | x ≥ 1 }
def B_def : Set ℝ := { x | x > 2 }

theorem B_subset_A : B_def ⊆ A_def :=
sorry

end B_subset_A_l21_21306


namespace no_opposite_identical_numbers_l21_21383

open Finset

theorem no_opposite_identical_numbers : 
  ∀ (f g : Fin 20 → Fin 20), 
  (∀ i : Fin 20, ∃ j : Fin 20, f j = i ∧ g j = (i + j) % 20) → 
  ∃ k : ℤ, ∀ i : Fin 20, f (i + k) % 20 ≠ g i 
  := by
    sorry

end no_opposite_identical_numbers_l21_21383


namespace sqrt_factorial_mul_squared_l21_21810

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l21_21810


namespace det_E_eq_25_l21_21627

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l21_21627


namespace second_number_is_650_l21_21410

theorem second_number_is_650 (x : ℝ) (h1 : 0.20 * 1600 = 0.20 * x + 190) : x = 650 :=
by sorry

end second_number_is_650_l21_21410


namespace no_integer_sided_triangle_with_odd_perimeter_1995_l21_21869

theorem no_integer_sided_triangle_with_odd_perimeter_1995 :
  ¬ ∃ (a b c : ℕ), (a + b + c = 1995) ∧ (∃ (h1 h2 h3 : ℕ), true) :=
by
  sorry

end no_integer_sided_triangle_with_odd_perimeter_1995_l21_21869


namespace semi_circle_radius_l21_21531

theorem semi_circle_radius (P : ℝ) (π : ℝ) (r : ℝ) (hP : P = 10.797344572538567) (hπ : π = 3.14159) :
  (π + 2) * r = P → r = 2.1 :=
by
  intro h
  sorry

end semi_circle_radius_l21_21531


namespace product_of_x_y_l21_21472

theorem product_of_x_y (x y : ℝ) :
  (54 = 5 * y^2 + 20) →
  (8 * x^2 + 2 = 38) →
  x * y = Real.sqrt (30.6) :=
by
  intros h1 h2
  -- these would be the proof steps
  sorry

end product_of_x_y_l21_21472


namespace people_in_room_proof_l21_21253

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end people_in_room_proof_l21_21253


namespace largest_square_area_l21_21914

theorem largest_square_area (XY XZ YZ : ℝ)
  (h1 : XZ^2 = 2 * XY^2)
  (h2 : XY^2 + YZ^2 = XZ^2)
  (h3 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  -- Proof skipped
  sorry

end largest_square_area_l21_21914


namespace line_equation_direction_point_l21_21886

theorem line_equation_direction_point 
  (d : ℝ × ℝ) (A : ℝ × ℝ) :
  d = (2, -1) →
  A = (1, 0) →
  ∃ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = -1 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 ↔ x + 2 * y - 1 = 0 :=
by
  sorry

end line_equation_direction_point_l21_21886


namespace employee_pays_correct_amount_l21_21103

def wholesale_cost : ℝ := 200
def markup_percentage : ℝ := 0.20
def discount_percentage : ℝ := 0.10

def retail_price (wholesale: ℝ) (markup_percentage: ℝ) : ℝ :=
  wholesale * (1 + markup_percentage)

def discount_amount (price: ℝ) (discount_percentage: ℝ) : ℝ :=
  price * discount_percentage

def final_price (retail: ℝ) (discount: ℝ) : ℝ :=
  retail - discount

theorem employee_pays_correct_amount : final_price (retail_price wholesale_cost markup_percentage) 
                                                     (discount_amount (retail_price wholesale_cost markup_percentage) discount_percentage) = 216 := 
by
  sorry

end employee_pays_correct_amount_l21_21103


namespace jack_sugar_l21_21481

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l21_21481


namespace Mr_Lee_probability_l21_21224

noncomputable def probability_more_grandsons_or_granddaughters : ℚ :=
  let n := 12
  let p := 1 / 2
  let num_ways_6_boys := Nat.choose n (n / 2)
  let total_ways := 2^n
  let prob_equal_boys_and_girls := (num_ways_6_boys : ℚ) / (total_ways : ℚ)
  1 - prob_equal_boys_and_girls

theorem Mr_Lee_probability : probability_more_grandsons_or_granddaughters = 793 / 1024 := by
  sorry

end Mr_Lee_probability_l21_21224


namespace number_of_arrangements_SEES_l21_21284

theorem number_of_arrangements_SEES : 
  ∃ n : ℕ, 
    (∀ (total_letters E S : ℕ), 
      total_letters = 4 ∧ E = 2 ∧ S = 2 → 
      n = Nat.factorial total_letters / (Nat.factorial E * Nat.factorial S)) → 
    n = 6 := 
by 
  sorry

end number_of_arrangements_SEES_l21_21284


namespace right_triangle_right_angles_l21_21458

theorem right_triangle_right_angles (T : Triangle) (h1 : T.is_right_triangle) :
  T.num_right_angles = 1 :=
sorry

end right_triangle_right_angles_l21_21458


namespace min_sum_factors_l21_21953

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l21_21953


namespace steps_in_five_days_l21_21358

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end steps_in_five_days_l21_21358


namespace portion_of_larger_jar_full_l21_21104

noncomputable def smaller_jar_capacity (S L : ℝ) : Prop :=
  (1 / 5) * S = (1 / 4) * L

noncomputable def larger_jar_capacity (L : ℝ) : ℝ :=
  (1 / 5) * (5 / 4) * L

theorem portion_of_larger_jar_full (S L : ℝ) 
  (h1 : smaller_jar_capacity S L) : 
  (1 / 4) * L + (1 / 4) * L = (1 / 2) * L := 
sorry

end portion_of_larger_jar_full_l21_21104


namespace price_ratio_l21_21147

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l21_21147


namespace common_ratio_of_infinite_geometric_series_l21_21876

theorem common_ratio_of_infinite_geometric_series 
  (a b : ℚ) 
  (h1 : a = 8 / 10) 
  (h2 : b = -6 / 15) 
  (h3 : b = a * r) : 
  r = -1 / 2 :=
by
  -- The proof goes here
  sorry

end common_ratio_of_infinite_geometric_series_l21_21876


namespace real_solutions_l21_21294

theorem real_solutions (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 7) →
  ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 3) * (x - 7) * (x - 3)) = 1 →
  x = 3 + Real.sqrt 3 ∨ x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 :=
by
  sorry

end real_solutions_l21_21294


namespace proof_problem_l21_21452

noncomputable def f (x a : ℝ) : ℝ := (1 + x^2) * Real.exp x - a
noncomputable def f' (x a : ℝ) : ℝ := (1 + 2 * x + x^2) * Real.exp x
noncomputable def k_OP (a : ℝ) : ℝ := a - 2 / Real.exp 1
noncomputable def g (m : ℝ) : ℝ := Real.exp m - (m + 1)

theorem proof_problem (a m : ℝ) (h₁ : a > 0) (h₂ : f' (-1) a = 0) (h₃ : f' m a = k_OP a) 
  : m + 1 ≤ 3 * a - 2 / Real.exp 1 := by
  sorry

end proof_problem_l21_21452


namespace pen_price_ratio_l21_21165

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l21_21165


namespace genevieve_coffee_drink_l21_21730

theorem genevieve_coffee_drink :
  let gallons := 4.5
  let small_thermos_count := 12
  let small_thermos_capacity_ml := 250
  let large_thermos_count := 6
  let large_thermos_capacity_ml := 500
  let genevieve_small_thermos_drink_count := 2
  let genevieve_large_thermos_drink_count := 1
  let ounces_per_gallon := 128
  let mls_per_ounce := 29.5735
  let total_mls := (gallons * ounces_per_gallon) * mls_per_ounce
  let genevieve_ml_drink := (genevieve_small_thermos_drink_count * small_thermos_capacity_ml) 
                            + (genevieve_large_thermos_drink_count * large_thermos_capacity_ml)
  let genevieve_ounces_drink := genevieve_ml_drink / mls_per_ounce
  genevieve_ounces_drink = 33.814 :=
by sorry

end genevieve_coffee_drink_l21_21730


namespace sunflower_count_l21_21606

theorem sunflower_count (r l d : ℕ) (t : ℕ) (h1 : r + l + d = 40) (h2 : t = 160) : 
  t - (r + l + d) = 120 := by
  sorry

end sunflower_count_l21_21606


namespace max_length_sequence_y_l21_21290

noncomputable def sequence (b1 b2 : ℕ) : ℕ → ℤ
| 1     := b1
| 2     := b2
| (n+3) := sequence (n+1) - sequence (n+2)

theorem max_length_sequence_y :
  ∃ y : ℕ, y = 1236 ∧ 
    ∀ n : ℕ, n ≤ 11 → 
    (sequence 2000 y n >= 0 ∧ sequence 2000 y (n + 2) - sequence 2000 y (n + 1) >= 0 ∧ 
    sequence 2000 y (n + 3) = sequence 2000 y (n + 1) - sequence 2000 y (n + 2)) :=
begin
  sorry
end

end max_length_sequence_y_l21_21290


namespace probability_asian_country_l21_21380

theorem probability_asian_country : 
  let cards := ["China", "USA", "UK", "South Korea"]
  let asian_countries := ["China", "South Korea"]
  let total_cards := 4
  let favorable_outcomes := 2
  (favorable_outcomes / total_cards : ℚ) = 1 / 2 := 
by 
  sorry

end probability_asian_country_l21_21380


namespace sum_le_square_l21_21947

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end sum_le_square_l21_21947


namespace more_elements_in_set_N_l21_21968

theorem more_elements_in_set_N 
  (M N : Finset ℕ) 
  (h_partition : ∀ x, x ∈ M ∨ x ∈ N) 
  (h_disjoint : ∀ x, x ∈ M → x ∉ N) 
  (h_total_2000 : M.card + N.card = 10^2000 - 10^1999) 
  (h_total_1000 : (10^1000 - 10^999) * (10^1000 - 10^999) < 10^2000 - 10^1999) : 
  N.card > M.card :=
by { sorry }

end more_elements_in_set_N_l21_21968


namespace geometric_sequence_strictly_increasing_iff_l21_21063

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^(n-1)

theorem geometric_sequence_strictly_increasing_iff (a_1 q : ℝ) :
  (∀ n : ℕ, geometric_sequence a_1 q (n+2) > geometric_sequence a_1 q n) ↔ 
  (∀ n : ℕ, geometric_sequence a_1 q (n+1) > geometric_sequence a_1 q n) := 
by
  sorry

end geometric_sequence_strictly_increasing_iff_l21_21063


namespace gel_pen_ratio_l21_21127

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l21_21127


namespace garden_length_l21_21677

theorem garden_length (P B : ℕ) (h₁ : P = 600) (h₂ : B = 95) : (∃ L : ℕ, 2 * (L + B) = P ∧ L = 205) :=
by
  sorry

end garden_length_l21_21677


namespace reasoning_is_inductive_l21_21788

-- Define conditions
def conducts_electricity (metal : String) : Prop :=
  metal = "copper" ∨ metal = "iron" ∨ metal = "aluminum" ∨ metal = "gold" ∨ metal = "silver"

-- Define the inductive reasoning type
def is_inductive_reasoning : Prop := 
  ∀ metals, conducts_electricity metals → (∀ m : String, conducts_electricity m → conducts_electricity m)

-- The theorem to prove
theorem reasoning_is_inductive : is_inductive_reasoning :=
by
  sorry

end reasoning_is_inductive_l21_21788


namespace num_integers_satisfying_inequality_l21_21895

theorem num_integers_satisfying_inequality :
  ∃ (x : ℕ), ∀ (y: ℤ), (-3 ≤ 3 * y + 2 → 3 * y + 2 ≤ 8) ↔ 4 = x :=
by
  sorry

end num_integers_satisfying_inequality_l21_21895


namespace find_constant_l21_21875

theorem find_constant (N : ℝ) (C : ℝ) (h1 : N = 12.0) (h2 : C + 0.6667 * N = 0.75 * N) : C = 0.9996 :=
by
  sorry

end find_constant_l21_21875


namespace sqrt_factorial_mul_squared_l21_21809

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l21_21809


namespace jack_sugar_final_l21_21478

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l21_21478


namespace division_remainder_l21_21168

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end division_remainder_l21_21168


namespace sold_on_saturday_l21_21504

-- Define all the conditions provided in the question
def amount_sold_thursday : ℕ := 210
def amount_sold_friday : ℕ := 2 * amount_sold_thursday
def amount_sold_sunday (S : ℕ) : ℕ := (S / 2)
def total_planned_sold : ℕ := 500
def excess_sold : ℕ := 325

-- Total sold is the sum of sold amounts from Thursday to Sunday
def total_sold (S : ℕ) : ℕ := amount_sold_thursday + amount_sold_friday + S + amount_sold_sunday S

-- The theorem to prove
theorem sold_on_saturday : ∃ S : ℕ, total_sold S = total_planned_sold + excess_sold ∧ S = 130 :=
by
  sorry

end sold_on_saturday_l21_21504


namespace fifth_rectangle_is_square_l21_21563

-- Define the conditions
variables (s : ℝ) (a b : ℝ)
variables (R1 R2 R3 R4 : Set (ℝ × ℝ))
variables (R5 : Set (ℝ × ℝ))

-- Assume the areas of the corner rectangles are equal
def equal_area (R : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), R = {p | p.1 < a ∧ p.2 < b} ∧ a * b = k

-- State the conditions
axiom h1 : equal_area R1 a
axiom h2 : equal_area R2 a
axiom h3 : equal_area R3 a
axiom h4 : equal_area R4 a

axiom h5 : ∀ (p : ℝ × ℝ), p ∈ R5 → p.1 ≠ 0 → p.2 ≠ 0

-- Prove that the fifth rectangle is a square
theorem fifth_rectangle_is_square : ∃ c : ℝ, ∀ r1 r2, r1 ∈ R5 → r2 ∈ R5 → r1.1 - r2.1 = c ∧ r1.2 - r2.2 = c :=
by sorry

end fifth_rectangle_is_square_l21_21563


namespace students_all_three_classes_l21_21050

variables (H M E HM HE ME HME : ℕ)

-- Conditions from the problem
def student_distribution : Prop :=
  H = 12 ∧
  M = 17 ∧
  E = 36 ∧
  HM + HE + ME = 3 ∧
  86 = H + M + E - (HM + HE + ME) + HME

-- Prove the number of students registered for all three classes
theorem students_all_three_classes (h : student_distribution H M E HM HE ME HME) : HME = 24 :=
  by sorry

end students_all_three_classes_l21_21050


namespace jack_sugar_amount_l21_21484

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l21_21484


namespace collinear_magnitude_a_perpendicular_magnitude_b_l21_21307

noncomputable section

open Real

-- Defining the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Defining the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Given conditions and respective proofs
theorem collinear_magnitude_a (x : ℝ) (h : 1 * 3 = x ^ 2) : magnitude (a x) = 2 :=
by sorry

theorem perpendicular_magnitude_b (x : ℝ) (h : 1 * x + x * 3 = 0) : magnitude (b x) = 3 :=
by sorry

end collinear_magnitude_a_perpendicular_magnitude_b_l21_21307


namespace problem_inequality_l21_21490

theorem problem_inequality (n a b : ℕ) (h₁ : n ≥ 2) 
  (h₂ : ∀ m, 2^m ∣ 5^n - 3^n → m ≤ a) 
  (h₃ : ∀ m, 2^m ≤ n → m ≤ b) : a ≤ b + 3 :=
sorry

end problem_inequality_l21_21490


namespace female_officers_on_police_force_l21_21505

theorem female_officers_on_police_force
  (percent_on_duty : ℝ)
  (total_on_duty : ℕ)
  (half_female_on_duty : ℕ)
  (h1 : percent_on_duty = 0.16)
  (h2 : total_on_duty = 160)
  (h3 : half_female_on_duty = total_on_duty / 2)
  (h4 : half_female_on_duty = 80)
  :
  ∃ (total_female_officers : ℕ), total_female_officers = 500 :=
by
  sorry

end female_officers_on_police_force_l21_21505


namespace factor_count_l21_21433

theorem factor_count (x : ℤ) : 
  (x^12 - x^3) = x^3 * (x - 1) * (x^2 + x + 1) * (x^6 + x^3 + 1) -> 4 = 4 :=
by
  sorry

end factor_count_l21_21433


namespace intersection_of_A_and_B_l21_21463

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_of_A_and_B : A ∩ B = {2, 6} :=
by
  sorry

end intersection_of_A_and_B_l21_21463


namespace symmetric_axis_of_g_l21_21182

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 6))

theorem symmetric_axis_of_g :
  ∃ k : ℤ, (∃ x : ℝ, g x = 2 * Real.sin (k * Real.pi + (Real.pi / 2)) ∧ x = (k * Real.pi) / 2 + (Real.pi / 3)) :=
sorry

end symmetric_axis_of_g_l21_21182


namespace volume_shaded_part_rotated_l21_21123

noncomputable def volume_of_solid_rotated_around_CD (BC AB : ℝ) (pi_val : ℝ) : ℝ :=
  let r := BC / 2
  let volume_cone := (1 / 3) * pi_val * (r ^ 2) * AB
  2 * volume_cone

theorem volume_shaded_part_rotated (BC AB : ℝ) (pi_val : ℝ) :
  BC = 6 → AB = 10 → pi_val = 3.14 → 
  volume_of_solid_rotated_around_CD BC AB pi_val = 188.4 :=
by
  intros hBC hAB hPi
  rw [hBC, hAB, hPi]
  have r := 6 / 2 
  have volume_cone := (1 / 3) * 3.14 * (r ^ 2) * 10
  have volume := 2 * volume_cone
  sorry

end volume_shaded_part_rotated_l21_21123


namespace equation_solution_system_solution_l21_21237

theorem equation_solution (x : ℚ) :
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 :=
by sorry

theorem system_solution (x y : ℚ) :
  (3 * x - 4 * y = 14) ∧ (5 * x + 4 * y = 2) ↔
  (x = 2) ∧ (y = -2) :=
by sorry

end equation_solution_system_solution_l21_21237


namespace barbata_interest_rate_l21_21998

theorem barbata_interest_rate (r : ℝ) : 
  let initial_investment := 2800
  let additional_investment := 1400
  let total_investment := initial_investment + additional_investment
  let annual_income := 0.06 * total_investment
  let additional_interest_rate := 0.08
  let income_from_initial := initial_investment * r
  let income_from_additional := additional_investment * additional_interest_rate
  income_from_initial + income_from_additional = annual_income → 
  r = 0.05 :=
by
  intros
  sorry

end barbata_interest_rate_l21_21998


namespace flashlight_price_percentage_l21_21170

theorem flashlight_price_percentage 
  (hoodie_price boots_price total_spent flashlight_price : ℝ)
  (discount_rate : ℝ)
  (h1 : hoodie_price = 80)
  (h2 : boots_price = 110)
  (h3 : discount_rate = 0.10)
  (h4 : total_spent = 195) 
  (h5 : total_spent = hoodie_price + ((1 - discount_rate) * boots_price) + flashlight_price) : 
  (flashlight_price / hoodie_price) * 100 = 20 :=
by
  sorry

end flashlight_price_percentage_l21_21170


namespace race_result_l21_21264

-- Definitions based on conditions
variable (hare_won : Bool)
variable (fox_second : Bool)
variable (hare_second : Bool)
variable (moose_first : Bool)

-- Condition that each squirrel had one error.
axiom owl_statement : xor hare_won fox_second ∧ xor hare_second moose_first

-- The final proof problem
theorem race_result : moose_first = true ∧ fox_second = true :=
by {
  -- Proving based on the owl's statement that each squirrel had one error
  sorry
}

end race_result_l21_21264


namespace find_number_l21_21722

-- Define the problem conditions
def problem_condition (x : ℝ) : Prop := 2 * x - x / 2 = 45

-- Main theorem statement
theorem find_number : ∃ (x : ℝ), problem_condition x ∧ x = 30 :=
by
  existsi 30
  -- Include the problem condition and the solution check
  unfold problem_condition
  -- We are skipping the proof using sorry to just provide the statement
  sorry

end find_number_l21_21722


namespace range_of_a_l21_21324

theorem range_of_a (x a : ℝ) (h₀ : x < 0) (h₁ : 2^x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l21_21324


namespace inequality_proof_l21_21533

theorem inequality_proof (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 → x > 5 / 7 := 
by
  sorry

end inequality_proof_l21_21533


namespace four_students_same_acquaintances_l21_21003

theorem four_students_same_acquaintances
  (students : Finset ℕ)
  (acquainted : ∀ s ∈ students, (students \ {s}).card ≥ 68)
  (count : students.card = 102) :
  ∃ n, ∃ cnt, cnt ≥ 4 ∧ (∃ S, S ⊆ students ∧ S.card = cnt ∧ ∀ x ∈ S, (students \ {x}).card = n) :=
sorry

end four_students_same_acquaintances_l21_21003


namespace range_of_a_l21_21475

noncomputable def acute_angle_condition (a : ℝ) : Prop :=
  let M := (-2, 0)
  let N := (0, 2)
  let A := (-1, 1)
  (a > 0) ∧ (∀ P : ℝ × ℝ, (P.1 - a) ^ 2 + P.2 ^ 2 = 2 →
    (dist P A) > 2 * Real.sqrt 2)

theorem range_of_a (a : ℝ) : acute_angle_condition a ↔ a > Real.sqrt 7 - 1 :=
by sorry

end range_of_a_l21_21475


namespace min_sum_of_factors_l21_21964

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21964


namespace sum_first_49_nat_nums_l21_21019

theorem sum_first_49_nat_nums : (Finset.range 50).sum (fun x => x) = 1225 := 
by
  sorry

end sum_first_49_nat_nums_l21_21019


namespace evaluate_expression_l21_21011

theorem evaluate_expression :
  abs ((4^2 - 8 * (3^2 - 12))^2) - abs (Real.sin (5 * Real.pi / 6) - Real.cos (11 * Real.pi / 3)) = 1600 :=
by
  sorry

end evaluate_expression_l21_21011


namespace parallel_lines_perpendicular_lines_l21_21892

theorem parallel_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 = s2) →
  a = 2 :=
by
  intros
  -- Proof goes here
  sorry

theorem perpendicular_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 * s2 = -1) →
  a = 0 ∨ a = -3 :=
by
  intros
  -- Proof goes here
  sorry

end parallel_lines_perpendicular_lines_l21_21892


namespace gel_pen_price_ratio_l21_21155

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l21_21155


namespace annual_rent_per_square_foot_l21_21091

theorem annual_rent_per_square_foot 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (area : ℕ)
  (annual_rent : ℕ) : 
  monthly_rent = 3600 → 
  length = 18 → 
  width = 20 → 
  area = length * width → 
  annual_rent = monthly_rent * 12 → 
  annual_rent / area = 120 :=
by
  sorry

end annual_rent_per_square_foot_l21_21091


namespace container_unoccupied_volume_is_628_l21_21619

def rectangular_prism_volume (length width height : ℕ) : ℕ :=
  length * width * height

def water_volume (total_volume : ℕ) : ℕ :=
  total_volume / 3

def ice_cubes_volume (number_of_cubes volume_per_cube : ℕ) : ℕ :=
  number_of_cubes * volume_per_cube

def unoccupied_volume (total_volume occupied_volume : ℕ) : ℕ :=
  total_volume - occupied_volume

theorem container_unoccupied_volume_is_628 :
  let length := 12
  let width := 10
  let height := 8
  let number_of_ice_cubes := 12
  let volume_per_ice_cube := 1
  let V := rectangular_prism_volume length width height
  let V_water := water_volume V
  let V_ice := ice_cubes_volume number_of_ice_cubes volume_per_ice_cube
  let V_occupied := V_water + V_ice
  unoccupied_volume V V_occupied = 628 :=
by
  sorry

end container_unoccupied_volume_is_628_l21_21619


namespace pages_for_15_dollars_l21_21755

theorem pages_for_15_dollars 
  (cpg : ℚ) -- cost per 5 pages in cents
  (budget : ℚ) -- budget in cents
  (h_cpg_pos : cpg = 7 * 1) -- 7 cents for 5 pages
  (h_budget_pos : budget = 1500 * 1) -- $15 = 1500 cents
  : (budget * (5 / cpg)).floor = 1071 :=
by {
  sorry
}

end pages_for_15_dollars_l21_21755


namespace benny_picked_proof_l21_21999

-- Define the number of apples Dan picked
def dan_picked: ℕ := 9

-- Define the total number of apples picked
def total_apples: ℕ := 11

-- Define the number of apples Benny picked
def benny_picked (dan_picked total_apples: ℕ): ℕ :=
  total_apples - dan_picked

-- The theorem we need to prove
theorem benny_picked_proof: benny_picked dan_picked total_apples = 2 :=
by
  -- We calculate the number of apples Benny picked
  sorry

end benny_picked_proof_l21_21999


namespace maximum_t_l21_21041

theorem maximum_t {a b t : ℝ} (ha : 0 < a) (hb : a < b) (ht : b < t)
  (h_condition : b * Real.log a < a * Real.log b) : t ≤ Real.exp 1 :=
sorry

end maximum_t_l21_21041


namespace gel_pen_ratio_l21_21128

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l21_21128


namespace lemonade_syrup_parts_l21_21113

theorem lemonade_syrup_parts (L : ℝ) :
  (L = 2 / 0.75) →
  (L = 2.6666666666666665) :=
by
  sorry

end lemonade_syrup_parts_l21_21113


namespace shortest_distance_proof_l21_21221

noncomputable def shortest_distance (k : ℝ) : ℝ :=
  let p := (k - 6) / 2
  let f_p := -p^2 + (6 - k) * p + 18
  let d := |f_p|
  d / (Real.sqrt (k^2 + 1))

theorem shortest_distance_proof (k : ℝ) :
  shortest_distance k = 
  |(-(k - 6) / 2^2 + (6 - k) * (k - 6) / 2 + 18)| / (Real.sqrt (k^2 + 1)) :=
sorry

end shortest_distance_proof_l21_21221


namespace collection_count_l21_21928

-- Variables and setups for all elements in the problem
def vowels := multiset.mk ['O', 'U', 'A', 'I', 'O', 'A']
def consonants := multiset.mk ['C', 'M', 'P', 'T', 'T', 'N']

-- Combinatoric functions for choosing elements, considering combinations with repetition
noncomputable def count_combinations {A : Type} (s : multiset A) (n : ℕ) : ℕ :=
  (multiset.powerset_len n s).card

/-- Theorem stating the number of distinct collections -/
theorem collection_count :
  let distinct_vowel_ways := count_combinations vowels 3,
      distinct_consonant_ways := count_combinations consonants 3
  in distinct_vowel_ways * distinct_consonant_ways = 196 :=
begin
  -- Production of actual counts for verification
  sorry
end

end collection_count_l21_21928


namespace squares_difference_l21_21223

theorem squares_difference (x y z : ℤ) 
  (h1 : x + y = 10) 
  (h2 : x - y = 8) 
  (h3 : y + z = 15) : 
  x^2 - z^2 = -115 :=
by 
  sorry

end squares_difference_l21_21223


namespace exists_natural_number_starting_and_ending_with_pattern_l21_21641

theorem exists_natural_number_starting_and_ending_with_pattern (n : ℕ) : 
  ∃ (m : ℕ), 
  (m % 10 = 1) ∧ 
  (∃ t : ℕ, 
    m^2 / 10^t = 10^(n - 1) * (10^n - 1) / 9) ∧ 
  (m^2 % 10^n = 1 ∨ m^2 % 10^n = 2) :=
sorry

end exists_natural_number_starting_and_ending_with_pattern_l21_21641


namespace determine_n_l21_21261

noncomputable def P : ℤ → ℤ := sorry

theorem determine_n (n : ℕ) (P : ℤ → ℤ)
  (h_deg : ∀ x : ℤ, P x = 2 ∨ P x = 1 ∨ P x = 0)
  (h0 : ∀ k : ℕ, k ≤ n → P (3 * k) = 2)
  (h1 : ∀ k : ℕ, k < n → P (3 * k + 1) = 1)
  (h2 : ∀ k : ℕ, k < n → P (3 * k + 2) = 0)
  (h_f : P (3 * n + 1) = 730) :
  n = 4 := 
sorry

end determine_n_l21_21261


namespace proportional_parts_l21_21692

theorem proportional_parts (A B C D : ℕ) (number : ℕ) (h1 : A = 5 * x) (h2 : B = 7 * x) (h3 : C = 4 * x) (h4 : D = 8 * x) (h5 : C = 60) : number = 360 := by
  sorry

end proportional_parts_l21_21692


namespace jill_sod_area_l21_21336

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end jill_sod_area_l21_21336


namespace perimeter_range_l21_21299

variable (a b x : ℝ)
variable (a_gt_b : a > b)
variable (triangle_ineq : a - b < x ∧ x < a + b)

theorem perimeter_range : 2 * a < a + b + x ∧ a + b + x < 2 * (a + b) :=
by
  sorry

end perimeter_range_l21_21299


namespace present_age_of_A_l21_21241

theorem present_age_of_A {x : ℕ} (h₁ : ∃ (x : ℕ), 5 * x = A ∧ 3 * x = B)
                         (h₂ : ∀ (A B : ℕ), (A + 6) / (B + 6) = 7 / 5) : A = 15 :=
by sorry

end present_age_of_A_l21_21241


namespace increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l21_21725

variables {k: ℝ} -- Thermal conductivity
variables {A A': ℝ} -- Original and increased contact area
variables {dT: ℝ} -- Temperature difference
variables {dx: ℝ} -- Thickness of the skillet handle

-- Define the heat flow rate according to Fourier's law of heat conduction
def heat_flow_rate (k: ℝ) (A: ℝ) (dT: ℝ) (dx: ℝ) : ℝ :=
  -k * A * (dT / dx)

theorem increased_contact_area_increases_heat_flow 
  (h₁: A' > A) -- Increased contact area
  (h₂: dT / dx > 0) -- Positive temperature gradient
  : heat_flow_rate k A' dT dx > heat_flow_rate k A dT dx :=
by
  -- Proof to show that increased area increases heat flow rate
  sorry

theorem handle_felt_hotter_no_thermodynamic_contradiction 
  (h₁: A' > A)
  (h₂: dT / dx > 0)
  : ¬(heat_flow_rate k A' dT dx contradicts thermodynamic laws) :=
by
  -- Proof to show no contradiction with the laws of thermodynamics
  sorry

end increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l21_21725


namespace min_sum_of_factors_l21_21960

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21960


namespace imaginary_part_of_z_l21_21371

open Complex

-- Definition of the complex number as per the problem statement
def z : ℂ := (2 - 3 * Complex.I) * Complex.I

-- The theorem stating that the imaginary part of the given complex number is 2
theorem imaginary_part_of_z : z.im = 2 :=
by
  sorry

end imaginary_part_of_z_l21_21371


namespace alex_needs_more_coins_l21_21425

-- Define the conditions and problem statement 
def num_friends : ℕ := 15
def coins_alex_has : ℕ := 95 

-- The total number of coins required is
def total_coins_needed : ℕ := num_friends * (num_friends + 1) / 2

-- The minimum number of additional coins needed
def additional_coins_needed : ℕ := total_coins_needed - coins_alex_has

-- Formalize the theorem 
theorem alex_needs_more_coins : additional_coins_needed = 25 := by
  -- Here we would provide the actual proof steps
  sorry

end alex_needs_more_coins_l21_21425


namespace pen_price_ratio_l21_21163

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l21_21163


namespace pyramid_volume_l21_21468

def area_SAB : ℝ := 9
def area_SBC : ℝ := 9
def area_SCD : ℝ := 27
def area_SDA : ℝ := 27
def area_ABCD : ℝ := 36
def dihedral_angle_equal := ∀ (α β γ δ: ℝ), α = β ∧ β = γ ∧ γ = δ

theorem pyramid_volume (h_eq_dihedral : dihedral_angle_equal)
  (area_conditions : area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27)
  (area_quadrilateral : area_ABCD = 36) :
  (1 / 3 * area_ABCD * 4.5) = 54 :=
sorry

end pyramid_volume_l21_21468


namespace gel_pen_ratio_l21_21124

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l21_21124


namespace abs_diff_31st_terms_l21_21796

/-- Sequence C is an arithmetic sequence with a starting term 100 and a common difference 15. --/
def seqC (n : ℕ) : ℤ :=
  100 + 15 * (n - 1)

/-- Sequence D is an arithmetic sequence with a starting term 100 and a common difference -20. --/
def seqD (n : ℕ) : ℤ :=
  100 - 20 * (n - 1)

/-- Absolute value of the difference between the 31st terms of sequences C and D is 1050. --/
theorem abs_diff_31st_terms : |seqC 31 - seqD 31| = 1050 := by
  sorry

end abs_diff_31st_terms_l21_21796


namespace digit_sum_equality_l21_21935

-- Definitions for the conditions
def is_permutation_of_digits (a b : ℕ) : Prop :=
  -- Assume implementation that checks if b is a permutation of the digits of a
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  -- Assume implementation that computes the sum of digits of n
  sorry

-- The theorem statement
theorem digit_sum_equality (a b : ℕ)
  (h : is_permutation_of_digits a b) :
  sum_of_digits (5 * a) = sum_of_digits (5 * b) :=
sorry

end digit_sum_equality_l21_21935


namespace additional_tiles_needed_l21_21694

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end additional_tiles_needed_l21_21694


namespace sequence_problem_l21_21599

theorem sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1)) (h_eq : a 100 = a 96) :
  a 2018 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_problem_l21_21599


namespace comb_15_6_eq_5005_perm_6_eq_720_l21_21976

open Nat

-- Prove that \frac{15!}{6!(15-6)!} = 5005
theorem comb_15_6_eq_5005 : (factorial 15) / (factorial 6 * factorial (15 - 6)) = 5005 := by
  sorry

-- Prove that the number of ways to arrange 6 items in a row is 720
theorem perm_6_eq_720 : factorial 6 = 720 := by
  sorry

end comb_15_6_eq_5005_perm_6_eq_720_l21_21976


namespace min_sum_of_factors_l21_21966

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21966


namespace area_of_cyclic_quadrilateral_l21_21093

theorem area_of_cyclic_quadrilateral (A B C D : Point) (R : ℝ) (φ : ℝ)
  (h_inscribed : IsInscribed A B C D R) (h_phi : AngleBetweenDiagonals A B C D = φ) :
  let S := 2 * R^2 * sin (angle A B C) * sin (angle B C D) * sin φ in
  area A B C D = S := by
  sorry

end area_of_cyclic_quadrilateral_l21_21093


namespace solve_equation_l21_21397

theorem solve_equation (x : ℝ) : x * (x + 1) = 12 → (x = -4 ∨ x = 3) :=
by
  sorry

end solve_equation_l21_21397


namespace susan_remaining_spaces_l21_21778

def susan_first_turn_spaces : ℕ := 15
def susan_second_turn_spaces : ℕ := 7 - 5
def susan_third_turn_spaces : ℕ := 20
def susan_fourth_turn_spaces : ℕ := 0
def susan_fifth_turn_spaces : ℕ := 10 - 8
def susan_sixth_turn_spaces : ℕ := 0
def susan_seventh_turn_roll : ℕ := 6
def susan_seventh_turn_spaces : ℕ := susan_seventh_turn_roll * 2
def susan_total_moved_spaces : ℕ := susan_first_turn_spaces + susan_second_turn_spaces + susan_third_turn_spaces + susan_fourth_turn_spaces + susan_fifth_turn_spaces + susan_sixth_turn_spaces + susan_seventh_turn_spaces
def game_total_spaces : ℕ := 100

theorem susan_remaining_spaces : susan_total_moved_spaces = 51 ∧ (game_total_spaces - susan_total_moved_spaces) = 49 := by
  sorry

end susan_remaining_spaces_l21_21778


namespace cylinder_height_comparison_l21_21098

theorem cylinder_height_comparison (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by {
  -- Proof steps here, not required per instruction
  sorry
}

end cylinder_height_comparison_l21_21098


namespace solve_for_q_l21_21366

theorem solve_for_q 
  (n m q : ℕ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m + n) / 90)
  (h3 : 5 / 6 = (q - m) / 150) : 
  q = 150 :=
sorry

end solve_for_q_l21_21366


namespace factorial_sqrt_sq_l21_21827

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l21_21827


namespace product_not_50_l21_21405

theorem product_not_50 :
  (1 / 2 * 100 = 50) ∧
  (-5 * -10 = 50) ∧
  ¬(5 * 11 = 50) ∧
  (2 * 25 = 50) ∧
  (5 / 2 * 20 = 50) :=
by
  sorry

end product_not_50_l21_21405


namespace positive_difference_proof_l21_21970

noncomputable def solve_system : Prop :=
  ∃ (x y : ℝ), 
  (x + y = 40) ∧ 
  (3 * y - 4 * x = 10) ∧ 
  abs (y - x) = 8.58

theorem positive_difference_proof : solve_system := 
  sorry

end positive_difference_proof_l21_21970


namespace arrange_digits_divisible_by_2_to_18_l21_21995

theorem arrange_digits_divisible_by_2_to_18: 
  ∃ n: ℕ, 
  (∃ lst: List ℕ, lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] ∧
   (lst.permutations.any (λ p, ∃ (m: ℕ) (h: m = p.foldl (λ acc d, acc * 10 + d) 0), 
    (∀ k in (list.range 17).map (+2), m % k = 0)))) := 
begin
  sorry
end

end arrange_digits_divisible_by_2_to_18_l21_21995


namespace grocer_initial_stock_l21_21984

theorem grocer_initial_stock 
  (x : ℝ) 
  (h1 : 0.20 * x + 70 = 0.30 * (x + 100)) : 
  x = 400 := by
  sorry

end grocer_initial_stock_l21_21984


namespace solve_system_l21_21553

theorem solve_system :
  (∀ x y : ℝ, log 4 x - log 2 y = 0 ∧ x^2 - 5 * y^2 + 4 = 0 → 
    (x, y) = (1, 1) ∨ (x, y) = (4, 2)) :=
by
  intros x y h
  cases h with Hlog Heq
  sorry

end solve_system_l21_21553


namespace largest_of_7_consecutive_numbers_with_average_20_l21_21649

variable (n : ℤ) 

theorem largest_of_7_consecutive_numbers_with_average_20
  (h_avg : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6))/7 = 20) : 
  (n + 6) = 23 :=
by
  -- Placeholder for the actual proof
  sorry

end largest_of_7_consecutive_numbers_with_average_20_l21_21649


namespace triangle_area_is_zero_l21_21718

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def scalar_vector_mult (k : ℝ) (v : Point3D) : Point3D := {
  x := k * v.x,
  y := k * v.y,
  z := k * v.z
}

theorem triangle_area_is_zero : 
  let u := Point3D.mk 2 1 (-1)
  let v := Point3D.mk 5 4 1
  let w := Point3D.mk 11 10 5
  vector_sub w u = scalar_vector_mult 3 (vector_sub v u) →
-- If the points u, v, w are collinear, the area of the triangle formed by these points is zero:
  ∃ area : ℝ, area = 0 :=
by {
  sorry
}

end triangle_area_is_zero_l21_21718


namespace factorization_of_expression_l21_21744

theorem factorization_of_expression
  (a b c : ℝ)
  (expansion : (b+c)*(c+a)*(a+b) + abc = (a+b+c)*(ab+ac+bc)) : 
  ∃ (m l : ℝ), (m = 0 ∧ l = a + b + c ∧ 
  (b+c)*(c+a)*(a+b) + abc = m*(a^2 + b^2 + c^2) + l*(ab + ac + bc)) :=
by
  sorry

end factorization_of_expression_l21_21744


namespace arrangement_count_l21_21558

-- Definitions corresponding to the conditions in a)
def num_students : ℕ := 8
def max_per_activity : ℕ := 5

-- Lean statement reflecting the target theorem in c)
theorem arrangement_count (n : ℕ) (max : ℕ) 
  (h1 : n = num_students)
  (h2 : max = max_per_activity) :
  ∃ total : ℕ, total = 182 :=
sorry

end arrangement_count_l21_21558


namespace AdjacentComplementaryAnglesAreComplementary_l21_21978

-- Definitions of angles related to the propositions

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def complementary (θ φ : ℝ) : Prop := θ + φ = π / 2
def adjacent_complementary (θ φ : ℝ) : Prop := complementary θ φ ∧ θ ≠ φ  -- Simplified for demonstration
def corresponding (θ φ : ℝ) : Prop := sorry  -- Definition placeholder
def interior_alternate (θ φ : ℝ) : Prop := sorry  -- Definition placeholder

theorem AdjacentComplementaryAnglesAreComplementary :
  ∀ θ φ : ℝ, adjacent_complementary θ φ → complementary θ φ :=
by
  intros θ φ h
  cases h with h_complementary h_adjacent
  exact h_complementary

end AdjacentComplementaryAnglesAreComplementary_l21_21978


namespace greatest_root_of_gx_l21_21721

theorem greatest_root_of_gx :
  ∃ x : ℝ, (10 * x^4 - 16 * x^2 + 3 = 0) ∧ (∀ y : ℝ, (10 * y^4 - 16 * y^2 + 3 = 0) → x ≥ y) ∧ x = Real.sqrt (3 / 5) := 
sorry

end greatest_root_of_gx_l21_21721


namespace mark_bananas_equals_mike_matt_fruits_l21_21570

theorem mark_bananas_equals_mike_matt_fruits :
  (∃ (bananas_mike matt_apples mark_bananas : ℕ),
    bananas_mike = 3 ∧
    matt_apples = 2 * bananas_mike ∧
    mark_bananas = 18 - (bananas_mike + matt_apples) ∧
    mark_bananas = (bananas_mike + matt_apples)) :=
sorry

end mark_bananas_equals_mike_matt_fruits_l21_21570


namespace smallest_square_area_l21_21555

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 4) (h4 : d = 5) :
  ∃ s, s^2 = 81 ∧ (a ≤ s ∧ b ≤ s ∧ c ≤ s ∧ d ≤ s ∧ (a + c) ≤ s ∧ (b + d) ≤ s) :=
sorry

end smallest_square_area_l21_21555


namespace monotonically_increasing_interval_l21_21786

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^3

theorem monotonically_increasing_interval : ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

end monotonically_increasing_interval_l21_21786


namespace jessy_initial_earrings_l21_21273

theorem jessy_initial_earrings (E : ℕ) (h₁ : 20 + E + (2 / 3 : ℚ) * E + (2 / 15 : ℚ) * E = 57) : E = 20 :=
by
  sorry

end jessy_initial_earrings_l21_21273


namespace Kolya_is_CollectionAgency_l21_21488

-- Define the roles
inductive Role
| FinancialPyramid
| CollectionAgency
| Bank
| InsuranceCompany

-- Define the conditions parametrically
structure Scenario where
  lent_books : Bool
  promise_broken : Bool
  mediator_requested : Bool
  reward_requested : Bool

-- Define the theorem statement
theorem Kolya_is_CollectionAgency
  (scenario : Scenario)
  (h1 : scenario.lent_books = true)
  (h2 : scenario.promise_broken = true)
  (h3 : scenario.mediator_requested = true)
  (h4 : scenario.reward_requested = true) :
  Kolya_is_CollectionAgency :=
  begin
    -- Proof not required
    sorry
  end

end Kolya_is_CollectionAgency_l21_21488


namespace total_files_on_flash_drive_l21_21852

theorem total_files_on_flash_drive :
  ∀ (music_files video_files picture_files : ℝ),
    music_files = 4.0 ∧ video_files = 21.0 ∧ picture_files = 23.0 →
    music_files + video_files + picture_files = 48.0 :=
by
  sorry

end total_files_on_flash_drive_l21_21852


namespace intersect_inverse_l21_21528

theorem intersect_inverse (c d : ℤ) (h1 : 2 * (-4) + c = d) (h2 : 2 * d + c = -4) : d = -4 := 
by
  sorry

end intersect_inverse_l21_21528


namespace center_of_rotation_l21_21711

noncomputable def f (z : ℂ) : ℂ := ((-1 - (Complex.I * Real.sqrt 3)) * z + (2 * Real.sqrt 3 - 12 * Complex.I)) / 2

theorem center_of_rotation :
  ∃ c : ℂ, f c = c ∧ c = -5 * Real.sqrt 3 / 2 - 7 / 2 * Complex.I :=
by
  sorry

end center_of_rotation_l21_21711


namespace ways_to_select_computers_l21_21085

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the number of Type A and Type B computers
def num_type_a := 4
def num_type_b := 5

-- Define the total number of computers to select
def total_selected := 3

-- Define the calculation for number of ways to select the computers ensuring both types are included
def ways_to_select := binomial num_type_a 2 * binomial num_type_b 1 + binomial num_type_a 1 * binomial num_type_b 2

-- State the theorem
theorem ways_to_select_computers : ways_to_select = 70 :=
by
  -- Proof will be provided here
  sorry

end ways_to_select_computers_l21_21085


namespace drowning_ratio_l21_21681

variable (total_sheep total_cows total_dogs drowned_sheep drowned_cows total_animals : ℕ)

-- Conditions provided
variable (initial_conditions : total_sheep = 20 ∧ total_cows = 10 ∧ total_dogs = 14)
variable (sheep_drowned_condition : drowned_sheep = 3)
variable (dogs_shore_condition : total_dogs = 14)
variable (total_made_it_shore : total_animals = 35)

theorem drowning_ratio (h1 : total_sheep = 20) (h2 : total_cows = 10) (h3 : total_dogs = 14) 
    (h4 : drowned_sheep = 3) (h5 : total_animals = 35) 
    : (drowned_cows = 2 * drowned_sheep) :=
by
  sorry

end drowning_ratio_l21_21681


namespace cooking_people_count_l21_21326

variables (P Y W : ℕ)

def people_practicing_yoga := 25
def people_studying_weaving := 8
def people_studying_only_cooking := 2
def people_studying_cooking_and_yoga := 7
def people_studying_cooking_and_weaving := 3
def people_studying_all_curriculums := 3

theorem cooking_people_count :
  P = people_studying_only_cooking + (people_studying_cooking_and_yoga - people_studying_all_curriculums)
    + (people_studying_cooking_and_weaving - people_studying_all_curriculums) + people_studying_all_curriculums →
  P = 9 :=
by
  intro h
  unfold people_studying_only_cooking people_studying_cooking_and_yoga people_studying_cooking_and_weaving people_studying_all_curriculums at h
  sorry

end cooking_people_count_l21_21326


namespace sum_999_is_1998_l21_21791

theorem sum_999_is_1998 : 999 + 999 = 1998 :=
by
  sorry

end sum_999_is_1998_l21_21791


namespace polynomial_division_l21_21671

noncomputable def polynomial_div_quotient (p q : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.divByMonic p q

theorem polynomial_division 
  (p q : Polynomial ℚ)
  (hq : q = Polynomial.C 3 * Polynomial.X - Polynomial.C 4)
  (hp : p = 10 * Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 8 * Polynomial.X - 9) :
  polynomial_div_quotient p q = (10 / 3) * Polynomial.X ^ 2 - (55 / 9) * Polynomial.X - (172 / 27) :=
by
  sorry

end polynomial_division_l21_21671


namespace probability_even_product_l21_21579

def is_even (n : ℕ) : Prop := n % 2 = 0

def chips_box_A := {1, 2, 4}
def chips_box_B := {1, 3, 5}

def total_outcomes : ℕ := chips_box_A.card * chips_box_B.card

def favorable_outcomes : ℕ :=
  (chips_box_A.filter is_even).card * chips_box_B.card

theorem probability_even_product : 
  (favorable_outcomes.to_rat / total_outcomes.to_rat) = (2 : ℚ / 3 : ℚ) :=
by
  sorry

end probability_even_product_l21_21579


namespace find_k_l21_21263

theorem find_k (k : ℝ) (h : 32 / k = 4) : k = 8 := sorry

end find_k_l21_21263


namespace sqrt_meaningful_range_l21_21254

theorem sqrt_meaningful_range (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l21_21254


namespace simplify_and_evaluate_expression_l21_21363

def a : ℚ := 1 / 3
def b : ℚ := -1
def expr : ℚ := 4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b)

theorem simplify_and_evaluate_expression : expr = -3 := 
by
  sorry

end simplify_and_evaluate_expression_l21_21363


namespace solve_for_b_l21_21246

theorem solve_for_b (b : ℚ) : 
  (∃ m1 m2 : ℚ, 3 * m1 - 2 * 1 + 4 = 0 ∧ 5 * m2 + b * 1 - 1 = 0 ∧ m1 * m2 = -1) → b = 15 / 2 :=
by
  sorry

end solve_for_b_l21_21246


namespace total_hours_watched_l21_21414

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end total_hours_watched_l21_21414


namespace sequence_value_is_correct_l21_21333

theorem sequence_value_is_correct (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2) : a 8 = 15 :=
sorry

end sequence_value_is_correct_l21_21333


namespace roots_product_eq_348_l21_21666

theorem roots_product_eq_348 (d e : ℤ) 
  (h : ∀ (s : ℂ), s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) : 
  d * e = 348 :=
sorry

end roots_product_eq_348_l21_21666


namespace factorial_sqrt_sq_l21_21828

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l21_21828


namespace jake_correct_speed_l21_21353

noncomputable def distance (d t : ℝ) : Prop :=
  d = 50 * (t + 4/60) ∧ d = 70 * (t - 4/60)

noncomputable def correct_speed (d t : ℝ) : ℝ :=
  d / t

theorem jake_correct_speed (d t : ℝ) (h1 : distance d t) : correct_speed d t = 58 :=
by
  sorry

end jake_correct_speed_l21_21353


namespace gel_pen_ratio_l21_21125

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l21_21125


namespace g_g_g_3_eq_71_l21_21222

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1 else 2 * n + 5

theorem g_g_g_3_eq_71 : g (g (g 3)) = 71 := 
by
  sorry

end g_g_g_3_eq_71_l21_21222


namespace probability_reaching_five_without_returning_to_zero_l21_21335

def reach_position_without_return_condition (tosses : ℕ) (target : ℤ) (return_limit : ℤ) : ℕ :=
  -- Ideally we should implement the logic to find the number of valid paths here (as per problem constraints)
  sorry

theorem probability_reaching_five_without_returning_to_zero {a b : ℕ} (h_rel_prime : Nat.gcd a b = 1)
    (h_paths_valid : reach_position_without_return_condition 10 5 3 = 15) :
    a = 15 ∧ b = 256 ∧ a + b = 271 :=
by
  sorry

end probability_reaching_five_without_returning_to_zero_l21_21335


namespace charges_are_equal_l21_21871

variable (a : ℝ)  -- original price for both travel agencies

def charge_A (a : ℝ) : ℝ := a + 2 * 0.7 * a
def charge_B (a : ℝ) : ℝ := 3 * 0.8 * a

theorem charges_are_equal : charge_A a = charge_B a :=
by
  sorry

end charges_are_equal_l21_21871


namespace largest_product_of_three_l21_21568

-- Definitions of the numbers in the set
def numbers : List Int := [-5, 1, -3, 5, -2, 2]

-- Define a function to calculate the product of a list of three integers
def product_of_three (a b c : Int) : Int := a * b * c

-- Define a predicate to state that 75 is the largest product of any three numbers from the given list
theorem largest_product_of_three :
  ∃ (a b c : Int), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ product_of_three a b c = 75 :=
sorry

end largest_product_of_three_l21_21568


namespace remainder_12345678901_mod_101_l21_21802

theorem remainder_12345678901_mod_101 : 12345678901 % 101 = 24 :=
by
  sorry

end remainder_12345678901_mod_101_l21_21802


namespace find_a_and_an_l21_21323

-- Given Sequences
def S (n : ℕ) (a : ℝ) : ℝ := 3^n - a

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop := ∃ a1 q, q ≠ 1 ∧ ∀ n, a_n n = a1 * q^n

-- The main statement
theorem find_a_and_an (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, S_n n = 3^n - a) ∧ is_geometric_sequence a_n →
  ∃ a, a = 1 ∧ ∀ n, a_n n = 2 * 3^(n-1) :=
by
  sorry

end find_a_and_an_l21_21323


namespace smallest_two_digit_integer_l21_21259

-- Define the problem parameters and condition
theorem smallest_two_digit_integer (n : ℕ) (a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
  (h6 : 19 * a = 8 * b + 3) : 
  n = 12 :=
sorry

end smallest_two_digit_integer_l21_21259


namespace engineer_thought_of_l21_21993

def isProperDivisor (n k : ℕ) : Prop :=
  k ≠ 1 ∧ k ≠ n ∧ k ∣ n

def transformDivisors (n m : ℕ) : Prop :=
  ∀ k, isProperDivisor n k → isProperDivisor m (k + 1)

theorem engineer_thought_of (n : ℕ) :
  (∀ m : ℕ, n = 2^2 ∨ n = 2^3 → transformDivisors n m → (m % 2 = 1)) :=
by
  sorry

end engineer_thought_of_l21_21993


namespace reciprocal_of_neg_one_third_l21_21661

theorem reciprocal_of_neg_one_third : 
  ∃ x : ℚ, (-1 / 3) * x = 1 :=
begin
  use -3,
  sorry
end

end reciprocal_of_neg_one_third_l21_21661


namespace product_of_translated_roots_l21_21632

noncomputable def roots (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_translated_roots
  {d e : ℝ}
  (h_d : roots 3 4 (-7) d)
  (h_e : roots 3 4 (-7) e)
  (sum_roots : d + e = -4 / 3)
  (product_roots : d * e = -7 / 3) :
  (d - 1) * (e - 1) = 1 :=
by
  sorry

end product_of_translated_roots_l21_21632


namespace volume_inside_sphere_outside_cylinder_l21_21112

noncomputable def sphere_radius := 6
noncomputable def cylinder_diameter := 8
noncomputable def sphere_volume := 4/3 * Real.pi * (sphere_radius ^ 3)
noncomputable def cylinder_height := Real.sqrt ((sphere_radius * 2) ^ 2 - (cylinder_diameter) ^ 2)
noncomputable def cylinder_volume := Real.pi * ((cylinder_diameter / 2) ^ 2) * cylinder_height
noncomputable def volume_difference := sphere_volume - cylinder_volume

theorem volume_inside_sphere_outside_cylinder:
  volume_difference = (288 - 64 * Real.sqrt 5) * Real.pi :=
sorry

end volume_inside_sphere_outside_cylinder_l21_21112


namespace sandy_total_spent_on_clothes_l21_21079

theorem sandy_total_spent_on_clothes :
  let shorts := 13.99
  let shirt := 12.14 
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := 
by
  sorry

end sandy_total_spent_on_clothes_l21_21079


namespace point_P_on_x_axis_l21_21748

noncomputable def point_on_x_axis (m : ℝ) : ℝ × ℝ := (4, m + 1)

theorem point_P_on_x_axis (m : ℝ) (h : point_on_x_axis m = (4, 0)) : m = -1 := 
by
  sorry

end point_P_on_x_axis_l21_21748


namespace max_visible_unit_cubes_l21_21980

def cube_size := 11
def total_unit_cubes := cube_size ^ 3

def visible_unit_cubes (n : ℕ) : ℕ :=
  (n * n) + (n * (n - 1)) + ((n - 1) * (n - 1))

theorem max_visible_unit_cubes : 
  visible_unit_cubes cube_size = 331 := by
  sorry

end max_visible_unit_cubes_l21_21980


namespace net_income_in_June_l21_21926

theorem net_income_in_June 
  (daily_milk_production : ℕ) 
  (price_per_gallon : ℝ) 
  (daily_expense : ℝ) 
  (days_in_month : ℕ)
  (monthly_expense : ℝ) : 
  daily_milk_production = 200 →
  price_per_gallon = 3.55 →
  daily_expense = daily_milk_production * price_per_gallon →
  days_in_month = 30 →
  monthly_expense = 3000 →
  (daily_expense * days_in_month - monthly_expense) = 18300 :=
begin
  intros h_prod h_price h_daily_inc h_days h_monthly_exp,
  sorry
end

end net_income_in_June_l21_21926


namespace translation_invariant_line_l21_21678

theorem translation_invariant_line (k : ℝ) :
  (∀ x : ℝ, k * (x - 2) + 5 = k * x + 2) → k = 3 / 2 :=
by
  sorry

end translation_invariant_line_l21_21678


namespace average_score_l21_21049

theorem average_score (T : ℝ) (M F : ℝ) (avgM avgF : ℝ) 
  (h1 : M = 0.4 * T) 
  (h2 : M + F = T) 
  (h3 : avgM = 75) 
  (h4 : avgF = 80) : 
  (75 * M + 80 * F) / T = 78 := 
  by 
  sorry

end average_score_l21_21049


namespace building_time_l21_21120

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end building_time_l21_21120


namespace no_separation_sister_chromatids_first_meiotic_l21_21675

-- Definitions for the steps happening during the first meiotic division
def first_meiotic_division :=
  ∃ (prophase_I : Prop) (metaphase_I : Prop) (anaphase_I : Prop) (telophase_I : Prop),
    prophase_I ∧ metaphase_I ∧ anaphase_I ∧ telophase_I

def pairing_homologous_chromosomes (prophase_I : Prop) := prophase_I
def crossing_over (prophase_I : Prop) := prophase_I
def separation_homologous_chromosomes (anaphase_I : Prop) := anaphase_I
def separation_sister_chromatids (mitosis : Prop) (second_meiotic_division : Prop) :=
  mitosis ∨ second_meiotic_division

-- Theorem to prove that the separation of sister chromatids does not occur during the first meiotic division
theorem no_separation_sister_chromatids_first_meiotic
  (prophase_I metaphase_I anaphase_I telophase_I mitosis second_meiotic_division : Prop)
  (h1: first_meiotic_division)
  (h2 : pairing_homologous_chromosomes prophase_I)
  (h3 : crossing_over prophase_I)
  (h4 : separation_homologous_chromosomes anaphase_I)
  (h5 : separation_sister_chromatids mitosis second_meiotic_division) : 
  ¬ separation_sister_chromatids prophase_I anaphase_I :=
by
  sorry

end no_separation_sister_chromatids_first_meiotic_l21_21675


namespace functional_equation_solution_l21_21585

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (f (xy - x)) + f (x + y) = y * f (x) + f (y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end functional_equation_solution_l21_21585


namespace common_ratio_l21_21185

theorem common_ratio
  (a b : ℝ)
  (h_arith : 2 * a = 1 + b)
  (h_geom : (a + 2) ^ 2 = 3 * (b + 5))
  (h_non_zero_a : a + 2 ≠ 0)
  (h_non_zero_b : b + 5 ≠ 0) :
  (a = 4 ∧ b = 7) ∧ (b + 5) / (a + 2) = 2 :=
by {
  sorry
}

end common_ratio_l21_21185


namespace parallelogram_height_l21_21437

theorem parallelogram_height (A b h : ℝ) (hA : A = 288) (hb : b = 18) : h = 16 :=
by
  sorry

end parallelogram_height_l21_21437


namespace sector_arc_length_120_degrees_radius_3_l21_21321

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem sector_arc_length_120_degrees_radius_3 :
  arc_length 120 3 = 2 * Real.pi :=
by
  sorry

end sector_arc_length_120_degrees_radius_3_l21_21321


namespace min_value_frac_expr_l21_21631

theorem min_value_frac_expr (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a < 1) (h₃ : 0 ≤ b) (h₄ : b < 1) (h₅ : 0 ≤ c) (h₆ : c < 1) :
  (1 / ((2 - a) * (2 - b) * (2 - c)) + 1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1 / 8 :=
sorry

end min_value_frac_expr_l21_21631


namespace sqrt_factorial_mul_square_l21_21823

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l21_21823


namespace no_solution_fraction_eq_l21_21521

theorem no_solution_fraction_eq (x : ℝ) : 
  (1 / (x - 2) = (1 - x) / (2 - x) - 3) → False := 
by 
  sorry

end no_solution_fraction_eq_l21_21521


namespace number_of_factors_l21_21575

theorem number_of_factors (a b c d : ℕ) (h₁ : a = 6) (h₂ : b = 6) (h₃ : c = 5) (h₄ : d = 1) :
  ((a + 1) * (b + 1) * (c + 1) * (d + 1) = 588) :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

end number_of_factors_l21_21575


namespace prob_sin_ge_half_l21_21559

theorem prob_sin_ge_half : 
  let a := -Real.pi / 6
  let b := Real.pi / 2
  let p := (Real.pi / 2 - Real.pi / 6) / (Real.pi / 2 + Real.pi / 6)
  a ≤ b ∧ a = -Real.pi / 6 ∧ b = Real.pi / 2 → p = 1 / 2 :=
by
  sorry

end prob_sin_ge_half_l21_21559


namespace sum_symmetric_prob_43_l21_21944

def prob_symmetric_sum_43_with_20 : Prop :=
  let n_dice := 9
  let min_sum := n_dice * 1
  let max_sum := n_dice * 6
  let midpoint := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * midpoint - 20
  symmetric_sum = 43

theorem sum_symmetric_prob_43 (n_dice : ℕ) (h₁ : n_dice = 9) (h₂ : ∀ i : ℕ, i ≥ 1 ∧ i ≤ 6) :
  prob_symmetric_sum_43_with_20 :=
by
  sorry

end sum_symmetric_prob_43_l21_21944


namespace baseball_cards_given_l21_21546

theorem baseball_cards_given
  (initial_cards : ℕ)
  (maria_take : ℕ)
  (peter_cards : ℕ)
  (paul_triples : ℕ)
  (final_cards : ℕ)
  (h1 : initial_cards = 15)
  (h2 : maria_take = (initial_cards + 1) / 2)
  (h3 : final_cards = 3 * (initial_cards - maria_take - peter_cards))
  (h4 : final_cards = 18) :
  peter_cards = 1 := 
sorry

end baseball_cards_given_l21_21546


namespace staircase_steps_180_toothpicks_l21_21233

-- Condition definition: total number of toothpicks for \( n \) steps is \( n(n + 1) \)
def total_toothpicks (n : ℕ) : ℕ := n * (n + 1)

-- Theorem statement: for 180 toothpicks, the number of steps \( n \) is 12
theorem staircase_steps_180_toothpicks : ∃ n : ℕ, total_toothpicks n = 180 ∧ n = 12 :=
by sorry

end staircase_steps_180_toothpicks_l21_21233


namespace additional_tiles_needed_l21_21695

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end additional_tiles_needed_l21_21695


namespace geometric_series_common_ratio_l21_21701

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by
  rw [ha, hS] at hS_eq
  -- This statement follows from algebraic manipulation outlined in the solution steps.
  sorry

end geometric_series_common_ratio_l21_21701


namespace remaining_pieces_total_l21_21201

noncomputable def initial_pieces : Nat := 16
noncomputable def kennedy_lost_pieces : Nat := 4 + 1 + 2
noncomputable def riley_lost_pieces : Nat := 1 + 1 + 1

theorem remaining_pieces_total : (initial_pieces - kennedy_lost_pieces) + (initial_pieces - riley_lost_pieces) = 22 := by
  sorry

end remaining_pieces_total_l21_21201


namespace julia_played_tag_l21_21341

/-
Problem:
Let m be the number of kids Julia played with on Monday.
Let t be the number of kids Julia played with on Tuesday.
m = 24
m = t + 18
Show that t = 6
-/

theorem julia_played_tag (m t : ℕ) (h1 : m = 24) (h2 : m = t + 18) : t = 6 :=
by
  sorry

end julia_played_tag_l21_21341


namespace kaleb_tickets_l21_21996

variable (T : Nat)
variable (tickets_left : Nat) (ticket_cost : Nat) (total_spent : Nat)

theorem kaleb_tickets : tickets_left = 3 → ticket_cost = 9 → total_spent = 27 → T = 6 :=
by
  sorry

end kaleb_tickets_l21_21996


namespace percentage_problem_l21_21195

-- Define the main proposition
theorem percentage_problem (n : ℕ) (a : ℕ) (b : ℕ) (P : ℕ) :
  n = 6000 →
  a = (50 * n) / 100 →
  b = (30 * a) / 100 →
  (P * b) / 100 = 90 →
  P = 10 :=
by
  intros h_n h_a h_b h_Pb
  sorry

end percentage_problem_l21_21195


namespace solve_monetary_prize_problem_l21_21691

def monetary_prize_problem : Prop :=
  ∃ (P x y : ℝ), 
    P = x + y + 30000 ∧
    x = (1/2) * P - (3/22) * (y + 30000) ∧
    y = (1/4) * P + (1/56) * x ∧
    P = 95000 ∧
    x = 40000 ∧
    y = 25000

theorem solve_monetary_prize_problem : monetary_prize_problem :=
  sorry

end solve_monetary_prize_problem_l21_21691


namespace seating_arrangement_l21_21010

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l21_21010


namespace sum_a_b_l21_21459

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) : a + b = 176 / 57 :=
by
  sorry

end sum_a_b_l21_21459


namespace infinite_series_sum_l21_21286

theorem infinite_series_sum :
  (∑' n : ℕ, (3:ℝ)^n / (1 + (3:ℝ)^n + (3:ℝ)^(n+1) + (3:ℝ)^(2*n+2))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l21_21286


namespace ratio_of_nuts_to_raisins_l21_21708

theorem ratio_of_nuts_to_raisins 
  (R N : ℝ) 
  (h_ratio : 3 * R = 0.2727272727272727 * (3 * R + 4 * N)) : 
  N = 2 * R := 
sorry

end ratio_of_nuts_to_raisins_l21_21708


namespace no_coprime_odd_numbers_for_6_8_10_l21_21265

theorem no_coprime_odd_numbers_for_6_8_10 :
  ∀ (m n : ℤ), m > n ∧ n > 0 ∧ (m.gcd n = 1) ∧ (m % 2 = 1) ∧ (n % 2 = 1) →
    (1 / 2 : ℚ) * (m^2 - n^2) ≠ 6 ∨ (m * n) ≠ 8 ∨ (1 / 2 : ℚ) * (m^2 + n^2) ≠ 10 :=
by
  sorry

end no_coprime_odd_numbers_for_6_8_10_l21_21265


namespace parking_lot_perimeter_l21_21845

theorem parking_lot_perimeter (a b : ℝ) 
  (h_diag : a^2 + b^2 = 784) 
  (h_area : a * b = 180) : 
  2 * (a + b) = 68 := 
by 
  sorry

end parking_lot_perimeter_l21_21845


namespace average_of_first_16_even_numbers_l21_21551

theorem average_of_first_16_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30 + 32) / 16 = 17 := 
by sorry

end average_of_first_16_even_numbers_l21_21551


namespace num_lines_in_grid_l21_21032

theorem num_lines_in_grid (columns rows : ℕ) (H1 : columns = 4) (H2 : rows = 3) 
    (total_points : ℕ) (H3 : total_points = columns * rows) :
    ∃ lines, lines = 40 :=
by
  sorry

end num_lines_in_grid_l21_21032


namespace find_A_B_l21_21292

theorem find_A_B :
  ∀ (A B : ℝ), (∀ (x : ℝ), 1 < x → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  (A = 0) ∧ (B = 1) :=
by
  sorry

end find_A_B_l21_21292


namespace train1_speed_l21_21797

noncomputable def total_distance_in_kilometers : ℝ :=
  (630 + 100 + 200) / 1000

noncomputable def time_in_hours : ℝ :=
  13.998880089592832 / 3600

noncomputable def relative_speed : ℝ :=
  total_distance_in_kilometers / time_in_hours

noncomputable def speed_of_train2 : ℝ :=
  72

noncomputable def speed_of_train1 : ℝ :=
  relative_speed - speed_of_train2

theorem train1_speed : speed_of_train1 = 167.076 := by 
  sorry

end train1_speed_l21_21797


namespace identity_of_polynomials_l21_21513

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l21_21513


namespace compute_sqrt_factorial_square_l21_21824

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l21_21824


namespace second_statue_weight_l21_21454

theorem second_statue_weight (S : ℕ) :
  ∃ S : ℕ,
    (80 = 10 + S + 15 + 15 + 22) → S = 18 :=
by
  sorry

end second_statue_weight_l21_21454


namespace factorize_expression_l21_21013

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end factorize_expression_l21_21013


namespace players_either_left_handed_or_throwers_l21_21536

theorem players_either_left_handed_or_throwers (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 34) (h3 : ∀ n, n = total_players - throwers → 1 / 3 * n = n / 3) :
  ∃ n, n = 46 := 
sorry

end players_either_left_handed_or_throwers_l21_21536


namespace team_win_percentage_remaining_l21_21114

theorem team_win_percentage_remaining (won_first_30: ℝ) (total_games: ℝ) (total_wins: ℝ)
  (h1: won_first_30 = 0.40 * 30)
  (h2: total_games = 120)
  (h3: total_wins = 0.70 * total_games) :
  (total_wins - won_first_30) / (total_games - 30) * 100 = 80 :=
by
  sorry


end team_win_percentage_remaining_l21_21114


namespace contrapositive_of_given_condition_l21_21503

-- Definitions
variable (P Q : Prop)

-- Given condition: If Jane answered all questions correctly, she will get a prize
axiom h : P → Q

-- Statement to be proven: If Jane did not get a prize, she answered at least one question incorrectly
theorem contrapositive_of_given_condition : ¬ Q → ¬ P := by
  sorry

end contrapositive_of_given_condition_l21_21503


namespace functional_expression_result_l21_21245

theorem functional_expression_result {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y) :
  ∀ t : ℝ, (f (4 * t) - f t) / (f (3 * t) - f (2 * t)) = 3 :=
sorry

end functional_expression_result_l21_21245


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21152

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21152


namespace simplify_expression_l21_21087

variable (a : ℝ)

theorem simplify_expression : 3 * a^2 - a * (2 * a - 1) = a^2 + a :=
by
  sorry

end simplify_expression_l21_21087


namespace find_integers_l21_21443

theorem find_integers (a b : ℤ) (h1 : a * b = a + b) (h2 : a * b = a - b) : a = 0 ∧ b = 0 :=
by 
  sorry

end find_integers_l21_21443


namespace find_y_l21_21898

theorem find_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 :=
sorry

end find_y_l21_21898


namespace factorial_expression_value_l21_21814

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l21_21814


namespace upper_limit_of_sixth_powers_l21_21693

theorem upper_limit_of_sixth_powers :
  ∃ b : ℕ, (∀ n : ℕ, (∃ a : ℕ, a^6 = n) ∧ n ≤ b → n = 46656) :=
by
  sorry

end upper_limit_of_sixth_powers_l21_21693


namespace evaluate_expression_l21_21175

theorem evaluate_expression : (900^2 / (153^2 - 147^2)) = 450 := by
  sorry

end evaluate_expression_l21_21175


namespace bananas_to_oranges_l21_21617

theorem bananas_to_oranges (B A O : ℕ) 
    (h1 : 4 * B = 3 * A) 
    (h2 : 7 * A = 5 * O) : 
    28 * B = 15 * O :=
by
  sorry

end bananas_to_oranges_l21_21617


namespace jake_present_weight_l21_21461

theorem jake_present_weight (J S B : ℝ) (h1 : J - 20 = 2 * S) (h2 : B = 0.5 * J) (h3 : J + S + B = 330) :
  J = 170 :=
by sorry

end jake_present_weight_l21_21461


namespace rectangle_square_area_ratio_eq_one_l21_21099

theorem rectangle_square_area_ratio_eq_one (r l w s: ℝ) (h1: l = 2 * w) (h2: r ^ 2 = (l / 2) ^ 2 + w ^ 2) (h3: s ^ 2 = 2 * r ^ 2) : 
  (l * w) / (s ^ 2) = 1 :=
by
sorry

end rectangle_square_area_ratio_eq_one_l21_21099


namespace probability_given_A_l21_21022

-- Define the sample space
def sample_space : set (ℕ × ℕ) := {(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)}

-- Define events A and B
def event_A : set (ℕ × ℕ) := {x ∈ sample_space | (x.1 + x.2) % 2 = 0}
def event_B : set (ℕ × ℕ) := {x ∈ sample_space | x.1 % 2 = 0 ∧ x.2 % 2 = 0}

-- Define the intersection of A and B
def event_A_and_B : set (ℕ × ℕ) := event_A ∩ event_B

theorem probability_given_A (h : ∀ x ∈ event_A, ¬ x ∈ event_B) : 
  ∑' (x : event_B), (1 : ℝ) / (event_A ∩ event_B).to_finset.card =
  1/4 := 
sorry

end probability_given_A_l21_21022


namespace no_contradiction_to_thermodynamics_l21_21729

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l21_21729


namespace components_le_20_components_le_n_squared_div_4_l21_21056

-- Question part b: 8x8 grid, can the number of components be more than 20
theorem components_le_20 {c : ℕ} (h1 : c = 64 / 4) : c ≤ 20 := by
  sorry

-- Question part c: n x n grid, can the number of components be more than n^2 / 4
theorem components_le_n_squared_div_4 (n : ℕ) (h2 : n > 8) {c : ℕ} (h3 : c = n^2 / 4) : 
  c ≤ n^2 / 4 := by
  sorry

end components_le_20_components_le_n_squared_div_4_l21_21056


namespace swap_original_x_y_l21_21992

variables (x y z : ℕ)

theorem swap_original_x_y (x_original y_original : ℕ) 
  (step1 : z = x_original)
  (step2 : x = y_original)
  (step3 : y = z) :
  x = y_original ∧ y = x_original :=
sorry

end swap_original_x_y_l21_21992


namespace vector_perpendicular_vector_parallel_l21_21742


variables {x : ℝ}
def vector_a : ℝ × ℝ × ℝ := (2, -1, 5)
def vector_b : ℝ × ℝ × ℝ := (-4, 2, x)

-- Prove that if vectors are perpendicular, x = 2
theorem vector_perpendicular : (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 + vector_a.3 * vector_b.3 = 0) → x = 2 :=
by sorry

-- Prove that if vectors are parallel, x = -10
theorem vector_parallel : (vector_b.1 / vector_a.1 = vector_b.2 / vector_a.2 ∧ vector_b.1 / vector_a.1 = vector_b.3 / vector_a.3) → x = -10 :=
by sorry

end vector_perpendicular_vector_parallel_l21_21742


namespace compare_fractions_l21_21055

variable {a b : ℝ}

theorem compare_fractions (h1 : 3 * a > b) (h2 : b > 0) :
  (a / b) > ((a + 1) / (b + 3)) :=
by
  sorry

end compare_fractions_l21_21055


namespace number_of_integer_values_x_floor_2_sqrt_x_eq_12_l21_21296

theorem number_of_integer_values_x_floor_2_sqrt_x_eq_12 :
  ∃! n : ℕ, n = 7 ∧ (∀ x : ℕ, (⌊2 * Real.sqrt x⌋ = 12 ↔ 36 ≤ x ∧ x < 43)) :=
by 
  sorry

end number_of_integer_values_x_floor_2_sqrt_x_eq_12_l21_21296


namespace trigonometric_identity_l21_21302

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 := 
by 
  sorry

end trigonometric_identity_l21_21302


namespace sum_increased_consecutive_integers_product_990_l21_21092

theorem sum_increased_consecutive_integers_product_990 
  (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 990) :
  (a + 2) + (b + 2) + (c + 2) = 36 :=
sorry

end sum_increased_consecutive_integers_product_990_l21_21092


namespace bert_money_left_l21_21857

theorem bert_money_left (initial_money : ℕ) (spent_hardware : ℕ) (spent_cleaners : ℕ) (spent_grocery : ℕ) :
  initial_money = 52 →
  spent_hardware = initial_money * 1 / 4 →
  spent_cleaners = 9 →
  spent_grocery = (initial_money - spent_hardware - spent_cleaners) / 2 →
  initial_money - spent_hardware - spent_cleaners - spent_grocery = 15 := 
by
  intros h_initial h_hardware h_cleaners h_grocery
  rw [h_initial, h_hardware, h_cleaners, h_grocery]
  sorry

end bert_money_left_l21_21857


namespace number_of_sodas_bought_l21_21673

-- Definitions based on conditions
def cost_sandwich : ℝ := 1.49
def cost_two_sandwiches : ℝ := 2 * cost_sandwich
def cost_soda : ℝ := 0.87
def total_cost : ℝ := 6.46

-- We need to prove that the number of sodas bought is 4 given these conditions
theorem number_of_sodas_bought : (total_cost - cost_two_sandwiches) / cost_soda = 4 := by
  sorry

end number_of_sodas_bought_l21_21673


namespace number_of_five_digit_numbers_with_one_odd_digit_l21_21455

def odd_digits : List ℕ := [1, 3, 5, 7, 9]
def even_digits : List ℕ := [0, 2, 4, 6, 8]

def five_digit_numbers_with_one_odd_digit : ℕ :=
  let num_1st_position := odd_digits.length * even_digits.length ^ 4
  let num_other_positions := 4 * odd_digits.length * (even_digits.length - 1) * (even_digits.length ^ 3)
  num_1st_position + num_other_positions

theorem number_of_five_digit_numbers_with_one_odd_digit :
  five_digit_numbers_with_one_odd_digit = 10625 :=
by
  sorry

end number_of_five_digit_numbers_with_one_odd_digit_l21_21455


namespace find_y_intercept_l21_21018

-- Conditions
def line_equation (x y : ℝ) : Prop := 4 * x + 7 * y - 3 * x * y = 28

-- Statement (Proof Problem)
theorem find_y_intercept : ∃ y : ℝ, line_equation 0 y ∧ (0, y) = (0, 4) := by
  sorry

end find_y_intercept_l21_21018


namespace bill_toilet_paper_duration_l21_21574

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end bill_toilet_paper_duration_l21_21574


namespace max_volume_tetrahedron_l21_21205

-- Definitions and conditions
def SA : ℝ := 4
def AB : ℝ := 5
def SB_min : ℝ := 7
def SC_min : ℝ := 9
def BC_max : ℝ := 6
def AC_max : ℝ := 8

-- Proof statement
theorem max_volume_tetrahedron {SB SC BC AC : ℝ} (hSB : SB ≥ SB_min) (hSC : SC ≥ SC_min) (hBC : BC ≤ BC_max) (hAC : AC ≤ AC_max) :
  ∃ V : ℝ, V = 8 * Real.sqrt 6 ∧ V ≤ (1/3) * (1/2) * SA * AB * (2 * Real.sqrt 6) * BC := by
  sorry

end max_volume_tetrahedron_l21_21205


namespace quadratic_function_m_value_l21_21891

theorem quadratic_function_m_value :
  ∃ m : ℝ, (m - 3 ≠ 0) ∧ (m^2 - 7 = 2) ∧ m = -3 :=
by
  sorry

end quadratic_function_m_value_l21_21891


namespace Eunji_score_equals_56_l21_21473

theorem Eunji_score_equals_56 (Minyoung_score Yuna_score : ℕ) (Eunji_score : ℕ) 
  (h1 : Minyoung_score = 55) (h2 : Yuna_score = 57)
  (h3 : Eunji_score > Minyoung_score) (h4 : Eunji_score < Yuna_score) : Eunji_score = 56 := by
  -- Given the hypothesis, it is a fact that Eunji's score is 56.
  sorry

end Eunji_score_equals_56_l21_21473


namespace max_sum_unique_digits_expression_equivalent_l21_21584

theorem max_sum_unique_digits_expression_equivalent :
  ∃ (a b c d e : ℕ), (2 * 19 * 53 = 2014) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (2 * (b + c) * (d + e) = 2014) ∧
    (a + b + c + d + e = 35) ∧ 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) :=
by
  sorry

end max_sum_unique_digits_expression_equivalent_l21_21584


namespace triangle_rotation_sum_eq_l21_21668

/-
Triangles ΔDEF and ΔD'E'F' are positioned in the coordinate plane with vertices 
D(0,0), E(0,10), F(14,0), D'(20,20), E'(30,20), F'(20,8). Determine the angle 
of rotation n degrees clockwise around the point (p,q) where 0<n<180, 
that transforms ΔDEF to ΔD'E'F'. 
Find n + p + q.
-/

-- Definitions based on the problem statement
def Point := (ℝ, ℝ)

def D : Point := (0, 0)
def E : Point := (0, 10)
def F : Point := (14, 0)
def D' : Point := (20, 20)
def E' : Point := (30, 20)
def F' : Point := (20, 8)

-- Lean 4 statement that asserts the rotation and sum condition
theorem triangle_rotation_sum_eq :
  ∃ (n p q : ℝ), 0 < n ∧ n < 180 ∧
  n = 90 ∧ p = 20 ∧ q = -20 ∧
  n + p + q = 90 :=
by {
  use [90, 20, -20],
  simp,
  sorry
}

end triangle_rotation_sum_eq_l21_21668


namespace units_digit_7_pow_3_pow_4_l21_21442

theorem units_digit_7_pow_3_pow_4 :
  (7 ^ (3 ^ 4)) % 10 = 7 :=
by
  -- Here's the proof placeholder
  sorry

end units_digit_7_pow_3_pow_4_l21_21442


namespace daisies_bought_l21_21702

theorem daisies_bought (cost_per_flower roses total_cost : ℕ) 
  (h1 : cost_per_flower = 3) 
  (h2 : roses = 8) 
  (h3 : total_cost = 30) : 
  (total_cost - (roses * cost_per_flower)) / cost_per_flower = 2 :=
by
  sorry

end daisies_bought_l21_21702


namespace original_ratio_l21_21973

theorem original_ratio (x y : ℤ) (h₁ : y = 72) (h₂ : (x + 6) / y = 1 / 3) : y / x = 4 := 
by
  sorry

end original_ratio_l21_21973


namespace prob_le_45_l21_21612

-- Define the probability conditions
def prob_between_1_and_45 : ℚ := 7 / 15
def prob_ge_1 : ℚ := 14 / 15

-- State the theorem to prove
theorem prob_le_45 : prob_between_1_and_45 = 7 / 15 := by
  sorry

end prob_le_45_l21_21612


namespace translate_quadratic_function_l21_21795

theorem translate_quadratic_function :
  ∀ x : ℝ, (y = (1 / 3) * x^2) →
          (y₂ = (1 / 3) * (x - 1)^2) →
          (y₃ = y₂ + 3) →
          y₃ = (1 / 3) * (x - 1)^2 + 3 := 
by 
  intros x h₁ h₂ h₃ 
  sorry

end translate_quadratic_function_l21_21795


namespace correct_conditions_for_cubic_eq_single_root_l21_21761

noncomputable def hasSingleRealRoot (a b : ℝ) : Prop :=
  let f := λ x : ℝ => x^3 - a * x + b
  let f' := λ x : ℝ => 3 * x^2 - a
  ∀ (x y : ℝ), f' x = 0 → f' y = 0 → x = y

theorem correct_conditions_for_cubic_eq_single_root :
  (hasSingleRealRoot 0 2) ∧ 
  (hasSingleRealRoot (-3) 2) ∧ 
  (hasSingleRealRoot 3 (-3)) :=
  by 
    sorry

end correct_conditions_for_cubic_eq_single_root_l21_21761


namespace max_binomial_term_l21_21874

noncomputable def binomial_term (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * (real.sqrt 11) ^ k

theorem max_binomial_term :
  (argmax (binomial_term 208) = 160) :=
by
  -- Proof required here
  sorry

end max_binomial_term_l21_21874


namespace smallest_piece_length_l21_21116

theorem smallest_piece_length {x : ℝ} : (5 - x) + (12 - x) ≤ (13 - x) → 
                                     x ≥ 4 :=
by
  intro h
  have h1 : 17 - 2 * x ≤ 13 - x := h
  linarith

end smallest_piece_length_l21_21116


namespace speed_of_first_train_l21_21539

-- Define the conditions
def distance_pq := 110 -- km
def speed_q := 25 -- km/h
def meet_time := 10 -- hours from midnight
def start_p := 7 -- hours from midnight
def start_q := 8 -- hours from midnight

-- Define the total travel time for each train
def travel_time_p := meet_time - start_p -- hours
def travel_time_q := meet_time - start_q -- hours

-- Define the distance covered by each train
def distance_covered_p (V_p : ℕ) : ℕ := V_p * travel_time_p
def distance_covered_q := speed_q * travel_time_q

-- Theorem to prove the speed of the first train
theorem speed_of_first_train (V_p : ℕ) : distance_covered_p V_p + distance_covered_q = distance_pq → V_p = 20 :=
sorry

end speed_of_first_train_l21_21539


namespace evaluate_expression_l21_21235

variable (a b : ℤ)

-- Define the original expression
def orig_expr (a b : ℤ) : ℤ :=
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (b^2 * a - 2 * a^2 * b + 1)

-- Specify the values for a and b
def a_val : ℤ := -1
def b_val : ℤ := 1

-- Prove that the expression evaluates to 10 when a = -1 and b = 1
theorem evaluate_expression : orig_expr a_val b_val = 10 := 
  by sorry

end evaluate_expression_l21_21235


namespace not_inequality_l21_21522

theorem not_inequality (x : ℝ) : ¬ (x^2 + 2*x - 3 < 0) :=
sorry

end not_inequality_l21_21522


namespace correct_option_l21_21931

-- Define the operations as functions to be used in the Lean statement.
def optA : ℕ := 3 + 5 * 7 + 9
def optB : ℕ := 3 + 5 + 7 * 9
def optC : ℕ := 3 * 5 * 7 - 9
def optD : ℕ := 3 * 5 * 7 + 9
def optE : ℕ := 3 * 5 + 7 * 9

-- The theorem to prove that the correct option is (E).
theorem correct_option : optE = 78 ∧ optA ≠ 78 ∧ optB ≠ 78 ∧ optC ≠ 78 ∧ optD ≠ 78 := by {
  sorry
}

end correct_option_l21_21931


namespace intersection_point_exists_l21_21777

def h : ℝ → ℝ := sorry  -- placeholder for the function h
def j : ℝ → ℝ := sorry  -- placeholder for the function j

-- Conditions
axiom h_3_eq : h 3 = 3
axiom j_3_eq : j 3 = 3
axiom h_6_eq : h 6 = 9
axiom j_6_eq : j 6 = 9
axiom h_9_eq : h 9 = 18
axiom j_9_eq : j 9 = 18

-- Theorem
theorem intersection_point_exists :
  ∃ a b : ℝ, a = 2 ∧ h (3 * a) = 3 * j (a) ∧ h (3 * a) = b ∧ 3 * j (a) = b ∧ a + b = 11 :=
  sorry

end intersection_point_exists_l21_21777


namespace machines_work_together_l21_21971

theorem machines_work_together (x : ℝ) (h₁ : 1/(x+4) + 1/(x+2) + 1/(x+3) = 1/x) : x = 1 :=
sorry

end machines_work_together_l21_21971


namespace simplify_and_evaluate_l21_21236

noncomputable def expr (x : ℝ) : ℝ :=
  (x + 3) * (x - 2) + x * (4 - x)

theorem simplify_and_evaluate (x : ℝ) (hx : x = 2) : expr x = 4 :=
by
  rw [hx]
  show expr 2 = 4
  sorry

end simplify_and_evaluate_l21_21236


namespace percentage_basketball_l21_21906

theorem percentage_basketball (total_students : ℕ) (chess_percentage : ℝ) (students_like_chess_basketball : ℕ) 
  (percentage_conversion : ∀ p : ℝ, 0 ≤ p → p / 100 = p) 
  (h_total : total_students = 250) 
  (h_chess : chess_percentage = 10) 
  (h_chess_basketball : students_like_chess_basketball = 125) :
  ∃ (basketball_percentage : ℝ), basketball_percentage = 40 := by
  sorry

end percentage_basketball_l21_21906


namespace distance_between_parallel_lines_l21_21738

theorem distance_between_parallel_lines (A B C1 C2 : ℝ) (hA : A = 2) (hB : B = 4)
  (hC1 : C1 = -8) (hC2 : C2 = 7) : 
  (|C2 - C1| / (Real.sqrt (A^2 + B^2)) = 3 * Real.sqrt 5 / 2) :=
by
  rw [hA, hB, hC1, hC2]
  sorry

end distance_between_parallel_lines_l21_21738


namespace solve_for_x_l21_21590

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end solve_for_x_l21_21590


namespace log_relation_l21_21187

theorem log_relation (a b c: ℝ) (h₁: a = (Real.log 2) / 2) (h₂: b = (Real.log 3) / 3) (h₃: c = (Real.log 5) / 5) : c < a ∧ a < b :=
by
  sorry

end log_relation_l21_21187


namespace original_quantity_of_ghee_l21_21752

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end original_quantity_of_ghee_l21_21752


namespace min_value_of_sum_of_squares_l21_21188

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 10) : 
  x^2 + y^2 + z^2 ≥ 100 / 29 :=
sorry

end min_value_of_sum_of_squares_l21_21188


namespace unique_and_double_solutions_l21_21561

theorem unique_and_double_solutions (a : ℝ) :
  (∃ (x : ℝ), 5 + |x - 2| = a ∧ ∀ y, 5 + |y - 2| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 7 - |2*x1 + 6| = a ∧ 7 - |2*x2 + 6| = a)) ∨
  (∃ (x : ℝ), 7 - |2*x + 6| = a ∧ ∀ y, 7 - |2*y + 6| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 5 + |x1 - 2| = a ∧ 5 + |x2 - 2| = a)) ↔ a = 5 ∨ a = 7 :=
by
  sorry

end unique_and_double_solutions_l21_21561


namespace identity_eq_l21_21514

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l21_21514


namespace evaluate_expression_l21_21583

theorem evaluate_expression :
  - (20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 :=
by
  sorry

end evaluate_expression_l21_21583


namespace abs_five_minus_two_e_l21_21582

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_two_e : |5 - 2 * e| = 0.436 := by
  sorry

end abs_five_minus_two_e_l21_21582


namespace triangle_BC_length_l21_21913

open_locale real

/-- In a triangle ABC with sides AB = 2 and AC = 3, where the median from A to BC has the same length
    as BC and the perimeter of the triangle is 10, the length of BC is 5. --/
theorem triangle_BC_length (BC : ℝ) (AB : ℝ) (AC : ℝ) (median_condition : ℝ) 
  (perimeter_condition : ℝ) (hAB : AB = 2) (hAC : AC = 3) 
  (hmedian : median_condition = BC) (hperimeter : AB + AC + BC = 10) : 
  BC = 5 := 
by
  sorry

end triangle_BC_length_l21_21913


namespace tan_2A_cos_pi3_minus_A_l21_21762

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

end tan_2A_cos_pi3_minus_A_l21_21762


namespace count_even_digits_in_base_5_of_567_l21_21439

def is_even (n : ℕ) : Bool := n % 2 = 0

def base_5_representation (n : ℕ) : List ℕ :=
  if h : n > 0 then
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc else loop (n / 5) ((n % 5) :: acc)
    loop n []
  else [0]

def count_even_digits_in_base_5 (n : ℕ) : ℕ :=
  (base_5_representation n).filter is_even |>.length

theorem count_even_digits_in_base_5_of_567 :
  count_even_digits_in_base_5 567 = 2 := by
  sorry

end count_even_digits_in_base_5_of_567_l21_21439


namespace duty_arrangements_l21_21842

theorem duty_arrangements (science_teachers : Finset ℕ) (liberal_arts_teachers : Finset ℕ) :
  science_teachers.card = 6 →
  liberal_arts_teachers.card = 2 →
  (∃ arrangements : Finset (Finset ℕ × Finset ℕ × Finset ℕ),
    arrangements.card = 540) :=
by
  intros h_science h_liberal
  sorry

end duty_arrangements_l21_21842


namespace difference_in_ages_l21_21653

variables (J B : ℕ)

-- The conditions: Jack's age is twice Bill's age, and in eight years, Jack will be three times Bill's age then.
axiom condition1 : J = 2 * B
axiom condition2 : J + 8 = 3 * (B + 8)

-- The theorem statement we are proving: The difference in their current ages is 16.
theorem difference_in_ages : J - B = 16 :=
by
  sorry

end difference_in_ages_l21_21653


namespace solve_fisherman_problem_l21_21645

def fisherman_problem : Prop :=
  ∃ (x y z : ℕ), x + y + z = 16 ∧ 13 * x + 5 * y + 4 * z = 113 ∧ x = 5 ∧ y = 4 ∧ z = 7

theorem solve_fisherman_problem : fisherman_problem :=
sorry

end solve_fisherman_problem_l21_21645


namespace alice_bob_probability_l21_21657

noncomputable def probability_of_exactly_two_sunny_days : ℚ :=
  let p_sunny := 3 / 5
  let p_rain := 2 / 5
  3 * (p_sunny^2 * p_rain)

theorem alice_bob_probability :
  probability_of_exactly_two_sunny_days = 54 / 125 := 
sorry

end alice_bob_probability_l21_21657


namespace output_of_code_snippet_is_six_l21_21544

-- Define the variables and the condition
def a : ℕ := 3
def y : ℕ := if a < 10 then 2 * a else a * a 

-- The statement to be proved
theorem output_of_code_snippet_is_six :
  y = 6 :=
by
  sorry

end output_of_code_snippet_is_six_l21_21544


namespace jen_total_birds_l21_21209

-- Define the initial conditions
def total_birds (c : ℕ) : ℕ :=
  let d := 10 + 4 * c in
  if d = 150 then c + d else 0

theorem jen_total_birds : total_birds 35 = 185 :=
  sorry

end jen_total_birds_l21_21209


namespace angle_measure_l21_21790

theorem angle_measure (x : ℝ) 
  (h1 : 5 * x + 12 = 180 - x) : x = 28 := by
  sorry

end angle_measure_l21_21790


namespace min_sum_of_factors_l21_21963

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21963


namespace smallest_composite_no_prime_factors_lt_20_l21_21867

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n : ℕ, (n > 1 ∧ ¬ Prime n ∧ (∀ p : ℕ, Prime p → p < 20 → p ∣ n → False)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_lt_20_l21_21867


namespace expand_polynomial_l21_21873

theorem expand_polynomial :
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 :=
by
  sorry

end expand_polynomial_l21_21873


namespace min_sum_of_factors_l21_21959

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21959


namespace no_contradiction_to_thermodynamics_l21_21728

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l21_21728


namespace find_matrix_find_eigenvalue_transform_line_l21_21298

noncomputable section

open Matrix

def M : Matrix (Fin 2) (Fin 2) ℝ := ![![6, 2], ![4, 4]]

def eigenvalue1 : ℝ := 8
def eigenvector1 : Fin 2 → ℝ := ![1, 1]

def point1 : Fin 2 → ℝ := ![-1, 2]
def point2 : Fin 2 → ℝ := ![-2, 4]

def line1 (x y : ℝ) : Prop := 2 * x - 4 * y + 1 = 0
def line2 (x' y' : ℝ) : Prop := x' - y' + 2 = 0

theorem find_matrix :
  (M ⬝ (col_vector eigenvector1) = eigenvalue1 • (col_vector eigenvector1)) ∧
  (M ⬝ (col_vector point1) = col_vector point2) →
  M = ![![6, 2], ![4, 4]] :=
sorry

theorem find_eigenvalue (λ : ℝ) (eigenvector2 : Fin 2 → ℝ) :
  (M ⬝ (col_vector eigenvector2) = λ • (col_vector eigenvector2)) →
  λ = 2 ∧ (2 * eigenvector2 0 + eigenvector2 1 = 0) :=
sorry

theorem transform_line (x y x' y' : ℝ) :
  line1 x y →
  (M ⬝ ![x, y] = ![x', y']) →
  line2 x' y' :=
sorry

end find_matrix_find_eigenvalue_transform_line_l21_21298


namespace multiplier_is_3_l21_21652

theorem multiplier_is_3 (x : ℝ) (num : ℝ) (difference : ℝ) (h1 : num = 15.0) (h2 : difference = 40) (h3 : x * num - 5 = difference) : x = 3 := 
by 
  sorry

end multiplier_is_3_l21_21652


namespace odd_function_increasing_function_l21_21491

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 0.5

theorem odd_function (x : ℝ) : f (-x) = -f (x) :=
  by sorry

theorem increasing_function : ∀ x y : ℝ, x < y → f x < f y :=
  by sorry

end odd_function_increasing_function_l21_21491


namespace time_for_one_paragraph_l21_21062

-- Definitions for the given conditions
def short_answer_time := 3 -- minutes
def essay_time := 60 -- minutes
def total_homework_time := 240 -- minutes
def essays_assigned := 2
def paragraphs_assigned := 5
def short_answers_assigned := 15

-- Function to calculate total time from given conditions
def total_time_for_essays (essays : ℕ) : ℕ :=
  essays * essay_time

def total_time_for_short_answers (short_answers : ℕ) : ℕ :=
  short_answers * short_answer_time

def total_time_for_paragraphs (paragraphs : ℕ) : ℕ :=
  total_homework_time - (total_time_for_essays essays_assigned + total_time_for_short_answers short_answers_assigned)

def time_per_paragraph (paragraphs : ℕ) : ℕ :=
  total_time_for_paragraphs paragraphs / paragraphs_assigned

-- Proving the question part
theorem time_for_one_paragraph : 
  time_per_paragraph paragraphs_assigned = 15 := by
  sorry

end time_for_one_paragraph_l21_21062


namespace fraction_to_decimal_l21_21407

theorem fraction_to_decimal : (5 / 8 : ℝ) = 0.625 := 
  by sorry

end fraction_to_decimal_l21_21407


namespace S_gt_inverse_1988_cubed_l21_21386

theorem S_gt_inverse_1988_cubed (a b c d : ℕ) (hb: 0 < b) (hd: 0 < d) 
  (h1: a + c < 1988) (h2: 1 - (a / b) - (c / d) > 0) : 
  1 - (a / b) - (c / d) > 1 / (1988^3) := 
sorry

end S_gt_inverse_1988_cubed_l21_21386


namespace volume_of_prism_l21_21543

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 20) (h3 : c * a = 12) (h4 : a + b + c = 11) :
  a * b * c = 12 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l21_21543


namespace total_hours_watched_l21_21413

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end total_hours_watched_l21_21413


namespace probability_green_dinosaur_or_blue_robot_l21_21665

theorem probability_green_dinosaur_or_blue_robot (t: ℕ) (blue_dinosaurs green_robots blue_robots: ℕ) 
(h1: blue_dinosaurs = 16) (h2: green_robots = 14) (h3: blue_robots = 36) (h4: t = 93):
  t = 93 → (blue_dinosaurs = 16) → (green_robots = 14) → (blue_robots = 36) → 
  (∃ green_dinosaurs: ℕ, t = blue_dinosaurs + green_robots + blue_robots + green_dinosaurs ∧ 
    (∃ k: ℕ, k = (green_dinosaurs + blue_robots) / (t / 31) ∧ k = 21 / 31)) := sorry

end probability_green_dinosaur_or_blue_robot_l21_21665


namespace roots_in_interval_l21_21731

def P (x : ℝ) : ℝ := x^2014 - 100 * x + 1

theorem roots_in_interval : 
  ∀ x : ℝ, P x = 0 → (1/100) ≤ x ∧ x ≤ 100^(1 / 2013) := 
  sorry

end roots_in_interval_l21_21731


namespace total_sales_first_three_days_total_earnings_seven_days_l21_21406

def planned_daily_sales : Int := 100

def deviation : List Int := [4, -3, -5, 14, -8, 21, -6]

def selling_price_per_pound : Int := 8
def freight_cost_per_pound : Int := 3

-- Part (1): Proof statement for the total amount sold in the first three days
theorem total_sales_first_three_days :
  let monday_sales := planned_daily_sales + deviation.head!
  let tuesday_sales := planned_daily_sales + (deviation.drop 1).head!
  let wednesday_sales := planned_daily_sales + (deviation.drop 2).head!
  monday_sales + tuesday_sales + wednesday_sales = 296 := by
  sorry

-- Part (2): Proof statement for Xiaoming's total earnings for the seven days
theorem total_earnings_seven_days :
  let total_sales := (List.sum (deviation.map (λ x => planned_daily_sales + x)))
  total_sales * (selling_price_per_pound - freight_cost_per_pound) = 3585 := by
  sorry

end total_sales_first_three_days_total_earnings_seven_days_l21_21406


namespace base8_addition_l21_21015

theorem base8_addition (X Y : ℕ) 
  (h1 : 5 * 8 + X + Y + 3 * 8 + 2 = 6 * 64 + 4 * 8 + X) :
  X + Y = 16 := by
  sorry

end base8_addition_l21_21015


namespace correct_statement_l21_21879

noncomputable def f (x : ℝ) := Real.exp x - x
noncomputable def g (x : ℝ) := Real.log x + x + 1

def proposition_p := ∀ x : ℝ, f x > 0
def proposition_q := ∃ x0 : ℝ, 0 < x0 ∧ g x0 = 0

theorem correct_statement : (proposition_p ∧ proposition_q) :=
by
  sorry

end correct_statement_l21_21879


namespace jack_sugar_amount_l21_21483

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l21_21483


namespace relationship_of_variables_l21_21024

variable {a b c d : ℝ}

theorem relationship_of_variables 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := 
sorry

end relationship_of_variables_l21_21024


namespace solve_inequality_l21_21440

theorem solve_inequality (x : ℝ) (h : |2 * x + 6| < 10) : -8 < x ∧ x < 2 :=
sorry

end solve_inequality_l21_21440


namespace sector_area_correct_l21_21888

noncomputable def sector_area (r α : ℝ) : ℝ :=
  (1 / 2) * r^2 * α

theorem sector_area_correct :
  sector_area 3 2 = 9 :=
by
  sorry

end sector_area_correct_l21_21888


namespace compute_sqrt_factorial_square_l21_21825

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l21_21825


namespace minimum_value_ineq_l21_21633

open Real

theorem minimum_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
    (x + 1 / (y * y)) * (x + 1 / (y * y) - 500) + (y + 1 / (x * x)) * (y + 1 / (x * x) - 500) ≥ -125000 :=
by 
  sorry

end minimum_value_ineq_l21_21633


namespace event_probability_l21_21419

noncomputable def probability_event : ℝ :=
  let a : ℝ := (1 : ℝ) / 2
  let b : ℝ := (3 : ℝ) / 2
  let interval_length : ℝ := 2
  (b - a) / interval_length

theorem event_probability :
  probability_event = (3 : ℝ) / 4 :=
by
  -- Proof step will be supplied here
  sorry

end event_probability_l21_21419


namespace difference_of_sums_l21_21670

def even_numbers_sum (n : ℕ) : ℕ := (n * (n + 1))
def odd_numbers_sum (n : ℕ) : ℕ := n^2

theorem difference_of_sums : 
  even_numbers_sum 3003 - odd_numbers_sum 3003 = 7999 := 
by {
  sorry 
}

end difference_of_sums_l21_21670


namespace keith_total_cost_correct_l21_21489

noncomputable def total_cost_keith_purchases : Real :=
  let discount_toy := 6.51
  let price_toy := discount_toy / 0.90
  let pet_food := 5.79
  let cage_price := 12.51
  let tax_rate := 0.08
  let cage_tax := cage_price * tax_rate
  let price_with_tax := cage_price + cage_tax
  let water_bottle := 4.99
  let bedding := 7.65
  let discovered_money := 1.0
  let total_cost := discount_toy + pet_food + price_with_tax + water_bottle + bedding
  total_cost - discovered_money

theorem keith_total_cost_correct :
  total_cost_keith_purchases = 37.454 :=
by
  sorry -- Proof of the theorem will go here

end keith_total_cost_correct_l21_21489


namespace arithmetic_expression_l21_21470

theorem arithmetic_expression : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_expression_l21_21470


namespace gel_pen_price_relation_b_l21_21133

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l21_21133


namespace number_of_correct_answers_l21_21550

theorem number_of_correct_answers (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 110) : c = 34 :=
by
  -- placeholder for proof
  sorry

end number_of_correct_answers_l21_21550


namespace sqrt_factorial_product_squared_l21_21816

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l21_21816


namespace BECD_is_rhombus_l21_21912

-- Definitions of points and trapezoid
variables {α : Type*} [linear_ordered_field α]

noncomputable def is_trapezoid (A B C D : α → α → Type*) : Prop :=
∃ (AB BC CD DA : line α), (_on_line A B AB) ∧ (on_line B C BC) ∧ (on_line C D CD) ∧ (on_line D A DA) ∧ parallel AB CD

noncomputable def is_equal_length (P Q R S : α → α → Type*) : Prop :=
distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P

-- Given conditions
variables {A B C D E O : α → α → Type*}

-- Definitions for trapezoid and conditions
def trapezoid_ABC_exists_side_equal := is_trapezoid A B C D ∧ is_equal_length A B C D
def diagonals_intersect (O : α → α → Type*) := ∃ (diag_1 diag_2 : line α), (on_line A C diag_1) ∧ (on_line B D diag_2) ∧ intersect diag_1 diag_2 O

def circumcircle_intersects_base (A B O E : α → α → Type*) : Prop := 
∃ (circ : circle α), on_circle A circ ∧ on_circle B circ ∧ on_circle O circ ∧ on_circle E circ

-- Theorem statement
theorem BECD_is_rhombus
  (h1 : trapezoid_ABC_exists_side_equal)
  (h2 : diagonals_intersect O)
  (h3 : circumcircle_intersects_base A B O E) :
  is_rhombus B E C D :=
sorry

end BECD_is_rhombus_l21_21912


namespace remainder_is_one_l21_21986

theorem remainder_is_one (N : ℤ) (R : ℤ)
  (h1 : N % 100 = R)
  (h2 : N % R = 1) :
  R = 1 :=
by
  sorry

end remainder_is_one_l21_21986


namespace gel_pen_price_ratio_l21_21159

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l21_21159


namespace luke_clothing_distribution_l21_21636

theorem luke_clothing_distribution (total_clothing: ℕ) (first_load: ℕ) (num_loads: ℕ) 
  (remaining_clothing : total_clothing - first_load = 30)
  (equal_load_per_small_load: (total_clothing - first_load) / num_loads = 6) : 
  total_clothing = 47 ∧ first_load = 17 ∧ num_loads = 5 :=
by
  have h1 : total_clothing - first_load = 30 := remaining_clothing
  have h2 : (total_clothing - first_load) / num_loads = 6 := equal_load_per_small_load
  sorry

end luke_clothing_distribution_l21_21636


namespace people_in_room_proof_l21_21252

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end people_in_room_proof_l21_21252


namespace part1_part2_i_part2_ii_l21_21908

theorem part1 :
  ¬ ∃ x : ℝ, - (4 / x) = x := 
sorry

theorem part2_i (a c : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, x = a * (x^2) + 6 * x + c ∧ x = 5 / 2) ↔ (a = -1 ∧ c = -25 / 4) :=
sorry

theorem part2_ii (m : ℝ) :
  (∃ (a c : ℝ), a = -1 ∧ c = - 25 / 4 ∧
    ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → - (x^2) + 6 * x - 25 / 4 + 1/4 ≥ -1 ∧ - (x^2) + 6 * x - 25 / 4 + 1/4 ≤ 3) ↔
    (3 ≤ m ∧ m ≤ 5) :=
sorry

end part1_part2_i_part2_ii_l21_21908


namespace range_of_a_not_empty_solution_set_l21_21043

theorem range_of_a_not_empty_solution_set :
  {a : ℝ | ∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0} =
  {a : ℝ | a ∈ {a : ℝ | a < -2} ∪ {a : ℝ | a ≥ 6 / 5}} :=
sorry

end range_of_a_not_empty_solution_set_l21_21043


namespace prime_pairs_solution_l21_21016

def is_prime (n : ℕ) : Prop := Nat.Prime n

def conditions (p q : ℕ) : Prop := 
  p^2 ∣ q^3 + 1 ∧ q^2 ∣ p^6 - 1

theorem prime_pairs_solution :
  ({(p, q) | is_prime p ∧ is_prime q ∧ conditions p q} = {(3, 2), (2, 3)}) :=
by
  sorry

end prime_pairs_solution_l21_21016


namespace total_amount_spent_is_300_l21_21833

-- Definitions of conditions
def S : ℕ := 97
def H : ℕ := 2 * S + 9

-- The total amount spent
def total_spent : ℕ := S + H

-- Proof statement
theorem total_amount_spent_is_300 : total_spent = 300 :=
by
  sorry

end total_amount_spent_is_300_l21_21833


namespace cost_per_pound_beef_is_correct_l21_21348

variable (budget initial_chicken_cost pounds_beef remaining_budget_after_purchase : ℝ)
variable (spending_on_beef cost_per_pound_beef : ℝ)

axiom h1 : budget = 80
axiom h2 : initial_chicken_cost = 12
axiom h3 : pounds_beef = 5
axiom h4 : remaining_budget_after_purchase = 53
axiom h5 : spending_on_beef = budget - initial_chicken_cost - remaining_budget_after_purchase
axiom h6 : cost_per_pound_beef = spending_on_beef / pounds_beef

theorem cost_per_pound_beef_is_correct : cost_per_pound_beef = 3 :=
by
  sorry

end cost_per_pound_beef_is_correct_l21_21348


namespace correct_transformation_l21_21832

theorem correct_transformation (a b m : ℝ) (h : m ≠ 0) : (am / bm) = (a / b) :=
by sorry

end correct_transformation_l21_21832


namespace sqrt_factorial_sq_l21_21804

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l21_21804


namespace division_remainder_l21_21169

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end division_remainder_l21_21169


namespace suzie_store_revenue_l21_21647

theorem suzie_store_revenue 
  (S B : ℝ) 
  (h1 : B = S + 15) 
  (h2 : 22 * S + 16 * B = 460) : 
  8 * S + 32 * B = 711.60 :=
by
  sorry

end suzie_store_revenue_l21_21647


namespace largest_integer_satisfying_condition_l21_21492

-- Definition of the conditions
def has_four_digits_in_base_10 (n : ℕ) : Prop :=
  10^3 ≤ n^2 ∧ n^2 < 10^4

-- Proof statement: N is the largest integer satisfying the condition
theorem largest_integer_satisfying_condition : ∃ (N : ℕ), 
  has_four_digits_in_base_10 N ∧ (∀ (m : ℕ), has_four_digits_in_base_10 m → m ≤ N) ∧ N = 99 := 
sorry

end largest_integer_satisfying_condition_l21_21492


namespace bubble_sort_prob_l21_21367

theorem bubble_sort_prob :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (↑p / ↑q : ℚ) = 1 / 132 ∧ (p + q = 133) :=
by
  sorry

end bubble_sort_prob_l21_21367


namespace initial_balance_before_check_deposit_l21_21408

theorem initial_balance_before_check_deposit (new_balance : ℝ) (initial_balance : ℝ) : 
  (50 = 1 / 4 * new_balance) → (initial_balance = new_balance - 50) → initial_balance = 150 :=
by
  sorry

end initial_balance_before_check_deposit_l21_21408


namespace find_y_l21_21831

theorem find_y : ∀ (x y : ℤ), x > 0 ∧ y > 0 ∧ x % y = 9 ∧ (x:ℝ) / (y:ℝ) = 96.15 → y = 60 :=
by
  intros x y h
  sorry

end find_y_l21_21831


namespace tailwind_speed_l21_21560

-- Define the given conditions
def plane_speed_with_wind (P W : ℝ) : Prop := P + W = 460
def plane_speed_against_wind (P W : ℝ) : Prop := P - W = 310

-- Theorem stating the proof problem
theorem tailwind_speed (P W : ℝ) 
  (h1 : plane_speed_with_wind P W) 
  (h2 : plane_speed_against_wind P W) : 
  W = 75 :=
sorry

end tailwind_speed_l21_21560


namespace prove_a_eq_b_l21_21918

theorem prove_a_eq_b 
    (a b : ℕ) 
    (h_pos : a > 0 ∧ b > 0) 
    (h_multiple : ∃ k : ℤ, a^2 + a * b + 1 = k * (b^2 + b * a + 1)) : 
    a = b := 
sorry

end prove_a_eq_b_l21_21918


namespace stadium_ticket_price_l21_21247

theorem stadium_ticket_price
  (original_price : ℝ)
  (decrease_rate : ℝ)
  (increase_rate : ℝ)
  (new_price : ℝ) 
  (h1 : original_price = 400)
  (h2 : decrease_rate = 0.2)
  (h3 : increase_rate = 0.05) 
  (h4 : (original_price * (1 + increase_rate) / (1 - decrease_rate)) = new_price) :
  new_price = 525 := 
by
  -- Proof omitted for this task.
  sorry

end stadium_ticket_price_l21_21247


namespace diagonal_BD_size_cos_A_value_l21_21329

noncomputable def AB := 250
noncomputable def CD := 250
noncomputable def angle_A := 120
noncomputable def angle_C := 120
noncomputable def AD := 150
noncomputable def BC := 150
noncomputable def perimeter := 800

/-- The size of the diagonal BD in isosceles trapezoid ABCD is 350, given the conditions -/
theorem diagonal_BD_size (AB CD AD BC : ℕ) (angle_A angle_C : ℝ) :
  AB = 250 → CD = 250 → AD = 150 → BC = 150 →
  angle_A = 120 → angle_C = 120 →
  ∃ BD : ℝ, BD = 350 :=
by
  sorry

/-- The cosine of angle A is -0.5, given the angle is 120 degrees -/
theorem cos_A_value (angle_A : ℝ) :
  angle_A = 120 → ∃ cos_A : ℝ, cos_A = -0.5 :=
by
  sorry

end diagonal_BD_size_cos_A_value_l21_21329


namespace min_value_reciprocal_sum_l21_21601

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

end min_value_reciprocal_sum_l21_21601


namespace x_squared_minus_y_squared_l21_21036

theorem x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 2 / 5) (h2 : x - y = 1 / 10) : x ^ 2 - y ^ 2 = 1 / 25 :=
by
  sorry

end x_squared_minus_y_squared_l21_21036


namespace simplify_and_evaluate_l21_21364

noncomputable def a := 2 * Real.sin (Real.pi / 4) + (1 / 2) ^ (-1 : ℤ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l21_21364


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21136

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21136


namespace standard_equation_of_ellipse_l21_21836

-- Define the conditions of the ellipse
def ellipse_condition_A (m n : ℝ) : Prop := n * (5 / 3) ^ 2 = 1
def ellipse_condition_B (m n : ℝ) : Prop := m + n = 1

-- The theorem to prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (m n : ℝ) (hA : ellipse_condition_A m n) (hB : ellipse_condition_B m n) :
  m = 16 / 25 ∧ n = 9 / 25 :=
sorry

end standard_equation_of_ellipse_l21_21836


namespace drainage_capacity_per_day_l21_21667

theorem drainage_capacity_per_day
  (capacity : ℝ)
  (rain_1 : ℝ)
  (rain_2 : ℝ)
  (rain_3 : ℝ)
  (rain_4_min : ℝ)
  (total_days : ℕ) 
  (days_to_drain : ℕ)
  (feet_to_inches : ℝ := 12)
  (required_rain_capacity : ℝ) 
  (drain_capacity_per_day : ℝ)

  (h1: capacity = 6 * feet_to_inches)
  (h2: rain_1 = 10)
  (h3: rain_2 = 2 * rain_1)
  (h4: rain_3 = 1.5 * rain_2)
  (h5: rain_4_min = 21)
  (h6: total_days = 4)
  (h7: days_to_drain = 3)
  (h8: required_rain_capacity = capacity - (rain_1 + rain_2 + rain_3))

  : drain_capacity_per_day = (rain_1 + rain_2 + rain_3 - required_rain_capacity + rain_4_min) / days_to_drain :=
sorry

end drainage_capacity_per_day_l21_21667


namespace students_answered_both_correctly_l21_21471

theorem students_answered_both_correctly :
  ∀ (total_students set_problem function_problem both_incorrect x : ℕ),
    total_students = 50 → 
    set_problem = 40 →
    function_problem = 31 →
    both_incorrect = 4 →
    x = total_students - both_incorrect - (set_problem + function_problem - total_students) →
    x = 25 :=
by
  intros total_students set_problem function_problem both_incorrect x
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end students_answered_both_correctly_l21_21471


namespace sqrt_factorial_product_squared_l21_21815

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l21_21815


namespace star_shell_arrangements_l21_21340

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Conditions
def outward_points : ℕ := 6
def inward_points : ℕ := 6
def total_points : ℕ := outward_points + inward_points
def unique_shells : ℕ := 12

-- The problem statement translated into Lean 4:
theorem star_shell_arrangements : (factorial unique_shells / 12 = 39916800) :=
by
  sorry

end star_shell_arrangements_l21_21340


namespace min_sum_abs_l21_21658

theorem min_sum_abs (x : ℝ) : ∃ m, m = 4 ∧ ∀ x : ℝ, |x + 2| + |x - 2| + |x - 1| ≥ m := 
sorry

end min_sum_abs_l21_21658


namespace building_time_l21_21119

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end building_time_l21_21119


namespace abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l21_21280

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 :=
sorry

theorem neg_one_pow_2023_eq_neg_one : (-1 : ℤ) ^ 2023 = -1 :=
sorry

end abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l21_21280


namespace number_of_diagonals_l21_21615

-- Define the regular pentagonal prism and its properties
def regular_pentagonal_prism : Type := sorry

-- Define what constitutes a diagonal in this context
def is_diagonal (p : regular_pentagonal_prism) (v1 v2 : Nat) : Prop :=
  sorry -- We need to detail what counts as a diagonal based on the conditions

-- Hypothesis on the structure specifying that there are 5 vertices on the top and 5 on the bottom
axiom vertices_on_top_and_bottom (p : regular_pentagonal_prism) : sorry -- We need the precise formalization

-- The main theorem
theorem number_of_diagonals (p : regular_pentagonal_prism) : ∃ n, n = 10 :=
  sorry

end number_of_diagonals_l21_21615


namespace value_of_a_l21_21597

theorem value_of_a (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {a, a^2}) (hB : B = {1, b}) (hAB : A = B) : a = -1 := 
by 
  sorry

end value_of_a_l21_21597


namespace probability_two_red_one_blue_l21_21838

theorem probability_two_red_one_blue
  (total_reds : ℕ) (total_blues : ℕ) (total_draws : ℕ)
  (h_reds : total_reds = 12) (h_blues : total_blues = 8) (h_draws : total_draws = 3) :
  let total_marbles := total_reds + total_blues in
  (total_reds * (total_reds - 1) * total_blues) / (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 44 / 95 := sorry

end probability_two_red_one_blue_l21_21838


namespace dessert_menu_count_l21_21688

def desserts := ["cake", "pie", "ice cream", "pudding"]

def menu_possible (desserts : List String) (days : ℕ) : ℕ := sorry

theorem dessert_menu_count : menu_possible desserts 7 = 972 := sorry

end dessert_menu_count_l21_21688


namespace find_weight_A_l21_21552

noncomputable def weight_of_A (a b c d e : ℕ) : Prop :=
  (a + b + c) / 3 = 84 ∧
  (a + b + c + d) / 4 = 80 ∧
  e = d + 5 ∧
  (b + c + d + e) / 4 = 79 →
  a = 77

theorem find_weight_A (a b c d e : ℕ) : weight_of_A a b c d e :=
by
  sorry

end find_weight_A_l21_21552


namespace pet_store_cats_left_l21_21420

theorem pet_store_cats_left :
  let initial_siamese := 13.5
  let initial_house := 5.25
  let added_cats := 10.75
  let discount := 0.5
  let initial_total := initial_siamese + initial_house
  let new_total := initial_total + added_cats
  let final_total := new_total - discount
  final_total = 29 :=
by sorry

end pet_store_cats_left_l21_21420


namespace factorial_expression_value_l21_21812

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l21_21812


namespace total_cost_eq_898_80_l21_21242

theorem total_cost_eq_898_80 (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 21) :
  4 * M + 3 * R + 5 * F = 898.80 :=
by
  sorry

end total_cost_eq_898_80_l21_21242


namespace compare_polynomials_l21_21566

theorem compare_polynomials (x : ℝ) : 2 * x^2 - 2 * x + 1 > x^2 - 2 * x := 
by
  sorry

end compare_polynomials_l21_21566


namespace coins_division_remainder_l21_21270

theorem coins_division_remainder :
  ∃ n : ℕ, (n % 8 = 6 ∧ n % 7 = 5 ∧ n % 9 = 0) :=
sorry

end coins_division_remainder_l21_21270


namespace sum_equals_120_l21_21174

def rectangular_parallelepiped := (3, 4, 5)

def face_dimensions : List (ℕ × ℕ) := [(4, 5), (3, 5), (3, 4)]

def number_assignment (d : ℕ × ℕ) : ℕ :=
  if d = (4, 5) then 9
  else if d = (3, 5) then 8
  else if d = (3, 4) then 5
  else 0

def sum_checkerboard_ring_one_width (rect_dims : ℕ × ℕ × ℕ) (number_assignment : ℕ × ℕ → ℕ) : ℕ :=
  let (x, y, z) := rect_dims
  let l1 := number_assignment (4, 5) * 2 * (4 * 5)
  let l2 := number_assignment (3, 5) * 2 * (3 * 5)
  let l3 := number_assignment (3, 4) * 2 * (3 * 4) 
  l1 + l2 + l3

theorem sum_equals_120 : ∀ rect_dims number_assignment,
  rect_dims = rectangular_parallelepiped → sum_checkerboard_ring_one_width rect_dims number_assignment = 720 := sorry

end sum_equals_120_l21_21174


namespace boys_and_girls_solution_l21_21418

theorem boys_and_girls_solution (x y : ℕ) 
  (h1 : 3 * x + y > 24) 
  (h2 : 7 * x + 3 * y < 60) : x = 8 ∧ y = 1 :=
by
  sorry

end boys_and_girls_solution_l21_21418


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21139

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21139


namespace pills_in_a_week_l21_21311

def insulin_pills_per_day : Nat := 2
def blood_pressure_pills_per_day : Nat := 3
def anticonvulsant_pills_per_day : Nat := 2 * blood_pressure_pills_per_day

def total_pills_per_day : Nat := insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsant_pills_per_day

theorem pills_in_a_week : total_pills_per_day * 7 = 77 := by
  sorry

end pills_in_a_week_l21_21311


namespace LCM_of_two_numbers_l21_21106

theorem LCM_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 14) (h2 : a * b = 2562) : Nat.lcm a b = 183 :=
by
  sorry

end LCM_of_two_numbers_l21_21106


namespace incorrect_statement_C_l21_21073

theorem incorrect_statement_C :
  (∀ (b h : ℝ), b > 0 → h > 0 → 2 * (b * h) = (2 * b) * h) ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → 2 * (π * r^2 * h) = π * r^2 * (2 * h)) ∧
  (∀ (a : ℝ), a > 0 → 4 * (a^3) ≠ (2 * a)^3) ∧
  (∀ (a b : ℚ), b ≠ 0 → a / (2 * b) ≠ (a / 2) / b) ∧
  (∀ (x : ℝ), x < 0 → 2 * x < x) :=
by
  sorry

end incorrect_statement_C_l21_21073


namespace number_of_ways_to_form_team_l21_21565

-- Defining the conditions
def total_employees : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def team_size : ℕ := 6
def men_in_team : ℕ := 4
def women_in_team : ℕ := 2

-- Using binomial coefficient to represent combinations
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proved
theorem number_of_ways_to_form_team :
  (choose num_men men_in_team) * (choose num_women women_in_team) = 
  choose 10 4 * choose 5 2 :=
by
  sorry

end number_of_ways_to_form_team_l21_21565


namespace average_score_of_remaining_students_correct_l21_21466

noncomputable def average_score_remaining_students (n : ℕ) (h_n : n > 15) (avg_all : ℚ) (avg_subgroup : ℚ) : ℚ :=
if h_avg_all : avg_all = 10 ∧ avg_subgroup = 16 then
  (10 * n - 240) / (n - 15)
else
  0

theorem average_score_of_remaining_students_correct (n : ℕ) (h_n : n > 15) :
  (average_score_remaining_students n h_n 10 16) = (10 * n - 240) / (n - 15) :=
by
  dsimp [average_score_remaining_students]
  split_ifs with h_avg
  · sorry
  · sorry

end average_score_of_remaining_students_correct_l21_21466


namespace valid_words_count_l21_21655

noncomputable def count_valid_words : Nat :=
  let total_possible_words : Nat := ((25^1) + (25^2) + (25^3) + (25^4) + (25^5))
  let total_possible_words_without_B : Nat := ((24^1) + (24^2) + (24^3) + (24^4) + (24^5))
  total_possible_words - total_possible_words_without_B

theorem valid_words_count : count_valid_words = 1864701 :=
by
  let total_1_letter_words := 25^1
  let total_2_letter_words := 25^2
  let total_3_letter_words := 25^3
  let total_4_letter_words := 25^4
  let total_5_letter_words := 25^5

  let total_words_without_B_1_letter := 24^1
  let total_words_without_B_2_letter := 24^2
  let total_words_without_B_3_letter := 24^3
  let total_words_without_B_4_letter := 24^4
  let total_words_without_B_5_letter := 24^5

  let valid_1_letter_words := total_1_letter_words - total_words_without_B_1_letter
  let valid_2_letter_words := total_2_letter_words - total_words_without_B_2_letter
  let valid_3_letter_words := total_3_letter_words - total_words_without_B_3_letter
  let valid_4_letter_words := total_4_letter_words - total_words_without_B_4_letter
  let valid_5_letter_words := total_5_letter_words - total_words_without_B_5_letter

  let valid_words := valid_1_letter_words + valid_2_letter_words + valid_3_letter_words + valid_4_letter_words + valid_5_letter_words
  sorry

end valid_words_count_l21_21655


namespace min_sum_of_factors_l21_21965

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21965


namespace largest_very_prime_is_373_l21_21634

def very_prime (n : ℕ) : Prop :=
  ∀ (d : ℕ), (d > 0 ∧ d ≤ Nat.digits 10 n).count (λ i, Nat.Prime (Nat.digit 10 n (i - 1))) > 0

theorem largest_very_prime_is_373 : ∀ n : ℕ, (very_prime n) → n ≤ 373 :=
sorry

end largest_very_prime_is_373_l21_21634


namespace jellybean_ratio_l21_21232

theorem jellybean_ratio (gigi_je : ℕ) (rory_je : ℕ) (lorelai_je : ℕ) (h_gigi : gigi_je = 15) (h_rory : rory_je = gigi_je + 30) (h_lorelai : lorelai_je = 180) : lorelai_je / (rory_je + gigi_je) = 3 :=
by
  -- Introduce the given hypotheses
  rw [h_gigi, h_rory, h_lorelai]
  -- Simplify the expression
  sorry

end jellybean_ratio_l21_21232


namespace original_quantity_of_ghee_l21_21751

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end original_quantity_of_ghee_l21_21751


namespace min_value_expression_l21_21220

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ( (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ) ≥ 7 :=
sorry

end min_value_expression_l21_21220


namespace sqrt_factorial_sq_l21_21805

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l21_21805


namespace gel_pen_price_relation_b_l21_21135

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l21_21135


namespace ice_cost_l21_21860

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end ice_cost_l21_21860


namespace bullseye_points_l21_21002

theorem bullseye_points (B : ℝ) (h : B + B / 2 = 75) : B = 50 :=
by
  sorry

end bullseye_points_l21_21002


namespace percentage_of_part_of_whole_l21_21674

theorem percentage_of_part_of_whole :
  let part := 375.2
  let whole := 12546.8
  (part / whole) * 100 = 2.99 :=
by
  sorry

end percentage_of_part_of_whole_l21_21674


namespace factor_1_factor_2_l21_21291

theorem factor_1 {x : ℝ} : x^2 - 4*x + 3 = (x - 1) * (x - 3) :=
sorry

theorem factor_2 {x : ℝ} : 4*x^2 + 12*x - 7 = (2*x + 7) * (2*x - 1) :=
sorry

end factor_1_factor_2_l21_21291


namespace roots_difference_squared_l21_21034

theorem roots_difference_squared
  {Φ ϕ : ℝ}
  (hΦ : Φ^2 - Φ - 2 = 0)
  (hϕ : ϕ^2 - ϕ - 2 = 0)
  (h_diff : Φ ≠ ϕ) :
  (Φ - ϕ)^2 = 9 :=
by sorry

end roots_difference_squared_l21_21034


namespace interest_rate_per_annum_l21_21111

theorem interest_rate_per_annum
  (P : ℕ := 450) 
  (t : ℕ := 8) 
  (I : ℕ := P - 306) 
  (simple_interest : ℕ := P * r * t / 100) :
  r = 4 :=
by
  sorry

end interest_rate_per_annum_l21_21111


namespace integer_solution_abs_lt_sqrt2_l21_21732

theorem integer_solution_abs_lt_sqrt2 (x : ℤ) (h : |x| < Real.sqrt 2) : x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end integer_solution_abs_lt_sqrt2_l21_21732


namespace arc_length_parametric_curve_l21_21979

noncomputable def arcLength (x y : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, Real.sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem arc_length_parametric_curve :
    (∫ t in (0 : ℝ)..(3 * Real.pi), 
        Real.sqrt ((deriv (fun t => (t ^ 2 - 2) * Real.sin t + 2 * t * Real.cos t) t) ^ 2 +
                   (deriv (fun t => (2 - t ^ 2) * Real.cos t + 2 * t * Real.sin t) t) ^ 2)) =
    9 * Real.pi ^ 3 :=
by
  -- The proof is omitted
  sorry

end arc_length_parametric_curve_l21_21979


namespace complete_the_square_l21_21398

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end complete_the_square_l21_21398


namespace volume_ratio_l21_21097

theorem volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) : b^3 / a^3 = 125 / 27 :=
by
  -- Skipping the proof by adding 'sorry'
  sorry

end volume_ratio_l21_21097


namespace prove_b_value_l21_21897

theorem prove_b_value (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end prove_b_value_l21_21897


namespace apples_in_blue_basket_l21_21792

-- Define the number of bananas in the blue basket
def bananas := 12

-- Define the total number of fruits in the blue basket
def totalFruits := 20

-- Define the number of apples as total fruits minus bananas
def apples := totalFruits - bananas

-- Prove that the number of apples in the blue basket is 8
theorem apples_in_blue_basket : apples = 8 := by
  sorry

end apples_in_blue_basket_l21_21792


namespace T_sum_correct_l21_21581

-- Defining the sequence T_n
def T (n : ℕ) : ℤ := 
(-1)^n * 2 * n + (-1)^(n + 1) * n

-- Values to compute
def n1 : ℕ := 27
def n2 : ℕ := 43
def n3 : ℕ := 60

-- Sum of particular values
def T_sum : ℤ := T n1 + T n2 + T n3

-- Placeholder value until actual calculation
def expected_sum : ℤ := -42 -- Replace with the correct calculated result

theorem T_sum_correct : T_sum = expected_sum := sorry

end T_sum_correct_l21_21581


namespace minimum_a_for_cube_in_tetrahedron_l21_21687

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  (Real.sqrt 6 / 12) * a

theorem minimum_a_for_cube_in_tetrahedron (a : ℝ) (r : ℝ) 
  (h_radius : r = radius_of_circumscribed_sphere a)
  (h_diag : Real.sqrt 3 = 2 * r) :
  a = 3 * Real.sqrt 2 :=
by
  sorry

end minimum_a_for_cube_in_tetrahedron_l21_21687


namespace min_a_value_l21_21885

theorem min_a_value 
  (a x y : ℤ) 
  (h1 : x - y^2 = a) 
  (h2 : y - x^2 = a) 
  (h3 : x ≠ y) 
  (h4 : |x| ≤ 10) : 
  a = -111 :=
sorry

end min_a_value_l21_21885


namespace polynomial_coeff_sum_abs_l21_21217

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℤ) 
  (h : (2*x - 1)^5 + (x + 2)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  |a| + |a_2| + |a_4| = 30 :=
sorry

end polynomial_coeff_sum_abs_l21_21217


namespace cages_used_l21_21275

-- Define the initial conditions
def total_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

-- State the theorem to prove the number of cages used
theorem cages_used : (total_puppies - puppies_sold) / puppies_per_cage = 3 := by
  sorry

end cages_used_l21_21275


namespace value_of_40th_expression_l21_21381

-- Define the sequence
def minuend (n : ℕ) : ℕ := 100 - (n - 1)
def subtrahend (n : ℕ) : ℕ := n
def expression_value (n : ℕ) : ℕ := minuend n - subtrahend n

-- Theorem: The value of the 40th expression in the sequence is 21
theorem value_of_40th_expression : expression_value 40 = 21 := by
  show 100 - (40 - 1) - 40 = 21
  sorry

end value_of_40th_expression_l21_21381


namespace race_time_difference_l21_21046

-- Define Malcolm's speed, Joshua's speed, and the distance
def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 7 -- minutes per mile
def race_distance := 15 -- miles

-- Statement of the theorem
theorem race_time_difference :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 15 :=
by sorry

end race_time_difference_l21_21046


namespace smaller_number_l21_21376

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end smaller_number_l21_21376


namespace total_population_l21_21616

variable (b g t : ℕ)

-- Conditions: 
axiom boys_to_girls (h1 : b = 4 * g) : Prop
axiom girls_to_teachers (h2 : g = 8 * t) : Prop

theorem total_population (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * b / 32 :=
sorry

end total_population_l21_21616


namespace coins_in_distinct_colors_l21_21215

theorem coins_in_distinct_colors 
  (n : ℕ)  (h1 : 1 < n) (h2 : n < 2010) : (∃ k : ℕ, 2010 = n * k) ↔ 
  ∀ i : ℕ, i < 2010 → (∃ f : ℕ → ℕ, ∀ j : ℕ, j < n → f (j + i) % n = j % n) :=
sorry

end coins_in_distinct_colors_l21_21215


namespace percentage_of_men_l21_21903

theorem percentage_of_men (M : ℝ) 
  (h1 : 0 < M ∧ M < 1) 
  (h2 : 0.2 * M + 0.4 * (1 - M) = 0.3) : M = 0.5 :=
by
  sorry

end percentage_of_men_l21_21903


namespace find_dividend_and_divisor_l21_21384

theorem find_dividend_and_divisor (quotient : ℕ) (remainder : ℕ) (total : ℕ) (dividend divisor : ℕ) :
  quotient = 13 ∧ remainder = 6 ∧ total = 137 ∧ (dividend + divisor + quotient + remainder = total)
  ∧ dividend = 13 * divisor + remainder → 
  dividend = 110 ∧ divisor = 8 :=
by
  intro h
  sorry

end find_dividend_and_divisor_l21_21384


namespace mr_rainwater_chickens_l21_21225

theorem mr_rainwater_chickens :
  ∃ (Ch : ℕ), (∀ (C G : ℕ), C = 9 ∧ G = 4 * C ∧ G = 2 * Ch → Ch = 18) :=
by
  sorry

end mr_rainwater_chickens_l21_21225


namespace division_remainder_l21_21260

theorem division_remainder :
  ∃ (r : ℝ), ∀ (z : ℝ), (4 * z^3 - 5 * z^2 - 17 * z + 4) = (4 * z + 6) * (z^2 - 4 * z + 1/2) + r ∧ r = 1 :=
sorry

end division_remainder_l21_21260


namespace b_profit_share_l21_21991

theorem b_profit_share (total_capital : ℝ) (profit : ℝ) (A_invest : ℝ) (B_invest : ℝ) (C_invest : ℝ) (D_invest : ℝ)
 (A_time : ℝ) (B_time : ℝ) (C_time : ℝ) (D_time : ℝ) :
  total_capital = 100000 ∧
  A_invest = B_invest + 10000 ∧
  B_invest = C_invest + 5000 ∧
  D_invest = A_invest + 8000 ∧
  A_time = 12 ∧
  B_time = 10 ∧
  C_time = 8 ∧
  D_time = 6 ∧
  profit = 50000 →
  (B_invest * B_time / (A_invest * A_time + B_invest * B_time + C_invest * C_time + D_invest * D_time)) * profit = 10925 :=
by
  sorry

end b_profit_share_l21_21991


namespace sqrt_factorial_squared_l21_21819

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l21_21819


namespace An_odd_iff_even_perfect_square_l21_21877

/-- For any integer n ≥ 2, we define A_n as the number of positive integers m such that the distance 
from n to the nearest non-negative multiple of m is equal to the distance from n^3 to the nearest 
non-negative multiple of m. This statement proves that A_n is odd if and only if n is an even perfect 
square. -/
theorem An_odd_iff_even_perfect_square (n: ℕ) (h: n ≥ 2) : 
    let A_n := ∑ m in Finset.range (n^3 - n + 1), 
                  if ((n % m = n^3 % m) ∨ (n % m = (m - n^3 % m) % m)) then 1 else 0
    in A_n % 2 = 1 ↔ ∃ k, n = 4 * k^2 :=
by sorry

end An_odd_iff_even_perfect_square_l21_21877


namespace age_ratio_in_two_years_l21_21476

-- Definitions based on conditions
def lennon_age_current : ℕ := 8
def ophelia_age_current : ℕ := 38
def lennon_age_in_two_years := lennon_age_current + 2
def ophelia_age_in_two_years := ophelia_age_current + 2

-- Statement to prove
theorem age_ratio_in_two_years : 
  (ophelia_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 4 ∧
  (lennon_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 1 := 
by 
  sorry

end age_ratio_in_two_years_l21_21476


namespace history_paper_pages_l21_21775

theorem history_paper_pages (days: ℕ) (pages_per_day: ℕ) (h₁: days = 3) (h₂: pages_per_day = 27) : days * pages_per_day = 81 := 
by
  sorry

end history_paper_pages_l21_21775


namespace days_of_supply_l21_21571

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end days_of_supply_l21_21571


namespace normal_prob_neg1_to_1_l21_21448

noncomputable def normal_distribution_0_sigma (σ : ℝ) : ProbabilityTheory.ProbabilitySpace ℝ :=
ProbabilityTheory.ProbabilitySpace.normal 0 σ^2

theorem normal_prob_neg1_to_1 {σ : ℝ} (hσ : 0 < σ) :
  (ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, x < -1) = 0.2) →
  (ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, -1 < x ∧ x < 1) = 0.6) :=
begin
  intros h1,
  have h2 : ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, x > 1) = 0.2,
  { sorry },  -- skipped proof of symmetry, which relies on properties of the normal distribution
  have h_total : ProbabilityTheory.ProbabilitySpace.probability (normal_distribution_0_sigma σ) (λ x, true) = 1 := 
    ProbabilityTheory.ProbabilitySpace.probability_univ (normal_distribution_0_sigma σ),
  sorry  -- skipped proof,
end

end normal_prob_neg1_to_1_l21_21448


namespace sqrt_factorial_product_squared_l21_21806

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l21_21806


namespace length_linear_function_alpha_increase_l21_21057

variable (l : ℝ) (l₀ : ℝ) (t : ℝ) (α : ℝ)

theorem length_linear_function 
  (h_formula : l = l₀ * (1 + α * t)) : 
  ∃ (f : ℝ → ℝ), (∀ t, f t = l₀ + l₀ * α * t ∧ (l = f t)) :=
by {
  -- Proof would go here
  sorry
}

theorem alpha_increase 
  (h_formula : l = l₀ * (1 + α * t))
  (h_initial : t = 1) :
  α = (l - l₀) / l₀ :=
by {
  -- Proof would go here
  sorry
}

end length_linear_function_alpha_increase_l21_21057


namespace coeff_x_term_l21_21651

noncomputable def binomial_coeff (n k : ℕ) : ℕ := (nat.choose n k)

theorem coeff_x_term :
  let f := (1 - 2 / x^2) * (2 + sqrt x)^6 in
  coeff (λ x, f) 1 = 238 :=
by
  sorry

end coeff_x_term_l21_21651


namespace maximum_value_of_expression_l21_21922

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression (x y z : ℝ ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) :
  problem_statement x y z ≤ 81 / 4 :=
sorry

end maximum_value_of_expression_l21_21922


namespace john_spent_money_on_soap_l21_21621

def number_of_bars : ℕ := 20
def weight_per_bar : ℝ := 1.5
def cost_per_pound : ℝ := 0.5

theorem john_spent_money_on_soap :
  let total_weight := number_of_bars * weight_per_bar in
  let total_cost := total_weight * cost_per_pound in
  total_cost = 15 :=
by
  sorry

end john_spent_money_on_soap_l21_21621


namespace train_speed_clicks_l21_21787

theorem train_speed_clicks (x : ℝ) (rail_length_feet : ℝ := 40) (clicks_per_mile : ℝ := 5280/ 40) :
  15 ≤ (2400/5280) * 60  * clicks_per_mile ∧ (2400/5280) * 60 * clicks_per_mile ≤ 30 :=
by {
  sorry
}

end train_speed_clicks_l21_21787


namespace range_of_a_l21_21630

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x + a / x + 7 else x + a / x - 7

theorem range_of_a (a : ℝ) (ha : 0 < a)
  (hodd : ∀ x : ℝ, f (-x) a = -f x a)
  (hcond : ∀ x : ℝ, 0 ≤ x → f x a ≥ 1 - a) :
  4 ≤ a := sorry

end range_of_a_l21_21630


namespace skier_total_time_l21_21537

variable (t1 t2 t3 : ℝ)

-- Conditions
def condition1 : Prop := t1 + t2 = 40.5
def condition2 : Prop := t2 + t3 = 37.5
def condition3 : Prop := 1 / t2 = 2 / (t1 + t3)

-- Theorem to prove total time is 58.5 minutes
theorem skier_total_time (h1 : condition1 t1 t2) (h2 : condition2 t2 t3) (h3 : condition3 t1 t2 t3) : t1 + t2 + t3 = 58.5 := 
by
  sorry

end skier_total_time_l21_21537


namespace max_gcd_b_n_b_n_plus_1_l21_21496

noncomputable def b (n : ℕ) : ℚ := (2 ^ n - 1) / 3

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, Int.gcd (b n).num (b (n + 1)).num = 1 :=
by
  sorry

end max_gcd_b_n_b_n_plus_1_l21_21496


namespace arith_seq_formula_l21_21909

noncomputable def arith_seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + a (n + 2) = 4 * n + 6

theorem arith_seq_formula (a : ℕ → ℤ) (h : arith_seq a) : ∀ n : ℕ, a n = 2 * n + 1 :=
by
  intros
  sorry

end arith_seq_formula_l21_21909


namespace average_score_of_class_l21_21549

theorem average_score_of_class : 
  ∀ (total_students assigned_students make_up_students : ℕ)
    (assigned_avg_score make_up_avg_score : ℚ),
    total_students = 100 →
    assigned_students = 70 →
    make_up_students = total_students - assigned_students →
    assigned_avg_score = 60 →
    make_up_avg_score = 80 →
    (assigned_students * assigned_avg_score + make_up_students * make_up_avg_score) / total_students = 66 :=
by
  intro total_students assigned_students make_up_students assigned_avg_score make_up_avg_score
  intros h_total_students h_assigned_students h_make_up_students h_assigned_avg_score h_make_up_avg_score
  sorry

end average_score_of_class_l21_21549


namespace ribbon_count_proof_l21_21052

theorem ribbon_count_proof :
  (∃ N : ℕ, (1 / 4) * N + (1 / 3) * N + (1 / 6) * N + 40 = N ∧ (∃ orange_ribbons : ℕ, orange_ribbons = (1 / 6) * N ∧ orange_ribbons ≈ 27)) :=
sorry

end ribbon_count_proof_l21_21052


namespace ana_bonita_age_gap_l21_21569

theorem ana_bonita_age_gap (A B n : ℚ) (h1 : A = 2 * B + 3) (h2 : A - 2 = 6 * (B - 2)) (h3 : A = B + n) : n = 6.25 :=
by
  sorry

end ana_bonita_age_gap_l21_21569


namespace measure_angle_ABG_l21_21766

-- Formalizing the conditions
def is_regular_octagon (polygon : Fin 8 → ℝ × ℝ) : Prop :=
  let vertices := [polygon 0, polygon 1, polygon 2, polygon 3, polygon 4, polygon 5, polygon 6, polygon 7]
  (∀ i, ∥vertices ((i + 1) % 8) - vertices i∥ = ∥vertices 1 - vertices 0∥) ∧ 
  (∀ i, ∠ (vertices (i + 1) % 8) (vertices i) (vertices (i - 1 + 8) % 8) = 135)

-- Define angle_measure, considering the numbering polygon Fin 8 from 0 to 7
def angle_measure_polygon (polygon : Fin 8 → ℝ × ℝ) (i j k : Fin 8) : ℝ :=
  ∠ (polygon j) (polygon i) (polygon k)

-- The proof problem statement
theorem measure_angle_ABG (polygon : Fin 8 → ℝ × ℝ) (h : is_regular_octagon polygon) : 
  angle_measure_polygon polygon 0 1 6 = 22.5 :=
sorry

end measure_angle_ABG_l21_21766


namespace passing_marks_required_l21_21682

theorem passing_marks_required (T : ℝ)
  (h1 : 0.30 * T + 60 = 0.40 * T)
  (h2 : 0.40 * T = passing_mark)
  (h3 : 0.50 * T - 40 = passing_mark) :
  passing_mark = 240 := by
  sorry

end passing_marks_required_l21_21682


namespace find_new_bottle_caps_l21_21710

theorem find_new_bottle_caps (initial caps_thrown current : ℕ) (h_initial : initial = 69)
  (h_thrown : caps_thrown = 60) (h_current : current = 67) :
  ∃ n, initial - caps_thrown + n = current ∧ n = 58 := by
sorry

end find_new_bottle_caps_l21_21710


namespace _l21_21272

noncomputable def height_of_cone (r : ℝ) (θ : ℝ) : ℝ :=
  let L := (θ / 360) * 2 * Real.pi * r in      -- Arc length, which is the circumference of the base of the cone
  let r_base := L / (2 * Real.pi) in           -- Radius of the base of the cone
  let h_sq := r^2 - r_base^2 in                -- Using Pythagorean theorem to find height^2
  Real.sqrt h_sq

example : height_of_cone 5 (162 * Real.pi / 180) = Real.sqrt 319 / 4 :=
by sorry

end _l21_21272


namespace identity_of_polynomials_l21_21511

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l21_21511


namespace solve_for_x_l21_21591

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end solve_for_x_l21_21591


namespace bonus_tasks_l21_21501

-- Definition for earnings without bonus
def earnings_without_bonus (tasks : ℕ) : ℕ := tasks * 2

-- Definition for calculating the total bonus received
def total_bonus (tasks : ℕ) (earnings : ℕ) : ℕ := earnings - earnings_without_bonus tasks

-- Definition for the number of bonuses received given the total bonus and a single bonus amount
def number_of_bonuses (total_bonus : ℕ) (bonus_amount : ℕ) : ℕ := total_bonus / bonus_amount

-- The theorem we want to prove
theorem bonus_tasks (tasks : ℕ) (earnings : ℕ) (bonus_amount : ℕ) (bonus_tasks : ℕ) :
  earnings = 78 →
  tasks = 30 →
  bonus_amount = 6 →
  bonus_tasks = tasks / (number_of_bonuses (total_bonus tasks earnings) bonus_amount) →
  bonus_tasks = 10 :=
by
  intros h_earnings h_tasks h_bonus_amount h_bonus_tasks
  sorry

end bonus_tasks_l21_21501


namespace smallest_total_cells_marked_l21_21084

-- Definitions based on problem conditions
def grid_height : ℕ := 8
def grid_width : ℕ := 13

def squares_per_height : ℕ := grid_height / 2
def squares_per_width : ℕ := grid_width / 2

def initial_marked_cells_per_square : ℕ := 1
def additional_marked_cells_per_square : ℕ := 1

def number_of_squares : ℕ := squares_per_height * squares_per_width
def initial_marked_cells : ℕ := number_of_squares * initial_marked_cells_per_square
def additional_marked_cells : ℕ := number_of_squares * additional_marked_cells_per_square

def total_marked_cells : ℕ := initial_marked_cells + additional_marked_cells

-- Statement of the proof problem
theorem smallest_total_cells_marked : total_marked_cells = 48 := by 
    -- Proof is not required as per the instruction
    sorry

end smallest_total_cells_marked_l21_21084


namespace total_viewing_time_l21_21412

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end total_viewing_time_l21_21412


namespace bees_hatch_every_day_l21_21982

   /-- 
   Given:
   - The queen loses 900 bees every day.
   - The initial number of bees is 12500.
   - After 7 days, the total number of bees is 27201.
   
   Prove:
   - The number of bees hatching from the queen's eggs every day is 3001.
   -/
   
   theorem bees_hatch_every_day :
     ∃ x : ℕ, 12500 + 7 * (x - 900) = 27201 → x = 3001 :=
   sorry
   
end bees_hatch_every_day_l21_21982


namespace find_initial_red_balloons_l21_21622

-- Define the initial state of balloons and the assumption.
def initial_blue_balloons : ℕ := 4
def red_balloons_after_inflation (R : ℕ) : ℕ := R + 2
def blue_balloons_after_inflation : ℕ := initial_blue_balloons + 2
def total_balloons (R : ℕ) : ℕ := red_balloons_after_inflation R + blue_balloons_after_inflation

-- Define the likelihood condition.
def likelihood_red (R : ℕ) : Prop := (red_balloons_after_inflation R : ℚ) / (total_balloons R : ℚ) = 0.4

-- Statement of the problem.
theorem find_initial_red_balloons (R : ℕ) (h : likelihood_red R) : R = 2 := by
  sorry

end find_initial_red_balloons_l21_21622


namespace subtraction_identity_l21_21799

theorem subtraction_identity : 3.57 - 1.14 - 0.23 = 2.20 := sorry

end subtraction_identity_l21_21799


namespace find_a_l21_21344

theorem find_a (a : ℤ) (h_range : 0 ≤ a ∧ a < 13) (h_div : (51 ^ 2022 + a) % 13 = 0) : a = 12 := 
by
  sorry

end find_a_l21_21344


namespace sequence_first_five_terms_l21_21603

noncomputable def a_n (n : ℕ) : ℤ := (-1) ^ n + (n : ℤ)

theorem sequence_first_five_terms :
  a_n 1 = 0 ∧
  a_n 2 = 3 ∧
  a_n 3 = 2 ∧
  a_n 4 = 5 ∧
  a_n 5 = 4 :=
by
  sorry

end sequence_first_five_terms_l21_21603


namespace min_sum_of_factors_of_2310_l21_21956

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l21_21956


namespace johns_quadratic_l21_21211

theorem johns_quadratic (d e : ℤ) (h1 : d^2 = 16) (h2 : 2 * d * e = -40) : d * e = -20 :=
sorry

end johns_quadratic_l21_21211


namespace probability_of_product_divisible_by_4_l21_21385

-- Define a 6-sided die
def die := {1, 2, 3, 4, 5, 6}

-- Define the event that the product of 5 rolls is divisible by 4
def product_divisible_by_4 (rolls : Vector ℕ 5) : Prop :=
  (rolls.toList.prod % 4 = 0)

-- Calculate the probability of the event that the product is divisible by 4
def probability_divisible_by_4 : ℚ :=
  -- Here, we should compute the probability
  sorry

-- Statement of the theorem
theorem probability_of_product_divisible_by_4 :
  probability_divisible_by_4 = 11 / 12 :=
sorry

end probability_of_product_divisible_by_4_l21_21385


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21153

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21153


namespace find_coordinates_of_D_l21_21765

theorem find_coordinates_of_D :
  ∃ (D : ℝ × ℝ),
    (∃ (λ : ℝ), 0 ≤ λ ∧ λ ≤ 1 ∧ 
    D = (P.1 + λ * (Q.1 - P.1), P.2 + λ * (Q.2 - P.2))
    ∧ dist D P = 2 * dist D Q) ∧
    D = (3, 7) :=
by
  let P : ℝ × ℝ := (-3, -2)
  let Q : ℝ × ℝ := (5, 10)
  use (3, 7)
  use 0.75 -- because PD = 2DQ happens when λ = 0.75
  sorry

end find_coordinates_of_D_l21_21765


namespace sum_of_cubes_identity_l21_21495

theorem sum_of_cubes_identity (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end sum_of_cubes_identity_l21_21495


namespace no_third_degree_polynomial_exists_l21_21679

theorem no_third_degree_polynomial_exists (a b c d : ℤ) (h : a ≠ 0) :
  ¬(p 15 = 3 ∧ p 21 = 12 ∧ p = λ x => a * x ^ 3 + b * x ^ 2 + c * x + d) :=
sorry

end no_third_degree_polynomial_exists_l21_21679


namespace Joan_spent_68_353_on_clothing_l21_21210

theorem Joan_spent_68_353_on_clothing :
  let shorts := 15.00
  let jacket := 14.82 * 0.9
  let shirt := 12.51 * 0.5
  let shoes := 21.67 - 3
  let hat := 8.75
  let belt := 6.34
  shorts + jacket + shirt + shoes + hat + belt = 68.353 :=
sorry

end Joan_spent_68_353_on_clothing_l21_21210


namespace exists_dense_H_with_chords_disjoint_from_K_l21_21343

open Set

variable (K : Set (EuclideanSpace ℝ 3)) (S2 : Set (EuclideanSpace ℝ 3))

-- Assume K is a closed subset of the closed unit ball in R^3
axiom closed_K : IsClosed K
axiom K_subset_closed_unit_ball : K ⊆ Metric.ball (0 : EuclideanSpace ℝ 3) 1

-- Define the property of the family of chords Ω
variable (Ω : Set (Set (EuclideanSpace ℝ 3)))

axiom chord_property : ∀ X Y ∈ S2, ∃ (X' Y' : EuclideanSpace ℝ 3), X' ∈ S2 ∧ Y' ∈ S2 ∧ Metric.dist X X' < 1 ∧ Metric.dist Y Y' < 1 ∧ ({X', Y'} ∈ Ω) ∧ Disjoint ({X', Y'} : Set (EuclideanSpace ℝ 3)) K

-- Define \( H \) is dense in \( S^2 \)
def dense_in_S2 (H : Set (EuclideanSpace ℝ 3)) : Prop :=
  ∀ x ∈ S2, ∀ ε > 0, ∃ y ∈ H, Metric.dist x y < ε

-- Define property of \( H \)
def chords_disjoint_from_K (H : Set (EuclideanSpace ℝ 3)) : Prop :=
  ∀ x y ∈ H, Disjoint ({x, y} : Set (EuclideanSpace ℝ 3)) K

-- Main theorem
theorem exists_dense_H_with_chords_disjoint_from_K : 
  ∃ H ⊆ S2, dense_in_S2 S2 H ∧ chords_disjoint_from_K K H := sorry

end exists_dense_H_with_chords_disjoint_from_K_l21_21343


namespace value_of_square_of_sum_l21_21745

theorem value_of_square_of_sum (x y: ℝ) 
(h1: 2 * x * (x + y) = 58) 
(h2: 3 * y * (x + y) = 111):
  (x + y)^2 = (169/5)^2 := by
  sorry

end value_of_square_of_sum_l21_21745


namespace at_least_one_not_less_than_two_l21_21028

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a + (1 / b) ≥ 2 ∨ b + (1 / a) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l21_21028


namespace box_dimensions_l21_21068

theorem box_dimensions (x y z : ℝ) (h1 : x * y * z = 160) 
  (h2 : y * z = 80) (h3 : x * z = 40) (h4 : x * y = 32) : 
  x = 4 ∧ y = 8 ∧ z = 10 :=
by
  -- Placeholder for the actual proof steps
  sorry

end box_dimensions_l21_21068


namespace leaves_blew_away_correct_l21_21924

-- Definitions based on conditions
def original_leaves : ℕ := 356
def leaves_left : ℕ := 112
def leaves_blew_away : ℕ := original_leaves - leaves_left

-- Theorem statement based on the question and correct answer
theorem leaves_blew_away_correct : leaves_blew_away = 244 := by {
  -- Proof goes here (omitted for now)
  sorry
}

end leaves_blew_away_correct_l21_21924


namespace range_of_a_l21_21042

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (∀ x y, 3 ≤ x ∧ x ≤ y → (x^2 - 2*a*x + 2) ≤ (y^2 - 2*a*y + 2)) → a ≤ 3 := 
sorry

end range_of_a_l21_21042


namespace total_students_l21_21378

-- Lean statement: Prove the number of students given the conditions.
theorem total_students (num_classrooms : ℕ) (num_buses : ℕ) (seats_per_bus : ℕ) 
  (students : ℕ) (h1 : num_classrooms = 87) (h2 : num_buses = 29) 
  (h3 : seats_per_bus = 2) (h4 : students = num_classrooms * num_buses * seats_per_bus) :
  students = 5046 :=
by
  sorry

end total_students_l21_21378


namespace collinear_points_l21_21453

variables (a b : ℝ × ℝ) (A B C D : ℝ × ℝ)

-- Define the vectors
noncomputable def vec_AB : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def vec_BC : ℝ × ℝ := (2 * a.1 + 8 * b.1, 2 * a.2 + 8 * b.2)
noncomputable def vec_CD : ℝ × ℝ := (3 * (a.1 - b.1), 3 * (a.2 - b.2))

-- Define the collinearity condition
def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Translate the problem statement into Lean
theorem collinear_points (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) (h₂ : ¬ (a.1 * b.2 - a.2 * b.1 = 0)):
  collinear (6 * (a.1 + b.1), 6 * (a.2 + b.2)) (5 * (a.1 + b.1, a.2 + b.2)) :=
sorry

end collinear_points_l21_21453


namespace emily_cards_l21_21004

theorem emily_cards (initial_cards : ℕ) (total_cards : ℕ) (given_cards : ℕ) 
  (h1 : initial_cards = 63) (h2 : total_cards = 70) 
  (h3 : total_cards = initial_cards + given_cards) : 
  given_cards = 7 := 
by 
  sorry

end emily_cards_l21_21004


namespace problem_a4_inv_a4_l21_21313

theorem problem_a4_inv_a4 (a : ℝ) (h : (a + 1/a)^4 = 16) : (a^4 + 1/a^4) = 2 := 
by 
  sorry

end problem_a4_inv_a4_l21_21313


namespace cube_surface_area_l21_21911

/-- A cube with an edge length of 10 cm has smaller cubes with edge length 2 cm 
    dug out from the middle of each face. The surface area of the new shape is 696 cm². -/
theorem cube_surface_area (original_edge : ℝ) (small_cube_edge : ℝ)
  (original_edge_eq : original_edge = 10) (small_cube_edge_eq : small_cube_edge = 2) :
  let original_surface := 6 * original_edge ^ 2
  let removed_area := 6 * small_cube_edge ^ 2
  let added_area := 6 * 5 * small_cube_edge ^ 2
  let new_surface := original_surface - removed_area + added_area
  new_surface = 696 := by
  sorry

end cube_surface_area_l21_21911


namespace original_price_l21_21985

theorem original_price (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 60) 
  (h2 : rate_of_profit = 0.20) 
  (h3 : SP = CP * (1 + rate_of_profit)) : CP = 50 := by
  sorry

end original_price_l21_21985


namespace symmetry_of_F_l21_21609

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
|f x| + f (|x|)

theorem symmetry_of_F (f : ℝ → ℝ) (h : is_odd_function f) :
    ∀ x : ℝ, F f x = F f (-x) :=
by
  sorry

end symmetry_of_F_l21_21609


namespace prime_sum_divisible_l21_21709

theorem prime_sum_divisible (p : Fin 2021 → ℕ) (prime : ∀ i, Nat.Prime (p i))
  (h : 6060 ∣ Finset.univ.sum (fun i => (p i)^4)) : 4 ≤ Finset.card (Finset.univ.filter (fun i => p i < 2021)) :=
sorry

end prime_sum_divisible_l21_21709


namespace problem1_l21_21427

theorem problem1 :
  (15 * (-3 / 4) + (-15) * (3 / 2) + 15 / 4) = -30 :=
by
  sorry

end problem1_l21_21427


namespace parity_of_expression_l21_21064

theorem parity_of_expression (o1 o2 n : ℕ) (h1 : o1 % 2 = 1) (h2 : o2 % 2 = 1) : 
  ((o1 * o1 + n * (o1 * o2)) % 2 = 1 ↔ n % 2 = 0) :=
by sorry

end parity_of_expression_l21_21064


namespace min_sum_of_factors_l21_21962

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21962


namespace price_ratio_l21_21144

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l21_21144


namespace train_travel_time_l21_21115

theorem train_travel_time
  (a : ℝ) (s : ℝ) (t : ℝ)
  (ha : a = 3)
  (hs : s = 27)
  (h0 : ∀ t, 0 ≤ t) :
  t = Real.sqrt 18 :=
by
  sorry

end train_travel_time_l21_21115


namespace classroom_gpa_l21_21656

theorem classroom_gpa (n : ℕ) (x : ℝ)
  (h1 : n > 0)
  (h2 : (1/3 : ℝ) * n * 45 + (2/3 : ℝ) * n * x = n * 55) : x = 60 :=
by
  sorry

end classroom_gpa_l21_21656


namespace beanie_babies_total_l21_21498

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end beanie_babies_total_l21_21498


namespace min_sum_of_factors_of_2310_l21_21958

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l21_21958


namespace maria_total_money_l21_21350

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end maria_total_money_l21_21350


namespace divisible_iff_l21_21076

theorem divisible_iff (m n k : ℕ) (h : m > n) : 
  (3^(k+1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
sorry

end divisible_iff_l21_21076


namespace rectangle_area_given_conditions_l21_21784

theorem rectangle_area_given_conditions
  (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_given_conditions_l21_21784


namespace divisor_of_1053_added_with_5_is_2_l21_21389

theorem divisor_of_1053_added_with_5_is_2 :
  ∃ d : ℕ, d > 1 ∧ ∀ (x : ℝ), x = 5.000000000000043 → (1053 + x) % d = 0 → d = 2 :=
by
  sorry

end divisor_of_1053_added_with_5_is_2_l21_21389


namespace lola_wins_probability_l21_21048

theorem lola_wins_probability (p : ℚ) (h : p = 3 / 7) : 1 - p = 4 / 7 :=
by
  rw [h]
  norm_num
  sorry

end lola_wins_probability_l21_21048


namespace sector_area_l21_21887

theorem sector_area (theta : ℝ) (L : ℝ) (h_theta : theta = π / 3) (h_L : L = 4) :
  ∃ r : ℝ, (L = r * theta ∧ ∃ A : ℝ, A = 1/2 * r^2 * theta ∧ A = 24 / π) := by
  sorry

end sector_area_l21_21887


namespace PropA_impl_PropB_not_PropB_impl_PropA_l21_21183

variable {x : ℝ}

def PropA (x : ℝ) : Prop := abs (x - 1) < 5
def PropB (x : ℝ) : Prop := abs (abs x - 1) < 5

theorem PropA_impl_PropB : PropA x → PropB x :=
by sorry

theorem not_PropB_impl_PropA : ¬(PropB x → PropA x) :=
by sorry

end PropA_impl_PropB_not_PropB_impl_PropA_l21_21183


namespace right_triangle_has_one_right_angle_l21_21456

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_l21_21456


namespace jack_sugar_final_l21_21479

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l21_21479


namespace gel_pen_ratio_l21_21126

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l21_21126


namespace raine_steps_l21_21360

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end raine_steps_l21_21360


namespace tangent_planes_of_surface_and_given_plane_l21_21435

-- Define the surface and the given plane
def surface (x y z : ℝ) := (x^2 + 4 * y^2 + 9 * z^2 = 1)
def given_plane (x y z : ℝ) := (x + y + 2 * z = 1)

-- Define the tangent plane equations to be proved
def tangent_plane_1 (x y z : ℝ) := (x + y + 2 * z - (109 / (6 * Real.sqrt 61)) = 0)
def tangent_plane_2 (x y z : ℝ) := (x + y + 2 * z + (109 / (6 * Real.sqrt 61)) = 0)

-- The statement to be proved
theorem tangent_planes_of_surface_and_given_plane :
  ∀ x y z, surface x y z ∧ given_plane x y z →
    tangent_plane_1 x y z ∨ tangent_plane_2 x y z :=
sorry

end tangent_planes_of_surface_and_given_plane_l21_21435


namespace clara_cookies_l21_21428

theorem clara_cookies (n : ℕ) :
  (15 * n - 1) % 11 = 0 → n = 3 := 
sorry

end clara_cookies_l21_21428


namespace sum_is_zero_l21_21345

noncomputable def z : ℂ := Complex.cos (3 * Real.pi / 8) + Complex.sin (3 * Real.pi / 8) * Complex.I

theorem sum_is_zero (hz : z^8 = 1) (hz1 : z ≠ 1) :
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^12)) = 0 :=
by
  sorry

end sum_is_zero_l21_21345


namespace evaluate_expression_l21_21580

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a * b + b^2) = 97 / 7 := by
  sorry

example : (3^4 + 2^4) / (3^2 - 3 * 2 + 2^2) = 97 / 7 := evaluate_expression 3 2 rfl rfl

end evaluate_expression_l21_21580


namespace pies_baked_l21_21072

/-- Mrs. Hilt baked 16.0 pecan pies and 14.0 apple pies. She needs 5.0 times this amount.
    Prove that the total number of pies she has to bake is 150.0. -/
theorem pies_baked (pecan_pies : ℝ) (apple_pies : ℝ) (times : ℝ)
  (h1 : pecan_pies = 16.0) (h2 : apple_pies = 14.0) (h3 : times = 5.0) :
  times * (pecan_pies + apple_pies) = 150.0 := by
  sorry

end pies_baked_l21_21072


namespace businessman_expenditure_l21_21090

theorem businessman_expenditure (P : ℝ) (h1 : P * 1.21 = 24200) : P = 20000 := 
by sorry

end businessman_expenditure_l21_21090


namespace prod_div_sum_le_square_l21_21950

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end prod_div_sum_le_square_l21_21950


namespace f_expression_f_odd_l21_21189

noncomputable def f (x : ℝ) (a b : ℝ) := (2^x + b) / (2^x + a)

theorem f_expression :
  ∃ a b, f 1 a b = 1 / 3 ∧ f 0 a b = 0 ∧ (∀ x, f x a b = (2^x - 1) / (2^x + 1)) :=
by
  sorry

theorem f_odd :
  ∀ x, f x 1 (-1) = (2^x - 1) / (2^x + 1) ∧ f (-x) 1 (-1) = -f x 1 (-1) :=
by
  sorry

end f_expression_f_odd_l21_21189


namespace sum_x_y_m_l21_21921

theorem sum_x_y_m (a b x y m : ℕ) (ha : a - b = 3) (hx : x = 10 * a + b) (hy : y = 10 * b + a) (hxy : x^2 - y^2 = m^2) : x + y + m = 178 := sorry

end sum_x_y_m_l21_21921


namespace zero_in_interval_l21_21929

theorem zero_in_interval (x y : ℝ) (hx_lt_0 : x < 0) (hy_gt_0 : 0 < y) (hy_lt_1 : y < 1) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) : x^5 < 0 ∧ 0 < y^8 :=
by
  sorry

end zero_in_interval_l21_21929


namespace child_is_late_l21_21193

theorem child_is_late 
  (distance : ℕ)
  (rate1 rate2 : ℕ) 
  (early_arrival : ℕ)
  (time_late_at_rate1 : ℕ)
  (time_required_by_rate1 : ℕ)
  (time_required_by_rate2 : ℕ)
  (actual_time : ℕ)
  (T : ℕ) :
  distance = 630 ∧ 
  rate1 = 5 ∧ 
  rate2 = 7 ∧ 
  early_arrival = 30 ∧
  (time_required_by_rate1 = distance / rate1) ∧
  (time_required_by_rate2 = distance / rate2) ∧
  (actual_time + T = time_required_by_rate1) ∧
  (actual_time - early_arrival = time_required_by_rate2) →
  T = 6 := 
by
  intros
  sorry

end child_is_late_l21_21193


namespace find_value_of_a2_plus_b2_plus_c2_l21_21216

variables (a b c : ℝ)

-- Define the conditions
def conditions := (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a + b + c = 0) ∧ (a^3 + b^3 + c^3 = a^5 + b^5 + c^5)

-- State the theorem we need to prove
theorem find_value_of_a2_plus_b2_plus_c2 (h : conditions a b c) : a^2 + b^2 + c^2 = 6 / 5 :=
  sorry

end find_value_of_a2_plus_b2_plus_c2_l21_21216


namespace jen_total_birds_l21_21208

-- Define the number of chickens and ducks
variables (C D : ℕ)

-- Define the conditions
def ducks_condition (C D : ℕ) : Prop := D = 4 * C + 10
def num_ducks (D : ℕ) : Prop := D = 150

-- Define the total number of birds
def total_birds (C D : ℕ) : ℕ := C + D

-- Prove that the total number of birds is 185 given the conditions
theorem jen_total_birds (C D : ℕ) (h1 : ducks_condition C D) (h2 : num_ducks D) : total_birds C D = 185 :=
by
  sorry

end jen_total_birds_l21_21208


namespace zero_in_interval_x5_y8_l21_21930

theorem zero_in_interval_x5_y8
    (x y : ℝ)
    (h1 : x^5 < y^8)
    (h2 : y^8 < y^3)
    (h3 : y^3 < x^6)
    (h4 : x < 0)
    (h5 : 0 < y)
    (h6 : y < 1) :
    0 ∈ set.Ioo (x^5) (y^8) :=
by
  sorry

end zero_in_interval_x5_y8_l21_21930


namespace solve_for_cubic_l21_21899

theorem solve_for_cubic (x y : ℝ) (h₁ : x * (x + y) = 49) (h₂: y * (x + y) = 63) : (x + y)^3 = 448 * Real.sqrt 7 := 
sorry

end solve_for_cubic_l21_21899


namespace algebra_expression_value_l21_21749

theorem algebra_expression_value (x y : ℤ) (h : x - 2 * y + 2 = 5) : 2 * x - 4 * y - 1 = 5 :=
by
  sorry

end algebra_expression_value_l21_21749


namespace intersection_A_B_find_coefficients_a_b_l21_21347

open Set

variable {X : Type} (x : X)

def setA : Set ℝ := { x | x^2 < 9 }
def setB : Set ℝ := { x | (x - 2) * (x + 4) < 0 }
def A_inter_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def A_union_B_solution_set : Set ℝ := { x | -4 < x ∧ x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | -3 < x ∧ x < 2 } :=
sorry

theorem find_coefficients_a_b (a b : ℝ) :
  (∀ x, 2 * x^2 + a * x + b < 0 ↔ -4 < x ∧ x < 3) → 
  a = 2 ∧ b = -24 :=
sorry

end intersection_A_B_find_coefficients_a_b_l21_21347


namespace deers_distribution_l21_21525

theorem deers_distribution (a_1 d a_2 a_5 : ℚ) 
  (h1 : a_2 = a_1 + d)
  (h2 : 5 * a_1 + 10 * d = 5)
  (h3 : a_2 = 2 / 3) :
  a_5 = 1 / 3 :=
sorry

end deers_distribution_l21_21525


namespace repeating_decimal_as_fraction_l21_21436

theorem repeating_decimal_as_fraction : 
  (0.\overline{36} : ℝ) = (4/11 : ℚ) := 
sorry

end repeating_decimal_as_fraction_l21_21436


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21151

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21151


namespace room_length_l21_21782

theorem room_length (L : ℝ) (width height door_area window_area cost_per_sq_ft total_cost : ℝ) 
    (num_windows : ℕ) (door_w window_w door_h window_h : ℝ)
    (h_width : width = 15) (h_height : height = 12) 
    (h_cost_per_sq_ft : cost_per_sq_ft = 9)
    (h_door_area : door_area = door_w * door_h)
    (h_window_area : window_area = window_w * window_h)
    (h_num_windows : num_windows = 3)
    (h_door_dim : door_w = 6 ∧ door_h = 3)
    (h_window_dim : window_w = 4 ∧ window_h = 3)
    (h_total_cost : total_cost = 8154) :
    (2 * height * (L + width) - (door_area + num_windows * window_area)) * cost_per_sq_ft = total_cost →
    L = 25 := 
by
  intros h_cost_eq
  sorry

end room_length_l21_21782


namespace arrangement_count_l21_21382

-- Define the problem conditions
def students : ℕ := 3
def villages : ℕ := 2
def total_arrangements : ℕ := 6

-- Define the property we want to prove
theorem arrangement_count :
  ∃ (s v : ℕ), s = students ∧ v = villages ∧ ∑ k in finset.range ((nat.choose students 2) * (nat.choose (students-2) 1) * nat.factorial villages), k = total_arrangements :=
by
  sorry

end arrangement_count_l21_21382


namespace group_A_can_form_triangle_l21_21530

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem group_A_can_form_triangle : can_form_triangle 9 6 13 :=
by
  sorry

end group_A_can_form_triangle_l21_21530


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21150

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21150


namespace eval_expr_l21_21037

theorem eval_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x^(2 * y) * y^(3 * x) / (y^(2 * y) * x^(3 * x))) = x^(2 * y - 3 * x) * y^(3 * x - 2 * y) :=
by
  sorry

end eval_expr_l21_21037


namespace function_defined_for_all_reals_l21_21866

theorem function_defined_for_all_reals (m : ℝ) :
  (∀ x : ℝ, 7 * x ^ 2 + m - 6 ≠ 0) → m > 6 :=
by
  sorry

end function_defined_for_all_reals_l21_21866


namespace smaller_factor_of_4851_l21_21373

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end smaller_factor_of_4851_l21_21373


namespace pen_price_ratio_l21_21162

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l21_21162


namespace pyramid_volume_l21_21524

noncomputable def volume_of_pyramid (S α β : ℝ) : ℝ :=
  (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β)))

theorem pyramid_volume 
  (S α β : ℝ)
  (base_area : S > 0)
  (equal_lateral_edges : true)
  (dihedral_angles : α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2) :
  volume_of_pyramid S α β = (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β))) :=
by
  sorry

end pyramid_volume_l21_21524


namespace cube_volume_surface_area_value_l21_21000

theorem cube_volume_surface_area_value (x : ℝ) : 
  (∃ s : ℝ, s = (6 * x)^(1 / 3) ∧ 6 * s^2 = 2 * x) → 
  x = 1 / 972 :=
by {
  sorry
}

end cube_volume_surface_area_value_l21_21000


namespace n_fraction_sum_l21_21102

theorem n_fraction_sum {n : ℝ} {lst : List ℝ} (h_len : lst.length = 21) 
(h_mem : n ∈ lst) 
(h_avg : n = 4 * (lst.erase n).sum / 20) :
  n = (lst.sum) / 6 :=
by
  sorry

end n_fraction_sum_l21_21102


namespace initial_average_mark_l21_21942

-- Define the conditions
def total_students := 13
def average_mark := 72
def excluded_students := 5
def excluded_students_average := 40
def remaining_students := total_students - excluded_students
def remaining_students_average := 92

-- Define the total marks calculations
def initial_total_marks (A : ℕ) : ℕ := total_students * A
def excluded_total_marks : ℕ := excluded_students * excluded_students_average
def remaining_total_marks : ℕ := remaining_students * remaining_students_average

-- Prove the initial average mark
theorem initial_average_mark : 
  initial_total_marks average_mark = excluded_total_marks + remaining_total_marks →
  average_mark = 72 :=
by
  sorry

end initial_average_mark_l21_21942


namespace sum_of_series_l21_21706

theorem sum_of_series : 
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := by
  sorry

end sum_of_series_l21_21706


namespace percent_sum_l21_21746

theorem percent_sum (A B C : ℝ)
  (hA : 0.45 * A = 270)
  (hB : 0.35 * B = 210)
  (hC : 0.25 * C = 150) :
  0.75 * A + 0.65 * B + 0.45 * C = 1110 := by
  sorry

end percent_sum_l21_21746


namespace evaluate_square_of_sum_l21_21736

theorem evaluate_square_of_sum (x y : ℕ) (h1 : x + y = 20) (h2 : 2 * x + y = 27) : (x + y) ^ 2 = 400 :=
by
  sorry

end evaluate_square_of_sum_l21_21736


namespace product_even_probability_l21_21181

theorem product_even_probability 
  (dice1 dice2 : ℕ) (h1 : dice1 ∈ set.univ ∩ {1..10}) (h2 : dice2 ∈ set.univ ∩ {1..10}) :
  (75 : ℚ) / 100 = 3 / 4 :=
sorry

end product_even_probability_l21_21181


namespace part_b_part_c_part_d_l21_21303

variables (A B : Event) (P : Probability)

axiom PA : P A = 0.3
axiom PB : P B = 0.6

theorem part_b (h : P (A ∩ B) = 0.18) : Independent A B := sorry

theorem part_c (h : CondProbability P B A = 0.6) : Independent A B := sorry

theorem part_d (h : Independent A B) : P (A ∪ B) = 0.72 := by
  have PA_inter_B : P (A ∩ B) = P A * P B := Independent.mul_inter h
  calc
    P (A ∪ B)
        = P A + P B - P (A ∩ B) := ProbabilityUnion P A B
    ... = 0.3 + 0.6 - (0.3 * 0.6) := by rw [PA, PB, PA_inter_B]
    ... = 0.3 + 0.6 - 0.18 := by norm_num
    ... = 0.72 := by norm_num

end part_b_part_c_part_d_l21_21303


namespace abc_is_772_l21_21035

noncomputable def find_abc (a b c : ℝ) : ℝ :=
if h₁ : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * (b + c) = 160 ∧ b * (c + a) = 168 ∧ c * (a + b) = 180
then 772 else 0

theorem abc_is_772 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
(h₄ : a * (b + c) = 160) (h₅ : b * (c + a) = 168) (h₆ : c * (a + b) = 180) :
  find_abc a b c = 772 := by
  sorry

end abc_is_772_l21_21035


namespace compute_sqrt_factorial_square_l21_21826

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l21_21826


namespace sqrt_factorial_squared_l21_21818

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l21_21818


namespace cookie_problem_l21_21865

theorem cookie_problem : 
  ∃ (B : ℕ), B = 130 ∧ B - 80 = 50 ∧ B/2 + 20 = 85 :=
by
  sorry

end cookie_problem_l21_21865


namespace seven_pow_fifty_one_mod_103_l21_21934

theorem seven_pow_fifty_one_mod_103 : (7^51 - 1) % 103 = 0 := 
by
  -- Fermat's Little Theorem: If p is a prime number and a is an integer not divisible by p,
  -- then a^(p-1) ≡ 1 ⧸ p.
  -- 103 is prime, so for 7 which is not divisible by 103, we have 7^102 ≡ 1 ⧸ 103.
sorry

end seven_pow_fifty_one_mod_103_l21_21934


namespace star_three_four_eq_zero_l21_21460

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - 2 * a * b

theorem star_three_four_eq_zero : star 3 4 = 0 := sorry

end star_three_four_eq_zero_l21_21460


namespace smaller_number_l21_21375

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end smaller_number_l21_21375


namespace common_ratio_geometric_sequence_l21_21203

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

theorem common_ratio_geometric_sequence
  (a3_eq : a 3 = 2 * S 2 + 1)
  (a4_eq : a 4 = 2 * S 3 + 1)
  (geometric_seq : ∀ n, a (n+1) = a 1 * (q ^ n))
  (h₀ : a 1 ≠ 0)
  (h₁ : q ≠ 0) :
  q = 3 :=
sorry

end common_ratio_geometric_sequence_l21_21203


namespace largest_two_digit_integer_l21_21801

theorem largest_two_digit_integer
  (a b : ℕ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 3 * (10 * a + b) = 10 * b + a + 5) :
  10 * a + b = 13 :=
by {
  -- Sorry is placed here to indicate that the proof is not provided
  sorry
}

end largest_two_digit_integer_l21_21801


namespace perfect_square_difference_l21_21494

theorem perfect_square_difference (m n : ℕ) (h : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, k^2 = m - n :=
sorry

end perfect_square_difference_l21_21494


namespace solve_problem_l21_21705

noncomputable def problem_statement : ℤ :=
  (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2

theorem solve_problem : problem_statement = -166 :=
by 
  -- Proof omitted
  sorry

end solve_problem_l21_21705


namespace profit_in_december_l21_21902

variable (a : ℝ)

theorem profit_in_december (h_a: a > 0):
  (1 - 0.06) * (1 + 0.10) * a = (1 - 0.06) * (1 + 0.10) * a :=
by
  sorry

end profit_in_december_l21_21902


namespace smallest_integer_value_of_m_l21_21322

theorem smallest_integer_value_of_m (x y m : ℝ) 
  (h1 : 3*x + y = m + 8) 
  (h2 : 2*x + 2*y = 2*m + 5) 
  (h3 : x - y < 1) : 
  m >= 3 := 
sorry

end smallest_integer_value_of_m_l21_21322


namespace cost_price_eq_560_l21_21117

variables (C SP1 SP2 : ℝ)
variables (h1 : SP1 = 0.79 * C) (h2 : SP2 = SP1 + 140) (h3 : SP2 = 1.04 * C)

theorem cost_price_eq_560 : C = 560 :=
by 
  sorry

end cost_price_eq_560_l21_21117


namespace inequality_proof_l21_21229

theorem inequality_proof
  (a b c d : ℝ)
  (ha : abs a > 1)
  (hb : abs b > 1)
  (hc : abs c > 1)
  (hd : abs d > 1)
  (h : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
sorry

end inequality_proof_l21_21229


namespace arithmetic_sequence_general_term_l21_21907

theorem arithmetic_sequence_general_term (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = 3 * n^2 + 2 * n) →
  a 1 = S 1 ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) →
  ∀ n, a n = 6 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l21_21907


namespace sum_of_first_four_terms_of_geometric_sequence_l21_21739

noncomputable def geometric_sum_first_four (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : q > 0) 
  (h3 : a 2 = 1) 
  (h4 : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  geometric_sum_first_four a q = 15 / 2 :=
sorry

end sum_of_first_four_terms_of_geometric_sequence_l21_21739


namespace find_number_l21_21394

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end find_number_l21_21394


namespace min_value_expression_l21_21173

theorem min_value_expression (x y : ℝ) : ∃ (a b : ℝ), x = a ∧ y = b ∧ (x^2 + y^2 - 8*x - 6*y + 30 = 5) :=
by
  sorry

end min_value_expression_l21_21173


namespace replace_stars_with_identity_l21_21509

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l21_21509


namespace painting_time_l21_21894

theorem painting_time (t₁₂ : ℕ) (h : t₁₂ = 6) (r : ℝ) (hr : r = t₁₂ / 12) (n : ℕ) (hn : n = 20) : 
  t₁₂ + n * r = 16 := by
  sorry

end painting_time_l21_21894


namespace repeating_decimal_to_fraction_denominator_l21_21368

theorem repeating_decimal_to_fraction_denominator :
  ∀ (S : ℚ), (S = 0.27) → (∃ a b : ℤ, b ≠ 0 ∧ S = a / b ∧ Int.gcd a b = 1 ∧ b = 3) :=
by
  sorry

end repeating_decimal_to_fraction_denominator_l21_21368


namespace probability_of_winning_l21_21660

theorem probability_of_winning (P_lose : ℚ) (h1 : P_lose = 5 / 8) : 1 - P_lose = 3 / 8 :=
by
  rw [h1]
  norm_num
  sorry -- This sorry avoids using the solution steps directly

#print probability_of_winning -- This can help in verifying that the statement is properly defined

end probability_of_winning_l21_21660


namespace chess_team_boys_l21_21683

variable (B G : ℕ)

theorem chess_team_boys (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 20) : B = 15 := by
  sorry

end chess_team_boys_l21_21683


namespace emily_final_score_l21_21834

theorem emily_final_score :
  16 + 33 - 48 = 1 :=
by
  -- proof skipped
  sorry

end emily_final_score_l21_21834


namespace sin_cos_identity_l21_21707

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) 
  - Real.cos (200 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 := 
by
  -- This would be where the proof goes
  sorry

end sin_cos_identity_l21_21707


namespace horner_eval_v3_at_minus4_l21_21255

def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

def horner_form (x : ℤ) : ℤ :=
  let a6 := 3
  let a5 := 5
  let a4 := 6
  let a3 := 79
  let a2 := -8
  let a1 := 35
  let a0 := 12
  let v := a6
  let v1 := v * x + a5
  let v2 := v1 * x + a4
  let v3 := v2 * x + a3
  let v4 := v3 * x + a2
  let v5 := v4 * x + a1
  let v6 := v5 * x + a0
  v3

theorem horner_eval_v3_at_minus4 :
  horner_form (-4) = -57 :=
by
  sorry

end horner_eval_v3_at_minus4_l21_21255


namespace valid_conditions_x_y_z_l21_21100

theorem valid_conditions_x_y_z (x y z : ℤ) :
  x = y - 1 ∧ z = y + 1 ∨ x = y ∧ z = y + 1 ↔ x * (x - y) + y * (y - x) + z * (z - y) = 1 :=
sorry

end valid_conditions_x_y_z_l21_21100


namespace pen_price_ratio_l21_21160

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l21_21160


namespace initial_erasers_count_l21_21500

noncomputable def erasers_lost := 42
noncomputable def erasers_ended_up_with := 53

theorem initial_erasers_count (initial_erasers : ℕ) : 
  initial_erasers_ended_up_with = initial_erasers - erasers_lost → initial_erasers = 95 :=
by
  sorry

end initial_erasers_count_l21_21500


namespace det_matrixE_l21_21625

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l21_21625


namespace infinitely_many_pairs_l21_21228

theorem infinitely_many_pairs : ∀ b : ℕ, ∃ a : ℕ, 2019 < 2^a / 3^b ∧ 2^a / 3^b < 2020 := 
by
  sorry

end infinitely_many_pairs_l21_21228


namespace maximum_sum_of_numbers_in_grid_l21_21045

theorem maximum_sum_of_numbers_in_grid :
  ∀ (grid : List (List ℕ)) (rect_cover : (ℕ × ℕ) → (ℕ × ℕ) → Prop),
  (∀ x y, rect_cover x y → x ≠ y → x.1 < 6 → x.2 < 6 → y.1 < 6 → y.2 < 6) →
  (∀ x y z w, rect_cover x y ∧ rect_cover z w → 
    (x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∨ (x.1 = z.1 ∨ x.2 = z.2) → 
    (x.1 = z.1 ∧ x.2 = y.2 ∨ x.2 = z.2 ∧ x.1 = y.1)) → False) →
  (36 = 6 * 6) →
  18 = 36 / 2 →
  342 = (18 * 19) :=
by
  intro grid rect_cover h_grid h_no_common_edge h_grid_size h_num_rectangles
  sorry

end maximum_sum_of_numbers_in_grid_l21_21045


namespace probability_of_rain_l21_21432

-- Define the conditions in Lean
variables (x : ℝ) -- probability of rain

-- Known condition: taking an umbrella 20% of the time
def takes_umbrella : Prop := 0.2 = x + ((1 - x) * x)

-- The desired problem statement
theorem probability_of_rain : takes_umbrella x → x = 1 / 9 :=
by
  -- placeholder for the proof
  intro h
  sorry

end probability_of_rain_l21_21432


namespace volume_of_prism_l21_21846

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 15) (hwh : w * h = 20) (hlh : l * h = 24) : l * w * h = 60 := 
sorry

end volume_of_prism_l21_21846


namespace days_of_supply_l21_21572

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end days_of_supply_l21_21572


namespace dollar_symmetric_l21_21878

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric {x y : ℝ} : dollar (x + y) (y + x) = 0 :=
by
  sorry

end dollar_symmetric_l21_21878


namespace three_million_times_three_million_l21_21680

theorem three_million_times_three_million : 
  (3 * 10^6) * (3 * 10^6) = 9 * 10^12 := 
by
  sorry

end three_million_times_three_million_l21_21680


namespace proof_problem_l21_21474

variable {a : ℕ → ℝ} -- sequence a
variable {S : ℕ → ℝ} -- partial sums sequence S 
variable {n : ℕ} -- index

-- Define the conditions
def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n+1) = a n + d

def S_is_partial_sum (a S : ℕ → ℝ) : Prop := 
  ∀ n, S (n+1) = S n + a (n+1)

-- The properties given in the problem
def conditions (a S : ℕ → ℝ) : Prop :=
  is_arith_seq a ∧ 
  S_is_partial_sum a S ∧ 
  S 6 < S 7 ∧ 
  S 7 > S 8

-- The conclusions that need to be proved
theorem proof_problem (a S : ℕ → ℝ) (h : conditions a S) : 
  S 9 < S 6 ∧
  (∀ n, a 1 ≥ a (n+1)) ∧
  (∀ m, S 7 ≥ S m) := by 
  sorry

end proof_problem_l21_21474


namespace pills_in_a_week_l21_21312

def insulin_pills_per_day : Nat := 2
def blood_pressure_pills_per_day : Nat := 3
def anticonvulsant_pills_per_day : Nat := 2 * blood_pressure_pills_per_day

def total_pills_per_day : Nat := insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsant_pills_per_day

theorem pills_in_a_week : total_pills_per_day * 7 = 77 := by
  sorry

end pills_in_a_week_l21_21312


namespace squared_expression_is_matching_string_l21_21426

theorem squared_expression_is_matching_string (n : ℕ) (h : n > 0) :
  let a := (10^n - 1) / 9
  let term1 := 4 * a * (9 * a + 2)
  let term2 := 10 * a + 1
  let term3 := 6 * a
  let exp := term1 + term2 - term3
  Nat.sqrt exp = 6 * a + 1 := by
  sorry

end squared_expression_is_matching_string_l21_21426


namespace largest_of_consecutive_numbers_l21_21650

theorem largest_of_consecutive_numbers (avg : ℕ) (n : ℕ) (h1 : n = 7) (h2 : avg = 20) :
  let sum := n * avg in
  let middle := sum / n in
  let largest := middle + 3 in
  largest = 23 :=
by
  -- Introduce locals to use 
  let sum := n * avg
  let middle := sum / n
  let largest := middle + 3
  -- Add the proof placeholder
  sorry

end largest_of_consecutive_numbers_l21_21650


namespace juan_speed_l21_21487

-- Statement of given distances and time
def distance : ℕ := 80
def time : ℕ := 8

-- Desired speed in miles per hour
def expected_speed : ℕ := 10

-- Theorem statement: Speed is distance divided by time and should equal 10 miles per hour
theorem juan_speed : distance / time = expected_speed :=
  by
  sorry

end juan_speed_l21_21487


namespace exists_integer_n_tangent_l21_21588
open Real

noncomputable def degree_to_radian (d : ℝ) : ℝ :=
  d * (π / 180)

theorem exists_integer_n_tangent :
  ∃ (n : ℤ), -90 < (n : ℝ) ∧ (n : ℝ) < 90 ∧ tan (degree_to_radian (n : ℝ)) = tan (degree_to_radian 345) ∧ n = -15 :=
by
  sorry

end exists_integer_n_tangent_l21_21588


namespace min_value_expr_l21_21445

theorem min_value_expr (x y : ℝ) : 
  ∃ min_val, min_val = 2 ∧ min_val ≤ (x + y)^2 + (x - 1/y)^2 :=
sorry

end min_value_expr_l21_21445


namespace robin_albums_l21_21770

theorem robin_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums_created : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : pics_per_album = 8)
  (h4 : total_pics = phone_pics + camera_pics)
  (h5 : albums_created = total_pics / pics_per_album) : albums_created = 5 := 
sorry

end robin_albums_l21_21770


namespace area_bounded_arcsin_cos_l21_21587

noncomputable def area_arcsin_cos (a b : ℝ) : ℝ :=
  ∫ x in a .. b, Real.arcsin (Real.cos x)

theorem area_bounded_arcsin_cos :
  area_arcsin_cos 0 (3 * Real.pi) = (3 * Real.pi^2) / 4 :=
by
  sorry

end area_bounded_arcsin_cos_l21_21587


namespace sqrt_factorial_mul_square_l21_21822

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l21_21822


namespace leak_takes_3_hours_to_empty_l21_21274

noncomputable def leak_emptying_time (inlet_rate_per_minute: ℕ) (tank_empty_time_with_inlet: ℕ) (tank_capacity: ℕ) : ℕ :=
  let inlet_rate_per_hour := inlet_rate_per_minute * 60
  let effective_empty_rate := tank_capacity / tank_empty_time_with_inlet
  let leak_rate := inlet_rate_per_hour + effective_empty_rate
  tank_capacity / leak_rate

theorem leak_takes_3_hours_to_empty:
  leak_emptying_time 6 12 1440 = 3 := 
sorry

end leak_takes_3_hours_to_empty_l21_21274


namespace proof_problem_l21_21733

noncomputable def a_n (n : ℕ) : ℕ := n + 2
noncomputable def b_n (n : ℕ) : ℕ := 2 * n + 3
noncomputable def C_n (n : ℕ) : ℚ := 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))
noncomputable def T_n (n : ℕ) : ℚ := (1/4) * (1 - (1/(2 * n + 1)))

theorem proof_problem :
  (∀ n, a_n n = n + 2) ∧
  (∀ n, b_n n = 2 * n + 3) ∧
  (∀ n, C_n n = 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))) ∧
  (∀ n, T_n n = (1/4) * (1 - (1/(2 * n + 1)))) ∧
  (∀ n, (T_n n > k / 54) ↔ k < 9) :=
by
  sorry

end proof_problem_l21_21733


namespace second_option_feasible_l21_21008

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l21_21008


namespace min_children_see_ear_l21_21330

theorem min_children_see_ear (n : ℕ) : ∃ (k : ℕ), k = n + 2 :=
by
  sorry

end min_children_see_ear_l21_21330


namespace inverse_of_parallel_lines_l21_21403

theorem inverse_of_parallel_lines 
  (P Q : Prop) 
  (parallel_impl_alt_angles : P → Q) :
  (Q → P) := 
by
  sorry

end inverse_of_parallel_lines_l21_21403


namespace exists_real_solution_real_solution_specific_values_l21_21644

theorem exists_real_solution (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

theorem real_solution_specific_values  (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

end exists_real_solution_real_solution_specific_values_l21_21644


namespace isosceles_triangle_area_l21_21118

-- Definitions
def isosceles_triangle (b h : ℝ) : Prop :=
∃ a : ℝ, a * b / 2 = a * h

def square_of_area_one (a : ℝ) : Prop :=
a = 1

def centroids_coincide (g_triangle g_square : ℝ × ℝ) : Prop :=
g_triangle = g_square

-- The statement of the problem
theorem isosceles_triangle_area
  (b h : ℝ)
  (s : ℝ)
  (triangle_centroid : ℝ × ℝ)
  (square_centroid : ℝ × ℝ)
  (H1 : isosceles_triangle b h)
  (H2 : square_of_area_one s)
  (H3 : centroids_coincide triangle_centroid square_centroid)
  : b * h / 2 = 9 / 4 :=
by
  sorry

end isosceles_triangle_area_l21_21118


namespace andrew_donuts_l21_21853

/--
Andrew originally asked for 3 donuts for each of his 2 friends, Brian and Samuel. 
Then invited 2 more friends and asked for the same amount of donuts for them. 
Andrew’s mother wants to buy one more donut for each of Andrew’s friends. 
Andrew's mother is also going to buy the same amount of donuts for Andrew as everybody else.
Given these conditions, the total number of donuts Andrew’s mother needs to buy is 20.
-/
theorem andrew_donuts : (3 * 2) + (3 * 2) + 4 + 4 = 20 :=
by
  -- Given:
  -- 1. Andrew asked for 3 donuts for each of his two friends, Brian and Samuel.
  -- 2. He later invited 2 more friends and asked for the same amount of donuts for them.
  -- 3. Andrew’s mother wants to buy one more donut for each of Andrew’s friends.
  -- 4. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else.
  -- Prove: The total number of donuts Andrew’s mother needs to buy is 20.
  sorry

end andrew_donuts_l21_21853


namespace gcd_seq_finitely_many_values_l21_21624

def gcd_seq_finite_vals (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  (∀ n ≥ 2, x (n + 1) = A * Nat.gcd (x n) (x (n-1)) + B) →
  ∃ N : ℕ, ∀ m n, m ≥ N → n ≥ N → x m = x n

theorem gcd_seq_finitely_many_values (A B : ℕ) (x : ℕ → ℕ) :
  gcd_seq_finite_vals A B x :=
by
  intros h
  sorry

end gcd_seq_finitely_many_values_l21_21624


namespace percentage_below_cost_l21_21798

variable (CP SP : ℝ)

-- Given conditions
def cost_price : ℝ := 5625
def more_for_profit : ℝ := 1800
def profit_percentage : ℝ := 0.16
def expected_SP : ℝ := cost_price + (cost_price * profit_percentage)
def actual_SP : ℝ := expected_SP - more_for_profit

-- Statement to prove
theorem percentage_below_cost (h1 : CP = cost_price) (h2 : SP = actual_SP) :
  (CP - SP) / CP * 100 = 16 := by
sorry

end percentage_below_cost_l21_21798


namespace two_hundredth_digit_of_7_over_29_l21_21257

theorem two_hundredth_digit_of_7_over_29 :
  (decimal_places ⟨7, 29⟩ 200) = 1 :=
sorry

end two_hundredth_digit_of_7_over_29_l21_21257


namespace gel_pen_price_ratio_l21_21154

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l21_21154


namespace solution_to_equation_l21_21774

noncomputable def solve_equation (x : ℝ) : Prop :=
  x + 2 = 1 / (x - 2) ∧ x ≠ 2

theorem solution_to_equation (x : ℝ) (h : solve_equation x) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 :=
sorry

end solution_to_equation_l21_21774


namespace list_size_is_2017_l21_21843

def has_sum (L : List ℤ) (n : ℤ) : Prop :=
  List.sum L = n

def has_product (L : List ℤ) (n : ℤ) : Prop :=
  List.prod L = n

def includes (L : List ℤ) (n : ℤ) : Prop :=
  n ∈ L

theorem list_size_is_2017 
(L : List ℤ) :
  has_sum L 2018 ∧ 
  has_product L 2018 ∧ 
  includes L 2018 
  → L.length = 2017 :=
by 
  sorry

end list_size_is_2017_l21_21843


namespace number_of_partners_l21_21785

def total_profit : ℝ := 80000
def majority_owner_share := 0.25 * total_profit
def remaining_profit := total_profit - majority_owner_share
def partner_share := 0.25 * remaining_profit
def combined_share := majority_owner_share + 2 * partner_share

theorem number_of_partners : combined_share = 50000 → remaining_profit / partner_share = 4 := by
  intro h1
  have h_majority : majority_owner_share = 0.25 * total_profit := by sorry
  have h_remaining : remaining_profit = total_profit - majority_owner_share := by sorry
  have h_partner : partner_share = 0.25 * remaining_profit := by sorry
  have h_combined : combined_share = majority_owner_share + 2 * partner_share := by sorry
  calc
    remaining_profit / partner_share = _ := by sorry
    4 = 4 := by sorry

end number_of_partners_l21_21785


namespace subgroup_in_center_l21_21214

-- Definitions corresponding to the conditions
variables {G : Type*} [Group G] 
variables {H : Subgroup G}

def Z (G : Type*) [Group G] := {a : G | ∀ x : G, a * x = x * a}

theorem subgroup_in_center
  (n : ℕ) (hn : 2 ≤ n)
  (p : ℕ) (hp : Nat.Prime p) (hp_dvd : p ∣ n)
  (hH : H.order = p)
  (unique_H : ∀ K : Subgroup G, K.order = p → K = H) : 
  H ≤ Z G :=
sorry

end subgroup_in_center_l21_21214


namespace exists_x0_l21_21075

theorem exists_x0 : ∃ x0 : ℝ, x0^2 + 2*x0 + 1 ≤ 0 :=
sorry

end exists_x0_l21_21075


namespace cell_phone_height_l21_21249

theorem cell_phone_height (width perimeter : ℕ) (h1 : width = 9) (h2 : perimeter = 46) : 
  ∃ length : ℕ, length = 14 ∧ perimeter = 2 * (width + length) :=
by
  sorry

end cell_phone_height_l21_21249


namespace sqrt_factorial_mul_square_l21_21821

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l21_21821


namespace angle_C_of_triangle_l21_21058

theorem angle_C_of_triangle (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := 
by
  sorry

end angle_C_of_triangle_l21_21058


namespace nail_polish_count_l21_21623

-- Definitions from conditions
def K : ℕ := 25
def H : ℕ := K + 8
def Ka : ℕ := K - 6
def L : ℕ := 2 * K
def S : ℕ := 13 + 10  -- Since 25 / 2 = 12.5, rounded to 13 for practical purposes

-- Statement to prove
def T : ℕ := H + Ka + L + S

theorem nail_polish_count : T = 125 := by
  sorry

end nail_polish_count_l21_21623


namespace gel_pen_price_ratio_l21_21156

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l21_21156


namespace volume_relation_l21_21469

variable {x y z V : ℝ}

theorem volume_relation
  (top_area : x * y = A)
  (side_area : y * z = B)
  (volume : x * y * z = V) :
  (y * z) * (x * y * z)^2 = z^3 * V := by
  sorry

end volume_relation_l21_21469


namespace mass_percentage_Cl_in_HClO2_is_51_78_l21_21589

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_HClO2 : ℝ :=
  molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

noncomputable def mass_percentage_Cl_in_HClO2 : ℝ :=
  (molar_mass_Cl / molar_mass_HClO2) * 100

theorem mass_percentage_Cl_in_HClO2_is_51_78 :
  mass_percentage_Cl_in_HClO2 = 51.78 := 
sorry

end mass_percentage_Cl_in_HClO2_is_51_78_l21_21589


namespace people_in_room_l21_21743

variable (total_chairs occupied_chairs people_present : ℕ)
variable (h1 : total_chairs = 28)
variable (h2 : occupied_chairs = 14)
variable (h3 : (2 / 3 : ℚ) * people_present = 14)
variable (h4 : total_chairs = 2 * occupied_chairs)

theorem people_in_room : people_present = 21 := 
by 
  --proof will be here
  sorry

end people_in_room_l21_21743


namespace factorize_expression_l21_21014

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end factorize_expression_l21_21014


namespace exam_full_marks_l21_21204

variables {A B C D F : ℝ}

theorem exam_full_marks
  (hA : A = 0.90 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.80 * D)
  (hA_val : A = 360)
  (hD : D = 0.80 * F) 
  : F = 500 :=
sorry

end exam_full_marks_l21_21204


namespace a_n_geometric_sequence_b_n_general_term_l21_21449

theorem a_n_geometric_sequence (t : ℝ) (h : t ≠ 0 ∧ t ≠ 1) :
  (∀ n, ∃ r : ℝ, a_n = t^n) :=
sorry

theorem b_n_general_term (t : ℝ) (h1 : t ≠ 0 ∧ t ≠ 1) (h2 : ∀ n, a_n = t^n)
  (h3 : ∃ q : ℝ, q = (2 * t^2 + t) / 2) :
  (∀ n, b_n = (t^(n + 1) * (2 * t + 1)^(n - 1)) / 2^(n - 2)) :=
sorry

end a_n_geometric_sequence_b_n_general_term_l21_21449


namespace jane_brown_sheets_l21_21486

theorem jane_brown_sheets :
  ∀ (total_sheets yellow_sheets brown_sheets : ℕ),
    total_sheets = 55 →
    yellow_sheets = 27 →
    brown_sheets = total_sheets - yellow_sheets →
    brown_sheets = 28 := 
by
  intros total_sheets yellow_sheets brown_sheets ht hy hb
  rw [ht, hy] at hb
  simp at hb
  exact hb

end jane_brown_sheets_l21_21486


namespace prove_values_of_a_l21_21864

-- Definitions of the conditions
def condition_1 (a x y : ℝ) : Prop := (x * y)^(1/3) = a^(a^2)
def condition_2 (a x y : ℝ) : Prop := (Real.log x / Real.log a * Real.log y / Real.log a) + (Real.log y / Real.log a * Real.log x / Real.log a) = 3 * a^3

-- The proof problem
theorem prove_values_of_a (a x y : ℝ) (h1 : condition_1 a x y) (h2 : condition_2 a x y) : a > 0 ∧ a ≤ 2/3 :=
sorry

end prove_values_of_a_l21_21864


namespace f_f_0_eq_zero_number_of_zeros_l21_21304

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then 1 - 1/x else (a - 1) * x + 1

theorem f_f_0_eq_zero (a : ℝ) : f a (f a 0) = 0 := by
  sorry

theorem number_of_zeros (a : ℝ) : 
  if a = 1 then ∃! x, f a x = 0 else
  if a > 1 then ∃! x1, ∃! x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 else
  ∃! x, f a x = 0 := by sorry

end f_f_0_eq_zero_number_of_zeros_l21_21304


namespace original_useful_item_is_pencil_l21_21611

def code_language (x : String) : String :=
  if x = "item" then "pencil"
  else if x = "pencil" then "mirror"
  else if x = "mirror" then "board"
  else x

theorem original_useful_item_is_pencil : 
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") ∧
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") 
  → "mirror" = "pencil" :=
by sorry

end original_useful_item_is_pencil_l21_21611


namespace find_pos_ints_l21_21017

theorem find_pos_ints (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
    (((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n)) →
    (a^m + 1 ∣ (a + 1)^n)) :=
by
  sorry

end find_pos_ints_l21_21017


namespace solve_system_l21_21554

theorem solve_system :
  (∀ x y : ℝ, log 4 x - log 2 y = 0 ∧ x^2 - 5 * y^2 + 4 = 0 → 
    (x, y) = (1, 1) ∨ (x, y) = (4, 2)) :=
by
  intros x y h
  cases h with Hlog Heq
  sorry

end solve_system_l21_21554


namespace range_of_m_l21_21218

theorem range_of_m (m : ℝ) :
  (∃ x y, y = x^2 + m * x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) → m ≤ -1 :=
by
  sorry

end range_of_m_l21_21218


namespace sqrt_factorial_sq_l21_21803

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l21_21803


namespace geometric_sequence_first_term_l21_21789

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 5) 
  (h2 : a * r^3 = 45) : 
  a = 5 / (3^(2/3)) := 
by
  -- proof steps to be filled here
  sorry

end geometric_sequence_first_term_l21_21789


namespace snow_probability_l21_21613

theorem snow_probability :
  let p₁ := 1/2
  let p₂ := 2/3
  let p₃ := 3/4
  let p₄ := 4/5
  let p₅ := 5/6
  let p₆ := 7/8
  let p₇ := 7/8
  1 - (p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇) = 139 / 384 := by
sorry

end snow_probability_l21_21613


namespace classroom_students_count_l21_21467

-- Definitions from the conditions
def students (C S Sh : ℕ) : Prop :=
  S = 2 * C ∧
  S = Sh + 8 ∧
  Sh = C + 19

-- Proof statement
theorem classroom_students_count (C S Sh : ℕ) 
  (h : students C S Sh) : 3 * C = 81 :=
by
  sorry

end classroom_students_count_l21_21467


namespace min_calls_required_l21_21715

-- Define the set of people involved in the communication
inductive Person
| A | B | C | D | E | F

-- Function to calculate the minimum number of calls for everyone to know all pieces of gossip
def minCalls : ℕ :=
  9

-- Theorem stating the minimum number of calls required
theorem min_calls_required : minCalls = 9 := by
  sorry

end min_calls_required_l21_21715


namespace find_y_intercept_l21_21422

theorem find_y_intercept (m b x y : ℝ) (h1 : m = 2) (h2 : (x, y) = (239, 480)) (line_eq : y = m * x + b) : b = 2 :=
by
  sorry

end find_y_intercept_l21_21422


namespace relationship_of_y_values_l21_21734

theorem relationship_of_y_values 
  (k : ℝ) (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_pos : k > 0) 
  (hA : y1 = k / x1) 
  (hB : y2 = k / x2) 
  (hC : y3 = k / x3) 
  (h_order : x1 < 0 ∧ 0 < x2 ∧ x2 < x3) : y1 < y3 ∧ y3 < y2 := 
by
  sorry

end relationship_of_y_values_l21_21734


namespace price_ratio_l21_21145

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l21_21145


namespace equivalent_resistance_is_15_l21_21596

-- Definitions based on conditions
def R : ℝ := 5 -- Resistance of each resistor in Ohms
def num_resistors : ℕ := 4

-- The equivalent resistance due to the short-circuit path removing one resistor
def simplified_circuit_resistance : ℝ := (num_resistors - 1) * R

-- The statement to prove
theorem equivalent_resistance_is_15 :
  simplified_circuit_resistance = 15 :=
by
  sorry

end equivalent_resistance_is_15_l21_21596


namespace sara_spent_on_hotdog_l21_21080

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end sara_spent_on_hotdog_l21_21080


namespace identity_verification_l21_21518

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l21_21518


namespace transformed_sum_l21_21987

theorem transformed_sum (n : ℕ) (y : Fin n → ℝ) (s : ℝ) (h : s = (Finset.univ.sum (fun i => y i))) :
  Finset.univ.sum (fun i => 3 * (y i) + 30) = 3 * s + 30 * n :=
by 
  sorry

end transformed_sum_l21_21987


namespace findCostPrices_l21_21847

def costPriceOfApple (sp_a : ℝ) (cp_a : ℝ) : Prop :=
  sp_a = (5 / 6) * cp_a

def costPriceOfOrange (sp_o : ℝ) (cp_o : ℝ) : Prop :=
  sp_o = (3 / 4) * cp_o

def costPriceOfBanana (sp_b : ℝ) (cp_b : ℝ) : Prop :=
  sp_b = (9 / 8) * cp_b

theorem findCostPrices (sp_a sp_o sp_b : ℝ) (cp_a cp_o cp_b : ℝ) :
  costPriceOfApple sp_a cp_a → 
  costPriceOfOrange sp_o cp_o → 
  costPriceOfBanana sp_b cp_b → 
  sp_a = 20 → sp_o = 15 → sp_b = 6 → 
  cp_a = 24 ∧ cp_o = 20 ∧ cp_b = 16 / 3 :=
by 
  intro h1 h2 h3 sp_a_eq sp_o_eq sp_b_eq
  -- proof goes here
  sorry

end findCostPrices_l21_21847


namespace intersection_at_7_m_l21_21945

def f (x : Int) (d : Int) : Int := 4 * x + d

theorem intersection_at_7_m (d m : Int) (h₁ : f 7 d = m) (h₂ : 7 = f m d) : m = 7 := by
  sorry

end intersection_at_7_m_l21_21945


namespace find_three_digit_number_l21_21941

noncomputable def three_digit_number := ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ 100 * x + 10 * y + z = 345 ∧
  (100 * z + 10 * y + x = 100 * x + 10 * y + z + 198) ∧
  (100 * x + 10 * z + y = 100 * x + 10 * y + z + 9) ∧
  (x^2 + y^2 + z^2 - 2 = 4 * (x + y + z))

theorem find_three_digit_number : three_digit_number :=
sorry

end find_three_digit_number_l21_21941


namespace geometric_sequence_sum_squared_l21_21740

theorem geometric_sequence_sum_squared (a : ℕ → ℕ) (n : ℕ) (q : ℕ) 
    (h_geometric: ∀ n, a (n + 1) = a n * q)
    (h_a1 : a 1 = 2)
    (h_a3 : a 3 = 4) :
    (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 = 1020 :=
by
  sorry

end geometric_sequence_sum_squared_l21_21740


namespace production_cost_per_performance_l21_21238

def overhead_cost := 81000
def income_per_performance := 16000
def performances_needed := 9

theorem production_cost_per_performance :
  ∃ P, 9 * income_per_performance = overhead_cost + 9 * P ∧ P = 7000 :=
by
  sorry

end production_cost_per_performance_l21_21238


namespace adjacent_complementary_is_complementary_l21_21977

/-- Two angles are complementary if their sum is 90 degrees. -/
def complementary (α β : ℝ) : Prop :=
  α + β = 90

/-- Two angles are adjacent complementary if they are complementary and adjacent. -/
def adjacent_complementary (α β : ℝ) : Prop :=
  complementary α β ∧ α > 0 ∧ β > 0

/-- Prove that adjacent complementary angles are complementary. -/
theorem adjacent_complementary_is_complementary (α β : ℝ) : adjacent_complementary α β → complementary α β :=
by
  sorry

end adjacent_complementary_is_complementary_l21_21977


namespace tangent_line_y_intercept_l21_21684

def circle1Center: ℝ × ℝ := (3, 0)
def circle1Radius: ℝ := 3
def circle2Center: ℝ × ℝ := (7, 0)
def circle2Radius: ℝ := 2

theorem tangent_line_y_intercept
    (tangent_line: ℝ × ℝ -> ℝ) 
    (P : tangent_line (circle1Center.1, circle1Center.2 + circle1Radius) = 0) -- Tangent condition for Circle 1
    (Q : tangent_line (circle2Center.1, circle2Center.2 + circle2Radius) = 0) -- Tangent condition for Circle 2
    :
    tangent_line (0, 4.5) = 0 := 
sorry

end tangent_line_y_intercept_l21_21684


namespace intersection_A_B_l21_21735

def A : Set ℝ := { x | 2 * x^2 - 5 * x < 0 }
def B : Set ℝ := { x | 3^(x - 1) ≥ Real.sqrt 3 }

theorem intersection_A_B : A ∩ B = Set.Ico (3 / 2) (5 / 2) := 
by
  sorry

end intersection_A_B_l21_21735


namespace locus_of_feet_of_perpendiculars_from_focus_l21_21021

def parabola_locus (p : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 = (p / 2) * x)

theorem locus_of_feet_of_perpendiculars_from_focus (p : ℝ) :
    parabola_locus p :=
by
  sorry

end locus_of_feet_of_perpendiculars_from_focus_l21_21021


namespace total_spent_amount_l21_21179

-- Define the conditions
def spent_relation (B D : ℝ) : Prop := D = 0.75 * B
def payment_difference (B D : ℝ) : Prop := B = D + 12.50

-- Define the theorem to prove
theorem total_spent_amount (B D : ℝ) 
  (h1 : spent_relation B D) 
  (h2 : payment_difference B D) : 
  B + D = 87.50 :=
sorry

end total_spent_amount_l21_21179


namespace math_problem_l21_21446

noncomputable def proof_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  a^2 + 4 * b^2 + 1 / (a * b) ≥ 4

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : proof_problem a b ha hb :=
by
  sorry

end math_problem_l21_21446


namespace increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l21_21724

variables {k: ℝ} -- Thermal conductivity
variables {A A': ℝ} -- Original and increased contact area
variables {dT: ℝ} -- Temperature difference
variables {dx: ℝ} -- Thickness of the skillet handle

-- Define the heat flow rate according to Fourier's law of heat conduction
def heat_flow_rate (k: ℝ) (A: ℝ) (dT: ℝ) (dx: ℝ) : ℝ :=
  -k * A * (dT / dx)

theorem increased_contact_area_increases_heat_flow 
  (h₁: A' > A) -- Increased contact area
  (h₂: dT / dx > 0) -- Positive temperature gradient
  : heat_flow_rate k A' dT dx > heat_flow_rate k A dT dx :=
by
  -- Proof to show that increased area increases heat flow rate
  sorry

theorem handle_felt_hotter_no_thermodynamic_contradiction 
  (h₁: A' > A)
  (h₂: dT / dx > 0)
  : ¬(heat_flow_rate k A' dT dx contradicts thermodynamic laws) :=
by
  -- Proof to show no contradiction with the laws of thermodynamics
  sorry

end increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l21_21724


namespace goteborg_to_stockholm_distance_l21_21526

/-- 
Given that the distance from Goteborg to Jonkoping on a map is 100 cm 
and the distance from Jonkoping to Stockholm is 150 cm, with a map scale of 1 cm: 20 km,
prove that the total distance from Goteborg to Stockholm passing through Jonkoping is 5000 km.
-/
theorem goteborg_to_stockholm_distance :
  let distance_G_to_J := 100 -- distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- distance from Jonkoping to Stockholm in cm
  let scale := 20 -- scale of the map, 1 cm : 20 km
  distance_G_to_J * scale + distance_J_to_S * scale = 5000 := 
by 
  let distance_G_to_J := 100 -- defining the distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- defining the distance from Jonkoping to Stockholm in cm
  let scale := 20 -- defining the scale of the map, 1 cm : 20 km
  sorry

end goteborg_to_stockholm_distance_l21_21526


namespace probability_interval_constant_term_l21_21447

noncomputable def expand_constant_term : ℝ :=
  (nat.choose 4 1) * (1 : ℝ / (2 : ℝ))^3 * (6 : ℝ)

theorem probability_interval_constant_term :
  let X := measure_theory.gaussian_measure (1 : ℝ) (1 : ℝ) in
  P(3 < X < 4) = 0.0214 :=
by
  sorry

end probability_interval_constant_term_l21_21447


namespace increase_in_value_l21_21916

-- Define the conditions
def starting_weight : ℝ := 400
def weight_multiplier : ℝ := 1.5
def price_per_pound : ℝ := 3

-- Define new weight and values
def new_weight : ℝ := starting_weight * weight_multiplier
def value_at_starting_weight : ℝ := starting_weight * price_per_pound
def value_at_new_weight : ℝ := new_weight * price_per_pound

-- Theorem to prove
theorem increase_in_value : value_at_new_weight - value_at_starting_weight = 600 := by
  sorry

end increase_in_value_l21_21916


namespace eval_expression_l21_21176

theorem eval_expression : 5 - 7 * (8 - 12 / 3^2) * 6 = -275 := by
  sorry

end eval_expression_l21_21176


namespace sunland_more_plates_than_moonland_l21_21635

theorem sunland_more_plates_than_moonland : 
  let sunland_plates := 26^4 * 10^2
  let moonland_plates := 26^3 * 10^3
  (sunland_plates - moonland_plates) = 7321600 := 
by
  sorry

end sunland_more_plates_than_moonland_l21_21635


namespace identity_eq_l21_21515

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l21_21515


namespace maria_total_money_l21_21349

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end maria_total_money_l21_21349


namespace expected_same_as_sixth_die_l21_21365

open ProbabilityTheory

/- Define the probability space -/
noncomputable theory
def die : Probₓ ℕ := uniform Icc 1 6

/- Define the expected_value function -/
def expected_value (n : ℕ) (p : Probₓ ℕ) : ℝ :=
  ∑ i in finₓRange n, (p i * i)

theorem expected_same_as_sixth_die:
  let X : Finₓ 6 → Probₓ ℕ := λ i, if i.val == 5 then uniform Icc 1 6 else dirac (1 / 6),
  expected_value 6 (X 0) + expected_value 6 (X 1) + expected_value 6 (X 2) +
  expected_value 6 (X 3) + expected_value 6 (X 4) + expected_value 6 (X 5) =
  11 / 6 :=
by
  sorry

end expected_same_as_sixth_die_l21_21365


namespace range_of_a_l21_21320

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l21_21320


namespace a1_is_1_l21_21297

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = (2^n - 1)

theorem a1_is_1 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_sum a S) : 
  a 1 = 1 :=
by 
  sorry

end a1_is_1_l21_21297


namespace initial_percentage_proof_l21_21556

noncomputable def initialPercentageAntifreeze (P : ℝ) : Prop :=
  let initial_fluid : ℝ := 4
  let drained_fluid : ℝ := 2.2857
  let added_antifreeze_fluid : ℝ := 2.2857 * 0.8
  let final_percentage : ℝ := 0.5
  let final_fluid : ℝ := 4
  
  let initial_antifreeze : ℝ := initial_fluid * P
  let drained_antifreeze : ℝ := drained_fluid * P
  let total_antifreeze_after_replacement : ℝ := initial_antifreeze - drained_antifreeze + added_antifreeze_fluid
  
  total_antifreeze_after_replacement = final_fluid * final_percentage

-- Prove that the initial percentage is 0.1
theorem initial_percentage_proof : initialPercentageAntifreeze 0.1 :=
by
  dsimp [initialPercentageAntifreeze]
  simp
  exact sorry

end initial_percentage_proof_l21_21556


namespace smallest_piece_length_l21_21698

theorem smallest_piece_length (x : ℕ) :
  (9 - x) + (14 - x) ≤ (16 - x) → x ≥ 7 :=
by
  sorry

end smallest_piece_length_l21_21698


namespace degree_g_greater_than_5_l21_21917

-- Definitions according to the given conditions
variables {f g : Polynomial ℤ}
variables (h : Polynomial ℤ)
variables (r : Fin 81 → ℤ)

-- Condition 1: g(x) divides f(x), meaning there exists an h(x) such that f(x) = g(x) * h(x)
def divides (g f : Polynomial ℤ) := ∃ (h : Polynomial ℤ), f = g * h

-- Condition 2: f(x) - 2008 has at least 81 distinct integer roots
def has_81_distinct_roots (f : Polynomial ℤ) (roots : Fin 81 → ℤ) : Prop :=
  ∀ i : Fin 81, f.eval (roots i) = 2008 ∧ Function.Injective roots

-- The theorem to prove
theorem degree_g_greater_than_5 (nonconst_f : f.degree > 0) (nonconst_g : g.degree > 0) 
  (g_div_f : divides g f) (f_has_roots : has_81_distinct_roots (f - Polynomial.C 2008) r) :
  g.degree > 5 :=
sorry

end degree_g_greater_than_5_l21_21917


namespace AF_over_FB_l21_21754

variables {A B C D F P : Type}
variables [AffineSpace ℝ (A B C D F P)]

-- Condition: AP/PD = 4/3
def ratio_AP_PD (P A D : ℝ) : Prop := (A - P) / (P - D) = 4 / 3

-- Condition: FP/PC = 1/2
def ratio_FP_PC (P F C : ℝ) : Prop := (F - P) / (P - C) = 1 / 2

-- Theorem: AF/FB = 5/9 given the conditions
theorem AF_over_FB (P A D F C B : ℝ) (h1 : ratio_AP_PD P A D) (h2 : ratio_FP_PC P F C) : (F - A) / (B - F) = 5 / 9 := by
  sorry

end AF_over_FB_l21_21754


namespace gel_pen_price_relation_b_l21_21131

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l21_21131


namespace largest_n_exists_l21_21283

theorem largest_n_exists :
  ∃ (n : ℕ), 
  (∀ (x y z : ℕ), n^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14) → n = 9 :=
sorry

end largest_n_exists_l21_21283


namespace music_books_cost_l21_21212

theorem music_books_cost
  (total_money : ℕ) (maths_books_count : ℕ) (maths_books_price : ℕ)
  (science_books_extra_count : ℕ) (science_books_price : ℕ)
  (art_books_multiplier : ℕ) (art_books_price : ℕ) :
  total_money = 500 →
  maths_books_count = 4 →
  maths_books_price = 20 →
  science_books_extra_count = 6 →
  science_books_price = 10 →
  art_books_multiplier = 2 →
  art_books_price = 20 →
  let
    maths_books_cost := maths_books_count * maths_books_price
    science_books_cost := (maths_books_count + science_books_extra_count) * science_books_price
    art_books_cost := (art_books_multiplier * maths_books_count) * art_books_price
    total_cost_excluding_music := maths_books_cost + science_books_cost + art_books_cost
    music_books_cost := total_money - total_cost_excluding_music
  in music_books_cost = 160 :=
by
  intros
  sorry

end music_books_cost_l21_21212


namespace no_integer_solution_l21_21033

theorem no_integer_solution :
  ∀ (x y : ℤ), ¬(x^4 + x + y^2 = 3 * y - 1) :=
by
  intros x y
  sorry

end no_integer_solution_l21_21033


namespace exists_finite_group_with_normal_subgroup_GT_Aut_l21_21714

noncomputable def finite_group_G (n : ℕ) : Type := sorry -- Specific construction details omitted
noncomputable def normal_subgroup_H (n : ℕ) : Type := sorry -- Specific construction details omitted

def Aut_G (n : ℕ) : ℕ := sorry -- Number of automorphisms of G
def Aut_H (n : ℕ) : ℕ := sorry -- Number of automorphisms of H

theorem exists_finite_group_with_normal_subgroup_GT_Aut (n : ℕ) :
  ∃ G H, finite_group_G n = G ∧ normal_subgroup_H n = H ∧ Aut_H n > Aut_G n := sorry

end exists_finite_group_with_normal_subgroup_GT_Aut_l21_21714


namespace younger_brother_age_l21_21969

variable (x y : ℕ)

theorem younger_brother_age :
  x + y = 46 →
  y = x / 3 + 10 →
  y = 19 :=
by
  intros h1 h2
  sorry

end younger_brother_age_l21_21969


namespace team_X_finishes_with_more_points_l21_21872

open Probability

theorem team_X_finishes_with_more_points (X Y : ℕ) 
  (h_conditions : 
    8 = 7 + 1 ∧
    ∀ (x y : ℕ), x ≠ y → (P(team_X_wins (x, y)) = 0.5)) : 
  Probability.event (team_X_points > team_Y_points) = 610 / 1024 :=
sorry

end team_X_finishes_with_more_points_l21_21872


namespace distance_ratio_l21_21262

-- Defining the conditions
def speedA : ℝ := 50 -- Speed of Car A in km/hr
def timeA : ℝ := 6 -- Time taken by Car A in hours

def speedB : ℝ := 100 -- Speed of Car B in km/hr
def timeB : ℝ := 1 -- Time taken by Car B in hours

-- Calculating the distances
def distanceA : ℝ := speedA * timeA -- Distance covered by Car A
def distanceB : ℝ := speedB * timeB -- Distance covered by Car B

-- Statement to prove the ratio of distances
theorem distance_ratio : (distanceA / distanceB) = 3 :=
by
  -- Calculations here might be needed, but we use sorry to indicate proof is pending
  sorry

end distance_ratio_l21_21262


namespace palindrome_probability_divisible_by_11_l21_21844

namespace PalindromeProbability

-- Define the concept of a five-digit palindrome and valid digits
def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 10001 * a + 1010 * b + 100 * c

-- Define the condition for a number being divisible by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Count all five-digit palindromes
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10  -- There are 9 choices for a (1-9), and 10 choices for b and c (0-9)

-- Count five-digit palindromes that are divisible by 11
def count_divisible_by_11_five_digit_palindromes : ℕ :=
  9 * 10  -- There are 9 choices for a, and 10 valid (b, c) pairs for divisibility by 11

-- Calculate the probability
theorem palindrome_probability_divisible_by_11 :
  (count_divisible_by_11_five_digit_palindromes : ℚ) / count_five_digit_palindromes = 1 / 10 :=
  by sorry -- Proof goes here

end PalindromeProbability

end palindrome_probability_divisible_by_11_l21_21844


namespace find_length_second_platform_l21_21981

noncomputable def length_second_platform : Prop :=
  let train_length := 500  -- in meters
  let time_cross_platform := 35  -- in seconds
  let time_cross_pole := 8  -- in seconds
  let second_train_length := 250  -- in meters
  let time_cross_second_train := 45  -- in seconds
  let platform1_scale := 0.75
  let time_cross_platform1 := 27  -- in seconds
  let train_speed := train_length / time_cross_pole
  let platform1_length := train_speed * time_cross_platform1 - train_length
  let platform2_length := platform1_length / platform1_scale
  platform2_length = 1583.33

/- The proof is omitted -/
theorem find_length_second_platform : length_second_platform := sorry

end find_length_second_platform_l21_21981


namespace find_number_l21_21779

theorem find_number (A : ℕ) (B : ℕ) (H1 : B = 300) (H2 : Nat.lcm A B = 2310) (H3 : Nat.gcd A B = 30) : A = 231 := 
by 
  sorry

end find_number_l21_21779


namespace two_hundredth_digit_of_7_over_29_l21_21258

theorem two_hundredth_digit_of_7_over_29 :
  (decimal_places ⟨7, 29⟩ 200) = 1 :=
sorry

end two_hundredth_digit_of_7_over_29_l21_21258


namespace smallest_positive_period_of_y_l21_21713

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.sin (-x / 2 + Real.pi / 4)

-- Statement we need to prove
theorem smallest_positive_period_of_y :
  ∃ T > 0, ∀ x : ℝ, y (x + T) = y x ∧ T = 4 * Real.pi := sorry

end smallest_positive_period_of_y_l21_21713


namespace jack_sugar_l21_21480

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l21_21480


namespace jack_sugar_amount_l21_21485

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l21_21485


namespace check_triangle_345_l21_21545

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem check_triangle_345 : satisfies_triangle_inequality 3 4 5 := by
  sorry

end check_triangle_345_l21_21545


namespace total_travel_time_is_19_hours_l21_21429

-- Define the distances and speeds as constants
def distance_WA_ID := 640
def speed_WA_ID := 80
def distance_ID_NV := 550
def speed_ID_NV := 50

-- Define the times based on the given distances and speeds
def time_WA_ID := distance_WA_ID / speed_WA_ID
def time_ID_NV := distance_ID_NV / speed_ID_NV

-- Define the total time
def total_time := time_WA_ID + time_ID_NV

-- Prove that the total travel time is 19 hours
theorem total_travel_time_is_19_hours : total_time = 19 := by
  sorry

end total_travel_time_is_19_hours_l21_21429


namespace rationalize_sqrt_l21_21936

theorem rationalize_sqrt (h : Real.sqrt 35 ≠ 0) : 35 / Real.sqrt 35 = Real.sqrt 35 := 
by 
sorry

end rationalize_sqrt_l21_21936


namespace union_A_B_l21_21219

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | x^2 - 1 < 0}
def A_union_B := {x : ℝ | (Real.log x ≤ 0) ∨ (x^2 - 1 < 0)}

theorem union_A_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- proof to be added
  sorry

end union_A_B_l21_21219


namespace evaluate_special_operation_l21_21497

-- Define the operation @
def special_operation (a b : ℕ) : ℚ := (a * b) / (a - b)

-- State the theorem
theorem evaluate_special_operation : special_operation 6 3 = 6 := by
  sorry

end evaluate_special_operation_l21_21497


namespace minimum_value_of_reciprocals_l21_21025

theorem minimum_value_of_reciprocals {m n : ℝ} 
  (hmn : m > 0 ∧ n > 0 ∧ (m * n > 0)) 
  (hline : 2 * m + 2 * n = 1) : 
  (1 / m + 1 / n) = 8 :=
sorry

end minimum_value_of_reciprocals_l21_21025


namespace prime_triple_l21_21586

theorem prime_triple (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (p * r - 1)) (h3 : r ∣ (p * q - 1)) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
sorry

end prime_triple_l21_21586


namespace greatest_value_l21_21541

theorem greatest_value (y : ℝ) (h : 4 * y^2 + 4 * y + 3 = 1) : (y + 1)^2 = 1/4 :=
sorry

end greatest_value_l21_21541


namespace allison_upload_rate_l21_21851

theorem allison_upload_rate (x : ℕ) (h1 : 15 * x + 30 * x = 450) : x = 10 :=
by
  sorry

end allison_upload_rate_l21_21851


namespace tangent_line_equation_l21_21654

open Set Filter

-- Define the function representing the curve y = x^2
def curve (x : ℝ) : ℝ := x^2

-- Define the point of tangency (1, 1)
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Define the tangent line equation in standard form 2x - y - 1 = 0
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- The statement of our theorem: proving that the tangent line to the curve at the
-- point (1,1) has the equation 2x - y - 1 = 0
theorem tangent_line_equation :
  let x0 := 1
  let y0 := curve x0
  let m := deriv curve x0
  ∀ y, tangent_line x0 y :=
by
  sorry

end tangent_line_equation_l21_21654


namespace mr_smith_payment_l21_21071

theorem mr_smith_payment {balance : ℝ} {percentage : ℝ} 
  (h_bal : balance = 150) (h_percent : percentage = 0.02) :
  (balance + balance * percentage) = 153 :=
by
  sorry

end mr_smith_payment_l21_21071


namespace work_completion_days_l21_21316

-- We assume D is a certain number of days and W is some amount of work
variables (D W : ℕ)

-- Define the rate at which 3 people can do 3W work in D days
def rate_3_people : ℚ := 3 * W / D

-- Define the rate at which 5 people can do 5W work in D days
def rate_5_people : ℚ := 5 * W / D

-- The problem states that both rates must be equal
theorem work_completion_days : (3 * D) = D / 3 :=
by sorry

end work_completion_days_l21_21316


namespace gel_pen_price_relation_b_l21_21130

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l21_21130


namespace prob_correct_l21_21697

noncomputable def prob_train_there_when_sam_arrives : ℚ :=
  let total_area := (60 : ℚ) * 60
  let triangle_area := (1 / 2 : ℚ) * 15 * 15
  let parallelogram_area := (30 : ℚ) * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem prob_correct : prob_train_there_when_sam_arrives = 25 / 160 :=
  sorry

end prob_correct_l21_21697


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21148

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21148


namespace identity_eq_l21_21516

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l21_21516


namespace crazy_silly_school_diff_books_movies_l21_21664

theorem crazy_silly_school_diff_books_movies 
    (total_books : ℕ) (total_movies : ℕ)
    (hb : total_books = 36)
    (hm : total_movies = 25) :
    total_books - total_movies = 11 :=
by {
  sorry
}

end crazy_silly_school_diff_books_movies_l21_21664


namespace find_m_value_l21_21105

theorem find_m_value :
  62519 * 9999 = 625127481 :=
  by sorry

end find_m_value_l21_21105


namespace sqrt_2_minus_x_domain_l21_21900

theorem sqrt_2_minus_x_domain (x : ℝ) : (∃ y, y = sqrt (2 - x)) → x ≤ 2 := by
  sorry

end sqrt_2_minus_x_domain_l21_21900


namespace pen_price_ratio_l21_21161

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l21_21161


namespace number_of_boys_and_girls_l21_21704

theorem number_of_boys_and_girls (b g : ℕ) 
    (h1 : ∀ n : ℕ, (n ≥ 1) → ∃ (a_n : ℕ), a_n = 2 * n + 1)
    (h2 : (2 * b + 1 = g))
    : b = (g - 1) / 2 :=
by
  sorry

end number_of_boys_and_girls_l21_21704


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21140

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21140


namespace area_enclosed_by_line_and_curve_l21_21915

theorem area_enclosed_by_line_and_curve :
  ∃ area, ∀ (x : ℝ), x^2 = 4 * (x - 4/2) → 
    area = ∫ (t : ℝ) in Set.Icc (-1 : ℝ) 2, (1/4 * t + 1/2 - 1/4 * t^2) :=
sorry

end area_enclosed_by_line_and_curve_l21_21915


namespace compute_matrix_vector_l21_21629

open Matrix

-- Aliases for vector and matrix types
def Vector2 := Fin 2 → ℝ
def Matrix2 := Matrix (Fin 2) (Fin 2) ℝ

-- Given assumptions
variables (N : Matrix2)
variables (p q : Vector2)
variables (hp : N.mulVec p = ![3, -4])
variables (hq : N.mulVec q = ![-2, 6])

-- Goal
theorem compute_matrix_vector : N.mulVec (3 • p - 2 • q) = ![13, -24] :=
by 
  sorry

end compute_matrix_vector_l21_21629


namespace focus_of_given_parabola_is_correct_l21_21191

-- Define the problem conditions
def parabolic_equation (x y : ℝ) : Prop := y = 4 * x^2

-- Define what it means for a point to be the focus of the given parabola
def is_focus_of_parabola (x0 y0 : ℝ) : Prop := 
    x0 = 0 ∧ y0 = 1 / 16

-- Define the theorem to be proven
theorem focus_of_given_parabola_is_correct : 
  ∃ x0 y0, parabolic_equation x0 y0 ∧ is_focus_of_parabola x0 y0 :=
sorry

end focus_of_given_parabola_is_correct_l21_21191


namespace annie_building_time_l21_21121

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end annie_building_time_l21_21121


namespace complete_the_square_l21_21400

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end complete_the_square_l21_21400


namespace cost_per_box_l21_21858

theorem cost_per_box (trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℕ) (box_cost : ℝ) 
  (h1 : trays = 3) 
  (h2 : cookies_per_tray = 80) 
  (h3 : cookies_per_box = 60)
  (h4 : total_cost = 14) 
  (h5 : (trays * cookies_per_tray) = 240)
  (h6 : (240 / cookies_per_box : ℕ) = 4) 
  (h7 : (total_cost / 4 : ℝ) = box_cost) : 
  box_cost = 3.5 := 
by sorry

end cost_per_box_l21_21858


namespace sum_of_coordinates_of_D_l21_21356

theorem sum_of_coordinates_of_D (x y : ℝ) (h1 : (x + 6) / 2 = 2) (h2 : (y + 2) / 2 = 6) :
  x + y = 8 := 
by
  sorry

end sum_of_coordinates_of_D_l21_21356


namespace sqrt_factorial_product_squared_l21_21807

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l21_21807


namespace jim_gave_away_675_cards_l21_21059

def total_cards_gave_away
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

theorem jim_gave_away_675_cards
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  (h_brother : sets_to_brother = 15)
  (h_sister : sets_to_sister = 8)
  (h_friend : sets_to_friend = 4)
  (h_cards_per_set : cards_per_set = 25)
  : total_cards_gave_away cards_per_set sets_to_brother sets_to_sister sets_to_friend = 675 :=
by
  sorry

end jim_gave_away_675_cards_l21_21059


namespace each_friend_gave_bella_2_roses_l21_21424

-- Define the given conditions
def total_roses_from_parents : ℕ := 2 * 12
def total_roses_bella_received : ℕ := 44
def number_of_dancer_friends : ℕ := 10

-- Define the mathematical goal
def roses_from_each_friend (total_roses_from_parents total_roses_bella_received number_of_dancer_friends : ℕ) : ℕ :=
  (total_roses_bella_received - total_roses_from_parents) / number_of_dancer_friends

-- Prove that each dancer friend gave Bella 2 roses
theorem each_friend_gave_bella_2_roses :
  roses_from_each_friend total_roses_from_parents total_roses_bella_received number_of_dancer_friends = 2 :=
by
  sorry

end each_friend_gave_bella_2_roses_l21_21424


namespace second_increase_is_40_l21_21372

variable (P : ℝ) (x : ℝ)

def second_increase (P : ℝ) (x : ℝ) : Prop :=
  1.30 * P * (1 + x / 100) = 1.82 * P

theorem second_increase_is_40 (P : ℝ) : ∃ x, second_increase P x ∧ x = 40 := by
  use 40
  sorry

end second_increase_is_40_l21_21372


namespace find_lesser_fraction_l21_21532

theorem find_lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 := 
by 
  sorry

end find_lesser_fraction_l21_21532


namespace complete_the_square_l21_21401

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end complete_the_square_l21_21401


namespace exists_pythagorean_triple_rational_k_l21_21923

theorem exists_pythagorean_triple_rational_k (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ), (a^2 + b^2 = c^2) ∧ ((a + c : ℚ) / b = k) := by
  sorry

end exists_pythagorean_triple_rational_k_l21_21923


namespace area_of_trapezoid_PQRS_is_147_l21_21054

variables (P Q R S T : Type) [metric_space T]
variables (area : T → T → T → ℝ)

def trapezoid (PQRS : Prop) := 
  ∃ (P Q R S: T), PQRS = (P, Q, R, S) ∧ 
  (∃ (PR_inter_QS_at_T : Prop), PR_inter_QS_at_T = (P, R, Q, S, T))

-- Conditions
axiom PQ_parallel_RS : ∀ (P Q R S : T), trapezoid (P, Q, R, S)
axiom area_pqt : area P Q T = 75
axiom area_pst : area P S T = 30

-- Question
theorem area_of_trapezoid_PQRS_is_147 :
  ∀ (P Q R S T : T), PQ_parallel_RS P Q R S → area PQRS = 147 := 
by 
  sorry  -- Proof to be filled

end area_of_trapezoid_PQRS_is_147_l21_21054


namespace find_T_l21_21764

theorem find_T (T : ℝ) : (1 / 2) * (1 / 7) * T = (1 / 3) * (1 / 5) * 90 → T = 84 :=
by sorry

end find_T_l21_21764


namespace graphs_intersection_l21_21640

theorem graphs_intersection 
  (a b c d x y : ℝ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) 
  (h1: y = ax^2 + bx + c) 
  (h2: y = ax^2 - bx + c + d) 
  : x = d / (2 * b) ∧ y = (a * d^2) / (4 * b^2) + d / 2 + c := 
sorry

end graphs_intersection_l21_21640


namespace ends_with_two_zeros_l21_21020

theorem ends_with_two_zeros (x y : ℕ) (h : (x^2 + x * y + y^2) % 10 = 0) : (x^2 + x * y + y^2) % 100 = 0 :=
sorry

end ends_with_two_zeros_l21_21020


namespace line_through_point_with_equal_intercepts_l21_21696

theorem line_through_point_with_equal_intercepts (x y : ℝ) :
  (∃ b : ℝ, 3 * x + y = 0) ∨ (∃ b : ℝ, x - y + 4 = 0) ∨ (∃ b : ℝ, x + y - 2 = 0) :=
  sorry

end line_through_point_with_equal_intercepts_l21_21696


namespace price_ratio_l21_21142

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l21_21142


namespace range_of_a_l21_21747

theorem range_of_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2)
    (hxy : x * y = 2) (h : ∀ x y, 2 - x ≥ a / (4 - y)) : a ≤ 0 :=
sorry

end range_of_a_l21_21747


namespace function_solution_l21_21434

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))

theorem function_solution (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x + 2 * f ((x - 1) / x) = 3 * x) →
  (∃ c : ℝ, ∀ x : ℝ, f x = if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))) :=
by
  intro h
  use (f 0)
  intro x
  split_ifs with h0 h1
  rotate_left -- to handle the cases x ≠ 0, 1 at first.
  sorry -- Additional proof steps required here.
  sorry -- Use the given conditions and functional equation to conclude f(0) = c.
  sorry -- Use the given conditions and functional equation to conclude f(1) = 3 - 2c.

end function_solution_l21_21434


namespace total_income_in_june_l21_21925

-- Establishing the conditions
def daily_production : ℕ := 200
def days_in_june : ℕ := 30
def price_per_gallon : ℝ := 3.55

-- Defining total milk production in June as a function of daily production and days in June
def total_milk_production_in_june : ℕ :=
  daily_production * days_in_june

-- Defining total income as a function of milk production and price per gallon
def total_income (milk_production : ℕ) (price : ℝ) : ℝ :=
  milk_production * price

-- Stating the theorem that we need to prove
theorem total_income_in_june :
  total_income total_milk_production_in_june price_per_gallon = 21300 := 
sorry

end total_income_in_june_l21_21925


namespace geometric_sequence_ratio_l21_21202

theorem geometric_sequence_ratio (a1 q : ℝ) (h : (a1 * (1 - q^3) / (1 - q)) / (a1 * (1 - q^2) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 := by
  sorry

end geometric_sequence_ratio_l21_21202


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21141

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21141


namespace kevin_marbles_l21_21937

theorem kevin_marbles (M : ℕ) (h1 : 40 * 3 = 120) (h2 : 4 * M = 320 - 120) :
  M = 50 :=
by {
  sorry
}

end kevin_marbles_l21_21937


namespace elizabeth_money_l21_21716

theorem elizabeth_money :
  (∀ (P N : ℝ), P = 5 → N = 6 → 
    (P * 1.60 + N * 2.00) = 20.00) :=
by
  sorry

end elizabeth_money_l21_21716


namespace part1_part2_part3_l21_21890

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 1 - x - a * x^2

theorem part1 (x : ℝ) : f x 0 ≥ 0 :=
sorry

theorem part2 {a : ℝ} (h : ∀ x ≥ 0, f x a ≥ 0) : a ≤ 1 / 2 :=
sorry

theorem part3 (x : ℝ) (hx : x > 0) : (Real.exp x - 1) * Real.log (x + 1) > x^2 :=
sorry

end part1_part2_part3_l21_21890


namespace travel_time_difference_is_58_minutes_l21_21502

-- Define the distances and speeds for Minnie
def minnie_uphill_distance := 15
def minnie_uphill_speed := 10
def minnie_downhill_distance := 25
def minnie_downhill_speed := 40
def minnie_flat_distance := 30
def minnie_flat_speed := 25

-- Define the distances and speeds for Penny
def penny_flat_distance := 30
def penny_flat_speed := 35
def penny_downhill_distance := 25
def penny_downhill_speed := 50
def penny_uphill_distance := 15
def penny_uphill_speed := 15

-- Calculate Minnie's total travel time in hours
def minnie_time := (minnie_uphill_distance / minnie_uphill_speed) + 
                   (minnie_downhill_distance / minnie_downhill_speed) + 
                   (minnie_flat_distance / minnie_flat_speed)

-- Calculate Penny's total travel time in hours
def penny_time := (penny_flat_distance / penny_flat_speed) + 
                  (penny_downhill_distance / penny_downhill_speed) +
                  (penny_uphill_distance / penny_uphill_speed)

-- Calculate difference in minutes
def time_difference_minutes := (minnie_time - penny_time) * 60

-- The proof statement
theorem travel_time_difference_is_58_minutes :
  time_difference_minutes = 58 := by
  sorry

end travel_time_difference_is_58_minutes_l21_21502


namespace vehicle_flow_mod_15_l21_21074

theorem vehicle_flow_mod_15
  (vehicle_length : ℝ := 5)
  (max_speed : ℕ := 100)
  (speed_interval : ℕ := 10)
  (distance_multiplier : ℕ := 10)
  (N : ℕ := 2000) :
  (N % 15) = 5 := 
sorry

end vehicle_flow_mod_15_l21_21074


namespace angle_ABG_in_regular_octagon_l21_21768

theorem angle_ABG_in_regular_octagon (N : ℕ) (hN : N = 8) (regular_octagon : RegularPolygon N) : 
  angle ABG = 22.5 :=
by
  sorry

end angle_ABG_in_regular_octagon_l21_21768


namespace remainder_polynomial_2047_l21_21285

def f (r : ℤ) : ℤ := r ^ 11 - 1

theorem remainder_polynomial_2047 : f 2 = 2047 :=
by
  sorry

end remainder_polynomial_2047_l21_21285


namespace taxi_ride_cost_l21_21989

noncomputable def fixed_cost : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def distance_traveled : ℝ := 8

theorem taxi_ride_cost :
  fixed_cost + (cost_per_mile * distance_traveled) = 4.40 := by
  sorry

end taxi_ride_cost_l21_21989


namespace remainder_of_1234567_div_123_l21_21167

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end remainder_of_1234567_div_123_l21_21167


namespace yolanda_walking_rate_correct_l21_21409

-- Definitions and conditions
def distance_XY : ℕ := 65
def bobs_walking_rate : ℕ := 7
def bobs_distance_when_met : ℕ := 35
def yolanda_start_time (t: ℕ) : ℕ := t + 1 -- Yolanda starts walking 1 hour earlier

-- Yolanda's walking rate calculation
def yolandas_walking_rate : ℕ := 5

theorem yolanda_walking_rate_correct { time_bob_walked : ℕ } 
  (h1 : distance_XY = 65)
  (h2 : bobs_walking_rate = 7)
  (h3 : bobs_distance_when_met = 35) 
  (h4 : time_bob_walked = bobs_distance_when_met / bobs_walking_rate)
  (h5 : yolanda_start_time time_bob_walked = 6) -- since bob walked 5 hours, yolanda walked 6 hours
  (h6 : distance_XY - bobs_distance_when_met = 30) :
  yolandas_walking_rate = ((distance_XY - bobs_distance_when_met) / yolanda_start_time time_bob_walked) := 
sorry

end yolanda_walking_rate_correct_l21_21409


namespace greatest_3_digit_base7_divisible_by_7_l21_21387

def base7ToDec (a b c : ℕ) : ℕ := a * 7^2 + b * 7^1 + c * 7^0

theorem greatest_3_digit_base7_divisible_by_7 :
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 7 ∧ b < 7 ∧ c < 7 ∧
  base7ToDec a b c % 7 = 0 ∧ base7ToDec a b c = 342 :=
begin
  use [6, 6, 6],
  split, { repeat { norm_num } }, -- a ≠ 0
  split, { norm_num }, -- a < 7
  split, { norm_num }, -- b < 7
  split, { norm_num }, -- c < 7
  split,
  { norm_num },
  norm_num,
end

end greatest_3_digit_base7_divisible_by_7_l21_21387


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21138

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21138


namespace ellipse_constants_sum_l21_21370

/-- Given the center of the ellipse at (h, k) = (3, -5),
    the semi-major axis a = 7,
    and the semi-minor axis b = 4,
    prove that h + k + a + b = 9. -/
theorem ellipse_constants_sum :
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  h + k + a + b = 9 :=
by
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  sorry

end ellipse_constants_sum_l21_21370


namespace sum_not_complete_residue_system_l21_21066

theorem sum_not_complete_residue_system
  (n : ℕ) (hn : Even n)
  (a b : Fin n → Fin n)
  (ha : ∀ i : Fin n, ∃ j : Fin n, a j = i)
  (hb : ∀ i : Fin n, ∃ j : Fin n, b j = i) :
  ¬ (∀ k : Fin n, ∃ i : Fin n, a i + b i = k) :=
sorry

end sum_not_complete_residue_system_l21_21066


namespace correct_statement_l21_21101

theorem correct_statement :
  (∃ (A : Prop), A = (2 * x^3 - 4 * x - 3 ≠ 3)) ∧
  (∃ (B : Prop), B = ((2 + 3) ≠ 6)) ∧
  (∃ (C : Prop), C = (-4 * x^2 * y = -4)) ∧
  (∃ (D : Prop), D = (1 = 1 ∧ 1 = 1 / 8)) →
  (C) :=
by sorry

end correct_statement_l21_21101


namespace cubics_identity_l21_21029

variable (a b c x y z : ℝ)

theorem cubics_identity (X Y Z : ℝ)
  (h1 : X = a * x + b * y + c * z)
  (h2 : Y = a * y + b * z + c * x)
  (h3 : Z = a * z + b * x + c * y) :
  X^3 + Y^3 + Z^3 - 3 * X * Y * Z = 
  (x^3 + y^3 + z^3 - 3 * x * y * z) * (a^3 + b^3 + c^3 - 3 * a * b * c) :=
sorry

end cubics_identity_l21_21029


namespace quadratic_condition_l21_21780

theorem quadratic_condition (a b c : ℝ) : (a ≠ 0) ↔ ∃ (x : ℝ), ax^2 + bx + c = 0 :=
by sorry

end quadratic_condition_l21_21780


namespace min_sum_of_factors_of_2310_l21_21955

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l21_21955


namespace odometer_trip_l21_21288

variables (d e f : ℕ) (x : ℕ)

-- Define the conditions
def start_odometer (d e f : ℕ) : ℕ := 100 * d + 10 * e + f
def end_odometer (d e f : ℕ) : ℕ := 100 * f + 10 * e + d
def distance_travelled (x : ℕ) : ℕ := 65 * x
def valid_trip (d e f x : ℕ) : Prop := 
  d ≥ 1 ∧ d + e + f ≤ 9 ∧ 
  end_odometer d e f - start_odometer d e f = distance_travelled x

-- The final statement to prove
theorem odometer_trip (h : valid_trip d e f x) : d^2 + e^2 + f^2 = 41 := 
sorry

end odometer_trip_l21_21288


namespace sqrt_factorial_product_squared_l21_21817

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l21_21817


namespace pen_price_ratio_l21_21164

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l21_21164


namespace max_abs_c_l21_21493

theorem max_abs_c (a b c d e : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -1 ≤ a * x^4 + b * x^3 + c * x^2 + d * x + e ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e ≤ 1) : abs c ≤ 8 :=
by {
  sorry
}

end max_abs_c_l21_21493


namespace factorization_example_l21_21404

theorem factorization_example : 
  ∀ (a : ℝ), a^2 - 6 * a + 9 = (a - 3)^2 :=
by
  intro a
  sorry

end factorization_example_l21_21404


namespace smallest_four_digit_palindrome_divisible_by_8_l21_21542

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem smallest_four_digit_palindrome_divisible_by_8 : ∃ (n : ℕ), is_palindrome n ∧ is_four_digit n ∧ divisible_by_8 n ∧ n = 4004 := by
  sorry

end smallest_four_digit_palindrome_divisible_by_8_l21_21542


namespace Sarah_consumed_one_sixth_l21_21362

theorem Sarah_consumed_one_sixth (total_slices : ℕ) (slices_sarah_ate : ℕ) (shared_slices : ℕ) :
  total_slices = 20 → slices_sarah_ate = 3 → shared_slices = 1 → 
  ((slices_sarah_ate + shared_slices / 3) / total_slices : ℚ) = 1 / 6 :=
by
  intros h1 h2 h3
  sorry

end Sarah_consumed_one_sixth_l21_21362


namespace circumcircle_eq_l21_21598

-- Definitions and conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P : (ℝ × ℝ) := (4, 2)
def is_tangent_point (x y : ℝ) : Prop := sorry -- You need a proper definition for tangency

theorem circumcircle_eq :
  ∃ (hA : is_tangent_point 0 2) (hB : ∃ x y, is_tangent_point x y),
  ∃ (x y : ℝ), (circle_eq 0 2 ∧ circle_eq x y) ∧ (x-2)^2 + (y-1)^2 = 5 :=
  sorry

end circumcircle_eq_l21_21598


namespace jack_sugar_l21_21482

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l21_21482


namespace externally_tangent_intersect_two_points_l21_21300

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def circle2 (x y r : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = r^2 ∧ r > 0

theorem externally_tangent (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) →
  (∃ x y : ℝ, circle1 x y) → 
  (dist (1, 1) (4, 5) = r + 1) → 
  r = 4 := 
sorry

theorem intersect_two_points (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) → 
  (∃ x y : ℝ, circle1 x y) → 
  (|r - 1| < dist (1, 1) (4, 5) ∧ dist (1, 1) (4, 5) < r + 1) → 
  4 < r ∧ r < 6 :=
sorry

end externally_tangent_intersect_two_points_l21_21300


namespace part_one_solution_set_part_two_lower_bound_l21_21741

def f (x a b : ℝ) : ℝ := abs (x - a) + abs (x + b)

-- Part (I)
theorem part_one_solution_set (a b x : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (f x a b ≤ 5) ↔ -3 ≤ x ∧ x ≤ 2 := by
  rw [h1, h2]
  sorry

-- Part (II)
theorem part_two_lower_bound (a b x : ℝ) (h : a > 0) (h' : b > 0) (h'' : a + 4 * b = 2 * a * b) :
  f x a b ≥ 9 / 2 := by
  sorry

end part_one_solution_set_part_two_lower_bound_l21_21741


namespace thirty_percent_of_forty_percent_of_x_l21_21317

theorem thirty_percent_of_forty_percent_of_x (x : ℝ) (h : 0.12 * x = 24) : 0.30 * 0.40 * x = 24 :=
sorry

end thirty_percent_of_forty_percent_of_x_l21_21317


namespace determine_k_l21_21305

variable (x y z k : ℝ)

theorem determine_k (h1 : 7 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := 
by 
  sorry

end determine_k_l21_21305


namespace simple_and_compound_interest_difference_l21_21396

theorem simple_and_compound_interest_difference (r : ℝ) :
  let P := 3600
  let t := 2
  let SI := P * r * t / 100
  let CI := P * (1 + r / 100)^t - P
  CI - SI = 225 → r = 25 := by
  intros
  sorry

end simple_and_compound_interest_difference_l21_21396


namespace holly_pills_per_week_l21_21310

theorem holly_pills_per_week 
  (insulin_pills_per_day : ℕ)
  (blood_pressure_pills_per_day : ℕ)
  (anticonvulsants_per_day : ℕ)
  (H1 : insulin_pills_per_day = 2)
  (H2 : blood_pressure_pills_per_day = 3)
  (H3 : anticonvulsants_per_day = 2 * blood_pressure_pills_per_day) :
  (insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsants_per_day) * 7 = 77 := 
by
  sorry

end holly_pills_per_week_l21_21310


namespace problem_l21_21884

theorem problem (a : ℝ) (h : a^2 - 2 * a - 2 = 0) :
  (1 - 1 / (a + 1)) / (a^3 / (a^2 + 2 * a + 1)) = 1 / 2 :=
by
  sorry

end problem_l21_21884


namespace height_relationship_l21_21856

theorem height_relationship (B V G : ℝ) (h1 : B = 2 * V) (h2 : V = (2 / 3) * G) : B = (4 / 3) * G :=
sorry

end height_relationship_l21_21856


namespace convert_speed_kmph_to_mps_l21_21177

def kilometers_to_meters := 1000
def hours_to_seconds := 3600
def speed_kmph := 18
def expected_speed_mps := 5

theorem convert_speed_kmph_to_mps :
  speed_kmph * (kilometers_to_meters / hours_to_seconds) = expected_speed_mps :=
by
  sorry

end convert_speed_kmph_to_mps_l21_21177


namespace find_n_l21_21994

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem find_n 
  (n : ℕ)
  (h1 : (binom (n-6) 7) / binom n 7 = (6 * binom (n-7) 6) / binom n 7)
  : n = 48 := by
  sorry

end find_n_l21_21994


namespace smaller_factor_of_4851_l21_21374

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end smaller_factor_of_4851_l21_21374


namespace bill_toilet_paper_duration_l21_21573

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end bill_toilet_paper_duration_l21_21573


namespace distinct_primes_sum_reciprocal_l21_21712

open Classical

theorem distinct_primes_sum_reciprocal (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (hineq: (1 / p : ℚ) + (1 / q) + (1 / r) ≥ 1) 
  : (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨
    (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) := 
sorry

end distinct_primes_sum_reciprocal_l21_21712


namespace stock_price_return_to_initial_l21_21053

variable (P₀ : ℝ) -- Initial price
variable (y : ℝ) -- Percentage increase during the fourth week

/-- The main theorem stating the required percentage increase in the fourth week -/
theorem stock_price_return_to_initial
  (h1 : P₀ * 1.30 * 0.75 * 1.20 = 117) -- Condition after three weeks
  (h2 : P₃ = P₀) : -- Price returns to initial
  y = -15 := 
by
  sorry

end stock_price_return_to_initial_l21_21053


namespace average_charge_per_person_l21_21107

-- Define the given conditions
def charge_first_day : ℝ := 15
def charge_second_day : ℝ := 7.5
def charge_third_day : ℝ := 2.5

def attendance_ratio_first_day : ℕ := 2
def attendance_ratio_second_day : ℕ := 5
def attendance_ratio_third_day : ℕ := 13

-- Average charge per person statement
theorem average_charge_per_person (x : ℝ) :
  let visitors_first_day := attendance_ratio_first_day * x,
      visitors_second_day := attendance_ratio_second_day * x,
      visitors_third_day := attendance_ratio_third_day * x,
      total_revenue := (visitors_first_day * charge_first_day) +
                       (visitors_second_day * charge_second_day) +
                       (visitors_third_day * charge_third_day),
      total_visitors := visitors_first_day + visitors_second_day + visitors_third_day in
  (total_revenue / total_visitors) = 5 := 
by
  -- proof goes here
  sorry

end average_charge_per_person_l21_21107


namespace pentagon_arithmetic_progression_angle_l21_21089

theorem pentagon_arithmetic_progression_angle (a n : ℝ) 
  (h1 : a + (a + n) + (a + 2 * n) + (a + 3 * n) + (a + 4 * n) = 540) :
  a + 2 * n = 108 :=
by
  sorry

end pentagon_arithmetic_progression_angle_l21_21089


namespace r_squared_plus_s_squared_l21_21067

theorem r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end r_squared_plus_s_squared_l21_21067


namespace quadratic_one_real_root_l21_21197

theorem quadratic_one_real_root (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 2 * x + 1 = 0) ↔ ((a = 0) ∨ (a = 1))) :=
sorry

end quadratic_one_real_root_l21_21197


namespace prove_length_square_qp_l21_21750

noncomputable def length_square_qp (r1 r2 d : ℝ) (x : ℝ) : Prop :=
  r1 = 10 ∧ r2 = 8 ∧ d = 15 ∧ (2*r1*x - (x^2 + r2^2 - d^2) = 0) → x^2 = 164

theorem prove_length_square_qp : length_square_qp 10 8 15 x :=
sorry

end prove_length_square_qp_l21_21750


namespace intersection_empty_implies_range_l21_21023

-- Define the sets A and B
def setA := {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Prove that if A ∩ B = ∅, then 1 < a < 2
theorem intersection_empty_implies_range (a : ℝ) (h : setA ∩ setB a = ∅) : 1 < a ∧ a < 2 :=
by
  sorry

end intersection_empty_implies_range_l21_21023


namespace fraction_of_pizza_covered_by_pepperoni_l21_21837

theorem fraction_of_pizza_covered_by_pepperoni :
  ∀ (d_pizza d_pepperoni : ℝ) (n_pepperoni : ℕ) (overlap_fraction : ℝ),
  d_pizza = 16 ∧ d_pepperoni = d_pizza / 8 ∧ n_pepperoni = 32 ∧ overlap_fraction = 0.25 →
  (π * d_pepperoni^2 / 4 * (1 - overlap_fraction) * n_pepperoni) / (π * (d_pizza / 2)^2) = 3 / 8 :=
by
  intro d_pizza d_pepperoni n_pepperoni overlap_fraction
  intro h
  sorry

end fraction_of_pizza_covered_by_pepperoni_l21_21837


namespace jack_sugar_final_l21_21477

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l21_21477


namespace part_I_part_II_l21_21086

-- Part (I)
theorem part_I (x a : ℝ) (h_a : a = 3) (h : abs (x - a) + abs (x + 5) ≥ 2 * abs (x + 5)) : x ≤ -1 := 
sorry

-- Part (II)
theorem part_II (a : ℝ) (h : ∀ x : ℝ, abs (x - a) + abs (x + 5) ≥ 6) : a ≥ 1 ∨ a ≤ -11 := 
sorry

end part_I_part_II_l21_21086


namespace average_stickers_per_pack_l21_21938

-- Define the conditions given in the problem
def pack1 := 5
def pack2 := 7
def pack3 := 7
def pack4 := 10
def pack5 := 11
def num_packs := 5
def total_stickers := pack1 + pack2 + pack3 + pack4 + pack5

-- Statement to prove the average number of stickers per pack
theorem average_stickers_per_pack :
  (total_stickers / num_packs) = 8 := by
  sorry

end average_stickers_per_pack_l21_21938


namespace holly_pills_per_week_l21_21309

theorem holly_pills_per_week 
  (insulin_pills_per_day : ℕ)
  (blood_pressure_pills_per_day : ℕ)
  (anticonvulsants_per_day : ℕ)
  (H1 : insulin_pills_per_day = 2)
  (H2 : blood_pressure_pills_per_day = 3)
  (H3 : anticonvulsants_per_day = 2 * blood_pressure_pills_per_day) :
  (insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsants_per_day) * 7 = 77 := 
by
  sorry

end holly_pills_per_week_l21_21309


namespace greatest_radius_of_circle_area_lt_90pi_l21_21196

theorem greatest_radius_of_circle_area_lt_90pi : ∃ (r : ℤ), (∀ (r' : ℤ), (π * (r':ℝ)^2 < 90 * π ↔ (r' ≤ r))) ∧ (π * (r:ℝ)^2 < 90 * π) ∧ (r = 9) :=
sorry

end greatest_radius_of_circle_area_lt_90pi_l21_21196


namespace cost_of_one_dozen_pens_l21_21943

noncomputable def cost_of_one_pen_and_one_pencil_ratio := 5

theorem cost_of_one_dozen_pens
  (cost_pencil : ℝ)
  (cost_3_pens_5_pencils : 3 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) + 5 * cost_pencil = 200) :
  12 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l21_21943


namespace ribbons_count_l21_21051

theorem ribbons_count (ribbons : ℕ) 
  (yellow_frac purple_frac orange_frac : ℚ)
  (black_ribbons : ℕ)
  (h1 : yellow_frac = 1/4)
  (h2 : purple_frac = 1/3)
  (h3 : orange_frac = 1/6)
  (h4 : ribbons - (yellow_frac * ribbons + purple_frac * ribbons + orange_frac * ribbons) = black_ribbons) :
  ribbons * orange_frac = 160 / 6 :=
by {
  sorry
}

end ribbons_count_l21_21051


namespace back_seat_can_hold_8_people_l21_21047

def totalPeopleOnSides : ℕ :=
  let left_seats := 15
  let right_seats := left_seats - 3
  let people_per_seat := 3
  (left_seats + right_seats) * people_per_seat

def bus_total_capacity : ℕ := 89

def back_seat_capacity : ℕ :=
  bus_total_capacity - totalPeopleOnSides

theorem back_seat_can_hold_8_people : back_seat_capacity = 8 := by
  sorry

end back_seat_can_hold_8_people_l21_21047


namespace find_number_l21_21395

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end find_number_l21_21395


namespace jill_sod_area_needed_l21_21339

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end jill_sod_area_needed_l21_21339


namespace part1_part2_l21_21180

noncomputable section
def g1 (x : ℝ) : ℝ := Real.log x

noncomputable def f (t : ℝ) : ℝ := 
  if g1 t = t then 1 else sorry  -- Assuming g1(x) = t has exactly one root.

theorem part1 (t : ℝ) : f t = 1 :=
by sorry

def g2 (x : ℝ) (a : ℝ) : ℝ := 
  if x ≤ 0 then x else -x^2 + 2*a*x + a

theorem part2 (a : ℝ) (h : ∃ t : ℝ, f (t + 2) > f t) : a > 1 :=
by sorry

end part1_part2_l21_21180


namespace midpoint_product_l21_21919

theorem midpoint_product (x y z : ℤ) 
  (h1 : (2 + x) / 2 = 4) 
  (h2 : (10 + y) / 2 = 6) 
  (h3 : (5 + z) / 2 = 3) : 
  x * y * z = 12 := 
by
  sorry

end midpoint_product_l21_21919


namespace rebecca_has_22_eggs_l21_21077

-- Define the conditions
def number_of_groups : ℕ := 11
def eggs_per_group : ℕ := 2

-- Define the total number of eggs calculated from the conditions.
def total_eggs : ℕ := number_of_groups * eggs_per_group

-- State the theorem and provide the proof outline.
theorem rebecca_has_22_eggs : total_eggs = 22 := by {
  -- Proof will go here, but for now we put sorry to indicate it is not yet provided.
  sorry
}

end rebecca_has_22_eggs_l21_21077


namespace exists_seq_two_reals_l21_21287

theorem exists_seq_two_reals (x y : ℝ) (a : ℕ → ℝ) (h_recur : ∀ n, a (n + 2) = x * a (n + 1) + y * a n) :
  (∀ r > 0, ∃ i j : ℕ, 0 < |a i| ∧ |a i| < r ∧ r < |a j|) → ∃ x y : ℝ, ∃ a : ℕ → ℝ, (∀ n, a (n + 2) = x * a (n + 1) + y * a n) :=
by
  sorry

end exists_seq_two_reals_l21_21287


namespace intersection_is_correct_l21_21301

-- Conditions definitions
def setA : Set ℝ := {x | 2 < x ∧ x < 8}
def setB : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Intersection definition
def intersection : Set ℝ := {x | 2 < x ∧ x ≤ 6}

-- Theorem statement
theorem intersection_is_correct : setA ∩ setB = intersection := 
by
  sorry

end intersection_is_correct_l21_21301


namespace factorial_expression_value_l21_21813

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l21_21813


namespace positive_difference_between_loans_l21_21757

noncomputable def loan_amount : ℝ := 12000

noncomputable def option1_interest_rate : ℝ := 0.08
noncomputable def option1_years_1 : ℕ := 3
noncomputable def option1_years_2 : ℕ := 9

noncomputable def option2_interest_rate : ℝ := 0.09
noncomputable def option2_years : ℕ := 12

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate)^years

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal + principal * rate * years

noncomputable def payment_at_year_3 : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 / 3

noncomputable def remaining_balance_after_3_years : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 - payment_at_year_3

noncomputable def total_payment_option1 : ℝ :=
  payment_at_year_3 + compound_interest remaining_balance_after_3_years option1_interest_rate option1_years_2

noncomputable def total_payment_option2 : ℝ :=
  simple_interest loan_amount option2_interest_rate option2_years

noncomputable def positive_difference : ℝ :=
  abs (total_payment_option1 - total_payment_option2)

theorem positive_difference_between_loans : positive_difference = 1731 := by
  sorry

end positive_difference_between_loans_l21_21757


namespace total_female_students_l21_21276

def total_students : ℕ := 1600
def sample_size : ℕ := 200
def fewer_girls : ℕ := 10

theorem total_female_students (x : ℕ) (sampled_girls sampled_boys : ℕ) (h_total_sample : sampled_girls + sampled_boys = sample_size)
                             (h_fewer_girls : sampled_girls + fewer_girls = sampled_boys) :
  sampled_girls * 8 = 760 :=
by
  sorry

end total_female_students_l21_21276


namespace replace_stars_with_identity_l21_21510

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l21_21510


namespace perimeter_triangle_formed_by_parallel_lines_l21_21538

-- Defining the side lengths of the triangle ABC
def AB := 150
def BC := 270
def AC := 210

-- Defining the lengths of the segments formed by intersections with lines parallel to the sides of ABC
def length_lA := 65
def length_lB := 60
def length_lC := 20

-- The perimeter of the triangle formed by the intersection of the lines
theorem perimeter_triangle_formed_by_parallel_lines :
  let perimeter : ℝ := 5.71 + 20 + 83.33 + 65 + 91 + 60 + 5.71
  perimeter = 330.75 := by
  sorry

end perimeter_triangle_formed_by_parallel_lines_l21_21538


namespace cricket_innings_count_l21_21983

theorem cricket_innings_count (n : ℕ) (h_avg_current : ∀ (total_runs : ℕ), total_runs = 32 * n)
  (h_runs_needed : ∀ (total_runs : ℕ), total_runs + 116 = 36 * (n + 1)) : n = 20 :=
by
  sorry

end cricket_innings_count_l21_21983


namespace count_diff_squares_l21_21607

/-- 
  The number of integers between 1 and 1500 that can be expressed as the difference
  of the squares of two positive integers is 1125.
-/
theorem count_diff_squares (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1500) :
  (finset.filter (λ k, ∃ a b : ℕ, k = (a+1)^2 - a^2 ∨ k = (b+1)^2 - (b-1)^2) (finset.range 1501)).card = 1125 :=
sorry

end count_diff_squares_l21_21607


namespace boxes_count_l21_21578

theorem boxes_count (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) : (total_notebooks / notebooks_per_box) = 3 :=
by
  sorry

end boxes_count_l21_21578


namespace smallest_possible_N_l21_21920

theorem smallest_possible_N (p q r s t : ℕ) (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0)
(h_sum : p + q + r + s + t = 2022) :
    ∃ N : ℕ, N = 506 ∧ N = max (p + q) (max (q + r) (max (r + s) (s + t))) :=
by
    sorry

end smallest_possible_N_l21_21920


namespace min_sum_of_factors_of_2310_l21_21957

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l21_21957


namespace sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l21_21967

def original_price : ℝ := 150
def discount_monday_to_wednesday : ℝ := 0.20
def tax_monday_to_wednesday : ℝ := 0.05
def discount_thursday_to_saturday : ℝ := 0.15
def tax_thursday_to_saturday : ℝ := 0.04
def discount_super_saver_sunday1 : ℝ := 0.25
def discount_super_saver_sunday2 : ℝ := 0.10
def tax_super_saver_sunday : ℝ := 0.03
def discount_festive_friday : ℝ := 0.20
def tax_festive_friday : ℝ := 0.04
def additional_discount_festive_friday : ℝ := 0.05

theorem sale_price_monday_to_wednesday : (original_price * (1 - discount_monday_to_wednesday)) * (1 + tax_monday_to_wednesday) = 126 :=
by sorry

theorem sale_price_thursday_to_saturday : (original_price * (1 - discount_thursday_to_saturday)) * (1 + tax_thursday_to_saturday) = 132.60 :=
by sorry

theorem sale_price_super_saver_sunday : ((original_price * (1 - discount_super_saver_sunday1)) * (1 - discount_super_saver_sunday2)) * (1 + tax_super_saver_sunday) = 104.29 :=
by sorry

theorem sale_price_festive_friday_selected : ((original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday)) * (1 - additional_discount_festive_friday) = 118.56 :=
by sorry

theorem sale_price_festive_friday_non_selected : (original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday) = 124.80 :=
by sorry

end sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l21_21967


namespace product_of_fractions_l21_21390

theorem product_of_fractions :
  (2 / 3) * (5 / 8) * (1 / 4) = 5 / 48 := by
  sorry

end product_of_fractions_l21_21390


namespace raking_yard_time_l21_21206

theorem raking_yard_time (your_rate : ℚ) (brother_rate : ℚ) (combined_rate : ℚ) (combined_time : ℚ) :
  your_rate = 1 / 30 ∧ 
  brother_rate = 1 / 45 ∧ 
  combined_rate = your_rate + brother_rate ∧ 
  combined_time = 1 / combined_rate → 
  combined_time = 18 := 
by 
  sorry

end raking_yard_time_l21_21206


namespace infinite_rationals_sqrt_rational_l21_21939

theorem infinite_rationals_sqrt_rational : ∃ᶠ x : ℚ in Filter.atTop, ∃ y : ℚ, y = Real.sqrt (x^2 + x + 1) :=
sorry

end infinite_rationals_sqrt_rational_l21_21939


namespace students_met_goal_l21_21417

def money_needed_per_student : ℕ := 450
def number_of_students : ℕ := 6
def collective_expenses : ℕ := 3000
def amount_raised_day1 : ℕ := 600
def amount_raised_day2 : ℕ := 900
def amount_raised_day3 : ℕ := 400
def days_remaining : ℕ := 4
def half_of_first_three_days : ℕ :=
  (amount_raised_day1 + amount_raised_day2 + amount_raised_day3) / 2

def total_needed : ℕ :=
  money_needed_per_student * number_of_students + collective_expenses
def total_raised : ℕ :=
  amount_raised_day1 + amount_raised_day2 + amount_raised_day3 + (half_of_first_three_days * days_remaining)

theorem students_met_goal : total_raised >= total_needed := by
  sorry

end students_met_goal_l21_21417


namespace find_f_of_functions_l21_21314

theorem find_f_of_functions
  (f g : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = - f x)
  (h_even : ∀ x, g (-x) = g x)
  (h_eq : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := 
sorry

end find_f_of_functions_l21_21314


namespace find_first_number_of_sequence_l21_21277

theorem find_first_number_of_sequence
    (a : ℕ → ℕ)
    (h1 : ∀ n, 3 ≤ n → a n = a (n-1) * a (n-2))
    (h2 : a 8 = 36)
    (h3 : a 9 = 1296)
    (h4 : a 10 = 46656) :
    a 1 = 60466176 := 
sorry

end find_first_number_of_sequence_l21_21277


namespace find_extrema_of_f_l21_21450

noncomputable def f (x : ℝ) := x^2 - 4 * x - 2

theorem find_extrema_of_f : 
  (∀ x, (1 ≤ x ∧ x ≤ 4) → f x ≤ -2) ∧ 
  (∃ x, (1 ≤ x ∧ x ≤ 4 ∧ f x = -6)) :=
by sorry

end find_extrema_of_f_l21_21450


namespace how_long_to_grow_more_l21_21355

def current_length : ℕ := 14
def length_to_donate : ℕ := 23
def desired_length_after_donation : ℕ := 12

theorem how_long_to_grow_more : 
  (desired_length_after_donation + length_to_donate - current_length) = 21 := 
by
  -- Leave the proof part for later
  sorry

end how_long_to_grow_more_l21_21355


namespace paul_initial_books_l21_21932

theorem paul_initial_books (sold_books : ℕ) (left_books : ℕ) (initial_books : ℕ) 
  (h_sold_books : sold_books = 109)
  (h_left_books : left_books = 27)
  (h_initial_books_formula : initial_books = sold_books + left_books) : 
  initial_books = 136 :=
by
  rw [h_sold_books, h_left_books] at h_initial_books_formula
  exact h_initial_books_formula

end paul_initial_books_l21_21932


namespace find_point_C_coordinates_l21_21662

/-- Given vertices A and B of a triangle, and the centroid G of the triangle, 
prove the coordinates of the third vertex C. 
-/
theorem find_point_C_coordinates : 
  ∀ (x y : ℝ),
  let A := (2, 3)
  let B := (-4, -2)
  let G := (2, -1)
  (2 + -4 + x) / 3 = 2 →
  (3 + -2 + y) / 3 = -1 →
  (x, y) = (8, -4) :=
by
  intro x y A B G h1 h2
  sorry

end find_point_C_coordinates_l21_21662


namespace find_A_l21_21523

noncomputable def A_value (A B C : ℝ) := (A = 1/4) 

theorem find_A : 
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / (x - 3)^2)) →
  A_value A B C :=
by 
  sorry

end find_A_l21_21523


namespace determine_xyz_l21_21577

theorem determine_xyz (x y z : ℝ) 
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 12) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) : 
  x * y * z = -4 / 3 := 
sorry

end determine_xyz_l21_21577


namespace intersection_complement_l21_21604

open Set

def U : Set ℤ := univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_complement :
  P ∩ (U \ M) = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l21_21604


namespace multiply_expression_l21_21226

variable {x : ℝ}

theorem multiply_expression :
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 :=
by
  sorry

end multiply_expression_l21_21226


namespace mixed_oil_rate_per_litre_l21_21548

variables (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ)

def total_cost (v p : ℝ) : ℝ := v * p
def total_volume (v1 v2 : ℝ) : ℝ := v1 + v2

theorem mixed_oil_rate_per_litre (h1 : volume1 = 10) (h2 : price1 = 55) (h3 : volume2 = 5) (h4 : price2 = 66) :
  (total_cost volume1 price1 + total_cost volume2 price2) / total_volume volume1 volume2 = 58.67 := 
by
  sorry

end mixed_oil_rate_per_litre_l21_21548


namespace shelves_needed_l21_21231

variable (total_books : Nat) (books_taken : Nat) (books_per_shelf : Nat)

theorem shelves_needed (h1 : total_books = 14) 
                       (h2 : books_taken = 2) 
                       (h3 : books_per_shelf = 3) : 
    (total_books - books_taken) / books_per_shelf = 4 := by
  sorry

end shelves_needed_l21_21231


namespace quadratic_inequality_solution_l21_21723

noncomputable def solve_inequality (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x > -1/2 ∧ x < 1/3) → (a * x^2 + b * x + 2 > 0)) →
  (a = -12) ∧ (b = -2)

theorem quadratic_inequality_solution :
   solve_inequality (-12) (-2) :=
by
  intro h
  sorry

end quadratic_inequality_solution_l21_21723


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21149

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21149


namespace portion_left_l21_21756

theorem portion_left (john_portion emma_portion final_portion : ℝ) (H1 : john_portion = 0.6) (H2 : emma_portion = 0.5 * (1 - john_portion)) :
  final_portion = 1 - john_portion - emma_portion :=
by
  sorry

end portion_left_l21_21756


namespace roses_ordered_l21_21207

theorem roses_ordered (tulips carnations roses : ℕ) (cost_per_flower total_expenses : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : cost_per_flower = 2)
  (h4 : total_expenses = 1890)
  (h5 : total_expenses = (tulips + carnations + roses) * cost_per_flower) :
  roses = 320 :=
by 
  -- Using the mathematical equivalence and conditions provided
  sorry

end roses_ordered_l21_21207


namespace frustum_surface_area_l21_21689

noncomputable def total_surface_area_of_frustum
  (R r h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (R - r)^2)
  let A_lateral := Real.pi * (R + r) * s
  let A_top := Real.pi * r^2
  let A_bottom := Real.pi * R^2
  A_lateral + A_top + A_bottom

theorem frustum_surface_area :
  total_surface_area_of_frustum 8 2 5 = 10 * Real.pi * Real.sqrt 61 + 68 * Real.pi :=
  sorry

end frustum_surface_area_l21_21689


namespace gel_pen_price_relation_b_l21_21132

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l21_21132


namespace hyperbola_same_foci_as_ellipse_eccentricity_two_l21_21441

theorem hyperbola_same_foci_as_ellipse_eccentricity_two
  (a b c e : ℝ)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = 5 ∧ b = 3 ∧ c = 4))
  (eccentricity_eq : e = 2) :
  ∃ x y : ℝ, (x^2 / (c / e)^2 - y^2 / (c^2 - (c / e)^2) = 1) ↔ (x^2 / 4 - y^2 / 12 = 1) :=
by
  sorry

end hyperbola_same_foci_as_ellipse_eccentricity_two_l21_21441


namespace expected_surnames_not_repositioned_l21_21850

theorem expected_surnames_not_repositioned (n : ℕ) :
  (∑ k in Finset.range n, (1 / (k + 1) : ℝ)) = 
  (1 + ∑ i in Finset.range (n-1), (1 / (i + 2) : ℝ)) :=
sorry

end expected_surnames_not_repositioned_l21_21850


namespace power_function_value_l21_21451

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_value (α : ℝ) (h : 2 ^ α = (Real.sqrt 2) / 2) : f 4 α = 1 / 2 := 
by 
  sorry

end power_function_value_l21_21451


namespace triangular_difference_30_28_l21_21830

noncomputable def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_difference_30_28 : triangular 30 - triangular 28 = 59 :=
by
  sorry

end triangular_difference_30_28_l21_21830


namespace Murtha_pebble_collection_l21_21227

def sum_of_first_n_natural_numbers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem Murtha_pebble_collection : sum_of_first_n_natural_numbers 20 = 210 := by
  sorry

end Murtha_pebble_collection_l21_21227


namespace current_speed_l21_21379

-- Define the constants based on conditions
def rowing_speed_kmph : Float := 24
def distance_meters : Float := 40
def time_seconds : Float := 4.499640028797696

-- Intermediate calculation: Convert rowing speed from km/h to m/s
def rowing_speed_mps : Float := rowing_speed_kmph * 1000 / 3600

-- Calculate downstream speed
def downstream_speed_mps : Float := distance_meters / time_seconds

-- Define the expected speed of the current
def expected_current_speed : Float := 2.22311111

-- The theorem to prove
theorem current_speed : 
  (downstream_speed_mps - rowing_speed_mps) = expected_current_speed :=
by 
  -- skipping the proof steps, as instructed
  sorry

end current_speed_l21_21379


namespace distance_A_B_l21_21881

noncomputable def distance_3d (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_A_B :
  distance_3d 4 1 9 10 (-1) 6 = 7 :=
by
  sorry

end distance_A_B_l21_21881


namespace julio_twice_james_in_years_l21_21061

noncomputable def years_until_julio_twice_james := 
  let x := 14
  (36 + x = 2 * (11 + x))

theorem julio_twice_james_in_years : 
  years_until_julio_twice_james := 
  by 
  sorry

end julio_twice_james_in_years_l21_21061


namespace smallest_coin_remainder_l21_21269

theorem smallest_coin_remainder
  (c : ℕ)
  (h1 : c % 8 = 6)
  (h2 : c % 7 = 5)
  (h3 : ∀ d : ℕ, (d % 8 = 6) → (d % 7 = 5) → d ≥ c) :
  c % 9 = 2 :=
sorry

end smallest_coin_remainder_l21_21269


namespace largest_M_bound_l21_21882

noncomputable def largest_constant_M : ℝ :=
  2019 ^ (-(1 / 2019))

theorem largest_M_bound (b : ℕ → ℝ) :
  (∀ k, 0 ≤ k ∧ k ≤ 2019 → 1 ≤ b k) ∧
  (∀ k j, 0 ≤ k ∧ k < j ∧ j ≤ 2019 → b k < b j) →
  let z : ℕ → ℂ := λ k, (roots (∑ k in finset.range 2020, (b k) * X ^ k)).nth k in 
  (1 / 2019) * ∑ k in finset.range 2019, ∥z k∥ ≥ largest_constant_M :=
sorry

end largest_M_bound_l21_21882


namespace ice_cost_l21_21859

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end ice_cost_l21_21859


namespace min_sum_factors_l21_21951

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l21_21951


namespace smallest_number_meeting_both_conditions_l21_21602

theorem smallest_number_meeting_both_conditions :
  ∃ n, (n = 2019) ∧
    (∃ a b c d e f : ℕ,
      n = a^4 + b^4 + c^4 + d^4 + e^4 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
      c ≠ d ∧ c ≠ e ∧
      d ≠ e ∧
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (∃ x y z u v w : ℕ,
      y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
      n = x + y + z + u + v + w) ∧
    (¬ ∃ m, m < 2019 ∧
      (∃ a b c d e f : ℕ,
        m = a^4 + b^4 + c^4 + d^4 + e^4 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
        c ≠ d ∧ c ≠ e ∧
        d ≠ e ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
      (∃ x y z u v w : ℕ,
        y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
        m = x + y + z + u + v + w)) :=
by
  sorry

end smallest_number_meeting_both_conditions_l21_21602


namespace right_triangle_one_right_angle_l21_21457

theorem right_triangle_one_right_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90 ∨ B = 90 ∨ C = 90) : (interp A B C).count (90) = 1 :=
by
  sorry

end right_triangle_one_right_angle_l21_21457


namespace polynomial_remainder_l21_21065

noncomputable def h (x : ℕ) := x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℕ) : (h (x^10)) % (h x) = 5 :=
sorry

end polynomial_remainder_l21_21065


namespace steps_in_five_days_l21_21357

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end steps_in_five_days_l21_21357


namespace percentage_increase_edge_length_l21_21194

theorem percentage_increase_edge_length (a a' : ℝ) (h : 6 * (a')^2 = 6 * a^2 + 1.25 * 6 * a^2) : a' = 1.5 * a :=
by sorry

end percentage_increase_edge_length_l21_21194


namespace range_of_k_for_empty_solution_set_l21_21430

theorem range_of_k_for_empty_solution_set :
  ∀ (k : ℝ), (∀ (x : ℝ), k * x^2 - 2 * |x - 1| + 3 * k < 0 → False) ↔ k ≥ 1 :=
by sorry

end range_of_k_for_empty_solution_set_l21_21430


namespace intersecting_point_value_l21_21783

theorem intersecting_point_value (c d : ℤ) (h1 : d = 5 * (-5) + c) (h2 : -5 = 5 * d + c) : 
  d = -5 := 
sorry

end intersecting_point_value_l21_21783


namespace arithmetic_geometric_sequence_problem_l21_21753

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (q : ℚ)
  (h1 : ∀ n m : ℕ, a (n + m) = a n * (q ^ m))
  (h2 : a 2 * a 3 * a 4 = 27 / 64)
  (h3 : q = 2)
  (h4 : ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h5 : b 7 = a 5) : 
  b 3 + b 11 = 6 := 
sorry

end arithmetic_geometric_sequence_problem_l21_21753


namespace remainder_of_1234567_div_123_l21_21166

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end remainder_of_1234567_div_123_l21_21166


namespace julia_tag_kids_monday_l21_21060

-- Definitions based on conditions
def total_tag_kids (M T : ℕ) : Prop := M + T = 20
def tag_kids_Tuesday := 13

-- Problem statement
theorem julia_tag_kids_monday (M : ℕ) : total_tag_kids M tag_kids_Tuesday → M = 7 := 
by
  intro h
  sorry

end julia_tag_kids_monday_l21_21060


namespace prod_div_sum_le_square_l21_21949

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end prod_div_sum_le_square_l21_21949


namespace equation_of_parabola_max_slope_OQ_l21_21893

section parabola

variable (p : ℝ)
variable (y : ℝ) (x : ℝ)
variable (n : ℝ) (m : ℝ)

-- Condition: p > 0 and distance from focus F to directrix being 2
axiom positive_p : p > 0
axiom distance_focus_directrix : ∀ {F : ℝ}, F = 2 * p → 2 * p = 2

-- Prove these two statements
theorem equation_of_parabola : (y^2 = 4 * x) :=
  sorry

theorem max_slope_OQ : (∃ K : ℝ, K = 1 / 3) :=
  sorry

end parabola

end equation_of_parabola_max_slope_OQ_l21_21893


namespace find_some_ounce_size_l21_21863

variable (x : ℕ)
variable (h_total : 122 = 6 * 5 + 4 * x + 15 * 4)

theorem find_some_ounce_size : x = 8 := by
  sorry

end find_some_ounce_size_l21_21863


namespace gel_pen_price_relation_b_l21_21134

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l21_21134


namespace min_omega_sin_two_max_l21_21438

theorem min_omega_sin_two_max (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ k : ℤ, (ω * x = (2 + 2 * k) * π)) →
  ∃ ω_min : ℝ, ω_min = 4 * π :=
by
  sorry

end min_omega_sin_two_max_l21_21438


namespace blake_spent_on_apples_l21_21279

noncomputable def apples_spending_problem : Prop :=
  let initial_amount := 300
  let change_received := 150
  let oranges_cost := 40
  let mangoes_cost := 60
  let total_spent := initial_amount - change_received
  let other_fruits_cost := oranges_cost + mangoes_cost
  let apples_cost := total_spent - other_fruits_cost
  apples_cost = 50

theorem blake_spent_on_apples : apples_spending_problem :=
by
  sorry

end blake_spent_on_apples_l21_21279


namespace percent_decrease_in_hours_l21_21110

theorem percent_decrease_in_hours (W H : ℝ) 
  (h1 : W > 0) 
  (h2 : H > 0)
  (new_wage : ℝ := W * 1.25)
  (H_new : ℝ := H / 1.25)
  (total_income_same : W * H = new_wage * H_new) :
  ((H - H_new) / H) * 100 = 20 := 
by
  sorry

end percent_decrease_in_hours_l21_21110


namespace avg_weight_A_l21_21534

-- Define the conditions
def num_students_A : ℕ := 40
def num_students_B : ℕ := 20
def avg_weight_B : ℝ := 40
def avg_weight_whole_class : ℝ := 46.67

-- State the theorem using these definitions
theorem avg_weight_A :
  ∃ W_A : ℝ,
    (num_students_A * W_A + num_students_B * avg_weight_B = (num_students_A + num_students_B) * avg_weight_whole_class) ∧
    W_A = 50.005 :=
by
  sorry

end avg_weight_A_l21_21534


namespace smallest_h_divisible_by_primes_l21_21672

theorem smallest_h_divisible_by_primes :
  ∃ h k : ℕ, (∀ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p > 8 ∧ q > 11 ∧ r > 24 → (h + k) % (p * q * r) = 0 ∧ h = 1) :=
by
  sorry

end smallest_h_divisible_by_primes_l21_21672


namespace pencils_per_student_l21_21773

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ)
    (h1 : total_pencils = 125)
    (h2 : students = 25)
    (h3 : pencils_per_student = total_pencils / students) :
    pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l21_21773


namespace two_perfect_squares_not_two_perfect_cubes_l21_21772

-- Define the initial conditions as Lean assertions
def isSumOfTwoPerfectSquares (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^2

def isSumOfTwoPerfectCubes (n : ℕ) := ∃ a b : ℕ, n = a^3 + b^3

-- Lean 4 statement to show 2005^2005 is a sum of two perfect squares
theorem two_perfect_squares :
  isSumOfTwoPerfectSquares (2005^2005) :=
sorry

-- Lean 4 statement to show 2005^2005 is not a sum of two perfect cubes
theorem not_two_perfect_cubes :
  ¬ isSumOfTwoPerfectCubes (2005^2005) :=
sorry

end two_perfect_squares_not_two_perfect_cubes_l21_21772


namespace hexagon_inequality_l21_21758

noncomputable def ABCDEF := 3 * Real.sqrt 3 / 2
noncomputable def ACE := Real.sqrt 3
noncomputable def BDF := Real.sqrt 3
noncomputable def R₁ := Real.sqrt 3 / 4
noncomputable def R₂ := -Real.sqrt 3 / 4

theorem hexagon_inequality :
  min ACE BDF + R₂ - R₁ ≤ 3 * Real.sqrt 3 / 4 :=
by
  sorry

end hexagon_inequality_l21_21758


namespace speed_of_man_upstream_l21_21690

def speed_of_man_in_still_water : ℝ := 32
def speed_of_man_downstream : ℝ := 39

theorem speed_of_man_upstream (V_m V_s : ℝ) :
  V_m = speed_of_man_in_still_water →
  V_m + V_s = speed_of_man_downstream →
  V_m - V_s = 25 :=
sorry

end speed_of_man_upstream_l21_21690


namespace min_homework_assignments_l21_21109

variable (p1 p2 p3 : Nat)

-- Define the points and assignments
def points_first_10 : Nat := 10
def assignments_first_10 : Nat := 10 * 1

def points_second_10 : Nat := 10
def assignments_second_10 : Nat := 10 * 2

def points_third_10 : Nat := 10
def assignments_third_10 : Nat := 10 * 3

def total_points : Nat := points_first_10 + points_second_10 + points_third_10
def total_assignments : Nat := assignments_first_10 + assignments_second_10 + assignments_third_10

theorem min_homework_assignments (hp1 : points_first_10 = 10) (ha1 : assignments_first_10 = 10) 
  (hp2 : points_second_10 = 10) (ha2 : assignments_second_10 = 20)
  (hp3 : points_third_10 = 10) (ha3 : assignments_third_10 = 30)
  (tp : total_points = 30) : 
  total_assignments = 60 := 
by sorry

end min_homework_assignments_l21_21109


namespace det_E_eq_25_l21_21628

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l21_21628


namespace mul_mod_correct_l21_21391

theorem mul_mod_correct :
  (2984 * 3998) % 1000 = 32 :=
by
  sorry

end mul_mod_correct_l21_21391


namespace sitting_people_l21_21006

variables {M I P A : Prop}

-- Conditions
axiom M_not_sitting : ¬ M
axiom A_not_sitting : ¬ A
axiom if_M_not_sitting_then_I_sitting : ¬ M → I
axiom if_I_sitting_then_P_sitting : I → P

theorem sitting_people : I ∧ P :=
by
  have I_sitting : I := if_M_not_sitting_then_I_sitting M_not_sitting
  have P_sitting : P := if_I_sitting_then_P_sitting I_sitting
  exact ⟨I_sitting, P_sitting⟩

end sitting_people_l21_21006


namespace roots_cubic_eq_l21_21315

theorem roots_cubic_eq (r s p q : ℝ) (h1 : r + s = p) (h2 : r * s = q) :
    r^3 + s^3 = p^3 - 3 * q * p :=
by
    -- Placeholder for proof
    sorry

end roots_cubic_eq_l21_21315


namespace perfect_square_trinomial_m_l21_21038

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a : ℤ, x^2 + (m - 2) * x + 9 = (x + a)^2) ∨
  (∃ a : ℤ, x^2 + (m - 2) * x + 9 = (x - a)^2) ↔ (m = 8 ∨ m = -4) :=
sorry

end perfect_square_trinomial_m_l21_21038


namespace bread_consumption_l21_21096

-- Definitions using conditions
def members := 4
def slices_snacks := 2
def slices_per_loaf := 12
def total_loaves := 5
def total_days := 3

-- The main theorem to prove
theorem bread_consumption :
  (3 * members * (B + slices_snacks) = total_loaves * slices_per_loaf) → B = 3 :=
by
  intro h
  sorry

end bread_consumption_l21_21096


namespace symmetric_point_origin_l21_21331

-- Define the original point P with given coordinates
def P : ℝ × ℝ := (-2, 3)

-- Define the symmetric point P' with respect to the origin
def P'_symmetric (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The theorem states that the symmetric point of P is (2, -3)
theorem symmetric_point_origin : P'_symmetric P = (2, -3) := 
by
  sorry

end symmetric_point_origin_l21_21331


namespace part1_l21_21190

theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Ici (1/2 : ℝ), 2 * x + a / (x + 1) ≥ 0) → a ≥ -3 / 2 :=
sorry

end part1_l21_21190


namespace partI_partII_partIII_l21_21346

-- Define the basic sets and their operations
variable {α : Type*}

-- Defining set A and conditions
noncomputable def A : Set ℝ := {-1, 1}
noncomputable def Aplus (A : Set ℝ) : Set ℝ := {x | ∃ (a b : ℝ), a ∈ A ∧ b ∈ A ∧ x = a + b}
noncomputable def Aminus (A : Set ℝ) : Set ℝ := {x | ∃ (a b : ℝ), a ∈ A ∧ b ∈ A ∧ x = abs (a - b)}

-- Part (I)
theorem partI :
  A = {-1, 1} →
  Aplus A = {-2, 0, 2} ∧ 
  Aminus A = {0, 2} := 
by 
  intro h,
  sorry

-- Defining A and ordering for Part (II)
variable {x1 x2 x3 x4 : ℝ}
variable h_order : x1 < x2 ∧ x2 < x3 ∧ x3 < x4

-- Part (II)
theorem partII (A : Set ℝ) :
  A = {x1, x2, x3, x4} ∧
  Aminus A = A →
  x1 + x4 = x2 + x3 := 
by 
  intro h,
  sorry

-- Defining the universal set and conditions for Part (III)
noncomputable def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 2023}
noncomputable def Amax (A : Set ℕ) : ℕ := A.card

-- Part (III)
theorem partIII (A : Set ℕ) :
  A ⊆ U ∧ Aplus A ∩ Aminus A = ∅ →
  Amax A ≤ 1349 := 
by 
  intro h,
  sorry

end partI_partII_partIII_l21_21346


namespace train_length_is_360_l21_21564

-- Conditions from the problem
variable (speed_kmph : ℕ) (time_sec : ℕ) (platform_length_m : ℕ)

-- Definitions to be used for the conditions
def speed_ms (speed_kmph : ℕ) : ℤ := (speed_kmph * 1000) / 3600 -- Speed in m/s
def total_distance (speed_ms : ℤ) (time_sec : ℕ) : ℤ := speed_ms * (time_sec : ℤ) -- Total distance covered
def train_length (total_distance : ℤ) (platform_length : ℤ) : ℤ := total_distance - platform_length -- Length of the train

-- Assertion statement
theorem train_length_is_360 : train_length (total_distance (speed_ms speed_kmph) time_sec) platform_length_m = 360 := 
  by sorry

end train_length_is_360_l21_21564


namespace ab_greater_than_a_plus_b_l21_21039

theorem ab_greater_than_a_plus_b (a b : ℝ) (h₁ : a ≥ 2) (h₂ : b > 2) : a * b > a + b :=
  sorry

end ab_greater_than_a_plus_b_l21_21039


namespace stickers_distribution_l21_21506

-- Definitions for initial sticker quantities and stickers given to first four friends
def initial_space_stickers : ℕ := 120
def initial_cat_stickers : ℕ := 80
def initial_dinosaur_stickers : ℕ := 150
def initial_superhero_stickers : ℕ := 45

def given_space_stickers : ℕ := 25
def given_cat_stickers : ℕ := 13
def given_dinosaur_stickers : ℕ := 33
def given_superhero_stickers : ℕ := 29

-- Definitions for remaining stickers calculation
def remaining_space_stickers : ℕ := initial_space_stickers - given_space_stickers
def remaining_cat_stickers : ℕ := initial_cat_stickers - given_cat_stickers
def remaining_dinosaur_stickers : ℕ := initial_dinosaur_stickers - given_dinosaur_stickers
def remaining_superhero_stickers : ℕ := initial_superhero_stickers - given_superhero_stickers

def total_remaining_stickers : ℕ := remaining_space_stickers + remaining_cat_stickers + remaining_dinosaur_stickers + remaining_superhero_stickers

-- Definition for number of each type of new sticker
def each_new_type_stickers : ℕ := total_remaining_stickers / 4
def remainder_stickers : ℕ := total_remaining_stickers % 4

-- Statement to be proved
theorem stickers_distribution :
  ∃ X : ℕ, X = 3 ∧ each_new_type_stickers = 73 :=
by
  sorry

end stickers_distribution_l21_21506


namespace candy_store_price_per_pound_fudge_l21_21839

theorem candy_store_price_per_pound_fudge 
  (fudge_pounds : ℕ)
  (truffles_dozen : ℕ)
  (truffles_price_each : ℝ)
  (pretzels_dozen : ℕ)
  (pretzels_price_each : ℝ)
  (total_revenue : ℝ) 
  (truffles_total : ℕ := truffles_dozen * 12)
  (pretzels_total : ℕ := pretzels_dozen * 12)
  (truffles_revenue : ℝ := truffles_total * truffles_price_each)
  (pretzels_revenue : ℝ := pretzels_total * pretzels_price_each)
  (fudge_revenue : ℝ := total_revenue - (truffles_revenue + pretzels_revenue))
  (fudge_price_per_pound : ℝ := fudge_revenue / fudge_pounds) :
  fudge_pounds = 20 →
  truffles_dozen = 5 →
  truffles_price_each = 1.50 →
  pretzels_dozen = 3 →
  pretzels_price_each = 2.00 →
  total_revenue = 212 →
  fudge_price_per_pound = 2.5 :=
by 
  sorry

end candy_store_price_per_pound_fudge_l21_21839


namespace complete_the_square_l21_21399

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end complete_the_square_l21_21399


namespace students_need_to_raise_each_l21_21416

def initial_amount_needed (num_students : ℕ) (amount_per_student : ℕ) (misc_expenses : ℕ) : ℕ :=
  (num_students * amount_per_student) + misc_expenses

def amount_raised_first_three_days (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  day1 + day2 + day3

def amount_raised_next_four_days (first_three_days_total : ℕ) : ℕ :=
  first_three_days_total / 2

def total_amount_raised_in_week (first_three_days_total : ℕ) (next_four_days_total : ℕ) : ℕ :=
  first_three_days_total + next_four_days_total

def amount_each_student_still_needs_to_raise 
  (total_needed : ℕ) (total_raised : ℕ) (num_students : ℕ) : ℕ :=
  if num_students > 0 then (total_needed - total_raised) / num_students else 0

theorem students_need_to_raise_each 
  (num_students : ℕ) (amount_per_student : ℕ) (misc_expenses : ℕ)
  (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) (next_half_factor : ℕ)
  (h_num_students : num_students = 6)
  (h_amount_per_student : amount_per_student = 450)
  (h_misc_expenses : misc_expenses = 3000)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 900)
  (h_day3 : day3 = 400)
  (h_next_half_factor : next_half_factor = 2) :
  amount_each_student_still_needs_to_raise
    (initial_amount_needed num_students amount_per_student misc_expenses)
    (total_amount_raised_in_week
      (amount_raised_first_three_days day1 day2 day3)
      (amount_raised_next_four_days (amount_raised_first_three_days day1 day2 day3 / h_next_half_factor)))
    num_students = 475 :=
by sorry

end students_need_to_raise_each_l21_21416


namespace find_num_tables_l21_21352

-- Definitions based on conditions
def num_students_in_class : ℕ := 47
def num_girls_bathroom : ℕ := 3
def num_students_canteen : ℕ := 3 * 3
def num_students_new_groups : ℕ := 2 * 4
def num_students_exchange : ℕ := 3 * 3 + 3 * 3 + 3 * 3

-- Calculation of the number of tables (corresponding to the answer)
def num_missing_students : ℕ := num_girls_bathroom + num_students_canteen + num_students_new_groups + num_students_exchange

def num_students_currently_in_class : ℕ := num_students_in_class - num_missing_students
def students_per_table : ℕ := 3

def num_tables : ℕ := num_students_currently_in_class / students_per_table

-- The theorem we want to prove
theorem find_num_tables : num_tables = 6 := by
  -- Proof steps would go here
  sorry

end find_num_tables_l21_21352


namespace quadratic_eq_with_given_roots_l21_21319

theorem quadratic_eq_with_given_roots (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 12) :
    (a + b = 16) ∧ (a * b = 144) ∧ (∀ (x : ℝ), x^2 - (a + b) * x + (a * b) = 0 ↔ x^2 - 16 * x + 144 = 0) := by
  sorry

end quadratic_eq_with_given_roots_l21_21319


namespace weight_of_sparrow_l21_21910

variable (a b : ℝ)

-- Define the conditions as Lean statements
-- 1. Six sparrows and seven swallows are balanced
def balanced_initial : Prop :=
  6 * b = 7 * a

-- 2. Sparrows are heavier than swallows
def sparrows_heavier : Prop :=
  b > a

-- 3. If one sparrow and one swallow are exchanged, the balance is maintained
def balanced_after_exchange : Prop :=
  5 * b + a = 6 * a + b

-- The theorem to prove the weight of one sparrow in terms of the weight of one swallow
theorem weight_of_sparrow (h1 : balanced_initial a b) (h2 : sparrows_heavier a b) (h3 : balanced_after_exchange a b) : 
  b = (5 / 4) * a :=
sorry

end weight_of_sparrow_l21_21910


namespace sarah_copies_total_pages_l21_21520

noncomputable def total_pages_copied (people : ℕ) (pages_first : ℕ) (copies_first : ℕ) (pages_second : ℕ) (copies_second : ℕ) : ℕ :=
  (pages_first * (copies_first * people)) + (pages_second * (copies_second * people))

theorem sarah_copies_total_pages :
  total_pages_copied 20 30 3 45 2 = 3600 := by
  sorry

end sarah_copies_total_pages_l21_21520


namespace distance_between_front_contestants_l21_21199

noncomputable def position_a (pd : ℝ) : ℝ := pd - 10
def position_b (pd : ℝ) : ℝ := pd - 40
def position_c (pd : ℝ) : ℝ := pd - 60
def position_d (pd : ℝ) : ℝ := pd

theorem distance_between_front_contestants (pd : ℝ):
  position_d pd - position_a pd = 10 :=
by
  sorry

end distance_between_front_contestants_l21_21199


namespace increased_contact_area_effect_l21_21727

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l21_21727


namespace cost_of_each_soccer_ball_l21_21648

theorem cost_of_each_soccer_ball (total_amount_paid : ℕ) (change_received : ℕ) (number_of_balls : ℕ)
  (amount_spent := total_amount_paid - change_received)
  (unit_price := amount_spent / number_of_balls) :
  total_amount_paid = 100 →
  change_received = 20 →
  number_of_balls = 2 →
  unit_price = 40 := by
  sorry

end cost_of_each_soccer_ball_l21_21648


namespace total_amount_l21_21278

theorem total_amount (x y z : ℝ) (hx : y = 45 / 0.45)
  (hy : z = (45 / 0.45) * 0.30)
  (hx_total : y = 45) :
  x + y + z = 175 :=
by
  -- Proof is omitted as per instructions
  sorry

end total_amount_l21_21278


namespace polygon_side_intersections_l21_21642

theorem polygon_side_intersections :
  let m6 := 6
  let m7 := 7
  let m8 := 8
  let m9 := 9
  let pairs := [(m6, m7), (m6, m8), (m6, m9), (m7, m8), (m7, m9), (m8, m9)]
  let count_intersections (m n : ℕ) : ℕ := 2 * min m n
  let total_intersections := pairs.foldl (fun total pair => total + count_intersections pair.1 pair.2) 0
  total_intersections = 80 :=
by
  sorry

end polygon_side_intersections_l21_21642


namespace resulting_figure_perimeter_l21_21849

def original_square_side : ℕ := 100

def original_square_area : ℕ := original_square_side * original_square_side

def rect1_side1 : ℕ := original_square_side
def rect1_side2 : ℕ := original_square_side / 2

def rect2_side1 : ℕ := original_square_side
def rect2_side2 : ℕ := original_square_side / 2

def new_figure_perimeter : ℕ :=
  3 * original_square_side + 4 * (original_square_side / 2)

theorem resulting_figure_perimeter :
  new_figure_perimeter = 500 :=
by {
    sorry
}

end resulting_figure_perimeter_l21_21849


namespace rearrange_digits_to_perfect_square_l21_21361

theorem rearrange_digits_to_perfect_square :
  ∃ n : ℤ, 2601 = n ^ 2 ∧ (∃ (perm : List ℤ), perm = [2, 0, 1, 6] ∧ perm.permutations ≠ List.nil) :=
by
  sorry

end rearrange_digits_to_perfect_square_l21_21361


namespace cylindrical_log_distance_l21_21415

def cylinder_radius := 3
def R₁ := 104
def R₂ := 64
def R₃ := 84
def straight_segment := 100

theorem cylindrical_log_distance :
  let adjusted_radius₁ := R₁ - cylinder_radius
  let adjusted_radius₂ := R₂ + cylinder_radius
  let adjusted_radius₃ := R₃ - cylinder_radius
  let arc_distance₁ := π * adjusted_radius₁
  let arc_distance₂ := π * adjusted_radius₂
  let arc_distance₃ := π * adjusted_radius₃
  let total_distance := arc_distance₁ + arc_distance₂ + arc_distance₃ + straight_segment
  total_distance = 249 * π + 100 :=
sorry

end cylindrical_log_distance_l21_21415


namespace solution_for_x_l21_21593

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end solution_for_x_l21_21593


namespace arrangement_schemes_count_l21_21868

open Finset

-- Definitions based on the given conditions
def teachers : Finset ℕ := {1, 2}
def students : Finset ℕ := {1, 2, 3, 4}
def choose_two (s : Finset ℕ) := s.choose 2

-- The theorem to prove the total number of different arrangement schemes is 12
theorem arrangement_schemes_count : 
  (teachers.card.choose 1) * ((students.card.choose 2)) = 12 :=
by
  -- Number of ways to select teachers
  have teachers_ways : teachers.card.choose 1 = 2 := by sorry
  -- Number of ways to select students
  have students_ways : students.card.choose 2 = 6 := by sorry
  -- Multiplying the ways
  rw [teachers_ways, students_ways]
  norm_num

end arrangement_schemes_count_l21_21868


namespace min_sum_factors_l21_21952

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l21_21952


namespace find_number_l21_21393

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end find_number_l21_21393


namespace find_12th_term_l21_21243

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ
| 0 => a
| (n+1) => r * geometric_sequence a r n

theorem find_12th_term : ∃ a r, geometric_sequence a r 4 = 5 ∧ geometric_sequence a r 7 = 40 ∧ geometric_sequence a r 11 = 640 :=
by
  -- statement only, no proof provided
  sorry

end find_12th_term_l21_21243


namespace quadratic_sum_terms_l21_21248

theorem quadratic_sum_terms (a b c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) → a + b + c = -46 :=
by
  sorry

end quadratic_sum_terms_l21_21248


namespace counties_no_rain_l21_21904

theorem counties_no_rain 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) :
  P_A = 0.7 → P_B = 0.5 → P_A_and_B = 0.4 →
  (1 - (P_A + P_B - P_A_and_B) = 0.2) :=
by intros h1 h2 h3; sorry

end counties_no_rain_l21_21904


namespace interior_triangle_area_l21_21889

theorem interior_triangle_area (a b c : ℝ)
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (hpythagorean : a^2 + b^2 = c^2) :
  1/2 * a * b = 24 :=
by
  sorry

end interior_triangle_area_l21_21889


namespace problem_l21_21699

variables (A B C D E : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := E > B ∧ B > D
def condition3 := D > A
def condition4 := C > B

-- Proof goal: Dana (D) and Beth (B) have the same amount of money
theorem problem (h1 : condition1 A C) (h2 : condition2 E B D) (h3 : condition3 D A) (h4 : condition4 C B) : D = B :=
sorry

end problem_l21_21699


namespace parabola_focus_segment_length_l21_21040

theorem parabola_focus_segment_length (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∀ x, abs x * abs (1 / a) = 4) : a = 1/4 := 
sorry

end parabola_focus_segment_length_l21_21040


namespace greatest_base7_3_digit_divisible_by_7_l21_21388

theorem greatest_base7_3_digit_divisible_by_7 :
  ∃ n : ℕ, n < 7^3 ∧ n ≥ 7^2 ∧ 7 ∣ n ∧ nat.to_digits 7 n = [6, 6, 0] := 
sorry

end greatest_base7_3_digit_divisible_by_7_l21_21388


namespace other_acute_angle_measure_l21_21328

-- Definitions based on the conditions
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90
def is_right_triangle (a b : ℝ) : Prop := right_triangle_sum a b ∧ a = 20

-- The statement to prove
theorem other_acute_angle_measure {a b : ℝ} (h : is_right_triangle a b) : b = 70 :=
sorry

end other_acute_angle_measure_l21_21328


namespace greatest_length_of_cords_l21_21069

theorem greatest_length_of_cords (a b c : ℝ) (h₁ : a = Real.sqrt 20) (h₂ : b = Real.sqrt 50) (h₃ : c = Real.sqrt 98) :
  ∃ (d : ℝ), d = 1 ∧ ∀ (k : ℝ), (k = a ∨ k = b ∨ k = c) → ∃ (n m : ℕ), k = d * (n : ℝ) ∧ d * (m : ℝ) = (m : ℝ) := by
sorry

end greatest_length_of_cords_l21_21069


namespace units_digit_of_result_is_7_l21_21529

theorem units_digit_of_result_is_7 (a b c : ℕ) (h : a = c + 3) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  (original - reversed) % 10 = 7 :=
by
  sorry

end units_digit_of_result_is_7_l21_21529


namespace power_of_7_mod_8_l21_21975

theorem power_of_7_mod_8 : 7^123 % 8 = 7 :=
by sorry

end power_of_7_mod_8_l21_21975


namespace John_spent_15_dollars_on_soap_l21_21620

theorem John_spent_15_dollars_on_soap (number_of_bars : ℕ) (weight_per_bar : ℝ) (cost_per_pound : ℝ)
  (h1 : number_of_bars = 20) (h2 : weight_per_bar = 1.5) (h3 : cost_per_pound = 0.5) :
  (number_of_bars * weight_per_bar * cost_per_pound) = 15 :=
by
  sorry

end John_spent_15_dollars_on_soap_l21_21620


namespace hyperbola_eccentricity_l21_21737

-- Define the conditions of the hyperbola and points
variables {a b c m d : ℝ} (ha : a > 0) (hb : b > 0) 
noncomputable def F1 : ℝ := sorry -- Placeholder for focus F1
noncomputable def F2 : ℝ := sorry -- Placeholder for focus F2
noncomputable def P : ℝ := sorry  -- Placeholder for point P

-- Define the sides of the triangle in terms of an arithmetic progression
def PF2 (m d : ℝ) : ℝ := m - d
def PF1 (m : ℝ) : ℝ := m
def F1F2 (m d : ℝ) : ℝ := m + d

-- Prove that the eccentricity is 5 given the conditions
theorem hyperbola_eccentricity 
  (m d : ℝ) (hc : c = (5 / 2) * d )  
  (h1 : PF1 m = 2 * a)
  (h2 : F1F2 m d = 2 * c)
  (h3 : (PF2 m d)^2 + (PF1 m)^2 = (F1F2 m d)^2 ) :
  (c / a) = 5 := 
sorry

end hyperbola_eccentricity_l21_21737


namespace gel_pen_price_ratio_l21_21157

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l21_21157


namespace debby_drinking_days_l21_21172

def starting_bottles := 264
def daily_consumption := 15
def bottles_left := 99

theorem debby_drinking_days : (starting_bottles - bottles_left) / daily_consumption = 11 :=
by
  -- proof steps will go here
  sorry

end debby_drinking_days_l21_21172


namespace find_number_l21_21392

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end find_number_l21_21392


namespace total_square_miles_of_plains_l21_21377

-- Defining conditions
def region_east_of_b : ℕ := 200
def region_east_of_a : ℕ := region_east_of_b - 50

-- To test this statement in Lean 4
theorem total_square_miles_of_plains : region_east_of_a + region_east_of_b = 350 := by
  sorry

end total_square_miles_of_plains_l21_21377


namespace fish_tank_ratio_l21_21239

theorem fish_tank_ratio :
  ∀ (F1 F2 F3: ℕ),
  F1 = 15 →
  F3 = 10 →
  (F3 = (1 / 3 * F2)) →
  F2 / F1 = 2 :=
by
  intros F1 F2 F3 hF1 hF3 hF2
  sorry

end fish_tank_ratio_l21_21239


namespace drug_ineffectiveness_probability_l21_21308

open Probability

-- Define the binomial distribution with 10 trials and success probability 0.8
def binomial_10_0_8 : Probability.ℙ (Fin 11) := binomial 10 0.8

-- Calculate the probability of getting x < 5 successes
def probability_drug_ineffective : ℝ := ∑ x in (Finset.range 5), binomial_10_0_8 x

-- The theorem to be proved
theorem drug_ineffectiveness_probability : probability_drug_ineffective = 0.006 := sorry

end drug_ineffectiveness_probability_l21_21308


namespace sixteen_k_plus_eight_not_perfect_square_l21_21595

theorem sixteen_k_plus_eight_not_perfect_square (k : ℕ) (hk : 0 < k) : ¬ ∃ m : ℕ, (16 * k + 8) = m * m := sorry

end sixteen_k_plus_eight_not_perfect_square_l21_21595


namespace sum_of_first_9_terms_of_arithmetic_sequence_l21_21027

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_of_first_9_terms_of_arithmetic_sequence 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 2 + a 8 = 18) 
  (h3 : sum_of_first_n_terms a S) :
  S 9 = 81 :=
sorry

end sum_of_first_9_terms_of_arithmetic_sequence_l21_21027


namespace max_green_socks_l21_21557

theorem max_green_socks (g y : ℕ) (h_t : g + y ≤ 2000) (h_prob : (g * (g - 1) + y * (y - 1) = (g + y) * (g + y - 1) / 3)) :
  g ≤ 19 := by
  sorry

end max_green_socks_l21_21557


namespace no_line_normal_to_both_curves_l21_21281

theorem no_line_normal_to_both_curves :
  ¬ ∃ a b : ℝ, ∃ (l : ℝ → ℝ),
    -- normal to y = cosh x at x = a
    (∀ x : ℝ, l x = -1 / (Real.sinh a) * (x - a) + Real.cosh a) ∧
    -- normal to y = sinh x at x = b
    (∀ x : ℝ, l x = -1 / (Real.cosh b) * (x - b) + Real.sinh b) := 
  sorry

end no_line_normal_to_both_curves_l21_21281


namespace find_ruv_l21_21576

theorem find_ruv (u v : ℝ) : 
  (∃ u v : ℝ, 
    (3 + 8 * u + 5, 1 - 4 * u + 2) = (4 + -3 * v + 5, 2 + 4 * v + 2)) →
  (u = -1/2 ∧ v = -1) :=
by
  intros H
  sorry

end find_ruv_l21_21576


namespace sqrt_factorial_product_squared_l21_21808

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l21_21808


namespace number_of_zeros_of_f_l21_21030

noncomputable def f (x : ℝ) : ℝ := 2^x - 3*x

theorem number_of_zeros_of_f : ∃ a b : ℝ, (f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧ ∀ x : ℝ, f x = 0 → x = a ∨ x = b :=
sorry

end number_of_zeros_of_f_l21_21030


namespace clock_spoke_angle_l21_21686

-- Define the parameters of the clock face and the problem.
def num_spokes := 10
def total_degrees := 360
def degrees_per_spoke := total_degrees / num_spokes
def position_3_oclock := 3 -- the third spoke
def halfway_45_oclock := 5 -- approximately the fifth spoke
def spokes_between := halfway_45_oclock - position_3_oclock
def smaller_angle := spokes_between * degrees_per_spoke
def expected_angle := 72

-- Statement of the problem
theorem clock_spoke_angle :
  smaller_angle = expected_angle := by
    -- Proof is omitted
    sorry

end clock_spoke_angle_l21_21686


namespace mika_stickers_l21_21351

def s1 : ℝ := 20.5
def s2 : ℝ := 26.3
def s3 : ℝ := 19.75
def s4 : ℝ := 6.25
def s5 : ℝ := 57.65
def s6 : ℝ := 15.8

theorem mika_stickers 
  (M : ℝ)
  (hM : M = s1 + s2 + s3 + s4 + s5 + s6) 
  : M = 146.25 :=
sorry

end mika_stickers_l21_21351


namespace total_donuts_needed_l21_21854

theorem total_donuts_needed :
  (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts_per_friend : ℕ) 
  (donuts_for_Andrew : ℕ) (total_friends : ℕ) 
  (h1 : initial_friends = 2)
  (h2 : additional_friends = 2)
  (h3 : total_friends = initial_friends + additional_friends)
  (h4 : donuts_per_friend = 3)
  (h5 : extra_donuts_per_friend = 1)
  (h6 : donuts_for_Andrew = donuts_per_friend + extra_donuts_per_friend)
  :
  let initial_donuts := total_friends * donuts_per_friend,
      extra_donuts := total_friends * extra_donuts_per_friend,
      total_donuts_for_friends := initial_donuts + extra_donuts,
      total_donuts := total_donuts_for_friends + donuts_for_Andrew 
  in total_donuts = 20 := by
  sorry

end total_donuts_needed_l21_21854


namespace min_sum_factors_l21_21954

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l21_21954


namespace orchids_cut_l21_21794

-- Define initial and final number of orchids in the vase
def initialOrchids : ℕ := 2
def finalOrchids : ℕ := 21

-- Formulate the claim to prove the number of orchids Jessica cut
theorem orchids_cut : finalOrchids - initialOrchids = 19 := by
  sorry

end orchids_cut_l21_21794


namespace hyperbola_center_l21_21295

theorem hyperbola_center :
  ∃ (center : ℝ × ℝ), center = (2.5, 4) ∧
    (∀ x y : ℝ, 9 * x^2 - 45 * x - 16 * y^2 + 128 * y + 207 = 0 ↔ 
      (1/1503) * (36 * (x - 2.5)^2 - 64 * (y - 4)^2) = 1) :=
sorry

end hyperbola_center_l21_21295


namespace vika_made_84_dollars_l21_21078

-- Define the amount of money Saheed, Kayla, and Vika made
variable (S K V : ℕ)

-- Given conditions
def condition1 : Prop := S = 4 * K
def condition2 : Prop := K = V - 30
def condition3 : Prop := S = 216

-- Statement to prove
theorem vika_made_84_dollars (S K V : ℕ) (h1 : condition1 S K) (h2 : condition2 K V) (h3 : condition3 S) : 
  V = 84 :=
by sorry

end vika_made_84_dollars_l21_21078


namespace tetrahedron_volume_is_zero_l21_21610

noncomputable def volume_of_tetrahedron (p q r : ℝ) : ℝ :=
  (1 / 6) * p * q * r

theorem tetrahedron_volume_is_zero (p q r : ℝ)
  (hpq : p^2 + q^2 = 36)
  (hqr : q^2 + r^2 = 64)
  (hrp : r^2 + p^2 = 100) :
  volume_of_tetrahedron p q r = 0 := by
  sorry

end tetrahedron_volume_is_zero_l21_21610


namespace inequality_solution_set_l21_21250

theorem inequality_solution_set (x : ℝ) : 
  (∃ x, (2 < x ∧ x < 3)) ↔ 
  ((x - 2) * (x - 3) / (x^2 + 1) < 0) :=
by sorry

end inequality_solution_set_l21_21250


namespace additional_books_l21_21354

theorem additional_books (initial_books total_books additional_books : ℕ)
  (h_initial : initial_books = 54)
  (h_total : total_books = 77) :
  additional_books = total_books - initial_books :=
by
  sorry

end additional_books_l21_21354


namespace gel_pen_ratio_l21_21129

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l21_21129


namespace identity_of_polynomials_l21_21512

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l21_21512


namespace jill_sod_area_needed_l21_21338

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end jill_sod_area_needed_l21_21338


namespace number_of_ways_to_pair_is_13_l21_21663

noncomputable def number_of_ways_to_pair (n : ℕ) : ℕ := 
  if n = 12 then 13 else 0

theorem number_of_ways_to_pair_is_13 :
  number_of_ways_to_pair 12 = 13 :=
by
  -- conditions
  let people := Finset.fin 12
  let knows := λ (a b : Fin 12), (a.val + 1) % 12 = b.val ∨ (a.val + 11) % 12 = b.val ∨ (a.val + 2) % 12 = b.val ∨ (a.val + 10) % 12 = b.val
  -- proof that respects the conditions (knowledge relationships amongst people)
  sorry

end number_of_ways_to_pair_is_13_l21_21663


namespace unique_root_condition_l21_21293

theorem unique_root_condition (a : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 - 4*a*x + a^2 - 4 = 0 → ∃! x₀ : ℝ, x = x₀) ↔ a < 1 :=
by sorry

end unique_root_condition_l21_21293


namespace max_length_sequence_l21_21289

def seq_term (n : ℕ) (y : ℤ) : ℤ :=
  match n with
  | 0 => 2000
  | 1 => y
  | k + 2 => seq_term (k + 1) y - seq_term k y

theorem max_length_sequence (y : ℤ) :
  1200 < y ∧ y < 1334 ∧ (∀ n, seq_term n y ≥ 0 ∨ seq_term (n + 1) y < 0) ↔ y = 1333 :=
by
  sorry

end max_length_sequence_l21_21289


namespace chad_total_spend_on_ice_l21_21861

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end chad_total_spend_on_ice_l21_21861


namespace min_sum_a_b_c_l21_21600

open Nat

theorem min_sum_a_b_c (a b c : ℕ) (h_lcm : lcm (lcm a b) c = 48)
  (h_gcd_ab : gcd a b = 4) (h_gcd_bc : gcd b c = 3) : a + b + c = 31 :=
  sorry

end min_sum_a_b_c_l21_21600


namespace min_sum_of_factors_l21_21961

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l21_21961


namespace regular_polygon_num_sides_l21_21700

theorem regular_polygon_num_sides (angle : ℝ) (h : angle = 45) : 
  (∃ n : ℕ, n = 360 / angle ∧ n ≠ 0) → n = 8 :=
by
  sorry

end regular_polygon_num_sides_l21_21700


namespace gardener_cabbages_increased_by_197_l21_21841

theorem gardener_cabbages_increased_by_197 (x : ℕ) (last_year_cabbages : ℕ := x^2) (increase : ℕ := 197) :
  (x + 1)^2 = x^2 + increase → (x + 1)^2 = 9801 :=
by
  intros h
  sorry

end gardener_cabbages_increased_by_197_l21_21841


namespace min_value_abs_2a_minus_b_l21_21184

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) : ∃ c : ℝ, c = |2 * a - b| ∧ c = 1 := 
sorry

end min_value_abs_2a_minus_b_l21_21184


namespace combinations_sol_eq_l21_21444

theorem combinations_sol_eq (x : ℕ) (h : Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) : x = 1 ∨ x = 3 := sorry

end combinations_sol_eq_l21_21444


namespace sara_spent_on_hotdog_l21_21081

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end sara_spent_on_hotdog_l21_21081


namespace aram_fraction_of_fine_l21_21547

theorem aram_fraction_of_fine
  (F : ℝ)
  (Joe_payment : ℝ := (1 / 4) * F + 3)
  (Peter_payment : ℝ := (1 / 3) * F - 3)
  (Aram_payment : ℝ := (1 / 2) * F - 4)
  (sum_payments_eq_F : Joe_payment + Peter_payment + Aram_payment = F):
  (Aram_payment / F) = (5 / 12) :=
by
  sorry

end aram_fraction_of_fine_l21_21547


namespace ninety_eight_times_ninety_eight_l21_21717

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 := 
by
  sorry

end ninety_eight_times_ninety_eight_l21_21717


namespace series_converges_uniformly_l21_21760

noncomputable def xi (ω : Ω) (n : ℕ) : ℝ := sorry -- Suppose xi is defined as needed

theorem series_converges_uniformly (h_iid : ∀ n, i.i.d. (xi ω n)) (h_vals : ∀ n, xi ω n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
(h_prob : ∀ n, prob {ω | xi ω n = k} = 1 / 10):
  ∃ X, (∀ ω, (∀ ε > 0, ∃ N, ∀ n ≥ N, |(∑ i in finset.range n, xi ω i / (10 ^ i)) - X ω| < ε)) ∧ (uniform_distribution X [0, 1]) :=
begin
  sorry
end

end series_converges_uniformly_l21_21760


namespace rosie_laps_l21_21012

theorem rosie_laps (lou_distance : ℝ) (track_length : ℝ) (lou_speed_factor : ℝ) (rosie_speed_multiplier : ℝ) 
    (number_of_laps_by_lou : ℝ) (number_of_laps_by_rosie : ℕ) :
  lou_distance = 3 ∧ 
  track_length = 1 / 4 ∧ 
  lou_speed_factor = 0.75 ∧ 
  rosie_speed_multiplier = 2 ∧ 
  number_of_laps_by_lou = lou_distance / track_length ∧ 
  number_of_laps_by_rosie = rosie_speed_multiplier * number_of_laps_by_lou → 
  number_of_laps_by_rosie = 18 := 
sorry

end rosie_laps_l21_21012


namespace gcd_48_30_is_6_l21_21669

/-- Prove that the Greatest Common Divisor (GCD) of 48 and 30 is 6. -/
theorem gcd_48_30_is_6 : Int.gcd 48 30 = 6 := by
  sorry

end gcd_48_30_is_6_l21_21669


namespace sales_tax_difference_l21_21781

theorem sales_tax_difference
  (item_price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.0725)
  (h_rate2 : rate2 = 0.0675)
  (h_item_price : item_price = 40) :
  item_price * rate1 - item_price * rate2 = 0.20 :=
by
  -- Since we are required to skip the proof, we put sorry here.
  sorry

end sales_tax_difference_l21_21781


namespace Sandy_pumpkins_l21_21643

-- Definitions from the conditions
def Mike_pumpkins : ℕ := 23
def Total_pumpkins : ℕ := 74

-- Theorem to prove the number of pumpkins Sandy grew
theorem Sandy_pumpkins : ∃ (n : ℕ), n + Mike_pumpkins = Total_pumpkins :=
by
  existsi 51
  sorry

end Sandy_pumpkins_l21_21643


namespace greyson_spent_on_fuel_l21_21605

theorem greyson_spent_on_fuel : ∀ (cost_per_refill times_refilled total_cost : ℕ), 
  cost_per_refill = 10 → 
  times_refilled = 4 → 
  total_cost = cost_per_refill * times_refilled → 
  total_cost = 40 :=
by
  intro cost_per_refill times_refilled total_cost
  intro h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end greyson_spent_on_fuel_l21_21605


namespace raine_steps_l21_21359

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end raine_steps_l21_21359


namespace a_alone_days_l21_21267

theorem a_alone_days 
  (B_days : ℕ)
  (B_days_eq : B_days = 8)
  (C_payment : ℝ)
  (C_payment_eq : C_payment = 450)
  (total_payment : ℝ)
  (total_payment_eq : total_payment = 3600)
  (combined_days : ℕ)
  (combined_days_eq : combined_days = 3)
  (combined_rate_eq : (1 / A + 1 / B_days + C = 1 / combined_days)) 
  (rate_proportion : (1 / A) / (1 / B_days) = 7 / 1) 
  : A = 56 :=
sorry

end a_alone_days_l21_21267


namespace correct_calculation_l21_21402

theorem correct_calculation :
  (∀ x, (x = (\sqrt 3)^2 → x ≠ 9)) ∧
  (∀ y, (y = \sqrt ((-2)^2) → y ≠ -2)) ∧
  (∀ z, (z = \sqrt 3 * \sqrt 2 → z ≠ 6)) ∧
  (∀ w, (w = \sqrt 8 / \sqrt 2 → w = 2)) :=
by
  sorry

end correct_calculation_l21_21402


namespace price_ratio_l21_21143

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l21_21143


namespace stans_average_speed_l21_21776

noncomputable def average_speed (distance1 distance2 distance3 : ℝ) (time1_hrs time1_mins time2 time3_hrs time3_mins : ℝ) : ℝ :=
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1_hrs + time1_mins / 60 + time2 + time3_hrs + time3_mins / 60
  total_distance / total_time

theorem stans_average_speed  :
  average_speed 350 420 330 5 40 7 5 30 = 60.54 :=
by
  -- sorry block indicates missing proof
  sorry

end stans_average_speed_l21_21776


namespace length_of_bridge_l21_21848

theorem length_of_bridge (ship_length : ℝ) (ship_speed_kmh : ℝ) (time : ℝ) (bridge_length : ℝ) :
  ship_length = 450 → ship_speed_kmh = 24 → time = 202.48 → bridge_length = (6.67 * 202.48 - 450) → bridge_length = 900.54 :=
by
  intros h1 h2 h3 h4
  sorry

end length_of_bridge_l21_21848


namespace gcd_51457_37958_l21_21719

theorem gcd_51457_37958 : Nat.gcd 51457 37958 = 1 := 
  sorry

end gcd_51457_37958_l21_21719


namespace max_sum_square_pyramid_addition_l21_21988

def square_pyramid_addition_sum (faces edges vertices : ℕ) : ℕ :=
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices

theorem max_sum_square_pyramid_addition :
  square_pyramid_addition_sum 6 12 8 = 34 :=
by
  sorry

end max_sum_square_pyramid_addition_l21_21988


namespace exchange_yen_for_yuan_l21_21332

-- Define the condition: 100 Japanese yen could be exchanged for 7.2 yuan
def exchange_rate : ℝ := 7.2
def yen_per_100_yuan : ℝ := 100

-- Define the amount in yuan we want to exchange
def yuan_amount : ℝ := 720

-- The mathematical assertion (proof problem)
theorem exchange_yen_for_yuan : 
  (yuan_amount / exchange_rate) * yen_per_100_yuan = 10000 :=
by
  sorry

end exchange_yen_for_yuan_l21_21332


namespace replace_stars_with_identity_l21_21508

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l21_21508


namespace det_matrixE_l21_21626

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l21_21626


namespace population_net_increase_l21_21327

-- Definitions of conditions
def birth_rate := 7 / 2 -- 7 people every 2 seconds
def death_rate := 1 / 2 -- 1 person every 2 seconds
def seconds_in_a_day := 86400 -- Number of seconds in one day

-- Definition of the total births in one day
def total_births_per_day := birth_rate * seconds_in_a_day

-- Definition of the total deaths in one day
def total_deaths_per_day := death_rate * seconds_in_a_day

-- Proposition to prove the net population increase in one day
theorem population_net_increase : total_births_per_day - total_deaths_per_day = 259200 := by
  sorry

end population_net_increase_l21_21327


namespace train_speed_proof_l21_21972

def identical_trains_speed : Real :=
  11.11

theorem train_speed_proof :
  ∀ (v : ℝ),
  (∀ (t t' : ℝ), 
  (t = 150 / v) ∧ 
  (t' = 300 / v) ∧ 
  ((t' + 100 / v) = 36)) → v = identical_trains_speed :=
by
  sorry

end train_speed_proof_l21_21972


namespace monomial_same_type_l21_21198

theorem monomial_same_type (a b : ℕ) (h1 : a + 1 = 3) (h2 : b = 3) : a + b = 5 :=
by 
  -- proof goes here
  sorry

end monomial_same_type_l21_21198


namespace problem_fraction_eq_l21_21540

theorem problem_fraction_eq (x : ℝ) :
  (x * (3 / 4) * (1 / 2) * 5060 = 759.0000000000001) ↔ (x = 0.4) :=
by
  sorry

end problem_fraction_eq_l21_21540


namespace min_value_of_expr_l21_21974

-- Define the expression
def expr (x y : ℝ) : ℝ := (x * y + 1)^2 + (x - y)^2

-- Statement to prove that the minimum value of the expression is 1
theorem min_value_of_expr : ∃ x y : ℝ, expr x y = 1 ∧ ∀ a b : ℝ, expr a b ≥ 1 :=
by
  -- Here the proof would be provided, but we leave it as sorry as per instructions.
  sorry

end min_value_of_expr_l21_21974


namespace sitting_people_l21_21005

variables {M I P A : Prop}

-- Conditions
axiom M_not_sitting : ¬ M
axiom A_not_sitting : ¬ A
axiom if_M_not_sitting_then_I_sitting : ¬ M → I
axiom if_I_sitting_then_P_sitting : I → P

theorem sitting_people : I ∧ P :=
by
  have I_sitting : I := if_M_not_sitting_then_I_sitting M_not_sitting
  have P_sitting : P := if_I_sitting_then_P_sitting I_sitting
  exact ⟨I_sitting, P_sitting⟩

end sitting_people_l21_21005


namespace prime_p_p_plus_15_l21_21659

theorem prime_p_p_plus_15 (p : ℕ) (hp : Nat.Prime p) (hp15 : Nat.Prime (p + 15)) : p = 2 :=
sorry

end prime_p_p_plus_15_l21_21659


namespace P_lt_Q_l21_21883

noncomputable def P (a : ℝ) : ℝ := (Real.sqrt (a + 41)) - (Real.sqrt (a + 40))
noncomputable def Q (a : ℝ) : ℝ := (Real.sqrt (a + 39)) - (Real.sqrt (a + 38))

theorem P_lt_Q (a : ℝ) (h : a > -38) : P a < Q a := by sorry

end P_lt_Q_l21_21883


namespace find_ratio_l21_21070
   
   -- Given Conditions
   variable (S T F : ℝ)
   variable (H1 : 30 + S + T + F = 450)
   variable (H2 : S > 30)
   variable (H3 : T > S)
   variable (H4 : F > T)
   
   -- The goal is to find the ratio S / 30
   theorem find_ratio :
     ∃ r : ℝ, r = S / 30 ↔ false :=
   by
     sorry
   
end find_ratio_l21_21070


namespace greatest_integer_property_l21_21720

theorem greatest_integer_property :
  ∃ n : ℤ, n < 1000 ∧ (∃ m : ℤ, 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧ 
  (∀ k : ℤ, k < 1000 ∧ (∃ m : ℤ, 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) → k ≤ n) := by
  -- skipped the proof with sorry
  sorry

end greatest_integer_property_l21_21720


namespace vector_field_lines_l21_21594

noncomputable def vector_lines : Prop :=
  ∃ (C_1 C_2 : ℝ), ∀ (x y z : ℝ), (9 * z^2 + 4 * y^2 = C_1) ∧ (x = C_2)

-- We state the proof goal as follows:
theorem vector_field_lines :
  ∀ (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ), 
    (∀ (x y z : ℝ), a (x, y, z) = (0, 9 * z, -4 * y)) →
    vector_lines :=
by
  intro a ha
  sorry

end vector_field_lines_l21_21594


namespace max_sides_of_polygon_in_1950_gon_l21_21200

theorem max_sides_of_polygon_in_1950_gon (n : ℕ) (h : n = 1950) :
  ∃ (m : ℕ), (m ≤ 1949) ∧ (∀ k, k > m → k ≤ 1949) :=
sorry

end max_sides_of_polygon_in_1950_gon_l21_21200


namespace rectangle_area_l21_21905

-- Define the vertices of the rectangle
def V1 : ℝ × ℝ := (-7, 1)
def V2 : ℝ × ℝ := (1, 1)
def V3 : ℝ × ℝ := (1, -6)
def V4 : ℝ × ℝ := (-7, -6)

-- Define the function to compute the area of the rectangle given the vertices
noncomputable def area_of_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  let length := abs (B.1 - A.1)
  let width := abs (A.2 - D.2)
  length * width

-- The statement to prove
theorem rectangle_area : area_of_rectangle V1 V2 V3 V4 = 56 := by
  sorry

end rectangle_area_l21_21905


namespace alexis_initial_budget_l21_21567

-- Define all the given conditions
def cost_shirt : Int := 30
def cost_pants : Int := 46
def cost_coat : Int := 38
def cost_socks : Int := 11
def cost_belt : Int := 18
def cost_shoes : Int := 41
def amount_left : Int := 16

-- Define the total expenses
def total_expenses : Int := cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + cost_shoes

-- Define the initial budget
def initial_budget : Int := total_expenses + amount_left

-- The proof statement
theorem alexis_initial_budget : initial_budget = 200 := by
  sorry

end alexis_initial_budget_l21_21567


namespace range_m_condition_l21_21507

theorem range_m_condition {x y m : ℝ} (h1 : x^2 + (y - 1)^2 = 1) (h2 : x + y + m ≥ 0) : -1 < m :=
by
  sorry

end range_m_condition_l21_21507


namespace coins_division_remainder_l21_21271

theorem coins_division_remainder :
  ∃ n : ℕ, (n % 8 = 6 ∧ n % 7 = 5 ∧ n % 9 = 0) :=
sorry

end coins_division_remainder_l21_21271


namespace sqrt_factorial_squared_l21_21820

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l21_21820


namespace friends_truth_l21_21251

-- Definitions for the truth values of the friends
def F₁_truth (a x₁ x₂ x₃ : Prop) : Prop := a ↔ ¬ (x₁ ∨ x₂ ∨ x₃)
def F₂_truth (b x₁ x₂ x₃ : Prop) : Prop := b ↔ (x₂ ∧ ¬ x₁ ∧ ¬ x₃)
def F₃_truth (c x₁ x₂ x₃ : Prop) : Prop := c ↔ x₃

-- Main theorem statement
theorem friends_truth (a b c x₁ x₂ x₃ : Prop) 
  (H₁ : F₁_truth a x₁ x₂ x₃) 
  (H₂ : F₂_truth b x₁ x₂ x₃) 
  (H₃ : F₃_truth c x₁ x₂ x₃)
  (H₄ : a ∨ b ∨ c) 
  (H₅ : ¬ (a ∧ b ∧ c)) : a ∧ ¬b ∧ ¬c ∨ ¬a ∧ b ∧ ¬c ∨ ¬a ∧ ¬b ∧ c :=
sorry

end friends_truth_l21_21251


namespace max_pots_l21_21240

theorem max_pots (x y z : ℕ) (h₁ : 3 * x + 4 * y + 9 * z = 100) (h₂ : 1 ≤ x) (h₃ : 1 ≤ y) (h₄ : 1 ≤ z) : 
  z ≤ 10 :=
sorry

end max_pots_l21_21240


namespace campers_difference_l21_21108

theorem campers_difference 
       (total : ℕ)
       (campers_two_weeks_ago : ℕ) 
       (campers_last_week : ℕ) 
       (diff: ℕ)
       (h_total : total = 150)
       (h_two_weeks_ago : campers_two_weeks_ago = 40) 
       (h_last_week : campers_last_week = 80) : 
       diff = campers_two_weeks_ago - (total - campers_two_weeks_ago - campers_last_week) :=
by
  sorry

end campers_difference_l21_21108


namespace second_option_feasible_l21_21007

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l21_21007


namespace train_cross_time_in_seconds_l21_21990

-- Definitions based on conditions
def train_speed_kph : ℚ := 60
def train_length_m : ℚ := 450

-- Statement: prove that the time to cross the pole is 27 seconds
theorem train_cross_time_in_seconds (train_speed_kph train_length_m : ℚ) :
  train_speed_kph = 60 →
  train_length_m = 450 →
  (train_length_m / (train_speed_kph * 1000 / 3600)) = 27 :=
by
  intros h_speed h_length
  rw [h_speed, h_length]
  sorry

end train_cross_time_in_seconds_l21_21990


namespace stickers_per_student_l21_21763

theorem stickers_per_student (G S B N: ℕ) (hG: G = 50) (hS: S = 2 * G) (hB: B = S - 20) (hN: N = 5) : 
  (G + S + B) / N = 46 := by
  sorry

end stickers_per_student_l21_21763


namespace june_spent_on_music_books_l21_21213

theorem june_spent_on_music_books
  (total_budget : ℤ)
  (math_books_cost : ℤ)
  (science_books_cost : ℤ)
  (art_books_cost : ℤ)
  (music_books_cost : ℤ)
  (h_total_budget : total_budget = 500)
  (h_math_books_cost : math_books_cost = 80)
  (h_science_books_cost : science_books_cost = 100)
  (h_art_books_cost : art_books_cost = 160)
  (h_total_cost : music_books_cost = total_budget - (math_books_cost + science_books_cost + art_books_cost)) :
  music_books_cost = 160 :=
sorry

end june_spent_on_music_books_l21_21213


namespace jill_sod_area_l21_21337

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end jill_sod_area_l21_21337


namespace fraction_eq_four_l21_21896

theorem fraction_eq_four (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) : 
  (2 * a + b) / b = 4 := 
by 
  sorry

end fraction_eq_four_l21_21896


namespace solve_system_l21_21940

theorem solve_system : ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 8 * x + 5 * y = 7 ∧ x = 1 / 4 ∧ y = 1 :=
by
  sorry

end solve_system_l21_21940


namespace price_ratio_l21_21146

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l21_21146


namespace recipe_flour_cups_l21_21638

theorem recipe_flour_cups (F : ℕ) : 
  (exists (sugar : ℕ) (flourAdded : ℕ) (sugarExtra : ℕ), sugar = 11 ∧ flourAdded = 4 ∧ sugarExtra = 6 ∧ ((F - flourAdded) + sugarExtra = sugar)) →
  F = 9 :=
sorry

end recipe_flour_cups_l21_21638


namespace graph_EQ_a_l21_21933

theorem graph_EQ_a (x y : ℝ) : (x - 2) * (y + 3) = 0 ↔ x = 2 ∨ y = -3 :=
by sorry

end graph_EQ_a_l21_21933


namespace AM_GM_problem_l21_21044

theorem AM_GM_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := 
sorry

end AM_GM_problem_l21_21044


namespace gel_pen_price_ratio_l21_21158

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l21_21158


namespace euclidean_steps_arbitrarily_large_l21_21769

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

theorem euclidean_steps_arbitrarily_large (n : ℕ) (h : n ≥ 2) :
  gcd (fib (n+1)) (fib n) = gcd (fib 1) (fib 0) := 
sorry

end euclidean_steps_arbitrarily_large_l21_21769


namespace number_of_bad_carrots_l21_21835

-- Definitions for conditions
def olivia_picked : ℕ := 20
def mother_picked : ℕ := 14
def good_carrots : ℕ := 19

-- Sum of total carrots picked
def total_carrots : ℕ := olivia_picked + mother_picked

-- Theorem stating the number of bad carrots
theorem number_of_bad_carrots : total_carrots - good_carrots = 15 :=
by
  sorry

end number_of_bad_carrots_l21_21835


namespace ratio_of_speeds_l21_21421

theorem ratio_of_speeds (v_A v_B : ℝ) (t : ℝ) (hA : v_A = 120 / t) (hB : v_B = 60 / t) : v_A / v_B = 2 :=
by {
  sorry
}

end ratio_of_speeds_l21_21421


namespace area_of_Q1Q3Q5Q7_l21_21562

def regular_octagon_apothem : ℝ := 3

def area_of_quadrilateral (a : ℝ) : Prop :=
  let s := 6 * (1 - Real.sqrt 2)
  let side_length := s * Real.sqrt 2
  let area := side_length ^ 2
  area = 72 * (3 - 2 * Real.sqrt 2)

theorem area_of_Q1Q3Q5Q7 : area_of_quadrilateral regular_octagon_apothem :=
  sorry

end area_of_Q1Q3Q5Q7_l21_21562


namespace remaining_gallons_to_fill_tank_l21_21997

-- Define the conditions as constants
def tank_capacity : ℕ := 50
def rate_seconds_per_gallon : ℕ := 20
def time_poured_minutes : ℕ := 6

-- Define the number of gallons poured per minute
def gallons_per_minute : ℕ := 60 / rate_seconds_per_gallon

def gallons_poured (minutes : ℕ) : ℕ :=
  minutes * gallons_per_minute

-- The main statement to prove the remaining gallons needed
theorem remaining_gallons_to_fill_tank : 
  tank_capacity - gallons_poured time_poured_minutes = 32 :=
by
  sorry

end remaining_gallons_to_fill_tank_l21_21997


namespace find_value_l21_21759

theorem find_value (x : ℝ) (hx : x + 1/x = 4) : x^3 + 1/x^3 = 52 := 
by 
  sorry

end find_value_l21_21759


namespace sqrt_meaningful_l21_21901

theorem sqrt_meaningful (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end sqrt_meaningful_l21_21901


namespace polygon_sides_arithmetic_progression_l21_21946

theorem polygon_sides_arithmetic_progression
  (n : ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 172 - (i - 1) * 8 > 0) -- Each angle in the sequence is positive
  (h2 : (∀ i, 1 ≤ i → i ≤ n → (172 - (i - 1) * 8) < 180)) -- Each angle < 180 degrees
  (h3 : n * (172 - (n-1) * 4) = 180 * (n - 2)) -- Sum of interior angles formula
  : n = 10 :=
sorry

end polygon_sides_arithmetic_progression_l21_21946


namespace pencils_per_person_l21_21646

theorem pencils_per_person (x : ℕ) (h : 3 * x = 24) : x = 8 :=
by
  -- sorry we are skipping the actual proof
  sorry

end pencils_per_person_l21_21646


namespace circle_radii_order_l21_21171

theorem circle_radii_order (r_A r_B r_C : ℝ) 
  (h1 : r_A = Real.sqrt 10) 
  (h2 : 2 * Real.pi * r_B = 10 * Real.pi)
  (h3 : Real.pi * r_C^2 = 16 * Real.pi) : 
  r_C < r_A ∧ r_A < r_B := 
  sorry

end circle_radii_order_l21_21171


namespace mr_smith_total_cost_l21_21639

noncomputable def total_cost : ℝ :=
  let adult_price := 30
  let child_price := 15
  let teen_price := 25
  let senior_discount := 0.10
  let college_discount := 0.05
  let senior_price := adult_price * (1 - senior_discount)
  let college_price := adult_price * (1 - college_discount)
  let soda_price := 2
  let iced_tea_price := 3
  let coffee_price := 4
  let juice_price := 1.50
  let wine_price := 6
  let buffet_cost := 2 * adult_price + 2 * senior_price + 3 * child_price + teen_price + 2 * college_price
  let drinks_cost := 3 * soda_price + 2 * iced_tea_price + coffee_price + juice_price + 2 * wine_price
  buffet_cost + drinks_cost

theorem mr_smith_total_cost : total_cost = 270.50 :=
by
  sorry

end mr_smith_total_cost_l21_21639


namespace sum_le_square_l21_21948

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end sum_le_square_l21_21948


namespace solution_for_x_l21_21592

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end solution_for_x_l21_21592


namespace distance_between_points_l21_21282

theorem distance_between_points :
  let point1 := (2, -3)
  let point2 := (8, 9)
  dist point1 point2 = 6 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_l21_21282


namespace find_constants_l21_21244

theorem find_constants :
  ∃ (A B C : ℚ), 
  (A = 1 ∧ B = 4 ∧ C = 1) ∧ 
  (∀ x, x ≠ -1 → x ≠ 3/2 → x ≠ 2 → 
    (6 * x^2 - 13 * x + 6) / (2 * x^3 + 3 * x^2 - 11 * x - 6) = 
    (A / (x + 1) + B / (2 * x - 3) + C / (x - 2))) :=
by
  sorry

end find_constants_l21_21244


namespace gasoline_price_increase_l21_21423

theorem gasoline_price_increase (high low : ℝ) (high_eq : high = 24) (low_eq : low = 18) : 
  ((high - low) / low) * 100 = 33.33 := 
  sorry

end gasoline_price_increase_l21_21423


namespace hat_value_in_rice_l21_21614

variables (f l r h : ℚ)

theorem hat_value_in_rice :
  (4 * f = 3 * l) →
  (l = 5 * r) →
  (5 * f = 7 * h) →
  h = (75 / 28) * r :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end hat_value_in_rice_l21_21614


namespace cos_double_angle_l21_21026

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
by 
  sorry

end cos_double_angle_l21_21026


namespace larger_number_l21_21094

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end larger_number_l21_21094


namespace seven_not_spheric_spheric_power_spheric_l21_21256

def is_spheric (r : ℚ) : Prop := ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := 
sorry

theorem spheric_power_spheric (r : ℚ) (n : ℕ) (h : is_spheric r) (hn : n > 1) : is_spheric (r ^ n) := 
sorry

end seven_not_spheric_spheric_power_spheric_l21_21256


namespace fillets_per_fish_l21_21431

-- Definitions for the conditions
def fish_caught_per_day := 2
def days := 30
def total_fish_caught : Nat := fish_caught_per_day * days
def total_fish_fillets := 120

-- The proof problem statement
theorem fillets_per_fish (h1 : total_fish_caught = 60) (h2 : total_fish_fillets = 120) : 
  (total_fish_fillets / total_fish_caught) = 2 := sorry

end fillets_per_fish_l21_21431


namespace factorial_sqrt_sq_l21_21829

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l21_21829


namespace regular_octagon_angle_ABG_l21_21767

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end regular_octagon_angle_ABG_l21_21767


namespace angle_east_northwest_l21_21685

def num_spokes : ℕ := 12
def central_angle : ℕ := 360 / num_spokes
def angle_between (start_dir end_dir : ℕ) : ℕ := (end_dir - start_dir) * central_angle

theorem angle_east_northwest : angle_between 3 9 = 90 := sorry

end angle_east_northwest_l21_21685


namespace abcd_product_l21_21186

theorem abcd_product :
  let A := (Real.sqrt 3003 + Real.sqrt 3004)
  let B := (-Real.sqrt 3003 - Real.sqrt 3004)
  let C := (Real.sqrt 3003 - Real.sqrt 3004)
  let D := (Real.sqrt 3004 - Real.sqrt 3003)
  A * B * C * D = 1 := 
by
  sorry

end abcd_product_l21_21186


namespace shelter_total_cats_l21_21342

theorem shelter_total_cats (total_adult_cats num_female_cats num_litters avg_kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 150) 
  (h2 : num_female_cats = 2 * total_adult_cats / 3)
  (h3 : num_litters = 2 * num_female_cats / 3)
  (h4 : avg_kittens_per_litter = 5):
  total_adult_cats + num_litters * avg_kittens_per_litter = 480 :=
by
  sorry

end shelter_total_cats_l21_21342


namespace smallest_coin_remainder_l21_21268

theorem smallest_coin_remainder
  (c : ℕ)
  (h1 : c % 8 = 6)
  (h2 : c % 7 = 5)
  (h3 : ∀ d : ℕ, (d % 8 = 6) → (d % 7 = 5) → d ≥ c) :
  c % 9 = 2 :=
sorry

end smallest_coin_remainder_l21_21268


namespace nelly_payment_is_correct_l21_21927

-- Given definitions and conditions
def joes_bid : ℕ := 160000
def additional_amount : ℕ := 2000

-- Nelly's total payment
def nellys_payment : ℕ := (3 * joes_bid) + additional_amount

-- The proof statement we need to prove that Nelly's payment equals 482000 dollars
theorem nelly_payment_is_correct : nellys_payment = 482000 :=
by
  -- This is a placeholder for the actual proof.
  -- You can fill in the formal proof here.
  sorry

end nelly_payment_is_correct_l21_21927


namespace exists_disk_of_radius_one_containing_1009_points_l21_21535

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end exists_disk_of_radius_one_containing_1009_points_l21_21535


namespace no_solution_exists_l21_21001

theorem no_solution_exists :
  ¬ ∃ a b : ℝ, a^2 + 3 * b^2 + 2 = 3 * a * b :=
by
  sorry

end no_solution_exists_l21_21001


namespace beanie_babies_total_l21_21499

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end beanie_babies_total_l21_21499


namespace modulus_of_2_plus_i_over_1_plus_2i_l21_21880

open Complex

noncomputable def modulus_of_complex_fraction : ℂ := 
  let z : ℂ := (2 + I) / (1 + 2 * I)
  abs z

theorem modulus_of_2_plus_i_over_1_plus_2i :
  modulus_of_complex_fraction = 1 := by
  sorry

end modulus_of_2_plus_i_over_1_plus_2i_l21_21880


namespace chad_total_spend_on_ice_l21_21862

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end chad_total_spend_on_ice_l21_21862


namespace james_milk_left_l21_21334

@[simp] def ounces_in_gallon : ℕ := 128
@[simp] def gallons_james_has : ℕ := 3
@[simp] def ounces_drank : ℕ := 13

theorem james_milk_left :
  (gallons_james_has * ounces_in_gallon - ounces_drank) = 371 :=
by
  sorry

end james_milk_left_l21_21334


namespace gel_pen_is_eight_times_ballpoint_pen_l21_21137

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l21_21137


namespace ratio_of_rectangle_to_square_l21_21230

theorem ratio_of_rectangle_to_square (s w h : ℝ) 
  (hs : h = s / 2)
  (shared_area_ABCD_EFGH_1 : 0.25 * s^2 = 0.4 * w * h)
  (shared_area_ABCD_EFGH_2 : 0.25 * s^2 = 0.4 * w * h) :
  w / h = 2.5 :=
by
  -- Proof goes here
  sorry

end ratio_of_rectangle_to_square_l21_21230


namespace water_pouring_problem_l21_21840

theorem water_pouring_problem : ∃ n : ℕ, n = 3 ∧
  (1 / (2 * n - 1) = 1 / 5) :=
by
  sorry

end water_pouring_problem_l21_21840


namespace identity_verification_l21_21517

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l21_21517


namespace ratio_eq_one_l21_21608

theorem ratio_eq_one {a b : ℝ} (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0 ∧ b ≠ 0) : (a^2 / 5) / (b^3 / 4) = 1 :=
by
  sorry

end ratio_eq_one_l21_21608


namespace min_value_x_plus_2y_l21_21462

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2y_l21_21462


namespace diagonal_length_l21_21178

theorem diagonal_length (d : ℝ) 
  (offset1 offset2 : ℝ) 
  (area : ℝ) 
  (h_offsets : offset1 = 11) 
  (h_offsets2 : offset2 = 9) 
  (h_area : area = 400) : d = 40 :=
by 
  sorry

end diagonal_length_l21_21178


namespace range_of_a_l21_21464

variable (a x : ℝ)

theorem range_of_a (h : ax > 2) (h_transform: ax > 2 → x < 2/a) : a < 0 :=
sorry

end range_of_a_l21_21464


namespace sqrt_factorial_mul_squared_l21_21811

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l21_21811


namespace Sara_spent_on_hotdog_l21_21083

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end Sara_spent_on_hotdog_l21_21083


namespace point_in_fourth_quadrant_l21_21618

def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant :
  in_fourth_quadrant (1, -2) ∧
  ¬ in_fourth_quadrant (2, 1) ∧
  ¬ in_fourth_quadrant (-2, 1) ∧
  ¬ in_fourth_quadrant (-1, -3) :=
by
  sorry

end point_in_fourth_quadrant_l21_21618


namespace find_certain_number_l21_21800

theorem find_certain_number (x : ℝ) : 136 - 0.35 * x = 31 -> x = 300 :=
by
  intro h
  sorry

end find_certain_number_l21_21800


namespace total_viewing_time_l21_21411

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end total_viewing_time_l21_21411


namespace theo_eggs_needed_l21_21703

def customers_first_hour : ℕ := 5
def customers_second_hour : ℕ := 7
def customers_third_hour : ℕ := 3
def customers_fourth_hour : ℕ := 8
def eggs_per_3_egg_omelette : ℕ := 3
def eggs_per_4_egg_omelette : ℕ := 4

theorem theo_eggs_needed :
  (customers_first_hour * eggs_per_3_egg_omelette) +
  (customers_second_hour * eggs_per_4_egg_omelette) +
  (customers_third_hour * eggs_per_3_egg_omelette) +
  (customers_fourth_hour * eggs_per_4_egg_omelette) = 84 := by
  sorry

end theo_eggs_needed_l21_21703


namespace ticket_distribution_l21_21793

noncomputable def num_dist_methods (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : ℕ := sorry

theorem ticket_distribution :
  num_dist_methods 18 5 6 7 10 = 140 := sorry

end ticket_distribution_l21_21793


namespace expected_time_for_bob_l21_21266

noncomputable def expected_waiting_time (times : List ℝ) : ℝ :=
  (List.sum (List.map (λ x, x / 2) (times.eraseNth 1))) + times.nthLe 1 sorry

theorem expected_time_for_bob :
  let times := [5, 7, 1, 12, 5]
  expected_waiting_time times = 18.5 := 
by
  sorry

end expected_time_for_bob_l21_21266


namespace fraction_halfway_between_l21_21527

theorem fraction_halfway_between (a b : ℚ) (h₁ : a = 1 / 6) (h₂ : b = 2 / 5) : (a + b) / 2 = 17 / 60 :=
by {
  sorry
}

end fraction_halfway_between_l21_21527


namespace total_pieces_of_paper_l21_21234

/-- Definitions according to the problem's conditions -/
def pieces_after_first_cut : Nat := 10

def pieces_after_second_cut (initial_pieces : Nat) : Nat := initial_pieces + 9

def pieces_after_third_cut (after_second_cut_pieces : Nat) : Nat := after_second_cut_pieces + 9

def pieces_after_fourth_cut (after_third_cut_pieces : Nat) : Nat := after_third_cut_pieces + 9

/-- The main theorem stating the desired result -/
theorem total_pieces_of_paper : 
  pieces_after_fourth_cut (pieces_after_third_cut (pieces_after_second_cut pieces_after_first_cut)) = 37 := 
by 
  -- The proof would go here, but it's omitted as per the instructions.
  sorry

end total_pieces_of_paper_l21_21234


namespace annie_building_time_l21_21122

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end annie_building_time_l21_21122


namespace identity_verification_l21_21519

theorem identity_verification (x : ℝ) :
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  have h₁ : (2 * x - 1)^3 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    calc
      (2 * x - 1)^3 = (2 * x)^3 + 3 * (2 * x)^2 * (-1) + 3 * (2 * x) * (-1)^2 + (-1)^3 : by ring
                  ... = 8 * x^3 - 12 * x^2 + 6 * x - 1 : by ring

  have h₂ : 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x =
           5 * x^3 + 3 * x^3 - 3 * x^2 - 3 * x + x^2 - x - 1 - 10 * x^2 + 10 * x := by
    ring

  have h₃ : 5 * x^3 + 3 * x^3 + x^2 - 13 * x^2 + 7 * x - 1 = 8 * x^3 - 12 * x^2 + 6 * x - 1 := by
    ring

  rw [h₁, h₂, h₃]
  exact rfl

end identity_verification_l21_21519


namespace find_x_l21_21031

-- Define vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 5)
def c (x : ℝ) : ℝ × ℝ := (3, x)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Compute 8a - b
def sum_vec : ℝ × ℝ :=
  (8 * a.1 - b.1, 8 * a.2 - b.2)

-- Prove that x = 4 given condition
theorem find_x (x : ℝ) (h : dot_product sum_vec (c x) = 30) : x = 4 :=
by
  sorry

end find_x_l21_21031


namespace digits_difference_l21_21369

theorem digits_difference (X Y : ℕ) (h : 10 * X + Y - (10 * Y + X) = 90) : X - Y = 10 :=
by
  sorry

end digits_difference_l21_21369


namespace calculate_teena_speed_l21_21192

noncomputable def Teena_speed (t c t_ahead_in_1_5_hours : ℝ) : ℝ :=
  let distance_initial_gap := 7.5
  let coe_speed := 40
  let time_in_hours := 1.5
  let distance_coe_travels := coe_speed * time_in_hours
  let total_distance_teena_needs := distance_coe_travels + distance_initial_gap + t_ahead_in_1_5_hours
  total_distance_teena_needs / time_in_hours

theorem calculate_teena_speed :
  (Teena_speed 7.5 40 15) = 55 :=
  by
  -- skipped proof
  sorry

end calculate_teena_speed_l21_21192


namespace lemonade_second_intermission_l21_21870

theorem lemonade_second_intermission (first_intermission third_intermission total_lemonade second_intermission : ℝ) 
  (h1 : first_intermission = 0.25) 
  (h2 : third_intermission = 0.25) 
  (h3 : total_lemonade = 0.92) 
  (h4 : second_intermission = total_lemonade - (first_intermission + third_intermission)) : 
  second_intermission = 0.42 := 
by 
  sorry

end lemonade_second_intermission_l21_21870


namespace committee_form_count_l21_21855

def numWaysToFormCommittee (departments : Fin 4 → (ℕ × ℕ)) : ℕ :=
  let waysCase1 := 6 * 81 * 81
  let waysCase2 := 6 * 9 * 9 * 2 * 9 * 9
  waysCase1 + waysCase2

theorem committee_form_count (departments : Fin 4 → (ℕ × ℕ)) 
  (h : ∀ i, departments i = (3, 3)) :
  numWaysToFormCommittee departments = 48114 := 
by
  sorry

end committee_form_count_l21_21855
