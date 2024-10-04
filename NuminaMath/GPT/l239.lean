import Mathlib

namespace system_of_equations_soln_l239_239983

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l239_239983


namespace geo_seq_sum_l239_239379

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_sum (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 0 + a 1 = 30) (h4 : a 3 + a 4 = 120) :
  a 6 + a 7 = 480 :=
sorry

end geo_seq_sum_l239_239379


namespace min_n_of_inequality_l239_239835

theorem min_n_of_inequality : 
  ∀ (n : ℕ), (1 ≤ n) → (1 / n - 1 / (n + 1) < 1 / 10) → (n = 3 ∨ ∃ (k : ℕ), k ≥ 3 ∧ n = k) :=
by
  sorry

end min_n_of_inequality_l239_239835


namespace brenda_ends_with_12_skittles_l239_239213

def initial_skittles : ℕ := 7
def bought_skittles : ℕ := 8
def given_away_skittles : ℕ := 3

theorem brenda_ends_with_12_skittles :
  initial_skittles + bought_skittles - given_away_skittles = 12 := by
  sorry

end brenda_ends_with_12_skittles_l239_239213


namespace volume_ratio_of_cube_and_cuboid_l239_239124

theorem volume_ratio_of_cube_and_cuboid :
  let edge_length_meter := 1
  let edge_length_cm := edge_length_meter * 100 -- Convert meter to centimeters
  let cube_volume := edge_length_cm^3
  let cuboid_width := 50
  let cuboid_length := 50
  let cuboid_height := 20
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume = 20 * cuboid_volume := 
by
  sorry

end volume_ratio_of_cube_and_cuboid_l239_239124


namespace gain_percent_l239_239784

theorem gain_percent (C S : ℝ) (h : 50 * C = 15 * S) :
  (S > C) →
  ((S - C) / C * 100) = 233.33 := 
sorry

end gain_percent_l239_239784


namespace not_recurring_decimal_l239_239630

-- Definitions based on the provided conditions
def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ d m n : ℕ, d ≠ 0 ∧ (x * d) % 10 ^ n = m

-- Condition: 0.89898989
def number_0_89898989 : ℝ := 0.89898989

-- Proof statement to show 0.89898989 is not a recurring decimal
theorem not_recurring_decimal : ¬ is_recurring_decimal number_0_89898989 :=
sorry

end not_recurring_decimal_l239_239630


namespace even_square_is_even_l239_239243

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l239_239243


namespace find_k_l239_239845

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) (h2 : k ≠ 0) :
  k = 2 :=
sorry

end find_k_l239_239845


namespace harly_dogs_final_count_l239_239575

theorem harly_dogs_final_count (initial_dogs : ℕ) (adopted_percentage : ℕ) (returned_dogs : ℕ) (adoption_rate : adopted_percentage = 40) (initial_count : initial_dogs = 80) (returned_count : returned_dogs = 5) :
  initial_dogs - (initial_dogs * adopted_percentage / 100) + returned_dogs = 53 :=
by
  sorry

end harly_dogs_final_count_l239_239575


namespace distinct_integers_sum_l239_239261

theorem distinct_integers_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 357) : a + b + c + d = 28 :=
by
  sorry

end distinct_integers_sum_l239_239261


namespace participants_in_robbery_l239_239960

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l239_239960


namespace extra_chairs_added_l239_239069

theorem extra_chairs_added (rows cols total_chairs extra_chairs : ℕ) 
  (h1 : rows = 7) 
  (h2 : cols = 12) 
  (h3 : total_chairs = 95) 
  (h4 : extra_chairs = total_chairs - rows * cols) : 
  extra_chairs = 11 := by 
  sorry

end extra_chairs_added_l239_239069


namespace used_computer_lifespan_l239_239274

-- Problem statement
theorem used_computer_lifespan (cost_new : ℕ) (lifespan_new : ℕ) (cost_used : ℕ) (num_used : ℕ) (savings : ℕ) :
  cost_new = 600 →
  lifespan_new = 6 →
  cost_used = 200 →
  num_used = 2 →
  savings = 200 →
  ((cost_new - savings = num_used * cost_used) → (2 * (lifespan_new / 2) = 6) → lifespan_new / 2 = 3)
:= by
  intros
  sorry

end used_computer_lifespan_l239_239274


namespace negation_of_proposition_l239_239521

-- Conditions
variable {x : ℝ}

-- The proposition
def proposition : Prop := ∃ x : ℝ, Real.exp x > x

-- The proof problem: proving the negation of the proposition
theorem negation_of_proposition : (¬ proposition) ↔ ∀ x : ℝ, Real.exp x ≤ x := by
  sorry

end negation_of_proposition_l239_239521


namespace girls_in_class4_1_l239_239297

theorem girls_in_class4_1 (total_students grade: ℕ)
    (total_girls: ℕ)
    (students_class4_1: ℕ)
    (boys_class4_2: ℕ)
    (h1: total_students = 72)
    (h2: total_girls = 35)
    (h3: students_class4_1 = 36)
    (h4: boys_class4_2 = 19) :
    (total_girls - (total_students - students_class4_1 - boys_class4_2) = 18) :=
by
    sorry

end girls_in_class4_1_l239_239297


namespace pasta_needed_for_family_reunion_l239_239661

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end pasta_needed_for_family_reunion_l239_239661


namespace ce_ad_ratio_l239_239850

theorem ce_ad_ratio (CD DB AE EB CP PE : ℝ) (h₁ : CD / DB = 4 / 1) (h₂ : AE / EB = 2 / 3) :
  CP / PE = 10 := 
sorry

end ce_ad_ratio_l239_239850


namespace largest_multiple_of_12_neg_gt_neg_150_l239_239923

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end largest_multiple_of_12_neg_gt_neg_150_l239_239923


namespace simplify_expression_l239_239141

variable {s r : ℝ}

theorem simplify_expression :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := 
by
  sorry

end simplify_expression_l239_239141


namespace greatest_sum_consecutive_lt_400_l239_239619

noncomputable def greatest_sum_of_consecutive_integers (n : ℤ) : ℤ :=
if n * (n + 1) < 400 then n + (n + 1) else 0

theorem greatest_sum_consecutive_lt_400 : ∃ n : ℤ, n * (n + 1) < 400 ∧ greatest_sum_of_consecutive_integers n = 39 :=
by
  sorry

end greatest_sum_consecutive_lt_400_l239_239619


namespace males_only_in_band_l239_239758

theorem males_only_in_band
  (females_in_band : ℕ)
  (males_in_band : ℕ)
  (females_in_orchestra : ℕ)
  (males_in_orchestra : ℕ)
  (females_in_both : ℕ)
  (total_students : ℕ)
  (total_students_in_either : ℕ)
  (hf_in_band : females_in_band = 120)
  (hm_in_band : males_in_band = 90)
  (hf_in_orchestra : females_in_orchestra = 100)
  (hm_in_orchestra : males_in_orchestra = 130)
  (hf_in_both : females_in_both = 80)
  (h_total_students : total_students = 260) :
  total_students_in_either = 260 → 
  (males_in_band - (90 + 130 + 80 - 260 - 120)) = 30 :=
by
  intros h_total_students_in_either
  sorry

end males_only_in_band_l239_239758


namespace bank_robbery_participants_l239_239958

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l239_239958


namespace living_room_size_is_96_l239_239874

-- Define the total area of the apartment
def total_area : ℕ := 16 * 10

-- Define the number of units
def units : ℕ := 5

-- Define the size of one unit
def size_of_one_unit : ℕ := total_area / units

-- Define the size of the living room
def living_room_size : ℕ := size_of_one_unit * 3

-- Proving that the living room size is indeed 96 square feet
theorem living_room_size_is_96 : living_room_size = 96 := 
by
  -- not providing proof, thus using sorry
  sorry

end living_room_size_is_96_l239_239874


namespace largest_multiple_of_12_negation_greater_than_150_l239_239922

theorem largest_multiple_of_12_negation_greater_than_150 : 
  ∃ (k : ℤ), (k * 12 = 144) ∧ (-k * 12 > -150) :=
by
  -- Definitions and conditions
  let multiple_12 (k : ℤ) := k * 12
  have condition : -multiple_12 (-12) > -150 := by sorry
  existsi -12
  exact ⟨rfl, condition⟩

end largest_multiple_of_12_negation_greater_than_150_l239_239922


namespace no_five_consecutive_divisible_by_2025_l239_239253

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 : 
  ¬ ∃ (a : ℕ), (∀ (i : ℕ), i < 5 → 2025 ∣ seq (a + i)) := 
sorry

end no_five_consecutive_divisible_by_2025_l239_239253


namespace remainder_when_x_squared_div_30_l239_239666

theorem remainder_when_x_squared_div_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  (x^2) % 30 = 21 := 
by 
  sorry

end remainder_when_x_squared_div_30_l239_239666


namespace problem_proof_l239_239834

variable {a1 a2 b1 b2 b3 : ℝ}

theorem problem_proof 
  (h1 : ∃ d, -7 + d = a1 ∧ a1 + d = a2 ∧ a2 + d = -1)
  (h2 : ∃ r, -4 * r = b1 ∧ b1 * r = b2 ∧ b2 * r = b3 ∧ b3 * r = -1)
  (ha : a2 - a1 = 2)
  (hb : b2 = -2) :
  (a2 - a1) / b2 = -1 :=
by
  sorry

end problem_proof_l239_239834


namespace money_bounds_l239_239140

variables (c d : ℝ)

theorem money_bounds :
  (7 * c + d > 84) ∧ (5 * c - d = 35) → (c > 9.92 ∧ d > 14.58) :=
by
  intro h
  sorry

end money_bounds_l239_239140


namespace right_triangle_leg_length_l239_239264

theorem right_triangle_leg_length
  (a : ℕ) (c : ℕ) (h₁ : a = 8) (h₂ : c = 17) :
  ∃ b : ℕ, a^2 + b^2 = c^2 ∧ b = 15 :=
by
  sorry

end right_triangle_leg_length_l239_239264


namespace determine_x_l239_239821

theorem determine_x (x : ℚ) : 
  x + 5 / 8 = 2 + 3 / 16 - 2 / 3 → 
  x = 43 / 48 := 
by
  intro h
  sorry

end determine_x_l239_239821


namespace solve_for_x_l239_239181

theorem solve_for_x (x : ℤ) (h : x + 1 = 4) : x = 3 :=
sorry

end solve_for_x_l239_239181


namespace robbery_participants_l239_239965

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l239_239965


namespace find_y_coordinate_of_P_l239_239273

theorem find_y_coordinate_of_P (P Q : ℝ × ℝ)
  (h1 : ∀ x, y = 0.8 * x) -- line equation
  (h2 : P.1 = 4) -- x-coordinate of P
  (h3 : P = Q) -- P and Q are equidistant from the line
  : P.2 = 3.2 := sorry

end find_y_coordinate_of_P_l239_239273


namespace factorize_quadratic_example_l239_239824

theorem factorize_quadratic_example (x : ℝ) :
  4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 :=
by
  sorry

end factorize_quadratic_example_l239_239824


namespace rowing_speed_in_still_water_l239_239016

theorem rowing_speed_in_still_water (d t1 t2 : ℝ) 
  (h1 : d = 750) (h2 : t1 = 675) (h3 : t2 = 450) : 
  (d / t1 + (d / t2 - d / t1) / 2) = 1.389 := 
by
  sorry

end rowing_speed_in_still_water_l239_239016


namespace team_selection_correct_l239_239876

-- Define the basic sets and experienced members
def boys := finset.range 7  -- 7 boys
def girls := finset.range 10 -- 10 girls
def experienced_boy := 0
def experienced_girl := 0

-- Define the conditions for the selection
def n_select_boy := boys.erase experienced_boy
def n_select_girl := girls.erase experienced_girl

-- Binomial coefficient
def binomial (n k : ℕ) : ℕ := nat.choose n k

-- Definition of the number of ways to select 3 boys and 3 girls including the experienced members
def ways_to_select_team : ℕ :=
  1 * (binomial 6 2) * 1 * (binomial 9 2)

theorem team_selection_correct : 
  ways_to_select_team = 540 :=
by
  sorry

end team_selection_correct_l239_239876


namespace prove_g_of_f_g_l239_239745

noncomputable def f (x : Polynomial ℤ) : Polynomial ℤ := x^2 + 2 * x + 1 -- Example polynomial for f(x)
def g (x : Polynomial ℤ) : Polynomial ℤ := x^2 + 20 * x - 20 -- g(x) we want to prove

theorem prove_g_of_f_g (f g : Polynomial ℤ)
  (h₁ : ∀ x, f.eval (g.eval x) = (f.eval x) * (g.eval x))
  (h₂ : g.eval 3 = 50) :
  g = Polynomial.C(1) * x^2 + Polynomial.C(20) * x - Polynomial.C(20) := 
sorry

end prove_g_of_f_g_l239_239745


namespace sum_of_vertices_l239_239012

theorem sum_of_vertices (vertices_rectangle : ℕ) (vertices_pentagon : ℕ) 
  (h_rect : vertices_rectangle = 4) (h_pent : vertices_pentagon = 5) : 
  vertices_rectangle + vertices_pentagon = 9 :=
by
  sorry

end sum_of_vertices_l239_239012


namespace rhombus_area_2sqrt2_l239_239778

structure Rhombus (α : Type _) :=
  (side_length : ℝ)
  (angle : ℝ)

theorem rhombus_area_2sqrt2 (R : Rhombus ℝ) (h_side : R.side_length = 2) (h_angle : R.angle = 45) :
  ∃ A : ℝ, A = 2 * Real.sqrt 2 :=
by
  let A := 2 * Real.sqrt 2
  existsi A
  sorry

end rhombus_area_2sqrt2_l239_239778


namespace participants_in_robbery_l239_239962

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l239_239962


namespace problem_statement_l239_239589

def A : ℕ := 9 * 10 * 10 * 5
def B : ℕ := 9 * 10 * 10 * 2 / 3

theorem problem_statement : A + B = 5100 := by
  sorry

end problem_statement_l239_239589


namespace expenses_categorization_l239_239546

-- Define the sets of expenses and their categories
inductive Expense
| home_internet
| travel
| camera_rental
| domain_payment
| coffee_shop
| loan
| tax
| qualification_courses

open Expense

-- Define the conditions under which expenses can or cannot be economized
def isEconomizable : Expense → Prop
| home_internet        := true
| travel               := true
| camera_rental        := true
| domain_payment       := true
| coffee_shop          := true
| loan                 := false
| tax                  := false
| qualification_courses:= false

-- The main theorem statement
theorem expenses_categorization
  (expenses : list Expense) :
  (∀ e ∈ [home_internet, travel, camera_rental, domain_payment, coffee_shop], isEconomizable e) ∧
  (∀ e ∈ [loan, tax, qualification_courses], ¬ isEconomizable e) :=
by
  intros,
  simp [isEconomizable],
  exact ⟨
    (λ e he, by cases he; exact trivial),
    (λ e he, by cases he; tauto)⟩

end expenses_categorization_l239_239546


namespace quadrilateral_has_four_sides_and_angles_l239_239643

-- Define the conditions based on the characteristics of a quadrilateral
def quadrilateral (sides angles : Nat) : Prop :=
  sides = 4 ∧ angles = 4

-- Statement: Verify the property of a quadrilateral
theorem quadrilateral_has_four_sides_and_angles (sides angles : Nat) (h : quadrilateral sides angles) : sides = 4 ∧ angles = 4 :=
by
  -- We provide a proof by the characteristics of a quadrilateral
  sorry

end quadrilateral_has_four_sides_and_angles_l239_239643


namespace tax_percentage_first_tier_l239_239362

theorem tax_percentage_first_tier
  (car_price : ℝ)
  (total_tax : ℝ)
  (first_tier_level : ℝ)
  (second_tier_rate : ℝ)
  (first_tier_tax : ℝ)
  (T : ℝ)
  (h_car_price : car_price = 30000)
  (h_total_tax : total_tax = 5500)
  (h_first_tier_level : first_tier_level = 10000)
  (h_second_tier_rate : second_tier_rate = 0.15)
  (h_first_tier_tax : first_tier_tax = (T / 100) * first_tier_level) :
  T = 25 :=
by
  sorry

end tax_percentage_first_tier_l239_239362


namespace probability_heads_and_multiple_of_five_l239_239529

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def coin_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

def die_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

theorem probability_heads_and_multiple_of_five :
  coin_is_fair ∧ die_is_fair →
  (1 / 2) * (1 / 6) = (1 / 12) :=
by
  intro h
  sorry

end probability_heads_and_multiple_of_five_l239_239529


namespace no_such_rectangle_l239_239562

theorem no_such_rectangle (a b x y : ℝ) (ha : a < b)
  (hx : x < a / 2) (hy : y < a / 2)
  (h_perimeter : 2 * (x + y) = a + b)
  (h_area : x * y = (a * b) / 2) :
  false :=
sorry

end no_such_rectangle_l239_239562


namespace solve_equation_l239_239228

theorem solve_equation (x : ℝ) :
  (1 / (x ^ 2 + 14 * x - 10)) + (1 / (x ^ 2 + 3 * x - 10)) + (1 / (x ^ 2 - 16 * x - 10)) = 0
  ↔ (x = 5 ∨ x = -2 ∨ x = 2 ∨ x = -5) :=
sorry

end solve_equation_l239_239228


namespace middle_number_is_9_l239_239910

-- Define the problem conditions
variable (x y z : ℕ)

-- Lean proof statement
theorem middle_number_is_9 
  (h1 : x + y = 16)
  (h2 : x + z = 21)
  (h3 : y + z = 23)
  (h4 : x < y)
  (h5 : y < z) : y = 9 :=
by
  sorry

end middle_number_is_9_l239_239910


namespace group_A_percentage_l239_239110

/-!
In an examination, there are 100 questions divided into 3 groups A, B, and C such that each group contains at least one question. 
Each question in group A carries 1 mark, each question in group B carries 2 marks, and each question in group C carries 3 marks. 
It is known that:
- Group B contains 23 questions
- Group C contains 1 question.
Prove that the percentage of the total marks that the questions in group A carry is 60.8%.
-/

theorem group_A_percentage :
  ∃ (a b c : ℕ), b = 23 ∧ c = 1 ∧ (a + b + c = 100) ∧ ((a * 1) + (b * 2) + (c * 3) = 125) ∧ ((a : ℝ) / 125 * 100 = 60.8) :=
by
  sorry

end group_A_percentage_l239_239110


namespace solve_for_x_l239_239750

theorem solve_for_x (x : ℚ) : x^2 + 125 = (x - 15)^2 → x = 10 / 3 := by
  sorry

end solve_for_x_l239_239750


namespace find_number_of_even_numbers_l239_239894

-- Define the average of the first n even numbers
def average_of_first_n_even (n : ℕ) : ℕ :=
  (n * (1 + n)) / n

-- The given condition: The average is 21
def average_is_21 (n : ℕ) : Prop :=
  average_of_first_n_even n = 21

-- The theorem to prove: If the average is 21, then n = 20
theorem find_number_of_even_numbers (n : ℕ) (h : average_is_21 n) : n = 20 :=
  sorry

end find_number_of_even_numbers_l239_239894


namespace tank_A_height_l239_239893

theorem tank_A_height :
  ∀ (h_B : ℝ) (C_A C_B : ℝ) (V_ratio : ℝ),
    C_A = 9 →
    C_B = 10 →
    h_B = 9 →
    V_ratio = 0.9000000000000001 →
    let r_A := C_A / (2 * π),
        r_B := C_B / (2 * π),
        V_A := π * r_A^2 * h_A,
        V_B := π * r_B^2 * h_B
    in V_A = V_ratio * V_B →
       h_A = 8.1 :=
by
  intros h_B C_A C_B V_ratio hCAB hCBB hBB hVR
  simp [hCAB, hCBB, hBB, hVR]
  sorry

end tank_A_height_l239_239893


namespace units_digit_of_8_pow_120_l239_239013

theorem units_digit_of_8_pow_120 : (8 ^ 120) % 10 = 6 := 
by
  sorry

end units_digit_of_8_pow_120_l239_239013


namespace area_of_triangle_ABC_l239_239369

def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (2, -7)

theorem area_of_triangle_ABC : 
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := |v.1 * w.2 - v.2 * w.1|
  let triangle_area := parallelogram_area / 2
  triangle_area = 15 :=
by
  sorry

end area_of_triangle_ABC_l239_239369


namespace complex_subtraction_l239_239249

open Complex

def z1 : ℂ := 3 + 4 * I
def z2 : ℂ := 1 + I

theorem complex_subtraction : z1 - z2 = 2 + 3 * I := by
  sorry

end complex_subtraction_l239_239249


namespace decreasing_cubic_function_l239_239147

theorem decreasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → a ≤ 0 :=
sorry

end decreasing_cubic_function_l239_239147


namespace number_of_even_factors_of_n_l239_239691

noncomputable def n := 2^3 * 3^2 * 7^3

theorem number_of_even_factors_of_n : 
  (∃ (a : ℕ), (1 ≤ a ∧ a ≤ 3)) ∧ 
  (∃ (b : ℕ), (0 ≤ b ∧ b ≤ 2)) ∧ 
  (∃ (c : ℕ), (0 ≤ c ∧ c ≤ 3)) → 
  (even_nat_factors_count : ℕ) = 36 :=
by
  sorry

end number_of_even_factors_of_n_l239_239691


namespace ant_trip_ratio_l239_239616

theorem ant_trip_ratio (A B : ℕ) (x c : ℕ) (h1 : A * x = c) (h2 : B * (3 / 2 * x) = 3 * c) :
  B = 2 * A :=
by
  sorry

end ant_trip_ratio_l239_239616


namespace problem1_problem2_l239_239218

theorem problem1 : -20 - (-8) + (-4) = -16 := by
  sorry

theorem problem2 : -1^3 * (-2)^2 / (4 / 3 : ℚ) + |5 - 8| = 0 := by
  sorry

end problem1_problem2_l239_239218


namespace baba_yaga_departure_and_speed_l239_239040

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end baba_yaga_departure_and_speed_l239_239040


namespace geometric_sequence_problem_l239_239728

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 * a 7 = 8)
  (h2 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1)):
  a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l239_239728


namespace inequality_proof_l239_239833

noncomputable def given_condition_1 (a b c u : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ x, (a * x^2 - b * x + c = 0)) ∧
  a * u^2 - b * u + c ≤ 0

noncomputable def given_condition_2 (A B C v : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ (∃ x, (A * x^2 - B * x + C = 0)) ∧
  A * v^2 - B * v + C ≤ 0

theorem inequality_proof (a b c A B C u v : ℝ) (h1 : given_condition_1 a b c u) (h2 : given_condition_2 A B C v) :
  (a * u + A * v) * (c / u + C / v) ≤ (b + B) ^ 2 / 4 :=
by
    sorry

end inequality_proof_l239_239833


namespace height_difference_is_correct_l239_239125

-- Define the heights of the trees as rational numbers.
def maple_tree_height : ℚ := 10 + 1 / 4
def spruce_tree_height : ℚ := 14 + 1 / 2

-- Prove that the spruce tree is 19 3/4 feet taller than the maple tree.
theorem height_difference_is_correct :
  spruce_tree_height - maple_tree_height = 19 + 3 / 4 := 
sorry

end height_difference_is_correct_l239_239125


namespace greening_investment_equation_l239_239195

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end greening_investment_equation_l239_239195


namespace words_written_first_two_hours_l239_239426

def essay_total_words : ℕ := 1200
def words_per_hour_first_two_hours (W : ℕ) : ℕ := 2 * W
def words_per_hour_next_two_hours : ℕ := 2 * 200

theorem words_written_first_two_hours (W : ℕ) (h : words_per_hour_first_two_hours W + words_per_hour_next_two_hours = essay_total_words) : W = 400 := 
by 
  sorry

end words_written_first_two_hours_l239_239426


namespace janet_more_siblings_than_carlos_l239_239423

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end janet_more_siblings_than_carlos_l239_239423


namespace part1_part2_l239_239867

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 2 :=
by
  sorry

theorem part2 (x : ℝ) : (∀ b : ℝ, b ≠ 0 → f x ≥ (|2 * b + 1| - |1 - b|) / |b|) → (x ≤ -1.5 ∨ x ≥ 1.5) :=
by
  sorry

end part1_part2_l239_239867


namespace depth_of_well_l239_239028

noncomputable def volume_of_cylinder (radius : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * radius^2 * depth

theorem depth_of_well (volume depth : ℝ) (r : ℝ) : 
  r = 1 ∧ volume = 25.132741228718345 ∧ 2 * r = 2 → depth = 8 :=
by
  intros h
  sorry

end depth_of_well_l239_239028


namespace part_one_part_two_l239_239385

-- 1. Prove that 1 + 2x^4 >= 2x^3 + x^2 for all real numbers x
theorem part_one (x : ℝ) : 1 + 2 * x^4 ≥ 2 * x^3 + x^2 := sorry

-- 2. Given x + 2y + 3z = 6, prove that x^2 + y^2 + z^2 ≥ 18 / 7
theorem part_two (x y z : ℝ) (h : x + 2 * y + 3 * z = 6) : x^2 + y^2 + z^2 ≥ 18 / 7 := sorry

end part_one_part_two_l239_239385


namespace students_chemistry_or_physics_not_both_l239_239671

variables (total_chemistry total_both total_physics_only : ℕ)

theorem students_chemistry_or_physics_not_both
  (h1 : total_chemistry = 30)
  (h2 : total_both = 15)
  (h3 : total_physics_only = 18) :
  total_chemistry - total_both + total_physics_only = 33 :=
by
  sorry

end students_chemistry_or_physics_not_both_l239_239671


namespace P_X_gt_3_l239_239386

noncomputable section

open ProbabilityTheory

-- Define normal distribution
def normalDist (μ σ : ℝ) : Measure ℝ := Measure.normal μ σ

-- Define X as a random variable
axiom X : ℝ → ℝ

-- Given assumptions
axiom X_normal : ∀ t, ∫ x in Set.Iic (X t), x = real_density_normal 2 1
axiom P_X_gt_1 : P (λ x, X x > 1) = 0.8413

-- The proof goal
theorem P_X_gt_3 : P (λ x, X x > 3) = 0.1587 :=
  sorry

end P_X_gt_3_l239_239386


namespace least_positive_integer_l239_239620

open Nat

theorem least_positive_integer (n : ℕ) (h1 : n ≡ 2 [MOD 5]) (h2 : n ≡ 2 [MOD 4]) (h3 : n ≡ 0 [MOD 3]) : n = 42 :=
sorry

end least_positive_integer_l239_239620


namespace anie_days_to_finish_task_l239_239772

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_finish_task_l239_239772


namespace Vitya_catchup_mom_in_5_l239_239493

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l239_239493


namespace shanna_tomato_ratio_l239_239285

-- Define the initial conditions
def initial_tomato_plants : ℕ := 6
def initial_eggplant_plants : ℕ := 2
def initial_pepper_plants : ℕ := 4
def pepper_plants_died : ℕ := 1
def vegetables_per_plant : ℕ := 7
def total_vegetables_harvested : ℕ := 56

-- Define the number of tomato plants that died
def tomato_plants_died (total_vegetables : ℕ) (veg_per_plant : ℕ) (initial_tomato : ℕ) 
  (initial_eggplant : ℕ) (initial_pepper : ℕ) (pepper_died : ℕ) : ℕ :=
  let surviving_plants := total_vegetables / veg_per_plant
  let surviving_pepper := initial_pepper - pepper_died
  let surviving_tomato := surviving_plants - (initial_eggplant + surviving_pepper)
  initial_tomato - surviving_tomato

-- Define the ratio
def ratio_tomato_plants_died_to_initial (tomato_died : ℕ) (initial_tomato : ℕ) : ℚ :=
  (tomato_died : ℚ) / (initial_tomato : ℚ)

theorem shanna_tomato_ratio :
  ratio_tomato_plants_died_to_initial (tomato_plants_died total_vegetables_harvested vegetables_per_plant 
    initial_tomato_plants initial_eggplant_plants initial_pepper_plants pepper_plants_died) initial_tomato_plants 
  = 1 / 2 := by
  sorry

end shanna_tomato_ratio_l239_239285


namespace sin_gt_cos_range_l239_239115

theorem sin_gt_cos_range (x : ℝ) : 
  0 < x ∧ x < 2 * Real.pi → (Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4)) := by
  sorry

end sin_gt_cos_range_l239_239115


namespace initial_processing_capacity_l239_239331

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end initial_processing_capacity_l239_239331


namespace age_of_15th_student_l239_239625

theorem age_of_15th_student 
  (avg_age_all : ℕ → ℕ → ℕ)
  (avg_age : avg_age_all 15 15 = 15)
  (avg_age_4 : avg_age_all 4 14 = 14)
  (avg_age_10 : avg_age_all 10 16 = 16) : 
  ∃ age15 : ℕ, age15 = 9 := 
by
  sorry

end age_of_15th_student_l239_239625


namespace license_plate_calculation_l239_239692

def license_plate_count : ℕ :=
  let letter_choices := 26^3
  let first_digit_choices := 5
  let remaining_digit_combinations := 5 * 5
  letter_choices * first_digit_choices * remaining_digit_combinations

theorem license_plate_calculation :
  license_plate_count = 455625 :=
by
  sorry

end license_plate_calculation_l239_239692


namespace xiaoming_additional_games_l239_239511

variable (total_games games_won target_percentage : ℕ)

theorem xiaoming_additional_games :
  total_games = 20 →
  games_won = 95 * total_games / 100 →
  target_percentage = 96 →
  ∃ additional_games, additional_games = 5 ∧
    (games_won + additional_games) / (total_games + additional_games) = target_percentage / 100 :=
by
  sorry

end xiaoming_additional_games_l239_239511


namespace binom_n_n_minus_2_l239_239503

theorem binom_n_n_minus_2 (n : ℕ) (h : n > 0) : nat.choose n (n-2) = n * (n-1) / 2 :=
by sorry

end binom_n_n_minus_2_l239_239503


namespace red_toys_removed_l239_239917

theorem red_toys_removed (R W : ℕ) (h1 : R + W = 134) (h2 : 2 * W = 88) (h3 : R - 2 * W / 2 = 88) : R - 88 = 2 :=
by {
  sorry
}

end red_toys_removed_l239_239917


namespace solve_quadratic_l239_239890

theorem solve_quadratic : ∀ (x : ℝ), x^2 - 5 * x + 1 = 0 →
  (x = (5 + Real.sqrt 21) / 2) ∨ (x = (5 - Real.sqrt 21) / 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l239_239890


namespace tori_needs_more_correct_answers_l239_239854

theorem tori_needs_more_correct_answers :
  let total_questions := 80
  let arithmetic_questions := 20
  let algebra_questions := 25
  let geometry_questions := 35
  let arithmetic_correct := 0.60 * arithmetic_questions
  let algebra_correct := Float.round (0.50 * algebra_questions)
  let geometry_correct := Float.round (0.70 * geometry_questions)
  let correct_answers := arithmetic_correct + algebra_correct + geometry_correct
  let passing_percentage := 0.65
  let required_correct := passing_percentage * total_questions
-- assertion
  required_correct - correct_answers = 2 := 
by 
  sorry

end tori_needs_more_correct_answers_l239_239854


namespace odometer_problem_l239_239545

theorem odometer_problem
  (a b c : ℕ) -- a, b, c are natural numbers
  (h1 : 1 ≤ a) -- condition (a ≥ 1)
  (h2 : a + b + c ≤ 7) -- condition (a + b + c ≤ 7)
  (h3 : 99 * (c - a) % 55 = 0) -- 99(c - a) must be divisible by 55
  (h4 : 100 * a + 10 * b + c < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  (h5 : 100 * c + 10 * b + a < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  : a^2 + b^2 + c^2 = 37 := sorry

end odometer_problem_l239_239545


namespace simple_interest_rate_l239_239293

theorem simple_interest_rate (P T A R : ℝ) (hT : T = 15) (hA : A = 4 * P)
  (hA_simple_interest : A = P + (P * R * T / 100)) : R = 20 :=
by
  sorry

end simple_interest_rate_l239_239293


namespace rational_x_sqrt3_x_sq_sqrt3_l239_239227

theorem rational_x_sqrt3_x_sq_sqrt3 (x : ℝ) : (∃ a b : ℚ, x + real.sqrt 3 = a ∧ x^2 + real.sqrt 3 = b) ↔ x = (1 / 2) - real.sqrt 3 :=
by
  sorry

end rational_x_sqrt3_x_sq_sqrt3_l239_239227


namespace masha_comb_teeth_count_l239_239276

theorem masha_comb_teeth_count (katya_teeth : ℕ) (masha_to_katya_ratio : ℕ) 
  (katya_teeth_eq : katya_teeth = 11) 
  (masha_to_katya_ratio_eq : masha_to_katya_ratio = 5) : 
  ∃ masha_teeth : ℕ, masha_teeth = 53 :=
by
  have katya_segments := 2 * katya_teeth - 1
  have masha_segments := masha_to_katya_ratio * katya_segments
  let masha_teeth := (masha_segments + 1) / 2
  use masha_teeth
  have masha_teeth_eq := (2 * masha_teeth - 1 = 105)
  sorry

end masha_comb_teeth_count_l239_239276


namespace problem_statement_l239_239846

variables (x y : ℚ)

theorem problem_statement 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 105) : 
  x^2 - y^2 = 8 / 1575 :=
sorry

end problem_statement_l239_239846


namespace molecular_weight_one_mole_of_AlPO4_l239_239506

theorem molecular_weight_one_mole_of_AlPO4
  (molecular_weight_4_moles : ℝ)
  (h : molecular_weight_4_moles = 488) :
  molecular_weight_4_moles / 4 = 122 :=
by
  sorry

end molecular_weight_one_mole_of_AlPO4_l239_239506


namespace sqrt_expression_is_869_l239_239815

theorem sqrt_expression_is_869 :
  (31 * 30 * 29 * 28 + 1) = 869 := 
sorry

end sqrt_expression_is_869_l239_239815


namespace victor_draw_order_count_l239_239617

-- Definitions based on the problem conditions
def num_piles : ℕ := 3
def num_cards_per_pile : ℕ := 3
def total_cards : ℕ := num_piles * num_cards_per_pile

-- The cardinality of the set of valid sequences where within each pile cards must be drawn in order
def valid_sequences_count : ℕ :=
  Nat.factorial total_cards / (Nat.factorial num_cards_per_pile ^ num_piles)

-- Now we state the problem: proving the valid sequences count is 1680
theorem victor_draw_order_count :
  valid_sequences_count = 1680 :=
by
  sorry

end victor_draw_order_count_l239_239617


namespace number_of_n_not_divisible_by_2_or_3_l239_239160

theorem number_of_n_not_divisible_by_2_or_3 :
  let count := (999 - (499 + 333 - 166))
  in count = 333 :=
sorry

end number_of_n_not_divisible_by_2_or_3_l239_239160


namespace sum_of_four_triangles_l239_239800

theorem sum_of_four_triangles (x y : ℝ) (h1 : 3 * x + 2 * y = 27) (h2 : 2 * x + 3 * y = 23) : 4 * y = 12 :=
sorry

end sum_of_four_triangles_l239_239800


namespace smallest_num_is_1113805958_l239_239929

def smallest_num (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧ (n + 5) % 73 = 0 ∧ (n + 5) % 101 = 0 ∧ (n + 5) % 89 = 0

theorem smallest_num_is_1113805958 : ∃ n, smallest_num n ∧ n = 1113805958 :=
by
  use 1113805958
  unfold smallest_num
  simp
  sorry

end smallest_num_is_1113805958_l239_239929


namespace number_of_divisibles_by_7_l239_239089

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l239_239089


namespace carla_zoo_l239_239048

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l239_239048


namespace second_number_is_12_l239_239186

noncomputable def expression := (26.3 * 12 * 20) / 3 + 125

theorem second_number_is_12 :
  expression = 2229 → 12 = 12 :=
by sorry

end second_number_is_12_l239_239186


namespace solve_trig_problem_l239_239551

open Real

theorem solve_trig_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := sorry

end solve_trig_problem_l239_239551


namespace number_of_children_l239_239749

def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6
def number_of_adults := 2
def total_cost := 77

theorem number_of_children : 
  ∃ (x : ℕ), cost_of_child_ticket * x + cost_of_adult_ticket * number_of_adults = total_cost ∧ x = 3 :=
by
  sorry

end number_of_children_l239_239749


namespace alcohol_quantity_in_mixture_l239_239182

theorem alcohol_quantity_in_mixture 
  (A W : ℝ)
  (h1 : A / W = 4 / 3)
  (h2 : A / (W + 4) = 4 / 5)
  : A = 8 :=
sorry

end alcohol_quantity_in_mixture_l239_239182


namespace solution_set_is_circle_with_exclusion_l239_239989

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l239_239989


namespace y_star_definition_l239_239233

def y_star (y : Real) : Real := y - 1

theorem y_star_definition (y : Real) : (5 : Real) - y_star 5 = 1 :=
  by sorry

end y_star_definition_l239_239233


namespace segment_area_l239_239531

noncomputable def area_segment_above_triangle (a b c : ℝ) (triangle_area : ℝ) (y : ℝ) :=
  let ellipse_area := Real.pi * a * b
  ellipse_area - triangle_area

theorem segment_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) :
  let y := (4 * Real.sqrt 2) / 3
  let triangle_area := (1 / 2) * (2 * (b - y))
  area_segment_above_triangle a b c triangle_area y = 6 * Real.pi - 2 + (4 * Real.sqrt 2) / 3 := by
  sorry

end segment_area_l239_239531


namespace fraction_meaningful_l239_239776

theorem fraction_meaningful (x : ℝ) : (¬ (x - 2 = 0)) ↔ (x ≠ 2) :=
by
  sorry

end fraction_meaningful_l239_239776


namespace solution_set_is_circle_with_exclusion_l239_239988

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l239_239988


namespace mary_carrots_correct_l239_239888

def sandy_carrots := 8
def total_carrots := 14

def mary_carrots := total_carrots - sandy_carrots

theorem mary_carrots_correct : mary_carrots = 6 := by
  unfold mary_carrots
  unfold total_carrots
  unfold sandy_carrots
  sorry

end mary_carrots_correct_l239_239888


namespace survey_respondents_l239_239751

theorem survey_respondents
  (X Y Z : ℕ) 
  (h1 : X = 360) 
  (h2 : X * 4 = Y * 9) 
  (h3 : X * 3 = Z * 9) : 
  X + Y + Z = 640 :=
by
  sorry

end survey_respondents_l239_239751


namespace expected_matches_l239_239185

-- Definitions based on problem conditions
def matchbox : Type := Fin 60 -- Each matchbox contains 60 matches.
def matchboxes (n : ℕ) : list matchbox := list.replicate n (Fin 60)

-- Random selection condition
constant probability_of_selecting_either_box : ℚ := 0.5

-- Expected value theorem
theorem expected_matches (M N : ℕ) (hM : M = 60) (hN : N = 60) (P : ℚ) (hP : P = probability_of_selecting_either_box) :
  ∃ μ : ℚ, μ = 7.795 ∧ 
  (∃ (X : ℕ → ℕ), (∀ k : ℕ, k ≤ 60 → (X k = k * P * (binom N k) / (2^(M+k))) ∧ 
  (∑ k in finset.range (N + 1), (N - k) * (X k * P * (binom N k) / (2^(M+k)))) = 7.795)) := sorry

end expected_matches_l239_239185


namespace angle_between_AD_and_BC_l239_239108

variables {a b c : ℝ} 
variables {θ : ℝ}
variables {α β γ δ ε ζ : ℝ} -- representing the angles

-- Conditions of the problem
def conditions (a b c : ℝ) (α β γ δ ε ζ : ℝ) : Prop :=
  (α + β + γ = 180) ∧ (δ + ε + ζ = 180) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Definition of the theorem to prove the angle between AD and BC
theorem angle_between_AD_and_BC
  (a b c : ℝ) (α β γ δ ε ζ : ℝ)
  (h : conditions a b c α β γ δ ε ζ) :
  θ = Real.arccos ((|b^2 - c^2|) / a^2) :=
sorry

end angle_between_AD_and_BC_l239_239108


namespace num_divisible_by_7_200_to_400_l239_239094

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l239_239094


namespace participants_in_robbery_l239_239961

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l239_239961


namespace no_odd_m_solution_l239_239287

theorem no_odd_m_solution : ∀ (m n : ℕ), 0 < m → 0 < n → (5 * n = m * n - 3 * m) → ¬ Odd m :=
by
  intros m n hm hn h_eq
  sorry

end no_odd_m_solution_l239_239287


namespace sales_overlap_l239_239581

-- Define the conditions
def bookstore_sale_days : List ℕ := [2, 6, 10, 14, 18, 22, 26, 30]
def shoe_store_sale_days : List ℕ := [1, 8, 15, 22, 29]

-- Define the statement to prove
theorem sales_overlap : (bookstore_sale_days ∩ shoe_store_sale_days).length = 1 := 
by
  sorry

end sales_overlap_l239_239581


namespace octal_742_to_decimal_l239_239789

theorem octal_742_to_decimal : (7 * 8^2 + 4 * 8^1 + 2 * 8^0 = 482) :=
by
  sorry

end octal_742_to_decimal_l239_239789


namespace abs_neg_2023_l239_239348

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l239_239348


namespace total_weight_of_fish_is_correct_l239_239861

noncomputable def totalWeightInFirstTank := 15 * 0.08 + 12 * 0.05

noncomputable def totalWeightInSecondTank := 2 * 15 * 0.08 + 3 * 12 * 0.05

noncomputable def totalWeightInThirdTank := 3 * 15 * 0.08 + 2 * 12 * 0.05 + 5 * 0.14

noncomputable def totalWeightAllTanks := totalWeightInFirstTank + totalWeightInSecondTank + totalWeightInThirdTank

theorem total_weight_of_fish_is_correct : 
  totalWeightAllTanks = 11.5 :=
by         
  sorry

end total_weight_of_fish_is_correct_l239_239861


namespace lawnmower_blades_l239_239199

theorem lawnmower_blades (B : ℤ) (h : 8 * B + 7 = 39) : B = 4 :=
by 
  sorry

end lawnmower_blades_l239_239199


namespace geometric_sequence_a4_l239_239406

theorem geometric_sequence_a4 {a_2 a_6 a_4 : ℝ} 
  (h1 : ∃ a_1 r : ℝ, a_2 = a_1 * r ∧ a_6 = a_1 * r^5) 
  (h2 : a_2 * a_6 = 64) 
  (h3 : a_2 = a_1 * r)
  (h4 : a_6 = a_1 * r^5)
  : a_4 = 8 :=
by
  sorry

end geometric_sequence_a4_l239_239406


namespace gasoline_price_increase_percentage_l239_239768

theorem gasoline_price_increase_percentage : 
  ∀ (highest_price lowest_price : ℝ), highest_price = 24 → lowest_price = 18 → 
  ((highest_price - lowest_price) / lowest_price) * 100 = 33.33 :=
by
  intros highest_price lowest_price h_highest h_lowest
  rw [h_highest, h_lowest]
  -- To be completed in the proof
  sorry

end gasoline_price_increase_percentage_l239_239768


namespace total_swim_distance_five_weeks_total_swim_time_five_weeks_l239_239737

-- Definitions of swim distances and times based on Jasmine's routine 
def monday_laps : ℕ := 10
def tuesday_laps : ℕ := 15
def tuesday_aerobics_time : ℕ := 20
def wednesday_laps : ℕ := 12
def wednesday_time_per_lap : ℕ := 2
def thursday_laps : ℕ := 18
def friday_laps : ℕ := 20

-- Proving total swim distance for five weeks
theorem total_swim_distance_five_weeks : (5 * (monday_laps + tuesday_laps + wednesday_laps + thursday_laps + friday_laps)) = 375 := 
by 
  sorry

-- Proving total swim time for five weeks (partially solvable)
theorem total_swim_time_five_weeks : (5 * (tuesday_aerobics_time + wednesday_laps * wednesday_time_per_lap)) = 220 := 
by 
  sorry

end total_swim_distance_five_weeks_total_swim_time_five_weeks_l239_239737


namespace even_square_is_even_l239_239245

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l239_239245


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l239_239468

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l239_239468


namespace pond_to_field_ratio_l239_239609

theorem pond_to_field_ratio 
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l = 28)
  (side_pond : ℝ := 7) 
  (A_pond : ℝ := side_pond ^ 2) 
  (A_field : ℝ := l * w):
  (A_pond / A_field) = 1 / 8 :=
by
  sorry

end pond_to_field_ratio_l239_239609


namespace z_neq_5_for_every_k_l239_239820

theorem z_neq_5_for_every_k (z : ℕ) (h₁ : z = 5) :
  ¬ (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (∃ m, n ^ 9 % 10 ^ k = z * (10 ^ m))) :=
by
  intro h
  sorry

end z_neq_5_for_every_k_l239_239820


namespace probability_divisible_by_5_l239_239462

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l239_239462


namespace initial_processing_capacity_l239_239332

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end initial_processing_capacity_l239_239332


namespace triangle_b_value_triangle_area_value_l239_239731

noncomputable def triangle_b (a : ℝ) (cosA : ℝ) : ℝ :=
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := cosA
  (a * sinB) / sinA

noncomputable def triangle_area (a b c : ℝ) (sinC : ℝ) : ℝ :=
  0.5 * a * b * sinC

-- Given conditions
variable (A B : ℝ) (a : ℝ := 3) (cosA : ℝ := Real.sqrt 6 / 3) (B := A + Real.pi / 2)

-- The assertions to prove
theorem triangle_b_value :
  triangle_b a cosA = 3 * Real.sqrt 2 :=
sorry

theorem triangle_area_value :
  triangle_area 3 (3 * Real.sqrt 2) 1 (1 / 3) = (3 * Real.sqrt 2) / 2 :=
sorry

end triangle_b_value_triangle_area_value_l239_239731


namespace smallest_integer_divisible_l239_239294

theorem smallest_integer_divisible:
  ∃ n : ℕ, n > 1 ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 6 = 1) ∧ n = 61 :=
by
  sorry

end smallest_integer_divisible_l239_239294


namespace Vitya_catches_mother_l239_239497

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l239_239497


namespace ellipse_equation_and_m_value_l239_239999

variable {a b : ℝ}
variable (e : ℝ) (F : ℝ × ℝ) (h1 : e = Real.sqrt 2 / 2) (h2 : F = (1, 0))

theorem ellipse_equation_and_m_value (h3 : a > b) (h4 : b > 0) 
  (h5 : (x y : ℝ) → (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 → (x - 1) ^ 2 + y ^ 2 = 1) :
  (a = Real.sqrt 2 ∧ b = 1) ∧
  (∀ m : ℝ, (y = x + m) → 
  ((∃ A B : ℝ × ℝ, A = (x₁, x₁ + m) ∧ B = (x₂, x₂ + m) ∧
  (x₁ ^ 2) / 2 + (x₁ + m) ^ 2 = 1 ∧ (x₂ ^ 2) / 2 + (x₂ + m) ^ 2 = 1 ∧
  x₁ * x₂ + (x₁ + m) * (x₂ + m) = -1) ↔ m = Real.sqrt 3 / 3 ∨ m = - Real.sqrt 3 / 3))
  :=
sorry

end ellipse_equation_and_m_value_l239_239999


namespace vitya_catch_up_l239_239484

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l239_239484


namespace FB_length_correct_l239_239730

-- Define a structure for the problem context
structure Triangle (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] where
  AB : ℝ
  CD : ℝ
  AE : ℝ
  altitude_CD : C -> (A -> B -> Prop)  -- CD is an altitude to AB
  altitude_AE : E -> (B -> C -> Prop)  -- AE is an altitude to BC
  angle_bisector_AF : F -> (B -> C -> Prop)  -- AF is the angle bisector of ∠BAC intersecting BC at F
  intersect_AF_BC_at_F : (F -> B -> Prop)  -- AF intersects BC at F

noncomputable def length_of_FB (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : ℝ := 
  2  -- From given conditions and conclusion

-- The main theorem to prove
theorem FB_length_correct (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : 
  t.AB = 8 ∧ t.CD = 3 ∧ t.AE = 4 → length_of_FB A B C D E F t = 2 :=
by
  intro h
  obtain ⟨AB_eq, CD_eq, AE_eq⟩ := h
  sorry

end FB_length_correct_l239_239730


namespace max_of_2xy_l239_239375

theorem max_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 2 * x * y ≤ 8 :=
by
  sorry

end max_of_2xy_l239_239375


namespace max_value_of_n_l239_239078

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_cond : a 11 / a 10 < -1)
  (h_maximum : ∃ N, ∀ n > N, S n ≤ S N) :
  ∃ N, S N > 0 ∧ ∀ m, S m > 0 → m ≤ N :=
by
  sorry

end max_value_of_n_l239_239078


namespace range_of_b_l239_239246

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a) / (2^x + 1)
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := Real.log (x^2 - b)

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x1 x2 : ℝ, f x1 a ≤ g x2 b) → b ≤ -Real.exp 1 :=
by
  sorry

end range_of_b_l239_239246


namespace acute_angle_sine_l239_239242
--import Lean library

-- Define the problem conditions and statement
theorem acute_angle_sine (a : ℝ) (h1 : 0 < a) (h2 : a < π / 2) (h3 : Real.sin a = 0.6) :
  π / 6 < a ∧ a < π / 4 :=
by 
  sorry

end acute_angle_sine_l239_239242


namespace Robinson_age_l239_239256

theorem Robinson_age (R : ℕ)
    (brother : ℕ := R + 2)
    (sister : ℕ := R + 6)
    (mother : ℕ := R + 20)
    (avg_age_yesterday : ℕ := 39)
    (total_age_yesterday : ℕ := 156)
    (eq : (R - 1) + (brother - 1) + (sister - 1) + (mother - 1) = total_age_yesterday) :
  R = 33 :=
by
  sorry

end Robinson_age_l239_239256


namespace correct_propositions_count_l239_239154

theorem correct_propositions_count (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0) ∧ -- original proposition
  (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0) ∧ -- converse proposition
  (¬(x ≠ 0 ∨ y ≠ 0) ∨ x^2 + y^2 = 0) ∧ -- negation proposition
  (¬(x^2 + y^2 = 0) ∨ x ≠ 0 ∨ y ≠ 0) -- inverse proposition
  := by
  sorry

end correct_propositions_count_l239_239154


namespace sum_of_a_b_either_1_or_neg1_l239_239288

theorem sum_of_a_b_either_1_or_neg1 (a b : ℝ) (h1 : a + a = 0) (h2 : b * b = 1) : a + b = 1 ∨ a + b = -1 :=
by {
  sorry
}

end sum_of_a_b_either_1_or_neg1_l239_239288


namespace Iggy_miles_on_Monday_l239_239396

theorem Iggy_miles_on_Monday 
  (tuesday_miles : ℕ)
  (wednesday_miles : ℕ)
  (thursday_miles : ℕ)
  (friday_miles : ℕ)
  (monday_minutes : ℕ)
  (pace : ℕ)
  (total_hours : ℕ)
  (total_minutes : ℕ)
  (total_tuesday_to_friday_miles : ℕ)
  (total_tuesday_to_friday_minutes : ℕ) :
  tuesday_miles = 4 →
  wednesday_miles = 6 →
  thursday_miles = 8 →
  friday_miles = 3 →
  pace = 10 →
  total_hours = 4 →
  total_minutes = total_hours * 60 →
  total_tuesday_to_friday_miles = tuesday_miles + wednesday_miles + thursday_miles + friday_miles →
  total_tuesday_to_friday_minutes = total_tuesday_to_friday_miles * pace →
  monday_minutes = total_minutes - total_tuesday_to_friday_minutes →
  (monday_minutes / pace) = 3 := sorry

end Iggy_miles_on_Monday_l239_239396


namespace sum_of_roots_l239_239352

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l239_239352


namespace vitya_catch_up_l239_239482

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l239_239482


namespace share_difference_l239_239530

theorem share_difference (x : ℕ) (p q r : ℕ) 
  (h1 : 3 * x = p) 
  (h2 : 7 * x = q) 
  (h3 : 12 * x = r) 
  (h4 : q - p = 2800) : 
  r - q = 3500 := by {
  sorry
}

end share_difference_l239_239530


namespace simplify_and_evaluate_l239_239602

theorem simplify_and_evaluate (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3 * b) * (2 * a - b) - 2 * (a - b)^2 = -23 := by
  sorry

end simplify_and_evaluate_l239_239602


namespace dorchester_puppy_count_l239_239223

/--
  Dorchester works at a puppy wash. He is paid $40 per day + $2.25 for each puppy he washes.
  On Wednesday, Dorchester earned $76. Prove that Dorchester washed 16 puppies that day.
-/
theorem dorchester_puppy_count
  (total_earnings : ℝ)
  (base_pay : ℝ)
  (pay_per_puppy : ℝ)
  (puppies_washed : ℕ)
  (h1 : total_earnings = 76)
  (h2 : base_pay = 40)
  (h3 : pay_per_puppy = 2.25) :
  total_earnings - base_pay = (puppies_washed : ℝ) * pay_per_puppy :=
sorry

example :
  dorchester_puppy_count 76 40 2.25 16 := by
  rw [dorchester_puppy_count, sub_self, mul_zero]

end dorchester_puppy_count_l239_239223


namespace amount_of_b_l239_239516

variable (A B : ℝ)

theorem amount_of_b (h₁ : A + B = 2530) (h₂ : (3 / 5) * A = (2 / 7) * B) : B = 1714 :=
sorry

end amount_of_b_l239_239516


namespace cos_seven_pi_over_six_l239_239809

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end cos_seven_pi_over_six_l239_239809


namespace janet_more_siblings_than_carlos_l239_239420

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end janet_more_siblings_than_carlos_l239_239420


namespace system_solution_l239_239171

theorem system_solution :
  (∀ x y : ℝ, (2 * x + 3 * y = 19) ∧ (3 * x + 4 * y = 26) → x = 2 ∧ y = 5) →
  (∃ x y : ℝ, (2 * (2 * x + 4) + 3 * (y + 3) = 19) ∧ (3 * (2 * x + 4) + 4 * (y + 3) = 26) ∧ x = -1 ∧ y = 2) :=
by
  sorry

end system_solution_l239_239171


namespace product_of_largest_and_second_largest_l239_239474

theorem product_of_largest_and_second_largest (a b c : ℕ) (h₁ : a = 10) (h₂ : b = 11) (h₃ : c = 12) :
  (max (max a b) c * (max (min a (max b c)) (min b (max a c)))) = 132 :=
by
  sorry

end product_of_largest_and_second_largest_l239_239474


namespace find_c_value_l239_239648

theorem find_c_value (x y n m c : ℕ) 
  (h1 : 10 * x + y = 8 * n) 
  (h2 : 10 + x + y = 9 * m) 
  (h3 : c = x + y) : 
  c = 8 := 
by
  sorry

end find_c_value_l239_239648


namespace value_of_z_sub_y_add_x_l239_239543

-- Represent 312 in base 3
def base3_representation : List ℕ := [1, 0, 1, 2, 1, 0] -- 312 in base 3 is 101210

-- Define x, y, z
def x : ℕ := (base3_representation.count 0)
def y : ℕ := (base3_representation.count 1)
def z : ℕ := (base3_representation.count 2)

-- Proposition to be proved
theorem value_of_z_sub_y_add_x : z - y + x = 2 := by
  sorry

end value_of_z_sub_y_add_x_l239_239543


namespace maria_payment_l239_239278

noncomputable def calculate_payment : ℝ :=
  let regular_price := 15
  let first_discount := 0.40 * regular_price
  let after_first_discount := regular_price - first_discount
  let holiday_discount := 0.10 * after_first_discount
  let after_holiday_discount := after_first_discount - holiday_discount
  after_holiday_discount + 2

theorem maria_payment : calculate_payment = 10.10 :=
by
  sorry

end maria_payment_l239_239278


namespace pentagon_triangle_ratio_l239_239948

theorem pentagon_triangle_ratio (p t s : ℝ) 
  (h₁ : 5 * p = 30) 
  (h₂ : 3 * t = 30)
  (h₃ : 4 * s = 30) : 
  p / t = 3 / 5 := by
  sorry

end pentagon_triangle_ratio_l239_239948


namespace halfway_point_l239_239174

theorem halfway_point (x1 x2 : ℚ) (h1 : x1 = 1 / 6) (h2 : x2 = 5 / 6) : 
  (x1 + x2) / 2 = 1 / 2 :=
by
  sorry

end halfway_point_l239_239174


namespace num_candidates_l239_239322

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l239_239322


namespace chocolate_chip_cookie_price_l239_239943

noncomputable def price_of_chocolate_chip_cookies :=
  let total_boxes := 1585
  let total_revenue := 1586.75
  let plain_boxes := 793.375
  let price_of_plain := 0.75
  let revenue_plain := plain_boxes * price_of_plain
  let choco_boxes := total_boxes - plain_boxes
  (993.71875 - revenue_plain) / choco_boxes

theorem chocolate_chip_cookie_price :
  price_of_chocolate_chip_cookies = 1.2525 :=
by sorry

end chocolate_chip_cookie_price_l239_239943


namespace number_solution_l239_239306

theorem number_solution (x : ℝ) : (x / 5 + 4 = x / 4 - 4) → x = 160 := by
  intros h
  sorry

end number_solution_l239_239306


namespace pencils_in_each_box_l239_239653

theorem pencils_in_each_box (n : ℕ) (h : 10 * n - 10 = 40) : n = 5 := by
  sorry

end pencils_in_each_box_l239_239653


namespace carla_zoo_l239_239050

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l239_239050


namespace proof_problem_l239_239211

noncomputable def problem_statement : Prop :=
  let p1 := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0
  let p2 := ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1
  let p3 := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3
  let p4 := ∀ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c > 0 → ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0
  p3 = true ∧ p1 = false ∧ p2 = false ∧ p4 = false

theorem proof_problem : problem_statement := 
sorry

end proof_problem_l239_239211


namespace taxi_fare_l239_239295

theorem taxi_fare (x : ℝ) (h : x > 3) : 
  let starting_price := 6
  let additional_fare_per_km := 1.4
  let fare := starting_price + additional_fare_per_km * (x - 3)
  fare = 1.4 * x + 1.8 :=
by
  sorry

end taxi_fare_l239_239295


namespace books_written_l239_239512

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end books_written_l239_239512


namespace abs_neg_2023_l239_239344

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239344


namespace katya_sold_glasses_l239_239585

-- Definitions based on the conditions specified in the problem
def ricky_sales : ℕ := 9

def tina_sales (K : ℕ) : ℕ := 2 * (K + ricky_sales)

def katya_sales_eq (K : ℕ) : Prop := tina_sales K = K + 26

-- Lean statement to prove Katya sold 8 glasses of lemonade
theorem katya_sold_glasses : ∃ (K : ℕ), katya_sales_eq K ∧ K = 8 :=
by
  sorry

end katya_sold_glasses_l239_239585


namespace xiao_dong_actual_jump_distance_l239_239112

-- Conditions are defined here
def standard_jump_distance : ℝ := 4.00
def xiao_dong_recorded_result : ℝ := -0.32

-- Here we structure our problem
theorem xiao_dong_actual_jump_distance :
  standard_jump_distance + xiao_dong_recorded_result = 3.68 :=
by
  sorry

end xiao_dong_actual_jump_distance_l239_239112


namespace f_specification_l239_239657

open Function

def f : ℕ → ℕ := sorry -- define function f here

axiom f_involution (n : ℕ) : f (f n) = n

axiom f_functional_property (n : ℕ) : f (f n + 1) = if n % 2 = 0 then n - 1 else n + 3

axiom f_bijective : Bijective f

axiom f_not_two (n : ℕ) : f (f n + 1) ≠ 2

axiom f_one_eq_two : f 1 = 2

theorem f_specification (n : ℕ) : 
  f n = if n % 2 = 1 then n + 1 else n - 1 :=
sorry

end f_specification_l239_239657


namespace find_slope_of_line_l239_239568

theorem find_slope_of_line (k m x0 : ℝ) (P Q : ℝ × ℝ) 
  (hP : P.2^2 = 4 * P.1) 
  (hQ : Q.2^2 = 4 * Q.1) 
  (hMid : (P.1 + Q.1) / 2 = x0 ∧ (P.2 + Q.2) / 2 = 2) 
  (hLineP : P.2 = k * P.1 + m) 
  (hLineQ : Q.2 = k * Q.1 + m) : k = 1 :=
by sorry

end find_slope_of_line_l239_239568


namespace minimum_value_l239_239431

theorem minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ x : ℝ, 
    (x = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2) ∧ 
    (∀ y, y = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2 → x ≤ y) ∧ 
    x = 7 :=
by 
  sorry

end minimum_value_l239_239431


namespace inequality_problem_l239_239100

theorem inequality_problem (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by {
  sorry
}

end inequality_problem_l239_239100


namespace line_through_two_quadrants_l239_239718

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l239_239718


namespace sqrt_addition_l239_239303

theorem sqrt_addition :
  (Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3) := 
by sorry

end sqrt_addition_l239_239303


namespace radius_distance_relation_l239_239864

variables {A B C : Point} (Γ₁ Γ₂ ω₀ : Circle)
variables (ω : ℕ → Circle)
variables (r d : ℕ → ℝ)

def diam_circle (P Q : Point) : Circle := sorry  -- This is to define a circle with diameter PQ
def tangent (κ κ' κ'' : Circle) : Prop := sorry  -- This is to define that three circles are mutually tangent

-- Defining the properties as given in the conditions
axiom Γ₁_def : Γ₁ = diam_circle A B
axiom Γ₂_def : Γ₂ = diam_circle A C
axiom ω₀_def : ω₀ = diam_circle B C
axiom ω_def : ∀ n : ℕ, tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₁ (ω n) ∧ tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₂ (ω n) -- ωₙ is tangent to previous circle, Γ₁ and Γ₂

-- The main proof statement
theorem radius_distance_relation (n : ℕ) : r n = 2 * n * d n :=
sorry

end radius_distance_relation_l239_239864


namespace pizza_order_cost_l239_239886

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end pizza_order_cost_l239_239886


namespace arithmetic_mean_end_number_l239_239762

theorem arithmetic_mean_end_number (n : ℤ) :
  (100 + n) / 2 = 150 + 100 → n = 400 := by
  sorry

end arithmetic_mean_end_number_l239_239762


namespace julia_birth_year_is_1979_l239_239851

-- Definitions based on conditions
def wayne_age_in_2021 : ℕ := 37
def wayne_birth_year : ℕ := 2021 - wayne_age_in_2021
def peter_birth_year : ℕ := wayne_birth_year - 3
def julia_birth_year : ℕ := peter_birth_year - 2

-- Theorem to prove
theorem julia_birth_year_is_1979 : julia_birth_year = 1979 := by
  sorry

end julia_birth_year_is_1979_l239_239851


namespace log_comparison_l239_239832

theorem log_comparison 
  (a : ℝ := 1 / 6 * Real.log 8)
  (b : ℝ := 1 / 2 * Real.log 5)
  (c : ℝ := Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := 
by
  sorry

end log_comparison_l239_239832


namespace order_of_nums_l239_239997

variable (a b : ℝ)

theorem order_of_nums (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := 
sorry

end order_of_nums_l239_239997


namespace remaining_apples_l239_239607

-- Define the initial number of apples
def initialApples : ℕ := 356

-- Define the number of apples given away as a mixed number converted to a fraction
def applesGivenAway : ℚ := 272 + 3/5

-- Prove that the remaining apples after giving away are 83
theorem remaining_apples
  (initialApples : ℕ)
  (applesGivenAway : ℚ) :
  initialApples - applesGivenAway = 83 := 
sorry

end remaining_apples_l239_239607


namespace coordinates_of_Q_l239_239134

theorem coordinates_of_Q (m : ℤ) (P Q : ℤ × ℤ) (hP : P = (m + 2, 2 * m + 4))
  (hQ_move : Q = (P.1, P.2 + 2)) (hQ_x_axis : Q.2 = 0) : Q = (-1, 0) :=
sorry

end coordinates_of_Q_l239_239134


namespace cost_of_graphing_calculator_l239_239225

/-
  Everton college paid $1625 for an order of 45 calculators.
  Each scientific calculator costs $10.
  The order included 20 scientific calculators and 25 graphing calculators.
  We need to prove that each graphing calculator costs $57.
-/

namespace EvertonCollege

theorem cost_of_graphing_calculator
  (total_cost : ℕ)
  (cost_scientific : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (cost_graphing : ℕ)
  (h_order : total_cost = 1625)
  (h_cost_scientific : cost_scientific = 10)
  (h_num_scientific : num_scientific = 20)
  (h_num_graphing : num_graphing = 25)
  (h_total_calc : num_scientific + num_graphing = 45)
  (h_pay : total_cost = num_scientific * cost_scientific + num_graphing * cost_graphing) :
  cost_graphing = 57 :=
by
  sorry

end EvertonCollege

end cost_of_graphing_calculator_l239_239225


namespace work_hours_l239_239122

namespace JohnnyWork

variable (dollarsPerHour : ℝ) (totalDollars : ℝ)

theorem work_hours 
  (h_wage : dollarsPerHour = 3.25)
  (h_earned : totalDollars = 26) 
  : (totalDollars / dollarsPerHour) = 8 := 
by
  rw [h_wage, h_earned]
  -- proof goes here
  sorry

end JohnnyWork

end work_hours_l239_239122


namespace monkeys_more_than_giraffes_l239_239046

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l239_239046


namespace probability_of_A_l239_239553

variable (A B : Prop)
variable (P : Prop → ℝ)

-- Given conditions
variable (h1 : P (A ∧ B) = 0.72)
variable (h2 : P (A ∧ ¬B) = 0.18)

theorem probability_of_A: P A = 0.90 := sorry

end probability_of_A_l239_239553


namespace algebraic_expression_value_l239_239258

theorem algebraic_expression_value (a : ℝ) (h : (a^2 - 3) * (a^2 + 1) = 0) : a^2 = 3 :=
by
  sorry

end algebraic_expression_value_l239_239258


namespace ab_proof_l239_239744

theorem ab_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 90 < a + b) (h4 : a + b < 99) 
  (h5 : 0.9 < (a : ℝ) / b) (h6 : (a : ℝ) / b < 0.91) : a * b = 2346 :=
sorry

end ab_proof_l239_239744


namespace books_received_l239_239284

theorem books_received (students : ℕ) (books_per_student : ℕ) (books_fewer : ℕ) (expected_books : ℕ) (received_books : ℕ) :
  students = 20 →
  books_per_student = 15 →
  books_fewer = 6 →
  expected_books = students * books_per_student →
  received_books = expected_books - books_fewer →
  received_books = 294 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_received_l239_239284


namespace sophomores_selected_correct_l239_239030

-- Define the number of students in each grade and the total spots for the event
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300
def totalSpots : ℕ := 40

-- Calculate the total number of students
def totalStudents : ℕ := freshmen + sophomores + juniors

-- The correct answer we want to prove
def numberOfSophomoresSelected : ℕ := (sophomores * totalSpots) / totalStudents

-- Statement to be proved
theorem sophomores_selected_correct : numberOfSophomoresSelected = 26 := by
  -- Proof is omitted
  sorry

end sophomores_selected_correct_l239_239030


namespace lake_width_l239_239424

theorem lake_width
  (W : ℝ)
  (janet_speed : ℝ) (sister_speed : ℝ) (wait_time : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : wait_time = 3)
  (h4 : W / sister_speed = W / janet_speed + wait_time) :
  W = 60 := 
sorry

end lake_width_l239_239424


namespace quadrant_of_point_C_l239_239263

theorem quadrant_of_point_C
  (a b : ℝ)
  (h1 : -(a-2) = -1)
  (h2 : b+5 = 3) :
  a = 3 ∧ b = -2 ∧ 0 < a ∧ b < 0 :=
by {
  sorry
}

end quadrant_of_point_C_l239_239263


namespace num_natural_numbers_divisible_by_7_l239_239096

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l239_239096


namespace remaining_sweet_potatoes_l239_239439

def harvested_sweet_potatoes : ℕ := 80
def sold_sweet_potatoes_mrs_adams : ℕ := 20
def sold_sweet_potatoes_mr_lenon : ℕ := 15
def traded_sweet_potatoes : ℕ := 10
def donated_sweet_potatoes : ℕ := 5

theorem remaining_sweet_potatoes :
  harvested_sweet_potatoes - (sold_sweet_potatoes_mrs_adams + sold_sweet_potatoes_mr_lenon + traded_sweet_potatoes + donated_sweet_potatoes) = 30 :=
by
  sorry

end remaining_sweet_potatoes_l239_239439


namespace train_speed_l239_239645

theorem train_speed (length time : ℝ) (h_length : length = 120) (h_time : time = 11.999040076793857) :
  (length / time) * 3.6 = 36.003 :=
by
  sorry

end train_speed_l239_239645


namespace mark_spending_l239_239595

theorem mark_spending (initial_money : ℕ) (first_store_half : ℕ) (first_store_additional : ℕ) 
                      (second_store_third : ℕ) (remaining_money : ℕ) (total_spent : ℕ) : 
  initial_money = 180 ∧ 
  first_store_half = 90 ∧ 
  first_store_additional = 14 ∧ 
  total_spent = first_store_half + first_store_additional ∧
  remaining_money = initial_money - total_spent ∧
  second_store_third = 60 ∧ 
  remaining_money - second_store_third = 16 ∧ 
  initial_money - (total_spent + second_store_third + 16) = 0 → 
  remaining_money - second_store_third = 16 :=
by
  intro h
  sorry

end mark_spending_l239_239595


namespace track_length_is_500_l239_239042

-- Given Conditions
variables (B_speed S_speed : ℝ) -- Brenda and Sally's speeds
variable (track_length : ℝ) -- Length of the track
-- Brenda and Sally start at diametrically opposite points and run in opposite directions
-- They meet for the first time after Brenda has run 100 meters
variable h_meet_first : B_speed * 100 / (B_speed + S_speed) = track_length / 2
-- They meet for the second time after Sally has run 150 meters past their first meeting point
variable h_meet_second : S_speed * 150 / (B_speed + S_speed) + track_length / 2 = track_length

-- Theorem to be proved: the length of the track is 500 meters
theorem track_length_is_500 (h1 : B_speed ≠ 0) (h2 : S_speed ≠ 0) :
  track_length = 500 := 
sorry

end track_length_is_500_l239_239042


namespace track_length_l239_239041

theorem track_length (x : ℕ) 
  (diametrically_opposite : ∃ a b : ℕ, a + b = x)
  (first_meeting : ∃ b : ℕ, b = 100)
  (second_meeting : ∃ s s' : ℕ, s = 150 ∧ s' = (x / 2 - 100 + s))
  (constant_speed : ∀ t₁ t₂ : ℕ, t₁ / t₂ = 100 / (x / 2 - 100)) :
  x = 400 := 
by sorry

end track_length_l239_239041


namespace ratio_twelfth_term_geometric_sequence_l239_239429

theorem ratio_twelfth_term_geometric_sequence (G H : ℕ → ℝ) (n : ℕ) (a r b s : ℝ)
  (hG : ∀ n, G n = a * (r^n - 1) / (r - 1))
  (hH : ∀ n, H n = b * (s^n - 1) / (s - 1))
  (ratio_condition : ∀ n, G n / H n = (5 * n + 3) / (3 * n + 17)) :
  (a * r^11) / (b * s^11) = 2 / 5 :=
by 
  sorry

end ratio_twelfth_term_geometric_sequence_l239_239429


namespace range_of_a_plus_3b_l239_239241

theorem range_of_a_plus_3b :
  ∀ (a b : ℝ),
    -1 ≤ a + b ∧ a + b ≤ 1 ∧ 1 ≤ a - 2 * b ∧ a - 2 * b ≤ 3 →
    -11 / 3 ≤ a + 3 * b ∧ a + 3 * b ≤ 7 / 3 :=
by
  sorry

end range_of_a_plus_3b_l239_239241


namespace cost_of_iced_coffee_for_2_weeks_l239_239168

def cost_to_last_for_2_weeks (servings_per_bottle servings_per_day price_per_bottle duration_in_days : ℕ) : ℕ :=
  let total_servings_needed := servings_per_day * duration_in_days
  let bottles_needed := total_servings_needed / servings_per_bottle
  bottles_needed * price_per_bottle

theorem cost_of_iced_coffee_for_2_weeks :
  cost_to_last_for_2_weeks 6 3 3 14 = 21 :=
by
  sorry

end cost_of_iced_coffee_for_2_weeks_l239_239168


namespace find_three_digit_numbers_l239_239980

theorem find_three_digit_numbers :
  ∃ A, (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376) ∨ (A = 625) :=
by
  sorry

end find_three_digit_numbers_l239_239980


namespace stamps_difference_l239_239151

theorem stamps_difference (x : ℕ) (h1: 5 * x / 3 * x = 5 / 3)
(h2: (5 * x - 12) / (3 * x + 12) = 4 / 3) : 
(5 * x - 12) - (3 * x + 12) = 32 := by
sorry

end stamps_difference_l239_239151


namespace smallest_value_is_A_l239_239210

def A : ℤ := -(-3 - 2)^2
def B : ℤ := (-3) * (-2)
def C : ℚ := ((-3)^2 : ℚ) / (-2)^2
def D : ℚ := ((-3)^2 : ℚ) / (-2)

theorem smallest_value_is_A : A < B ∧ A < C ∧ A < D :=
by
  sorry

end smallest_value_is_A_l239_239210


namespace Arman_total_earnings_two_weeks_l239_239428

theorem Arman_total_earnings_two_weeks :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let this_week_increase := 0.5
  let initial_rate := 10
  let this_week_rate := initial_rate + this_week_increase
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := this_week_hours * this_week_rate
  let total_earnings := last_week_earnings + this_week_earnings
  total_earnings = 770 := 
by
  sorry

end Arman_total_earnings_two_weeks_l239_239428


namespace Leah_coins_value_in_cents_l239_239126

theorem Leah_coins_value_in_cents (p n : ℕ) (h₁ : p + n = 15) (h₂ : p = n + 2) : p + 5 * n = 44 :=
by
  sorry

end Leah_coins_value_in_cents_l239_239126


namespace monkeys_more_than_giraffes_l239_239047

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l239_239047


namespace train_crosses_pole_l239_239797

theorem train_crosses_pole
  (speed_kmph : ℝ)
  (train_length_meters : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (time_seconds : ℝ)
  (h1 : speed_kmph = 270)
  (h2 : train_length_meters = 375.03)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_mps = speed_kmph * conversion_factor)
  (h5 : time_seconds = train_length_meters / speed_mps)
  : time_seconds = 5.0004 :=
by
  sorry

end train_crosses_pole_l239_239797


namespace binary_difference_l239_239971

theorem binary_difference (n : ℕ) (b_2 : List ℕ) (x y : ℕ) (h1 : n = 157)
  (h2 : b_2 = [1, 0, 0, 1, 1, 1, 0, 1])
  (hx : x = b_2.count 0)
  (hy : y = b_2.count 1) : y - x = 2 := by
  sorry

end binary_difference_l239_239971


namespace probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l239_239787

variable {p q : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1)

theorem probability_A_miss_at_least_once :
  1 - p^4 = (1 - p^4) := by
sorry

theorem probability_A_2_hits_B_3_hits :
  24 * p^2 * q^3 * (1 - p)^2 * (1 - q) = 24 * p^2 * q^3 * (1 - p)^2 * (1 - q) := by
sorry

end probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l239_239787


namespace book_loss_percentage_l239_239763

theorem book_loss_percentage 
  (C S : ℝ) 
  (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := 
by 
  sorry

end book_loss_percentage_l239_239763


namespace find_abc_l239_239064

theorem find_abc (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by 
  sorry

end find_abc_l239_239064


namespace stacy_savings_for_3_pairs_l239_239142

-- Define the cost per pair of shorts
def cost_per_pair : ℕ := 10

-- Define the discount percentage as a decimal
def discount_percentage : ℝ := 0.1

-- Function to calculate the total cost without discount for n pairs
def total_cost_without_discount (n : ℕ) : ℕ := cost_per_pair * n

-- Function to calculate the total cost with discount for n pairs
noncomputable def total_cost_with_discount (n : ℕ) : ℝ :=
  if n >= 3 then
    let discount := discount_percentage * (cost_per_pair * n : ℝ)
    (cost_per_pair * n : ℝ) - discount
  else
    cost_per_pair * n

-- Function to calculate the savings for buying n pairs at once compared to individually
noncomputable def savings (n : ℕ) : ℝ :=
  (total_cost_without_discount n : ℝ) - total_cost_with_discount n

-- Proof statement
theorem stacy_savings_for_3_pairs : savings 3 = 3 := by
  sorry

end stacy_savings_for_3_pairs_l239_239142


namespace books_written_l239_239513

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end books_written_l239_239513


namespace solve_fraction_zero_l239_239265

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end solve_fraction_zero_l239_239265


namespace solve_system_l239_239891

-- Define the conditions of the system of equations
def condition1 (x y : ℤ) := 4 * x - 3 * y = -13
def condition2 (x y : ℤ) := 5 * x + 3 * y = -14

-- Define the proof goal using the conditions
theorem solve_system : ∃ (x y : ℤ), condition1 x y ∧ condition2 x y ∧ x = -3 ∧ y = 1 / 3 :=
by
  sorry

end solve_system_l239_239891


namespace coeff_x2_expansion_l239_239895

theorem coeff_x2_expansion :
  let polynomial := ((Polynomial.X + Polynomial.C 1) ^ 5 * (Polynomial.X - Polynomial.C 2))
  Polynomial.coeff polynomial 2 = -15 :=
by
  sorry

end coeff_x2_expansion_l239_239895


namespace sequence_form_l239_239818

theorem sequence_form {a : ℕ → ℚ} (h_eq : ∀ n : ℕ, a n * x ^ 2 - a (n + 1) * x + 1 = 0) 
  (h_roots : ∀ α β : ℚ, 6 * α - 2 * α * β + 6 * β = 3 ) (h_a1 : a 1 = 7 / 6) :
  ∀ n : ℕ, a n = (1 / 2) ^ n + 2 / 3 :=
by
  sorry

end sequence_form_l239_239818


namespace anie_days_to_finish_task_l239_239773

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_finish_task_l239_239773


namespace evaluate_dollar_l239_239234

variable {R : Type} [CommRing R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : 
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) :=
by
  sorry

end evaluate_dollar_l239_239234


namespace abs_neg_2023_l239_239349

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l239_239349


namespace monkeys_more_than_giraffes_l239_239045

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l239_239045


namespace plant_branches_l239_239025

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 91) : 1 + x + x^2 = 91 :=
by sorry

end plant_branches_l239_239025


namespace categorization_proof_l239_239547

-- Definitions of expenses
inductive ExpenseType
| Fixed
| Variable

-- Expense data structure
structure Expense where
  name : String
  type : ExpenseType

-- List of expenses
def fixed_expenses : List Expense :=
  [ { name := "Utility payments", type := ExpenseType.Fixed },
    { name := "Loan payments", type := ExpenseType.Fixed },
    { name := "Taxes", type := ExpenseType.Fixed } ]

def variable_expenses : List Expense :=
  [ { name := "Entertainment", type := ExpenseType.Variable },
    { name := "Travel", type := ExpenseType.Variable },
    { name := "Purchasing non-essential items", type := ExpenseType.Variable },
    { name := "Renting professional video cameras", type := ExpenseType.Variable },
    { name := "Maintenance of the blog", type := ExpenseType.Variable } ]

-- Expenses that can be economized
def economizable_expenses : List String :=
  [ "Payment for home internet and internet traffic",
    "Travel expenses",
    "Renting professional video cameras for a year",
    "Domain payment for blog maintenance",
    "Visiting coffee shops (4 times a month)" ]

-- Expenses that cannot be economized
def non_economizable_expenses : List String :=
  [ "Loan payments",
    "Tax payments",
    "Courses for qualification improvement in blogger school (onsite training)" ]

-- Additional expenses and economizing suggestions
def additional_expenses : List String :=
  [ "Professional development workshops",
    "Marketing and advertising costs",
    "Office supplies",
    "Subscription services" ]

-- Lean statement for the problem
theorem categorization_proof :
  (∀ exp ∈ fixed_expenses, exp.name ∈ non_economizable_expenses) ∧
  (∀ exp ∈ variable_expenses, exp.name ∈ economizable_expenses) ∧
  (∃ exp ∈ additional_expenses, true) :=
by
  sorry

end categorization_proof_l239_239547


namespace geometric_sum_sequence_l239_239146

theorem geometric_sum_sequence (n : ℕ) (a : ℕ → ℕ) (a1 : a 1 = 2) (a4 : a 4 = 16) :
    (∃ q : ℕ, a 2 = a 1 * q) → (∃ S_n : ℕ, S_n = 2 * (2 ^ n - 1)) :=
by
  sorry

end geometric_sum_sequence_l239_239146


namespace abs_neg_2023_l239_239345

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239345


namespace find_p_and_q_l239_239388

theorem find_p_and_q (p q : ℝ)
    (M : Set ℝ := {x | x^2 + p * x - 2 = 0})
    (N : Set ℝ := {x | x^2 - 2 * x + q = 0})
    (h : M ∪ N = {-1, 0, 2}) :
    p = -1 ∧ q = 0 :=
sorry

end find_p_and_q_l239_239388


namespace x_days_worked_l239_239519

theorem x_days_worked (W : ℝ) :
  let x_work_rate := W / 20
  let y_work_rate := W / 24
  let y_days := 12
  let y_work_done := y_work_rate * y_days
  let total_work := W
  let work_done_by_x := (W - y_work_done) / x_work_rate
  work_done_by_x = 10 := 
by
  sorry

end x_days_worked_l239_239519


namespace investment_amounts_proof_l239_239310

noncomputable def investment_proof_statement : Prop :=
  let p_investment_first_year := 52000
  let q_investment := (5/4) * p_investment_first_year
  let r_investment := (6/4) * p_investment_first_year;
  let p_investment_second_year := p_investment_first_year + (20/100) * p_investment_first_year;
  (q_investment = 65000) ∧ (r_investment = 78000) ∧ (q_investment = 65000) ∧ (r_investment = 78000)

theorem investment_amounts_proof : investment_proof_statement :=
  by
    sorry

end investment_amounts_proof_l239_239310


namespace not_odd_not_even_min_value_3_l239_239872

def f (x : ℝ) : ℝ := x^2 + abs (x - 2) - 1

-- Statement 1: Prove that the function is neither odd nor even.
theorem not_odd_not_even : 
  ¬(∀ x, f (-x) = -f x) ∧ ¬(∀ x, f (-x) = f x) :=
sorry

-- Statement 2: Prove that the minimum value of the function is 3.
theorem min_value_3 : ∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≥ 3 :=
sorry

end not_odd_not_even_min_value_3_l239_239872


namespace valid_documents_count_l239_239541

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end valid_documents_count_l239_239541


namespace map_distance_to_actual_distance_l239_239453

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_inches : ℝ)
  (scale_miles : ℝ)
  (actual_distance : ℝ)
  (h_scale : scale_inches = 0.5)
  (h_scale_miles : scale_miles = 10)
  (h_map_distance : map_distance = 20) :
  actual_distance = 400 :=
by
  sorry

end map_distance_to_actual_distance_l239_239453


namespace multiplicative_inverse_modulo_2799_l239_239905

theorem multiplicative_inverse_modulo_2799 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 2799 ∧ (225 * n) % 2799 = 1 :=
by {
  -- conditions are expressed directly in the theorem assumption
  sorry
}

end multiplicative_inverse_modulo_2799_l239_239905


namespace stationery_shop_costs_l239_239649

theorem stationery_shop_costs (p n : ℝ) 
  (h1 : 9 * p + 6 * n = 3.21)
  (h2 : 8 * p + 5 * n = 2.84) :
  12 * p + 9 * n = 4.32 :=
sorry

end stationery_shop_costs_l239_239649


namespace cos_seven_pi_over_six_l239_239808

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end cos_seven_pi_over_six_l239_239808


namespace circle_radius_5_l239_239370

theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10 * x + y^2 + 2 * y + c = 0) → 
  (∀ x y : ℝ, (x + 5)^2 + (y + 1)^2 = 25) → 
  c = 51 :=
sorry

end circle_radius_5_l239_239370


namespace minimum_value_of_z_l239_239434

theorem minimum_value_of_z
  (x y : ℝ)
  (h1 : 3 * x + y - 6 ≥ 0)
  (h2 : x - y - 2 ≤ 0)
  (h3 : y - 3 ≤ 0) :
  ∃ z, z = 4 * x + y ∧ z = 7 :=
sorry

end minimum_value_of_z_l239_239434


namespace divisible_by_7_in_range_200_to_400_l239_239091

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l239_239091


namespace find_original_production_planned_l239_239207

-- Definition of the problem
variables (x : ℕ)
noncomputable def original_production_planned (x : ℕ) :=
  (6000 / (x + 500)) = (4500 / x)

-- The theorem to prove the original number planned is 1500
theorem find_original_production_planned (x : ℕ) (h : original_production_planned x) : x = 1500 :=
sorry

end find_original_production_planned_l239_239207


namespace bottles_needed_l239_239022

theorem bottles_needed (runners : ℕ) (bottles_needed_per_runner : ℕ) (bottles_available : ℕ)
  (h_runners : runners = 14)
  (h_bottles_needed_per_runner : bottles_needed_per_runner = 5)
  (h_bottles_available : bottles_available = 68) :
  runners * bottles_needed_per_runner - bottles_available = 2 :=
by
  sorry

end bottles_needed_l239_239022


namespace no_real_roots_iff_k_gt_2_l239_239676

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end no_real_roots_iff_k_gt_2_l239_239676


namespace deanna_wins_l239_239799

theorem deanna_wins (A B C D : ℕ) (total_games : ℕ) (total_wins : ℕ) (A_wins : A = 5) (B_wins : B = 2)
  (C_wins : C = 1) (total_games_def : total_games = 6) (total_wins_def : total_wins = 12)
  (total_wins_eq : A + B + C + D = total_wins) : D = 4 :=
by
  sorry

end deanna_wins_l239_239799


namespace length_of_AP_in_right_triangle_l239_239400

theorem length_of_AP_in_right_triangle 
  (A B C : ℝ × ℝ)
  (hA : A = (0, 2))
  (hB : B = (0, 0))
  (hC : C = (2, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0 ∧ M.2 = 0)
  (inc : ℝ × ℝ)
  (hinc : inc = (1, 1)) :
  ∃ P : ℝ × ℝ, (P.1 = 0 ∧ P.2 = 1) ∧ dist A P = 1 := by
  sorry

end length_of_AP_in_right_triangle_l239_239400


namespace smallest_model_length_l239_239753

theorem smallest_model_length 
  (full_size_length : ℕ)
  (mid_size_ratio : ℚ)
  (smallest_size_ratio : ℚ)
  (H1 : full_size_length = 240)
  (H2 : mid_size_ratio = 1/10)
  (H3 : smallest_size_ratio = 1/2) 
  : full_size_length * mid_size_ratio * smallest_size_ratio = 12 :=
by
  sorry

end smallest_model_length_l239_239753


namespace exists_a_solution_iff_l239_239367

theorem exists_a_solution_iff (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1 / 4 := 
by 
  sorry

end exists_a_solution_iff_l239_239367


namespace carla_zoo_l239_239049

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l239_239049


namespace carol_total_peanuts_l239_239054

-- Conditions as definitions
def carol_initial_peanuts : Nat := 2
def carol_father_peanuts : Nat := 5

-- Theorem stating that the total number of peanuts Carol has is 7
theorem carol_total_peanuts : carol_initial_peanuts + carol_father_peanuts = 7 := by
  -- Proof would go here, but we use sorry to skip
  sorry

end carol_total_peanuts_l239_239054


namespace total_food_per_day_l239_239823

def num_dogs : ℝ := 2
def food_per_dog_per_day : ℝ := 0.12

theorem total_food_per_day : (num_dogs * food_per_dog_per_day) = 0.24 :=
by sorry

end total_food_per_day_l239_239823


namespace exactly_one_solves_problem_l239_239644

theorem exactly_one_solves_problem (pA pB pC : ℝ) (hA : pA = 1 / 2) (hB : pB = 1 / 3) (hC : pC = 1 / 4) :
  (pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC) = 11 / 24 :=
by
  sorry

end exactly_one_solves_problem_l239_239644


namespace length_of_wall_l239_239320

theorem length_of_wall (side_mirror length_wall width_wall : ℕ) 
  (mirror_area wall_area : ℕ) (H1 : side_mirror = 54) 
  (H2 : mirror_area = side_mirror * side_mirror) 
  (H3 : wall_area = 2 * mirror_area) 
  (H4 : width_wall = 68) 
  (H5 : wall_area = length_wall * width_wall) : 
  length_wall = 86 :=
by
  sorry

end length_of_wall_l239_239320


namespace new_person_weight_is_90_l239_239183

-- Define the weight of the replaced person
def replaced_person_weight : ℝ := 40

-- Define the increase in average weight when the new person replaces the replaced person
def increase_in_average_weight : ℝ := 10

-- Define the increase in total weight as 5 times the increase in average weight
def increase_in_total_weight (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

-- Define the weight of the new person
def new_person_weight (replaced_w : ℝ) (total_increase : ℝ) : ℝ := replaced_w + total_increase

-- Prove that the weight of the new person is 90 kg
theorem new_person_weight_is_90 :
  new_person_weight replaced_person_weight (increase_in_total_weight 5 increase_in_average_weight) = 90 := 
by 
  -- sorry will skip the proof, as required
  sorry

end new_person_weight_is_90_l239_239183


namespace scientific_notation_of_viewers_l239_239877

def million : ℝ := 10^6
def viewers : ℝ := 70.62 * million

theorem scientific_notation_of_viewers : viewers = 7.062 * 10^7 := by
  sorry

end scientific_notation_of_viewers_l239_239877


namespace cos_neg_570_eq_neg_sqrt3_div_2_l239_239674

theorem cos_neg_570_eq_neg_sqrt3_div_2 :
  Real.cos (-(570 : ℝ) * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_neg_570_eq_neg_sqrt3_div_2_l239_239674


namespace speed_of_train_l239_239325

def distance : ℝ := 80
def time : ℝ := 6
def expected_speed : ℝ := 13.33

theorem speed_of_train : distance / time = expected_speed :=
by
  sorry

end speed_of_train_l239_239325


namespace millet_more_than_half_l239_239350

def daily_millet (n : ℕ) : ℝ :=
  1 - (0.7)^n

theorem millet_more_than_half (n : ℕ) : daily_millet 2 > 0.5 :=
by {
  sorry
}

end millet_more_than_half_l239_239350


namespace remainder_1234567_div_145_l239_239654

theorem remainder_1234567_div_145 : 1234567 % 145 = 67 := by
  sorry

end remainder_1234567_div_145_l239_239654


namespace first_digit_base12_1025_l239_239010

theorem first_digit_base12_1025 : (1025 : ℕ) / (12^2 : ℕ) = 7 := by
  sorry

end first_digit_base12_1025_l239_239010


namespace N_properties_l239_239552

def N : ℕ := 3625

theorem N_properties :
  (N % 32 = 21) ∧ (N % 125 = 0) ∧ (N^2 % 8000 = N % 8000) :=
by
  sorry

end N_properties_l239_239552


namespace tan_theta_equation_l239_239973

theorem tan_theta_equation (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) :
  Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  sorry

end tan_theta_equation_l239_239973


namespace abs_neg_2023_l239_239346

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l239_239346


namespace abs_eq_two_iff_l239_239395

theorem abs_eq_two_iff (a : ℝ) : |a| = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_two_iff_l239_239395


namespace units_digit_42_3_plus_27_2_l239_239302

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_42_3_plus_27_2 : units_digit (42^3 + 27^2) = 7 :=
by
  sorry

end units_digit_42_3_plus_27_2_l239_239302


namespace work_time_A_and_C_together_l239_239190

theorem work_time_A_and_C_together
  (A_work B_work C_work : ℝ)
  (hA : A_work = 1/3)
  (hB : B_work = 1/6)
  (hBC : B_work + C_work = 1/3) :
  1 / (A_work + C_work) = 2 := by
  sorry

end work_time_A_and_C_together_l239_239190


namespace example_problem_l239_239006

-- Definitions and conditions derived from the original problem statement
def smallest_integer_with_two_divisors (m : ℕ) : Prop := m = 2
def second_largest_integer_with_three_divisors_less_than_100 (n : ℕ) : Prop := n = 25

theorem example_problem (m n : ℕ) 
    (h1 : smallest_integer_with_two_divisors m) 
    (h2 : second_largest_integer_with_three_divisors_less_than_100 n) : 
    m + n = 27 :=
by sorry

end example_problem_l239_239006


namespace total_eyes_l239_239257

def boys := 23
def girls := 18
def cats := 10
def spiders := 5

def boy_eyes := 2
def girl_eyes := 2
def cat_eyes := 2
def spider_eyes := 8

theorem total_eyes : (boys * boy_eyes) + (girls * girl_eyes) + (cats * cat_eyes) + (spiders * spider_eyes) = 142 := by
  sorry

end total_eyes_l239_239257


namespace find_y_of_set_with_mean_l239_239382

theorem find_y_of_set_with_mean (y : ℝ) (h : ((8 + 15 + 20 + 6 + y) / 5 = 12)) : y = 11 := 
by 
    sorry

end find_y_of_set_with_mean_l239_239382


namespace rationalize_denominator_correct_l239_239881

noncomputable def rationalize_denominator : Prop :=
  (1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2)

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l239_239881


namespace athlete_with_most_stable_performance_l239_239804

def variance_A : ℝ := 0.78
def variance_B : ℝ := 0.2
def variance_C : ℝ := 1.28

theorem athlete_with_most_stable_performance : variance_B < variance_A ∧ variance_B < variance_C :=
by {
  -- Variance comparisons:
  -- 0.2 < 0.78
  -- 0.2 < 1.28
  sorry
}

end athlete_with_most_stable_performance_l239_239804


namespace Carly_running_distance_l239_239812

theorem Carly_running_distance :
  let week1 := 2 in
  let week2 := 2 * week1 + 3 in
  let week3 := (9 / 7) * week2 in
  let week4 := week3 - 5 in
  week4 = 4 :=
by
  let week1 := 2
  let week2 := 2 * week1 + 3
  let week3 := (9 / 7) * week2
  let week4 := week3 - 5
  show week4 = 4
  sorry

end Carly_running_distance_l239_239812


namespace bank_robbery_participants_l239_239957

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l239_239957


namespace average_percentage_25_students_l239_239934

theorem average_percentage_25_students (s1 s2 : ℕ) (p1 p2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : p1 = 75) (h3 : s2 = 10) (h4 : p2 = 95) (h5 : n = 25) :
  ((s1 * p1 + s2 * p2) / n) = 83 := 
by
  sorry

end average_percentage_25_students_l239_239934


namespace even_square_is_even_l239_239244

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l239_239244


namespace least_repeating_block_length_l239_239452

theorem least_repeating_block_length (n d : ℚ) (h1 : n = 7) (h2 : d = 13) (h3 : (n / d).isRepeatingDecimal) : 
  ∃ k : ℕ, k = 6 ∧ ∃ m : ℕ, lenRecBlock (fractionToDecimal (n / d)) m k := 
by 
  sorry

end least_repeating_block_length_l239_239452


namespace lemonade_percentage_l239_239794

theorem lemonade_percentage (V : ℝ) (L : ℝ) :
  (0.80 * 0.40 * V + (100 - L) / 100 * 0.60 * V = 0.65 * V) →
  L = 99.45 :=
by
  intro h
  -- The proof would go here
  sorry

end lemonade_percentage_l239_239794


namespace max_sundays_in_51_days_l239_239921

theorem max_sundays_in_51_days (days_in_week: ℕ) (total_days: ℕ) 
  (start_on_first: Bool) (first_day_sunday: Prop) 
  (is_sunday: ℕ → Bool) :
  days_in_week = 7 ∧ total_days = 51 ∧ start_on_first = tt ∧ first_day_sunday → 
  (∃ n, ∀ i < total_days, is_sunday i → n ≤ 8) ∧ 
  (∀ j, j ≤ total_days → is_sunday j → j ≤ 8) := by
  sorry

end max_sundays_in_51_days_l239_239921


namespace circle_equation_through_origin_l239_239998

theorem circle_equation_through_origin (focus : ℝ × ℝ) (radius : ℝ) (x y : ℝ) 
  (h1 : focus = (1, 0)) 
  (h2 : (x - 1)^2 + y^2 = radius^2) : 
  x^2 + y^2 - 2*x = 0 :=
by
  sorry

end circle_equation_through_origin_l239_239998


namespace triangle_angle_contradiction_l239_239136

theorem triangle_angle_contradiction (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) :
  false :=
by
  have h : α + β + γ > 180 := by
  { linarith }
  linarith

end triangle_angle_contradiction_l239_239136


namespace solve_system_l239_239446

theorem solve_system : 
  ∃ x y : ℚ, (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ∧ x = 1/2 ∧ y = -3 :=
by
  sorry

end solve_system_l239_239446


namespace find_sum_of_m_and_k_l239_239769

theorem find_sum_of_m_and_k
  (d m k : ℤ)
  (h : (9 * d^2 - 5 * d + m) * (4 * d^2 + k * d - 6) = 36 * d^4 + 11 * d^3 - 59 * d^2 + 10 * d + 12) :
  m + k = -7 :=
by sorry

end find_sum_of_m_and_k_l239_239769


namespace inequality_sqrt_ab_l239_239600

theorem inequality_sqrt_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
sorry

end inequality_sqrt_ab_l239_239600


namespace distribution_equiv_implies_constants_l239_239590

open ProbabilityTheory

noncomputable section

theorem distribution_equiv_implies_constants (ξ : ℝ → ℝ) 
  (hnd : ¬∀ x, ξ x = ξ 0) -- ξ is non-degenerate
  (a : ℝ) (ha : a > 0) (b : ℝ)
  (h : ∀ x, ξ x = ξ (a * x + b)) : a = 1 ∧ b = 0 :=
by
  sorry

end distribution_equiv_implies_constants_l239_239590


namespace jill_vs_jack_arrival_time_l239_239732

def distance_to_park : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem jill_vs_jack_arrival_time :
  let jill_time := distance_to_park / jill_speed
  let jack_time := distance_to_park / jack_speed
  let jill_time_minutes := jill_time * 60
  let jack_time_minutes := jack_time * 60
  jill_time_minutes < jack_time_minutes ∧ jack_time_minutes - jill_time_minutes = 5.4 :=
by
  sorry

end jill_vs_jack_arrival_time_l239_239732


namespace factor_expression_l239_239542

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) :=
by
  sorry

end factor_expression_l239_239542


namespace vaishali_total_stripes_l239_239481

theorem vaishali_total_stripes
  (hats1 : ℕ) (stripes1 : ℕ)
  (hats2 : ℕ) (stripes2 : ℕ)
  (hats3 : ℕ) (stripes3 : ℕ)
  (hats4 : ℕ) (stripes4 : ℕ)
  (total_stripes : ℕ) :
  hats1 = 4 → stripes1 = 3 →
  hats2 = 3 → stripes2 = 4 →
  hats3 = 6 → stripes3 = 0 →
  hats4 = 2 → stripes4 = 5 →
  total_stripes = (hats1 * stripes1) + (hats2 * stripes2) + (hats3 * stripes3) + (hats4 * stripes4) →
  total_stripes = 34 := by
  sorry

end vaishali_total_stripes_l239_239481


namespace cos_decreasing_intervals_l239_239665

open Real

def is_cos_decreasing_interval (k : ℤ) : Prop := 
  let f (x : ℝ) := cos (π / 4 - 2 * x)
  ∀ x y : ℝ, (k * π + π / 8 ≤ x) → (x ≤ k * π + 5 * π / 8) → 
             (k * π + π / 8 ≤ y) → (y ≤ k * π + 5 * π / 8) → 
             x < y → f x > f y

theorem cos_decreasing_intervals : ∀ k : ℤ, is_cos_decreasing_interval k :=
by
  sorry

end cos_decreasing_intervals_l239_239665


namespace plane_difference_correct_l239_239975

noncomputable def max_planes : ℕ := 27
noncomputable def min_planes : ℕ := 7
noncomputable def diff_planes : ℕ := max_planes - min_planes

theorem plane_difference_correct : diff_planes = 20 := by
  sorry

end plane_difference_correct_l239_239975


namespace price_difference_l239_239912

theorem price_difference (total_cost shirt_price : ℝ) (h1 : total_cost = 80.34) (h2 : shirt_price = 36.46) :
  (total_cost - shirt_price) - shirt_price = 7.42 :=
by
  sorry

end price_difference_l239_239912


namespace initial_white_cookies_l239_239659

theorem initial_white_cookies (B W : ℕ) 
  (h1 : B = W + 50)
  (h2 : (1 / 2 : ℚ) * B + (1 / 4 : ℚ) * W = 85) :
  W = 80 :=
by
  sorry

end initial_white_cookies_l239_239659


namespace pet_store_cages_l239_239020

-- Definitions and conditions
def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def puppies_per_cage : ℕ := 4
def remaining_puppies : ℕ := initial_puppies - sold_puppies
def cages_used : ℕ := remaining_puppies / puppies_per_cage

-- Theorem statement
theorem pet_store_cages : cages_used = 8 := by sorry

end pet_store_cages_l239_239020


namespace sin_cos_identity_l239_239785

theorem sin_cos_identity : (Real.sin (65 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) 
  - Real.cos (65 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_cos_identity_l239_239785


namespace morleys_theorem_l239_239015

def is_trisector (A B C : Point) (p : Point) : Prop :=
sorry -- Definition that this point p is on one of the trisectors of ∠BAC

def triangle (A B C : Point) : Prop :=
sorry -- Definition that points A, B, C form a triangle

def equilateral (A B C : Point) : Prop :=
sorry -- Definition that triangle ABC is equilateral

theorem morleys_theorem (A B C D E F : Point)
  (hABC : triangle A B C)
  (hD : is_trisector A B C D)
  (hE : is_trisector B C A E)
  (hF : is_trisector C A B F) :
  equilateral D E F :=
sorry

end morleys_theorem_l239_239015


namespace k_positive_first_third_quadrants_l239_239707

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l239_239707


namespace lisa_needs_change_probability_l239_239162

theorem lisa_needs_change_probability :
  let quarters := 16
  let toy_prices := List.range' 2 10 |> List.map (fun n => n * 25) -- List of toy costs: (50,75,...,300)
  let favorite_toy_price := 275
  let factorial := Nat.factorial
  let favorable := (factorial 9) + 9 * (factorial 8)
  let total_permutations := factorial 10
  let p_no_change := (favorable.toFloat / total_permutations.toFloat) -- Convert to Float for probability calculations
  let p_change_needed := Float.round ((1.0 - p_no_change) * 100.0) / 100.0
  p_change_needed = 4.0 / 5.0 := sorry

end lisa_needs_change_probability_l239_239162


namespace painted_faces_cube_eq_54_l239_239313

def painted_faces (n : ℕ) : ℕ :=
  if n = 5 then (3 * 3) * 6 else 0

theorem painted_faces_cube_eq_54 : painted_faces 5 = 54 := by {
  sorry
}

end painted_faces_cube_eq_54_l239_239313


namespace sara_has_8_balloons_l239_239167

-- Define the number of yellow balloons Tom has.
def tom_balloons : ℕ := 9 

-- Define the total number of yellow balloons.
def total_balloons : ℕ := 17

-- Define the number of yellow balloons Sara has.
def sara_balloons : ℕ := total_balloons - tom_balloons

-- Theorem stating that Sara has 8 yellow balloons.
theorem sara_has_8_balloons : sara_balloons = 8 := by
  -- Proof goes here. Adding sorry for now to skip the proof.
  sorry

end sara_has_8_balloons_l239_239167


namespace max_type_A_pieces_max_profit_l239_239024

noncomputable def type_A_cost := 80
noncomputable def type_A_sell := 120
noncomputable def type_B_cost := 60
noncomputable def type_B_sell := 90
noncomputable def total_clothes := 100
noncomputable def min_type_A := 65
noncomputable def max_cost := 7500

/-- The maximum number of type A clothing pieces that can be purchased --/
theorem max_type_A_pieces (x : ℕ) : 
  type_A_cost * x + type_B_cost * (total_clothes - x) ≤ max_cost → 
  x ≤ 75 := by 
sorry

variable (a : ℝ) (h_a : 0 < a ∧ a < 10)

/-- The optimal purchase strategy to maximize profit --/
theorem max_profit (x y : ℕ) : 
  (x + y = total_clothes) ∧ 
  (type_A_cost * x + type_B_cost * y ≤ max_cost) ∧
  (min_type_A ≤ x) ∧ 
  (x ≤ 75) → 
  (type_A_sell - type_A_cost - a) * x + (type_B_sell - type_B_cost) * y 
  ≤ (type_A_sell - type_A_cost - a) * 75 + (type_B_sell - type_B_cost) * 25 := by 
sorry

end max_type_A_pieces_max_profit_l239_239024


namespace cheapest_book_price_l239_239830

theorem cheapest_book_price
  (n : ℕ) (c : ℕ) (d : ℕ)
  (h1 : n = 40)
  (h2 : d = 3)
  (h3 : c + d * 19 = 75) :
  c = 18 :=
sorry

end cheapest_book_price_l239_239830


namespace more_silverfish_than_goldfish_l239_239109

variable (n G S R : ℕ)

-- Condition 1: If the cat eats all the goldfish, the number of remaining fish is \(\frac{2}{3}\)n - 1
def condition1 := n - G = (2 * n) / 3 - 1

-- Condition 2: If the cat eats all the redfish, the number of remaining fish is \(\frac{2}{3}\)n + 4
def condition2 := n - R = (2 * n) / 3 + 4

-- The goal: Silverfish are more numerous than goldfish by 2
theorem more_silverfish_than_goldfish (h1 : condition1 n G) (h2 : condition2 n R) :
  S = (n / 3) + 3 → G = (n / 3) + 1 → S - G = 2 :=
by
  sorry

end more_silverfish_than_goldfish_l239_239109


namespace find_a2_l239_239567

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def sum_geom_seq (a : ℕ → ℕ) (q : ℕ) (n : ℕ) := (a 0 * (1 - q^(n + 1))) / (1 - q)

-- Given conditions
def a_n : ℕ → ℕ := sorry -- Define the sequence a_n
def q : ℕ := 2
def S_4 := 60

-- The theorem to be proved
theorem find_a2 (h1: is_geometric_sequence a_n q)
                (h2: sum_geom_seq a_n q 3 = S_4) : 
                a_n 1 = 8 :=
sorry

end find_a2_l239_239567


namespace line_in_first_and_third_quadrants_l239_239712

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l239_239712


namespace apartment_living_room_size_l239_239873

theorem apartment_living_room_size :
  (∀ (a_total r_total r_living : ℝ), a_total = 160 → r_total = 6 → (∃ r_other, r_total = 5 + 1 ∧ r_living = 3 * r_other) → r_living = 60) :=
by
  intros a_total r_total r_living a_total_eq r_total_eq h
  cases h with r_other h'
  cases h' with r_total_eq' r_living_eq
  have r_other_eq : r_other = 20 :=
    by
      exact (by linarith : r_other = 20)
  rw [r_other_eq] at r_living_eq
  exact r_living_eq

end apartment_living_room_size_l239_239873


namespace find_quadratic_function_l239_239561

def quad_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_quadratic_function : ∃ (a b c : ℝ), 
  (∀ x : ℝ, quad_function a b c x = 2 * x^2 + 4 * x - 1) ∧ 
  (quad_function a b c (-1) = -3) ∧ 
  (quad_function a b c 1 = 5) :=
sorry

end find_quadratic_function_l239_239561


namespace quadratic_inequality_solution_l239_239255

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ ax^2 + bx + c > 0) :
  ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - bx + c > 0 := 
sorry

end quadratic_inequality_solution_l239_239255


namespace take_home_pay_is_correct_l239_239415

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l239_239415


namespace horse_food_per_day_l239_239212

theorem horse_food_per_day
  (total_horse_food_per_day : ℕ)
  (sheep_count : ℕ)
  (sheep_to_horse_ratio : ℕ)
  (horse_to_sheep_ratio : ℕ)
  (horse_food_per_horse_per_day : ℕ) :
  sheep_to_horse_ratio * horse_food_per_horse_per_day = total_horse_food_per_day / (sheep_count / sheep_to_horse_ratio * horse_to_sheep_ratio) :=
by
  -- Given
  let total_horse_food_per_day := 12880
  let sheep_count := 24
  let sheep_to_horse_ratio := 3
  let horse_to_sheep_ratio := 7

  -- We need to show that horse_food_per_horse_per_day = 230
  have horse_count : ℕ := (sheep_count / sheep_to_horse_ratio) * horse_to_sheep_ratio
  have horse_food_per_horse_per_day : ℕ := total_horse_food_per_day / horse_count

  -- Desired proof statement
  sorry

end horse_food_per_day_l239_239212


namespace Wendy_bouquets_l239_239937

def num_flowers_before : ℕ := 45
def num_wilted_flowers : ℕ := 35
def flowers_per_bouquet : ℕ := 5

theorem Wendy_bouquets : (num_flowers_before - num_wilted_flowers) / flowers_per_bouquet = 2 := by
  sorry

end Wendy_bouquets_l239_239937


namespace positive_numbers_l239_239083

theorem positive_numbers 
    (a b c : ℝ) 
    (h1 : a + b + c > 0) 
    (h2 : ab + bc + ca > 0) 
    (h3 : abc > 0) 
    : a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end positive_numbers_l239_239083


namespace jebb_take_home_pay_l239_239412

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l239_239412


namespace scientific_notation_of_308000000_l239_239978

theorem scientific_notation_of_308000000 :
  ∃ (a : ℝ) (n : ℤ), (a = 3.08) ∧ (n = 8) ∧ (308000000 = a * 10 ^ n) :=
by
  sorry

end scientific_notation_of_308000000_l239_239978


namespace vitya_catches_up_in_5_minutes_l239_239496

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l239_239496


namespace smallest_composite_square_side_length_l239_239237

theorem smallest_composite_square_side_length (n : ℕ) (h : ∃ k, 14 * n = k^2) : 
  ∃ m : ℕ, n = 14 ∧ m = 14 :=
by
  sorry

end smallest_composite_square_side_length_l239_239237


namespace Q_div_P_eq_10_over_3_l239_239892

noncomputable def solve_Q_over_P (P Q : ℤ) :=
  (Q / P = 10 / 3)

theorem Q_div_P_eq_10_over_3 (P Q : ℤ) (x : ℝ) :
  (∀ x, x ≠ 3 → x ≠ 4 → (P / (x + 3) + Q / (x^2 - 10 * x + 16) = (x^2 - 6 * x + 18) / (x^3 - 7 * x^2 + 14 * x - 48))) →
  solve_Q_over_P P Q :=
sorry

end Q_div_P_eq_10_over_3_l239_239892


namespace simplify_eval_l239_239601

theorem simplify_eval (a : ℝ) (h : a = Real.sqrt 3 / 3) : (a + 1) ^ 2 + a * (1 - a) = Real.sqrt 3 + 1 := 
by
  sorry

end simplify_eval_l239_239601


namespace min_f_value_l239_239697

noncomputable def f (a b : ℝ) := 
  Real.sqrt (2 * a^2 - 8 * a + 10) + 
  Real.sqrt (b^2 - 6 * b + 10) + 
  Real.sqrt (2 * a^2 - 2 * a * b + b^2)

theorem min_f_value : ∃ a b : ℝ, f a b = 2 * Real.sqrt 5 :=
sorry

end min_f_value_l239_239697


namespace both_locks_stall_time_l239_239740

-- Definitions of the conditions
def first_lock_time : ℕ := 5
def second_lock_time : ℕ := 3 * first_lock_time - 3
def both_locks_time : ℕ := 5 * second_lock_time

-- The proof statement
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end both_locks_stall_time_l239_239740


namespace serving_ways_correct_l239_239606

open Finset

def meal := {b: ℕ // b < 3}

def orders : Finset (Fin 10 × meal) := 
  {(⟨0, _⟩, ⟨0, _⟩), (⟨1, _⟩, ⟨0, _⟩), (⟨2, _⟩, ⟨0, _⟩), (⟨3, _⟩, ⟨0, _⟩), 
   (⟨4, _⟩, ⟨1, _⟩), (⟨5, _⟩, ⟨1, _⟩), (⟨6, _⟩, ⟨1, _⟩), 
   (⟨7, _⟩, ⟨2, _⟩), (⟨8, _⟩, ⟨2, _⟩), (⟨9, _⟩, ⟨2, _⟩)}

def servers := univ.perm

noncomputable def valid_serving_ways : ℕ := 
  let ways := (servers.filter (λ f: perm (Fin 10), (orders.filter (λ (o: Fin 10 × meal), (f o.1).fst = o.1)).card = 2)).card
  in ways

theorem serving_ways_correct : valid_serving_ways = 288 := sorry

end serving_ways_correct_l239_239606


namespace total_toys_is_correct_l239_239734

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end total_toys_is_correct_l239_239734


namespace maximum_sum_is_42_l239_239970

-- Definitions according to the conditions in the problem

def initial_faces : ℕ := 7 -- 2 pentagonal + 5 rectangular
def initial_vertices : ℕ := 10 -- 5 at the top and 5 at the bottom
def initial_edges : ℕ := 15 -- 5 for each pentagon and 5 linking them

def added_faces : ℕ := 5 -- 5 new triangular faces
def added_vertices : ℕ := 1 -- 1 new vertex at the apex of the pyramid
def added_edges : ℕ := 5 -- 5 new edges connecting the new vertex to the pentagon's vertices

-- New quantities after adding the pyramid
def new_faces : ℕ := initial_faces - 1 + added_faces
def new_vertices : ℕ := initial_vertices + added_vertices
def new_edges : ℕ := initial_edges + added_edges

-- Sum of the new shape's characteristics
def sum_faces_vertices_edges : ℕ := new_faces + new_vertices + new_edges

-- Statement to be proved
theorem maximum_sum_is_42 : sum_faces_vertices_edges = 42 := by
  sorry

end maximum_sum_is_42_l239_239970


namespace impossible_to_have_same_number_of_each_color_l239_239204

-- Define the initial number of coins Laura has
def initial_green : Nat := 1

-- Define the net gain in coins per transaction
def coins_gain_per_transaction : Nat := 4

-- Define a function that calculates the total number of coins after n transactions
def total_coins (n : Nat) : Nat :=
  initial_green + n * coins_gain_per_transaction

-- Define the theorem to prove that it's impossible for Laura to have the same number of red and green coins
theorem impossible_to_have_same_number_of_each_color :
  ¬ ∃ n : Nat, ∃ red green : Nat, red = green ∧ total_coins n = red + green := by
  sorry

end impossible_to_have_same_number_of_each_color_l239_239204


namespace verify_BG_BF_verify_FG_EG_find_x_l239_239857

noncomputable def verify_angles (CBG GBE EBF BCF FCE : ℝ) :=
  CBG = 20 ∧ GBE = 40 ∧ EBF = 20 ∧ BCF = 50 ∧ FCE = 30

theorem verify_BG_BF (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → BG = BF :=
by
  sorry

theorem verify_FG_EG (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → FG = EG :=
by
  sorry

theorem find_x (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → x = 30 :=
by
  sorry

end verify_BG_BF_verify_FG_EG_find_x_l239_239857


namespace pizza_order_cost_l239_239887

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end pizza_order_cost_l239_239887


namespace expected_value_X_l239_239558

open Probability

-- Define the conditions of the problem
def bag : Set ℕ := {1, 2, 3}  -- Representing red = 1, yellow = 2, blue = 3
def equal_prob : Measure ℕ := Measure.ofFinset bag (λ _, 1/3)

-- Define the event of drawing two consecutive red balls
def twoConseReds (draws : List ℕ) : Prop :=
  ∃ i, i < draws.length - 1 ∧ draws.nth i = some 1 ∧ draws.nth (i + 1) = some 1

-- Define the random variable X as the number of draws needed to achieve two consecutive red balls
noncomputable def X : Measure ℕ := Measure.map length {draws | twoConseReds draws}

-- Define the expected value of X
noncomputable def E_X : ℝ := ∫⁻ x, x.toReal ∂X

-- The proof statement
theorem expected_value_X : E_X = 12 :=
  sorry

end expected_value_X_l239_239558


namespace intersection_M_N_l239_239254

def M := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def N := { y : ℝ | y > 0 }

theorem intersection_M_N : (M ∩ N) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l239_239254


namespace total_pizza_order_cost_l239_239884

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end total_pizza_order_cost_l239_239884


namespace largest_k_for_sum_of_consecutive_integers_l239_239067

theorem largest_k_for_sum_of_consecutive_integers 
  (k : ℕ) (h : 3 ^ 12 = (finset.range k).sum(λ i, i + n)) : k = 729 :=
sorry

end largest_k_for_sum_of_consecutive_integers_l239_239067


namespace problem_solution_l239_239390

theorem problem_solution :
  (∑ k in Finset.range 6, (Nat.choose 5 k) ^ 3) =
  ∑ k, (Nat.choose 5 k) ^ 3 :=
by
  sorry

end problem_solution_l239_239390


namespace dale_pasta_l239_239663

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end dale_pasta_l239_239663


namespace find_fraction_l239_239522

theorem find_fraction
  (N : ℝ)
  (hN : N = 30)
  (h : 0.5 * N = (x / y) * N + 10):
  x / y = 1 / 6 :=
by
  sorry

end find_fraction_l239_239522


namespace total_selections_eq_525_l239_239070

open Finset

noncomputable def count_valid_selections : ℕ :=
  let S := range 21 \ {0}
  let quadruples := { x ∈ (S.product (S.product (S.product S))).to_finset | 
                    let a := x.1
                    let b := x.2.1
                    let c := x.2.2.1
                    let d := x.2.2.2
                    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ (a + c = b + d) 
                  }
  quadruples.card / 2

theorem total_selections_eq_525 : count_valid_selections = 525 :=
  sorry

end total_selections_eq_525_l239_239070


namespace percentage_increase_of_gross_sales_l239_239942

theorem percentage_increase_of_gross_sales 
  (P R : ℝ) 
  (orig_gross new_price new_qty new_gross : ℝ)
  (h1 : new_price = 0.8 * P)
  (h2 : new_qty = 1.8 * R)
  (h3 : orig_gross = P * R)
  (h4 : new_gross = new_price * new_qty) :
  ((new_gross - orig_gross) / orig_gross) * 100 = 44 :=
by sorry

end percentage_increase_of_gross_sales_l239_239942


namespace r20_moves_to_r30_probability_l239_239377

noncomputable def sequence : List ℝ := sorry

def isDistinct (l : List ℝ) : Prop :=
  l.Nodup

def swapOperation (l : List ℝ) : List ℝ :=
  List.foldl (λ l' i =>
    if i < l'.length - 1 ∧ l'.nthLe i sorry > l'.nthLe (i + 1) sorry then
      l'.updateNth i (l'.nthLe (i + 1) sorry)
        |> List.updateNth (i + 1) (l'.nthLe i sorry)
    else l') l (List.range (l.length - 1))

def probability : ℚ :=
  1 / 930

theorem r20_moves_to_r30_probability (l : List ℝ) (h : isDistinct l) (h_length : l.length = 40) :
  let result := swapOperation l
  classical.some (Rat.mk_eq probability (1 / 930)) + classical.some (Rat.mk_eq probability (1 / 930)) = 931 := 
  sorry

end r20_moves_to_r30_probability_l239_239377


namespace dealer_sold_70_hondas_l239_239198

theorem dealer_sold_70_hondas
  (total_cars: ℕ)
  (percent_audi percent_toyota percent_acura percent_honda : ℝ)
  (total_audi := total_cars * percent_audi)
  (total_toyota := total_cars * percent_toyota)
  (total_acura := total_cars * percent_acura)
  (total_honda := total_cars * percent_honda )
  (h1 : total_cars = 200)
  (h2 : percent_audi = 0.15)
  (h3 : percent_toyota = 0.22)
  (h4 : percent_acura = 0.28)
  (h5 : percent_honda = 1 - (percent_audi + percent_toyota + percent_acura))
  : total_honda = 70 := 
  by
  sorry

end dealer_sold_70_hondas_l239_239198


namespace range_of_m_l239_239394

noncomputable def f (x : ℝ) : ℝ := sorry -- to be defined as an odd, decreasing function

theorem range_of_m 
  (hf_odd : ∀ x, f (-x) = -f x) -- f is odd
  (hf_decreasing : ∀ x y, x < y → f y < f x) -- f is strictly decreasing
  (h_condition : ∀ m, f (1 - m) + f (1 - m^2) < 0) :
  ∀ m, (0 < m ∧ m < 1) :=
sorry

end range_of_m_l239_239394


namespace frankie_pets_l239_239996

variable {C S P D : ℕ}

theorem frankie_pets (h1 : S = C + 6) (h2 : P = C - 1) (h3 : C + D = 6) (h4 : C + S + P + D = 19) : 
  C + S + P + D = 19 :=
  by sorry

end frankie_pets_l239_239996


namespace max_value_of_f_l239_239150

noncomputable def f (x : Real) := 2 * (Real.sin x) ^ 2 - (Real.tan x) ^ 2

theorem max_value_of_f : 
  ∃ (x : Real), f x = 3 - 2 * Real.sqrt 2 := 
sorry

end max_value_of_f_l239_239150


namespace range_of_a_minus_b_l239_239694

theorem range_of_a_minus_b {a b : ℝ} (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) : -6 < a - b ∧ a - b < 1 :=
by
  sorry -- The proof is skipped as per the instructions.

end range_of_a_minus_b_l239_239694


namespace int_squares_l239_239063

theorem int_squares (n : ℕ) (h : ∃ k : ℕ, n^4 - n^3 + 3 * n^2 + 5 = k^2) : n = 2 := by
  sorry

end int_squares_l239_239063


namespace rice_wheat_ratio_l239_239143

theorem rice_wheat_ratio (total_shi : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) (total_sample : ℕ) : 
  total_shi = 1512 ∧ sample_size = 216 ∧ wheat_in_sample = 27 ∧ total_sample = 1512 * (wheat_in_sample / sample_size) →
  total_sample = 189 :=
by
  intros h
  sorry

end rice_wheat_ratio_l239_239143


namespace inscribed_circle_theta_l239_239525

/-- Given that a circle inscribed in triangle ABC is tangent to sides BC, CA, and AB at points
    where the tangential angles are 120 degrees, 130 degrees, and theta degrees respectively,
    we need to prove that theta is 110 degrees. -/
theorem inscribed_circle_theta 
  (ABC : Type)
  (A B C : ABC)
  (theta : ℝ)
  (tangent_angle_BC : ℝ)
  (tangent_angle_CA : ℝ) 
  (tangent_angle_AB : ℝ) 
  (h1 : tangent_angle_BC = 120)
  (h2 : tangent_angle_CA = 130) 
  (h3 : tangent_angle_AB = theta) : 
  theta = 110 :=
by
  sorry

end inscribed_circle_theta_l239_239525


namespace friend_cutoff_fraction_l239_239478

-- Definitions based on problem conditions
def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def days_biking : ℕ := 1
def days_bus : ℕ := 3
def days_friend : ℕ := 1
def total_weekly_commuting_time : ℕ := 160

-- Lean theorem statement
theorem friend_cutoff_fraction (F : ℕ) (hF : days_biking * biking_time + days_bus * bus_time + days_friend * F = total_weekly_commuting_time) :
  (biking_time - F) / biking_time = 2 / 3 :=
by
  sorry

end friend_cutoff_fraction_l239_239478


namespace difference_of_squares_example_l239_239628

theorem difference_of_squares_example : 169^2 - 168^2 = 337 :=
by
  -- The proof steps using the difference of squares formula is omitted here.
  sorry

end difference_of_squares_example_l239_239628


namespace ratio_of_flour_to_eggs_l239_239972

theorem ratio_of_flour_to_eggs (F E : ℕ) (h1 : E = 60) (h2 : F + E = 90) : F / 30 = 1 ∧ E / 30 = 2 := by
  sorry

end ratio_of_flour_to_eggs_l239_239972


namespace find_number_l239_239031

variable (x : ℝ)

theorem find_number (h : 20 * (x / 5) = 40) : x = 10 := by
  sorry

end find_number_l239_239031


namespace first_part_is_7613_l239_239946

theorem first_part_is_7613 :
  ∃ (n : ℕ), ∃ (d : ℕ), d = 3 ∧ (761 * 10 + d) * 1000 + 829 = n ∧ (n % 9 = 0) ∧ (761 * 10 + d = 7613) := 
by
  sorry

end first_part_is_7613_l239_239946


namespace sum_values_l239_239579

noncomputable def abs_eq_4 (x : ℝ) : Prop := |x| = 4
noncomputable def abs_eq_5 (x : ℝ) : Prop := |x| = 5

theorem sum_values (a b : ℝ) (h₁ : abs_eq_4 a) (h₂ : abs_eq_5 b) :
  a + b = 9 ∨ a + b = -1 ∨ a + b = 1 ∨ a + b = -9 := 
by
  -- Proof is omitted
  sorry

end sum_values_l239_239579


namespace length_of_AD_l239_239401

theorem length_of_AD (AB BC AC AD DC : ℝ)
    (h1 : AB = BC)
    (h2 : AD = 2 * DC)
    (h3 : AC = AD + DC)
    (h4 : AC = 27) : AD = 18 := 
by
  sorry

end length_of_AD_l239_239401


namespace T_2016_eq_l239_239156

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def b_n (n : ℕ) : ℚ :=
  1 / ((n + 1) * a_n n)

noncomputable def T_n (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, b_n (i + 1))

theorem T_2016_eq : T_n 2016 = 2016 / 2017 := by
  sorry

end T_2016_eq_l239_239156


namespace arrival_time_at_work_l239_239132

-- Conditions
def pickup_time : String := "06:00"
def travel_to_station : ℕ := 40 -- in minutes
def travel_from_station : ℕ := 140 -- in minutes

-- Prove the arrival time is 9:00 a.m.
theorem arrival_time_at_work : 
  (arrival_time : String) :=
by
  -- Assume the conditions
  let initial_time := "06:00"
  let first_station_time := "06:40"
  let work_arrival_time := "09:00"
  sorry -- Proof

end arrival_time_at_work_l239_239132


namespace gcd_fact_8_10_l239_239555

-- Definitions based on the conditions in a)
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Question and conditions translated to a proof problem in Lean
theorem gcd_fact_8_10 : Nat.gcd (fact 8) (fact 10) = 40320 := by
  sorry

end gcd_fact_8_10_l239_239555


namespace parabola_equation_l239_239526

theorem parabola_equation (p : ℝ) (hp : 0 < p)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (A B : ℝ × ℝ)
  (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (h_intersect : y1^2 = 2*p*x1 ∧ y2^2 = 2*p*x2)
  (M : ℝ × ℝ) (hM : M = ((x1 + x2) / 2, (y1 + y2) / 2))
  (hM_coords : M = (3, 2)) :
  p = 2 ∨ p = 4 :=
sorry

end parabola_equation_l239_239526


namespace expected_value_m_plus_n_l239_239034

-- Define the main structures and conditions
def spinner_sectors : List ℚ := [-1.25, -1, 0, 1, 1.25]
def initial_value : ℚ := 1

-- Define a function that returns the largest expected value on the paper
noncomputable def expected_largest_written_value (sectors : List ℚ) (initial : ℚ) : ℚ :=
  -- The expected value calculation based on the problem and solution analysis
  11/6  -- This is derived from the correct solution steps not shown here

-- Define the final claim
theorem expected_value_m_plus_n :
  let m := 11
  let n := 6
  expected_largest_written_value spinner_sectors initial_value = 11/6 → m + n = 17 :=
by sorry

end expected_value_m_plus_n_l239_239034


namespace find_floor_at_same_time_l239_239605

def timeTaya (n : ℕ) : ℕ := 15 * (n - 22)
def timeJenna (n : ℕ) : ℕ := 120 + 3 * (n - 22)

theorem find_floor_at_same_time (n : ℕ) : n = 32 :=
by
  -- The goal is to show that Taya and Jenna arrive at the same floor at the same time
  have ht : 15 * (n - 22) = timeTaya n := rfl
  have hj : 120 + 3 * (n - 22) = timeJenna n := rfl
  -- equate the times
  have h : timeTaya n = timeJenna n := by sorry
  -- solving the equation for n = 32
  sorry

end find_floor_at_same_time_l239_239605


namespace complex_square_l239_239696

theorem complex_square (a b : ℤ) (i : ℂ) (h1: a = 5) (h2: b = 3) (h3: i^2 = -1) :
  ((↑a) + (↑b) * i)^2 = 16 + 30 * i := by
  sorry

end complex_square_l239_239696


namespace circle_center_coordinates_l239_239896

theorem circle_center_coordinates (x y : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → (x = 1 ∧ y = -2) :=
by
  sorry

end circle_center_coordinates_l239_239896


namespace triangle_inradius_l239_239460

theorem triangle_inradius (p A : ℝ) (h_p : p = 20) (h_A : A = 30) : 
  ∃ r : ℝ, r = 3 ∧ A = r * p / 2 :=
by
  sorry

end triangle_inradius_l239_239460


namespace system_of_equations_soln_l239_239986

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l239_239986


namespace total_toys_is_correct_l239_239733

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end total_toys_is_correct_l239_239733


namespace greatest_three_digit_multiple_of_17_l239_239172

theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ≤ 999 ∧ n ≥ 100 ∧ (∃ k : ℕ, n = 17 * k) ∧ 
  (∀ m : ℕ, m ≤ 999 → m ≥ 100 → (∃ k : ℕ, m = 17 * k) → m ≤ n) ∧ n = 986 := 
sorry

end greatest_three_digit_multiple_of_17_l239_239172


namespace soccer_team_arrangements_l239_239577

theorem soccer_team_arrangements : 
  ∃ (n : ℕ), n = 2 * (Nat.factorial 11)^2 := 
sorry

end soccer_team_arrangements_l239_239577


namespace domain_implies_range_a_range_implies_range_a_l239_239684

theorem domain_implies_range_a {a : ℝ} :
  (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 :=
sorry

theorem range_implies_range_a {a : ℝ} :
  (∀ y : ℝ, ∃ x : ℝ, ax^2 + 2 * a * x + 1 = y) → 1 ≤ a :=
sorry

end domain_implies_range_a_range_implies_range_a_l239_239684


namespace tangent_line_eqn_l239_239376

theorem tangent_line_eqn (r x0 y0 : ℝ) (h : x0^2 + y0^2 = r^2) : 
  ∃ a b c : ℝ, a = x0 ∧ b = y0 ∧ c = r^2 ∧ (a*x + b*y = c) :=
sorry

end tangent_line_eqn_l239_239376


namespace arithmetic_seq_problem_l239_239869

theorem arithmetic_seq_problem (S : ℕ → ℤ) (n : ℕ) (h1 : S 6 = 36) 
                               (h2 : S n = 324) (h3 : S (n - 6) = 144) (hn : n > 6) : 
  n = 18 := 
sorry

end arithmetic_seq_problem_l239_239869


namespace determinant_condition_l239_239391

variable (p q r s : ℝ)

theorem determinant_condition (h: p * s - q * r = 5) :
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 20 :=
by
  sorry

end determinant_condition_l239_239391


namespace minimum_dot_product_l239_239384

-- Definitions of points A and B
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (2, 0)

-- Definition of condition that P lies on the line x - y + 1 = 0
def onLineP (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

-- Definition of dot product between vectors PA and PB
def dotProduct (P A B : ℝ × ℝ) : ℝ := 
  let PA := (P.1 - A.1, P.2 - A.2)
  let PB := (P.1 - B.1, P.2 - B.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- Lean 4 theorem statement
theorem minimum_dot_product (P : ℝ × ℝ) (hP : onLineP P) : 
  dotProduct P pointA pointB = 0 := 
sorry

end minimum_dot_product_l239_239384


namespace triangle_area_hyperbola_l239_239686

noncomputable def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 24) = 1

def foci_F1 : ℝ × ℝ := (-5, 0)
def foci_F2 : ℝ × ℝ := (5, 0)

theorem triangle_area_hyperbola 
  (P : ℝ × ℝ)
  (on_hyperbola : hyperbola_eq P.1 P.2)
  (right_branch : P.1 > 0)
  (distance_relation : ∀ P : ℝ × ℝ, dist P foci_F1 = (4/3) * dist P foci_F2) :
  ∃ (area : ℝ), area = 24 :=
by
  sorry

end triangle_area_hyperbola_l239_239686


namespace solution_set_of_inequality_l239_239612

theorem solution_set_of_inequality (x : ℝ) : x * (x + 2) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 0 := 
sorry

end solution_set_of_inequality_l239_239612


namespace max_ab_upper_bound_l239_239559

noncomputable def circle_center_coords : ℝ × ℝ :=
  let center_x := -1
  let center_y := 2
  (center_x, center_y)

noncomputable def max_ab_value (a b : ℝ) : ℝ :=
  if a = 1 - 2 * b then a * b else 0

theorem max_ab_upper_bound :
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let line_cond : ℝ × ℝ := (-1, 2)
  (circle_center_coords = line_cond) →
  (∀ a b : ℝ, max_ab_value a b ≤ 1 / 8) :=
by
  intro circle line_cond h
  -- Proof is omitted as per instruction
  sorry

end max_ab_upper_bound_l239_239559


namespace matching_times_l239_239586

noncomputable def chargeAtTime (t : Nat) : ℚ :=
  100 - t / 6

def isMatchingTime (hh mm : Nat) : Prop :=
  hh * 60 + mm = 100 - (hh * 60 + mm) / 6

theorem matching_times:
  isMatchingTime 4 52 ∨
  isMatchingTime 5 43 ∨
  isMatchingTime 6 35 ∨
  isMatchingTime 7 26 ∨
  isMatchingTime 9 9 :=
by
  repeat { sorry }

end matching_times_l239_239586


namespace exists_duplicate_in_grid_of_differences_bounded_l239_239852

theorem exists_duplicate_in_grid_of_differences_bounded :
  ∀ (f : ℕ × ℕ → ℤ), 
  (∀ i j, i < 10 → j < 10 → (i + 1 < 10 → (abs (f (i, j) - f (i + 1, j)) ≤ 5)) 
                             ∧ (j + 1 < 10 → (abs (f (i, j) - f (i, j + 1)) ≤ 5))) → 
  ∃ x y : ℕ × ℕ, x ≠ y ∧ f x = f y :=
by
  intros
  sorry -- Proof goes here

end exists_duplicate_in_grid_of_differences_bounded_l239_239852


namespace unit_prices_max_books_l239_239636

-- Definitions based on conditions 1 and 2
def unit_price_A (x : ℝ) : Prop :=
  x > 5 ∧ (1200 / x = 900 / (x - 5))

-- Definitions based on conditions 3, 4, and 5
def max_books_A (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 300 ∧ 0.9 * 20 * y + 15 * (300 - y) ≤ 5100

theorem unit_prices
  (x : ℝ)
  (h : unit_price_A x) :
  x = 20 ∧ x - 5 = 15 :=
sorry

theorem max_books
  (y : ℝ)
  (hy : max_books_A y) :
  y ≤ 200 :=
sorry

end unit_prices_max_books_l239_239636


namespace scientific_notation_of_area_l239_239759

theorem scientific_notation_of_area :
  (0.0000064 : ℝ) = 6.4 * 10 ^ (-6) := 
sorry

end scientific_notation_of_area_l239_239759


namespace k_positive_first_third_quadrants_l239_239708

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l239_239708


namespace system_solve_l239_239578

theorem system_solve (x y : ℚ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 2 * y = 12) : x + y = 3 / 7 :=
by
  -- The proof will go here, but we skip it for now.
  sorry

end system_solve_l239_239578


namespace contradiction_proof_l239_239170

theorem contradiction_proof (a b : ℝ) : a + b = 12 → ¬ (a < 6 ∧ b < 6) :=
by
  intro h
  intro h_contra
  sorry

end contradiction_proof_l239_239170


namespace minimum_value_of_f_l239_239432

noncomputable def f (x y z : ℝ) : ℝ := (1 / (x + y)) + (1 / (x + z)) + (1 / (y + z)) - (x * y * z)

theorem minimum_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → f x y z = 1 / 2 :=
by
  sorry

end minimum_value_of_f_l239_239432


namespace janet_more_siblings_than_carlos_l239_239421

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end janet_more_siblings_than_carlos_l239_239421


namespace B_finish_work_in_10_days_l239_239634

variable (W : ℝ) -- amount of work
variable (x : ℝ) -- number of days B can finish the work alone

theorem B_finish_work_in_10_days (h1 : ∀ A_rate, A_rate = W / 4)
                                (h2 : ∀ B_rate, B_rate = W / x)
                                (h3 : ∀ Work_done_together Remaining_work,
                                      Work_done_together = 2 * (W / 4 + W / x) ∧
                                      Remaining_work = W - Work_done_together ∧
                                      Remaining_work = (W / x) * 3.0000000000000004) :
  x = 10 :=
by
  sorry

end B_finish_work_in_10_days_l239_239634


namespace line_equation_l239_239454

theorem line_equation (m b : ℝ) (h_slope : m = 3) (h_intercept : b = 4) :
  3 * x - y + 4 = 0 :=
by
  sorry

end line_equation_l239_239454


namespace task_D_is_suitable_l239_239510

-- Definitions of the tasks
def task_A := "Investigating the age distribution of your classmates"
def task_B := "Understanding the ratio of male to female students in the eighth grade of your school"
def task_C := "Testing the urine samples of athletes who won championships at the Olympics"
def task_D := "Investigating the sleeping conditions of middle school students in Lishui City"

-- Definition of suitable_for_sampling_survey condition
def suitable_for_sampling_survey (task : String) : Prop :=
  task = task_D

-- Theorem statement
theorem task_D_is_suitable : suitable_for_sampling_survey task_D := by
  -- the proof is omitted
  sorry

end task_D_is_suitable_l239_239510


namespace kaleb_lives_left_l239_239014

theorem kaleb_lives_left (initial_lives : ℕ) (lives_lost : ℕ) (remaining_lives : ℕ) :
  initial_lives = 98 → lives_lost = 25 → remaining_lives = initial_lives - lives_lost → remaining_lives = 73 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end kaleb_lives_left_l239_239014


namespace revenue_growth_20_percent_l239_239305

noncomputable def revenue_increase (R2000 R2003 R2005 : ℝ) : ℝ :=
  ((R2005 - R2003) / R2003) * 100

theorem revenue_growth_20_percent (R2000 : ℝ) (h1 : R2003 = 1.5 * R2000) (h2 : R2005 = 1.8 * R2000) :
  revenue_increase R2000 R2003 R2005 = 20 :=
by
  sorry

end revenue_growth_20_percent_l239_239305


namespace exam_correct_answers_l239_239403

theorem exam_correct_answers (C W : ℕ) 
  (h1 : C + W = 60)
  (h2 : 4 * C - W = 160) : 
  C = 44 :=
sorry

end exam_correct_answers_l239_239403


namespace greening_investment_growth_l239_239192

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end greening_investment_growth_l239_239192


namespace tim_total_payment_correct_l239_239004

-- Define the conditions stated in the problem
def doc_visit_cost : ℝ := 300
def insurance_coverage_percent : ℝ := 0.75
def cat_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60

-- Define the amounts covered by insurance 
def insurance_coverage_amount : ℝ := doc_visit_cost * insurance_coverage_percent
def tim_payment_for_doc_visit : ℝ := doc_visit_cost - insurance_coverage_amount
def tim_payment_for_cat_visit : ℝ := cat_visit_cost - pet_insurance_coverage

-- Define the total payment Tim needs to make
def tim_total_payment : ℝ := tim_payment_for_doc_visit + tim_payment_for_cat_visit

-- State the main theorem
theorem tim_total_payment_correct : tim_total_payment = 135 := by
  sorry

end tim_total_payment_correct_l239_239004


namespace masha_number_l239_239130

theorem masha_number (x : ℝ) (n : ℤ) (ε : ℝ) (h1 : 0 ≤ ε) (h2 : ε < 1) (h3 : x = n + ε) (h4 : (n : ℝ) = 0.57 * x) : x = 100 / 57 :=
by
  sorry

end masha_number_l239_239130


namespace largest_k_consecutive_sum_l239_239066

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum_l239_239066


namespace sum_g_h_l239_239337

theorem sum_g_h (d g h : ℝ) 
  (h1 : (8 * d^2 - 4 * d + g) * (4 * d^2 + h * d + 7) = 32 * d^4 + (4 * h - 16) * d^3 - (14 * d^2 - 28 * d - 56)) :
  g + h = -8 :=
sorry

end sum_g_h_l239_239337


namespace find_a_l239_239570

-- Define the function f given a parameter a
def f (x a : ℝ) : ℝ := x^3 - 3*x^2 + a

-- Condition: f(x+1) is an odd function
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f (-(x+1)) a = -f (x+1) a) : a = 2 := 
sorry

end find_a_l239_239570


namespace inverse_function_log_l239_239251

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem inverse_function_log (a : ℝ) (g : ℝ → ℝ) (x : ℝ) (y : ℝ) :
  (a > 0) → (a ≠ 1) → 
  (f 2 a = 4) → 
  (f y a = x) → 
  (g x = y) → 
  g x = Real.logb 2 x := 
by
  intros ha hn hfx hfy hg
  sorry

end inverse_function_log_l239_239251


namespace negation_of_proposition_l239_239372

-- Definitions of the conditions
variables (a b c : ℝ) 

-- Prove the mathematically equivalent statement:
theorem negation_of_proposition :
  (a + b + c ≠ 1) → (a^2 + b^2 + c^2 > 1 / 9) :=
sorry

end negation_of_proposition_l239_239372


namespace circle_passing_through_pole_l239_239116

noncomputable def equation_of_circle (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.cos θ

theorem circle_passing_through_pole :
  equation_of_circle 2 θ := 
sorry

end circle_passing_through_pole_l239_239116


namespace fraction_zero_implies_x_is_minus_5_l239_239268

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end fraction_zero_implies_x_is_minus_5_l239_239268


namespace expression_value_l239_239538

def a : ℕ := 45
def b : ℕ := 18
def c : ℕ := 10

theorem expression_value :
  (a + b)^2 - (a^2 + b^2 + c) = 1610 := by
  sorry

end expression_value_l239_239538


namespace example_theorem_l239_239571

def not_a_term : Prop := ∀ n : ℕ, ¬ (24 - 2 * n = 3)

theorem example_theorem : not_a_term :=
  by sorry

end example_theorem_l239_239571


namespace flagpole_height_l239_239314

theorem flagpole_height :
  ∃ (AB AC AD DE DC : ℝ), 
    AC = 5 ∧
    AD = 3 ∧ 
    DE = 1.8 ∧
    DC = AC - AD ∧
    AB = (DE * AC) / DC ∧
    AB = 4.5 :=
by
  exists 4.5, 5, 3, 1.8, 2
  simp
  sorry

end flagpole_height_l239_239314


namespace sum_of_roots_l239_239360

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l239_239360


namespace part_one_solution_set_part_two_m_range_l239_239387

theorem part_one_solution_set (m : ℝ) (x : ℝ) (h : m = 0) : ((m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

theorem part_two_m_range (m : ℝ) : (∀ x : ℝ, (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
by
  sorry

end part_one_solution_set_part_two_m_range_l239_239387


namespace B_is_1_and_2_number_of_sets_A_l239_239380

open Set

variable (A B : Set ℝ)
variable (f : ℝ → ℝ)

-- Define the function
noncomputable def f (x : ℝ) : ℝ := abs x + 1

-- Condition: A = {-1, 0, 1}
def cond_A : A = ({-1, 0, 1} : Set ℝ) := rfl

-- Condition: B has exactly 2 elements
def cond_B_two_elements : B.Card = 2 := sorry

-- Prove that if A = {-1, 0, 1} and B has 2 elements, then B = {1, 2}
theorem B_is_1_and_2 (hA : A = {-1, 0, 1}) (hb : B.Card = 2) : B = {1, 2} :=
  sorry

-- Prove number of sets A that map to {1, 2} under the function f
theorem number_of_sets_A :
  let B := {1, 2}
  in (card (filter (λ (s : Set ℝ), image (f) s = B) (powerset {-1, 0, 1}))) = 7 :=
  sorry

end B_is_1_and_2_number_of_sets_A_l239_239380


namespace vitya_catchup_time_l239_239486

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l239_239486


namespace sum_of_numbers_l239_239068

theorem sum_of_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := 
by
  sorry

end sum_of_numbers_l239_239068


namespace total_weight_of_ripe_apples_is_1200_l239_239549

def total_apples : Nat := 14
def weight_ripe_apple : Nat := 150
def weight_unripe_apple : Nat := 120
def unripe_apples : Nat := 6
def ripe_apples : Nat := total_apples - unripe_apples
def total_weight_ripe_apples : Nat := ripe_apples * weight_ripe_apple

theorem total_weight_of_ripe_apples_is_1200 :
  total_weight_ripe_apples = 1200 := by
  sorry

end total_weight_of_ripe_apples_is_1200_l239_239549


namespace largest_b_for_denom_has_nonreal_roots_l239_239230

theorem largest_b_for_denom_has_nonreal_roots :
  ∃ b : ℤ, 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) 
  ∧ (∀ b' : ℤ, (∀ x : ℝ, x^2 + (b' : ℝ) * x + 12 ≠ 0) → b' ≤ b)
  ∧ b = 6 :=
sorry

end largest_b_for_denom_has_nonreal_roots_l239_239230


namespace change_in_y_when_x_increases_l239_239816

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- State the theorem
theorem change_in_y_when_x_increases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -5 :=
by
  sorry

end change_in_y_when_x_increases_l239_239816


namespace parity_of_expression_l239_239842

theorem parity_of_expression (a b c : ℕ) (ha : a % 2 = 1) (hb : b % 2 = 0) :
  (3 ^ a + (b - 1) ^ 2 * (c + 1)) % 2 = if c % 2 = 0 then 1 else 0 :=
by
  sorry

end parity_of_expression_l239_239842


namespace correct_statement_is_B_l239_239622

-- Define integers and zero
def is_integer (n : ℤ) : Prop := True
def is_zero (n : ℤ) : Prop := n = 0

-- Define rational numbers
def is_rational (q : ℚ) : Prop := True

-- Positive and negative zero cannot co-exist
def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0

-- Statement A: Integers and negative integers are collectively referred to as integers.
def statement_A : Prop :=
  ∀ n : ℤ, (is_positive n ∨ is_negative n) ↔ is_integer n

-- Statement B: Integers and fractions are collectively referred to as rational numbers.
def statement_B : Prop :=
  ∀ q : ℚ, is_rational q

-- Statement C: Zero can be either a positive integer or a negative integer.
def statement_C : Prop :=
  ∀ n : ℤ, is_zero n → (is_positive n ∨ is_negative n)

-- Statement D: A rational number is either a positive number or a negative number.
def statement_D : Prop :=
  ∀ q : ℚ, (q ≠ 0 → (is_positive q.num ∨ is_negative q.num))

-- The problem is to prove that statement B is the only correct statement.
theorem correct_statement_is_B : statement_B ∧ ¬statement_A ∧ ¬statement_C ∧ ¬statement_D :=
by sorry

end correct_statement_is_B_l239_239622


namespace harlys_dogs_left_l239_239574

-- Define the initial conditions
def initial_dogs : ℕ := 80
def adoption_percentage : ℝ := 0.40
def dogs_taken_back : ℕ := 5

-- Compute the number of dogs left
theorem harlys_dogs_left : (80 - (int((0.40 * 80).to_nat) - 5)) = 53 :=
by
  sorry

end harlys_dogs_left_l239_239574


namespace problem_statement_l239_239786

-- Definitions from the problem conditions
variable (r : ℝ) (A B C : ℝ)

-- Problem condition that A, B are endpoints of the diameter of the circle
-- Defining the length AB being the diameter -> length AB = 2r
def AB := 2 * r

-- Condition that ABC is inscribed in a circle and AB is the diameter implies the angle ACB = 90°
-- Using Thales' theorem we know that A, B, C satisfy certain geometric properties in a right triangle
-- AC and BC are the other two sides with H right angle at C.

-- Proving the target equation
theorem problem_statement (h : C ≠ A ∧ C ≠ B) : (AC + BC)^2 ≤ 8 * r^2 := 
sorry


end problem_statement_l239_239786


namespace dorchester_puppies_washed_l239_239224

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end dorchester_puppies_washed_l239_239224


namespace johns_weekly_earnings_percentage_increase_l239_239425

theorem johns_weekly_earnings_percentage_increase (initial final : ℝ) :
  initial = 30 →
  final = 50 →
  ((final - initial) / initial) * 100 = 66.67 :=
by
  intros h_initial h_final
  rw [h_initial, h_final]
  norm_num
  sorry

end johns_weekly_earnings_percentage_increase_l239_239425


namespace find_start_time_l239_239007

def time_first_train_started 
  (distance_pq : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (start_time_train2 : ℝ) 
  (meeting_time : ℝ) 
  (T : ℝ) : ℝ :=
  T

theorem find_start_time 
  (distance_pq : ℝ := 200)
  (speed_train1 : ℝ := 20)
  (speed_train2 : ℝ := 25)
  (start_time_train2 : ℝ := 8)
  (meeting_time : ℝ := 12) 
  : time_first_train_started distance_pq speed_train1 speed_train2 start_time_train2 meeting_time 7 = 7 :=
by
  sorry

end find_start_time_l239_239007


namespace compute_a_l239_239837

theorem compute_a (a : ℝ) (h : 2.68 * 0.74 = a) : a = 1.9832 :=
by
  -- Here skip the proof steps
  sorry

end compute_a_l239_239837


namespace total_fencing_l239_239163

def playground_side_length : ℕ := 27
def garden_length : ℕ := 12
def garden_width : ℕ := 9

def perimeter_square (side : ℕ) : ℕ := 4 * side
def perimeter_rectangle (length width : ℕ) : ℕ := 2 * length + 2 * width

theorem total_fencing (side playground_side_length : ℕ) (garden_length garden_width : ℕ) :
  perimeter_square playground_side_length + perimeter_rectangle garden_length garden_width = 150 :=
by
  sorry

end total_fencing_l239_239163


namespace find_number_of_students_l239_239106

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end find_number_of_students_l239_239106


namespace river_lengths_l239_239900

theorem river_lengths (x : ℝ) (dnieper don : ℝ)
  (h1 : dnieper = (5 / (19 / 3)) * x)
  (h2 : don = (6.5 / 9.5) * x)
  (h3 : dnieper - don = 300) :
  x = 2850 ∧ dnieper = 2250 ∧ don = 1950 :=
by
  sorry

end river_lengths_l239_239900


namespace num_sequences_eq_15_l239_239767

noncomputable def num_possible_sequences : ℕ :=
  let angles_increasing_arith_seq := ∃ (x d : ℕ), x > 0 ∧ x + 4 * d < 140 ∧ 5 * x + 10 * d = 540 ∧ d ≠ 0
  by sorry

theorem num_sequences_eq_15 : num_possible_sequences = 15 := 
  by sorry

end num_sequences_eq_15_l239_239767


namespace plane_equation_l239_239455

theorem plane_equation (x y z : ℝ) (A B C D : ℤ) (h1 : A = 9) (h2 : B = -6) (h3 : C = 4) (h4 : D = -133) (A_pos : A > 0) (gcd_condition : Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) : 
  A * x + B * y + C * z + D = 0 :=
sorry

end plane_equation_l239_239455


namespace salt_fraction_l239_239909

variables {a x : ℝ}

-- First condition: the shortfall in salt the first time
def shortfall_first (a x : ℝ) : ℝ := a - x

-- Second condition: the shortfall in salt the second time
def shortfall_second (a x : ℝ) : ℝ := a - 2 * x

-- Third condition: relationship given by the problem
axiom condition : shortfall_first a x = 2 * shortfall_second a x

-- Prove fraction of necessary salt added the first time is 1/3
theorem salt_fraction (a x : ℝ) (h : shortfall_first a x = 2 * shortfall_second a x) : x = a / 3 :=
by
  sorry

end salt_fraction_l239_239909


namespace center_radius_sum_l239_239865

theorem center_radius_sum (a b r : ℝ) (h : ∀ x y : ℝ, (x^2 - 8*x - 4*y = -y^2 + 2*y + 13) ↔ (x - 4)^2 + (y - 3)^2 = 38) :
  a = 4 ∧ b = 3 ∧ r = Real.sqrt 38 → a + b + r = 7 + Real.sqrt 38 :=
by
  sorry

end center_radius_sum_l239_239865


namespace sodas_to_take_back_l239_239232

def num_sodas_brought : ℕ := 50
def num_sodas_drank : ℕ := 38

theorem sodas_to_take_back : (num_sodas_brought - num_sodas_drank) = 12 := by
  sorry

end sodas_to_take_back_l239_239232


namespace geometric_sequence_condition_l239_239866

variable (a b c : ℝ)

-- Condition: For a, b, c to form a geometric sequence.
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ≠ 0) ∧ (b^2 = a * c)

-- Given that a, b, c are real numbers
-- Prove that ac = b^2 is a necessary but not sufficient condition for a, b, c to form a geometric sequence.
theorem geometric_sequence_condition (a b c : ℝ) (h : a * c = b^2) :
  ¬ (∃ b : ℝ, b^2 = a * c → (is_geometric_sequence a b c)) :=
sorry

end geometric_sequence_condition_l239_239866


namespace prob_both_correct_prob_at_least_one_correct_prob_exactly_3_correct_l239_239480

-- Define probabilities for satellite A and B
axiom P_A : ℚ
axiom P_B : ℚ
axiom independent_A_B : IndepEvents P_A P_B
axiom P_A_val : P_A = 4/5
axiom P_B_val : P_B = 3/4

-- Problem I
theorem prob_both_correct : P(⋂₁ (λ(ω : event_univ), A ω ∧ B ω)) = 3/5
  := by
     sorry

-- Problem II
theorem prob_at_least_one_correct : 1 - (1 - P_A) * (1 - P_B) = 19/20
  := by
     sorry

-- Define binomial probability conditions
axiom p : ℚ
axiom n : ℕ
axiom k : ℕ
axiom trials_independence : IndepTrials p n
axiom p_val : p = 4/5
axiom n_val : n = 4
axiom k_val : k = 3

-- Problem III
theorem prob_exactly_3_correct : binomial_pmf n k p = 256/625
  := by
     sorry

end prob_both_correct_prob_at_least_one_correct_prob_exactly_3_correct_l239_239480


namespace sum_term_ratio_equals_four_l239_239240

variable {a_n : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S_n : ℕ → ℝ} -- The sum of the first n terms S_n
variable {d : ℝ} -- The common difference of the sequence
variable {a_1 : ℝ} -- The first term of the sequence

-- The conditions as hypotheses
axiom a_n_formula (n : ℕ) : a_n n = a_1 + (n - 1) * d
axiom S_n_formula (n : ℕ) : S_n n = n * (a_1 + (n - 1) * d / 2)
axiom non_zero_d : d ≠ 0
axiom condition_a10_S4 : a_n 10 = S_n 4

-- The proof statement
theorem sum_term_ratio_equals_four : (S_n 8) / (a_n 9) = 4 :=
by
  sorry

end sum_term_ratio_equals_four_l239_239240


namespace original_number_of_men_l239_239640

theorem original_number_of_men (M : ℕ) : 
  (∀ t : ℕ, (t = 8) -> (8:ℕ) * M = 8 * 10 / (M - 3) ) -> ( M = 12 ) :=
by sorry

end original_number_of_men_l239_239640


namespace probability_2x_less_y_equals_one_over_eight_l239_239317

noncomputable def probability_2x_less_y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 3 * 1.5
  let area_rectangle : ℚ := 6 * 3
  area_triangle / area_rectangle

theorem probability_2x_less_y_equals_one_over_eight :
  probability_2x_less_y_in_rectangle = 1 / 8 :=
by
  sorry

end probability_2x_less_y_equals_one_over_eight_l239_239317


namespace y_relationship_range_of_x_l239_239073

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end y_relationship_range_of_x_l239_239073


namespace scientific_notation_l239_239760

theorem scientific_notation :
  (0.0000064 : ℝ) = 6.4 * 10^(-6) :=
by
  sorry

end scientific_notation_l239_239760


namespace monster_perimeter_l239_239724

theorem monster_perimeter (r : ℝ) (theta : ℝ) (h₁ : r = 2) (h₂ : theta = 90 * π / 180) :
  2 * r + (3 / 4) * (2 * π * r) = 3 * π + 4 := by
  -- Sorry to skip the proof.
  sorry

end monster_perimeter_l239_239724


namespace flour_per_new_bread_roll_l239_239632

theorem flour_per_new_bread_roll (p1 f1 p2 f2 c : ℚ)
  (h1 : p1 = 40)
  (h2 : f1 = 1 / 8)
  (h3 : p2 = 25)
  (h4 : c = p1 * f1)
  (h5 : c = p2 * f2) :
  f2 = 1 / 5 :=
by
  sorry

end flour_per_new_bread_roll_l239_239632


namespace major_axis_length_of_ellipse_l239_239250

theorem major_axis_length_of_ellipse :
  ∀ {y x : ℝ},
  (y^2 / 25 + x^2 / 15 = 1) → 
  2 * Real.sqrt 25 = 10 :=
by
  intro y x h
  sorry

end major_axis_length_of_ellipse_l239_239250


namespace prob_correct_l239_239829

-- Define the individual probabilities.
def prob_first_ring := 1 / 10
def prob_second_ring := 3 / 10
def prob_third_ring := 2 / 5
def prob_fourth_ring := 1 / 10

-- Define the total probability of answering within the first four rings.
def prob_answer_within_four_rings := 
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring

-- State the theorem.
theorem prob_correct : prob_answer_within_four_rings = 9 / 10 :=
by
  -- We insert a placeholder for the proof.
  sorry

end prob_correct_l239_239829


namespace toothbrushes_difference_l239_239976

theorem toothbrushes_difference
  (total : ℕ)
  (jan : ℕ)
  (feb : ℕ)
  (mar : ℕ)
  (apr_may_sum : total = jan + feb + mar + 164)
  (apr_may_half : 164 / 2 = 82)
  (busy_month_given : feb = 67)
  (slow_month_given : mar = 46) :
  feb - mar = 21 :=
by
  sorry

end toothbrushes_difference_l239_239976


namespace find_m_plus_b_l239_239766

-- Define the given equation
def given_line (x y : ℝ) : Prop := x - 3 * y + 11 = 0

-- Define the reflection of the given line about the x-axis
def reflected_line (x y : ℝ) : Prop := x + 3 * y + 11 = 0

-- Define the slope-intercept form of the reflected line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- State the theorem to prove
theorem find_m_plus_b (m b : ℝ) :
  (∀ x y : ℝ, reflected_line x y ↔ slope_intercept_form m b x y) → m + b = -4 :=
by
  sorry

end find_m_plus_b_l239_239766


namespace quadratic_real_roots_l239_239995

variable (a b : ℝ)

theorem quadratic_real_roots (h : ∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) : b ≤ -1/8 :=
by
  sorry

end quadratic_real_roots_l239_239995


namespace max_visible_sum_is_128_l239_239236

-- Define the structure of the problem
structure Cube :=
  (faces : Fin 6 → Nat)
  (bottom_face : Nat)
  (all_faces : ∀ i : Fin 6, i ≠ ⟨0, by decide⟩ → faces i = bottom_face → False)

-- Define the problem conditions
noncomputable def problem_conditions : Prop :=
  let cubes := [Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry]
  -- Cube stacking in two layers, with two cubes per layer
  
  true

-- Define the theorem to be proved
theorem max_visible_sum_is_128 (h : problem_conditions) : 
  ∃ (total_sum : Nat), total_sum = 128 := 
sorry

end max_visible_sum_is_128_l239_239236


namespace mixed_number_evaluation_l239_239550

theorem mixed_number_evaluation :
  let a := (4 + 1 / 3 : ℚ)
  let b := (3 + 2 / 7 : ℚ)
  let c := (2 + 5 / 6 : ℚ)
  let d := (1 + 1 / 2 : ℚ)
  let e := (5 + 1 / 4 : ℚ)
  let f := (3 + 2 / 5 : ℚ)
  (a + b - c) * (d + e) / f = 9 + 198 / 317 :=
by {
  let a : ℚ := 4 + 1 / 3
  let b : ℚ := 3 + 2 / 7
  let c : ℚ := 2 + 5 / 6
  let d : ℚ := 1 + 1 / 2
  let e : ℚ := 5 + 1 / 4
  let f : ℚ := 3 + 2 / 5
  sorry
}

end mixed_number_evaluation_l239_239550


namespace find_u_l239_239057

-- Definitions for given points lying on a straight line
def point := (ℝ × ℝ)

-- Points
def p1 : point := (2, 8)
def p2 : point := (6, 20)
def p3 : point := (10, 32)

-- Function to check if point is on the line derived from p1, p2, p3
def is_on_line (x y : ℝ) : Prop :=
  ∃ m b : ℝ, y = m * x + b ∧
  p1.2 = m * p1.1 + b ∧ 
  p2.2 = m * p2.1 + b ∧
  p3.2 = m * p3.1 + b

-- Statement to prove
theorem find_u (u : ℝ) (hu : is_on_line 50 u) : u = 152 :=
sorry

end find_u_l239_239057


namespace problem_solved_l239_239685

-- Define the function f with the given conditions
def satisfies_conditions(f : ℝ × ℝ × ℝ → ℝ) :=
  (∀ x y z t : ℝ, f (x + t, y + t, z + t) = t + f (x, y, z)) ∧
  (∀ x y z t : ℝ, f (t * x, t * y, t * z) = t * f (x, y, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (y, x, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (x, z, y))

-- We'll state the main result to be proven, without giving the proof
theorem problem_solved (f : ℝ × ℝ × ℝ → ℝ) (h : satisfies_conditions f) : f (2000, 2001, 2002) = 2001 :=
  sorry

end problem_solved_l239_239685


namespace moscow_probability_higher_l239_239936

def total_combinations : ℕ := 64 * 63

def invalid_combinations_ural : ℕ := 8 * 7 + 8 * 7

def valid_combinations_moscow : ℕ := total_combinations

def valid_combinations_ural : ℕ := total_combinations - invalid_combinations_ural

def probability_moscow : ℚ := valid_combinations_moscow / total_combinations

def probability_ural : ℚ := valid_combinations_ural / total_combinations

theorem moscow_probability_higher :
  probability_moscow > probability_ural :=
by
  unfold probability_moscow probability_ural
  unfold valid_combinations_moscow valid_combinations_ural invalid_combinations_ural total_combinations
  sorry

end moscow_probability_higher_l239_239936


namespace count_multiples_of_7_l239_239092

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l239_239092


namespace solution_set_l239_239664

open Nat

def is_solution (a b c : ℕ) : Prop :=
  a ^ (b + 20) * (c - 1) = c ^ (b + 21) - 1

theorem solution_set (a b c : ℕ) : 
  (is_solution a b c) ↔ ((c = 0 ∧ a = 1) ∨ (c = 1)) := 
sorry

end solution_set_l239_239664


namespace largest_n_for_factoring_l239_239231

theorem largest_n_for_factoring :
  ∃ (n : ℤ), 
    (∀ A B : ℤ, (5 * B + A = n ∧ A * B = 60) → (5 * B + A ≤ n)) ∧
    n = 301 :=
by sorry

end largest_n_for_factoring_l239_239231


namespace dorchester_puppy_washing_l239_239222

-- Define the conditions
def daily_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def wednesday_total_pay : ℝ := 76

-- Define the true statement
theorem dorchester_puppy_washing :
  let earnings_from_puppy_washing := wednesday_total_pay - daily_pay in
  let number_of_puppies := earnings_from_puppy_washing / pay_per_puppy in
  number_of_puppies = 16 :=
by
  -- Placeholder for the proof
  sorry

end dorchester_puppy_washing_l239_239222


namespace sharon_trip_distance_l239_239443

theorem sharon_trip_distance
  (h1 : ∀ (d : ℝ), (180 * d) = 1 ∨ (d = 0))  -- Any distance traveled in 180 minutes follows 180d=1 (usual speed)
  (h2 : ∀ (d : ℝ), (276 * (d - 20 / 60)) = 1 ∨ (d = 0))  -- With reduction in speed due to snowstorm too follows a similar relation
  (h3: ∀ (total_time : ℝ), total_time = 276 ∨ total_time = 0)  -- Total time is 276 minutes
  : ∃ (x : ℝ), x = 135 := sorry

end sharon_trip_distance_l239_239443


namespace geometric_sequence_sum_div_l239_239560

theorem geometric_sequence_sum_div :
  ∀ {a : ℕ → ℝ} {q : ℝ},
  (∀ n, a (n + 1) = a n * q) →
  q = -1 / 3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros a q geometric_seq common_ratio
  sorry

end geometric_sequence_sum_div_l239_239560


namespace five_fourths_of_eight_thirds_is_correct_l239_239982

-- Define the given fractions
def five_fourths : ℚ := 5 / 4
def eight_thirds : ℚ := 8 / 3

-- Define the expected result
def expected_result : ℚ := 10 / 3

-- Theorem to prove correctness of the computation
theorem five_fourths_of_eight_thirds_is_correct : five_fourths * eight_thirds = expected_result := by
  sorry

end five_fourths_of_eight_thirds_is_correct_l239_239982


namespace total_meat_supply_l239_239105

-- Definitions of the given conditions
def lion_consumption_per_day : ℕ := 25
def tiger_consumption_per_day : ℕ := 20
def duration_days : ℕ := 2

-- Statement of the proof problem
theorem total_meat_supply :
  (lion_consumption_per_day + tiger_consumption_per_day) * duration_days = 90 :=
by
  sorry

end total_meat_supply_l239_239105


namespace solution_set_circle_l239_239992

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l239_239992


namespace largest_band_members_l239_239793

def band_formation (m r x : ℕ) : Prop :=
  m < 100 ∧ m = r * x + 2 ∧ (r - 2) * (x + 1) = m ∧ r - 2 * x = 4

theorem largest_band_members : ∃ (r x m : ℕ), band_formation m r x ∧ m = 98 := 
  sorry

end largest_band_members_l239_239793


namespace total_journey_distance_l239_239626

variable (D : ℚ) (lateTime : ℚ := 1/4)

theorem total_journey_distance :
  (∃ (T : ℚ), T = D / 40 ∧ T + lateTime = D / 35) →
  D = 70 :=
by
  intros h
  obtain ⟨T, h1, h2⟩ := h
  have h3 : T = D / 40 := h1
  have h4 : T + lateTime = D / 35 := h2
  sorry

end total_journey_distance_l239_239626


namespace ramon_current_age_is_26_l239_239583

-- Definitions based on the problem conditions
def loui_age : Nat := 23
def ramon_age_in_20_years (ramon_current_age : Nat) : Nat := ramon_current_age + 20
def twice_loui_age : Nat := 2 * loui_age
def ramon_condition (ramon_current_age : Nat) : Prop := ramon_age_in_20_years(ramon_current_age) = twice_loui_age

-- The theorem stating the proof problem
theorem ramon_current_age_is_26 (r : Nat) 
  (h1 : loui_age = 23) 
  (h2 : ramon_condition r) : 
  r = 26 :=
sorry

end ramon_current_age_is_26_l239_239583


namespace june_eggs_count_l239_239863

theorem june_eggs_count :
  (2 * 5) + 3 + 4 = 17 := 
by 
  sorry

end june_eggs_count_l239_239863


namespace total_amount_l239_239477

noncomputable def initial_amounts (a j t : ℕ) := (t = 24)
noncomputable def redistribution_amounts (a j t a' j' t' : ℕ) :=
  a' = 3 * (2 * (a - 2 * j - 24)) ∧
  j' = 3 * (3 * j - (a - 2 * j - 24 + 48)) ∧
  t' = 144 - (6 * (a - 2 * j - 24) + 9 * j - 3 * (a - 2 * j - 24 + 48))

theorem total_amount (a j t a' j' t' : ℕ) (h1 : t = 24)
  (h2 : redistribution_amounts a j t a' j' t')
  (h3 : t' = 24) : 
  a + j + t = 72 :=
sorry

end total_amount_l239_239477


namespace parallel_lines_a_value_l239_239389

theorem parallel_lines_a_value :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
      (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l239_239389


namespace system_of_equations_soln_l239_239985

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l239_239985


namespace silvia_escalator_time_l239_239119

noncomputable def total_time_standing (v s : ℝ) : ℝ := 
  let d := 80 * v
  d / s

theorem silvia_escalator_time (v s t : ℝ) (h1 : 80 * v = 28 * (v + s)) (h2 : t = total_time_standing v s) : 
  t = 43 := by
  sorry

end silvia_escalator_time_l239_239119


namespace line_intersects_ellipse_if_and_only_if_l239_239902

theorem line_intersects_ellipse_if_and_only_if (k : ℝ) (m : ℝ) :
  (∀ x, ∃ y, y = k * x + 1 ∧ (x^2 / 5 + y^2 / m = 1)) ↔ (m ≥ 1 ∧ m ≠ 5) := 
sorry

end line_intersects_ellipse_if_and_only_if_l239_239902


namespace janet_more_siblings_than_carlos_l239_239422

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end janet_more_siblings_than_carlos_l239_239422


namespace costPrice_of_bat_is_152_l239_239950

noncomputable def costPriceOfBatForA (priceC : ℝ) (profitA : ℝ) (profitB : ℝ) : ℝ :=
  priceC / (1 + profitB) / (1 + profitA)

theorem costPrice_of_bat_is_152 :
  costPriceOfBatForA 228 0.20 0.25 = 152 :=
by
  -- Placeholder for the proof
  sorry

end costPrice_of_bat_is_152_l239_239950


namespace no_real_solution_f_of_f_f_eq_x_l239_239252

-- Defining the quadratic polynomial f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Stating the main theorem
theorem no_real_solution_f_of_f_f_eq_x (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
by 
  -- Proof will go here
  sorry

end no_real_solution_f_of_f_f_eq_x_l239_239252


namespace abigail_lost_money_l239_239205

theorem abigail_lost_money (initial_amount spent_first_store spent_second_store remaining_amount_lost: ℝ) 
  (h_initial : initial_amount = 50) 
  (h_spent_first : spent_first_store = 15.25) 
  (h_spent_second : spent_second_store = 8.75) 
  (h_remaining : remaining_amount_lost = 16) : (initial_amount - spent_first_store - spent_second_store - remaining_amount_lost = 10) :=
by
  sorry

end abigail_lost_money_l239_239205


namespace ralph_socks_l239_239137

theorem ralph_socks
  (x y w z : ℕ)
  (h1 : x + y + w + z = 15)
  (h2 : x + 2 * y + 3 * w + 4 * z = 36)
  (hx : x ≥ 1) (hy : y ≥ 1) (hw : w ≥ 1) (hz : z ≥ 1) :
  x = 5 :=
sorry

end ralph_socks_l239_239137


namespace hydrogen_atoms_in_compound_l239_239196

theorem hydrogen_atoms_in_compound :
  ∀ (molecular_weight_of_compound atomic_weight_Al atomic_weight_O atomic_weight_H : ℕ)
    (num_Al num_O num_H : ℕ),
    molecular_weight_of_compound = 78 →
    atomic_weight_Al = 27 →
    atomic_weight_O = 16 →
    atomic_weight_H = 1 →
    num_Al = 1 →
    num_O = 3 →
    molecular_weight_of_compound = 
      (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H) →
    num_H = 3 := by
  intros
  sorry

end hydrogen_atoms_in_compound_l239_239196


namespace ashley_age_l239_239624

theorem ashley_age (A M : ℕ) (h1 : 4 * M = 7 * A) (h2 : A + M = 22) : A = 8 :=
sorry

end ashley_age_l239_239624


namespace determine_f_2048_l239_239907

theorem determine_f_2048 (f : ℕ → ℝ)
  (A1 : ∀ a b n : ℕ, a > 0 → b > 0 → a * b = 2^n → f a + f b = n^2)
  : f 2048 = 121 := by
  sorry

end determine_f_2048_l239_239907


namespace distance_between_foci_l239_239761

-- Define the conditions
def is_asymptote (y x : ℝ) (slope intercept : ℝ) : Prop := y = slope * x + intercept

def passes_through_point (x y x0 y0 : ℝ) : Prop := x = x0 ∧ y = y0

-- The hyperbola conditions
axiom asymptote1 : ∀ x y : ℝ, is_asymptote y x 2 3
axiom asymptote2 : ∀ x y : ℝ, is_asymptote y x (-2) 5
axiom hyperbola_passes : passes_through_point 2 9 2 9

-- The proof problem statement: distance between the foci
theorem distance_between_foci : ∀ {a b c : ℝ}, ∃ c, (c^2 = 22.75 + 22.75) → 2 * c = 2 * Real.sqrt 45.5 :=
by
  sorry

end distance_between_foci_l239_239761


namespace sphere_volume_in_cone_l239_239203

theorem sphere_volume_in_cone (d : ℝ) (r : ℝ) (π : ℝ) (V : ℝ) (h1 : d = 12) (h2 : r = d / 2) (h3 : V = (4 / 3) * π * r^3) :
  V = 288 * π :=
by 
  sorry

end sphere_volume_in_cone_l239_239203


namespace ramon_current_age_l239_239584

variable (R : ℕ) (L : ℕ)

theorem ramon_current_age :
  (L = 23) → (R + 20 = 2 * L) → R = 26 :=
by
  intro hL hR
  rw [hL] at hR
  have : R + 20 = 46 := by linarith
  linarith

end ramon_current_age_l239_239584


namespace theodoreEarningsCorrect_l239_239159

noncomputable def theodoreEarnings : ℝ := 
  let s := 10
  let ps := 20
  let w := 20
  let pw := 5
  let b := 15
  let pb := 15
  let m := 150
  let l := 200
  let t := 0.10
  let totalEarnings := (s * ps) + (w * pw) + (b * pb)
  let expenses := m + l
  let earningsBeforeTaxes := totalEarnings - expenses
  let taxes := t * earningsBeforeTaxes
  earningsBeforeTaxes - taxes

theorem theodoreEarningsCorrect :
  theodoreEarnings = 157.50 :=
by sorry

end theodoreEarningsCorrect_l239_239159


namespace price_of_first_doughnut_l239_239608

theorem price_of_first_doughnut 
  (P : ℕ)  -- Price of the first doughnut
  (total_doughnuts : ℕ := 48)  -- Total number of doughnuts
  (price_per_dozen : ℕ := 6)  -- Price per dozen of additional doughnuts
  (total_cost : ℕ := 24)  -- Total cost spent
  (doughnuts_left : ℕ := total_doughnuts - 1)  -- Doughnuts left after the first one
  (dozens : ℕ := doughnuts_left / 12)  -- Number of whole dozens
  (cost_of_dozens : ℕ := dozens * price_per_dozen)  -- Cost of the dozens of doughnuts
  (cost_after_first : ℕ := total_cost - cost_of_dozens)  -- Remaining cost after dozens
  : P = 6 := 
by
  -- Proof to be filled in
  sorry

end price_of_first_doughnut_l239_239608


namespace collete_age_ratio_l239_239441

theorem collete_age_ratio (Ro R C : ℕ) (h1 : R = 2 * Ro) (h2 : Ro = 8) (h3 : R - C = 12) :
  C / Ro = 1 / 2 := by
sorry

end collete_age_ratio_l239_239441


namespace solution_set_circle_l239_239991

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l239_239991


namespace range_of_m_l239_239564

def p (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2^x - m + 1 > 0

def q (m : ℝ) : Prop :=
  5 - 2 * m > 1

theorem range_of_m (m : ℝ) (hpq : p m ∧ q m) : m ≤ 1 := sorry

end range_of_m_l239_239564


namespace fish_ratio_bobby_sarah_l239_239335

-- Defining the conditions
variables (bobby sarah tony billy : ℕ)

-- Condition: Billy has 10 fish.
def billy_has_10_fish : billy = 10 := by sorry

-- Condition: Tony has 3 times as many fish as Billy.
def tony_has_3_times_billy : tony = 3 * billy := by sorry

-- Condition: Sarah has 5 more fish than Tony.
def sarah_has_5_more_than_tony : sarah = tony + 5 := by sorry

-- Condition: All 4 people have 145 fish together.
def total_fish : bobby + sarah + tony + billy = 145 := by sorry

-- The theorem we want to prove
theorem fish_ratio_bobby_sarah : (bobby : ℚ) / sarah = 2 / 1 := by
  -- You can write out the entire proof step by step here, but initially, we'll just put sorry.
  sorry

end fish_ratio_bobby_sarah_l239_239335


namespace largest_4_digit_number_divisible_by_24_l239_239780

theorem largest_4_digit_number_divisible_by_24 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 24 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 24 = 0 → m ≤ n :=
sorry

end largest_4_digit_number_divisible_by_24_l239_239780


namespace y_payment_is_approximately_272_73_l239_239518

noncomputable def calc_y_payment : ℝ :=
  let total_payment : ℝ := 600
  let percent_x_to_y : ℝ := 1.2
  total_payment / (percent_x_to_y + 1)

theorem y_payment_is_approximately_272_73
  (total_payment : ℝ)
  (percent_x_to_y : ℝ)
  (h1 : total_payment = 600)
  (h2 : percent_x_to_y = 1.2) :
  calc_y_payment = 272.73 :=
by
  sorry

end y_payment_is_approximately_272_73_l239_239518


namespace largest_n_l239_239925

theorem largest_n : ∃ (n : ℕ), n < 1000 ∧ (∃ (m : ℕ), lcm m n = 3 * m * gcd m n) ∧ (∀ k, k < 1000 ∧ (∃ (m' : ℕ), lcm m' k = 3 * m' * gcd m' k) → k ≤ 972) := sorry

end largest_n_l239_239925


namespace product_xyz_l239_239214

theorem product_xyz {x y z a b c : ℝ} 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = b^2) 
  (h3 : x^3 + y^3 + z^3 = c^3) : 
  x * y * z = (a^3 - 3 * a * b^2 + 2 * c^3) / 6 :=
by
  sorry

end product_xyz_l239_239214


namespace coordinates_of_P_l239_239280

-- Define the conditions and the question as a Lean theorem
theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) (h1 : P = (m + 3, m + 1)) (h2 : P.2 = 0) :
  P = (2, 0) := 
sorry

end coordinates_of_P_l239_239280


namespace initial_numbers_is_five_l239_239449

theorem initial_numbers_is_five : 
  ∀ (n S : ℕ), 
    (12 * n = S) →
    (10 * (n - 1) = S - 20) → 
    n = 5 := 
by sorry

end initial_numbers_is_five_l239_239449


namespace hydrogen_atoms_in_compound_l239_239940

theorem hydrogen_atoms_in_compound :
  ∀ (H_atoms Br_atoms O_atoms total_molecular_weight weight_H weight_Br weight_O : ℝ),
  Br_atoms = 1 ∧ O_atoms = 3 ∧ total_molecular_weight = 129 ∧ 
  weight_H = 1 ∧ weight_Br = 79.9 ∧ weight_O = 16 →
  H_atoms = 1 :=
by
  sorry

end hydrogen_atoms_in_compound_l239_239940


namespace Carly_injured_week_miles_l239_239813

def week1_miles : ℕ := 2
def week2_miles : ℕ := week1_miles * 2 + 3
def week3_miles : ℕ := week2_miles * 9 / 7
def week4_miles : ℕ := week3_miles - 5

theorem Carly_injured_week_miles : week4_miles = 4 :=
  by
    sorry

end Carly_injured_week_miles_l239_239813


namespace total_biscuits_l239_239627

-- Define the number of dogs and biscuits per dog
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem stating the total number of biscuits needed
theorem total_biscuits : num_dogs * biscuits_per_dog = 6 := by
  -- sorry to skip the proof
  sorry

end total_biscuits_l239_239627


namespace find_x_such_that_ceil_mul_x_eq_168_l239_239366

theorem find_x_such_that_ceil_mul_x_eq_168 (x : ℝ) (h_pos : x > 0)
  (h_eq : ⌈x⌉ * x = 168) (h_ceil: ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) :
  x = 168 / 13 :=
by
  sorry

end find_x_such_that_ceil_mul_x_eq_168_l239_239366


namespace dusting_days_l239_239534

theorem dusting_days 
    (vacuuming_minutes_per_day : ℕ) 
    (vacuuming_days_per_week : ℕ)
    (dusting_minutes_per_day : ℕ)
    (total_cleaning_minutes_per_week : ℕ)
    (x : ℕ) :
    vacuuming_minutes_per_day = 30 →
    vacuuming_days_per_week = 3 →
    dusting_minutes_per_day = 20 →
    total_cleaning_minutes_per_week = 130 →
    (vacuuming_minutes_per_day * vacuuming_days_per_week + dusting_minutes_per_day * x = total_cleaning_minutes_per_week) →
    x = 2 :=
by
  -- Proof steps go here
  sorry

end dusting_days_l239_239534


namespace solve_system_of_equations_l239_239604

theorem solve_system_of_equations :
  ∃ x y : ℝ, 
  (4 * x - 3 * y = -0.5) ∧ 
  (5 * x + 7 * y = 10.3) ∧ 
  (|x - 0.6372| < 1e-4) ∧ 
  (|y - 1.0163| < 1e-4) :=
by
  sorry

end solve_system_of_equations_l239_239604


namespace sufficient_condition_for_odd_l239_239782

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log (Real.sqrt (x^2 + a^2) - x)

theorem sufficient_condition_for_odd (a : ℝ) :
  (∀ x : ℝ, f 1 (-x) = -f 1 x) ∧
  (∀ x : ℝ, f (-1) (-x) = -f (-1) x) → 
  (a = 1 → ∀ x : ℝ, f a (-x) = -f a x) ∧ 
  (a ≠ 1 → ∃ x : ℝ, f a (-x) ≠ -f a x) :=
by
  sorry

end sufficient_condition_for_odd_l239_239782


namespace race_track_radius_l239_239148

theorem race_track_radius (C_inner : ℝ) (width : ℝ) (r_outer : ℝ) : 
  C_inner = 440 ∧ width = 14 ∧ r_outer = (440 / (2 * Real.pi) + 14) → r_outer = 84 :=
by
  intros
  sorry

end race_track_radius_l239_239148


namespace max_ben_cupcakes_l239_239166

theorem max_ben_cupcakes (total_cupcakes : ℕ) (ben_cupcakes charles_cupcakes diana_cupcakes : ℕ)
    (h1 : total_cupcakes = 30)
    (h2 : diana_cupcakes = 2 * ben_cupcakes)
    (h3 : charles_cupcakes = diana_cupcakes)
    (h4 : total_cupcakes = ben_cupcakes + charles_cupcakes + diana_cupcakes) :
    ben_cupcakes = 6 :=
by
  -- Proof steps would go here
  sorry

end max_ben_cupcakes_l239_239166


namespace range_of_a_l239_239248

noncomputable def quadratic_inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2 * (a - 2) * x + a > 0

theorem range_of_a :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → quadratic_inequality_condition a x) ↔ (1 < a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l239_239248


namespace solution_set_is_circle_with_exclusion_l239_239990

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l239_239990


namespace total_packs_of_groceries_l239_239436

-- Definitions for the conditions
def packs_of_cookies : ℕ := 2
def packs_of_cake : ℕ := 12

-- Theorem stating the total packs of groceries
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake = 14 :=
by sorry

end total_packs_of_groceries_l239_239436


namespace zoo_problem_l239_239051

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l239_239051


namespace sum_of_roots_of_poly_eq_14_over_3_l239_239355

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l239_239355


namespace find_a_l239_239839

theorem find_a (a : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) 
    (h_curve : ∀ x, y x = x^4 + a * x^2 + 1)
    (h_derivative : ∀ x, y' x = (4 * x^3 + 2 * a * x))
    (h_tangent_slope : y' (-1) = 8) :
    a = -6 :=
by
  -- To be proven
  sorry

end find_a_l239_239839


namespace maddy_credits_to_graduate_l239_239128

theorem maddy_credits_to_graduate (semesters : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ)
  (semesters_eq : semesters = 8)
  (credits_per_class_eq : credits_per_class = 3)
  (classes_per_semester_eq : classes_per_semester = 5) :
  semesters * (classes_per_semester * credits_per_class) = 120 :=
by
  -- Placeholder for proof
  sorry

end maddy_credits_to_graduate_l239_239128


namespace parabola_vertex_sum_l239_239062

theorem parabola_vertex_sum (p q r : ℝ) 
  (h1 : ∃ (a b c : ℝ), ∀ (x : ℝ), a * x ^ 2 + b * x + c = y)
  (h2 : ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = -1)
  (h3 : ∀ (x : ℝ), y = p * x ^ 2 + q * x + r)
  (h4 : y = p * (0 - 3) ^ 2 + r - 1)
  (h5 : y = 8)
  : p + q + r = 3 := 
by
  sorry

end parabola_vertex_sum_l239_239062


namespace abs_neg_2023_l239_239343

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239343


namespace percentage_of_rotten_oranges_l239_239647

theorem percentage_of_rotten_oranges
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (percentage_good_condition : ℕ)
  (rotted_percentage_bananas : ℕ)
  (total_fruits : ℕ)
  (good_condition_fruits : ℕ)
  (rotted_fruits : ℕ)
  (rotted_bananas : ℕ)
  (rotted_oranges : ℕ)
  (percentage_rotten_oranges : ℕ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : percentage_good_condition = 89)
  (h4 : rotted_percentage_bananas = 5)
  (h5 : total_fruits = total_oranges + total_bananas)
  (h6 : good_condition_fruits = percentage_good_condition * total_fruits / 100)
  (h7 : rotted_fruits = total_fruits - good_condition_fruits)
  (h8 : rotted_bananas = rotted_percentage_bananas * total_bananas / 100)
  (h9 : rotted_oranges = rotted_fruits - rotted_bananas)
  (h10 : percentage_rotten_oranges = rotted_oranges * 100 / total_oranges) : 
  percentage_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l239_239647


namespace square_side_length_l239_239901

theorem square_side_length (x S : ℕ) (h1 : S > 0) (h2 : x = 4) (h3 : 4 * S = 6 * x) : S = 6 := by
  subst h2
  sorry

end square_side_length_l239_239901


namespace greening_investment_growth_l239_239193

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end greening_investment_growth_l239_239193


namespace average_salary_is_8000_l239_239290

def average_salary_all_workers (A : ℝ) :=
  let total_workers := 30
  let technicians := 10
  let technician_salary := 12000
  let rest_workers := total_workers - technicians
  let rest_salary := 6000
  let total_salary := (technicians * technician_salary) + (rest_workers * rest_salary)
  A = total_salary / total_workers

theorem average_salary_is_8000 : average_salary_all_workers 8000 :=
by
  sorry

end average_salary_is_8000_l239_239290


namespace relationship_between_a_b_c_l239_239679

noncomputable def a : ℝ := Real.exp (-2)

noncomputable def b : ℝ := a ^ a

noncomputable def c : ℝ := a ^ b

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by {
  sorry
}

end relationship_between_a_b_c_l239_239679


namespace p_2015_coordinates_l239_239652

namespace AaronWalk

def position (n : ℕ) : ℤ × ℤ :=
sorry

theorem p_2015_coordinates : position 2015 = (22, 57) := 
sorry

end AaronWalk

end p_2015_coordinates_l239_239652


namespace speed_ratio_l239_239746

theorem speed_ratio :
  ∀ (v_A v_B : ℝ), (v_A / v_B = 3 / 2) ↔ (v_A = 3 * v_B / 2) :=
by
  intros
  sorry

end speed_ratio_l239_239746


namespace ellipse_eccentricity_l239_239698

-- Define the geometric sequence condition and the ellipse properties
theorem ellipse_eccentricity :
  ∀ (a b c e : ℝ), 
  (b^2 = a * c) ∧ (a^2 - c^2 = b^2) ∧ (e = c / a) ∧ (0 < e ∧ e < 1) →
  e = (Real.sqrt 5 - 1) / 2 := 
by 
  sorry

end ellipse_eccentricity_l239_239698


namespace initial_percentage_increase_l239_239292

variable (P : ℝ) (x : ℝ)

theorem initial_percentage_increase :
  (P * (1 + x / 100) * 1.3 = P * 1.625) → (x = 25) := by
  sorry

end initial_percentage_increase_l239_239292


namespace tax_rate_correct_l239_239323

def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 89.6

def taxable_amount : ℝ := total_value - non_taxable_amount

theorem tax_rate_correct : (tax_paid / taxable_amount) * 100 = 8 := by
  sorry

end tax_rate_correct_l239_239323


namespace books_sold_on_tuesday_l239_239121

theorem books_sold_on_tuesday (total_stock : ℕ) (monday_sold : ℕ) (wednesday_sold : ℕ)
  (thursday_sold : ℕ) (friday_sold : ℕ) (percent_unsold : ℚ) (tuesday_sold : ℕ) :
  total_stock = 1100 →
  monday_sold = 75 →
  wednesday_sold = 64 →
  thursday_sold = 78 →
  friday_sold = 135 →
  percent_unsold = 63.45 →
  tuesday_sold = total_stock - (monday_sold + wednesday_sold + thursday_sold + friday_sold + (total_stock * percent_unsold / 100)) :=
by sorry

end books_sold_on_tuesday_l239_239121


namespace part1_part2_l239_239765

-- Definition of the conditions given
def february_parcels : ℕ := 200000
def april_parcels : ℕ := 338000
def monthly_growth_rate : ℝ := 0.3

-- Problem 1: Proving the monthly growth rate is 0.3
theorem part1 (x : ℝ) (h : february_parcels * (1 + x)^2 = april_parcels) : x = monthly_growth_rate :=
  sorry

-- Problem 2: Proving the number of parcels in May is less than 450,000 with the given growth rate
theorem part2 (h : monthly_growth_rate = 0.3 ) : february_parcels * (1 + monthly_growth_rate)^3 < 450000 :=
  sorry

end part1_part2_l239_239765


namespace fraction_zero_implies_x_is_minus_5_l239_239267

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end fraction_zero_implies_x_is_minus_5_l239_239267


namespace trapezoid_perimeter_l239_239239

-- Define the problem conditions
variables (A B C D : Point) (BC AD : Line) (AB CD : Segment)

-- Conditions
def is_parallel (L1 L2 : Line) : Prop := sorry
def is_right_angle (A B C : Point) : Prop := sorry
def is_angle_150 (A B C : Point) : Prop := sorry

noncomputable def length (s : Segment) : ℝ := sorry

def trapezoid_conditions (A B C D : Point) (BC AD : Line) (AB CD : Segment) : Prop :=
  is_parallel BC AD ∧ is_angle_150 A B C ∧ is_right_angle C D B ∧
  length AB = 4 ∧ length BC = 3 - Real.sqrt 3

-- Perimeter calculation
noncomputable def perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) : ℝ :=
  length AB + length BC + length CD + length AD

-- Lean statement for the math proof problem
theorem trapezoid_perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) :
  trapezoid_conditions A B C D BC AD AB CD → perimeter A B C D BC AD AB CD = 12 :=
sorry

end trapezoid_perimeter_l239_239239


namespace relation_between_x_and_y_l239_239101

noncomputable def x : ℝ := 2 + Real.sqrt 3
noncomputable def y : ℝ := 1 / (2 - Real.sqrt 3)

theorem relation_between_x_and_y : x = y := sorry

end relation_between_x_and_y_l239_239101


namespace arman_two_weeks_earnings_l239_239427

theorem arman_two_weeks_earnings :
  let hourly_rate := 10
  let last_week_hours := 35
  let this_week_hours := 40
  let increase := 0.5
  let first_week_earnings := last_week_hours * hourly_rate
  let new_hourly_rate := hourly_rate + increase
  let second_week_earnings := this_week_hours * new_hourly_rate
  let total_earnings := first_week_earnings + second_week_earnings
  total_earnings = 770 := 
by
  -- Definitions based on conditions
  let hourly_rate := 10
  let last_week_hours := 35
  let this_week_hours := 40
  let increase := 0.5
  let first_week_earnings := last_week_hours * hourly_rate
  let new_hourly_rate := hourly_rate + increase
  let second_week_earnings := this_week_hours * new_hourly_rate
  let total_earnings := first_week_earnings + second_week_earnings
  sorry

end arman_two_weeks_earnings_l239_239427


namespace baba_yaga_departure_and_speed_l239_239039

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end baba_yaga_departure_and_speed_l239_239039


namespace total_homework_problems_l239_239650

-- Define the conditions as Lean facts
def finished_problems : ℕ := 45
def ratio_finished_to_left := (9, 4)
def problems_left (L : ℕ) := finished_problems * ratio_finished_to_left.2 = L * ratio_finished_to_left.1 

-- State the theorem
theorem total_homework_problems (L : ℕ) (h : problems_left L) : finished_problems + L = 65 :=
sorry

end total_homework_problems_l239_239650


namespace mike_total_games_l239_239875

-- Define the number of games Mike went to this year
def games_this_year : ℕ := 15

-- Define the number of games Mike went to last year
def games_last_year : ℕ := 39

-- Prove the total number of games Mike went to
theorem mike_total_games : games_this_year + games_last_year = 54 :=
by
  sorry

end mike_total_games_l239_239875


namespace abs_neg_2023_l239_239338

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239338


namespace tom_initial_money_l239_239299

-- Defining the given values
def super_nintendo_value : ℝ := 150
def store_percentage : ℝ := 0.80
def nes_price : ℝ := 160
def game_value : ℝ := 30
def change_received : ℝ := 10

-- Calculate the credit received for the Super Nintendo
def credit_received := store_percentage * super_nintendo_value

-- Calculate the remaining amount Tom needs to pay for the NES after using the credit
def remaining_amount := nes_price - credit_received

-- Calculate the total amount Tom needs to pay, including the game value
def total_amount_needed := remaining_amount + game_value

-- Proving that the initial money Tom gave is $80
theorem tom_initial_money : total_amount_needed + change_received = 80 :=
by
    sorry

end tom_initial_money_l239_239299


namespace percentage_error_in_side_measurement_l239_239802

theorem percentage_error_in_side_measurement :
  (forall (S S' : ℝ) (A A' : ℝ), 
    A = S^2 ∧ A' = S'^2 ∧ (A' - A) / A * 100 = 25.44 -> 
    (S' - S) / S * 100 = 12.72) :=
by
  intros S S' A A' h
  sorry

end percentage_error_in_side_measurement_l239_239802


namespace line_in_first_and_third_quadrants_l239_239711

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l239_239711


namespace value_of_expression_l239_239201

variables (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

theorem value_of_expression (h : f a b c d (-2) = -3) : 8 * a - 4 * b + 2 * c - d = 3 :=
by {
  sorry
}

end value_of_expression_l239_239201


namespace geometric_sequence_property_l239_239858

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ)
  (H_geo : ∀ n, a (n + 1) = a n * q)
  (H_cond1 : a 5 * a 7 = 2)
  (H_cond2 : a 2 + a 10 = 3) :
  (a 12 / a 4 = 2) ∨ (a 12 / a 4 = 1/2) :=
sorry

end geometric_sequence_property_l239_239858


namespace Bayes_theorem_2_white_balls_l239_239165

noncomputable def P (A_i : ℕ → bool) : ℝ := do 
  sorry 

noncomputable def BayesTheorem {A : Type*} (P : A → ℝ) (B : A → ℝ) : A → ℝ :=
  λ x, P x * B x / ∑ y in A, P y * B y

theorem Bayes_theorem_2_white_balls :
  let A := ℕ → bool,
      B := 2 * (∑ j in 0, 1, 2, P (λ j, true) * P (λ j : 0, 1, 2 : Φ, true)) :=
  BayesTheorem (λ A, 3 / 10 * 2 / 5) (∑ j in 0 to 2, P (λ A, true) (λ i, B)) 
    = 18 / 37 :=
  sorry

end Bayes_theorem_2_white_balls_l239_239165


namespace packs_per_box_l239_239533

theorem packs_per_box (total_cost : ℝ) (num_boxes : ℕ) (cost_per_pack : ℝ) 
  (num_tissues_per_pack : ℕ) (cost_per_tissue : ℝ) (total_packs : ℕ) :
  total_cost = 1000 ∧ num_boxes = 10 ∧ cost_per_pack = num_tissues_per_pack * cost_per_tissue ∧ 
  num_tissues_per_pack = 100 ∧ cost_per_tissue = 0.05 ∧ total_packs * cost_per_pack = total_cost / num_boxes →
  total_packs = 20 :=
by
  sorry

end packs_per_box_l239_239533


namespace find_x_l239_239847

def op (a b : ℕ) : ℕ := a * b - b + b ^ 2

theorem find_x (x : ℕ) : (∃ x : ℕ, op x 8 = 80) :=
  sorry

end find_x_l239_239847


namespace box_dimensions_sum_l239_239036

theorem box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30) 
  (h2 : A * C = 50)
  (h3 : B * C = 90) : 
  A + B + C = (58 * Real.sqrt 15) / 3 :=
sorry

end box_dimensions_sum_l239_239036


namespace problem1_problem2_l239_239113

section ArithmeticSequence

variable {a : ℕ → ℤ} {a1 a5 a8 a6 a4 d : ℤ}

-- Problem 1: Prove that if a_5 = -1 and a_8 = 2, then a_1 = -5 and d = 1
theorem problem1 
  (h1 : a 5 = -1) 
  (h2 : a 8 = 2)
  (h3 : ∀ n, a n = a1 + n * d) : 
  a1 = -5 ∧ d = 1 := 
sorry 

-- Problem 2: Prove that if a_1 + a_6 = 12 and a_4 = 7, then a_9 = 17
theorem problem2 
  (h1 : a1 + a 6 = 12) 
  (h2 : a 4 = 7)
  (h3 : ∀ n, a n = a1 + n * d) 
  (h4 : ∀ m (hm : m ≠ 0), a1 = a 1): 
   a 9 = 17 := 
sorry

end ArithmeticSequence

end problem1_problem2_l239_239113


namespace unique_alphabets_count_l239_239805

theorem unique_alphabets_count
  (total_alphabets : ℕ)
  (each_written_times : ℕ)
  (total_written : total_alphabets * each_written_times = 10) :
  total_alphabets = 5 := by
  -- The proof would be filled in here.
  sorry

end unique_alphabets_count_l239_239805


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l239_239721

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l239_239721


namespace eleven_y_minus_x_eq_one_l239_239175

theorem eleven_y_minus_x_eq_one 
  (x y : ℤ) 
  (hx_pos : x > 0)
  (h1 : x = 7 * y + 3)
  (h2 : 2 * x = 6 * (3 * y) + 2) : 
  11 * y - x = 1 := 
by 
  sorry

end eleven_y_minus_x_eq_one_l239_239175


namespace smallest_y_l239_239507

theorem smallest_y (y : ℕ) : (27^y > 3^24) ↔ (y ≥ 9) :=
sorry

end smallest_y_l239_239507


namespace probability_divisible_by_five_l239_239466

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l239_239466


namespace max_value_of_3x_plus_4y_l239_239680

theorem max_value_of_3x_plus_4y (x y : ℝ) 
(h : x^2 + y^2 = 14 * x + 6 * y + 6) : 
3 * x + 4 * y ≤ 73 := 
sorry

end max_value_of_3x_plus_4y_l239_239680


namespace karen_locks_problem_l239_239741

theorem karen_locks_problem :
  let T1 := 5 in
  let T2 := 3 * T1 - 3 in
  let Combined_Locks_Time := 5 * T2 in
  Combined_Locks_Time = 60 := by
    let T1 := 5
    let T2 := 3 * T1 - 3
    let Combined_Locks_Time := 5 * T2
    sorry

end karen_locks_problem_l239_239741


namespace root_conditions_imply_sum_l239_239104

-- Define the variables a and b in the context that their values fit the given conditions.
def a : ℝ := 5
def b : ℝ := -6

-- Define the quadratic equation and conditions on roots.
def quadratic_eq (x : ℝ) := x^2 - a * x - b

-- Given that 2 and 3 are the roots of the quadratic equation.
def roots_condition := (quadratic_eq 2 = 0) ∧ (quadratic_eq 3 = 0)

-- The theorem to prove.
theorem root_conditions_imply_sum :
  roots_condition → a + b = -1 :=
by
sorry

end root_conditions_imply_sum_l239_239104


namespace tanker_filling_rate_l239_239796

theorem tanker_filling_rate :
  let barrels_per_minute := 5
  let liters_per_barrel := 159
  let minutes_per_hour := 60
  let liters_per_cubic_meter := 1000
  (barrels_per_minute * liters_per_barrel * minutes_per_hour) / 
  liters_per_cubic_meter = 47.7 :=
by
  sorry

end tanker_filling_rate_l239_239796


namespace ratio_of_coconut_flavored_red_jelly_beans_l239_239918

theorem ratio_of_coconut_flavored_red_jelly_beans :
  ∀ (total_jelly_beans jelly_beans_coconut_flavored : ℕ)
    (three_fourths_red : total_jelly_beans > 0 ∧ (3/4 : ℝ) * total_jelly_beans = 3 * (total_jelly_beans / 4))
    (h1 : jelly_beans_coconut_flavored = 750)
    (h2 : total_jelly_beans = 4000),
  (250 : ℝ)/(3000 : ℝ) = 1/4 :=
by
  intros total_jelly_beans jelly_beans_coconut_flavored three_fourths_red h1 h2
  sorry

end ratio_of_coconut_flavored_red_jelly_beans_l239_239918


namespace pirate_coins_total_l239_239752

def total_coins (y : ℕ) := 6 * y

theorem pirate_coins_total : 
  (∃ y : ℕ, y ≠ 0 ∧ y * (y + 1) / 2 = 5 * y) →
  total_coins 9 = 54 :=
by
  sorry

end pirate_coins_total_l239_239752


namespace vitya_catchup_time_l239_239490

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l239_239490


namespace candy_comparison_l239_239043

variable (skittles_bryan : ℕ)
variable (gummy_bears_bryan : ℕ)
variable (chocolate_bars_bryan : ℕ)
variable (mms_ben : ℕ)
variable (jelly_beans_ben : ℕ)
variable (lollipops_ben : ℕ)

def bryan_total_candies := skittles_bryan + gummy_bears_bryan + chocolate_bars_bryan
def ben_total_candies := mms_ben + jelly_beans_ben + lollipops_ben

def difference_skittles_mms := skittles_bryan - mms_ben
def difference_gummy_jelly := jelly_beans_ben - gummy_bears_bryan
def difference_choco_lollipops := chocolate_bars_bryan - lollipops_ben

def sum_of_differences := difference_skittles_mms + difference_gummy_jelly + difference_choco_lollipops

theorem candy_comparison
  (h_bryan_skittles : skittles_bryan = 50)
  (h_bryan_gummy_bears : gummy_bears_bryan = 25)
  (h_bryan_choco_bars : chocolate_bars_bryan = 15)
  (h_ben_mms : mms_ben = 20)
  (h_ben_jelly_beans : jelly_beans_ben = 30)
  (h_ben_lollipops : lollipops_ben = 10) :
  bryan_total_candies = 90 ∧
  ben_total_candies = 60 ∧
  bryan_total_candies > ben_total_candies ∧
  difference_skittles_mms = 30 ∧
  difference_gummy_jelly = 5 ∧
  difference_choco_lollipops = 5 ∧
  sum_of_differences = 40 := by
  sorry

end candy_comparison_l239_239043


namespace abs_neg_2023_l239_239339

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239339


namespace gcd_228_1995_base3_to_base6_conversion_l239_239631

-- Proof Problem 1: GCD of 228 and 1995 is 57
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

-- Proof Problem 2: Converting base-3 number 11102 to base-6
theorem base3_to_base6_conversion : Nat.ofDigits 6 [3, 1, 5] = Nat.ofDigits 10 [1, 1, 1, 0, 2] :=
by
  sorry

end gcd_228_1995_base3_to_base6_conversion_l239_239631


namespace probability_only_one_l239_239269

-- Define the probabilities
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complement probabilities
def not_P (P : ℚ) : ℚ := 1 - P
def P_not_A := not_P P_A
def P_not_B := not_P P_B
def P_not_C := not_P P_C

-- Expressions for probabilities where only one student solves the problem
def only_A_solves : ℚ := P_A * P_not_B * P_not_C
def only_B_solves : ℚ := P_B * P_not_A * P_not_C
def only_C_solves : ℚ := P_C * P_not_A * P_not_B

-- Total probability that only one student solves the problem
def P_only_one : ℚ := only_A_solves + only_B_solves + only_C_solves

-- The theorem to prove that the total probability matches
theorem probability_only_one : P_only_one = 11 / 24 := by
  sorry

end probability_only_one_l239_239269


namespace retailer_selling_price_l239_239945

theorem retailer_selling_price
  (cost_price_manufacturer : ℝ)
  (manufacturer_profit_rate : ℝ)
  (wholesaler_profit_rate : ℝ)
  (retailer_profit_rate : ℝ)
  (manufacturer_selling_price : ℝ)
  (wholesaler_selling_price : ℝ)
  (retailer_selling_price : ℝ)
  (h1 : cost_price_manufacturer = 17)
  (h2 : manufacturer_profit_rate = 0.18)
  (h3 : wholesaler_profit_rate = 0.20)
  (h4 : retailer_profit_rate = 0.25)
  (h5 : manufacturer_selling_price = cost_price_manufacturer + (manufacturer_profit_rate * cost_price_manufacturer))
  (h6 : wholesaler_selling_price = manufacturer_selling_price + (wholesaler_profit_rate * manufacturer_selling_price))
  (h7 : retailer_selling_price = wholesaler_selling_price + (retailer_profit_rate * wholesaler_selling_price)) :
  retailer_selling_price = 30.09 :=
by {
  sorry
}

end retailer_selling_price_l239_239945


namespace total_books_written_l239_239515

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end total_books_written_l239_239515


namespace total_books_written_l239_239514

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end total_books_written_l239_239514


namespace avg_remaining_two_is_correct_avg_new_set_is_correct_l239_239144

-- Definitions derived from the conditions
def avg_all_numbers := 4.60
def total_numbers := 10
def sum_all_numbers := avg_all_numbers * total_numbers

def avg_first_three := 3.4
def num_first_three := 3
def sum_first_three := avg_first_three * num_first_three

def avg_next_two := 3.8
def num_next_two := 2
def sum_next_two := avg_next_two * num_next_two

def avg_another_three := 4.2
def num_another_three := 3
def sum_another_three := avg_another_three * num_another_three

def sum_known_eight := sum_first_three + sum_next_two + sum_another_three
def sum_remaining_two := sum_all_numbers - sum_known_eight
def avg_remaining_two := sum_remaining_two / 2

def new_set_size := 3
def sum_new_set := avg_all_numbers * new_set_size
def avg_new_set := sum_new_set / new_set_size

-- Lean statements for the proof problems
theorem avg_remaining_two_is_correct : avg_remaining_two = 7.8 := by
  sorry

theorem avg_new_set_is_correct : avg_new_set = 4.60 := by
  sorry

end avg_remaining_two_is_correct_avg_new_set_is_correct_l239_239144


namespace total_weight_on_scale_l239_239327

-- Define the weights of Alexa and Katerina
def alexa_weight : ℕ := 46
def katerina_weight : ℕ := 49

-- State the theorem to prove the total weight on the scale
theorem total_weight_on_scale : alexa_weight + katerina_weight = 95 := by
  sorry

end total_weight_on_scale_l239_239327


namespace total_balls_in_box_l239_239524

theorem total_balls_in_box :
  ∀ (W B R : ℕ), 
    W = 16 →
    B = W + 12 →
    R = 2 * B →
    W + B + R = 100 :=
by
  intros W B R hW hB hR
  sorry

end total_balls_in_box_l239_239524


namespace total_height_correct_l239_239952

-- Stack and dimensions setup
def height_of_disc_stack (top_diameter bottom_diameter disc_thickness : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  num_discs * disc_thickness

def total_height (top_diameter bottom_diameter disc_thickness cylinder_height : ℕ) : ℕ :=
  height_of_disc_stack top_diameter bottom_diameter disc_thickness + cylinder_height

-- Given conditions
def top_diameter := 15
def bottom_diameter := 1
def disc_thickness := 2
def cylinder_height := 10
def correct_answer := 26

-- Proof problem
theorem total_height_correct :
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = correct_answer :=
by
  sorry

end total_height_correct_l239_239952


namespace meaningful_range_l239_239906

theorem meaningful_range (x : ℝ) : (x < 4) ↔ (4 - x > 0) := 
by sorry

end meaningful_range_l239_239906


namespace odds_against_C_win_l239_239725

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C_win (pA pB : ℚ) (hA : pA = 1/5) (hB : pB = 2/3) :
  odds_against_winning (1 - pA - pB) = 13 / 2 :=
by
  sorry

end odds_against_C_win_l239_239725


namespace find_x_rational_l239_239226

theorem find_x_rational (x : ℝ) (h1 : ∃ (a : ℚ), x + Real.sqrt 3 = a)
  (h2 : ∃ (b : ℚ), x^2 + Real.sqrt 3 = b) :
  x = (1 / 2 : ℝ) - Real.sqrt 3 :=
sorry

end find_x_rational_l239_239226


namespace krish_remaining_money_l239_239435

variable (initial_amount sweets stickers friends each_friend charity : ℝ)

theorem krish_remaining_money :
  initial_amount = 200.50 →
  sweets = 35.25 →
  stickers = 10.75 →
  friends = 4 →
  each_friend = 25.20 →
  charity = 15.30 →
  initial_amount - (sweets + stickers + friends * each_friend + charity) = 38.40 :=
by
  intros h_initial h_sweets h_stickers h_friends h_each_friend h_charity
  sorry

end krish_remaining_money_l239_239435


namespace total_drivers_l239_239613

theorem total_drivers (N : ℕ) (A : ℕ) (sA sB sC sD : ℕ) (total_sampled : ℕ)
  (hA : A = 96) (hsA : sA = 12) (hsB : sB = 21) (hsC : sC = 25) (hsD : sD = 43) (htotal : total_sampled = sA + sB + sC + sD)
  (hsA_proportion : (sA : ℚ) / A = (total_sampled : ℚ) / N) : N = 808 := by
  sorry

end total_drivers_l239_239613


namespace friend_selling_price_correct_l239_239316

-- Definition of the original cost price
def original_cost_price : ℕ := 50000

-- Definition of the loss percentage
def loss_percentage : ℕ := 10

-- Definition of the gain percentage
def gain_percentage : ℕ := 20

-- Definition of the man's selling price after loss
def man_selling_price : ℕ := original_cost_price - (original_cost_price * loss_percentage / 100)

-- Definition of the friend's selling price after gain
def friend_selling_price : ℕ := man_selling_price + (man_selling_price * gain_percentage / 100)

theorem friend_selling_price_correct : friend_selling_price = 54000 := by
  sorry

end friend_selling_price_correct_l239_239316


namespace integral_part_odd_l239_239880

theorem integral_part_odd (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, (⌊(3 + Real.sqrt 5)^n⌋ = 2 * m + 1) := 
by
  -- Sorry used since the proof steps are not required in the task
  sorry

end integral_part_odd_l239_239880


namespace gcd_factorial_8_10_l239_239554

-- Define the concept of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n 

-- Statement of the problem 
theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_factorial_8_10_l239_239554


namespace symmetric_point_coordinates_l239_239672

theorem symmetric_point_coordinates (Q : ℝ × ℝ × ℝ) 
  (P : ℝ × ℝ × ℝ := (-6, 7, -9)) 
  (A : ℝ × ℝ × ℝ := (1, 3, -1)) 
  (B : ℝ × ℝ × ℝ := (6, 5, -2)) 
  (C : ℝ × ℝ × ℝ := (0, -3, -5)) : Q = (2, -5, 7) :=
sorry

end symmetric_point_coordinates_l239_239672


namespace n_four_minus_n_squared_l239_239221

theorem n_four_minus_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
by 
  sorry

end n_four_minus_n_squared_l239_239221


namespace expression_varies_l239_239899

noncomputable def expr (x : ℝ) : ℝ := (3 * x^2 - 2 * x - 5) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3))

theorem expression_varies (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) : ∃ y : ℝ, expr x = y ∧ ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → expr x₁ ≠ expr x₂ :=
by
  sorry

end expression_varies_l239_239899


namespace ratio_correct_l239_239440

def my_age : ℕ := 35
def son_age_next_year : ℕ := 8
def son_age_now : ℕ := son_age_next_year - 1
def ratio_of_ages : ℕ := my_age / son_age_now

theorem ratio_correct : ratio_of_ages = 5 :=
by
  -- Add proof here
  sorry

end ratio_correct_l239_239440


namespace oil_depth_solution_l239_239639

theorem oil_depth_solution
  (length diameter surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 12)
  (h_diameter : diameter = 4)
  (h_surface_area : surface_area = 24)
  (r : ℝ := diameter / 2)
  (c : ℝ := surface_area / length) :
  (h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3) :=
by
  sorry

end oil_depth_solution_l239_239639


namespace ratio_sheep_horses_l239_239536

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end ratio_sheep_horses_l239_239536


namespace arithmetic_sequence_20th_term_l239_239055

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 5
  let n := 20
  let a_n := a + (n - 1) * d
  a_n = 97 := by
  sorry

end arithmetic_sequence_20th_term_l239_239055


namespace solution_set_of_inequality_l239_239908

theorem solution_set_of_inequality :
  ∀ x : ℝ, |2 * x^2 - 1| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end solution_set_of_inequality_l239_239908


namespace congruence_solution_l239_239098

theorem congruence_solution (x : ℤ) (h : 5 * x + 11 ≡ 3 [ZMOD 19]) : 3 * x + 7 ≡ 6 [ZMOD 19] :=
sorry

end congruence_solution_l239_239098


namespace children_count_l239_239188

theorem children_count (W C n : ℝ) (h1 : 4 * W = 1 / 7) (h2 : n * C = 1 / 14) (h3 : 5 * W + 10 * C = 1 / 4) : n = 10 :=
by
  sorry

end children_count_l239_239188


namespace total_tax_in_cents_l239_239528

-- Declare the main variables and constants
def wage_per_hour_cents : ℕ := 2500
def local_tax_rate : ℝ := 0.02
def state_tax_rate : ℝ := 0.005

-- Define the total tax calculation as a proof statement
theorem total_tax_in_cents :
  local_tax_rate * wage_per_hour_cents + state_tax_rate * wage_per_hour_cents = 62.5 :=
by sorry

end total_tax_in_cents_l239_239528


namespace min_alterations_to_make_sums_unique_l239_239056

def initial_matrix : matrix (fin 3) (fin 3) ℕ :=
  ![![4, 9, 2], ![9, 1, 6], ![4, 5, 7]]

def row_sums (m : matrix (fin 3) (fin 3) ℕ) : fin 3 → ℕ :=
  λ i, (finset.univ.sum (λ j, m i j))

def col_sums (m : matrix (fin 3) (fin 3) ℕ) : fin 3 → ℕ :=
  λ j, (finset.univ.sum (λ i, m i j))

theorem min_alterations_to_make_sums_unique :
  ∃ m' : matrix (fin 3) (fin 3) ℕ,
    (∃ i j₁ j₂, m' = initial_matrix.update i j₁ 10 .update i j₂ 4) ∧
    (∀ i₁ i₂, i₁ ≠ i₂ → row_sums m' i₁ ≠ row_sums m' i₂) ∧
    (∀ j₁ j₂, j₁ ≠ j₂ → col_sums m' j₁ ≠ col_sums m' j₂) :=
begin
  sorry, -- proof not required
end

end min_alterations_to_make_sums_unique_l239_239056


namespace full_capacity_l239_239180

def oil_cylinder_capacity (C : ℝ) :=
  (4 / 5) * C - (3 / 4) * C = 4

theorem full_capacity : oil_cylinder_capacity 80 :=
by
  simp [oil_cylinder_capacity]
  sorry

end full_capacity_l239_239180


namespace voter_ratio_l239_239726

theorem voter_ratio (Vx Vy : ℝ) (hx : 0.72 * Vx + 0.36 * Vy = 0.60 * (Vx + Vy)) : Vx = 2 * Vy :=
by
sorry

end voter_ratio_l239_239726


namespace all_statements_imply_negation_l239_239219

theorem all_statements_imply_negation :
  let s1 := (true ∧ true ∧ false)
  let s2 := (false ∧ true ∧ true)
  let s3 := (true ∧ false ∧ true)
  let s4 := (false ∧ false ∧ true)
  (s1 → ¬(true ∧ true ∧ true)) ∧
  (s2 → ¬(true ∧ true ∧ true)) ∧
  (s3 → ¬(true ∧ true ∧ true)) ∧
  (s4 → ¬(true ∧ true ∧ true)) :=
by sorry

end all_statements_imply_negation_l239_239219


namespace simplify_polynomials_l239_239286

theorem simplify_polynomials (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 :=
by 
  sorry

end simplify_polynomials_l239_239286


namespace triangle_problem_l239_239859

theorem triangle_problem (n : ℕ) (h : 1 < n ∧ n < 4) : n = 2 ∨ n = 3 :=
by
  -- Valid realizability proof omitted
  sorry

end triangle_problem_l239_239859


namespace take_home_pay_l239_239417

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l239_239417


namespace length_of_square_side_is_correct_l239_239882

noncomputable def length_of_square_side : ℚ :=
  let PQ : ℚ := 7
  let QR : ℚ := 24
  let hypotenuse := (PQ^2 + QR^2).sqrt
  (25 * 175) / (24 * 32)

theorem length_of_square_side_is_correct :
  length_of_square_side = 4375 / 768 := 
by 
  sorry

end length_of_square_side_is_correct_l239_239882


namespace student_ticket_cost_l239_239757

def general_admission_ticket_cost : ℕ := 6
def total_tickets_sold : ℕ := 525
def total_revenue : ℕ := 2876
def general_admission_tickets_sold : ℕ := 388

def number_of_student_tickets_sold : ℕ := total_tickets_sold - general_admission_tickets_sold
def revenue_from_general_admission : ℕ := general_admission_tickets_sold * general_admission_ticket_cost

theorem student_ticket_cost : ∃ S : ℕ, number_of_student_tickets_sold * S + revenue_from_general_admission = total_revenue ∧ S = 4 :=
by
  sorry

end student_ticket_cost_l239_239757


namespace pasta_needed_for_family_reunion_l239_239660

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end pasta_needed_for_family_reunion_l239_239660


namespace clock_hands_straight_twenty_four_hours_l239_239853

noncomputable def hands_straight_per_day : ℕ :=
  2 * 22

theorem clock_hands_straight_twenty_four_hours :
  hands_straight_per_day = 44 :=
by
  sorry

end clock_hands_straight_twenty_four_hours_l239_239853


namespace two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l239_239471

-- Define the number of boys and girls
def boys : ℕ := 2
def girls : ℕ := 3
def total_people : ℕ := boys + girls

-- Define assumptions about arrangements
def arrangements_in_two_rows : ℕ := sorry
def arrangements_with_person_A_not_head_tail : ℕ := sorry
def arrangements_with_girls_together : ℕ := sorry
def arrangements_with_boys_not_adjacent : ℕ := sorry

-- State the mathematical equivalence proof problems
theorem two_rows_arrangement : arrangements_in_two_rows = 60 := 
  sorry

theorem person_A_not_head_tail_arrangement : arrangements_with_person_A_not_head_tail = 72 := 
  sorry

theorem girls_together_arrangement : arrangements_with_girls_together = 36 := 
  sorry

theorem boys_not_adjacent_arrangement : arrangements_with_boys_not_adjacent = 72 := 
  sorry

end two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l239_239471


namespace basil_pots_count_l239_239333

theorem basil_pots_count (B : ℕ) (h1 : 9 * 18 + 6 * 30 + 4 * B = 354) : B = 3 := 
by 
  -- This is just the signature of the theorem. The proof is omitted.
  sorry

end basil_pots_count_l239_239333


namespace take_home_pay_is_correct_l239_239414

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l239_239414


namespace ratio_of_weights_l239_239157

variable (x y : ℝ)

theorem ratio_of_weights (h : x + y = 7 * (x - y)) (h1 : x > y) : x / y = 4 / 3 :=
sorry

end ratio_of_weights_l239_239157


namespace current_time_l239_239860

theorem current_time (t : ℝ) 
  (h1 : 6 * (t + 10) - (90 + 0.5 * (t - 5)) = 90 ∨ 6 * (t + 10) - (90 + 0.5 * (t - 5)) = -90) :
  t = 3 + 11 / 60 := sorry

end current_time_l239_239860


namespace quadratic_inequality_solution_set_l239_239461

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 5*x - 14 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 7} :=
by
  -- proof to be filled here
  sorry

end quadratic_inequality_solution_set_l239_239461


namespace abs_neg_2023_l239_239341

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239341


namespace playground_area_l239_239450

theorem playground_area (L B : ℕ) (h1 : B = 6 * L) (h2 : B = 420)
  (A_total A_playground : ℕ) (h3 : A_total = L * B) 
  (h4 : A_playground = A_total / 7) :
  A_playground = 4200 :=
by sorry

end playground_area_l239_239450


namespace probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l239_239777

-- Definitions for balls and initial conditions
def totalBalls : ℕ := 10
def redBalls : ℕ := 2
def whiteBalls : ℕ := 3
def yellowBalls : ℕ := 5

-- Drawing without replacement
noncomputable def probability_second_ball_red : ℚ :=
  (2/10) * (1/9) + (8/10) * (2/9)

-- Probabilities for each case
noncomputable def probability_first_prize : ℚ := 
  (redBalls.choose 1 * whiteBalls.choose 1) / (totalBalls.choose 2)

noncomputable def probability_second_prize : ℚ := 
  (redBalls.choose 2) / (totalBalls.choose 2)

noncomputable def probability_third_prize : ℚ := 
  (whiteBalls.choose 2) / (totalBalls.choose 2)

-- Probability of at least one yellow ball (no prize)
noncomputable def probability_no_prize : ℚ := 
  1 - probability_first_prize - probability_second_prize - probability_third_prize

-- Probability distribution and expectation for number of winners X
noncomputable def winning_probability : ℚ := probability_first_prize + probability_second_prize + probability_third_prize

noncomputable def P_X (n : ℕ) : ℚ :=
  if n = 0 then (7/9)^3
  else if n = 1 then 3 * (2/9) * (7/9)^2
  else if n = 2 then 3 * (2/9)^2 * (7/9)
  else if n = 3 then (2/9)^3
  else 0

noncomputable def expectation_X : ℚ := 
  3 * winning_probability

-- Lean statements
theorem probability_of_second_ball_red_is_correct :
  probability_second_ball_red = 1 / 5 := by
  sorry

theorem probabilities_of_winning_prizes :
  probability_first_prize = 2 / 15 ∧
  probability_second_prize = 1 / 45 ∧
  probability_third_prize = 1 / 15 := by
  sorry

theorem distribution_and_expectation_of_X :
  P_X 0 = 343 / 729 ∧
  P_X 1 = 294 / 729 ∧
  P_X 2 = 84 / 729 ∧
  P_X 3 = 8 / 729 ∧
  expectation_X = 2 / 3 := by
  sorry

end probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l239_239777


namespace probability_divisible_by_five_l239_239465

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l239_239465


namespace correct_inequality_l239_239247

theorem correct_inequality (x : ℝ) : (1 / (x^2 + 1)) > (1 / (x^2 + 2)) :=
by {
  -- Lean proof steps would be here, but we will use 'sorry' instead to indicate the proof is omitted.
  sorry
}

end correct_inequality_l239_239247


namespace alex_shirts_count_l239_239953

theorem alex_shirts_count (j a b : ℕ) (h1 : j = a + 3) (h2 : b = j + 8) (h3 : b = 15) : a = 4 :=
by
  sorry

end alex_shirts_count_l239_239953


namespace exists_non_monochromatic_coloring_l239_239563

-- Defining the set of points and properties
def P := Finset (Fin 1994)
axiom noncollinear : ∀ₓ P₁ P₂ P₃ ∈ P, P₁ ≠ P₂ → P₂ ≠ P₃ → P₁ ≠ P₃ → ¬ collinear P₁ P₂ P₃

-- Partition into 83 groups
def partition : Finset (Finset (Fin 1994)) := sorry
axiom partition_props : partition.card = 83 ∧ ∀ₓ g ∈ partition, 3 ≤ g.card

-- Defining the graph G*
def G_star : SimpleGraph (Fin 1994) := sorry
axiom G_star_min_tris : (G_star.num_triangles P) = (min_tris G_star)

-- Coloring axiom
axiom coloring : ∃ (coloring : G_star.Edge → Fin 4), ∀ₓ (a b c : Fin 1994), a ≠ b → b ≠ c → a ≠ c → ¬ monochromatic_triangle coloring a b c

-- Main theorem statement
theorem exists_non_monochromatic_coloring : ∃ coloring, ∀ₓ (a b c ∈ P), G_star.Adj a b → G_star.Adj b c → G_star.Adj a c → coloring (G_star.EdgeSet a b) ≠ coloring (G_star.EdgeSet b c) ∨ coloring (G_star.EdgeSet b c) ≠ coloring (G_star.EdgeSet a c) ∨ coloring (G_star.EdgeSet a c) ≠ coloring (G_star.EdgeSet a b) :=
sorry

end exists_non_monochromatic_coloring_l239_239563


namespace vitya_catch_up_time_l239_239502

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l239_239502


namespace circles_intersection_distance_squared_l239_239361

open Real

-- Definitions of circles
def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 25

def circle2 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 6)^2 = 9

-- Theorem to prove
theorem circles_intersection_distance_squared :
  ∃ A B : (ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B) ∧
  (dist A B)^2 = 675 / 49 :=
sorry

end circles_intersection_distance_squared_l239_239361


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l239_239720

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l239_239720


namespace scientific_notation_l239_239611

theorem scientific_notation (n : ℕ) (h : n = 27000000) : 
  ∃ (m : ℝ) (e : ℤ), n = m * (10 : ℝ) ^ e ∧ m = 2.7 ∧ e = 7 :=
by 
  use 2.7 
  use 7
  sorry

end scientific_notation_l239_239611


namespace largest_coefficient_term_l239_239911

theorem largest_coefficient_term:
  (∃ r: ℕ, r = 4 ∧ ∀ k: ℕ, 0 ≤ k ∧ k ≤ 7 → nat.choose 7 k ≤ nat.choose 7 4) :=
sorry

end largest_coefficient_term_l239_239911


namespace socks_total_is_51_l239_239275

-- Define initial conditions for John and Mary
def john_initial_socks : Nat := 33
def john_thrown_away_socks : Nat := 19
def john_new_socks : Nat := 13

def mary_initial_socks : Nat := 20
def mary_thrown_away_socks : Nat := 6
def mary_new_socks : Nat := 10

-- Define the total socks function
def total_socks (john_initial john_thrown john_new mary_initial mary_thrown mary_new : Nat) : Nat :=
  (john_initial - john_thrown + john_new) + (mary_initial - mary_thrown + mary_new)

-- Statement to prove
theorem socks_total_is_51 : 
  total_socks john_initial_socks john_thrown_away_socks john_new_socks 
              mary_initial_socks mary_thrown_away_socks mary_new_socks = 51 := 
by
  sorry

end socks_total_is_51_l239_239275


namespace susan_age_in_5_years_l239_239023

-- Definitions of the given conditions
def james_age_in_15_years : ℕ := 37
def years_until_james_is_37 : ℕ := 15
def years_ago_james_twice_janet : ℕ := 8
def susan_born_when_janet_turned : ℕ := 3
def years_to_future_susan_age : ℕ := 5

-- Calculate the current age of people involved
def james_current_age : ℕ := james_age_in_15_years - years_until_james_is_37
def james_age_8_years_ago : ℕ := james_current_age - years_ago_james_twice_janet
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def janet_current_age : ℕ := janet_age_8_years_ago + years_ago_james_twice_janet
def susan_current_age : ℕ := janet_current_age - susan_born_when_janet_turned

-- Prove that Susan will be 17 years old in 5 years
theorem susan_age_in_5_years (james_age_future : james_age_in_15_years = 37)
  (years_until_james_37 : years_until_james_is_37 = 15)
  (years_ago_twice_janet : years_ago_james_twice_janet = 8)
  (susan_born_janet : susan_born_when_janet_turned = 3)
  (years_future : years_to_future_susan_age = 5) :
  susan_current_age + years_to_future_susan_age = 17 := by
  -- The proof is omitted
  sorry

end susan_age_in_5_years_l239_239023


namespace abs_neg_2023_l239_239342

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239342


namespace obtuse_triangle_existence_l239_239209

theorem obtuse_triangle_existence :
  ∃ (a b c : ℝ), (a = 2 ∧ b = 6 ∧ c = 7 ∧ 
  (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)) ∧
  ¬(6^2 + 7^2 < 8^2 ∨ 7^2 + 8^2 < 6^2 ∨ 8^2 + 6^2 < 7^2) ∧
  ¬(7^2 + 8^2 < 10^2 ∨ 8^2 + 10^2 < 7^2 ∨ 10^2 + 7^2 < 8^2) ∧
  ¬(5^2 + 12^2 < 13^2 ∨ 12^2 + 13^2 < 5^2 ∨ 13^2 + 5^2 < 12^2) :=
sorry

end obtuse_triangle_existence_l239_239209


namespace unique_triplet_satisfying_conditions_l239_239281

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
                 (c ∣ a * b + 1) ∧
                 (b ∣ c * a + 1) ∧
                 (a ∣ b * c + 1) ∧
                 a = 2 ∧ b = 3 ∧ c = 7 :=
by
  sorry

end unique_triplet_satisfying_conditions_l239_239281


namespace AM_GM_Ineq_l239_239879

theorem AM_GM_Ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_Ineq_l239_239879


namespace total_dogs_is_28_l239_239058

def number_of_boxes : ℕ := 7
def dogs_per_box : ℕ := 4
def total_dogs (boxes : ℕ) (dogs_in_each : ℕ) : ℕ := boxes * dogs_in_each

theorem total_dogs_is_28 : total_dogs number_of_boxes dogs_per_box = 28 :=
by
  sorry

end total_dogs_is_28_l239_239058


namespace Aiyanna_has_more_cookies_l239_239801

theorem Aiyanna_has_more_cookies (cookies_Alyssa : ℕ) (cookies_Aiyanna : ℕ) (h1 : cookies_Alyssa = 129) (h2 : cookies_Aiyanna = cookies_Alyssa + 11) : cookies_Aiyanna = 140 := by
  sorry

end Aiyanna_has_more_cookies_l239_239801


namespace watch_cost_price_l239_239933

theorem watch_cost_price (CP : ℝ) (h1 : (0.90 * CP) + 280 = 1.04 * CP) : CP = 2000 := 
by 
  sorry

end watch_cost_price_l239_239933


namespace find_exponent_l239_239262

theorem find_exponent (y : ℕ) (h : (1/8) * (2: ℝ)^36 = (2: ℝ)^y) : y = 33 :=
by sorry

end find_exponent_l239_239262


namespace zoo_problem_l239_239053

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l239_239053


namespace triangle_inequality_sides_l239_239277

theorem triangle_inequality_sides
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := 
by
  sorry

end triangle_inequality_sides_l239_239277


namespace sufficient_not_necessary_condition_of_sin_l239_239871

open Real

theorem sufficient_not_necessary_condition_of_sin (θ : ℝ) :
  (abs (θ - π / 12) < π / 12) → (sin θ < 1 / 2) :=
sorry

end sufficient_not_necessary_condition_of_sin_l239_239871


namespace line_passing_through_first_and_third_quadrants_l239_239702

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l239_239702


namespace solution_set_circle_l239_239994

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l239_239994


namespace problem_a_problem_b_l239_239017

-- Define the conditions for problem (a):
variable (x y z : ℝ)
variable (h_xyz : x * y * z = 1)

theorem problem_a (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 :=
sorry

-- Define the conditions for problem (b):
variable (a b c : ℚ)

theorem problem_b (h_abc : a * b * c = 1) :
  ∃ (x y z : ℚ), x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ (x * y * z = 1) ∧ 
  (x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 = 1) :=
sorry

end problem_a_problem_b_l239_239017


namespace abs_neg_2023_l239_239340

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l239_239340


namespace zero_in_interval_2_3_l239_239158

noncomputable def f (x : ℝ) : ℝ := log (x / 2) - 1 / x

theorem zero_in_interval_2_3 :
  (∃ x, 2 < x ∧ x < 3 ∧ f x = 0) :=
by {
  have h2 : f 2 = log 1 - 1 / 2 := by simp [f, log],
  have h2_neg : f 2 < 0 := by {
    rw [h2, log_one],
    norm_num,
  },
  have h3 : f 3 = log (3 / 2) - 1 / 3 := by simp [f],
  have h3_pos : f 3 > 0 := by {
    rw [h3],
    norm_num,
    exact log_pos (by norm_num : 3 / 2 > 1),
  },
  exact exists_Ioo_zero_of_continuous f (by norm_num : 2 < 3) h2_neg h3_pos,
  sorry,
}

end zero_in_interval_2_3_l239_239158


namespace average_of_sequence_l239_239065

theorem average_of_sequence (z : ℝ) : 
  (0 + 3 * z + 9 * z + 27 * z + 81 * z) / 5 = 24 * z :=
by
  sorry

end average_of_sequence_l239_239065


namespace number_of_males_is_one_part_l239_239153

-- Define the total population
def population : ℕ := 480

-- Define the number of divided parts
def parts : ℕ := 3

-- Define the population part represented by one square.
def part_population (total_population : ℕ) (n_parts : ℕ) : ℕ :=
  total_population / n_parts

-- The Lean statement for the problem
theorem number_of_males_is_one_part : part_population population parts = 160 :=
by
  -- Proof omitted
  sorry

end number_of_males_is_one_part_l239_239153


namespace pair_with_15_l239_239459

theorem pair_with_15 (s : List ℕ) (h : s = [49, 29, 9, 40, 22, 15, 53, 33, 13, 47]) :
  ∃ (t : List (ℕ × ℕ)), (∀ (x y : ℕ), (x, y) ∈ t → x + y = 62) ∧ (15, 47) ∈ t := by
  sorry

end pair_with_15_l239_239459


namespace probability_of_correct_match_l239_239944

noncomputable def probability_correct_match : ℚ :=
  1 / (Finset.univ : Finset (Equiv.Perm (Fin 4))).card

theorem probability_of_correct_match :
  probability_correct_match = 1 / 24 := by
  sorry

end probability_of_correct_match_l239_239944


namespace flowers_in_each_row_l239_239596

theorem flowers_in_each_row (rows : ℕ) (total_remaining_flowers : ℕ) 
  (percentage_remaining : ℚ) (correct_rows : rows = 50) 
  (correct_remaining : total_remaining_flowers = 8000) 
  (correct_percentage : percentage_remaining = 0.40) :
  (total_remaining_flowers : ℚ) / percentage_remaining / (rows : ℚ) = 400 := 
by {
 sorry
}

end flowers_in_each_row_l239_239596


namespace students_scoring_80_percent_l239_239635

theorem students_scoring_80_percent
  (x : ℕ)
  (h1 : 10 * 90 + x * 80 = 25 * 84)
  (h2 : x + 10 = 25) : x = 15 := 
by {
  -- Proof goes here
  sorry
}

end students_scoring_80_percent_l239_239635


namespace math_proof_problem_l239_239080

noncomputable def a_value := 1
noncomputable def b_value := 2

-- Defining the primary conditions
def condition1 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3 * x + 2 > 0) ↔ (x < 1 ∨ x > b)

def condition2 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - (2 * b - a) * x - 2 * b < 0) ↔ (-1 < x ∧ x < 4)

-- Defining the main goal
theorem math_proof_problem :
  ∃ a b : ℝ, a = a_value ∧ b = b_value ∧ condition1 a b ∧ condition2 a b := 
sorry

end math_proof_problem_l239_239080


namespace negate_universal_proposition_l239_239458

open Classical

def P (x : ℝ) : Prop := x^3 - 3*x > 0

theorem negate_universal_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

end negate_universal_proposition_l239_239458


namespace original_price_of_trouser_l239_239862

-- Define conditions
def sale_price : ℝ := 20
def discount : ℝ := 0.80

-- Define what the proof aims to show
theorem original_price_of_trouser (P : ℝ) (h : sale_price = P * (1 - discount)) : P = 100 :=
sorry

end original_price_of_trouser_l239_239862


namespace probability_not_red_light_l239_239803

theorem probability_not_red_light :
  ∀ (red_light yellow_light green_light : ℕ),
    red_light = 30 →
    yellow_light = 5 →
    green_light = 40 →
    (yellow_light + green_light) / (red_light + yellow_light + green_light) = (3 : ℚ) / 5 :=
by intros red_light yellow_light green_light h_red h_yellow h_green
   sorry

end probability_not_red_light_l239_239803


namespace simplify_and_evaluate_l239_239889

-- Defining the conditions
def a : Int := -3
def b : Int := -2

-- Defining the expression
def expr (a b : Int) : Int := (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2)

-- Stating the theorem/proof problem
theorem simplify_and_evaluate : expr a b = -6 := by
  sorry

end simplify_and_evaluate_l239_239889


namespace geometric_sum_over_term_l239_239770

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def geometric_term (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem geometric_sum_over_term (a₁ : ℝ) (q : ℝ) (h₁ : q = 3) :
  (geometric_sum a₁ q 4) / (geometric_term a₁ q 4) = 40 / 27 := by
  sorry

end geometric_sum_over_term_l239_239770


namespace base7_to_base10_l239_239618

theorem base7_to_base10 (a b c d e : ℕ) (h : 45321 = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0)
  (ha : a = 4) (hb : b = 5) (hc : c = 3) (hd : d = 2) (he : e = 1) : 
  a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 = 11481 := 
by 
  sorry

end base7_to_base10_l239_239618


namespace probability_odd_sum_probability_even_product_l239_239920
open Classical

noncomputable def number_of_possible_outcomes : ℕ := 36
noncomputable def number_of_odd_sum_outcomes : ℕ := 18
noncomputable def number_of_even_product_outcomes : ℕ := 27

theorem probability_odd_sum (n : ℕ) (m_1 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_1 = number_of_odd_sum_outcomes) : (m_1 : ℝ) / n = 1 / 2 :=
by
  sorry

theorem probability_even_product (n : ℕ) (m_2 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_2 = number_of_even_product_outcomes) : (m_2 : ℝ) / n = 3 / 4 :=
by
  sorry

end probability_odd_sum_probability_even_product_l239_239920


namespace smallest_model_length_l239_239754

theorem smallest_model_length :
  ∀ (full_size mid_size smallest : ℕ),
  full_size = 240 →
  mid_size = full_size / 10 →
  smallest = mid_size / 2 →
  smallest = 12 :=
by
  intros full_size mid_size smallest h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end smallest_model_length_l239_239754


namespace number_of_marbles_pat_keeps_l239_239878

theorem number_of_marbles_pat_keeps 
  (x : ℕ) 
  (h1 : x / 6 = 9) 
  : x / 3 = 18 :=
by
  sorry

end number_of_marbles_pat_keeps_l239_239878


namespace problem_1_problem_2_l239_239311

-- First Proof Problem
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x^2 + 1) : 
  f x = 2 * x^2 - 4 * x + 3 :=
sorry

-- Second Proof Problem
theorem problem_2 {a b : ℝ} (f : ℝ → ℝ) (hf : ∀ x, f x = x / (a * x + b))
  (h1 : f 2 = 1) (h2 : ∃! x, f x = x) : 
  f x = 2 * x / (x + 2) :=
sorry

end problem_1_problem_2_l239_239311


namespace abs_neg_2023_l239_239347

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l239_239347


namespace length_of_garden_side_l239_239615

theorem length_of_garden_side (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 112) (h2 : perimeter = 4 * side_length) : 
  side_length = 28 :=
by
  sorry

end length_of_garden_side_l239_239615


namespace possible_values_of_k_l239_239706

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l239_239706


namespace monthly_earnings_l239_239598

variable (e : ℕ) (s : ℕ) (p : ℕ) (t : ℕ)

-- conditions
def half_monthly_savings := s = e / 2
def car_price := p = 16000
def saving_months := t = 8
def total_saving := s * t = p

theorem monthly_earnings : ∀ (e s p t : ℕ), 
  half_monthly_savings e s → 
  car_price p → 
  saving_months t → 
  total_saving s t p → 
  e = 4000 :=
by
  intros e s p t h1 h2 h3 h4
  sorry

end monthly_earnings_l239_239598


namespace last_number_l239_239179

theorem last_number (A B C D E F G : ℕ)
  (h1 : A + B + C + D = 52)
  (h2 : D + E + F + G = 60)
  (h3 : E + F + G = 55)
  (h4 : D^2 = G) : G = 25 :=
by
  sorry

end last_number_l239_239179


namespace ratio_third_second_l239_239597

theorem ratio_third_second (k : ℝ) (x y z : ℝ) (h1 : y = 4 * x) (h2 : x = 18) (h3 : z = k * y) (h4 : (x + y + z) / 3 = 78) :
  z = 2 * y :=
by
  sorry

end ratio_third_second_l239_239597


namespace students_with_both_pets_l239_239668

theorem students_with_both_pets :
  ∀ (total_students students_with_dog students_with_cat students_with_both : ℕ),
    total_students = 45 →
    students_with_dog = 25 →
    students_with_cat = 34 →
    total_students = students_with_dog + students_with_cat - students_with_both →
    students_with_both = 14 :=
by
  intros total_students students_with_dog students_with_cat students_with_both
  sorry

end students_with_both_pets_l239_239668


namespace inequality_a5_b5_c5_l239_239557

theorem inequality_a5_b5_c5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3 * b * c + a * b^3 * c + a * b * c^3 :=
by
  sorry

end inequality_a5_b5_c5_l239_239557


namespace system_linear_eq_sum_l239_239689

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end system_linear_eq_sum_l239_239689


namespace josie_initial_amount_is_correct_l239_239123

def cost_of_milk := 4.00 / 2
def cost_of_bread := 3.50
def cost_of_detergent_after_coupon := 10.25 - 1.25
def cost_of_bananas := 2 * 0.75
def total_cost := cost_of_milk + cost_of_bread + cost_of_detergent_after_coupon + cost_of_bananas
def leftover := 4.00
def initial_amount := total_cost + leftover

theorem josie_initial_amount_is_correct :
  initial_amount = 20.00 := by
  sorry

end josie_initial_amount_is_correct_l239_239123


namespace sum_of_roots_l239_239359

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l239_239359


namespace expression_equals_value_l239_239215

theorem expression_equals_value : 97^3 + 3 * (97^2) + 3 * 97 + 1 = 940792 := 
by
  sorry

end expression_equals_value_l239_239215


namespace sum_of_possible_values_of_x_l239_239479

-- Define the concept of an isosceles triangle with specific angles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the angle sum property of a triangle
def angle_sum_property (a b c : ℝ) : Prop := 
  a + b + c = 180

-- State the problem using the given conditions and the required proof
theorem sum_of_possible_values_of_x :
  ∀ (x : ℝ), 
    is_isosceles_triangle 70 70 x ∨
    is_isosceles_triangle 70 x x ∨
    is_isosceles_triangle x 70 70 →
    angle_sum_property 70 70 x →
    angle_sum_property 70 x x →
    angle_sum_property x 70 70 →
    (70 + 55 + 40) = 165 :=
  by
    sorry

end sum_of_possible_values_of_x_l239_239479


namespace additional_hours_equal_five_l239_239966

-- The total hovering time constraint over two days
def total_time : ℕ := 24

-- Hovering times for each zone on the first day
def day1_mountain_time : ℕ := 3
def day1_central_time : ℕ := 4
def day1_eastern_time : ℕ := 2

-- Additional hours on the second day (variables M, C, E)
variables (M C E : ℕ)

-- The main proof statement
theorem additional_hours_equal_five 
  (h : day1_mountain_time + M + day1_central_time + C + day1_eastern_time + E = total_time) :
  M = 5 ∧ C = 5 ∧ E = 5 :=
by
  sorry

end additional_hours_equal_five_l239_239966


namespace take_home_pay_is_correct_l239_239416

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l239_239416


namespace total_earmuffs_l239_239334

theorem total_earmuffs {a b c : ℕ} (h1 : a = 1346) (h2 : b = 6444) (h3 : c = a + b) : c = 7790 := by
  sorry

end total_earmuffs_l239_239334


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l239_239469

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l239_239469


namespace roger_piles_of_quarters_l239_239442

theorem roger_piles_of_quarters (Q : ℕ) 
  (h₀ : ∃ Q : ℕ, True) 
  (h₁ : ∀ p, (p = Q) → True)
  (h₂ : ∀ c, (c = 7) → True) 
  (h₃ : Q * 14 = 42) : 
  Q = 3 := 
sorry

end roger_piles_of_quarters_l239_239442


namespace area_of_triangle_correct_l239_239368

def area_of_triangle : ℝ :=
  let A := (4, -3)
  let B := (-1, 2)
  let C := (2, -7)
  let v := (2 : ℝ, 4 : ℝ)
  let w := (-3 : ℝ, 9 : ℝ)
  let det := (2 * 9 + 3 * 4 : ℝ)
  (det / 2)

theorem area_of_triangle_correct : area_of_triangle = 15 := by
  let A := (4, -3)
  let B := (-1, 2)
  let C := (2, -7)
  let v := (2 : ℝ, 4 : ℝ)
  let w := (-3 : ℝ, 9 : ℝ)
  let det := (2 * 9 + 3 * 4 : ℝ)
  have h : area_of_triangle = (det / 2) := rfl
  rw [h]
  norm_num
  sorry

end area_of_triangle_correct_l239_239368


namespace number_of_baskets_l239_239614

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : total_apples / apples_per_basket = 37 :=
  by sorry

end number_of_baskets_l239_239614


namespace exists_five_numbers_l239_239304

theorem exists_five_numbers :
  ∃ a1 a2 a3 a4 a5 : ℤ,
  a1 + a2 < 0 ∧
  a2 + a3 < 0 ∧
  a3 + a4 < 0 ∧
  a4 + a5 < 0 ∧
  a5 + a1 < 0 ∧
  a1 + a2 + a3 + a4 + a5 > 0 :=
by
  sorry

end exists_five_numbers_l239_239304


namespace valid_documents_count_l239_239540

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end valid_documents_count_l239_239540


namespace find_y_l239_239152

-- Definitions based on conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

-- Lean statement capturing the problem
theorem find_y
  (h1 : inversely_proportional x y)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (h4 : x = -12) :
  y = -56.25 :=
sorry  -- Proof omitted

end find_y_l239_239152


namespace total_toys_is_60_l239_239736

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end total_toys_is_60_l239_239736


namespace range_of_f_is_real_l239_239077

noncomputable def f (x : ℝ) (m : ℝ) := Real.log (5^x + 4 / 5^x + m)

theorem range_of_f_is_real (m : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x m = y) ↔ m ≤ -4 :=
sorry

end range_of_f_is_real_l239_239077


namespace anie_days_to_complete_l239_239774

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_complete_l239_239774


namespace total_surface_area_of_cylinder_l239_239949

theorem total_surface_area_of_cylinder 
  (r h : ℝ) 
  (hr : r = 3) 
  (hh : h = 8) : 
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 66 * Real.pi := by
  sorry

end total_surface_area_of_cylinder_l239_239949


namespace arrangement_ways_l239_239363

def green_marbles : Nat := 7
noncomputable def N_max_blue_marbles : Nat := 924

theorem arrangement_ways (N : Nat) (blue_marbles : Nat) (total_marbles : Nat)
  (h1 : total_marbles = green_marbles + blue_marbles) 
  (h2 : ∃ b_gap, b_gap = blue_marbles - (total_marbles - green_marbles - 1))
  (h3 : blue_marbles ≥ 6)
  : N = N_max_blue_marbles := 
sorry

end arrangement_ways_l239_239363


namespace sin_value_l239_239238

theorem sin_value (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 :=
by
  sorry

end sin_value_l239_239238


namespace pupils_like_burgers_total_l239_239296

theorem pupils_like_burgers_total (total_pupils pizza_lovers both_lovers : ℕ) :
  total_pupils = 200 →
  pizza_lovers = 125 →
  both_lovers = 40 →
  (pizza_lovers - both_lovers) + (total_pupils - pizza_lovers - both_lovers) + both_lovers = 115 :=
by
  intros h_total h_pizza h_both
  rw [h_total, h_pizza, h_both]
  sorry

end pupils_like_burgers_total_l239_239296


namespace no_n_exists_11_div_mod_l239_239629

theorem no_n_exists_11_div_mod (n : ℕ) (h1 : n > 0) (h2 : 3^5 ≡ 1 [MOD 11]) (h3 : 4^5 ≡ 1 [MOD 11]) : ¬ (11 ∣ (3^n + 4^n)) := 
sorry

end no_n_exists_11_div_mod_l239_239629


namespace solve_quadratic_inequality_l239_239755

theorem solve_quadratic_inequality :
  { x : ℝ | -3 * x^2 + 8 * x + 5 < 0 } = { x : ℝ | x < -1 ∨ x > 5 / 3 } :=
sorry

end solve_quadratic_inequality_l239_239755


namespace cannot_be_the_lengths_l239_239932

theorem cannot_be_the_lengths (x y z : ℝ) (h1 : x^2 + y^2 = 16) (h2 : x^2 + z^2 = 25) (h3 : y^2 + z^2 = 49) : false :=
by
  sorry

end cannot_be_the_lengths_l239_239932


namespace savings_in_july_l239_239005

-- Definitions based on the conditions
def savings_june : ℕ := 27
def savings_august : ℕ := 21
def expenses_books : ℕ := 5
def expenses_shoes : ℕ := 17
def final_amount_left : ℕ := 40

-- Main theorem stating the problem
theorem savings_in_july (J : ℕ) : 
  savings_june + J + savings_august - (expenses_books + expenses_shoes) = final_amount_left → 
  J = 14 :=
by
  sorry

end savings_in_july_l239_239005


namespace number_of_divisibles_by_7_l239_239088

theorem number_of_divisibles_by_7 (a b : ℕ) (h1 : a = 200) (h2 : b = 400) : 
  (nat.card {n | a ≤ 7 * n ∧ 7 * n ≤ b}) = 29 := 
by sorry

end number_of_divisibles_by_7_l239_239088


namespace triangle_BD_length_l239_239405

noncomputable def triangle_length_BD : ℝ :=
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  BD

theorem triangle_BD_length : triangle_length_BD = 63 :=
by
  -- Definitions and assumptions
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)

  -- Formal proof logic corresponding to solution steps
  sorry

end triangle_BD_length_l239_239405


namespace triangle_inequality_condition_l239_239318

theorem triangle_inequality_condition (a b : ℝ) (h : a + b = 1) (ha : a ≥ 0) (hb : b ≥ 0) :
    a + b > 1 → a + 1 > b ∧ b + 1 > a := by
  sorry

end triangle_inequality_condition_l239_239318


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l239_239470

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l239_239470


namespace length_of_AB_l239_239410

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l239_239410


namespace find_integers_a_b_c_l239_239220

theorem find_integers_a_b_c :
  ∃ a b c : ℤ, ((x - a) * (x - 12) + 1 = (x + b) * (x + c)) ∧ 
  ((b + 12) * (c + 12) = 1 → ((b = -11 ∧ c = -11) → a = 10) ∧ 
  ((b = -13 ∧ c = -13) → a = 14)) :=
by
  sorry

end find_integers_a_b_c_l239_239220


namespace movies_in_first_box_l239_239977

theorem movies_in_first_box (x : ℕ) 
  (cost_first : ℕ) (cost_second : ℕ) 
  (num_second : ℕ) (avg_price : ℕ)
  (h_cost_first : cost_first = 2)
  (h_cost_second : cost_second = 5)
  (h_num_second : num_second = 5)
  (h_avg_price : avg_price = 3)
  (h_total_eq : cost_first * x + cost_second * num_second = avg_price * (x + num_second)) :
  x = 5 :=
by
  sorry

end movies_in_first_box_l239_239977


namespace cannot_be_simultaneous_squares_l239_239968

theorem cannot_be_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + y = a^2 ∧ y^2 + x = b^2) :=
by
  sorry

end cannot_be_simultaneous_squares_l239_239968


namespace fourteen_root_of_unity_l239_239822

theorem fourteen_root_of_unity (n : ℕ) (hn : n < 14) :
  (∃ k : ℤ, (tan (π / 7) + complex.I) / (tan (π / 7) - complex.I) =
            complex.exp (complex.I * ↑(2 * k * π / 14)) ∧
            (0 ≤ n ∧ n ≤ 13)) :=
by
  use 4
  sorry

end fourteen_root_of_unity_l239_239822


namespace triangle_side_length_l239_239118

theorem triangle_side_length (y z : ℝ) (cos_Y_minus_Z : ℝ) (h_y : y = 7) (h_z : z = 6) (h_cos : cos_Y_minus_Z = 17 / 18) : 
  ∃ x : ℝ, x = Real.sqrt 65 :=
by
  sorry

end triangle_side_length_l239_239118


namespace domain_of_g_x_l239_239229

theorem domain_of_g_x :
  ∀ x, (x ≤ 6 ∧ x ≥ -19) ↔ -19 ≤ x ∧ x ≤ 6 :=
by 
  -- Statement only, no proof
  sorry

end domain_of_g_x_l239_239229


namespace additional_interest_rate_l239_239756

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

end additional_interest_rate_l239_239756


namespace volume_of_cuboid_l239_239145

-- Define the edges of the cuboid
def edge1 : ℕ := 6
def edge2 : ℕ := 5
def edge3 : ℕ := 6

-- Define the volume formula for a cuboid
def volume (a b c : ℕ) : ℕ := a * b * c

-- State the theorem
theorem volume_of_cuboid : volume edge1 edge2 edge3 = 180 := by
  sorry

end volume_of_cuboid_l239_239145


namespace Vitya_catchup_mom_in_5_l239_239492

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l239_239492


namespace avocados_per_serving_l239_239071

-- Definitions for the conditions
def original_avocados : ℕ := 5
def additional_avocados : ℕ := 4
def total_avocados : ℕ := original_avocados + additional_avocados
def servings : ℕ := 3

-- Theorem stating the result
theorem avocados_per_serving : (total_avocados / servings) = 3 :=
by
  sorry

end avocados_per_serving_l239_239071


namespace luis_finish_fourth_task_l239_239437

-- Define the starting and finishing times
def start_time : ℕ := 540  -- 9:00 AM is 540 minutes from midnight
def finish_third_task : ℕ := 750  -- 12:30 PM is 750 minutes from midnight
def duration_one_task : ℕ := (750 - 540) / 3  -- Time for one task

-- Define the problem statement
theorem luis_finish_fourth_task :
  start_time = 540 →
  finish_third_task = 750 →
  3 * duration_one_task = finish_third_task - start_time →
  finish_third_task + duration_one_task = 820 :=
by
  -- You can place the proof for the theorem here
  sorry

end luis_finish_fourth_task_l239_239437


namespace problem_expression_value_l239_239836

variable (m n p q : ℝ)
variable (h1 : m + n = 0) (h2 : m / n = -1)
variable (h3 : p * q = 1) (h4 : m ≠ n)

theorem problem_expression_value : 
  (m + n) / m + 2 * p * q - m / n = 3 :=
by sorry

end problem_expression_value_l239_239836


namespace problem1_problem2_l239_239539

-- Define Problem 1 statement
theorem problem1 : 
  (\sqrt 75 + \sqrt 27 - \sqrt (1/2) * \sqrt 12 + \sqrt 24 = 8 * \sqrt 3 + \sqrt 6) :=
sorry

-- Define Problem 2 statement
theorem problem2 : 
  ((\sqrt 3 + \sqrt 2) * (\sqrt 3 - \sqrt 2) - (\sqrt 5 - 1)^2 = 2 * \sqrt 5 - 5) :=
sorry

end problem1_problem2_l239_239539


namespace steinburg_marching_band_l239_239914

theorem steinburg_marching_band :
  ∃ n : ℤ, n > 0 ∧ 30 * n < 1200 ∧ 30 * n % 34 = 6 ∧ 30 * n = 720 := by
  sorry

end steinburg_marching_band_l239_239914


namespace remainder_of_division_l239_239897

theorem remainder_of_division :
  ∀ (L S R : ℕ), 
  L = 1575 → 
  L - S = 1365 → 
  S * 7 + R = L → 
  R = 105 :=
by
  intros L S R h1 h2 h3
  sorry

end remainder_of_division_l239_239897


namespace increasing_intervals_decreasing_interval_l239_239610

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem increasing_intervals : 
  (∀ x, x < -1/3 → deriv f x > 0) ∧ 
  (∀ x, x > 1 → deriv f x > 0) :=
sorry

theorem decreasing_interval : 
  ∀ x, -1/3 < x ∧ x < 1 → deriv f x < 0 :=
sorry

end increasing_intervals_decreasing_interval_l239_239610


namespace player_matches_l239_239792

theorem player_matches (n : ℕ) :
  (34 * n + 78 = 38 * (n + 1)) → n = 10 :=
by
  intro h
  have h1 : 34 * n + 78 = 38 * n + 38 := by sorry
  have h2 : 78 = 4 * n + 38 := by sorry
  have h3 : 40 = 4 * n := by sorry
  have h4 : n = 10 := by sorry
  exact h4

end player_matches_l239_239792


namespace g_triple_3_eq_31_l239_239433

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 1 else 2 * n - 3

theorem g_triple_3_eq_31 : g (g (g 3)) = 31 := by
  sorry

end g_triple_3_eq_31_l239_239433


namespace vitya_catch_up_time_l239_239500

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l239_239500


namespace length_of_AB_l239_239409

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l239_239409


namespace part1_part2_l239_239430

-- Definitions and conditions
variables {A B C a b c : ℝ}
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A)) -- Given condition

-- Part (1): If A = 2B, then find C
theorem part1 (h2 : A = 2 * B) : C = (5 / 8) * π := by
  sorry

-- Part (2): Prove that 2a² = b² + c²
theorem part2 : 2 * a^2 = b^2 + c^2 := by
  sorry

end part1_part2_l239_239430


namespace pizza_eating_group_l239_239791

theorem pizza_eating_group (x y : ℕ) (h1 : 6 * x + 2 * y ≥ 49) (h2 : 7 * x + 3 * y ≤ 59) : x = 8 ∧ y = 2 := by
  sorry

end pizza_eating_group_l239_239791


namespace an_geometric_l239_239072

-- Define the functions and conditions
def f (x : ℝ) (b : ℝ) : ℝ := b * x + 1

def g (n : ℕ) (b : ℝ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => f (g n b) b

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ :=
  g (n + 1) b - g n b

-- Prove that a_n is a geometric sequence
theorem an_geometric (b : ℝ) (h : b ≠ 1) : 
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) b = q * a n b :=
sorry

end an_geometric_l239_239072


namespace code_word_MEET_l239_239904

def translate_GREAT_TIME : String → ℕ 
| "G" => 0
| "R" => 1
| "E" => 2
| "A" => 3
| "T" => 4
| "I" => 5
| "M" => 6
| _   => 0 -- Default case for simplicity, not strictly necessary

theorem code_word_MEET : translate_GREAT_TIME "M" = 6 ∧ translate_GREAT_TIME "E" = 2 ∧ translate_GREAT_TIME "T" = 4 →
  let MEET : ℕ := (translate_GREAT_TIME "M" * 1000) + 
                  (translate_GREAT_TIME "E" * 100) + 
                  (translate_GREAT_TIME "E" * 10) + 
                  (translate_GREAT_TIME "T")
  MEET = 6224 :=
sorry

end code_word_MEET_l239_239904


namespace vitya_catch_up_l239_239483

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l239_239483


namespace profit_without_discount_l239_239307

theorem profit_without_discount (CP SP_with_discount SP_without_discount : ℝ) (h1 : CP = 100) (h2 : SP_with_discount = CP + 0.235 * CP) (h3 : SP_with_discount = 0.95 * SP_without_discount) : (SP_without_discount - CP) / CP * 100 = 30 :=
by
  sorry

end profit_without_discount_l239_239307


namespace maximum_smallest_angle_l239_239103

-- Definition of points on the plane
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

-- Function to calculate the angle between three points (p1, p2, p3)
def angle (p1 p2 p3 : Point2D) : ℝ := 
  -- Placeholder for the actual angle calculation
  sorry

-- Condition: Given five points on a plane
variables (A B C D E : Point2D)

-- Maximum value of the smallest angle formed by any triple is 36 degrees
theorem maximum_smallest_angle :
  ∃ α : ℝ, (∀ p1 p2 p3 : Point2D, α ≤ angle p1 p2 p3) ∧ α = 36 :=
sorry

end maximum_smallest_angle_l239_239103


namespace rubber_boat_lost_time_l239_239646

theorem rubber_boat_lost_time (a b : ℝ) (x : ℝ) (h : (5 - x) * (a - b) + (6 - x) * b = a + b) : x = 4 :=
  sorry

end rubber_boat_lost_time_l239_239646


namespace cost_when_q_is_2_l239_239155

-- Defining the cost function
def cost (q : ℕ) : ℕ := q^3 + q - 1

-- Theorem to prove the cost when q = 2
theorem cost_when_q_is_2 : cost 2 = 9 :=
by
  -- placeholder for the proof
  sorry

end cost_when_q_is_2_l239_239155


namespace express_in_scientific_notation_l239_239747

theorem express_in_scientific_notation (n : ℝ) (h : n = 456.87 * 10^6) : n = 4.5687 * 10^8 :=
by 
  -- sorry to skip the proof
  sorry

end express_in_scientific_notation_l239_239747


namespace distance_between_two_cars_l239_239935

theorem distance_between_two_cars 
    (initial_distance : ℝ) 
    (first_car_distance1 : ℝ) 
    (first_car_distance2 : ℝ)
    (second_car_distance : ℝ) 
    (final_distance : ℝ) :
    initial_distance = 150 →
    first_car_distance1 = 25 →
    first_car_distance2 = 25 →
    second_car_distance = 35 →
    final_distance = initial_distance - (first_car_distance1 + first_car_distance2 + second_car_distance) →
    final_distance = 65 :=
by
  intros h_initial h_first1 h_first2 h_second h_final
  sorry

end distance_between_two_cars_l239_239935


namespace vitya_catches_up_in_5_minutes_l239_239494

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l239_239494


namespace sunzi_system_l239_239114

variable (x y : ℝ)

theorem sunzi_system :
  (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  sorry

end sunzi_system_l239_239114


namespace range_of_a_l239_239855

open Real

theorem range_of_a (a x y : ℝ)
  (h1 : (x - a) ^ 2 + (y - (a + 2)) ^ 2 = 1)
  (h2 : ∃ M : ℝ × ℝ, (M.1 - a) ^ 2 + (M.2 - (a + 2)) ^ 2 = 1
                       ∧ dist M (0, 3) = 2 * dist M (0, 0)) :
  -3 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l239_239855


namespace anie_days_to_complete_l239_239775

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_complete_l239_239775


namespace intersecting_lines_l239_239903

theorem intersecting_lines (c d : ℝ) :
  (∀ x y, (x = (1/3) * y + c) ∧ (y = (1/3) * x + d) → x = 3 ∧ y = 6) →
  c + d = 6 :=
by
  sorry

end intersecting_lines_l239_239903


namespace smallest_solution_l239_239173

theorem smallest_solution (x : ℝ) (h : x^2 + 10 * x - 24 = 0) : x = -12 :=
sorry

end smallest_solution_l239_239173


namespace no_largest_integer_exists_l239_239926

/--
  Define a predicate to check whether an integer is a non-square.
-/
def is_non_square (n : ℕ) : Prop :=
  ¬ ∃ m : ℕ, m * m = n

/--
  Define the main theorem which states that there is no largest positive integer
  that cannot be expressed as the sum of a positive integral multiple of 36
  and a positive non-square integer less than 36.
-/
theorem no_largest_integer_exists : ¬ ∃ (n : ℕ), 
  ∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b < 36 ∧ is_non_square b →
  n ≠ 36 * a + b :=
sorry

end no_largest_integer_exists_l239_239926


namespace harry_walks_9_dogs_on_thursday_l239_239085

-- Define the number of dogs Harry walks on specific days
def dogs_monday : Nat := 7
def dogs_wednesday : Nat := 7
def dogs_friday : Nat := 7
def dogs_tuesday : Nat := 12

-- Define the payment per dog
def payment_per_dog : Nat := 5

-- Define total weekly earnings
def total_weekly_earnings : Nat := 210

-- Define the number of dogs Harry walks on Thursday
def dogs_thursday : Nat := 9

-- Define the total earnings for Monday, Wednesday, Friday, and Tuesday
def earnings_first_four_days : Nat := (dogs_monday + dogs_wednesday + dogs_friday + dogs_tuesday) * payment_per_dog

-- Now we state the theorem that we need to prove
theorem harry_walks_9_dogs_on_thursday :
  (total_weekly_earnings - earnings_first_four_days) / payment_per_dog = dogs_thursday :=
by
  -- Proof omitted
  sorry

end harry_walks_9_dogs_on_thursday_l239_239085


namespace find_digit_property_l239_239673

theorem find_digit_property (a x : ℕ) (h : 10 * a + x = a + x + a * x) : x = 9 :=
sorry

end find_digit_property_l239_239673


namespace original_number_of_laborers_l239_239197

theorem original_number_of_laborers (L : ℕ) 
  (h : L * 9 = (L - 6) * 15) : L = 15 :=
sorry

end original_number_of_laborers_l239_239197


namespace grasshopper_opposite_corner_moves_l239_239919

noncomputable def grasshopper_jump_count : ℕ :=
  Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9)

theorem grasshopper_opposite_corner_moves :
  grasshopper_jump_count = Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9) :=
by
  -- The detailed proof would go here.
  sorry

end grasshopper_opposite_corner_moves_l239_239919


namespace original_number_of_players_l239_239473

theorem original_number_of_players 
    (n : ℕ) (W : ℕ)
    (h1 : W = n * 112)
    (h2 : W + 110 + 60 = (n + 2) * 106) : 
    n = 7 :=
by
  sorry

end original_number_of_players_l239_239473


namespace fraction_equals_i_l239_239076

theorem fraction_equals_i (m n : ℝ) (i : ℂ) (h : i * i = -1) (h_cond : m * (1 + i) = (11 + n * i)) :
  (m + n * i) / (m - n * i) = i :=
sorry

end fraction_equals_i_l239_239076


namespace sunny_lead_l239_239397

-- Define the context of the race
variables {s m : ℝ}  -- s: Sunny's speed, m: Misty's speed
variables (distance_first : ℝ) (distance_ahead_first : ℝ)
variables (additional_distance_sunny_second : ℝ) (correct_answer : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first = 400 ∧
  distance_ahead_first = 20 ∧
  additional_distance_sunny_second = 40 ∧
  correct_answer = 20 

-- The math proof problem in Lean 4
theorem sunny_lead (h : conditions distance_first distance_ahead_first additional_distance_sunny_second correct_answer) :
  ∀ s m : ℝ, s / m = (400 / 380 : ℝ) → 
  (s / m) * 400 + additional_distance_sunny_second = (m / s) * 440 + correct_answer :=
sorry

end sunny_lead_l239_239397


namespace natural_numbers_divisible_by_7_between_200_400_l239_239086

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l239_239086


namespace num_female_students_l239_239289

theorem num_female_students (F : ℕ) (h1: 8 * 85 + F * 92 = (8 + F) * 90) : F = 20 := 
by
  sorry

end num_female_students_l239_239289


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l239_239719

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l239_239719


namespace children_tickets_sold_l239_239176

theorem children_tickets_sold {A C : ℕ} (h1 : 6 * A + 4 * C = 104) (h2 : A + C = 21) : C = 11 :=
by
  sorry

end children_tickets_sold_l239_239176


namespace red_light_at_A_prob_calc_l239_239001

-- Defining the conditions
def count_total_permutations : ℕ := Nat.factorial 4 / Nat.factorial 1
def count_favorable_permutations : ℕ := Nat.factorial 3 / Nat.factorial 1

-- Calculating the probability
def probability_red_at_A : ℚ := count_favorable_permutations / count_total_permutations

-- Statement to be proved
theorem red_light_at_A_prob_calc : probability_red_at_A = 1 / 4 :=
by
  sorry

end red_light_at_A_prob_calc_l239_239001


namespace range_of_a_l239_239838

noncomputable def f (x : ℝ) : ℝ := sorry -- The actual definition of the function f is not given
def g (a x : ℝ) : ℝ := a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc (-2 : ℝ) 2 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2 : ℝ) 2 ∧ g a x₀ = f x₁) ↔
  a ≤ -1/2 ∨ 5/2 ≤ a :=
by 
  sorry

end range_of_a_l239_239838


namespace seat_number_X_l239_239472

theorem seat_number_X (X : ℕ) (h1 : 42 - 30 = X - 6) : X = 18 :=
by
  sorry

end seat_number_X_l239_239472


namespace barium_atoms_in_compound_l239_239638

noncomputable def barium_atoms (total_molecular_weight : ℝ) (weight_ba_per_atom : ℝ) (weight_br_per_atom : ℝ) (num_br_atoms : ℕ) : ℝ :=
  (total_molecular_weight - (num_br_atoms * weight_br_per_atom)) / weight_ba_per_atom

theorem barium_atoms_in_compound :
  barium_atoms 297 137.33 79.90 2 = 1 :=
by
  unfold barium_atoms
  norm_num
  sorry

end barium_atoms_in_compound_l239_239638


namespace probability_not_exceed_60W_l239_239771

noncomputable def total_bulbs : ℕ := 250
noncomputable def bulbs_100W : ℕ := 100
noncomputable def bulbs_60W : ℕ := 50
noncomputable def bulbs_25W : ℕ := 50
noncomputable def bulbs_15W : ℕ := 50

noncomputable def probability_of_event (event : ℕ) (total : ℕ) : ℝ := 
  event / total

noncomputable def P_A : ℝ := probability_of_event bulbs_60W total_bulbs
noncomputable def P_B : ℝ := probability_of_event bulbs_25W total_bulbs
noncomputable def P_C : ℝ := probability_of_event bulbs_15W total_bulbs
noncomputable def P_D : ℝ := probability_of_event bulbs_100W total_bulbs

theorem probability_not_exceed_60W : 
  P_A + P_B + P_C = 3 / 5 :=
by
  sorry

end probability_not_exceed_60W_l239_239771


namespace relationship_y1_y2_l239_239687

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem relationship_y1_y2 :
  ∀ (a b c x₀ x₁ x₂ : ℝ),
    (quadratic_function a b c 0 = 4) →
    (quadratic_function a b c 1 = 1) →
    (quadratic_function a b c 2 = 0) →
    1 < x₁ → 
    x₁ < 2 → 
    3 < x₂ → 
    x₂ < 4 → 
    (quadratic_function a b c x₁ < quadratic_function a b c x₂) :=
by 
  sorry

end relationship_y1_y2_l239_239687


namespace diet_soda_ratio_l239_239790

def total_bottles : ℕ := 60
def diet_soda_bottles : ℕ := 14

theorem diet_soda_ratio : (diet_soda_bottles * 30) = (total_bottles * 7) :=
by {
  -- We're given that total_bottles = 60 and diet_soda_bottles = 14
  -- So to prove the ratio 14/60 is equivalent to 7/30:
  -- Multiplying both sides by 30 and 60 simplifies the arithmetic.
  sorry
}

end diet_soda_ratio_l239_239790


namespace rabbit_distribution_problem_l239_239282

-- Define the rabbits and stores
constant rabbits : Fin 6
constant stores : Fin 5

-- Define the parent and child relationships
constant is_parent : Fin 6 → Bool

-- Define the distribution function
noncomputable def distribute_rabbits (distribution : Fin 6 → Fin 5) : Prop :=
  ∀ i j : Fin 6, (is_parent i = is_parent j → distribution i ≠ distribution j) ∧ (∀ s : Fin 5, ∃! r : Fin 6, distribution r = s → r < 3)

-- Statement of the problem
theorem rabbit_distribution_problem :
  (∃ distribution : Fin 6 → Fin 5, distribute_rabbits distribution) → ∃! (ways : ℕ), ways = 446 :=
sorry

end rabbit_distribution_problem_l239_239282


namespace fruits_in_good_condition_percentage_l239_239178

theorem fruits_in_good_condition_percentage (total_oranges total_bananas rotten_oranges_percentage rotten_bananas_percentage : ℝ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percentage = 0.15) 
  (h4 : rotten_bananas_percentage = 0.08) : 
  (1 - ((rotten_oranges_percentage * total_oranges + rotten_bananas_percentage * total_bananas) / (total_oranges + total_bananas))) * 100 = 87.8 :=
by 
  sorry

end fruits_in_good_condition_percentage_l239_239178


namespace chocolate_bars_count_l239_239129

theorem chocolate_bars_count (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
    (h_milk : milk_chocolate = 25)
    (h_almond : almond_chocolate = 25)
    (h_white : white_chocolate = 25)
    (h_percent : milk_chocolate = almond_chocolate ∧ almond_chocolate = white_chocolate ∧ white_chocolate = dark_chocolate) :
    dark_chocolate = 25 := by
  sorry

end chocolate_bars_count_l239_239129


namespace length_of_segment_AB_l239_239408

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l239_239408


namespace robbery_proof_l239_239954

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l239_239954


namespace minimum_value_1_minimum_value_2_l239_239374

noncomputable section

open Real -- Use the real numbers

theorem minimum_value_1 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + y^2 + z^2 >= 8 / 3 :=
by
  sorry  -- Proof omitted
 
theorem minimum_value_2 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + (y - 1)^2 + z^2 >= 6 :=
by
  sorry  -- Proof omitted

end minimum_value_1_minimum_value_2_l239_239374


namespace arithmetic_seq_sum_a3_a15_l239_239569

theorem arithmetic_seq_sum_a3_a15 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_eq : a 1 - a 5 + a 9 - a 13 + a 17 = 117) :
  a 3 + a 15 = 234 :=
sorry

end arithmetic_seq_sum_a3_a15_l239_239569


namespace system_solution_unique_l239_239826

theorem system_solution_unique : 
  ∀ (x y z : ℝ),
  (4 * x^2) / (1 + 4 * x^2) = y ∧
  (4 * y^2) / (1 + 4 * y^2) = z ∧
  (4 * z^2) / (1 + 4 * z^2) = x 
  → (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solution_unique_l239_239826


namespace complement_set_example_l239_239572

open Set

variable (U M : Set ℕ)

def complement (U M : Set ℕ) := U \ M

theorem complement_set_example :
  (U = {1, 2, 3, 4, 5, 6}) → 
  (M = {1, 3, 5}) → 
  (complement U M = {2, 4, 6}) := by
  intros hU hM
  rw [complement, hU, hM]
  sorry

end complement_set_example_l239_239572


namespace least_positive_integer_mod_cond_l239_239927

theorem least_positive_integer_mod_cond (N : ℕ) :
  (N % 6 = 5) ∧ 
  (N % 7 = 6) ∧ 
  (N % 8 = 7) ∧ 
  (N % 9 = 8) ∧ 
  (N % 10 = 9) ∧ 
  (N % 11 = 10) →
  N = 27719 :=
by
  sorry

end least_positive_integer_mod_cond_l239_239927


namespace relationship_between_x_and_y_l239_239102

variables (x y : ℝ)

theorem relationship_between_x_and_y (h1 : x + y > 2 * x) (h2 : x - y < 2 * y) : y > x := 
sorry

end relationship_between_x_and_y_l239_239102


namespace vitya_catchup_time_l239_239489

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l239_239489


namespace k_positive_first_third_quadrants_l239_239709

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l239_239709


namespace range_of_a_l239_239682

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 3 → log (x - 1) + log (3 - x) = log (a - x)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3) →
  3 < a ∧ a < 13 / 4 :=
by
  sorry

end range_of_a_l239_239682


namespace y_in_terms_of_w_l239_239200

theorem y_in_terms_of_w (y w : ℝ) (h1 : y = 3^2 - 1) (h2 : w = 2) : y = 4 * w :=
by
  sorry

end y_in_terms_of_w_l239_239200


namespace complement_union_l239_239690

open Set

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ)
  (hA : A = { x : ℝ | x^2 - 3 * x < 4 })
  (hB : B = { x : ℝ | |x| ≥ 2 }) :
  (compl B ∪ A) = Ioo (-2 : ℝ) 4 :=
by
  -- We state that complement and union is as required.
  sorry

end complement_union_l239_239690


namespace sum_of_integers_is_19_l239_239723

theorem sum_of_integers_is_19
  (a b : ℕ) 
  (h1 : a > b) 
  (h2 : a - b = 5) 
  (h3 : a * b = 84) : 
  a + b = 19 :=
sorry

end sum_of_integers_is_19_l239_239723


namespace triangle_inequality_holds_l239_239594

theorem triangle_inequality_holds (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^3 + b^3 + c^3 + 4 * a * b * c ≤ (9 / 32) * (a + b + c)^3 :=
by {
  sorry
}

end triangle_inequality_holds_l239_239594


namespace sales_price_reduction_l239_239191

theorem sales_price_reduction
  (current_sales : ℝ := 20)
  (current_profit_per_shirt : ℝ := 40)
  (sales_increase_per_dollar : ℝ := 2)
  (desired_profit : ℝ := 1200) :
  ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1200 ∧ x = 20 :=
by
  use 20
  sorry

end sales_price_reduction_l239_239191


namespace decimal_to_vulgar_fraction_l239_239658

theorem decimal_to_vulgar_fraction (h : (34 / 100 : ℚ) = 0.34) : (0.34 : ℚ) = 17 / 50 := by
  sorry

end decimal_to_vulgar_fraction_l239_239658


namespace ratio_sheep_horses_l239_239535

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end ratio_sheep_horses_l239_239535


namespace sum_of_f_l239_239373

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f :
  f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 3 * Real.sqrt 2 :=
by
  sorry

end sum_of_f_l239_239373


namespace find_other_endpoint_of_diameter_l239_239656

noncomputable def circle_center : (ℝ × ℝ) := (4, -2)
noncomputable def one_endpoint_of_diameter : (ℝ × ℝ) := (7, 5)
noncomputable def other_endpoint_of_diameter : (ℝ × ℝ) := (1, -9)

theorem find_other_endpoint_of_diameter :
  let (cx, cy) := circle_center
  let (x1, y1) := one_endpoint_of_diameter
  let (x2, y2) := other_endpoint_of_diameter
  (x2, y2) = (2 * cx - x1, 2 * cy - y1) :=
by
  sorry

end find_other_endpoint_of_diameter_l239_239656


namespace jebb_take_home_pay_l239_239413

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l239_239413


namespace f_neg_def_l239_239868

variable (f : ℝ → ℝ)
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_def_pos : ∀ x : ℝ, 0 < x → f x = x * (1 + x)

theorem f_neg_def (x : ℝ) (hx : x < 0) : f x = x * (1 - x) := by
  sorry

end f_neg_def_l239_239868


namespace line_passing_through_first_and_third_quadrants_l239_239700

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l239_239700


namespace eq_fraction_l239_239843

def f(x : ℤ) : ℤ := 3 * x + 4
def g(x : ℤ) : ℤ := 2 * x - 1

theorem eq_fraction : (f (g (f 3))) / (g (f (g 3))) = 79 / 37 := by
  sorry

end eq_fraction_l239_239843


namespace take_home_pay_l239_239419

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l239_239419


namespace susan_age_l239_239447

theorem susan_age (S J B : ℝ) 
  (h1 : S = 2 * J)
  (h2 : S + J + B = 60) 
  (h3 : B = J + 10) : 
  S = 25 := sorry

end susan_age_l239_239447


namespace number_of_friends_gave_money_l239_239828

-- Definition of given data in conditions
def amount_per_friend : ℕ := 6
def total_amount : ℕ := 30

-- Theorem to be proved
theorem number_of_friends_gave_money : total_amount / amount_per_friend = 5 :=
by
  sorry

end number_of_friends_gave_money_l239_239828


namespace vitya_catchup_time_l239_239488

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l239_239488


namespace sum_of_roots_l239_239351

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l239_239351


namespace hexagon_angle_Q_l239_239548

theorem hexagon_angle_Q
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : a1 = 134) 
  (h2 : a2 = 98) 
  (h3 : a3 = 120) 
  (h4 : a4 = 110) 
  (h5 : a5 = 96) 
  (sum_hexagon_angles : a1 + a2 + a3 + a4 + a5 + Q = 720) : 
  Q = 162 := by {
  sorry
}

end hexagon_angle_Q_l239_239548


namespace packaging_combinations_l239_239939

-- Conditions
def wrapping_paper_choices : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

-- Question and proof
theorem packaging_combinations : wrapping_paper_choices * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end packaging_combinations_l239_239939


namespace no_common_points_l239_239235

theorem no_common_points 
  (x x_o y y_o : ℝ) 
  (h_parabola : y^2 = 4 * x) 
  (h_inside : y_o^2 < 4 * x_o) : 
  ¬ ∃ (x y : ℝ), y * y_o = 2 * (x + x_o) ∧ y^2 = 4 * x :=
by
  sorry

end no_common_points_l239_239235


namespace num_natural_numbers_divisible_by_7_l239_239097

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l239_239097


namespace wrapping_paper_area_l239_239633

theorem wrapping_paper_area (a : ℝ) (h : ℝ) : h = a ∧ 1 ≥ 0 → 4 * a^2 = 4 * a^2 :=
by sorry

end wrapping_paper_area_l239_239633


namespace part1_part2_l239_239378

variable {α : Type*}
def A : Set ℝ := {x | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1)
theorem part1 : B 5 ∩ A = {x | 6 ≤ x ∧ x < 9} := 
sorry

-- Part (2)
theorem part2 (m : ℝ): A ∩ B m = B m ↔ m < 5 :=
sorry

end part1_part2_l239_239378


namespace rodney_lifting_capacity_l239_239139

theorem rodney_lifting_capacity 
  (R O N : ℕ)
  (h1 : R + O + N = 239)
  (h2 : R = 2 * O)
  (h3 : O = 4 * N - 7) : 
  R = 146 := 
by
  sorry

end rodney_lifting_capacity_l239_239139


namespace find_a_range_l239_239074

-- Definitions of sets A and B
def A (a x : ℝ) : Prop := a + 1 ≤ x ∧ x ≤ 2 * a - 1
def B (x : ℝ) : Prop := x ≤ 3 ∨ x > 5

-- Condition p: A ⊆ B
def p (a : ℝ) : Prop := ∀ x, A a x → B x

-- The function f(x) = x^2 - 2ax + 1
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Condition q: f(x) is increasing on (1/2, +∞)
def q (a : ℝ) : Prop := ∀ x y, 1/2 < x → x < y → f a x ≤ f a y

-- The given propositions
def prop1 (a : ℝ) : Prop := p a
def prop2 (a : ℝ) : Prop := q a

-- Given conditions
def given_conditions (a : ℝ) : Prop := ¬ (prop1 a ∧ prop2 a) ∧ (prop1 a ∨ prop2 a)

-- Proof statement: Find the range of values for 'a' according to the given conditions
theorem find_a_range (a : ℝ) :
  given_conditions a →
  (1/2 < a ∧ a ≤ 2) ∨ (4 < a) :=
sorry

end find_a_range_l239_239074


namespace line_passing_through_first_and_third_quadrants_l239_239699

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l239_239699


namespace solution_set_is_circle_with_exclusion_l239_239987

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l239_239987


namespace equation_in_terms_of_y_l239_239599

theorem equation_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x :=
sorry

end equation_in_terms_of_y_l239_239599


namespace total_tiles_is_1352_l239_239527

noncomputable def side_length_of_floor := 39

noncomputable def total_tiles_covering_floor (n : ℕ) : ℕ :=
  (n ^ 2) - ((n / 3) ^ 2)

theorem total_tiles_is_1352 :
  total_tiles_covering_floor side_length_of_floor = 1352 := by
  sorry

end total_tiles_is_1352_l239_239527


namespace possible_values_of_k_l239_239705

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l239_239705


namespace first_team_engineers_l239_239035

theorem first_team_engineers (E : ℕ) 
  (teamQ_engineers : ℕ := 16) 
  (work_days_teamQ : ℕ := 30) 
  (work_days_first_team : ℕ := 32) 
  (working_capacity_ratio : ℚ := 3 / 2) :
  E * work_days_first_team * 3 = teamQ_engineers * work_days_teamQ * 2 → 
  E = 10 :=
by
  sorry

end first_team_engineers_l239_239035


namespace ball_travel_distance_fourth_hit_l239_239795

theorem ball_travel_distance_fourth_hit :
  let initial_height := 150
  let rebound_ratio := 1 / 3
  let distances := [initial_height, 
                    initial_height * rebound_ratio, 
                    initial_height * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio]
  distances.sum = 294 + 1 / 3 := by
  sorry

end ball_travel_distance_fourth_hit_l239_239795


namespace option_e_is_perfect_square_l239_239621

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem option_e_is_perfect_square :
  is_perfect_square (4^10 * 5^5 * 6^10) :=
sorry

end option_e_is_perfect_square_l239_239621


namespace binomial_square_b_value_l239_239393

theorem binomial_square_b_value (b : ℝ) (h : ∃ c : ℝ, (9 * x^2 + 24 * x + b) = (3 * x + c) ^ 2) : b = 16 :=
sorry

end binomial_square_b_value_l239_239393


namespace factorization_correct_l239_239364

noncomputable def factor_polynomial : Polynomial ℝ :=
  Polynomial.X^6 - 64

theorem factorization_correct : 
  factor_polynomial = 
  (Polynomial.X - 2) * 
  (Polynomial.X + 2) * 
  (Polynomial.X^4 + 4 * Polynomial.X^2 + 16) :=
by
  sorry

end factorization_correct_l239_239364


namespace length_of_segment_AB_l239_239407

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l239_239407


namespace man_age_twice_son_age_l239_239315

theorem man_age_twice_son_age (S M : ℕ) (h1 : M = S + 24) (h2 : S = 22) : 
  ∃ Y : ℕ, M + Y = 2 * (S + Y) ∧ Y = 2 :=
by 
  sorry

end man_age_twice_son_age_l239_239315


namespace sum_of_roots_l239_239358

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l239_239358


namespace rich_avg_time_per_mile_l239_239138

-- Define the total time in minutes and the total distance
def total_minutes : ℕ := 517
def total_miles : ℕ := 50

-- Define a function to calculate the average time per mile
def avg_time_per_mile (total_time : ℕ) (distance : ℕ) : ℚ :=
  total_time / distance

-- Theorem statement
theorem rich_avg_time_per_mile :
  avg_time_per_mile total_minutes total_miles = 10.34 :=
by
  -- Proof steps go here
  sorry

end rich_avg_time_per_mile_l239_239138


namespace complement_intersection_l239_239082

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) (hM : M = {1, 3, 5, 7}) (hN : N = {2, 5, 8}) :
  (U \ M) ∩ N = {2, 8} :=
by
  sorry

end complement_intersection_l239_239082


namespace floral_arrangement_carnations_percentage_l239_239399

theorem floral_arrangement_carnations_percentage :
  ∀ (F : ℕ),
  (1 / 4) * (7 / 10) * F + (2 / 3) * (3 / 10) * F = (29 / 40) * F :=
by
  sorry

end floral_arrangement_carnations_percentage_l239_239399


namespace zoo_camels_l239_239915

theorem zoo_camels (x y : ℕ) (h1 : x - y = 10) (h2 : x + 2 * y = 55) : x + y = 40 :=
by sorry

end zoo_camels_l239_239915


namespace line_through_two_quadrants_l239_239717

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l239_239717


namespace Vitya_catchup_mom_in_5_l239_239491

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l239_239491


namespace num_divisible_by_7_200_to_400_l239_239095

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l239_239095


namespace cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l239_239974

-- 1) Cylinder
theorem cylinder_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 = r^2 :=
sorry

-- 2) Sphere
theorem sphere_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 + z^2 = r^2 :=
sorry

-- 3) Hyperbolic Cylinder
theorem hyperbolic_cylinder_is_defined (m : ℝ) :
  ∀ (x y z : ℝ), xy = m → ∃ (k : ℝ), k = m ∧ xy = k :=
sorry

-- 4) Parabolic Cylinder
theorem parabolic_cylinder_is_defined :
  ∀ (x z : ℝ), z = x^2 → ∃ (k : ℝ), k = 1 ∧ z = k*x^2 :=
sorry

end cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l239_239974


namespace smallest_solution_l239_239509

noncomputable def equation (x : ℝ) := x^4 - 40 * x^2 + 400

theorem smallest_solution : ∃ x : ℝ, equation x = 0 ∧ ∀ y : ℝ, equation y = 0 → -2 * Real.sqrt 5 ≤ y :=
by
  sorry

end smallest_solution_l239_239509


namespace shem_wage_multiple_kem_l239_239444

-- Define the hourly wages and conditions
def kem_hourly_wage : ℝ := 4
def shem_daily_wage : ℝ := 80
def shem_workday_hours : ℝ := 8

-- Prove the multiple of Shem's hourly wage compared to Kem's hourly wage
theorem shem_wage_multiple_kem : (shem_daily_wage / shem_workday_hours) / kem_hourly_wage = 2.5 := by
  sorry

end shem_wage_multiple_kem_l239_239444


namespace pages_read_per_day_l239_239814

-- Define the total number of pages in the book
def total_pages := 96

-- Define the number of days it took to finish the book
def number_of_days := 12

-- Define pages read per day for Charles
def pages_per_day := total_pages / number_of_days

-- Prove that the number of pages read per day is equal to 8
theorem pages_read_per_day : pages_per_day = 8 :=
by
  sorry

end pages_read_per_day_l239_239814


namespace problem_1_problem_2_problem_3_problem_4_l239_239655

theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 :=
by sorry

theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 :=
by sorry

theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 :=
by sorry

theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l239_239655


namespace find_x_l239_239573

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

-- Definition for the condition of parallel vectors
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Mathematical statement to prove
theorem find_x (x : ℝ) 
  (h_parallel : are_parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2)) : 
  x = -1 :=
sorry

end find_x_l239_239573


namespace good_coloring_count_l239_239916

noncomputable def c_n (n : ℕ) : ℤ :=
  1 / 2 * (3^(n + 1) + (-1)^(n + 1))

theorem good_coloring_count (n : ℕ) : 
  ∃ c : ℕ → ℤ, c n = c_n n := sorry

end good_coloring_count_l239_239916


namespace sum_of_roots_l239_239353

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l239_239353


namespace robbery_participants_l239_239963

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l239_239963


namespace added_water_is_18_l239_239523

def capacity : ℕ := 40

def initial_full_percent : ℚ := 0.30

def final_full_fraction : ℚ := 3/4

def initial_water (capacity : ℕ) (initial_full_percent : ℚ) : ℚ :=
  initial_full_percent * capacity

def final_water (capacity : ℕ) (final_full_fraction : ℚ) : ℚ :=
  final_full_fraction * capacity

def water_added (initial_water : ℚ) (final_water : ℚ) : ℚ :=
  final_water - initial_water

theorem added_water_is_18 :
  water_added (initial_water capacity initial_full_percent) (final_water capacity final_full_fraction) = 18 := by
  sorry

end added_water_is_18_l239_239523


namespace find_integer_pairs_l239_239059

theorem find_integer_pairs (x y : ℤ) (h_xy : x ≤ y) (h_eq : (1 : ℚ)/x + (1 : ℚ)/y = 1/4) :
  (x, y) = (5, 20) ∨ (x, y) = (6, 12) ∨ (x, y) = (8, 8) ∨ (x, y) = (-4, 2) ∨ (x, y) = (-12, 3) :=
sorry

end find_integer_pairs_l239_239059


namespace age_problem_l239_239544

theorem age_problem
    (D X : ℕ) 
    (h1 : D = 4 * X) 
    (h2 : D = X + 30) : D = 40 ∧ X = 10 := by
  sorry

end age_problem_l239_239544


namespace ap_80th_term_l239_239849

/--
If the sum of the first 20 terms of an arithmetic progression is 200,
and the sum of the first 60 terms is 180, then the 80th term is -573/40.
-/
theorem ap_80th_term (S : ℤ → ℚ) (a d : ℚ)
  (h1 : S 20 = 200)
  (h2 : S 60 = 180)
  (hS : ∀ n, S n = n / 2 * (2 * a + (n - 1) * d)) :
  a + 79 * d = -573 / 40 :=
by {
  sorry
}

end ap_80th_term_l239_239849


namespace Jason_total_money_l239_239120

theorem Jason_total_money :
  let quarter_value := 0.25
  let dime_value := 0.10
  let nickel_value := 0.05
  let initial_quarters := 49
  let initial_dimes := 32
  let initial_nickels := 18
  let additional_quarters := 25
  let additional_dimes := 15
  let additional_nickels := 10
  let initial_money := initial_quarters * quarter_value + initial_dimes * dime_value + initial_nickels * nickel_value
  let additional_money := additional_quarters * quarter_value + additional_dimes * dime_value + additional_nickels * nickel_value
  initial_money + additional_money = 24.60 :=
by
  sorry

end Jason_total_money_l239_239120


namespace min_oranges_in_new_box_l239_239002

theorem min_oranges_in_new_box (m n : ℕ) (x : ℕ) (h1 : m + n ≤ 60) 
    (h2 : 59 * m = 60 * n + x) : x = 30 :=
sorry

end min_oranges_in_new_box_l239_239002


namespace polar_eq_of_circle_product_of_distances_MA_MB_l239_239117

noncomputable def circle_center := (2, Real.pi / 3)
noncomputable def circle_radius := 2

-- Polar equation of the circle
theorem polar_eq_of_circle :
  ∀ (ρ θ : ℝ),
    (circle_center.snd = Real.pi / 3) →
    ρ = 2 * 2 * Real.cos (θ - circle_center.snd) → 
    ρ = 4 * Real.cos (θ - (Real.pi / 3)) :=
by 
  sorry

noncomputable def point_M := (1, -2)

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ := 
  (1 + 1/2 * t, -2 + Real.sqrt 3 / 2 * t)

noncomputable def cartesian_center := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
noncomputable def cartesian_radius := 2

-- Cartesian form of the circle equation from the polar coordinates
noncomputable def cartesian_eq (x y : ℝ) : Prop :=
  (x - cartesian_center.fst)^2 + (y - cartesian_center.snd)^2 = circle_radius^2

-- Product of distances |MA| * |MB|
theorem product_of_distances_MA_MB :
  ∃ (t1 t2 : ℝ),
  (∀ t, parametric_line t ∈ {p : ℝ × ℝ | cartesian_eq p.fst p.snd}) → 
  (point_M.fst, point_M.snd) = (1, -2) →
  t1 * t2 = 3 + 4 * Real.sqrt 3 :=
by
  sorry

end polar_eq_of_circle_product_of_distances_MA_MB_l239_239117


namespace sum_of_roots_l239_239354

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l239_239354


namespace range_of_a_l239_239565

noncomputable def p (a : ℝ) : Prop :=
∀ (x : ℝ), x > -1 → (x^2) / (x + 1) ≥ a

noncomputable def q (a : ℝ) : Prop :=
∃ (x : ℝ), (a*x^2 - a*x + 1 = 0)

theorem range_of_a (a : ℝ) :
  ¬ p a ∧ ¬ q a ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
by sorry

end range_of_a_l239_239565


namespace vectors_form_basis_l239_239330

-- Define the vectors in set B
def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (3, 7)

-- Define a function that checks if two vectors form a basis
def form_basis (v1 v2 : ℝ × ℝ) : Prop :=
  let det := v1.1 * v2.2 - v1.2 * v2.1
  det ≠ 0

-- State the theorem that vectors e1 and e2 form a basis
theorem vectors_form_basis : form_basis e1 e2 :=
by
  -- Add the proof here
  sorry

end vectors_form_basis_l239_239330


namespace part_a_part_b_l239_239312

theorem part_a (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℤ), a < m * α - n ∧ m * α - n < b :=
sorry

theorem part_b (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end part_a_part_b_l239_239312


namespace find_sin_E_floor_l239_239398

variable {EF GH EH FG : ℝ}
variable (E G : ℝ)

-- Conditions from the problem
def is_convex_quadrilateral (EF GH EH FG : ℝ) : Prop := true
def angles_congruent (E G : ℝ) : Prop := E = G
def sides_equal (EF GH : ℝ) : Prop := EF = GH ∧ EF = 200
def sides_not_equal (EH FG : ℝ) : Prop := EH ≠ FG
def perimeter (EF GH EH FG : ℝ) : Prop := EF + GH + EH + FG = 800

-- The theorem to be proved
theorem find_sin_E_floor (h_convex : is_convex_quadrilateral EF GH EH FG)
                         (h_angles : angles_congruent E G)
                         (h_sides : sides_equal EF GH)
                         (h_sides_ne : sides_not_equal EH FG)
                         (h_perimeter : perimeter EF GH EH FG) :
  ⌊ 1000 * Real.sin E ⌋ = 0 := by
  sorry

end find_sin_E_floor_l239_239398


namespace parallel_vectors_l239_239678

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors {m : ℝ} (h : (∃ k : ℝ, vector_a = k • vector_b m)) : m = -2 :=
by
  sorry

end parallel_vectors_l239_239678


namespace natural_numbers_divisible_by_7_between_200_400_l239_239087

theorem natural_numbers_divisible_by_7_between_200_400 : 
  { n : ℕ | 200 <= n ∧ n <= 400 ∧ n % 7 = 0 }.to_finset.card = 29 := 
  sorry

end natural_numbers_divisible_by_7_between_200_400_l239_239087


namespace robbery_proof_l239_239956

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l239_239956


namespace race_car_cost_l239_239202

variable (R : ℝ)
variable (Mater_cost SallyMcQueen_cost : ℝ)

-- Conditions
def Mater_cost_def : Mater_cost = 0.10 * R := by sorry
def SallyMcQueen_cost_def : SallyMcQueen_cost = 3 * Mater_cost := by sorry
def SallyMcQueen_cost_val : SallyMcQueen_cost = 42000 := by sorry

-- Theorem to prove the race car cost
theorem race_car_cost : R = 140000 :=
  by
    -- Use the conditions to prove
    sorry

end race_car_cost_l239_239202


namespace line_in_first_and_third_quadrants_l239_239714

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l239_239714


namespace alcohol_quantity_l239_239457

theorem alcohol_quantity (A W : ℕ) (h1 : 4 * W = 3 * A) (h2 : 4 * (W + 8) = 5 * A) : A = 16 := 
by
  sorry

end alcohol_quantity_l239_239457


namespace total_toys_is_60_l239_239735

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end total_toys_is_60_l239_239735


namespace problem_l239_239021

-- Define the main problem conditions
variables {a b c : ℝ}
axiom h1 : a^2 + b^2 + c^2 = 63
axiom h2 : 2 * a + 3 * b + 6 * c = 21 * Real.sqrt 7

-- Define the goal
theorem problem :
  (a / c) ^ (a / b) = (1 / 3) ^ (2 / 3) :=
sorry

end problem_l239_239021


namespace train_speed_l239_239324

theorem train_speed (length : ℝ) (time_seconds : ℝ) (speed : ℝ) :
  length = 320 → time_seconds = 16 → speed = 72 :=
by 
  sorry

end train_speed_l239_239324


namespace inequality_solution_l239_239667

theorem inequality_solution (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
    (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := by
  sorry

end inequality_solution_l239_239667


namespace symmetric_line_equation_l239_239764

def line_1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line_2 (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem symmetric_line_equation :
  ∀ x y : ℝ, line_1 x y → line_2 x y → symmetric_line x y := 
sorry

end symmetric_line_equation_l239_239764


namespace mrs_sheridan_total_cats_l239_239279

-- Definitions from the conditions
def original_cats : Nat := 17
def additional_cats : Nat := 14

-- The total number of cats is the sum of the original and additional cats
def total_cats : Nat := original_cats + additional_cats

-- Statement to prove
theorem mrs_sheridan_total_cats : total_cats = 31 := by
  sorry

end mrs_sheridan_total_cats_l239_239279


namespace local_maximum_at_neg2_l239_239283

noncomputable def y (x : ℝ) : ℝ :=
  (1/3) * x^3 - 4 * x + 4

theorem local_maximum_at_neg2 :
  ∃ x : ℝ, x = -2 ∧ 
           y x = 28/3 ∧
           (∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 2) < δ → y z < y (-2)) := by
  sorry

end local_maximum_at_neg2_l239_239283


namespace number_of_polynomials_satisfying_P_neg1_eq_neg12_l239_239817

noncomputable def count_polynomials_satisfying_condition : ℕ := 
  sorry

theorem number_of_polynomials_satisfying_P_neg1_eq_neg12 :
  count_polynomials_satisfying_condition = 455 := 
  sorry

end number_of_polynomials_satisfying_P_neg1_eq_neg12_l239_239817


namespace value_of_a_minus_b_l239_239259

theorem value_of_a_minus_b 
  (a b : ℤ)
  (h1 : 1010 * a + 1014 * b = 1018)
  (h2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 :=
sorry

end value_of_a_minus_b_l239_239259


namespace find_smallest_A_divisible_by_51_l239_239931

theorem find_smallest_A_divisible_by_51 :
  ∃ (x y : ℕ), (A = 1100 * x + 11 * y) ∧ 
    (0 ≤ x) ∧ (x ≤ 9) ∧ 
    (0 ≤ y) ∧ (y ≤ 9) ∧ 
    (A % 51 = 0) ∧ 
    (A = 1122) :=
sorry

end find_smallest_A_divisible_by_51_l239_239931


namespace area_of_rectangle_l239_239456

-- Definitions and conditions
def side_of_square : ℕ := 50
def radius_of_circle : ℕ := side_of_square
def length_of_rectangle : ℕ := (2 * radius_of_circle) / 5
def breadth_of_rectangle : ℕ := 10

-- Theorem statement
theorem area_of_rectangle :
  (length_of_rectangle * breadth_of_rectangle = 200) := by
  sorry

end area_of_rectangle_l239_239456


namespace max_value_of_function_l239_239381

theorem max_value_of_function (x : ℝ) (h : x < 5 / 4) :
    (∀ y, y = 4 * x - 2 + 1 / (4 * x - 5) → y ≤ 1):=
sorry

end max_value_of_function_l239_239381


namespace x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l239_239844

theorem x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq (x y : ℝ) :
  ¬((x > y) → (x^2 > y^2)) ∧ ¬((x^2 > y^2) → (x > y)) :=
by
  sorry

end x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l239_239844


namespace solve_equation_l239_239603

theorem solve_equation (x : ℝ) :
  x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3 * x + 2 > 0 ↔ x = -3 ∨ x = 0 ∨ x = 5 :=
by
  sorry

end solve_equation_l239_239603


namespace parabola_focus_distance_l239_239898

theorem parabola_focus_distance (p m : ℝ) (h1 : p > 0) (h2 : (2 - (-p/2)) = 4) : p = 4 := 
by
  sorry

end parabola_focus_distance_l239_239898


namespace cone_slant_height_l239_239029

theorem cone_slant_height (radius : ℝ) (theta : ℝ) (h_radius : radius = 6) (h_theta : theta = 240) : 
  let circumference_base := 2 * Real.pi * radius in
  let arc_length := (theta / 360) * 2 * Real.pi * slant_height in 
  12 * Real.pi = arc_length → 
  slant_height = 9 := sorry

end cone_slant_height_l239_239029


namespace trajectory_equation_l239_239593

variable (m x y : ℝ)
def a := (m * x, y + 1)
def b := (x, y - 1)
def is_perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

theorem trajectory_equation 
  (h1: is_perpendicular (a m x y) (b x y)) : 
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l239_239593


namespace problem_fixed_values_problem_arbitrary_values_l239_239742

noncomputable def minimal_value_fixed (x y z : ℝ) (m n p: ℝ) :=
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem problem_fixed_values (x y z m n p : ℝ) 
  (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) 
  (xyz_eq_8 : x * y * z = 8) (mnp_eq_8 : m * n * p = 8)
  (m_eq_2 : m = 2) (n_eq_2 : n = 2) (p_eq_2 : p = 2) :
  minimal_value_fixed x y z m n p = 36 :=
sorry

theorem problem_arbitrary_values (x y z m n p : ℝ) 
  (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) 
  (xyz_eq_8 : x * y * z = 8) (mnp_eq_8 : m * n * p = 8) :
  minimal_value_fixed x y z m n p = 
  6 * real.cbrt 2 * (real.cbrt (m^2) + real.cbrt (n^2) + real.cbrt (p^2)) :=
sorry

end problem_fixed_values_problem_arbitrary_values_l239_239742


namespace sarah_more_than_cecily_l239_239003

theorem sarah_more_than_cecily (t : ℕ) (ht : t = 144) :
  let s := (1 / 3 : ℚ) * t
  let a := (3 / 8 : ℚ) * t
  let c := t - (s + a)
  s - c = 6 := by
  sorry

end sarah_more_than_cecily_l239_239003


namespace vitya_catchup_time_l239_239487

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l239_239487


namespace anna_bought_five_chocolate_bars_l239_239532

noncomputable section

def initial_amount : ℝ := 10
def price_chewing_gum : ℝ := 1
def price_candy_cane : ℝ := 0.5
def remaining_amount : ℝ := 1

def chewing_gum_cost : ℝ := 3 * price_chewing_gum
def candy_cane_cost : ℝ := 2 * price_candy_cane

def total_spent : ℝ := initial_amount - remaining_amount
def known_items_cost : ℝ := chewing_gum_cost + candy_cane_cost
def chocolate_bars_cost : ℝ := total_spent - known_items_cost
def price_chocolate_bar : ℝ := 1

def chocolate_bars_bought : ℝ := chocolate_bars_cost / price_chocolate_bar

theorem anna_bought_five_chocolate_bars : chocolate_bars_bought = 5 := 
by
  sorry

end anna_bought_five_chocolate_bars_l239_239532


namespace slab_length_l239_239189

noncomputable def area_of_one_slab (total_area: ℝ) (num_slabs: ℕ) : ℝ :=
  total_area / num_slabs

noncomputable def length_of_one_slab (slab_area : ℝ) : ℝ :=
  Real.sqrt slab_area

theorem slab_length (total_area : ℝ) (num_slabs : ℕ)
  (h_total_area : total_area = 98)
  (h_num_slabs : num_slabs = 50) :
  length_of_one_slab (area_of_one_slab total_area num_slabs) = 1.4 :=
by
  sorry

end slab_length_l239_239189


namespace color_change_probability_l239_239038

-- Definitions based directly on conditions in a)
def light_cycle_duration := 93
def change_intervals_duration := 15
def expected_probability := 5 / 31

-- The Lean 4 statement for the proof problem
theorem color_change_probability :
  (change_intervals_duration / light_cycle_duration) = expected_probability :=
by
  sorry

end color_change_probability_l239_239038


namespace revised_lemonade_calories_l239_239371

def lemonade (lemon_grams sugar_grams water_grams lemon_calories_per_50grams sugar_calories_per_100grams : ℕ) :=
  let lemon_cals := lemon_calories_per_50grams
  let sugar_cals := (sugar_grams / 100) * sugar_calories_per_100grams
  let water_cals := 0
  lemon_cals + sugar_cals + water_cals

def lemonade_weight (lemon_grams sugar_grams water_grams : ℕ) :=
  lemon_grams + sugar_grams + water_grams

def caloric_density (total_calories : ℕ) (total_weight : ℕ) := (total_calories : ℚ) / total_weight

def calories_in_serving (density : ℚ) (serving : ℕ) := density * serving

theorem revised_lemonade_calories :
  let lemon_calories := 32
  let sugar_calories := 579
  let total_calories := lemonade 50 150 300 lemon_calories sugar_calories
  let total_weight := lemonade_weight 50 150 300
  let density := caloric_density total_calories total_weight
  let serving_calories := calories_in_serving density 250
  serving_calories = 305.5 := sorry

end revised_lemonade_calories_l239_239371


namespace quadratic_real_roots_exists_l239_239670

theorem quadratic_real_roots_exists :
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 * x1 - 6 * x1 + 8 = 0) ∧ (x2 * x2 - 6 * x2 + 8 = 0) :=
by
  sorry

end quadratic_real_roots_exists_l239_239670


namespace sum_of_roots_of_poly_eq_14_over_3_l239_239356

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l239_239356


namespace area_of_triangle_BP_Q_is_24_l239_239032

open Real

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem area_of_triangle_BP_Q_is_24
  (A B C P H Q : ℝ × ℝ)
  (h_triangle_ABC_right : C.1 = 0 ∧ C.2 = 0 ∧ B.2 = 0 ∧ A.2 ≠ 0)
  (h_BC_diameter : distance B C = 26)
  (h_tangent_AP : distance P B = distance P C ∧ P ≠ C)
  (h_PH_perpendicular_BC : P.1 = H.1 ∧ H.2 = 0)
  (h_PH_intersects_AB_at_Q : H.1 = Q.1 ∧ Q.2 ≠ 0)
  (h_BH_CH_ratio : 4 * distance B H = 9 * distance C H)
  : triangle_area B P Q = 24 :=
sorry

end area_of_triangle_BP_Q_is_24_l239_239032


namespace vitya_catches_up_in_5_minutes_l239_239495

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l239_239495


namespace solution_set_circle_l239_239993

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l239_239993


namespace probability_divisible_by_five_l239_239467

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l239_239467


namespace sum_of_roots_l239_239357

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l239_239357


namespace home_electronics_budget_l239_239637

theorem home_electronics_budget (deg_ba: ℝ) (b_deg: ℝ) (perc_me: ℝ) (perc_fa: ℝ) (perc_gm: ℝ) (perc_il: ℝ) : 
  deg_ba = 43.2 → 
  b_deg = 360 → 
  perc_me = 12 →
  perc_fa = 15 →
  perc_gm = 29 →
  perc_il = 8 →
  (b_deg / 360 * 100 = 12) → 
  perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100) = 76 →
  100 - (perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100)) = 24 :=
by
  intro h_deg_ba h_b_deg h_perc_me h_perc_fa h_perc_gm h_perc_il h_ba_12perc h_total_76perc
  sorry

end home_electronics_budget_l239_239637


namespace trader_profit_percent_equal_eight_l239_239037

-- Defining the initial conditions
def original_price (P : ℝ) := P
def purchased_price (P : ℝ) := 0.60 * original_price P
def selling_price (P : ℝ) := 1.80 * purchased_price P

-- Statement to be proved
theorem trader_profit_percent_equal_eight (P : ℝ) (h : P > 0) :
  ((selling_price P - original_price P) / original_price P) * 100 = 8 :=
by
  sorry

end trader_profit_percent_equal_eight_l239_239037


namespace length_of_each_part_l239_239177

theorem length_of_each_part (ft : ℕ) (inch : ℕ) (parts : ℕ) (total_length : ℕ) (part_length : ℕ) :
  ft = 6 → inch = 8 → parts = 5 → total_length = 12 * ft + inch → part_length = total_length / parts → part_length = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_each_part_l239_239177


namespace similar_triangles_x_value_l239_239856

theorem similar_triangles_x_value
  (x : ℝ)
  (h_similar : ∀ (AB BC DE EF : ℝ), AB / BC = DE / EF)
  (h_AB : AB = x)
  (h_BC : BC = 33)
  (h_DE : DE = 96)
  (h_EF : EF = 24) :
  x = 132 :=
by
  -- Proof steps will be here
  sorry

end similar_triangles_x_value_l239_239856


namespace Vitya_catches_mother_l239_239499

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l239_239499


namespace Billy_has_10_fish_l239_239537

def Billy_has_fish (Bobby Sarah Tony Billy : ℕ) : Prop :=
  Bobby = 2 * Sarah ∧
  Sarah = Tony + 5 ∧
  Tony = 3 * Billy ∧
  Bobby + Sarah + Tony + Billy = 145

theorem Billy_has_10_fish : ∃ (Billy : ℕ), Billy_has_fish (2 * (3 * Billy + 5)) (3 * Billy + 5) (3 * Billy) Billy ∧ Billy = 10 :=
by
  sorry

end Billy_has_10_fish_l239_239537


namespace robbery_proof_l239_239955

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l239_239955


namespace number_of_students_earning_B_l239_239582

variables (a b c : ℕ) -- since we assume we only deal with whole numbers

-- Given conditions:
-- 1. The probability of earning an A is twice the probability of earning a B.
axiom h1 : a = 2 * b
-- 2. The probability of earning a C is equal to the probability of earning a B.
axiom h2 : c = b
-- 3. The only grades are A, B, or C and there are 45 students in the class.
axiom h3 : a + b + c = 45

-- Prove that the number of students earning a B is 11.
theorem number_of_students_earning_B : b = 11 :=
by
    sorry

end number_of_students_earning_B_l239_239582


namespace least_multiple_of_15_greater_than_500_l239_239928

theorem least_multiple_of_15_greater_than_500 : 
  ∃ (n : ℕ), n > 500 ∧ (∃ (k : ℕ), n = 15 * k) ∧ (n = 510) :=
by
  sorry

end least_multiple_of_15_greater_than_500_l239_239928


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l239_239722

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l239_239722


namespace smallest_positive_integer_l239_239508

theorem smallest_positive_integer (n : ℕ) : 13 * n ≡ 567 [MOD 5] ↔ n = 4 := by
  sorry

end smallest_positive_integer_l239_239508


namespace moores_law_transistors_l239_239206

-- Define the initial conditions
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2 -- in years
def transistors_doubling (n : ℕ) : ℕ := initial_transistors * 2^n

-- Calculate the number of doubling events from 1995 to 2010
def years_spanned : ℕ := 15
def number_of_doublings : ℕ := years_spanned / doubling_period

-- Expected number of transistors in 2010
def expected_transistors_in_2010 : ℕ := 64000000

theorem moores_law_transistors :
  transistors_doubling number_of_doublings = expected_transistors_in_2010 :=
sorry

end moores_law_transistors_l239_239206


namespace mary_needs_more_sugar_l239_239748

theorem mary_needs_more_sugar 
  (sugar_needed flour_needed salt_needed already_added_flour : ℕ)
  (h1 : sugar_needed = 11)
  (h2 : flour_needed = 6)
  (h3 : salt_needed = 9)
  (h4 : already_added_flour = 12) :
  (sugar_needed - salt_needed) = 2 :=
by
  sorry

end mary_needs_more_sugar_l239_239748


namespace possible_values_of_k_l239_239703

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l239_239703


namespace zoo_problem_l239_239052

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l239_239052


namespace cricket_throwers_l239_239133

theorem cricket_throwers (T L R : ℕ) 
  (h1 : T + L + R = 55)
  (h2 : T + R = 49) 
  (h3 : L = (1/3) * (L + R))
  (h4 : R = (2/3) * (L + R)) :
  T = 37 :=
by sorry

end cricket_throwers_l239_239133


namespace total_beakers_count_l239_239641

variable (total_beakers_with_ions : ℕ) 
variable (drops_per_test : ℕ)
variable (total_drops_used : ℕ) 
variable (beakers_without_ions : ℕ)

theorem total_beakers_count
  (h1 : total_beakers_with_ions = 8)
  (h2 : drops_per_test = 3)
  (h3 : total_drops_used = 45)
  (h4 : beakers_without_ions = 7) : 
  (total_drops_used / drops_per_test) = (total_beakers_with_ions + beakers_without_ions) :=
by
  -- Proof to be filled in
  sorry

end total_beakers_count_l239_239641


namespace minimum_value_l239_239743

open Real

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 9) :
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z) ≥ 9 :=
by sorry

end minimum_value_l239_239743


namespace joe_lowest_score_dropped_l239_239018

theorem joe_lowest_score_dropped (A B C D : ℕ) 
  (h1 : A + B + C + D = 160)
  (h2 : A + B + C = 135) 
  (h3 : D ≤ A ∧ D ≤ B ∧ D ≤ C) :
  D = 25 :=
sorry

end joe_lowest_score_dropped_l239_239018


namespace Vitya_catches_mother_l239_239498

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l239_239498


namespace find_n_l239_239580

theorem find_n (n : ℕ) : 
  (1/5 : ℝ)^35 * (1/4 : ℝ)^n = (1 : ℝ) / (2 * 10^35) → n = 18 :=
by
  intro h
  sorry

end find_n_l239_239580


namespace factorize_expression_l239_239365

theorem factorize_expression (a b : ℝ) : 3 * a ^ 2 - 3 * b ^ 2 = 3 * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l239_239365


namespace midpoint_of_segment_l239_239505

def A : ℝ × ℝ × ℝ := (10, -3, 5)
def B : ℝ × ℝ × ℝ := (-2, 7, -4)

theorem midpoint_of_segment :
  let M_x := (10 + -2 : ℝ) / 2
  let M_y := (-3 + 7 : ℝ) / 2
  let M_z := (5 + -4 : ℝ) / 2
  (M_x, M_y, M_z) = (4, 2, 0.5) :=
by
  let M_x : ℝ := (10 + -2) / 2
  let M_y : ℝ := (-3 + 7) / 2
  let M_z : ℝ := (5 + -4) / 2
  show (M_x, M_y, M_z) = (4, 2, 0.5)
  repeat { sorry }

end midpoint_of_segment_l239_239505


namespace greening_investment_equation_l239_239194

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end greening_investment_equation_l239_239194


namespace investment_months_l239_239798

theorem investment_months (i_a i_b i_c a_gain total_gain : ℝ) (m : ℝ) :
  i_a = 1 ∧ i_b = 2 * i_a ∧ i_c = 3 * i_a ∧ a_gain = 6100 ∧ total_gain = 18300 ∧ m * i_b * (12 - m) + i_c * 3 * 4 = 12200 →
  a_gain / total_gain = i_a * 12 / (i_a * 12 + i_b * (12 - m) + i_c * 4) → m = 6 :=
by
  intros h1 h2
  obtain ⟨ha, hb, hc, hag, htg, h⟩ := h1
  -- proof omitted
  sorry

end investment_months_l239_239798


namespace three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l239_239336

def three_digit_odd_nums (digits : Finset ℕ) : ℕ :=
  let odd_digits := digits.filter (λ n => n % 2 = 1)
  let num_choices_for_units_place := odd_digits.card
  let remaining_digits := digits \ odd_digits
  let num_choices_for_hundreds_tens_places := remaining_digits.card * (remaining_digits.card - 1)
  num_choices_for_units_place * num_choices_for_hundreds_tens_places

theorem three_digit_odd_nums_using_1_2_3_4_5_without_repetition :
  three_digit_odd_nums {1, 2, 3, 4, 5} = 36 :=
by
  -- Proof is skipped
  sorry

end three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l239_239336


namespace k_positive_first_third_quadrants_l239_239710

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l239_239710


namespace cannot_be_value_of_A_plus_P_l239_239291

theorem cannot_be_value_of_A_plus_P (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (a_neq_b: a ≠ b) :
  let A : ℕ := a * b
  let P : ℕ := 2 * a + 2 * b
  A + P ≠ 102 :=
by
  sorry

end cannot_be_value_of_A_plus_P_l239_239291


namespace find_integer_n_l239_239009

theorem find_integer_n (n : ℕ) (hn1 : 0 ≤ n) (hn2 : n < 102) (hmod : 99 * n % 102 = 73) : n = 97 :=
  sorry

end find_integer_n_l239_239009


namespace least_positive_integer_exists_l239_239011

theorem least_positive_integer_exists 
  (exists_k : ∃ k, (1 ≤ k ∧ k ≤ 2 * 5) ∧ (5^2 - 5 + k) % k = 0)
  (not_all_k : ¬(∀ k, (1 ≤ k ∧ k ≤ 2 * 5) → (5^2 - 5 + k) % k = 0)) :
  5 = 5 := 
by
  trivial

end least_positive_integer_exists_l239_239011


namespace valid_license_plates_count_l239_239651

/--
The problem is to prove that the total number of valid license plates under the given format is equal to 45,697,600.
The given conditions are:
1. A valid license plate in Xanadu consists of three letters followed by two digits, and then one more letter at the end.
2. There are 26 choices of letters for each letter spot.
3. There are 10 choices of digits for each digit spot.

We need to conclude that the number of possible license plates is:
26^4 * 10^2 = 45,697,600.
-/

def num_valid_license_plates : Nat :=
  let letter_choices := 26
  let digit_choices := 10
  let total_choices := letter_choices ^ 3 * digit_choices ^ 2 * letter_choices
  total_choices

theorem valid_license_plates_count : num_valid_license_plates = 45697600 := by
  sorry

end valid_license_plates_count_l239_239651


namespace daniel_practices_total_minutes_in_week_l239_239819

theorem daniel_practices_total_minutes_in_week :
  let school_minutes_per_day := 15
  let school_days := 5
  let weekend_minutes_per_day := 2 * school_minutes_per_day
  let weekend_days := 2
  let total_school_week_minutes := school_minutes_per_day * school_days
  let total_weekend_minutes := weekend_minutes_per_day * weekend_days
  total_school_week_minutes + total_weekend_minutes = 135 :=
by
  sorry

end daniel_practices_total_minutes_in_week_l239_239819


namespace number_of_saturday_sales_l239_239841

def caricatures_sold_on_saturday (total_earnings weekend_earnings price_per_drawing sunday_sales : ℕ) : ℕ :=
  (total_earnings - (sunday_sales * price_per_drawing)) / price_per_drawing

theorem number_of_saturday_sales : caricatures_sold_on_saturday 800 800 20 16 = 24 := 
by 
  sorry

end number_of_saturday_sales_l239_239841


namespace take_home_pay_l239_239418

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l239_239418


namespace fib_fact_last_two_sum_is_five_l239_239930

def fib_fact_last_two_sum (s : List (Fin 100)) : Fin 100 :=
  s.sum

theorem fib_fact_last_two_sum_is_five :
  fib_fact_last_two_sum [1, 1, 2, 6, 20, 20, 0] = 5 :=
by 
  sorry

end fib_fact_last_two_sum_is_five_l239_239930


namespace cream_ratio_l239_239739

noncomputable def joe_coffee_initial := 14
noncomputable def joe_coffee_drank := 3
noncomputable def joe_cream_added := 3

noncomputable def joann_coffee_initial := 14
noncomputable def joann_cream_added := 3
noncomputable def joann_mixture_stirred := 17
noncomputable def joann_amount_drank := 3

theorem cream_ratio (joe_coffee_initial joe_coffee_drank joe_cream_added 
                     joann_coffee_initial joann_cream_added joann_mixture_stirred 
                     joann_amount_drank : ℝ) : 
  (joe_coffee_initial - joe_coffee_drank + joe_cream_added) / 
  (joann_cream_added - (joann_amount_drank * (joann_cream_added / joann_mixture_stirred))) = 17 / 14 :=
by
  -- Prove the theorem statement
  sorry

end cream_ratio_l239_239739


namespace find_integer_solutions_l239_239588

-- Axiom stating that p is prime
axiom is_prime (p : ℕ) : Prop

theorem find_integer_solutions (n k p : ℕ) (hn : Int) (hk : Nat) (hp : Nat) (is_prime p) :
  (|6 * hn^2 - 17 * hn - 39| = p^k) ↔ ((hn, hp, hk) ∈ [(-1, 2, 4), (-2, 19, 1), (4, 11, 1), (2, 7, 2), (-4, 5, 3)]) :=
sorry

end find_integer_solutions_l239_239588


namespace problem1_problem2_min_value_l239_239938

theorem problem1 (x : ℝ) : |x + 1| + |x - 2| ≥ 3 := sorry

theorem problem2 (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 14 := sorry

theorem min_value (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) :
  ∃ x y z, x^2 + y^2 + z^2 = 1 / 14 := sorry

end problem1_problem2_min_value_l239_239938


namespace weight_of_each_bag_is_7_l239_239319

-- Defining the conditions
def morning_bags : ℕ := 29
def afternoon_bags : ℕ := 17
def total_weight : ℕ := 322

-- Defining the question in terms of proving a specific weight per bag
def bags_sold := morning_bags + afternoon_bags
def weight_per_bag (w : ℕ) := total_weight = bags_sold * w

-- Proving the question == answer under the given conditions
theorem weight_of_each_bag_is_7 :
  ∃ w : ℕ, weight_per_bag w ∧ w = 7 :=
by
  sorry

end weight_of_each_bag_is_7_l239_239319


namespace max_a_condition_l239_239693

theorem max_a_condition (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) → a = -1 :=
by
  sorry

end max_a_condition_l239_239693


namespace binomial_n_choose_n_sub_2_l239_239504

theorem binomial_n_choose_n_sub_2 (n : ℕ) (h : 2 ≤ n) : Nat.choose n (n - 2) = n * (n - 1) / 2 :=
by
  sorry

end binomial_n_choose_n_sub_2_l239_239504


namespace group_size_systematic_sampling_l239_239967

-- Define the total number of viewers
def total_viewers : ℕ := 10000

-- Define the number of viewers to be selected
def selected_viewers : ℕ := 10

-- Lean statement to prove the group size for systematic sampling
theorem group_size_systematic_sampling (n_total n_selected : ℕ) : n_total = total_viewers → n_selected = selected_viewers → (n_total / n_selected) = 1000 :=
by
  intros h_total h_selected
  rw [h_total, h_selected]
  sorry

end group_size_systematic_sampling_l239_239967


namespace max_collection_l239_239623

theorem max_collection : 
  let Yoongi := 4 
  let Jungkook := 6 / 3 
  let Yuna := 5 
  max Yoongi (max Jungkook Yuna) = 5 :=
by 
  let Yoongi := 4
  let Jungkook := (6 / 3) 
  let Yuna := 5
  show max Yoongi (max Jungkook Yuna) = 5
  sorry

end max_collection_l239_239623


namespace smallest_positive_period_pi_interval_extrema_l239_239081

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3

theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem interval_extrema :
  ∃ x_max x_min : ℝ, 
  -Real.pi / 4 ≤ x_max ∧ x_max ≤ Real.pi / 6 ∧ f x_max = 2 ∧
  -Real.pi / 4 ≤ x_min ∧ x_min ≤ Real.pi / 6 ∧ f x_min = -1 ∧ 
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 2 ∧ f x ≥ -1) :=
sorry

end smallest_positive_period_pi_interval_extrema_l239_239081


namespace probability_divisible_by_5_l239_239464

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l239_239464


namespace cos_seven_pi_over_six_l239_239807

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l239_239807


namespace range_of_reciprocal_sum_l239_239260

theorem range_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
    4 ≤ (1/x + 1/y) :=
by
  sorry

end range_of_reciprocal_sum_l239_239260


namespace maximum_tied_teams_round_robin_l239_239107

noncomputable def round_robin_tournament_max_tied_teams (n : ℕ) : ℕ := 
  sorry

theorem maximum_tied_teams_round_robin (h : n = 8) : round_robin_tournament_max_tied_teams n = 7 :=
sorry

end maximum_tied_teams_round_robin_l239_239107


namespace ellipse_foci_y_axis_range_l239_239681

theorem ellipse_foci_y_axis_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1) ↔ (m < -1 ∨ (1 < m ∧ m < 3 / 2)) :=
sorry

end ellipse_foci_y_axis_range_l239_239681


namespace range_of_m_satisfies_inequality_l239_239848

theorem range_of_m_satisfies_inequality (m : ℝ) :
  ((∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) ↔ (m ≤ -1 ∨ m > 5/3)) :=
sorry

end range_of_m_satisfies_inequality_l239_239848


namespace best_possible_overall_standing_l239_239270

noncomputable def N : ℕ := 100 -- number of participants
noncomputable def M : ℕ := 14  -- number of stages

-- Define a competitor finishing 93rd in each stage
def finishes_93rd_each_stage (finishes : ℕ → ℕ) : Prop :=
  ∀ i, i < M → finishes i = 93

-- Define the best possible overall standing
theorem best_possible_overall_standing
  (finishes : ℕ → ℕ) -- function representing stage finishes for the competitor
  (h : finishes_93rd_each_stage finishes) :
  ∃ k, k = 2 := 
sorry

end best_possible_overall_standing_l239_239270


namespace total_pizza_order_cost_l239_239885

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end total_pizza_order_cost_l239_239885


namespace find_triplets_l239_239827

noncomputable def triplets_solution (x y z : ℝ) : Prop := 
  (x^2 + y^2 = -x + 3*y + z) ∧ 
  (y^2 + z^2 = x + 3*y - z) ∧ 
  (x^2 + z^2 = 2*x + 2*y - z) ∧ 
  (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

theorem find_triplets : 
  { (x, y, z) : ℝ × ℝ × ℝ | triplets_solution x y z } = 
  { (0, 1, -2), (-3/2, 5/2, -1/2) } :=
sorry

end find_triplets_l239_239827


namespace tan_theta_value_l239_239695

theorem tan_theta_value (θ k : ℝ) 
  (h1 : Real.sin θ = (k + 1) / (k - 3)) 
  (h2 : Real.cos θ = (k - 1) / (k - 3)) 
  (h3 : (Real.sin θ ≠ 0) ∧ (Real.cos θ ≠ 0)) : 
  Real.tan θ = 3 / 4 := 
sorry

end tan_theta_value_l239_239695


namespace find_b_plus_m_l239_239169

def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 7
def line2 (b : ℝ) (x : ℝ) : ℝ := 4 * x + b

theorem find_b_plus_m :
  ∃ (m b : ℝ), line1 m 8 = 11 ∧ line2 b 8 = 11 ∧ b + m = -20.5 :=
sorry

end find_b_plus_m_l239_239169


namespace probability_divisible_by_5_l239_239463

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l239_239463


namespace factorize_expression_l239_239061

theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x^2 + 8 * x = 2 * x * (x - 2) ^ 2 := 
sorry

end factorize_expression_l239_239061


namespace probability_even_sum_l239_239979

-- Defining the probabilities for the first wheel
def P_even_1 : ℚ := 2/3
def P_odd_1 : ℚ := 1/3

-- Defining the probabilities for the second wheel
def P_even_2 : ℚ := 1/2
def P_odd_2 : ℚ := 1/2

-- Prove that the probability that the sum of the two selected numbers is even is 1/2
theorem probability_even_sum : 
  P_even_1 * P_even_2 + P_odd_1 * P_odd_2 = 1/2 :=
by
  sorry

end probability_even_sum_l239_239979


namespace green_caps_percentage_l239_239329

variable (total_caps : ℕ) (red_caps : ℕ)

def green_caps (total_caps red_caps: ℕ) : ℕ :=
  total_caps - red_caps

def percentage_of_green_caps (total_caps green_caps: ℕ) : ℕ :=
  (green_caps * 100) / total_caps

theorem green_caps_percentage :
  (total_caps = 125) →
  (red_caps = 50) →
  percentage_of_green_caps total_caps (green_caps total_caps red_caps) = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  exact sorry  -- The proof is omitted 

end green_caps_percentage_l239_239329


namespace geometric_sequence_condition_l239_239729

theorem geometric_sequence_condition (a : ℕ → ℝ) :
  (∀ n ≥ 2, a n = 2 * a (n-1)) → 
  (∃ r, r = 2 ∧ ∀ n ≥ 2, a n = r * a (n-1)) ∧ 
  (∃ b, b ≠ 0 ∧ ∀ n, a n = 0) :=
sorry

end geometric_sequence_condition_l239_239729


namespace simplify_expansion_l239_239060

-- Define the variables and expressions
variable (x : ℝ)

-- The main statement
theorem simplify_expansion : (x + 5) * (4 * x - 12) = 4 * x^2 + 8 * x - 60 :=
by sorry

end simplify_expansion_l239_239060


namespace possible_values_of_k_l239_239704

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l239_239704


namespace each_child_plays_equally_l239_239475

theorem each_child_plays_equally (total_time : ℕ) (num_children : ℕ)
  (play_group_size : ℕ) (play_time : ℕ) :
  num_children = 6 ∧ play_group_size = 3 ∧ total_time = 120 ∧ play_time = (total_time * play_group_size) / num_children →
  play_time = 60 :=
by
  intros h
  sorry

end each_child_plays_equally_l239_239475


namespace find_views_multiplier_l239_239208

theorem find_views_multiplier (M: ℝ) (h: 4000 * M + 50000 = 94000) : M = 11 :=
by
  sorry

end find_views_multiplier_l239_239208


namespace line_through_two_quadrants_l239_239715

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l239_239715


namespace find_y_l239_239727

-- Definitions of the given conditions
def is_straight_line (A B : Point) : Prop := 
  ∃ C D, A ≠ C ∧ B ≠ D

def angle (A B C : Point) : ℝ := sorry -- Assume angle is a function providing the angle in degrees

-- The proof problem statement
theorem find_y
  (A B C D X Y Z : Point)
  (hAB : is_straight_line A B)
  (hCD : is_straight_line C D)
  (hAXB : angle A X B = 180) 
  (hYXZ : angle Y X Z = 70)
  (hCYX : angle C Y X = 110) :
  angle X Y Z = 40 :=
sorry

end find_y_l239_239727


namespace percentage_of_annual_decrease_is_10_l239_239448

-- Define the present population and future population
def P_present : ℕ := 500
def P_future : ℕ := 450 

-- Calculate the percentage decrease
def percentage_decrease (P_present P_future : ℕ) : ℕ :=
  ((P_present - P_future) * 100) / P_present

-- Lean statement to prove the percentage decrease is 10%
theorem percentage_of_annual_decrease_is_10 :
  percentage_decrease P_present P_future = 10 :=
by
  unfold percentage_decrease
  sorry

end percentage_of_annual_decrease_is_10_l239_239448


namespace problem_solution_l239_239161

def boys_A := 5
def girls_A := 3
def boys_B := 6
def girls_B := 2

noncomputable def selected_ways (boys_A girls_A boys_B girls_B : ℕ) : ℕ :=
  (nat.choose girls_A 1) * (nat.choose boys_A 1) * (nat.choose boys_B 2) + 
  (nat.choose boys_A 2) * (nat.choose girls_B 1) * (nat.choose boys_B 1)

theorem problem_solution : selected_ways boys_A girls_A boys_B girls_B = 345 := by
  sorry

end problem_solution_l239_239161


namespace max_cos_sin_volume_of_solid_l239_239587

noncomputable def f (x : ℝ) : ℝ := 
  Real.Analysis.Limit1 (cos x ^ n + sin x ^ n) ^ (1 / n)

theorem max_cos_sin {x : ℝ} (hx : 0 ≤ x ∧ x ≤ π / 2) : 
  f(x) = max (cos x) (sin x) := by
  sorry

theorem volume_of_solid {a b : ℝ} (ha : a = sqrt 2 / 2) (hb : b = 1) :
  2 * π * (∫ (y : ℝ) in ha..hb, Real.arc_cos y * y) + 
  2 * π * (∫ (y : ℝ) in 0..ha, Real.arc_sin y * y) := by
  sorry

end max_cos_sin_volume_of_solid_l239_239587


namespace mindy_earns_k_times_more_than_mork_l239_239131

-- Given the following conditions:
-- Mork's tax rate: 0.45
-- Mindy's tax rate: 0.25
-- Combined tax rate: 0.29
-- Mindy earns k times more than Mork

theorem mindy_earns_k_times_more_than_mork (M : ℝ) (k : ℝ) (hM : M > 0) :
  (0.45 * M + 0.25 * k * M) / (M * (1 + k)) = 0.29 → k = 4 :=
by
  sorry

end mindy_earns_k_times_more_than_mork_l239_239131


namespace total_limes_picked_l239_239831

def Fred_limes : ℕ := 36
def Alyssa_limes : ℕ := 32
def Nancy_limes : ℕ := 35
def David_limes : ℕ := 42
def Eileen_limes : ℕ := 50

theorem total_limes_picked :
  Fred_limes + Alyssa_limes + Nancy_limes + David_limes + Eileen_limes = 195 :=
by
  sorry

end total_limes_picked_l239_239831


namespace ellipse_foci_coordinates_l239_239451

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
    x^2 / 16 + y^2 / 25 = 1 → (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3) :=
by
  sorry

end ellipse_foci_coordinates_l239_239451


namespace oranges_sold_in_the_morning_eq_30_l239_239301

variable (O : ℝ)  -- Denote the number of oranges Wendy sold in the morning

-- Conditions as assumptions
def price_per_apple : ℝ := 1.5
def price_per_orange : ℝ := 1
def morning_apples_sold : ℝ := 40
def afternoon_apples_sold : ℝ := 50
def afternoon_oranges_sold : ℝ := 40
def total_sales_for_day : ℝ := 205

-- Prove that O, satisfying the given conditions, equals 30
theorem oranges_sold_in_the_morning_eq_30 (h : 
    (morning_apples_sold * price_per_apple) +
    (O * price_per_orange) +
    (afternoon_apples_sold * price_per_apple) +
    (afternoon_oranges_sold * price_per_orange) = 
    total_sales_for_day
  ) : O = 30 :=
by
  sorry

end oranges_sold_in_the_morning_eq_30_l239_239301


namespace find_geometric_sequence_values_l239_239383

theorem find_geometric_sequence_values :
  ∃ (a b c : ℤ), (∃ q : ℤ, q ≠ 0 ∧ 2 * q ^ 4 = 32 ∧ a = 2 * q ∧ b = 2 * q ^ 2 ∧ c = 2 * q ^ 3)
                 ↔ ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end find_geometric_sequence_values_l239_239383


namespace probability_is_pi_over_32_l239_239947

open Set MeasureTheory ProbabilityTheory

-- Define the rectangle as a set in ℝ²
def rectangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the event as a set in ℝ² where x² + y² < y
def event : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 < p.2}

-- The measure of the rectangle is the Lebesgue measure in 2 dimensions
def rect_measure := volume (rectangle : Set (ℝ × ℝ))

-- The measure of the event in the rectangle
def event_measure := volume (event ∩ rectangle)

-- Probability that x² + y² < y given (x, y) is randomly picked from the rectangle
def prob : ℝ := event_measure / rect_measure

theorem probability_is_pi_over_32 : prob = π / 32 :=
by
  sorry

end probability_is_pi_over_32_l239_239947


namespace sum_consecutive_numbers_last_digit_diff_l239_239969

theorem sum_consecutive_numbers_last_digit_diff (a : ℕ) : 
    (2015 * (a + 1007) % 10) ≠ (2019 * (a + 3024) % 10) := 
by 
  sorry

end sum_consecutive_numbers_last_digit_diff_l239_239969


namespace seats_needed_l239_239000

def flute_players : ℕ := 5
def trumpet_players : ℕ := 3 * flute_players
def trombone_players : ℕ := trumpet_players - 8
def drummers : ℕ := trombone_players + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players : ℕ := trombone_players + 3
def total_seats_needed : ℕ := flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players

theorem seats_needed (s : ℕ) (h : s = 65) : total_seats_needed = s :=
by {
  have h_flutes : flute_players = 5 := rfl,
  have h_trumpets : trumpet_players = 3 * flute_players := rfl,
  have h_trombones : trombone_players = trumpet_players - 8 := rfl,
  have h_drums : drummers = trombone_players + 11 := rfl,
  have h_clarinets : clarinet_players = 2 * flute_players := rfl,
  have h_french_horns : french_horn_players = trombone_players + 3 := rfl,
  have h_total : total_seats_needed = flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players := rfl,
  rw [h_flutes, h_trumpets, h_trombones, h_drums, h_clarinets, h_french_horns] at h_total,
  simp only [flute_players, trumpet_players, trombone_players, drummers, clarinet_players, french_horn_players] at h_total,
  norm_num at h_total,
  exact h,
}

end seats_needed_l239_239000


namespace examine_points_l239_239044

variable (Bryan Jen Sammy mistakes : ℕ)

def problem_conditions : Prop :=
  Bryan = 20 ∧ Jen = Bryan + 10 ∧ Sammy = Jen - 2 ∧ mistakes = 7

theorem examine_points (h : problem_conditions Bryan Jen Sammy mistakes) : ∃ total_points : ℕ, total_points = Sammy + mistakes :=
by {
  sorry
}

end examine_points_l239_239044


namespace dale_pasta_l239_239662

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end dale_pasta_l239_239662


namespace condition_necessary_but_not_sufficient_l239_239099

theorem condition_necessary_but_not_sufficient (a : ℝ) :
  ((1 / a > 1) → (a < 1)) ∧ (∃ (a : ℝ), a < 1 ∧ 1 / a < 1) :=
by
  sorry

end condition_necessary_but_not_sufficient_l239_239099


namespace system_linear_eq_sum_l239_239688

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end system_linear_eq_sum_l239_239688


namespace min_value_f_l239_239811

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem min_value_f (h : ∀ x > 10, f x ≥ 40) : ∀ x > 10, f x = 40 → x = 20 :=
by
  sorry

end min_value_f_l239_239811


namespace find_point_C_l239_239149

-- Definitions of the conditions
def line_eq (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def on_parabola (C : ℝ × ℝ) : Prop := parabola_eq C.1 C.2
def perpendicular_at_C (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Points A and B satisfy both the line and parabola equations
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_eq A.1 A.2 ∧ parabola_eq A.1 A.2 ∧
  line_eq B.1 B.2 ∧ parabola_eq B.1 B.2

-- Statement to be proven
theorem find_point_C (A B : ℝ × ℝ) (hA : intersection_points A B) :
  ∃ C : ℝ × ℝ, on_parabola C ∧ perpendicular_at_C A B C ∧
    (C = (1, -2) ∨ C = (9, -6)) :=
by
  sorry

end find_point_C_l239_239149


namespace smaller_cube_volume_is_correct_l239_239033

noncomputable def inscribed_smaller_cube_volume 
  (edge_length_outer_cube : ℝ)
  (h : edge_length_outer_cube = 12) : ℝ := 
  let diameter_sphere := edge_length_outer_cube
  let radius_sphere := diameter_sphere / 2
  let space_diagonal_smaller_cube := diameter_sphere
  let side_length_smaller_cube := space_diagonal_smaller_cube / (Real.sqrt 3)
  let volume_smaller_cube := side_length_smaller_cube ^ 3
  volume_smaller_cube

theorem smaller_cube_volume_is_correct 
  (h : 12 = 12) : inscribed_smaller_cube_volume 12 h = 192 * Real.sqrt 3 :=
by
  sorry

end smaller_cube_volume_is_correct_l239_239033


namespace infinitely_many_sum_form_l239_239870

theorem infinitely_many_sum_form {a : ℕ → ℕ} (h : ∀ n, a n < a (n + 1)) :
  ∀ i, ∃ᶠ n in at_top, ∃ r s j, r > 0 ∧ s > 0 ∧ i < j ∧ a n = r * a i + s * a j := 
by
  sorry

end infinitely_many_sum_form_l239_239870


namespace eval_polynomial_at_3_l239_239779

def f (x : ℝ) : ℝ := 2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

theorem eval_polynomial_at_3 : f 3 = 130 :=
by
  -- proof can be completed here following proper steps or using Horner's method
  sorry

end eval_polynomial_at_3_l239_239779


namespace jebb_take_home_pay_l239_239411

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l239_239411


namespace cone_volume_l239_239079

theorem cone_volume (l : ℝ) (circumference : ℝ) (radius : ℝ) (height : ℝ) (volume : ℝ) 
  (h1 : l = 8) 
  (h2 : circumference = 6 * Real.pi) 
  (h3 : radius = circumference / (2 * Real.pi))
  (h4 : height = Real.sqrt (l^2 - radius^2)) 
  (h5 : volume = (1 / 3) * Real.pi * radius^2 * height) :
  volume = 3 * Real.sqrt 55 * Real.pi := 
  by 
    sorry

end cone_volume_l239_239079


namespace sum_of_angles_l239_239781

theorem sum_of_angles (p q r s t u v w x y : ℝ)
  (H1 : p + r + t + v + x = 360)
  (H2 : q + s + u + w + y = 360) :
  p + q + r + s + t + u + v + w + x + y = 720 := 
by sorry

end sum_of_angles_l239_239781


namespace no_positive_integer_solution_l239_239520

theorem no_positive_integer_solution (m n : ℕ) (h : 0 < m) (h1 : 0 < n) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2006) :=
sorry

end no_positive_integer_solution_l239_239520


namespace robbie_weekly_fat_intake_l239_239883

theorem robbie_weekly_fat_intake
  (morning_cups : ℕ) (afternoon_cups : ℕ) (evening_cups : ℕ)
  (fat_per_cup : ℕ) (days_per_week : ℕ) :
  morning_cups = 3 →
  afternoon_cups = 2 →
  evening_cups = 5 →
  fat_per_cup = 10 →
  days_per_week = 7 →
  (morning_cups * fat_per_cup + afternoon_cups * fat_per_cup + evening_cups * fat_per_cup) * days_per_week = 700 :=
by
  intros
  sorry

end robbie_weekly_fat_intake_l239_239883


namespace no_real_roots_iff_k_gt_2_l239_239675

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end no_real_roots_iff_k_gt_2_l239_239675


namespace fractions_sum_to_one_l239_239476

theorem fractions_sum_to_one :
  ∃ (a b c : ℕ), (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ ((a, b, c) = (2, 3, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (3, 6, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (6, 3, 2)) :=
by
  sorry

end fractions_sum_to_one_l239_239476


namespace bus_stop_time_l239_239309

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) (h1: speed_without_stoppages = 48) (h2: speed_with_stoppages = 24) :
  ∃ (minutes_stopped_per_hour : ℝ), minutes_stopped_per_hour = 30 :=
by
  sorry

end bus_stop_time_l239_239309


namespace count_multiples_of_7_l239_239093

theorem count_multiples_of_7 (low high : ℕ) (hlow : low = 200) (hhigh : high = 400) : 
  (card {n | low ≤ n ∧ n ≤ high ∧ n % 7 = 0}) = 29 := by
  sorry

end count_multiples_of_7_l239_239093


namespace Problem1_Problem2_l239_239187

-- Given conditions as definitions in Lean
structure Square3D :=
(AD AR RQ : ℝ)
(hAD_AR : AD = AR)
(hAR_RQ : AR = 2 * RQ)
(hAD_val : AD = 2)

structure Point (α : Type) :=
(x y z : α)

noncomputable def midpoint (P1 P2 : Point ℝ) : Point ℝ :=
{ x := (P1.x + P2.x) / 2,
  y := (P1.y + P2.y) / 2,
  z := (P1.z + P2.z) / 2 }

noncomputable def onLineSegment (P Q : Point ℝ) (M : Point ℝ) : Prop :=
∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ M = { x := P.x + λ * (Q.x - P.x), y := P.y + λ * (Q.y - P.y), z := P.z + λ * (Q.z - P.z)}

-- Definitions in the problem
variables (R A B C D Q M: Point ℝ)
variables (SQR: Square3D)
variables (E : Point ℝ := midpoint B R)
variables (onM : onLineSegment B Q M)

-- Theorem for Part 1
theorem Problem1 : (R.x = A.x ∧ R.y = A.y ∧ R.z ≠ A.z) →
                   (RQ.x - R.x = AD.x - A.x ∧ RQ.y - R.y = AD.y - A.y ∧ RQ.z - R.z = AD.z - A.z ∧ 
                   AD.y = A.y ∧ AD.z = A.z ∧ AD.x = A.x + 2) →
                   A.z = 0 →
                   E = { x := (B.x + R.x) / 2, y := (B.y + R.y) / 2, z := (B.z + R.z) / 2 } →
                   (M.z = 0) → 
                   AD = AR ∧ AR = 2 * RQ ∧ AD = 2 →
                   (A ≠ E) →
                   A.z = 0 →
                   B.z = 0 →
                   C.z = 0 →
                   D.z = 0 →
                   M.z = 0 →
                   E.z ≠ 0 →
                   A ≠ R →
                   A ≠ B →
                   A ≠ C →
                   A ≠ D →
                   A E⊥ C M :=
sorry

-- Theorem for Part 2
theorem Problem2 : (R.x = A.x ∧ R.y = A.y ∧ R.z ≠ A.z) →
                   (RQ.x - R.x = AD.x - A.x ∧ RQ.y - R.y = AD.y - A.y ∧ RQ.z - R.z = AD.z - A.z ∧ 
                   AD.y = A.y ∧ AD.z = A.z ∧ AD.x = A.x + 2) →
                   A.z = 0 →
                   E = { x := (B.x + R.x) / 2, y := (B.y + R.y) / 2, z := (B.z + R.z) / 2} →
                   (M.x = λ * (Q.x - B.x) + B.x ∧ M.y = λ * (Q.y - B.y) + B.y ∧ M.z = λ * (Q.z - B.z) + B.z) →
                   AD = AR ∧ AR = 2 * RQ ∧ AD = 2 →
                   A ≠ E →
                   A.z = 0 →
                   B.z = 0 →
                   C.z = 0 →
                   D.z = 0 →
                   0 ≤ λ ∧ λ ≤ 1 →
                   4 / 9 ≤ |(normalizedAngle {(C.x - M.x, C.y - M.y, C.z - M.z) • (2, 2, 1)} / (||{C.x - M.x, C.y - M.y, C.z - M.z}|| * ||{2, 2, 1}||))| ∧ |normalizedAngle {(C.x - M.x, C.y - M.y, C.z - M.z) • (2, 2, 1)} / (||{C.x - M.x, C.y - M.y, C.z - M.z}|| * ||{2, 2, 1}||)| ≤ sqrt(2) / 2 :=
sorry

end Problem1_Problem2_l239_239187


namespace robbery_participants_l239_239964

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l239_239964


namespace zero_in_tens_place_l239_239404

variable {A B : ℕ} {m : ℕ}

-- Define the conditions
def condition1 (A : ℕ) (B : ℕ) (m : ℕ) : Prop :=
  ∀ A B : ℕ, ∀ m : ℕ, A * 10^(m+1) + B = 9 * (A * 10^m + B)

theorem zero_in_tens_place (A B : ℕ) (m : ℕ) :
  condition1 A B m → m = 1 :=
by
  intro h
  sorry

end zero_in_tens_place_l239_239404


namespace rubles_greater_than_seven_l239_239135

theorem rubles_greater_than_seven (x : ℕ) (h : x > 7) : ∃ a b : ℕ, x = 3 * a + 5 * b :=
sorry

end rubles_greater_than_seven_l239_239135


namespace number_of_adults_l239_239164

theorem number_of_adults
  (A C : ℕ)
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) :
  A = 350 :=
by
  sorry

end number_of_adults_l239_239164


namespace range_of_x_for_positive_function_value_l239_239566

variable {R : Type*} [LinearOrderedField R]

def even_function (f : R → R) := ∀ x, f (-x) = f x

def monotonically_decreasing_on_nonnegatives (f : R → R) := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem range_of_x_for_positive_function_value (f : R → R)
  (hf_even : even_function f)
  (hf_monotonic : monotonically_decreasing_on_nonnegatives f)
  (hf_at_2 : f 2 = 0)
  (hf_positive : ∀ x, f (x - 1) > 0) :
  ∀ x, -1 < x ∧ x < 3 := sorry

end range_of_x_for_positive_function_value_l239_239566


namespace line_in_first_and_third_quadrants_l239_239713

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l239_239713


namespace line_through_two_quadrants_l239_239716

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l239_239716


namespace sufficient_not_necessary_condition_for_positive_quadratic_l239_239392

variables {a b c : ℝ}

theorem sufficient_not_necessary_condition_for_positive_quadratic 
  (ha : a > 0)
  (hb : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x ^ 2 + b * x + c > 0) 
  ∧ ¬ (∀ x : ℝ, ∃ a b c : ℝ, a > 0 ∧ b^2 - 4 * a * c ≥ 0 ∧ (a * x ^ 2 + b * x + c > 0)) :=
by
  sorry

end sufficient_not_necessary_condition_for_positive_quadratic_l239_239392


namespace vitya_catchup_time_l239_239485

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l239_239485


namespace evens_in_triangle_l239_239272

theorem evens_in_triangle (a : ℕ → ℕ → ℕ) (h : ∀ i j, a i.succ j = (a i (j - 1) + a i j + a i (j + 1)) % 2) :
  ∀ n ≥ 2, ∃ j, a n j % 2 = 0 :=
  sorry

end evens_in_triangle_l239_239272


namespace length_of_second_train_is_approximately_159_98_l239_239008

noncomputable def length_of_second_train : ℝ :=
  let length_first_train := 110 -- meters
  let speed_first_train := 60 -- km/hr
  let speed_second_train := 40 -- km/hr
  let time_to_cross := 9.719222462203025 -- seconds
  let km_per_hr_to_m_per_s := 5 / 18 -- conversion factor from km/hr to m/s
  let relative_speed := (speed_first_train + speed_second_train) * km_per_hr_to_m_per_s -- relative speed in m/s
  let total_distance := relative_speed * time_to_cross -- total distance covered
  total_distance - length_first_train -- length of the second train

theorem length_of_second_train_is_approximately_159_98 :
  abs (length_of_second_train - 159.98) < 0.01 := 
by
  sorry -- Placeholder for the actual proof

end length_of_second_train_is_approximately_159_98_l239_239008


namespace roses_cut_from_garden_l239_239298

-- Define the variables and conditions
variables {x : ℕ} -- x is the number of freshly cut roses

def initial_roses : ℕ := 17
def roses_thrown_away : ℕ := 8
def roses_final_vase : ℕ := 42
def roses_given_away : ℕ := 6

-- The condition that describes the total roses now
def condition (x : ℕ) : Prop :=
  initial_roses - roses_thrown_away + (1/3 : ℚ) * x = roses_final_vase

-- The verification step that checks the total roses concerning given away roses
def verification (x : ℕ) : Prop :=
  (1/3 : ℚ) * x + roses_given_away = roses_final_vase + roses_given_away

-- The main theorem to prove the number of roses cut
theorem roses_cut_from_garden (x : ℕ) (h1 : condition x) (h2 : verification x) : x = 99 :=
  sorry

end roses_cut_from_garden_l239_239298


namespace problem_statement_l239_239217

theorem problem_statement (h1 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2) :
  (Real.pi / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 := by
  sorry

end problem_statement_l239_239217


namespace amount_spent_on_tumbler_l239_239438

def initial_amount : ℕ := 50
def spent_on_coffee : ℕ := 10
def amount_left : ℕ := 10
def total_spent : ℕ := initial_amount - amount_left

theorem amount_spent_on_tumbler : total_spent - spent_on_coffee = 30 := by
  sorry

end amount_spent_on_tumbler_l239_239438


namespace wet_surface_area_is_correct_l239_239783

-- Define the dimensions of the cistern
def cistern_length : ℝ := 6  -- in meters
def cistern_width  : ℝ := 4  -- in meters
def water_depth    : ℝ := 1.25  -- in meters

-- Compute areas for each surface in contact with water
def bottom_area : ℝ := cistern_length * cistern_width
def long_sides_area : ℝ := 2 * (cistern_length * water_depth)
def short_sides_area : ℝ := 2 * (cistern_width * water_depth)

-- Calculate the total area of the wet surface
def total_wet_surface_area : ℝ := bottom_area + long_sides_area + short_sides_area

-- Statement to prove
theorem wet_surface_area_is_correct : total_wet_surface_area = 49 := by
  sorry

end wet_surface_area_is_correct_l239_239783


namespace bank_robbery_participants_l239_239959

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l239_239959


namespace largest_multiple_of_12_negation_l239_239924

theorem largest_multiple_of_12_negation (k : ℤ) (h1 : 12 * k = 144) (h2 : -12 * k > -150) : 12 * k = 144 :=
by
  unfold has_mul.mul
  unfold has_neg.neg
  sorry

end largest_multiple_of_12_negation_l239_239924


namespace LCM_of_numbers_with_HCF_and_ratio_l239_239184

theorem LCM_of_numbers_with_HCF_and_ratio (a b x : ℕ)
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x)
  (h3 : ∀ y : ℕ, y ∣ a → y ∣ b → y ∣ x)
  (hx : x = 5) :
  Nat.lcm a b = 60 := 
by
  sorry

end LCM_of_numbers_with_HCF_and_ratio_l239_239184


namespace x5_plus_y5_l239_239075

theorem x5_plus_y5 (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 252 :=
by
  -- Placeholder for the proof
  sorry

end x5_plus_y5_l239_239075


namespace value_of_6_inch_cube_is_1688_l239_239941

noncomputable def cube_value (side_length : ℝ) : ℝ :=
  let volume := side_length ^ 3
  (volume / 64) * 500

-- Main statement
theorem value_of_6_inch_cube_is_1688 :
  cube_value 6 = 1688 := by
  sorry

end value_of_6_inch_cube_is_1688_l239_239941


namespace isosceles_triangle_angle_ABC_36_l239_239913

noncomputable theory
open Triangle

variables {A B C D F : Point} {a b : Real}

/-- The triangle ABC is isosceles with AB = BC,
    D is the foot of the altitude from B to AC,
    F is the foot of the internal bisector from A, and
    AF = 2BD. Prove the angle ABC is 36 degrees. -/
theorem isosceles_triangle_angle_ABC_36 (h1 : is_isosceles ∠ABC B A B)
(h2 : is_altitude B D AC)
(h3 : is_internal_bisector A F BC)
(h4 : AF = 2 * BD) :
  ∠ABC = 36 := sorry

end isosceles_triangle_angle_ABC_36_l239_239913


namespace count_integer_triangles_with_perimeter_12_l239_239576

theorem count_integer_triangles_with_perimeter_12 : 
  ∃! (sides : ℕ × ℕ × ℕ), sides.1 + sides.2.1 + sides.2.2 = 12 ∧ sides.1 + sides.2.1 > sides.2.2 ∧ sides.1 + sides.2.2 > sides.2.1 ∧ sides.2.1 + sides.2.2 > sides.1 ∧
  (sides = (2, 5, 5) ∨ sides = (3, 4, 5) ∨ sides = (4, 4, 4)) :=
by 
  exists 3
  sorry

end count_integer_triangles_with_perimeter_12_l239_239576


namespace fraction_B_compared_to_A_and_C_l239_239026

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end fraction_B_compared_to_A_and_C_l239_239026


namespace undefined_values_of_expression_l239_239556

theorem undefined_values_of_expression (a : ℝ) :
  a^2 - 9 = 0 ↔ a = -3 ∨ a = 3 := 
sorry

end undefined_values_of_expression_l239_239556


namespace radius_smaller_circle_l239_239027

theorem radius_smaller_circle (A₁ A₂ A₃ : ℝ) (s : ℝ)
  (h1 : A₁ + A₂ = 12 * Real.pi)
  (h2 : A₃ = (Real.sqrt 3 / 4) * s^2)
  (h3 : 2 * A₂ = A₁ + A₁ + A₂ + A₃) :
  ∃ r : ℝ, r = Real.sqrt (6 - (Real.sqrt 3 / 8) * s^2) := by
  sorry

end radius_smaller_circle_l239_239027


namespace expression_equals_39_l239_239810

def expression : ℤ := (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 3 + 2^1 + 2^2 + 2^3 + 2^4

theorem expression_equals_39 : expression = 39 := by 
  sorry

end expression_equals_39_l239_239810


namespace divisible_by_7_in_range_200_to_400_l239_239090

theorem divisible_by_7_in_range_200_to_400 : 
  ∃ n : ℕ, 
    (∀ (x : ℕ), (200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ finset.range (201)) ∧ finset.card (finset.filter (λ x, 200 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0) (finset.range 401)) = 29) := 
begin
  sorry
end

end divisible_by_7_in_range_200_to_400_l239_239090


namespace solution_l239_239216

noncomputable def problem_statement : Prop :=
  ( (π / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 )

theorem solution : problem_statement := by
  sorry

end solution_l239_239216


namespace length_of_room_l239_239738

def area_of_room : ℝ := 10
def width_of_room : ℝ := 2

theorem length_of_room : width_of_room * 5 = area_of_room :=
by
  sorry

end length_of_room_l239_239738


namespace Alexis_mangoes_l239_239328

-- Define the variables for the number of mangoes each person has.
variable (A D Ash : ℕ)

-- Conditions given in the problem.
axiom h1 : A = 4 * (D + Ash)
axiom h2 : A + D + Ash = 75

-- The proof goal.
theorem Alexis_mangoes : A = 60 :=
sorry

end Alexis_mangoes_l239_239328


namespace rectangle_perimeter_l239_239951

-- Define the conditions
variables (z w : ℕ)
-- Define the side lengths of the rectangles
def rectangle_long_side := z - w
def rectangle_short_side := w

-- Theorem: The perimeter of one of the four rectangles
theorem rectangle_perimeter : 2 * (rectangle_long_side z w) + 2 * (rectangle_short_side w) = 2 * z :=
by sorry

end rectangle_perimeter_l239_239951


namespace monotonic_intervals_of_f_range_of_m_for_unique_solution_l239_239683

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (1 + a) * x + Real.log x

-- Monotonic intervals specification
theorem monotonic_intervals_of_f (a : ℝ) (h : 0 ≤ a) :
  (∀ x : ℝ, 0 < x → derivative (λ x, f a x) x ≥ 0 ↔
   (a = 0 ∧ 0 < x ∧ x < 1) ∨
   (0 < a ∧ a < 1 ∧ 0 < x ∧ x < 1) ∨
   (0 < a ∧ a < 1 ∧ x > 1 / a) ∨ 
   (a = 1) ∨
   (a > 1 ∧ 0 < x ∧ x < 1 / a) ∨
   (a > 1 ∧ x > 1)) ∧

  (∀ x : ℝ, 0 < x → derivative (λ x, f a x) x ≤ 0 ↔ 
   (a = 0 ∧ x > 1) ∨
   (0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
   (a > 1 ∧ 1 / a < x ∧ x < 1)) :=
sorry

-- Range of values for m given the unique solution condition
theorem range_of_m_for_unique_solution (a : ℝ) (h : a = 0) :
  ∀ (m : ℝ), (∃ unique x : ℝ, x ∈ Icc 1 (Real.exp 2) ∧ f a x = m * x) ↔
  (m = (Real.log 1 - 1) ∧ m = Real.log 1) ∨
  (m ≥ -1 ∧ m < (2 / Real.exp 2) - 1 ∨ m = (1 / Real.exp 1) - 1) :=
sorry

end monotonic_intervals_of_f_range_of_m_for_unique_solution_l239_239683


namespace speed_against_current_l239_239642

noncomputable def man's_speed_with_current : ℝ := 20
noncomputable def current_speed : ℝ := 1

theorem speed_against_current :
  (man's_speed_with_current - 2 * current_speed) = 18 := by
sorry

end speed_against_current_l239_239642


namespace smallest_possible_area_of_2020th_square_l239_239321

theorem smallest_possible_area_of_2020th_square :
  ∃ A : ℕ, (∃ n : ℕ, n * n = 2019 + A) ∧ A ≠ 1 ∧
  ∀ A' : ℕ, A' > 0 ∧ (∃ n : ℕ, n * n = 2019 + A') ∧ A' ≠ 1 → A ≤ A' :=
by
  sorry

end smallest_possible_area_of_2020th_square_l239_239321


namespace equation_line_through_intersections_l239_239084

theorem equation_line_through_intersections (A1 B1 A2 B2 : ℝ)
  (h1 : 2 * A1 + 3 * B1 = 1)
  (h2 : 2 * A2 + 3 * B2 = 1) :
  ∃ (a b c : ℝ), a = 2 ∧ b = 3 ∧ c = -1 ∧ (a * x + b * y + c = 0) := 
sorry

end equation_line_through_intersections_l239_239084


namespace quadratic_to_vertex_form_l239_239300

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the vertex form of the quadratic function.
def vertex_form (x : ℝ) : ℝ := (x - 1)^2 + 2

-- State the equivalence we want to prove.
theorem quadratic_to_vertex_form :
  ∀ x : ℝ, quadratic_function x = vertex_form x :=
by
  intro x
  show quadratic_function x = vertex_form x
  sorry

end quadratic_to_vertex_form_l239_239300


namespace line_passing_through_first_and_third_quadrants_l239_239701

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l239_239701


namespace lily_pad_growth_rate_l239_239271

theorem lily_pad_growth_rate 
  (day_37_covers_full : ℕ → ℝ)
  (day_36_covers_half : ℕ → ℝ)
  (exponential_growth : day_37_covers_full = 2 * day_36_covers_half) :
  (2 - 1) / 1 * 100 = 100 :=
by sorry

end lily_pad_growth_rate_l239_239271


namespace estimate_blue_balls_l239_239111

theorem estimate_blue_balls (total_balls : ℕ) (prob_yellow : ℚ)
  (h_total : total_balls = 80)
  (h_prob_yellow : prob_yellow = 0.25) :
  total_balls * (1 - prob_yellow) = 60 :=
by
  -- proof
  sorry

end estimate_blue_balls_l239_239111


namespace remaining_two_by_two_square_exists_l239_239677

theorem remaining_two_by_two_square_exists (grid_size : ℕ) (cut_squares : ℕ) : grid_size = 29 → cut_squares = 99 → 
  ∃ remaining_square : ℕ, remaining_square = 1 :=
by
  intros
  sorry

end remaining_two_by_two_square_exists_l239_239677


namespace train_speed_l239_239308

-- Defining the lengths and time
def length_train : ℕ := 100
def length_bridge : ℕ := 300
def time_crossing : ℕ := 15

-- Defining the total distance
def total_distance : ℕ := length_train + length_bridge

-- Proving the speed of the train
theorem train_speed : (total_distance / time_crossing : ℚ) = 26.67 := by
  sorry

end train_speed_l239_239308


namespace calculate_expression_l239_239591

def inequality_holds (a b c d x : ℝ) : Prop :=
  (x - a) * (x - b) * (x - d) / (x - c) ≥ 0

theorem calculate_expression : 
  ∀ (a b c d : ℝ),
    a < b ∧ b < d ∧
    (∀ x : ℝ, 
      (inequality_holds a b c d x ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32))) →
    a + 2 * b + 3 * c + 4 * d = 160 :=
sorry

end calculate_expression_l239_239591


namespace necessary_but_not_sufficient_l239_239592

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a + b > 4) ↔ (¬ (a > 2 ∧ b > 2)) ∧ ((a > 2 ∧ b > 2) → (a + b > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l239_239592


namespace evaluate_256_pow_5_div_8_l239_239669

theorem evaluate_256_pow_5_div_8 (h : 256 = 2^8) : 256^(5/8) = 32 :=
by
  sorry

end evaluate_256_pow_5_div_8_l239_239669


namespace megatek_manufacturing_percentage_l239_239019

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ)
    (h_proportional : total_degrees = 360)
    (h_manufacturing_degrees : manufacturing_degrees = 180) :
    (manufacturing_degrees / total_degrees) * 100 = 50 := by
  -- The proof will go here.
  sorry

end megatek_manufacturing_percentage_l239_239019


namespace new_mixture_concentration_l239_239517

def vessel1_capacity : ℝ := 2
def vessel1_concentration : ℝ := 0.30
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.40
def total_volume : ℝ := 8
def expected_concentration : ℝ := 37.5

theorem new_mixture_concentration :
  ((vessel1_capacity * vessel1_concentration + vessel2_capacity * vessel2_concentration) / total_volume) * 100 = expected_concentration :=
by
  sorry

end new_mixture_concentration_l239_239517


namespace unique_triple_l239_239981

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def find_triples (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  is_prime x ∧ is_prime y ∧ is_prime z ∧
  is_prime (x - y) ∧ is_prime (y - z) ∧ is_prime (x - z)

theorem unique_triple :
  ∀ (x y z : ℕ), find_triples x y z → (x, y, z) = (7, 5, 2) :=
by
  sorry

end unique_triple_l239_239981


namespace compare_fx_l239_239840

noncomputable def f (a x : ℝ) := a * x ^ 2 + 2 * a * x + 4

theorem compare_fx (a x1 x2 : ℝ) (h₁ : -3 < a) (h₂ : a < 0) (h₃ : x1 < x2) (h₄ : x1 + x2 ≠ 1 + a) :
  f a x1 > f a x2 :=
sorry

end compare_fx_l239_239840


namespace limit_to_infinity_zero_l239_239127

variable (f : ℝ → ℝ)

theorem limit_to_infinity_zero (h_continuous : Continuous f)
  (h_alpha : ∀ (α : ℝ), α > 0 → Filter.Tendsto (fun n : ℕ => f (n * α)) Filter.atTop (nhds 0)) :
  Filter.Tendsto f Filter.atTop (nhds 0) :=
sorry

end limit_to_infinity_zero_l239_239127


namespace solve_diophantine_l239_239825

theorem solve_diophantine : ∀ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ (x^3 - y^3 = x * y + 61) → (x, y) = (6, 5) :=
by
  intros x y h
  sorry

end solve_diophantine_l239_239825


namespace solve_fraction_zero_l239_239266

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end solve_fraction_zero_l239_239266


namespace decagon_area_l239_239788

theorem decagon_area 
    (perimeter_square : ℝ) 
    (side_division : ℕ) 
    (side_length : ℝ) 
    (triangle_area : ℝ) 
    (total_triangle_area : ℝ) 
    (square_area : ℝ)
    (decagon_area : ℝ) :
    perimeter_square = 150 →
    side_division = 5 →
    side_length = perimeter_square / 4 →
    triangle_area = 1 / 2 * (side_length / side_division) * (side_length / side_division) →
    total_triangle_area = 8 * triangle_area →
    square_area = side_length * side_length →
    decagon_area = square_area - total_triangle_area →
    decagon_area = 1181.25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end decagon_area_l239_239788


namespace a5_equals_2_l239_239402

variable {a : ℕ → ℝ}  -- a_n represents the nth term of the arithmetic sequence

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a 1 + n * d 

-- Given condition
axiom arithmetic_condition (h : is_arithmetic_sequence a) : a 1 + a 5 + a 9 = 6

-- The goal is to prove a_5 = 2
theorem a5_equals_2 (h : is_arithmetic_sequence a) (h_cond : a 1 + a 5 + a 9 = 6) : a 5 = 2 := 
by 
  sorry

end a5_equals_2_l239_239402


namespace cos_seven_pi_over_six_l239_239806

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l239_239806


namespace simplify_expression_l239_239445

variable (q : Int) -- condition that q is an integer

theorem simplify_expression (q : Int) : 
  ((7 * q + 3) - 3 * q * 2) * 4 + (5 - 2 / 4) * (8 * q - 12) = 40 * q - 42 :=
  by
  sorry

end simplify_expression_l239_239445


namespace vendor_sales_first_day_l239_239326

theorem vendor_sales_first_day (A S: ℝ) (h1: S = S / 100) 
  (h2: 0.20 * A * (1 - S / 100) = 0.42 * A - 0.50 * A * (0.80 * (1 - S / 100)))
  (h3: 0 < S) (h4: S < 100) : 
  S = 30 := 
by
  sorry

end vendor_sales_first_day_l239_239326


namespace vitya_catch_up_time_l239_239501

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l239_239501


namespace system_of_equations_soln_l239_239984

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l239_239984
