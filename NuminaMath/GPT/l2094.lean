import Mathlib

namespace time_ratio_krishan_nandan_l2094_209439

theorem time_ratio_krishan_nandan 
  (N T k : ℝ) 
  (H1 : N * T = 6000) 
  (H2 : N * T + 6 * N * k * T = 78000) 
  : k = 2 := 
by 
sorry

end time_ratio_krishan_nandan_l2094_209439


namespace solve_fisherman_problem_l2094_209403

def fisherman_problem : Prop :=
  ∃ (x y z : ℕ), x + y + z = 16 ∧ 13 * x + 5 * y + 4 * z = 113 ∧ x = 5 ∧ y = 4 ∧ z = 7

theorem solve_fisherman_problem : fisherman_problem :=
sorry

end solve_fisherman_problem_l2094_209403


namespace problem_l2094_209445

theorem problem (C D : ℝ) (h : ∀ x : ℝ, x ≠ 4 → 
  (C / (x - 4)) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) : 
  C + D = 174 :=
sorry

end problem_l2094_209445


namespace min_employees_needed_l2094_209421

-- Definitions for the problem conditions
def hardware_employees : ℕ := 150
def software_employees : ℕ := 130
def both_employees : ℕ := 50

-- Statement of the proof problem
theorem min_employees_needed : hardware_employees + software_employees - both_employees = 230 := 
by 
  -- Calculation skipped with sorry
  sorry

end min_employees_needed_l2094_209421


namespace final_price_is_correct_l2094_209440

def original_price : ℝ := 450
def discounts : List ℝ := [0.10, 0.20, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

noncomputable def final_sale_price (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem final_price_is_correct:
  final_sale_price original_price discounts = 307.8 :=
by
  sorry

end final_price_is_correct_l2094_209440


namespace cost_of_each_pant_l2094_209455

theorem cost_of_each_pant (shirts pants : ℕ) (cost_shirt cost_total : ℕ) (cost_pant : ℕ) :
  shirts = 10 ∧ pants = (shirts / 2) ∧ cost_shirt = 6 ∧ cost_total = 100 →
  (shirts * cost_shirt + pants * cost_pant = cost_total) →
  cost_pant = 8 :=
by
  sorry

end cost_of_each_pant_l2094_209455


namespace negation_of_proposition_l2094_209400

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x > 0 → (x - 2) / x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ (0 ≤ x ∧ x < 2) := 
sorry

end negation_of_proposition_l2094_209400


namespace total_cost_of_items_l2094_209469

variables (E P M : ℝ)

-- Conditions
def condition1 : Prop := E + 3 * P + 2 * M = 240
def condition2 : Prop := 2 * E + 5 * P + 4 * M = 440

-- Question to prove
def question (E P M : ℝ) : ℝ := 3 * E + 4 * P + 6 * M

theorem total_cost_of_items (E P M : ℝ) :
  condition1 E P M →
  condition2 E P M →
  question E P M = 520 := 
by 
  intros h1 h2
  sorry

end total_cost_of_items_l2094_209469


namespace blackRhinoCount_correct_l2094_209485

noncomputable def numberOfBlackRhinos : ℕ :=
  let whiteRhinoCount := 7
  let whiteRhinoWeight := 5100
  let blackRhinoWeightInTons := 1
  let totalWeight := 51700
  let oneTonInPounds := 2000
  let totalWhiteRhinoWeight := whiteRhinoCount * whiteRhinoWeight
  let totalBlackRhinoWeight := totalWeight - totalWhiteRhinoWeight
  totalBlackRhinoWeight / (blackRhinoWeightInTons * oneTonInPounds)

theorem blackRhinoCount_correct : numberOfBlackRhinos = 8 := by
  sorry

end blackRhinoCount_correct_l2094_209485


namespace problem1_problem2_l2094_209418

def f (x a : ℝ) := |x - 1| + |x - a|

/-
  Problem 1:
  Prove that if a = 3, the solution set to the inequality f(x) ≥ 4 is 
  {x | x ≤ 0 ∨ x ≥ 4}.
-/
theorem problem1 (f : ℝ → ℝ → ℝ) (a : ℝ) (h : a = 3) : 
  {x : ℝ | f x a ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := 
sorry

/-
  Problem 2:
  Prove that for any x₁ ∈ ℝ, if f(x₁) ≥ 2 holds true, the range of values for
  a is {a | a ≥ 3 ∨ a ≤ -1}.
-/
theorem problem2 (f : ℝ → ℝ → ℝ) (x₁ : ℝ) :
  (∀ x₁ : ℝ, f x₁ a ≥ 2) ↔ (a ≥ 3 ∨ a ≤ -1) :=
sorry

end problem1_problem2_l2094_209418


namespace ellipse_foci_y_axis_range_l2094_209489

theorem ellipse_foci_y_axis_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1) ↔ (m < -1 ∨ (1 < m ∧ m < 3 / 2)) :=
sorry

end ellipse_foci_y_axis_range_l2094_209489


namespace shortest_side_of_similar_triangle_l2094_209470

def Triangle (a b c : ℤ) : Prop := a^2 + b^2 = c^2
def SimilarTriangles (a b c a' b' c' : ℤ) : Prop := ∃ k : ℤ, k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c 

theorem shortest_side_of_similar_triangle (a b c a' b' c' : ℤ)
  (h₀ : Triangle 15 b 17)
  (h₁ : SimilarTriangles 15 b 17 a' b' c')
  (h₂ : c' = 51) : a' = 24 :=
by
  sorry

end shortest_side_of_similar_triangle_l2094_209470


namespace find_triples_l2094_209407

theorem find_triples (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  (x! + y!) / n! = (3:ℕ)^n ↔ (x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end find_triples_l2094_209407


namespace mary_final_books_l2094_209456

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end mary_final_books_l2094_209456


namespace cost_of_gravelling_path_l2094_209405

theorem cost_of_gravelling_path (length width path_width : ℝ) (cost_per_sq_m : ℝ)
  (h1 : length = 110) (h2 : width = 65) (h3 : path_width = 2.5) (h4 : cost_per_sq_m = 0.50) :
  (length * width - (length - 2 * path_width) * (width - 2 * path_width)) * cost_per_sq_m = 425 := by
  sorry

end cost_of_gravelling_path_l2094_209405


namespace inequality_reversal_l2094_209460

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by
  sorry

end inequality_reversal_l2094_209460


namespace smallest_integer_satisfying_mod_conditions_l2094_209490

theorem smallest_integer_satisfying_mod_conditions :
  ∃ n : ℕ, n > 0 ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  n = 1154 := 
sorry

end smallest_integer_satisfying_mod_conditions_l2094_209490


namespace field_trip_buses_needed_l2094_209412

def fifth_graders : Nat := 109
def sixth_graders : Nat := 115
def seventh_graders : Nat := 118
def teachers_per_grade : Nat := 4
def parents_per_grade : Nat := 2
def total_grades : Nat := 3
def seats_per_bus : Nat := 72

def total_students : Nat := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : Nat := (teachers_per_grade + parents_per_grade) * total_grades
def total_people : Nat := total_students + total_chaperones
def buses_needed : Nat := (total_people + seats_per_bus - 1) / seats_per_bus  -- ceiling division

theorem field_trip_buses_needed : buses_needed = 5 := by
  sorry

end field_trip_buses_needed_l2094_209412


namespace isosceles_triangle_side_length_l2094_209468

theorem isosceles_triangle_side_length (P : ℕ := 53) (base : ℕ := 11) (x : ℕ)
  (h1 : x + x + base = P) : x = 21 :=
by {
  -- The proof goes here.
  sorry
}

end isosceles_triangle_side_length_l2094_209468


namespace find_integer_values_l2094_209480

theorem find_integer_values (a : ℤ) (h : ∃ (n : ℤ), (a + 9) = n * (a + 6)) :
  a = -5 ∨ a = -7 ∨ a = -3 ∨ a = -9 :=
by
  sorry

end find_integer_values_l2094_209480


namespace max_n_is_2_l2094_209432

def is_prime_seq (q : ℕ → ℕ) : Prop :=
  ∀ i, Nat.Prime (q i)

def gen_seq (q0 : ℕ) : ℕ → ℕ
  | 0 => q0
  | (i + 1) => (gen_seq q0 i - 1)^3 + 3

theorem max_n_is_2 (q0 : ℕ) (hq0 : q0 > 0) :
  ∀ (q1 q2 : ℕ), q1 = gen_seq q0 1 → q2 = gen_seq q0 2 → 
  is_prime_seq (gen_seq q0) → q2 = (q1 - 1)^3 + 3 := 
  sorry

end max_n_is_2_l2094_209432


namespace regular_nine_sided_polygon_has_27_diagonals_l2094_209484

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l2094_209484


namespace minimum_value_side_c_l2094_209413

open Real

noncomputable def minimum_side_c (a b c : ℝ) (B : ℝ) (S : ℝ) : ℝ := c

theorem minimum_value_side_c (a b c B : ℝ) (h1 : c * cos B = a + 1 / 2 * b)
  (h2 : S = sqrt 3 / 12 * c) :
  minimum_side_c a b c B S >= 1 :=
by
  -- Precise translation of mathematical conditions and required proof. 
  -- The actual steps to prove the theorem would be here.
  sorry

end minimum_value_side_c_l2094_209413


namespace non_organic_chicken_price_l2094_209434

theorem non_organic_chicken_price :
  ∀ (x : ℝ), (0.75 * x = 9) → (2 * (0.9 * x) = 21.6) :=
by
  intro x hx
  sorry

end non_organic_chicken_price_l2094_209434


namespace students_shorter_than_yoongi_l2094_209464

variable (total_students taller_than_yoongi : Nat)

theorem students_shorter_than_yoongi (h₁ : total_students = 20) (h₂ : taller_than_yoongi = 11) : 
    total_students - (taller_than_yoongi + 1) = 8 :=
by
  -- Here would be the proof
  sorry

end students_shorter_than_yoongi_l2094_209464


namespace height_of_parallelogram_l2094_209419

theorem height_of_parallelogram (A B H : ℕ) (hA : A = 308) (hB : B = 22) (h_eq : H = A / B) : H = 14 := 
by sorry

end height_of_parallelogram_l2094_209419


namespace sahil_selling_price_correct_l2094_209458

-- Define the conditions as constants
def cost_of_machine : ℕ := 13000
def cost_of_repair : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Define the total cost calculation
def total_cost : ℕ := cost_of_machine + cost_of_repair + transportation_charges

-- Define the profit calculation
def profit : ℕ := total_cost * profit_percentage / 100

-- Define the selling price calculation
def selling_price : ℕ := total_cost + profit

-- Now we express our proof problem
theorem sahil_selling_price_correct :
  selling_price = 28500 := by
  -- sorries to skip the proof.
  sorry

end sahil_selling_price_correct_l2094_209458


namespace intersection_setA_setB_l2094_209438

-- Define set A
def setA : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B as the domain of the function y = log(x - 1)
def setB : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem intersection_setA_setB : setA ∩ setB = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_setA_setB_l2094_209438


namespace geese_more_than_ducks_l2094_209409

theorem geese_more_than_ducks (initial_ducks: ℕ) (initial_geese: ℕ) (initial_swans: ℕ) (additional_ducks: ℕ)
  (additional_geese: ℕ) (leaving_swans: ℕ) (leaving_geese: ℕ) (returning_geese: ℕ) (returning_swans: ℕ)
  (final_leaving_ducks: ℕ) (final_leaving_swans: ℕ)
  (initial_ducks_eq: initial_ducks = 25)
  (initial_geese_eq: initial_geese = 2 * initial_ducks - 10)
  (initial_swans_eq: initial_swans = 3 * initial_ducks + 8)
  (additional_ducks_eq: additional_ducks = 4)
  (additional_geese_eq: additional_geese = 7)
  (leaving_swans_eq: leaving_swans = 9)
  (leaving_geese_eq: leaving_geese = 5)
  (returning_geese_eq: returning_geese = 15)
  (returning_swans_eq: returning_swans = 11)
  (final_leaving_ducks_eq: final_leaving_ducks = 2 * (initial_ducks + additional_ducks))
  (final_leaving_swans_eq: final_leaving_swans = (initial_swans + returning_swans) / 2):
  (initial_geese + additional_geese + returning_geese - leaving_geese - final_leaving_geese + returning_geese) -
  (initial_ducks + additional_ducks - final_leaving_ducks) = 57 :=
by
  sorry

end geese_more_than_ducks_l2094_209409


namespace symmetric_function_expression_l2094_209476

variable (f : ℝ → ℝ)
variable (h_sym : ∀ x y, f (-2 - x) = - f x)
variable (h_def : ∀ x, 0 < x → f x = 1 / x)

theorem symmetric_function_expression : ∀ x, x < -2 → f x = 1 / (2 + x) :=
by
  intro x
  intro hx
  sorry

end symmetric_function_expression_l2094_209476


namespace find_x_l2094_209463

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l2094_209463


namespace ned_washed_shirts_l2094_209488

theorem ned_washed_shirts (short_sleeve long_sleeve not_washed: ℕ) (h1: short_sleeve = 9) (h2: long_sleeve = 21) (h3: not_washed = 1) : 
    (short_sleeve + long_sleeve - not_washed = 29) :=
by
  sorry

end ned_washed_shirts_l2094_209488


namespace percentage_equivalence_l2094_209433

theorem percentage_equivalence (x : ℝ) :
  (70 / 100) * 600 = (x / 100) * 1050 → x = 40 :=
by
  sorry

end percentage_equivalence_l2094_209433


namespace cost_per_box_types_l2094_209471

-- Definitions based on conditions
def cost_type_B := 1500
def cost_type_A := cost_type_B + 500

-- Given conditions
def condition1 : cost_type_A = cost_type_B + 500 := by sorry
def condition2 : 6000 / (cost_type_B + 500) = 4500 / cost_type_B := by sorry

-- Theorem to be proved
theorem cost_per_box_types :
  cost_type_A = 2000 ∧ cost_type_B = 1500 ∧
  (∃ (m : ℕ), 20 ≤ m ∧ m ≤ 25 ∧ 2000 * (50 - m) + 1500 * m ≤ 90000) ∧
  (∃ (a b : ℕ), 2500 * a + 3500 * b = 87500 ∧ a + b ≤ 33) :=
sorry

end cost_per_box_types_l2094_209471


namespace total_animals_correct_l2094_209427

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end total_animals_correct_l2094_209427


namespace rainfall_on_tuesday_l2094_209443

noncomputable def R_Tuesday (R_Sunday : ℝ) (D1 : ℝ) : ℝ := 
  R_Sunday + D1

noncomputable def R_Thursday (R_Tuesday : ℝ) (D2 : ℝ) : ℝ :=
  R_Tuesday + D2

noncomputable def total_rainfall (R_Sunday R_Tuesday R_Thursday : ℝ) : ℝ :=
  R_Sunday + R_Tuesday + R_Thursday

theorem rainfall_on_tuesday : R_Tuesday 2 3.75 = 5.75 := 
by 
  sorry -- Proof goes here

end rainfall_on_tuesday_l2094_209443


namespace area_ratio_of_shapes_l2094_209457

theorem area_ratio_of_shapes (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * π * r) (h2 : l = 3 * w) :
  (l * w) / (π * r^2) = (3 * π) / 16 :=
by sorry

end area_ratio_of_shapes_l2094_209457


namespace sum_of_squares_of_geometric_progression_l2094_209465

theorem sum_of_squares_of_geometric_progression 
  {b_1 q S_1 S_2 : ℝ} 
  (h1 : |q| < 1) 
  (h2 : S_1 = b_1 / (1 - q))
  (h3 : S_2 = b_1 / (1 + q)) : 
  (b_1^2 / (1 - q^2)) = S_1 * S_2 := 
by
  sorry

end sum_of_squares_of_geometric_progression_l2094_209465


namespace fewer_twos_to_hundred_l2094_209462

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l2094_209462


namespace mrs_wilsborough_tickets_l2094_209474

theorem mrs_wilsborough_tickets :
  ∀ (saved vip_ticket_cost regular_ticket_cost vip_tickets left : ℕ),
    saved = 500 →
    vip_ticket_cost = 100 →
    regular_ticket_cost = 50 →
    vip_tickets = 2 →
    left = 150 →
    (saved - left - (vip_tickets * vip_ticket_cost)) / regular_ticket_cost = 3 :=
by
  intros saved vip_ticket_cost regular_ticket_cost vip_tickets left
  sorry

end mrs_wilsborough_tickets_l2094_209474


namespace mgp_inequality_l2094_209475

theorem mgp_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (1 / Real.sqrt (1 / 2 + a + a * b + a * b * c) + 
   1 / Real.sqrt (1 / 2 + b + b * c + b * c * d) + 
   1 / Real.sqrt (1 / 2 + c + c * d + c * d * a) + 
   1 / Real.sqrt (1 / 2 + d + d * a + d * a * b)) 
  ≥ Real.sqrt 2 := 
sorry

end mgp_inequality_l2094_209475


namespace gcd_of_powers_l2094_209491

theorem gcd_of_powers (a b : ℕ) (h1 : a = 2^300 - 1) (h2 : b = 2^315 - 1) :
  gcd a b = 32767 :=
by
  sorry

end gcd_of_powers_l2094_209491


namespace larry_wins_game_l2094_209444

-- Defining probabilities for Larry and Julius
def larry_throw_prob : ℚ := 2 / 3
def julius_throw_prob : ℚ := 1 / 3

-- Calculating individual probabilities based on the description
def p1 : ℚ := larry_throw_prob
def p3 : ℚ := (julius_throw_prob ^ 2) * larry_throw_prob
def p5 : ℚ := (julius_throw_prob ^ 4) * larry_throw_prob

-- Aggregating the probability that Larry wins the game
def larry_wins_prob : ℚ := p1 + p3 + p5

-- The proof statement
theorem larry_wins_game : larry_wins_prob = 170 / 243 := by
  sorry

end larry_wins_game_l2094_209444


namespace income_to_expenditure_ratio_l2094_209415

variable (I E S : ℕ)

def Ratio (a b : ℕ) : ℚ := a / (b : ℚ)

theorem income_to_expenditure_ratio (h1 : I = 14000) (h2 : S = 2000) (h3 : S = I - E) : 
  Ratio I E = 7 / 6 :=
by
  sorry

end income_to_expenditure_ratio_l2094_209415


namespace minimize_fuel_consumption_l2094_209414

-- Define conditions as constants
def cargo_total : ℕ := 157
def cap_large : ℕ := 5
def cap_small : ℕ := 2
def fuel_large : ℕ := 20
def fuel_small : ℕ := 10

-- Define truck counts
def n_large : ℕ := 31
def n_small : ℕ := 1

-- Theorem: the number of large and small trucks that minimize fuel consumption
theorem minimize_fuel_consumption : 
  n_large * cap_large + n_small * cap_small = cargo_total ∧
  (∀ m_large m_small, m_large * cap_large + m_small * cap_small = cargo_total → 
    m_large * fuel_large + m_small * fuel_small ≥ n_large * fuel_large + n_small * fuel_small) :=
by
  -- Statement to be proven
  sorry

end minimize_fuel_consumption_l2094_209414


namespace min_a2_k2b2_l2094_209431

variable (a b t k : ℝ)
variable (hk : 0 < k)
variable (h : a + k * b = t)

theorem min_a2_k2b2 (a b t k : ℝ) (hk : 0 < k) (h : a + k * b = t) :
  a^2 + (k * b)^2 ≥ (1 + k^2) * (t^2) / ((1 + k)^2) :=
sorry

end min_a2_k2b2_l2094_209431


namespace cards_per_page_l2094_209499

noncomputable def total_cards (new_cards old_cards : ℕ) : ℕ := new_cards + old_cards

theorem cards_per_page
  (new_cards old_cards : ℕ)
  (total_pages : ℕ)
  (h_new_cards : new_cards = 3)
  (h_old_cards : old_cards = 13)
  (h_total_pages : total_pages = 2) :
  total_cards new_cards old_cards / total_pages = 8 :=
by
  rw [h_new_cards, h_old_cards, h_total_pages]
  rfl

end cards_per_page_l2094_209499


namespace geometric_sequence_b_value_l2094_209435

theorem geometric_sequence_b_value :
  ∀ (a b c : ℝ),
  (a = 5 + 2 * Real.sqrt 6) →
  (c = 5 - 2 * Real.sqrt 6) →
  (b * b = a * c) →
  (b = 1 ∨ b = -1) :=
by
  intros a b c ha hc hgeometric
  sorry

end geometric_sequence_b_value_l2094_209435


namespace product_of_possible_values_of_x_l2094_209493

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end product_of_possible_values_of_x_l2094_209493


namespace unique_four_digit_perfect_cube_divisible_by_16_and_9_l2094_209467

theorem unique_four_digit_perfect_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 16 = 0 ∧ n % 9 = 0 ∧ n = 1728 :=
by sorry

end unique_four_digit_perfect_cube_divisible_by_16_and_9_l2094_209467


namespace max_min_y_l2094_209429

noncomputable def y (x : ℝ) : ℝ :=
  7 - 4 * (Real.sin x) * (Real.cos x) + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

theorem max_min_y :
  (∃ x : ℝ, y x = 10) ∧ (∃ x : ℝ, y x = 6) := by
  sorry

end max_min_y_l2094_209429


namespace find_x_squared_plus_y_squared_l2094_209494

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 75) : x^2 + y^2 = 3205 / 121 :=
by
  sorry

end find_x_squared_plus_y_squared_l2094_209494


namespace triangle_properties_l2094_209492

open Real

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (b = c)

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def area (a b c : ℝ) (A : ℝ) : ℝ :=
  1/2 * b * c * sin A

theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h1 : sin B * sin C = 1/4) 
  (h2 : tan B * tan C = 1/3) 
  (h3 : a = 4 * sqrt 3) 
  (h4 : A + B + C = π) 
  (isosceles : is_isosceles_triangle A B C a b c) :
  is_isosceles_triangle A B C a b c ∧ 
  perimeter a b c = 8 + 4 * sqrt 3 ∧ 
  area a b c A = 4 * sqrt 3 :=
sorry

end triangle_properties_l2094_209492


namespace find_c_l2094_209447

def sum_of_digits (n : ℕ) : ℕ := (n.digits 10).sum

theorem find_c :
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 5 :=
by
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  sorry

end find_c_l2094_209447


namespace range_of_a_l2094_209472

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → (a > 2) :=
by
  sorry

end range_of_a_l2094_209472


namespace solution_set_of_inequality_l2094_209496

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l2094_209496


namespace nat_numbers_l2094_209459

theorem nat_numbers (n : ℕ) (h1 : n ≥ 2) (h2 : ∃a b : ℕ, a * b = n ∧ ∀ c : ℕ, 1 < c ∧ c ∣ n → a ≤ c ∧ n = a^2 + b^2) : 
  n = 5 ∨ n = 8 ∨ n = 20 :=
by
  sorry

end nat_numbers_l2094_209459


namespace sufficient_but_not_necessary_l2094_209406

theorem sufficient_but_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l2094_209406


namespace three_digit_number_increase_l2094_209461

theorem three_digit_number_increase (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n * 1001 / n) = 1001 :=
by
  sorry

end three_digit_number_increase_l2094_209461


namespace alpha_range_midpoint_trajectory_l2094_209422

noncomputable def circle_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
  ⟨Real.cos θ, Real.sin θ⟩

theorem alpha_range (α : ℝ) (h1 : 0 < α ∧ α < 2 * Real.pi) :
  (Real.tan α) > 1 ∨ (Real.tan α) < -1 ↔ (Real.pi / 4 < α ∧ α < 3 * Real.pi / 4) ∨ 
                                          (5 * Real.pi / 4 < α ∧ α < 7 * Real.pi / 4) := 
  sorry

theorem midpoint_trajectory (m : ℝ) (h2 : -1 < m ∧ m < 1) :
  ∃ x y : ℝ, x = (Real.sqrt 2 * m) / (m^2 + 1) ∧ 
             y = -(Real.sqrt 2 * m^2) / (m^2 + 1) :=
  sorry

end alpha_range_midpoint_trajectory_l2094_209422


namespace average_speed_of_rocket_l2094_209401

theorem average_speed_of_rocket
  (ascent_speed : ℕ)
  (ascent_time : ℕ)
  (descent_distance : ℕ)
  (descent_time : ℕ)
  (average_speed : ℕ)
  (h_ascent_speed : ascent_speed = 150)
  (h_ascent_time : ascent_time = 12)
  (h_descent_distance : descent_distance = 600)
  (h_descent_time : descent_time = 3)
  (h_average_speed : average_speed = 160) :
  (ascent_speed * ascent_time + descent_distance) / (ascent_time + descent_time) = average_speed :=
by
  sorry

end average_speed_of_rocket_l2094_209401


namespace travel_time_on_third_day_l2094_209416

-- Definitions based on conditions
def speed_first_day : ℕ := 5
def time_first_day : ℕ := 7
def distance_first_day : ℕ := speed_first_day * time_first_day

def speed_second_day_part1 : ℕ := 6
def time_second_day_part1 : ℕ := 6
def distance_second_day_part1 : ℕ := speed_second_day_part1 * time_second_day_part1

def speed_second_day_part2 : ℕ := 3
def time_second_day_part2 : ℕ := 3
def distance_second_day_part2 : ℕ := speed_second_day_part2 * time_second_day_part2

def distance_second_day : ℕ := distance_second_day_part1 + distance_second_day_part2
def total_distance_first_two_days : ℕ := distance_first_day + distance_second_day

def total_distance : ℕ := 115
def distance_third_day : ℕ := total_distance - total_distance_first_two_days

def speed_third_day : ℕ := 7
def time_third_day : ℕ := distance_third_day / speed_third_day

-- The statement to be proven
theorem travel_time_on_third_day : time_third_day = 5 := by
  sorry

end travel_time_on_third_day_l2094_209416


namespace value_of_g_3x_minus_5_l2094_209423

variable (R : Type) [Field R]
variable (g : R → R)
variable (x y : R)

-- Given condition: g(x) = -3 for all real numbers x
axiom g_is_constant : ∀ x : R, g x = -3

-- Prove that g(3x - 5) = -3
theorem value_of_g_3x_minus_5 : g (3 * x - 5) = -3 :=
by
  sorry

end value_of_g_3x_minus_5_l2094_209423


namespace find_Q_l2094_209441

variable (Q U P k : ℝ)

noncomputable def varies_directly_and_inversely : Prop :=
  P = k * (Q / U)

theorem find_Q (h : varies_directly_and_inversely Q U P k)
  (h1 : P = 6) (h2 : Q = 8) (h3 : U = 4)
  (h4 : P = 18) (h5 : U = 9) :
  Q = 54 :=
sorry

end find_Q_l2094_209441


namespace find_a_l2094_209477

-- Definitions from conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def directrix : ℝ := 1

-- Statement to prove
theorem find_a (a : ℝ) (h : directrix = 1) : a = -1/4 :=
sorry

end find_a_l2094_209477


namespace x_equals_l2094_209428

variable (x y: ℝ)

theorem x_equals:
  (x / (x - 2) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 1)) → x = 2 * y^2 + 6 * y + 2 := by
  sorry

end x_equals_l2094_209428


namespace trainB_destination_time_l2094_209473

def trainA_speed : ℕ := 90
def trainB_speed : ℕ := 135
def trainA_time_after_meeting : ℕ := 9
def trainB_time_after_meeting (x : ℕ) : ℕ := 18 - 3 * x

theorem trainB_destination_time : (trainA_time_after_meeting, trainA_speed) = (9, 90) → 
  (trainB_speed, trainB_time_after_meeting 3) = (135, 3) := by
  sorry

end trainB_destination_time_l2094_209473


namespace problem_solution_l2094_209498

-- Define the sets and the conditions given in the problem
def setA : Set ℝ := 
  {y | ∃ (x : ℝ), (x ∈ Set.Icc (3 / 4) 2) ∧ (y = x^2 - (3 / 2) * x + 1)}

def setB (m : ℝ) : Set ℝ := 
  {x | x + m^2 ≥ 1}

-- The proof statement contains two parts
theorem problem_solution (m : ℝ) :
  -- Part (I) - Prove the set A
  setA = Set.Icc (7 / 16) 2
  ∧
  -- Part (II) - Prove the range for m
  (∀ x, x ∈ setA → x ∈ setB m) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
  sorry

end problem_solution_l2094_209498


namespace Cherry_weekly_earnings_l2094_209452

theorem Cherry_weekly_earnings :
  let charge_small_cargo := 2.50
  let charge_large_cargo := 4.00
  let daily_small_cargo := 4
  let daily_large_cargo := 2
  let days_in_week := 7
  let daily_earnings := (charge_small_cargo * daily_small_cargo) + (charge_large_cargo * daily_large_cargo)
  let weekly_earnings := daily_earnings * days_in_week
  weekly_earnings = 126 := sorry

end Cherry_weekly_earnings_l2094_209452


namespace marcie_and_martin_in_picture_l2094_209466

noncomputable def marcie_prob_in_picture : ℚ :=
  let marcie_lap_time := 100
  let martin_lap_time := 75
  let start_time := 720
  let end_time := 780
  let picture_duration := 60
  let marcie_position_720 := (720 % marcie_lap_time) / marcie_lap_time
  let marcie_in_pic_start := 0
  let marcie_in_pic_end := 20 + 33 + 1/3
  let martin_position_720 := (720 % martin_lap_time) / martin_lap_time
  let martin_in_pic_start := 20
  let martin_in_pic_end := 45 + 25
  let overlap_start := max marcie_in_pic_start martin_in_pic_start
  let overlap_end := min marcie_in_pic_end martin_in_pic_end
  let overlap_duration := overlap_end - overlap_start
  overlap_duration / picture_duration

theorem marcie_and_martin_in_picture :
  marcie_prob_in_picture = 111 / 200 :=
by
  sorry

end marcie_and_martin_in_picture_l2094_209466


namespace equation_of_chord_l2094_209495

-- Define the ellipse equation and point P
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def P : ℝ × ℝ := (3, 2)
def is_midpoint (A B P : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2
def on_chord (A B : ℝ × ℝ) (x y : ℝ) : Prop := (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)

-- Lean Statement
theorem equation_of_chord :
  ∀ A B : ℝ × ℝ,
    ellipse_eq A.1 A.2 →
    ellipse_eq B.1 B.2 →
    is_midpoint A B P →
    ∀ x y : ℝ,
      on_chord A B x y →
      2 * x + 3 * y = 12 :=
by
  sorry

end equation_of_chord_l2094_209495


namespace integer_solutions_to_equation_l2094_209487

theorem integer_solutions_to_equation :
  { p : ℤ × ℤ | (p.1 ^ 2 * p.2 + 1 = p.1 ^ 2 + 2 * p.1 * p.2 + 2 * p.1 + p.2) } =
  { (-1, -1), (0, 1), (1, -1), (2, -7), (3, 7) } :=
by
  sorry

end integer_solutions_to_equation_l2094_209487


namespace remainder_when_divided_by_100_l2094_209411

/-- A basketball team has 15 available players. A fixed set of 5 players starts the game, while the other 
10 are available as substitutes. During the game, the coach may make up to 4 substitutions. No player 
removed from the game may reenter, and no two substitutions can happen simultaneously. The players 
involved and the order of substitutions matter. -/
def num_substitution_sequences : ℕ :=
  let a_0 := 1
  let a_1 := 5 * 10
  let a_2 := a_1 * 4 * 9
  let a_3 := a_2 * 3 * 8
  let a_4 := a_3 * 2 * 7
  a_0 + a_1 + a_2 + a_3 + a_4

theorem remainder_when_divided_by_100 : num_substitution_sequences % 100 = 51 :=
by
  -- proof to be written
  sorry

end remainder_when_divided_by_100_l2094_209411


namespace find_z_l2094_209479

theorem find_z : 
    ∃ z : ℝ, ( ( 2 ^ 5 ) * ( 9 ^ 2 ) ) / ( z * ( 3 ^ 5 ) ) = 0.16666666666666666 ↔ z = 64 :=
by
    sorry

end find_z_l2094_209479


namespace odd_number_difference_of_squares_not_unique_l2094_209481

theorem odd_number_difference_of_squares_not_unique :
  ∀ n : ℤ, Odd n → ∃ X Y X' Y' : ℤ, (n = X^2 - Y^2) ∧ (n = X'^2 - Y'^2) ∧ (X ≠ X' ∨ Y ≠ Y') :=
sorry

end odd_number_difference_of_squares_not_unique_l2094_209481


namespace number_of_special_permutations_l2094_209483

noncomputable def count_special_permutations : ℕ :=
  (Nat.choose 12 6)

theorem number_of_special_permutations : count_special_permutations = 924 :=
  by
    sorry

end number_of_special_permutations_l2094_209483


namespace max_sum_of_squares_l2094_209486

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 85) 
  (h3 : ad + bc = 196) 
  (h4 : cd = 120) : 
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 918 :=
by {
  sorry
}

end max_sum_of_squares_l2094_209486


namespace value_standard_deviations_less_than_mean_l2094_209425

-- Definitions of the given conditions
def mean : ℝ := 15
def std_dev : ℝ := 1.5
def value : ℝ := 12

-- Lean 4 statement to prove the question
theorem value_standard_deviations_less_than_mean :
  (mean - value) / std_dev = 2 := by
  sorry

end value_standard_deviations_less_than_mean_l2094_209425


namespace angle_ACD_l2094_209437

theorem angle_ACD (E : ℝ) (arc_eq : ∀ (AB BC CD : ℝ), AB = BC ∧ BC = CD) (angle_eq : E = 40) : ∃ (ACD : ℝ), ACD = 15 :=
by
  sorry

end angle_ACD_l2094_209437


namespace jerry_claims_years_of_salary_l2094_209408

theorem jerry_claims_years_of_salary
  (Y : ℝ)
  (salary_damage_per_year : ℝ := 50000)
  (medical_bills : ℝ := 200000)
  (punitive_damages : ℝ := 3 * (salary_damage_per_year * Y + medical_bills))
  (total_damages : ℝ := salary_damage_per_year * Y + medical_bills + punitive_damages)
  (received_amount : ℝ := 0.8 * total_damages)
  (actual_received_amount : ℝ := 5440000) :
  received_amount = actual_received_amount → Y = 30 := 
by
  sorry

end jerry_claims_years_of_salary_l2094_209408


namespace find_other_number_l2094_209446

noncomputable def HCF : ℕ := 14
noncomputable def LCM : ℕ := 396
noncomputable def one_number : ℕ := 154
noncomputable def product_of_numbers : ℕ := HCF * LCM

theorem find_other_number (other_number : ℕ) :
  HCF * LCM = one_number * other_number → other_number = 36 :=
by
  sorry

end find_other_number_l2094_209446


namespace find_a_l2094_209448

theorem find_a (a : ℝ) (h1 : a + 3 > 0) (h2 : abs (a + 3) = 5) : a = 2 := 
by
  sorry

end find_a_l2094_209448


namespace round_robin_tournament_l2094_209404

theorem round_robin_tournament (n : ℕ)
  (total_points_1 : ℕ := 3086) (total_points_2 : ℕ := 2018) (total_points_3 : ℕ := 1238)
  (pair_avg_1 : ℕ := (3086 + 1238) / 2) (pair_avg_2 : ℕ := (3086 + 2018) / 2) (pair_avg_3 : ℕ := (1238 + 2018) / 2)
  (overall_avg : ℕ := (3086 + 2018 + 1238) / 3)
  (all_pairwise_diff : pair_avg_1 ≠ pair_avg_2 ∧ pair_avg_1 ≠ pair_avg_3 ∧ pair_avg_2 ≠ pair_avg_3) :
  n = 47 :=
by
  sorry

end round_robin_tournament_l2094_209404


namespace sample_capacity_l2094_209478

theorem sample_capacity (n : ℕ) (A B C : ℕ) (h_ratio : A / (A + B + C) = 3 / 14) (h_A : A = 15) : n = 70 :=
by
  sorry

end sample_capacity_l2094_209478


namespace functional_eq_zero_l2094_209402

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_zero :
  (∀ x y : ℝ, f (x + y) = f x - f y) →
  (∀ x : ℝ, f x = 0) :=
by
  intros h x
  sorry

end functional_eq_zero_l2094_209402


namespace paths_from_A_to_B_no_revisits_l2094_209442

noncomputable def numPaths : ℕ :=
  16

theorem paths_from_A_to_B_no_revisits : numPaths = 16 :=
by
  sorry

end paths_from_A_to_B_no_revisits_l2094_209442


namespace factorization_correct_l2094_209449

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end factorization_correct_l2094_209449


namespace sarah_bus_time_l2094_209430

noncomputable def totalTimeAway : ℝ := (4 + 15/60) + (5 + 15/60)  -- 9.5 hours
noncomputable def totalTimeAwayInMinutes : ℝ := totalTimeAway * 60  -- 570 minutes

noncomputable def timeInClasses : ℝ := 8 * 45  -- 360 minutes
noncomputable def timeInLunch : ℝ := 30  -- 30 minutes
noncomputable def timeInExtracurricular : ℝ := 1.5 * 60  -- 90 minutes
noncomputable def totalTimeInSchoolActivities : ℝ := timeInClasses + timeInLunch + timeInExtracurricular  -- 480 minutes

noncomputable def timeOnBus : ℝ := totalTimeAwayInMinutes - totalTimeInSchoolActivities  -- 90 minutes

theorem sarah_bus_time : timeOnBus = 90 := by
  sorry

end sarah_bus_time_l2094_209430


namespace marcella_shoes_l2094_209497

theorem marcella_shoes :
  ∀ (original_pairs lost_shoes : ℕ), original_pairs = 27 → lost_shoes = 9 → 
  ∃ (remaining_pairs : ℕ), remaining_pairs = 18 ∧ remaining_pairs ≤ original_pairs - lost_shoes / 2 :=
by
  intros original_pairs lost_shoes h1 h2
  use 18
  constructor
  . exact rfl
  . sorry

end marcella_shoes_l2094_209497


namespace twelve_position_in_circle_l2094_209454

theorem twelve_position_in_circle (a : ℕ → ℕ) (h_cyclic : ∀ i, a (i + 20) = a i)
  (h_sum_six : ∀ i, a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) = 24)
  (h_first : a 1 = 1) :
  a 12 = 7 :=
sorry

end twelve_position_in_circle_l2094_209454


namespace ellipse_standard_equation_chord_length_range_l2094_209410

-- Conditions for question 1
def ellipse_center (O : ℝ × ℝ) : Prop := O = (0, 0)
def major_axis_x (major_axis : ℝ) : Prop := major_axis = 1
def eccentricity (e : ℝ) : Prop := e = (Real.sqrt 2) / 2
def perp_chord_length (AA' : ℝ) : Prop := AA' = Real.sqrt 2

-- Lean statement for question 1
theorem ellipse_standard_equation (O : ℝ × ℝ) (major_axis : ℝ) (e : ℝ) (AA' : ℝ) :
  ellipse_center O → major_axis_x major_axis → eccentricity e → perp_chord_length AA' →
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / (a^2)) + y^2 / (b^2) = 1) := sorry

-- Conditions for question 2
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def max_area_triangle (S : ℝ) : Prop := S = 1 / 2

-- Lean statement for question 2
theorem chord_length_range (x y z w : ℝ) (E F G H : ℝ × ℝ) :
  circle_eq x y → ellipse_eq z w → max_area_triangle ((E.1 * F.1) * (Real.sin (E.2 * F.2))) →
  ( ∃ min_chord max_chord : ℝ, min_chord = Real.sqrt 3 ∧ max_chord = 2 ∧
    ∀ x1 y1 x2 y2 : ℝ, (G.1 = x1 ∧ H.1 = x2 ∧ G.2 = y1 ∧ H.2 = y2) →
    (min_chord ≤ (Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2)))) ∧
         Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2))) ≤ max_chord )) := sorry

end ellipse_standard_equation_chord_length_range_l2094_209410


namespace max_a_value_l2094_209426

theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 - 2 * x - 3 > 0) →
  (¬ (∀ x : ℝ, x^2 - 2 * x - 3 > 0 → x < a)) →
  a = -1 :=
by
  sorry

end max_a_value_l2094_209426


namespace shopkeeper_gain_percent_l2094_209420

theorem shopkeeper_gain_percent
    (SP₁ SP₂ CP : ℝ)
    (h₁ : SP₁ = 187)
    (h₂ : SP₂ = 264)
    (h₃ : SP₁ = 0.85 * CP) :
    ((SP₂ - CP) / CP) * 100 = 20 := by 
  sorry

end shopkeeper_gain_percent_l2094_209420


namespace fuel_efficiency_l2094_209436

noncomputable def gas_cost_per_gallon : ℝ := 4
noncomputable def money_spent_on_gas : ℝ := 42
noncomputable def miles_traveled : ℝ := 336

theorem fuel_efficiency : (miles_traveled / (money_spent_on_gas / gas_cost_per_gallon)) = 32 := by
  sorry

end fuel_efficiency_l2094_209436


namespace teacher_age_is_45_l2094_209424

def avg_age_of_students := 14
def num_students := 30
def avg_age_with_teacher := 15
def num_people_with_teacher := 31

def total_age_of_students := avg_age_of_students * num_students
def total_age_with_teacher := avg_age_with_teacher * num_people_with_teacher

theorem teacher_age_is_45 : (total_age_with_teacher - total_age_of_students = 45) :=
by
  sorry

end teacher_age_is_45_l2094_209424


namespace no_solution_equation_l2094_209453

theorem no_solution_equation (m : ℝ) : 
  ¬∃ x : ℝ, x ≠ 2 ∧ (x - 3) / (x - 2) = m / (2 - x) → m = 1 := 
by 
  sorry

end no_solution_equation_l2094_209453


namespace divisor_of_51234_plus_3_l2094_209482

theorem divisor_of_51234_plus_3 : ∃ d : ℕ, d > 1 ∧ (51234 + 3) % d = 0 ∧ d = 3 :=
by {
  sorry
}

end divisor_of_51234_plus_3_l2094_209482


namespace difference_of_squares_l2094_209417

noncomputable def product_of_consecutive_integers (n : ℕ) := n * (n + 1)

theorem difference_of_squares (h : ∃ n : ℕ, product_of_consecutive_integers n = 2720) :
  ∃ a b : ℕ, product_of_consecutive_integers a = 2720 ∧ (b = a + 1) ∧ (b * b - a * a = 103) :=
by
  sorry

end difference_of_squares_l2094_209417


namespace problem_solving_ratio_l2094_209450

theorem problem_solving_ratio 
  (total_mcqs : ℕ) (total_psqs : ℕ)
  (written_mcqs_fraction : ℚ) (total_remaining_questions : ℕ)
  (h1 : total_mcqs = 35)
  (h2 : total_psqs = 15)
  (h3 : written_mcqs_fraction = 2/5)
  (h4 : total_remaining_questions = 31) :
  (5 : ℚ) / 15 = (1 : ℚ) / 3 := 
by {
  -- given that 5 is the number of problem-solving questions already written,
  -- and 15 is the total number of problem-solving questions
  sorry
}

end problem_solving_ratio_l2094_209450


namespace correct_statements_l2094_209451
noncomputable def is_pythagorean_triplet (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem correct_statements {a b c : ℕ} (h1 : is_pythagorean_triplet a b c) (h2 : a^2 + b^2 = c^2) :
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → a^2 + b^2 = c^2)) ∧
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → is_pythagorean_triplet (2 * a) (2 * b) (2 * c))) :=
by sorry

end correct_statements_l2094_209451
