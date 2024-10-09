import Mathlib

namespace express_An_l1484_148450

noncomputable def A_n (A : ℝ) (n : ℤ) : ℝ :=
  (1 / 2^n) * ((A + (A^2 - 4).sqrt)^n + (A - (A^2 - 4).sqrt)^n)

theorem express_An (a : ℝ) (A : ℝ) (n : ℤ) (h : a + a⁻¹ = A) :
  (a^n + a^(-n)) = A_n A n := 
sorry

end express_An_l1484_148450


namespace Zhenya_Venya_are_truth_tellers_l1484_148493

-- Definitions
def is_truth_teller(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = true

def is_liar(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = false

noncomputable def BenyaStatement := "V is a liar"
noncomputable def ZhenyaStatement := "B is a liar"
noncomputable def SenyaStatement1 := "B and V are liars"
noncomputable def SenyaStatement2 := "Zh is a liar"

-- Conditions and proving the statement
theorem Zhenya_Venya_are_truth_tellers (truth_teller : String → Bool) :
  (∀ dwarf, truth_teller dwarf = true ∨ truth_teller dwarf = false) →
  (is_truth_teller "Benya" truth_teller → is_liar "Venya" truth_teller) →
  (is_truth_teller "Zhenya" truth_teller → is_liar "Benya" truth_teller) →
  (is_truth_teller "Senya" truth_teller → 
    is_liar "Benya" truth_teller ∧ is_liar "Venya" truth_teller ∧ is_liar "Zhenya" truth_teller) →
  is_truth_teller "Zhenya" truth_teller ∧ is_truth_teller "Venya" truth_teller :=
by
  sorry

end Zhenya_Venya_are_truth_tellers_l1484_148493


namespace Michelle_initial_crayons_l1484_148457

variable (M : ℕ)  -- M is the number of crayons Michelle initially has
variable (J : ℕ := 2)  -- Janet has 2 crayons
variable (final_crayons : ℕ := 4)  -- After Janet gives her crayons to Michelle, Michelle has 4 crayons

theorem Michelle_initial_crayons : M + J = final_crayons → M = 2 :=
by
  intro h1
  sorry

end Michelle_initial_crayons_l1484_148457


namespace circle_diameter_l1484_148470

theorem circle_diameter (r : ℝ) (h : π * r^2 = 9 * π) : 2 * r = 6 :=
by sorry

end circle_diameter_l1484_148470


namespace algebra_expression_l1484_148413

theorem algebra_expression (a b : ℝ) (h : a - b = 3) : 1 + a - b = 4 :=
sorry

end algebra_expression_l1484_148413


namespace maximum_area_of_rectangular_farm_l1484_148403

theorem maximum_area_of_rectangular_farm :
  ∃ l w : ℕ, 2 * (l + w) = 160 ∧ l * w = 1600 :=
by
  sorry

end maximum_area_of_rectangular_farm_l1484_148403


namespace min_director_games_l1484_148454

theorem min_director_games (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 325) (h2 : (26 * 25) / 2 = 325) : k = 0 :=
by {
  -- The conditions are provided in the hypothesis, and the goal is proving the minimum games by director equals 0.
  sorry
}

end min_director_games_l1484_148454


namespace relationship_between_roots_l1484_148482

-- Define the number of real roots of the equations
def number_real_roots_lg_eq_sin : ℕ := 3
def number_real_roots_x_eq_sin : ℕ := 1
def number_real_roots_x4_eq_sin : ℕ := 2

-- Define the variables
def a : ℕ := number_real_roots_lg_eq_sin
def b : ℕ := number_real_roots_x_eq_sin
def c : ℕ := number_real_roots_x4_eq_sin

-- State the theorem
theorem relationship_between_roots : a > c ∧ c > b :=
by
  -- the proof is skipped
  sorry

end relationship_between_roots_l1484_148482


namespace maximum_elements_in_A_l1484_148497

theorem maximum_elements_in_A (n : ℕ) (h : n > 0)
  (A : Finset (Finset (Fin n))) 
  (hA : ∀ a ∈ A, ∀ b ∈ A, a ≠ b → ¬ a ⊆ b) :  
  A.card ≤ Nat.choose n (n / 2) :=
sorry

end maximum_elements_in_A_l1484_148497


namespace lily_milk_quantity_l1484_148485

theorem lily_milk_quantity :
  let init_gallons := (5 : ℝ)
  let given_away := (18 / 4 : ℝ)
  let received_back := (7 / 4 : ℝ)
  init_gallons - given_away + received_back = 2 + 1 / 4 :=
by
  sorry

end lily_milk_quantity_l1484_148485


namespace unique_functional_equation_l1484_148488

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
sorry

end unique_functional_equation_l1484_148488


namespace inequality_example_l1484_148499

theorem inequality_example (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) (h4 : b < 0) : a + b < b + c := 
by sorry

end inequality_example_l1484_148499


namespace train_speed_kmph_l1484_148447

-- The conditions
def speed_m_s : ℝ := 52.5042
def conversion_factor : ℝ := 3.6

-- The theorem we need to prove
theorem train_speed_kmph : speed_m_s * conversion_factor = 189.01512 := 
  sorry

end train_speed_kmph_l1484_148447


namespace minor_premise_l1484_148416

-- Definitions
def Rectangle : Type := sorry
def Square : Type := sorry
def Parallelogram : Type := sorry

axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle
axiom square_is_parallelogram : Square → Parallelogram

-- Problem statement
theorem minor_premise : ∀ (S : Square), ∃ (R : Rectangle), square_is_rectangle S = R :=
by
  sorry

end minor_premise_l1484_148416


namespace divides_sequence_l1484_148414

theorem divides_sequence (a : ℕ → ℕ) (n k: ℕ) (h0 : a 0 = 0) (h1 : a 1 = 1) 
  (hrec : ∀ m, a (m + 2) = 2 * a (m + 1) + a m) :
  (2^k ∣ a n) ↔ (2^k ∣ n) :=
sorry

end divides_sequence_l1484_148414


namespace missy_yells_at_obedient_dog_12_times_l1484_148469

theorem missy_yells_at_obedient_dog_12_times (x : ℕ) (h : x + 4 * x = 60) : x = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end missy_yells_at_obedient_dog_12_times_l1484_148469


namespace part1_solution_set_part2_a_range_l1484_148419

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := abs x + 2 * abs (x + 2 - a)

-- Part 1: When a = 3, solving the inequality
theorem part1_solution_set (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

-- Part 2: Finding the range of a such that f(x) = g(x-2) >= 1 for all x in ℝ
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := g (x - 2) a

theorem part2_a_range : (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end part1_solution_set_part2_a_range_l1484_148419


namespace complex_square_l1484_148425

theorem complex_square (a b : ℝ) (i : ℂ) (h1 : a + b * i - 2 * i = 2 - b * i) : 
  (a + b * i) ^ 2 = 3 + 4 * i := 
by {
  -- Proof steps skipped (using sorry to indicate proof is required)
  sorry
}

end complex_square_l1484_148425


namespace find_c2_given_d4_l1484_148492

theorem find_c2_given_d4 (c d k : ℝ) (h : c^2 * d^4 = k) (hc8 : c = 8) (hd2 : d = 2) (hd4 : d = 4):
  c^2 = 4 :=
by
  sorry

end find_c2_given_d4_l1484_148492


namespace combination_simplify_l1484_148495

theorem combination_simplify : (Nat.choose 6 2) + 3 = 18 := by
  sorry

end combination_simplify_l1484_148495


namespace donny_spent_total_on_friday_and_sunday_l1484_148473

noncomputable def daily_savings (initial: ℚ) (increase_rate: ℚ) (days: List ℚ) : List ℚ :=
days.scanl (λ acc day => acc * increase_rate + acc) initial

noncomputable def thursday_savings : ℚ := (daily_savings 15 (1 + 0.1) [15, 15, 15]).sum

noncomputable def friday_spent : ℚ := thursday_savings * 0.5

noncomputable def remaining_after_friday : ℚ := thursday_savings - friday_spent

noncomputable def saturday_savings (thursday: ℚ) : ℚ := thursday * (1 - 0.20)

noncomputable def total_savings_saturday : ℚ := remaining_after_friday + saturday_savings thursday_savings

noncomputable def sunday_spent : ℚ := total_savings_saturday * 0.40

noncomputable def total_spent : ℚ := friday_spent + sunday_spent

theorem donny_spent_total_on_friday_and_sunday : total_spent = 55.13 := by
  sorry

end donny_spent_total_on_friday_and_sunday_l1484_148473


namespace mystical_mountain_creatures_l1484_148433

-- Definitions for conditions
def nineHeadedBirdHeads : Nat := 9
def nineHeadedBirdTails : Nat := 1
def nineTailedFoxHeads : Nat := 1
def nineTailedFoxTails : Nat := 9

-- Prove the number of Nine-Tailed Foxes
theorem mystical_mountain_creatures (x y : Nat)
  (h1 : 9 * x + (y - 1) = 36 * (y - 1) + 4 * x)
  (h2 : 9 * (x - 1) + y = 3 * (9 * y + (x - 1))) :
  x = 14 :=
by
  sorry

end mystical_mountain_creatures_l1484_148433


namespace simplify_fraction_l1484_148453

theorem simplify_fraction (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := 
sorry

end simplify_fraction_l1484_148453


namespace least_area_exists_l1484_148408

-- Definition of the problem conditions
def is_rectangle (l w : ℕ) : Prop :=
  2 * (l + w) = 120

def area (l w : ℕ) := l * w

-- Statement of the proof problem
theorem least_area_exists :
  ∃ (l w : ℕ), is_rectangle l w ∧ (∀ (l' w' : ℕ), is_rectangle l' w' → area l w ≤ area l' w') ∧ area l w = 59 :=
sorry

end least_area_exists_l1484_148408


namespace sum_of_vertices_l1484_148420

theorem sum_of_vertices (n : ℕ) (h1 : 6 * n + 12 * n = 216) : 8 * n = 96 :=
by
  -- Proof is omitted intentionally
  sorry

end sum_of_vertices_l1484_148420


namespace range_of_m_l1484_148445

theorem range_of_m (m : ℝ) :
  (∃ x y : ℤ, (x ≠ y) ∧ (x ≥ m ∧ y ≥ m) ∧ (3 - 2 * x ≥ 0) ∧ (3 - 2 * y ≥ 0)) ↔ (-1 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l1484_148445


namespace circle_center_radius_sum_l1484_148480

theorem circle_center_radius_sum :
  let D := { p : ℝ × ℝ | (p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34) }
  let c := 7
  let d := -5
  let s := 2 * Real.sqrt 10
  (c + d + s = 2 + 2 * Real.sqrt 10) :=
by
  sorry

end circle_center_radius_sum_l1484_148480


namespace ellipse_hyperbola_equation_l1484_148476

-- Definitions for the Ellipse and Hyperbola
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / 10 + (y^2) / m = 1
def hyperbola (x y : ℝ) (b : ℝ) : Prop := (x^2) - (y^2) / b = 1

-- Conditions
def same_foci (c1 c2 : ℝ) : Prop := c1 = c2
def intersection_at_p (x y : ℝ) : Prop := x = (Real.sqrt 10) / 3 ∧ (ellipse x y 1 ∧ hyperbola x y 8)

-- Theorem stating the mathematically equivalent proof problem
theorem ellipse_hyperbola_equation :
  ∀ (m b : ℝ) (x y : ℝ), ellipse x y m ∧ hyperbola x y b ∧ same_foci (Real.sqrt (10 - m)) (Real.sqrt (1 + b)) ∧ intersection_at_p x y
  → (m = 1) ∧ (b = 8) := 
by
  intros m b x y h
  sorry

end ellipse_hyperbola_equation_l1484_148476


namespace find_polynomial_value_l1484_148424

theorem find_polynomial_value (x y : ℝ) 
  (h1 : 3 * x + y = 12) 
  (h2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := 
by 
  sorry

end find_polynomial_value_l1484_148424


namespace valid_triangles_from_10_points_l1484_148441

noncomputable def number_of_valid_triangles (n : ℕ) (h : n = 10) : ℕ :=
  if n = 10 then 100 else 0

theorem valid_triangles_from_10_points :
  number_of_valid_triangles 10 rfl = 100 := 
sorry

end valid_triangles_from_10_points_l1484_148441


namespace shooting_test_performance_l1484_148434

theorem shooting_test_performance (m n : ℝ)
    (h1 : m > 9.7)
    (h2 : n < 0.25) :
    (m = 9.9 ∧ n = 0.2) :=
sorry

end shooting_test_performance_l1484_148434


namespace circle_line_intersection_symmetric_l1484_148472

theorem circle_line_intersection_symmetric (m n p x y : ℝ)
    (h_intersects : ∃ x y, x = m * y - 1 ∧ x^2 + y^2 + m * x + n * y + p = 0)
    (h_symmetric : ∀ A B : ℝ × ℝ, A = (x, y) ∧ B = (y, x) → y = x) :
    p < -3 / 2 :=
by
  sorry

end circle_line_intersection_symmetric_l1484_148472


namespace evaluate_expression_l1484_148448

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) : 
  z * (z - 4 * x) = 36 := by
  sorry

end evaluate_expression_l1484_148448


namespace max_connected_stations_l1484_148460

theorem max_connected_stations (n : ℕ) 
  (h1 : ∀ s : ℕ, s ≤ n → s ≤ 3) 
  (h2 : ∀ x y : ℕ, x < y → ∃ z : ℕ, z < 3 ∧ z ≤ n) : 
  n = 10 :=
by 
  sorry

end max_connected_stations_l1484_148460


namespace average_price_of_towels_l1484_148412

-- Definitions based on conditions
def cost_towel1 : ℕ := 3 * 100
def cost_towel2 : ℕ := 5 * 150
def cost_towel3 : ℕ := 2 * 600
def total_cost : ℕ := cost_towel1 + cost_towel2 + cost_towel3
def total_towels : ℕ := 3 + 5 + 2
def average_price : ℕ := total_cost / total_towels

-- Statement to be proved
theorem average_price_of_towels :
  average_price = 225 :=
by
  sorry

end average_price_of_towels_l1484_148412


namespace trapezoid_base_count_l1484_148496

theorem trapezoid_base_count (A h : ℕ) (multiple : ℕ) (bases_sum pairs_count : ℕ) : 
  A = 1800 ∧ h = 60 ∧ multiple = 10 ∧ pairs_count = 4 ∧ 
  bases_sum = (A / (1/2 * h)) / multiple → pairs_count > 3 := 
by 
  sorry

end trapezoid_base_count_l1484_148496


namespace solve_for_x_l1484_148455

theorem solve_for_x (b x : ℝ) (h1 : b > 1) (h2 : x > 0)
    (h3 : (4 * x) ^ (Real.log 4 / Real.log b) = (6 * x) ^ (Real.log 6 / Real.log b)) :
    x = 1 / 6 :=
by
  sorry

end solve_for_x_l1484_148455


namespace α_eq_β_plus_two_l1484_148490

-- Definitions based on the given conditions:
-- α(n): number of ways n can be expressed as a sum of the integers 1 and 2, considering different orders as distinct ways.
-- β(n): number of ways n can be expressed as a sum of integers greater than 1, considering different orders as distinct ways.

def α (n : ℕ) : ℕ := sorry
def β (n : ℕ) : ℕ := sorry

-- The proof statement that needs to be proved.
theorem α_eq_β_plus_two (n : ℕ) (h : 0 < n) : α n = β (n + 2) := 
  sorry

end α_eq_β_plus_two_l1484_148490


namespace dividend_value_l1484_148409

def dividend (divisor quotient remainder : ℝ) := (divisor * quotient) + remainder

theorem dividend_value :
  dividend 35.8 21.65 11.3 = 786.47 :=
by
  sorry

end dividend_value_l1484_148409


namespace no_solution_exists_l1484_148486

theorem no_solution_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x ^ 2 + f y) = 2 * x - f y :=
by
  sorry

end no_solution_exists_l1484_148486


namespace find_x_l1484_148494

theorem find_x :
  ∀ x : ℝ, (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
  8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) →
  x = 1486 / 225 :=
by
  sorry

end find_x_l1484_148494


namespace company_total_parts_l1484_148443

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

end company_total_parts_l1484_148443


namespace probability_not_touch_outer_edge_l1484_148405

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end probability_not_touch_outer_edge_l1484_148405


namespace min_soldiers_needed_l1484_148466

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l1484_148466


namespace not_directly_nor_inversely_proportional_l1484_148421

theorem not_directly_nor_inversely_proportional :
  ∀ (x y : ℝ),
    ((2 * x + y = 5) ∨ (2 * x + 3 * y = 12)) ∧
    ((¬ (∃ k : ℝ, x = k * y)) ∧ (¬ (∃ k : ℝ, x * y = k))) := sorry

end not_directly_nor_inversely_proportional_l1484_148421


namespace cos_C_in_triangle_l1484_148411

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end cos_C_in_triangle_l1484_148411


namespace bitcoin_donation_l1484_148401

theorem bitcoin_donation (x : ℝ) (h : 3 * (80 - x) / 2 - 10 = 80) : x = 20 :=
sorry

end bitcoin_donation_l1484_148401


namespace negation_of_proposition_p_l1484_148436

variable (x : ℝ)

def proposition_p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0

theorem negation_of_proposition_p : ¬ proposition_p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end negation_of_proposition_p_l1484_148436


namespace make_up_set_money_needed_l1484_148478

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end make_up_set_money_needed_l1484_148478


namespace find_2008_star_2010_l1484_148484

-- Define the operation
def operation_star (x y : ℕ) : ℕ := sorry  -- We insert a sorry here because the precise definition is given by the conditions

-- The properties given in the problem
axiom property1 : operation_star 2 2010 = 1
axiom property2 : ∀ n : ℕ, operation_star (2 * (n + 1)) 2010 = 3 * operation_star (2 * n) 2010

-- The main proof statement
theorem find_2008_star_2010 : operation_star 2008 2010 = 3 ^ 1003 :=
by
  -- Here we would provide the proof, but it's omitted.
  sorry

end find_2008_star_2010_l1484_148484


namespace randy_trips_l1484_148498

def trips_per_month
  (initial : ℕ) -- Randy initially had $200 in his piggy bank
  (final : ℕ)   -- Randy had $104 left in his piggy bank after a year
  (spend_per_trip : ℕ) -- Randy spends $2 every time he goes to the store
  (months_in_year : ℕ) -- Number of months in a year, which is 12
  (total_trips_per_year : ℕ) -- Total trips he makes in a year
  (trips_per_month : ℕ) -- Trips to the store every month
  : Prop :=
  initial = 200 ∧ final = 104 ∧ spend_per_trip = 2 ∧ months_in_year = 12 ∧
  total_trips_per_year = (initial - final) / spend_per_trip ∧ 
  trips_per_month = total_trips_per_year / months_in_year ∧
  trips_per_month = 4

theorem randy_trips :
  trips_per_month 200 104 2 12 ((200 - 104) / 2) (48 / 12) :=
by 
  sorry

end randy_trips_l1484_148498


namespace calculate_expression_l1484_148431

theorem calculate_expression : 2 * Real.sin (60 * Real.pi / 180) + (-1/2)⁻¹ + abs (2 - Real.sqrt 3) = 0 :=
by
  sorry

end calculate_expression_l1484_148431


namespace one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l1484_148491

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem one_zero_implies_a_eq_pm2 (a : ℝ) : (∃! x, f a x = 0) → (a = 2 ∨ a = -2) := by
  sorry

theorem zero_in_interval_implies_a_in_open_interval (a : ℝ) : (∃ x, f a x = 0 ∧ 0 < x ∧ x < 1) → 2 < a := by
  sorry

end one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l1484_148491


namespace horizontal_distance_travelled_l1484_148444

theorem horizontal_distance_travelled (r : ℝ) (θ : ℝ) (d : ℝ)
  (h_r : r = 2) (h_θ : θ = Real.pi / 6) :
  d = 2 * Real.sqrt 3 * Real.pi := sorry

end horizontal_distance_travelled_l1484_148444


namespace inequality_geq_8_l1484_148452

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end inequality_geq_8_l1484_148452


namespace no_unique_solution_l1484_148475

theorem no_unique_solution (d : ℝ) (x y : ℝ) :
  (3 * (3 * x + 4 * y) = 36) ∧ (9 * x + 12 * y = d) ↔ d ≠ 36 := sorry

end no_unique_solution_l1484_148475


namespace Joey_SAT_Weeks_l1484_148410

theorem Joey_SAT_Weeks
    (hours_per_night : ℕ) (nights_per_week : ℕ)
    (hours_per_weekend_day : ℕ) (days_per_weekend : ℕ)
    (total_hours : ℕ) (weekly_hours : ℕ) (weeks : ℕ)
    (h1 : hours_per_night = 2) (h2 : nights_per_week = 5)
    (h3 : hours_per_weekend_day = 3) (h4 : days_per_weekend = 2)
    (h5 : total_hours = 96) (h6 : weekly_hours = 16)
    (h7 : weekly_hours = (hours_per_night * nights_per_week) + (hours_per_weekend_day * days_per_weekend)) :
  weeks = total_hours / weekly_hours :=
sorry

end Joey_SAT_Weeks_l1484_148410


namespace triangle_angle_eq_pi_over_3_l1484_148463

theorem triangle_angle_eq_pi_over_3
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = a * b)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ C : ℝ, C = 2 * Real.pi / 3 ∧ 
            Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
by
  -- Proof goes here
  sorry

end triangle_angle_eq_pi_over_3_l1484_148463


namespace polynomial_symmetric_equiv_l1484_148474

variable {R : Type*} [CommRing R]

def symmetric_about (P : R → R) (a b : R) : Prop :=
  ∀ x, P (2 * a - x) = 2 * b - P x

def polynomial_form (P : R → R) (a b : R) (Q : R → R) : Prop :=
  ∀ x, P x = b + (x - a) * Q ((x - a) * (x - a))

theorem polynomial_symmetric_equiv (P Q : R → R) (a b : R) :
  (symmetric_about P a b ↔ polynomial_form P a b Q) :=
sorry

end polynomial_symmetric_equiv_l1484_148474


namespace sum_of_all_four_numbers_is_zero_l1484_148487

theorem sum_of_all_four_numbers_is_zero 
  {a b c d : ℝ}
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b = c + d)
  (h_prod : a * c = b * d) 
  : a + b + c + d = 0 := 
by
  sorry

end sum_of_all_four_numbers_is_zero_l1484_148487


namespace most_stable_yield_l1484_148404

theorem most_stable_yield (S_A S_B S_C S_D : ℝ)
  (h₁ : S_A = 3.6)
  (h₂ : S_B = 2.89)
  (h₃ : S_C = 13.4)
  (h₄ : S_D = 20.14) : 
  S_B < S_A ∧ S_B < S_C ∧ S_B < S_D :=
by {
  sorry -- Proof skipped as per instructions
}

end most_stable_yield_l1484_148404


namespace calculate_minimal_total_cost_l1484_148456

structure GardenSection where
  area : ℕ
  flower_cost : ℚ

def garden := [
  GardenSection.mk 10 2.75, -- Orchids
  GardenSection.mk 14 2.25, -- Violets
  GardenSection.mk 14 1.50, -- Hyacinths
  GardenSection.mk 15 1.25, -- Tulips
  GardenSection.mk 25 0.75  -- Sunflowers
]

def total_cost (sections : List GardenSection) : ℚ :=
  sections.foldr (λ s acc => s.area * s.flower_cost + acc) 0

theorem calculate_minimal_total_cost :
  total_cost garden = 117.5 := by
  sorry

end calculate_minimal_total_cost_l1484_148456


namespace sufficientButNotNecessary_l1484_148446

theorem sufficientButNotNecessary (x : ℝ) : ((x + 1) * (x - 3) < 0) → x < 3 ∧ ¬(x < 3 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficientButNotNecessary_l1484_148446


namespace cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l1484_148464

-- Definition of size-n tromino
def tromino_area (n : ℕ) := (4 * 4 * n - 1)

-- Problem (a): Can a size-5 tromino be tiled by size-1 trominos
theorem cannot_tile_size5_with_size1_trominos :
  ¬ (∃ (count : ℕ), count * 3 = tromino_area 5) :=
by sorry

-- Problem (b): Can a size-2013 tromino be tiled by size-1 trominos
theorem can_tile_size2013_with_size1_trominos :
  ∃ (count : ℕ), count * 3 = tromino_area 2013 :=
by sorry

end cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l1484_148464


namespace binom_8_5_eq_56_l1484_148479

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l1484_148479


namespace remainder_division_l1484_148422

theorem remainder_division (N : ℤ) (hN : N % 899 = 63) : N % 29 = 5 := 
by 
  sorry

end remainder_division_l1484_148422


namespace total_weight_of_load_l1484_148483

def weight_of_crate : ℕ := 4
def weight_of_carton : ℕ := 3
def number_of_crates : ℕ := 12
def number_of_cartons : ℕ := 16

theorem total_weight_of_load :
  number_of_crates * weight_of_crate + number_of_cartons * weight_of_carton = 96 :=
by sorry

end total_weight_of_load_l1484_148483


namespace min_value_of_reciprocal_sum_l1484_148400

-- Define the problem
theorem min_value_of_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y : ℝ, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (2 * a * x - b * y + 2 = 0)):
  ∃ (m : ℝ), m = 4 ∧ (1 / a + 1 / b) ≥ m :=
by
  sorry

end min_value_of_reciprocal_sum_l1484_148400


namespace find_common_ratio_l1484_148402

variable {α : Type*} [LinearOrderedField α] [NormedLinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop := ∀ n, a (n+1) = q * a n

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop := ∀ n, S n = (Finset.range n).sum a

theorem find_common_ratio
  (a : ℕ → α)
  (S : ℕ → α)
  (q : α)
  (pos_terms : ∀ n, 0 < a n)
  (geometric_seq : geometric_sequence a q)
  (sum_eq : sum_first_n_terms a S)
  (eqn : S 1 + 2 * S 5 = 3 * S 3) :
  q = (2:α)^(3 / 2) / 2^(3 / 2) :=
by
  sorry

end find_common_ratio_l1484_148402


namespace lucy_run_base10_eq_1878_l1484_148449

-- Define a function to convert a base-8 numeral to base-10.
def base8_to_base10 (n: Nat) : Nat :=
  (3 * 8^3) + (5 * 8^2) + (2 * 8^1) + (6 * 8^0)

-- Define the base-8 number.
def lucy_run (n : Nat) : Nat := n

-- Prove that the base-10 equivalent of the base-8 number 3526 is 1878.
theorem lucy_run_base10_eq_1878 : base8_to_base10 (lucy_run 3526) = 1878 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end lucy_run_base10_eq_1878_l1484_148449


namespace time_with_walkway_l1484_148477

theorem time_with_walkway (v w : ℝ) (t : ℕ) :
  (80 = 120 * (v - w)) → 
  (80 = 60 * v) → 
  t = 80 / (v + w) → 
  t = 40 :=
by
  sorry

end time_with_walkway_l1484_148477


namespace min_birthdays_on_wednesday_l1484_148462

theorem min_birthdays_on_wednesday (n x w: ℕ) (h_n : n = 61) 
  (h_ineq : w > x) (h_sum : 6 * x + w = n) : w ≥ 13 :=
by
  sorry

end min_birthdays_on_wednesday_l1484_148462


namespace bus_travel_time_l1484_148468

theorem bus_travel_time (D1 D2: ℝ) (T: ℝ) (h1: D1 + D2 = 250) (h2: D1 >= 0) (h3: D2 >= 0) :
  T = D1 / 40 + D2 / 60 ↔ D1 + D2 = 250 := 
by
  sorry

end bus_travel_time_l1484_148468


namespace angle_CDE_proof_l1484_148437

theorem angle_CDE_proof
    (A B C D E : Type)
    (angle_A angle_B angle_C : ℝ)
    (angle_AEB : ℝ)
    (angle_BED : ℝ)
    (angle_BDE : ℝ) :
    angle_A = 90 ∧
    angle_B = 90 ∧
    angle_C = 90 ∧
    angle_AEB = 50 ∧
    angle_BED = 2 * angle_BDE →
    ∃ angle_CDE : ℝ, angle_CDE = 70 :=
by
  sorry

end angle_CDE_proof_l1484_148437


namespace range_of_x_l1484_148481

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x < -1/2 ∨ x > 1/4 :=
by
  sorry

end range_of_x_l1484_148481


namespace rabbits_clear_land_in_21_days_l1484_148442

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end rabbits_clear_land_in_21_days_l1484_148442


namespace speed_plane_east_l1484_148439

-- Definitions of the conditions
def speed_west : ℕ := 275
def time_hours : ℝ := 3.5
def distance_apart : ℝ := 2100

-- Theorem statement to prove the speed of the plane traveling due East
theorem speed_plane_east (v: ℝ) 
  (h: (v + speed_west) * time_hours = distance_apart) : 
  v = 325 :=
  sorry

end speed_plane_east_l1484_148439


namespace price_of_second_variety_l1484_148461

-- Define prices and conditions
def price_first : ℝ := 126
def price_third : ℝ := 175.5
def mixture_price : ℝ := 153
def total_weight : ℝ := 4

-- Define unknown price
variable (x : ℝ)

-- Definition of the weighted mixture price
theorem price_of_second_variety :
  (1 * price_first) + (1 * x) + (2 * price_third) = total_weight * mixture_price →
  x = 135 :=
by
  sorry

end price_of_second_variety_l1484_148461


namespace phil_quarters_l1484_148430

def initial_quarters : ℕ := 50

def quarters_after_first_year (initial : ℕ) : ℕ := 2 * initial

def quarters_collected_second_year : ℕ := 3 * 12

def quarters_collected_third_year : ℕ := 12 / 3

def total_quarters_before_loss (initial : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ := 
  quarters_after_first_year initial + second_year + third_year

def lost_quarters (total : ℕ) : ℕ := total / 4

def quarters_left (total : ℕ) (lost : ℕ) : ℕ := total - lost

theorem phil_quarters : 
  quarters_left 
    (total_quarters_before_loss 
      initial_quarters 
      quarters_collected_second_year 
      quarters_collected_third_year)
    (lost_quarters 
      (total_quarters_before_loss 
        initial_quarters 
        quarters_collected_second_year 
        quarters_collected_third_year))
  = 105 :=
by
  sorry

end phil_quarters_l1484_148430


namespace monthly_average_decrease_rate_l1484_148407

-- Conditions
def january_production : Float := 1.6 * 10^6
def march_production : Float := 0.9 * 10^6
def rate_decrease : Float := 0.25

-- Proof Statement: we need to prove that the monthly average decrease rate x = 0.25 satisfies the given condition
theorem monthly_average_decrease_rate :
  january_production * (1 - rate_decrease) * (1 - rate_decrease) = march_production := by
  sorry

end monthly_average_decrease_rate_l1484_148407


namespace annual_interest_rate_l1484_148429

theorem annual_interest_rate 
  (P A : ℝ) 
  (hP : P = 136) 
  (hA : A = 150) 
  : (A - P) / P = 0.10 :=
by sorry

end annual_interest_rate_l1484_148429


namespace number_of_remaining_red_points_l1484_148415

/-- 
Given a grid where the distance between any two adjacent points in a row or column is 1,
and any green point can turn points within a distance of no more than 1 into green every second.
Initial state of the grid is given. Determine the number of red points after 4 seconds.
-/
def remaining_red_points_after_4_seconds (initial_state : List (List Bool)) : Nat := 
41 -- assume this is the computed number after applying the infection rule for 4 seconds

theorem number_of_remaining_red_points (initial_state : List (List Bool)) :
  remaining_red_points_after_4_seconds initial_state = 41 := 
sorry

end number_of_remaining_red_points_l1484_148415


namespace car_Y_average_speed_l1484_148440

theorem car_Y_average_speed 
  (car_X_speed : ℝ)
  (car_X_time_before_Y : ℝ)
  (car_X_distance_when_Y_starts : ℝ)
  (car_X_total_distance : ℝ)
  (car_X_travel_time : ℝ)
  (car_Y_distance : ℝ)
  (car_Y_travel_time : ℝ)
  (h_car_X_speed : car_X_speed = 35)
  (h_car_X_time_before_Y : car_X_time_before_Y = 72 / 60)
  (h_car_X_distance_when_Y_starts : car_X_distance_when_Y_starts = car_X_speed * car_X_time_before_Y)
  (h_car_X_total_distance : car_X_total_distance = car_X_distance_when_Y_starts + car_X_distance_when_Y_starts)
  (h_car_X_travel_time : car_X_travel_time = car_X_total_distance / car_X_speed)
  (h_car_Y_distance : car_Y_distance = 490)
  (h_car_Y_travel_time : car_Y_travel_time = car_X_travel_time) :
  (car_Y_distance / car_Y_travel_time) = 32.24 := 
sorry

end car_Y_average_speed_l1484_148440


namespace sum_of_coefficients_at_1_l1484_148417

def P (x : ℝ) := 2 * (4 * x^8 - 3 * x^5 + 9)
def Q (x : ℝ) := 9 * (x^6 + 2 * x^3 - 8)
def R (x : ℝ) := P x + Q x

theorem sum_of_coefficients_at_1 : R 1 = -25 := by
  sorry

end sum_of_coefficients_at_1_l1484_148417


namespace haniMoreSitupsPerMinute_l1484_148489

-- Define the conditions given in the problem
def totalSitups : Nat := 110
def situpsByDiana : Nat := 40
def rateDianaPerMinute : Nat := 4

-- Define the derived conditions from the solution steps
def timeDianaMinutes := situpsByDiana / rateDianaPerMinute -- 10 minutes
def situpsByHani := totalSitups - situpsByDiana -- 70 situps
def rateHaniPerMinute := situpsByHani / timeDianaMinutes -- 7 situps per minute

-- The theorem we need to prove
theorem haniMoreSitupsPerMinute : rateHaniPerMinute - rateDianaPerMinute = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end haniMoreSitupsPerMinute_l1484_148489


namespace midpoint_trajectory_of_moving_point_l1484_148428

/-- Given a fixed point A (4, -3) and a moving point B on the circle (x+1)^2 + y^2 = 4, prove that 
    the equation of the trajectory of the midpoint M of the line segment AB is 
    (x - 3/2)^2 + (y + 3/2)^2 = 1. -/
theorem midpoint_trajectory_of_moving_point {x y : ℝ} :
  (∃ (B : ℝ × ℝ), (B.1 + 1)^2 + B.2^2 = 4 ∧ 
    (x, y) = ((B.1 + 4) / 2, (B.2 - 3) / 2)) →
  (x - 3/2)^2 + (y + 3/2)^2 = 1 :=
by sorry

end midpoint_trajectory_of_moving_point_l1484_148428


namespace total_pencils_l1484_148426

variable (C Y M D : ℕ)

-- Conditions
def cheryl_has_thrice_as_cyrus (h1 : C = 3 * Y) : Prop := true
def madeline_has_half_of_cheryl (h2 : M = 63 ∧ C = 2 * M) : Prop := true
def daniel_has_25_percent_of_total (h3 : D = (C + Y + M) / 4) : Prop := true

-- Total number of pencils for all four
theorem total_pencils (h1 : C = 3 * Y) (h2 : M = 63 ∧ C = 2 * M) (h3 : D = (C + Y + M) / 4) :
  C + Y + M + D = 289 :=
by { sorry }

end total_pencils_l1484_148426


namespace max_value_of_trig_expr_l1484_148427

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end max_value_of_trig_expr_l1484_148427


namespace evaluate_expression_l1484_148418

theorem evaluate_expression (x : ℤ) (h : x = 5) : 
  3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2 = 1457 := 
by
  rw [h]
  sorry

end evaluate_expression_l1484_148418


namespace calen_more_pencils_l1484_148432

def calen_pencils (C B D: ℕ) :=
  D = 9 ∧
  B = 2 * D - 3 ∧
  C - 10 = 10

theorem calen_more_pencils (C B D : ℕ) (h : calen_pencils C B D) : C = B + 5 :=
by
  obtain ⟨hD, hB, hC⟩ := h
  simp only [hD, hB, hC]
  sorry

end calen_more_pencils_l1484_148432


namespace trapezium_area_l1484_148451

theorem trapezium_area (a b h : ℝ) (h₁ : a = 20) (h₂ : b = 16) (h₃ : h = 15) : 
  (1/2 * (a + b) * h = 270) :=
by
  rw [h₁, h₂, h₃]
  -- The following lines of code are omitted as they serve as solving this proof, and the requirement is to provide the statement only. 
  sorry

end trapezium_area_l1484_148451


namespace value_of_2022_plus_a_minus_b_l1484_148435

theorem value_of_2022_plus_a_minus_b (x a b : ℚ) (h_distinct : x ≠ a ∧ x ≠ b ∧ a ≠ b) 
  (h_gt : a > b) (h_min : ∀ y : ℚ, |y - a| + |y - b| ≥ 2 ∧ |x - a| + |x - b| = 2) :
  2022 + a - b = 2024 := 
by 
  sorry

end value_of_2022_plus_a_minus_b_l1484_148435


namespace length_of_AB_l1484_148423

theorem length_of_AB (A B P Q : ℝ) 
  (hp : 0 < P) (hp' : P < 1) 
  (hq : 0 < Q) (hq' : Q < 1) 
  (H1 : P = 3 / 7) (H2 : Q = 5 / 12)
  (H3 : P * (1 - Q) + Q * (1 - P) = 4) : 
  (B - A) = 336 / 11 :=
by
  sorry

end length_of_AB_l1484_148423


namespace positive_difference_l1484_148459

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l1484_148459


namespace triangle_largest_angle_l1484_148465

theorem triangle_largest_angle 
  (a1 a2 a3 : ℝ) 
  (h_sum : a1 + a2 + a3 = 180)
  (h_arith_seq : 2 * a2 = a1 + a3)
  (h_one_angle : a1 = 28) : 
  max a1 (max a2 a3) = 92 := 
by
  sorry

end triangle_largest_angle_l1484_148465


namespace complement_union_eq_l1484_148458

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_union_eq : (U \ (A ∪ B)) = {6,8} := by
  sorry

end complement_union_eq_l1484_148458


namespace annie_serious_accident_probability_l1484_148467

theorem annie_serious_accident_probability :
  (∀ temperature : ℝ, temperature < 32 → ∃ skid_chance_increase : ℝ, skid_chance_increase = 5 * ⌊ (32 - temperature) / 3 ⌋ / 100) →
  (∀ control_regain_chance : ℝ, control_regain_chance = 0.4) →
  (∀ control_loss_chance : ℝ, control_loss_chance = 1 - control_regain_chance) →
  (temperature = 8) →
  (serious_accident_probability = skid_chance_increase * control_loss_chance) →
  serious_accident_probability = 0.24 := by
  sorry

end annie_serious_accident_probability_l1484_148467


namespace remainder_when_divided_by_2000_l1484_148406

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end remainder_when_divided_by_2000_l1484_148406


namespace correct_answer_l1484_148471

noncomputable def sqrt_2 : ℝ := Real.sqrt 2

def P : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem correct_answer : {sqrt_2} ⊆ P :=
sorry

end correct_answer_l1484_148471


namespace find_k_l1484_148438

theorem find_k (k : ℕ) : (1 / 2) ^ 16 * (1 / 81) ^ k = 1 / 18 ^ 16 → k = 8 :=
by
  intro h
  sorry

end find_k_l1484_148438
