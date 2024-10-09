import Mathlib

namespace Cooper_age_l1693_169354

variable (X : ℕ)
variable (Dante : ℕ)
variable (Maria : ℕ)

theorem Cooper_age (h1 : Dante = 2 * X) (h2 : Maria = 2 * X + 1) (h3 : X + Dante + Maria = 31) : X = 6 :=
by
  -- Proof is omitted as indicated
  sorry

end Cooper_age_l1693_169354


namespace total_pokemon_cards_l1693_169307

-- Definitions based on conditions
def jenny_cards : ℕ := 6
def orlando_cards : ℕ := jenny_cards + 2
def richard_cards : ℕ := 3 * orlando_cards

-- The theorem stating the total number of cards
theorem total_pokemon_cards : jenny_cards + orlando_cards + richard_cards = 38 :=
by
  sorry

end total_pokemon_cards_l1693_169307


namespace find_abc_of_N_l1693_169311

theorem find_abc_of_N :
  ∃ N : ℕ, (N % 10000) = (N + 2) % 10000 ∧ 
            (N % 16 = 15 ∧ (N + 2) % 16 = 1) ∧ 
            ∃ abc : ℕ, (100 ≤ abc ∧ abc < 1000) ∧ 
            (N % 1000) = 100 * abc + 99 := sorry

end find_abc_of_N_l1693_169311


namespace paidAmount_Y_l1693_169322

theorem paidAmount_Y (X Y : ℝ) (h1 : X + Y = 638) (h2 : X = 1.2 * Y) : Y = 290 :=
by
  sorry

end paidAmount_Y_l1693_169322


namespace angle_complement_supplement_l1693_169332

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l1693_169332


namespace math_competition_rankings_l1693_169314

noncomputable def rankings (n : ℕ) : ℕ → Prop := sorry

theorem math_competition_rankings :
  (∀ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    
    -- A's guesses
    (rankings A 1 → rankings B 3 ∧ rankings C 5) →
    -- B's guesses
    (rankings B 2 → rankings E 4 ∧ rankings D 5) →
    -- C's guesses
    (rankings C 3 → rankings A 1 ∧ rankings E 4) →
    -- D's guesses
    (rankings D 4 → rankings C 1 ∧ rankings D 2) →
    -- E's guesses
    (rankings E 5 → rankings A 3 ∧ rankings D 4) →
    -- Condition that each position is guessed correctly by someone
    (∃ i, rankings A i) ∧
    (∃ i, rankings B i) ∧
    (∃ i, rankings C i) ∧
    (∃ i, rankings D i) ∧
    (∃ i, rankings E i) →
    
    -- The actual placing according to derived solution
    rankings A 1 ∧ 
    rankings D 2 ∧ 
    rankings B 3 ∧ 
    rankings E 4 ∧ 
    rankings C 5) :=
sorry

end math_competition_rankings_l1693_169314


namespace pipe_A_fill_time_l1693_169306

theorem pipe_A_fill_time (B C : ℝ) (hB : B = 8) (hC : C = 14.4) (hB_not_zero : B ≠ 0) (hC_not_zero : C ≠ 0) :
  ∃ (A : ℝ), (1 / A + 1 / B = 1 / C) ∧ A = 24 :=
by
  sorry

end pipe_A_fill_time_l1693_169306


namespace f_50_value_l1693_169359

def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9 * x^2 - 15 * x

theorem f_50_value : f 50 = 146 :=
by
  sorry

end f_50_value_l1693_169359


namespace find_k_l1693_169331

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l1693_169331


namespace largest_m_l1693_169305

noncomputable def max_min_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : ℝ :=
  min (a * b) (min (b * c) (c * a))

theorem largest_m (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : max_min_ab_bc_ca a b c ha hb hc h1 h2 = 6.75 :=
by
  sorry

end largest_m_l1693_169305


namespace sum_of_coefficients_is_zero_l1693_169375

noncomputable def expansion : Polynomial ℚ := (Polynomial.X^2 + Polynomial.X + 1) * (2*Polynomial.X - 2)^5

theorem sum_of_coefficients_is_zero :
  (expansion.coeff 0) + (expansion.coeff 1) + (expansion.coeff 2) + (expansion.coeff 3) + 
  (expansion.coeff 4) + (expansion.coeff 5) + (expansion.coeff 6) + (expansion.coeff 7) = 0 :=
by
  sorry

end sum_of_coefficients_is_zero_l1693_169375


namespace revenue_from_full_price_tickets_l1693_169319

theorem revenue_from_full_price_tickets (f h p : ℕ) 
    (h1 : f + h = 160) 
    (h2 : f * p + h * (p / 2) = 2400) 
    (h3 : h = 160 - f)
    (h4 : 2 * 2400 = 4800) :
  f * p = 800 := 
sorry

end revenue_from_full_price_tickets_l1693_169319


namespace largest_whole_number_l1693_169368

theorem largest_whole_number (x : ℕ) : 9 * x < 150 → x ≤ 16 :=
by sorry

end largest_whole_number_l1693_169368


namespace cars_in_first_section_l1693_169392

noncomputable def first_section_rows : ℕ := 15
noncomputable def first_section_cars_per_row : ℕ := 10
noncomputable def total_cars_first_section : ℕ := first_section_rows * first_section_cars_per_row

theorem cars_in_first_section : total_cars_first_section = 150 :=
by
  sorry

end cars_in_first_section_l1693_169392


namespace man_l1693_169373

noncomputable def speed_in_still_water (current_speed_kmph : ℝ) (distance_m : ℝ) (time_seconds : ℝ) : ℝ :=
   let current_speed_mps := current_speed_kmph * 1000 / 3600
   let downstream_speed_mps := distance_m / time_seconds
   let still_water_speed_mps := downstream_speed_mps - current_speed_mps
   let still_water_speed_kmph := still_water_speed_mps * 3600 / 1000
   still_water_speed_kmph

theorem man's_speed_in_still_water :
  speed_in_still_water 6 100 14.998800095992323 = 18 := by
  sorry

end man_l1693_169373


namespace cost_of_figurine_l1693_169358

noncomputable def cost_per_tv : ℝ := 50
noncomputable def num_tvs : ℕ := 5
noncomputable def num_figurines : ℕ := 10
noncomputable def total_spent : ℝ := 260

theorem cost_of_figurine : 
  ((total_spent - (num_tvs * cost_per_tv)) / num_figurines) = 1 := 
by
  sorry

end cost_of_figurine_l1693_169358


namespace circles_intersect_l1693_169360

def circle1 := { x : ℝ × ℝ | (x.1 - 1)^2 + (x.2 + 2)^2 = 1 }
def circle2 := { x : ℝ × ℝ | (x.1 - 2)^2 + (x.2 + 1)^2 = 1 / 4 }

theorem circles_intersect :
  ∃ x : ℝ × ℝ, x ∈ circle1 ∧ x ∈ circle2 :=
sorry

end circles_intersect_l1693_169360


namespace three_pipes_time_l1693_169371

variable (R : ℝ) (T : ℝ)

-- Condition: Two pipes fill the tank in 18 hours
def two_pipes_fill : Prop := 2 * R * 18 = 1

-- Question: How long does it take for three pipes to fill the tank?
def three_pipes_fill : Prop := 3 * R * T = 1

theorem three_pipes_time (h : two_pipes_fill R) : three_pipes_fill R 12 :=
by
  sorry

end three_pipes_time_l1693_169371


namespace infinite_solutions_ax2_by2_eq_z3_l1693_169301

theorem infinite_solutions_ax2_by2_eq_z3 
  (a b : ℤ) 
  (coprime_ab : Int.gcd a b = 1) :
  ∃ (x y z : ℤ), (∀ n : ℤ, ∃ (x y z : ℤ), a * x^2 + b * y^2 = z^3 
  ∧ Int.gcd x y = 1) := 
sorry

end infinite_solutions_ax2_by2_eq_z3_l1693_169301


namespace line_through_intersection_and_origin_l1693_169313

theorem line_through_intersection_and_origin :
  ∃ (x y : ℝ), (2*x + y = 3) ∧ (x + 4*y = 2) ∧ (x - 10*y = 0) :=
by
  sorry

end line_through_intersection_and_origin_l1693_169313


namespace alex_cakes_l1693_169323

theorem alex_cakes :
  let slices_first_cake := 8
  let slices_second_cake := 12
  let given_away_friends_first := slices_first_cake / 4
  let remaining_after_friends_first := slices_first_cake - given_away_friends_first
  let given_away_family_first := remaining_after_friends_first / 2
  let remaining_after_family_first := remaining_after_friends_first - given_away_family_first
  let stored_in_freezer_first := remaining_after_family_first / 4
  let remaining_after_freezer_first := remaining_after_family_first - stored_in_freezer_first
  let remaining_after_eating_first := remaining_after_freezer_first - 2
  
  let given_away_friends_second := slices_second_cake / 3
  let remaining_after_friends_second := slices_second_cake - given_away_friends_second
  let given_away_family_second := remaining_after_friends_second / 6
  let remaining_after_family_second := remaining_after_friends_second - given_away_family_second
  let stored_in_freezer_second := remaining_after_family_second / 4
  let remaining_after_freezer_second := remaining_after_family_second - stored_in_freezer_second
  let remaining_after_eating_second := remaining_after_freezer_second - 1

  remaining_after_eating_first + stored_in_freezer_first + remaining_after_eating_second + stored_in_freezer_second = 7 :=
by
  -- Proof goes here
  sorry

end alex_cakes_l1693_169323


namespace evaluate_expression_l1693_169367

noncomputable def f : ℝ → ℝ := sorry

lemma f_condition (a : ℝ) : f (a + 1) = f a * f 1 := sorry

lemma f_one : f 1 = 2 := sorry

theorem evaluate_expression :
  (f 2018 / f 2017) + (f 2019 / f 2018) + (f 2020 / f 2019) = 6 :=
sorry

end evaluate_expression_l1693_169367


namespace dogwood_tree_count_l1693_169336

theorem dogwood_tree_count (n d1 d2 d3 d4 d5: ℕ) 
  (h1: n = 39)
  (h2: d1 = 24)
  (h3: d2 = d1 / 2)
  (h4: d3 = 4 * d2)
  (h5: d4 = 5)
  (h6: d5 = 15):
  n + d1 + d2 + d3 + d4 + d5 = 143 :=
by
  sorry

end dogwood_tree_count_l1693_169336


namespace time_difference_between_car_and_minivan_arrival_l1693_169343

variable (car_speed : ℝ := 40)
variable (minivan_speed : ℝ := 50)
variable (pass_time : ℝ := 1 / 6) -- in hours

theorem time_difference_between_car_and_minivan_arrival :
  (60 * (1 / 6 - (20 / 3 / 50))) = 2 := sorry

end time_difference_between_car_and_minivan_arrival_l1693_169343


namespace florist_initial_roses_l1693_169398

theorem florist_initial_roses : 
  ∀ (R : ℕ), (R - 16 + 19 = 40) → (R = 37) :=
by
  intro R
  intro h
  sorry

end florist_initial_roses_l1693_169398


namespace complementary_angles_positive_difference_l1693_169335

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l1693_169335


namespace twenty_four_point_game_l1693_169349

theorem twenty_four_point_game : (9 + 7) * 3 / 2 = 24 := by
  sorry -- Proof to be provided

end twenty_four_point_game_l1693_169349


namespace part1_monotonicity_part2_find_range_l1693_169318

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l1693_169318


namespace cab_company_charge_l1693_169329

-- Defining the conditions
def total_cost : ℝ := 23
def base_price : ℝ := 3
def distance_to_hospital : ℝ := 5

-- Theorem stating the cost per mile
theorem cab_company_charge : 
  (total_cost - base_price) / distance_to_hospital = 4 :=
by
  -- Proof is omitted
  sorry

end cab_company_charge_l1693_169329


namespace inequality_solution_set_l1693_169334

theorem inequality_solution_set :
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} := 
sorry

end inequality_solution_set_l1693_169334


namespace slope_of_tangent_at_1_0_l1693_169387

noncomputable def f (x : ℝ) : ℝ :=
2 * x^2 - 2 * x

def derivative_f (x : ℝ) : ℝ :=
4 * x - 2

theorem slope_of_tangent_at_1_0 : derivative_f 1 = 2 :=
by
  sorry

end slope_of_tangent_at_1_0_l1693_169387


namespace k_range_l1693_169362

theorem k_range (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 0 ≤ 2 * x - 2 * k) → k ≤ 1 :=
by
  intro h
  have h1 := h 1 (by simp)
  have h3 := h 3 (by simp)
  sorry

end k_range_l1693_169362


namespace range_of_a_for_inequality_l1693_169310

theorem range_of_a_for_inequality : 
  ∃ a : ℝ, (∀ x : ℤ, (a * x - 1) ^ 2 < x ^ 2) ↔ 
    (a > -3 / 2 ∧ a ≤ -4 / 3) ∨ (4 / 3 ≤ a ∧ a < 3 / 2) :=
by
  sorry

end range_of_a_for_inequality_l1693_169310


namespace quadratic_real_roots_l1693_169302

variable (a b : ℝ)

theorem quadratic_real_roots (h : ∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) : b ≤ -1/8 :=
by
  sorry

end quadratic_real_roots_l1693_169302


namespace geometric_sequence_term_l1693_169363

noncomputable def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_term {a : ℕ → ℤ} {q : ℤ}
  (h1 : geometric_sequence a q)
  (h2 : a 7 = 10)
  (h3 : q = -2) :
  a 10 = -80 :=
by
  sorry

end geometric_sequence_term_l1693_169363


namespace wholesale_cost_proof_l1693_169393

-- Definitions based on conditions
def wholesale_cost (W : ℝ) := W
def retail_price (W : ℝ) := 1.20 * W
def employee_paid (R : ℝ) := 0.90 * R

-- Theorem statement: given the conditions, prove that the wholesale cost is $200.
theorem wholesale_cost_proof : 
  ∃ W : ℝ, (retail_price W = 1.20 * W) ∧ (employee_paid (retail_price W) = 216) ∧ W = 200 :=
by 
  let W := 200
  have hp : retail_price W = 1.20 * W := by sorry
  have ep : employee_paid (retail_price W) = 216 := by sorry
  exact ⟨W, hp, ep, rfl⟩

end wholesale_cost_proof_l1693_169393


namespace system_solutions_l1693_169366

theorem system_solutions : {p : ℝ × ℝ | p.snd ^ 2 = p.fst ∧ p.snd = p.fst} = {⟨1, 1⟩, ⟨0, 0⟩} :=
by
  sorry

end system_solutions_l1693_169366


namespace find_inscription_l1693_169309

-- Definitions for the conditions
def identical_inscriptions (box1 box2 : String) : Prop :=
  box1 = box2

def conclusion_same_master (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini"

def cannot_identify_master (box : String) : Prop :=
  ¬(∀ (made_by : String → Prop), made_by "Bellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Cellini")

def single_casket_indeterminate (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini")

-- Inscription on the boxes
def inscription := "At least one of these boxes was made by Cellini's son."

-- The Lean statement for the proof
theorem find_inscription (box1 box2 : String)
  (h1 : identical_inscriptions box1 box2)
  (h2 : conclusion_same_master box1)
  (h3 : cannot_identify_master box1)
  (h4 : single_casket_indeterminate box1) :
  box1 = inscription :=
sorry

end find_inscription_l1693_169309


namespace certain_event_l1693_169380

-- Definitions for a line and plane
inductive Line
| mk : Line

inductive Plane
| mk : Plane

-- Definitions for parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p₁ p₂ : Plane) : Prop := sorry

-- Given conditions and the proof statement
theorem certain_event (l : Line) (α β : Plane) (h1 : perpendicular l α) (h2 : perpendicular l β) : plane_parallel α β :=
sorry

end certain_event_l1693_169380


namespace travel_time_l1693_169365

theorem travel_time (speed distance time : ℕ) (h_speed : speed = 60) (h_distance : distance = 180) : 
  time = distance / speed → time = 3 := by
  sorry

end travel_time_l1693_169365


namespace find_x_value_l1693_169348

noncomputable def solve_some_number (x : ℝ) : Prop :=
  let expr := (x - (8 / 7) * 5 + 10)
  expr = 13.285714285714286

theorem find_x_value : ∃ x : ℝ, solve_some_number x ∧ x = 9 := by
  sorry

end find_x_value_l1693_169348


namespace abs_diff_kth_power_l1693_169347

theorem abs_diff_kth_power (k : ℕ) (a b : ℤ) (x y : ℤ)
  (hk : 2 ≤ k)
  (ha : a ≠ 0) (hb : b ≠ 0)
  (hab_odd : (a + b) % 2 = 1)
  (hxy : 0 < |x - y| ∧ |x - y| ≤ 2)
  (h_eq : a^k * x - b^k * y = a - b) :
  ∃ m : ℤ, |a - b| = m^k :=
sorry

end abs_diff_kth_power_l1693_169347


namespace quadratic_has_two_distinct_roots_l1693_169321

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end quadratic_has_two_distinct_roots_l1693_169321


namespace trip_total_time_l1693_169337

theorem trip_total_time 
  (x : ℕ) 
  (h1 : 30 * 5 = 150) 
  (h2 : 42 * x + 150 = 38 * (x + 5)) 
  (h3 : 38 = (150 + 42 * x) / (5 + x)) : 
  5 + x = 15 := by
  sorry

end trip_total_time_l1693_169337


namespace smallest_n_divisible_by_2022_l1693_169304

theorem smallest_n_divisible_by_2022 (n : ℕ) (h1 : n > 1) (h2 : (n^7 - 1) % 2022 = 0) : n = 79 :=
sorry

end smallest_n_divisible_by_2022_l1693_169304


namespace percentage_increase_in_population_due_to_birth_is_55_l1693_169378

/-- The initial population at the start of the period is 100,000 people. -/
def initial_population : ℕ := 100000

/-- The period of observation is 10 years. -/
def period : ℕ := 10

/-- The number of people leaving the area each year due to emigration is 2000. -/
def emigration_per_year : ℕ := 2000

/-- The number of people coming into the area each year due to immigration is 2500. -/
def immigration_per_year : ℕ := 2500

/-- The population at the end of the period is 165,000 people. -/
def final_population : ℕ := 165000

/-- The net migration per year is calculated by subtracting emigration from immigration. -/
def net_migration_per_year : ℕ := immigration_per_year - emigration_per_year

/-- The total net migration over the period is obtained by multiplying net migration per year by the number of years. -/
def net_migration_over_period : ℕ := net_migration_per_year * period

/-- The total population increase is the difference between the final and initial population. -/
def total_population_increase : ℕ := final_population - initial_population

/-- The increase in population due to birth is calculated by subtracting net migration over the period from the total population increase. -/
def increase_due_to_birth : ℕ := total_population_increase - net_migration_over_period

/-- The percentage increase in population due to birth is calculated by dividing the increase due to birth by the initial population, and then multiplying by 100 to convert to percentage. -/
def percentage_increase_due_to_birth : ℕ := (increase_due_to_birth * 100) / initial_population

/-- The final Lean statement to prove. -/
theorem percentage_increase_in_population_due_to_birth_is_55 :
  percentage_increase_due_to_birth = 55 := by
sorry

end percentage_increase_in_population_due_to_birth_is_55_l1693_169378


namespace james_sheets_of_paper_l1693_169352

noncomputable def sheets_of_paper (books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) : ℕ :=
  (books * pages_per_book) / (pages_per_side * sides_per_sheet)

theorem james_sheets_of_paper :
  sheets_of_paper 2 600 4 2 = 150 :=
by
  sorry

end james_sheets_of_paper_l1693_169352


namespace LanceCents_l1693_169356

noncomputable def MargaretCents : ℕ := 75
noncomputable def GuyCents : ℕ := 60
noncomputable def BillCents : ℕ := 60
noncomputable def TotalCents : ℕ := 265

theorem LanceCents (lanceCents : ℕ) :
  MargaretCents + GuyCents + BillCents + lanceCents = TotalCents → lanceCents = 70 :=
by
  intros
  sorry

end LanceCents_l1693_169356


namespace find_y_l1693_169339

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 12) (h2 : x = 6) : y = 3 :=
by
  sorry

end find_y_l1693_169339


namespace total_games_played_l1693_169345

-- Define the number of teams
def num_teams : ℕ := 12

-- Define the number of games each team plays with each other team
def games_per_pair : ℕ := 4

-- The theorem stating the total number of games played
theorem total_games_played : num_teams * (num_teams - 1) / 2 * games_per_pair = 264 :=
by
  sorry

end total_games_played_l1693_169345


namespace max_x2_plus_4y_plus_3_l1693_169374

theorem max_x2_plus_4y_plus_3 
  (x y : ℝ) 
  (h : x^2 + y^2 = 1) : 
  x^2 + 4*y + 3 ≤ 7 := sorry

end max_x2_plus_4y_plus_3_l1693_169374


namespace prove_ax5_by5_l1693_169353

variables {a b x y : ℝ}

theorem prove_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 30)
                      (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 :=
sorry

end prove_ax5_by5_l1693_169353


namespace smallest_positive_number_is_correct_l1693_169328

noncomputable def smallest_positive_number : ℝ := 20 - 5 * Real.sqrt 15

theorem smallest_positive_number_is_correct :
  ∀ n,
    (n = 12 - 3 * Real.sqrt 12 ∨ n = 3 * Real.sqrt 12 - 11 ∨ n = 20 - 5 * Real.sqrt 15 ∨ n = 55 - 11 * Real.sqrt 30 ∨ n = 11 * Real.sqrt 30 - 55) →
    n > 0 → smallest_positive_number ≤ n :=
by
  sorry

end smallest_positive_number_is_correct_l1693_169328


namespace find_multiple_of_sum_l1693_169355

-- Define the conditions and the problem statement in Lean
theorem find_multiple_of_sum (a b m : ℤ) 
  (h1 : b = 8) 
  (h2 : b - a = 3) 
  (h3 : a * b = 14 + m * (a + b)) : 
  m = 2 :=
by
  sorry

end find_multiple_of_sum_l1693_169355


namespace defective_units_shipped_for_sale_l1693_169340

theorem defective_units_shipped_for_sale (d p : ℝ) (h1 : d = 0.09) (h2 : p = 0.04) : (d * p * 100 = 0.36) :=
by 
  -- Assuming some calculation steps 
  sorry

end defective_units_shipped_for_sale_l1693_169340


namespace a_1995_eq_l1693_169350

def a_3 : ℚ := (2 + 3) / (1 + 6)

def a (n : ℕ) : ℚ :=
  if n = 3 then a_3
  else if n ≥ 4 then
    let a_n_minus_1 := a (n - 1)
    (a_n_minus_1 + n) / (1 + n * a_n_minus_1)
  else
    0 -- We only care about n ≥ 3 in this problem

-- The problem itself
theorem a_1995_eq :
  a 1995 = 1991009 / 1991011 :=
by
  sorry

end a_1995_eq_l1693_169350


namespace matrix_sum_correct_l1693_169396

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![1, 2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-5, -7],
  ![4, -9]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, -7],
  ![5, -7]
]

theorem matrix_sum_correct : A + B = C := by 
  sorry

end matrix_sum_correct_l1693_169396


namespace age_of_15th_student_l1693_169390

theorem age_of_15th_student (T : ℕ) (T8 : ℕ) (T6 : ℕ)
  (avg_15_students : T / 15 = 15)
  (avg_8_students : T8 / 8 = 14)
  (avg_6_students : T6 / 6 = 16) :
  (T - (T8 + T6)) = 17 := by
  sorry

end age_of_15th_student_l1693_169390


namespace smallest_y_l1693_169342

noncomputable def x : ℕ := 3 * 40 * 75

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k^3 = n

theorem smallest_y (y : ℕ) (hy : y = 3) :
  ∀ (x : ℕ), x = 3 * 40 * 75 → is_perfect_cube (x * y) :=
by
  intro x hx
  unfold is_perfect_cube
  exists 5 -- This is just a placeholder value; the proof would find the correct k
  sorry

end smallest_y_l1693_169342


namespace maximum_value_of_expression_l1693_169369

noncomputable def max_function_value (x y z : ℝ) : ℝ := 
  (x^3 - x * y^2 + y^3) * (x^3 - x * z^2 + z^3) * (y^3 - y * z^2 + z^3)

theorem maximum_value_of_expression : 
  ∃ x y z : ℝ, (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 3) 
  ∧ max_function_value x y z = 2916 / 2187 := 
sorry

end maximum_value_of_expression_l1693_169369


namespace sheila_hourly_wage_l1693_169357

-- Definition of conditions
def hours_per_day_mon_wed_fri := 8
def days_mon_wed_fri := 3
def hours_per_day_tue_thu := 6
def days_tue_thu := 2
def weekly_earnings := 432

-- Variables derived from conditions
def total_hours_mon_wed_fri := hours_per_day_mon_wed_fri * days_mon_wed_fri
def total_hours_tue_thu := hours_per_day_tue_thu * days_tue_thu
def total_hours_per_week := total_hours_mon_wed_fri + total_hours_tue_thu

-- Proof statement
theorem sheila_hourly_wage : (weekly_earnings / total_hours_per_week) = 12 := 
sorry

end sheila_hourly_wage_l1693_169357


namespace contrapositive_of_proposition_l1693_169382

-- Proposition: If xy=0, then x=0
def proposition (x y : ℝ) : Prop := x * y = 0 → x = 0

-- Contrapositive: If x ≠ 0, then xy ≠ 0
def contrapositive (x y : ℝ) : Prop := x ≠ 0 → x * y ≠ 0

-- Proof that contrapositive of the given proposition holds
theorem contrapositive_of_proposition (x y : ℝ) : proposition x y ↔ contrapositive x y :=
by {
  sorry
}

end contrapositive_of_proposition_l1693_169382


namespace race_distance_l1693_169395

variables (a b c d : ℝ)
variables (h1 : d / a = (d - 30) / b)
variables (h2 : d / b = (d - 15) / c)
variables (h3 : d / a = (d - 40) / c)

theorem race_distance : d = 90 :=
by 
  sorry

end race_distance_l1693_169395


namespace cost_price_of_watch_l1693_169370

variable (CP SP1 SP2 : ℝ)

theorem cost_price_of_watch (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.04 * CP)
  (h3 : SP2 = SP1 + 200) : CP = 10000 / 7 := 
by
  sorry

end cost_price_of_watch_l1693_169370


namespace region_Z_probability_l1693_169399

variable (P : Type) [Field P]
variable (P_X P_Y P_W P_Z : P)

theorem region_Z_probability :
  P_X = 1 / 3 → P_Y = 1 / 4 → P_W = 1 / 6 → P_X + P_Y + P_Z + P_W = 1 → P_Z = 1 / 4 := by
  sorry

end region_Z_probability_l1693_169399


namespace max_dot_product_on_circle_l1693_169364

theorem max_dot_product_on_circle :
  (∃(x y : ℝ),
    x^2 + (y - 3)^2 = 1 ∧
    2 ≤ y ∧ y ≤ 4 ∧
    (∀(y : ℝ), (2 ≤ y ∧ y ≤ 4 →
      (x^2 + y^2 - 4) ≤ 12))) := by
  sorry

end max_dot_product_on_circle_l1693_169364


namespace largest_of_three_consecutive_integers_l1693_169381

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l1693_169381


namespace problem_2023_divisible_by_consecutive_integers_l1693_169333

theorem problem_2023_divisible_by_consecutive_integers :
  ∃ (n : ℕ), (n = 2022 ∨ n = 2023 ∨ n = 2024) ∧ (2023^2023 - 2023^2021) % n = 0 :=
sorry

end problem_2023_divisible_by_consecutive_integers_l1693_169333


namespace algebraic_expression_evaluation_l1693_169379

theorem algebraic_expression_evaluation (a b : ℝ) (h : 1 / a + 1 / (2 * b) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1 / 2 := 
by
  sorry

end algebraic_expression_evaluation_l1693_169379


namespace find_m_l1693_169316

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end find_m_l1693_169316


namespace daily_salary_of_manager_l1693_169385

theorem daily_salary_of_manager
  (M : ℕ)
  (salary_clerk : ℕ)
  (num_managers : ℕ)
  (num_clerks : ℕ)
  (total_salary : ℕ)
  (h1 : salary_clerk = 2)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16)
  (h5 : 2 * M + 3 * salary_clerk = total_salary) :
  M = 5 := 
  sorry

end daily_salary_of_manager_l1693_169385


namespace discount_is_10_percent_l1693_169315

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end discount_is_10_percent_l1693_169315


namespace sum_digits_of_3n_l1693_169391

noncomputable def sum_digits (n : ℕ) : ℕ :=
sorry  -- Placeholder for a proper implementation of sum_digits

theorem sum_digits_of_3n (n : ℕ) 
  (h1 : sum_digits n = 100) 
  (h2 : sum_digits (44 * n) = 800) : 
  sum_digits (3 * n) = 300 := 
by
  sorry

end sum_digits_of_3n_l1693_169391


namespace weight_of_replaced_person_is_correct_l1693_169327

-- Define a constant representing the number of persons in the group.
def num_people : ℕ := 10
-- Define a constant representing the weight of the new person.
def new_person_weight : ℝ := 110
-- Define a constant representing the increase in average weight when the new person joins.
def avg_weight_increase : ℝ := 5
-- Define the weight of the person who was replaced.
noncomputable def replaced_person_weight : ℝ :=
  new_person_weight - num_people * avg_weight_increase

-- Prove that the weight of the replaced person is 60 kg.
theorem weight_of_replaced_person_is_correct : replaced_person_weight = 60 :=
by
  -- Skip the detailed proof steps.
  sorry

end weight_of_replaced_person_is_correct_l1693_169327


namespace no_prize_for_A_l1693_169394

variable (A B C D : Prop)

theorem no_prize_for_A 
  (hA : A → B) 
  (hB : B → C) 
  (hC : ¬D → ¬C) 
  (exactly_one_did_not_win : (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)) 
: ¬A := 
sorry

end no_prize_for_A_l1693_169394


namespace min_period_f_and_max_value_g_l1693_169312

open Real

noncomputable def f (x : ℝ) : ℝ := abs (sin x) + abs (cos x)
noncomputable def g (x : ℝ) : ℝ := sin x ^ 3 - sin x

theorem min_period_f_and_max_value_g :
  (∀ m : ℝ, (∀ x : ℝ, f (x + m) = f x) -> m = π / 2) ∧ 
  (∃ n : ℝ, ∀ x : ℝ, g x ≤ n ∧ (∃ x : ℝ, g x = n)) ∧ 
  (∃ mn : ℝ, mn = (π / 2) * (2 * sqrt 3 / 9)) := 
by sorry

end min_period_f_and_max_value_g_l1693_169312


namespace philip_farm_animal_count_l1693_169372

def number_of_cows : ℕ := 20

def number_of_ducks : ℕ := number_of_cows * 3 / 2

def total_cows_and_ducks : ℕ := number_of_cows + number_of_ducks

def number_of_pigs : ℕ := total_cows_and_ducks / 5

def total_animals : ℕ := total_cows_and_ducks + number_of_pigs

theorem philip_farm_animal_count : total_animals = 60 := by
  sorry

end philip_farm_animal_count_l1693_169372


namespace grasshopper_jump_distance_l1693_169388

theorem grasshopper_jump_distance (g f m : ℕ)
    (h1 : f = g + 32)
    (h2 : m = f - 26)
    (h3 : m = 31) : g = 25 :=
by
  sorry

end grasshopper_jump_distance_l1693_169388


namespace basketball_game_points_l1693_169397

variable (J T K : ℕ)

theorem basketball_game_points (h1 : T = J + 20) (h2 : J + T + K = 100) (h3 : T = 30) : 
  T / K = 1 / 2 :=
by sorry

end basketball_game_points_l1693_169397


namespace value_computation_l1693_169317

theorem value_computation (N : ℝ) (h1 : 1.20 * N = 2400) : 0.20 * N = 400 := 
by
  sorry

end value_computation_l1693_169317


namespace commute_time_l1693_169308

theorem commute_time (d w t : ℝ) (x : ℝ) (h_distance : d = 1.5) (h_walking_speed : w = 3) (h_train_speed : t = 20)
  (h_extra_time : 30 = 4.5 + x + 2) : x = 25.5 :=
by {
  -- Add the statement of the proof
  sorry
}

end commute_time_l1693_169308


namespace least_k_divisible_by_2160_l1693_169300

theorem least_k_divisible_by_2160 (k : ℤ) : k^3 ∣ 2160 → k ≥ 60 := by
  sorry

end least_k_divisible_by_2160_l1693_169300


namespace inequality_abc_l1693_169384

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end inequality_abc_l1693_169384


namespace average_speed_of_rocket_l1693_169389

def distance_soared (speed_soaring : ℕ) (time_soaring : ℕ) : ℕ :=
  speed_soaring * time_soaring

def distance_plummeted : ℕ := 600

def total_distance (distance_soared : ℕ) (distance_plummeted : ℕ) : ℕ :=
  distance_soared + distance_plummeted

def total_time (time_soaring : ℕ) (time_plummeting : ℕ) : ℕ :=
  time_soaring + time_plummeting

def average_speed (total_distance : ℕ) (total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_rocket :
  let speed_soaring := 150
  let time_soaring := 12
  let time_plummeting := 3
  distance_soared speed_soaring time_soaring +
  distance_plummeted = 2400
  →
  total_time time_soaring time_plummeting = 15
  →
  average_speed (distance_soared speed_soaring time_soaring + distance_plummeted)
                (total_time time_soaring time_plummeting) = 160 :=
by
  sorry

end average_speed_of_rocket_l1693_169389


namespace set_intersection_complement_l1693_169325

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

theorem set_intersection_complement :
  ((U \ A) ∩ B) = {1, 3, 7} :=
by
  sorry

end set_intersection_complement_l1693_169325


namespace height_difference_l1693_169320

-- Definitions of the terms and conditions
variables {b h : ℝ} -- base and height of Triangle B
variables {b' h' : ℝ} -- base and height of Triangle A

-- Given conditions:
-- Triangle A's base is 10% greater than Triangle B's base
def base_relation (b' : ℝ) (b : ℝ) := b' = 1.10 * b

-- The area of Triangle A is 1% less than the area of Triangle B
def area_relation (b h b' h' : ℝ) := (1 / 2) * b' * h' = (1 / 2) * b * h - 0.01 * (1 / 2) * b * h

-- Proof statement
theorem height_difference (b h b' h' : ℝ) (H_base: base_relation b' b) (H_area: area_relation b h b' h') :
  h' = 0.9 * h := 
sorry

end height_difference_l1693_169320


namespace total_number_of_fleas_l1693_169351

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end total_number_of_fleas_l1693_169351


namespace cactus_species_minimum_l1693_169386

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l1693_169386


namespace find_value_l1693_169344

variable (N : ℝ)

def condition : Prop := (1 / 4) * (1 / 3) * (2 / 5) * N = 16

theorem find_value (h : condition N) : (1 / 3) * (2 / 5) * N = 64 :=
sorry

end find_value_l1693_169344


namespace tetrahedron_inequality_l1693_169376

theorem tetrahedron_inequality (t1 t2 t3 t4 τ1 τ2 τ3 τ4 : ℝ) 
  (ht1 : t1 > 0) (ht2 : t2 > 0) (ht3 : t3 > 0) (ht4 : t4 > 0)
  (hτ1 : τ1 > 0) (hτ2 : τ2 > 0) (hτ3 : τ3 > 0) (hτ4 : τ4 > 0)
  (sphere_inscribed : ∀ {x y : ℝ}, x > 0 → y > 0 → x^2 / y^2 ≤ (x - 2 * y) ^ 2 / x ^ 2) :
  (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4) ≥ 1 
  ∧ (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4 = 1 ↔ t1 = t2 ∧ t2 = t3 ∧ t3 = t4) := by
  sorry

end tetrahedron_inequality_l1693_169376


namespace fourth_term_geometric_progression_l1693_169330

theorem fourth_term_geometric_progression
  (x : ℝ)
  (h : ∀ n : ℕ, n ≥ 0 → (3 * x * (n : ℝ) + 3 * (n : ℝ)) = (6 * x * ((n - 1) : ℝ) + 6 * ((n - 1) : ℝ))) :
  (((3*x + 3)^2 = (6*x + 6) * x) ∧ x = -3) → (∀ n : ℕ, n = 4 → (2^(n-3) * (6*x + 6)) = -24) :=
by
  sorry

end fourth_term_geometric_progression_l1693_169330


namespace tamia_bell_pepper_pieces_l1693_169341

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l1693_169341


namespace total_payment_leila_should_pay_l1693_169361

-- Definitions of the conditions
def chocolateCakes := 3
def chocolatePrice := 12
def strawberryCakes := 6
def strawberryPrice := 22

-- Mathematical equivalent proof problem
theorem total_payment_leila_should_pay : 
  chocolateCakes * chocolatePrice + strawberryCakes * strawberryPrice = 168 := 
by 
  sorry

end total_payment_leila_should_pay_l1693_169361


namespace relationship_between_a_b_l1693_169303

theorem relationship_between_a_b (a b x : ℝ) (h1 : 2 * x = a + b) (h2 : 2 * x^2 = a^2 - b^2) : 
  a = -b ∨ a = 3 * b :=
  sorry

end relationship_between_a_b_l1693_169303


namespace max_A_k_value_l1693_169346

noncomputable def A_k (k : ℕ) : ℝ := (19^k + 66^k) / k.factorial

theorem max_A_k_value : 
  ∃ k : ℕ, (∀ m : ℕ, (A_k m ≤ A_k k)) ∧ k = 65 :=
by
  sorry

end max_A_k_value_l1693_169346


namespace johnny_money_left_l1693_169383

def total_saved (september october november : ℕ) : ℕ := september + october + november

def money_left (total amount_spent : ℕ) : ℕ := total - amount_spent

theorem johnny_money_left 
    (saved_september : ℕ)
    (saved_october : ℕ)
    (saved_november : ℕ)
    (spent_video_game : ℕ)
    (h1 : saved_september = 30)
    (h2 : saved_october = 49)
    (h3 : saved_november = 46)
    (h4 : spent_video_game = 58) :
    money_left (total_saved saved_september saved_october saved_november) spent_video_game = 67 := 
by sorry

end johnny_money_left_l1693_169383


namespace planting_rate_l1693_169326

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l1693_169326


namespace nearest_integer_to_expression_correct_l1693_169324

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l1693_169324


namespace smallest_bdf_l1693_169377

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l1693_169377


namespace factor_equivalence_l1693_169338

noncomputable def given_expression (x : ℝ) :=
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5)

noncomputable def target_form (x : ℝ) :=
  7 * x^2 * (x + 68 / 7)

theorem factor_equivalence (x : ℝ) : given_expression x = target_form x :=
by
  sorry

end factor_equivalence_l1693_169338
