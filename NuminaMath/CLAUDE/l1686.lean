import Mathlib

namespace alyssa_total_games_l1686_168661

/-- The total number of soccer games Alyssa will attend over three years -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Proof that Alyssa will attend 39 soccer games in total -/
theorem alyssa_total_games :
  total_games 11 13 15 = 39 := by
  sorry

end alyssa_total_games_l1686_168661


namespace negation_and_contrary_l1686_168606

def last_digit (n : ℤ) : ℕ := (n % 10).natAbs

def divisible_by_five (n : ℤ) : Prop := n % 5 = 0

def original_statement : Prop :=
  ∀ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n

theorem negation_and_contrary :
  (¬original_statement ↔ ∃ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) ∧ ¬divisible_by_five n) ∧
  (∀ n : ℤ, (last_digit n ≠ 0 ∧ last_digit n ≠ 5) → ¬divisible_by_five n) :=
sorry

end negation_and_contrary_l1686_168606


namespace game_draw_fraction_l1686_168630

theorem game_draw_fraction (ben_wins tom_wins : ℚ) 
  (h1 : ben_wins = 4/9) 
  (h2 : tom_wins = 1/3) : 
  1 - (ben_wins + tom_wins) = 2/9 := by
sorry

end game_draw_fraction_l1686_168630


namespace students_taking_only_history_l1686_168677

theorem students_taking_only_history (total : ℕ) (history : ℕ) (statistics : ℕ) (physics : ℕ) (chemistry : ℕ)
  (hist_stat : ℕ) (hist_phys : ℕ) (hist_chem : ℕ) (stat_phys : ℕ) (stat_chem : ℕ) (phys_chem : ℕ) (all_four : ℕ)
  (h_total : total = 500)
  (h_history : history = 150)
  (h_statistics : statistics = 130)
  (h_physics : physics = 120)
  (h_chemistry : chemistry = 100)
  (h_hist_stat : hist_stat = 60)
  (h_hist_phys : hist_phys = 50)
  (h_hist_chem : hist_chem = 40)
  (h_stat_phys : stat_phys = 35)
  (h_stat_chem : stat_chem = 30)
  (h_phys_chem : phys_chem = 25)
  (h_all_four : all_four = 20) :
  history - hist_stat - hist_phys - hist_chem + all_four = 20 := by
  sorry

end students_taking_only_history_l1686_168677


namespace simplify_expression_l1686_168695

theorem simplify_expression (x y z : ℝ) :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 30 * x - 10 * z := by
  sorry

end simplify_expression_l1686_168695


namespace fifteenth_term_of_geometric_sequence_l1686_168637

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem fifteenth_term_of_geometric_sequence :
  let a := 5
  let r := (1/2 : ℚ)
  let n := 15
  geometricSequenceTerm a r n = 5/16384 := by
sorry

end fifteenth_term_of_geometric_sequence_l1686_168637


namespace penny_percentage_theorem_l1686_168672

theorem penny_percentage_theorem (initial_pennies : ℕ) 
                                 (old_pennies : ℕ) 
                                 (final_pennies : ℕ) : 
  initial_pennies = 200 →
  old_pennies = 30 →
  final_pennies = 136 →
  (initial_pennies - old_pennies) * (1 - 20 / 100) = final_pennies :=
by
  sorry

end penny_percentage_theorem_l1686_168672


namespace quadratic_equation_consequence_l1686_168659

theorem quadratic_equation_consequence (m : ℝ) (h : m^2 + 2*m - 1 = 0) :
  2*m^2 + 4*m - 3 = -1 := by
  sorry

end quadratic_equation_consequence_l1686_168659


namespace travel_time_is_50_minutes_l1686_168694

/-- Represents a tram system with stations A and B -/
structure TramSystem where
  departure_interval : ℕ  -- Interval between tram departures from A in minutes
  journey_time : ℕ        -- Time for a tram to travel from A to B in minutes

/-- Represents a person cycling from B to A -/
structure Cyclist where
  trams_encountered : ℕ   -- Number of trams encountered during the journey

/-- Calculates the time taken for the cyclist to travel from B to A -/
def travel_time (system : TramSystem) (cyclist : Cyclist) : ℕ :=
  cyclist.trams_encountered * system.departure_interval

/-- Theorem stating the travel time for the given scenario -/
theorem travel_time_is_50_minutes 
  (system : TramSystem) 
  (cyclist : Cyclist) 
  (h1 : system.departure_interval = 5)
  (h2 : system.journey_time = 15)
  (h3 : cyclist.trams_encountered = 10) :
  travel_time system cyclist = 50 := by
  sorry

#eval travel_time ⟨5, 15⟩ ⟨10⟩

end travel_time_is_50_minutes_l1686_168694


namespace normas_cards_l1686_168666

/-- Proves that Norma's total number of cards is 158.0 given the initial and found amounts -/
theorem normas_cards (initial_cards : Real) (found_cards : Real) 
  (h1 : initial_cards = 88.0) 
  (h2 : found_cards = 70.0) : 
  initial_cards + found_cards = 158.0 := by
sorry

end normas_cards_l1686_168666


namespace vector_sum_proof_l1686_168696

/-- Given two 2D vectors a and b, prove that 3a + 4b equals the specified result -/
theorem vector_sum_proof (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-3, 4) → (3 • a + 4 • b : ℝ × ℝ) = (-6, 19) := by
  sorry

end vector_sum_proof_l1686_168696


namespace total_fish_fillets_l1686_168638

theorem total_fish_fillets (team1 team2 team3 : ℕ) 
  (h1 : team1 = 189) 
  (h2 : team2 = 131) 
  (h3 : team3 = 180) : 
  team1 + team2 + team3 = 500 := by
  sorry

end total_fish_fillets_l1686_168638


namespace even_monotone_function_inequality_l1686_168682

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem even_monotone_function_inequality (f : ℝ → ℝ) (m : ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_ineq : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 := by
  sorry

end even_monotone_function_inequality_l1686_168682


namespace rectangle_area_l1686_168689

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 206) : L * B = 2520 := by
  sorry

end rectangle_area_l1686_168689


namespace commodity_price_equality_l1686_168683

/-- The year when commodity X costs 40 cents more than commodity Y -/
def target_year : ℕ := 2007

/-- The base year for price comparison -/
def base_year : ℕ := 2001

/-- The initial price of commodity X in dollars -/
def initial_price_X : ℚ := 4.20

/-- The initial price of commodity Y in dollars -/
def initial_price_Y : ℚ := 4.40

/-- The yearly price increase of commodity X in dollars -/
def price_increase_X : ℚ := 0.30

/-- The yearly price increase of commodity Y in dollars -/
def price_increase_Y : ℚ := 0.20

/-- The price difference between X and Y in the target year, in dollars -/
def price_difference : ℚ := 0.40

theorem commodity_price_equality :
  initial_price_X + price_increase_X * (target_year - base_year : ℚ) =
  initial_price_Y + price_increase_Y * (target_year - base_year : ℚ) + price_difference :=
by sorry

end commodity_price_equality_l1686_168683


namespace probability_of_triangle_formation_l1686_168617

/-- Regular 15-gon with unit circumradius -/
def regular_15gon : Set (ℝ × ℝ) := sorry

/-- Set of all segments in the 15-gon -/
def segments (poly : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

/-- Function to calculate the length of a segment -/
def segment_length (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Predicate to check if three segments form a triangle with positive area -/
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

/-- The total number of ways to choose 3 segments from the 15-gon -/
def total_combinations : ℕ := Nat.choose 105 3

/-- The number of valid triangles formed by three segments -/
def valid_triangles : ℕ := sorry

theorem probability_of_triangle_formation :
  (valid_triangles : ℚ) / total_combinations = 323 / 429 := by sorry

end probability_of_triangle_formation_l1686_168617


namespace smallest_cube_root_integer_l1686_168616

theorem smallest_cube_root_integer (m n : ℕ) (r : ℝ) : 
  (∃ m : ℕ, ∃ r : ℝ, 
    m > 0 ∧ 
    r > 0 ∧ 
    r < 1/1000 ∧ 
    (m : ℝ)^(1/3 : ℝ) = n + r) → 
  n ≥ 19 :=
sorry

end smallest_cube_root_integer_l1686_168616


namespace trader_gain_percentage_l1686_168608

/-- The gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gain : ℕ) : ℚ :=
  (num_gain : ℚ) / (num_sold : ℚ) * 100

/-- Theorem: The trader's gain percentage is 33.33% -/
theorem trader_gain_percentage : 
  ∃ (ε : ℚ), abs (gain_percentage 90 30 - 100/3) < ε ∧ ε < 1/100 := by
  sorry

end trader_gain_percentage_l1686_168608


namespace subset_union_equality_l1686_168645

theorem subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
  (⋃ (i ∈ I), A i) = (⋃ (j ∈ J), A j) := by
sorry

end subset_union_equality_l1686_168645


namespace traffic_light_change_probability_l1686_168613

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_time : ℕ
  change_time : ℕ
  num_changes : ℕ

/-- Calculates the probability of observing a color change -/
def probability_of_change (cycle : TrafficLightCycle) : ℚ :=
  (cycle.change_time * cycle.num_changes : ℚ) / cycle.total_time

/-- Theorem: The probability of observing a color change in the given traffic light cycle is 2/9 -/
theorem traffic_light_change_probability :
  let cycle : TrafficLightCycle := ⟨90, 5, 4⟩
  probability_of_change cycle = 2 / 9 := by
  sorry

end traffic_light_change_probability_l1686_168613


namespace harry_potion_kits_l1686_168684

/-- The number of spellbooks Harry needs to buy -/
def num_spellbooks : ℕ := 5

/-- The cost of one spellbook in gold -/
def cost_spellbook : ℕ := 5

/-- The cost of one potion kit in silver -/
def cost_potion_kit : ℕ := 20

/-- The cost of one owl in gold -/
def cost_owl : ℕ := 28

/-- The number of silver in one gold -/
def silver_per_gold : ℕ := 9

/-- The total amount Harry will pay in silver -/
def total_cost : ℕ := 537

/-- The number of potion kits Harry needs to buy -/
def num_potion_kits : ℕ := (total_cost - (num_spellbooks * cost_spellbook * silver_per_gold + cost_owl * silver_per_gold)) / cost_potion_kit

theorem harry_potion_kits : num_potion_kits = 3 := by
  sorry

end harry_potion_kits_l1686_168684


namespace roy_school_days_l1686_168635

/-- Represents the number of hours Roy spends on sports activities in school each day -/
def daily_sports_hours : ℕ := 2

/-- Represents the number of days Roy missed within a week -/
def missed_days : ℕ := 2

/-- Represents the total hours Roy spent on sports in school for the week -/
def weekly_sports_hours : ℕ := 6

/-- Represents the number of days Roy goes to school in a week -/
def school_days : ℕ := 5

theorem roy_school_days : 
  daily_sports_hours * (school_days - missed_days) = weekly_sports_hours ∧ 
  school_days = 5 := by
  sorry

end roy_school_days_l1686_168635


namespace cornmeal_mixture_proof_l1686_168623

/-- Proves that mixing 40 pounds of cornmeal with soybean meal results in a 280 lb mixture
    that is 13% protein, given that soybean meal is 14% protein and cornmeal is 7% protein. -/
theorem cornmeal_mixture_proof (total_weight : ℝ) (soybean_protein : ℝ) (cornmeal_protein : ℝ)
    (desired_protein : ℝ) (cornmeal_weight : ℝ) :
  total_weight = 280 →
  soybean_protein = 0.14 →
  cornmeal_protein = 0.07 →
  desired_protein = 0.13 →
  cornmeal_weight = 40 →
  let soybean_weight := total_weight - cornmeal_weight
  (soybean_protein * soybean_weight + cornmeal_protein * cornmeal_weight) / total_weight = desired_protein :=
by sorry

end cornmeal_mixture_proof_l1686_168623


namespace identity_is_unique_solution_l1686_168688

/-- A function satisfying the given functional equation for all real numbers -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + f (f y)) = 2 * x + f (f y) - f (f x)

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → (∀ x : ℝ, f x = x) :=
sorry

end identity_is_unique_solution_l1686_168688


namespace hyperbola_asymptote_angle_l1686_168668

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan ((2 * b / a) / (1 - b^2 / a^2)) = π / 4) →
  a / b = 1 + Real.sqrt 2 := by
sorry

end hyperbola_asymptote_angle_l1686_168668


namespace unique_three_digit_number_l1686_168625

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Moves the last digit to the front -/
def ThreeDigitNumber.rotateDigits (n : ThreeDigitNumber) : ThreeDigitNumber :=
  ⟨n.ones, n.hundreds, n.tens, by sorry⟩

theorem unique_three_digit_number :
  ∃! (n : ThreeDigitNumber),
    n.ones = 1 ∧
    (n.toNat - n.rotateDigits.toNat : Int) = (10 * (3 ^ 2) : Int) ∧
    n.toNat = 211 := by sorry

end unique_three_digit_number_l1686_168625


namespace school_pupils_count_school_pupils_count_proof_l1686_168679

theorem school_pupils_count : ℕ → ℕ → ℕ → Prop :=
  fun num_girls girls_boys_diff total_pupils =>
    (num_girls = 868) →
    (girls_boys_diff = 281) →
    (total_pupils = num_girls + (num_girls - girls_boys_diff)) →
    total_pupils = 1455

-- The proof is omitted
theorem school_pupils_count_proof : school_pupils_count 868 281 1455 := by
  sorry

end school_pupils_count_school_pupils_count_proof_l1686_168679


namespace right_triangle_vector_problem_l1686_168605

/-- Given a right-angled triangle ABC where AB is the hypotenuse,
    vector CA = (3, -9), and vector CB = (-3, x), prove that x = -1. -/
theorem right_triangle_vector_problem (A B C : ℝ × ℝ) (x : ℝ) :
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 
    + (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 →
  (C.1 - A.1, C.2 - A.2) = (3, -9) →
  (C.1 - B.1, C.2 - B.2) = (-3, x) →
  x = -1 :=
by sorry

end right_triangle_vector_problem_l1686_168605


namespace round_trip_time_l1686_168602

/-- Proves that given a round trip with specified conditions, the outbound journey takes 180 minutes -/
theorem round_trip_time (speed_out speed_return : ℝ) (total_time : ℝ) : 
  speed_out = 100 →
  speed_return = 150 →
  total_time = 5 →
  (total_time * speed_out * speed_return) / (speed_out + speed_return) / speed_out * 60 = 180 := by
  sorry

end round_trip_time_l1686_168602


namespace complement_M_intersect_N_l1686_168664

-- Define the universe type
inductive Universe : Type
  | a | b | c | d | e

-- Define the sets
def I : Set Universe := {Universe.a, Universe.b, Universe.c, Universe.d, Universe.e}
def M : Set Universe := {Universe.a, Universe.b, Universe.c}
def N : Set Universe := {Universe.b, Universe.d, Universe.e}

-- State the theorem
theorem complement_M_intersect_N :
  (I \ M) ∩ N = {Universe.d, Universe.e} := by
  sorry

end complement_M_intersect_N_l1686_168664


namespace second_integer_value_l1686_168681

/-- Given three consecutive odd integers where the sum of the first and third is 144,
    prove that the second integer is 72. -/
theorem second_integer_value (a b c : ℤ) : 
  (∃ n : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2) →  -- consecutive odd integers
  (a + c = 144) →                            -- sum of first and third is 144
  b = 72 :=                                  -- second integer is 72
by sorry

end second_integer_value_l1686_168681


namespace indeterminate_disjunction_l1686_168624

theorem indeterminate_disjunction (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (r : Prop), r ↔ (p ∨ q) :=
sorry

end indeterminate_disjunction_l1686_168624


namespace mildred_spending_l1686_168699

def total_given : ℕ := 100
def amount_left : ℕ := 40
def candice_spent : ℕ := 35

theorem mildred_spending :
  total_given - amount_left - candice_spent = 25 :=
by sorry

end mildred_spending_l1686_168699


namespace quadratic_function_satisfies_conditions_l1686_168640

def f (x : ℝ) : ℝ := -x^2 + 3

theorem quadratic_function_satisfies_conditions :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  f 0 = 3 := by sorry

end quadratic_function_satisfies_conditions_l1686_168640


namespace park_population_l1686_168680

theorem park_population (lions leopards elephants zebras : ℕ) : 
  lions = 200 →
  lions = 2 * leopards →
  elephants = (lions + leopards) / 2 →
  zebras = elephants + leopards →
  lions + leopards + elephants + zebras = 700 := by
sorry

end park_population_l1686_168680


namespace cookies_taken_theorem_l1686_168601

/-- Calculates the number of cookies taken out in 6 days given the initial count,
    remaining count after 10 days, and assuming equal daily consumption. -/
def cookies_taken_in_six_days (initial_count : ℕ) (remaining_count : ℕ) : ℕ :=
  let total_taken := initial_count - remaining_count
  let daily_taken := total_taken / 10
  6 * daily_taken

/-- Theorem stating that given 150 initial cookies and 45 remaining after 10 days,
    the number of cookies taken in 6 days is 63. -/
theorem cookies_taken_theorem :
  cookies_taken_in_six_days 150 45 = 63 := by
  sorry

#eval cookies_taken_in_six_days 150 45

end cookies_taken_theorem_l1686_168601


namespace sin_30_plus_cos_60_quadratic_equation_solutions_l1686_168621

-- Problem 1
theorem sin_30_plus_cos_60 : Real.sin (π / 6) + Real.cos (π / 3) = 1 := by sorry

-- Problem 2
theorem quadratic_equation_solutions (x : ℝ) : 
  x^2 - 4*x = 12 ↔ x = 6 ∨ x = -2 := by sorry

end sin_30_plus_cos_60_quadratic_equation_solutions_l1686_168621


namespace incenter_coords_l1686_168634

/-- Triangle DEF with side lengths d, e, f -/
structure Triangle where
  d : ℝ
  e : ℝ
  f : ℝ

/-- Barycentric coordinates -/
structure BarycentricCoords where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : BarycentricCoords :=
  sorry

/-- The theorem stating that the barycentric coordinates of the incenter
    of triangle DEF with side lengths 8, 15, 17 are (8/40, 15/40, 17/40) -/
theorem incenter_coords :
  let t : Triangle := { d := 8, e := 15, f := 17 }
  let i : BarycentricCoords := incenter t
  i.x = 8/40 ∧ i.y = 15/40 ∧ i.z = 17/40 ∧ i.x + i.y + i.z = 1 := by
  sorry

end incenter_coords_l1686_168634


namespace fruit_pie_theorem_l1686_168639

/-- Represents the quantities of fruits used in pie making -/
structure FruitQuantities where
  apples : ℕ
  peaches : ℕ
  pears : ℕ
  plums : ℕ

/-- The ratio of fruits used per apple in pie making -/
structure FruitRatio where
  peaches_per_apple : ℕ
  pears_per_apple : ℕ
  plums_per_apple : ℕ

/-- Calculate the quantities of fruits used given the number of apples and the ratio -/
def calculate_used_fruits (apples_used : ℕ) (ratio : FruitRatio) : FruitQuantities :=
  { apples := apples_used,
    peaches := apples_used * ratio.peaches_per_apple,
    pears := apples_used * ratio.pears_per_apple,
    plums := apples_used * ratio.plums_per_apple }

theorem fruit_pie_theorem (initial_apples initial_peaches initial_pears initial_plums : ℕ)
                          (ratio : FruitRatio)
                          (apples_left : ℕ) :
  initial_apples = 40 →
  initial_peaches = 54 →
  initial_pears = 60 →
  initial_plums = 48 →
  ratio.peaches_per_apple = 2 →
  ratio.pears_per_apple = 3 →
  ratio.plums_per_apple = 4 →
  apples_left = 39 →
  calculate_used_fruits (initial_apples - apples_left) ratio =
    { apples := 1, peaches := 2, pears := 3, plums := 4 } :=
by sorry


end fruit_pie_theorem_l1686_168639


namespace quadratic_complete_square_l1686_168607

/-- Given a quadratic equation 16x^2 + 32x - 1280 = 0, prove that when rewritten
    in the form (x + r)^2 = s by completing the square, the value of s is 81. -/
theorem quadratic_complete_square (x : ℝ) :
  (16 * x^2 + 32 * x - 1280 = 0) →
  ∃ (r s : ℝ), ((x + r)^2 = s ∧ s = 81) :=
by sorry

end quadratic_complete_square_l1686_168607


namespace bike_sharing_selection_l1686_168657

theorem bike_sharing_selection (yellow_bikes : ℕ) (blue_bikes : ℕ) (inspect_yellow : ℕ) (inspect_blue : ℕ) :
  yellow_bikes = 6 →
  blue_bikes = 4 →
  inspect_yellow = 4 →
  inspect_blue = 4 →
  (Nat.choose blue_bikes 2 * Nat.choose yellow_bikes 2 +
   Nat.choose blue_bikes 3 * Nat.choose yellow_bikes 1 +
   Nat.choose blue_bikes 4) = 115 :=
by sorry

end bike_sharing_selection_l1686_168657


namespace millet_majority_on_tuesday_l1686_168663

/-- Represents the proportion of millet seeds remaining after birds eat -/
def milletRemaining : ℝ := 0.7

/-- Calculates the amount of millet seeds in the feeder after n days -/
def milletAmount (n : ℕ) : ℝ := 1 - milletRemaining ^ n

/-- The day when more than half the seeds are millet -/
def milletMajorityDay : ℕ := 2

theorem millet_majority_on_tuesday :
  milletAmount milletMajorityDay > 0.5 ∧
  ∀ k : ℕ, k < milletMajorityDay → milletAmount k ≤ 0.5 :=
sorry

end millet_majority_on_tuesday_l1686_168663


namespace sqrt_equation_solution_l1686_168609

theorem sqrt_equation_solution :
  ∃! x : ℚ, Real.sqrt (5 - 4 * x) = 6 :=
by
  use -31/4
  sorry

end sqrt_equation_solution_l1686_168609


namespace system_solution_l1686_168615

theorem system_solution :
  let x₁ : ℝ := 5
  let x₂ : ℝ := -5
  let x₃ : ℝ := 0
  let x₄ : ℝ := 2
  let x₅ : ℝ := -1
  let x₆ : ℝ := 1
  (x₁ + x₃ + 2*x₄ + 3*x₅ - 4*x₆ = 20) ∧
  (2*x₁ + x₂ - 3*x₃ + x₅ + 2*x₆ = -13) ∧
  (5*x₁ - x₂ + x₃ + 2*x₄ + 6*x₅ = 20) ∧
  (2*x₁ - 2*x₂ + 3*x₃ + 2*x₅ + 2*x₆ = 13) :=
by
  sorry

end system_solution_l1686_168615


namespace factorization_of_3x2_minus_12y2_l1686_168600

theorem factorization_of_3x2_minus_12y2 (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end factorization_of_3x2_minus_12y2_l1686_168600


namespace nested_fraction_equals_seven_halves_l1686_168658

theorem nested_fraction_equals_seven_halves :
  2 + 2 / (1 + 1 / (2 + 1)) = 7 / 2 := by
  sorry

end nested_fraction_equals_seven_halves_l1686_168658


namespace output_is_72_l1686_168627

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 38 then step1 * 2 else step1 - 10

theorem output_is_72 : function_machine 12 = 72 := by
  sorry

end output_is_72_l1686_168627


namespace cup_purchase_properties_prize_purchase_properties_l1686_168691

/-- Represents the cost and quantity of insulated cups --/
structure CupPurchase where
  cost_a : ℕ  -- Cost of A type cup
  cost_b : ℕ  -- Cost of B type cup
  quantity_a : ℕ  -- Quantity of A type cups
  quantity_b : ℕ  -- Quantity of B type cups

/-- Theorem stating the properties of the cup purchase --/
theorem cup_purchase_properties :
  ∃ (purchase : CupPurchase),
    -- B type cup costs 10 yuan more than A type cup
    purchase.cost_b = purchase.cost_a + 10 ∧
    -- 1200 yuan buys 1.5 times as many A cups as 1000 yuan buys B cups
    1200 / purchase.cost_a = (3/2) * (1000 / purchase.cost_b) ∧
    -- Company buys 9 fewer B cups than A cups
    purchase.quantity_b = purchase.quantity_a - 9 ∧
    -- Number of A cups is not less than 38
    purchase.quantity_a ≥ 38 ∧
    -- Total cost does not exceed 3150 yuan
    purchase.cost_a * purchase.quantity_a + purchase.cost_b * purchase.quantity_b ≤ 3150 ∧
    -- Cost of A type cup is 40 yuan
    purchase.cost_a = 40 ∧
    -- Cost of B type cup is 50 yuan
    purchase.cost_b = 50 ∧
    -- There are exactly three valid purchasing schemes
    (∃ (scheme1 scheme2 scheme3 : CupPurchase),
      scheme1.quantity_a = 38 ∧ scheme1.quantity_b = 29 ∧
      scheme2.quantity_a = 39 ∧ scheme2.quantity_b = 30 ∧
      scheme3.quantity_a = 40 ∧ scheme3.quantity_b = 31 ∧
      ∀ (other : CupPurchase),
        (other.quantity_a ≥ 38 ∧
         other.quantity_b = other.quantity_a - 9 ∧
         other.cost_a * other.quantity_a + other.cost_b * other.quantity_b ≤ 3150) →
        (other = scheme1 ∨ other = scheme2 ∨ other = scheme3)) :=
by
  sorry

/-- Represents the quantity of prizes --/
structure PrizePurchase where
  quantity_a : ℕ  -- Quantity of A type prizes
  quantity_b : ℕ  -- Quantity of B type prizes

/-- Theorem stating the properties of the prize purchase --/
theorem prize_purchase_properties :
  ∃ (prize : PrizePurchase),
    -- A type prize costs 270 yuan
    -- B type prize costs 240 yuan
    -- Total cost of prizes equals minimum cost from part 2 (2970 yuan)
    270 * prize.quantity_a + 240 * prize.quantity_b = 2970 ∧
    -- There are 3 A type prizes and 9 B type prizes
    prize.quantity_a = 3 ∧
    prize.quantity_b = 9 :=
by
  sorry

end cup_purchase_properties_prize_purchase_properties_l1686_168691


namespace omega_identity_l1686_168612

theorem omega_identity (ω : ℂ) (h : ω = -1/2 + Complex.I * (Real.sqrt 3) / 2) :
  1 + ω = -1/ω := by sorry

end omega_identity_l1686_168612


namespace line_intersects_at_least_one_l1686_168674

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (contained_in : Line → Plane → Prop)
variable (intersects : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (plane_intersection : Plane → Plane → Line → Prop)

-- State the theorem
theorem line_intersects_at_least_one 
  (a b l : Line) (α β : Plane) :
  skew a b →
  contained_in a α →
  contained_in b β →
  plane_intersection α β l →
  (intersects l a) ∨ (intersects l b) :=
sorry

end line_intersects_at_least_one_l1686_168674


namespace sufficient_condition_for_inequality_l1686_168652

theorem sufficient_condition_for_inequality (x : ℝ) : 
  (∀ x, x > 1 → 1 - 1/x > 0) ∧ 
  (∃ x, 1 - 1/x > 0 ∧ ¬(x > 1)) :=
sorry

end sufficient_condition_for_inequality_l1686_168652


namespace polar_to_cartesian_circle_l1686_168692

/-- The polar equation ρ = 4cosθ is equivalent to the Cartesian equation (x-2)^2 + y^2 = 4 -/
theorem polar_to_cartesian_circle :
  ∀ (x y ρ θ : ℝ), 
  (ρ = 4 * Real.cos θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) → 
  ((x - 2)^2 + y^2 = 4) :=
by sorry

end polar_to_cartesian_circle_l1686_168692


namespace intersection_contains_two_elements_l1686_168614

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1^2}
def N : Set (ℝ × ℝ) := {p | p.2 = 5 * p.1}

-- State the theorem
theorem intersection_contains_two_elements :
  ∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ M ∩ N ∧ b ∈ M ∩ N ∧
  ∀ c, c ∈ M ∩ N → c = a ∨ c = b :=
sorry

end intersection_contains_two_elements_l1686_168614


namespace sum_of_preceding_terms_l1686_168678

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define our specific sequence
def our_sequence (a : ℕ → ℕ) : Prop :=
  arithmetic_sequence a ∧ a 1 = 3 ∧ a 2 = 8 ∧ ∃ k : ℕ, a k = 33 ∧ ∀ m : ℕ, m > k → a m > 33

theorem sum_of_preceding_terms (a : ℕ → ℕ) (h : our_sequence a) :
  ∃ n : ℕ, a n + a (n + 1) = 51 ∧ a (n + 2) = 33 :=
sorry

end sum_of_preceding_terms_l1686_168678


namespace hexagon_segment_probability_l1686_168656

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of elements in T -/
def T_size : ℕ := 15

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 17/35

theorem hexagon_segment_probability :
  (num_sides / T_size) * ((num_sides - 1) / (T_size - 1)) +
  (num_diagonals / T_size) * ((num_diagonals - 1) / (T_size - 1)) = prob_same_length := by
  sorry

end hexagon_segment_probability_l1686_168656


namespace chocolate_sales_l1686_168647

theorem chocolate_sales (cost_price selling_price : ℝ) (N : ℕ) : 
  (121 * cost_price = N * selling_price) →
  (selling_price = cost_price * (1 + 57.142857142857146 / 100)) →
  N = 77 := by
  sorry

end chocolate_sales_l1686_168647


namespace smallest_m_is_16_l1686_168698

/-- The set T of complex numbers -/
def T : Set ℂ :=
  {z : ℂ | ∃ (u v : ℝ), z = u + v * Complex.I ∧ Real.sqrt 3 / 3 ≤ u ∧ u ≤ Real.sqrt 3 / 2}

/-- The property P(n) that should hold for all n ≥ m -/
def P (n : ℕ) : Prop :=
  ∃ z ∈ T, z ^ (2 * n) = 1

/-- The theorem stating that 16 is the smallest positive integer m satisfying the condition -/
theorem smallest_m_is_16 :
  (∀ n ≥ 16, P n) ∧ ∀ m < 16, ¬(∀ n ≥ m, P n) :=
sorry

end smallest_m_is_16_l1686_168698


namespace software_contract_probability_l1686_168618

theorem software_contract_probability
  (p_hardware : ℝ)
  (p_at_least_one : ℝ)
  (p_both : ℝ)
  (h1 : p_hardware = 4/5)
  (h2 : p_at_least_one = 5/6)
  (h3 : p_both = 11/30) :
  1 - (p_at_least_one - p_hardware + p_both) = 3/5 := by
sorry

end software_contract_probability_l1686_168618


namespace jake_weight_ratio_l1686_168687

/-- Jake's weight problem -/
theorem jake_weight_ratio :
  let jake_present_weight : ℚ := 196
  let total_weight : ℚ := 290
  let weight_loss : ℚ := 8
  let jake_new_weight := jake_present_weight - weight_loss
  let sister_weight := total_weight - jake_present_weight
  jake_new_weight / sister_weight = 2 := by
  sorry

end jake_weight_ratio_l1686_168687


namespace sqrt_12_equals_2_sqrt_3_l1686_168655

theorem sqrt_12_equals_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_12_equals_2_sqrt_3_l1686_168655


namespace max_coins_identifiable_l1686_168633

/-- The maximum number of coins that can be tested to identify one counterfeit (lighter) coin -/
def max_coins (n : ℕ) : ℕ := 2 * n^2 + 1

/-- A balance scale used for weighing coins -/
structure BalanceScale :=
  (weigh : ℕ → ℕ → Bool)

/-- Represents the process of identifying a counterfeit coin -/
structure CoinIdentification :=
  (n : ℕ)  -- Number of weighings allowed
  (coins : ℕ)  -- Total number of coins
  (scale : BalanceScale)
  (max_weighings_per_coin : ℕ := 2)  -- Maximum number of times each coin can be weighed

/-- Theorem stating the maximum number of coins that can be tested -/
theorem max_coins_identifiable (ci : CoinIdentification) :
  ci.coins ≤ max_coins ci.n ↔
  ∃ (strategy : Unit), true  -- Placeholder for the existence of a valid identification strategy
:= by sorry

end max_coins_identifiable_l1686_168633


namespace compute_expression_l1686_168620

theorem compute_expression : 
  20 * (200 / 3 + 36 / 9 + 16 / 25 + 3) = 13212.8 := by
  sorry

end compute_expression_l1686_168620


namespace johns_dancing_time_l1686_168642

theorem johns_dancing_time (john_initial : ℝ) (john_after : ℝ) (james : ℝ) 
  (h1 : john_after = 5)
  (h2 : james = john_initial + 1 + john_after + (1/3) * (john_initial + 1 + john_after))
  (h3 : john_initial + john_after + james = 20) :
  john_initial = 3 := by
  sorry

end johns_dancing_time_l1686_168642


namespace system_of_inequalities_solution_l1686_168631

theorem system_of_inequalities_solution (x : ℝ) :
  (x^2 > x + 2 ∧ 4*x^2 ≤ 4*x + 15) ↔ 
  (x ∈ Set.Icc (-3/2) (-1) ∪ Set.Ioc 2 (5/2)) :=
sorry

end system_of_inequalities_solution_l1686_168631


namespace circle_equation_with_given_diameter_l1686_168685

/-- The standard equation of a circle with diameter endpoints (0,2) and (4,4) -/
theorem circle_equation_with_given_diameter :
  let p₁ : ℝ × ℝ := (0, 2)
  let p₂ : ℝ × ℝ := (4, 4)
  let M : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (t : ℝ), p = (1 - t) • p₁ + t • p₂ ∧ 0 ≤ t ∧ t ≤ 1}
  ∀ (x y : ℝ), (x, y) ∈ M ↔ (x - 2)^2 + (y - 3)^2 = 5 :=
by sorry

end circle_equation_with_given_diameter_l1686_168685


namespace triangle_angle_sequence_range_l1686_168671

theorem triangle_angle_sequence_range (A B C a b c k : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧  -- sum of angles in a triangle
  B - A = C - B ∧  -- arithmetic sequence
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a^2 + c^2 = k * b^2 →  -- given equation
  1 < k ∧ k ≤ 2 := by sorry

end triangle_angle_sequence_range_l1686_168671


namespace equilibrium_exists_l1686_168636

/-- Represents the equilibrium state of two connected vessels with different liquids -/
def EquilibriumState (H : ℝ) : Prop :=
  ∃ (h_water h_gasoline : ℝ),
    -- Initial conditions
    0 < H ∧
    -- Valve position
    0.15 * H < 0.9 * H ∧
    -- Initial liquid levels
    0.9 * H = 0.9 * H ∧
    -- Densities
    let ρ_water : ℝ := 1000
    let ρ_gasoline : ℝ := 600
    -- Equilibrium condition
    ρ_water * (0.75 * H - (0.9 * H - h_water)) = 
      ρ_water * (h_water - 0.15 * H) + ρ_gasoline * (H - h_water) ∧
    -- Final water level
    h_water = 0.69 * H ∧
    -- Final gasoline level
    h_gasoline = H

/-- Theorem stating that the equilibrium state exists for any positive vessel height -/
theorem equilibrium_exists (H : ℝ) (h_pos : 0 < H) : EquilibriumState H := by
  sorry

#check equilibrium_exists

end equilibrium_exists_l1686_168636


namespace hyperbola_focal_distance_l1686_168603

/-- Given a hyperbola with equation x²/36 - y²/b² = 1 where b > 0,
    eccentricity e = 5/3, and a point P on the hyperbola such that |PF₁| = 15,
    prove that |PF₂| = 27 -/
theorem hyperbola_focal_distance (b : ℝ) (P : ℝ × ℝ) :
  b > 0 →
  (P.1^2 / 36 - P.2^2 / b^2 = 1) →
  (Real.sqrt (36 + b^2) / 6 = 5 / 3) →
  (Real.sqrt ((P.1 + Real.sqrt (36 + b^2))^2 + P.2^2) = 15) →
  Real.sqrt ((P.1 - Real.sqrt (36 + b^2))^2 + P.2^2) = 27 :=
by sorry

end hyperbola_focal_distance_l1686_168603


namespace class_size_is_37_l1686_168641

/-- Represents the number of students in a class with specific age distribution. -/
def number_of_students (common_age : ℕ) (total_age_sum : ℕ) : ℕ :=
  (total_age_sum + 3) / common_age

/-- Theorem stating the number of students in the class is 37. -/
theorem class_size_is_37 :
  ∃ (common_age : ℕ),
    common_age > 0 ∧
    number_of_students common_age 330 = 37 ∧
    330 = 7 * (common_age - 1) + 2 * (common_age + 2) + (37 - 9) * common_age :=
sorry

end class_size_is_37_l1686_168641


namespace three_numbers_sum_divisible_by_three_l1686_168697

def set_of_numbers : Finset ℕ := Finset.range 20

theorem three_numbers_sum_divisible_by_three (set_of_numbers : Finset ℕ) :
  (Finset.filter (fun s : Finset ℕ => s.card = 3 ∧ 
    (s.sum id) % 3 = 0 ∧ 
    s ⊆ set_of_numbers) (Finset.powerset set_of_numbers)).card = 384 := by
  sorry

end three_numbers_sum_divisible_by_three_l1686_168697


namespace distance_between_points_l1686_168665

theorem distance_between_points : Real.sqrt ((0 - 6)^2 + (18 - 0)^2) = 6 * Real.sqrt 10 := by
  sorry

end distance_between_points_l1686_168665


namespace arithmetic_mean_problem_l1686_168662

/-- Given a set {6, 13, 18, 4, x} where 10 is the arithmetic mean, prove that x = 9 -/
theorem arithmetic_mean_problem (x : ℝ) : 
  (6 + 13 + 18 + 4 + x) / 5 = 10 → x = 9 := by
  sorry

end arithmetic_mean_problem_l1686_168662


namespace inequality_addition_l1686_168646

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end inequality_addition_l1686_168646


namespace student_number_problem_l1686_168651

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 112 → x = 125 := by
  sorry

end student_number_problem_l1686_168651


namespace rectangle_area_rectangle_area_proof_l1686_168628

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1600 10 = 160 := by
  sorry

end rectangle_area_rectangle_area_proof_l1686_168628


namespace sid_computer_accessories_cost_l1686_168649

/-- Calculates the amount spent on computer accessories given the initial amount,
    snack cost, and remaining amount after purchases. -/
def computer_accessories_cost (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - snack_cost - remaining_amount

/-- Proves that Sid spent $12 on computer accessories given the problem conditions. -/
theorem sid_computer_accessories_cost :
  let initial_amount : ℕ := 48
  let snack_cost : ℕ := 8
  let remaining_amount : ℕ := (initial_amount / 2) + 4
  computer_accessories_cost initial_amount snack_cost remaining_amount = 12 := by
  sorry

#eval computer_accessories_cost 48 8 28

end sid_computer_accessories_cost_l1686_168649


namespace smallest_prime_factor_of_2903_l1686_168670

theorem smallest_prime_factor_of_2903 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2903 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2903 → p ≤ q :=
by sorry

end smallest_prime_factor_of_2903_l1686_168670


namespace solutions_equation1_solution_equation2_l1686_168622

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 2)^2 = 36
def equation2 (x : ℝ) : Prop := (2*x - 1)^3 = -125

-- Statement for the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = 8 ∨ x = -4)) :=
sorry

-- Statement for the second equation
theorem solution_equation2 : 
  (∃ x : ℝ, equation2 x) ∧
  (∀ x : ℝ, equation2 x ↔ x = -2) :=
sorry

end solutions_equation1_solution_equation2_l1686_168622


namespace two_three_four_forms_triangle_one_two_three_not_triangle_two_two_four_not_triangle_two_three_six_not_triangle_triangle_formation_theorem_l1686_168673

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem stating that (2, 3, 4) can form a triangle
theorem two_three_four_forms_triangle :
  can_form_triangle 2 3 4 := by sorry

-- Theorem stating that (1, 2, 3) cannot form a triangle
theorem one_two_three_not_triangle :
  ¬ can_form_triangle 1 2 3 := by sorry

-- Theorem stating that (2, 2, 4) cannot form a triangle
theorem two_two_four_not_triangle :
  ¬ can_form_triangle 2 2 4 := by sorry

-- Theorem stating that (2, 3, 6) cannot form a triangle
theorem two_three_six_not_triangle :
  ¬ can_form_triangle 2 3 6 := by sorry

-- Main theorem combining all results
theorem triangle_formation_theorem :
  can_form_triangle 2 3 4 ∧
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 2 2 4 ∧
  ¬ can_form_triangle 2 3 6 := by sorry

end two_three_four_forms_triangle_one_two_three_not_triangle_two_two_four_not_triangle_two_three_six_not_triangle_triangle_formation_theorem_l1686_168673


namespace negative_a_range_l1686_168619

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x < 1/2 ∨ x > 3}

-- Theorem statement
theorem negative_a_range (a : ℝ) (h_neg : a < 0) :
  (complementA ∩ B a = B a) ↔ -1/4 ≤ a ∧ a < 0 :=
by sorry

end negative_a_range_l1686_168619


namespace math_club_female_members_l1686_168653

theorem math_club_female_members :
  ∀ (female_members male_members : ℕ),
    female_members > 0 →
    male_members = 2 * female_members →
    female_members + male_members = 18 →
    female_members = 6 := by
  sorry

end math_club_female_members_l1686_168653


namespace reflect_distance_C_l1686_168644

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflect_distance (p : ℝ × ℝ) : ℝ :=
  2 * |p.2|

theorem reflect_distance_C : reflect_distance (-3, 2) = 4 := by
  sorry

end reflect_distance_C_l1686_168644


namespace sculpture_height_proof_l1686_168643

/-- The height of the sculpture in inches -/
def sculpture_height : ℝ := 34

/-- The height of the base in inches -/
def base_height : ℝ := 4

/-- The combined height of the sculpture and base in feet -/
def total_height_feet : ℝ := 3.1666666666666665

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

theorem sculpture_height_proof :
  sculpture_height = total_height_feet * feet_to_inches - base_height :=
by sorry

end sculpture_height_proof_l1686_168643


namespace largest_number_l1686_168693

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_square_of_prime (n : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ n = p * p

theorem largest_number (P Q R S T : ℕ) : 
  (2 ≤ P ∧ P ≤ 19) →
  (2 ≤ Q ∧ Q ≤ 19) →
  (2 ≤ R ∧ R ≤ 19) →
  (2 ≤ S ∧ S ≤ 19) →
  (2 ≤ T ∧ T ≤ 19) →
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T →
  (P ≥ 10 ∧ P < 100 ∧ is_prime P ∧ is_prime (digit_sum P)) →
  (∃ k : ℕ, Q = 5 * k) →
  (R % 2 = 1 ∧ ¬is_prime R) →
  is_square_of_prime S →
  (is_prime T ∧ T = (P + Q) / 2) →
  Q ≥ P ∧ Q ≥ R ∧ Q ≥ S ∧ Q ≥ T :=
by sorry

end largest_number_l1686_168693


namespace sum_of_reciprocals_zero_l1686_168629

theorem sum_of_reciprocals_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) : 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 0 := by
  sorry

end sum_of_reciprocals_zero_l1686_168629


namespace pirate_treasure_distribution_l1686_168654

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pirate_treasure_distribution : 
  ∃ (x : ℕ), 
    x > 0 ∧ 
    sum_of_first_n x = 3 * x ∧ 
    x + 3 * x = 20 := by
  sorry

end pirate_treasure_distribution_l1686_168654


namespace inequality_solution_set_l1686_168675

theorem inequality_solution_set (x : ℝ) : 
  (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) :=
by sorry

end inequality_solution_set_l1686_168675


namespace system_solutions_correct_l1686_168626

theorem system_solutions_correct :
  -- System (1)
  let x₁ := 1
  let y₁ := 2
  -- System (2)
  let x₂ := (1 : ℚ) / 2
  let y₂ := 5
  -- Prove that these solutions satisfy the equations
  (x₁ = 5 - 2 * y₁ ∧ 3 * x₁ - y₁ = 1) ∧
  (2 * x₂ - y₂ = -4 ∧ 4 * x₂ - 5 * y₂ = -23) := by
  sorry


end system_solutions_correct_l1686_168626


namespace inequality_equivalence_l1686_168660

theorem inequality_equivalence (x : ℝ) :
  -1 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 1 ↔ x > 1 := by
  sorry

end inequality_equivalence_l1686_168660


namespace line_ellipse_intersection_l1686_168676

theorem line_ellipse_intersection (k : ℝ) : ∃ (x y : ℝ), 
  (y = k * x + 1 - k) ∧ (x^2 / 9 + y^2 / 4 = 1) := by
  sorry

#check line_ellipse_intersection

end line_ellipse_intersection_l1686_168676


namespace complex_square_one_minus_i_l1686_168611

theorem complex_square_one_minus_i :
  (1 - Complex.I) ^ 2 = -2 * Complex.I :=
sorry

end complex_square_one_minus_i_l1686_168611


namespace train_length_l1686_168650

/-- Proves that a train traveling at 45 km/hr crossing a 255 m bridge in 30 seconds has a length of 120 m -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 120 := by
  sorry

#check train_length

end train_length_l1686_168650


namespace max_teams_is_eight_l1686_168610

/-- Represents the number of teams that can be formed given the number of climbers in each skill level and the required composition of each team. -/
def max_teams (advanced intermediate beginner : ℕ) 
              (adv_per_team int_per_team beg_per_team : ℕ) : ℕ :=
  min (advanced / adv_per_team)
      (min (intermediate / int_per_team)
           (beginner / beg_per_team))

/-- Theorem stating that the maximum number of teams that can be formed is 8. -/
theorem max_teams_is_eight : 
  max_teams 45 70 57 5 8 5 = 8 := by
  sorry

end max_teams_is_eight_l1686_168610


namespace angle_range_in_third_quadrant_l1686_168669

theorem angle_range_in_third_quadrant (θ : Real) (k : Int) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin (θ/4) < Real.cos (θ/4)) →  -- sin(θ/4) < cos(θ/4)
  (∃ k : Int, 
    ((2*k*π + 5*π/4 < θ/4 ∧ θ/4 < 2*k*π + 11*π/8) ∨ 
     (2*k*π + 7*π/4 < θ/4 ∧ θ/4 < 2*k*π + 15*π/8))) := by
  sorry

end angle_range_in_third_quadrant_l1686_168669


namespace min_expected_weight_l1686_168667

theorem min_expected_weight (x y e : ℝ) :
  y = 0.85 * x - 88 + e →
  |e| ≤ 4 →
  x = 160 →
  ∃ y_min : ℝ, y_min = 44 ∧ ∀ y' : ℝ, (∃ e' : ℝ, y' = 0.85 * x - 88 + e' ∧ |e'| ≤ 4) → y' ≥ y_min :=
by sorry

end min_expected_weight_l1686_168667


namespace unique_natural_number_a_l1686_168648

theorem unique_natural_number_a : ∃! (a : ℕ), 
  (1000 ≤ 4 * a^2) ∧ (4 * a^2 < 10000) ∧ 
  (1000 ≤ (4/3) * a^3) ∧ ((4/3) * a^3 < 10000) ∧
  (∃ (n : ℕ), (4/3) * a^3 = n) :=
by sorry

end unique_natural_number_a_l1686_168648


namespace paving_cost_l1686_168686

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 400) :
  length * width * rate = 8250 := by
  sorry

end paving_cost_l1686_168686


namespace largest_divisible_n_l1686_168632

theorem largest_divisible_n : ∃ (n : ℕ), n = 910 ∧ 
  (∀ m : ℕ, m > n → ¬(m - 10 ∣ m^3 - 100)) ∧ 
  (n - 10 ∣ n^3 - 100) :=
by sorry

end largest_divisible_n_l1686_168632


namespace magnitude_a_minus_b_l1686_168604

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, -2)

theorem magnitude_a_minus_b : 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 := by
  sorry

end magnitude_a_minus_b_l1686_168604


namespace ellipse_parabola_intersection_bounds_l1686_168690

/-- If an ellipse and a parabola have a common point, then the parameter 'a' of the ellipse is bounded. -/
theorem ellipse_parabola_intersection_bounds (a : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) → 
  -1 ≤ a ∧ a ≤ 17/8 := by
sorry

end ellipse_parabola_intersection_bounds_l1686_168690
