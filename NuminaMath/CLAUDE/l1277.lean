import Mathlib

namespace NUMINAMATH_CALUDE_married_men_count_l1277_127799

theorem married_men_count (total : ℕ) (tv : ℕ) (radio : ℕ) (ac : ℕ) (all_and_married : ℕ) 
  (h_total : total = 100)
  (h_tv : tv = 75)
  (h_radio : radio = 85)
  (h_ac : ac = 70)
  (h_all_and_married : all_and_married = 12)
  (h_all_and_married_le_total : all_and_married ≤ total) :
  ∃ (married : ℕ), married ≥ all_and_married ∧ married ≤ total :=
by
  sorry

end NUMINAMATH_CALUDE_married_men_count_l1277_127799


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l1277_127733

theorem quadrilateral_side_length (A B C D : ℝ × ℝ) : 
  let angle (p q r : ℝ × ℝ) := Real.arccos ((p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)) / 
    (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) * Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2))
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  angle D A B = π/3 ∧ 
  angle A B C = π/2 ∧ 
  angle B C D = π/2 ∧ 
  dist B C = 2 ∧ 
  dist C D = 3 →
  dist A B = 8 / Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_quadrilateral_side_length_l1277_127733


namespace NUMINAMATH_CALUDE_complement_union_M_N_l1277_127754

open Set

-- Define the universal set as ℝ
universe u
variable {α : Type u}

-- Define sets M and N
def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {x | x > 2}

-- State the theorem
theorem complement_union_M_N :
  (M ∪ N)ᶜ = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l1277_127754


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l1277_127796

theorem min_value_quadratic_function :
  ∀ (x y z : ℝ), 
    x^2 + 4*x*y + 3*y^2 + 2*z^2 - 8*x - 4*y + 6*z ≥ -13.5 ∧
    (x^2 + 4*x*y + 3*y^2 + 2*z^2 - 8*x - 4*y + 6*z = -13.5 ↔ x = 1 ∧ y = 3/2 ∧ z = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l1277_127796


namespace NUMINAMATH_CALUDE_geometric_sequence_101st_term_l1277_127783

def geometric_sequence (a : ℕ → ℝ) := ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_101st_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_3rd : a 3 = 3)
  (h_sum : a 2016 + a 2017 = 0) :
  a 101 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_101st_term_l1277_127783


namespace NUMINAMATH_CALUDE_pokemon_cards_distribution_l1277_127750

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 56) (h2 : num_friends = 4) :
  total_cards / num_friends = 14 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_distribution_l1277_127750


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_minimum_years_proof_l1277_127779

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def payment (t : ℝ) : ℝ := t

def remaining_capital (n : ℕ) (t : ℝ) : ℝ :=
  if n = 0 then initial_capital
  else (1 + growth_rate) * remaining_capital (n - 1) t - payment t

theorem geometric_sequence_proof (t : ℝ) (h : 0 < t ∧ t < 2500) :
  ∀ n : ℕ, (remaining_capital (n + 1) t - 2 * t) / (remaining_capital n t - 2 * t) = 3 / 2 :=
sorry

theorem minimum_years_proof :
  let t := 1500
  (∃ m : ℕ, remaining_capital m t > 21000) ∧
  (∀ k : ℕ, k < 6 → remaining_capital k t ≤ 21000) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_minimum_years_proof_l1277_127779


namespace NUMINAMATH_CALUDE_rent_share_ratio_l1277_127714

/-- Proves that the ratio of Sheila's share to Purity's share is 5:1 given the rent conditions --/
theorem rent_share_ratio (total_rent : ℝ) (rose_share : ℝ) (purity_share : ℝ) (sheila_share : ℝ) :
  total_rent = 5400 →
  rose_share = 1800 →
  rose_share = 3 * purity_share →
  total_rent = purity_share + rose_share + sheila_share →
  sheila_share / purity_share = 5 := by
  sorry

#check rent_share_ratio

end NUMINAMATH_CALUDE_rent_share_ratio_l1277_127714


namespace NUMINAMATH_CALUDE_two_fruits_from_five_l1277_127784

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruit types available -/
def num_fruits : ℕ := 5

/-- The number of fruits to be chosen -/
def fruits_to_choose : ℕ := 2

theorem two_fruits_from_five :
  choose num_fruits fruits_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_fruits_from_five_l1277_127784


namespace NUMINAMATH_CALUDE_upper_bound_y_l1277_127736

theorem upper_bound_y (x y : ℤ) (U : ℤ) : 
  (3 < x ∧ x < 6) → 
  (6 < y ∧ y < U) → 
  (∀ (x' y' : ℤ), (3 < x' ∧ x' < 6) → (6 < y' ∧ y' < U) → y' - x' ≤ 4) →
  (∃ (x' y' : ℤ), (3 < x' ∧ x' < 6) ∧ (6 < y' ∧ y' < U) ∧ y' - x' = 4) →
  U = 10 :=
by sorry

end NUMINAMATH_CALUDE_upper_bound_y_l1277_127736


namespace NUMINAMATH_CALUDE_logan_gas_budget_l1277_127769

/-- Calculates the amount Logan can spend on gas annually --/
def gas_budget (current_income rent groceries desired_savings income_increase : ℕ) : ℕ :=
  (current_income + income_increase) - (rent + groceries + desired_savings)

/-- Proves that Logan's gas budget is $8,000 given his financial constraints --/
theorem logan_gas_budget :
  gas_budget 65000 20000 5000 42000 10000 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_logan_gas_budget_l1277_127769


namespace NUMINAMATH_CALUDE_frisbee_price_l1277_127712

/-- Given the conditions of frisbee sales, prove the price of non-$4 frisbees -/
theorem frisbee_price (total_frisbees : ℕ) (total_receipts : ℕ) (price_known : ℕ) (min_known_price : ℕ) :
  total_frisbees = 64 →
  total_receipts = 196 →
  price_known = 4 →
  min_known_price = 4 →
  ∃ (price_unknown : ℕ),
    price_unknown * (total_frisbees - min_known_price) + price_known * min_known_price = total_receipts ∧
    price_unknown = 3 := by
  sorry

#check frisbee_price

end NUMINAMATH_CALUDE_frisbee_price_l1277_127712


namespace NUMINAMATH_CALUDE_not_in_second_quadrant_l1277_127729

/-- A linear function y = x - 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of y = x - 2 does not pass through the second quadrant -/
theorem not_in_second_quadrant :
  ¬ ∃ (x : ℝ), second_quadrant x (f x) := by
  sorry

end NUMINAMATH_CALUDE_not_in_second_quadrant_l1277_127729


namespace NUMINAMATH_CALUDE_students_liking_both_sea_and_mountains_l1277_127732

theorem students_liking_both_sea_and_mountains 
  (total_students : ℕ)
  (sea_lovers : ℕ)
  (mountain_lovers : ℕ)
  (neither_lovers : ℕ)
  (h1 : total_students = 500)
  (h2 : sea_lovers = 337)
  (h3 : mountain_lovers = 289)
  (h4 : neither_lovers = 56) :
  sea_lovers + mountain_lovers - (total_students - neither_lovers) = 182 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_sea_and_mountains_l1277_127732


namespace NUMINAMATH_CALUDE_min_value_theorem_l1277_127722

theorem min_value_theorem (a : ℝ) (ha : a > 0) :
  a + (a + 4) / a ≥ 5 ∧ ∃ a₀ > 0, a₀ + (a₀ + 4) / a₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1277_127722


namespace NUMINAMATH_CALUDE_paper_folding_holes_l1277_127744

/-- The number of small squares along each side after folding a square paper n times -/
def squares_per_side (n : ℕ) : ℕ := 2^n

/-- The number of internal edges along each side after folding -/
def internal_edges (n : ℕ) : ℕ := squares_per_side n - 1

/-- The total number of holes in the middle of the paper after folding n times -/
def total_holes (n : ℕ) : ℕ := internal_edges n * squares_per_side n

/-- Theorem: When a square piece of paper is folded in half 6 times and a notch is cut along
    each edge of the resulting small square, the number of small holes in the middle
    when unfolded is 4032. -/
theorem paper_folding_holes :
  total_holes 6 = 4032 := by sorry

end NUMINAMATH_CALUDE_paper_folding_holes_l1277_127744


namespace NUMINAMATH_CALUDE_max_silver_tokens_l1277_127797

/-- Represents the state of tokens --/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth --/
structure ExchangeBooth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Defines if an exchange is possible given a token state and an exchange booth --/
def canExchange (state : TokenState) (booth : ExchangeBooth) : Prop :=
  state.red ≥ booth.redIn ∧ state.blue ≥ booth.blueIn

/-- Defines the result of an exchange --/
def exchangeResult (state : TokenState) (booth : ExchangeBooth) : TokenState :=
  { red := state.red - booth.redIn + booth.redOut,
    blue := state.blue - booth.blueIn + booth.blueOut,
    silver := state.silver + booth.silverOut }

/-- Defines if a state is final (no more exchanges possible) --/
def isFinalState (state : TokenState) (booths : List ExchangeBooth) : Prop :=
  ∀ booth ∈ booths, ¬(canExchange state booth)

/-- The main theorem --/
theorem max_silver_tokens : 
  ∃ (finalState : TokenState),
    let initialState : TokenState := { red := 100, blue := 50, silver := 0 }
    let booth1 : ExchangeBooth := { redIn := 4, blueIn := 0, redOut := 0, blueOut := 3, silverOut := 1 }
    let booth2 : ExchangeBooth := { redIn := 0, blueIn := 2, redOut := 1, blueOut := 0, silverOut := 1 }
    let booths : List ExchangeBooth := [booth1, booth2]
    (isFinalState finalState booths) ∧
    (finalState.silver = 143) ∧
    (∀ (otherFinalState : TokenState),
      (isFinalState otherFinalState booths) →
      (otherFinalState.silver ≤ finalState.silver)) := by
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l1277_127797


namespace NUMINAMATH_CALUDE_triangle_most_stable_l1277_127727

-- Define the possible shapes
inductive Shape
  | Heptagon
  | Hexagon
  | Pentagon
  | Triangle

-- Define a property for stability
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => true
  | _ => false

-- Theorem stating that the triangle is the most stable shape
theorem triangle_most_stable :
  ∀ s : Shape, is_stable s → s = Shape.Triangle :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_most_stable_l1277_127727


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_four_l1277_127762

theorem sum_of_squares_divisible_by_four (n : ℤ) :
  ∃ k : ℤ, (2*n)^2 + (2*n + 2)^2 + (2*n + 4)^2 = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_four_l1277_127762


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l1277_127748

theorem sum_and_ratio_to_difference (x y : ℚ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 4/5) :
  y - x = 500/9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l1277_127748


namespace NUMINAMATH_CALUDE_equal_sums_exist_l1277_127775

/-- Represents a cell in the table -/
inductive Cell
  | Neg : Cell  -- Represents -1
  | Zero : Cell -- Represents 0
  | Pos : Cell  -- Represents 1

/-- Represents a (2n+1) × (2n+1) table -/
def Table (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → Cell

/-- Calculates the sum of a row or column -/
def sum_line (t : Table n) (is_row : Bool) (i : Fin (2*n+1)) : ℤ :=
  sorry

/-- The main theorem -/
theorem equal_sums_exist (n : ℕ) (t : Table n) :
  ∃ (i j : Fin (2*n+1)) (b₁ b₂ : Bool), 
    (i ≠ j ∨ b₁ ≠ b₂) ∧ sum_line t b₁ i = sum_line t b₂ j :=
sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l1277_127775


namespace NUMINAMATH_CALUDE_union_subset_iff_m_nonpositive_no_m_exists_for_equality_l1277_127740

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | |x - 1| ≤ m}

-- Question 1
theorem union_subset_iff_m_nonpositive (m : ℝ) :
  (P ∪ S m) ⊆ P ↔ m ≤ 0 :=
sorry

-- Question 2
theorem no_m_exists_for_equality :
  ¬∃ m : ℝ, P = S m :=
sorry

end NUMINAMATH_CALUDE_union_subset_iff_m_nonpositive_no_m_exists_for_equality_l1277_127740


namespace NUMINAMATH_CALUDE_counterfeit_probability_l1277_127721

def total_bills : ℕ := 20
def counterfeit_bills : ℕ := 5
def selected_bills : ℕ := 2

def prob_both_counterfeit : ℚ := (counterfeit_bills.choose selected_bills : ℚ) / (total_bills.choose selected_bills)
def prob_at_least_one_counterfeit : ℚ := 1 - ((total_bills - counterfeit_bills).choose selected_bills : ℚ) / (total_bills.choose selected_bills)

theorem counterfeit_probability :
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_probability_l1277_127721


namespace NUMINAMATH_CALUDE_peppers_total_weight_l1277_127734

theorem peppers_total_weight : 
  let green_peppers : Float := 0.3333333333333333
  let red_peppers : Float := 0.3333333333333333
  let yellow_peppers : Float := 0.25
  let orange_peppers : Float := 0.5
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.4166666666666665 := by
  sorry

end NUMINAMATH_CALUDE_peppers_total_weight_l1277_127734


namespace NUMINAMATH_CALUDE_male_sample_size_in_given_scenario_l1277_127709

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  female_count : ℕ
  sample_size : ℕ
  h_female_count : female_count ≤ total_population
  h_sample_size : sample_size ≤ total_population

/-- Calculates the number of male students to be drawn in a stratified sample -/
def male_sample_size (s : StratifiedSample) : ℕ :=
  ((s.total_population - s.female_count) * s.sample_size) / s.total_population

/-- Theorem stating the number of male students to be drawn in the given scenario -/
theorem male_sample_size_in_given_scenario :
  let s : StratifiedSample := {
    total_population := 900,
    female_count := 400,
    sample_size := 45,
    h_female_count := by norm_num,
    h_sample_size := by norm_num
  }
  male_sample_size s = 25 := by
  sorry

end NUMINAMATH_CALUDE_male_sample_size_in_given_scenario_l1277_127709


namespace NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l1277_127743

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The coordinates of point P -/
def P : ℝ × ℝ := (-2, 3)

/-- Theorem: The coordinates of P(-2, 3) with respect to the x-axis are (-2, -3) -/
theorem coordinates_wrt_x_axis : reflect_x P = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l1277_127743


namespace NUMINAMATH_CALUDE_cylinder_volume_l1277_127747

/-- The volume of a cylinder with base radius 1 cm and height 2 cm is 2π cm³ -/
theorem cylinder_volume : 
  let r : ℝ := 1  -- base radius in cm
  let h : ℝ := 2  -- height in cm
  let V : ℝ := π * r^2 * h  -- volume formula
  V = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1277_127747


namespace NUMINAMATH_CALUDE_tanker_fill_time_l1277_127706

/-- Proves that two pipes with filling rates of 1/30 and 1/15 of a tanker per hour
    will fill the tanker in 10 hours when used simultaneously. -/
theorem tanker_fill_time (fill_time_A fill_time_B : ℝ) 
  (h_A : fill_time_A = 30) 
  (h_B : fill_time_B = 15) : 
  1 / (1 / fill_time_A + 1 / fill_time_B) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tanker_fill_time_l1277_127706


namespace NUMINAMATH_CALUDE_ninas_pet_insect_eyes_l1277_127788

/-- The number of eyes among Nina's pet insects -/
def total_eyes (num_spiders : ℕ) (spider_eyes : ℕ) (num_ants : ℕ) (ant_eyes : ℕ) : ℕ :=
  num_spiders * spider_eyes + num_ants * ant_eyes

/-- Theorem stating the total number of eyes among Nina's pet insects -/
theorem ninas_pet_insect_eyes :
  total_eyes 3 8 50 2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_ninas_pet_insect_eyes_l1277_127788


namespace NUMINAMATH_CALUDE_min_correct_answers_to_pass_l1277_127773

/-- Represents a test with given parameters -/
structure Test where
  total_questions : ℕ
  points_correct : ℕ
  points_wrong : ℕ
  passing_score : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : Test) (correct_answers : ℕ) : ℤ :=
  (test.points_correct * correct_answers : ℤ) - 
  (test.points_wrong * (test.total_questions - correct_answers) : ℤ)

/-- Theorem stating the minimum number of correct answers needed to pass the test -/
theorem min_correct_answers_to_pass (test : Test) 
  (h1 : test.total_questions = 20)
  (h2 : test.points_correct = 5)
  (h3 : test.points_wrong = 3)
  (h4 : test.passing_score = 60) :
  ∀ n : ℕ, n ≥ 15 ↔ calculate_score test n ≥ test.passing_score := by
  sorry

#check min_correct_answers_to_pass

end NUMINAMATH_CALUDE_min_correct_answers_to_pass_l1277_127773


namespace NUMINAMATH_CALUDE_quarters_count_l1277_127772

/-- Proves that given 21 coins consisting of nickels and quarters with a total value of $3.65, the number of quarters is 13. -/
theorem quarters_count (total_coins : ℕ) (total_value : ℚ) (nickels : ℕ) (quarters : ℕ) : 
  total_coins = 21 →
  total_value = 365/100 →
  total_coins = nickels + quarters →
  total_value = (5 * nickels + 25 * quarters) / 100 →
  quarters = 13 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l1277_127772


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1277_127711

theorem quadratic_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0) ↔ (a = -1 ∨ a = 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1277_127711


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1277_127757

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}

theorem intersection_M_complement_N : M ∩ (U \ N) = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1277_127757


namespace NUMINAMATH_CALUDE_inequality_proof_l1277_127755

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2 * Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1277_127755


namespace NUMINAMATH_CALUDE_min_bottles_for_27_people_min_bottles_sufficient_l1277_127720

/-- The minimum number of bottles needed to be purchased for a given number of people,
    given that 3 empty bottles can be exchanged for 1 full bottle -/
def min_bottles_to_purchase (num_people : ℕ) : ℕ :=
  (2 * num_people + 2) / 3

/-- Proof that for 27 people, the minimum number of bottles to purchase is 18 -/
theorem min_bottles_for_27_people :
  min_bottles_to_purchase 27 = 18 := by
  sorry

/-- Proof that the calculated minimum number of bottles is sufficient for all people -/
theorem min_bottles_sufficient (num_people : ℕ) :
  min_bottles_to_purchase num_people + (min_bottles_to_purchase num_people) / 2 ≥ num_people := by
  sorry

end NUMINAMATH_CALUDE_min_bottles_for_27_people_min_bottles_sufficient_l1277_127720


namespace NUMINAMATH_CALUDE_birds_remaining_count_l1277_127742

/-- The number of grey birds initially in the cage -/
def grey_birds : ℕ := 40

/-- The number of white birds next to the cage -/
def white_birds : ℕ := grey_birds + 6

/-- The number of grey birds remaining after half are freed -/
def remaining_grey_birds : ℕ := grey_birds / 2

/-- The total number of birds remaining after ten minutes -/
def total_remaining_birds : ℕ := remaining_grey_birds + white_birds

theorem birds_remaining_count : total_remaining_birds = 66 := by
  sorry

end NUMINAMATH_CALUDE_birds_remaining_count_l1277_127742


namespace NUMINAMATH_CALUDE_carrot_harvest_calculation_l1277_127756

/-- Calculates the expected carrot harvest from a rectangular backyard --/
theorem carrot_harvest_calculation 
  (length_paces width_paces : ℕ) 
  (pace_to_feet : ℝ) 
  (carrot_yield_per_sqft : ℝ) : 
  length_paces = 25 → 
  width_paces = 30 → 
  pace_to_feet = 2.5 → 
  carrot_yield_per_sqft = 0.5 → 
  (length_paces : ℝ) * pace_to_feet * (width_paces : ℝ) * pace_to_feet * carrot_yield_per_sqft = 2343.75 := by
  sorry

#check carrot_harvest_calculation

end NUMINAMATH_CALUDE_carrot_harvest_calculation_l1277_127756


namespace NUMINAMATH_CALUDE_markup_percentage_is_45_l1277_127745

/-- Given a cost price, discount, and profit percentage, calculate the markup percentage. -/
def calculate_markup_percentage (cost_price discount : ℚ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage)
  let marked_price := selling_price + discount
  let markup := marked_price - cost_price
  (markup / cost_price) * 100

/-- Theorem: Given the specific values in the problem, the markup percentage is 45%. -/
theorem markup_percentage_is_45 :
  let cost_price : ℚ := 180
  let discount : ℚ := 45
  let profit_percentage : ℚ := 0.20
  calculate_markup_percentage cost_price discount profit_percentage = 45 := by
  sorry

#eval calculate_markup_percentage 180 45 0.20

end NUMINAMATH_CALUDE_markup_percentage_is_45_l1277_127745


namespace NUMINAMATH_CALUDE_lemonade_ratio_l1277_127741

/-- Given that 36 lemons make 48 gallons of lemonade, this theorem proves that 4.5 lemons are needed for 6 gallons of lemonade. -/
theorem lemonade_ratio (lemons : ℝ) (gallons : ℝ) 
  (h1 : lemons / gallons = 36 / 48) 
  (h2 : gallons = 6) : 
  lemons = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_ratio_l1277_127741


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1277_127746

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1277_127746


namespace NUMINAMATH_CALUDE_stone_statue_cost_is_20_l1277_127735

/-- The cost of a stone statue -/
def stone_statue_cost : ℚ := 20

/-- The number of stone statues produced monthly -/
def stone_statues_per_month : ℕ := 10

/-- The number of wooden statues produced monthly -/
def wooden_statues_per_month : ℕ := 20

/-- The cost of a wooden statue -/
def wooden_statue_cost : ℚ := 5

/-- The tax rate as a decimal -/
def tax_rate : ℚ := 1/10

/-- The monthly earnings after taxes -/
def monthly_earnings_after_taxes : ℚ := 270

/-- Theorem stating that the cost of a stone statue is $20 -/
theorem stone_statue_cost_is_20 :
  stone_statue_cost * stone_statues_per_month +
  wooden_statue_cost * wooden_statues_per_month =
  monthly_earnings_after_taxes / (1 - tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_stone_statue_cost_is_20_l1277_127735


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l1277_127718

/-- The probability of selecting at least one woman when choosing 4 people at random from a group of 10 men and 5 women -/
theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  (1 : ℚ) - (men.choose selected : ℚ) / (total_people.choose selected : ℚ) = 84 / 91 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l1277_127718


namespace NUMINAMATH_CALUDE_extremum_and_derivative_not_equivalent_l1277_127798

-- Define a function type that represents real-valued functions of a real variable
def RealFunction := ℝ → ℝ

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : RealFunction) (a : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - a| < ε → f a ≤ f x ∨ f a ≥ f x

-- Define the derivative of a function at a point
noncomputable def has_derivative_at (f : RealFunction) (a : ℝ) (f' : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a - f' * (x - a)| ≤ ε * |x - a|

-- Theorem statement
theorem extremum_and_derivative_not_equivalent :
  ∃ (f : RealFunction) (a : ℝ),
    (has_extremum f a ∧ ¬(has_derivative_at f a 0)) ∧
    ∃ (g : RealFunction) (b : ℝ),
      (has_derivative_at g b 0 ∧ ¬(has_extremum g b)) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_derivative_not_equivalent_l1277_127798


namespace NUMINAMATH_CALUDE_f_geq_2_iff_max_m_value_l1277_127713

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the first part of the problem
theorem f_geq_2_iff (x : ℝ) :
  f x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 :=
sorry

-- Theorem for the second part of the problem
theorem max_m_value :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ -2*x^2 + m) ∧
  (∀ m : ℝ, (∀ x : ℝ, f x ≥ -2*x^2 + m) → m ≤ 5/2) :=
sorry

end NUMINAMATH_CALUDE_f_geq_2_iff_max_m_value_l1277_127713


namespace NUMINAMATH_CALUDE_food_distribution_l1277_127795

/-- Given a total amount of food and a number of full boxes, calculates the amount of food per box. -/
def food_per_box (total_food : ℕ) (num_boxes : ℕ) : ℚ :=
  (total_food : ℚ) / (num_boxes : ℚ)

/-- Proves that given 777 kilograms of food and 388 full boxes, each box contains 2 kilograms of food. -/
theorem food_distribution (total_food : ℕ) (num_boxes : ℕ) 
  (h1 : total_food = 777) (h2 : num_boxes = 388) : 
  food_per_box total_food num_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_food_distribution_l1277_127795


namespace NUMINAMATH_CALUDE_valid_triplet_configurations_l1277_127703

/-- A structure representing a configuration of triplet subsets satisfying the given conditions -/
structure TripletConfiguration (n : ℕ) :=
  (m : ℕ)
  (subsets : Fin m → Finset (Fin n))
  (cover_pairs : ∀ (i j : Fin n), i ≠ j → ∃ (k : Fin m), {i, j} ⊆ subsets k)
  (subset_size : ∀ (k : Fin m), (subsets k).card = 3)
  (intersect_one : ∀ (k₁ k₂ : Fin m), k₁ ≠ k₂ → (subsets k₁ ∩ subsets k₂).card = 1)

/-- The theorem stating that the only valid configurations are (1, 3) and (7, 7) -/
theorem valid_triplet_configurations :
  {n : ℕ | ∃ (c : TripletConfiguration n), True} = {3, 7} :=
sorry

end NUMINAMATH_CALUDE_valid_triplet_configurations_l1277_127703


namespace NUMINAMATH_CALUDE_product_of_numbers_l1277_127768

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1277_127768


namespace NUMINAMATH_CALUDE_ben_initial_eggs_l1277_127701

/-- The number of eggs Ben had initially -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Ben ate in the morning -/
def morning_eggs : ℕ := 4

/-- The number of eggs Ben ate in the afternoon -/
def afternoon_eggs : ℕ := 3

/-- The number of eggs Ben has left -/
def remaining_eggs : ℕ := 13

/-- Theorem stating that Ben initially had 20 eggs -/
theorem ben_initial_eggs : initial_eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_ben_initial_eggs_l1277_127701


namespace NUMINAMATH_CALUDE_cube_function_properties_l1277_127774

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cube_function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x y : ℝ, x < y → f x < f y)  -- f is monotonically increasing
  := by sorry

end NUMINAMATH_CALUDE_cube_function_properties_l1277_127774


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l1277_127770

theorem angle_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) 
  (h2 : Real.sin α + Real.cos α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l1277_127770


namespace NUMINAMATH_CALUDE_distance_ratio_bound_l1277_127700

/-- Given n points on a plane with maximum distance D and minimum distance d between any two points,
    the ratio of maximum to minimum distance is greater than (√(nπ)/2) - 1. -/
theorem distance_ratio_bound (n : ℕ) (D d : ℝ) (h_pos : 0 < d) (h_max : d ≤ D) :
  D / d > Real.sqrt (n * Real.pi) / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_bound_l1277_127700


namespace NUMINAMATH_CALUDE_negation_of_cubic_inequality_l1277_127765

theorem negation_of_cubic_inequality :
  (¬ (∀ x : ℝ, x^3 - x ≥ 0)) ↔ (∃ x : ℝ, x^3 - x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_cubic_inequality_l1277_127765


namespace NUMINAMATH_CALUDE_greatest_x_value_l1277_127767

theorem greatest_x_value (x : ℤ) (h : (6.1 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 620) :
  x ≤ 2 ∧ ∃ y : ℤ, y > 2 → (6.1 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 620 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1277_127767


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1277_127778

/-- 
An arithmetic sequence is a sequence where the difference between 
any two consecutive terms is constant.
-/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/--
Given an arithmetic sequence a_n where a_5 = 10 and a_12 = 31,
the common difference d is equal to 3.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a5 : a 5 = 10) 
  (h_a12 : a 12 = 31) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1277_127778


namespace NUMINAMATH_CALUDE_expand_polynomial_l1277_127791

theorem expand_polynomial (x : ℝ) : (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1277_127791


namespace NUMINAMATH_CALUDE_train_A_time_l1277_127708

/-- Represents the properties of a train journey --/
structure TrainJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup --/
def trainProblem (routeLength : ℝ) (meetingPoint : ℝ) (trainBTime : ℝ) : Prop :=
  ∃ (trainA trainB : TrainJourney),
    -- Total route length
    routeLength = 75 ∧
    -- Train B's journey
    trainB.distance = routeLength ∧
    trainB.time = trainBTime ∧
    trainB.speed = trainB.distance / trainB.time ∧
    -- Train A's journey
    trainA.distance = routeLength ∧
    -- Meeting point
    meetingPoint = 30 ∧
    -- Trains meet at the same time
    meetingPoint / trainA.speed = (routeLength - meetingPoint) / trainB.speed ∧
    -- Train A's time is the total distance divided by its speed
    trainA.time = trainA.distance / trainA.speed

/-- The theorem to prove --/
theorem train_A_time : 
  ∀ (routeLength meetingPoint trainBTime : ℝ),
    trainProblem routeLength meetingPoint trainBTime →
    ∃ (trainA : TrainJourney), trainA.time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_A_time_l1277_127708


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l1277_127782

theorem infinitely_many_primes_4k_plus_3 : 
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Prime p ∧ ∃ k : ℕ, p = 4 * k + 3 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l1277_127782


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l1277_127716

theorem sum_of_possible_x_values (x z : ℝ) (h1 : |x - z| = 100) (h2 : |z - 12| = 60) : 
  ∃ (x1 x2 x3 x4 : ℝ), 
    (|x1 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x2 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x3 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x4 - z| = 100 ∧ |z - 12| = 60) ∧
    x1 + x2 + x3 + x4 = 48 ∧
    (∀ y : ℝ, (|y - z| = 100 ∧ |z - 12| = 60) → (y = x1 ∨ y = x2 ∨ y = x3 ∨ y = x4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l1277_127716


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l1277_127725

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides and interior angles of 150°. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (perimeter side_length : ℝ) (interior_angle : ℝ),
    perimeter = 180 →
    side_length = 15 →
    n * side_length = perimeter →
    interior_angle = (n - 2) * 180 / n →
    n = 12 ∧ interior_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l1277_127725


namespace NUMINAMATH_CALUDE_geli_workout_results_l1277_127793

/-- Represents a workout routine with push-ups and runs -/
structure WorkoutRoutine where
  workoutsPerWeek : ℕ
  weeks : ℕ
  initialPushups : ℕ
  pushupIncrement : ℕ
  pushupsMileRatio : ℕ

/-- Calculates the total number of push-ups for a given workout routine -/
def totalPushups (routine : WorkoutRoutine) : ℕ :=
  let totalDays := routine.workoutsPerWeek * routine.weeks
  let lastDayPushups := routine.initialPushups + (totalDays - 1) * routine.pushupIncrement
  totalDays * (routine.initialPushups + lastDayPushups) / 2

/-- Calculates the number of one-mile runs based on the total push-ups -/
def totalRuns (routine : WorkoutRoutine) : ℕ :=
  totalPushups routine / routine.pushupsMileRatio

/-- Theorem stating the results for Geli's specific workout routine -/
theorem geli_workout_results :
  let routine : WorkoutRoutine := {
    workoutsPerWeek := 3,
    weeks := 4,
    initialPushups := 10,
    pushupIncrement := 5,
    pushupsMileRatio := 30
  }
  totalPushups routine = 450 ∧ totalRuns routine = 15 := by
  sorry


end NUMINAMATH_CALUDE_geli_workout_results_l1277_127793


namespace NUMINAMATH_CALUDE_tractor_production_proof_l1277_127724

/-- The number of tractors produced in October -/
def october_production : ℕ := 1000

/-- The additional number of tractors planned to be produced in November and December -/
def additional_production : ℕ := 2310

/-- The percentage increase of the additional production compared to the original plan -/
def percentage_increase : ℚ := 21 / 100

/-- The monthly growth rate for November and December -/
def monthly_growth_rate : ℚ := 1 / 10

/-- The original annual production plan -/
def original_annual_plan : ℕ := 11000

theorem tractor_production_proof :
  (october_production * (1 + monthly_growth_rate) + october_production * (1 + monthly_growth_rate)^2 = additional_production) ∧
  (original_annual_plan + original_annual_plan * percentage_increase = original_annual_plan + additional_production) :=
by sorry

end NUMINAMATH_CALUDE_tractor_production_proof_l1277_127724


namespace NUMINAMATH_CALUDE_evaluate_expression_l1277_127752

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 5) : 
  y * (2 * y - 5 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1277_127752


namespace NUMINAMATH_CALUDE_total_flowers_received_l1277_127719

/-- The number of types of flowers bought -/
def num_flower_types : ℕ := 4

/-- The number of pieces bought for each type of flower -/
def pieces_per_type : ℕ := 40

/-- Theorem: The total number of flowers received by the orphanage is 160 -/
theorem total_flowers_received : num_flower_types * pieces_per_type = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_received_l1277_127719


namespace NUMINAMATH_CALUDE_fiveDigitIntegersCount_eq_ten_l1277_127758

/-- The number of permutations of n elements with repetitions, where r₁, r₂, ..., rₖ
    are the repetition counts of each repeated element. -/
def permutationsWithRepetition (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The number of different five-digit integers formed using the digits 3, 3, 3, 8, and 8. -/
def fiveDigitIntegersCount : ℕ :=
  permutationsWithRepetition 5 [3, 2]

theorem fiveDigitIntegersCount_eq_ten : fiveDigitIntegersCount = 10 := by
  sorry

end NUMINAMATH_CALUDE_fiveDigitIntegersCount_eq_ten_l1277_127758


namespace NUMINAMATH_CALUDE_necklaces_made_l1277_127749

def total_beads : ℕ := 52
def beads_per_necklace : ℕ := 2

theorem necklaces_made : total_beads / beads_per_necklace = 26 := by
  sorry

end NUMINAMATH_CALUDE_necklaces_made_l1277_127749


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1277_127759

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1277_127759


namespace NUMINAMATH_CALUDE_eric_egg_collection_l1277_127776

/-- Represents the types of birds on Eric's farm -/
inductive BirdType
  | Chicken
  | Duck
  | Goose

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def num_birds (b : BirdType) : Nat :=
  match b with
  | BirdType.Chicken => 6
  | BirdType.Duck => 4
  | BirdType.Goose => 2

def normal_laying_rate (b : BirdType) : Nat :=
  match b with
  | BirdType.Chicken => 3
  | BirdType.Duck => 2
  | BirdType.Goose => 1

def is_sunday (d : Day) : Bool :=
  match d with
  | Day.Sunday => true
  | _ => false

def laying_rate (b : BirdType) (d : Day) : Nat :=
  if is_sunday d then
    max (normal_laying_rate b - 1) 0
  else
    normal_laying_rate b

def daily_eggs (d : Day) : Nat :=
  (num_birds BirdType.Chicken * laying_rate BirdType.Chicken d) +
  (num_birds BirdType.Duck * laying_rate BirdType.Duck d) +
  (num_birds BirdType.Goose * laying_rate BirdType.Goose d)

def weekly_eggs : Nat :=
  daily_eggs Day.Monday +
  daily_eggs Day.Tuesday +
  daily_eggs Day.Wednesday +
  daily_eggs Day.Thursday +
  daily_eggs Day.Friday +
  daily_eggs Day.Saturday +
  daily_eggs Day.Sunday

theorem eric_egg_collection : weekly_eggs = 184 := by
  sorry

end NUMINAMATH_CALUDE_eric_egg_collection_l1277_127776


namespace NUMINAMATH_CALUDE_integral_convergence_l1277_127738

/-- The floor function, returning the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The integrand of our improper integral -/
noncomputable def f (x : ℝ) : ℝ :=
  (-1 : ℝ) ^ (floor (1 / x)) / x

/-- Statement of the convergence properties of our improper integral -/
theorem integral_convergence :
  ¬ (∃ (I : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (a : ℝ), 0 < a ∧ a < δ → |∫ x in a..1, |f x| - I| < ε) ∧
  (∃ (I : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (a : ℝ), 0 < a ∧ a < δ → |∫ x in a..1, f x - I| < ε) :=
by sorry

end NUMINAMATH_CALUDE_integral_convergence_l1277_127738


namespace NUMINAMATH_CALUDE_e_2i_in_second_quadrant_l1277_127764

open Complex

theorem e_2i_in_second_quadrant :
  let z : ℂ := Complex.exp (2 * I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_e_2i_in_second_quadrant_l1277_127764


namespace NUMINAMATH_CALUDE_angle_PQR_is_60_degrees_l1277_127726

-- Define the points
def P : ℝ × ℝ × ℝ := (-3, 1, 7)
def Q : ℝ × ℝ × ℝ := (-4, 0, 3)
def R : ℝ × ℝ × ℝ := (-5, 0, 4)

-- Define the angle PQR in radians
def angle_PQR : ℝ := sorry

theorem angle_PQR_is_60_degrees :
  angle_PQR = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_PQR_is_60_degrees_l1277_127726


namespace NUMINAMATH_CALUDE_total_friends_count_l1277_127730

-- Define the number of friends who can pay Rs. 60 each
def standard_payers : ℕ := 10

-- Define the amount each standard payer would pay
def standard_payment : ℕ := 60

-- Define the extra amount paid by one friend
def extra_payment : ℕ := 50

-- Define the total amount paid by the friend who paid extra
def total_extra_payer_amount : ℕ := 115

-- Theorem to prove
theorem total_friends_count : 
  ∃ (n : ℕ), 
    n = standard_payers + 1 ∧ 
    n * (total_extra_payer_amount - extra_payment) = 
      standard_payers * standard_payment + extra_payment :=
by
  sorry

#check total_friends_count

end NUMINAMATH_CALUDE_total_friends_count_l1277_127730


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l1277_127710

/-- Given points A(-3,8) and B(2,2), prove that M(1,0) on the x-axis minimizes |AM| + |BM| -/
theorem minimize_sum_distances (A B M : ℝ × ℝ) : 
  A = (-3, 8) → 
  B = (2, 2) → 
  M.2 = 0 → 
  M = (1, 0) → 
  ∀ P : ℝ × ℝ, P.2 = 0 → 
    Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) + Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) ≤ 
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_minimize_sum_distances_l1277_127710


namespace NUMINAMATH_CALUDE_peter_lost_marbles_l1277_127704

/-- The number of marbles Peter lost -/
def marbles_lost (initial : ℕ) (current : ℕ) : ℕ := initial - current

/-- Theorem stating that the number of marbles Peter lost is the difference between his initial and current marbles -/
theorem peter_lost_marbles (initial : ℕ) (current : ℕ) (h : initial ≥ current) :
  marbles_lost initial current = initial - current :=
by sorry

end NUMINAMATH_CALUDE_peter_lost_marbles_l1277_127704


namespace NUMINAMATH_CALUDE_number_of_divisors_2310_l1277_127771

/-- The number of positive divisors of 2310 is 32. -/
theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_2310_l1277_127771


namespace NUMINAMATH_CALUDE_triangle_side_length_l1277_127792

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1277_127792


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1277_127723

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1277_127723


namespace NUMINAMATH_CALUDE_dans_cards_after_purchase_l1277_127763

/-- The number of baseball cards Dan has after Sam's purchase -/
def dans_remaining_cards (initial_cards sam_bought : ℕ) : ℕ :=
  initial_cards - sam_bought

/-- Theorem: Dan's remaining cards is the difference between his initial cards and those Sam bought -/
theorem dans_cards_after_purchase (initial_cards sam_bought : ℕ) 
  (h : sam_bought ≤ initial_cards) : 
  dans_remaining_cards initial_cards sam_bought = initial_cards - sam_bought := by
  sorry

end NUMINAMATH_CALUDE_dans_cards_after_purchase_l1277_127763


namespace NUMINAMATH_CALUDE_complex_equation_square_sum_l1277_127785

theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : 
  a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_square_sum_l1277_127785


namespace NUMINAMATH_CALUDE_store_payback_time_l1277_127731

/-- Calculates the time required to pay back an initial investment given monthly revenue and expenses -/
def payback_time (initial_cost : ℕ) (monthly_revenue : ℕ) (monthly_expenses : ℕ) : ℕ :=
  let monthly_profit := monthly_revenue - monthly_expenses
  initial_cost / monthly_profit

theorem store_payback_time :
  payback_time 25000 4000 1500 = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_payback_time_l1277_127731


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1277_127790

theorem max_value_quadratic (q : ℝ) : -3 * q^2 + 18 * q + 5 ≤ 32 ∧ ∃ q₀ : ℝ, -3 * q₀^2 + 18 * q₀ + 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1277_127790


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_is_56_l1277_127789

/-- The number of people the Ferris wheel can seat -/
def ferris_wheel_capacity (total_waiting : ℕ) (not_riding : ℕ) : ℕ :=
  total_waiting - not_riding

/-- Theorem: The Ferris wheel capacity is 56 people given the problem conditions -/
theorem ferris_wheel_capacity_is_56 :
  ferris_wheel_capacity 92 36 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_is_56_l1277_127789


namespace NUMINAMATH_CALUDE_almond_vs_peanut_butter_cost_difference_l1277_127715

/-- The cost difference per batch between almond butter cookies and peanut butter cookies -/
theorem almond_vs_peanut_butter_cost_difference 
  (peanut_butter_cost : ℝ) 
  (almond_butter_cost : ℝ) 
  (batch_jar_ratio : ℝ) 
  (h1 : peanut_butter_cost = 3)
  (h2 : almond_butter_cost = 3 * peanut_butter_cost)
  (h3 : batch_jar_ratio = 1 / 2) : 
  batch_jar_ratio * almond_butter_cost - batch_jar_ratio * peanut_butter_cost = 3 := by
  sorry

#check almond_vs_peanut_butter_cost_difference

end NUMINAMATH_CALUDE_almond_vs_peanut_butter_cost_difference_l1277_127715


namespace NUMINAMATH_CALUDE_odometer_puzzle_l1277_127787

theorem odometer_puzzle (a b c d : ℕ) : 
  a ≥ 1 →
  a + b + c + d ≤ 10 →
  (1000 * d + 100 * c + 10 * b + a) - (1000 * a + 100 * b + 10 * c + d) % 60 = 0 →
  a^2 + b^2 + c^2 + d^2 = 83 := by
  sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l1277_127787


namespace NUMINAMATH_CALUDE_fraction_operations_l1277_127766

/-- Define the † operation for fractions -/
def dagger (a b c d : ℚ) : ℚ := a * c * (d / b)

/-- Define the * operation for fractions -/
def star (a b c d : ℚ) : ℚ := a * c * (b / d)

/-- Theorem stating that (5/6)†(7/9)*(2/3) = 140 -/
theorem fraction_operations : 
  star (dagger (5/6) (7/9)) (2/3) = 140 := by sorry

end NUMINAMATH_CALUDE_fraction_operations_l1277_127766


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1277_127705

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a) 
  (h_sum : a 2 + a 3 + a 7 = 6) : 
  a 1 + a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1277_127705


namespace NUMINAMATH_CALUDE_surface_points_is_75_l1277_127794

/-- Represents a cube with faces marked with points -/
structure Cube where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the assembled shape of cubes -/
structure AssembledShape where
  cubes : Fin 7 → Cube
  glued_pairs : Fin 9 → Fin 7 × Fin 6 × Fin 7 × Fin 6
  glued_pairs_same_points : ∀ i : Fin 9,
    let (c1, f1, c2, f2) := glued_pairs i
    (cubes c1).faces f1 = (cubes c2).faces f2

/-- The total number of points on the surface of the assembled shape -/
def surface_points (shape : AssembledShape) : Nat :=
  sorry

/-- Theorem stating that the total number of points on the surface is 75 -/
theorem surface_points_is_75 (shape : AssembledShape) :
  surface_points shape = 75 := by
  sorry

end NUMINAMATH_CALUDE_surface_points_is_75_l1277_127794


namespace NUMINAMATH_CALUDE_sarah_money_l1277_127760

/-- Given that Bridge and Sarah have 300 cents in total, and Bridge has 50 cents more than Sarah,
    prove that Sarah has 125 cents. -/
theorem sarah_money : 
  ∀ (sarah_cents bridge_cents : ℕ), 
    sarah_cents + bridge_cents = 300 →
    bridge_cents = sarah_cents + 50 →
    sarah_cents = 125 := by
  sorry

end NUMINAMATH_CALUDE_sarah_money_l1277_127760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_and_sum_l1277_127786

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 = 11 →
  a 2 + a 6 = 18 →
  (∀ n : ℕ, b n = a n + 3^n) →
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, S n = n^2 + 2*n - 3/2 + 3^(n+1)/2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_and_sum_l1277_127786


namespace NUMINAMATH_CALUDE_expression_value_l1277_127737

theorem expression_value (y : ℝ) (some_variable : ℝ) 
  (h1 : some_variable / (2 * y) = 3 / 2)
  (h2 : (7 * some_variable + 5 * y) / (some_variable - 2 * y) = 26) :
  some_variable = 3 * y :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1277_127737


namespace NUMINAMATH_CALUDE_simplify_fraction_l1277_127728

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 5) :
  15 * x^3 * y^2 / (10 * x^2 * y^4) = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1277_127728


namespace NUMINAMATH_CALUDE_average_score_is_8_1_l1277_127777

theorem average_score_is_8_1 (shooters_7 shooters_8 shooters_9 shooters_10 : ℕ)
  (h1 : shooters_7 = 4)
  (h2 : shooters_8 = 2)
  (h3 : shooters_9 = 3)
  (h4 : shooters_10 = 1) :
  let total_points := 7 * shooters_7 + 8 * shooters_8 + 9 * shooters_9 + 10 * shooters_10
  let total_shooters := shooters_7 + shooters_8 + shooters_9 + shooters_10
  (total_points : ℚ) / total_shooters = 81 / 10 :=
by sorry

end NUMINAMATH_CALUDE_average_score_is_8_1_l1277_127777


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l1277_127780

theorem roots_polynomial_sum (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0 → 3*α^4 + 8*β^3 = 876 := by
  sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l1277_127780


namespace NUMINAMATH_CALUDE_divisible_by_72_l1277_127781

theorem divisible_by_72 (X Y : Nat) : 
  X < 10 ∧ Y < 10 →
  (42000 + X * 100 + 40 + Y) % 72 = 0 ↔ 
  ((X = 8 ∧ Y = 0) ∨ (X = 0 ∧ Y = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_72_l1277_127781


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1277_127717

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) * (Real.sqrt 5 / Real.sqrt 12) = Real.sqrt 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1277_127717


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l1277_127702

/-- A linear function y = (m-1)x + m^2 - 1 passing through the origin has m = -1 -/
theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, y = (m - 1) * x + m^2 - 1) →
  (m - 1 ≠ 0) →
  (0 : ℝ) = (m - 1) * 0 + m^2 - 1 →
  m = -1 := by
  sorry

#check linear_function_through_origin

end NUMINAMATH_CALUDE_linear_function_through_origin_l1277_127702


namespace NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l1277_127751

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem x_neg_one_is_local_minimum :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1 → |x - (-1)| < δ → f x ≥ f (-1) :=
sorry

end NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l1277_127751


namespace NUMINAMATH_CALUDE_f_decreasing_after_one_l1277_127739

def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem f_decreasing_after_one :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_after_one_l1277_127739


namespace NUMINAMATH_CALUDE_chord_bisected_by_M_l1277_127753

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a chord of the ellipse
def is_chord (A B : ℝ × ℝ) : Prop :=
  is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2

-- Define the midpoint of a chord
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define a line by its equation ax + by + c = 0
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- The main theorem
theorem chord_bisected_by_M :
  ∀ A B : ℝ × ℝ,
  is_chord A B →
  is_midpoint M A B →
  line_equation 1 2 (-4) A.1 A.2 ∧ line_equation 1 2 (-4) B.1 B.2 :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_M_l1277_127753


namespace NUMINAMATH_CALUDE_min_bailing_rate_solution_l1277_127707

/-- Represents the problem of determining the minimum bailing rate for a boat --/
def MinBailingRateProblem (distance_to_shore : ℝ) (rowing_speed : ℝ) (water_intake_rate : ℝ) (max_water_capacity : ℝ) : Prop :=
  let time_to_shore : ℝ := distance_to_shore / rowing_speed
  let total_water_intake : ℝ := water_intake_rate * time_to_shore
  let excess_water : ℝ := total_water_intake - max_water_capacity
  let min_bailing_rate : ℝ := excess_water / time_to_shore
  min_bailing_rate = 2

/-- The theorem stating the minimum bailing rate for the given problem --/
theorem min_bailing_rate_solution :
  MinBailingRateProblem 0.5 6 12 50 := by
  sorry


end NUMINAMATH_CALUDE_min_bailing_rate_solution_l1277_127707


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1277_127761

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 2}

theorem complement_of_A_in_U : 
  {x : Int | x ∈ U ∧ x ∉ A} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1277_127761
