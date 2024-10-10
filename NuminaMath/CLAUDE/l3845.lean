import Mathlib

namespace patients_ages_problem_l3845_384588

theorem patients_ages_problem : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x - y = 44 ∧ x * y = 1280 ∧ x = 64 ∧ y = 20 := by
  sorry

end patients_ages_problem_l3845_384588


namespace class_composition_l3845_384577

theorem class_composition (d m : ℕ) : 
  (d : ℚ) / (d + m : ℚ) = 3/5 →
  ((d - 1 : ℚ) / (d + m - 3 : ℚ) = 5/8) →
  d = 21 ∧ m = 14 := by
sorry

end class_composition_l3845_384577


namespace frank_reading_speed_l3845_384555

-- Define the parameters of the problem
def total_pages : ℕ := 193
def total_chapters : ℕ := 15
def total_days : ℕ := 660

-- Define the function to calculate chapters read per day
def chapters_per_day : ℚ := total_chapters / total_days

-- Theorem statement
theorem frank_reading_speed :
  chapters_per_day = 15 / 660 := by
  sorry

end frank_reading_speed_l3845_384555


namespace quadrilateral_offset_l3845_384540

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 20 →
  offset1 = 6 →
  area = 150 →
  ∃ offset2 : ℝ, 
    area = (diagonal * (offset1 + offset2)) / 2 ∧
    offset2 = 9 := by
  sorry

end quadrilateral_offset_l3845_384540


namespace remainder_5_2024_mod_17_l3845_384568

theorem remainder_5_2024_mod_17 : 5^2024 % 17 = 16 := by sorry

end remainder_5_2024_mod_17_l3845_384568


namespace simplify_square_roots_l3845_384512

theorem simplify_square_roots : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_square_roots_l3845_384512


namespace factor_implies_m_value_l3845_384585

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 3 := by
  sorry

end factor_implies_m_value_l3845_384585


namespace cubic_polynomial_integer_root_l3845_384529

/-- Given a cubic polynomial x^3 + px + q = 0 where p and q are rational,
    if 3 - √5 is a root and the polynomial has an integer root,
    then this integer root must be -6. -/
theorem cubic_polynomial_integer_root
  (p q : ℚ)
  (h1 : ∃ (x : ℝ), x^3 + p*x + q = 0)
  (h2 : (3 - Real.sqrt 5)^3 + p*(3 - Real.sqrt 5) + q = 0)
  (h3 : ∃ (r : ℤ), r^3 + p*r + q = 0) :
  ∃ (r : ℤ), r^3 + p*r + q = 0 ∧ r = -6 := by
sorry

end cubic_polynomial_integer_root_l3845_384529


namespace problem_statement_l3845_384508

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem problem_statement :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ m : ℝ, (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) → m ≤ -1/3) ∧
  (∀ a > 0, (∃ x₀ ≥ 1, f x₀ < a * (-x₀^2 + 3*x₀)) → 
    ((1/2*(Real.exp 1 + Real.exp (-1)) < a ∧ a < Real.exp 1 → a^(Real.exp 1 - 1) > Real.exp (a - 1)) ∧
     (a > Real.exp 1 → a^(Real.exp 1 - 1) < Real.exp (a - 1)))) :=
by sorry

end problem_statement_l3845_384508


namespace lowest_cost_per_pack_10_plus_cartons_cost_10_plus_lower_than_5_to_9_l3845_384565

/-- Represents the number of boxes per carton -/
def boxes_per_carton : ℕ := 15

/-- Represents the number of packs per box -/
def packs_per_box : ℕ := 12

/-- Represents the total cost for 12 cartons before discounts -/
def total_cost_12_cartons : ℝ := 3000

/-- Represents the quantity discount for 5 or more cartons -/
def quantity_discount_5_plus : ℝ := 0.10

/-- Represents the quantity discount for 10 or more cartons -/
def quantity_discount_10_plus : ℝ := 0.15

/-- Represents the gold tier membership discount -/
def gold_tier_discount : ℝ := 0.10

/-- Represents the seasonal promotion discount -/
def seasonal_discount : ℝ := 0.03

/-- Theorem stating that purchasing 10 or more cartons results in the lowest cost per pack -/
theorem lowest_cost_per_pack_10_plus_cartons :
  let cost_per_carton := total_cost_12_cartons / 12
  let packs_per_carton := boxes_per_carton * packs_per_box
  let total_discount := quantity_discount_10_plus + gold_tier_discount + seasonal_discount
  let cost_per_carton_after_discount := cost_per_carton * (1 - total_discount)
  cost_per_carton_after_discount / packs_per_carton = 1 :=
sorry

/-- Theorem stating that the cost per pack for 10 or more cartons is lower than for 5-9 cartons -/
theorem cost_10_plus_lower_than_5_to_9 :
  let cost_per_carton := total_cost_12_cartons / 12
  let packs_per_carton := boxes_per_carton * packs_per_box
  let total_discount_10_plus := quantity_discount_10_plus + gold_tier_discount + seasonal_discount
  let total_discount_5_to_9 := quantity_discount_5_plus + gold_tier_discount + seasonal_discount
  let cost_per_pack_10_plus := (cost_per_carton * (1 - total_discount_10_plus)) / packs_per_carton
  let cost_per_pack_5_to_9 := (cost_per_carton * (1 - total_discount_5_to_9)) / packs_per_carton
  cost_per_pack_10_plus < cost_per_pack_5_to_9 :=
sorry

end lowest_cost_per_pack_10_plus_cartons_cost_10_plus_lower_than_5_to_9_l3845_384565


namespace pet_show_big_dogs_l3845_384595

theorem pet_show_big_dogs :
  ∀ (big_dogs small_dogs : ℕ),
  (big_dogs : ℚ) / small_dogs = 3 / 17 →
  big_dogs + small_dogs = 80 →
  big_dogs = 12 :=
by
  sorry

end pet_show_big_dogs_l3845_384595


namespace total_students_is_540_l3845_384515

/-- Represents the student population of a high school. -/
structure StudentPopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The conditions of the student population problem. -/
def studentPopulationProblem (p : StudentPopulation) : Prop :=
  p.sophomores = 144 ∧
  p.freshmen = (125 * p.juniors) / 100 ∧
  p.sophomores = (90 * p.freshmen) / 100 ∧
  p.seniors = (20 * (p.freshmen + p.sophomores + p.juniors + p.seniors)) / 100

/-- The theorem stating that the total number of students is 540. -/
theorem total_students_is_540 (p : StudentPopulation) 
  (h : studentPopulationProblem p) : 
  p.freshmen + p.sophomores + p.juniors + p.seniors = 540 := by
  sorry


end total_students_is_540_l3845_384515


namespace x_plus_3_over_x_is_fraction_l3845_384564

/-- A fraction is an expression with a variable in the denominator. -/
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, f x = (n x) / (d x) ∧ d x ≠ 0

/-- The expression (x + 3) / x is a fraction. -/
theorem x_plus_3_over_x_is_fraction :
  is_fraction (λ x => (x + 3) / x) :=
sorry

end x_plus_3_over_x_is_fraction_l3845_384564


namespace max_acute_angles_octagon_l3845_384553

/-- A convex octagon is a polygon with 8 sides where all interior angles are less than 180 degrees. -/
def ConvexOctagon : Type := Unit

/-- An acute angle is an angle less than 90 degrees. -/
def AcuteAngle : Type := Unit

/-- The number of acute angles in a convex octagon. -/
def num_acute_angles (octagon : ConvexOctagon) : ℕ := sorry

/-- The theorem stating that the maximum number of acute angles in a convex octagon is 4. -/
theorem max_acute_angles_octagon :
  ∀ (octagon : ConvexOctagon), num_acute_angles octagon ≤ 4 :=
sorry

end max_acute_angles_octagon_l3845_384553


namespace lollipops_left_after_sharing_l3845_384510

def raspberry_lollipops : ℕ := 51
def mint_lollipops : ℕ := 121
def chocolate_lollipops : ℕ := 9
def blueberry_lollipops : ℕ := 232
def num_friends : ℕ := 13

theorem lollipops_left_after_sharing :
  (raspberry_lollipops + mint_lollipops + chocolate_lollipops + blueberry_lollipops) % num_friends = 10 := by
  sorry

end lollipops_left_after_sharing_l3845_384510


namespace average_of_20_and_22_l3845_384544

theorem average_of_20_and_22 : (20 + 22) / 2 = 21 := by
  sorry

end average_of_20_and_22_l3845_384544


namespace gold_award_middle_sum_l3845_384514

/-- Represents the sequence of gold awards --/
def gold_sequence (n : ℕ) : ℚ := sorry

theorem gold_award_middle_sum :
  (∀ i j : ℕ, i < j → i < 10 → j < 10 → gold_sequence j - gold_sequence i = (j - i) * (gold_sequence 1 - gold_sequence 0)) →
  gold_sequence 7 + gold_sequence 8 + gold_sequence 9 = 12 →
  gold_sequence 0 + gold_sequence 1 + gold_sequence 2 + gold_sequence 3 = 12 →
  gold_sequence 4 + gold_sequence 5 + gold_sequence 6 = 83/26 := by
  sorry

end gold_award_middle_sum_l3845_384514


namespace lcm_of_9_12_15_l3845_384505

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by sorry

end lcm_of_9_12_15_l3845_384505


namespace exactly_three_proper_sets_l3845_384506

/-- A set of weights is proper if it can balance any weight from 1 to 200 grams uniquely -/
def IsProperSet (s : Multiset ℕ) : Prop :=
  (s.sum = 200) ∧
  (∀ w : ℕ, w ≥ 1 ∧ w ≤ 200 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The number of different proper sets of weights -/
def NumberOfProperSets : ℕ := 3

/-- Theorem stating that there are exactly 3 different proper sets of weights -/
theorem exactly_three_proper_sets :
  (∃ (sets : Finset (Multiset ℕ)), sets.card = NumberOfProperSets ∧
    (∀ s : Multiset ℕ, s ∈ sets ↔ IsProperSet s)) ∧
  (¬∃ (sets : Finset (Multiset ℕ)), sets.card > NumberOfProperSets ∧
    (∀ s : Multiset ℕ, s ∈ sets ↔ IsProperSet s)) :=
sorry

end exactly_three_proper_sets_l3845_384506


namespace negation_of_existence_is_universal_not_l3845_384516

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |a*x + 1|

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem negation_of_existence_is_universal_not :
  (¬ ∃ a : ℝ, is_even_function (f a)) ↔ (∀ a : ℝ, ¬ is_even_function (f a)) := by sorry

end negation_of_existence_is_universal_not_l3845_384516


namespace basketball_game_difference_l3845_384504

theorem basketball_game_difference (total_games won_games lost_games : ℕ) : 
  total_games = 62 →
  won_games > lost_games →
  won_games = 45 →
  lost_games = 17 →
  won_games - lost_games = 28 :=
by sorry

end basketball_game_difference_l3845_384504


namespace sqrt_3_5_7_not_arithmetic_sequence_l3845_384594

theorem sqrt_3_5_7_not_arithmetic_sequence : 
  ¬ ∃ (d : ℝ), Real.sqrt 5 - Real.sqrt 3 = d ∧ Real.sqrt 7 - Real.sqrt 5 = d :=
by
  sorry

end sqrt_3_5_7_not_arithmetic_sequence_l3845_384594


namespace gain_percent_proof_l3845_384554

/-- Given that the cost of 20 articles equals the selling price of 10 articles,
    prove that the gain percent is 100%. -/
theorem gain_percent_proof (cost : ℝ) (sell_price : ℝ) : 
  (20 * cost = 10 * sell_price) → (sell_price - cost) / cost * 100 = 100 := by
  sorry

end gain_percent_proof_l3845_384554


namespace restaurant_production_difference_l3845_384591

/-- Represents the daily production of a restaurant -/
structure DailyProduction where
  pizzas : ℕ
  hotDogs : ℕ
  pizzasMoreThanHotDogs : pizzas > hotDogs

/-- Represents the monthly production of a restaurant -/
def MonthlyProduction (d : DailyProduction) (days : ℕ) : ℕ :=
  days * (d.pizzas + d.hotDogs)

/-- Theorem: The restaurant makes 40 more pizzas than hot dogs every day -/
theorem restaurant_production_difference (d : DailyProduction) 
    (h1 : d.hotDogs = 60)
    (h2 : MonthlyProduction d 30 = 4800) :
  d.pizzas - d.hotDogs = 40 := by
  sorry

end restaurant_production_difference_l3845_384591


namespace fifteenth_student_age_l3845_384509

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℚ) 
  (num_group1 : Nat) 
  (avg_age_group1 : ℚ) 
  (num_group2 : Nat) 
  (avg_age_group2 : ℚ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  : ℚ :=
  by
    sorry

#check fifteenth_student_age

end fifteenth_student_age_l3845_384509


namespace mario_age_is_four_l3845_384579

/-- Mario and Maria's ages satisfy the given conditions -/
structure AgesProblem where
  mario : ℕ
  maria : ℕ
  sum_ages : mario + maria = 7
  age_difference : mario = maria + 1

/-- Mario's age is 4 given the conditions -/
theorem mario_age_is_four (p : AgesProblem) : p.mario = 4 := by
  sorry

end mario_age_is_four_l3845_384579


namespace volleyball_lineup_combinations_l3845_384575

theorem volleyball_lineup_combinations (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 5) : 
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
  sorry

end volleyball_lineup_combinations_l3845_384575


namespace integer_solutions_of_inequalities_l3845_384523

theorem integer_solutions_of_inequalities :
  {x : ℤ | (2 * x - 1 < x + 1) ∧ (1 - 2 * (x - 1) ≤ 3)} = {0, 1} := by
  sorry

end integer_solutions_of_inequalities_l3845_384523


namespace probability_all_cocaptains_l3845_384534

def team1_size : ℕ := 6
def team2_size : ℕ := 9
def team3_size : ℕ := 10
def cocaptains_per_team : ℕ := 3
def num_teams : ℕ := 3
def selected_members : ℕ := 3

theorem probability_all_cocaptains :
  (1 / num_teams) * (
    1 / (team1_size.choose selected_members) +
    1 / (team2_size.choose selected_members) +
    1 / (team3_size.choose selected_members)
  ) = 53 / 2520 := by
  sorry

end probability_all_cocaptains_l3845_384534


namespace corrected_mean_l3845_384549

/-- Given 100 observations with an initial mean of 45, and three incorrect recordings
    (60 as 35, 52 as 25, and 85 as 40), the corrected mean is 45.97. -/
theorem corrected_mean (n : ℕ) (initial_mean : ℝ) 
  (error1 error2 error3 : ℝ) (h1 : n = 100) (h2 : initial_mean = 45)
  (h3 : error1 = 60 - 35) (h4 : error2 = 52 - 25) (h5 : error3 = 85 - 40) :
  let total_error := error1 + error2 + error3
  let initial_sum := n * initial_mean
  let corrected_sum := initial_sum + total_error
  corrected_sum / n = 45.97 := by
sorry

end corrected_mean_l3845_384549


namespace sum_of_valid_starting_values_l3845_384542

def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 4 * n + 1

def apply_transform (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | m + 1 => transform (apply_transform n m)

def valid_starting_values : List ℕ :=
  (List.range 100).filter (λ n => apply_transform n 6 = 1)

theorem sum_of_valid_starting_values :
  valid_starting_values.sum = 85 := by sorry

end sum_of_valid_starting_values_l3845_384542


namespace equation_solution_l3845_384519

theorem equation_solution : ∃ (a b : ℤ), a^2 * b^2 + a^2 + b^2 + 1 = 2005 ∧ (a = 2 ∧ b = 20) := by
  sorry

end equation_solution_l3845_384519


namespace inequality_solution_l3845_384525

theorem inequality_solution :
  ∀ x : ℝ, (2 < (3 * x) / (4 * x - 7) ∧ (3 * x) / (4 * x - 7) ≤ 9) ↔ 
    (21 / 11 < x ∧ x ≤ 14 / 5) := by
  sorry

end inequality_solution_l3845_384525


namespace zahar_process_terminates_l3845_384532

/-- Represents the state of the notebooks -/
def NotebookState := List Nat

/-- Represents a single operation in Zahar's process -/
def ZaharOperation (state : NotebookState) : Option NotebookState := sorry

/-- Predicate to check if the notebooks are in ascending order -/
def IsAscendingOrder (state : NotebookState) : Prop := sorry

/-- Predicate to check if a state is valid (contains numbers 1 to n) -/
def IsValidState (state : NotebookState) : Prop := sorry

/-- The main theorem stating that Zahar's process will terminate -/
theorem zahar_process_terminates (n : Nat) (initial_state : NotebookState) :
  n ≥ 1 →
  IsValidState initial_state →
  ∃ (final_state : NotebookState) (steps : Nat),
    (∀ k : Nat, k < steps → ∃ intermediate_state, ZaharOperation (intermediate_state) ≠ none) ∧
    ZaharOperation final_state = none ∧
    IsAscendingOrder final_state :=
  sorry

end zahar_process_terminates_l3845_384532


namespace pure_imaginary_ratio_l3845_384511

theorem pure_imaginary_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : ∃ (z : ℂ), z.re = 0 ∧ z = (5 - 9 * Complex.I) * (x + y * Complex.I)) : 
  x / y = -9 / 5 := by
sorry

end pure_imaginary_ratio_l3845_384511


namespace digit_difference_in_base_d_l3845_384584

/-- Represents a digit in base d -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number AB in base d to its decimal representation -/
def toDecimal (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d 
  (d : ℕ) 
  (h_d : d > 7) 
  (A B : Digit d) 
  (h_sum : toDecimal d A B + toDecimal d A A = 1 * d * d + 7 * d + 2) :
  (A.val - B.val : ℤ) = 4 :=
sorry

end digit_difference_in_base_d_l3845_384584


namespace retirement_total_is_70_l3845_384521

/-- The required total of age and years of employment for retirement -/
def retirement_total : ℕ := 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1988

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible for retirement -/
def retirement_year : ℕ := 2007

theorem retirement_total_is_70 :
  retirement_total = 
    (retirement_year - hire_year) + -- Years of employment
    (retirement_year - hire_year + hire_age) -- Age at retirement
  := by sorry

end retirement_total_is_70_l3845_384521


namespace tower_height_difference_l3845_384524

/-- Given the heights of three towers and their relationships, prove the height difference between two of them. -/
theorem tower_height_difference 
  (cn_tower_height : ℝ)
  (cn_space_needle_diff : ℝ)
  (eiffel_tower_height : ℝ)
  (h1 : cn_tower_height = 553)
  (h2 : cn_space_needle_diff = 369)
  (h3 : eiffel_tower_height = 330) :
  eiffel_tower_height - (cn_tower_height - cn_space_needle_diff) = 146 := by
  sorry

end tower_height_difference_l3845_384524


namespace unique_row_with_41_l3845_384590

/-- The number of rows in Pascal's Triangle containing 41 -/
def rows_containing_41 : ℕ := 1

/-- 41 is prime -/
axiom prime_41 : Nat.Prime 41

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- 41 appears as a binomial coefficient -/
axiom exists_41_binomial : ∃ n k : ℕ, binomial n k = 41

theorem unique_row_with_41 : 
  (∃! r : ℕ, ∃ k : ℕ, binomial r k = 41) ∧ rows_containing_41 = 1 :=
sorry

end unique_row_with_41_l3845_384590


namespace exactly_two_numbers_satisfy_l3845_384545

/-- A function that returns true if a number satisfies the given property --/
def satisfies_property (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
  ∃ (a b : ℕ),
    n = 10 * a + b ∧  -- n is represented as 10a + b
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧  -- a and b are single digits
    (n - (a + b) / 2) % 10 = 4  -- the property holds

/-- The theorem stating that exactly two numbers satisfy the property --/
theorem exactly_two_numbers_satisfy :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n, n ∈ s ↔ satisfies_property n :=
sorry

end exactly_two_numbers_satisfy_l3845_384545


namespace inequality_solution_sets_l3845_384527

theorem inequality_solution_sets (a : ℝ) :
  let f := fun x => a * x^2 - (a + 2) * x + 2
  (a = -1 → {x : ℝ | f x < 0} = {x : ℝ | x < -2 ∨ x > 1}) ∧
  (a = 0 → {x : ℝ | f x < 0} = {x : ℝ | x > 1}) ∧
  (a < 0 → {x : ℝ | f x < 0} = {x : ℝ | x < 2/a ∨ x > 1}) ∧
  (0 < a ∧ a < 2 → {x : ℝ | f x < 0} = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → {x : ℝ | f x < 0} = ∅) ∧
  (a > 2 → {x : ℝ | f x < 0} = {x : ℝ | 2/a < x ∧ x < 1}) := by
  sorry

end inequality_solution_sets_l3845_384527


namespace bd_squared_equals_sixteen_l3845_384503

theorem bd_squared_equals_sixteen
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 5)
  (h3 : 3 * a - 2 * b + 4 * c - d = 17)
  : (b - d)^2 = 16 := by
  sorry

end bd_squared_equals_sixteen_l3845_384503


namespace one_fourth_in_one_eighth_l3845_384573

theorem one_fourth_in_one_eighth : (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by sorry

end one_fourth_in_one_eighth_l3845_384573


namespace negative_fraction_comparison_l3845_384539

theorem negative_fraction_comparison : -3/5 > -3/4 := by
  sorry

end negative_fraction_comparison_l3845_384539


namespace square_difference_l3845_384566

theorem square_difference (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 := by
  sorry

end square_difference_l3845_384566


namespace probability_score_difference_not_exceeding_three_l3845_384520

def group_A : List ℕ := [88, 89, 90]
def group_B : List ℕ := [87, 88, 92]

def total_possibilities : ℕ := group_A.length * group_B.length

def favorable_outcomes : ℕ :=
  (group_A.length * group_B.length) - 
  (group_A.filter (λ x => x = 88)).length * 
  (group_B.filter (λ x => x = 92)).length

theorem probability_score_difference_not_exceeding_three :
  (favorable_outcomes : ℚ) / total_possibilities = 8 / 9 := by
  sorry

end probability_score_difference_not_exceeding_three_l3845_384520


namespace star_value_of_a_l3845_384569

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^3

-- Theorem statement
theorem star_value_of_a : 
  ∃ a : ℝ, star a 3 = 15 ∧ a = 21 :=
by sorry

end star_value_of_a_l3845_384569


namespace complex_square_calculation_l3845_384582

theorem complex_square_calculation (z : ℂ) : z = 2 + 3*I → z^2 = -5 + 12*I := by
  sorry

end complex_square_calculation_l3845_384582


namespace circle_equation_m_range_l3845_384502

/-- Given that the equation x^2 + y^2 - x + y + m = 0 represents a circle,
    prove that m < 1/2 -/
theorem circle_equation_m_range (m : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - 1/2)^2 + (y + 1/2)^2 = r^2) →
  m < 1/2 :=
by sorry

end circle_equation_m_range_l3845_384502


namespace number_exceeding_percentage_l3845_384552

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 21 ∧ x = 25 := by
  sorry

end number_exceeding_percentage_l3845_384552


namespace remaining_fuel_after_three_hours_remaining_fuel_formula_l3845_384518

/-- Represents the fuel consumption model of a car -/
structure CarFuelModel where
  initial_fuel : ℝ
  consumption_rate : ℝ

/-- Calculates the remaining fuel after a given time -/
def remaining_fuel (model : CarFuelModel) (t : ℝ) : ℝ :=
  model.initial_fuel - model.consumption_rate * t

/-- Theorem stating the remaining fuel after 3 hours for a specific car model -/
theorem remaining_fuel_after_three_hours
  (model : CarFuelModel)
  (h1 : model.initial_fuel = 100)
  (h2 : model.consumption_rate = 6) :
  remaining_fuel model 3 = 82 := by
  sorry

/-- Theorem proving the general formula for remaining fuel -/
theorem remaining_fuel_formula
  (model : CarFuelModel)
  (h1 : model.initial_fuel = 100)
  (h2 : model.consumption_rate = 6)
  (t : ℝ) :
  remaining_fuel model t = 100 - 6 * t := by
  sorry

end remaining_fuel_after_three_hours_remaining_fuel_formula_l3845_384518


namespace canoe_trip_time_rita_canoe_trip_time_l3845_384558

/-- Calculates the total time for a round trip given upstream and downstream speeds and distance -/
theorem canoe_trip_time (upstream_speed downstream_speed distance : ℝ) :
  upstream_speed > 0 →
  downstream_speed > 0 →
  distance > 0 →
  (distance / upstream_speed) + (distance / downstream_speed) =
    (upstream_speed + downstream_speed) * distance / (upstream_speed * downstream_speed) := by
  sorry

/-- Proves that Rita's canoe trip takes 8 hours -/
theorem rita_canoe_trip_time :
  let upstream_speed : ℝ := 3
  let downstream_speed : ℝ := 9
  let distance : ℝ := 18
  (distance / upstream_speed) + (distance / downstream_speed) = 8 := by
  sorry

end canoe_trip_time_rita_canoe_trip_time_l3845_384558


namespace smallest_solution_of_equation_l3845_384507

theorem smallest_solution_of_equation : ∃ x : ℝ, 
  (∀ y : ℝ, y^4 - 26*y^2 + 169 = 0 → x ≤ y) ∧ 
  x^4 - 26*x^2 + 169 = 0 ∧ 
  x = -Real.sqrt 13 := by
  sorry

end smallest_solution_of_equation_l3845_384507


namespace decimal_sum_l3845_384557

theorem decimal_sum : 0.3 + 0.08 + 0.007 = 0.387 := by
  sorry

end decimal_sum_l3845_384557


namespace original_number_proof_l3845_384592

theorem original_number_proof (x : ℝ) : 1 + 1/x = 11/5 → x = 5/6 := by
  sorry

end original_number_proof_l3845_384592


namespace interest_problem_l3845_384580

/-- Proves that given the conditions of the interest problem, the principal amount must be 400 -/
theorem interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) : 
  (P * (R + 6) * 10 / 100 - P * R * 10 / 100 = 240) → P = 400 := by
  sorry

end interest_problem_l3845_384580


namespace tenth_term_value_l3845_384598

theorem tenth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = n * (2 * n + 1)) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  a 10 = 39 := by
  sorry

end tenth_term_value_l3845_384598


namespace quadratic_equivalence_l3845_384533

theorem quadratic_equivalence :
  ∀ x y : ℝ, y = x^2 - 8*x - 1 ↔ y = (x - 4)^2 - 17 := by
sorry

end quadratic_equivalence_l3845_384533


namespace johns_trip_distance_l3845_384593

theorem johns_trip_distance : ∃ (total_distance : ℝ), 
  (total_distance / 2) + 40 + (total_distance / 4) = total_distance ∧ 
  total_distance = 160 := by
sorry

end johns_trip_distance_l3845_384593


namespace builder_nuts_boxes_l3845_384530

/-- Represents the number of boxes of nuts purchased by the builder. -/
def boxes_of_nuts : ℕ := sorry

/-- Represents the number of boxes of bolts purchased by the builder. -/
def boxes_of_bolts : ℕ := 7

/-- Represents the number of bolts in each box. -/
def bolts_per_box : ℕ := 11

/-- Represents the number of nuts in each box. -/
def nuts_per_box : ℕ := 15

/-- Represents the number of bolts left over after the project. -/
def bolts_leftover : ℕ := 3

/-- Represents the number of nuts left over after the project. -/
def nuts_leftover : ℕ := 6

/-- Represents the total number of bolts and nuts used in the project. -/
def total_used : ℕ := 113

theorem builder_nuts_boxes : 
  boxes_of_nuts = 3 ∧
  boxes_of_bolts * bolts_per_box - bolts_leftover + 
  boxes_of_nuts * nuts_per_box - nuts_leftover = total_used :=
sorry

end builder_nuts_boxes_l3845_384530


namespace geometric_sum_problem_l3845_384574

theorem geometric_sum_problem : 
  let a : ℚ := 1/2
  let r : ℚ := -1/3
  let n : ℕ := 6
  let S := (a * (1 - r^n)) / (1 - r)
  S = 91/243 := by sorry

end geometric_sum_problem_l3845_384574


namespace prob_difference_games_l3845_384589

/-- Probability of getting heads on a single toss of the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails on a single toss of the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game A -/
def p_win_game_a : ℚ := p_heads^4 + p_tails^4

/-- Probability of winning Game C -/
def p_win_game_c : ℚ := p_heads^5 + p_tails^5 + p_heads^3 * p_tails^2 + p_tails^3 * p_heads^2

/-- The difference in probabilities between winning Game A and Game C -/
theorem prob_difference_games : p_win_game_a - p_win_game_c = 3/64 := by sorry

end prob_difference_games_l3845_384589


namespace polar_line_through_point_parallel_to_axis_l3845_384546

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- The polar equation of a line parallel to the polar axis -/
def isPolarLineParallelToAxis (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ θ, f θ * Real.sin θ = k

theorem polar_line_through_point_parallel_to_axis 
  (P : PolarPoint) 
  (h_P : P.ρ = 2 ∧ P.θ = π/3) :
  isPolarLineParallelToAxis (fun θ ↦ Real.sqrt 3 / Real.sin θ) ∧ 
  (Real.sqrt 3 / Real.sin P.θ) * Real.sin P.θ = P.ρ * Real.sin P.θ :=
sorry

end polar_line_through_point_parallel_to_axis_l3845_384546


namespace problem_1_problem_2_l3845_384551

-- Problem 1
theorem problem_1 : (1 - Real.sqrt 3) ^ 0 + |-Real.sqrt 2| - 2 * Real.cos (π / 4) + (1 / 4)⁻¹ = 5 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x₁ x₂ : ℝ, x₁ = (3 + 2 * Real.sqrt 3) / 3 ∧ 
                                 x₂ = (3 - 2 * Real.sqrt 3) / 3 ∧ 
                                 3 * x₁^2 - 6 * x₁ - 1 = 0 ∧
                                 3 * x₂^2 - 6 * x₂ - 1 = 0 := by
  sorry

end problem_1_problem_2_l3845_384551


namespace wrong_height_calculation_l3845_384583

theorem wrong_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 180)
  (h3 : real_avg = 178)
  (h4 : actual_height = 106)
  : ∃ wrong_height : ℝ,
    (n * initial_avg - wrong_height + actual_height) / n = real_avg ∧ 
    wrong_height = 176 := by
  sorry

end wrong_height_calculation_l3845_384583


namespace free_flowers_per_dozen_l3845_384513

def flowers_per_dozen : ℕ := 12

theorem free_flowers_per_dozen 
  (bought_dozens : ℕ) 
  (total_flowers : ℕ) 
  (h1 : bought_dozens = 3) 
  (h2 : total_flowers = 42) : ℕ := by
  sorry

#check free_flowers_per_dozen

end free_flowers_per_dozen_l3845_384513


namespace symmetric_line_equation_l3845_384586

/-- Given a line l symmetric to the line 2x - 3y + 4 = 0 with respect to the line x = 1,
    prove that the equation of line l is 2x + 3y - 8 = 0 -/
theorem symmetric_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (2 - x, y) ∈ {(x, y) | 2*x - 3*y + 4 = 0}) →
  l = {(x, y) | 2*x + 3*y - 8 = 0} :=
by sorry

end symmetric_line_equation_l3845_384586


namespace geometric_sequence_middle_term_l3845_384578

theorem geometric_sequence_middle_term (b : ℝ) (h : b > 0) :
  (∃ s : ℝ, s ≠ 0 ∧ 10 * s = b ∧ b * s = 1/3) → b = Real.sqrt (10/3) :=
by sorry

end geometric_sequence_middle_term_l3845_384578


namespace delta_phi_equation_solution_l3845_384576

-- Define the functions δ and φ
def δ (x : ℚ) : ℚ := 4 * x + 9
def φ (x : ℚ) : ℚ := 9 * x + 6

-- State the theorem
theorem delta_phi_equation_solution :
  ∃ x : ℚ, δ (φ x) = 10 ∧ x = -23 / 36 := by
  sorry

end delta_phi_equation_solution_l3845_384576


namespace point_in_region_range_l3845_384556

theorem point_in_region_range (a : ℝ) : 
  (2 * a + 2 < 4) → (a ∈ Set.Iio 1) :=
by sorry

end point_in_region_range_l3845_384556


namespace constant_function_l3845_384587

def BoundedAbove (f : ℤ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℤ, f n ≤ M

theorem constant_function (f : ℤ → ℝ) 
  (h_bound : BoundedAbove f)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n :=
sorry

end constant_function_l3845_384587


namespace function_inequality_l3845_384597

/-- Given a function f : ℝ → ℝ with derivative f', prove that if f'(x) < f(x) for all x,
    then f(2) < e^2 * f(0) and f(2012) < e^2012 * f(0) -/
theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hf' : ∀ x, deriv f x = f' x) (h : ∀ x, f' x < f x) : 
    f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 := by
  sorry

end function_inequality_l3845_384597


namespace max_prime_area_rectangle_l3845_384563

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def rectangleArea (l w : ℕ) : ℕ := l * w

def rectanglePerimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_prime_area_rectangle (l w : ℕ) :
  rectanglePerimeter l w = 40 →
  isPrime (rectangleArea l w) →
  rectangleArea l w ≤ 19 ∧
  (rectangleArea l w = 19 → (l = 1 ∧ w = 19) ∨ (l = 19 ∧ w = 1)) :=
sorry

end max_prime_area_rectangle_l3845_384563


namespace garden_flowers_l3845_384528

/-- Represents a rectangular garden with a rose planted in it. -/
structure Garden where
  rows_front : ℕ  -- Number of rows in front of the rose
  rows_back : ℕ   -- Number of rows behind the rose
  cols_right : ℕ  -- Number of columns to the right of the rose
  cols_left : ℕ   -- Number of columns to the left of the rose

/-- Calculates the total number of flowers in the garden. -/
def total_flowers (g : Garden) : ℕ :=
  (g.rows_front + g.rows_back + 1) * (g.cols_right + g.cols_left + 1)

/-- Theorem stating that a garden with the given properties has 462 flowers. -/
theorem garden_flowers :
  ∀ (g : Garden),
    g.rows_front = 6 ∧
    g.rows_back = 15 ∧
    g.cols_right = 12 ∧
    g.cols_left = 8 →
    total_flowers g = 462 := by
  sorry

end garden_flowers_l3845_384528


namespace trigonometric_equation_solution_l3845_384543

theorem trigonometric_equation_solution (x : ℝ) :
  (5.32 * Real.sin (2 * x) * Real.sin (6 * x) * Real.cos (4 * x) + (1/4) * Real.cos (12 * x) = 0) ↔
  (∃ k : ℤ, x = (π / 8) * (2 * k + 1)) ∨
  (∃ k : ℤ, x = (π / 12) * (6 * k + 1) ∨ x = (π / 12) * (6 * k - 1)) :=
by sorry

end trigonometric_equation_solution_l3845_384543


namespace trigonometric_sum_zero_l3845_384501

theorem trigonometric_sum_zero : 
  Real.sin (29/6 * Real.pi) + Real.cos (-29/3 * Real.pi) + Real.tan (-25/4 * Real.pi) = 0 := by
  sorry

end trigonometric_sum_zero_l3845_384501


namespace correct_regression_equation_l3845_384535

/-- Represents the selling price of a product in yuan/piece -/
def selling_price : ℝ → Prop :=
  λ x => x > 0

/-- Represents the sales volume of a product in pieces -/
def sales_volume : ℝ → Prop :=
  λ y => y > 0

/-- Represents a negative correlation between sales volume and selling price -/
def negative_correlation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The regression equation for sales volume based on selling price -/
def regression_equation (x : ℝ) : ℝ :=
  -10 * x + 200

theorem correct_regression_equation :
  (∀ x, selling_price x → sales_volume (regression_equation x)) ∧
  negative_correlation regression_equation :=
sorry

end correct_regression_equation_l3845_384535


namespace parallel_vectors_k_value_l3845_384550

/-- Given two parallel vectors a and b, prove that k = -1/2 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-1, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ a = t • b) →
  k = -1/2 := by
  sorry

end parallel_vectors_k_value_l3845_384550


namespace sufficient_but_not_necessary_condition_l3845_384560

/-- A sequence of 8 positive real numbers -/
def Sequence := Fin 8 → ℝ

/-- Predicate to check if a sequence is positive -/
def is_positive (s : Sequence) : Prop :=
  ∀ i, s i > 0

/-- Predicate to check if a sequence is geometric -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ i : Fin 7, s (i + 1) = q * s i

theorem sufficient_but_not_necessary_condition (s : Sequence) 
  (h_positive : is_positive s) :
  (s 0 + s 7 < s 3 + s 4 → ¬ is_geometric s) ∧
  ∃ s' : Sequence, is_positive s' ∧ ¬ is_geometric s' ∧ s' 0 + s' 7 ≥ s' 3 + s' 4 :=
by sorry

end sufficient_but_not_necessary_condition_l3845_384560


namespace triangle_base_calculation_l3845_384538

/-- Given a triangle with area 46 cm² and height 10 cm, prove its base is 9.2 cm -/
theorem triangle_base_calculation (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 46 →
  height = 10 →
  area = (base * height) / 2 →
  base = 9.2 := by
sorry

end triangle_base_calculation_l3845_384538


namespace second_cook_selection_l3845_384548

theorem second_cook_selection (n : ℕ) (k : ℕ) : n = 9 ∧ k = 1 → Nat.choose n k = 9 := by
  sorry

end second_cook_selection_l3845_384548


namespace symmetric_point_polar_axis_l3845_384500

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Reflects a polar point about the polar axis -/
def reflectAboutPolarAxis (p : PolarPoint) : PolarPoint :=
  { r := p.r, θ := -p.θ }

theorem symmetric_point_polar_axis (A : PolarPoint) (h : A = { r := 1, θ := π/3 }) :
  reflectAboutPolarAxis A = { r := 1, θ := -π/3 } := by
  sorry

end symmetric_point_polar_axis_l3845_384500


namespace distance_equals_abs_l3845_384531

theorem distance_equals_abs (x : ℝ) : |x - 0| = |x| := by
  sorry

end distance_equals_abs_l3845_384531


namespace bound_on_c_l3845_384581

theorem bound_on_c (a b c : ℝ) 
  (sum_condition : a + 2 * b + c = 1) 
  (square_sum_condition : a^2 + b^2 + c^2 = 1) : 
  -2/3 ≤ c ∧ c ≤ 1 := by
  sorry

end bound_on_c_l3845_384581


namespace light_distance_theorem_l3845_384596

/-- The distance light travels in one year in miles -/
def light_year_miles : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℝ := 500

/-- The conversion factor from miles to kilometers -/
def miles_to_km : ℝ := 1.60934

/-- The distance light travels in the given number of years in kilometers -/
def light_distance_km : ℝ := light_year_miles * years * miles_to_km

theorem light_distance_theorem : 
  light_distance_km = 4.723e15 := by sorry

end light_distance_theorem_l3845_384596


namespace work_completion_l3845_384561

/-- Represents the number of days it takes to complete the entire work -/
def total_days : ℕ := 40

/-- Represents the number of days y takes to finish the remaining work -/
def remaining_days : ℕ := 32

/-- Represents the fraction of work completed in one day -/
def daily_work_rate : ℚ := 1 / total_days

theorem work_completion (x_days : ℕ) : 
  x_days * daily_work_rate + remaining_days * daily_work_rate = 1 → 
  x_days = 8 := by sorry

end work_completion_l3845_384561


namespace grid_game_winner_parity_l3845_384570

/-- Represents the state of a string in the grid game -/
inductive StringState
| Uncut
| Cut

/-- Represents a player in the grid game -/
inductive Player
| First
| Second

/-- Represents the grid game state -/
structure GridGame where
  m : ℕ
  n : ℕ
  strings : Array (Array StringState)

/-- Determines the winner of the grid game based on the dimensions -/
def gridGameWinner (game : GridGame) : Player :=
  if (game.m + game.n) % 2 == 0 then Player.Second else Player.First

/-- The main theorem: The winner of the grid game is determined by the parity of m + n -/
theorem grid_game_winner_parity (game : GridGame) :
  gridGameWinner game = 
    if (game.m + game.n) % 2 == 0 then Player.Second else Player.First :=
by sorry

end grid_game_winner_parity_l3845_384570


namespace expand_product_l3845_384541

theorem expand_product (x : ℝ) : (x + 4) * (x^2 - 5*x - 6) = x^3 - x^2 - 26*x - 24 := by
  sorry

end expand_product_l3845_384541


namespace factors_of_polynomial_l3845_384572

theorem factors_of_polynomial (x : ℝ) : 
  (x^4 - 4*x^2 + 4 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2)) ∧ 
  (x^4 - 4*x^2 + 4 ≠ (x - 1) * (x^3 + x^2 + x + 1)) ∧
  (x^4 - 4*x^2 + 4 ≠ (x^2 + 2) * (x^2 - 2)) := by
  sorry

end factors_of_polynomial_l3845_384572


namespace least_subtraction_for_divisibility_l3845_384517

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 509) :
  ∃ (k : ℕ), k = 14 ∧
  (∀ m : ℕ, m < k → ¬((n - m) % 9 = 0 ∧ (n - m) % 15 = 0)) ∧
  (n - k) % 9 = 0 ∧ (n - k) % 15 = 0 :=
by sorry

end least_subtraction_for_divisibility_l3845_384517


namespace returning_players_l3845_384522

theorem returning_players (new_players : ℕ) (total_groups : ℕ) (players_per_group : ℕ) : 
  new_players = 48 → total_groups = 9 → players_per_group = 6 →
  total_groups * players_per_group - new_players = 6 :=
by
  sorry

end returning_players_l3845_384522


namespace sum_reciprocals_of_quadratic_roots_l3845_384537

theorem sum_reciprocals_of_quadratic_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b)
  (eq_a : a^2 + a - 2007 = 0) (eq_b : b^2 + b - 2007 = 0) :
  1/a + 1/b = 1/2007 := by sorry

end sum_reciprocals_of_quadratic_roots_l3845_384537


namespace inequalities_for_ordered_reals_l3845_384599

theorem inequalities_for_ordered_reals 
  (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 0 > c) 
  (h4 : c > d) : 
  (a + c > b + d) ∧ 
  (a * d^2 > b * c^2) ∧ 
  ((1 : ℝ) / (b * c) < (1 : ℝ) / (a * d)) := by
  sorry

end inequalities_for_ordered_reals_l3845_384599


namespace complex_modulus_three_fourths_minus_two_fifths_i_l3845_384559

theorem complex_modulus_three_fourths_minus_two_fifths_i :
  Complex.abs (3/4 - (2/5)*Complex.I) = 17/20 := by
  sorry

end complex_modulus_three_fourths_minus_two_fifths_i_l3845_384559


namespace fraction_equality_l3845_384547

theorem fraction_equality (x y : ℝ) (h : x / y = 1 / 2) :
  (x - y) / (x + y) = -1 / 3 := by sorry

end fraction_equality_l3845_384547


namespace sin_cos_shift_l3845_384536

theorem sin_cos_shift (x : ℝ) : Real.sin (x/2) = Real.cos ((x-π)/2 - π/4) := by sorry

end sin_cos_shift_l3845_384536


namespace chord_length_implies_a_value_l3845_384562

/-- Given a polar coordinate system with a line θ = π/3 and a circle ρ = 2a * sin(θ),
    where the chord length intercepted by the line on the circle is 2√3,
    prove that a = 2. -/
theorem chord_length_implies_a_value (a : ℝ) (h1 : a > 0) : 
  (∃ (ρ : ℝ → ℝ) (θ : ℝ), 
    (θ = π / 3) ∧ 
    (ρ θ = 2 * a * Real.sin θ) ∧
    (∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3)) → 
  a = 2 := by
  sorry

end chord_length_implies_a_value_l3845_384562


namespace largest_prime_and_composite_under_20_l3845_384567

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem largest_prime_and_composite_under_20 :
  (∀ n : ℕ, is_two_digit n → n < 20 → is_prime n → n ≤ 19) ∧
  (is_prime 19) ∧
  (∀ n : ℕ, is_two_digit n → n < 20 → is_composite n → n ≤ 18) ∧
  (is_composite 18) :=
sorry

end largest_prime_and_composite_under_20_l3845_384567


namespace total_salary_after_layoffs_l3845_384526

def total_employees : ℕ := 450
def employees_2000 : ℕ := 150
def employees_2500 : ℕ := 200
def employees_3000 : ℕ := 100

def layoff_round1_2000 : ℚ := 0.20
def layoff_round1_2500 : ℚ := 0.25
def layoff_round1_3000 : ℚ := 0.15

def layoff_round2_2000 : ℚ := 0.10
def layoff_round2_2500 : ℚ := 0.15
def layoff_round2_3000 : ℚ := 0.05

def salary_2000 : ℕ := 2000
def salary_2500 : ℕ := 2500
def salary_3000 : ℕ := 3000

theorem total_salary_after_layoffs :
  let remaining_2000 := employees_2000 - ⌊employees_2000 * layoff_round1_2000⌋ - ⌊(employees_2000 - ⌊employees_2000 * layoff_round1_2000⌋) * layoff_round2_2000⌋
  let remaining_2500 := employees_2500 - ⌊employees_2500 * layoff_round1_2500⌋ - ⌊(employees_2500 - ⌊employees_2500 * layoff_round1_2500⌋) * layoff_round2_2500⌋
  let remaining_3000 := employees_3000 - ⌊employees_3000 * layoff_round1_3000⌋ - ⌊(employees_3000 - ⌊employees_3000 * layoff_round1_3000⌋) * layoff_round2_3000⌋
  remaining_2000 * salary_2000 + remaining_2500 * salary_2500 + remaining_3000 * salary_3000 = 776500 := by
sorry

end total_salary_after_layoffs_l3845_384526


namespace xyz_value_l3845_384571

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10)
  (h3 : x + y = 2 * z) :
  x * y * z = 6 := by sorry

end xyz_value_l3845_384571
