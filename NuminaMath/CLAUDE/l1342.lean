import Mathlib

namespace baby_grab_theorem_l1342_134246

/-- Represents the number of possible outcomes when a baby grabs one item from a set of items -/
def possible_outcomes (educational living entertainment : ℕ) : ℕ :=
  educational + living + entertainment

/-- Theorem: The number of possible outcomes when a baby grabs one item
    is equal to the sum of educational, living, and entertainment items -/
theorem baby_grab_theorem (educational living entertainment : ℕ) :
  possible_outcomes educational living entertainment =
  educational + living + entertainment := by
  sorry

end baby_grab_theorem_l1342_134246


namespace denis_neighbors_l1342_134297

-- Define the set of children
inductive Child : Type
  | Anya : Child
  | Borya : Child
  | Vera : Child
  | Gena : Child
  | Denis : Child

-- Define the line as a function from position (1 to 5) to Child
def Line : Type := Fin 5 → Child

-- Define what it means for two children to be next to each other
def NextTo (line : Line) (c1 c2 : Child) : Prop :=
  ∃ i : Fin 4, (line i = c1 ∧ line (i.succ) = c2) ∨ (line i = c2 ∧ line (i.succ) = c1)

-- Define the conditions
def LineConditions (line : Line) : Prop :=
  (line 0 = Child.Borya) ∧ 
  (NextTo line Child.Vera Child.Anya) ∧
  (¬ NextTo line Child.Vera Child.Gena) ∧
  (¬ NextTo line Child.Anya Child.Borya) ∧
  (¬ NextTo line Child.Anya Child.Gena) ∧
  (¬ NextTo line Child.Borya Child.Gena)

-- Theorem statement
theorem denis_neighbors 
  (line : Line) 
  (h : LineConditions line) : 
  (NextTo line Child.Denis Child.Anya) ∧ (NextTo line Child.Denis Child.Gena) :=
sorry

end denis_neighbors_l1342_134297


namespace equality_of_squares_l1342_134229

theorem equality_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
sorry

end equality_of_squares_l1342_134229


namespace a_age_is_eleven_l1342_134207

/-- Represents a person in the problem -/
inductive Person
  | A
  | B
  | C

/-- Represents a statement made by a person -/
structure Statement where
  person : Person
  content : Nat → Nat → Nat → Prop

/-- The set of all statements made by the three people -/
def statements : List Statement := sorry

/-- Predicate to check if a set of ages is consistent with the true statements -/
def consistent (a b c : Nat) : Prop := sorry

/-- Theorem stating that A's age is 11 -/
theorem a_age_is_eleven :
  ∃ (a b c : Nat),
    consistent a b c ∧
    (∀ (x y z : Nat), consistent x y z → (x = a ∧ y = b ∧ z = c)) ∧
    a = 11 := by sorry

end a_age_is_eleven_l1342_134207


namespace distribute_volunteers_eq_twelve_l1342_134230

/-- The number of ways to distribute 8 volunteer positions to 3 schools -/
def distribute_volunteers : ℕ :=
  let total_positions := 8
  let num_schools := 3
  let total_partitions := Nat.choose (total_positions - 1) (num_schools - 1)
  let equal_allocations := 3 * 3  -- (1,1,6), (2,2,4), (3,3,2)
  total_partitions - equal_allocations

/-- Theorem: The number of ways to distribute 8 volunteer positions to 3 schools,
    with each school receiving at least one position and the allocations being unequal, is 12 -/
theorem distribute_volunteers_eq_twelve : distribute_volunteers = 12 := by
  sorry

end distribute_volunteers_eq_twelve_l1342_134230


namespace min_sum_at_six_l1342_134261

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  h1 : a 1 + a 5 = -14  -- Given condition
  h2 : S 9 = -27  -- Given condition
  h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Arithmetic sequence property

/-- The theorem stating that S_n is minimized when n = 6 -/
theorem min_sum_at_six (seq : ArithmeticSequence) : 
  ∃ (n : ℕ), ∀ (m : ℕ), seq.S n ≤ seq.S m ∧ n = 6 :=
sorry

end min_sum_at_six_l1342_134261


namespace library_books_l1342_134222

theorem library_books (initial_books : ℕ) : 
  initial_books - 124 + 22 = 234 → initial_books = 336 := by
  sorry

end library_books_l1342_134222


namespace no_divisible_sum_difference_l1342_134212

theorem no_divisible_sum_difference : 
  ¬∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ 
  ((∃ k : ℤ, A = k * (A + B) ∧ ∃ m : ℤ, B = m * (A - B)) ∨
   (∃ k : ℤ, B = k * (A + B) ∧ ∃ m : ℤ, A = m * (A - B))) :=
by sorry

end no_divisible_sum_difference_l1342_134212


namespace percentage_relationship_l1342_134243

theorem percentage_relationship (p j t : ℝ) (r : ℝ) 
  (h1 : j = p * (1 - 0.25))
  (h2 : j = t * (1 - 0.20))
  (h3 : t = p * (1 - r / 100)) :
  r = 6.25 := by
  sorry

end percentage_relationship_l1342_134243


namespace inequality_solutions_imply_range_l1342_134202

theorem inequality_solutions_imply_range (a : ℝ) : 
  (∃ x₁ x₂ : ℕ+, x₁ ≠ x₂ ∧ 
    (∀ x : ℕ+, 2 * (x : ℝ) + a ≤ 1 ↔ (x = x₁ ∨ x = x₂))) →
  -5 < a ∧ a ≤ -3 := by
sorry

end inequality_solutions_imply_range_l1342_134202


namespace power_multiplication_l1342_134211

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l1342_134211


namespace greg_ate_four_halves_l1342_134265

/-- Represents the number of whole cookies made -/
def total_cookies : ℕ := 14

/-- Represents the number of halves each cookie is cut into -/
def halves_per_cookie : ℕ := 2

/-- Represents the number of halves Brad ate -/
def brad_halves : ℕ := 6

/-- Represents the number of halves left -/
def left_halves : ℕ := 18

/-- Theorem stating that Greg ate 4 halves -/
theorem greg_ate_four_halves : 
  total_cookies * halves_per_cookie - brad_halves - left_halves = 4 := by
  sorry

end greg_ate_four_halves_l1342_134265


namespace league_games_count_l1342_134270

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem league_games_count :
  let total_teams : ℕ := 8
  let teams_per_game : ℕ := 2
  number_of_games total_teams = 28 := by
  sorry

end league_games_count_l1342_134270


namespace value_of_expression_l1342_134268

theorem value_of_expression (a b : ℝ) 
  (h1 : |a| = 2) 
  (h2 : |-b| = 5) 
  (h3 : a < b) : 
  2*a - 3*b = -11 ∨ 2*a - 3*b = -19 := by
  sorry

end value_of_expression_l1342_134268


namespace carla_project_days_l1342_134280

/-- The number of days needed to complete a project given the number of items to collect and items collected per day. -/
def daysNeeded (leaves : ℕ) (bugs : ℕ) (itemsPerDay : ℕ) : ℕ :=
  (leaves + bugs) / itemsPerDay

/-- Theorem: Carla needs 10 days to complete the project. -/
theorem carla_project_days : daysNeeded 30 20 5 = 10 := by
  sorry

end carla_project_days_l1342_134280


namespace cubic_inequality_l1342_134249

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end cubic_inequality_l1342_134249


namespace stocking_discount_percentage_l1342_134247

-- Define the given conditions
def num_grandchildren : ℕ := 5
def num_children : ℕ := 4
def stockings_per_person : ℕ := 5
def stocking_price : ℚ := 20
def monogram_price : ℚ := 5
def total_cost_after_discount : ℚ := 1035

-- Define the theorem
theorem stocking_discount_percentage :
  let total_people := num_grandchildren + num_children
  let total_stockings := total_people * stockings_per_person
  let stocking_cost := total_stockings * stocking_price
  let monogram_cost := total_stockings * monogram_price
  let total_cost_before_discount := stocking_cost + monogram_cost
  let discount_amount := total_cost_before_discount - total_cost_after_discount
  let discount_percentage := (discount_amount / total_cost_before_discount) * 100
  discount_percentage = 8 := by
sorry

end stocking_discount_percentage_l1342_134247


namespace octagon_area_equal_perimeter_l1342_134278

theorem octagon_area_equal_perimeter (s : Real) (o : Real) : 
  s > 0 → o > 0 →
  s^2 = 16 →
  4 * s = 8 * o →
  2 * (1 + Real.sqrt 2) * o^2 = 8 + 8 * Real.sqrt 2 := by
  sorry

end octagon_area_equal_perimeter_l1342_134278


namespace inequality_proof_l1342_134295

theorem inequality_proof (a b c d : ℝ) (h : a + b + c + d = 8) :
  a / (8 + b - d)^(1/3) + b / (8 + c - a)^(1/3) + c / (8 + d - b)^(1/3) + d / (8 + a - c)^(1/3) ≥ 4 := by
  sorry

end inequality_proof_l1342_134295


namespace chair_table_cost_fraction_l1342_134289

theorem chair_table_cost_fraction :
  let table_cost : ℚ := 140
  let total_cost : ℚ := 220
  let chair_cost : ℚ := (total_cost - table_cost) / 4
  chair_cost / table_cost = 1 / 7 := by
sorry

end chair_table_cost_fraction_l1342_134289


namespace absolute_value_equation_solutions_l1342_134272

theorem absolute_value_equation_solutions (z : ℝ) :
  ∃ (x y : ℝ), (|x - y^2| = z*x + y^2 ∧ z*x + y^2 ≥ 0) ↔
  ((x = 0 ∧ y = 0) ∨
   (∃ (y : ℝ), x = 2*y^2/(1-z) ∧ z ≠ 1 ∧ z > -1)) :=
by sorry

end absolute_value_equation_solutions_l1342_134272


namespace tan_alpha_2_implications_l1342_134208

theorem tan_alpha_2_implications (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6/11 ∧
  (1/4) * (Real.sin α)^2 + (1/3) * Real.sin α * Real.cos α + (1/2) * (Real.cos α)^2 + 1 = 43/30 := by
  sorry

end tan_alpha_2_implications_l1342_134208


namespace inequality_solution_set_l1342_134288

theorem inequality_solution_set (x : ℝ) : 6 + 5*x - x^2 > 0 ↔ -1 < x ∧ x < 6 := by
  sorry

end inequality_solution_set_l1342_134288


namespace expression_value_l1342_134282

theorem expression_value : 
  let a : ℝ := 1.69
  let b : ℝ := 1.73
  let c : ℝ := 0.48
  1 / (a^2 - a*c - a*b + b*c) + 2 / (b^2 - a*b - b*c + a*c) + 1 / (c^2 - a*c - b*c + a*b) = 20 :=
by sorry

end expression_value_l1342_134282


namespace complex_quadrant_l1342_134260

theorem complex_quadrant (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_quadrant_l1342_134260


namespace infinitely_many_special_integers_l1342_134223

theorem infinitely_many_special_integers (k : ℕ) (hk : k > 1) :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ x ∈ S, 
    (∃ (a b : ℕ), x = a^k - b^k) ∧ 
    (¬∃ (c d : ℕ), x = c^k + d^k)) :=
sorry

end infinitely_many_special_integers_l1342_134223


namespace polynomial_evaluation_l1342_134257

theorem polynomial_evaluation : let x : ℝ := 3
  x^6 - 4*x^2 + 3*x = 702 := by sorry

end polynomial_evaluation_l1342_134257


namespace lucy_age_l1342_134254

def FriendGroup : Type := Fin 6 → Nat

def validAges (group : FriendGroup) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 6)), ∀ i, group (perm i) = [4, 6, 8, 10, 12, 14].get i

def skateparkCondition (group : FriendGroup) : Prop :=
  ∃ i j, i ≠ j ∧ group i + group j = 18

def swimmingPoolCondition (group : FriendGroup) : Prop :=
  ∃ i j, i ≠ j ∧ group i < 12 ∧ group j < 12

def libraryCondition (group : FriendGroup) (lucyIndex : Fin 6) : Prop :=
  ∃ i, i ≠ lucyIndex ∧ group i = 6

theorem lucy_age (group : FriendGroup) (lucyIndex : Fin 6) :
  validAges group →
  skateparkCondition group →
  swimmingPoolCondition group →
  libraryCondition group lucyIndex →
  group lucyIndex = 12 :=
by sorry

end lucy_age_l1342_134254


namespace trigonometric_identity_l1342_134252

theorem trigonometric_identity (α : ℝ) : 
  4 * Real.sin (2 * α - 3/2 * Real.pi) * Real.sin (Real.pi/6 + 2 * α) * Real.sin (Real.pi/6 - 2 * α) = Real.cos (6 * α) := by
  sorry

end trigonometric_identity_l1342_134252


namespace unique_function_on_rationals_l1342_134285

theorem unique_function_on_rationals
  (f : ℚ → ℝ)
  (h1 : ∀ x y : ℚ, f (x + y) - y * f x - x * f y = f x * f y - x - y + x * y)
  (h2 : ∀ x : ℚ, f x = 2 * f (x + 1) + 2 + x)
  (h3 : f 1 + 1 > 0) :
  ∀ x : ℚ, f x = -x / 2 := by sorry

end unique_function_on_rationals_l1342_134285


namespace steven_peach_count_l1342_134209

-- Define the number of peaches Jake and Steven have
def jake_peaches : ℕ := 7
def steven_peaches : ℕ := jake_peaches + 12

-- Theorem to prove
theorem steven_peach_count : steven_peaches = 19 := by
  sorry

end steven_peach_count_l1342_134209


namespace salary_change_percentage_l1342_134266

theorem salary_change_percentage (x : ℝ) : 
  (1 - x/100) * (1 + x/100) = 0.75 → x = 50 := by
  sorry

end salary_change_percentage_l1342_134266


namespace max_product_under_constraint_l1342_134255

theorem max_product_under_constraint :
  ∀ x y : ℕ+, 
  7 * x + 4 * y = 140 → 
  x * y ≤ 168 := by
sorry

end max_product_under_constraint_l1342_134255


namespace converse_correctness_l1342_134215

-- Define the original proposition
def original_prop (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- Define the converse proposition
def converse_prop (a b : ℝ) : Prop := (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0)

-- Theorem stating that the converse_prop is indeed the converse of original_prop
theorem converse_correctness : 
  ∀ (a b : ℝ), converse_prop a b ↔ (¬(a^2 + b^2 = 0) → ¬(a = 0 ∧ b = 0)) :=
by sorry

end converse_correctness_l1342_134215


namespace jason_initial_cards_l1342_134220

/-- The number of Pokemon cards Jason had initially -/
def initial_cards : ℕ := sorry

/-- The number of Pokemon cards Alyssa bought from Jason -/
def cards_bought : ℕ := 224

/-- The number of Pokemon cards Jason has now -/
def remaining_cards : ℕ := 452

/-- Theorem stating that Jason's initial number of Pokemon cards was 676 -/
theorem jason_initial_cards : initial_cards = 676 := by
  sorry

end jason_initial_cards_l1342_134220


namespace steak_cooking_time_l1342_134267

def waffle_time : ℕ := 10
def total_time : ℕ := 28
def num_steaks : ℕ := 3

theorem steak_cooking_time :
  ∃ (steak_time : ℕ), steak_time * num_steaks + waffle_time = total_time ∧ steak_time = 6 := by
  sorry

end steak_cooking_time_l1342_134267


namespace symmetry_implies_a_equals_one_monotonic_increasing_implies_a_leq_one_max_value_on_interval_l1342_134239

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1: If f(1+x) = f(1-x) for all x, then a = 1
theorem symmetry_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a (1+x) = f a (1-x)) → a = 1 := by sorry

-- Theorem 2: If f is monotonically increasing on [1, +∞), then a ≤ 1
theorem monotonic_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x < f a y) → a ≤ 1 := by sorry

-- Theorem 3: The maximum value of f on [-1, 1] is 2
theorem max_value_on_interval (a : ℝ) :
  ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ m := by sorry

end symmetry_implies_a_equals_one_monotonic_increasing_implies_a_leq_one_max_value_on_interval_l1342_134239


namespace board_game_ratio_l1342_134251

theorem board_game_ratio (total_students : ℕ) (reading_students : ℕ) (homework_students : ℕ) :
  total_students = 24 →
  reading_students = total_students / 2 →
  homework_students = 4 →
  (total_students - (reading_students + homework_students)) * 3 = total_students :=
by
  sorry

end board_game_ratio_l1342_134251


namespace permutation_game_winning_strategy_l1342_134213

/-- The game on permutation group S_n -/
def PermutationGame (n : ℕ) : Prop :=
  n > 1 ∧
  ∃ (strategy : ℕ → Bool),
    (n ≥ 4 ∧ Odd n → strategy n = false) ∧
    (n = 2 ∨ n = 3 → strategy n = true)

/-- Theorem stating the winning strategies for different values of n -/
theorem permutation_game_winning_strategy :
  ∀ n : ℕ, PermutationGame n :=
sorry

end permutation_game_winning_strategy_l1342_134213


namespace yoyo_cost_l1342_134286

/-- Given that Mrs. Hilt bought a yoyo and a whistle for a total of 38 cents,
    and the whistle costs 14 cents, prove that the yoyo costs 24 cents. -/
theorem yoyo_cost (total : ℕ) (whistle : ℕ) (yoyo : ℕ)
    (h1 : total = 38)
    (h2 : whistle = 14)
    (h3 : total = whistle + yoyo) :
  yoyo = 24 := by
  sorry

end yoyo_cost_l1342_134286


namespace sum_of_roots_quadratic_l1342_134293

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 16*x - 4) → (∃ y : ℝ, y^2 = 16*y - 4 ∧ x + y = 16) := by
  sorry

end sum_of_roots_quadratic_l1342_134293


namespace system_solution_sum_l1342_134279

theorem system_solution_sum (a b : ℝ) : 
  (a * 1 + b * 2 = 4 ∧ b * 1 - a * 2 = 7) → a + b = 1 := by
  sorry

end system_solution_sum_l1342_134279


namespace angle_sum_theorem_l1342_134240

theorem angle_sum_theorem (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_equation : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π/3 := by
sorry

end angle_sum_theorem_l1342_134240


namespace x_value_l1342_134283

def A (x : ℝ) : Set ℝ := {2, x, x^2 - 30}

theorem x_value (x : ℝ) (h : -5 ∈ A x) : x = 5 := by
  sorry

end x_value_l1342_134283


namespace ten_cent_coin_count_l1342_134218

theorem ten_cent_coin_count :
  ∀ (x y : ℕ),
  x + y = 20 →
  10 * x + 50 * y = 800 →
  x = 5 :=
by
  sorry

end ten_cent_coin_count_l1342_134218


namespace total_badges_sum_l1342_134227

/-- The total number of spelling badges for Hermione, Luna, and Celestia -/
def total_badges (hermione_badges luna_badges celestia_badges : ℕ) : ℕ :=
  hermione_badges + luna_badges + celestia_badges

/-- Theorem stating that the total number of spelling badges is 83 -/
theorem total_badges_sum : total_badges 14 17 52 = 83 := by
  sorry

end total_badges_sum_l1342_134227


namespace toms_weekly_distance_l1342_134205

/-- Represents Tom's weekly exercise schedule --/
structure ExerciseSchedule where
  monday_run_morning : Real
  monday_run_evening : Real
  wednesday_run_morning : Real
  wednesday_run_evening : Real
  friday_run_first : Real
  friday_run_second : Real
  friday_run_third : Real
  tuesday_cycle_morning : Real
  tuesday_cycle_evening : Real
  thursday_cycle_morning : Real
  thursday_cycle_evening : Real

/-- Calculates the total distance Tom runs and cycles in a week --/
def total_distance (schedule : ExerciseSchedule) : Real :=
  schedule.monday_run_morning + schedule.monday_run_evening +
  schedule.wednesday_run_morning + schedule.wednesday_run_evening +
  schedule.friday_run_first + schedule.friday_run_second + schedule.friday_run_third +
  schedule.tuesday_cycle_morning + schedule.tuesday_cycle_evening +
  schedule.thursday_cycle_morning + schedule.thursday_cycle_evening

/-- Tom's actual exercise schedule --/
def toms_schedule : ExerciseSchedule :=
  { monday_run_morning := 6
  , monday_run_evening := 4
  , wednesday_run_morning := 5.25
  , wednesday_run_evening := 5
  , friday_run_first := 3
  , friday_run_second := 4.5
  , friday_run_third := 2
  , tuesday_cycle_morning := 10
  , tuesday_cycle_evening := 8
  , thursday_cycle_morning := 7
  , thursday_cycle_evening := 12
  }

/-- Theorem stating that Tom's total weekly distance is 66.75 miles --/
theorem toms_weekly_distance : total_distance toms_schedule = 66.75 := by
  sorry


end toms_weekly_distance_l1342_134205


namespace store_ordered_15_boxes_of_pencils_l1342_134235

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 80

/-- The cost of each pencil in dollars -/
def pencil_cost : ℕ := 4

/-- The cost of each pen in dollars -/
def pen_cost : ℕ := 5

/-- The additional number of pens ordered beyond twice the number of pencils -/
def additional_pens : ℕ := 300

/-- The total cost of the stationery order in dollars -/
def total_cost : ℕ := 18300

/-- Proves that the store ordered 15 boxes of pencils given the conditions -/
theorem store_ordered_15_boxes_of_pencils :
  ∃ (x : ℕ),
    x * pencils_per_box * pencil_cost +
    (2 * x * pencils_per_box + additional_pens) * pen_cost = total_cost ∧
    x = 15 := by
  sorry

end store_ordered_15_boxes_of_pencils_l1342_134235


namespace tangent_slope_implies_a_l1342_134233

/-- Given a curve y = x^4 + ax + 1 with a tangent at (-1, a+2) having slope 8, prove a = -6 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + a*x + 1
  let point : ℝ × ℝ := (-1, a + 2)
  let slope : ℝ := 8
  (f (-1) = a + 2) ∧ 
  (deriv f (-1) = slope) → 
  a = -6 := by
sorry

end tangent_slope_implies_a_l1342_134233


namespace kerosene_cost_calculation_l1342_134276

/-- The cost of a certain number of eggs in cents -/
def egg_cost : ℝ := sorry

/-- The cost of a pound of rice in cents -/
def rice_cost : ℝ := 24

/-- The cost of a half-liter of kerosene in cents -/
def half_liter_kerosene_cost : ℝ := sorry

/-- The cost of a liter of kerosene in cents -/
def liter_kerosene_cost : ℝ := sorry

theorem kerosene_cost_calculation :
  (egg_cost = rice_cost) →
  (half_liter_kerosene_cost = 6 * egg_cost) →
  (liter_kerosene_cost = 2 * half_liter_kerosene_cost) →
  liter_kerosene_cost = 288 := by
  sorry

end kerosene_cost_calculation_l1342_134276


namespace area_under_sine_curve_l1342_134244

theorem area_under_sine_curve : 
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := 2 * π / 3
  let curve (x : ℝ) := 2 * Real.sin x
  ∫ x in lower_bound..upper_bound, curve x = 3 := by
  sorry

end area_under_sine_curve_l1342_134244


namespace coffee_stock_problem_l1342_134263

/-- Proves that the weight of the second batch of coffee is 100 pounds given the initial conditions --/
theorem coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) (total_decaf_percent : ℝ) 
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.20)
  (h3 : second_batch_decaf_percent = 0.70)
  (h4 : total_decaf_percent = 0.30) : 
  ∃ (second_batch : ℝ), 
    initial_decaf_percent * initial_stock + second_batch_decaf_percent * second_batch = 
    total_decaf_percent * (initial_stock + second_batch) ∧ 
    second_batch = 100 := by
  sorry

end coffee_stock_problem_l1342_134263


namespace problem_statement_l1342_134237

theorem problem_statement (a b c k : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) 
  (h_eq : 2 * a * b * c + k * (a^2 + b^2 + c^2) = k^3) : 
  Real.sqrt ((k - a) * (k - b) / ((k + a) * (k + b))) + 
  Real.sqrt ((k - b) * (k - c) / ((k + b) * (k + c))) + 
  Real.sqrt ((k - c) * (k - a) / ((k + c) * (k + a))) = 1 := by
  sorry

end problem_statement_l1342_134237


namespace workshop_percentage_approx_29_l1342_134225

/-- Calculates the percentage of a work day spent in workshops -/
def workshop_percentage (work_day_hours : ℕ) (workshop1_minutes : ℕ) (workshop2_multiplier : ℕ) : ℚ :=
  let work_day_minutes : ℕ := work_day_hours * 60
  let workshop2_minutes : ℕ := workshop1_minutes * workshop2_multiplier
  let total_workshop_minutes : ℕ := workshop1_minutes + workshop2_minutes
  (total_workshop_minutes : ℚ) / (work_day_minutes : ℚ) * 100

/-- The percentage of the work day spent in workshops is approximately 29% -/
theorem workshop_percentage_approx_29 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |workshop_percentage 8 35 3 - 29| < ε :=
sorry

end workshop_percentage_approx_29_l1342_134225


namespace dolls_distribution_l1342_134291

theorem dolls_distribution (total_dolls : ℕ) (defective_dolls : ℕ) (num_stores : ℕ) : 
  total_dolls = 40 → defective_dolls = 4 → num_stores = 4 →
  (total_dolls - defective_dolls) / num_stores = 9 := by
  sorry

end dolls_distribution_l1342_134291


namespace logarithm_sum_equality_l1342_134296

theorem logarithm_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end logarithm_sum_equality_l1342_134296


namespace a_share_profit_l1342_134216

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_share_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem: A's share in the profit is 3660 given the investments and total profit -/
theorem a_share_profit (investment_a investment_b investment_c total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 12200) :
  calculate_share_profit investment_a investment_b investment_c total_profit = 3660 := by
  sorry

#eval calculate_share_profit 6300 4200 10500 12200

end a_share_profit_l1342_134216


namespace fraction_inequality_l1342_134299

theorem fraction_inequality (x : ℝ) : 
  -3 ≤ x ∧ x ≤ 1 ∧ (3 * x + 8 ≥ 3 * (5 - 2 * x)) → 7/9 ≤ x ∧ x ≤ 1 :=
by sorry

end fraction_inequality_l1342_134299


namespace sum_of_reciprocals_of_roots_l1342_134231

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 6 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 6 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 6) := by
sorry

end sum_of_reciprocals_of_roots_l1342_134231


namespace initial_birds_on_fence_l1342_134210

theorem initial_birds_on_fence (initial_birds storks additional_birds final_birds : ℕ) :
  initial_birds + storks > 0 ∧ 
  storks = 46 ∧ 
  additional_birds = 6 ∧ 
  final_birds = 10 ∧ 
  initial_birds + additional_birds = final_birds →
  initial_birds = 4 := by
sorry

end initial_birds_on_fence_l1342_134210


namespace shirt_cost_calculation_l1342_134298

theorem shirt_cost_calculation (C : ℝ) : 
  (C * (1 + 0.3) * 0.5 = 13) → C = 20 :=
by
  sorry

end shirt_cost_calculation_l1342_134298


namespace complex_roots_quadratic_l1342_134264

theorem complex_roots_quadratic (a b : ℝ) : 
  (∃ z₁ z₂ : ℂ, z₁ = a + 3*I ∧ z₂ = b + 7*I ∧ 
   z₁^2 - (10 + 10*I)*z₁ + (70 + 16*I) = 0 ∧
   z₂^2 - (10 + 10*I)*z₂ + (70 + 16*I) = 0) →
  a = -3.5 ∧ b = 13.5 := by
sorry

end complex_roots_quadratic_l1342_134264


namespace max_take_home_pay_l1342_134273

/-- The income that yields the maximum take-home pay given a specific tax rate -/
theorem max_take_home_pay :
  let income : ℝ → ℝ := λ x => 1000 * x
  let tax_rate : ℝ → ℝ := λ x => 0.02 * x
  let tax : ℝ → ℝ := λ x => (tax_rate x) * (income x)
  let take_home_pay : ℝ → ℝ := λ x => (income x) - (tax x)
  ∃ x : ℝ, x = 25 ∧ ∀ y : ℝ, take_home_pay y ≤ take_home_pay x :=
by sorry

end max_take_home_pay_l1342_134273


namespace min_bags_for_candy_distribution_l1342_134256

theorem min_bags_for_candy_distribution : ∃ (n : ℕ), n > 0 ∧ 
  77 % n = 0 ∧ (7 * n) % 77 = 0 ∧ (11 * n) % 77 = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → 
    77 % m ≠ 0 ∨ (7 * m) % 77 ≠ 0 ∨ (11 * m) % 77 ≠ 0 :=
by sorry

end min_bags_for_candy_distribution_l1342_134256


namespace worker_daily_hours_l1342_134292

/-- Represents the number of work hours per day for a worker -/
def daily_hours (total_hours : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℚ :=
  total_hours / (days_per_week * weeks_per_month)

/-- Theorem stating that under given conditions, a worker's daily work hours are 10 -/
theorem worker_daily_hours :
  let total_hours : ℕ := 200
  let days_per_week : ℕ := 5
  let weeks_per_month : ℕ := 4
  daily_hours total_hours days_per_week weeks_per_month = 10 := by
  sorry

end worker_daily_hours_l1342_134292


namespace apple_percentage_after_removal_l1342_134253

/-- Represents a bowl of fruit with apples and oranges -/
structure FruitBowl where
  apples : ℕ
  oranges : ℕ

/-- Calculates the percentage of apples in a fruit bowl -/
def applePercentage (bowl : FruitBowl) : ℚ :=
  (bowl.apples : ℚ) / ((bowl.apples + bowl.oranges) : ℚ) * 100

theorem apple_percentage_after_removal :
  let initialBowl : FruitBowl := { apples := 12, oranges := 23 }
  let removedOranges : ℕ := 15
  let finalBowl : FruitBowl := { apples := initialBowl.apples, oranges := initialBowl.oranges - removedOranges }
  applePercentage finalBowl = 60 := by
  sorry

end apple_percentage_after_removal_l1342_134253


namespace janet_owes_22000_l1342_134224

/-- Calculates the total amount Janet owes for wages and taxes for one month -/
def total_owed (warehouse_workers : ℕ) (managers : ℕ) (warehouse_wage : ℚ) (manager_wage : ℚ)
  (days_per_month : ℕ) (hours_per_day : ℕ) (fica_tax_rate : ℚ) : ℚ :=
  let total_hours := days_per_month * hours_per_day
  let warehouse_total := warehouse_workers * warehouse_wage * total_hours
  let manager_total := managers * manager_wage * total_hours
  let total_wages := warehouse_total + manager_total
  let fica_taxes := total_wages * fica_tax_rate
  total_wages + fica_taxes

theorem janet_owes_22000 :
  total_owed 4 2 15 20 25 8 (1/10) = 22000 := by
  sorry

end janet_owes_22000_l1342_134224


namespace infinite_non_triangular_arithmetic_sequence_l1342_134269

-- Define triangular numbers
def isTriangular (k : ℕ) : Prop :=
  ∃ n : ℕ, k = n * (n - 1) / 2

-- Define an arithmetic sequence
def isArithmeticSequence (s : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, s (n + 1) = s n + d

-- Theorem statement
theorem infinite_non_triangular_arithmetic_sequence :
  ∃ s : ℕ → ℕ, isArithmeticSequence s ∧ (∀ n : ℕ, ¬ isTriangular (s n)) :=
sorry

end infinite_non_triangular_arithmetic_sequence_l1342_134269


namespace coin_toss_heads_l1342_134258

theorem coin_toss_heads (total_tosses : ℕ) (tail_count : ℕ) (head_count : ℕ) :
  total_tosses = 10 →
  tail_count = 7 →
  head_count = total_tosses - tail_count →
  head_count = 3 := by
sorry

end coin_toss_heads_l1342_134258


namespace driver_speed_ratio_l1342_134200

/-- Two drivers meet halfway between cities A and B. The first driver left earlier
    than the second driver by an amount of time equal to half the time it would have
    taken them to meet if they had left simultaneously. This theorem proves the ratio
    of their speeds. -/
theorem driver_speed_ratio
  (x : ℝ)  -- Distance between cities A and B
  (v₁ v₂ : ℝ)  -- Speeds of the first and second driver respectively
  (h₁ : v₁ > 0)  -- First driver's speed is positive
  (h₂ : v₂ > 0)  -- Second driver's speed is positive
  (h₃ : x > 0)  -- Distance between cities is positive
  (h₄ : x / (2 * v₁) = x / (2 * v₂) + x / (2 * (v₁ + v₂)))  -- Meeting condition
  : v₂ / v₁ = (1 + Real.sqrt 5) / 2 := by
  sorry

end driver_speed_ratio_l1342_134200


namespace triangle_similarity_and_area_l1342_134219

/-- Triangle similarity and area theorem -/
theorem triangle_similarity_and_area (PQ QR YZ : ℝ) (area_XYZ : ℝ) :
  PQ = 8 →
  QR = 16 →
  YZ = 24 →
  area_XYZ = 144 →
  ∃ (XY : ℝ),
    (XY / PQ = YZ / QR) ∧
    (area_XYZ = (1/2) * YZ * (2 * area_XYZ / YZ)) ∧
    XY = 12 := by
  sorry

end triangle_similarity_and_area_l1342_134219


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l1342_134236

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l1342_134236


namespace limit_special_function_l1342_134214

/-- The limit of (x^2 + 2x - 3) / (x^2 + 4x - 5) raised to the power of 1 / (2-x) as x approaches 1 is equal to 2/3 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(((x^2 + 2*x - 3) / (x^2 + 4*x - 5))^(1/(2-x))) - (2/3)| < ε :=
by sorry

end limit_special_function_l1342_134214


namespace circle_radius_problem_l1342_134250

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if two circles are congruent -/
def are_congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Checks if a point is on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

theorem circle_radius_problem (A B C D : Circle) : 
  are_externally_tangent A B ∧ 
  are_externally_tangent B C ∧ 
  are_externally_tangent A C ∧
  is_internally_tangent A D ∧
  is_internally_tangent B D ∧
  is_internally_tangent C D ∧
  are_congruent B C ∧
  A.radius = 2 ∧
  point_on_circle D.center A →
  B.radius = 16/9 := by
  sorry

end circle_radius_problem_l1342_134250


namespace purchase_ways_count_l1342_134232

/-- Represents the number of oreo flavors --/
def oreo_flavors : ℕ := 6

/-- Represents the number of milk flavors --/
def milk_flavors : ℕ := 4

/-- Represents the total number of products they purchase collectively --/
def total_products : ℕ := 4

/-- Represents the maximum number of same flavor items Alpha can order --/
def alpha_max_same_flavor : ℕ := 2

/-- Function to calculate the number of ways Alpha and Beta can purchase products --/
def purchase_ways : ℕ := sorry

/-- Theorem stating the correct number of ways to purchase products --/
theorem purchase_ways_count : purchase_ways = 2143 := by sorry

end purchase_ways_count_l1342_134232


namespace smallest_fraction_l1342_134277

theorem smallest_fraction (x : ℝ) (h : x = 9) : 
  min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min (x/8) (x^2/64)))) = 8/(x+2) := by
  sorry

end smallest_fraction_l1342_134277


namespace smallest_n_sqrt_difference_l1342_134234

theorem smallest_n_sqrt_difference (n : ℕ) : 
  (n ≥ 2501) ↔ (Real.sqrt n - Real.sqrt (n - 1) < 0.01) :=
sorry

end smallest_n_sqrt_difference_l1342_134234


namespace sum_of_scores_l1342_134275

/-- The sum of scores in a guessing game -/
theorem sum_of_scores (hajar_score : ℕ) (score_difference : ℕ) : 
  hajar_score = 24 →
  score_difference = 21 →
  hajar_score + (hajar_score + score_difference) = 69 := by
  sorry

#check sum_of_scores

end sum_of_scores_l1342_134275


namespace smallest_in_S_l1342_134206

def S : Set Int := {0, -17, 4, 3, -2}

theorem smallest_in_S : ∀ x ∈ S, -17 ≤ x := by
  sorry

end smallest_in_S_l1342_134206


namespace ellipse_condition_l1342_134226

/-- A curve represented by the equation x²/(7-m) + y²/(m-3) = 1 is an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  7 - m > 0 ∧ m - 3 > 0 ∧ 7 - m ≠ m - 3

/-- The condition 3 < m < 7 is necessary but not sufficient for the curve to be an ellipse -/
theorem ellipse_condition (m : ℝ) :
  (is_ellipse m → 3 < m ∧ m < 7) ∧
  ¬(3 < m ∧ m < 7 → is_ellipse m) :=
sorry

end ellipse_condition_l1342_134226


namespace geography_textbook_cost_l1342_134259

/-- The cost of a geography textbook given the following conditions:
  1. 35 English textbooks and 35 geography textbooks are ordered
  2. An English book costs $7.50
  3. The total amount of the order is $630
-/
theorem geography_textbook_cost :
  let num_books : ℕ := 35
  let english_book_cost : ℚ := 7.5
  let total_cost : ℚ := 630
  let geography_book_cost : ℚ := (total_cost - num_books * english_book_cost) / num_books
  geography_book_cost = 10.5 := by
  sorry

end geography_textbook_cost_l1342_134259


namespace arithmetic_sequence_product_l1342_134262

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  (b 5 * b 6 = 21) →  -- given condition
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = -11) :=
by sorry

end arithmetic_sequence_product_l1342_134262


namespace find_first_number_l1342_134203

/-- A sequence where the sum of two numbers is always 1 less than their actual arithmetic sum -/
def SpecialSequence (a b c : ℕ) : Prop := a + b = c + 1

/-- The theorem to prove -/
theorem find_first_number (x : ℕ) :
  SpecialSequence x 9 16 → x = 8 := by
  sorry

end find_first_number_l1342_134203


namespace gcd_polynomial_and_multiple_l1342_134245

theorem gcd_polynomial_and_multiple (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a^3 + 2 * a^2 + 6 * a + 76) a = 76 := by
  sorry

end gcd_polynomial_and_multiple_l1342_134245


namespace two_sets_of_points_l1342_134228

/-- Given two sets of points in a plane, if the total number of connecting lines
    is 136 and the sum of connecting lines between the groups is 66,
    then one set contains 10 points and the other contains 7 points. -/
theorem two_sets_of_points (x y : ℕ) : 
  x + y = 17 ∧ 
  (x * (x - 1) + y * (y - 1)) / 2 = 136 ∧ 
  x * y = 66 →
  (x = 10 ∧ y = 7) ∨ (x = 7 ∧ y = 10) :=
by sorry

end two_sets_of_points_l1342_134228


namespace x_range_l1342_134287

theorem x_range (x y : ℝ) (h : x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)) :
  x ∈ Set.Icc 0 20 := by
  sorry

end x_range_l1342_134287


namespace arithmetic_sequence_common_difference_l1342_134284

/-- Given an arithmetic sequence {a_n} where a_2 = 3 and a_6 = 13, 
    prove that the common difference is 5/2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h1 : a 2 = 3) 
  (h2 : a 6 = 13) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) :
  ∃ d : ℚ, d = 5/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end arithmetic_sequence_common_difference_l1342_134284


namespace smallest_integer_980_divisors_l1342_134290

theorem smallest_integer_980_divisors (n m k : ℕ) : 
  (∀ i < n, (Nat.divisors i).card ≠ 980) →
  (Nat.divisors n).card = 980 →
  n = m * 4^k →
  ¬(4 ∣ m) →
  (∀ j, j < n → (Nat.divisors j).card = 980 → j = n) →
  m + k = 649 :=
by sorry

end smallest_integer_980_divisors_l1342_134290


namespace shaded_area_calculation_l1342_134201

/-- The area of the shaded region in a grid composed of three rectangles minus a triangle --/
theorem shaded_area_calculation (bottom_height bottom_width middle_height middle_width top_height top_width triangle_base triangle_height : ℕ) 
  (h_bottom : bottom_height = 3 ∧ bottom_width = 5)
  (h_middle : middle_height = 4 ∧ middle_width = 7)
  (h_top : top_height = 5 ∧ top_width = 12)
  (h_triangle : triangle_base = 12 ∧ triangle_height = 5) :
  (bottom_height * bottom_width + middle_height * middle_width + top_height * top_width) - (triangle_base * triangle_height / 2) = 73 := by
  sorry

end shaded_area_calculation_l1342_134201


namespace nested_root_simplification_l1342_134248

theorem nested_root_simplification (b : ℝ) (h : b > 0) :
  (((b^16)^(1/3))^(1/4))^3 * (((b^16)^(1/4))^(1/3))^3 = b^8 := by
sorry

end nested_root_simplification_l1342_134248


namespace star_difference_l1342_134238

def star (x y : ℤ) : ℤ := x * y - 2 * x + y ^ 2

theorem star_difference : (star 7 4) - (star 4 7) = -39 := by
  sorry

end star_difference_l1342_134238


namespace geometric_sequence_11th_term_l1342_134294

/-- 
Given a geometric sequence where:
  a₅ = 5 (5th term is 5)
  a₈ = 40 (8th term is 40)
Prove that a₁₁ = 320 (11th term is 320)
-/
theorem geometric_sequence_11th_term 
  (a : ℕ → ℝ) -- The geometric sequence
  (h₁ : a 5 = 5) -- 5th term is 5
  (h₂ : a 8 = 40) -- 8th term is 40
  (h₃ : ∀ n m : ℕ, a (n + m) = a n * (a 6 / a 5) ^ m) -- Geometric sequence property
  : a 11 = 320 := by
  sorry


end geometric_sequence_11th_term_l1342_134294


namespace percentage_decrease_of_b_l1342_134241

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) :
  a > 0 ∧ b > 0 ∧
  a / b = 4 / 5 ∧
  x = a * 1.25 ∧
  m = b * (1 - p / 100) ∧
  m / x = 0.2
  → p = 80 := by sorry

end percentage_decrease_of_b_l1342_134241


namespace perpendicular_to_countless_lines_perpendicular_to_intersection_perpendicular_to_plane_l1342_134271

-- Define two perpendicular planes
axiom Plane1 : Type
axiom Plane2 : Type
axiom perpendicular_planes : Plane1 → Plane2 → Prop

-- Define a line
axiom Line : Type

-- Define a line being in a plane
axiom line_in_plane : Line → Plane1 → Prop
axiom line_in_plane2 : Line → Plane2 → Prop

-- Define perpendicularity between lines
axiom perpendicular_lines : Line → Line → Prop

-- Define perpendicularity between a line and a plane
axiom perpendicular_line_plane : Line → Plane1 → Prop
axiom perpendicular_line_plane2 : Line → Plane2 → Prop

-- Define the intersection line of two planes
axiom intersection_line : Plane1 → Plane2 → Line

-- Define a point
axiom Point : Type

-- Define a point being in a plane
axiom point_in_plane : Point → Plane1 → Prop

-- Define drawing a perpendicular line from a point to a line
axiom perpendicular_from_point : Point → Line → Line

-- Theorem 1: A line in one plane must be perpendicular to countless lines in the other plane
theorem perpendicular_to_countless_lines 
  (p1 : Plane1) (p2 : Plane2) (l : Line) 
  (h1 : perpendicular_planes p1 p2) 
  (h2 : line_in_plane l p1) : 
  ∃ (S : Set Line), (∀ l' ∈ S, line_in_plane2 l' p2 ∧ perpendicular_lines l l') ∧ Set.Infinite S :=
sorry

-- Theorem 2: If a perpendicular to the intersection line is drawn from any point in one plane, 
-- then this perpendicular must be perpendicular to the other plane
theorem perpendicular_to_intersection_perpendicular_to_plane 
  (p1 : Plane1) (p2 : Plane2) (pt : Point) 
  (h1 : perpendicular_planes p1 p2) 
  (h2 : point_in_plane pt p1) :
  let i := intersection_line p1 p2
  let perp := perpendicular_from_point pt i
  perpendicular_line_plane2 perp p2 :=
sorry

end perpendicular_to_countless_lines_perpendicular_to_intersection_perpendicular_to_plane_l1342_134271


namespace range_of_a_l1342_134274

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 1) →
  (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a y < f a x) →
  (∀ x ∈ Set.Icc 1 (a + 1), ∀ y ∈ Set.Icc 1 (a + 1), |f a x - f a y| ≤ 4) →
  a ∈ Set.Icc 2 3 :=
by sorry

end range_of_a_l1342_134274


namespace susan_bought_sixty_peaches_l1342_134242

/-- Represents the number of peaches in Susan's knapsack -/
def knapsack_peaches : ℕ := 12

/-- Represents the number of cloth bags Susan has -/
def num_cloth_bags : ℕ := 2

/-- Calculates the number of peaches in each cloth bag -/
def peaches_per_cloth_bag : ℕ := 2 * knapsack_peaches

/-- Calculates the total number of peaches Susan bought -/
def total_peaches : ℕ := num_cloth_bags * peaches_per_cloth_bag + knapsack_peaches

/-- Theorem stating that Susan bought 60 peaches in total -/
theorem susan_bought_sixty_peaches : total_peaches = 60 := by
  sorry

end susan_bought_sixty_peaches_l1342_134242


namespace range_of_m_l1342_134204

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |3 - x| + |5 + x| > m) ↔ m < 8 :=
sorry

end range_of_m_l1342_134204


namespace quadratic_equation_solution_l1342_134221

theorem quadratic_equation_solution : 
  ∃ y : ℝ, y^2 - 6*y + 5 = 0 ↔ y = 1 ∨ y = 5 := by
sorry

end quadratic_equation_solution_l1342_134221


namespace can_capacity_is_eight_litres_l1342_134281

/-- Represents the contents and capacity of a can containing a mixture of milk and water. -/
structure Can where
  initial_milk : ℝ
  initial_water : ℝ
  capacity : ℝ

/-- Proves that the capacity of the can is 8 litres given the specified conditions. -/
theorem can_capacity_is_eight_litres (can : Can)
  (h1 : can.initial_milk / can.initial_water = 1 / 5)
  (h2 : (can.initial_milk + 2) / can.initial_water = 3 / 5)
  (h3 : can.capacity = can.initial_milk + can.initial_water + 2) :
  can.capacity = 8 := by
  sorry

#check can_capacity_is_eight_litres

end can_capacity_is_eight_litres_l1342_134281


namespace parallel_cuts_three_pieces_intersecting_cuts_four_pieces_l1342_134217

-- Define a square
def Square : Type := Unit

-- Define a straight cut from edge to edge
def StraightCut (s : Square) : Type := Unit

-- Define parallel cuts
def ParallelCuts (s : Square) (c1 c2 : StraightCut s) : Prop := sorry

-- Define intersecting cuts
def IntersectingCuts (s : Square) (c1 c2 : StraightCut s) : Prop := sorry

-- Define the number of pieces resulting from cuts
def NumberOfPieces (s : Square) (c1 c2 : StraightCut s) : ℕ := sorry

-- Theorem for parallel cuts
theorem parallel_cuts_three_pieces (s : Square) (c1 c2 : StraightCut s) 
  (h : ParallelCuts s c1 c2) : NumberOfPieces s c1 c2 = 3 := by sorry

-- Theorem for intersecting cuts
theorem intersecting_cuts_four_pieces (s : Square) (c1 c2 : StraightCut s) 
  (h : IntersectingCuts s c1 c2) : NumberOfPieces s c1 c2 = 4 := by sorry

end parallel_cuts_three_pieces_intersecting_cuts_four_pieces_l1342_134217
