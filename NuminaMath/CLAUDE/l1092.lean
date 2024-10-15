import Mathlib

namespace NUMINAMATH_CALUDE_male_employees_count_l1092_109242

/-- Proves the number of male employees in a company given certain conditions --/
theorem male_employees_count :
  ∀ (m f : ℕ),
  (m : ℚ) / f = 7 / 8 →
  ((m + 3 : ℚ) / f = 8 / 9) →
  m = 189 := by
sorry

end NUMINAMATH_CALUDE_male_employees_count_l1092_109242


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_l1092_109266

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials :
  unitsDigit (sumFactorials 100) = unitsDigit (sumFactorials 4) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_l1092_109266


namespace NUMINAMATH_CALUDE_inverse_sum_product_l1092_109247

theorem inverse_sum_product (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3*a + b/3 ≠ 0) :
  (3*a + b/3)⁻¹ * ((3*a)⁻¹ + (b/3)⁻¹) = (a*b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l1092_109247


namespace NUMINAMATH_CALUDE_baker_cakes_l1092_109248

/-- The number of cakes Baker made initially -/
def initial_cakes : ℕ := sorry

/-- The number of cakes Baker's friend bought -/
def friend_bought : ℕ := 140

/-- The number of cakes Baker still has -/
def remaining_cakes : ℕ := 15

/-- Theorem stating that the initial number of cakes is 155 -/
theorem baker_cakes : initial_cakes = friend_bought + remaining_cakes := by sorry

end NUMINAMATH_CALUDE_baker_cakes_l1092_109248


namespace NUMINAMATH_CALUDE_billy_ate_twenty_apples_l1092_109294

/-- The number of apples Billy ate on each day of the week --/
structure BillyApples where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Billy's apple consumption --/
def billyConditions (b : BillyApples) : Prop :=
  b.monday = 2 ∧
  b.tuesday = 2 * b.monday ∧
  b.wednesday = 9 ∧
  b.thursday = 4 * b.friday ∧
  b.friday = b.monday / 2

/-- The total number of apples Billy ate in the week --/
def totalApples (b : BillyApples) : ℕ :=
  b.monday + b.tuesday + b.wednesday + b.thursday + b.friday

/-- Theorem stating that Billy ate 20 apples in total --/
theorem billy_ate_twenty_apples :
  ∃ b : BillyApples, billyConditions b ∧ totalApples b = 20 := by
  sorry


end NUMINAMATH_CALUDE_billy_ate_twenty_apples_l1092_109294


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l1092_109208

theorem quadratic_equation_coefficient (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + p) = x^2 + q*x + 12) → q = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l1092_109208


namespace NUMINAMATH_CALUDE_min_value_theorem_l1092_109290

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1092_109290


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_system_l1092_109255

theorem unique_solution_absolute_value_system :
  ∃! (x y : ℝ), 
    (abs (x + y) + abs (1 - x) = 6) ∧
    (abs (x + y + 1) + abs (1 - y) = 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_system_l1092_109255


namespace NUMINAMATH_CALUDE_puppies_feeding_theorem_l1092_109286

/-- Given the number of formula portions, puppies, and days, calculate the number of feedings per day. -/
def feedings_per_day (portions : ℕ) (puppies : ℕ) (days : ℕ) : ℚ :=
  (portions : ℚ) / (puppies * days)

/-- Theorem stating that given 105 portions of formula for 7 puppies over 5 days, the number of feedings per day is equal to 3. -/
theorem puppies_feeding_theorem :
  feedings_per_day 105 7 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_puppies_feeding_theorem_l1092_109286


namespace NUMINAMATH_CALUDE_soda_price_calculation_l1092_109260

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- The cost of a side dish in cents -/
def side_dish_cost : ℕ := 30

theorem soda_price_calculation :
  (3 * burger_cost + 2 * soda_cost + side_dish_cost = 510) →
  (2 * burger_cost + 3 * soda_cost = 540) →
  soda_cost = 132 := by sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l1092_109260


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_six_implies_product_l1092_109226

theorem sqrt_sum_equals_six_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (15 - x) = 6 →
  (8 + x) * (15 - x) = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_six_implies_product_l1092_109226


namespace NUMINAMATH_CALUDE_inequality_proof_l1092_109222

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1092_109222


namespace NUMINAMATH_CALUDE_point_b_coordinates_l1092_109288

/-- Given point A and vector a, if vector AB = 2a, then point B has specific coordinates -/
theorem point_b_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (1, -3) →
  a = (3, 4) →
  B - A = 2 • a →
  B = (7, 5) := by
  sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l1092_109288


namespace NUMINAMATH_CALUDE_correct_num_double_burgers_l1092_109217

/-- Represents the number of double burgers Caleb bought. -/
def num_double_burgers : ℕ := 37

/-- Represents the number of single burgers Caleb bought. -/
def num_single_burgers : ℕ := 50 - num_double_burgers

/-- The total cost of all burgers in cents. -/
def total_cost : ℕ := 6850

/-- The cost of a single burger in cents. -/
def single_burger_cost : ℕ := 100

/-- The cost of a double burger in cents. -/
def double_burger_cost : ℕ := 150

/-- The total number of burgers. -/
def total_burgers : ℕ := 50

theorem correct_num_double_burgers :
  num_single_burgers * single_burger_cost + num_double_burgers * double_burger_cost = total_cost ∧
  num_single_burgers + num_double_burgers = total_burgers :=
by sorry

end NUMINAMATH_CALUDE_correct_num_double_burgers_l1092_109217


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1092_109244

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = -1 + Real.sqrt 2 / 2 ∨ x = -1 - Real.sqrt 2 / 2) ∧
  (∀ x : ℝ, x^2 + 6 * x = 5 ↔ x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1092_109244


namespace NUMINAMATH_CALUDE_population_increase_l1092_109274

theorem population_increase (x : ℝ) : 
  (3 + 3 * x / 100 = 12) → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l1092_109274


namespace NUMINAMATH_CALUDE_debby_spent_14_tickets_l1092_109203

/-- The number of tickets Debby spent on a hat -/
def hat_tickets : ℕ := 2

/-- The number of tickets Debby spent on a stuffed animal -/
def stuffed_animal_tickets : ℕ := 10

/-- The number of tickets Debby spent on a yoyo -/
def yoyo_tickets : ℕ := 2

/-- The total number of tickets Debby spent -/
def total_tickets : ℕ := hat_tickets + stuffed_animal_tickets + yoyo_tickets

theorem debby_spent_14_tickets : total_tickets = 14 := by
  sorry

end NUMINAMATH_CALUDE_debby_spent_14_tickets_l1092_109203


namespace NUMINAMATH_CALUDE_total_riding_time_two_weeks_l1092_109285

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the riding time in minutes for a given day -/
def ridingTime (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 60
  | Day.Tuesday   => 30
  | Day.Wednesday => 60
  | Day.Thursday  => 30
  | Day.Friday    => 60
  | Day.Saturday  => 120
  | Day.Sunday    => 0

/-- Calculates the total riding time for one week in minutes -/
def weeklyRidingTime : ℕ :=
  (ridingTime Day.Monday) + (ridingTime Day.Tuesday) + (ridingTime Day.Wednesday) +
  (ridingTime Day.Thursday) + (ridingTime Day.Friday) + (ridingTime Day.Saturday) +
  (ridingTime Day.Sunday)

/-- Theorem: Bethany rides for 12 hours in total over a 2-week period -/
theorem total_riding_time_two_weeks :
  (2 * weeklyRidingTime) / 60 = 12 := by sorry

end NUMINAMATH_CALUDE_total_riding_time_two_weeks_l1092_109285


namespace NUMINAMATH_CALUDE_light_flash_time_l1092_109238

/-- The time taken for a light to flash 600 times, given that it flashes every 6 seconds, is equal to 1 hour -/
theorem light_flash_time (flash_interval : ℕ) (total_flashes : ℕ) (seconds_per_hour : ℕ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_per_hour = 3600 →
  (flash_interval * total_flashes) / seconds_per_hour = 1 :=
by sorry

end NUMINAMATH_CALUDE_light_flash_time_l1092_109238


namespace NUMINAMATH_CALUDE_unique_quaternary_polynomial_l1092_109299

/-- A polynomial with coefficients in {0, 1, 2, 3} -/
def QuaternaryPolynomial := List (Fin 4)

/-- Evaluate a quaternary polynomial at x = 2 -/
def evalAt2 (p : QuaternaryPolynomial) : ℕ :=
  p.enum.foldl (fun acc (i, coef) => acc + coef.val * 2^i) 0

theorem unique_quaternary_polynomial (n : ℕ) (hn : n > 0) :
  ∃! p : QuaternaryPolynomial, evalAt2 p = n := by sorry

end NUMINAMATH_CALUDE_unique_quaternary_polynomial_l1092_109299


namespace NUMINAMATH_CALUDE_smallest_positive_b_l1092_109209

/-- Circle w1 defined by the equation x^2+y^2+6x-8y-23=0 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y - 23 = 0

/-- Circle w2 defined by the equation x^2+y^2-6x-8y+65=0 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 65 = 0

/-- A circle is externally tangent to w2 -/
def externally_tangent_w2 (x y r : ℝ) : Prop := 
  r + 2 = Real.sqrt ((x - 3)^2 + (y - 4)^2)

/-- A circle is internally tangent to w1 -/
def internally_tangent_w1 (x y r : ℝ) : Prop := 
  6 - r = Real.sqrt ((x + 3)^2 + (y - 4)^2)

/-- The line y = bx contains the center (x, y) of the tangent circle -/
def center_on_line (x y b : ℝ) : Prop := y = b * x

theorem smallest_positive_b : 
  ∃ (b : ℝ), b > 0 ∧ 
  (∀ (b' : ℝ), b' > 0 → 
    (∃ (x y r : ℝ), externally_tangent_w2 x y r ∧ 
                    internally_tangent_w1 x y r ∧ 
                    center_on_line x y b') 
    → b ≤ b') ∧
  b = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_l1092_109209


namespace NUMINAMATH_CALUDE_line_point_k_value_l1092_109272

/-- A line contains the points (8,10), (0,k), and (-8,3). This theorem proves that k = 13/2. -/
theorem line_point_k_value : 
  ∀ (k : ℚ), 
  (∃ (line : Set (ℚ × ℚ)), 
    (8, 10) ∈ line ∧ 
    (0, k) ∈ line ∧ 
    (-8, 3) ∈ line ∧ 
    (∀ (x y z : ℚ × ℚ), x ∈ line → y ∈ line → z ∈ line → 
      (x.2 - y.2) * (y.1 - z.1) = (y.2 - z.2) * (x.1 - y.1))) → 
  k = 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l1092_109272


namespace NUMINAMATH_CALUDE_points_per_round_l1092_109264

/-- Given a card game where:
  * Jane ends up with 60 points
  * She lost 20 points
  * She played 8 rounds
  Prove that the number of points awarded for winning one round is 10. -/
theorem points_per_round (final_points : ℕ) (lost_points : ℕ) (rounds : ℕ) :
  final_points = 60 →
  lost_points = 20 →
  rounds = 8 →
  (final_points + lost_points) / rounds = 10 :=
by sorry

end NUMINAMATH_CALUDE_points_per_round_l1092_109264


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l1092_109233

/-- Proves that the weight of electronics is 12 pounds given the conditions of the suitcase problem -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 7 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 3) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 8) = 2 * (B / C)) -- Ratio doubles after removing 8 pounds of clothes
  : E = 12 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l1092_109233


namespace NUMINAMATH_CALUDE_tims_photos_l1092_109211

theorem tims_photos (total : ℕ) (toms_photos : ℕ) (pauls_extra : ℕ) : 
  total = 152 → toms_photos = 38 → pauls_extra = 10 →
  ∃ (tims_photos : ℕ), 
    tims_photos + toms_photos + (tims_photos + pauls_extra) = total ∧ 
    tims_photos = 52 := by
  sorry

end NUMINAMATH_CALUDE_tims_photos_l1092_109211


namespace NUMINAMATH_CALUDE_probability_one_of_each_l1092_109224

def forks : ℕ := 8
def spoons : ℕ := 9
def knives : ℕ := 10
def teaspoons : ℕ := 7

def total_silverware : ℕ := forks + spoons + knives + teaspoons

theorem probability_one_of_each (forks spoons knives teaspoons : ℕ) 
  (h1 : forks = 8) (h2 : spoons = 9) (h3 : knives = 10) (h4 : teaspoons = 7) :
  (forks * spoons * knives * teaspoons : ℚ) / (Nat.choose total_silverware 4) = 40 / 367 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l1092_109224


namespace NUMINAMATH_CALUDE_certain_number_problem_l1092_109252

theorem certain_number_problem (x : ℝ) : 
  ((x + 20) * 2) / 2 - 2 = 88 / 2 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1092_109252


namespace NUMINAMATH_CALUDE_solve_equation_l1092_109253

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := (a + 5) * b

-- State the theorem
theorem solve_equation (x : ℝ) (h : at_op x 1.3 = 11.05) : x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1092_109253


namespace NUMINAMATH_CALUDE_two_faces_same_edge_count_l1092_109251

/-- A polyhedron with n faces, where each face has between 3 and n-1 edges. -/
structure Polyhedron (n : ℕ) where
  faces : Fin n → ℕ
  face_edge_count_lower_bound : ∀ i, faces i ≥ 3
  face_edge_count_upper_bound : ∀ i, faces i ≤ n - 1

/-- There exist at least two faces with the same number of edges in any polyhedron. -/
theorem two_faces_same_edge_count {n : ℕ} (h : n > 2) (P : Polyhedron n) :
  ∃ i j, i ≠ j ∧ P.faces i = P.faces j := by
  sorry

end NUMINAMATH_CALUDE_two_faces_same_edge_count_l1092_109251


namespace NUMINAMATH_CALUDE_new_player_weight_l1092_109271

/-- Represents a basketball team --/
structure BasketballTeam where
  players : ℕ
  averageWeight : ℝ
  totalWeight : ℝ

/-- Calculates the total weight of a team --/
def totalWeight (team : BasketballTeam) : ℝ :=
  team.players * team.averageWeight

/-- Represents the change in team composition --/
structure TeamChange where
  oldTeam : BasketballTeam
  newTeam : BasketballTeam
  replacedWeight1 : ℝ
  replacedWeight2 : ℝ
  newPlayerWeight : ℝ

/-- Theorem stating the weight of the new player --/
theorem new_player_weight (change : TeamChange) 
  (h1 : change.oldTeam.players = 12)
  (h2 : change.oldTeam.averageWeight = 80)
  (h3 : change.newTeam.players = change.oldTeam.players)
  (h4 : change.newTeam.averageWeight = change.oldTeam.averageWeight + 2.5)
  (h5 : change.replacedWeight1 = 65)
  (h6 : change.replacedWeight2 = 75) :
  change.newPlayerWeight = 170 := by
  sorry

end NUMINAMATH_CALUDE_new_player_weight_l1092_109271


namespace NUMINAMATH_CALUDE_transform_trig_function_l1092_109227

/-- Given a function f(x) = (√2/2)(sin x + cos x), 
    applying a horizontal stretch by a factor of 2 
    and a left shift by π/2 results in cos(x/2) -/
theorem transform_trig_function : 
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = (Real.sqrt 2 / 2) * (Real.sin x + Real.cos x)) ∧
    (∀ x, g x = f (x / 2 + π / 2)) ∧
    (∀ x, g x = Real.cos (x / 2)) := by
  sorry

end NUMINAMATH_CALUDE_transform_trig_function_l1092_109227


namespace NUMINAMATH_CALUDE_cafeteria_tables_l1092_109284

/-- The number of tables in a cafeteria --/
def num_tables : ℕ := 15

/-- The number of seats per table --/
def seats_per_table : ℕ := 10

/-- The fraction of seats usually left unseated --/
def unseated_fraction : ℚ := 1 / 10

/-- The number of seats usually taken --/
def seats_taken : ℕ := 135

/-- Theorem stating the number of tables in the cafeteria --/
theorem cafeteria_tables :
  num_tables = seats_taken / (seats_per_table * (1 - unseated_fraction)) := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_tables_l1092_109284


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_l1092_109281

/-- The number of pens given to Sharon in a pen collection scenario --/
theorem pens_given_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : 
  initial_pens = 25 →
  mike_pens = 22 →
  final_pens = 75 →
  (initial_pens + mike_pens) * 2 - final_pens = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_l1092_109281


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1092_109258

theorem age_ratio_problem (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 24) :
  (x : ℚ) / y = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1092_109258


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l1092_109257

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ |a| = 1) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l1092_109257


namespace NUMINAMATH_CALUDE_solve_annas_candy_problem_l1092_109213

def annas_candy_problem (initial_money : ℚ) 
                         (gum_price : ℚ) 
                         (gum_quantity : ℕ) 
                         (chocolate_price : ℚ) 
                         (chocolate_quantity : ℕ) 
                         (candy_cane_price : ℚ) 
                         (money_left : ℚ) : Prop :=
  let total_spent := gum_price * gum_quantity + chocolate_price * chocolate_quantity
  let money_for_candy_canes := initial_money - total_spent - money_left
  let candy_canes_bought := money_for_candy_canes / candy_cane_price
  candy_canes_bought = 2

theorem solve_annas_candy_problem : 
  annas_candy_problem 10 1 3 1 5 (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_annas_candy_problem_l1092_109213


namespace NUMINAMATH_CALUDE_area_between_quartic_and_line_l1092_109239

/-- The area between a quartic function and a line that touch at two points -/
theorem area_between_quartic_and_line 
  (a b c d e p q α β : ℝ) 
  (ha : a ≠ 0) 
  (hαβ : α < β) : 
  let f := fun (x : ℝ) ↦ a * x^4 + b * x^3 + c * x^2 + d * x + e
  let g := fun (x : ℝ) ↦ p * x + q
  (∃ (x : ℝ), x = α ∨ x = β → f x = g x ∧ (deriv f) x = (deriv g) x) →
  ∫ x in α..β, |f x - g x| = a * (β - α)^5 / 30 := by
sorry

end NUMINAMATH_CALUDE_area_between_quartic_and_line_l1092_109239


namespace NUMINAMATH_CALUDE_sum_of_squares_l1092_109287

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 35)
  (h2 : x^2 * y + x * y^2 = 210) : 
  x^2 + y^2 = 154 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1092_109287


namespace NUMINAMATH_CALUDE_sports_club_membership_l1092_109265

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 30)
  (h2 : badminton = 16)
  (h3 : tennis = 19)
  (h4 : both = 7) :
  total - (badminton + tennis - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l1092_109265


namespace NUMINAMATH_CALUDE_amanda_works_ten_hours_l1092_109225

/-- Amanda's work scenario -/
def amanda_scenario (hourly_rate : ℝ) (withheld_pay : ℝ) (hours_worked : ℝ) : Prop :=
  hourly_rate = 50 ∧
  withheld_pay = 400 ∧
  withheld_pay = 0.8 * (hourly_rate * hours_worked)

/-- Theorem: Amanda works 10 hours per day -/
theorem amanda_works_ten_hours :
  ∃ (hourly_rate withheld_pay hours_worked : ℝ),
    amanda_scenario hourly_rate withheld_pay hours_worked ∧
    hours_worked = 10 :=
sorry

end NUMINAMATH_CALUDE_amanda_works_ten_hours_l1092_109225


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_number_l1092_109237

theorem modulus_of_specific_complex_number :
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i + 2 / (1 + i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_number_l1092_109237


namespace NUMINAMATH_CALUDE_two_problems_without_conditional_l1092_109200

/-- Represents a mathematical problem that may or may not require conditional statements in its algorithm. -/
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| FunctionValue

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requiresConditional (p : Problem) : Bool :=
  match p with
  | Problem.OppositeNumber => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.FunctionValue => true

/-- The list of all problems given in the question. -/
def allProblems : List Problem :=
  [Problem.OppositeNumber, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.FunctionValue]

/-- Theorem stating that the number of problems not requiring conditional statements is 2. -/
theorem two_problems_without_conditional :
  (allProblems.filter (fun p => ¬requiresConditional p)).length = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_problems_without_conditional_l1092_109200


namespace NUMINAMATH_CALUDE_prime_divides_product_l1092_109254

theorem prime_divides_product (p a b : ℕ) : 
  Prime p → (p ∣ (a * b)) → (p ∣ a) ∨ (p ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_product_l1092_109254


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1092_109259

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be increasing
def IsIncreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- Define the condition a_{n+1} > |a_n|
def StrictlyGreaterThanAbs (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) > |a n|

-- Theorem statement
theorem condition_sufficient_not_necessary :
  (∀ a : Sequence, StrictlyGreaterThanAbs a → IsIncreasing a) ∧
  (∃ a : Sequence, IsIncreasing a ∧ ¬StrictlyGreaterThanAbs a) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1092_109259


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l1092_109246

/-- Theorem: Rectangle Width Decrease
Given a rectangle where:
- The length increases by 20%
- The area increases by 4%
Then the width must decrease by 40/3% (approximately 13.33%) -/
theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.2 * L) (h2 : L' * W' = 1.04 * L * W) :
  W' = (1 - 40 / 300) * W :=
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l1092_109246


namespace NUMINAMATH_CALUDE_volume_common_part_equal_cones_l1092_109261

/-- Given two equal cones with common height and parallel bases, 
    the volume of their common part is 1/4 of the volume of each cone. -/
theorem volume_common_part_equal_cones (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  let V_cone := (1/3) * π * R^2 * h
  let V_common := (1/12) * π * R^2 * h
  V_common = (1/4) * V_cone := by
  sorry

end NUMINAMATH_CALUDE_volume_common_part_equal_cones_l1092_109261


namespace NUMINAMATH_CALUDE_bakers_pastry_problem_l1092_109283

/-- Baker's pastry problem -/
theorem bakers_pastry_problem 
  (total_cakes : ℕ) 
  (total_pastries : ℕ) 
  (sold_pastries : ℕ) 
  (remaining_pastries : ℕ) 
  (h1 : total_cakes = 7)
  (h2 : total_pastries = 148)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : total_pastries = sold_pastries + remaining_pastries) :
  ¬∃! sold_cakes : ℕ, sold_cakes ≤ total_cakes :=
sorry

end NUMINAMATH_CALUDE_bakers_pastry_problem_l1092_109283


namespace NUMINAMATH_CALUDE_triangle_existence_l1092_109241

theorem triangle_existence (x : ℕ) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 12 ∧ c = x^3 + 1 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ (x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l1092_109241


namespace NUMINAMATH_CALUDE_system_equations_and_inequality_l1092_109292

theorem system_equations_and_inequality (a x y : ℝ) : 
  x - y = 1 + 3 * a →
  x + y = -7 - a →
  x ≤ 0 →
  y < 0 →
  (-2 < a ∧ a ≤ 3) →
  (∀ x, 2 * a * x + x > 2 * a + 1 ↔ x < 1) →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_system_equations_and_inequality_l1092_109292


namespace NUMINAMATH_CALUDE_lily_score_l1092_109282

/-- Represents the score for hitting a specific ring -/
structure RingScore where
  inner : ℕ
  middle : ℕ
  outer : ℕ

/-- Represents the number of hits for each ring -/
structure Hits where
  inner : ℕ
  middle : ℕ
  outer : ℕ

/-- Calculates the total score given ring scores and hits -/
def totalScore (rs : RingScore) (h : Hits) : ℕ :=
  rs.inner * h.inner + rs.middle * h.middle + rs.outer * h.outer

theorem lily_score 
  (rs : RingScore) 
  (tom_hits john_hits : Hits) 
  (h1 : tom_hits.inner + tom_hits.middle + tom_hits.outer = 6)
  (h2 : john_hits.inner + john_hits.middle + john_hits.outer = 6)
  (h3 : totalScore rs tom_hits = 46)
  (h4 : totalScore rs john_hits = 34)
  (h5 : totalScore rs { inner := 4, middle := 4, outer := 4 } = 80) :
  totalScore rs { inner := 2, middle := 2, outer := 2 } = 40 := by
  sorry

#check lily_score

end NUMINAMATH_CALUDE_lily_score_l1092_109282


namespace NUMINAMATH_CALUDE_unique_intersection_l1092_109212

/-- The function f(x) = 4 - 2x + x^2 -/
def f (x : ℝ) : ℝ := 4 - 2*x + x^2

/-- The function g(x) = 2 + 2x + x^2 -/
def g (x : ℝ) : ℝ := 2 + 2*x + x^2

theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    f p.1 = g p.1 ∧ 
    p = (1/2, 13/4) := by
  sorry

#check unique_intersection

end NUMINAMATH_CALUDE_unique_intersection_l1092_109212


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1092_109210

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 8 = 24) :
  w / 8 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1092_109210


namespace NUMINAMATH_CALUDE_josephus_69_l1092_109221

/-- The Josephus function that returns the last remaining number given n. -/
def josephus (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the Josephus number for n = 69 is 10. -/
theorem josephus_69 : josephus 69 = 10 := by
  sorry

end NUMINAMATH_CALUDE_josephus_69_l1092_109221


namespace NUMINAMATH_CALUDE_cos_50_cos_20_plus_sin_50_sin_20_l1092_109207

theorem cos_50_cos_20_plus_sin_50_sin_20 :
  Real.cos (50 * π / 180) * Real.cos (20 * π / 180) + Real.sin (50 * π / 180) * Real.sin (20 * π / 180) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cos_50_cos_20_plus_sin_50_sin_20_l1092_109207


namespace NUMINAMATH_CALUDE_card_distribution_implies_square_l1092_109235

theorem card_distribution_implies_square (n : ℕ) (m : ℕ) (h_n : n ≥ 3) 
  (h_m : m = n * (n - 1) / 2) (h_m_even : Even m) 
  (a : Fin n → ℕ) (h_a_range : ∀ i, 1 ≤ a i ∧ a i ≤ m) 
  (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_sums_distinct : ∀ i j k l, (i ≠ j ∧ k ≠ l) → (i, j) ≠ (k, l) → 
    (a i + a j) % m ≠ (a k + a l) % m) :
  ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_implies_square_l1092_109235


namespace NUMINAMATH_CALUDE_remaining_customers_l1092_109295

/-- Given an initial number of customers and a number of customers who left,
    prove that the remaining number of customers is equal to the
    initial number minus the number who left. -/
theorem remaining_customers
  (initial : ℕ) (left : ℕ) (h : left ≤ initial) :
  initial - left = initial - left :=
by sorry

end NUMINAMATH_CALUDE_remaining_customers_l1092_109295


namespace NUMINAMATH_CALUDE_cave_depth_l1092_109228

/-- The depth of the cave given the current depth and remaining distance -/
theorem cave_depth (current_depth remaining_distance : ℕ) 
  (h1 : current_depth = 588)
  (h2 : remaining_distance = 386) : 
  current_depth + remaining_distance = 974 := by
  sorry

end NUMINAMATH_CALUDE_cave_depth_l1092_109228


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt28_l1092_109289

theorem consecutive_integers_around_sqrt28 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 28) → (Real.sqrt 28 < b) → (a + b = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt28_l1092_109289


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l1092_109234

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ 2*x^2 - 6*x - 5
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 19) / 2 ∧ 
              x₂ = (3 - Real.sqrt 19) / 2 ∧ 
              f x₁ = 0 ∧ f x₂ = 0 :=
sorry

theorem linear_equation_solutions :
  let g : ℝ → ℝ := λ x ↦ 3*x*(4-x) - 2*(x-4)
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ 
              x₂ = -2/3 ∧ 
              g x₁ = 0 ∧ g x₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l1092_109234


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l1092_109262

/-- The maximum area of a right-angled triangle with perimeter √2 + 1 is 1/4 -/
theorem max_area_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a^2 + b^2 = c^2 → 
  a + b + c = Real.sqrt 2 + 1 → 
  (1/2 * a * b) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l1092_109262


namespace NUMINAMATH_CALUDE_scaling_transform_line_l1092_109280

/-- Scaling transformation that maps (x, y) to (x', y') -/
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem scaling_transform_line : 
  ∀ (x y : ℝ), x + y = 1 → 
  let (x', y') := scaling_transform x y
  2 * x' + 3 * y' = 6 := by
sorry

end NUMINAMATH_CALUDE_scaling_transform_line_l1092_109280


namespace NUMINAMATH_CALUDE_power_product_equality_l1092_109297

theorem power_product_equality : (3^5 * 4^5) * 6^2 = 8957952 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1092_109297


namespace NUMINAMATH_CALUDE_fantasy_creatures_gala_handshakes_l1092_109206

-- Define the number of gremlins and imps
def num_gremlins : ℕ := 30
def num_imps : ℕ := 20

-- Define the number of imps each imp shakes hands with
def imp_imp_handshakes : ℕ := 5

-- Calculate the number of handshakes between gremlins
def gremlin_gremlin_handshakes : ℕ := num_gremlins * (num_gremlins - 1) / 2

-- Calculate the number of handshakes between imps
def imp_imp_total_handshakes : ℕ := num_imps * imp_imp_handshakes / 2

-- Calculate the number of handshakes between gremlins and imps
def gremlin_imp_handshakes : ℕ := num_gremlins * num_imps

-- Define the total number of handshakes
def total_handshakes : ℕ := gremlin_gremlin_handshakes + imp_imp_total_handshakes + gremlin_imp_handshakes

-- Theorem statement
theorem fantasy_creatures_gala_handshakes : total_handshakes = 1085 := by
  sorry

end NUMINAMATH_CALUDE_fantasy_creatures_gala_handshakes_l1092_109206


namespace NUMINAMATH_CALUDE_game_cost_l1092_109277

theorem game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_cost : ℕ) (game_cost : ℕ) : 
  initial_money = 57 →
  num_toys = 5 →
  toy_cost = 6 →
  initial_money = game_cost + (num_toys * toy_cost) →
  game_cost = 27 := by
sorry

end NUMINAMATH_CALUDE_game_cost_l1092_109277


namespace NUMINAMATH_CALUDE_triangles_not_always_congruent_l1092_109270

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

-- Define the condition for the theorem
def satisfies_condition (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ 
   ((t1.a < t1.b ∧ t1.angle_A = t2.angle_A) ∨ 
    (t1.b < t1.a ∧ t1.angle_B = t2.angle_B)))

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

-- Theorem statement
theorem triangles_not_always_congruent :
  ∃ (t1 t2 : Triangle), satisfies_condition t1 t2 ∧ ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_triangles_not_always_congruent_l1092_109270


namespace NUMINAMATH_CALUDE_tens_digit_of_8_power_1701_l1092_109279

theorem tens_digit_of_8_power_1701 : ∃ n : ℕ, 8^1701 ≡ n [ZMOD 100] ∧ n < 100 ∧ (n / 10 : ℕ) = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_power_1701_l1092_109279


namespace NUMINAMATH_CALUDE_bryan_pushups_l1092_109218

theorem bryan_pushups (planned_sets : ℕ) (pushups_per_set : ℕ) (actual_total : ℕ)
  (h1 : planned_sets = 3)
  (h2 : pushups_per_set = 15)
  (h3 : actual_total = 40) :
  planned_sets * pushups_per_set - actual_total = 5 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l1092_109218


namespace NUMINAMATH_CALUDE_external_equilaterals_centers_theorem_l1092_109245

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (base : Point)
  (apex : Point)

/-- Returns the center of an equilateral triangle -/
def centerOfEquilateral (t : EquilateralTriangle) : Point := sorry

/-- Returns the centroid of a triangle -/
def centroid (t : Triangle) : Point := sorry

/-- Constructs equilateral triangles on the sides of a given triangle -/
def constructExternalEquilaterals (t : Triangle) : 
  (EquilateralTriangle × EquilateralTriangle × EquilateralTriangle) := sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (A B C : Point) : Prop := sorry

theorem external_equilaterals_centers_theorem (t : Triangle) :
  let (eqAB, eqBC, eqCA) := constructExternalEquilaterals t
  let centerAB := centerOfEquilateral eqAB
  let centerBC := centerOfEquilateral eqBC
  let centerCA := centerOfEquilateral eqCA
  isEquilateral centerAB centerBC centerCA ∧
  centroid (Triangle.mk centerAB centerBC centerCA) = centroid t := by sorry

end NUMINAMATH_CALUDE_external_equilaterals_centers_theorem_l1092_109245


namespace NUMINAMATH_CALUDE_boat_speed_solution_l1092_109296

def boat_problem (downstream_time upstream_time stream_speed : ℝ) : Prop :=
  downstream_time > 0 ∧ 
  upstream_time > 0 ∧ 
  stream_speed > 0 ∧
  ∃ (distance boat_speed : ℝ),
    distance > 0 ∧
    boat_speed > stream_speed ∧
    distance = (boat_speed + stream_speed) * downstream_time ∧
    distance = (boat_speed - stream_speed) * upstream_time

theorem boat_speed_solution :
  boat_problem 1 1.5 3 →
  ∃ (distance boat_speed : ℝ),
    boat_speed = 15 ∧
    distance > 0 ∧
    boat_speed > 3 ∧
    distance = (boat_speed + 3) * 1 ∧
    distance = (boat_speed - 3) * 1.5 :=
by
  sorry

#check boat_speed_solution

end NUMINAMATH_CALUDE_boat_speed_solution_l1092_109296


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1092_109249

theorem cubic_equation_solution : 
  ∃ (a : ℝ), (a^3 - 4*a^2 + 7*a - 28 = 0) ∧ 
  (∀ x : ℝ, x^3 - 4*x^2 + 7*x - 28 = 0 → x ≤ a) →
  2*a + 0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1092_109249


namespace NUMINAMATH_CALUDE_munchausen_polygon_exists_l1092_109202

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define a line as a set of points satisfying a linear equation
def Line : Type := Set (ℝ × ℝ)

-- Define what it means for a point to be inside a polygon
def inside (p : Point) (poly : Polygon) : Prop := sorry

-- Define what it means for a line to divide a polygon
def divides (l : Line) (poly : Polygon) : Prop := sorry

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Count the number of polygons resulting from dividing a polygon by a line
def count_divisions (l : Line) (poly : Polygon) : ℕ := sorry

-- The main theorem
theorem munchausen_polygon_exists :
  ∃ (P : Polygon) (O : Point),
    inside O P ∧
    ∀ (L : Line), passes_through L O →
      count_divisions L P = 3 := by sorry

end NUMINAMATH_CALUDE_munchausen_polygon_exists_l1092_109202


namespace NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l1092_109291

/-- Represents the cost of a pizza with toppings -/
def pizza_cost (base_cost : ℚ) (num_slices : ℕ) (first_topping_cost : ℚ) 
  (next_two_toppings_cost : ℚ) (rest_toppings_cost : ℚ) (num_toppings : ℕ) : ℚ :=
  let total_cost := base_cost + first_topping_cost + 
    (if num_toppings > 1 then min (num_toppings - 1) 2 * next_two_toppings_cost else 0) +
    (if num_toppings > 3 then (num_toppings - 3) * rest_toppings_cost else 0)
  total_cost / num_slices

theorem jimmy_pizza_cost_per_slice :
  pizza_cost 10 8 2 1 0.5 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l1092_109291


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1092_109220

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1092_109220


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_digit_swap_l1092_109263

theorem arithmetic_geometric_mean_digit_swap : ∃ (x₁ x₂ : ℕ), 
  x₁ ≠ x₂ ∧
  (let A := (x₁ + x₂) / 2
   let G := Int.sqrt (x₁ * x₂)
   10 ≤ A ∧ A < 100 ∧
   10 ≤ G ∧ G < 100 ∧
   ((A / 10 = G % 10 ∧ A % 10 = G / 10) ∨
    (A % 10 = G / 10 ∧ A / 10 = G % 10)) ∧
   x₁ = 98 ∧
   x₂ = 32) :=
by
  sorry

#eval (98 + 32) / 2  -- Expected output: 65
#eval Int.sqrt (98 * 32)  -- Expected output: 56

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_digit_swap_l1092_109263


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1092_109232

theorem fraction_subtraction : (4 + 6 + 8) / (3 + 5 + 7) - (3 + 5 + 7) / (4 + 6 + 8) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1092_109232


namespace NUMINAMATH_CALUDE_gala_trees_count_l1092_109243

theorem gala_trees_count (total : ℕ) (fuji gala honeycrisp : ℕ) : 
  total = fuji + gala + honeycrisp →
  fuji = (2 * total) / 3 →
  honeycrisp = total / 6 →
  fuji + (125 * fuji) / 1000 + (75 * fuji) / 1000 = 315 →
  gala = 66 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_count_l1092_109243


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1092_109204

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 15 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1092_109204


namespace NUMINAMATH_CALUDE_vasya_lowest_position_l1092_109230

/-- Represents a cyclist in the race -/
structure Cyclist :=
  (id : Nat)

/-- Represents a stage in the race -/
structure Stage :=
  (number : Nat)

/-- Represents the time a cyclist takes to complete a stage -/
structure StageTime :=
  (cyclist : Cyclist)
  (stage : Stage)
  (time : ℝ)

/-- Represents the total time a cyclist takes to complete all stages -/
structure TotalTime :=
  (cyclist : Cyclist)
  (time : ℝ)

/-- The number of cyclists in the race -/
def numCyclists : Nat := 500

/-- The number of stages in the race -/
def numStages : Nat := 15

/-- Vasya's position in each stage -/
def vasyaStagePosition : Nat := 7

/-- Function to get a cyclist's position in a stage -/
def stagePosition (c : Cyclist) (s : Stage) : Nat := sorry

/-- Function to get a cyclist's overall position -/
def overallPosition (c : Cyclist) : Nat := sorry

/-- Vasya's cyclist object -/
def vasya : Cyclist := ⟨0⟩  -- Assuming Vasya's ID is 0

/-- The main theorem -/
theorem vasya_lowest_position :
  (∀ s : Stage, stagePosition vasya s = vasyaStagePosition) →
  (∀ c1 c2 : Cyclist, ∀ s : Stage, c1 ≠ c2 → stagePosition c1 s ≠ stagePosition c2 s) →
  (∀ c1 c2 : Cyclist, c1 ≠ c2 → overallPosition c1 ≠ overallPosition c2) →
  overallPosition vasya ≤ 91 := sorry

end NUMINAMATH_CALUDE_vasya_lowest_position_l1092_109230


namespace NUMINAMATH_CALUDE_harry_worked_35_hours_l1092_109215

/-- Represents the pay structure and hours worked for Harry and James -/
structure PayStructure where
  x : ℝ  -- Base hourly rate
  james_overtime_rate : ℝ  -- James' overtime rate as a multiple of x
  harry_hours : ℕ  -- Total hours Harry worked
  harry_overtime : ℕ  -- Hours Harry worked beyond 21
  james_hours : ℕ  -- Total hours James worked
  james_overtime : ℕ  -- Hours James worked beyond 40

/-- Calculates Harry's total pay -/
def harry_pay (p : PayStructure) : ℝ :=
  21 * p.x + p.harry_overtime * (1.5 * p.x)

/-- Calculates James' total pay -/
def james_pay (p : PayStructure) : ℝ :=
  40 * p.x + p.james_overtime * (p.james_overtime_rate * p.x)

/-- Theorem stating that Harry worked 35 hours given the problem conditions -/
theorem harry_worked_35_hours :
  ∀ (p : PayStructure),
    p.james_hours = 41 →
    p.james_overtime = 1 →
    p.harry_hours = p.harry_overtime + 21 →
    harry_pay p = james_pay p →
    p.harry_hours = 35 := by
  sorry


end NUMINAMATH_CALUDE_harry_worked_35_hours_l1092_109215


namespace NUMINAMATH_CALUDE_derivative_at_zero_l1092_109231

/-- Given a function f(x) = e^x + sin x - cos x, prove that its derivative at x = 0 is 2 -/
theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x + Real.sin x - Real.cos x) :
  deriv f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l1092_109231


namespace NUMINAMATH_CALUDE_cube_divisibility_l1092_109236

theorem cube_divisibility (k : ℕ) (n : ℕ) : 
  (k ≥ 30) → 
  (∀ m : ℕ, m ≥ 30 → m < k → ¬(∃ p : ℕ, m^3 = p * n)) →
  (∃ q : ℕ, k^3 = q * n) →
  n = 27000 := by
sorry

end NUMINAMATH_CALUDE_cube_divisibility_l1092_109236


namespace NUMINAMATH_CALUDE_m_equals_three_l1092_109214

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) - m * (3 + 6 * Complex.I)

/-- Theorem: If z(m) is pure imaginary, then m = 3 -/
theorem m_equals_three (h : isPureImaginary (z m)) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_equals_three_l1092_109214


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l1092_109275

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l1092_109275


namespace NUMINAMATH_CALUDE_arithmetic_mean_change_l1092_109250

theorem arithmetic_mean_change (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 10 →
  (b + c + d) / 3 = 11 →
  (a + c + d) / 3 = 12 →
  (a + b + d) / 3 = 13 →
  (a + b + c) / 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_change_l1092_109250


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1092_109273

theorem fraction_evaluation : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1092_109273


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_and_cashback_l1092_109293

/-- The total cost of an iPhone 12 and an iWatch after discounts and cashback -/
theorem total_cost_after_discounts_and_cashback :
  let iphone_price : ℚ := 800
  let iwatch_price : ℚ := 300
  let iphone_discount : ℚ := 15 / 100
  let iwatch_discount : ℚ := 10 / 100
  let cashback_rate : ℚ := 2 / 100
  let iphone_discounted := iphone_price * (1 - iphone_discount)
  let iwatch_discounted := iwatch_price * (1 - iwatch_discount)
  let total_before_cashback := iphone_discounted + iwatch_discounted
  let cashback_amount := total_before_cashback * cashback_rate
  let final_cost := total_before_cashback - cashback_amount
  final_cost = 931 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_and_cashback_l1092_109293


namespace NUMINAMATH_CALUDE_gasoline_reduction_l1092_109256

theorem gasoline_reduction (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.2 * P
  let new_total_cost := 1.14 * (P * Q)
  let new_quantity := new_total_cost / new_price
  (Q - new_quantity) / Q = 0.05 := by
sorry

end NUMINAMATH_CALUDE_gasoline_reduction_l1092_109256


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1092_109229

/-- Given a line L1 with equation 3x + 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -1/2x - 2 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  (3 * x + 6 * y = 9) →  -- Equation of L1
  (y = -1/2 * x - 2) →   -- Equation of L2
  (∃ m b : ℝ, 3 * x + 6 * y = 9 ↔ y = m * x + b) →  -- L1 can be written in slope-intercept form
  ((-1/2) = m) →  -- Slopes are equal
  ((-1/2) * 2 - 2 = -3) →  -- L2 passes through (2, -3)
  (y = -1/2 * x - 2) ∧ (3 * 2 + 6 * (-3) = 9)  -- L2 is parallel to L1 and passes through (2, -3)
:= by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1092_109229


namespace NUMINAMATH_CALUDE_employee_income_change_l1092_109216

theorem employee_income_change 
  (payment_increase : Real) 
  (time_decrease : Real) : 
  payment_increase = 0.3333 → 
  time_decrease = 0.3333 → 
  let new_payment := 1 + payment_increase
  let new_time := 1 - time_decrease
  let income_change := new_payment * new_time - 1
  income_change = -0.1111 := by sorry

end NUMINAMATH_CALUDE_employee_income_change_l1092_109216


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1092_109223

/-- Given a hyperbola with center (1, 0), one focus at (1 + √41, 0), and one vertex at (-2, 0),
    prove that h + k + a + b = 1 + 0 + 3 + 4√2, where (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1
    is the equation of the hyperbola. -/
theorem hyperbola_sum (h k a b : ℝ) : 
  (1 : ℝ) = h ∧ (0 : ℝ) = k ∧  -- center at (1, 0)
  (1 + Real.sqrt 41 : ℝ) = h + Real.sqrt (c^2) ∧ -- focus at (1 + √41, 0)
  (-2 : ℝ) = h - a ∧ -- vertex at (-2, 0)
  (∀ x y : ℝ, (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) → -- equation of hyperbola
  h + k + a + b = 1 + 0 + 3 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1092_109223


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1092_109268

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x ∈ U | x^2 - 4*x + 3 = 0}

theorem complement_of_M_in_U : (U \ M) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1092_109268


namespace NUMINAMATH_CALUDE_negative_difference_equals_reversed_difference_l1092_109278

theorem negative_difference_equals_reversed_difference (a b : ℝ) : 
  -(a - b) = b - a := by sorry

end NUMINAMATH_CALUDE_negative_difference_equals_reversed_difference_l1092_109278


namespace NUMINAMATH_CALUDE_max_salary_320000_l1092_109269

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  num_players : ℕ
  min_salary : ℕ
  total_salary_cap : ℕ

/-- Calculates the maximum possible salary for a single player in a baseball team -/
def max_single_player_salary (team : BaseballTeam) : ℕ :=
  team.total_salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player in a specific baseball team -/
theorem max_salary_320000 :
  let team : BaseballTeam := ⟨25, 20000, 800000⟩
  max_single_player_salary team = 320000 := by
  sorry

#eval max_single_player_salary ⟨25, 20000, 800000⟩

end NUMINAMATH_CALUDE_max_salary_320000_l1092_109269


namespace NUMINAMATH_CALUDE_revenue_decrease_l1092_109276

/-- Proves that a 43.529411764705884% decrease to $48.0 billion results in an original revenue of $85.0 billion -/
theorem revenue_decrease (current_revenue : ℝ) (decrease_percentage : ℝ) (original_revenue : ℝ) :
  current_revenue = 48.0 ∧
  decrease_percentage = 43.529411764705884 ∧
  current_revenue = original_revenue * (1 - decrease_percentage / 100) →
  original_revenue = 85.0 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l1092_109276


namespace NUMINAMATH_CALUDE_power_sum_equality_l1092_109298

theorem power_sum_equality : (2 : ℕ)^(3^2) + (-1 : ℤ)^(2^3) = 513 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1092_109298


namespace NUMINAMATH_CALUDE_ellipse_parabola_tangent_lines_l1092_109267

/-- Given an ellipse and a parabola with specific properties, prove the equation of the parabola and its tangent lines. -/
theorem ellipse_parabola_tangent_lines :
  ∀ (b : ℝ) (p : ℝ),
  0 < b → b < 2 → p > 0 →
  (∀ (x y : ℝ), x^2 / 4 + y^2 / b^2 = 1 → (x^2 + y^2) / 4 = 3 / 4) →
  (∀ (x y : ℝ), x^2 = 2 * p * y) →
  (∃ (x₀ y₀ : ℝ), x₀^2 / 4 + y₀^2 / b^2 = 1 ∧ x₀^2 = 2 * p * y₀ ∧ (x₀ = 0 ∨ y₀ = 1 ∨ y₀ = -1)) →
  (∀ (x y : ℝ), x^2 = 4 * y) ∧
  (∀ (x y : ℝ), (y = 0 ∨ x + y + 1 = 0) → 
    (x + 1)^2 = 4 * y ∧ (x + 1 = -1 → y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_tangent_lines_l1092_109267


namespace NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l1092_109240

open Real

theorem log_sin_in_terms_of_m_n (α m n : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : log (1 + cos α) = m)
  (h4 : log (1 / (1 - cos α)) = n) :
  log (sin α) = (1 / 2) * (m - 1 / n) := by
  sorry

end NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l1092_109240


namespace NUMINAMATH_CALUDE_area_relationship_l1092_109205

/-- A circle circumscribed about a right triangle with sides 12, 35, and 37 -/
structure CircumscribedTriangle where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The area of the largest non-triangular region -/
  C : ℝ
  /-- The sum of the areas of the two smaller non-triangular regions -/
  A_plus_B : ℝ
  /-- The radius is half of the hypotenuse -/
  radius_eq : radius = 37 / 2
  /-- The largest non-triangular region is a semicircle -/
  C_eq : C = π * radius^2 / 2
  /-- The sum of all regions equals the circle's area -/
  area_eq : A_plus_B + 210 + C = π * radius^2

/-- The relationship between the areas of the non-triangular regions -/
theorem area_relationship (t : CircumscribedTriangle) : t.A_plus_B + 210 = t.C := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l1092_109205


namespace NUMINAMATH_CALUDE_commercial_break_duration_l1092_109219

theorem commercial_break_duration :
  let five_minute_commercials : ℕ := 3
  let two_minute_commercials : ℕ := 11
  let five_minute_duration : ℕ := 5
  let two_minute_duration : ℕ := 2
  (five_minute_commercials * five_minute_duration + two_minute_commercials * two_minute_duration : ℕ) = 37 :=
by sorry

end NUMINAMATH_CALUDE_commercial_break_duration_l1092_109219


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1092_109201

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁, a₃, a₉ form a geometric sequence,
    prove that (a₁ + a₃ + a₉) / (a₂ + a₄ + a₁₀) = 13/16. -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) 
  (d : ℚ) 
  (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : (a 3) ^ 2 = a 1 * a 9) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1092_109201
