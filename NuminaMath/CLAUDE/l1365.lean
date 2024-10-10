import Mathlib

namespace pet_store_feet_count_l1365_136562

/-- A pet store sells dogs and parakeets. -/
structure PetStore :=
  (dogs : ℕ)
  (parakeets : ℕ)

/-- Calculate the total number of feet in the pet store. -/
def total_feet (store : PetStore) : ℕ :=
  4 * store.dogs + 2 * store.parakeets

/-- Theorem: Given 15 total heads and 9 dogs, the total number of feet is 48. -/
theorem pet_store_feet_count :
  ∀ (store : PetStore),
  store.dogs + store.parakeets = 15 →
  store.dogs = 9 →
  total_feet store = 48 :=
by
  sorry


end pet_store_feet_count_l1365_136562


namespace collins_earnings_per_can_l1365_136502

/-- The amount of money Collin earns per aluminum can -/
def earnings_per_can (cans_home : ℕ) (cans_grandparents_multiplier : ℕ) (cans_neighbor : ℕ) (cans_dad_office : ℕ) (savings_amount : ℚ) : ℚ :=
  let total_cans := cans_home + cans_home * cans_grandparents_multiplier + cans_neighbor + cans_dad_office
  let total_earnings := 2 * savings_amount
  total_earnings / total_cans

/-- Theorem stating that Collin earns $0.25 per aluminum can -/
theorem collins_earnings_per_can :
  earnings_per_can 12 3 46 250 43 = 1/4 := by
  sorry

end collins_earnings_per_can_l1365_136502


namespace circle_radius_l1365_136548

/-- The radius of a circle satisfying the given condition -/
theorem circle_radius : ∃ (r : ℝ), r > 0 ∧ 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) ∧ r = 3 := by
  sorry

end circle_radius_l1365_136548


namespace white_area_calculation_l1365_136594

theorem white_area_calculation (total_area grey_area1 grey_area2 dark_grey_area : ℝ) 
  (h1 : total_area = 32)
  (h2 : grey_area1 = 16)
  (h3 : grey_area2 = 15)
  (h4 : dark_grey_area = 4) :
  total_area = grey_area1 + grey_area2 + (total_area - grey_area1 - grey_area2 + dark_grey_area) - dark_grey_area :=
by sorry

end white_area_calculation_l1365_136594


namespace sequence_length_l1365_136510

theorem sequence_length (n : ℕ) (b : ℕ → ℝ) : 
  (n > 0) →
  (b 0 = 45) →
  (b 1 = 80) →
  (b n = 0) →
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 901 := by
sorry

end sequence_length_l1365_136510


namespace campsite_return_strategy_l1365_136536

structure CampsiteScenario where
  num_students : ℕ
  time_remaining : ℕ
  num_roads : ℕ
  time_per_road : ℕ
  num_liars : ℕ

def has_reliable_strategy (scenario : CampsiteScenario) : Prop :=
  ∃ (strategy : CampsiteScenario → Bool),
    strategy scenario = true

theorem campsite_return_strategy 
  (scenario1 : CampsiteScenario)
  (scenario2 : CampsiteScenario)
  (h1 : scenario1.num_students = 8)
  (h2 : scenario1.time_remaining = 60)
  (h3 : scenario2.num_students = 4)
  (h4 : scenario2.time_remaining = 100)
  (h5 : scenario1.num_roads = 4)
  (h6 : scenario2.num_roads = 4)
  (h7 : scenario1.time_per_road = 20)
  (h8 : scenario2.time_per_road = 20)
  (h9 : scenario1.num_liars = 2)
  (h10 : scenario2.num_liars = 2) :
  has_reliable_strategy scenario1 ∧ has_reliable_strategy scenario2 :=
sorry

end campsite_return_strategy_l1365_136536


namespace birds_flew_up_l1365_136530

theorem birds_flew_up (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 231)
  (h2 : final_birds = 312)
  : final_birds - initial_birds = 81 := by
  sorry

end birds_flew_up_l1365_136530


namespace comic_book_ratio_l1365_136569

/-- Given the initial number of comic books, the number bought, and the final number,
    prove that the ratio of comic books sold to initial comic books is 1/2. -/
theorem comic_book_ratio 
  (initial : ℕ) (bought : ℕ) (final : ℕ) 
  (h1 : initial = 14) 
  (h2 : bought = 6) 
  (h3 : final = 13) : 
  (initial - (final - bought)) / initial = 1 / 2 := by
sorry

end comic_book_ratio_l1365_136569


namespace pauls_crayons_l1365_136506

theorem pauls_crayons (erasers_birthday : ℕ) (crayons_left : ℕ) (crayons_difference : ℕ) :
  erasers_birthday = 38 →
  crayons_left = 391 →
  crayons_difference = 353 →
  crayons_left = erasers_birthday + crayons_difference →
  crayons_left = 391 :=
by sorry

end pauls_crayons_l1365_136506


namespace intersection_set_characterization_l1365_136554

/-- The set of positive real numbers m for which the graphs of y = (mx-1)^2 and y = √x + m 
    have exactly one intersection point on the interval [0,1] -/
def IntersectionSet : Set ℝ :=
  {m : ℝ | m > 0 ∧ ∃! x : ℝ, x ∈ [0, 1] ∧ (m * x - 1)^2 = Real.sqrt x + m}

/-- The theorem stating that the IntersectionSet is equal to (0,1] ∪ [3, +∞) -/
theorem intersection_set_characterization :
  IntersectionSet = Set.Ioo 0 1 ∪ Set.Ici 3 := by
  sorry

end intersection_set_characterization_l1365_136554


namespace function_not_in_first_quadrant_l1365_136512

/-- The function f(x) = (1/5)^(x+1) + m does not pass through the first quadrant if and only if m ≤ -1/5 -/
theorem function_not_in_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -1/5 := by sorry

end function_not_in_first_quadrant_l1365_136512


namespace balls_in_boxes_l1365_136509

def to_base_7_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem balls_in_boxes (n : ℕ) (h : n = 3010) : 
  (to_base_7_digits n).sum = 16 := by
  sorry

end balls_in_boxes_l1365_136509


namespace square_difference_theorem_l1365_136505

theorem square_difference_theorem (a b A : ℝ) : 
  (5*a + 3*b)^2 = (5*a - 3*b)^2 + A → A = 60*a*b := by
  sorry

end square_difference_theorem_l1365_136505


namespace computer_sales_ratio_l1365_136531

theorem computer_sales_ratio (total : ℕ) (laptops : ℕ) (desktops : ℕ) (netbooks : ℕ) :
  total = 72 →
  laptops = total / 2 →
  desktops = 12 →
  netbooks = total - laptops - desktops →
  (netbooks : ℚ) / total = 1 / 3 := by
  sorry

end computer_sales_ratio_l1365_136531


namespace thumbtack_count_l1365_136552

theorem thumbtack_count (num_cans : ℕ) (boards_tested : ℕ) (tacks_per_board : ℕ) (remaining_tacks : ℕ) : 
  num_cans = 3 →
  boards_tested = 120 →
  tacks_per_board = 1 →
  remaining_tacks = 30 →
  (num_cans * (boards_tested * tacks_per_board + remaining_tacks) = 450) :=
by sorry

end thumbtack_count_l1365_136552


namespace fraction_product_equality_l1365_136555

theorem fraction_product_equality : (1 / 3 : ℚ)^3 * (1 / 7 : ℚ)^2 = 1 / 1323 := by
  sorry

end fraction_product_equality_l1365_136555


namespace ratio_equality_l1365_136599

theorem ratio_equality : 
  ∀ (a b c d x : ℚ), 
    a = 3 / 5 → 
    b = 6 / 7 → 
    c = 2 / 3 → 
    d = 7 / 15 → 
    (a / b = d / c) → 
    x = d := by
  sorry

end ratio_equality_l1365_136599


namespace quadratic_inequality_solution_general_quadratic_inequality_solution_l1365_136591

def quadratic_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 3 * x + 2 > 0}

def solution_set (b : ℝ) : Set ℝ :=
  {x | x < 1 ∨ x > b}

theorem quadratic_inequality_solution (a b : ℝ) :
  quadratic_inequality a b = solution_set b → a = 1 ∧ b = 2 :=
sorry

def general_quadratic_inequality (a b c : ℝ) : Set ℝ :=
  {x | x^2 - b * (a + c) * x + 4 * c > 0}

theorem general_quadratic_inequality_solution (a b c : ℝ) :
  a = 1 ∧ b = 2 →
  (c > 1 → general_quadratic_inequality a b c = {x | x < 2 ∨ x > 2 * c}) ∧
  (c = 1 → general_quadratic_inequality a b c = {x | x ≠ 2}) ∧
  (c < 1 → general_quadratic_inequality a b c = {x | x > 2 ∨ x < 2 * c}) :=
sorry

end quadratic_inequality_solution_general_quadratic_inequality_solution_l1365_136591


namespace function_property_l1365_136514

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = -f x

def is_monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_periodic_neg_one f) 
  (h3 : is_monotone_increasing_on f (-1) 0) :
  f 2 > f (Real.sqrt 2) ∧ f (Real.sqrt 2) > f 3 := by
  sorry

end function_property_l1365_136514


namespace unique_divisible_by_six_l1365_136534

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem unique_divisible_by_six : 
  ∀ B : ℕ, B < 10 → 
    (is_divisible_by (7520 + B) 6 ↔ B = 4) :=
by sorry

end unique_divisible_by_six_l1365_136534


namespace line_point_a_value_l1365_136528

/-- Given a line y = 0.75x + 1 and points (4, b), (a, 5), and (a, b + 1) on this line, prove that a = 16/3 -/
theorem line_point_a_value (b : ℝ) :
  (∃ (a : ℝ), (4 : ℝ) * (3/4) + 1 = b ∧ 
              a * (3/4) + 1 = 5 ∧ 
              a * (3/4) + 1 = b + 1) →
  ∃ (a : ℝ), a = 16/3 := by
  sorry

end line_point_a_value_l1365_136528


namespace maria_carrots_l1365_136567

def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  (initial - thrown_out) + new_picked

theorem maria_carrots (initial thrown_out new_picked : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out new_picked = initial - thrown_out + new_picked :=
by
  sorry

end maria_carrots_l1365_136567


namespace ab_length_l1365_136504

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom collinear : ∃ (t : ℝ), B = A + t • (D - A) ∧ C = A + t • (D - A)
axiom ab_eq_cd : dist A B = dist C D
axiom bc_length : dist B C = 15
axiom e_not_on_line : ¬∃ (t : ℝ), E = A + t • (D - A)
axiom be_eq_ce : dist B E = dist C E ∧ dist B E = 13

-- Define the perimeter function
def perimeter (X Y Z : ℝ × ℝ) : ℝ := dist X Y + dist Y Z + dist Z X

-- State the theorem
theorem ab_length :
  perimeter A E D = 1.5 * perimeter B E C →
  dist A B = 6.04 := by sorry

end ab_length_l1365_136504


namespace rulers_in_drawer_l1365_136515

/-- The number of rulers originally in the drawer -/
def original_rulers : ℕ := 71 - 25

theorem rulers_in_drawer : original_rulers = 46 := by
  sorry

end rulers_in_drawer_l1365_136515


namespace decimal_sum_difference_l1365_136566

theorem decimal_sum_difference : (0.5 : ℚ) - 0.03 + 0.007 = 0.477 := by sorry

end decimal_sum_difference_l1365_136566


namespace quadratic_roots_condition_l1365_136507

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
   x1^2 - x1 + 2*m - 2 = 0 ∧ 
   x2^2 - x2 + 2*m - 2 = 0) 
  ↔ m ≤ 9/8 :=
by sorry

end quadratic_roots_condition_l1365_136507


namespace correct_algebraic_notation_l1365_136529

/-- Predicate to check if an expression follows algebraic notation rules -/
def follows_algebraic_notation (expr : String) : Prop :=
  match expr with
  | "7/3 * x^2" => True
  | "a * 1/4" => False
  | "-2 1/6 * p" => False
  | "2y / z" => False
  | _ => False

/-- Theorem stating that 7/3 * x^2 follows algebraic notation rules -/
theorem correct_algebraic_notation :
  follows_algebraic_notation "7/3 * x^2" ∧
  ¬follows_algebraic_notation "a * 1/4" ∧
  ¬follows_algebraic_notation "-2 1/6 * p" ∧
  ¬follows_algebraic_notation "2y / z" :=
sorry

end correct_algebraic_notation_l1365_136529


namespace sufficient_not_necessary_condition_l1365_136540

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x^2020 > 0) ∧
  (∃ x, x^2020 > 0 ∧ ¬(x > 0)) := by
  sorry

end sufficient_not_necessary_condition_l1365_136540


namespace second_player_wins_123_l1365_136568

/-- A game where players color points on a circle. -/
structure ColorGame where
  num_points : ℕ
  first_player : Bool
  
/-- The result of the game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- Determine the winner of the color game. -/
def winner (game : ColorGame) : GameResult :=
  if game.num_points % 2 = 1 then GameResult.SecondPlayerWins
  else GameResult.FirstPlayerWins

/-- The main theorem stating that the second player wins in a game with 123 points. -/
theorem second_player_wins_123 :
  ∀ (game : ColorGame), game.num_points = 123 → winner game = GameResult.SecondPlayerWins :=
  sorry

end second_player_wins_123_l1365_136568


namespace complex_product_range_l1365_136551

theorem complex_product_range (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ < 1)
  (h₂ : Complex.abs z₂ < 1)
  (h₃ : ∃ (r : ℝ), z₁ + z₂ = r)
  (h₄ : z₁ + z₂ + z₁ * z₂ = 0) :
  ∃ (x : ℝ), z₁ * z₂ = x ∧ -1/2 < x ∧ x < 1 := by
  sorry

end complex_product_range_l1365_136551


namespace village_population_l1365_136557

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.2) = 3553 → P = 4678 := by
  sorry

end village_population_l1365_136557


namespace school_relationship_l1365_136500

/-- In a school with teachers and students, prove the relationship between the number of teachers,
    students, students per teacher, and common teachers between any two students. -/
theorem school_relationship (m n k l : ℕ) : 
  (∀ (teacher : Fin m), ∃! (students : Finset (Fin n)), students.card = k) →
  (∀ (student1 student2 : Fin n), student1 ≠ student2 → 
    ∃! (common_teachers : Finset (Fin m)), common_teachers.card = l) →
  m * k * (k - 1) = n * (n - 1) * l := by
  sorry

end school_relationship_l1365_136500


namespace first_consecutive_shot_probability_value_l1365_136577

/-- The probability of making a shot -/
def shot_probability : ℚ := 2/3

/-- The number of attempts before the first consecutive shot -/
def attempts : ℕ := 6

/-- The probability of making the first consecutive shot on the 7th attempt -/
def first_consecutive_shot_probability : ℚ :=
  (1 - shot_probability)^attempts * shot_probability^2

theorem first_consecutive_shot_probability_value :
  first_consecutive_shot_probability = 8/729 := by
  sorry

end first_consecutive_shot_probability_value_l1365_136577


namespace rick_ironing_theorem_l1365_136595

/-- The number of dress shirts Rick can iron in one hour -/
def shirts_per_hour : ℕ := 4

/-- The number of dress pants Rick can iron in one hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick has ironed -/
def total_pieces : ℕ := shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing_theorem : total_pieces = 27 := by
  sorry

end rick_ironing_theorem_l1365_136595


namespace intersection_of_M_and_N_l1365_136547

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | |x| < 2}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -1 ≤ x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l1365_136547


namespace probability_is_one_fourth_l1365_136571

/-- Represents a right triangle XYZ with given side lengths -/
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  angle_x_is_right : xy > 0 ∧ xz > 0

/-- Calculates the probability of a randomly chosen point P inside the right triangle XYZ
    forming a triangle PYZ with an area less than one-third of the area of XYZ -/
def probability_small_area (t : RightTriangle) : ℝ :=
  sorry

/-- The main theorem stating that for a right triangle with sides 6 and 8,
    the probability of forming a smaller triangle with area less than one-third
    of the original triangle's area is 1/4 -/
theorem probability_is_one_fourth :
  let t : RightTriangle := ⟨6, 8, by norm_num⟩
  probability_small_area t = 1/4 :=
sorry

end probability_is_one_fourth_l1365_136571


namespace line_passes_through_point_l1365_136503

theorem line_passes_through_point (a : ℝ) : (a + 2) * 1 + a * (-1) - 2 = 0 := by
  sorry

end line_passes_through_point_l1365_136503


namespace petrol_expense_l1365_136544

def monthly_expenses (rent milk groceries education misc petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + misc + petrol

def savings_percentage : ℚ := 1 / 10

theorem petrol_expense (rent milk groceries education misc savings : ℕ) 
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : misc = 3940)
  (h6 : savings = 2160)
  (h7 : ∃ (salary petrol : ℕ), savings_percentage * salary = savings ∧ 
        monthly_expenses rent milk groceries education misc petrol = salary - savings) :
  ∃ (petrol : ℕ), petrol = 2000 := by
sorry

end petrol_expense_l1365_136544


namespace unique_non_range_value_l1365_136598

/-- The function f defined by the given properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that 58 is the unique number not in the range of f -/
theorem unique_non_range_value
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h_19 : f a b c d 19 = 19)
  (h_97 : f a b c d 97 = 97)
  (h_inverse : ∀ x ≠ -d/c, f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 58 := by
  sorry

end unique_non_range_value_l1365_136598


namespace sqrt_special_sum_l1365_136532

def digits_to_num (d : ℕ) (n : ℕ) : ℕ := (10^n - 1) / (10 - 1) * d

theorem sqrt_special_sum (n : ℕ) (h : n > 0) :
  Real.sqrt (digits_to_num 4 (2*n) + digits_to_num 1 (n+1) - digits_to_num 6 n) = 
  digits_to_num 6 (n-1) + 7 :=
sorry

end sqrt_special_sum_l1365_136532


namespace sum_other_vertices_y_equals_14_l1365_136519

structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ

def Rectangle.sumOtherVerticesY (r : Rectangle) : ℝ :=
  r.vertex1.2 + r.vertex2.2

theorem sum_other_vertices_y_equals_14 (r : Rectangle) 
  (h1 : r.vertex1 = (2, 20))
  (h2 : r.vertex2 = (10, -6)) :
  r.sumOtherVerticesY = 14 := by
  sorry

#check sum_other_vertices_y_equals_14

end sum_other_vertices_y_equals_14_l1365_136519


namespace min_addition_to_prime_l1365_136535

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  n = 10 * a + b ∧ 2 * a * b = n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_addition_to_prime :
  ∃ n : ℕ, is_valid_number n ∧
  (∀ k : ℕ, k < 1 → ¬(is_prime (n + k))) ∧
  is_prime (n + 1) :=
sorry

end min_addition_to_prime_l1365_136535


namespace time_to_save_downpayment_l1365_136558

def salary : ℝ := 150000
def savings_rate : ℝ := 0.10
def house_cost : ℝ := 450000
def downpayment_rate : ℝ := 0.20

def yearly_savings : ℝ := salary * savings_rate
def required_downpayment : ℝ := house_cost * downpayment_rate

theorem time_to_save_downpayment :
  required_downpayment / yearly_savings = 6 := by sorry

end time_to_save_downpayment_l1365_136558


namespace chord_bisector_line_equation_l1365_136584

/-- Given an ellipse and a point inside it, this theorem proves the equation of the line
    on which the chord bisected by the point lies. -/
theorem chord_bisector_line_equation (x y : ℝ) :
  (x^2 / 16 + y^2 / 4 = 1) →  -- Ellipse equation
  (3^2 / 16 + 1^2 / 4 < 1) →  -- Point P(3,1) is inside the ellipse
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x^2 / 16 + y^2 / 4 = 1) ∧ 
    ((x + 3) / 2 = 3 ∧ (y + 1) / 2 = 1) → 
    y = m * x + b ∧ 
    3 * x + 4 * y - 13 = 0 :=
by sorry

end chord_bisector_line_equation_l1365_136584


namespace units_digit_of_seven_to_sixth_l1365_136560

theorem units_digit_of_seven_to_sixth (n : ℕ) : n = 7^6 → n % 10 = 9 := by
  sorry

end units_digit_of_seven_to_sixth_l1365_136560


namespace three_digit_divisibility_by_seven_l1365_136583

theorem three_digit_divisibility_by_seven :
  ∃ (start : ℕ), 
    (100 ≤ start) ∧ 
    (start + 127 ≤ 999) ∧ 
    (∀ k : ℕ, k < 128 → (start + k) % 7 = (start % 7)) ∧
    (start % 7 = 0) := by
  sorry

end three_digit_divisibility_by_seven_l1365_136583


namespace greatest_divisor_with_remainders_l1365_136573

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (h1 : a > r1) (h2 : b > r2) : 
  Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (a % (Nat.gcd (a - r1) (b - r2))) r1 ∧ 
    Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (b % (Nat.gcd (a - r1) (b - r2))) r2 → 
  Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (1642 - 6) (1856 - 4) := by
  sorry

#eval Nat.gcd (1642 - 6) (1856 - 4)

end greatest_divisor_with_remainders_l1365_136573


namespace functional_equation_solution_l1365_136543

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y + y * f x) = f x + f y + x * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end functional_equation_solution_l1365_136543


namespace equal_roots_implies_a_equals_negative_one_l1365_136574

/-- The quadratic equation with parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x * (x + 1) + a * x

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ := (1 + a)^2

theorem equal_roots_implies_a_equals_negative_one :
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ 
    ∀ y : ℝ, quadratic_equation a y = 0 → y = x) →
  discriminant a = 0 →
  a = -1 :=
sorry

end equal_roots_implies_a_equals_negative_one_l1365_136574


namespace square_not_always_positive_l1365_136549

theorem square_not_always_positive : ¬ ∀ x : ℝ, x^2 > 0 := by
  sorry

end square_not_always_positive_l1365_136549


namespace quadratic_inequality_not_always_negative_l1365_136572

theorem quadratic_inequality_not_always_negative :
  ¬ (∀ x : ℝ, x^2 + x - 1 < 0) :=
sorry

end quadratic_inequality_not_always_negative_l1365_136572


namespace equidistant_points_count_l1365_136511

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and distance from origin --/
structure Line where
  normal : ℝ × ℝ
  distance : ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Distance between a point and a line --/
def distancePointToLine (p : Point) (l : Line) : ℝ := sorry

/-- Distance between a point and a circle --/
def distancePointToCircle (p : Point) (c : Circle) : ℝ := sorry

/-- Check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- The main theorem --/
theorem equidistant_points_count
  (c : Circle)
  (t1 t2 : Line)
  (h1 : c.radius = 4)
  (h2 : isTangent t1 c)
  (h3 : isTangent t2 c)
  (h4 : t1.distance = 4)
  (h5 : t2.distance = 6)
  (h6 : t1.normal = t2.normal) :
  ∃! (s : Finset Point), 
    (∀ p ∈ s, distancePointToCircle p c = distancePointToLine p t1 ∧ 
                distancePointToCircle p c = distancePointToLine p t2) ∧
    s.card = 2 := by sorry

end equidistant_points_count_l1365_136511


namespace beth_cookie_price_l1365_136561

/-- Represents a cookie batch with a count and price per cookie -/
structure CookieBatch where
  count : ℕ
  price : ℚ

/-- Calculates the total earnings from a cookie batch -/
def totalEarnings (batch : CookieBatch) : ℚ :=
  batch.count * batch.price

theorem beth_cookie_price (alan_batch beth_batch : CookieBatch) : 
  alan_batch.count = 15 → 
  alan_batch.price = 1/2 → 
  beth_batch.count = 18 → 
  totalEarnings alan_batch = totalEarnings beth_batch → 
  beth_batch.price = 21/50 := by
sorry

#eval (21 : ℚ) / 50

end beth_cookie_price_l1365_136561


namespace election_winner_margin_l1365_136501

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) : 
  (2 : ℕ) ≤ total_votes →
  winner_votes = (75 * total_votes) / 100 →
  winner_votes = 750 →
  winner_votes - (total_votes - winner_votes) = 500 :=
by sorry

end election_winner_margin_l1365_136501


namespace square_perimeter_from_diagonal_l1365_136542

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  4 * (d / Real.sqrt 2) = 8 := by sorry

end square_perimeter_from_diagonal_l1365_136542


namespace arithmetic_mean_of_range_l1365_136517

def integer_range : List Int := List.range 14 |>.map (fun i => i - 6)

theorem arithmetic_mean_of_range : 
  (integer_range.sum : ℚ) / integer_range.length = 1/2 := by
  sorry

end arithmetic_mean_of_range_l1365_136517


namespace triangle_sine_sum_bound_l1365_136556

/-- Given a triangle with angles A, B, and C (in radians), 
    the sum of the sines of its angles is at most 3√3/2, 
    with equality if and only if the triangle is equilateral. -/
theorem triangle_sine_sum_bound (A B C : ℝ) 
    (h_angles : A + B + C = π) 
    (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 ∧ 
  (Real.sin A + Real.sin B + Real.sin C = 3 * Real.sqrt 3 / 2 ↔ A = B ∧ B = C) := by
sorry

end triangle_sine_sum_bound_l1365_136556


namespace inequality_solution_l1365_136579

theorem inequality_solution (x : ℝ) : (3*x + 7)/5 + 1 > x ↔ x < 6 := by sorry

end inequality_solution_l1365_136579


namespace unique_number_existence_l1365_136593

def digit_sum (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

def has_digit (n d : ℕ) : Prop := sorry

def all_nines_except_one (n : ℕ) (pos : ℕ) : Prop := sorry

theorem unique_number_existence :
  ∃! N : ℕ,
    (num_digits N = 1112) ∧
    (2000 ∣ digit_sum N) ∧
    (2000 ∣ digit_sum (N + 1)) ∧
    (has_digit N 1) ∧
    (all_nines_except_one N 890) :=
by
  sorry

end unique_number_existence_l1365_136593


namespace broken_stick_pairing_probability_l1365_136570

/-- The number of sticks --/
def n : ℕ := 5

/-- The probability of pairing each long part with a short part when rearranging broken sticks --/
theorem broken_stick_pairing_probability :
  (2^n : ℚ) / (Nat.choose (2*n) n : ℚ) = 8/63 := by sorry

end broken_stick_pairing_probability_l1365_136570


namespace xyz_value_l1365_136589

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 20 * Real.rpow 2 (1/3))
  (hxz : x * z = 35 * Real.rpow 2 (1/3))
  (hyz : y * z = 14 * Real.rpow 2 (1/3)) :
  x * y * z = 140 := by
sorry

end xyz_value_l1365_136589


namespace parallelogram_roots_l1365_136590

/-- The polynomial whose roots we're investigating -/
def P (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 13*b*z^2 - 3*(2*b^2 + 5*b - 3)*z - 1

/-- Predicate to check if a set of complex numbers forms a parallelogram -/
def formsParallelogram (roots : Finset ℂ) : Prop :=
  roots.card = 4 ∧ ∃ (w₁ w₂ : ℂ), roots = {w₁, -w₁, w₂, -w₂}

/-- The main theorem stating that 3/2 is the only real value of b for which
    the roots of P form a parallelogram -/
theorem parallelogram_roots :
  ∃! (b : ℝ), b = 3/2 ∧ 
    ∃ (roots : Finset ℂ), (∀ z ∈ roots, P b z = 0) ∧ formsParallelogram roots :=
sorry

end parallelogram_roots_l1365_136590


namespace sqrt_640000_equals_800_l1365_136538

theorem sqrt_640000_equals_800 : Real.sqrt 640000 = 800 := by
  sorry

end sqrt_640000_equals_800_l1365_136538


namespace jerry_max_showers_l1365_136524

/-- Represents the water usage scenario for Jerry in July --/
structure WaterUsage where
  total_allowance : ℕ
  drinking_cooking : ℕ
  shower_usage : ℕ
  pool_length : ℕ
  pool_width : ℕ
  pool_height : ℕ
  gallon_per_cubic_foot : ℕ
  leakage_rate : ℕ
  days_in_july : ℕ

/-- Calculates the maximum number of showers Jerry can take in July --/
def max_showers (w : WaterUsage) : ℕ :=
  let pool_volume := w.pool_length * w.pool_width * w.pool_height
  let pool_water := pool_volume * w.gallon_per_cubic_foot
  let total_leakage := w.leakage_rate * w.days_in_july
  let water_for_showers := w.total_allowance - w.drinking_cooking - pool_water - total_leakage
  water_for_showers / w.shower_usage

/-- Theorem stating that Jerry can take at most 7 showers in July --/
theorem jerry_max_showers :
  let w : WaterUsage := {
    total_allowance := 1000,
    drinking_cooking := 100,
    shower_usage := 20,
    pool_length := 10,
    pool_width := 10,
    pool_height := 6,
    gallon_per_cubic_foot := 1,
    leakage_rate := 5,
    days_in_july := 31
  }
  max_showers w = 7 := by
  sorry


end jerry_max_showers_l1365_136524


namespace boat_travel_time_l1365_136546

theorem boat_travel_time (v : ℝ) :
  let upstream_speed := v - 4
  let downstream_speed := v + 4
  let distance := 120
  let upstream_time (t : ℝ) := t + 1
  let downstream_time := 1
  (upstream_speed * upstream_time downstream_time = distance) ∧
  (downstream_speed * downstream_time = distance) →
  downstream_time = 1 :=
by
  sorry

end boat_travel_time_l1365_136546


namespace infinite_representations_l1365_136520

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 10*x^2 + 29*x - 25

-- Define the property for a number to be a root of f
def is_root (x : ℝ) : Prop := f x = 0

-- Define the property for two numbers to be distinct
def are_distinct (x y : ℝ) : Prop := x ≠ y

-- Define the property for a positive integer to have the required representation
def has_representation (n : ℕ) (α β : ℝ) : Prop :=
  ∃ (r s : ℤ), n = ⌊r * α⌋ ∧ n = ⌊s * β⌋

-- State the theorem
theorem infinite_representations :
  ∃ (α β : ℝ), is_root α ∧ is_root β ∧ are_distinct α β ∧
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → has_representation n α β :=
sorry

end infinite_representations_l1365_136520


namespace four_thirds_of_x_is_36_l1365_136592

theorem four_thirds_of_x_is_36 : ∃ x : ℚ, (4 / 3) * x = 36 ∧ x = 27 := by
  sorry

end four_thirds_of_x_is_36_l1365_136592


namespace recurring_decimal_fraction_sum_l1365_136581

theorem recurring_decimal_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 56 / 99 → 
  Nat.gcd a b = 1 → 
  (a : ℕ) + b = 155 := by
  sorry

end recurring_decimal_fraction_sum_l1365_136581


namespace shirt_cost_l1365_136575

theorem shirt_cost (initial_amount : ℕ) (socks_cost : ℕ) (amount_left : ℕ) :
  initial_amount = 100 →
  socks_cost = 11 →
  amount_left = 65 →
  initial_amount - amount_left - socks_cost = 24 := by
sorry

end shirt_cost_l1365_136575


namespace roots_expression_equals_one_l1365_136576

theorem roots_expression_equals_one (α β γ δ : ℝ) : 
  (α^2 - 2*α + 1 = 0) → 
  (β^2 - 2*β + 1 = 0) → 
  (γ^2 - 3*γ + 1 = 0) → 
  (δ^2 - 3*δ + 1 = 0) → 
  (α - γ)^2 * (β - δ)^2 = 1 := by
sorry

end roots_expression_equals_one_l1365_136576


namespace quadratic_equation_solution_l1365_136553

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 7 * x^2 + 13 * x - 30 = 0 :=
by
  -- The unique solution is x = 10/7
  use 10/7
  sorry

end quadratic_equation_solution_l1365_136553


namespace thyme_leaves_theorem_l1365_136518

/-- The number of leaves per thyme plant -/
def thyme_leaves_per_plant : ℕ :=
  let basil_pots : ℕ := 3
  let rosemary_pots : ℕ := 9
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let total_leaves : ℕ := 354
  let basil_leaves : ℕ := basil_pots * basil_leaves_per_pot
  let rosemary_leaves : ℕ := rosemary_pots * rosemary_leaves_per_pot
  let thyme_leaves : ℕ := total_leaves - basil_leaves - rosemary_leaves
  thyme_leaves / thyme_pots

theorem thyme_leaves_theorem : thyme_leaves_per_plant = 30 := by
  sorry

end thyme_leaves_theorem_l1365_136518


namespace intersection_point_is_7_neg8_l1365_136539

/-- Two lines in 2D space --/
structure TwoLines where
  line1 : ℝ → ℝ × ℝ
  line2 : ℝ → ℝ × ℝ

/-- The given two lines from the problem --/
def givenLines : TwoLines where
  line1 := λ t => (1 + 2*t, 1 - 3*t)
  line2 := λ u => (5 + 4*u, -9 + 2*u)

/-- Definition of intersection point --/
def isIntersectionPoint (p : ℝ × ℝ) (lines : TwoLines) : Prop :=
  ∃ t u, lines.line1 t = p ∧ lines.line2 u = p

/-- Theorem stating that (7, -8) is the intersection point of the given lines --/
theorem intersection_point_is_7_neg8 :
  isIntersectionPoint (7, -8) givenLines := by
  sorry

end intersection_point_is_7_neg8_l1365_136539


namespace lcm_gcf_ratio_240_360_l1365_136533

theorem lcm_gcf_ratio_240_360 : 
  (Nat.lcm 240 360) / (Nat.gcd 240 360) = 60 := by
  sorry

end lcm_gcf_ratio_240_360_l1365_136533


namespace ball_distribution_l1365_136587

theorem ball_distribution (n : ℕ) (k : ℕ) : 
  (∃ x y z : ℕ, x + y + z = n ∧ x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3) →
  (Nat.choose (n - 6 + k - 1) (k - 1) = 15) →
  (k = 3 ∧ n = 10) :=
by sorry

end ball_distribution_l1365_136587


namespace largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5_l1365_136545

theorem largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5 : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n % 18 = 0 ∧ 
    (26 : ℝ) < Real.sqrt n ∧ 
    Real.sqrt n ≤ 26.5 ∧
    ∀ (m : ℕ), m > 0 ∧ m % 18 = 0 ∧ (26 : ℝ) < Real.sqrt m ∧ Real.sqrt m ≤ 26.5 → m ≤ n :=
by
  sorry

end largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5_l1365_136545


namespace max_value_theorem_l1365_136596

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 2*a*b*Real.sqrt 3 + 2*a*c ≤ Real.sqrt 3 := by
  sorry

end max_value_theorem_l1365_136596


namespace total_bill_sum_l1365_136563

-- Define the variables for each person's bill
variable (alice_bill : ℝ) (bob_bill : ℝ) (charlie_bill : ℝ)

-- Define the conditions
axiom alice_tip : 0.15 * alice_bill = 3
axiom bob_tip : 0.25 * bob_bill = 5
axiom charlie_tip : 0.20 * charlie_bill = 4

-- Theorem statement
theorem total_bill_sum :
  alice_bill + bob_bill + charlie_bill = 60 :=
sorry

end total_bill_sum_l1365_136563


namespace correct_observation_value_l1365_136550

theorem correct_observation_value 
  (n : ℕ) 
  (original_mean : ℚ) 
  (incorrect_value : ℚ) 
  (corrected_mean : ℚ) 
  (h1 : n = 50) 
  (h2 : original_mean = 30) 
  (h3 : incorrect_value = 23) 
  (h4 : corrected_mean = 30.5) : 
  (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - incorrect_value) = 48 :=
by sorry

end correct_observation_value_l1365_136550


namespace mars_mission_cost_share_l1365_136565

-- Define the total cost in billions of dollars
def total_cost : ℕ := 25

-- Define the number of people sharing the cost in millions
def num_people : ℕ := 200

-- Define the conversion factor from billions to millions
def billion_to_million : ℕ := 1000

-- Theorem statement
theorem mars_mission_cost_share :
  (total_cost * billion_to_million) / num_people = 125 := by
sorry

end mars_mission_cost_share_l1365_136565


namespace flower_arrangement_count_l1365_136578

/-- The number of ways to choose k items from n items -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of ways to arrange k items in k distinct positions -/
def arrangements (k : ℕ) : ℕ := Nat.factorial k

theorem flower_arrangement_count :
  let n : ℕ := 5  -- Total number of flower types
  let k : ℕ := 2  -- Number of flowers to pick
  (combinations n k) * (arrangements k) = 20 := by
  sorry

#eval (combinations 5 2) * (arrangements 2)  -- Should output 20

end flower_arrangement_count_l1365_136578


namespace arrange_seven_white_five_black_l1365_136588

/-- The number of ways to arrange white and black balls with constraints -/
def arrangeBalls (white black : ℕ) : ℕ :=
  Nat.choose (white + 1) black

/-- Theorem stating the number of ways to arrange 7 white balls and 5 black balls -/
theorem arrange_seven_white_five_black :
  arrangeBalls 7 5 = 56 := by
  sorry

end arrange_seven_white_five_black_l1365_136588


namespace stock_price_return_l1365_136582

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_increase := initial_price * 1.25
  let price_after_decrease := price_after_increase * 0.8
  price_after_decrease = initial_price :=
by sorry

end stock_price_return_l1365_136582


namespace factor_representation_1000000_l1365_136525

/-- The number of ways to represent 1,000,000 as the product of three factors -/
def factor_representation (n : ℕ) (distinct_order : Bool) : ℕ :=
  if distinct_order then 784 else 139

/-- Theorem stating the number of ways to represent 1,000,000 as the product of three factors -/
theorem factor_representation_1000000 :
  (factor_representation 1000000 true = 784) ∧
  (factor_representation 1000000 false = 139) := by
  sorry

end factor_representation_1000000_l1365_136525


namespace roots_sum_absolute_values_l1365_136580

theorem roots_sum_absolute_values (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2013*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  abs a + abs b + abs c = 94 := by
sorry

end roots_sum_absolute_values_l1365_136580


namespace quadratic_inequality_implies_a_range_l1365_136586

theorem quadratic_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a < 1) := by
  sorry

end quadratic_inequality_implies_a_range_l1365_136586


namespace stock_price_after_two_years_l1365_136521

/-- Calculates the final stock price after two years of growth -/
def final_stock_price (initial_price : ℝ) (first_year_growth : ℝ) (second_year_growth : ℝ) : ℝ :=
  initial_price * (1 + first_year_growth) * (1 + second_year_growth)

/-- Theorem stating that the stock price after two years of growth is $247.50 -/
theorem stock_price_after_two_years :
  final_stock_price 150 0.5 0.1 = 247.50 := by
  sorry

#eval final_stock_price 150 0.5 0.1

end stock_price_after_two_years_l1365_136521


namespace parallelogram_below_line_l1365_136541

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

def BelowOrOnLine (p : Point) (y0 : ℝ) : Prop :=
  p.y ≤ y0

theorem parallelogram_below_line :
  let A : Point := ⟨4, 2⟩
  let B : Point := ⟨-2, -4⟩
  let C : Point := ⟨-8, -4⟩
  let D : Point := ⟨0, 4⟩
  let y0 : ℝ := -2
  Parallelogram A B C D →
  ∀ p : Point, (∃ t u : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ u ∧ u ≤ 1 ∧
    p.x = A.x + t * (B.x - A.x) + u * (D.x - A.x) ∧
    p.y = A.y + t * (B.y - A.y) + u * (D.y - A.y)) →
  BelowOrOnLine p y0 := by
sorry

end parallelogram_below_line_l1365_136541


namespace train_count_l1365_136523

theorem train_count (carriages_per_train : ℕ) (rows_per_carriage : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  carriages_per_train = 4 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  total_wheels = 240 →
  total_wheels / (carriages_per_train * rows_per_carriage * wheels_per_row) = 4 :=
by
  sorry

end train_count_l1365_136523


namespace quadratic_expression_value_l1365_136526

theorem quadratic_expression_value :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 + z^2 + 2*x*z = 26 := by
sorry

end quadratic_expression_value_l1365_136526


namespace no_natural_solution_for_x2_plus_y2_eq_7z2_l1365_136559

theorem no_natural_solution_for_x2_plus_y2_eq_7z2 :
  ¬ ∃ (x y z : ℕ), x^2 + y^2 = 7 * z^2 := by
  sorry

end no_natural_solution_for_x2_plus_y2_eq_7z2_l1365_136559


namespace marble_jar_theorem_l1365_136564

/-- Represents a jar of marbles with orange, purple, and yellow colors. -/
structure MarbleJar where
  orange : ℝ
  purple : ℝ
  yellow : ℝ

/-- The total number of marbles in the jar. -/
def MarbleJar.total (jar : MarbleJar) : ℝ :=
  jar.orange + jar.purple + jar.yellow

/-- A jar satisfying the given conditions. -/
def specialJar : MarbleJar :=
  { orange := 0,  -- placeholder values
    purple := 0,
    yellow := 0 }

theorem marble_jar_theorem (jar : MarbleJar) :
  jar.purple + jar.yellow = 7 →
  jar.orange + jar.yellow = 5 →
  jar.orange + jar.purple = 9 →
  jar.total = 10.5 := by
  sorry

#check marble_jar_theorem

end marble_jar_theorem_l1365_136564


namespace triangles_with_integer_sides_not_exceeding_two_l1365_136597

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_sides_not_exceeding_two (a b c : ℕ) : Prop :=
  a ≤ 2 ∧ b ≤ 2 ∧ c ≤ 2

theorem triangles_with_integer_sides_not_exceeding_two :
  ∃! (S : Set (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ 
      is_valid_triangle a b c ∧ 
      triangle_sides_not_exceeding_two a b c) ∧
    S = {(1, 1, 1), (2, 2, 1), (2, 2, 2)} :=
sorry

end triangles_with_integer_sides_not_exceeding_two_l1365_136597


namespace remainder_equality_l1365_136516

theorem remainder_equality (a b : ℕ+) :
  (∀ p : ℕ, Nat.Prime p → 
    (a : ℕ) % p ≤ (b : ℕ) % p) →
  a = b := by
sorry

end remainder_equality_l1365_136516


namespace triangle_area_l1365_136527

theorem triangle_area (A B C : Real) (angleC : A + B + C = Real.pi) 
  (sideAC sideAB : Real) (h_angleC : C = Real.pi / 6) 
  (h_sideAC : sideAC = 3 * Real.sqrt 3) (h_sideAB : sideAB = 3) :
  let area := (1 / 2) * sideAC * sideAB * Real.sin A
  area = (9 * Real.sqrt 3) / 2 ∨ area = (9 * Real.sqrt 3) / 4 := by
sorry

end triangle_area_l1365_136527


namespace b_days_proof_l1365_136522

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 6

/-- The total payment for the work -/
def total_payment : ℝ := 3600

/-- The number of days it takes A, B, and C to complete the work together -/
def abc_days : ℝ := 3

/-- The payment given to C -/
def c_payment : ℝ := 450

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 6

theorem b_days_proof :
  (1 / a_days + 1 / b_days) * abc_days = 1 ∧
  c_payment / total_payment = 1 - (1 / a_days + 1 / b_days) * abc_days :=
by sorry

end b_days_proof_l1365_136522


namespace converse_not_always_true_l1365_136508

theorem converse_not_always_true : ¬(∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by
  sorry

end converse_not_always_true_l1365_136508


namespace f_properties_l1365_136537

noncomputable section

open Real

/-- The function f(x) = ae^(2x) - ae^x - xe^x --/
def f (a : ℝ) (x : ℝ) : ℝ := a * exp (2 * x) - a * exp x - x * exp x

/-- The theorem stating the properties of f --/
theorem f_properties :
  ∀ a : ℝ, a ≥ 0 → (∀ x : ℝ, f a x ≥ 0) →
  ∃ x₀ : ℝ,
    a = 1 ∧
    (∀ x : ℝ, f 1 x ≤ f 1 x₀) ∧
    (∀ x : ℝ, x ≠ x₀ → f 1 x < f 1 x₀) ∧
    (log 2 / (2 * exp 1) + 1 / (4 * exp 1 ^ 2) ≤ f 1 x₀) ∧
    (f 1 x₀ < 1 / 4) := by
  sorry

end f_properties_l1365_136537


namespace angle_in_third_quadrant_l1365_136513

theorem angle_in_third_quadrant (α : Real) (h1 : π < α ∧ α < 3*π/2) : 
  (Real.sin (π/2 - α) * Real.cos (-α) * Real.tan (π + α)) / Real.cos (π - α) = 2 * Real.sqrt 5 / 5 →
  Real.cos α = -(Real.sqrt 5 / 5) := by
  sorry

end angle_in_third_quadrant_l1365_136513


namespace fraction_division_result_l1365_136585

theorem fraction_division_result (a : ℝ) 
  (h1 : a^2 + 4*a + 4 ≠ 0) 
  (h2 : a^2 + 5*a + 6 ≠ 0) : 
  (a^2 - 4) / (a^2 + 4*a + 4) / ((a^2 + a - 6) / (a^2 + 5*a + 6)) = 1 := by
  sorry

end fraction_division_result_l1365_136585
