import Mathlib

namespace symmetric_points_line_equation_l2188_218874

/-- Given two points are symmetric about a line, prove the equation of the line -/
theorem symmetric_points_line_equation (O A : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  O = (0, 0) →
  A = (-4, 2) →
  (∀ p : ℝ × ℝ, p ∈ l ↔ (p.1 - O.1) * (A.1 - O.1) + (p.2 - O.2) * (A.2 - O.2) = 0) →
  (∀ x y : ℝ, (x, y) ∈ l ↔ 2*x - y + 5 = 0) :=
by sorry

end symmetric_points_line_equation_l2188_218874


namespace delores_money_left_l2188_218860

/-- Calculates the money left after purchases given initial amount and costs --/
def money_left (initial_amount : ℕ) (computer_cost : ℕ) (printer_cost : ℕ) : ℕ :=
  initial_amount - (computer_cost + printer_cost)

theorem delores_money_left :
  money_left 450 400 40 = 10 := by
  sorry

end delores_money_left_l2188_218860


namespace constant_revenue_increase_l2188_218845

def revenue : Fin 14 → ℕ
  | 0  => 150000  -- January (year 1)
  | 1  => 180000  -- February (year 1)
  | 2  => 210000  -- March (year 1)
  | 3  => 240000  -- April (year 1)
  | 4  => 270000  -- May (year 1)
  | 5  => 300000  -- June (year 1)
  | 6  => 330000  -- July (year 1)
  | 7  => 300000  -- August (year 1)
  | 8  => 270000  -- September (year 1)
  | 9  => 300000  -- October (year 1)
  | 10 => 330000  -- November (year 1)
  | 11 => 360000  -- December (year 1)
  | 12 => 390000  -- January (year 2)
  | 13 => 420000  -- February (year 2)

theorem constant_revenue_increase :
  ∀ i : Fin 13, i.val ≠ 6 ∧ i.val ≠ 7 →
    revenue (i + 1) - revenue i = 30000 :=
by sorry

end constant_revenue_increase_l2188_218845


namespace main_result_l2188_218808

/-- Average of two numbers -/
def avg (a b : ℚ) : ℚ := (a + b) / 2

/-- Weighted average of four numbers with weights 1:2:1:2 -/
def wavg (a b c d : ℚ) : ℚ := (a + 2*b + c + 2*d) / 6

/-- The main theorem to prove -/
theorem main_result : wavg (wavg 2 2 1 1) (avg 1 2) 0 2 = 17/12 := by sorry

end main_result_l2188_218808


namespace eight_digit_increasing_remainder_l2188_218812

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of 8-digit positive integers with digits in increasing order -/
def M : ℕ := 9 * stars_and_bars 7 10

theorem eight_digit_increasing_remainder :
  M % 1000 = 960 := by sorry

end eight_digit_increasing_remainder_l2188_218812


namespace pencil_box_calculation_l2188_218897

/-- Given a total number of pencils and pencils per box, calculate the number of filled boxes -/
def filled_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) : ℕ :=
  total_pencils / pencils_per_box

/-- Theorem: Given 648 pencils and 4 pencils per box, the number of filled boxes is 162 -/
theorem pencil_box_calculation :
  filled_boxes 648 4 = 162 := by
  sorry

end pencil_box_calculation_l2188_218897


namespace robert_birth_year_l2188_218806

def first_amc8_year : ℕ := 1985

def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

def robert_age_at_tenth_amc8 : ℕ := 15

theorem robert_birth_year :
  ∃ (birth_year : ℕ),
    birth_year = amc8_year 10 - robert_age_at_tenth_amc8 ∧
    birth_year = 1979 :=
by sorry

end robert_birth_year_l2188_218806


namespace initial_cells_eq_one_l2188_218886

/-- Represents the doubling time of the bacteria in minutes -/
def doubling_time : ℕ := 20

/-- Represents the growth time in hours -/
def growth_time : ℕ := 4

/-- Represents the final number of bacterial cells -/
def final_cells : ℕ := 4096

/-- Calculates the number of doublings that occurred during the growth period -/
def num_doublings : ℕ := growth_time * 60 / doubling_time

/-- Represents the initial number of bacterial cells -/
def initial_cells : ℕ := final_cells / (2^num_doublings)

/-- Proves that the initial number of cells was 1 -/
theorem initial_cells_eq_one : initial_cells = 1 := by
  sorry

end initial_cells_eq_one_l2188_218886


namespace transformed_expr_at_one_l2188_218842

-- Define the original expression
def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

-- Define the transformed expression
def transformed_expr (x : ℚ) : ℚ := 
  (original_expr x + 2) / (original_expr x - 3)

-- Theorem statement
theorem transformed_expr_at_one :
  transformed_expr 1 = -1/9 := by sorry

end transformed_expr_at_one_l2188_218842


namespace mr_green_garden_yield_l2188_218824

/-- Represents the dimensions and expected yield of a rectangular garden -/
structure Garden where
  length_paces : ℕ
  width_paces : ℕ
  feet_per_pace : ℕ
  yield_per_sqft : ℚ

/-- Calculates the expected potato yield from a garden in pounds -/
def expected_yield (g : Garden) : ℚ :=
  (g.length_paces * g.feet_per_pace) *
  (g.width_paces * g.feet_per_pace) *
  g.yield_per_sqft

/-- Theorem stating the expected yield for Mr. Green's garden -/
theorem mr_green_garden_yield :
  let g : Garden := {
    length_paces := 18,
    width_paces := 25,
    feet_per_pace := 3,
    yield_per_sqft := 3/4
  }
  expected_yield g = 3037.5 := by sorry

end mr_green_garden_yield_l2188_218824


namespace expression_properties_l2188_218871

def expression_result (signs : List Bool) : Int :=
  let nums := List.range 9
  List.foldl (λ acc (n, sign) => if sign then acc + (n + 1) else acc - (n + 1)) 0 (List.zip nums signs)

theorem expression_properties :
  (∀ signs : List Bool, expression_result signs ≠ 0) ∧
  (∃ signs : List Bool, expression_result signs = 1) ∧
  (∀ n : Int, (n % 2 = 1 ∧ -45 ≤ n ∧ n ≤ 45) ↔ ∃ signs : List Bool, expression_result signs = n) := by
  sorry

end expression_properties_l2188_218871


namespace tom_time_ratio_l2188_218868

/-- The duration of the BS program in years -/
def bs_duration : ℕ := 3

/-- The duration of the Ph.D. program in years -/
def phd_duration : ℕ := 5

/-- Tom's total time to complete both programs in years -/
def tom_total_time : ℕ := 6

/-- The normal time to complete both programs -/
def normal_time : ℕ := bs_duration + phd_duration

theorem tom_time_ratio :
  (tom_total_time : ℚ) / (normal_time : ℚ) = 3 / 4 := by
  sorry

end tom_time_ratio_l2188_218868


namespace max_trip_weight_l2188_218854

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 1250

theorem max_trip_weight :
  max_crates * min_crate_weight = 6250 :=
by sorry

end max_trip_weight_l2188_218854


namespace otimes_inequality_solutions_l2188_218830

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

-- Define the set of non-negative integers
def NonNegIntegers : Set ℤ := {x : ℤ | x ≥ 0}

-- Theorem statement
theorem otimes_inequality_solutions :
  {x ∈ NonNegIntegers | otimes 2 x ≥ 3} = {0, 1} := by sorry

end otimes_inequality_solutions_l2188_218830


namespace factors_of_1320_l2188_218887

theorem factors_of_1320 : Finset.card (Nat.divisors 1320) = 24 := by
  sorry

end factors_of_1320_l2188_218887


namespace circle_arrangement_theorem_l2188_218896

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def circleA : Circle := { center := { x := 0, y := -1 }, radius := 1 }
def circleB : Circle := { center := { x := 5, y := 3 }, radius := 3 }
def circleC : Circle := { center := { x := 8, y := -4 }, radius := 4 }

def line_l : Line := { a := 0, b := 1, c := 0 }

def is_below (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

def is_above (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

def is_tangent (c : Circle) (l : Line) : Prop :=
  abs (l.a * c.center.x + l.b * c.center.y + l.c) = c.radius * (l.a^2 + l.b^2).sqrt

def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let dx := c1.center.x - c2.center.x
  let dy := c1.center.y - c2.center.y
  (dx^2 + dy^2).sqrt = c1.radius + c2.radius

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem circle_arrangement_theorem :
  is_below circleA.center line_l ∧
  is_below circleC.center line_l ∧
  is_above circleB.center line_l ∧
  is_tangent circleA line_l ∧
  is_tangent circleB line_l ∧
  is_tangent circleC line_l ∧
  are_externally_tangent circleB circleA ∧
  are_externally_tangent circleB circleC →
  triangle_area circleA.center circleB.center circleC.center = 23.5 := by
  sorry

end circle_arrangement_theorem_l2188_218896


namespace parabola_opens_downwards_l2188_218869

/-- A parabola opens downwards if its quadratic coefficient is negative -/
def opens_downwards (a b c : ℝ) : Prop :=
  a < 0

/-- The theorem states that for a = -3, the parabola y = ax^2 + bx + c opens downwards -/
theorem parabola_opens_downwards :
  let a : ℝ := -3
  opens_downwards a b c := by sorry

end parabola_opens_downwards_l2188_218869


namespace l_shaped_field_area_l2188_218864

theorem l_shaped_field_area :
  let field_length : ℕ := 10
  let field_width : ℕ := 7
  let removed_length_diff : ℕ := 3
  let removed_width_diff : ℕ := 2
  let removed_length : ℕ := field_length - removed_length_diff
  let removed_width : ℕ := field_width - removed_width_diff
  let total_area : ℕ := field_length * field_width
  let removed_area : ℕ := removed_length * removed_width
  let l_shaped_area : ℕ := total_area - removed_area
  l_shaped_area = 35 := by sorry

end l_shaped_field_area_l2188_218864


namespace quadratic_function_value_l2188_218838

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  f a b c 1 = 3 → f a b c 2 = 12 → f a b c 3 = 27 → f a b c 4 = 48 := by
  sorry

end quadratic_function_value_l2188_218838


namespace battery_factory_robots_l2188_218829

/-- The number of robots working simultaneously in a battery factory -/
def num_robots : ℕ :=
  let time_per_battery : ℕ := 15  -- 6 minutes for materials + 9 minutes for creation
  let total_time : ℕ := 300       -- 5 hours * 60 minutes
  let total_batteries : ℕ := 200
  total_batteries * time_per_battery / total_time

theorem battery_factory_robots :
  num_robots = 10 :=
sorry

end battery_factory_robots_l2188_218829


namespace range_of_m_l2188_218853

theorem range_of_m (m : ℝ) : 
  (¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0))) → 
  m ∈ Set.Iic (-2) ∪ Set.Ioi (-1) :=
sorry

end range_of_m_l2188_218853


namespace percent_relation_l2188_218820

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.6 * x := by
sorry

end percent_relation_l2188_218820


namespace cubic_expression_value_l2188_218846

theorem cubic_expression_value (m n : ℝ) 
  (h1 : m^2 = n + 2) 
  (h2 : n^2 = m + 2) 
  (h3 : m ≠ n) : 
  m^3 - 2*m*n + n^3 = -2 := by
  sorry

end cubic_expression_value_l2188_218846


namespace unique_two_digit_integer_l2188_218849

theorem unique_two_digit_integer (u : ℕ) : 
  (10 ≤ u ∧ u < 100) ∧ (13 * u) % 100 = 52 ↔ u = 4 := by sorry

end unique_two_digit_integer_l2188_218849


namespace max_player_salary_l2188_218893

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 25 →
  min_salary = 18000 →
  max_total = 1000000 →
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary = 568000 :=
by sorry

end max_player_salary_l2188_218893


namespace mail_order_cost_l2188_218870

/-- The total cost of mail ordering books with a shipping fee -/
def total_cost (unit_price : ℝ) (shipping_rate : ℝ) (num_books : ℝ) : ℝ :=
  unit_price * num_books * (1 + shipping_rate)

/-- Theorem: The total cost of mail ordering 'a' books with a unit price of 8 yuan and a 10% shipping fee is 8(1+10%)a yuan -/
theorem mail_order_cost (a : ℝ) : 
  total_cost 8 0.1 a = 8 * (1 + 0.1) * a := by
  sorry

end mail_order_cost_l2188_218870


namespace deepak_current_age_l2188_218839

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The ratio between Rahul and Deepak's ages -/
def age_ratio (ages : Ages) : ℚ :=
  ages.rahul / ages.deepak

/-- Rahul's age after 6 years -/
def rahul_future_age (ages : Ages) : ℕ :=
  ages.rahul + 6

theorem deepak_current_age (ages : Ages) :
  age_ratio ages = 4/3 →
  rahul_future_age ages = 42 →
  ages.deepak = 27 := by
sorry

end deepak_current_age_l2188_218839


namespace max_value_is_nine_l2188_218859

-- Define the set of possible values
def S : Finset ℕ := {1, 2, 4, 5}

-- Define the expression to be maximized
def f (x y z w : ℕ) : ℤ := x * y - y * z + z * w - w * x

-- Theorem statement
theorem max_value_is_nine :
  ∃ (x y z w : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ w ∈ S ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  f x y z w = 9 ∧
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  f a b c d ≤ 9 :=
by sorry

end max_value_is_nine_l2188_218859


namespace sticker_distribution_l2188_218807

theorem sticker_distribution (total : ℕ) (andrew_kept : ℕ) (daniel_received : ℕ) 
  (h1 : total = 750)
  (h2 : andrew_kept = 130)
  (h3 : daniel_received = 250) :
  total - andrew_kept - daniel_received - daniel_received = 120 :=
by sorry

end sticker_distribution_l2188_218807


namespace yulia_profit_is_44_l2188_218828

/-- Calculates Yulia's profit given her revenues and expenses -/
def yulia_profit (lemonade_revenue babysitting_revenue lemonade_expenses : ℕ) : ℕ :=
  (lemonade_revenue + babysitting_revenue) - lemonade_expenses

/-- Proves that Yulia's profit is $44 given the provided revenues and expenses -/
theorem yulia_profit_is_44 :
  yulia_profit 47 31 34 = 44 := by
  sorry

end yulia_profit_is_44_l2188_218828


namespace mileage_reimbursement_rate_calculation_l2188_218848

/-- Calculates the mileage reimbursement rate given daily mileages and total reimbursement -/
def mileage_reimbursement_rate (monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement : ℚ) : ℚ :=
  total_reimbursement / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem mileage_reimbursement_rate_calculation 
  (monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement : ℚ) :
  mileage_reimbursement_rate monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement =
  total_reimbursement / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) :=
by sorry

end mileage_reimbursement_rate_calculation_l2188_218848


namespace oil_price_rollback_l2188_218872

def current_price : ℝ := 1.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters : ℝ := liters_today + liters_friday
def total_spend : ℝ := 39

theorem oil_price_rollback :
  let friday_price := (total_spend - current_price * liters_today) / liters_friday
  current_price - friday_price = 0.4 := by sorry

end oil_price_rollback_l2188_218872


namespace alicia_science_books_l2188_218883

/-- Represents the number of science books Alicia bought -/
def science_books : ℕ := sorry

/-- Represents the cost of a math book -/
def math_book_cost : ℕ := 3

/-- Represents the cost of a science book -/
def science_book_cost : ℕ := 3

/-- Represents the cost of an art book -/
def art_book_cost : ℕ := 2

/-- Represents the number of math books Alicia bought -/
def math_books : ℕ := 2

/-- Represents the number of art books Alicia bought -/
def art_books : ℕ := 3

/-- Represents the total cost of all books -/
def total_cost : ℕ := 30

/-- Theorem stating that Alicia bought 6 science books -/
theorem alicia_science_books : 
  math_books * math_book_cost + art_books * art_book_cost + science_books * science_book_cost = total_cost → 
  science_books = 6 := by
  sorry

end alicia_science_books_l2188_218883


namespace intersection_point_in_interval_l2188_218843

open Real

theorem intersection_point_in_interval (f g : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (∀ x, g x = 2^x + 1) →
  f x₀ = g x₀ →
  1 < x₀ ∧ x₀ < 2 := by
sorry

end intersection_point_in_interval_l2188_218843


namespace bucket_fill_theorem_l2188_218894

/-- Given two buckets P and Q, where P has thrice the capacity of Q,
    and P alone takes 60 turns to fill a drum, prove that P and Q together
    take 45 turns to fill the same drum. -/
theorem bucket_fill_theorem (p q : ℕ) (drum : ℕ) : 
  p = 3 * q →  -- Bucket P has thrice the capacity of bucket Q
  60 * p = drum →  -- It takes 60 turns for bucket P to fill the drum
  45 * (p + q) = drum :=  -- It takes 45 turns for both buckets to fill the drum
by sorry

end bucket_fill_theorem_l2188_218894


namespace fixed_point_of_function_l2188_218827

theorem fixed_point_of_function (n : ℤ) (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => x^n + a^(x-1)
  f 1 = 2 := by sorry

end fixed_point_of_function_l2188_218827


namespace probability_diamond_spade_heart_l2188_218802

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Nat)
  (diamonds : Nat)
  (spades : Nat)
  (hearts : Nat)

/-- Calculates the probability of drawing a specific sequence of cards -/
def probability_specific_sequence (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.cards *
  (d.spades : ℚ) / (d.cards - 1) *
  (d.hearts : ℚ) / (d.cards - 2)

/-- A standard deck of 52 cards with 13 cards of each suit -/
def standard_deck : Deck :=
  { cards := 52,
    diamonds := 13,
    spades := 13,
    hearts := 13 }

theorem probability_diamond_spade_heart :
  probability_specific_sequence standard_deck = 2197 / 132600 := by
  sorry

end probability_diamond_spade_heart_l2188_218802


namespace luke_fillets_l2188_218850

/-- Calculates the total number of fish fillets Luke has after fishing for a given number of days. -/
def total_fillets (fish_per_day : ℕ) (days : ℕ) (fillets_per_fish : ℕ) : ℕ :=
  fish_per_day * days * fillets_per_fish

/-- Proves that Luke has 120 fish fillets after fishing for 30 days. -/
theorem luke_fillets : total_fillets 2 30 2 = 120 := by
  sorry

end luke_fillets_l2188_218850


namespace max_area_rectangle_d_l2188_218836

/-- Given a rectangle divided into four smaller rectangles A, B, C, and D,
    where the perimeters of A, B, and C are known, 
    prove that the maximum possible area of rectangle D is 16 cm². -/
theorem max_area_rectangle_d (perim_A perim_B perim_C : ℝ) 
  (h_perim_A : perim_A = 10)
  (h_perim_B : perim_B = 12)
  (h_perim_C : perim_C = 14) :
  ∃ (area_D : ℝ), area_D ≤ 16 ∧ 
  ∀ (other_area : ℝ), (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*(a+b) = perim_B + perim_C - perim_A ∧ other_area = a*b) 
  → other_area ≤ area_D := by
  sorry

end max_area_rectangle_d_l2188_218836


namespace max_min_sum_equals_22_5_l2188_218867

/-- Given real numbers x, y, and z satisfying 5(x + y + z) = x^2 + y^2 + z^2,
    the maximum value of xy + xz + yz plus 5 times the minimum value of xy + xz + yz equals 22.5 -/
theorem max_min_sum_equals_22_5 :
  ∃ (N n : ℝ),
    (∀ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 →
      x * y + x * z + y * z ≤ N ∧
      n ≤ x * y + x * z + y * z) ∧
    (∃ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 ∧ x * y + x * z + y * z = N) ∧
    (∃ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 ∧ x * y + x * z + y * z = n) ∧
    N + 5 * n = 22.5 :=
by sorry

end max_min_sum_equals_22_5_l2188_218867


namespace circle_series_area_sum_l2188_218826

/-- The sum of the areas of an infinite series of circles, where the first circle has a radius of 2 inches
    and each subsequent circle's radius is half of the previous one, is equal to 16π/3. -/
theorem circle_series_area_sum : 
  let radius : ℕ → ℝ := fun n => 2 / (2 ^ (n - 1))
  let area : ℕ → ℝ := fun n => π * (radius n)^2
  (∑' n, area n) = 16 * π / 3 := by
  sorry

end circle_series_area_sum_l2188_218826


namespace conjugate_complex_modulus_l2188_218895

theorem conjugate_complex_modulus (α β : ℂ) :
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (α^2 / β).im = 0 →                              -- α²/β is real
  Complex.abs (α - β) = 4 →                       -- |α - β| = 4
  Complex.abs α = 4 * Real.sqrt 3 / 3 :=          -- |α| = 4√3/3
by sorry

end conjugate_complex_modulus_l2188_218895


namespace hyperbola_eccentricity_l2188_218878

-- Define the hyperbola C
def C (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 2

-- Define a point on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  C a b 2 3

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  C a b 2 3 → focal_distance c → c / a = 2 :=
by sorry

end hyperbola_eccentricity_l2188_218878


namespace maria_has_four_dimes_l2188_218880

/-- Represents the number of coins of each type in Maria's piggy bank -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value in cents given a CoinCount -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.quarters * 25 + coins.nickels * 5

/-- Theorem stating that Maria has 4 dimes -/
theorem maria_has_four_dimes :
  ∃ (initial : CoinCount),
    initial.quarters = 4 ∧
    initial.nickels = 7 ∧
    totalValue { dimes := initial.dimes,
                 quarters := initial.quarters + 5,
                 nickels := initial.nickels } = 300 ∧
    initial.dimes = 4 := by
  sorry

end maria_has_four_dimes_l2188_218880


namespace nineteen_percent_female_officers_on_duty_l2188_218803

/-- Calculates the percentage of female officers on duty given the total officers on duty,
    the fraction of officers on duty who are female, and the total number of female officers. -/
def percentage_female_officers_on_duty (total_on_duty : ℕ) (fraction_female : ℚ) (total_female : ℕ) : ℚ :=
  (fraction_female * total_on_duty : ℚ) / total_female * 100

/-- Theorem stating that 19% of female officers were on duty that night. -/
theorem nineteen_percent_female_officers_on_duty :
  percentage_female_officers_on_duty 152 (1/2) 400 = 19 := by
  sorry

end nineteen_percent_female_officers_on_duty_l2188_218803


namespace unique_solution_system_l2188_218862

theorem unique_solution_system : 
  ∃! (x y : ℝ), x + y = 3 ∧ x^4 - y^4 = 8*x - y ∧ x = 2 ∧ y = 1 := by
  sorry

end unique_solution_system_l2188_218862


namespace constant_product_l2188_218821

-- Define the circle and points
variable (Circle : Type) (A B C D : Point)
variable (diameter : Circle → Point → Point → Prop)
variable (tangent : Circle → Point → Prop)
variable (on_circle : Circle → Point → Prop)
variable (on_tangent : Circle → Point → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem constant_product 
  (circle : Circle)
  (h1 : diameter circle A B)
  (h2 : tangent circle B)
  (h3 : on_circle circle C)
  (h4 : on_tangent circle D)
  : distance A C * distance A D = distance A B * distance A B :=
sorry

end constant_product_l2188_218821


namespace peach_distribution_l2188_218811

/-- Proves that given 60 peaches distributed among two equal-sized containers and one smaller container,
    where the smaller container holds half as many peaches as each of the equal-sized containers,
    the number of peaches in the smaller container is 12. -/
theorem peach_distribution (total_peaches : ℕ) (cloth_bag : ℕ) (knapsack : ℕ) : 
  total_peaches = 60 →
  2 * cloth_bag + knapsack = total_peaches →
  knapsack = cloth_bag / 2 →
  knapsack = 12 := by
sorry

end peach_distribution_l2188_218811


namespace min_value_of_expression_min_value_achievable_l2188_218852

theorem min_value_of_expression (x y : ℝ) : (x * y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end min_value_of_expression_min_value_achievable_l2188_218852


namespace simplify_expression_l2188_218809

theorem simplify_expression : (625 : ℝ) ^ (1/4) * (400 : ℝ) ^ (1/2) = 100 := by sorry

end simplify_expression_l2188_218809


namespace line_AB_not_through_point_B_l2188_218818

-- Define the circles C and M
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the condition that (a, b) is on circle C
def M_on_C (a b : ℝ) : Prop := circle_C a b

-- Define the line AB
def line_AB (a b x y : ℝ) : Prop := (2*a - 2)*x + 2*b*y - 3 = 0

-- Theorem statement
theorem line_AB_not_through_point_B (a b : ℝ) (h : M_on_C a b) :
  ¬ line_AB a b (1/2) (1/2) :=
sorry

end line_AB_not_through_point_B_l2188_218818


namespace pages_per_day_l2188_218891

/-- Given a book with 576 pages read over 72 days, prove that the number of pages read per day is 8 -/
theorem pages_per_day (total_pages : ℕ) (total_days : ℕ) (h1 : total_pages = 576) (h2 : total_days = 72) :
  total_pages / total_days = 8 := by
  sorry

end pages_per_day_l2188_218891


namespace stating_professor_seating_arrangements_l2188_218823

/-- Represents the number of chairs in a row -/
def num_chairs : ℕ := 10

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the effective number of chair positions professors can choose from -/
def effective_chairs : ℕ := 4

/-- 
Theorem stating that the number of ways professors can choose their chairs
under the given conditions is 24.
-/
theorem professor_seating_arrangements :
  (effective_chairs.choose num_professors) * num_professors.factorial = 24 :=
by sorry

end stating_professor_seating_arrangements_l2188_218823


namespace square_roots_problem_l2188_218857

theorem square_roots_problem (x a : ℝ) (hx : x > 0) :
  ((-a + 2)^2 = x ∧ (2*a - 1)^2 = x) → (a = -1 ∧ x = 9) := by
sorry

end square_roots_problem_l2188_218857


namespace hcf_of_three_numbers_l2188_218840

theorem hcf_of_three_numbers (a b c : ℕ+) : 
  (a + b + c : ℝ) = 60 →
  Nat.lcm (a : ℕ) (Nat.lcm b c) = 180 →
  (1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ)) = 11 / 120 →
  (a * b * c : ℕ) = 900 →
  Nat.gcd (a : ℕ) (Nat.gcd b c) = 5 := by
sorry


end hcf_of_three_numbers_l2188_218840


namespace symmetric_complex_product_l2188_218863

theorem symmetric_complex_product (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 1) → 
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) → 
  z₁ * z₂ = -2 := by
  sorry

end symmetric_complex_product_l2188_218863


namespace no_solutions_for_star_equation_l2188_218876

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

-- Theorem statement
theorem no_solutions_for_star_equation :
  ¬ ∃ y : ℝ, star 2 y = 20 := by
  sorry

end no_solutions_for_star_equation_l2188_218876


namespace truncated_hexahedron_property_l2188_218815

-- Define the structure of our polyhedron
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  H : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces

-- Define the properties of our specific polyhedron
def truncated_hexahedron : Polyhedron where
  V := 20
  E := 36
  F := 18
  H := 6
  T := 12

-- Theorem statement
theorem truncated_hexahedron_property (p : Polyhedron) 
  (euler : p.V - p.E + p.F = 2)
  (faces : p.F = 18)
  (hex_tri : p.H + p.T = p.F)
  (vertex_config : 2 * p.V = 3 * p.T + 6 * p.H) :
  100 * 2 + 10 * 2 + p.V = 240 := by
  sorry

#check truncated_hexahedron_property

end truncated_hexahedron_property_l2188_218815


namespace cubic_root_sum_squares_l2188_218873

/-- Given a cubic polynomial x^3 - 3x - 2 = 0 with roots a, b, and c,
    prove that a(b + c)^2 + b(c + a)^2 + c(a + b)^2 = 6 -/
theorem cubic_root_sum_squares (a b c : ℝ) : 
  a^3 - 3*a - 2 = 0 → 
  b^3 - 3*b - 2 = 0 → 
  c^3 - 3*c - 2 = 0 → 
  a*(b + c)^2 + b*(c + a)^2 + c*(a + b)^2 = 6 := by
  sorry

end cubic_root_sum_squares_l2188_218873


namespace quadratic_rational_root_parity_l2188_218865

theorem quadratic_rational_root_parity (a b c : ℤ) (x : ℚ) : 
  a ≠ 0 → 
  a * x^2 + b * x + c = 0 → 
  ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end quadratic_rational_root_parity_l2188_218865


namespace constant_covered_area_l2188_218890

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ
  center : ℝ × ℝ

/-- Represents the configuration of two squares as described in the problem -/
structure TwoSquaresConfig where
  bottom_square : Square
  top_square : Square
  rotation_angle : ℝ

/-- Calculates the total area covered by two squares in the given configuration -/
noncomputable def total_covered_area (config : TwoSquaresConfig) : ℝ :=
  sorry

/-- Theorem: The total covered area is constant regardless of the rotation angle -/
theorem constant_covered_area
  (bottom_square : Square)
  (top_square : Square)
  (h_identical : bottom_square.side_length = top_square.side_length)
  (h_diagonal_intersection : top_square.center = (bottom_square.center.1 + bottom_square.side_length / 2, bottom_square.center.2 + bottom_square.side_length / 2)) :
  ∀ θ₁ θ₂ : ℝ,
    total_covered_area { bottom_square := bottom_square, top_square := top_square, rotation_angle := θ₁ } =
    total_covered_area { bottom_square := bottom_square, top_square := top_square, rotation_angle := θ₂ } :=
  sorry

end constant_covered_area_l2188_218890


namespace difference_105th_100th_term_l2188_218810

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem difference_105th_100th_term :
  let a₁ := 3
  let d := 5
  (arithmeticSequence a₁ d 105) - (arithmeticSequence a₁ d 100) = 25 := by
  sorry

end difference_105th_100th_term_l2188_218810


namespace sales_tax_percentage_l2188_218856

theorem sales_tax_percentage (total_allowed : ℝ) (food_cost : ℝ) (tip_percentage : ℝ) :
  total_allowed = 75 →
  food_cost = 61.48 →
  tip_percentage = 15 →
  ∃ (sales_tax_percentage : ℝ),
    sales_tax_percentage ≤ 6.95 ∧
    food_cost * (1 + sales_tax_percentage / 100 + tip_percentage / 100) ≤ total_allowed :=
by sorry

end sales_tax_percentage_l2188_218856


namespace num_cars_in_parking_lot_l2188_218804

def num_bikes : ℕ := 10
def total_wheels : ℕ := 76
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem num_cars_in_parking_lot : 
  (total_wheels - num_bikes * wheels_per_bike) / wheels_per_car = 14 := by
  sorry

end num_cars_in_parking_lot_l2188_218804


namespace gerbil_weight_difference_gerbil_weight_difference_proof_l2188_218866

/-- The weight difference between Scruffy and Muffy given the conditions of the gerbil problem -/
theorem gerbil_weight_difference : ℝ → Prop :=
  fun weight_difference =>
    ∃ (muffy_weight : ℝ),
      let puffy_weight := muffy_weight + 5
      let scruffy_weight := 12
      puffy_weight + muffy_weight = 23 ∧
      weight_difference = scruffy_weight - muffy_weight ∧
      weight_difference = 3

/-- Proof of the gerbil weight difference theorem -/
theorem gerbil_weight_difference_proof : gerbil_weight_difference 3 := by
  sorry

end gerbil_weight_difference_gerbil_weight_difference_proof_l2188_218866


namespace A_intersect_B_eq_open_interval_l2188_218861

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define set B
def B : Set ℝ := {x | Real.rpow 2022 x > Real.sqrt 2022}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo (1/2 : ℝ) 6 := by sorry

end A_intersect_B_eq_open_interval_l2188_218861


namespace bob_cleaning_time_l2188_218875

def alice_time : ℝ := 30

theorem bob_cleaning_time :
  let bob_time := (3 / 4 : ℝ) * alice_time
  bob_time = 22.5 := by sorry

end bob_cleaning_time_l2188_218875


namespace remaining_income_calculation_l2188_218814

def remaining_income (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ) 
  (utilities_percent : ℝ) (transportation_percent : ℝ) (insurance_percent : ℝ) 
  (emergency_fund_percent : ℝ) : ℝ :=
  let initial_remaining := 1 - (food_percent + education_percent + transportation_percent)
  let rent_amount := rent_percent * initial_remaining
  let post_rent_remaining := initial_remaining - rent_amount
  let utilities_amount := utilities_percent * rent_amount
  let post_utilities_remaining := post_rent_remaining - utilities_amount
  let insurance_amount := insurance_percent * post_utilities_remaining
  let pre_emergency_remaining := post_utilities_remaining - insurance_amount
  let emergency_fund_amount := emergency_fund_percent * pre_emergency_remaining
  pre_emergency_remaining - emergency_fund_amount

theorem remaining_income_calculation :
  remaining_income 0.42 0.18 0.30 0.25 0.12 0.15 0.06 = 0.139825 := by
  sorry

#eval remaining_income 0.42 0.18 0.30 0.25 0.12 0.15 0.06

end remaining_income_calculation_l2188_218814


namespace unit_square_max_distance_l2188_218817

theorem unit_square_max_distance (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  min (min (min (Real.sqrt ((x - 0)^2 + (y - 0)^2))
                (Real.sqrt ((x - 1)^2 + (y - 0)^2)))
           (Real.sqrt ((x - 1)^2 + (y - 1)^2)))
      (Real.sqrt ((x - 0)^2 + (y - 1)^2))
  ≤ Real.sqrt 5 / 2 := by
sorry

end unit_square_max_distance_l2188_218817


namespace infinite_primes_dividing_polynomial_values_l2188_218822

/-- A polynomial with integer coefficients -/
def IntPolynomial := List ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ :=
  (p.length - 1).max 0

/-- Evaluate a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  p.enum.foldl (fun acc (i, a) => acc + a * x ^ i) 0

theorem infinite_primes_dividing_polynomial_values (p : IntPolynomial)
  (h : degree p ≥ 1) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    (∀ q ∈ S, Prime q ∧ ∃ n : ℕ, (eval p n) % q = 0) :=
  sorry

end infinite_primes_dividing_polynomial_values_l2188_218822


namespace factorization_theorem_l2188_218800

theorem factorization_theorem (x y : ℝ) :
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := by
  sorry

end factorization_theorem_l2188_218800


namespace inequality_equivalence_l2188_218882

theorem inequality_equivalence (x : ℝ) :
  (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end inequality_equivalence_l2188_218882


namespace smallest_common_multiple_of_8_and_6_l2188_218851

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 ∧ 8 ∣ m ∧ 6 ∣ m → n ≤ m :=
by
  use 24
  sorry

end smallest_common_multiple_of_8_and_6_l2188_218851


namespace base8_digit_product_8654_l2188_218834

/-- Convert a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8654₁₀ is 0 --/
theorem base8_digit_product_8654 :
  listProduct (toBase8 8654) = 0 :=
sorry

end base8_digit_product_8654_l2188_218834


namespace parabola_directrix_l2188_218885

/-- Given a parabola y² = 2px where p > 0, if a point M(1, m) on the parabola
    is at a distance of 5 from the focus, then the directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) (h1 : p > 0) (h2 : m^2 = 2*p) 
  (h3 : (1 - p/2)^2 + m^2 = 5^2) : 
  ∃ (x : ℝ), x = -4 ∧ ∀ (y : ℝ), (x + p/2)^2 = (1 - x)^2 + m^2 := by
  sorry

end parabola_directrix_l2188_218885


namespace max_salary_proof_l2188_218816

/-- The number of players in a team -/
def team_size : ℕ := 25

/-- The minimum salary for a player -/
def min_salary : ℕ := 15000

/-- The total salary cap for a team -/
def salary_cap : ℕ := 850000

/-- The maximum possible salary for a single player -/
def max_player_salary : ℕ := 490000

theorem max_salary_proof :
  (team_size - 1) * min_salary + max_player_salary = salary_cap ∧
  ∀ (x : ℕ), x > max_player_salary →
    (team_size - 1) * min_salary + x > salary_cap :=
by sorry

end max_salary_proof_l2188_218816


namespace remaining_area_in_square_l2188_218813

theorem remaining_area_in_square : 
  let large_square_side : ℝ := 3.5
  let small_square_side : ℝ := 2
  let rectangle_length : ℝ := 2
  let rectangle_width : ℝ := 1.5
  let triangle_leg : ℝ := 1
  let large_square_area := large_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let rectangle_area := rectangle_length * rectangle_width
  let triangle_area := 0.5 * triangle_leg * triangle_leg
  let occupied_area := small_square_area + rectangle_area + triangle_area
  large_square_area - occupied_area = 4.75 := by
sorry

end remaining_area_in_square_l2188_218813


namespace path_count_l2188_218835

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

/-- The set of points in the problem -/
inductive Point
| A
| B
| C
| D

/-- The total number of paths from A to C -/
def total_paths : ℕ := sorry

theorem path_count :
  (num_paths Point.A Point.B = 2) →
  (num_paths Point.B Point.D = 2) →
  (num_paths Point.D Point.C = 2) →
  (num_paths Point.A Point.C = 1 + num_paths Point.A Point.B * num_paths Point.B Point.D * num_paths Point.D Point.C) →
  (total_paths = 9) :=
by sorry

end path_count_l2188_218835


namespace water_tank_capacity_l2188_218879

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) : 
  (c / 3 : ℝ) / c = 1 / 3 ∧ 
  (c / 3 + 5 : ℝ) / c = 1 / 2 → 
  c = 30 := by
sorry

end water_tank_capacity_l2188_218879


namespace sin_sum_angles_l2188_218847

/-- Given a point A(1, 2) on the terminal side of angle α in the Cartesian plane,
    and angle β formed by rotating α's terminal side counterclockwise by π/2,
    prove that sin(α + β) = -3/5 -/
theorem sin_sum_angles (α β : Real) : 
  (∃ A : ℝ × ℝ, A = (1, 2) ∧ A.1 = Real.cos α * Real.sqrt (A.1^2 + A.2^2) ∧ 
                   A.2 = Real.sin α * Real.sqrt (A.1^2 + A.2^2)) →
  β = α + π/2 →
  Real.sin (α + β) = -3/5 := by
  sorry

end sin_sum_angles_l2188_218847


namespace original_mango_price_l2188_218844

/-- Represents the price increase rate -/
def price_increase_rate : ℝ := 0.15

/-- Represents the original price of an orange -/
def original_orange_price : ℝ := 40

/-- Represents the total cost of 10 oranges and 10 mangoes after price increase -/
def total_cost : ℝ := 1035

/-- Represents the quantity of each fruit -/
def quantity : ℕ := 10

/-- Calculates the new price after applying the price increase -/
def new_price (original_price : ℝ) : ℝ :=
  original_price * (1 + price_increase_rate)

/-- Theorem stating that the original price of a mango was $50 -/
theorem original_mango_price :
  ∃ (original_mango_price : ℝ),
    original_mango_price = 50 ∧
    (quantity : ℝ) * new_price original_orange_price +
    (quantity : ℝ) * new_price original_mango_price = total_cost := by
  sorry

end original_mango_price_l2188_218844


namespace smallest_x_value_exists_solution_l2188_218819

theorem smallest_x_value (x : ℝ) : 
  ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 → x ≥ 0 :=
by sorry

theorem exists_solution : 
  ∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 ∧ x = 0 :=
by sorry

end smallest_x_value_exists_solution_l2188_218819


namespace addition_and_subtraction_proof_l2188_218825

theorem addition_and_subtraction_proof :
  (1 + (-11) = -10) ∧ (0 - 4.5 = -4.5) := by sorry

end addition_and_subtraction_proof_l2188_218825


namespace f_1_equals_5_l2188_218832

-- Define the quadratic polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom quad_f : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
axiom quad_g : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c
axiom f_2_3 : f 2 = 2 ∧ f 3 = 2
axiom g_2_3 : g 2 = 2 ∧ g 3 = 2
axiom g_1 : g 1 = 2
axiom f_5 : f 5 = 7
axiom g_5 : g 5 = 2

-- State the theorem
theorem f_1_equals_5 : f 1 = 5 := by sorry

end f_1_equals_5_l2188_218832


namespace fraction_reducibility_implies_determinant_divisibility_l2188_218892

theorem fraction_reducibility_implies_determinant_divisibility
  (a b c d l k : ℤ) 
  (h : ∃ (m n : ℤ), a * l + b = k * m ∧ c * l + d = k * n) :
  k ∣ (a * d - b * c) := by
  sorry

end fraction_reducibility_implies_determinant_divisibility_l2188_218892


namespace complement_intersection_theorem_l2188_218877

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {-2, 2} := by sorry

end complement_intersection_theorem_l2188_218877


namespace red_bellied_minnows_count_l2188_218837

/-- Represents the number of minnows in a pond with different belly colors. -/
structure MinnowPond where
  total : ℕ
  red_percent : ℚ
  green_percent : ℚ
  white_count : ℕ

/-- Theorem stating the number of red-bellied minnows in the pond. -/
theorem red_bellied_minnows_count (pond : MinnowPond)
  (h1 : pond.red_percent = 2/5)
  (h2 : pond.green_percent = 3/10)
  (h3 : pond.white_count = 15)
  (h4 : pond.total * (1 - pond.red_percent - pond.green_percent) = pond.white_count) :
  pond.total * pond.red_percent = 20 := by
  sorry

end red_bellied_minnows_count_l2188_218837


namespace square_of_999999999_has_8_zeros_l2188_218833

theorem square_of_999999999_has_8_zeros :
  let n : ℕ := 999999999
  ∃ m : ℕ, n^2 = m * 10^8 ∧ m % 10 ≠ 0 ∧ m ≥ 10^9 ∧ m < 10^10 :=
by sorry

end square_of_999999999_has_8_zeros_l2188_218833


namespace unique_rectangle_pieces_l2188_218801

theorem unique_rectangle_pieces :
  ∀ (a b : ℕ),
    a < b →
    (49 * 51) % (a * b) = 0 →
    (99 * 101) % (a * b) = 0 →
    a = 1 ∧ b = 3 := by
  sorry

end unique_rectangle_pieces_l2188_218801


namespace money_sharing_problem_l2188_218881

/-- Represents the ratio of money shared among three people -/
structure MoneyRatio :=
  (a b c : ℕ)

/-- Calculates the total amount of money given a ratio and the first person's share -/
def totalAmount (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  firstShare * (ratio.a + ratio.b + ratio.c)

/-- Theorem: Given a money ratio of 1:2:7 and the first person's share of $20, 
    the total amount shared is $200 -/
theorem money_sharing_problem (ratio : MoneyRatio) (firstShare : ℕ) :
  ratio.a = 1 → ratio.b = 2 → ratio.c = 7 → firstShare = 20 →
  totalAmount ratio firstShare = 200 := by
  sorry

#eval totalAmount ⟨1, 2, 7⟩ 20

end money_sharing_problem_l2188_218881


namespace stratified_sample_sum_l2188_218888

/-- Represents the number of items in each category -/
def categories : List ℕ := [40, 10, 30, 20]

/-- Total number of items -/
def total : ℕ := categories.sum

/-- Sample size -/
def sample_size : ℕ := 20

/-- Calculates the number of items sampled from a category -/
def sampled_items (category_size : ℕ) : ℕ :=
  (category_size * sample_size) / total

/-- Theorem stating that the sum of sampled items from categories with 10 and 20 items is 6 -/
theorem stratified_sample_sum :
  sampled_items (categories[1]) + sampled_items (categories[3]) = 6 := by
  sorry

end stratified_sample_sum_l2188_218888


namespace traffic_class_total_l2188_218831

/-- The number of drunk drivers in the traffic class -/
def drunk_drivers : ℕ := 6

/-- The number of speeders in the traffic class -/
def speeders : ℕ := 7 * drunk_drivers - 3

/-- The total number of students in the traffic class -/
def total_students : ℕ := drunk_drivers + speeders

/-- Theorem stating that the total number of students in the traffic class is 45 -/
theorem traffic_class_total : total_students = 45 := by sorry

end traffic_class_total_l2188_218831


namespace necessary_but_not_sufficient_l2188_218899

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, (x - 1) * (x - 2) ≤ 0 → x^2 - 3*x ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 3*x ≤ 0 ∧ (x - 1) * (x - 2) > 0) := by
  sorry

end necessary_but_not_sufficient_l2188_218899


namespace right_triangle_from_angle_condition_l2188_218855

theorem right_triangle_from_angle_condition (A B C : Real) :
  -- Triangle condition
  A + B + C = 180 →
  -- Given angle condition
  A = B ∧ A = (1/2) * C →
  -- Conclusion: C is a right angle
  C = 90 := by
  sorry

end right_triangle_from_angle_condition_l2188_218855


namespace basketball_conference_games_l2188_218884

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 2

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem basketball_conference_games :
  total_games = 155 := by sorry

end basketball_conference_games_l2188_218884


namespace z_in_third_quadrant_l2188_218889

def complex_number_quadrant (z : ℂ) : Prop :=
  Real.sign z.re = -1 ∧ Real.sign z.im = -1

theorem z_in_third_quadrant :
  let z : ℂ := (-2 - Complex.I) * (3 + Complex.I)
  complex_number_quadrant z := by
  sorry

end z_in_third_quadrant_l2188_218889


namespace sqrt_a_plus_one_range_l2188_218841

theorem sqrt_a_plus_one_range :
  ∀ a : ℝ, (∃ x : ℝ, x^2 = a + 1) ↔ a ≥ -1 := by
sorry

end sqrt_a_plus_one_range_l2188_218841


namespace cafe_order_combinations_l2188_218805

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- The number of distinct meal combinations for two people choosing from a menu with a given number of items, where order matters and repetition is allowed -/
def meal_combinations (items : ℕ) : ℕ := items ^ num_people

theorem cafe_order_combinations :
  meal_combinations menu_items = 225 := by
  sorry

end cafe_order_combinations_l2188_218805


namespace intersection_distance_squared_l2188_218858

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_distance_squared :
  ∀ (A B M : ℝ × ℝ),
    curve_C A.1 A.2 →
    curve_C B.1 B.2 →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    line_l M.1 M.2 →
    y_axis M.1 →
    (Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) + Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2))^2 = 16 + 2 * Real.sqrt 3 :=
by sorry

end intersection_distance_squared_l2188_218858


namespace equation_solution_l2188_218898

theorem equation_solution :
  ∃! y : ℚ, y + 4/5 = 2/3 + y/6 :=
by
  -- The proof goes here
  sorry

end equation_solution_l2188_218898
