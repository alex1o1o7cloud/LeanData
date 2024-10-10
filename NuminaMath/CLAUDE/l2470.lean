import Mathlib

namespace equidistant_points_of_f_l2470_247064

/-- A point (x, y) is equidistant if |x| = |y| -/
def is_equidistant (x y : ℝ) : Prop := abs x = abs y

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem: The points (0,0), (-1,-1), and (-3,3) are equidistant points of f -/
theorem equidistant_points_of_f :
  is_equidistant 0 (f 0) ∧
  is_equidistant (-1) (f (-1)) ∧
  is_equidistant (-3) (f (-3)) :=
sorry

end equidistant_points_of_f_l2470_247064


namespace alex_jellybean_possibilities_l2470_247035

def total_money : ℕ := 100  -- in pence

def toffee_price : ℕ := 5
def bubblegum_price : ℕ := 3
def jellybean_price : ℕ := 2

def min_toffee_spend : ℕ := 35  -- ⌈100 / 3⌉ rounded up to nearest multiple of 5
def min_bubblegum_spend : ℕ := 27  -- ⌈100 / 4⌉ rounded up to nearest multiple of 3
def min_jellybean_spend : ℕ := 10  -- 100 / 10

def possible_jellybean_counts : Set ℕ := {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19}

theorem alex_jellybean_possibilities :
  ∀ n : ℕ, n ∈ possible_jellybean_counts ↔
    ∃ (t b j : ℕ),
      t * toffee_price + b * bubblegum_price + n * jellybean_price = total_money ∧
      t * toffee_price ≥ min_toffee_spend ∧
      b * bubblegum_price ≥ min_bubblegum_spend ∧
      n * jellybean_price ≥ min_jellybean_spend :=
by sorry

end alex_jellybean_possibilities_l2470_247035


namespace ping_pong_rackets_sold_l2470_247046

/-- The number of pairs of ping pong rackets sold -/
def num_pairs : ℕ := 55

/-- The total amount made from selling rackets in dollars -/
def total_amount : ℚ := 539

/-- The average price of a pair of rackets in dollars -/
def avg_price : ℚ := 9.8

/-- Theorem: The number of pairs of ping pong rackets sold is 55 -/
theorem ping_pong_rackets_sold :
  (total_amount / avg_price : ℚ) = num_pairs := by sorry

end ping_pong_rackets_sold_l2470_247046


namespace train_distance_problem_l2470_247091

theorem train_distance_problem :
  let fast_train_time : ℝ := 5
  let slow_train_time : ℝ := fast_train_time * (1 + 1/5)
  let stop_time : ℝ := 2
  let additional_distance : ℝ := 40
  let distance : ℝ := 150
  let fast_train_speed : ℝ := distance / fast_train_time
  let slow_train_speed : ℝ := distance / slow_train_time
  let fast_train_distance : ℝ := fast_train_speed * stop_time
  let slow_train_distance : ℝ := slow_train_speed * stop_time
  let remaining_distance : ℝ := distance - (fast_train_distance + slow_train_distance)
  remaining_distance = additional_distance :=
by
  sorry

#check train_distance_problem

end train_distance_problem_l2470_247091


namespace total_tickets_is_56_l2470_247004

/-- The total number of tickets spent during three trips to the arcade -/
def total_tickets : ℕ :=
  let first_trip := 2 + 10 + 2
  let second_trip := 3 + 7 + 5
  let third_trip := 8 + 15 + 4
  first_trip + second_trip + third_trip

/-- Theorem stating that the total number of tickets spent is 56 -/
theorem total_tickets_is_56 : total_tickets = 56 := by
  sorry

end total_tickets_is_56_l2470_247004


namespace two_distinct_real_roots_l2470_247079

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + x - 2 = m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ :=
  4 * m + 9

-- Theorem statement
theorem two_distinct_real_roots (m : ℝ) (h : m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m :=
sorry

end two_distinct_real_roots_l2470_247079


namespace earphone_cost_l2470_247015

/-- The cost of an earphone given weekly expenditure data -/
theorem earphone_cost (mean_expenditure : ℕ) (mon tue wed thu sat sun : ℕ) (pen notebook : ℕ) :
  mean_expenditure = 500 →
  mon = 450 →
  tue = 600 →
  wed = 400 →
  thu = 500 →
  sat = 550 →
  sun = 300 →
  pen = 30 →
  notebook = 50 →
  ∃ (earphone : ℕ), earphone = 7 * mean_expenditure - (mon + tue + wed + thu + sat + sun) - pen - notebook :=
by
  sorry

end earphone_cost_l2470_247015


namespace persimmons_count_l2470_247052

theorem persimmons_count (tangerines : ℕ) (total : ℕ) (h1 : tangerines = 19) (h2 : total = 37) :
  total - tangerines = 18 := by
  sorry

end persimmons_count_l2470_247052


namespace range_of_a_l2470_247033

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (∀ x, p x ↔ (3*x - 1)/(x - 2) ≤ 1) →
  (∀ x, q x ↔ x^2 - (2*a + 1)*x + a*(a + 1) < 0) →
  (∀ x, ¬(q x) → ¬(p x)) →
  (∃ x, ¬(q x) ∧ p x) →
  -1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l2470_247033


namespace race_distance_l2470_247096

/-- The race distance in meters -/
def d : ℝ := 75

/-- The speed of runner X -/
def x : ℝ := sorry

/-- The speed of runner Y -/
def y : ℝ := sorry

/-- The speed of runner Z -/
def z : ℝ := sorry

/-- Theorem stating that d is the correct race distance -/
theorem race_distance : 
  (d / x = (d - 25) / y) ∧ 
  (d / y = (d - 15) / z) ∧ 
  (d / x = (d - 35) / z) → 
  d = 75 := by sorry

end race_distance_l2470_247096


namespace base_conversion_l2470_247030

theorem base_conversion (b : ℝ) : 
  b > 0 ∧ (5 * 6 + 4 = 1 * b^2 + 2 * b + 1) → b = -1 + Real.sqrt 34 := by
  sorry

end base_conversion_l2470_247030


namespace polynomial_derivative_sum_l2470_247067

theorem polynomial_derivative_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end polynomial_derivative_sum_l2470_247067


namespace intersection_complement_equality_l2470_247042

open Set

def U : Set ℝ := univ
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {3, 4} := by sorry

end intersection_complement_equality_l2470_247042


namespace double_perimeter_polygon_exists_l2470_247048

/-- A grid point in 2D space --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A line segment on the grid --/
inductive GridSegment
  | Horizontal : GridPoint → GridPoint → GridSegment
  | Vertical : GridPoint → GridPoint → GridSegment
  | Diagonal1x1 : GridPoint → GridPoint → GridSegment
  | Diagonal1x2 : GridPoint → GridPoint → GridSegment

/-- A polygon on the grid --/
structure GridPolygon where
  vertices : List GridPoint
  edges : List GridSegment

/-- A triangle on the grid --/
structure GridTriangle where
  vertices : Fin 3 → GridPoint
  edges : Fin 3 → GridSegment

/-- Calculate the perimeter of a grid polygon --/
def perimeterOfPolygon (p : GridPolygon) : ℕ :=
  sorry

/-- Calculate the perimeter of a grid triangle --/
def perimeterOfTriangle (t : GridTriangle) : ℕ :=
  sorry

/-- Check if a polygon has double the perimeter of a triangle --/
def hasDoublePerimeter (p : GridPolygon) (t : GridTriangle) : Prop :=
  perimeterOfPolygon p = 2 * perimeterOfTriangle t

/-- Main theorem: Given a triangle on a grid, there exists a polygon with double its perimeter --/
theorem double_perimeter_polygon_exists (t : GridTriangle) : 
  ∃ (p : GridPolygon), hasDoublePerimeter p t :=
sorry

end double_perimeter_polygon_exists_l2470_247048


namespace condition_necessary_not_sufficient_l2470_247037

/-- Function f(x) = ax² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

/-- Condition: a ≥ 4 or a ≤ 0 -/
def condition (a : ℝ) : Prop := a ≥ 4 ∨ a ≤ 0

/-- f has zero points -/
def has_zero_points (a : ℝ) : Prop := ∃ x : ℝ, f a x = 0

theorem condition_necessary_not_sufficient :
  (∀ a : ℝ, has_zero_points a → condition a) ∧
  (∃ a : ℝ, condition a ∧ ¬has_zero_points a) :=
sorry

end condition_necessary_not_sufficient_l2470_247037


namespace quadratic_two_real_roots_l2470_247094

/-- 
Given a quadratic equation (m-1)x^2 - 2mx + m + 3 = 0,
prove that it has two real roots if and only if m ≤ 3/2 and m ≠ 1.
-/
theorem quadratic_two_real_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    (m - 1) * x^2 - 2 * m * x + m + 3 = 0 ∧ 
    (m - 1) * y^2 - 2 * m * y + m + 3 = 0) ↔ 
  (m ≤ 3/2 ∧ m ≠ 1) :=
sorry

end quadratic_two_real_roots_l2470_247094


namespace seventeen_in_base_three_l2470_247008

/-- Represents a number in base 3 as a list of digits (least significant digit first) -/
def BaseThreeRepresentation := List Nat

/-- Converts a base 3 representation to its decimal value -/
def toDecimal (rep : BaseThreeRepresentation) : Nat :=
  rep.enum.foldl (fun acc (i, digit) => acc + digit * (3^i)) 0

/-- Theorem: The base-3 representation of 17 is [2, 2, 1] (which represents 122₃) -/
theorem seventeen_in_base_three :
  ∃ (rep : BaseThreeRepresentation), toDecimal rep = 17 ∧ rep = [2, 2, 1] := by
  sorry

end seventeen_in_base_three_l2470_247008


namespace quadratic_completing_square_l2470_247028

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
sorry

end quadratic_completing_square_l2470_247028


namespace birds_on_fence_l2470_247075

theorem birds_on_fence : 
  let initial_sparrows : ℕ := 4
  let initial_storks : ℕ := 46
  let pigeons_joined : ℕ := 6
  let sparrows_left : ℕ := 3
  let storks_left : ℕ := 5
  let swans_came : ℕ := 8
  let ducks_came : ℕ := 2
  
  let total_birds : ℕ := 
    (initial_sparrows + initial_storks + pigeons_joined - sparrows_left - storks_left + swans_came + ducks_came)
  
  total_birds = 58 := by
  sorry

end birds_on_fence_l2470_247075


namespace president_and_vice_captain_selection_l2470_247058

/-- The number of people to choose from -/
def n : ℕ := 5

/-- The number of positions to fill -/
def k : ℕ := 2

/-- Theorem: The number of ways to select a class president and a vice-captain 
    from a group of n people, where one person cannot hold both positions, 
    is equal to n * (n - 1) -/
theorem president_and_vice_captain_selection : n * (n - 1) = 20 := by
  sorry

end president_and_vice_captain_selection_l2470_247058


namespace box_surface_area_l2470_247003

/-- Proves that a rectangular box with given edge sum and diagonal length has a specific surface area -/
theorem box_surface_area (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 168) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 1139 := by
  sorry

end box_surface_area_l2470_247003


namespace oliver_dish_count_l2470_247041

/-- Represents the buffet and Oliver's preferences -/
structure Buffet where
  total_dishes : ℕ
  mango_salsa_dishes : ℕ
  fresh_mango_dishes : ℕ
  mango_jelly_dishes : ℕ
  strawberry_dishes : ℕ
  pineapple_dishes : ℕ
  mango_dishes_oliver_can_eat : ℕ

/-- Calculates the number of dishes Oliver can eat -/
def dishes_for_oliver (b : Buffet) : ℕ :=
  b.total_dishes -
  (b.mango_salsa_dishes + b.fresh_mango_dishes + b.mango_jelly_dishes - b.mango_dishes_oliver_can_eat) -
  min b.strawberry_dishes b.pineapple_dishes

/-- Theorem stating the number of dishes Oliver can eat -/
theorem oliver_dish_count (b : Buffet) : dishes_for_oliver b = 28 :=
  by
    have h1 : b.total_dishes = 42 := by sorry
    have h2 : b.mango_salsa_dishes = 5 := by sorry
    have h3 : b.fresh_mango_dishes = 7 := by sorry
    have h4 : b.mango_jelly_dishes = 2 := by sorry
    have h5 : b.strawberry_dishes = 3 := by sorry
    have h6 : b.pineapple_dishes = 5 := by sorry
    have h7 : b.mango_dishes_oliver_can_eat = 3 := by sorry
    sorry

#eval dishes_for_oliver {
  total_dishes := 42,
  mango_salsa_dishes := 5,
  fresh_mango_dishes := 7,
  mango_jelly_dishes := 2,
  strawberry_dishes := 3,
  pineapple_dishes := 5,
  mango_dishes_oliver_can_eat := 3
}

end oliver_dish_count_l2470_247041


namespace existence_of_xy_for_function_l2470_247040

open Set

theorem existence_of_xy_for_function (f : ℝ → ℝ) 
  (hf : ∀ x, x > 0 → f x > 0) : 
  ∃ x y, x > 0 ∧ y > 0 ∧ f (x + y) < y * f (f x) := by
  sorry

end existence_of_xy_for_function_l2470_247040


namespace frank_five_dollar_bills_l2470_247082

def peanut_cost_per_pound : ℕ := 3
def days_in_week : ℕ := 7
def pounds_per_day : ℕ := 3
def one_dollar_bills : ℕ := 7
def ten_dollar_bills : ℕ := 2
def twenty_dollar_bills : ℕ := 1
def change_amount : ℕ := 4

def total_without_fives : ℕ := one_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills

def total_pounds_needed : ℕ := days_in_week * pounds_per_day

theorem frank_five_dollar_bills :
  ∃ (five_dollar_bills : ℕ),
    total_without_fives + 5 * five_dollar_bills - change_amount = peanut_cost_per_pound * total_pounds_needed ∧
    five_dollar_bills = 4 := by
  sorry

end frank_five_dollar_bills_l2470_247082


namespace adam_wall_area_l2470_247002

/-- The total area of four rectangular walls with given dimensions -/
def totalWallArea (w1_width w1_height w2_width w2_height w3_width w3_height w4_width w4_height : ℝ) : ℝ :=
  w1_width * w1_height + w2_width * w2_height + w3_width * w3_height + w4_width * w4_height

/-- Theorem: The total area of the walls with the given dimensions is 160 square feet -/
theorem adam_wall_area :
  totalWallArea 4 8 6 8 4 8 6 8 = 160 := by
  sorry

#eval totalWallArea 4 8 6 8 4 8 6 8

end adam_wall_area_l2470_247002


namespace min_sum_squares_min_sum_squares_attained_l2470_247013

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
    (h_sum : 3 * x₁ + 2 * x₂ + x₃ = 30) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 125 := by
  sorry

theorem min_sum_squares_attained (ε : ℝ) (h_pos : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
    3 * x₁ + 2 * x₂ + x₃ = 30 ∧ 
    x₁^2 + x₂^2 + x₃^2 < 125 + ε := by
  sorry

end min_sum_squares_min_sum_squares_attained_l2470_247013


namespace part_I_part_II_l2470_247009

-- Define the function f
def f (a b x : ℝ) := 2 * x^2 - 2 * a * x + b

-- Define set A
def A (a b : ℝ) := {x : ℝ | f a b x > 0}

-- Define set B
def B (t : ℝ) := {x : ℝ | |x - t| ≤ 1}

-- Theorem for part (I)
theorem part_I (a b : ℝ) (h1 : f a b (-1) = -8) (h2 : ∀ x : ℝ, f a b x ≥ f a b (-1)) :
  (Set.univ \ A a b) ∪ B 1 = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Theorem for part (II)
theorem part_II (a b : ℝ) (h1 : f a b (-1) = -8) (h2 : ∀ x : ℝ, f a b x ≥ f a b (-1)) :
  {t : ℝ | A a b ∩ B t = ∅} = {t : ℝ | -2 ≤ t ∧ t ≤ 0} :=
sorry

end part_I_part_II_l2470_247009


namespace congruence_problem_l2470_247018

theorem congruence_problem (m : ℕ) : 
  163 * 937 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 11 := by
  sorry

end congruence_problem_l2470_247018


namespace sum_equals_seven_x_l2470_247070

theorem sum_equals_seven_x (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : 
  x + y + z = 7 * x := by
  sorry

end sum_equals_seven_x_l2470_247070


namespace sqrt_sum_greater_than_sqrt_of_sum_l2470_247045

theorem sqrt_sum_greater_than_sqrt_of_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end sqrt_sum_greater_than_sqrt_of_sum_l2470_247045


namespace distinct_roots_isosceles_triangle_k_values_l2470_247036

/-- The quadratic equation x^2 - (2k+1)x + k^2 + k = 0 has two distinct real roots for all k -/
theorem distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - (2*k+1)*x₁ + k^2 + k = 0 ∧ x₂^2 - (2*k+1)*x₂ + k^2 + k = 0 :=
sorry

/-- When two roots of x^2 - (2k+1)x + k^2 + k = 0 form two sides of an isosceles triangle 
    with the third side of length 5, k = 4 or k = 5 -/
theorem isosceles_triangle_k_values :
  ∃ x₁ x₂ : ℝ, 
    x₁^2 - (2*4+1)*x₁ + 4^2 + 4 = 0 ∧
    x₂^2 - (2*4+1)*x₂ + 4^2 + 4 = 0 ∧
    ((x₁ = 5 ∧ x₂ = x₁) ∨ (x₂ = 5 ∧ x₁ = x₂))
  ∧
  ∃ y₁ y₂ : ℝ,
    y₁^2 - (2*5+1)*y₁ + 5^2 + 5 = 0 ∧
    y₂^2 - (2*5+1)*y₂ + 5^2 + 5 = 0 ∧
    ((y₁ = 5 ∧ y₂ = y₁) ∨ (y₂ = 5 ∧ y₁ = y₂))
  ∧
  ∀ k : ℝ, k ≠ 4 → k ≠ 5 →
    ¬∃ z₁ z₂ : ℝ,
      z₁^2 - (2*k+1)*z₁ + k^2 + k = 0 ∧
      z₂^2 - (2*k+1)*z₂ + k^2 + k = 0 ∧
      ((z₁ = 5 ∧ z₂ = z₁) ∨ (z₂ = 5 ∧ z₁ = z₂)) :=
sorry

end distinct_roots_isosceles_triangle_k_values_l2470_247036


namespace building_entrances_l2470_247073

/-- Represents a building with multiple entrances -/
structure Building where
  floors : ℕ
  apartments_per_floor : ℕ
  total_apartments : ℕ

/-- Calculates the number of entrances in a building -/
def number_of_entrances (b : Building) : ℕ :=
  b.total_apartments / (b.floors * b.apartments_per_floor)

/-- Theorem stating the number of entrances in the specific building -/
theorem building_entrances :
  let b : Building := {
    floors := 9,
    apartments_per_floor := 4,
    total_apartments := 180
  }
  number_of_entrances b = 5 := by
  sorry

end building_entrances_l2470_247073


namespace four_boxes_volume_l2470_247099

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a^3

/-- The total volume of n identical cubes with edge length a -/
def total_volume (n : ℕ) (a : ℝ) : ℝ := n * (cube_volume a)

/-- Theorem: The total volume of four cubic boxes, each with an edge length of 5 feet, is 500 cubic feet -/
theorem four_boxes_volume : total_volume 4 5 = 500 := by
  sorry

end four_boxes_volume_l2470_247099


namespace quadratic_real_roots_condition_l2470_247076

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 + 2*x - k = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation x k

-- Theorem statement
theorem quadratic_real_roots_condition (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 :=
sorry

end quadratic_real_roots_condition_l2470_247076


namespace range_of_a_l2470_247054

/-- The inequality (a-3)x^2 + 2(a-3)x - 4 < 0 has a solution set of all real numbers for x -/
def has_all_real_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  has_all_real_solutions a ↔ -1 < a ∧ a < 3 :=
sorry

end range_of_a_l2470_247054


namespace prob_not_face_card_is_ten_thirteenths_l2470_247098

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of face cards in a standard deck
def face_cards : ℕ := 12

-- Define the probability of not getting a face card
def prob_not_face_card : ℚ := (total_cards - face_cards) / total_cards

-- Theorem statement
theorem prob_not_face_card_is_ten_thirteenths :
  prob_not_face_card = 10 / 13 := by sorry

end prob_not_face_card_is_ten_thirteenths_l2470_247098


namespace double_money_in_20_years_l2470_247077

/-- The simple interest rate that doubles a sum of money in 20 years -/
def double_money_rate : ℚ := 5 / 100

/-- The time period in years -/
def time_period : ℕ := 20

/-- The final amount after applying simple interest -/
def final_amount (principal : ℚ) : ℚ :=
  principal * (1 + double_money_rate * time_period)

theorem double_money_in_20_years (principal : ℚ) (h : principal > 0) :
  final_amount principal = 2 * principal := by
  sorry

#check double_money_in_20_years

end double_money_in_20_years_l2470_247077


namespace coefficient_x_cubed_sum_binomial_l2470_247029

theorem coefficient_x_cubed_sum_binomial (n : ℕ) (hn : n ≥ 3) :
  (Finset.range (n - 2)).sum (fun k => Nat.choose (k + 3) 3) = Nat.choose (n + 1) 4 := by
  sorry

#check coefficient_x_cubed_sum_binomial 2005

end coefficient_x_cubed_sum_binomial_l2470_247029


namespace ellipse_major_axis_length_l2470_247012

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The length of the major axis of the ellipse is 10.5 -/
theorem ellipse_major_axis_length :
  major_axis_length 3 0.75 = 10.5 := by
  sorry

end ellipse_major_axis_length_l2470_247012


namespace hyperbola_eccentricity_l2470_247026

/-- The eccentricity of a hyperbola with equation x²/4 - y²/3 = 1 is √7/2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 2 := by sorry

end hyperbola_eccentricity_l2470_247026


namespace divisibility_by_8640_l2470_247000

theorem divisibility_by_8640 (x : ℤ) : ∃ k : ℤ, x^9 - 6*x^7 + 9*x^5 - 4*x^3 = 8640 * k := by
  sorry

end divisibility_by_8640_l2470_247000


namespace mad_hatter_waiting_time_l2470_247066

/-- The rate at which the Mad Hatter's clock runs compared to normal time -/
def mad_hatter_rate : ℚ := 5/4

/-- The rate at which the March Hare's clock runs compared to normal time -/
def march_hare_rate : ℚ := 5/6

/-- The agreed meeting time in hours after noon -/
def meeting_time : ℚ := 5

theorem mad_hatter_waiting_time :
  let mad_hatter_arrival := meeting_time / mad_hatter_rate
  let march_hare_arrival := meeting_time / march_hare_rate
  march_hare_arrival - mad_hatter_arrival = 2 := by sorry

end mad_hatter_waiting_time_l2470_247066


namespace johns_drive_distance_l2470_247024

/-- Represents the total distance of John's drive in miles -/
def total_distance : ℝ := 360

/-- Represents the initial distance driven on battery alone in miles -/
def battery_distance : ℝ := 60

/-- Represents the gasoline consumption rate in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- Represents the average fuel efficiency in miles per gallon -/
def avg_fuel_efficiency : ℝ := 40

/-- Theorem stating that given the conditions, the total distance of John's drive is 360 miles -/
theorem johns_drive_distance :
  total_distance = battery_distance + 
  (total_distance - battery_distance) * gasoline_rate * avg_fuel_efficiency :=
by sorry

end johns_drive_distance_l2470_247024


namespace system_solution_exists_l2470_247038

theorem system_solution_exists (b : ℝ) : 
  (∃ (a x y : ℝ), 
    x = 7 / b - abs (y + b) ∧ 
    x^2 + y^2 + 96 = -a * (2 * y + a) - 20 * x) ↔ 
  (b ≤ -7/12 ∨ b > 0) :=
sorry

end system_solution_exists_l2470_247038


namespace percent_teachers_without_conditions_l2470_247095

theorem percent_teachers_without_conditions (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 90)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 2667 / 100 :=
sorry

end percent_teachers_without_conditions_l2470_247095


namespace series_sum_equals_three_l2470_247020

theorem series_sum_equals_three (k : ℝ) (hk : k > 1) :
  (∑' n : ℕ, (n^2 + 3*n - 2) / k^n) = 2 → k = 3 := by
  sorry

end series_sum_equals_three_l2470_247020


namespace isosceles_triangle_perimeter_l2470_247074

/-- Given an equilateral triangle with perimeter 45 and an isosceles triangle sharing one side
    with the equilateral triangle and having a base of length 10, the perimeter of the isosceles
    triangle is 40. -/
theorem isosceles_triangle_perimeter
  (equilateral_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 45)
  (h_isosceles_base : isosceles_base = 10)
  (h_shared_side : ∃ (side : ℝ), side = equilateral_perimeter / 3 ∧
                   ∃ (leg : ℝ), leg = side) :
  ∃ (isosceles_perimeter : ℝ), isosceles_perimeter = 40 :=
by sorry

end isosceles_triangle_perimeter_l2470_247074


namespace negative_less_than_positive_l2470_247087

theorem negative_less_than_positive : ∀ x y : ℝ, x < 0 → 0 < y → x < y := by sorry

end negative_less_than_positive_l2470_247087


namespace nested_a_value_l2470_247086

-- Define the function a
def a (k : ℕ) : ℕ := (k + 1)^2

-- State the theorem
theorem nested_a_value :
  let k : ℕ := 1
  a (a (a (a k))) = 458329 := by
  sorry

end nested_a_value_l2470_247086


namespace distance_to_center_l2470_247057

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8*x - 4*y + 16

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The given point -/
def given_point : ℝ × ℝ := (3, -1)

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem distance_to_center : distance circle_center given_point = Real.sqrt 2 := by sorry

end distance_to_center_l2470_247057


namespace traffic_light_change_probability_l2470_247050

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the duration of color changes -/
def changeDuration (cycle : TrafficLightCycle) : ℕ :=
  3 * 5 -- 5 seconds at the end of each color

/-- Theorem: Probability of observing a color change -/
theorem traffic_light_change_probability (cycle : TrafficLightCycle)
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50)
    (h4 : cycleDuration cycle = 100) :
    (changeDuration cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

end traffic_light_change_probability_l2470_247050


namespace angle_theta_value_l2470_247071

theorem angle_theta_value (θ : Real) (A B : Set Real) : 
  A = {1, Real.cos θ} →
  B = {0, 1/2, 1} →
  A ⊆ B →
  0 < θ →
  θ < π/2 →
  θ = π/3 := by
sorry

end angle_theta_value_l2470_247071


namespace T_values_l2470_247022

theorem T_values (θ : Real) :
  (∃ T : Real, T = Real.sqrt (1 + Real.sin (2 * θ))) →
  (((Real.sin (π - θ) = 3/5 ∧ π/2 < θ ∧ θ < π) →
    Real.sqrt (1 + Real.sin (2 * θ)) = 1/5) ∧
   ((Real.cos (π/2 - θ) = m ∧ π/2 < θ ∧ θ < 3*π/4) →
    Real.sqrt (1 + Real.sin (2 * θ)) = m - Real.sqrt (1 - m^2)) ∧
   ((Real.cos (π/2 - θ) = m ∧ 3*π/4 < θ ∧ θ < π) →
    Real.sqrt (1 + Real.sin (2 * θ)) = -m + Real.sqrt (1 - m^2))) :=
by sorry

end T_values_l2470_247022


namespace rightmost_three_digits_of_7_to_1993_l2470_247019

theorem rightmost_three_digits_of_7_to_1993 : 7^1993 % 1000 = 343 := by
  sorry

end rightmost_three_digits_of_7_to_1993_l2470_247019


namespace triangle_area_in_circle_l2470_247093

/-- The area of a triangle inscribed in a circle with given radius and side ratio --/
theorem triangle_area_in_circle (r : ℝ) (a b c : ℝ) (h_radius : r = 2 * Real.sqrt 3) 
  (h_ratio : ∃ (k : ℝ), a = 3 * k ∧ b = 5 * k ∧ c = 7 * k) :
  ∃ (area : ℝ), area = (135 * Real.sqrt 3) / 49 ∧ 
  area = (1 / 2) * a * b * Real.sin (2 * π / 3) := by
  sorry

end triangle_area_in_circle_l2470_247093


namespace sandy_marks_per_correct_sum_l2470_247032

theorem sandy_marks_per_correct_sum :
  ∀ (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_attempts : ℕ),
    marks_per_incorrect = 2 →
    total_attempts = 30 →
    total_marks = 50 →
    correct_attempts = 22 →
    marks_per_correct * correct_attempts - marks_per_incorrect * (total_attempts - correct_attempts) = total_marks →
    marks_per_correct = 3 := by
  sorry

end sandy_marks_per_correct_sum_l2470_247032


namespace square_area_from_diagonal_l2470_247060

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d^2 / 2 : ℝ) = 50 := by
  sorry

#check square_area_from_diagonal

end square_area_from_diagonal_l2470_247060


namespace wooden_block_volume_l2470_247014

/-- Represents a rectangular wooden block -/
structure WoodenBlock where
  length : ℝ
  baseArea : ℝ

/-- Calculates the volume of a rectangular wooden block -/
def volume (block : WoodenBlock) : ℝ :=
  block.length * block.baseArea

/-- Theorem: The volume of the wooden block is 864 cubic decimeters -/
theorem wooden_block_volume :
  ∀ (block : WoodenBlock),
    block.length = 72 →
    (3 - 1) * 2 * block.baseArea = 48 →
    volume block = 864 := by
  sorry

end wooden_block_volume_l2470_247014


namespace puppies_per_dog_l2470_247090

theorem puppies_per_dog (num_dogs : ℕ) (sold_fraction : ℚ) (price_per_puppy : ℕ) (total_revenue : ℕ) :
  num_dogs = 2 →
  sold_fraction = 3 / 4 →
  price_per_puppy = 200 →
  total_revenue = 3000 →
  (total_revenue / price_per_puppy : ℚ) / sold_fraction / num_dogs = 10 :=
by sorry

end puppies_per_dog_l2470_247090


namespace consecutive_even_numbers_problem_l2470_247016

theorem consecutive_even_numbers_problem :
  ∀ (x y z : ℕ),
  (y = x + 2) →
  (z = y + 2) →
  (3 * x = 2 * z + 14) →
  z = 26 := by
sorry

end consecutive_even_numbers_problem_l2470_247016


namespace sum_of_digits_main_expression_l2470_247053

/-- Represents a string of digits --/
structure DigitString :=
  (length : ℕ)
  (digit : ℕ)
  (digit_valid : digit < 10)

/-- Calculates the product of two DigitStrings --/
def multiply_digit_strings (a b : DigitString) : ℕ := sorry

/-- Calculates the sum of digits in a natural number --/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents the expression (80 eights × 80 fives + 80 ones) --/
def main_expression : ℕ :=
  let eights : DigitString := ⟨80, 8, by norm_num⟩
  let fives : DigitString := ⟨80, 5, by norm_num⟩
  let ones : DigitString := ⟨80, 1, by norm_num⟩
  multiply_digit_strings eights fives + ones.length * ones.digit

/-- The main theorem to be proved --/
theorem sum_of_digits_main_expression :
  sum_of_digits main_expression = 400 := by sorry

end sum_of_digits_main_expression_l2470_247053


namespace slope_MN_constant_l2470_247080

/-- Curve C defined by y² = 4x -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point D on curve C -/
def D : ℝ × ℝ := (1, 2)

/-- Line with slope k passing through D -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - D.1) + D.2}

/-- Theorem: The slope of line MN is constant -/
theorem slope_MN_constant (k : ℝ) :
  k ≠ 0 →
  D ∈ C →
  ∃ (M N : ℝ × ℝ),
    M ∈ C ∧ M ∈ line k ∧
    N ∈ C ∧ N ∈ line (-1/k) ∧
    M ≠ D ∧ N ≠ D →
    (N.2 - M.2) / (N.1 - M.1) = -1 := by
  sorry

end slope_MN_constant_l2470_247080


namespace insufficient_information_for_unique_solution_l2470_247068

theorem insufficient_information_for_unique_solution :
  ∀ (x y z w : ℕ),
  x + y + z + w = 750 →
  10 * x + 20 * y + 50 * z + 100 * w = 27500 →
  ∃ (y' : ℕ), y ≠ y' ∧
  ∃ (x' z' w' : ℕ),
  x' + y' + z' + w' = 750 ∧
  10 * x' + 20 * y' + 50 * z' + 100 * w' = 27500 :=
by sorry

end insufficient_information_for_unique_solution_l2470_247068


namespace cyclic_inequality_l2470_247043

theorem cyclic_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x * y + y * z + z * x = x + y + z) : 
  1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) ≤ 1 ∧ 
  (1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end cyclic_inequality_l2470_247043


namespace general_admission_price_l2470_247056

/-- Proves that the price of a general admission seat is $21.85 -/
theorem general_admission_price : 
  ∀ (total_tickets : ℕ) 
    (total_revenue : ℚ) 
    (vip_price : ℚ) 
    (gen_price : ℚ),
  total_tickets = 320 →
  total_revenue = 7500 →
  vip_price = 45 →
  (∃ (vip_tickets gen_tickets : ℕ),
    vip_tickets + gen_tickets = total_tickets ∧
    vip_tickets = gen_tickets - 276 ∧
    vip_price * vip_tickets + gen_price * gen_tickets = total_revenue) →
  gen_price = 21.85 := by
sorry


end general_admission_price_l2470_247056


namespace notebook_length_for_12cm_span_l2470_247069

/-- Given a hand span and a notebook with a long side twice the span, calculate the length of the notebook's long side. -/
def notebook_length (hand_span : ℝ) : ℝ := 2 * hand_span

/-- Theorem stating that for a hand span of 12 cm, the notebook's long side is 24 cm. -/
theorem notebook_length_for_12cm_span :
  notebook_length 12 = 24 := by sorry

end notebook_length_for_12cm_span_l2470_247069


namespace total_capital_calculation_l2470_247085

/-- Represents the total capital at the end of the first year given an initial investment and profit rate. -/
def totalCapitalEndOfYear (initialInvestment : ℝ) (profitRate : ℝ) : ℝ :=
  initialInvestment * (1 + profitRate)

/-- Theorem stating that for an initial investment of 50 ten thousand yuan and profit rate P,
    the total capital at the end of the first year is 50(1+P) ten thousand yuan. -/
theorem total_capital_calculation (P : ℝ) :
  totalCapitalEndOfYear 50 P = 50 * (1 + P) := by
  sorry

end total_capital_calculation_l2470_247085


namespace fraction_proof_l2470_247023

theorem fraction_proof (x : ℝ) (f : ℝ) : 
  x = 300 → 0.70 * x = f * x + 110 → f = 1/3 := by
  sorry

end fraction_proof_l2470_247023


namespace remainder_theorem_l2470_247031

theorem remainder_theorem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + (2 * u + 1) * y) % y = v := by
  sorry

end remainder_theorem_l2470_247031


namespace isosceles_right_triangle_roots_l2470_247061

theorem isosceles_right_triangle_roots (p q : ℂ) (z₁ z₂ : ℂ) : 
  z₁^2 + 2*p*z₁ + q = 0 →
  z₂^2 + 2*p*z₂ + q = 0 →
  z₂ = Complex.I * z₁ →
  p^2 / q = 2 := by
  sorry

end isosceles_right_triangle_roots_l2470_247061


namespace percentage_of_men_speaking_french_l2470_247039

theorem percentage_of_men_speaking_french (E : ℝ) (E_pos : E > 0) :
  let men_percentage : ℝ := 70
  let french_speaking_percentage : ℝ := 40
  let women_not_speaking_french_percentage : ℝ := 83.33333333333331
  let men_count : ℝ := (men_percentage / 100) * E
  let french_speaking_count : ℝ := (french_speaking_percentage / 100) * E
  let women_count : ℝ := E - men_count
  let women_speaking_french_count : ℝ := (1 - women_not_speaking_french_percentage / 100) * women_count
  let men_speaking_french_count : ℝ := french_speaking_count - women_speaking_french_count
  (men_speaking_french_count / men_count) * 100 = 50 := by
sorry

end percentage_of_men_speaking_french_l2470_247039


namespace largest_of_three_l2470_247025

theorem largest_of_three (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x*y + y*z + z*x = -8)
  (prod_eq : x*y*z = -18) :
  max x (max y z) = Real.sqrt 5 :=
sorry

end largest_of_three_l2470_247025


namespace square_sum_equals_twenty_l2470_247055

theorem square_sum_equals_twenty (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end square_sum_equals_twenty_l2470_247055


namespace intersection_of_A_and_B_l2470_247062

def A : Set ℝ := {x | x^2 - 4*x > 0}
def B : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x | -1 ≤ x ∧ x < 0} := by sorry

end intersection_of_A_and_B_l2470_247062


namespace problem_statement_l2470_247006

theorem problem_statement (x y z : ℚ) : 
  x = 1/3 → y = 2/3 → z = x * y → 3 * x^2 * y^5 * z^3 = 768/1594323 := by
sorry

end problem_statement_l2470_247006


namespace perpendicular_slope_l2470_247017

/-- Given a line with equation 2x + 3y = 6, the slope of a perpendicular line is 3/2 -/
theorem perpendicular_slope (x y : ℝ) :
  (2 * x + 3 * y = 6) →
  ∃ m : ℝ, m = 3 / 2 ∧ m * (-2 / 3) = -1 :=
by sorry

end perpendicular_slope_l2470_247017


namespace arithmetic_sequence_property_l2470_247049

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 := by
sorry

end arithmetic_sequence_property_l2470_247049


namespace arithmetic_sum_l2470_247051

theorem arithmetic_sum : 5 * 12 + 7 * 9 + 8 * 4 + 6 * 7 + 2 * 13 = 223 := by
  sorry

end arithmetic_sum_l2470_247051


namespace cubed_49_plus_1_l2470_247089

theorem cubed_49_plus_1 : 49^3 + 3*(49^2) + 3*49 + 1 = 125000 := by
  sorry

end cubed_49_plus_1_l2470_247089


namespace difference_of_squares_l2470_247021

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l2470_247021


namespace cake_cost_l2470_247001

/-- Represents the duration of vacations in days -/
def vacation_duration_1 : ℕ := 7
def vacation_duration_2 : ℕ := 4

/-- Represents the payment in CZK for each vacation period -/
def payment_1 : ℕ := 700
def payment_2 : ℕ := 340

/-- Represents the daily rate for dog walking and rabbit feeding -/
def daily_rate : ℕ := 120

theorem cake_cost (cake_price : ℕ) : 
  (cake_price + payment_1) / vacation_duration_1 = 
  (cake_price + payment_2) / vacation_duration_2 → 
  cake_price = 140 := by
  sorry

end cake_cost_l2470_247001


namespace factorial_ratio_equals_119_factorial_l2470_247044

theorem factorial_ratio_equals_119_factorial : (Nat.factorial (Nat.factorial 5)) / (Nat.factorial 5) = Nat.factorial 119 := by
  sorry

end factorial_ratio_equals_119_factorial_l2470_247044


namespace marbles_given_l2470_247081

theorem marbles_given (initial_marbles : ℕ) (remaining_marbles : ℕ) : 
  initial_marbles = 87 → remaining_marbles = 79 → initial_marbles - remaining_marbles = 8 := by
sorry

end marbles_given_l2470_247081


namespace influenza_virus_diameter_l2470_247078

theorem influenza_virus_diameter (n : ℤ) : 0.000000203 = 2.03 * (10 : ℝ) ^ n → n = -7 := by
  sorry

end influenza_virus_diameter_l2470_247078


namespace range_of_a_l2470_247034

def M : Set ℝ := {x | |x| < 1}
def N (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : M ∪ N a = M) : a ∈ Set.Ioo (-1) 1 := by
  sorry

end range_of_a_l2470_247034


namespace elite_academy_games_l2470_247088

/-- The number of teams in the Elite Academy Basketball League -/
def num_teams : ℕ := 8

/-- The number of times each team plays every other team -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 3

/-- The total number of games in a season for the Elite Academy Basketball League -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem elite_academy_games :
  total_games = 108 := by sorry

end elite_academy_games_l2470_247088


namespace time_addition_theorem_l2470_247027

/-- Represents a date and time --/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime --/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- Checks if two DateTimes are equal --/
def dateTimeEqual (dt1 dt2 : DateTime) : Prop :=
  dt1.year = dt2.year ∧
  dt1.month = dt2.month ∧
  dt1.day = dt2.day ∧
  dt1.hour = dt2.hour ∧
  dt1.minute = dt2.minute

theorem time_addition_theorem :
  let start := DateTime.mk 2023 7 4 12 0
  let end_time := DateTime.mk 2023 7 6 21 36
  dateTimeEqual (addMinutes start 3456) end_time :=
by sorry

end time_addition_theorem_l2470_247027


namespace arithmetic_sequence_properties_l2470_247059

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  h1 : a 3 = 10
  h2 : a 12 = 31

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 1 = 16/3) ∧ 
  (∀ n : ℕ, seq.a (n + 1) - seq.a n = 7/3) ∧
  (∀ n : ℕ, seq.a n = 7/3 * n + 3) ∧
  (seq.a 18 = 45) ∧
  (∀ n : ℕ, seq.a n ≠ 85) := by
  sorry

#check arithmetic_sequence_properties

end arithmetic_sequence_properties_l2470_247059


namespace bisecting_line_tangent_lines_l2470_247047

-- Define the point P and circle C
def P : ℝ × ℝ := (-1, 4)
def C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define the center of circle C
def center : ℝ × ℝ := (2, 3)

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 3*y - 11 = 0
def line2 (x y : ℝ) : Prop := y - 4 = 0
def line3 (x y : ℝ) : Prop := 3*x + 4*y - 13 = 0

-- Theorem statements
theorem bisecting_line :
  line1 P.1 P.2 ∧ line1 center.1 center.2 := by sorry

theorem tangent_lines :
  (line2 P.1 P.2 ∧ (∃ (p : ℝ × ℝ), p ∈ C ∧ line2 p.1 p.2 ∧ (∀ (q : ℝ × ℝ), q ∈ C → line2 q.1 q.2 → q = p))) ∧
  (line3 P.1 P.2 ∧ (∃ (p : ℝ × ℝ), p ∈ C ∧ line3 p.1 p.2 ∧ (∀ (q : ℝ × ℝ), q ∈ C → line3 q.1 q.2 → q = p))) := by sorry

end bisecting_line_tangent_lines_l2470_247047


namespace kneading_time_is_ten_l2470_247083

/-- Represents the time in minutes for bread-making process --/
structure BreadTime where
  total : ℕ
  rising : ℕ
  baking : ℕ

/-- Calculates the kneading time given the bread-making times --/
def kneadingTime (bt : BreadTime) : ℕ :=
  bt.total - (2 * bt.rising + bt.baking)

/-- Theorem stating that the kneading time is 10 minutes for the given conditions --/
theorem kneading_time_is_ten :
  let bt : BreadTime := { total := 280, rising := 120, baking := 30 }
  kneadingTime bt = 10 := by sorry

end kneading_time_is_ten_l2470_247083


namespace frame_cells_l2470_247011

theorem frame_cells (n : ℕ) (h : n = 254) : 
  n^2 - (n - 2)^2 = 2016 :=
by sorry

end frame_cells_l2470_247011


namespace at_least_two_in_same_group_l2470_247092

theorem at_least_two_in_same_group 
  (n : ℕ) 
  (h_n : n = 28) 
  (partition1 partition2 partition3 : Fin n → Fin 3) 
  (h_diff1 : partition1 ≠ partition2) 
  (h_diff2 : partition2 ≠ partition3) 
  (h_diff3 : partition1 ≠ partition3) :
  ∃ i j : Fin n, i ≠ j ∧ 
    partition1 i = partition1 j ∧ 
    partition2 i = partition2 j ∧ 
    partition3 i = partition3 j :=
sorry

end at_least_two_in_same_group_l2470_247092


namespace trajectory_of_P_l2470_247084

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 15

-- Define point N
def point_N : ℝ × ℝ := (1, 0)

-- Define the property of point M being on circle C
def point_M_on_C (M : ℝ × ℝ) : Prop := circle_C M.1 M.2

-- Define point P as the intersection of perpendicular bisector of MN and CM
def point_P (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  point_M_on_C M ∧ 
  -- Additional conditions for P would be defined here, but we omit the detailed geometric conditions
  True

-- State the theorem
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, (∃ M : ℝ × ℝ, point_P M P) →
  (P.1^2 / 4 + P.2^2 / 3 = 1) :=
sorry

end trajectory_of_P_l2470_247084


namespace vocabulary_test_score_l2470_247072

theorem vocabulary_test_score (total_words : ℕ) (target_score : ℚ) 
  (h1 : total_words = 600) 
  (h2 : target_score = 90 / 100) : 
  ∃ (words_to_learn : ℕ), 
    (words_to_learn : ℚ) / total_words = target_score ∧ 
    words_to_learn = 540 := by
  sorry

end vocabulary_test_score_l2470_247072


namespace isosceles_triangle_proof_l2470_247010

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles --/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Checks if the triangle satisfies the triangle inequality --/
def Triangle.satisfiesInequality (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem isosceles_triangle_proof (rope_length : ℝ) 
  (h1 : rope_length = 18) 
  (h2 : ∃ t : Triangle, t.isIsosceles ∧ t.a + t.b + t.c = rope_length ∧ t.a = t.b ∧ t.a = 2 * t.c) :
  ∃ t : Triangle, t.isIsosceles ∧ t.satisfiesInequality ∧ t.a = 36/5 ∧ t.b = 36/5 ∧ t.c = 18/5 ∧
  ∃ t2 : Triangle, t2.isIsosceles ∧ t2.satisfiesInequality ∧ t2.a = 4 ∧ t2.b = 7 ∧ t2.c = 7 ∧
  t2.a + t2.b + t2.c = rope_length :=
by
  sorry


end isosceles_triangle_proof_l2470_247010


namespace divisibility_problem_l2470_247007

theorem divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (∃ k : ℕ, abc - 1 = k * ((a - 1) * (b - 1) * (c - 1))) →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end divisibility_problem_l2470_247007


namespace rotational_inertia_scaling_l2470_247097

/-- Represents a sphere with a given radius and rotational inertia about its center axis -/
structure Sphere where
  radius : ℝ
  rotationalInertia : ℝ

/-- Given two spheres with the same density, where the second sphere has twice the radius of the first,
    prove that the rotational inertia of the second sphere is 32 times that of the first sphere -/
theorem rotational_inertia_scaling (s1 s2 : Sphere) (h1 : s2.radius = 2 * s1.radius) :
  s2.rotationalInertia = 32 * s1.rotationalInertia := by
  sorry


end rotational_inertia_scaling_l2470_247097


namespace product_of_sum_of_logs_l2470_247005

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem product_of_sum_of_logs (a b : ℝ) (h : log10 a + log10 b = 1) : a * b = 10 := by
  sorry

end product_of_sum_of_logs_l2470_247005


namespace geometric_sequence_condition_l2470_247065

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) (c : ℝ) : ℝ := 3^n - c

/-- The nth term of the sequence -/
def a (n : ℕ) (c : ℝ) : ℝ := S (n + 1) c - S n c

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_condition (c : ℝ) :
  (c = 1 ↔ IsGeometric (a · c)) :=
sorry

end geometric_sequence_condition_l2470_247065


namespace hyperbola_incenter_theorem_l2470_247063

/-- Hyperbola C: x²/4 - y²/5 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- Point P is on the hyperbola in the first quadrant -/
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y
  first_quadrant : x > 0 ∧ y > 0

/-- F₁ and F₂ are the left and right foci of the hyperbola -/
structure Foci where
  f₁ : ℝ × ℝ
  f₂ : ℝ × ℝ

/-- I is the incenter of triangle PF₁F₂ -/
def incenter (p : PointOnHyperbola) (f : Foci) : ℝ × ℝ := sorry

/-- |PF₁| = 2|PF₂| -/
def focal_distance_condition (p : PointOnHyperbola) (f : Foci) : Prop :=
  let (x₁, y₁) := f.f₁
  let (x₂, y₂) := f.f₂
  ((p.x - x₁)^2 + (p.y - y₁)^2) = 4 * ((p.x - x₂)^2 + (p.y - y₂)^2)

/-- Vector PI = x * Vector PF₁ + y * Vector PF₂ -/
def vector_condition (p : PointOnHyperbola) (f : Foci) (x y : ℝ) : Prop :=
  let i := incenter p f
  let (x₁, y₁) := f.f₁
  let (x₂, y₂) := f.f₂
  (i.1 - p.x, i.2 - p.y) = (x * (x₁ - p.x) + y * (x₂ - p.x), x * (y₁ - p.y) + y * (y₂ - p.y))

/-- Main theorem -/
theorem hyperbola_incenter_theorem (p : PointOnHyperbola) (f : Foci) (x y : ℝ) :
  focal_distance_condition p f →
  vector_condition p f x y →
  y - x = 2/9 := by sorry

end hyperbola_incenter_theorem_l2470_247063
