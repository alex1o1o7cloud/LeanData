import Mathlib

namespace martha_coffee_spending_l864_86466

/-- The cost of an iced coffee that satisfies Martha's coffee spending reduction --/
def iced_coffee_cost : ℚ := by sorry

/-- Proves that the cost of an iced coffee is $2.00 --/
theorem martha_coffee_spending :
  let latte_cost : ℚ := 4
  let lattes_per_week : ℕ := 5
  let iced_coffees_per_week : ℕ := 3
  let weeks_per_year : ℕ := 52
  let spending_reduction_ratio : ℚ := 1 / 4
  let spending_reduction_amount : ℚ := 338

  let annual_latte_spending : ℚ := latte_cost * lattes_per_week * weeks_per_year
  let annual_iced_coffee_spending : ℚ := iced_coffee_cost * iced_coffees_per_week * weeks_per_year
  let total_annual_spending : ℚ := annual_latte_spending + annual_iced_coffee_spending

  (1 - spending_reduction_ratio) * total_annual_spending = total_annual_spending - spending_reduction_amount →
  iced_coffee_cost = 2 := by sorry

end martha_coffee_spending_l864_86466


namespace factorize_x_squared_plus_2x_l864_86481

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) := by
  sorry

end factorize_x_squared_plus_2x_l864_86481


namespace diagonals_bisect_implies_parallelogram_l864_86474

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A diagonal of a quadrilateral is a line segment connecting two non-adjacent vertices. -/
def Quadrilateral.diagonal (q : Quadrilateral) (i j : Fin 4) : ℝ × ℝ → ℝ × ℝ :=
  sorry

/-- Two line segments bisect each other if they intersect at their midpoints. -/
def bisect (seg1 seg2 : ℝ × ℝ → ℝ × ℝ) : Prop :=
  sorry

/-- A parallelogram is a quadrilateral with two pairs of parallel sides. -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- If the diagonals of a quadrilateral bisect each other, then it is a parallelogram. -/
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  (∃ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ 
    bisect (q.diagonal i k) (q.diagonal j l)) →
  is_parallelogram q :=
sorry

end diagonals_bisect_implies_parallelogram_l864_86474


namespace cuboid_properties_l864_86452

/-- Represents a cuboid with length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edge lengths of a cuboid -/
def sumEdgeLengths (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Theorem about a specific cuboid's properties -/
theorem cuboid_properties :
  ∃ c : Cuboid,
    c.length = 2 * c.width ∧
    c.width = c.height ∧
    sumEdgeLengths c = 48 ∧
    surfaceArea c = 90 ∧
    volume c = 54 := by
  sorry

end cuboid_properties_l864_86452


namespace cylinder_volume_approximation_l864_86485

/-- The volume of a cylinder with diameter 14 cm and height 2 cm is approximately 307.88 cubic centimeters. -/
theorem cylinder_volume_approximation :
  let d : ℝ := 14  -- diameter in cm
  let h : ℝ := 2   -- height in cm
  let r : ℝ := d / 2  -- radius in cm
  let π : ℝ := Real.pi
  let V : ℝ := π * r^2 * h  -- volume formula
  ∃ ε > 0, abs (V - 307.88) < ε ∧ ε < 0.01 :=
by sorry

end cylinder_volume_approximation_l864_86485


namespace freds_remaining_cards_l864_86469

/-- Calculates the number of baseball cards Fred has after Melanie's purchase. -/
def remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem stating that Fred's remaining cards is the difference between his initial cards and those bought by Melanie. -/
theorem freds_remaining_cards :
  remaining_cards 5 3 = 2 := by
  sorry

end freds_remaining_cards_l864_86469


namespace magazines_per_box_l864_86478

theorem magazines_per_box (total_magazines : ℕ) (num_boxes : ℕ) (h1 : total_magazines = 63) (h2 : num_boxes = 7) :
  total_magazines / num_boxes = 9 := by
  sorry

end magazines_per_box_l864_86478


namespace count_numbers_with_seven_800_l864_86406

def contains_seven (n : Nat) : Bool :=
  let digits := n.digits 10
  7 ∈ digits

def count_numbers_with_seven (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_seven |>.length

theorem count_numbers_with_seven_800 :
  count_numbers_with_seven 800 = 152 := by
  sorry

end count_numbers_with_seven_800_l864_86406


namespace max_yellow_apples_max_total_apples_l864_86431

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples taken from the basket -/
structure TakenApples :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if the condition for stopping is met -/
def stoppingCondition (taken : TakenApples) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial basket of apples -/
def initialBasket : Basket :=
  { green := 10, yellow := 13, red := 18 }

/-- Theorem for the maximum number of yellow apples that can be taken -/
theorem max_yellow_apples :
  ∃ (taken : TakenApples),
    taken.yellow = 13 ∧
    taken.yellow ≤ taken.red ∧
    taken.green ≤ taken.yellow ∧
    ∀ (other : TakenApples),
      other.yellow > 13 →
      other.yellow > other.red ∨ other.green > other.yellow :=
sorry

/-- Theorem for the maximum number of apples that can be taken in total -/
theorem max_total_apples :
  ∃ (taken : TakenApples),
    taken.green + taken.yellow + taken.red = 39 ∧
    ¬(stoppingCondition taken) ∧
    ∀ (other : TakenApples),
      other.green + other.yellow + other.red > 39 →
      stoppingCondition other :=
sorry

end max_yellow_apples_max_total_apples_l864_86431


namespace stadium_height_l864_86427

/-- The height of a rectangular stadium given its length, width, and the length of the longest pole that can fit diagonally. -/
theorem stadium_height (length width diagonal : ℝ) (h1 : length = 24) (h2 : width = 18) (h3 : diagonal = 34) :
  Real.sqrt (diagonal^2 - length^2 - width^2) = 16 := by
  sorry

end stadium_height_l864_86427


namespace no_integers_product_sum_20182017_l864_86440

theorem no_integers_product_sum_20182017 : ¬∃ (a b : ℤ), a * b * (a + b) = 20182017 := by
  sorry

end no_integers_product_sum_20182017_l864_86440


namespace volleyball_net_max_removable_edges_l864_86421

/-- Represents a volleyball net graph -/
structure VolleyballNet where
  rows : Nat
  cols : Nat

/-- Calculates the number of vertices in the volleyball net graph -/
def VolleyballNet.vertexCount (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1) + net.rows * net.cols

/-- Calculates the total number of edges in the volleyball net graph -/
def VolleyballNet.edgeCount (net : VolleyballNet) : Nat :=
  -- This is a placeholder. The actual calculation would be more complex.
  4 * net.rows * net.cols + net.rows * (net.cols - 1) + net.cols * (net.rows - 1)

/-- Theorem: The maximum number of edges that can be removed without disconnecting
    the graph for a 10x20 volleyball net is 800 -/
theorem volleyball_net_max_removable_edges :
  let net : VolleyballNet := { rows := 10, cols := 20 }
  ∃ (removable : Nat), removable = net.edgeCount - (net.vertexCount - 1) ∧ removable = 800 := by
  sorry


end volleyball_net_max_removable_edges_l864_86421


namespace joe_haircut_time_l864_86416

/-- The time it takes to cut different types of hair and the number of haircuts Joe performed --/
structure HaircutData where
  womenTime : ℕ  -- Time to cut a woman's hair
  menTime : ℕ    -- Time to cut a man's hair
  kidsTime : ℕ   -- Time to cut a kid's hair
  womenCount : ℕ -- Number of women's haircuts
  menCount : ℕ   -- Number of men's haircuts
  kidsCount : ℕ  -- Number of kids' haircuts

/-- Calculate the total time Joe spent cutting hair --/
def totalHaircutTime (data : HaircutData) : ℕ :=
  data.womenTime * data.womenCount +
  data.menTime * data.menCount +
  data.kidsTime * data.kidsCount

/-- Theorem stating that Joe spent 255 minutes cutting hair --/
theorem joe_haircut_time :
  let data : HaircutData := {
    womenTime := 50,
    menTime := 15,
    kidsTime := 25,
    womenCount := 3,
    menCount := 2,
    kidsCount := 3
  }
  totalHaircutTime data = 255 := by
  sorry


end joe_haircut_time_l864_86416


namespace tissue_box_price_l864_86410

-- Define the quantities and prices
def toilet_paper_rolls : ℕ := 10
def paper_towel_rolls : ℕ := 7
def tissue_boxes : ℕ := 3
def toilet_paper_price : ℚ := 1.5
def paper_towel_price : ℚ := 2
def total_cost : ℚ := 35

-- Theorem to prove
theorem tissue_box_price : 
  (total_cost - (toilet_paper_rolls * toilet_paper_price + paper_towel_rolls * paper_towel_price)) / tissue_boxes = 2 := by
  sorry

end tissue_box_price_l864_86410


namespace no_real_solutions_l864_86475

theorem no_real_solutions :
  ¬∃ (x : ℝ), (2*x - 6)^2 + 4 = -2*|x| := by
sorry

end no_real_solutions_l864_86475


namespace min_value_x_plus_2y_l864_86477

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  ∀ z w : ℝ, z > 0 ∧ w > 0 ∧ 2/z + 1/w = 1 → x + 2*y ≤ z + 2*w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2/a + 1/b = 1 ∧ a + 2*b = 8 :=
sorry

end min_value_x_plus_2y_l864_86477


namespace gretchen_to_rachelle_ratio_l864_86430

def pennies_problem (rachelle gretchen rocky : ℕ) : Prop :=
  rachelle = 180 ∧
  rocky = gretchen / 3 ∧
  rachelle + gretchen + rocky = 300

theorem gretchen_to_rachelle_ratio :
  ∀ rachelle gretchen rocky : ℕ,
  pennies_problem rachelle gretchen rocky →
  gretchen * 2 = rachelle :=
by
  sorry

end gretchen_to_rachelle_ratio_l864_86430


namespace factorial_800_trailing_zeros_l864_86467

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem factorial_800_trailing_zeros :
  trailingZeros 800 = 199 := by
  sorry

end factorial_800_trailing_zeros_l864_86467


namespace max_sin_a_is_one_l864_86487

theorem max_sin_a_is_one (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), Real.sin x ≤ m :=
sorry

end max_sin_a_is_one_l864_86487


namespace quiz_mcq_count_l864_86459

theorem quiz_mcq_count :
  ∀ (n : ℕ),
  (((1 : ℚ) / 3) ^ n * ((1 : ℚ) / 2) ^ 2 = (1 : ℚ) / 12) →
  n = 1 :=
by sorry

end quiz_mcq_count_l864_86459


namespace platform_length_l864_86436

/-- The length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 500)
  (h2 : time_platform = 45)
  (h3 : time_pole = 25) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 400 :=
by sorry

end platform_length_l864_86436


namespace probability_exactly_three_less_than_seven_l864_86486

def probability_less_than_7 : ℚ := 1 / 2

def number_of_dice : ℕ := 6

def target_count : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_exactly_three_less_than_seven :
  (choose number_of_dice target_count : ℚ) * probability_less_than_7^target_count * (1 - probability_less_than_7)^(number_of_dice - target_count) = 5 / 16 := by
  sorry

end probability_exactly_three_less_than_seven_l864_86486


namespace group_size_calculation_l864_86476

theorem group_size_calculation (T : ℕ) (L : ℕ) : 
  T = L + 90 → -- Total is sum of young and old
  (L : ℚ) / T = 1/4 → -- Probability of selecting young person
  T = 120 := by
  sorry

end group_size_calculation_l864_86476


namespace quadratic_root_condition_l864_86413

theorem quadratic_root_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) →
  c = -a :=
sorry

end quadratic_root_condition_l864_86413


namespace twentieth_term_of_sequence_l864_86451

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem twentieth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 5) :
  arithmeticSequence a₁ (a₂ - a₁) 20 = 98 :=
by
  sorry

end twentieth_term_of_sequence_l864_86451


namespace probability_not_distinct_roots_greater_than_two_l864_86488

def is_valid_pair (a c : ℤ) : Prop :=
  |a| ≤ 6 ∧ |c| ≤ 6

def has_distinct_roots_greater_than_two (a c : ℤ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 2 ∧ x₂ > 2 ∧ a * x₁^2 - 3 * a * x₁ + c = 0 ∧ a * x₂^2 - 3 * a * x₂ + c = 0

def total_pairs : ℕ := 169

def valid_pairs : ℕ := 2

theorem probability_not_distinct_roots_greater_than_two :
  (total_pairs - valid_pairs) / total_pairs = 167 / 169 :=
sorry

end probability_not_distinct_roots_greater_than_two_l864_86488


namespace complex_multiplication_l864_86425

theorem complex_multiplication : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 → 
  (2 + Complex.I) * (1 - 3 * Complex.I) = 5 - 5 * Complex.I := by
  sorry

end complex_multiplication_l864_86425


namespace first_number_in_sequence_l864_86456

def sequence_product (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → n ≤ 10 → a n = a (n-1) * a (n-2)

theorem first_number_in_sequence 
  (a : ℕ → ℚ) 
  (h_seq : sequence_product a) 
  (h_8 : a 8 = 36) 
  (h_9 : a 9 = 324) 
  (h_10 : a 10 = 11664) : 
  a 1 = 59049 / 65536 := by
sorry

end first_number_in_sequence_l864_86456


namespace rectangle_diagonal_l864_86496

theorem rectangle_diagonal (side1 : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side1 = 6 → area = 48 → diagonal = 10 → 
  ∃ (side2 : ℝ), 
    side1 * side2 = area ∧ 
    diagonal^2 = side1^2 + side2^2 := by
  sorry

end rectangle_diagonal_l864_86496


namespace factorization_of_2x_squared_minus_2_l864_86445

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end factorization_of_2x_squared_minus_2_l864_86445


namespace parallel_vectors_k_value_l864_86449

def a : Fin 2 → ℝ := ![1, -3]
def b : Fin 2 → ℝ := ![2, 1]

theorem parallel_vectors_k_value : 
  ∃ (k : ℝ), ∃ (c : ℝ), c ≠ 0 ∧ 
    (∀ i : Fin 2, (k * a i + b i) = c * (a i - 2 * b i)) → 
    k = -1/2 := by
  sorry

end parallel_vectors_k_value_l864_86449


namespace expression_equality_l864_86483

theorem expression_equality : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 27 + (Real.pi + 1)^0 = 4 * Real.sqrt 3 + 1 := by
  sorry

end expression_equality_l864_86483


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l864_86464

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l864_86464


namespace stream_speed_l864_86405

/-- Proves that the speed of a stream is 8 kmph given the conditions of a boat's travel --/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (h1 : boat_speed = 24)
  (h2 : downstream_distance = 64)
  (h3 : upstream_distance = 32)
  (h4 : downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) :
  x = 8 := by
  sorry

#check stream_speed

end stream_speed_l864_86405


namespace max_value_implies_a_l864_86419

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∃ x ∈ Set.Icc 0 4, f a x = 3) ∧
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) →
  a = 3 := by
sorry

end max_value_implies_a_l864_86419


namespace speedster_convertibles_count_l864_86489

theorem speedster_convertibles_count (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  (2 : ℚ) / 3 * total = speedsters →
  (4 : ℚ) / 5 * speedsters = convertibles →
  total - speedsters = 40 →
  convertibles = 64 := by
  sorry

end speedster_convertibles_count_l864_86489


namespace smallest_two_digit_reverse_diff_perfect_square_l864_86417

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_two_digit_reverse_diff_perfect_square :
  ∃ N : ℕ, is_two_digit N ∧
    is_perfect_square (N - reverse_digits N) ∧
    (N - reverse_digits N > 0) ∧
    (∀ M : ℕ, is_two_digit M →
      is_perfect_square (M - reverse_digits M) →
      (M - reverse_digits M > 0) →
      N ≤ M) ∧
    N = 90 := by
  sorry

end smallest_two_digit_reverse_diff_perfect_square_l864_86417


namespace hyperbola_condition_l864_86442

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 5) - y^2 / (k + 2) = 1 ∧ 
  (k - 5 > 0 ∧ k + 2 > 0)

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (is_hyperbola k → k > 5) ∧ 
  ¬(k > 5 → is_hyperbola k) :=
sorry

end hyperbola_condition_l864_86442


namespace comparison_of_powers_and_log_l864_86450

theorem comparison_of_powers_and_log : 7^(3/10) > (3/10)^7 ∧ (3/10)^7 > Real.log (3/10) := by
  sorry

end comparison_of_powers_and_log_l864_86450


namespace initial_ratio_men_to_women_l864_86465

/-- Proves that the initial ratio of men to women in a room was 4:5 --/
theorem initial_ratio_men_to_women :
  ∀ (initial_men initial_women : ℕ),
  (initial_women - 3) * 2 = 24 →
  initial_men + 2 = 14 →
  (initial_men : ℚ) / initial_women = 4 / 5 := by
sorry

end initial_ratio_men_to_women_l864_86465


namespace initial_interest_rate_l864_86495

/-- Proves that the initial interest rate is 5% given the problem conditions --/
theorem initial_interest_rate 
  (initial_investment : ℝ) 
  (additional_investment : ℝ) 
  (additional_rate : ℝ) 
  (total_rate : ℝ) 
  (h1 : initial_investment = 8000)
  (h2 : additional_investment = 4000)
  (h3 : additional_rate = 8)
  (h4 : total_rate = 6)
  : (initial_investment * (100 * total_rate - additional_investment * additional_rate) / 
    (100 * (initial_investment + additional_investment))) = 5 := by
  sorry

end initial_interest_rate_l864_86495


namespace consecutive_odd_numbers_l864_86401

theorem consecutive_odd_numbers (x : ℤ) : 
  (∃ y z : ℤ, y = x + 2 ∧ z = x + 4 ∧ 
   Odd x ∧ Odd y ∧ Odd z ∧
   11 * x = 3 * z + 4 * y + 16) → 
  x = 9 := by
sorry

end consecutive_odd_numbers_l864_86401


namespace birthday_money_calculation_l864_86457

def playstation_cost : ℝ := 500
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5
def games_to_sell : ℕ := 20

theorem birthday_money_calculation :
  let total_from_games : ℝ := game_price * (games_to_sell : ℝ)
  let remaining_money_needed : ℝ := playstation_cost - christmas_money - total_from_games
  remaining_money_needed = 200 := by sorry

end birthday_money_calculation_l864_86457


namespace arrangements_theorem_l864_86473

/-- The number of arrangements of 5 people in a row with exactly 1 person between A and B -/
def arrangements_count : ℕ := 36

/-- The total number of people in the row -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

/-- Theorem stating that the number of arrangements is 36 -/
theorem arrangements_theorem :
  (arrangements_count = 36) ∧
  (total_people = 5) ∧
  (people_between = 1) :=
sorry

end arrangements_theorem_l864_86473


namespace rectangle_dimensions_l864_86418

/-- Given a square tile with side length 10 dm, containing four identical rectangles and a small square,
    where the perimeter of the small square is five times smaller than the perimeter of the entire square,
    prove that the dimensions of the rectangles are 4 dm × 6 dm. -/
theorem rectangle_dimensions (tile_side : ℝ) (small_square_side : ℝ) (rect_short_side : ℝ) (rect_long_side : ℝ) :
  tile_side = 10 →
  small_square_side * 4 = tile_side * 4 / 5 →
  tile_side = small_square_side + 2 * rect_short_side →
  tile_side = rect_short_side + rect_long_side →
  rect_short_side = 4 ∧ rect_long_side = 6 :=
by sorry

end rectangle_dimensions_l864_86418


namespace train_speed_calculation_l864_86439

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length_train1 length_train2 speed_train2 crossing_time : ℝ) 
  (h1 : length_train1 = 150)
  (h2 : length_train2 = 350.04)
  (h3 : speed_train2 = 80)
  (h4 : crossing_time = 9)
  : ∃ (speed_train1 : ℝ), abs (speed_train1 - 120.016) < 0.001 := by
  sorry

end train_speed_calculation_l864_86439


namespace cube_root_unity_sum_l864_86411

/-- Given a nonreal cube root of unity ω, prove that (ω - 2ω^2 + 2)^4 + (2 + 2ω - ω^2)^4 = -257 -/
theorem cube_root_unity_sum (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (ω - 2*ω^2 + 2)^4 + (2 + 2*ω - ω^2)^4 = -257 := by
  sorry

end cube_root_unity_sum_l864_86411


namespace triangle_line_equations_l864_86434

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC, returns the equation of line AB -/
def line_AB (t : Triangle) : LineEquation :=
  { a := 3, b := 8, c := 15 }

/-- Given triangle ABC, returns the equation of the altitude from C to AB -/
def altitude_C (t : Triangle) : LineEquation :=
  { a := 8, b := -3, c := 6 }

theorem triangle_line_equations (t : Triangle) 
  (h1 : t.A = (-5, 0)) 
  (h2 : t.B = (3, -3)) 
  (h3 : t.C = (0, 2)) : 
  (line_AB t = { a := 3, b := 8, c := 15 }) ∧ 
  (altitude_C t = { a := 8, b := -3, c := 6 }) := by
  sorry

end triangle_line_equations_l864_86434


namespace min_distance_to_line_l864_86448

theorem min_distance_to_line (x y : ℝ) : 
  3 * x + 4 * y = 24 → x ≥ 0 → 
  ∃ (min_val : ℝ), min_val = 24 / 5 ∧ 
    ∀ (x' y' : ℝ), 3 * x' + 4 * y' = 24 → x' ≥ 0 → 
      Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_val := by
  sorry

end min_distance_to_line_l864_86448


namespace age_difference_proof_l864_86415

/-- Proves that the difference between twice John's current age and Tim's age is 15 years -/
theorem age_difference_proof (james_age_past : ℕ) (john_age_past : ℕ) (tim_age : ℕ) 
  (h1 : james_age_past = 23)
  (h2 : john_age_past = 35)
  (h3 : tim_age = 79)
  (h4 : ∃ (x : ℕ), tim_age + x = 2 * (john_age_past + (john_age_past - james_age_past))) :
  2 * (john_age_past + (john_age_past - james_age_past)) - tim_age = 15 := by
  sorry


end age_difference_proof_l864_86415


namespace circles_position_l864_86444

theorem circles_position (r₁ r₂ : ℝ) (h₁ : r₁ * r₂ = 3) (h₂ : r₁ + r₂ = 5) (h₃ : (r₁ - r₂)^2 = 13/4) :
  let d := 3
  r₁ + r₂ > d ∧ |r₁ - r₂| > d :=
by sorry

end circles_position_l864_86444


namespace arithmetic_sequence_sum_l864_86471

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 = 80) :
  a 1 + a 13 = 40 := by
  sorry

end arithmetic_sequence_sum_l864_86471


namespace ava_finishes_on_monday_l864_86435

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def days_after (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (days_after d n)

def reading_time (n : ℕ) : ℕ := 2 * n - 1

def total_reading_time (n : ℕ) : ℕ := 
  (List.range n).map reading_time |>.sum

theorem ava_finishes_on_monday : 
  days_after DayOfWeek.Sunday (total_reading_time 20) = DayOfWeek.Monday := by
  sorry


end ava_finishes_on_monday_l864_86435


namespace binomial_20_19_l864_86482

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end binomial_20_19_l864_86482


namespace board_sum_possible_l864_86428

theorem board_sum_possible : ∃ (a b : ℕ), 
  a ≤ 10 ∧ b ≤ 11 ∧ 
  (10 - a : ℝ) * 1.11 + (11 - b : ℝ) * 1.01 = 20.19 := by
sorry

end board_sum_possible_l864_86428


namespace sum_of_squares_remainder_l864_86433

theorem sum_of_squares_remainder (a b c d e : ℕ) (ha : a = 445876) (hb : b = 985420) (hc : c = 215546) (hd : d = 656452) (he : e = 387295) :
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 := by
sorry

end sum_of_squares_remainder_l864_86433


namespace sum_of_union_equals_31_l864_86437

def A : Finset ℕ := {2, 0, 1, 8}

def B : Finset ℕ := Finset.image (· * 2) A

theorem sum_of_union_equals_31 : (A ∪ B).sum id = 31 := by
  sorry

end sum_of_union_equals_31_l864_86437


namespace factory_B_is_better_l864_86468

/-- Represents a chicken leg factory --/
structure ChickenFactory where
  name : String
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ

/-- Determines if a factory is better based on its statistics --/
def isBetterFactory (f1 f2 : ChickenFactory) : Prop :=
  f1.mean = f2.mean ∧
  f1.variance < f2.variance ∧
  f1.median = f1.mean ∧
  f1.mode = f1.mean ∧
  (f2.median ≠ f2.mean ∨ f2.mode ≠ f2.mean)

/-- Factory A data --/
def factoryA : ChickenFactory :=
  { name := "A"
    mean := 75
    median := 74.5
    mode := 74
    variance := 3.4 }

/-- Factory B data --/
def factoryB : ChickenFactory :=
  { name := "B"
    mean := 75
    median := 75
    mode := 75
    variance := 2 }

/-- Theorem stating that Factory B is better than Factory A --/
theorem factory_B_is_better : isBetterFactory factoryB factoryA := by
  sorry

#check factory_B_is_better

end factory_B_is_better_l864_86468


namespace exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary_l864_86479

/-- Represents the outcome of tossing a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The sample space of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads),
   (CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads),
   (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event "Exactly one head is up" -/
def exactlyOneHead : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads)}

/-- The event "Exactly two heads are up" -/
def exactlyTwoHeads : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads)}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set TwoCoinsOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (A B : Set TwoCoinsOutcome) : Prop :=
  A ∪ B = sampleSpace

theorem exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary :
  mutuallyExclusive exactlyOneHead exactlyTwoHeads ∧
  ¬complementary exactlyOneHead exactlyTwoHeads :=
by sorry

end exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary_l864_86479


namespace fifteenth_number_base5_l864_86484

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The 15th number in base 5 counting system -/
def fifteenthNumberBase5 : List ℕ := toBase5 15

theorem fifteenth_number_base5 :
  fifteenthNumberBase5 = [3, 0] :=
sorry

end fifteenth_number_base5_l864_86484


namespace ice_cream_ratio_is_two_to_one_l864_86470

/-- The ratio of Victoria's ice cream scoops to Oli's ice cream scoops -/
def ice_cream_ratio : ℚ := by
  -- Define Oli's number of scoops
  let oli_scoops : ℕ := 4
  -- Define Victoria's number of scoops
  let victoria_scoops : ℕ := oli_scoops + 4
  -- Calculate the ratio
  exact (victoria_scoops : ℚ) / oli_scoops

/-- Theorem stating that the ice cream ratio is 2:1 -/
theorem ice_cream_ratio_is_two_to_one : ice_cream_ratio = 2 := by
  sorry

end ice_cream_ratio_is_two_to_one_l864_86470


namespace exponent_division_l864_86424

theorem exponent_division (a : ℝ) : a^10 / a^5 = a^5 := by
  sorry

end exponent_division_l864_86424


namespace average_after_removal_l864_86443

def originalList : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def removedNumber : ℕ := 1

def remainingList : List ℕ := originalList.filter (· ≠ removedNumber)

theorem average_after_removal :
  (remainingList.sum : ℚ) / remainingList.length = 15/2 := by sorry

end average_after_removal_l864_86443


namespace least_months_to_double_debt_l864_86429

def initial_amount : ℝ := 1200
def interest_rate : ℝ := 0.06

def compound_factor : ℝ := 1 + interest_rate

theorem least_months_to_double_debt : 
  (∀ n : ℕ, n < 12 → compound_factor ^ n ≤ 2) ∧ 
  compound_factor ^ 12 > 2 := by
  sorry

end least_months_to_double_debt_l864_86429


namespace base_b_square_l864_86403

theorem base_b_square (b : ℕ) : b > 0 → (3 * b + 3)^2 = b^3 + 2 * b^2 + 3 * b ↔ b = 9 := by
  sorry

end base_b_square_l864_86403


namespace divisibility_condition_l864_86407

theorem divisibility_condition (n : ℕ) : n ≥ 1 → (n ^ 2 ∣ 2 ^ n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end divisibility_condition_l864_86407


namespace expand_expression_l864_86414

theorem expand_expression (y : ℝ) : (11 * y + 18) * (3 * y) = 33 * y^2 + 54 * y := by
  sorry

end expand_expression_l864_86414


namespace simplify_expression_l864_86460

theorem simplify_expression (b : ℝ) (h : b ≠ 2) :
  2 - 2 / (2 + b / (2 - b)) = 4 / (4 - b) := by sorry

end simplify_expression_l864_86460


namespace three_points_determine_plane_line_and_point_determine_plane_trapezoid_determines_plane_circle_points_not_always_determine_plane_l864_86441

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- A circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D

/-- Three points are collinear if they lie on the same line -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- A point lies on a circle -/
def point_on_circle (p : Point3D) (c : Circle3D) : Prop := sorry

/-- Three points determine a unique plane -/
theorem three_points_determine_plane (p1 p2 p3 : Point3D) (h : ¬collinear p1 p2 p3) : 
  ∃! plane : Plane3D, point_on_plane p1 plane ∧ point_on_plane p2 plane ∧ point_on_plane p3 plane :=
sorry

/-- A line and a point not on the line determine a unique plane -/
theorem line_and_point_determine_plane (l : Line3D) (p : Point3D) (h : ¬point_on_line p l) :
  ∃! plane : Plane3D, (∀ q : Point3D, point_on_line q l → point_on_plane q plane) ∧ point_on_plane p plane :=
sorry

/-- A trapezoid determines a unique plane -/
theorem trapezoid_determines_plane (p1 p2 p3 p4 : Point3D) 
  (h1 : ∃ l1 l2 : Line3D, point_on_line p1 l1 ∧ point_on_line p2 l1 ∧ point_on_line p3 l2 ∧ point_on_line p4 l2) 
  (h2 : l1 ≠ l2) :
  ∃! plane : Plane3D, point_on_plane p1 plane ∧ point_on_plane p2 plane ∧ point_on_plane p3 plane ∧ point_on_plane p4 plane :=
sorry

/-- The center and two points on a circle do not always determine a unique plane -/
theorem circle_points_not_always_determine_plane :
  ∃ (c : Circle3D) (p1 p2 : Point3D), 
    point_on_circle p1 c ∧ point_on_circle p2 c ∧ 
    ¬(∃! plane : Plane3D, point_on_plane c.center plane ∧ point_on_plane p1 plane ∧ point_on_plane p2 plane) :=
sorry

end three_points_determine_plane_line_and_point_determine_plane_trapezoid_determines_plane_circle_points_not_always_determine_plane_l864_86441


namespace jack_euros_l864_86480

/-- Calculates the number of euros Jack has given his dollar amount, 
    the exchange rate, and his total amount in dollars. -/
def calculate_euros (dollars : ℕ) (exchange_rate : ℕ) (total : ℕ) : ℕ :=
  (total - dollars) / exchange_rate

/-- Proves that Jack has 36 euros given the problem conditions. -/
theorem jack_euros : calculate_euros 45 2 117 = 36 := by
  sorry

end jack_euros_l864_86480


namespace john_ate_12_ounces_l864_86423

/-- The amount of steak John ate given the original weight, burned portion, and eating percentage -/
def steak_eaten (original_weight : ℝ) (burned_portion : ℝ) (eating_percentage : ℝ) : ℝ :=
  (1 - burned_portion) * original_weight * eating_percentage

/-- Theorem stating that John ate 12 ounces of steak -/
theorem john_ate_12_ounces : 
  let original_weight : ℝ := 30
  let burned_portion : ℝ := 1/2
  let eating_percentage : ℝ := 0.8
  steak_eaten original_weight burned_portion eating_percentage = 12 := by
  sorry

end john_ate_12_ounces_l864_86423


namespace game_result_l864_86446

theorem game_result (x : ℝ) : ((x + 90) - 27 - x) * 11 / 3 = 231 := by
  sorry

end game_result_l864_86446


namespace percentage_calculation_l864_86420

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 100 → 
  (P / 100) * (3 / 5 * N) = 36 → 
  P = 60 := by
sorry

end percentage_calculation_l864_86420


namespace elf_nuts_problem_l864_86426

theorem elf_nuts_problem (nuts : Fin 10 → ℕ) 
  (h1 : (nuts 0) + (nuts 2) = 110)
  (h2 : (nuts 1) + (nuts 3) = 120)
  (h3 : (nuts 2) + (nuts 4) = 130)
  (h4 : (nuts 3) + (nuts 5) = 140)
  (h5 : (nuts 4) + (nuts 6) = 150)
  (h6 : (nuts 5) + (nuts 7) = 160)
  (h7 : (nuts 6) + (nuts 8) = 170)
  (h8 : (nuts 7) + (nuts 9) = 180)
  (h9 : (nuts 8) + (nuts 0) = 190)
  (h10 : (nuts 9) + (nuts 1) = 200) :
  nuts 5 = 55 := by
  sorry

end elf_nuts_problem_l864_86426


namespace weight_of_b_l864_86454

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 40 := by
sorry

end weight_of_b_l864_86454


namespace quadrilateral_area_l864_86494

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) :
  diagonal = 24 →
  offset1 = 9 →
  offset2 = 6 →
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 180 := by
  sorry

end quadrilateral_area_l864_86494


namespace ellipse_foci_coordinates_l864_86463

/-- 
Given an ellipse with equation mx^2 + ny^2 + mn = 0, where m < n < 0,
prove that the coordinates of its foci are (0, ±√(n-m)).
-/
theorem ellipse_foci_coordinates 
  (m n : ℝ) 
  (h1 : m < n) 
  (h2 : n < 0) : 
  let equation := fun (x y : ℝ) => m * x^2 + n * y^2 + m * n
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) ∧ 
      c^2 = n - m) :=
sorry

end ellipse_foci_coordinates_l864_86463


namespace sum_of_digits_up_to_1000_l864_86422

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of all numbers from 1 to 1000 is 13501 -/
theorem sum_of_digits_up_to_1000 : sumOfDigitsUpTo 1000 = 13501 := by sorry

end sum_of_digits_up_to_1000_l864_86422


namespace toms_books_l864_86453

/-- Given that Joan has 10 books and together with Tom they have 48 books,
    prove that Tom has 38 books. -/
theorem toms_books (joan_books : ℕ) (total_books : ℕ) (h1 : joan_books = 10) (h2 : total_books = 48) :
  total_books - joan_books = 38 := by
  sorry

end toms_books_l864_86453


namespace barbara_colored_paper_bundles_l864_86490

/-- Represents the number of sheets in different paper units -/
structure PaperUnits where
  sheets_per_bunch : ℕ
  sheets_per_bundle : ℕ
  sheets_per_heap : ℕ

/-- Represents the quantities of different types of paper -/
structure PaperQuantities where
  bunches_of_white : ℕ
  heaps_of_scrap : ℕ
  total_sheets_removed : ℕ

/-- Calculates the number of bundles of colored paper -/
def bundles_of_colored_paper (units : PaperUnits) (quantities : PaperQuantities) : ℕ :=
  let white_sheets := quantities.bunches_of_white * units.sheets_per_bunch
  let scrap_sheets := quantities.heaps_of_scrap * units.sheets_per_heap
  let colored_sheets := quantities.total_sheets_removed - (white_sheets + scrap_sheets)
  colored_sheets / units.sheets_per_bundle

/-- Theorem stating that Barbara found 3 bundles of colored paper -/
theorem barbara_colored_paper_bundles :
  let units := PaperUnits.mk 4 2 20
  let quantities := PaperQuantities.mk 2 5 114
  bundles_of_colored_paper units quantities = 3 := by
  sorry

end barbara_colored_paper_bundles_l864_86490


namespace barrel_division_exists_l864_86432

/-- Represents the fill state of a barrel -/
inductive BarrelState
  | Empty
  | Half
  | Full

/-- Represents a distribution of barrels to an heir -/
structure Distribution where
  empty : Nat
  half : Nat
  full : Nat

/-- Calculates the total wine in a distribution -/
def wineAmount (d : Distribution) : Nat :=
  d.full * 2 + d.half

/-- Checks if a distribution is valid (8 barrels total) -/
def isValidDistribution (d : Distribution) : Prop :=
  d.empty + d.half + d.full = 8

/-- Represents a complete division of barrels among three heirs -/
structure BarrelDivision where
  heir1 : Distribution
  heir2 : Distribution
  heir3 : Distribution

/-- Checks if a barrel division is valid -/
def isValidDivision (div : BarrelDivision) : Prop :=
  isValidDistribution div.heir1 ∧
  isValidDistribution div.heir2 ∧
  isValidDistribution div.heir3 ∧
  div.heir1.empty + div.heir2.empty + div.heir3.empty = 8 ∧
  div.heir1.half + div.heir2.half + div.heir3.half = 8 ∧
  div.heir1.full + div.heir2.full + div.heir3.full = 8 ∧
  wineAmount div.heir1 = wineAmount div.heir2 ∧
  wineAmount div.heir2 = wineAmount div.heir3

theorem barrel_division_exists : ∃ (div : BarrelDivision), isValidDivision div := by
  sorry

end barrel_division_exists_l864_86432


namespace malcolm_followers_difference_l864_86499

def malcolm_social_media (instagram_followers facebook_followers : ℕ) : Prop :=
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  ∃ (youtube_followers : ℕ),
    youtube_followers > tiktok_followers ∧
    instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 ∧
    youtube_followers - tiktok_followers = 510

theorem malcolm_followers_difference :
  malcolm_social_media 240 500 :=
by sorry

end malcolm_followers_difference_l864_86499


namespace square_side_differences_l864_86498

theorem square_side_differences (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > a₄)
  (diff₁ : a₁ - a₂ = 11) (diff₂ : a₂ - a₃ = 5) (diff₃ : a₃ - a₄ = 13) :
  a₁ - a₄ = 29 := by
sorry

end square_side_differences_l864_86498


namespace star_equation_solution_l864_86412

-- Define the star operation
noncomputable def star (x y : ℝ) : ℝ :=
  x + Real.sqrt (y + Real.sqrt (y + Real.sqrt y))

-- State the theorem
theorem star_equation_solution :
  ∃ h : ℝ, star 3 h = 8 ∧ h = 20 :=
by
  sorry

end star_equation_solution_l864_86412


namespace girls_in_class_l864_86408

theorem girls_in_class (boys : ℕ) (ways : ℕ) : boys = 15 → ways = 1050 → ∃ girls : ℕ,
  girls * (boys.choose 2) = ways ∧ girls = 10 := by sorry

end girls_in_class_l864_86408


namespace equation_with_72_l864_86472

/-- The first term of the nth equation in the sequence -/
def first_term (n : ℕ) : ℕ := 2 * n^2

/-- The equation number in which 72 appears as the first term -/
theorem equation_with_72 : {k : ℕ | first_term k = 72} = {6} := by sorry

end equation_with_72_l864_86472


namespace sam_distance_l864_86400

/-- Given Marguerite's travel details and Sam's driving time, prove Sam's distance traveled. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end sam_distance_l864_86400


namespace h1n1_vaccine_scientific_notation_l864_86461

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Checks if two ScientificNotation values are equal up to a certain number of significant figures -/
def equalUpToSigFigs (a b : ScientificNotation) (sigFigs : ℕ) : Prop :=
  sorry

theorem h1n1_vaccine_scientific_notation :
  equalUpToSigFigs (toScientificNotation (25.06 * 1000000) 3) 
                   { coefficient := 2.51, exponent := 7, is_valid := by sorry } 3 := by
  sorry

end h1n1_vaccine_scientific_notation_l864_86461


namespace sticker_distribution_l864_86409

/-- Represents the share of stickers each winner should receive -/
structure Share where
  al : Rat
  bert : Rat
  carl : Rat
  dan : Rat

/-- Calculates the remaining fraction of stickers after all winners have taken their perceived shares -/
def remaining_stickers (s : Share) : Rat :=
  let total := 1
  let bert_sees := total - s.al
  let carl_sees := bert_sees - (s.bert * bert_sees)
  let dan_sees := carl_sees - (s.carl * carl_sees)
  total - (s.al + s.bert * bert_sees + s.carl * carl_sees + s.dan * dan_sees)

/-- The theorem to be proved -/
theorem sticker_distribution (s : Share) 
  (h1 : s.al = 4/10)
  (h2 : s.bert = 3/10)
  (h3 : s.carl = 2/10)
  (h4 : s.dan = 1/10) :
  remaining_stickers s = 2844/10000 := by
  sorry

end sticker_distribution_l864_86409


namespace unique_n_existence_l864_86493

theorem unique_n_existence : ∃! n : ℤ,
  50 < n ∧ n < 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n % 4 = 1 ∧
  n = 105 := by
  sorry

end unique_n_existence_l864_86493


namespace arithmetic_sequence_first_term_l864_86438

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5)
  (h2 : sum_n seq 9 = 1) :
  seq.a 1 = -5/27 := by
sorry

end arithmetic_sequence_first_term_l864_86438


namespace rows_containing_47_l864_86492

-- Define Pascal's Triangle binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define a function to check if a number is in a row of Pascal's Triangle
def numberInRow (num row : ℕ) : Prop := sorry

-- Define primality
def isPrime (p : ℕ) : Prop := sorry

-- Theorem statement
theorem rows_containing_47 :
  isPrime 47 →
  (∃! row : ℕ, numberInRow 47 row) :=
by sorry

end rows_containing_47_l864_86492


namespace triangle_side_length_l864_86458

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = π / 3 →  -- 60° in radians
  B = π / 4 →  -- 45° in radians
  a = 3 →
  b = Real.sqrt 6 := by
  sorry

end triangle_side_length_l864_86458


namespace geometric_sequence_first_term_l864_86491

theorem geometric_sequence_first_term :
  ∀ (a r : ℝ),
    a * r^2 = 720 →
    a * r^5 = 5040 →
    a = 720 / 7^(2/3) :=
by
  sorry

end geometric_sequence_first_term_l864_86491


namespace exists_decreasing_function_always_ge_one_l864_86404

theorem exists_decreasing_function_always_ge_one :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x > f y) ∧ (∀ x : ℝ, f x ≥ 1) := by
  sorry

end exists_decreasing_function_always_ge_one_l864_86404


namespace factor_expression_1_factor_expression_2_l864_86402

-- For the first expression
theorem factor_expression_1 (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) := by
  sorry

-- For the second expression
theorem factor_expression_2 (x a : ℝ) :
  x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - 3 - a) := by
  sorry

end factor_expression_1_factor_expression_2_l864_86402


namespace quadratic_inequality_solution_set_l864_86447

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 5 * x - 2 < 0} = {x : ℝ | -1/3 < x ∧ x < 2} := by
  sorry

end quadratic_inequality_solution_set_l864_86447


namespace basketball_shooting_test_l864_86462

-- Define the probabilities of making a basket for students A and B
def prob_A : ℚ := 1/2
def prob_B : ℚ := 2/3

-- Define the number of shots for Part I
def shots_part_I : ℕ := 3

-- Define the number of chances for Part II
def chances_part_II : ℕ := 4

-- Define the function to calculate the probability of exactly k successes in n trials
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Define the probability of student A meeting the standard in Part I
def prob_A_meets_standard : ℚ :=
  binomial_probability shots_part_I 2 prob_A + binomial_probability shots_part_I 3 prob_A

-- Define the probability distribution of X (number of shots taken by B) in Part II
def prob_X (x : ℕ) : ℚ :=
  if x = 2 then prob_B^2
  else if x = 3 then prob_B * (1-prob_B) * prob_B + prob_B^2 * (1-prob_B) + (1-prob_B)^3
  else if x = 4 then (1-prob_B) * prob_B^2 + prob_B * (1-prob_B) * prob_B
  else 0

-- Define the expected value of X
def expected_X : ℚ :=
  2 * prob_X 2 + 3 * prob_X 3 + 4 * prob_X 4

-- Theorem statement
theorem basketball_shooting_test :
  prob_A_meets_standard = 1/2 ∧ expected_X = 25/9 := by sorry

end basketball_shooting_test_l864_86462


namespace cos_pi_sixth_eq_sin_shifted_l864_86497

theorem cos_pi_sixth_eq_sin_shifted (x : ℝ) : 
  Real.cos x + π/6 = Real.sin (x + 2*π/3) := by sorry

end cos_pi_sixth_eq_sin_shifted_l864_86497


namespace jongkooks_milk_consumption_l864_86455

/-- Converts liters to milliliters -/
def liters_to_ml (l : ℚ) : ℚ := 1000 * l

/-- Represents the amount of milk drunk in milliliters for each day -/
structure MilkConsumption where
  day1 : ℚ
  day2 : ℚ
  day3 : ℚ

/-- Calculates the total milk consumption in milliliters -/
def total_consumption (mc : MilkConsumption) : ℚ :=
  mc.day1 + mc.day2 + mc.day3

theorem jongkooks_milk_consumption :
  ∃ (mc : MilkConsumption),
    mc.day1 = liters_to_ml 3 + 7 ∧
    mc.day3 = 840 ∧
    total_consumption mc = liters_to_ml 6 + 30 ∧
    mc.day2 = 2183 := by
  sorry

end jongkooks_milk_consumption_l864_86455
