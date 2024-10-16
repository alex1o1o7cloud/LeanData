import Mathlib

namespace NUMINAMATH_CALUDE_poojas_speed_l2760_276034

/-- Proves that Pooja's speed is 3 km/hr given the conditions of the problem -/
theorem poojas_speed (roja_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  roja_speed = 7 →
  time = 4 →
  distance = 40 →
  ∃ pooja_speed : ℝ, pooja_speed = 3 ∧ (roja_speed + pooja_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_poojas_speed_l2760_276034


namespace NUMINAMATH_CALUDE_mothers_age_l2760_276001

/-- Proves that the mother's age is 42 given the conditions of the problem -/
theorem mothers_age (daughter_age : ℕ) (future_years : ℕ) (mother_age : ℕ) : 
  daughter_age = 8 →
  future_years = 9 →
  mother_age + future_years = 3 * (daughter_age + future_years) →
  mother_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l2760_276001


namespace NUMINAMATH_CALUDE_fair_ace_draw_l2760_276035

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)

/-- Represents the game setup -/
structure Game :=
  (deck : Deck)
  (num_players : ℕ)

/-- The probability of a player getting the first ace -/
def prob_first_ace (g : Game) (player : ℕ) : ℚ :=
  1 / g.num_players

theorem fair_ace_draw (g : Game) (player : ℕ) : 
  g.deck.total_cards = 32 ∧ 
  g.deck.num_aces = 4 ∧ 
  g.num_players = 4 ∧ 
  player > 0 ∧ 
  player ≤ g.num_players →
  prob_first_ace g player = 1/8 := by
  sorry

#check fair_ace_draw

end NUMINAMATH_CALUDE_fair_ace_draw_l2760_276035


namespace NUMINAMATH_CALUDE_sheep_count_l2760_276036

/-- Represents the number of animals on a boat and their fate after capsizing -/
structure BoatAnimals where
  sheep : ℕ
  cows : ℕ
  dogs : ℕ
  drownedSheep : ℕ
  drownedCows : ℕ
  survivedAnimals : ℕ

/-- Theorem stating the number of sheep on the boat given the conditions -/
theorem sheep_count (b : BoatAnimals) : b.sheep = 20 :=
  by
  have h1 : b.cows = 10 := sorry
  have h2 : b.dogs = 14 := sorry
  have h3 : b.drownedSheep = 3 := sorry
  have h4 : b.drownedCows = 2 * b.drownedSheep := sorry
  have h5 : b.survivedAnimals = 35 := sorry
  have h6 : b.survivedAnimals = b.sheep - b.drownedSheep + b.cows - b.drownedCows + b.dogs := sorry
  sorry

#check sheep_count

end NUMINAMATH_CALUDE_sheep_count_l2760_276036


namespace NUMINAMATH_CALUDE_scatter_plot_suitable_for_linear_relationship_only_scatter_plot_suitable_for_linear_relationship_l2760_276055

/-- A type representing different types of plots --/
inductive PlotType
  | ScatterPlot
  | StemAndLeafPlot
  | FrequencyDistributionHistogram
  | FrequencyDistributionLineChart

/-- A function that determines if a plot type is suitable for identifying linear relationships --/
def isSuitableForLinearRelationship (plot : PlotType) : Prop :=
  match plot with
  | PlotType.ScatterPlot => true
  | _ => false

/-- Theorem stating that a scatter plot is suitable for identifying linear relationships --/
theorem scatter_plot_suitable_for_linear_relationship :
  isSuitableForLinearRelationship PlotType.ScatterPlot :=
sorry

/-- Theorem stating that a scatter plot is the only suitable plot type for identifying linear relationships --/
theorem only_scatter_plot_suitable_for_linear_relationship (plot : PlotType) :
  isSuitableForLinearRelationship plot → plot = PlotType.ScatterPlot :=
sorry

end NUMINAMATH_CALUDE_scatter_plot_suitable_for_linear_relationship_only_scatter_plot_suitable_for_linear_relationship_l2760_276055


namespace NUMINAMATH_CALUDE_jerky_order_fulfillment_l2760_276074

/-- The number of days needed to fulfill a jerky order -/
def days_to_fulfill_order (bags_per_batch : ℕ) (order_size : ℕ) (bags_in_stock : ℕ) : ℕ :=
  let bags_to_make := order_size - bags_in_stock
  (bags_to_make + bags_per_batch - 1) / bags_per_batch

theorem jerky_order_fulfillment :
  days_to_fulfill_order 10 60 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerky_order_fulfillment_l2760_276074


namespace NUMINAMATH_CALUDE_power_product_equality_l2760_276080

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7 * 11 = 277200 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2760_276080


namespace NUMINAMATH_CALUDE_complex_equation_product_l2760_276060

/-- Given (1+3i)(a+bi) = 10i, where i is the imaginary unit and a, b ∈ ℝ, prove that ab = 3 -/
theorem complex_equation_product (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 + 3 * Complex.I) * (a + b * Complex.I) = 10 * Complex.I →
  a * b = 3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_product_l2760_276060


namespace NUMINAMATH_CALUDE_max_distance_from_unit_circle_to_point_l2760_276069

theorem max_distance_from_unit_circle_to_point (z : ℂ) :
  Complex.abs z = 1 →
  (∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w - (3 + 4*I)) ≤ Complex.abs (z - (3 + 4*I))) →
  Complex.abs (z - (3 + 4*I)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_unit_circle_to_point_l2760_276069


namespace NUMINAMATH_CALUDE_cuboid_edge_lengths_l2760_276012

theorem cuboid_edge_lengths :
  ∀ a b c : ℕ,
  (a * b * c + a * b + b * c + c * a + a + b + c = 2000) →
  ({a, b, c} : Finset ℕ) = {28, 22, 2} := by
sorry

end NUMINAMATH_CALUDE_cuboid_edge_lengths_l2760_276012


namespace NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l2760_276043

/-- Arithmetic sequence sum function -/
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Product of n and S_n -/
def nSn (a₁ d : ℚ) (n : ℕ) : ℚ := n * S a₁ d n

theorem min_nSn_arithmetic_sequence :
  ∃ (a₁ d : ℚ),
    S a₁ d 10 = 0 ∧
    S a₁ d 15 = 25 ∧
    (∀ (n : ℕ), n > 0 → nSn a₁ d n ≥ -48) ∧
    (∃ (n : ℕ), n > 0 ∧ nSn a₁ d n = -48) := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l2760_276043


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l2760_276007

theorem ellipse_fixed_point_intersection :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  k ≠ 0 →
  A ≠ (2, 0) →
  B ≠ (2, 0) →
  A.1^2 / 4 + A.2^2 / 3 = 1 →
  B.1^2 / 4 + B.2^2 / 3 = 1 →
  A.2 = k * (A.1 - 2/7) →
  B.2 = k * (B.1 - 2/7) →
  (A.1 - 2)^2 + A.2^2 = (B.1 - 2)^2 + B.2^2 →
  (A.1 - 2)^2 + A.2^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  ∃ (m : ℝ), A.2 = k * (A.1 - 2/7) ∧ B.2 = k * (B.1 - 2/7) := by
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l2760_276007


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2760_276022

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  1/a + 2/b + 9/c + 8/d + 18/e + 32/f ≥ 24 ∧ 
  ∃ (a' b' c' d' e' f' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 9 ∧
    1/a' + 2/b' + 9/c' + 8/d' + 18/e' + 32/f' = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2760_276022


namespace NUMINAMATH_CALUDE_last_problem_number_l2760_276058

theorem last_problem_number (start : ℕ) (problems_solved : ℕ) : 
  start = 80 → problems_solved = 46 → start + problems_solved - 1 = 125 := by
sorry

end NUMINAMATH_CALUDE_last_problem_number_l2760_276058


namespace NUMINAMATH_CALUDE_kiwi_weight_l2760_276085

theorem kiwi_weight (total_weight : ℝ) (apple_weight : ℝ) (orange_percent : ℝ) (strawberry_kiwi_percent : ℝ) (strawberry_kiwi_ratio : ℝ) :
  total_weight = 400 →
  apple_weight = 80 →
  orange_percent = 0.15 →
  strawberry_kiwi_percent = 0.40 →
  strawberry_kiwi_ratio = 3 / 5 →
  ∃ kiwi_weight : ℝ,
    kiwi_weight = 100 ∧
    kiwi_weight + (strawberry_kiwi_ratio * kiwi_weight) = strawberry_kiwi_percent * total_weight ∧
    kiwi_weight + (strawberry_kiwi_ratio * kiwi_weight) + (orange_percent * total_weight) + apple_weight = total_weight :=
by
  sorry

end NUMINAMATH_CALUDE_kiwi_weight_l2760_276085


namespace NUMINAMATH_CALUDE_smallest_a_parabola_l2760_276086

/-- The smallest possible value of 'a' for a parabola with specific conditions -/
theorem smallest_a_parabola : 
  ∀ (a b c : ℝ), 
  (∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ x = 3/2 ∧ y = -1/4) →  -- vertex condition
  (a > 0) →  -- a is positive
  (∃ (n : ℤ), 2*a + b + 3*c = n) →  -- 2a + b + 3c is an integer
  (∀ (a' : ℝ), 
    (∃ (b' c' : ℝ), 
      (∃ (x y : ℝ), y = a' * x^2 + b' * x + c' ∧ x = 3/2 ∧ y = -1/4) ∧
      (a' > 0) ∧
      (∃ (n : ℤ), 2*a' + b' + 3*c' = n)) → 
    a ≤ a') →
  a = 3/23 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_parabola_l2760_276086


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l2760_276057

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 110₂ -/
def binary_110 : List Bool := [false, true, true]

theorem binary_110_equals_6 :
  binary_to_decimal binary_110 = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l2760_276057


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l2760_276027

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l2760_276027


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2760_276076

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_sum : i + i^2 + i^3 = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2760_276076


namespace NUMINAMATH_CALUDE_cube_remainder_sum_quotient_l2760_276072

def cube_rem_16 (n : ℕ) : ℕ := (n^3) % 16

def distinct_remainders : Finset ℕ :=
  (Finset.range 15).image cube_rem_16

theorem cube_remainder_sum_quotient :
  (Finset.sum distinct_remainders id) / 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_sum_quotient_l2760_276072


namespace NUMINAMATH_CALUDE_polynomial_real_root_l2760_276008

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x + 1

/-- The theorem statement -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≤ -1.5 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l2760_276008


namespace NUMINAMATH_CALUDE_weston_penalty_kicks_l2760_276096

/-- Calculates the number of penalty kicks required for a football drill -/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: The number of penalty kicks for Weston Junior Football Club's drill is 92 -/
theorem weston_penalty_kicks :
  penalty_kicks 24 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_weston_penalty_kicks_l2760_276096


namespace NUMINAMATH_CALUDE_shearer_payment_l2760_276049

/-- Given the following conditions:
  - The number of sheep is 200
  - Each sheep produces 10 pounds of wool
  - The price of wool is $20 per pound
  - The profit is $38000
  Prove that the amount paid to the shearer is $2000 -/
theorem shearer_payment (num_sheep : ℕ) (wool_per_sheep : ℕ) (wool_price : ℕ) (profit : ℕ) :
  num_sheep = 200 →
  wool_per_sheep = 10 →
  wool_price = 20 →
  profit = 38000 →
  num_sheep * wool_per_sheep * wool_price - profit = 2000 := by
  sorry

end NUMINAMATH_CALUDE_shearer_payment_l2760_276049


namespace NUMINAMATH_CALUDE_multiples_imply_lower_bound_l2760_276090

theorem multiples_imply_lower_bound (n : ℕ) (a : ℕ) (h1 : n > 1) (h2 : a > n^2) :
  (∀ i ∈ Finset.range n, ∃ k ∈ Finset.range n, (a + k + 1) % (n^2 + i + 1) = 0) →
  a > n^4 - n^3 := by
  sorry

end NUMINAMATH_CALUDE_multiples_imply_lower_bound_l2760_276090


namespace NUMINAMATH_CALUDE_construction_paper_count_l2760_276091

/-- Represents the number of sheets in a pack of construction paper -/
structure ConstructionPaper where
  blue : ℕ
  red : ℕ

/-- Represents the daily usage of construction paper -/
structure DailyUsage where
  blue : ℕ
  red : ℕ

def initial_ratio (pack : ConstructionPaper) : Prop :=
  pack.blue * 7 = pack.red * 2

def daily_usage : DailyUsage :=
  { blue := 1, red := 3 }

def last_day_usage : DailyUsage :=
  { blue := 1, red := 3 }

def remaining_red : ℕ := 15

theorem construction_paper_count :
  ∃ (pack : ConstructionPaper),
    initial_ratio pack ∧
    ∃ (days : ℕ),
      pack.blue = daily_usage.blue * days + last_day_usage.blue ∧
      pack.red = daily_usage.red * days + last_day_usage.red + remaining_red ∧
      pack.blue + pack.red = 135 :=
sorry

end NUMINAMATH_CALUDE_construction_paper_count_l2760_276091


namespace NUMINAMATH_CALUDE_employee_pay_l2760_276065

theorem employee_pay (total : ℝ) (ratio : ℝ) (lower_pay : ℝ) : 
  total = 580 →
  ratio = 1.5 →
  total = lower_pay + ratio * lower_pay →
  lower_pay = 232 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l2760_276065


namespace NUMINAMATH_CALUDE_max_boat_distance_xiaohu_max_distance_l2760_276070

/-- Calculates the maximum distance a boat can travel in a river with given conditions --/
theorem max_boat_distance (total_time : ℝ) (boat_speed : ℝ) (current_speed : ℝ) 
  (paddle_time : ℝ) (break_time : ℝ) : ℝ :=
  let total_minutes : ℝ := total_time * 60
  let cycle_time : ℝ := paddle_time + break_time
  let num_cycles : ℝ := total_minutes / cycle_time
  let total_break_time : ℝ := num_cycles * break_time
  let effective_paddle_time : ℝ := total_minutes - total_break_time - total_break_time
  let upstream_speed : ℝ := boat_speed - current_speed
  let downstream_speed : ℝ := boat_speed + current_speed
  let downstream_ratio : ℝ := downstream_speed / (upstream_speed + downstream_speed)
  let downstream_paddle_time : ℝ := downstream_ratio * effective_paddle_time
  let downstream_distance : ℝ := downstream_speed * (downstream_paddle_time / 60)
  let drift_distance : ℝ := current_speed * (break_time / 60)
  downstream_distance + drift_distance

/-- Proves that the maximum distance Xiaohu can be from the rental place is 1.375 km --/
theorem xiaohu_max_distance : 
  max_boat_distance 2 3 1.5 30 10 = 1.375 := by sorry

end NUMINAMATH_CALUDE_max_boat_distance_xiaohu_max_distance_l2760_276070


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2760_276098

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) = r₁ + r₂

theorem circles_externally_tangent :
  let c₁ : ℝ × ℝ := (0, 8)
  let c₂ : ℝ × ℝ := (-6, 0)
  let r₁ : ℝ := 6
  let r₂ : ℝ := 2
  externally_tangent c₁ c₂ r₁ r₂ := by
  sorry

#check circles_externally_tangent

end NUMINAMATH_CALUDE_circles_externally_tangent_l2760_276098


namespace NUMINAMATH_CALUDE_one_carton_per_case_l2760_276093

/-- Given that each carton contains b boxes, each box contains 200 paper clips,
    and 400 paper clips are contained in 2 cases, prove that there is 1 carton in a case. -/
theorem one_carton_per_case (b : ℕ) (h1 : b > 0) :
  (∃ c : ℕ, c > 0 ∧ c * b * 200 = 200) → c = 1 :=
by sorry

end NUMINAMATH_CALUDE_one_carton_per_case_l2760_276093


namespace NUMINAMATH_CALUDE_set_A_representation_l2760_276030

def A : Set (ℝ × ℝ) := {(x, y) | 3 * x + y = 11 ∧ x - y = 1}

theorem set_A_representation : A = {(3, 2)} := by sorry

end NUMINAMATH_CALUDE_set_A_representation_l2760_276030


namespace NUMINAMATH_CALUDE_prob_A_nth_day_l2760_276097

/-- The probability of switching restaurants each day -/
def switch_prob : ℝ := 0.6

/-- The probability of choosing restaurant A on the n-th day -/
def prob_A (n : ℕ) : ℝ := 0.5 + 0.5 * (-0.2)^(n - 1)

/-- Theorem stating the probability of choosing restaurant A on the n-th day -/
theorem prob_A_nth_day (n : ℕ) :
  prob_A n = 0.5 + 0.5 * (-0.2)^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prob_A_nth_day_l2760_276097


namespace NUMINAMATH_CALUDE_fox_weasel_hunting_average_l2760_276006

/-- Proves that given the initial conditions and the number of animals remaining after 3 weeks,
    the average number of weasels caught by each fox per week is 4. -/
theorem fox_weasel_hunting_average :
  let initial_weasels : ℕ := 100
  let initial_rabbits : ℕ := 50
  let num_foxes : ℕ := 3
  let rabbits_per_fox_per_week : ℕ := 2
  let weeks : ℕ := 3
  let animals_left : ℕ := 96
  let total_animals_caught := initial_weasels + initial_rabbits - animals_left
  let total_rabbits_caught := num_foxes * rabbits_per_fox_per_week * weeks
  let total_weasels_caught := total_animals_caught - total_rabbits_caught
  let weasels_per_fox := total_weasels_caught / num_foxes
  let avg_weasels_per_fox_per_week := weasels_per_fox / weeks
  avg_weasels_per_fox_per_week = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_fox_weasel_hunting_average_l2760_276006


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2760_276031

theorem last_two_digits_sum (n : ℕ) : (9^n + 11^n) % 100 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2760_276031


namespace NUMINAMATH_CALUDE_green_tea_profit_maximization_l2760_276025

/-- The profit function for a green tea company -/
def profit (x : ℝ) : ℝ := -2 * x^2 + 340 * x - 12000

/-- The selling price that maximizes profit -/
def max_profit_price : ℝ := 85

theorem green_tea_profit_maximization :
  /- The profit function is correct -/
  (∀ x : ℝ, profit x = -2 * x^2 + 340 * x - 12000) ∧
  /- The maximum profit occurs at x = 85 -/
  (∀ x : ℝ, profit x ≤ profit max_profit_price) := by
  sorry


end NUMINAMATH_CALUDE_green_tea_profit_maximization_l2760_276025


namespace NUMINAMATH_CALUDE_debby_deleted_pictures_l2760_276000

theorem debby_deleted_pictures (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 24)
  (h2 : museum_pics = 12)
  (h3 : remaining_pics = 22) :
  zoo_pics + museum_pics - remaining_pics = 14 := by
  sorry

end NUMINAMATH_CALUDE_debby_deleted_pictures_l2760_276000


namespace NUMINAMATH_CALUDE_simplify_expression_l2760_276048

-- Define the left-hand side of the equation
def lhs (y : ℝ) : ℝ := 3*y + 4*y^2 + 2 - (8 - 3*y - 4*y^2 + y^3)

-- Define the right-hand side of the equation
def rhs (y : ℝ) : ℝ := -y^3 + 8*y^2 + 6*y - 6

-- Theorem statement
theorem simplify_expression (y : ℝ) : lhs y = rhs y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2760_276048


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_geq_neg_two_l2760_276092

theorem inequality_holds_iff_a_geq_neg_two :
  ∀ a : ℝ, (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_geq_neg_two_l2760_276092


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2760_276084

theorem quadratic_inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0) ↔ m ∈ Set.Ici 1 ∩ Set.Iio 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2760_276084


namespace NUMINAMATH_CALUDE_Q_proper_subset_P_l2760_276044

def P : Set ℝ := {x : ℝ | x ≥ 1}
def Q : Set ℝ := {2, 3}

theorem Q_proper_subset_P : Q ⊂ P := by sorry

end NUMINAMATH_CALUDE_Q_proper_subset_P_l2760_276044


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l2760_276056

/-- The number of unique arrangements of the letters in MISSISSIPPI -/
def mississippiArrangements : ℕ := 34650

/-- The total number of letters in MISSISSIPPI -/
def totalLetters : ℕ := 11

/-- The number of occurrences of 'I' in MISSISSIPPI -/
def countI : ℕ := 4

/-- The number of occurrences of 'S' in MISSISSIPPI -/
def countS : ℕ := 4

/-- The number of occurrences of 'P' in MISSISSIPPI -/
def countP : ℕ := 2

/-- The number of occurrences of 'M' in MISSISSIPPI -/
def countM : ℕ := 1

theorem mississippi_arrangements :
  mississippiArrangements = Nat.factorial totalLetters / (Nat.factorial countI * Nat.factorial countS * Nat.factorial countP) :=
by sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l2760_276056


namespace NUMINAMATH_CALUDE_car_speed_problem_l2760_276071

/-- Theorem: Given two cars starting from opposite ends of a 60-mile highway at the same time,
    with one car traveling at speed x mph and the other at 17 mph, if they meet after 2 hours,
    then the speed x of the first car is 13 mph. -/
theorem car_speed_problem (x : ℝ) :
  (x > 0) →  -- Assuming positive speed for the first car
  (2 * x + 2 * 17 = 60) →  -- Distance traveled by both cars equals highway length
  x = 13 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2760_276071


namespace NUMINAMATH_CALUDE_fourth_quarter_total_points_l2760_276028

/-- Represents the points scored by a team in each quarter -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- The game between Raiders and Wildcats -/
structure BasketballGame :=
  (raiders : TeamScores)
  (wildcats : TeamScores)

/-- Conditions of the game -/
def game_conditions (g : BasketballGame) : Prop :=
  let r := g.raiders
  let w := g.wildcats
  -- Game tied at halftime
  r.q1 + r.q2 = w.q1 + w.q2 ∧
  -- Raiders' points form an increasing arithmetic sequence
  ∃ (d : ℕ), r.q2 = r.q1 + d ∧ r.q3 = r.q2 + d ∧ r.q4 = r.q3 + d ∧
  -- Wildcats' points are equal in first two quarters, then decrease by same difference
  ∃ (j : ℕ), w.q1 = w.q2 ∧ w.q3 = w.q2 - j ∧ w.q4 = w.q3 - j ∧
  -- Wildcats won by exactly four points
  (w.q1 + w.q2 + w.q3 + w.q4) = (r.q1 + r.q2 + r.q3 + r.q4) + 4

theorem fourth_quarter_total_points (g : BasketballGame) :
  game_conditions g → g.raiders.q4 + g.wildcats.q4 = 28 :=
by sorry

end NUMINAMATH_CALUDE_fourth_quarter_total_points_l2760_276028


namespace NUMINAMATH_CALUDE_birds_in_tree_l2760_276099

theorem birds_in_tree (swallows bluebirds cardinals : ℕ) : 
  swallows = 2 →
  bluebirds = 2 * swallows →
  cardinals = 3 * bluebirds →
  swallows + bluebirds + cardinals = 18 :=
by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2760_276099


namespace NUMINAMATH_CALUDE_engine_capacity_proof_l2760_276052

/-- Represents the relationship between diesel volume, distance, and engine capacity -/
structure DieselEngineRelation where
  volume : ℝ  -- Volume of diesel in litres
  distance : ℝ  -- Distance in km
  capacity : ℝ  -- Engine capacity in cc

/-- The relation between diesel volume and engine capacity is directly proportional -/
axiom diesel_capacity_proportion (r1 r2 : DieselEngineRelation) :
  r1.volume / r1.capacity = r2.volume / r2.capacity

/-- Given data for the first scenario -/
def scenario1 : DieselEngineRelation :=
  { volume := 60, distance := 600, capacity := 800 }

/-- Given data for the second scenario -/
def scenario2 : DieselEngineRelation :=
  { volume := 120, distance := 800, capacity := 1600 }

/-- Theorem stating that the engine capacity for the second scenario is 1600 cc -/
theorem engine_capacity_proof :
  scenario2.capacity = 1600 :=
by sorry

end NUMINAMATH_CALUDE_engine_capacity_proof_l2760_276052


namespace NUMINAMATH_CALUDE_colonization_combinations_l2760_276089

/-- The number of habitable planets --/
def total_planets : ℕ := 18

/-- The number of Earth-like planets --/
def earth_like_planets : ℕ := 9

/-- The number of Mars-like planets --/
def mars_like_planets : ℕ := 9

/-- The resource units required to colonize an Earth-like planet --/
def earth_like_resource : ℕ := 3

/-- The resource units required to colonize a Mars-like planet --/
def mars_like_resource : ℕ := 2

/-- The total resource units available for colonization --/
def total_resources : ℕ := 27

/-- The number of Earth-like planets that can be colonized --/
def colonized_earth_like : ℕ := 7

/-- The number of Mars-like planets that can be colonized --/
def colonized_mars_like : ℕ := 3

theorem colonization_combinations : 
  (Nat.choose earth_like_planets colonized_earth_like) * 
  (Nat.choose mars_like_planets colonized_mars_like) = 3024 :=
by sorry

end NUMINAMATH_CALUDE_colonization_combinations_l2760_276089


namespace NUMINAMATH_CALUDE_common_solution_for_all_a_l2760_276023

/-- The linear equation (a-3)x + (2a-5)y + 6-a = 0 has a common solution (7, -3) for all values of a. -/
theorem common_solution_for_all_a :
  ∀ (a : ℝ), (a - 3) * 7 + (2 * a - 5) * (-3) + 6 - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_solution_for_all_a_l2760_276023


namespace NUMINAMATH_CALUDE_soda_survey_result_l2760_276075

/-- The number of people who chose "Soda" in a survey of 520 people,
    where the central angle of the "Soda" sector is 270° (to the nearest whole degree). -/
def soda_count : ℕ := 390

/-- The total number of people surveyed. -/
def total_surveyed : ℕ := 520

/-- The central angle of the "Soda" sector in degrees. -/
def soda_angle : ℕ := 270

theorem soda_survey_result :
  (soda_count : ℚ) / total_surveyed * 360 ≥ soda_angle - (1/2 : ℚ) ∧
  (soda_count : ℚ) / total_surveyed * 360 < soda_angle + (1/2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_soda_survey_result_l2760_276075


namespace NUMINAMATH_CALUDE_chessboard_probability_l2760_276016

theorem chessboard_probability (k : ℕ) : k ≥ 5 →
  (((k - 4)^2 - 1) / (2 * (k - 4)^2 : ℚ) = 48 / 100) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_probability_l2760_276016


namespace NUMINAMATH_CALUDE_rectangle_area_l2760_276021

theorem rectangle_area (x : ℝ) (h : x > 0) :
  ∃ (w : ℝ), w > 0 ∧ 
  w^2 + (3*w)^2 = x^2 ∧ 
  3*w^2 = (3/10)*x^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2760_276021


namespace NUMINAMATH_CALUDE_odd_function_implies_m_n_equal_one_f_is_decreasing_k_range_l2760_276077

noncomputable def f (m n x : ℝ) : ℝ := (m - 3^x) / (n + 3^x)

theorem odd_function_implies_m_n_equal_one 
  (h : ∀ x, f m n x = -f m n (-x)) : m = 1 ∧ n = 1 := by sorry

theorem f_is_decreasing : 
  ∀ x y, x < y → f 1 1 x > f 1 1 y := by sorry

theorem k_range (t : ℝ) (h1 : t ∈ Set.Icc 0 4) 
  (h2 : f 1 1 (k - 2*t^2) + f 1 1 (4*t - 2*t^2) < 0) : 
  k > -1 := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_n_equal_one_f_is_decreasing_k_range_l2760_276077


namespace NUMINAMATH_CALUDE_smallest_N_value_l2760_276087

/-- Represents a point in the rectangular array -/
structure Point where
  row : Nat
  col : Nat

/-- The original numbering function -/
def original_number (N : Nat) (p : Point) : Nat :=
  (p.row - 1) * N + p.col

/-- The new numbering function -/
def new_number (p : Point) : Nat :=
  5 * (p.col - 1) + p.row

/-- The theorem stating the smallest possible value of N -/
theorem smallest_N_value : ∃ (N : Nat) (p₁ p₂ p₃ p₄ p₅ : Point),
  N > 0 ∧
  p₁.row = 1 ∧ p₂.row = 2 ∧ p₃.row = 3 ∧ p₄.row = 4 ∧ p₅.row = 5 ∧
  p₁.col ≤ N ∧ p₂.col ≤ N ∧ p₃.col ≤ N ∧ p₄.col ≤ N ∧ p₅.col ≤ N ∧
  original_number N p₁ = new_number p₂ ∧
  original_number N p₂ = new_number p₁ ∧
  original_number N p₃ = new_number p₄ ∧
  original_number N p₄ = new_number p₅ ∧
  original_number N p₅ = new_number p₃ ∧
  (∀ (M : Nat) (q₁ q₂ q₃ q₄ q₅ : Point),
    M > 0 ∧
    q₁.row = 1 ∧ q₂.row = 2 ∧ q₃.row = 3 ∧ q₄.row = 4 ∧ q₅.row = 5 ∧
    q₁.col ≤ M ∧ q₂.col ≤ M ∧ q₃.col ≤ M ∧ q₄.col ≤ M ∧ q₅.col ≤ M ∧
    original_number M q₁ = new_number q₂ ∧
    original_number M q₂ = new_number q₁ ∧
    original_number M q₃ = new_number q₄ ∧
    original_number M q₄ = new_number q₅ ∧
    original_number M q₅ = new_number q₃ →
    M ≥ N) ∧
  N = 149 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_value_l2760_276087


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_eight_l2760_276013

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define the triangle inequality
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the quadratic equation
def is_root_of_equation (x : ℝ) : Prop :=
  x^2 - 4*x + 3 = 0

-- Theorem statement
theorem triangle_perimeter_is_eight :
  ∃ (t : Triangle), t.a = 2 ∧ t.b = 3 ∧ 
  is_root_of_equation t.c ∧ 
  is_valid_triangle t ∧
  perimeter t = 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_eight_l2760_276013


namespace NUMINAMATH_CALUDE_compute_expression_l2760_276053

theorem compute_expression : (12 : ℚ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2760_276053


namespace NUMINAMATH_CALUDE_apple_cost_price_l2760_276051

theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 18 →
  loss_fraction = 1/6 →
  selling_price = cost_price - (loss_fraction * cost_price) →
  cost_price = 21.6 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2760_276051


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2760_276062

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {y | ∃ x, y = Real.exp x + 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2760_276062


namespace NUMINAMATH_CALUDE_square_root_real_range_l2760_276019

theorem square_root_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 + x) → x ≥ -3 := by
sorry

end NUMINAMATH_CALUDE_square_root_real_range_l2760_276019


namespace NUMINAMATH_CALUDE_polynomial_integrality_l2760_276026

theorem polynomial_integrality (x : ℤ) : ∃ k : ℤ, (1/5 : ℚ) * x^5 + (1/3 : ℚ) * x^3 + (7/15 : ℚ) * x = k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integrality_l2760_276026


namespace NUMINAMATH_CALUDE_equation_has_real_root_l2760_276010

theorem equation_has_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a > b) (hbc : b > c) : 
  ∃ x : ℝ, (1 / (x + a) + 1 / (x + b) + 1 / (x + c) - 3 / x = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l2760_276010


namespace NUMINAMATH_CALUDE_common_tangent_sum_l2760_276017

/-- Given two functions f and g with a common tangent at (0, m), prove a + b = 1 -/
theorem common_tangent_sum (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let g : ℝ → ℝ := λ x ↦ x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ -a * Real.sin x
  let g' : ℝ → ℝ := λ x ↦ 2*x + b
  (∃ m : ℝ, f 0 = m ∧ g 0 = m ∧ f' 0 = g' 0) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l2760_276017


namespace NUMINAMATH_CALUDE_smallest_ellipse_area_l2760_276061

/-- The smallest area of an ellipse containing two specific circles -/
theorem smallest_ellipse_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) :
  ∃ k : ℝ, k = 1/2 ∧ ∀ a' b' : ℝ, (∀ x y : ℝ, x^2/a'^2 + y^2/b'^2 = 1 → 
    ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) → π * a' * b' ≥ k * π := by
  sorry


end NUMINAMATH_CALUDE_smallest_ellipse_area_l2760_276061


namespace NUMINAMATH_CALUDE_lcm_6_15_l2760_276079

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_6_15_l2760_276079


namespace NUMINAMATH_CALUDE_expression_evaluation_l2760_276032

def f (x : ℚ) : ℚ := (2 * x + 2) / (x - 2)

theorem expression_evaluation :
  let x : ℚ := 3
  let result := f (f x)
  result = 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2760_276032


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2760_276064

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) :
  x^2 + y^2 + z^2 ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2760_276064


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_15_9_l2760_276014

theorem gcd_lcm_sum_15_9 : 
  Nat.gcd 15 9 + 2 * Nat.lcm 15 9 = 93 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_15_9_l2760_276014


namespace NUMINAMATH_CALUDE_science_quiz_passing_requirement_l2760_276004

theorem science_quiz_passing_requirement (total_questions physics_questions chemistry_questions biology_questions : ℕ)
  (physics_correct_percent chemistry_correct_percent biology_correct_percent passing_percent : ℚ) :
  total_questions = 100 →
  physics_questions = 20 →
  chemistry_questions = 40 →
  biology_questions = 40 →
  physics_correct_percent = 80 / 100 →
  chemistry_correct_percent = 50 / 100 →
  biology_correct_percent = 70 / 100 →
  passing_percent = 65 / 100 →
  (passing_percent * total_questions).ceil -
    (physics_correct_percent * physics_questions +
     chemistry_correct_percent * chemistry_questions +
     biology_correct_percent * biology_questions) = 1 := by
  sorry

end NUMINAMATH_CALUDE_science_quiz_passing_requirement_l2760_276004


namespace NUMINAMATH_CALUDE_power_of_81_l2760_276015

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_power_of_81_l2760_276015


namespace NUMINAMATH_CALUDE_regular_polygon_radius_l2760_276054

/-- A regular polygon with the given properties --/
structure RegularPolygon where
  -- Number of sides
  n : ℕ
  -- Side length
  s : ℝ
  -- Radius
  r : ℝ
  -- Sum of interior angles is twice the sum of exterior angles
  interior_sum_twice_exterior : (n - 2) * 180 = 2 * 360
  -- Side length is 2
  side_length_is_two : s = 2

/-- The radius of the regular polygon with the given properties is 2 --/
theorem regular_polygon_radius (p : RegularPolygon) : p.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_radius_l2760_276054


namespace NUMINAMATH_CALUDE_distance_pole_to_line_rho_cos_theta_eq_two_l2760_276067

/-- The distance from the pole to a line in polar coordinates -/
def distance_pole_to_line (a : ℝ) : ℝ :=
  |a|

/-- Theorem: The distance from the pole to the line ρcosθ=2 is 2 -/
theorem distance_pole_to_line_rho_cos_theta_eq_two :
  distance_pole_to_line 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_pole_to_line_rho_cos_theta_eq_two_l2760_276067


namespace NUMINAMATH_CALUDE_island_puzzle_l2760_276073

-- Define the possible types for natives
inductive NativeType
  | Knight
  | Liar

-- Define a function to represent the truthfulness of a statement based on the native type
def isTruthful (t : NativeType) (s : Prop) : Prop :=
  match t with
  | NativeType.Knight => s
  | NativeType.Liar => ¬s

-- Define the statement made by A
def statementA (typeA typeB : NativeType) : Prop :=
  typeA = NativeType.Liar ∨ typeB = NativeType.Liar

-- Theorem stating that A is a knight and B is a liar
theorem island_puzzle :
  ∃ (typeA typeB : NativeType),
    isTruthful typeA (statementA typeA typeB) ∧
    typeA = NativeType.Knight ∧
    typeB = NativeType.Liar :=
  sorry

end NUMINAMATH_CALUDE_island_puzzle_l2760_276073


namespace NUMINAMATH_CALUDE_ed_conch_shells_ed_conch_shells_eq_8_l2760_276046

theorem ed_conch_shells (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (jacob_extra : ℕ) (total_shells : ℕ) : ℕ :=
  let ed_known := ed_limpet + ed_oyster
  let jacob_shells := ed_known + jacob_extra
  let known_shells := initial_shells + ed_known + jacob_shells
  total_shells - known_shells

theorem ed_conch_shells_eq_8 : 
  ed_conch_shells 2 7 2 2 30 = 8 := by sorry

end NUMINAMATH_CALUDE_ed_conch_shells_ed_conch_shells_eq_8_l2760_276046


namespace NUMINAMATH_CALUDE_b_over_c_equals_one_l2760_276002

theorem b_over_c_equals_one (a b c d : ℕ) : 
  0 < a ∧ a < 4 ∧ 
  0 < b ∧ b < 4 ∧ 
  0 < c ∧ c < 4 ∧ 
  0 < d ∧ d < 4 ∧ 
  4^a + 3^b + 2^c + 1^d = 78 → 
  b / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_over_c_equals_one_l2760_276002


namespace NUMINAMATH_CALUDE_continued_fraction_value_l2760_276095

theorem continued_fraction_value : 
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) → x = (3 + Real.sqrt 39) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l2760_276095


namespace NUMINAMATH_CALUDE_selection_problem_l2760_276041

def total_students : ℕ := 10
def selected_students : ℕ := 3
def students_excluding_c : ℕ := 9
def students_excluding_abc : ℕ := 7

theorem selection_problem :
  (Nat.choose students_excluding_c selected_students) -
  (Nat.choose students_excluding_abc selected_students) = 49 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l2760_276041


namespace NUMINAMATH_CALUDE_exactly_three_ways_l2760_276078

/-- The sum of consecutive integers from a to b, inclusive -/
def consecutiveSum (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- The predicate that checks if a pair (a, b) satisfies the conditions -/
def isValidPair (a b : ℕ) : Prop :=
  a < b ∧ consecutiveSum a b = 91

/-- The theorem stating that there are exactly 3 valid pairs -/
theorem exactly_three_ways :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 3 ∧ ∀ p, p ∈ s ↔ isValidPair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_ways_l2760_276078


namespace NUMINAMATH_CALUDE_x_to_neg_y_equals_half_l2760_276088

theorem x_to_neg_y_equals_half (x y : ℝ) (h : Real.sqrt (x + y - 3) = -(x - 2*y)^2) : 
  x^(-y) = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_x_to_neg_y_equals_half_l2760_276088


namespace NUMINAMATH_CALUDE_min_y_value_l2760_276063

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 72*y) : 
  ∀ y' : ℝ, (∃ x' : ℝ, x'^2 + y'^2 = 20*x' + 72*y') → y ≥ 36 - Real.sqrt 1396 := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l2760_276063


namespace NUMINAMATH_CALUDE_a_decreasing_l2760_276066

open BigOperators

def a (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / (k * (n + 1 - k))

theorem a_decreasing (n : ℕ) (h : n ≥ 2) : a (n + 1) < a n := by
  sorry

end NUMINAMATH_CALUDE_a_decreasing_l2760_276066


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2760_276081

theorem sum_of_solutions (x : ℝ) : 
  (∃ a b : ℝ, (4*x + 6) * (3*x - 8) = 0 ∧ x = a ∨ x = b) → 
  (∃ a b : ℝ, (4*x + 6) * (3*x - 8) = 0 ∧ x = a ∨ x = b ∧ a + b = 7/6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2760_276081


namespace NUMINAMATH_CALUDE_rabbit_carrot_consumption_l2760_276068

theorem rabbit_carrot_consumption :
  ∀ (rabbit_days deer_days : ℕ) (total_food : ℕ),
    rabbit_days = deer_days + 2 →
    5 * rabbit_days = total_food →
    6 * deer_days = total_food →
    5 * rabbit_days = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrot_consumption_l2760_276068


namespace NUMINAMATH_CALUDE_base4_sum_234_73_l2760_276047

/-- Converts a number from base 4 to base 10 -/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- The sum of two numbers in base 4 -/
def base4_sum (a b : ℕ) : ℕ :=
  base10_to_base4 (base4_to_base10 a + base4_to_base10 b)

theorem base4_sum_234_73 : base4_sum 234 73 = 10303 := by sorry

end NUMINAMATH_CALUDE_base4_sum_234_73_l2760_276047


namespace NUMINAMATH_CALUDE_distance_between_stations_l2760_276045

/-- The distance between two stations given the conditions of two trains meeting --/
theorem distance_between_stations
  (speed_train1 : ℝ)
  (speed_train2 : ℝ)
  (extra_distance : ℝ)
  (h1 : speed_train1 = 20)
  (h2 : speed_train2 = 25)
  (h3 : extra_distance = 70)
  (h4 : speed_train1 > 0)
  (h5 : speed_train2 > 0) :
  ∃ (time : ℝ),
    time > 0 ∧
    speed_train1 * time + speed_train2 * time = speed_train1 * time + extra_distance ∧
    speed_train1 * time + speed_train2 * time = 630 :=
by sorry


end NUMINAMATH_CALUDE_distance_between_stations_l2760_276045


namespace NUMINAMATH_CALUDE_same_terminal_side_l2760_276038

theorem same_terminal_side (θ : ℝ) : ∃ k : ℤ, θ = (23 * π / 3 : ℝ) + 2 * π * k ↔ θ = (5 * π / 3 : ℝ) + 2 * π * k := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l2760_276038


namespace NUMINAMATH_CALUDE_ahmed_hassan_apple_ratio_l2760_276029

/-- Ahmed's orchard has 8 orange trees and an unknown number of apple trees. -/
def ahmed_orange_trees : ℕ := 8

/-- Hassan's orchard has 1 apple tree. -/
def hassan_apple_trees : ℕ := 1

/-- Hassan's orchard has 2 orange trees. -/
def hassan_orange_trees : ℕ := 2

/-- The difference in total trees between Ahmed's and Hassan's orchards. -/
def tree_difference : ℕ := 9

/-- Ahmed's apple trees -/
def ahmed_apple_trees : ℕ := ahmed_orange_trees + tree_difference - (hassan_apple_trees + hassan_orange_trees)

theorem ahmed_hassan_apple_ratio :
  ahmed_apple_trees = 4 * hassan_apple_trees := by
  sorry

end NUMINAMATH_CALUDE_ahmed_hassan_apple_ratio_l2760_276029


namespace NUMINAMATH_CALUDE_victoria_gym_schedule_l2760_276083

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents Victoria's gym schedule -/
structure GymSchedule where
  startDay : DayOfWeek
  sessionsPlanned : Nat
  publicHolidays : Nat
  personalEvents : Nat

/-- Calculates the number of days until the completion of the gym schedule -/
def daysToComplete (schedule : GymSchedule) : Nat :=
  sorry

/-- Determines the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

/-- Main theorem: Victoria completes her 30th gym session on the 51st day, which is a Wednesday -/
theorem victoria_gym_schedule :
  let schedule := GymSchedule.mk DayOfWeek.Monday 30 3 2
  daysToComplete schedule = 51 ∧
  dayAfter DayOfWeek.Monday 51 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_victoria_gym_schedule_l2760_276083


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l2760_276040

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 35)
  (h2 : num_groups = 7)
  (h3 : caps_per_group * num_groups = total_caps) :
  caps_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l2760_276040


namespace NUMINAMATH_CALUDE_competition_participants_l2760_276018

theorem competition_participants (initial : ℕ) 
  (h1 : initial * 40 / 100 * 50 / 100 * 25 / 100 = 15) : 
  initial = 300 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l2760_276018


namespace NUMINAMATH_CALUDE_rational_solution_cosine_equation_l2760_276094

theorem rational_solution_cosine_equation (q : ℚ) 
  (h1 : 0 < q) (h2 : q < 1) 
  (h3 : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2/3 := by
sorry

end NUMINAMATH_CALUDE_rational_solution_cosine_equation_l2760_276094


namespace NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l2760_276003

theorem complex_exp_13pi_div_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l2760_276003


namespace NUMINAMATH_CALUDE_selection_count_theorem_l2760_276033

/-- Represents a grid of people -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a selection of people from the grid -/
structure Selection :=
  (grid : Grid)
  (num_selected : ℕ)

/-- Counts the number of valid selections -/
def count_valid_selections (s : Selection) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem selection_count_theorem (g : Grid) (s : Selection) :
  g.rows = 6 ∧ g.cols = 7 ∧ s.grid = g ∧ s.num_selected = 3 →
  count_valid_selections s = 4200 :=
sorry

end NUMINAMATH_CALUDE_selection_count_theorem_l2760_276033


namespace NUMINAMATH_CALUDE_grid_area_l2760_276082

/-- Calculate the area of a grid composed of a base rectangle, a top left rectangle, and a trapezoid -/
theorem grid_area (base_height base_width : ℝ)
                  (top_left_height top_left_width : ℝ)
                  (trapezoid_base1 trapezoid_base2 trapezoid_height : ℝ) :
  base_height = 2 →
  base_width = 12 →
  top_left_height = 1 →
  top_left_width = 4 →
  trapezoid_base1 = 7 →
  trapezoid_base2 = 8 →
  trapezoid_height = 2 →
  base_height * base_width +
  top_left_height * top_left_width +
  (trapezoid_base1 + trapezoid_base2) * trapezoid_height / 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_grid_area_l2760_276082


namespace NUMINAMATH_CALUDE_train_crossing_time_l2760_276005

theorem train_crossing_time (train_length : ℝ) (platform1_length : ℝ) (platform2_length : ℝ) (time1 : ℝ) :
  train_length = 100 →
  platform1_length = 350 →
  platform2_length = 500 →
  time1 = 15 →
  let total_distance1 := train_length + platform1_length
  let speed := total_distance1 / time1
  let total_distance2 := train_length + platform2_length
  let time2 := total_distance2 / speed
  time2 = 20 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2760_276005


namespace NUMINAMATH_CALUDE_min_translation_overlap_l2760_276024

theorem min_translation_overlap (φ : Real) : 
  (φ > 0) →
  (∀ x, Real.sin (2 * (x + φ)) = Real.sin (2 * x - 2 * φ + Real.pi / 3)) →
  φ ≥ Real.pi / 12 :=
sorry

end NUMINAMATH_CALUDE_min_translation_overlap_l2760_276024


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2760_276011

theorem modulus_of_complex_fraction (z : ℂ) : 
  z = (2.2 * Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2760_276011


namespace NUMINAMATH_CALUDE_smallest_period_of_one_minus_cos_2x_l2760_276059

/-- The smallest positive period of y = 1 - cos(2x) is π -/
theorem smallest_period_of_one_minus_cos_2x (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 1 - Real.cos (2 * x)
  ∃ T : ℝ, T > 0 ∧ T = π ∧ ∀ t : ℝ, f (t + T) = f t ∧ 
    ∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (t + S) = f t) → T ≤ S :=
by sorry

end NUMINAMATH_CALUDE_smallest_period_of_one_minus_cos_2x_l2760_276059


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2760_276037

theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2760_276037


namespace NUMINAMATH_CALUDE_kelvin_frog_paths_l2760_276039

/-- Represents a position in the coordinate plane -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Represents a move that Kelvin can make -/
inductive Move
  | Walk : Move
  | Jump : Move

/-- Defines the possible moves Kelvin can make from a given position -/
def possibleMoves (pos : Position) : List Position :=
  [
    {x := pos.x, y := pos.y + 1},     -- Walk up
    {x := pos.x + 1, y := pos.y},     -- Walk right
    {x := pos.x + 1, y := pos.y + 1}, -- Walk diagonally
    {x := pos.x, y := pos.y + 2},     -- Jump up
    {x := pos.x + 2, y := pos.y},     -- Jump right
    {x := pos.x + 1, y := pos.y + 1}  -- Jump diagonally
  ]

/-- Counts the number of ways to reach the target position from the start position -/
def countWays (start : Position) (target : Position) : ℕ :=
  sorry

theorem kelvin_frog_paths : countWays {x := 0, y := 0} {x := 6, y := 8} = 1831830 := by
  sorry

end NUMINAMATH_CALUDE_kelvin_frog_paths_l2760_276039


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l2760_276009

-- Define the structure for a survey
structure Survey where
  total_population : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool

-- Define the sampling methods
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling

-- Define the function to determine the appropriate sampling method
def appropriate_sampling_method (survey : Survey) : SamplingMethod :=
  if survey.has_distinct_groups && survey.total_population > survey.sample_size * 10
  then SamplingMethod.StratifiedSampling
  else SamplingMethod.SimpleRandomSampling

-- Define the surveys
def survey1 : Survey := {
  total_population := 125 + 280 + 95,
  sample_size := 100,
  has_distinct_groups := true
}

def survey2 : Survey := {
  total_population := 15,
  sample_size := 3,
  has_distinct_groups := false
}

-- Theorem to prove
theorem appropriate_sampling_methods :
  appropriate_sampling_method survey1 = SamplingMethod.StratifiedSampling ∧
  appropriate_sampling_method survey2 = SamplingMethod.SimpleRandomSampling :=
by sorry


end NUMINAMATH_CALUDE_appropriate_sampling_methods_l2760_276009


namespace NUMINAMATH_CALUDE_lisa_minimum_score_l2760_276050

def minimum_score_for_geometry (term1 term2 term3 term4 : ℝ) (required_average : ℝ) : ℝ :=
  5 * required_average - (term1 + term2 + term3 + term4)

theorem lisa_minimum_score :
  let term1 := 84
  let term2 := 80
  let term3 := 82
  let term4 := 87
  let required_average := 85
  minimum_score_for_geometry term1 term2 term3 term4 required_average = 92 := by
sorry

end NUMINAMATH_CALUDE_lisa_minimum_score_l2760_276050


namespace NUMINAMATH_CALUDE_bounded_quadratic_coef_sum_l2760_276042

/-- A quadratic polynomial f(x) = ax² + bx + c with |f(x)| ≤ 1 for all x in [0, 2] -/
def BoundedQuadratic (a b c : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |a * x^2 + b * x + c| ≤ 1

/-- The sum of absolute values of coefficients is at most 7 -/
theorem bounded_quadratic_coef_sum (a b c : ℝ) (h : BoundedQuadratic a b c) :
  |a| + |b| + |c| ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_bounded_quadratic_coef_sum_l2760_276042


namespace NUMINAMATH_CALUDE_determinant_equation_solution_l2760_276020

/-- Definition of a 2x2 determinant -/
def det (a b c d : ℚ) : ℚ := a * d - b * c

/-- Theorem: If |x-2 x+3; x+1 x-2| = 13, then x = -3/2 -/
theorem determinant_equation_solution :
  ∀ x : ℚ, det (x - 2) (x + 3) (x + 1) (x - 2) = 13 → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equation_solution_l2760_276020
