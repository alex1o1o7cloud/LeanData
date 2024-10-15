import Mathlib

namespace NUMINAMATH_CALUDE_intersection_equality_implies_range_l694_69477

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the range of a
def range_a : Set ℝ := {a | a = 1 ∨ a ≤ -1}

-- Theorem statement
theorem intersection_equality_implies_range (a : ℝ) : 
  A ∩ B a = B a → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_range_l694_69477


namespace NUMINAMATH_CALUDE_dinner_cost_calculation_l694_69491

/-- The total cost of dinner for Bret and his co-workers -/
def dinner_cost (num_people : ℕ) (main_meal_cost : ℚ) (num_appetizers : ℕ) (appetizer_cost : ℚ) (tip_percentage : ℚ) (rush_order_fee : ℚ) : ℚ :=
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := tip_percentage * subtotal
  subtotal + tip + rush_order_fee

/-- Theorem stating the total cost of dinner -/
theorem dinner_cost_calculation :
  dinner_cost 4 12 2 6 (1/5) 5 = 77 :=
by sorry

end NUMINAMATH_CALUDE_dinner_cost_calculation_l694_69491


namespace NUMINAMATH_CALUDE_donna_bananas_l694_69404

def total_bananas : ℕ := 350
def lydia_bananas : ℕ := 90
def dawn_extra_bananas : ℕ := 70

theorem donna_bananas :
  total_bananas - (lydia_bananas + (lydia_bananas + dawn_extra_bananas)) = 100 :=
by sorry

end NUMINAMATH_CALUDE_donna_bananas_l694_69404


namespace NUMINAMATH_CALUDE_triangle_radii_relation_l694_69474

/-- Given a triangle with side lengths a, b, c, semi-perimeter p, area S, and circumradius R,
    prove the relationship between the inradius τ, exradii τa, τb, τc, and other triangle properties. -/
theorem triangle_radii_relation
  (a b c p S R τ τa τb τc : ℝ)
  (h1 : S = τ * p)
  (h2 : S = τa * (p - a))
  (h3 : S = τb * (p - b))
  (h4 : S = τc * (p - c))
  (h5 : a * b * c / S = 4 * R) :
  1 / τ^3 - 1 / τa^3 - 1 / τb^3 - 1 / τc^3 = 12 * R / S^2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_radii_relation_l694_69474


namespace NUMINAMATH_CALUDE_michaels_blocks_l694_69485

/-- Given that Michael has some blocks stored in boxes, prove that the total number of blocks is 16 -/
theorem michaels_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (h1 : num_boxes = 8) (h2 : blocks_per_box = 2) :
  num_boxes * blocks_per_box = 16 := by
  sorry

end NUMINAMATH_CALUDE_michaels_blocks_l694_69485


namespace NUMINAMATH_CALUDE_factorization_proof_l694_69484

theorem factorization_proof (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l694_69484


namespace NUMINAMATH_CALUDE_scissors_cost_l694_69483

theorem scissors_cost (initial_amount : ℕ) (num_scissors : ℕ) (num_erasers : ℕ) 
  (eraser_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 100 → 
  num_scissors = 8 → 
  num_erasers = 10 → 
  eraser_cost = 4 → 
  remaining_amount = 20 → 
  ∃ (scissor_cost : ℕ), 
    scissor_cost = 5 ∧ 
    initial_amount = num_scissors * scissor_cost + num_erasers * eraser_cost + remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_scissors_cost_l694_69483


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l694_69486

theorem initial_mean_calculation (n : ℕ) (correct_value wrong_value : ℝ) (correct_mean : ℝ) (M : ℝ) :
  n = 30 →
  correct_value = 145 →
  wrong_value = 135 →
  correct_mean = 140.33333333333334 →
  n * M + (correct_value - wrong_value) = n * correct_mean →
  M = 140 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l694_69486


namespace NUMINAMATH_CALUDE_steves_speed_ratio_l694_69406

/-- Proves the ratio of Steve's speeds given the problem conditions -/
theorem steves_speed_ratio :
  let distance : ℝ := 10 -- km
  let total_time : ℝ := 6 -- hours
  let speed_back : ℝ := 5 -- km/h
  let speed_to_work : ℝ := distance / (total_time - distance / speed_back)
  speed_back / speed_to_work = 2
  := by sorry

end NUMINAMATH_CALUDE_steves_speed_ratio_l694_69406


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l694_69499

theorem real_part_of_complex_product : Complex.re ((1 + 3 * Complex.I) * Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l694_69499


namespace NUMINAMATH_CALUDE_total_hamburgers_for_lunch_l694_69414

theorem total_hamburgers_for_lunch : 
  let initial_beef : ℕ := 15
  let initial_veggie : ℕ := 12
  let additional_beef : ℕ := 5
  let additional_veggie : ℕ := 7
  initial_beef + initial_veggie + additional_beef + additional_veggie = 39
  := by sorry

end NUMINAMATH_CALUDE_total_hamburgers_for_lunch_l694_69414


namespace NUMINAMATH_CALUDE_calculation_proof_l694_69475

theorem calculation_proof : |Real.sqrt 3 - 2| - 2 * Real.tan (π / 3) + (π - 2023) ^ 0 + Real.sqrt 27 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l694_69475


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l694_69415

theorem trigonometric_expression_equality (θ : Real) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) = 17 * (Real.sqrt 10 + 1) / 24 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l694_69415


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l694_69409

/-- Given a train crossing a bridge, calculate the length of the bridge -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 145 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l694_69409


namespace NUMINAMATH_CALUDE_apple_ratio_is_half_l694_69496

/-- The number of apples Anna ate on Tuesday -/
def tuesday_apples : ℕ := 4

/-- The number of apples Anna ate on Wednesday -/
def wednesday_apples : ℕ := 2 * tuesday_apples

/-- The total number of apples Anna ate over the three days -/
def total_apples : ℕ := 14

/-- The number of apples Anna ate on Thursday -/
def thursday_apples : ℕ := total_apples - tuesday_apples - wednesday_apples

/-- The ratio of apples eaten on Thursday to Tuesday -/
def thursday_to_tuesday_ratio : ℚ := thursday_apples / tuesday_apples

theorem apple_ratio_is_half : thursday_to_tuesday_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_is_half_l694_69496


namespace NUMINAMATH_CALUDE_rent_increase_problem_l694_69418

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 850) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.25) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 800 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l694_69418


namespace NUMINAMATH_CALUDE_prism_lateral_edge_length_l694_69441

/-- A prism with 12 vertices and a sum of lateral edge lengths of 60 has lateral edges of length 10. -/
theorem prism_lateral_edge_length (num_vertices : ℕ) (sum_lateral_edges : ℝ) :
  num_vertices = 12 →
  sum_lateral_edges = 60 →
  ∃ (lateral_edge_length : ℝ), lateral_edge_length = 10 ∧
    lateral_edge_length * (num_vertices / 2) = sum_lateral_edges :=
by sorry


end NUMINAMATH_CALUDE_prism_lateral_edge_length_l694_69441


namespace NUMINAMATH_CALUDE_rebecca_eggs_l694_69458

/-- The number of groups Rebecca wants to split her eggs into -/
def num_groups : ℕ := 4

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 2

/-- Theorem: Rebecca has 8 eggs in total -/
theorem rebecca_eggs : num_groups * eggs_per_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l694_69458


namespace NUMINAMATH_CALUDE_johns_leftover_earnings_l694_69478

/-- Proves that given John spent 40% of his earnings on rent and 30% less than that on a dishwasher, he had 32% of his earnings left over. -/
theorem johns_leftover_earnings : 
  ∀ (total_earnings : ℝ) (rent_percent : ℝ) (dishwasher_percent : ℝ),
    rent_percent = 40 →
    dishwasher_percent = rent_percent - (0.3 * rent_percent) →
    100 - (rent_percent + dishwasher_percent) = 32 := by
  sorry

end NUMINAMATH_CALUDE_johns_leftover_earnings_l694_69478


namespace NUMINAMATH_CALUDE_existence_of_alpha_beta_l694_69426

-- Define the Intermediate Value Property
def has_intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y, (f a < y ∧ y < f b) ∨ (f b < y ∧ y < f a) → ∃ c, a < c ∧ c < b ∧ f c = y

-- State the theorem
theorem existence_of_alpha_beta
  (f : ℝ → ℝ) (a b : ℝ) (h_ivp : has_intermediate_value_property f a b)
  (h_sign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end NUMINAMATH_CALUDE_existence_of_alpha_beta_l694_69426


namespace NUMINAMATH_CALUDE_partnership_profit_l694_69471

/-- Given the investments of three partners and one partner's share of the profit,
    calculate the total profit of the partnership. -/
theorem partnership_profit
  (investment_A investment_B investment_C : ℕ)
  (profit_share_A : ℕ)
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 3660) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 12200 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l694_69471


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l694_69466

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l694_69466


namespace NUMINAMATH_CALUDE_sum_of_rational_roots_l694_69461

def h (x : ℚ) : ℚ := x^3 - 12*x^2 + 47*x - 60

theorem sum_of_rational_roots :
  ∃ (r₁ r₂ r₃ : ℚ),
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0 ∧
    (∀ r : ℚ, h r = 0 → r = r₁ ∨ r = r₂ ∨ r = r₃) ∧
    r₁ + r₂ + r₃ = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_rational_roots_l694_69461


namespace NUMINAMATH_CALUDE_simplify_expression_l694_69476

theorem simplify_expression (x y : ℝ) : 7*x + 8*y - 3*x + 4*y + 10 = 4*x + 12*y + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l694_69476


namespace NUMINAMATH_CALUDE_prob_at_most_one_first_class_l694_69427

/-- The probability of selecting at most one first-class product when randomly choosing 2 out of 5 products (3 first-class and 2 second-class) is 0.7 -/
theorem prob_at_most_one_first_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) :
  total = 5 →
  first_class = 3 →
  second_class = 2 →
  selected = 2 →
  (Nat.choose first_class 1 * Nat.choose second_class 1 + Nat.choose second_class 2) / Nat.choose total selected = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_first_class_l694_69427


namespace NUMINAMATH_CALUDE_monochromatic_four_cycle_exists_l694_69450

/-- A coloring of edges in a graph using two colors -/
def TwoColoring (V : Type*) := V → V → Bool

/-- A complete graph with 6 vertices -/
def CompleteGraph6 := Fin 6

/-- A 4-cycle in a graph -/
def FourCycle (V : Type*) := 
  (V × V × V × V)

/-- Predicate to check if a 4-cycle is monochromatic under a given coloring -/
def IsMonochromatic (c : TwoColoring CompleteGraph6) (cycle : FourCycle CompleteGraph6) : Prop :=
  let (a, b, d, e) := cycle
  c a b = c b d ∧ c b d = c d e ∧ c d e = c e a

/-- Main theorem: In a complete graph with 6 vertices where each edge is colored 
    with one of two colors, there exists a monochromatic 4-cycle -/
theorem monochromatic_four_cycle_exists :
  ∀ (c : TwoColoring CompleteGraph6),
  ∃ (cycle : FourCycle CompleteGraph6), IsMonochromatic c cycle :=
sorry


end NUMINAMATH_CALUDE_monochromatic_four_cycle_exists_l694_69450


namespace NUMINAMATH_CALUDE_coin_collection_l694_69470

theorem coin_collection (nickels dimes quarters : ℕ) (total_value : ℕ) : 
  nickels = dimes →
  quarters = 2 * nickels →
  total_value = 1950 →
  5 * nickels + 10 * dimes + 25 * quarters = total_value →
  nickels = 30 := by
sorry

end NUMINAMATH_CALUDE_coin_collection_l694_69470


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l694_69467

theorem quadratic_root_transformation (r s : ℝ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  (r + s = -4/3) →
  (r * s = 2/3) →
  ∃ q : ℝ, r^3 + s^3 = -16/27 ∧ r^3 * s^3 = q ∧ 
    ∀ x : ℝ, x^2 + (16/27) * x + q = 0 ↔ (x = r^3 ∨ x = s^3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l694_69467


namespace NUMINAMATH_CALUDE_arithmetic_sequence_vertex_l694_69479

/-- Given that a, b, c, d form an arithmetic sequence and (a, d) is the vertex of f(x) = x^2 - 2x,
    prove that b + c = 0 -/
theorem arithmetic_sequence_vertex (a b c d : ℝ) : 
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →  -- arithmetic sequence condition
  (a = 1 ∧ d = -1) →                              -- vertex condition
  b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_vertex_l694_69479


namespace NUMINAMATH_CALUDE_sin_315_degrees_l694_69492

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l694_69492


namespace NUMINAMATH_CALUDE_onion_weight_proof_l694_69435

/-- Proves that the total weight of onions on a scale is 7.68 kg given specific conditions --/
theorem onion_weight_proof (total_weight : ℝ) (remaining_onions : ℕ) (removed_onions : ℕ) 
  (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ) : 
  total_weight = 7.68 ∧ 
  remaining_onions = 35 ∧ 
  removed_onions = 5 ∧ 
  avg_weight_remaining = 0.190 ∧ 
  avg_weight_removed = 0.206 → 
  total_weight = (remaining_onions : ℝ) * avg_weight_remaining + 
                 (removed_onions : ℝ) * avg_weight_removed :=
by
  sorry

#check onion_weight_proof

end NUMINAMATH_CALUDE_onion_weight_proof_l694_69435


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l694_69498

theorem simplify_and_evaluate (a b : ℝ) : 
  a = Real.tan (π / 3) → 
  b = Real.sin (π / 3) → 
  ((b^2 + a^2) / a - 2 * b) / (1 - b / a) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l694_69498


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l694_69400

theorem cos_two_alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) :
  Real.cos (2 * α) = 2 * Real.sqrt 14 / 9 ∨ Real.cos (2 * α) = -2 * Real.sqrt 14 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l694_69400


namespace NUMINAMATH_CALUDE_abs_inequality_implies_quadratic_inequality_l694_69454

theorem abs_inequality_implies_quadratic_inequality :
  {x : ℝ | |x - 1| < 2} ⊂ {x : ℝ | (x + 2) * (x - 3) < 0} ∧
  {x : ℝ | |x - 1| < 2} ≠ {x : ℝ | (x + 2) * (x - 3) < 0} :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_implies_quadratic_inequality_l694_69454


namespace NUMINAMATH_CALUDE_portfolio_distribution_l694_69423

theorem portfolio_distribution (total_students : ℕ) (total_portfolios : ℕ) 
  (h1 : total_students = 120) 
  (h2 : total_portfolios = 8365) : 
  ∃ (regular_portfolios : ℕ) (special_portfolios : ℕ) (remaining_portfolios : ℕ),
    let regular_students : ℕ := (85 * total_students) / 100
    let special_students : ℕ := total_students - regular_students
    special_portfolios = regular_portfolios + 10 ∧
    regular_students * regular_portfolios + special_students * special_portfolios + remaining_portfolios = total_portfolios ∧
    remaining_portfolios = 25 :=
by sorry

end NUMINAMATH_CALUDE_portfolio_distribution_l694_69423


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l694_69480

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (9 * bowling_ball_weight = 5 * canoe_weight) →
    (4 * canoe_weight = 120) →
    bowling_ball_weight = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l694_69480


namespace NUMINAMATH_CALUDE_seniors_in_stratified_sample_l694_69421

/-- Represents the number of seniors in a stratified sample -/
def seniors_in_sample (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) : ℕ :=
  (total_seniors * sample_size) / total_students

/-- Theorem stating that in a school with 4500 students, of which 1500 are seniors,
    a stratified sample of 300 students will contain 100 seniors -/
theorem seniors_in_stratified_sample :
  seniors_in_sample 4500 1500 300 = 100 := by
  sorry

end NUMINAMATH_CALUDE_seniors_in_stratified_sample_l694_69421


namespace NUMINAMATH_CALUDE_solutions_of_f_of_f_eq_x_l694_69430

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 5*x + 1

-- State the theorem
theorem solutions_of_f_of_f_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = -2 - Real.sqrt 3 ∨ x = -2 + Real.sqrt 3 ∨ x = -3 - Real.sqrt 2 ∨ x = -3 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_solutions_of_f_of_f_eq_x_l694_69430


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_four_l694_69494

theorem sum_of_roots_equals_four :
  let f (x : ℝ) := (x^3 - 2*x^2 - 8*x) / (x + 2)
  (∃ a b : ℝ, (f a = 5 ∧ f b = 5 ∧ a ≠ b) ∧ a + b = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_four_l694_69494


namespace NUMINAMATH_CALUDE_money_distribution_l694_69463

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 900)
  (AC_sum : A + C = 400)
  (C_amount : C = 250) :
  B + C = 750 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l694_69463


namespace NUMINAMATH_CALUDE_apple_orange_ratio_l694_69440

/-- Given a basket of fruit with apples and oranges, prove the ratio of apples to oranges --/
theorem apple_orange_ratio (total_fruit : ℕ) (oranges : ℕ) : 
  total_fruit = 40 → oranges = 10 → (total_fruit - oranges) / oranges = 3 := by
  sorry

#check apple_orange_ratio

end NUMINAMATH_CALUDE_apple_orange_ratio_l694_69440


namespace NUMINAMATH_CALUDE_incorrect_multiplication_l694_69448

theorem incorrect_multiplication : 79133 * 111107 ≠ 8794230231 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_l694_69448


namespace NUMINAMATH_CALUDE_mary_max_earnings_l694_69405

/-- Calculates the maximum weekly earnings for Mary given her work conditions -/
theorem mary_max_earnings :
  let max_hours : ℕ := 60
  let regular_hours : ℕ := 20
  let regular_rate : ℚ := 8
  let overtime_rate_increase : ℚ := 0.25
  let overtime_rate : ℚ := regular_rate * (1 + overtime_rate_increase)
  let overtime_hours : ℕ := max_hours - regular_hours
  let regular_earnings : ℚ := regular_hours * regular_rate
  let overtime_earnings : ℚ := overtime_hours * overtime_rate
  regular_earnings + overtime_earnings = 560 := by
sorry

end NUMINAMATH_CALUDE_mary_max_earnings_l694_69405


namespace NUMINAMATH_CALUDE_shopping_theorem_l694_69453

def shopping_problem (initial_amount discount_rate tax_rate: ℝ)
  (sweater t_shirt shoes jeans scarf: ℝ) : Prop :=
  let discounted_t_shirt := t_shirt * (1 - discount_rate)
  let subtotal := sweater + discounted_t_shirt + shoes + jeans + scarf
  let total_with_tax := subtotal * (1 + tax_rate)
  let remaining := initial_amount - total_with_tax
  remaining = 30.11

theorem shopping_theorem :
  shopping_problem 200 0.1 0.05 36 12 45 52 18 :=
by sorry

end NUMINAMATH_CALUDE_shopping_theorem_l694_69453


namespace NUMINAMATH_CALUDE_train_passing_time_l694_69407

/-- The time taken for a faster train to catch and pass a slower train -/
theorem train_passing_time (train_length : ℝ) (speed_fast speed_slow : ℝ) : 
  train_length = 25 →
  speed_fast = 46 * (1000 / 3600) →
  speed_slow = 36 * (1000 / 3600) →
  speed_fast > speed_slow →
  (2 * train_length) / (speed_fast - speed_slow) = 18 := by
  sorry

#eval (2 * 25) / ((46 - 36) * (1000 / 3600))

end NUMINAMATH_CALUDE_train_passing_time_l694_69407


namespace NUMINAMATH_CALUDE_train_length_l694_69443

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) →
  platform_length = 240 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 280 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l694_69443


namespace NUMINAMATH_CALUDE_expected_draws_eq_sixteen_thirds_l694_69493

/-- The number of red balls in the bag -/
def num_red : ℕ := 2

/-- The number of black balls in the bag -/
def num_black : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_black

/-- The set of possible numbers of draws -/
def possible_draws : Finset ℕ := Finset.range (total_balls + 1) \ Finset.range num_red

/-- The probability of drawing a specific number of balls -/
noncomputable def prob_draw (n : ℕ) : ℚ :=
  if n ∈ possible_draws then
    -- This is a placeholder for the actual probability calculation
    1 / possible_draws.card
  else
    0

/-- The expected number of draws -/
noncomputable def expected_draws : ℚ :=
  Finset.sum possible_draws (λ n => n * prob_draw n)

theorem expected_draws_eq_sixteen_thirds :
  expected_draws = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_draws_eq_sixteen_thirds_l694_69493


namespace NUMINAMATH_CALUDE_range_of_m_l694_69472

theorem range_of_m (p q : Prop) (m : ℝ) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) 
  (h3 : p ↔ m < 0) 
  (h4 : q ↔ m < 2) : 
  0 ≤ m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l694_69472


namespace NUMINAMATH_CALUDE_system_solution_exists_l694_69401

theorem system_solution_exists : ∃ (x y : ℝ), 
  0 ≤ x ∧ x ≤ 6 ∧ 
  0 ≤ y ∧ y ≤ 4 ∧ 
  x + 2 * Real.sqrt y = 6 ∧ 
  Real.sqrt x + y = 4 ∧ 
  abs (x - 2.985) < 0.001 ∧ 
  abs (y - 2.272) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l694_69401


namespace NUMINAMATH_CALUDE_angle_AOD_measure_l694_69422

-- Define the angles
variable (AOB BOC COD AOD : ℝ)

-- Define the conditions
axiom angles_equal : AOB = BOC ∧ BOC = COD
axiom AOD_smaller : AOD = AOB / 3

-- Define the distinctness of rays (we can't directly represent this in angles, so we'll skip it)

-- Define the theorem
theorem angle_AOD_measure :
  (AOB + BOC + COD + AOD = 360 ∨ AOB + BOC + COD - AOD = 360) →
  AOD = 36 ∨ AOD = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOD_measure_l694_69422


namespace NUMINAMATH_CALUDE_problem_solution_l694_69410

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, Real.log x₀ ≥ x₀ - 1

-- Define proposition q
def q : Prop := ∀ θ : ℝ, Real.sin θ + Real.cos θ < 1

-- Theorem to prove
theorem problem_solution :
  p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l694_69410


namespace NUMINAMATH_CALUDE_rays_initial_cents_l694_69464

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The amount of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- The number of nickels Peter receives -/
def nickels_to_peter : ℕ := cents_to_peter / nickel_value

/-- The amount of cents Ray gives to Randi -/
def cents_to_randi : ℕ := 2 * cents_to_peter

/-- The number of nickels Randi receives -/
def nickels_to_randi : ℕ := cents_to_randi / nickel_value

/-- The difference in nickels between Randi and Peter -/
def nickel_difference : ℕ := 6

theorem rays_initial_cents :
  nickels_to_randi = nickels_to_peter + nickel_difference →
  cents_to_peter + cents_to_randi = 90 := by
  sorry

end NUMINAMATH_CALUDE_rays_initial_cents_l694_69464


namespace NUMINAMATH_CALUDE_largest_multiple_of_18_with_9_and_0_digits_l694_69434

def is_multiple_of_18 (n : ℕ) : Prop := ∃ k : ℕ, n = 18 * k

def digits_are_9_or_0 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 9 ∨ d = 0

theorem largest_multiple_of_18_with_9_and_0_digits :
  ∃ m : ℕ,
    is_multiple_of_18 m ∧
    digits_are_9_or_0 m ∧
    (∀ n : ℕ, is_multiple_of_18 n → digits_are_9_or_0 n → n ≤ m) ∧
    m = 900 ∧
    m / 18 = 50 := by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_18_with_9_and_0_digits_l694_69434


namespace NUMINAMATH_CALUDE_water_in_tank_after_rain_l694_69447

/-- Calculates the final amount of water in a tank after evaporation, draining, and rain. -/
def final_water_amount (initial_water evaporated_water drained_water rain_duration rain_rate : ℕ) : ℕ :=
  let remaining_after_evaporation := initial_water - evaporated_water
  let remaining_after_draining := remaining_after_evaporation - drained_water
  let rain_amount := (rain_duration / 10) * rain_rate
  remaining_after_draining + rain_amount

/-- Theorem stating that the final amount of water in the tank is 1550 liters. -/
theorem water_in_tank_after_rain :
  final_water_amount 6000 2000 3500 30 350 = 1550 := by
  sorry

end NUMINAMATH_CALUDE_water_in_tank_after_rain_l694_69447


namespace NUMINAMATH_CALUDE_inequality_proof_l694_69446

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  4 * a^3 * (a - b) ≥ a^4 - b^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l694_69446


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l694_69408

def total_crayons : ℕ := 15
def red_crayons : ℕ := 2
def selection_size : ℕ := 6

def select_crayons_with_red : ℕ := Nat.choose total_crayons selection_size - Nat.choose (total_crayons - red_crayons) selection_size

theorem crayon_selection_theorem : select_crayons_with_red = 2860 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l694_69408


namespace NUMINAMATH_CALUDE_equation_solution_l694_69465

theorem equation_solution (x : ℚ) : 64 * (x + 1)^3 - 27 = 0 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l694_69465


namespace NUMINAMATH_CALUDE_system_solutions_l694_69490

def system (x y z : ℚ) : Prop :=
  x^2 + 2*y*z = x ∧ y^2 + 2*z*x = y ∧ z^2 + 2*x*y = z

def solutions : List (ℚ × ℚ × ℚ) :=
  [(0, 0, 0), (1/3, 1/3, 1/3), (1, 0, 0), (0, 1, 0), (0, 0, 1),
   (2/3, -1/3, -1/3), (-1/3, 2/3, -1/3), (-1/3, -1/3, 2/3)]

theorem system_solutions :
  ∀ x y z : ℚ, system x y z ↔ (x, y, z) ∈ solutions := by sorry

end NUMINAMATH_CALUDE_system_solutions_l694_69490


namespace NUMINAMATH_CALUDE_maggie_subscriptions_to_parents_l694_69497

-- Define the price per subscription
def price_per_subscription : ℕ := 5

-- Define the number of subscriptions sold to different people
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_next_door : ℕ := 2
def subscriptions_to_another_neighbor : ℕ := 2 * subscriptions_to_next_door

-- Define the total earnings
def total_earnings : ℕ := 55

-- Define the number of subscriptions sold to parents
def subscriptions_to_parents : ℕ := 4

-- Theorem to prove
theorem maggie_subscriptions_to_parents :
  subscriptions_to_parents * price_per_subscription +
  (subscriptions_to_grandfather + subscriptions_to_next_door + subscriptions_to_another_neighbor) * price_per_subscription =
  total_earnings :=
by sorry

end NUMINAMATH_CALUDE_maggie_subscriptions_to_parents_l694_69497


namespace NUMINAMATH_CALUDE_line_passes_through_point_l694_69411

theorem line_passes_through_point (a b : ℝ) (h : 3 * a + 2 * b = 5) :
  a * 6 + b * 4 - 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l694_69411


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l694_69481

/-- Proves that for a rectangular field with a square pond, given specific conditions, the ratio of length to width is 2:1 -/
theorem field_length_width_ratio (field_length field_width pond_side : ℝ) : 
  field_length = 28 →
  pond_side = 7 →
  field_length * field_width = 8 * pond_side * pond_side →
  field_length / field_width = 2 := by
  sorry

#check field_length_width_ratio

end NUMINAMATH_CALUDE_field_length_width_ratio_l694_69481


namespace NUMINAMATH_CALUDE_predicted_weight_for_178cm_l694_69417

/-- Regression equation for weight prediction based on height -/
def weight_prediction (height : ℝ) : ℝ := 0.72 * height - 58.2

/-- Theorem: The predicted weight for a person with height 178 cm is 69.96 kg -/
theorem predicted_weight_for_178cm :
  weight_prediction 178 = 69.96 := by sorry

end NUMINAMATH_CALUDE_predicted_weight_for_178cm_l694_69417


namespace NUMINAMATH_CALUDE_function_inequality_l694_69459

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ≥ 1, f x = x * Real.log x) →
  (∀ x ≥ 1, f x ≥ a * x - 1) →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l694_69459


namespace NUMINAMATH_CALUDE_e1_e2_form_basis_l694_69495

def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (5, 7)

def is_non_collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 ≠ 0

def forms_basis (v w : ℝ × ℝ) : Prop :=
  is_non_collinear v w

theorem e1_e2_form_basis : forms_basis e1 e2 := by
  sorry

end NUMINAMATH_CALUDE_e1_e2_form_basis_l694_69495


namespace NUMINAMATH_CALUDE_tim_watch_time_l694_69428

/-- The number of shows Tim watches -/
def num_shows : ℕ := 2

/-- The duration of the short show in hours -/
def short_show_duration : ℚ := 1/2

/-- The duration of the long show in hours -/
def long_show_duration : ℕ := 1

/-- The number of episodes of the short show -/
def short_show_episodes : ℕ := 24

/-- The number of episodes of the long show -/
def long_show_episodes : ℕ := 12

/-- The total number of hours Tim watched TV -/
def total_watch_time : ℚ := short_show_duration * short_show_episodes + long_show_duration * long_show_episodes

theorem tim_watch_time :
  total_watch_time = 24 := by sorry

end NUMINAMATH_CALUDE_tim_watch_time_l694_69428


namespace NUMINAMATH_CALUDE_parabola_shift_l694_69469

def original_parabola (x : ℝ) : ℝ := -x^2

def shifted_parabola (x : ℝ) : ℝ := -(x - 2)^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l694_69469


namespace NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l694_69429

theorem product_of_squares_and_fourth_powers (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_sum_squares : a^2 + b^2 = 5)
  (h_sum_fourth_powers : a^4 + b^4 = 17) : 
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l694_69429


namespace NUMINAMATH_CALUDE_slope_angle_sqrt3_l694_69403

/-- The slope angle of a line with slope √3 is 60 degrees. -/
theorem slope_angle_sqrt3 : ∃ θ : Real, 
  0 ≤ θ ∧ θ < Real.pi ∧ 
  Real.tan θ = Real.sqrt 3 ∧ 
  θ = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_sqrt3_l694_69403


namespace NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l694_69436

theorem x_power_2187_minus_reciprocal (x : ℂ) (h : x - 1/x = 2*I*Real.sqrt 2) : 
  x^2187 - 1/(x^2187) = -22*I*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l694_69436


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l694_69468

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 
  1 / x + 1 / y = 8 / 75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l694_69468


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l694_69449

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 9) →
  (a 4 + a 5 = 27 ∨ a 4 + a 5 = -27) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l694_69449


namespace NUMINAMATH_CALUDE_stacy_berries_l694_69460

/-- The number of berries each person has -/
structure BerryDistribution where
  sophie : ℕ
  sylar : ℕ
  steve : ℕ
  stacy : ℕ

/-- The conditions of the berry distribution problem -/
def valid_distribution (b : BerryDistribution) : Prop :=
  b.sylar = 5 * b.sophie ∧
  b.steve = 2 * b.sylar ∧
  b.stacy = 4 * b.steve ∧
  b.sophie + b.sylar + b.steve + b.stacy = 2200

/-- Theorem stating that Stacy has 1560 berries -/
theorem stacy_berries (b : BerryDistribution) (h : valid_distribution b) : b.stacy = 1560 := by
  sorry

end NUMINAMATH_CALUDE_stacy_berries_l694_69460


namespace NUMINAMATH_CALUDE_linear_systems_solution_l694_69445

/-- Given two systems of linear equations with the same solution, 
    prove the solution, the values of a and b, and a related expression. -/
theorem linear_systems_solution :
  ∃ (x y a b : ℝ),
    -- First system of equations
    (2 * x + 5 * y = -26 ∧ a * x - b * y = -4) ∧
    -- Second system of equations
    (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
    -- The solution
    (x = 2 ∧ y = -6) ∧
    -- The values of a and b
    (a = 1 ∧ b = -1) ∧
    -- The value of the expression
    ((2 * a + b) ^ 2020 = 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_systems_solution_l694_69445


namespace NUMINAMATH_CALUDE_prob_no_shaded_correct_l694_69431

/-- Represents a rectangle in the 2 by 2005 grid -/
structure Rectangle where
  left : Fin 2006
  right : Fin 2006
  top : Fin 3
  bottom : Fin 3
  h_valid : left < right

/-- The total number of rectangles in the grid -/
def total_rectangles : ℕ := 3 * (1003 * 2005)

/-- The number of rectangles containing a shaded square -/
def shaded_rectangles : ℕ := 3 * (1003 * 1003)

/-- Predicate for whether a rectangle contains a shaded square -/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left ≤ 1003 ∧ r.right > 1003) ∨ (r.top = 0 ∧ r.bottom = 1) ∨ (r.top = 1 ∧ r.bottom = 2)

/-- The probability of choosing a rectangle that does not contain a shaded square -/
def prob_no_shaded : ℚ := 1002 / 2005

theorem prob_no_shaded_correct :
  (total_rectangles - shaded_rectangles : ℚ) / total_rectangles = prob_no_shaded := by
  sorry

end NUMINAMATH_CALUDE_prob_no_shaded_correct_l694_69431


namespace NUMINAMATH_CALUDE_unique_increasing_function_l694_69462

theorem unique_increasing_function :
  ∃! f : ℕ → ℕ,
    (∀ n m : ℕ, (2^m + 1) * f n * f (2^m * n) = 2^m * (f n)^2 + (f (2^m * n))^2 + (2^m - 1)^2 * n) ∧
    (∀ a b : ℕ, a < b → f a < f b) ∧
    (∀ n : ℕ, f n = n + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_increasing_function_l694_69462


namespace NUMINAMATH_CALUDE_oranges_in_sack_l694_69444

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 66

/-- The number of days of harvest -/
def harvest_days : ℕ := 87

/-- The total number of oranges after the harvest -/
def total_oranges : ℕ := 143550

/-- The number of oranges in each sack -/
def oranges_per_sack : ℕ := total_oranges / (sacks_per_day * harvest_days)

theorem oranges_in_sack : oranges_per_sack = 25 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_sack_l694_69444


namespace NUMINAMATH_CALUDE_odd_function_fixed_point_l694_69412

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The theorem states that if f is an odd function on ℝ,
    then (-1, -2) is a point on the graph of y = f(x+1) - 2 -/
theorem odd_function_fixed_point (f : ℝ → ℝ) (h : IsOdd f) :
  f 0 - 2 = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_fixed_point_l694_69412


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l694_69419

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, -2) and (-4, 8) is 6. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (10, -2)
  let p2 : ℝ × ℝ := (-4, 8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l694_69419


namespace NUMINAMATH_CALUDE_car_hire_cost_for_b_l694_69416

theorem car_hire_cost_for_b (total_cost : ℚ) (time_a time_b time_c : ℚ) : 
  total_cost = 520 →
  time_a = 7 →
  time_b = 8 →
  time_c = 11 →
  time_b / (time_a + time_b + time_c) * total_cost = 160 := by
  sorry

end NUMINAMATH_CALUDE_car_hire_cost_for_b_l694_69416


namespace NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l694_69456

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  ∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 ∧
  ∃ y : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l694_69456


namespace NUMINAMATH_CALUDE_employee_salaries_exist_l694_69455

/-- Proves the existence of salaries for three employees satisfying given conditions --/
theorem employee_salaries_exist :
  ∃ (m n p : ℝ),
    (m + n + p = 1750) ∧
    (m = 1.3 * n) ∧
    (p = 0.9 * (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_employee_salaries_exist_l694_69455


namespace NUMINAMATH_CALUDE_negation_of_universal_conditional_l694_69489

theorem negation_of_universal_conditional (P : ℝ → Prop) :
  (¬∀ x : ℝ, x ≥ 2 → P x) ↔ (∃ x : ℝ, x < 2 ∧ ¬P x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_conditional_l694_69489


namespace NUMINAMATH_CALUDE_tangent_point_bisects_second_side_l694_69437

/-- A pentagon inscribed around a circle -/
structure InscribedPentagon where
  /-- The lengths of the sides of the pentagon -/
  sides : Fin 5 → ℕ
  /-- The first and third sides have length 1 -/
  first_third_sides_one : sides 0 = 1 ∧ sides 2 = 1
  /-- The point where the circle touches the second side of the pentagon -/
  tangent_point : ℝ
  /-- The tangent point is between 0 and the length of the second side -/
  tangent_point_valid : 0 < tangent_point ∧ tangent_point < sides 1

/-- The theorem stating that the tangent point divides the second side into two equal segments -/
theorem tangent_point_bisects_second_side (p : InscribedPentagon) :
  p.tangent_point = (p.sides 1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_bisects_second_side_l694_69437


namespace NUMINAMATH_CALUDE_max_garden_area_l694_69488

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden given its dimensions. -/
def area (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular garden given its dimensions. -/
def perimeter (d : GardenDimensions) : ℝ := 2 * (d.length + d.width)

/-- Theorem: The maximum area of a rectangular garden with 320 feet of fencing
    and length no less than 100 feet is 6000 square feet, achieved when
    the length is 100 feet and the width is 60 feet. -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    perimeter d = 320 ∧
    d.length ≥ 100 ∧
    area d = 6000 ∧
    (∀ (d' : GardenDimensions), perimeter d' = 320 ∧ d'.length ≥ 100 → area d' ≤ area d) :=
by sorry

end NUMINAMATH_CALUDE_max_garden_area_l694_69488


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l694_69433

theorem arithmetic_sequence_common_difference :
  ∀ (a d : ℚ) (n : ℕ),
    a = 2 →
    a + (n - 1) * d = 20 →
    n * (a + (a + (n - 1) * d)) / 2 = 132 →
    d = 18 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l694_69433


namespace NUMINAMATH_CALUDE_not_prime_expression_l694_69413

theorem not_prime_expression (n k : ℤ) (h1 : n > 2) (h2 : k ≠ n) :
  ¬ Prime (n^2 - k*n + k - 1) :=
by sorry

end NUMINAMATH_CALUDE_not_prime_expression_l694_69413


namespace NUMINAMATH_CALUDE_area_triangle_STU_l694_69438

/-- Square pyramid with given dimensions and points -/
structure SquarePyramid where
  -- Base side length
  base : ℝ
  -- Altitude
  height : ℝ
  -- Point S position ratio
  s_ratio : ℝ
  -- Point T position ratio
  t_ratio : ℝ
  -- Point U position ratio
  u_ratio : ℝ

/-- Theorem stating the area of triangle STU in the square pyramid -/
theorem area_triangle_STU (p : SquarePyramid) 
  (h_base : p.base = 4)
  (h_height : p.height = 8)
  (h_s : p.s_ratio = 1/4)
  (h_t : p.t_ratio = 1/2)
  (h_u : p.u_ratio = 3/4) :
  ∃ (area : ℝ), area = 7.5 ∧ 
  area = (1/2) * Real.sqrt ((p.s_ratio * p.height)^2 + (p.base/2)^2) * 
         (p.u_ratio * Real.sqrt (p.height^2 + (p.base/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_STU_l694_69438


namespace NUMINAMATH_CALUDE_g_of_four_value_l694_69424

/-- A function from positive reals to positive reals satisfying certain conditions -/
def G := {g : ℝ → ℝ // ∀ x > 0, g x > 0 ∧ g 1 = 1 ∧ g (x^2 * g x) = x * g (x^2) + g x}

theorem g_of_four_value (g : G) : g.val 4 = 36/23 := by
  sorry

end NUMINAMATH_CALUDE_g_of_four_value_l694_69424


namespace NUMINAMATH_CALUDE_bike_cost_l694_69439

/-- The cost of Jenn's bike given her savings in quarters and leftover money -/
theorem bike_cost (num_jars : ℕ) (quarters_per_jar : ℕ) (quarter_value : ℚ) (leftover : ℕ) : 
  num_jars = 5 →
  quarters_per_jar = 160 →
  quarter_value = 1/4 →
  leftover = 20 →
  (num_jars * quarters_per_jar : ℕ) * quarter_value - leftover = 200 := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_l694_69439


namespace NUMINAMATH_CALUDE_max_min_values_l694_69451

theorem max_min_values (x y : ℝ) (h : |5*x + y| + |5*x - y| = 20) :
  (∃ (a b : ℝ), a^2 - a*b + b^2 = 124 ∧ 
   ∀ (c d : ℝ), |5*c + d| + |5*c - d| = 20 → c^2 - c*d + d^2 ≤ 124) ∧
  (∃ (a b : ℝ), a^2 - a*b + b^2 = 4 ∧ 
   ∀ (c d : ℝ), |5*c + d| + |5*c - d| = 20 → c^2 - c*d + d^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l694_69451


namespace NUMINAMATH_CALUDE_remainder_polynomial_l694_69442

-- Define the polynomials
variable (z : ℂ)
variable (Q : ℂ → ℂ)
variable (R : ℂ → ℂ)

-- State the theorem
theorem remainder_polynomial :
  (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) →
  (∃ a b : ℂ, ∀ z, R z = a * z + b) →
  (∀ z, R z = z + 1) :=
by sorry

end NUMINAMATH_CALUDE_remainder_polynomial_l694_69442


namespace NUMINAMATH_CALUDE_polynomial_simplification_l694_69487

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 4 * r^2 + 5 * r - 3) - (r^3 + 6 * r^2 + 8 * r - 7) = r^3 - 2 * r^2 - 3 * r + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l694_69487


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l694_69482

/-- Theorem: Error percent in rectangle area calculation --/
theorem rectangle_area_error_percent
  (L W : ℝ)  -- L and W represent the actual length and width of the rectangle
  (h_positive_L : L > 0)
  (h_positive_W : W > 0)
  (measured_length : ℝ := 1.05 * L)  -- Length measured 5% in excess
  (measured_width : ℝ := 0.96 * W)   -- Width measured 4% in deficit
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  : (calculated_area - actual_area) / actual_area * 100 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l694_69482


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l694_69425

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l694_69425


namespace NUMINAMATH_CALUDE_class_average_l694_69473

theorem class_average (students1 : ℕ) (average1 : ℚ) (students2 : ℕ) (average2 : ℚ) :
  students1 = 15 →
  average1 = 73/100 →
  students2 = 10 →
  average2 = 88/100 →
  (students1 * average1 + students2 * average2) / (students1 + students2) = 79/100 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l694_69473


namespace NUMINAMATH_CALUDE_total_harvest_is_2000_l694_69452

/-- Represents the harvest of tomatoes over three days -/
structure TomatoHarvest where
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total harvest over three days -/
def total_harvest (h : TomatoHarvest) : ℕ :=
  h.wednesday + h.thursday + h.friday

/-- Theorem stating the total harvest is 2000 kg given the conditions -/
theorem total_harvest_is_2000 (h : TomatoHarvest) 
  (h_wed : h.wednesday = 400)
  (h_thu : h.thursday = h.wednesday / 2)
  (h_fri : h.friday - 700 = 700) :
  total_harvest h = 2000 := by
  sorry

#check total_harvest_is_2000

end NUMINAMATH_CALUDE_total_harvest_is_2000_l694_69452


namespace NUMINAMATH_CALUDE_matrix_operation_result_l694_69420

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 6, 1]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-7, 8; 3, -5]

theorem matrix_operation_result : 
  2 • A + B = !![1, 2; 15, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_operation_result_l694_69420


namespace NUMINAMATH_CALUDE_line_slope_product_l694_69432

/-- Given two lines L₁ and L₂ with equations y = 3mx and y = nx respectively,
    where L₁ makes three times the angle with the horizontal as L₂,
    L₁ has 3 times the slope of L₂, and L₁ is not vertical,
    prove that the product mn equals 9/4. -/
theorem line_slope_product (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ 
               3 * m = Real.tan θ₁ ∧ 
               n = Real.tan θ₂ ∧ 
               m = 3 * n ∧ 
               m ≠ 0) →
  m * n = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_line_slope_product_l694_69432


namespace NUMINAMATH_CALUDE_toms_initial_investment_l694_69457

theorem toms_initial_investment (t j k : ℝ) : 
  t + j + k = 1200 →
  t - 150 + 3*j + 3*k = 1800 →
  t = 825 := by
sorry

end NUMINAMATH_CALUDE_toms_initial_investment_l694_69457


namespace NUMINAMATH_CALUDE_complex_simplification_l694_69402

theorem complex_simplification :
  (7 - 4 * Complex.I) - (2 + 6 * Complex.I) + (3 - 3 * Complex.I) = 8 - 13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l694_69402
