import Mathlib

namespace NUMINAMATH_CALUDE_total_buyers_is_140_l3425_342576

/-- The number of buyers in a grocery store over three consecutive days -/
structure BuyerCount where
  day_before_yesterday : ℕ
  yesterday : ℕ
  today : ℕ

/-- Conditions for the buyer count problem -/
def buyer_count_conditions (b : BuyerCount) : Prop :=
  b.today = b.yesterday + 40 ∧
  b.yesterday = b.day_before_yesterday / 2 ∧
  b.day_before_yesterday = 50

/-- The total number of buyers over three days -/
def total_buyers (b : BuyerCount) : ℕ :=
  b.day_before_yesterday + b.yesterday + b.today

/-- Theorem stating that given the conditions, the total number of buyers is 140 -/
theorem total_buyers_is_140 (b : BuyerCount) (h : buyer_count_conditions b) :
  total_buyers b = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_buyers_is_140_l3425_342576


namespace NUMINAMATH_CALUDE_square_root_equality_l3425_342541

theorem square_root_equality (a b : ℝ) : 
  (a^2 + b^2)^2 = (4*a - 6*b + 13)^2 → (a^2 + b^2)^2 = 169 := by
sorry

end NUMINAMATH_CALUDE_square_root_equality_l3425_342541


namespace NUMINAMATH_CALUDE_balloons_distribution_l3425_342587

theorem balloons_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 236) (h2 : num_friends = 10) :
  total_balloons % num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloons_distribution_l3425_342587


namespace NUMINAMATH_CALUDE_food_preferences_l3425_342503

theorem food_preferences (total students : ℕ)
  (french_fries burgers pizza tacos : ℕ)
  (fries_burgers fries_pizza fries_tacos : ℕ)
  (burgers_pizza burgers_tacos pizza_tacos : ℕ)
  (all_four : ℕ)
  (h_total : total = 30)
  (h_fries : french_fries = 18)
  (h_burgers : burgers = 12)
  (h_pizza : pizza = 14)
  (h_tacos : tacos = 10)
  (h_fries_burgers : fries_burgers = 8)
  (h_fries_pizza : fries_pizza = 6)
  (h_fries_tacos : fries_tacos = 4)
  (h_burgers_pizza : burgers_pizza = 5)
  (h_burgers_tacos : burgers_tacos = 3)
  (h_pizza_tacos : pizza_tacos = 7)
  (h_all_four : all_four = 2) :
  total - (french_fries + burgers + pizza + tacos
           - fries_burgers - fries_pizza - fries_tacos
           - burgers_pizza - burgers_tacos - pizza_tacos
           + all_four) = 11 := by
  sorry

end NUMINAMATH_CALUDE_food_preferences_l3425_342503


namespace NUMINAMATH_CALUDE_fifth_number_is_one_l3425_342520

def random_table : List (List Nat) := [
  [7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198],
  [3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481]
]

def is_valid_number (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 20

def extract_valid_numbers (lst : List Nat) : List Nat :=
  lst.filter (λ n => is_valid_number n)

def select_numbers (table : List (List Nat)) : List Nat :=
  let flattened := table.join
  let valid_numbers := extract_valid_numbers flattened
  valid_numbers.take 5

theorem fifth_number_is_one :
  (select_numbers random_table).get? 4 = some 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_number_is_one_l3425_342520


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l3425_342554

def geometric_series : ℕ → ℚ
  | 0 => 7/8
  | 1 => -21/32
  | 2 => 63/128
  | (n+3) => (-3/4) * geometric_series n

theorem common_ratio_of_geometric_series :
  ∀ n : ℕ, n ≥ 1 → geometric_series (n+1) / geometric_series n = -3/4 :=
by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l3425_342554


namespace NUMINAMATH_CALUDE_bottles_left_on_shelf_l3425_342557

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_purchase : ℕ) (harry_purchase : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_purchase = 5)
  (h3 : harry_purchase = 6) :
  initial_bottles - (jason_purchase + harry_purchase) = 24 :=
by sorry

end NUMINAMATH_CALUDE_bottles_left_on_shelf_l3425_342557


namespace NUMINAMATH_CALUDE_regular_bottle_is_16_oz_l3425_342532

/-- Represents Jon's drinking habits and fluid intake --/
structure DrinkingHabits where
  awake_hours : ℕ := 16
  drinking_interval : ℕ := 4
  larger_bottles_per_day : ℕ := 2
  larger_bottle_size_factor : ℚ := 1.25
  weekly_fluid_intake : ℕ := 728

/-- Calculates the size of Jon's regular water bottle in ounces --/
def regular_bottle_size (h : DrinkingHabits) : ℚ :=
  h.weekly_fluid_intake / (7 * (h.awake_hours / h.drinking_interval + h.larger_bottles_per_day * h.larger_bottle_size_factor))

/-- Theorem stating that Jon's regular water bottle size is 16 ounces --/
theorem regular_bottle_is_16_oz (h : DrinkingHabits) : regular_bottle_size h = 16 := by
  sorry

end NUMINAMATH_CALUDE_regular_bottle_is_16_oz_l3425_342532


namespace NUMINAMATH_CALUDE_power_function_properties_l3425_342552

-- Define the power function f(x) = x^α
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 3 / Real.log 9)

-- Theorem statement
theorem power_function_properties :
  -- The function passes through (9,3)
  f 9 = 3 ∧
  -- f(x) is increasing on its domain
  (∀ x y, x < y → x > 0 → y > 0 → f x < f y) ∧
  -- When x ≥ 4, f(x) ≥ 2
  (∀ x, x ≥ 4 → f x ≥ 2) ∧
  -- When x₂ > x₁ > 0, (f(x₁) + f(x₂))/2 < f((x₁ + x₂)/2)
  (∀ x₁ x₂, x₂ > x₁ → x₁ > 0 → (f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_power_function_properties_l3425_342552


namespace NUMINAMATH_CALUDE_loan_payback_calculation_l3425_342531

/-- Calculates the total amount to be paid back for a loan with interest -/
def total_payback (principal : ℝ) (interest_rate : ℝ) : ℝ :=
  principal * (1 + interest_rate)

/-- Theorem: Given a loan of $1200 with a 10% interest rate, the total amount to be paid back is $1320 -/
theorem loan_payback_calculation :
  total_payback 1200 0.1 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_loan_payback_calculation_l3425_342531


namespace NUMINAMATH_CALUDE_cone_height_l3425_342505

theorem cone_height (r l h : ℝ) : 
  r = 1 → l = 4 → l^2 = r^2 + h^2 → h = Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_cone_height_l3425_342505


namespace NUMINAMATH_CALUDE_distance_to_focus_l3425_342595

/-- Given a parabola x = (1/2)y², prove that the distance from a point P(1, y) on the parabola to its focus F is 3/2 -/
theorem distance_to_focus (y : ℝ) (h : 1 = (1/2) * y^2) : 
  let p : ℝ × ℝ := (1, y)
  let f : ℝ × ℝ := ((1/4), 0)  -- Focus of the parabola x = (1/2)y²
  ‖p - f‖ = 3/2 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3425_342595


namespace NUMINAMATH_CALUDE_beautiful_labeling_theorem_l3425_342572

/-- A beautiful labeling of n + 1 equally spaced points on a circle. -/
def BeautifulLabeling (n : ℕ) : Type := sorry

/-- The number of beautiful labelings for n + 1 points. -/
def M (n : ℕ) : ℕ := sorry

/-- The number of ordered pairs (x, y) of positive integers 
    such that x + y ≤ n and gcd(x, y) = 1. -/
def N (n : ℕ) : ℕ := sorry

/-- Theorem: For any integer n ≥ 3, M(n) = N(n) + 1 -/
theorem beautiful_labeling_theorem (n : ℕ) (h : n ≥ 3) : M n = N n + 1 := by
  sorry

end NUMINAMATH_CALUDE_beautiful_labeling_theorem_l3425_342572


namespace NUMINAMATH_CALUDE_jean_needs_four_more_packs_l3425_342511

/-- Represents the number of cupcakes in a small pack -/
def small_pack : ℕ := 10

/-- Represents the number of cupcakes in a large pack -/
def large_pack : ℕ := 15

/-- Represents the number of large packs Jean initially bought -/
def initial_packs : ℕ := 4

/-- Represents the total number of children in the orphanage -/
def total_children : ℕ := 100

/-- Calculates the number of additional packs of 10 cupcakes Jean needs to buy -/
def additional_packs_needed : ℕ :=
  (total_children - initial_packs * large_pack) / small_pack

theorem jean_needs_four_more_packs :
  additional_packs_needed = 4 :=
sorry

end NUMINAMATH_CALUDE_jean_needs_four_more_packs_l3425_342511


namespace NUMINAMATH_CALUDE_expected_heads_is_94_l3425_342579

/-- The probability of a coin landing on heads after at most four flips -/
def prob_heads : ℚ := 15 / 16

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The expected number of coins landing on heads -/
def expected_heads : ℚ := num_coins * prob_heads

theorem expected_heads_is_94 :
  ⌊expected_heads⌋ = 94 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_94_l3425_342579


namespace NUMINAMATH_CALUDE_triangle_side_length_l3425_342575

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: In triangle ABC, if AB = 2, BC = 5, and the perimeter is even, then AC = 5 -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 5)
  (h3 : ∃ n : ℕ, t.perimeter = 2 * n) :
  t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3425_342575


namespace NUMINAMATH_CALUDE_fred_balloons_l3425_342540

theorem fred_balloons (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 709 → given_away = 221 → remaining = initial - given_away → remaining = 488 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloons_l3425_342540


namespace NUMINAMATH_CALUDE_turtle_ratio_l3425_342530

/-- Prove that given the conditions, the ratio of turtles Kris has to Kristen has is 1:4 -/
theorem turtle_ratio : 
  ∀ (kris trey kristen : ℕ),
  trey = 5 * kris →
  kris + trey + kristen = 30 →
  kristen = 12 →
  kris.gcd kristen = 3 →
  (kris / 3 : ℚ) / (kristen / 3 : ℚ) = 1 / 4 := by
sorry


end NUMINAMATH_CALUDE_turtle_ratio_l3425_342530


namespace NUMINAMATH_CALUDE_fraction_inequality_l3425_342528

theorem fraction_inequality (a b : ℝ) :
  (b > 0 ∧ 0 > a → 1/a < 1/b) ∧
  (0 > a ∧ a > b → 1/a < 1/b) ∧
  (a > b ∧ b > 0 → 1/a < 1/b) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3425_342528


namespace NUMINAMATH_CALUDE_parabola_equation_l3425_342566

/-- A parabola with vertex at the origin and focus on the line x - 2y - 2 = 0 --/
structure Parabola where
  /-- The focus of the parabola lies on this line --/
  focus_line : {(x, y) : ℝ × ℝ | x - 2*y - 2 = 0}
  /-- The axis of symmetry is either the x-axis or y-axis --/
  symmetry_axis : (Unit → Prop) ⊕ (Unit → Prop)

/-- The standard equation of the parabola is either y² = 8x or x² = -4y --/
theorem parabola_equation (p : Parabola) :
  (∃ (x y : ℝ), y^2 = 8*x) ∨ (∃ (x y : ℝ), x^2 = -4*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3425_342566


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3425_342577

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 9*x + 14 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 9 ∧ s₁ * s₂ = 14 ∧ s₁^2 + s₂^2 = 53 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3425_342577


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3425_342586

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 5 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + m*y - 5 = 0 ∧ y = 5) :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3425_342586


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l3425_342507

/-- A triangle with side lengths a, b, and c is obtuse if and only if a² + b² < c² for some permutation of its sides. --/
def IsObtuseTriangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)

/-- The range of possible values for the third side of an obtuse triangle with two sides of length 3 and 4. --/
theorem obtuse_triangle_side_range :
  ∀ x : ℝ, IsObtuseTriangle 3 4 x ↔ (5 < x ∧ x < 7) ∨ (1 < x ∧ x < Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l3425_342507


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_l3425_342519

theorem triangle_tangent_sum (A B C : Real) : 
  A + B + C = π →  -- angle sum property of triangle
  A + C = 2 * B →  -- given condition
  Real.tan (A / 2) + Real.tan (C / 2) + Real.sqrt 3 * Real.tan (A / 2) * Real.tan (C / 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_l3425_342519


namespace NUMINAMATH_CALUDE_meenas_bottle_caps_l3425_342515

theorem meenas_bottle_caps (initial : ℕ) : 
  (initial : ℚ) * (1 + 0.4) * (1 - 0.2) = initial + 21 → initial = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_meenas_bottle_caps_l3425_342515


namespace NUMINAMATH_CALUDE_hard_hats_remaining_is_51_l3425_342549

/-- Calculates the remaining hard hats after transactions --/
def remaining_hard_hats (pink_initial green_initial yellow_initial : ℕ) 
  (carl_pink_taken john_pink_taken : ℕ) : ℕ :=
  let john_green_taken := 2 * john_pink_taken
  let pink_after_taken := pink_initial - carl_pink_taken - john_pink_taken
  let green_after_taken := green_initial - john_green_taken
  let carl_pink_returned := carl_pink_taken / 2
  let john_pink_returned := john_pink_taken / 3
  let john_green_returned := john_green_taken / 3
  let pink_final := pink_after_taken + carl_pink_returned + john_pink_returned
  let green_final := green_after_taken + john_green_returned
  pink_final + green_final + yellow_initial

/-- Theorem stating that the total number of hard hats remaining is 51 --/
theorem hard_hats_remaining_is_51 : 
  remaining_hard_hats 26 15 24 4 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_hard_hats_remaining_is_51_l3425_342549


namespace NUMINAMATH_CALUDE_nice_people_count_l3425_342578

/-- Represents the number of nice people for a given name and total count -/
def nice_count (name : String) (total : ℕ) : ℕ :=
  match name with
  | "Barry" => total
  | "Kevin" => total / 2
  | "Julie" => total * 3 / 4
  | "Joe" => total / 10
  | _ => 0

/-- The total number of nice people in the crowd -/
def total_nice_people : ℕ :=
  nice_count "Barry" 24 + nice_count "Kevin" 20 + nice_count "Julie" 80 + nice_count "Joe" 50

theorem nice_people_count : total_nice_people = 99 := by
  sorry

end NUMINAMATH_CALUDE_nice_people_count_l3425_342578


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3425_342539

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a - 2 * Complex.I) / (1 + 2 * Complex.I)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3425_342539


namespace NUMINAMATH_CALUDE_kekai_sales_ratio_l3425_342569

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := shirts_sold * shirt_price + pants_sold * pants_price

def money_given_to_parents : ℕ := total_earnings - money_left

theorem kekai_sales_ratio :
  (money_given_to_parents : ℚ) / total_earnings = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_kekai_sales_ratio_l3425_342569


namespace NUMINAMATH_CALUDE_total_books_is_24_l3425_342502

/-- The number of boxes Victor bought -/
def num_boxes : ℕ := 8

/-- The number of books in each box -/
def books_per_box : ℕ := 3

/-- Theorem: The total number of books Victor bought is 24 -/
theorem total_books_is_24 : num_boxes * books_per_box = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_24_l3425_342502


namespace NUMINAMATH_CALUDE_find_t_l3425_342501

theorem find_t : ∃ t : ℝ, 
  (∀ x : ℝ, |2*x + t| - t ≤ 8 ↔ -5 ≤ x ∧ x ≤ 4) → 
  t = 1 := by sorry

end NUMINAMATH_CALUDE_find_t_l3425_342501


namespace NUMINAMATH_CALUDE_alternating_student_arrangements_l3425_342585

def num_male_students : ℕ := 4
def num_female_students : ℕ := 5

theorem alternating_student_arrangements :
  (num_male_students.factorial * num_female_students.factorial : ℕ) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_alternating_student_arrangements_l3425_342585


namespace NUMINAMATH_CALUDE_range_of_x_in_triangle_l3425_342537

/-- Given a triangle ABC with vectors AB and AC, prove the range of x -/
theorem range_of_x_in_triangle (x : ℝ) : 
  let AB : ℝ × ℝ := (x, 2*x)
  let AC : ℝ × ℝ := (3*x, 2)
  -- Dot product is negative for obtuse angle
  (x * (3*x) + (2*x) * 2 < 0) →
  -- x is in the open interval (-4/3, 0)
  -4/3 < x ∧ x < 0 :=
by sorry


end NUMINAMATH_CALUDE_range_of_x_in_triangle_l3425_342537


namespace NUMINAMATH_CALUDE_oreo_milk_purchases_l3425_342592

/-- The number of different flavors of oreos --/
def oreo_flavors : ℕ := 6

/-- The number of different flavors of milk --/
def milk_flavors : ℕ := 4

/-- The total number of products Alpha and Beta purchased --/
def total_products : ℕ := 4

/-- The number of ways Alpha and Beta could have left the store --/
def purchase_combinations : ℕ := 2561

/-- Theorem stating the number of ways Alpha and Beta could have left the store --/
theorem oreo_milk_purchases :
  (oreo_flavors = 6) →
  (milk_flavors = 4) →
  (total_products = 4) →
  purchase_combinations = 2561 :=
by sorry

end NUMINAMATH_CALUDE_oreo_milk_purchases_l3425_342592


namespace NUMINAMATH_CALUDE_train_tunnel_time_l3425_342553

/-- Calculates the time taken for a train to pass through a tunnel -/
theorem train_tunnel_time (train_length : ℝ) (pole_passing_time : ℝ) (tunnel_length : ℝ) :
  train_length = 500 →
  pole_passing_time = 20 →
  tunnel_length = 500 →
  (train_length + tunnel_length) / (train_length / pole_passing_time) = 40 := by
  sorry


end NUMINAMATH_CALUDE_train_tunnel_time_l3425_342553


namespace NUMINAMATH_CALUDE_wire_shapes_area_difference_l3425_342550

theorem wire_shapes_area_difference :
  let wire_length : ℝ := 52
  let square_side : ℝ := wire_length / 4
  let rect_width : ℝ := 15
  let rect_length : ℝ := (wire_length / 2) - rect_width
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_width * rect_length
  square_area - rect_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_wire_shapes_area_difference_l3425_342550


namespace NUMINAMATH_CALUDE_no_intersection_condition_l3425_342534

theorem no_intersection_condition (k : ℝ) : 
  -1 ≤ k ∧ k ≤ 1 → 
  (∀ x : ℝ, x = k * π / 2 → ¬∃ y : ℝ, y = Real.tan (2 * x + π / 4)) ↔ 
  (k = 1 / 4 ∨ k = -3 / 4) := by
sorry

end NUMINAMATH_CALUDE_no_intersection_condition_l3425_342534


namespace NUMINAMATH_CALUDE_article_price_decrease_l3425_342522

theorem article_price_decrease (price_after_decrease : ℝ) (decrease_percentage : ℝ) :
  price_after_decrease = 200 ∧ decrease_percentage = 20 →
  (price_after_decrease / (1 - decrease_percentage / 100)) = 250 :=
by sorry

end NUMINAMATH_CALUDE_article_price_decrease_l3425_342522


namespace NUMINAMATH_CALUDE_glendas_average_speed_l3425_342565

/-- Calculates the average speed given initial and final odometer readings and total time -/
def average_speed (initial_reading : ℕ) (final_reading : ℕ) (total_time : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

/-- Theorem: Glenda's average speed is 55 miles per hour -/
theorem glendas_average_speed :
  let initial_reading := 1221
  let final_reading := 1881
  let total_time := 12
  average_speed initial_reading final_reading total_time = 55 := by
  sorry

end NUMINAMATH_CALUDE_glendas_average_speed_l3425_342565


namespace NUMINAMATH_CALUDE_square_difference_l3425_342596

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3425_342596


namespace NUMINAMATH_CALUDE_fraction_division_equality_l3425_342516

theorem fraction_division_equality : (-1/12 + 1/3 - 1/2) / (-1/18) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l3425_342516


namespace NUMINAMATH_CALUDE_intersection_point_l3425_342529

-- Define the two linear functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := -2*x + 6

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 ∧ p = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3425_342529


namespace NUMINAMATH_CALUDE_inserted_eights_composite_l3425_342584

theorem inserted_eights_composite (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (1880 * 10^n - 611) / 9 = a * b :=
sorry

end NUMINAMATH_CALUDE_inserted_eights_composite_l3425_342584


namespace NUMINAMATH_CALUDE_expression_simplification_l3425_342560

theorem expression_simplification (a c x z : ℝ) :
  (c * x * (a^3 * x^3 + 3 * a^3 * z^3 + c^3 * z^3) + a * z * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (c * x + a * z) = 
  a^3 * x^3 + 3 * a^3 * z^3 + c^3 * z^3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3425_342560


namespace NUMINAMATH_CALUDE_tylers_age_l3425_342556

theorem tylers_age (tyler_age brother_age : ℕ) : 
  tyler_age + 3 = brother_age →
  tyler_age + brother_age = 11 →
  tyler_age = 4 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l3425_342556


namespace NUMINAMATH_CALUDE_pinwheel_area_is_six_l3425_342544

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a kite in the pinwheel -/
structure Kite where
  center : GridPoint
  vertex1 : GridPoint
  vertex2 : GridPoint
  vertex3 : GridPoint

/-- Represents a pinwheel shape -/
structure Pinwheel where
  kites : Fin 4 → Kite
  grid_size : Nat
  h_grid_size : grid_size = 5

/-- Calculates the area of a pinwheel -/
noncomputable def pinwheel_area (p : Pinwheel) : ℝ :=
  sorry

/-- Theorem stating that the area of the described pinwheel is 6 square units -/
theorem pinwheel_area_is_six (p : Pinwheel) : pinwheel_area p = 6 := by
  sorry

end NUMINAMATH_CALUDE_pinwheel_area_is_six_l3425_342544


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3425_342506

theorem quadratic_inequality_solution (x : ℝ) :
  -5 * x^2 + 7 * x + 2 > 0 ↔ 
  x > ((-7 - Real.sqrt 89) / -10) ∧ x < ((-7 + Real.sqrt 89) / -10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3425_342506


namespace NUMINAMATH_CALUDE_theater_seat_interpretation_l3425_342581

/-- Represents a theater seat as an ordered pair of natural numbers -/
structure TheaterSeat :=
  (row : ℕ)
  (seat : ℕ)

/-- Interprets a TheaterSeat as a description -/
def interpret (s : TheaterSeat) : String :=
  s!"seat {s.seat} in row {s.row}"

theorem theater_seat_interpretation :
  interpret ⟨6, 2⟩ = "seat 2 in row 6" := by
  sorry

end NUMINAMATH_CALUDE_theater_seat_interpretation_l3425_342581


namespace NUMINAMATH_CALUDE_kamals_chemistry_marks_l3425_342546

theorem kamals_chemistry_marks 
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 65)
  (h3 : physics_marks = 82)
  (h4 : biology_marks = 85)
  (h5 : average_marks = 79)
  (h6 : num_subjects = 5)
  : ∃ (chemistry_marks : ℕ), 
    (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks) / num_subjects = average_marks ∧ 
    chemistry_marks = 67 :=
by sorry

end NUMINAMATH_CALUDE_kamals_chemistry_marks_l3425_342546


namespace NUMINAMATH_CALUDE_work_completion_proof_l3425_342598

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 9

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℕ := 72

/-- The number of men in the first group -/
def men_first_group : ℕ := 36

theorem work_completion_proof :
  (men_first_group : ℚ) * days_first_group = men_second_group * days_second_group :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3425_342598


namespace NUMINAMATH_CALUDE_sequence_sum_property_l3425_342533

/-- A sequence of positive terms satisfying a specific equation. -/
def sequence_a (n : ℕ) : ℝ :=
  sorry

/-- The sum of the first n terms of the sequence. -/
def S (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating the property of the sequence sum. -/
theorem sequence_sum_property :
  ∀ (n : ℕ), n ≥ 1 →
  (n * (n + 1) * (sequence_a n)^2 + (n^2 + n - 1) * sequence_a n - 1 = 0) →
  sequence_a n > 0 →
  2019 * S 2018 = 2018 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l3425_342533


namespace NUMINAMATH_CALUDE_problem_solving_percentage_l3425_342500

theorem problem_solving_percentage (total : ℕ) (multiple_choice : ℕ) : 
  total = 50 → multiple_choice = 10 → 
  (((total - multiple_choice) : ℚ) / total) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_percentage_l3425_342500


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l3425_342574

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 101)
  (h2 : finished_problems = 47)
  (h3 : remaining_pages = 6)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l3425_342574


namespace NUMINAMATH_CALUDE_square_times_square_minus_one_div_12_l3425_342512

theorem square_times_square_minus_one_div_12 (k : ℤ) : 
  12 ∣ (k^2 * (k^2 - 1)) := by sorry

end NUMINAMATH_CALUDE_square_times_square_minus_one_div_12_l3425_342512


namespace NUMINAMATH_CALUDE_problem_solution_l3425_342521

variable {S : Type*} [Inhabited S] [Nontrivial S]
variable (mul : S → S → S)

axiom mul_def : ∀ (a b : S), mul a (mul b a) = b

theorem problem_solution :
  (∀ (b : S), mul b (mul b b) = b) ∧
  (∀ (a b : S), mul (mul a b) (mul b (mul a b)) = b) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3425_342521


namespace NUMINAMATH_CALUDE_second_exam_sleep_for_average_85_l3425_342582

/-- Represents the relationship between sleep hours and test score -/
structure ExamData where
  sleep : ℝ
  score : ℝ

/-- The constant product of sleep hours and test score -/
def sleepScoreProduct (data : ExamData) : ℝ := data.sleep * data.score

theorem second_exam_sleep_for_average_85 
  (first_exam : ExamData)
  (h_first_exam : first_exam.sleep = 6 ∧ first_exam.score = 60)
  (h_inverse_relation : ∀ exam : ExamData, sleepScoreProduct exam = sleepScoreProduct first_exam)
  (second_exam : ExamData)
  (h_second_exam : second_exam.sleep = 3.3) :
  (first_exam.score + second_exam.score) / 2 = 85 := by
sorry

end NUMINAMATH_CALUDE_second_exam_sleep_for_average_85_l3425_342582


namespace NUMINAMATH_CALUDE_thirty_percent_of_two_hundred_l3425_342561

theorem thirty_percent_of_two_hundred : (30 / 100) * 200 = 60 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_two_hundred_l3425_342561


namespace NUMINAMATH_CALUDE_no_factorization_l3425_342551

/-- A polynomial in x and y with a parameter m -/
def polynomial (m : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + m*y + m

/-- A linear factor in x and y with integer coefficients -/
structure LinearFactor where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The product of two linear factors -/
def product (f g : LinearFactor) (x y : ℤ) : ℤ :=
  (f.a * x + f.b * y + f.c) * (g.a * x + g.b * y + g.c)

/-- Theorem stating that the polynomial cannot be factored for any integer m -/
theorem no_factorization :
  ∀ (m : ℤ), ¬∃ (f g : LinearFactor), ∀ (x y : ℤ),
    polynomial m x y = product f g x y :=
sorry

end NUMINAMATH_CALUDE_no_factorization_l3425_342551


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l3425_342568

theorem bernoulli_inequality (x : ℝ) (n : ℕ+) (h1 : x ≠ 0) (h2 : x > -1) :
  (1 + x)^(n : ℝ) ≥ n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l3425_342568


namespace NUMINAMATH_CALUDE_orange_cost_solution_l3425_342526

/-- Calculates the cost of an orange given the initial quantities, apple cost, and final earnings -/
def orange_cost (initial_apples initial_oranges : ℕ) (apple_cost : ℚ) 
  (final_apples final_oranges : ℕ) (total_earnings : ℚ) : ℚ :=
  let apples_sold := initial_apples - final_apples
  let oranges_sold := initial_oranges - final_oranges
  let apple_earnings := apples_sold * apple_cost
  let orange_earnings := total_earnings - apple_earnings
  orange_earnings / oranges_sold

theorem orange_cost_solution :
  orange_cost 50 40 (4/5) 10 6 49 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_solution_l3425_342526


namespace NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l3425_342555

-- Statement 1
theorem max_value_theorem (x : ℝ) (h : x < 1/2) :
  ∃ (max_val : ℝ), max_val = -1 ∧ 
  ∀ y : ℝ, y < 1/2 → 2*y + 1/(2*y - 1) ≤ max_val :=
sorry

-- Statement 2
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 2/b = 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2*Real.sqrt 2 ∧
  a*(b - 1) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l3425_342555


namespace NUMINAMATH_CALUDE_qr_equals_b_l3425_342508

-- Define the curve
def curve (c : ℝ) (x y : ℝ) : Prop := y / c = Real.cosh (x / c)

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem qr_equals_b (a b c : ℝ) (P Q R : Point) : 
  curve c P.x P.y →  -- P is on the curve
  curve c Q.x Q.y →  -- Q is on the curve
  P = Point.mk a b →  -- P has coordinates (a, b)
  Q = Point.mk 0 c →  -- Q has coordinates (0, c)
  R.y = 0 →  -- R is on the x-axis
  (∃ k : ℝ, R.x = k * Real.sinh (a / c)) →  -- R.x is proportional to sinh(a/c)
  (Q.y - R.y) / (Q.x - R.x) = -1 / Real.sinh (a / c) →  -- QR is parallel to normal at P
  Real.sqrt ((R.x - Q.x)^2 + (R.y - Q.y)^2) = b  -- Distance QR equals b
  := by sorry

end NUMINAMATH_CALUDE_qr_equals_b_l3425_342508


namespace NUMINAMATH_CALUDE_mlb_game_misses_l3425_342590

theorem mlb_game_misses (hits misses : ℕ) : 
  misses = 3 * hits → 
  hits + misses = 200 → 
  misses = 150 := by
sorry

end NUMINAMATH_CALUDE_mlb_game_misses_l3425_342590


namespace NUMINAMATH_CALUDE_exponent_calculation_correct_and_uses_operations_l3425_342523

-- Define the exponent operations
inductive ExponentOperation
  | multiplication
  | division
  | exponentiation
  | productExponentiation

-- Define a function to represent the calculation
def exponentCalculation (a : ℝ) : ℝ := (a^3 * a^2)^2

-- Define a function to represent the result of the calculation
def exponentResult (a : ℝ) : ℝ := a^10

-- Define a function to check if an operation is used in the calculation
def isOperationUsed (op : ExponentOperation) : Prop :=
  match op with
  | ExponentOperation.multiplication => True
  | ExponentOperation.exponentiation => True
  | ExponentOperation.productExponentiation => True
  | _ => False

-- Theorem stating that the calculation is correct and uses the specified operations
theorem exponent_calculation_correct_and_uses_operations (a : ℝ) :
  exponentCalculation a = exponentResult a ∧
  isOperationUsed ExponentOperation.multiplication ∧
  isOperationUsed ExponentOperation.exponentiation ∧
  isOperationUsed ExponentOperation.productExponentiation :=
by sorry

end NUMINAMATH_CALUDE_exponent_calculation_correct_and_uses_operations_l3425_342523


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_12_l3425_342591

theorem cos_squared_minus_sin_squared_pi_12 : 
  Real.cos (π / 12) ^ 2 - Real.sin (π / 12) ^ 2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_12_l3425_342591


namespace NUMINAMATH_CALUDE_function_identity_l3425_342536

theorem function_identity (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_identity_l3425_342536


namespace NUMINAMATH_CALUDE_caleb_picked_less_than_kayla_l3425_342589

/-- The number of apples picked by Kayla -/
def kayla_apples : ℕ := 20

/-- The number of apples picked by Suraya -/
def suraya_apples : ℕ := kayla_apples + 7

/-- The number of apples picked by Caleb -/
def caleb_apples : ℕ := suraya_apples - 12

theorem caleb_picked_less_than_kayla : kayla_apples - caleb_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_caleb_picked_less_than_kayla_l3425_342589


namespace NUMINAMATH_CALUDE_different_course_choices_l3425_342509

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses for one person -/
def ways_per_person : ℕ := Nat.choose total_courses courses_per_person

/-- The total number of ways both persons can choose courses -/
def total_ways : ℕ := ways_per_person * ways_per_person

/-- The number of ways to choose the same courses for both persons -/
def same_choices : ℕ := Nat.choose total_courses courses_per_person

/-- The number of ways to choose courses with at least one difference -/
def different_choices : ℕ := total_ways - same_choices

theorem different_course_choices : different_choices = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_course_choices_l3425_342509


namespace NUMINAMATH_CALUDE_subset_range_l3425_342542

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m - 3}

-- Theorem statement
theorem subset_range (m : ℝ) : B m ⊆ A ↔ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_subset_range_l3425_342542


namespace NUMINAMATH_CALUDE_video_rental_percentage_l3425_342588

theorem video_rental_percentage (a : ℕ) : 
  let action := a
  let drama := 5 * a
  let comedy := 10 * a
  let total := action + drama + comedy
  (comedy : ℚ) / total * 100 = 62.5 := by
sorry

end NUMINAMATH_CALUDE_video_rental_percentage_l3425_342588


namespace NUMINAMATH_CALUDE_expand_product_l3425_342562

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3425_342562


namespace NUMINAMATH_CALUDE_tan_sum_identity_l3425_342558

theorem tan_sum_identity (x : ℝ) : 
  Real.tan (18 * π / 180 - x) * Real.tan (12 * π / 180 + x) + 
  Real.sqrt 3 * (Real.tan (18 * π / 180 - x) + Real.tan (12 * π / 180 + x)) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l3425_342558


namespace NUMINAMATH_CALUDE_combined_yellow_ratio_approx_31_percent_l3425_342524

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the total number of yellow jelly beans in a bag -/
def yellow_count (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellow_ratio

/-- Theorem: The ratio of yellow jelly beans to all beans when three bags are combined -/
theorem combined_yellow_ratio_approx_31_percent 
  (bag1 bag2 bag3 : JellyBeanBag)
  (h1 : bag1 = ⟨26, 1/2⟩)
  (h2 : bag2 = ⟨28, 1/4⟩)
  (h3 : bag3 = ⟨30, 1/5⟩) :
  let total_yellow := yellow_count bag1 + yellow_count bag2 + yellow_count bag3
  let total_beans := bag1.total + bag2.total + bag3.total
  abs ((total_yellow / total_beans) - 31/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_combined_yellow_ratio_approx_31_percent_l3425_342524


namespace NUMINAMATH_CALUDE_apps_deleted_l3425_342545

theorem apps_deleted (initial_apps final_apps : ℝ) 
  (h1 : initial_apps = 300.5)
  (h2 : final_apps = 129.5) :
  initial_apps - final_apps = 171 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l3425_342545


namespace NUMINAMATH_CALUDE_A_inverse_correct_l3425_342593

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![3, -1, 3],
    ![2, -1, 4],
    ![1,  2, -3]]

def A_inv : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![ 1/2, -3/10,  1/10],
    ![-1,    6/5,   3/5],
    ![-1/2,  7/10,  1/10]]

theorem A_inverse_correct : A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_A_inverse_correct_l3425_342593


namespace NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l3425_342543

theorem cube_squared_equals_sixth_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l3425_342543


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3425_342547

/-- Given a line segment with one endpoint at (1, -3) and midpoint at (3, 5),
    the sum of the coordinates of the other endpoint is 18. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (3 = (x + 1) / 2) →  -- Midpoint x-coordinate condition
    (5 = (y - 3) / 2) →  -- Midpoint y-coordinate condition
    x + y = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3425_342547


namespace NUMINAMATH_CALUDE_birthday_crayons_l3425_342594

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 291

/-- The number of crayons Paul lost or gave away -/
def crayons_lost_or_given : ℕ := 315

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost_or_given

theorem birthday_crayons : total_crayons = 606 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l3425_342594


namespace NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l3425_342597

/-- Given a point P(m-1, m+1) that lies on the x-axis, 
    prove that its symmetric point with respect to the x-axis has coordinates (-2, 0) -/
theorem symmetric_point_on_x_axis (m : ℝ) :
  (m + 1 = 0) →  -- P lies on the x-axis
  ((-2 : ℝ), (0 : ℝ)) = (m - 1, -(m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l3425_342597


namespace NUMINAMATH_CALUDE_expression_evaluation_l3425_342564

theorem expression_evaluation (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6*m + 9) / (m - 2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3425_342564


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3425_342563

/-- Represents a configuration of square tiles --/
structure TileConfiguration where
  numTiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a configuration --/
def addTiles (config : TileConfiguration) (newTiles : ℕ) : TileConfiguration :=
  { numTiles := config.numTiles + newTiles
  , perimeter := config.perimeter + 2 * newTiles }

theorem perimeter_after_adding_tiles 
  (initialConfig : TileConfiguration) 
  (tilesAdded : ℕ) :
  initialConfig.numTiles = 9 →
  initialConfig.perimeter = 16 →
  tilesAdded = 3 →
  (addTiles initialConfig tilesAdded).perimeter = 22 := by
  sorry

#check perimeter_after_adding_tiles

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3425_342563


namespace NUMINAMATH_CALUDE_cookie_radius_l3425_342535

theorem cookie_radius (x y : ℝ) : 
  (x^2 + y^2 - 6.5 = x + 3*y) → 
  ∃ (h k r : ℝ), r = 3 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_radius_l3425_342535


namespace NUMINAMATH_CALUDE_finger_2004_is_index_l3425_342548

def finger_sequence : ℕ → String
| 0 => "pinky"
| 1 => "ring"
| 2 => "middle"
| 3 => "index"
| 4 => "thumb"
| 5 => "index"
| 6 => "middle"
| 7 => "ring"
| n + 8 => finger_sequence n

theorem finger_2004_is_index : finger_sequence 2003 = "index" := by
  sorry

end NUMINAMATH_CALUDE_finger_2004_is_index_l3425_342548


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3425_342513

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 393000

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation :=
  { coefficient := 3.93
    exponent := 5
    coefficient_range := by sorry }

/-- Theorem stating that the scientific notation is correct -/
theorem scientific_notation_correct :
  (scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3425_342513


namespace NUMINAMATH_CALUDE_triangle_rotation_l3425_342559

theorem triangle_rotation (α β γ : ℝ) (k m : ℤ) (h1 : α + β + γ = 180)
  (h2 : 15 * α = 360 * k) (h3 : 6 * β = 360 * m) :
  ∃ (n : ℕ) (l : ℤ), n * γ = 360 * l ∧ n = 5 ∧ ∀ (n' : ℕ) (l' : ℤ), n' * γ = 360 * l' → n ≤ n' := by
  sorry

end NUMINAMATH_CALUDE_triangle_rotation_l3425_342559


namespace NUMINAMATH_CALUDE_systematic_sampling_first_two_numbers_l3425_342504

/-- Systematic sampling function that returns the nth sample number -/
def systematicSample (populationSize sampleSize n : ℕ) : ℕ :=
  (n - 1) * (populationSize / sampleSize)

/-- Theorem stating the first two sample numbers in the given systematic sampling scenario -/
theorem systematic_sampling_first_two_numbers
  (populationSize : ℕ)
  (sampleSize : ℕ)
  (lastSampleNumber : ℕ)
  (h1 : populationSize = 8000)
  (h2 : sampleSize = 50)
  (h3 : lastSampleNumber = 7900) :
  systematicSample populationSize sampleSize 1 = 159 ∧
  systematicSample populationSize sampleSize 2 = 319 := by
  sorry

#eval systematicSample 8000 50 1  -- Expected: 159
#eval systematicSample 8000 50 2  -- Expected: 319

end NUMINAMATH_CALUDE_systematic_sampling_first_two_numbers_l3425_342504


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3425_342517

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3425_342517


namespace NUMINAMATH_CALUDE_percentage_problem_l3425_342527

theorem percentage_problem (p : ℝ) : p * 50 / 100 = 200 → p = 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3425_342527


namespace NUMINAMATH_CALUDE_arrangement_count_l3425_342510

def number_of_arrangements (men women : ℕ) : ℕ :=
  let group1 := men.choose 1 * women.choose 2
  let remaining_men := men - 1
  let remaining_women := women - 2
  let group2 := remaining_men.choose 1 * remaining_women.choose 2
  let group3 := 1  -- Only one way to arrange the last group
  group1 * group2 * group3

theorem arrangement_count :
  number_of_arrangements 4 5 = 360 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l3425_342510


namespace NUMINAMATH_CALUDE_function_properties_l3425_342583

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x / 2))^2 + Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) - 1/2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : is_periodic (f ω) Real.pi) (h3 : ∀ T, 0 < T → T < Real.pi → ¬ is_periodic (f ω) T) :
  (ω = 2) ∧ 
  (∀ x, f ω x ≤ 1) ∧
  (∀ x, f ω x ≥ -1) ∧
  (∃ x, f ω x = 1) ∧
  (∃ x, f ω x = -1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6),
    x ≤ y → f ω x ≤ f ω y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3425_342583


namespace NUMINAMATH_CALUDE_contradiction_assumptions_l3425_342567

theorem contradiction_assumptions :
  (∀ p q : ℝ, (p^3 + q^3 = 2) → (¬(p + q ≤ 2) ↔ p + q > 2)) ∧
  (∀ a b : ℝ, |a| + |b| < 1 →
    ∃ x₁ : ℝ, x₁^2 + a*x₁ + b = 0 ∧ |x₁| ≥ 1 →
      ∃ x₂ : ℝ, x₂^2 + a*x₂ + b = 0 ∧ |x₂| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumptions_l3425_342567


namespace NUMINAMATH_CALUDE_multiple_z_values_l3425_342525

/-- Given two four-digit integers x and y where y is the reverse of x, 
    z = |x - y| can take multiple distinct values. -/
theorem multiple_z_values (x y z : ℕ) : 
  (1000 ≤ x ∧ x ≤ 9999) →
  (1000 ≤ y ∧ y ≤ 9999) →
  (y = (x % 10) * 1000 + ((x / 10) % 10) * 100 + ((x / 100) % 10) * 10 + (x / 1000)) →
  (z = Int.natAbs (x - y)) →
  ∃ (z₁ z₂ : ℕ), z₁ ≠ z₂ ∧ 
    ∃ (x₁ y₁ x₂ y₂ : ℕ), 
      (1000 ≤ x₁ ∧ x₁ ≤ 9999) ∧
      (1000 ≤ y₁ ∧ y₁ ≤ 9999) ∧
      (y₁ = (x₁ % 10) * 1000 + ((x₁ / 10) % 10) * 100 + ((x₁ / 100) % 10) * 10 + (x₁ / 1000)) ∧
      (z₁ = Int.natAbs (x₁ - y₁)) ∧
      (1000 ≤ x₂ ∧ x₂ ≤ 9999) ∧
      (1000 ≤ y₂ ∧ y₂ ≤ 9999) ∧
      (y₂ = (x₂ % 10) * 1000 + ((x₂ / 10) % 10) * 100 + ((x₂ / 100) % 10) * 10 + (x₂ / 1000)) ∧
      (z₂ = Int.natAbs (x₂ - y₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_multiple_z_values_l3425_342525


namespace NUMINAMATH_CALUDE_complement_of_union_l3425_342514

def U : Set Int := {x | x^2 - 5*x - 6 ≤ 0}

def A : Set Int := {x | x*(2-x) ≥ 0}

def B : Set Int := {1, 2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {-1, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3425_342514


namespace NUMINAMATH_CALUDE_log_equation_solution_l3425_342538

theorem log_equation_solution :
  ∀ x : ℝ, (Real.log 729 / Real.log (3 * x) = x) →
    (x = 3 ∧ ¬ ∃ n : ℕ, x = n^2 ∧ ¬ ∃ m : ℕ, x = m^3 ∧ ∃ k : ℤ, x = k) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3425_342538


namespace NUMINAMATH_CALUDE_circle_line_slope_range_l3425_342518

/-- Given a circle and a line, if there are at least three distinct points on the circle
    with a specific distance from the line, then the slope of the line is within a certain range. -/
theorem circle_line_slope_range (a b : ℝ) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 4*x - 4*y - 10 = 0
  let line := fun (x y : ℝ) => a*x + b*y = 0
  let k := -a/b  -- slope of the line
  let distance_point_to_line := fun (x y : ℝ) => |a*x + b*y| / Real.sqrt (a^2 + b^2)
  (∃ (p q r : ℝ × ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    circle p.1 p.2 ∧ circle q.1 q.2 ∧ circle r.1 r.2 ∧
    distance_point_to_line p.1 p.2 = 2 * Real.sqrt 2 ∧
    distance_point_to_line q.1 q.2 = 2 * Real.sqrt 2 ∧
    distance_point_to_line r.1 r.2 = 2 * Real.sqrt 2) →
  2 - Real.sqrt 3 ≤ k ∧ k ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_slope_range_l3425_342518


namespace NUMINAMATH_CALUDE_solution_count_l3425_342571

theorem solution_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  4 * p.1 + 7 * p.2 = 600 ∧ 
  p.1 % 2 = 0 ∧ 
  p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l3425_342571


namespace NUMINAMATH_CALUDE_division_calculation_l3425_342580

theorem division_calculation : 250 / (5 + 15 * 3^2) = 25 / 14 := by
  sorry

end NUMINAMATH_CALUDE_division_calculation_l3425_342580


namespace NUMINAMATH_CALUDE_m_range_l3425_342570

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < -2 ∨ x > 10

def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0

-- Define the condition that ¬q is sufficient but not necessary for ¬p
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(q x m) → ¬(p x)) ∧ ∃ x, ¬(p x) ∧ q x m

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) ∧ (sufficient_not_necessary m) ↔ 0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3425_342570


namespace NUMINAMATH_CALUDE_unique_monic_polynomial_l3425_342599

/-- A monic polynomial of degree 2 satisfying f(0) = 10 and f(1) = 14 -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 10

/-- The theorem stating that f is the unique monic polynomial of degree 2 satisfying the given conditions -/
theorem unique_monic_polynomial :
  ∀ g : ℝ → ℝ, (∃ a b : ℝ, ∀ x, g x = x^2 + a*x + b) →
  g 0 = 10 → g 1 = 14 → g = f :=
by sorry

end NUMINAMATH_CALUDE_unique_monic_polynomial_l3425_342599


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l3425_342573

/-- The quadratic function f(x) = -(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

/-- Point P1 on the graph of f -/
def P1 : ℝ × ℝ := (-1, f (-1))

/-- Point P2 on the graph of f -/
def P2 : ℝ × ℝ := (3, f 3)

/-- Point P3 on the graph of f -/
def P3 : ℝ × ℝ := (5, f 5)

theorem quadratic_points_relationship : P1.2 = P2.2 ∧ P1.2 > P3.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l3425_342573
