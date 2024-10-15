import Mathlib

namespace NUMINAMATH_CALUDE_sin_power_five_expansion_l1655_165516

theorem sin_power_five_expansion (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 512 := by
  sorry

end NUMINAMATH_CALUDE_sin_power_five_expansion_l1655_165516


namespace NUMINAMATH_CALUDE_initial_speed_is_60_l1655_165569

/-- Represents the initial speed of a traveler given specific journey conditions -/
def initial_speed (D T : ℝ) : ℝ :=
  let remaining_time := T - T / 3
  let remaining_distance := D / 3
  60

/-- Theorem stating the initial speed under given conditions -/
theorem initial_speed_is_60 (D T : ℝ) (h1 : D > 0) (h2 : T > 0) :
  initial_speed D T = 60 := by
  sorry

#check initial_speed_is_60

end NUMINAMATH_CALUDE_initial_speed_is_60_l1655_165569


namespace NUMINAMATH_CALUDE_point_D_value_l1655_165534

/-- The number corresponding to point D on a number line, given that:
    - A corresponds to 5
    - B corresponds to 8
    - C corresponds to -10
    - The sum of the four numbers remains unchanged when the direction of the number line is reversed
-/
def point_D : ℝ := -3

/-- The sum of the numbers corresponding to points A, B, C, and D -/
def sum_forward (d : ℝ) : ℝ := 5 + 8 + (-10) + d

/-- The sum of the numbers corresponding to points A, B, C, and D when the direction is reversed -/
def sum_reversed (d : ℝ) : ℝ := (-5) + (-8) + 10 + (-d)

/-- Theorem stating that point D corresponds to -3 -/
theorem point_D_value : 
  sum_forward point_D = sum_reversed point_D :=
by sorry

end NUMINAMATH_CALUDE_point_D_value_l1655_165534


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_range_for_negative_f_l1655_165579

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |2 * x - a|

-- Theorem for part (1)
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x > 0} = {x : ℝ | 1 < x ∧ x < 5/3} := by sorry

-- Theorem for part (2)
theorem a_range_for_negative_f :
  (∀ x < 2, f a x < 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_range_for_negative_f_l1655_165579


namespace NUMINAMATH_CALUDE_three_integers_product_2008th_power_l1655_165582

theorem three_integers_product_2008th_power :
  ∃ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧  -- distinct
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- positive
    y = (x + z) / 2 ∧        -- one is average of other two
    ∃ (k : ℕ), x * y * z = k^2008 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_product_2008th_power_l1655_165582


namespace NUMINAMATH_CALUDE_relationship_holds_l1655_165593

/-- The relationship between x and y is defined by the function f --/
def f (x : ℕ) : ℕ := x^2 + 3*x + 1

/-- The set of x values --/
def X : Finset ℕ := {1, 2, 3, 4}

/-- The corresponding y values for each x in X --/
def Y : Finset ℕ := {5, 13, 25, 41}

/-- A function that checks if a given pair (x, y) satisfies the relationship --/
def satisfies_relationship (pair : ℕ × ℕ) : Prop :=
  f pair.1 = pair.2

theorem relationship_holds : ∀ (x : ℕ), x ∈ X → (x, f x) ∈ X.product Y ∧ satisfies_relationship (x, f x) := by
  sorry

end NUMINAMATH_CALUDE_relationship_holds_l1655_165593


namespace NUMINAMATH_CALUDE_kyle_total_laps_l1655_165591

-- Define the number of laps jogged in P.E. class
def pe_laps : ℝ := 1.12

-- Define the number of laps jogged during track practice
def track_laps : ℝ := 2.12

-- Define the total number of laps
def total_laps : ℝ := pe_laps + track_laps

-- Theorem statement
theorem kyle_total_laps : total_laps = 3.24 := by
  sorry

end NUMINAMATH_CALUDE_kyle_total_laps_l1655_165591


namespace NUMINAMATH_CALUDE_circle_area_is_one_l1655_165566

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (8 / (2 * Real.pi * r) + 2 * r = 6 * r) → π * r^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_is_one_l1655_165566


namespace NUMINAMATH_CALUDE_distributive_property_l1655_165514

theorem distributive_property (a b c : ℝ) : -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l1655_165514


namespace NUMINAMATH_CALUDE_cubic_fraction_value_l1655_165556

theorem cubic_fraction_value : 
  let a : ℝ := 8
  let b : ℝ := 8 - 1
  (a^3 + b^3) / (a^2 - a*b + b^2) = 15 := by sorry

end NUMINAMATH_CALUDE_cubic_fraction_value_l1655_165556


namespace NUMINAMATH_CALUDE_unique_root_implies_a_equals_three_a_equals_three_implies_unique_root_unique_root_iff_a_equals_three_l1655_165539

/-- The function f(x) defined by the equation --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

/-- Theorem stating that a = 3 is the only value for which f has a unique root --/
theorem unique_root_implies_a_equals_three :
  ∀ a : ℝ, (∃! x : ℝ, f a x = 0) → a = 3 :=
by sorry

/-- Theorem stating that when a = 3, f has a unique root --/
theorem a_equals_three_implies_unique_root :
  ∃! x : ℝ, f 3 x = 0 :=
by sorry

/-- The main theorem combining the above results --/
theorem unique_root_iff_a_equals_three :
  ∀ a : ℝ, (∃! x : ℝ, f a x = 0) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_a_equals_three_a_equals_three_implies_unique_root_unique_root_iff_a_equals_three_l1655_165539


namespace NUMINAMATH_CALUDE_selection_probabilities_l1655_165596

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The probability of selecting exactly 2 boys and 1 girl -/
def prob_2boys_1girl : ℚ := 3/5

/-- The probability of selecting at least 1 girl -/
def prob_at_least_1girl : ℚ := 4/5

theorem selection_probabilities :
  (Nat.choose num_boys 2 * Nat.choose num_girls 1) / Nat.choose total_people num_selected = prob_2boys_1girl ∧
  1 - (Nat.choose num_boys num_selected) / Nat.choose total_people num_selected = prob_at_least_1girl :=
by sorry

end NUMINAMATH_CALUDE_selection_probabilities_l1655_165596


namespace NUMINAMATH_CALUDE_spaghetti_pizza_ratio_l1655_165557

/-- The total number of students surveyed -/
def total_students : ℕ := 800

/-- The number of students preferring lasagna -/
def lasagna_preference : ℕ := 150

/-- The number of students preferring manicotti -/
def manicotti_preference : ℕ := 120

/-- The number of students preferring ravioli -/
def ravioli_preference : ℕ := 180

/-- The number of students preferring spaghetti -/
def spaghetti_preference : ℕ := 200

/-- The number of students preferring pizza -/
def pizza_preference : ℕ := 150

/-- Theorem stating that the ratio of students preferring spaghetti to students preferring pizza is 4/3 -/
theorem spaghetti_pizza_ratio : 
  (spaghetti_preference : ℚ) / (pizza_preference : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_spaghetti_pizza_ratio_l1655_165557


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1655_165586

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 2 →
  a 6 = 8 →
  a 3 * a 4 * a 5 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1655_165586


namespace NUMINAMATH_CALUDE_average_age_of_ten_students_l1655_165538

theorem average_age_of_ten_students
  (total_students : Nat)
  (total_average_age : ℝ)
  (nine_students_average : ℝ)
  (twentieth_student_age : ℝ)
  (h1 : total_students = 20)
  (h2 : total_average_age = 20)
  (h3 : nine_students_average = 11)
  (h4 : twentieth_student_age = 61) :
  let remaining_students := total_students - 10
  let total_age := total_students * total_average_age
  let nine_students_total_age := 9 * nine_students_average
  let ten_students_total_age := total_age - nine_students_total_age - twentieth_student_age
  ten_students_total_age / remaining_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_ten_students_l1655_165538


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1655_165501

theorem sufficient_condition_for_inequality (a : ℝ) (h : a ≥ 5) :
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1655_165501


namespace NUMINAMATH_CALUDE_prob_ace_king_queen_value_l1655_165547

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing Ace, King, Queen in order without replacement -/
def prob_ace_king_queen : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  NumKings / (StandardDeck - 1) *
  NumQueens / (StandardDeck - 2)

theorem prob_ace_king_queen_value :
  prob_ace_king_queen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_queen_value_l1655_165547


namespace NUMINAMATH_CALUDE_gcd_90_405_l1655_165542

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_405_l1655_165542


namespace NUMINAMATH_CALUDE_expression_equals_sum_l1655_165527

theorem expression_equals_sum (a b c : ℝ) (ha : a = 13) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l1655_165527


namespace NUMINAMATH_CALUDE_valid_division_exists_l1655_165533

/-- Represents a grid cell that can contain a symbol -/
inductive Cell
  | Empty
  | Star
  | Cross

/-- Represents a 7x7 grid -/
def Grid := Fin 7 → Fin 7 → Cell

/-- Represents a matchstick placement -/
structure Matchstick where
  row : Fin 8
  col : Fin 8
  horizontal : Bool

/-- Counts the number of matchsticks in a list -/
def count_matchsticks (placements : List Matchstick) : Nat :=
  placements.length

/-- Checks if two parts of the grid are of equal size and shape -/
def equal_parts (g : Grid) (placements : List Matchstick) : Prop :=
  sorry

/-- Checks if the symbols (stars and crosses) are placed correctly -/
def correct_symbol_placement (g : Grid) : Prop :=
  sorry

/-- The main theorem stating that a valid division exists -/
theorem valid_division_exists : ∃ (g : Grid) (placements : List Matchstick),
  count_matchsticks placements = 26 ∧
  equal_parts g placements ∧
  correct_symbol_placement g :=
  sorry

end NUMINAMATH_CALUDE_valid_division_exists_l1655_165533


namespace NUMINAMATH_CALUDE_max_product_sum_246_l1655_165563

theorem max_product_sum_246 : 
  ∀ x y : ℤ, x + y = 246 → x * y ≤ 15129 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_246_l1655_165563


namespace NUMINAMATH_CALUDE_necessary_to_sufficient_negation_l1655_165541

theorem necessary_to_sufficient_negation (A B : Prop) :
  (B → A) → (¬A → ¬B) := by sorry

end NUMINAMATH_CALUDE_necessary_to_sufficient_negation_l1655_165541


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1655_165590

def is_divisible_by_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

def unit_digit (n : ℕ) : ℕ := n % 10

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

def uniquely_determined_by_divisors (n : ℕ) : Prop :=
  ∀ m : ℕ, m < 60 → unit_digit m = unit_digit n → num_divisors m = num_divisors n → m = n

theorem unique_n_satisfying_conditions :
  ∃! n : ℕ, n < 60 ∧
    is_divisible_by_two_primes n ∧
    uniquely_determined_by_divisors n ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1655_165590


namespace NUMINAMATH_CALUDE_unique_solution_divisibility_l1655_165568

theorem unique_solution_divisibility : ∀ a b : ℕ+,
  (∃ k l : ℕ+, (a^2 + b^2 : ℕ) * k = a^3 + 1 ∧ (a^2 + b^2 : ℕ) * l = b^3 + 1) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_divisibility_l1655_165568


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l1655_165560

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_remainder : (List.sum (List.map factorial [1, 2, 3, 4, 5, 6])) % 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l1655_165560


namespace NUMINAMATH_CALUDE_nine_payment_methods_l1655_165504

/-- Represents the number of ways to pay an amount using given denominations -/
def paymentMethods (amount : ℕ) (denominations : List ℕ) : ℕ := sorry

/-- The cost of the book in yuan -/
def bookCost : ℕ := 20

/-- Available note denominations in yuan -/
def availableNotes : List ℕ := [10, 5, 1]

/-- Theorem stating that there are 9 ways to pay for the book -/
theorem nine_payment_methods : paymentMethods bookCost availableNotes = 9 := by sorry

end NUMINAMATH_CALUDE_nine_payment_methods_l1655_165504


namespace NUMINAMATH_CALUDE_rotateSemicircleDiameter_is_eight_l1655_165550

/-- The diameter of a solid figure obtained by rotating a semicircle around its diameter -/
def rotateSemicircleDiameter (radius : ℝ) : ℝ :=
  2 * radius

/-- Theorem: The diameter of a solid figure obtained by rotating a semicircle 
    with a radius of 4 centimeters once around its diameter is 8 centimeters -/
theorem rotateSemicircleDiameter_is_eight :
  rotateSemicircleDiameter 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rotateSemicircleDiameter_is_eight_l1655_165550


namespace NUMINAMATH_CALUDE_vacuum_time_calculation_l1655_165565

theorem vacuum_time_calculation (total_free_time dusting_time mopping_time cat_brushing_time num_cats remaining_free_time : ℕ) 
  (h1 : total_free_time = 180)
  (h2 : dusting_time = 60)
  (h3 : mopping_time = 30)
  (h4 : cat_brushing_time = 5)
  (h5 : num_cats = 3)
  (h6 : remaining_free_time = 30) :
  total_free_time - remaining_free_time - (dusting_time + mopping_time + cat_brushing_time * num_cats) = 45 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_time_calculation_l1655_165565


namespace NUMINAMATH_CALUDE_min_value_zero_at_one_sixth_l1655_165585

/-- The quadratic expression as a function of x, y, and c -/
def f (x y c : ℝ) : ℝ :=
  2 * x^2 - 4 * c * x * y + (2 * c^2 + 1) * y^2 - 2 * x - 6 * y + 9

/-- Theorem stating that 1/6 is the value of c that makes the minimum of f zero -/
theorem min_value_zero_at_one_sixth :
  ∃ (x y : ℝ), f x y (1/6) = 0 ∧ ∀ (x' y' : ℝ), f x' y' (1/6) ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_at_one_sixth_l1655_165585


namespace NUMINAMATH_CALUDE_f_at_two_equals_one_fourth_l1655_165570

/-- Given a function f(x) = 2^x + 2^(-x) - 4, prove that f(2) = 1/4 -/
theorem f_at_two_equals_one_fourth :
  let f : ℝ → ℝ := λ x ↦ 2^x + 2^(-x) - 4
  f 2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_equals_one_fourth_l1655_165570


namespace NUMINAMATH_CALUDE_poverty_definition_l1655_165500

-- Define poverty as a string
def poverty : String := "poverty"

-- State the theorem
theorem poverty_definition : poverty = "poverty" := by
  sorry

end NUMINAMATH_CALUDE_poverty_definition_l1655_165500


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l1655_165561

-- Define the original equation
def original_equation (x : ℝ) : Prop := 3 * x^2 - 2 = 4 * x

-- Define the general form of a quadratic equation
def general_form (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem statement
theorem quadratic_coefficients :
  ∃ (c : ℝ), ∀ (x : ℝ), 
    (original_equation x ↔ general_form 3 (-4) c x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l1655_165561


namespace NUMINAMATH_CALUDE_cage_cost_calculation_l1655_165509

/-- The cost of Keith's purchases -/
def total_cost : ℝ := 24.81

/-- The cost of the rabbit toy -/
def rabbit_toy_cost : ℝ := 6.51

/-- The cost of the pet food -/
def pet_food_cost : ℝ := 5.79

/-- The amount of money Keith found -/
def found_money : ℝ := 1.00

/-- The cost of the cage -/
def cage_cost : ℝ := total_cost - (rabbit_toy_cost + pet_food_cost) + found_money

theorem cage_cost_calculation : cage_cost = 13.51 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_calculation_l1655_165509


namespace NUMINAMATH_CALUDE_twelve_digit_159_div37_not_sum76_l1655_165540

-- Define a function to check if a number consists only of digits 1, 5, and 9
def only_159_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 5 ∨ d = 9

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Theorem statement
theorem twelve_digit_159_div37_not_sum76 (n : ℕ) :
  n ≥ 10^11 ∧ n < 10^12 ∧ only_159_digits n ∧ n % 37 = 0 →
  sum_of_digits n ≠ 76 := by
  sorry

end NUMINAMATH_CALUDE_twelve_digit_159_div37_not_sum76_l1655_165540


namespace NUMINAMATH_CALUDE_trig_identity_l1655_165574

theorem trig_identity (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.sin (α - π / 6) ^ 2 - Real.cos (5 * π / 6 + α) = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1655_165574


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1655_165554

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ)
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l1655_165554


namespace NUMINAMATH_CALUDE_angle_AEC_l1655_165562

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_360 : A + B + C + D = 360)

-- Define the exterior angle bisector point
def exterior_angle_bisector_point (q : Quadrilateral) : ℝ :=
  sorry

-- Theorem statement
theorem angle_AEC (q : Quadrilateral) :
  let E := exterior_angle_bisector_point q
  (360 - (q.B + q.D)) / 2 = E := by
  sorry

end NUMINAMATH_CALUDE_angle_AEC_l1655_165562


namespace NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l1655_165543

def admission_fee (adults children : ℕ) : ℕ := 30 * adults + 15 * children

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 2 ∧ children ≥ 2 ∧ admission_fee adults children = 2250

def ratio_difference (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_one_to_one :
  ∃ (a c : ℕ), is_valid_combination a c ∧
    ∀ (x y : ℕ), is_valid_combination x y →
      ratio_difference a c ≤ ratio_difference x y :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l1655_165543


namespace NUMINAMATH_CALUDE_length_PI_is_five_l1655_165549

/-- A right triangle with given side lengths and its incenter -/
structure RightTriangleWithIncenter where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Length of side QR (hypotenuse) -/
  qr : ℝ
  /-- The triangle is right-angled -/
  is_right : pq ^ 2 + pr ^ 2 = qr ^ 2
  /-- The given side lengths form a valid triangle -/
  triangle_inequality : pq + pr > qr ∧ pr + qr > pq ∧ qr + pq > pr
  /-- The incenter of the triangle -/
  incenter : ℝ × ℝ

/-- The length of segment PI in a right triangle with incenter -/
def length_PI (t : RightTriangleWithIncenter) : ℝ :=
  sorry

/-- Theorem: The length of segment PI is 5 for the given triangle -/
theorem length_PI_is_five (t : RightTriangleWithIncenter) 
  (h1 : t.pq = 15) (h2 : t.pr = 20) (h3 : t.qr = 25) : length_PI t = 5 := by
  sorry

end NUMINAMATH_CALUDE_length_PI_is_five_l1655_165549


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1655_165503

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) : 
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1655_165503


namespace NUMINAMATH_CALUDE_alcohol_dilution_l1655_165548

/-- Proves that adding 3 liters of water to 11 liters of a 42% alcohol solution 
    results in a new mixture with 33% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 11 ∧ 
  initial_concentration = 0.42 ∧ 
  added_water = 3 ∧ 
  final_concentration = 0.33 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l1655_165548


namespace NUMINAMATH_CALUDE_sum_of_products_zero_l1655_165517

theorem sum_of_products_zero 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 117) :
  x*y + y*z + x*z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_zero_l1655_165517


namespace NUMINAMATH_CALUDE_ink_blot_is_circle_l1655_165553

/-- A closed, bounded set in a plane -/
def InkBlot : Type := Set (ℝ × ℝ)

/-- The minimum distance from a point to the boundary of the ink blot -/
def min_distance (S : InkBlot) (p : ℝ × ℝ) : ℝ := sorry

/-- The maximum distance from a point to the boundary of the ink blot -/
def max_distance (S : InkBlot) (p : ℝ × ℝ) : ℝ := sorry

/-- The largest of all minimum distances -/
def largest_min_distance (S : InkBlot) : ℝ := sorry

/-- The smallest of all maximum distances -/
def smallest_max_distance (S : InkBlot) : ℝ := sorry

/-- A circle in the plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : InkBlot := sorry

theorem ink_blot_is_circle (S : InkBlot) :
  largest_min_distance S = smallest_max_distance S →
  ∃ (center : ℝ × ℝ) (radius : ℝ), S = Circle center radius :=
sorry

end NUMINAMATH_CALUDE_ink_blot_is_circle_l1655_165553


namespace NUMINAMATH_CALUDE_handshake_count_l1655_165544

/-- Represents the number of students in the class -/
def num_students : ℕ := 40

/-- Represents the length of the counting sequence -/
def sequence_length : ℕ := 4

/-- Calculates the number of initial pairs facing each other -/
def initial_pairs : ℕ := num_students / sequence_length

/-- Calculates the sum of handshakes in subsequent rounds -/
def subsequent_handshakes : ℕ := (initial_pairs * (initial_pairs + 1)) / 2

/-- Calculates the total number of handshakes -/
def total_handshakes : ℕ := initial_pairs + 3 * subsequent_handshakes

/-- Theorem stating that the total number of handshakes is 175 -/
theorem handshake_count : total_handshakes = 175 := by sorry

end NUMINAMATH_CALUDE_handshake_count_l1655_165544


namespace NUMINAMATH_CALUDE_trapezoid_area_l1655_165588

/-- A trapezoid with the given properties has an area of 260.4 square centimeters. -/
theorem trapezoid_area (h : ℝ) (b₁ b₂ : ℝ) :
  h = 12 →
  b₁ = 15 →
  b₂ = 13 →
  (b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt = 14 →
  (1/2) * (((b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt + b₁) + 
           ((b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt + b₂)) * h = 260.4 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1655_165588


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1655_165536

/-- Given a reflection of point (-2, 3) across line y = mx + b to point (4, -5), prove m + b = -1 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 4 ∧ y = -5 ∧ 
    (x - (-2))^2 + (y - 3)^2 = (x - (-2))^2 + (m * (x - (-2)) + b - 3)^2 ∧
    y = m * x + b) →
  m + b = -1 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1655_165536


namespace NUMINAMATH_CALUDE_parabola_reflection_theorem_l1655_165592

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
def Line (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)}

/-- Check if a point is on the parabola -/
def onParabola (par : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * par.p * P.x

/-- Check if a point is on the axis of symmetry -/
def onAxisOfSymmetry (P : Point) : Prop :=
  P.y = 0

/-- Reflection of a point about y-axis -/
def reflectAboutYAxis (P : Point) : Point :=
  ⟨-P.x, P.y⟩

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem parabola_reflection_theorem (par : Parabola) (A : Point)
  (h_A_on_axis : onAxisOfSymmetry A)
  (h_A_inside : onParabola par A → False) :
  let B := reflectAboutYAxis A
  (∀ P Q : Point, onParabola par P → onParabola par Q →
    P.x * Q.x > 0 → P ∈ Line A Q → Q ∈ Line A P →
    angle P B A = angle Q B A) ∧
  (∀ P Q : Point, onParabola par P → onParabola par Q →
    P.x * Q.x > 0 → P ∈ Line B Q → Q ∈ Line B P →
    angle P A B + angle Q A B = 180) :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_theorem_l1655_165592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1655_165578

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n-1)) / 2

/-- Theorem: If S_6/S_3 = 4 for an arithmetic sequence, then S_5/S_6 = 25/36 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 4) : 
  seq.S 5 / seq.S 6 = 25/36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1655_165578


namespace NUMINAMATH_CALUDE_savings_percentage_second_year_l1655_165532

/-- Proves that under the given conditions, the savings percentage in the second year is 15% -/
theorem savings_percentage_second_year 
  (salary_first_year : ℝ) 
  (savings_rate_first_year : ℝ) 
  (salary_increase_rate : ℝ) 
  (savings_increase_rate : ℝ) : 
  savings_rate_first_year = 0.1 →
  salary_increase_rate = 0.1 →
  savings_increase_rate = 1.65 →
  (savings_increase_rate * savings_rate_first_year * salary_first_year) / 
  ((1 + salary_increase_rate) * salary_first_year) = 0.15 := by
sorry


end NUMINAMATH_CALUDE_savings_percentage_second_year_l1655_165532


namespace NUMINAMATH_CALUDE_actual_tax_expectation_l1655_165587

/-- Represents the fraction of the population that are liars -/
def fraction_liars : ℝ := 0.1

/-- Represents the fraction of the population that are economists -/
def fraction_economists : ℝ := 1 - fraction_liars

/-- Represents the fraction of affirmative answers for raising taxes -/
def affirmative_taxes : ℝ := 0.4

/-- Represents the fraction of affirmative answers for increasing money supply -/
def affirmative_money : ℝ := 0.3

/-- Represents the fraction of affirmative answers for issuing bonds -/
def affirmative_bonds : ℝ := 0.5

/-- Represents the fraction of affirmative answers for spending gold reserves -/
def affirmative_gold : ℝ := 0

/-- The theorem stating that 30% of the population actually expects raising taxes -/
theorem actual_tax_expectation : 
  affirmative_taxes - fraction_liars = 0.3 := by sorry

end NUMINAMATH_CALUDE_actual_tax_expectation_l1655_165587


namespace NUMINAMATH_CALUDE_hiking_distance_proof_l1655_165537

theorem hiking_distance_proof (total_distance car_to_stream stream_to_meadow : ℝ) 
  (h1 : total_distance = 0.7)
  (h2 : car_to_stream = 0.2)
  (h3 : stream_to_meadow = 0.4) :
  total_distance - (car_to_stream + stream_to_meadow) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_hiking_distance_proof_l1655_165537


namespace NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l1655_165507

/-- Represents a cube made of unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_size := cube.size^2
  let edge_size := cube.size - 1
  3 * face_size - 3 * edge_size + 1

/-- Theorem: For a 9×9×9 cube, the maximum number of visible unit cubes is 220 -/
theorem max_visible_cubes_9x9x9 :
  max_visible_cubes ⟨9⟩ = 220 := by sorry

end NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l1655_165507


namespace NUMINAMATH_CALUDE_janes_pudding_purchase_l1655_165518

theorem janes_pudding_purchase (ice_cream_count : ℕ) (ice_cream_cost pudding_cost : ℚ) 
  (ice_cream_pudding_diff : ℚ) :
  ice_cream_count = 15 →
  ice_cream_cost = 5 →
  pudding_cost = 2 →
  ice_cream_count * ice_cream_cost = ice_cream_pudding_diff + pudding_count * pudding_cost →
  pudding_count = 5 :=
by
  sorry

#check janes_pudding_purchase

end NUMINAMATH_CALUDE_janes_pudding_purchase_l1655_165518


namespace NUMINAMATH_CALUDE_trig_values_special_angles_l1655_165581

theorem trig_values_special_angles :
  (Real.sin (π/6) = 1/2) ∧
  (Real.cos (π/6) = Real.sqrt 3 / 2) ∧
  (Real.tan (π/6) = Real.sqrt 3 / 3) ∧
  (Real.sin (π/4) = Real.sqrt 2 / 2) ∧
  (Real.cos (π/4) = Real.sqrt 2 / 2) ∧
  (Real.tan (π/4) = 1) ∧
  (Real.sin (π/3) = Real.sqrt 3 / 2) ∧
  (Real.cos (π/3) = 1/2) ∧
  (Real.tan (π/3) = Real.sqrt 3) ∧
  (Real.sin (π/2) = 1) ∧
  (Real.cos (π/2) = 0) := by
  sorry

-- Note: tan(π/2) is undefined, so it's not included in the theorem statement

end NUMINAMATH_CALUDE_trig_values_special_angles_l1655_165581


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l1655_165529

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ
  area_eq : area = (d1 * d2) / 2

/-- Theorem: In a rhombus with area 110 cm² and one diagonal 11 cm, the other diagonal is 20 cm -/
theorem rhombus_diagonal_length (r : Rhombus) 
    (h1 : r.d1 = 11) 
    (h2 : r.area = 110) : 
    r.d2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l1655_165529


namespace NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_has_positive_slope_l1655_165510

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  passes_through_first_quadrant : Bool
  passes_through_third_quadrant : Bool

/-- Definition of a line passing through first and third quadrants -/
def passes_through_first_and_third (l : Line) : Prop :=
  l.passes_through_first_quadrant ∧ l.passes_through_third_quadrant

/-- Theorem: If a line y = kx (k ≠ 0) passes through the first and third quadrants, then k > 0 -/
theorem line_through_first_and_third_quadrants_has_positive_slope (l : Line) 
    (h1 : l.slope ≠ 0) 
    (h2 : passes_through_first_and_third l) : 
    l.slope > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_has_positive_slope_l1655_165510


namespace NUMINAMATH_CALUDE_train_passengers_l1655_165522

theorem train_passengers (initial_passengers : ℕ) (num_stops : ℕ) : 
  initial_passengers = 64 → num_stops = 4 → 
  (initial_passengers : ℚ) * (2/3)^num_stops = 1024/81 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l1655_165522


namespace NUMINAMATH_CALUDE_paula_paint_theorem_l1655_165571

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCapacity where
  total_rooms : ℕ
  cans : ℕ

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cans_needed (initial : PaintCapacity) (lost_cans : ℕ) (rooms_to_paint : ℕ) : ℕ :=
  let rooms_per_can := initial.total_rooms / initial.cans
  rooms_to_paint / rooms_per_can

theorem paula_paint_theorem (initial : PaintCapacity) (lost_cans : ℕ) :
  initial.total_rooms = 40 →
  initial.cans = initial.cans - lost_cans + lost_cans →
  lost_cans = 6 →
  cans_needed initial lost_cans 30 = 18 := by
  sorry

#check paula_paint_theorem

end NUMINAMATH_CALUDE_paula_paint_theorem_l1655_165571


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1655_165546

/-- The line equation kx - y + 1 - 3k = 0 passes through the point (3, 1) for all k. -/
theorem line_passes_through_point :
  ∀ (k : ℝ), k * 3 - 1 + 1 - 3 * k = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1655_165546


namespace NUMINAMATH_CALUDE_train_distance_difference_l1655_165559

/-- Proves that the difference in distance traveled by two trains is 60 km -/
theorem train_distance_difference :
  ∀ (speed1 speed2 total_distance : ℝ),
  speed1 = 20 →
  speed2 = 25 →
  total_distance = 540 →
  ∃ (time : ℝ),
    time > 0 ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed2 * time - speed1 * time = 60 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l1655_165559


namespace NUMINAMATH_CALUDE_machines_completion_time_l1655_165583

theorem machines_completion_time 
  (time_A time_B time_C time_D time_E : ℝ) 
  (h_A : time_A = 4)
  (h_B : time_B = 12)
  (h_C : time_C = 6)
  (h_D : time_D = 8)
  (h_E : time_E = 18) :
  (1 / (1/time_A + 1/time_B + 1/time_C + 1/time_D + 1/time_E)) = 72/49 := by
  sorry

end NUMINAMATH_CALUDE_machines_completion_time_l1655_165583


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1655_165595

/-- Enumeration of sampling methods -/
inductive SamplingMethod
  | LotteryMethod
  | StratifiedSampling
  | RandomNumberMethod
  | SystematicSampling

/-- Scenario with total population and sample size -/
structure Scenario where
  total_population : ℕ
  sample_size : ℕ
  has_strata : Bool

/-- Function to determine the correct sampling method based on scenario -/
def correct_sampling_method (s : Scenario) : SamplingMethod :=
  if s.has_strata then
    SamplingMethod.StratifiedSampling
  else if s.total_population ≤ 30 then
    SamplingMethod.LotteryMethod
  else if s.sample_size ≤ 10 then
    SamplingMethod.RandomNumberMethod
  else
    SamplingMethod.SystematicSampling

/-- Theorem stating the correct sampling methods for given scenarios -/
theorem correct_sampling_methods :
  (correct_sampling_method ⟨30, 10, false⟩ = SamplingMethod.LotteryMethod) ∧
  (correct_sampling_method ⟨30, 10, true⟩ = SamplingMethod.StratifiedSampling) ∧
  (correct_sampling_method ⟨300, 10, false⟩ = SamplingMethod.RandomNumberMethod) ∧
  (correct_sampling_method ⟨300, 50, false⟩ = SamplingMethod.SystematicSampling) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1655_165595


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cuboid_l1655_165589

theorem sphere_surface_area_from_cuboid (a : ℝ) (h : a > 0) :
  let cuboid_dimensions := (2*a, a, a)
  let sphere_radius := Real.sqrt (3/2 * a^2)
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 6 * Real.pi * a^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cuboid_l1655_165589


namespace NUMINAMATH_CALUDE_a_51_equals_101_l1655_165545

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem a_51_equals_101 (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_a_51_equals_101_l1655_165545


namespace NUMINAMATH_CALUDE_min_value_problem_l1655_165572

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_sum : x + 2*y = 1) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1655_165572


namespace NUMINAMATH_CALUDE_x_value_l1655_165580

theorem x_value : ∃ x : ℝ, (x = 88 * (1 + 0.20)) ∧ (x = 105.6) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1655_165580


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1655_165575

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 1/x + 1/y = 1 → 1/(x-1) + 9/(y-1) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1655_165575


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1655_165573

theorem sqrt_expression_equality : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1655_165573


namespace NUMINAMATH_CALUDE_bus_rental_optimization_l1655_165506

theorem bus_rental_optimization (total_people : ℕ) (small_bus_seats small_bus_cost : ℕ)
  (large_bus_seats large_bus_cost : ℕ) (total_buses : ℕ) :
  total_people = 600 →
  small_bus_seats = 32 →
  large_bus_seats = 45 →
  small_bus_cost + 2 * large_bus_cost = 2800 →
  large_bus_cost = (125 * small_bus_cost) / 100 →
  total_buses = 14 →
  ∃ (small_buses large_buses : ℕ),
    small_buses + large_buses = total_buses ∧
    small_buses * small_bus_seats + large_buses * large_bus_seats ≥ total_people ∧
    small_buses * small_bus_cost + large_buses * large_bus_cost = 13600 ∧
    ∀ (other_small other_large : ℕ),
      other_small + other_large = total_buses →
      other_small * small_bus_seats + other_large * large_bus_seats ≥ total_people →
      other_small * small_bus_cost + other_large * large_bus_cost ≥ 13600 :=
by sorry

end NUMINAMATH_CALUDE_bus_rental_optimization_l1655_165506


namespace NUMINAMATH_CALUDE_sphere_division_theorem_l1655_165512

/-- The maximum number of regions into which a sphere can be divided by n great circles -/
def sphere_regions (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem: The maximum number of regions into which a sphere can be divided by n great circles is n^2 - n + 2 -/
theorem sphere_division_theorem (n : ℕ) : 
  sphere_regions n = n^2 - n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_division_theorem_l1655_165512


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l1655_165524

/-- Calculates the final alcohol percentage after adding pure alcohol to a solution -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_initial_percentage : initial_percentage = 35)
  (h_added_alcohol : added_alcohol = 1.8) :
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  final_alcohol / final_volume * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l1655_165524


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1655_165558

theorem min_sum_of_squares (x y : ℝ) (h : 4 * x^2 + 5 * x * y + 4 * y^2 = 5) :
  ∃ (S_min : ℝ), S_min = 10/13 ∧ x^2 + y^2 ≥ S_min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l1655_165558


namespace NUMINAMATH_CALUDE_equal_roots_values_l1655_165564

theorem equal_roots_values (x m : ℝ) : 
  (x^2 * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x^2 / m → 
  (∀ x, 2*x^2 - 4*x - m^2 - 2*m = 0) → 
  (m = -1 + Real.sqrt 3 ∨ m = -1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_values_l1655_165564


namespace NUMINAMATH_CALUDE_inequality_proof_l1655_165531

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1655_165531


namespace NUMINAMATH_CALUDE_line_k_equation_l1655_165599

/-- Given two lines in the xy-plane and conditions for a third line K, prove that
    the equation y = (4/15)x + (89/15) satisfies all conditions for line K. -/
theorem line_k_equation (x y : ℝ) : 
  let line1 : ℝ → ℝ := λ x => (4/5) * x + 3
  let line2 : ℝ → ℝ := λ x => (3/4) * x + 5
  let lineK : ℝ → ℝ := λ x => (4/15) * x + (89/15)
  (∀ x, lineK x = (1/3) * (line1 x - 3) + 3 * 3) ∧ 
  (lineK 4 = line2 4) ∧ 
  (lineK 4 = 7) := by
  sorry


end NUMINAMATH_CALUDE_line_k_equation_l1655_165599


namespace NUMINAMATH_CALUDE_intersection_distance_l1655_165502

/-- The distance between the intersection points of y = 5 and y = 5x^2 + 2x - 2 is 2.4 -/
theorem intersection_distance : 
  let f (x : ℝ) := 5*x^2 + 2*x - 2
  let g (x : ℝ) := 5
  let roots := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2.4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1655_165502


namespace NUMINAMATH_CALUDE_no_distinct_integers_divisibility_l1655_165525

theorem no_distinct_integers_divisibility : ¬∃ (a : Fin 2001 → ℕ+), 
  (∀ (i j : Fin 2001), i ≠ j → (a i).val * (a j).val ∣ 
    ((a i).val ^ 2000 - (a i).val ^ 1000 + 1) * 
    ((a j).val ^ 2000 - (a j).val ^ 1000 + 1)) ∧ 
  (∀ (i j : Fin 2001), i ≠ j → a i ≠ a j) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_integers_divisibility_l1655_165525


namespace NUMINAMATH_CALUDE_identity_iff_annihilator_l1655_165526

variable (R : Type) [Fintype R] [CommRing R]

def has_multiplicative_identity (R : Type) [Ring R] : Prop :=
  ∃ e : R, ∀ x : R, e * x = x ∧ x * e = x

def annihilator_is_zero (R : Type) [Ring R] : Prop :=
  ∀ a : R, (∀ x : R, a * x = 0) → a = 0

theorem identity_iff_annihilator (R : Type) [Fintype R] [CommRing R] :
  has_multiplicative_identity R ↔ annihilator_is_zero R :=
sorry

end NUMINAMATH_CALUDE_identity_iff_annihilator_l1655_165526


namespace NUMINAMATH_CALUDE_function_equality_l1655_165513

theorem function_equality :
  (∀ x : ℝ, x^2 = (x^6)^(1/3)) ∧
  (∀ x : ℝ, x = (x^3)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l1655_165513


namespace NUMINAMATH_CALUDE_cakes_served_total_l1655_165577

/-- The number of cakes served over two days in a restaurant -/
theorem cakes_served_total (lunch_today : ℕ) (dinner_today : ℕ) (yesterday : ℕ)
  (h1 : lunch_today = 5)
  (h2 : dinner_today = 6)
  (h3 : yesterday = 3) :
  lunch_today + dinner_today + yesterday = 14 :=
by sorry

end NUMINAMATH_CALUDE_cakes_served_total_l1655_165577


namespace NUMINAMATH_CALUDE_set_inclusion_theorem_l1655_165576

def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

theorem set_inclusion_theorem :
  (∀ x ∈ B, x ∈ A 1) ∧
  (∀ a : ℝ, (∀ x ∈ A a, x ∈ B) ↔ a < -8 ∨ a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_theorem_l1655_165576


namespace NUMINAMATH_CALUDE_bedroom_area_l1655_165505

/-- Proves that the area of each bedroom is 121 square feet given the specified house layout --/
theorem bedroom_area (total_area : ℝ) (num_bedrooms : ℕ) (num_bathrooms : ℕ)
  (bathroom_length bathroom_width : ℝ) (kitchen_area : ℝ) :
  total_area = 1110 →
  num_bedrooms = 4 →
  num_bathrooms = 2 →
  bathroom_length = 6 →
  bathroom_width = 8 →
  kitchen_area = 265 →
  ∃ (bedroom_area : ℝ),
    bedroom_area = 121 ∧
    total_area = num_bedrooms * bedroom_area + 
                 num_bathrooms * bathroom_length * bathroom_width +
                 2 * kitchen_area :=
by
  sorry

end NUMINAMATH_CALUDE_bedroom_area_l1655_165505


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l1655_165551

theorem soccer_league_female_fraction :
  let last_year_males : ℕ := 30
  let male_increase_rate : ℚ := 11/10
  let female_increase_rate : ℚ := 5/4
  let total_increase_rate : ℚ := 23/20
  let this_year_males : ℚ := last_year_males * male_increase_rate
  let last_year_females : ℚ := (total_increase_rate * (last_year_males : ℚ) - this_year_males) / (female_increase_rate - total_increase_rate)
  let this_year_females : ℚ := last_year_females * female_increase_rate
  let this_year_total : ℚ := this_year_males + this_year_females
  
  (this_year_females / this_year_total) = 75/207 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l1655_165551


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1655_165535

/-- Proves that the equation 3(x+1)² = 2(x+1) is equivalent to a quadratic equation in the standard form ax² + bx + c = 0, where a ≠ 0 -/
theorem equation_is_quadratic (x : ℝ) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (3 * (x + 1)^2 = 2 * (x + 1)) ↔ (a * x^2 + b * x + c = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l1655_165535


namespace NUMINAMATH_CALUDE_divisors_of_720_l1655_165520

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l1655_165520


namespace NUMINAMATH_CALUDE_problem_solution_l1655_165523

theorem problem_solution :
  ∀ (x a b : ℝ),
  (∃ y : ℝ, y^2 = x ∧ y = a + 3) ∧
  (∃ z : ℝ, z^2 = x ∧ z = 2*a - 15) ∧
  (3^2 = 2*b - 1) →
  (a = 4 ∧ b = 5 ∧ (a + b - 1)^(1/3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1655_165523


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l1655_165552

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five :
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l1655_165552


namespace NUMINAMATH_CALUDE_utilities_percentage_l1655_165598

def budget_circle_graph (transportation research_development equipment supplies salaries utilities : ℝ) : Prop :=
  transportation = 20 ∧
  research_development = 9 ∧
  equipment = 4 ∧
  supplies = 2 ∧
  salaries = 60 ∧
  transportation + research_development + equipment + supplies + salaries + utilities = 100

theorem utilities_percentage 
  (transportation research_development equipment supplies salaries utilities : ℝ)
  (h : budget_circle_graph transportation research_development equipment supplies salaries utilities)
  (h_salaries : salaries * 360 / 100 = 216) : utilities = 5 := by
  sorry

end NUMINAMATH_CALUDE_utilities_percentage_l1655_165598


namespace NUMINAMATH_CALUDE_right_triangle_sin_d_l1655_165584

theorem right_triangle_sin_d (D E F : Real) (h1 : 4 * Real.sin D = 5 * Real.cos D) :
  Real.sin D = 5 * Real.sqrt 41 / 41 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_d_l1655_165584


namespace NUMINAMATH_CALUDE_oranges_left_l1655_165567

/-- Proves that the number of oranges Joan is left with is equal to the number she picked minus the number Sara sold. -/
theorem oranges_left (joan_picked : ℕ) (sara_sold : ℕ) (joan_left : ℕ)
  (h1 : joan_picked = 37)
  (h2 : sara_sold = 10)
  (h3 : joan_left = 27) :
  joan_left = joan_picked - sara_sold :=
by sorry

end NUMINAMATH_CALUDE_oranges_left_l1655_165567


namespace NUMINAMATH_CALUDE_luke_weed_eating_money_l1655_165519

def mowing_money : ℕ := 9
def weeks_lasting : ℕ := 9
def weekly_spending : ℕ := 3

theorem luke_weed_eating_money :
  mowing_money + (weeks_lasting * weekly_spending - mowing_money) = 18 := by
  sorry

end NUMINAMATH_CALUDE_luke_weed_eating_money_l1655_165519


namespace NUMINAMATH_CALUDE_sum_abc_equals_22_l1655_165594

theorem sum_abc_equals_22 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (eq1 : a^2 + b*c = 115)
  (eq2 : b^2 + a*c = 127)
  (eq3 : c^2 + a*b = 115) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_22_l1655_165594


namespace NUMINAMATH_CALUDE_percent_of_percent_l1655_165555

theorem percent_of_percent (y : ℝ) (h : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1655_165555


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l1655_165521

theorem quadratic_equation_integer_roots :
  let S : Set ℝ := {a : ℝ | a > 0 ∧ ∃ x y : ℤ, x ≠ y ∧ a^2 * x^2 + a * x + 1 - 13 * a^2 = 0 ∧ a^2 * y^2 + a * y + 1 - 13 * a^2 = 0}
  S = {1, 1/3, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l1655_165521


namespace NUMINAMATH_CALUDE_winter_olympics_merchandise_l1655_165515

def total_items : ℕ := 180
def figurine_cost : ℕ := 80
def pendant_cost : ℕ := 50
def total_spent : ℕ := 11400
def figurine_price : ℕ := 100
def pendant_price : ℕ := 60
def min_profit : ℕ := 2900

theorem winter_olympics_merchandise (x y : ℕ) (m : ℕ) : 
  x + y = total_items ∧ 
  figurine_cost * x + pendant_cost * y = total_spent ∧
  (pendant_price - pendant_cost) * m + (figurine_price - figurine_cost) * (total_items - m) ≥ min_profit →
  x = 80 ∧ y = 100 ∧ m ≤ 70 := by
  sorry

end NUMINAMATH_CALUDE_winter_olympics_merchandise_l1655_165515


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_parallelepiped_l1655_165511

theorem sphere_surface_area_with_inscribed_parallelepiped (a b c : ℝ) (S : ℝ) :
  a = 1 →
  b = 2 →
  c = 2 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_parallelepiped_l1655_165511


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1655_165530

theorem simplify_sqrt_sum (h : π / 2 < 2 ∧ 2 < 3 * π / 4) :
  Real.sqrt (1 + Real.sin 4) + Real.sqrt (1 - Real.sin 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1655_165530


namespace NUMINAMATH_CALUDE_lions_count_l1655_165528

theorem lions_count (lions tigers cougars : ℕ) : 
  tigers = 14 →
  cougars = (lions + tigers) / 2 →
  lions + tigers + cougars = 39 →
  lions = 12 := by
sorry

end NUMINAMATH_CALUDE_lions_count_l1655_165528


namespace NUMINAMATH_CALUDE_opposite_of_three_l1655_165597

theorem opposite_of_three : -(3 : ℝ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1655_165597


namespace NUMINAMATH_CALUDE_opposite_sign_power_l1655_165508

theorem opposite_sign_power (x y : ℝ) : 
  (|x + 3| * (y - 2)^2 ≤ 0 ∧ |x + 3| + (y - 2)^2 = 0) → x^y = 9 := by sorry

end NUMINAMATH_CALUDE_opposite_sign_power_l1655_165508
