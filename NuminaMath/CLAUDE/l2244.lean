import Mathlib

namespace jerry_spent_two_tickets_l2244_224481

def tickets_spent (initial_tickets : ℕ) (won_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  initial_tickets + won_tickets - final_tickets

theorem jerry_spent_two_tickets :
  tickets_spent 4 47 49 = 2 := by
  sorry

end jerry_spent_two_tickets_l2244_224481


namespace rod_triangle_impossibility_l2244_224452

theorem rod_triangle_impossibility (L : ℝ) (a b : ℝ) 
  (h1 : L > 0) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a + b = L / 2) : 
  ¬(L / 2 + a > b ∧ L / 2 + b > a ∧ a + b > L / 2) := by
  sorry


end rod_triangle_impossibility_l2244_224452


namespace unique_center_symmetric_not_axis_symmetric_l2244_224486

-- Define the shapes
inductive Shape
  | Square
  | EquilateralTriangle
  | Circle
  | Parallelogram

-- Define the symmetry properties
def is_center_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Square => true
  | Shape.EquilateralTriangle => false
  | Shape.Circle => true
  | Shape.Parallelogram => true

def is_axis_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Square => true
  | Shape.EquilateralTriangle => true
  | Shape.Circle => true
  | Shape.Parallelogram => false

-- Theorem statement
theorem unique_center_symmetric_not_axis_symmetric :
  ∀ s : Shape, (is_center_symmetric s ∧ ¬is_axis_symmetric s) ↔ s = Shape.Parallelogram :=
sorry

end unique_center_symmetric_not_axis_symmetric_l2244_224486


namespace parallel_line_perpendicular_line_l2244_224465

-- Define the point P as the intersection of two lines
def P : ℝ × ℝ := (2, 1)

-- Define line l1
def l1 (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define the equation of a line passing through P with slope m
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Theorem for parallel case
theorem parallel_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ line_through_P 4 x y) ∧
                 a = 4 ∧ b = -1 ∧ c = -7 :=
sorry

-- Theorem for perpendicular case
theorem perpendicular_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ line_through_P (-1/4) x y) ∧
                 a = 1 ∧ b = 4 ∧ c = -6 :=
sorry

end parallel_line_perpendicular_line_l2244_224465


namespace greatest_difference_of_arithmetic_progression_l2244_224467

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has two distinct real roots -/
def hasTwoRoots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c > 0

/-- Generates all six quadratic equations with coefficients a, 2b, 4c in any order -/
def generateEquations (a b c : ℤ) : List QuadraticEquation :=
  [
    ⟨a, 2*b, 4*c⟩,
    ⟨a, 4*c, 2*b⟩,
    ⟨2*b, a, 4*c⟩,
    ⟨2*b, 4*c, a⟩,
    ⟨4*c, a, 2*b⟩,
    ⟨4*c, 2*b, a⟩
  ]

/-- The main theorem to be proved -/
theorem greatest_difference_of_arithmetic_progression
  (a b c : ℤ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_decreasing : a > b ∧ b > c)
  (h_arithmetic : ∃ d : ℤ, b = a + d ∧ c = a + 2*d)
  (h_two_roots : ∀ eq ∈ generateEquations a b c, hasTwoRoots eq) :
  ∃ (d : ℤ), d = -3 ∧ a = 4 ∧ b = 1 ∧ c = -2 ∧
  ∀ (d' : ℤ) (a' b' c' : ℤ),
    a' ≠ 0 → b' ≠ 0 → c' ≠ 0 →
    a' > b' → b' > c' →
    b' = a' + d' → c' = a' + 2*d' →
    (∀ eq ∈ generateEquations a' b' c', hasTwoRoots eq) →
    d' ≥ d :=
by
  sorry

end greatest_difference_of_arithmetic_progression_l2244_224467


namespace total_spent_is_89_10_l2244_224492

/-- The total amount spent by Edward and his friend after the discount -/
def total_spent_after_discount (
  trick_deck_price : ℝ) 
  (edward_decks : ℕ) 
  (edward_hat_price : ℝ)
  (friend_decks : ℕ)
  (friend_wand_price : ℝ)
  (discount_rate : ℝ) : ℝ :=
  let total_before_discount := 
    trick_deck_price * (edward_decks + friend_decks) + edward_hat_price + friend_wand_price
  total_before_discount * (1 - discount_rate)

/-- Theorem stating that the total amount spent after the discount is $89.10 -/
theorem total_spent_is_89_10 :
  total_spent_after_discount 9 4 12 4 15 0.1 = 89.10 := by
  sorry

end total_spent_is_89_10_l2244_224492


namespace carnival_tickets_l2244_224474

theorem carnival_tickets (num_friends : ℕ) (total_tickets : ℕ) (h1 : num_friends = 6) (h2 : total_tickets = 234) :
  (total_tickets / num_friends : ℕ) = 39 := by
  sorry

end carnival_tickets_l2244_224474


namespace bus_line_count_l2244_224435

theorem bus_line_count (people_in_front people_behind : ℕ) 
  (h1 : people_in_front = 6) 
  (h2 : people_behind = 5) : 
  people_in_front + 1 + people_behind = 12 := by
  sorry

end bus_line_count_l2244_224435


namespace perpendicular_lines_parallel_l2244_224422

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel
  (α : Plane) (a b : Line)
  (ha : perpendicular a α)
  (hb : perpendicular b α) :
  parallel a b :=
sorry

end perpendicular_lines_parallel_l2244_224422


namespace sugar_in_recipe_l2244_224415

theorem sugar_in_recipe (sugar_already_in : ℕ) (sugar_to_add : ℕ) : 
  sugar_already_in = 2 → sugar_to_add = 11 → sugar_already_in + sugar_to_add = 13 := by
  sorry

end sugar_in_recipe_l2244_224415


namespace exists_diff_same_num_prime_divisors_l2244_224423

/-- The number of distinct prime divisors of a natural number -/
def numDistinctPrimeDivisors (n : ℕ) : ℕ := sorry

/-- For any natural number n, there exist natural numbers a and b such that
    n = a - b and they have the same number of distinct prime divisors -/
theorem exists_diff_same_num_prime_divisors (n : ℕ) :
  ∃ a b : ℕ, n = a - b ∧ numDistinctPrimeDivisors a = numDistinctPrimeDivisors b := by
  sorry

end exists_diff_same_num_prime_divisors_l2244_224423


namespace f_2013_l2244_224425

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 6) ≤ f (x + 2) + 4) ∧
  (∀ x : ℝ, f (x + 4) ≥ f (x + 2) + 2) ∧
  (f 1 = 1)

theorem f_2013 (f : ℝ → ℝ) (h : f_properties f) : f 2013 = 2013 := by
  sorry

end f_2013_l2244_224425


namespace decaf_coffee_percentage_l2244_224491

/-- Proves that the percentage of decaffeinated coffee in the second batch is 70% --/
theorem decaf_coffee_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (second_batch : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : second_batch = 100)
  (h4 : final_decaf_percent = 30)
  (h5 : initial_stock * initial_decaf_percent / 100 + second_batch * x / 100 = 
        (initial_stock + second_batch) * final_decaf_percent / 100) :
  x = 70 := by
  sorry

#check decaf_coffee_percentage

end decaf_coffee_percentage_l2244_224491


namespace ceiling_floor_difference_l2244_224497

theorem ceiling_floor_difference : ⌈(15 : ℚ) / 7 * (-27 : ℚ) / 3⌉ - ⌊(15 : ℚ) / 7 * ⌈(-27 : ℚ) / 3⌉⌋ = 1 := by
  sorry

end ceiling_floor_difference_l2244_224497


namespace circle_area_increase_l2244_224409

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end circle_area_increase_l2244_224409


namespace double_wardrobe_with_socks_l2244_224434

/-- Represents the number of pairs of an item in a wardrobe -/
structure WardobeItem where
  pairs : Nat

/-- Represents a wardrobe with various clothing items -/
structure Wardrobe where
  socks : WardobeItem
  shoes : WardobeItem
  pants : WardobeItem
  tshirts : WardobeItem

/-- Calculates the total number of individual items in a wardrobe -/
def totalItems (w : Wardrobe) : Nat :=
  w.socks.pairs * 2 + w.shoes.pairs * 2 + w.pants.pairs + w.tshirts.pairs

/-- Theorem: Buying 35 pairs of socks doubles the number of items in Jonas' wardrobe -/
theorem double_wardrobe_with_socks (jonas : Wardrobe)
    (h1 : jonas.socks.pairs = 20)
    (h2 : jonas.shoes.pairs = 5)
    (h3 : jonas.pants.pairs = 10)
    (h4 : jonas.tshirts.pairs = 10) :
    totalItems { socks := ⟨jonas.socks.pairs + 35⟩,
                 shoes := jonas.shoes,
                 pants := jonas.pants,
                 tshirts := jonas.tshirts } = 2 * totalItems jonas := by
  sorry


end double_wardrobe_with_socks_l2244_224434


namespace not_always_equal_l2244_224466

-- Define the binary operation
def binary_op {S : Type} (op : S → S → S) : Prop :=
  ∀ (a b : S), ∃ (c : S), op a b = c

-- Define the property of the operation
def special_property {S : Type} (op : S → S → S) : Prop :=
  ∀ (a b : S), op a (op b a) = b

theorem not_always_equal {S : Type} [Inhabited S] (op : S → S → S) 
  (h1 : binary_op op) (h2 : special_property op) (h3 : ∃ (x y : S), x ≠ y) :
  ∃ (a b : S), op (op a b) a ≠ a := by
  sorry

end not_always_equal_l2244_224466


namespace expression_value_l2244_224431

theorem expression_value :
  let a : ℤ := 3
  let b : ℤ := 2
  let c : ℤ := 1
  (a + (b + c)^2) - ((a + b)^2 - c) = -12 :=
by sorry

end expression_value_l2244_224431


namespace class_size_is_40_l2244_224401

/-- Represents the heights of rectangles in a histogram --/
structure HistogramHeights where
  ratios : List Nat
  first_frequency : Nat

/-- Calculates the total number of students represented by a histogram --/
def totalStudents (h : HistogramHeights) : Nat :=
  let unit_frequency := h.first_frequency / h.ratios.head!
  unit_frequency * h.ratios.sum

/-- Theorem stating that for the given histogram, the total number of students is 40 --/
theorem class_size_is_40 (h : HistogramHeights) 
    (height_ratio : h.ratios = [4, 3, 7, 6]) 
    (first_freq : h.first_frequency = 8) : 
  totalStudents h = 40 := by
  sorry

#eval totalStudents { ratios := [4, 3, 7, 6], first_frequency := 8 }

end class_size_is_40_l2244_224401


namespace nested_sqrt_equality_l2244_224480

theorem nested_sqrt_equality : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * (3 ^ (1/4)) := by
  sorry

end nested_sqrt_equality_l2244_224480


namespace smallest_undefined_inverse_l2244_224426

def is_inverse_undefined (a n : ℕ) : Prop :=
  ¬ (∃ b : ℕ, a * b ≡ 1 [MOD n])

theorem smallest_undefined_inverse : 
  (∀ a : ℕ, 0 < a → a < 10 → 
    ¬(is_inverse_undefined a 55 ∧ is_inverse_undefined a 66)) ∧ 
  (is_inverse_undefined 10 55 ∧ is_inverse_undefined 10 66) := by
  sorry

end smallest_undefined_inverse_l2244_224426


namespace sara_quarters_remaining_l2244_224450

/-- Given that Sara initially had 783 quarters and her dad borrowed 271 quarters,
    prove that Sara now has 512 quarters. -/
theorem sara_quarters_remaining (initial_quarters borrowed_quarters : ℕ) 
    (h1 : initial_quarters = 783)
    (h2 : borrowed_quarters = 271) :
    initial_quarters - borrowed_quarters = 512 := by
  sorry

end sara_quarters_remaining_l2244_224450


namespace cheese_bread_solution_l2244_224411

/-- Represents the problem of buying cheese bread for a group of people. -/
structure CheeseBreadProblem where
  cost_per_100g : ℚ  -- Cost in R$ per 100g of cheese bread
  pieces_per_100g : ℕ  -- Number of pieces in 100g of cheese bread
  pieces_per_person : ℕ  -- Average number of pieces eaten per person
  total_people : ℕ  -- Total number of people
  scale_precision : ℕ  -- Precision of the bakery's scale in grams

/-- Calculates the amount to buy, cost, and leftover pieces for a given CheeseBreadProblem. -/
def solve_cheese_bread_problem (p : CheeseBreadProblem) :
  (ℕ × ℚ × ℕ) :=
  sorry

/-- Theorem stating the correct solution for the given problem. -/
theorem cheese_bread_solution :
  let problem := CheeseBreadProblem.mk 3.2 10 5 23 100
  let (amount, cost, leftover) := solve_cheese_bread_problem problem
  amount = 1200 ∧ cost = 38.4 ∧ leftover = 5 :=
sorry

end cheese_bread_solution_l2244_224411


namespace problem_statement_l2244_224432

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 2, m ≤ x^2 - 2*x

def q (m : ℝ) : Prop := ∃ x ≥ 0, 2^x + 3 = m

theorem problem_statement :
  (∀ m : ℝ, p m ↔ m ∈ Set.Iic (-1)) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Iic (-1) ∪ Set.Ici 4) :=
by sorry

end problem_statement_l2244_224432


namespace susan_age_proof_l2244_224442

def james_age_in_15_years : ℕ := 37

def james_current_age : ℕ := james_age_in_15_years - 15

def james_age_8_years_ago : ℕ := james_current_age - 8

def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2

def janet_current_age : ℕ := janet_age_8_years_ago + 8

def susan_current_age : ℕ := janet_current_age - 3

def susan_age_in_5_years : ℕ := susan_current_age + 5

theorem susan_age_proof : susan_age_in_5_years = 17 := by
  sorry

end susan_age_proof_l2244_224442


namespace lower_right_is_one_l2244_224440

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Check if a number appears exactly once in each row --/
def valid_rows (g : Grid) : Prop :=
  ∀ i : Fin 5, ∀ n : Fin 5, (∃! j : Fin 5, g i j = n)

/-- Check if a number appears exactly once in each column --/
def valid_columns (g : Grid) : Prop :=
  ∀ j : Fin 5, ∀ n : Fin 5, (∃! i : Fin 5, g i j = n)

/-- Check if no number repeats on the main diagonal --/
def valid_diagonal (g : Grid) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i i ≠ g j j

/-- Check if the grid satisfies the initial placements --/
def valid_initial (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 1 1 = 1 ∧ g 2 2 = 2 ∧ g 0 3 = 3 ∧ g 3 3 = 4 ∧ g 1 4 = 0

theorem lower_right_is_one (g : Grid) 
  (h_rows : valid_rows g) 
  (h_cols : valid_columns g) 
  (h_diag : valid_diagonal g) 
  (h_init : valid_initial g) : 
  g 4 4 = 0 :=
sorry

end lower_right_is_one_l2244_224440


namespace fraction_of_y_l2244_224406

theorem fraction_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end fraction_of_y_l2244_224406


namespace correct_quotient_proof_l2244_224427

theorem correct_quotient_proof (dividend : ℕ) (mistaken_divisor correct_divisor mistaken_quotient : ℕ) 
  (h1 : mistaken_divisor = 12)
  (h2 : correct_divisor = 21)
  (h3 : mistaken_quotient = 42)
  (h4 : dividend = mistaken_divisor * mistaken_quotient)
  (h5 : dividend % correct_divisor = 0) :
  dividend / correct_divisor = 24 := by
sorry

end correct_quotient_proof_l2244_224427


namespace total_chairs_l2244_224449

/-- Represents the number of chairs of each color in a classroom. -/
structure Classroom where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Defines a classroom with the given conditions. -/
def classroom : Classroom where
  blue := 10
  green := 3 * 10
  white := (3 * 10 + 10) - 13

/-- Theorem stating the total number of chairs in the classroom. -/
theorem total_chairs : classroom.blue + classroom.green + classroom.white = 67 := by
  sorry

#eval classroom.blue + classroom.green + classroom.white

end total_chairs_l2244_224449


namespace seventh_term_is_ten_l2244_224482

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  (a 2 = 2) ∧ 
  (a 4 + a 5 = 12)

/-- Theorem stating that the 7th term of the arithmetic sequence is 10 -/
theorem seventh_term_is_ten (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 7 = 10 := by
  sorry

end seventh_term_is_ten_l2244_224482


namespace sum_of_products_inequality_l2244_224458

theorem sum_of_products_inequality (a b c d : ℝ) (h : a + b + c + d = 1) :
  a * b + b * c + c * d + d * a ≤ 1 / 4 := by
  sorry

end sum_of_products_inequality_l2244_224458


namespace vasya_promotion_higher_revenue_l2244_224495

/-- Represents the revenue from candy box sales under different promotions -/
def candy_revenue (normal_revenue : ℝ) : Prop :=
  let vasya_revenue := normal_revenue * 2 * 0.8
  let kolya_revenue := normal_revenue * (8/3)
  (vasya_revenue = 16000) ∧ 
  (kolya_revenue = 13333.33333333333) ∧ 
  (vasya_revenue - normal_revenue = 6000)

/-- Theorem stating that Vasya's promotion leads to higher revenue -/
theorem vasya_promotion_higher_revenue :
  candy_revenue 10000 :=
sorry

end vasya_promotion_higher_revenue_l2244_224495


namespace instantaneous_velocity_at_3_l2244_224448

-- Define the displacement function
def s (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t^2

-- Theorem: The instantaneous velocity at t = 3 is 54
theorem instantaneous_velocity_at_3 : v 3 = 54 := by
  sorry

end instantaneous_velocity_at_3_l2244_224448


namespace equation_solution_l2244_224433

theorem equation_solution : ∃ x : ℝ, 11 + Real.sqrt (x + 6 * 4 / 3) = 13 ∧ x = -4 := by
  sorry

end equation_solution_l2244_224433


namespace specific_mountain_depth_l2244_224490

/-- Represents a cone-shaped mountain partially submerged in water -/
structure Mountain where
  totalHeight : ℝ
  baseRadius : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of the mountain -/
def oceanDepth (m : Mountain) : ℝ :=
  m.totalHeight * (1 - (m.aboveWaterVolumeFraction) ^ (1/3))

/-- Theorem stating the ocean depth for the specific mountain described in the problem -/
theorem specific_mountain_depth :
  let m : Mountain := {
    totalHeight := 10000,
    baseRadius := 3000,
    aboveWaterVolumeFraction := 1/10
  }
  oceanDepth m = 5360 := by
  sorry


end specific_mountain_depth_l2244_224490


namespace jack_needs_additional_money_l2244_224479

def socks_price : ℝ := 12.75
def shoes_price : ℝ := 145
def ball_price : ℝ := 38
def bag_price : ℝ := 47
def shoes_discount : ℝ := 0.05
def bag_discount : ℝ := 0.10
def jack_money : ℝ := 25

def total_cost : ℝ := 
  2 * socks_price + 
  shoes_price * (1 - shoes_discount) + 
  ball_price + 
  bag_price * (1 - bag_discount)

theorem jack_needs_additional_money : 
  total_cost - jack_money = 218.55 := by sorry

end jack_needs_additional_money_l2244_224479


namespace contacts_in_sphere_tetrahedron_l2244_224456

/-- The number of contacts in a tetrahedral stack of spheres -/
def tetrahedron_contacts (n : ℕ) : ℕ := n^3 - n

/-- 
Theorem: In a tetrahedron formed by stacking identical spheres, 
where each edge has n spheres, the total number of points of 
tangency between the spheres is n³ - n.
-/
theorem contacts_in_sphere_tetrahedron (n : ℕ) : 
  tetrahedron_contacts n = n^3 - n := by
  sorry

end contacts_in_sphere_tetrahedron_l2244_224456


namespace coefficient_of_x_cubed_l2244_224476

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 2*x) + 2 * (x + 3*x^3 - 2*x^2 + 4*x^5 - x^3) - 6 * (2 + x - 5*x^3 - 2*x^2)

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression))) 0 / 6 = 26 := by sorry

end coefficient_of_x_cubed_l2244_224476


namespace book_pages_l2244_224428

/-- The number of pages Mrs. Hilt read -/
def pages_read : ℕ := 11

/-- The number of pages Mrs. Hilt has left to read -/
def pages_left : ℕ := 6

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_read + pages_left

theorem book_pages : total_pages = 17 := by sorry

end book_pages_l2244_224428


namespace divide_ten_theorem_l2244_224447

theorem divide_ten_theorem (x : ℝ) : 
  x > 0 ∧ x < 10 →
  (10 - x)^2 + x^2 + (10 - x) / x = 72 →
  x = 2 := by
sorry


end divide_ten_theorem_l2244_224447


namespace submarine_age_conversion_l2244_224455

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ × ℕ × ℕ) : ℕ :=
  let (a, b, c) := octal
  a * 8^2 + b * 8^1 + c * 8^0

theorem submarine_age_conversion :
  octal_to_decimal (3, 6, 7) = 247 := by
  sorry

end submarine_age_conversion_l2244_224455


namespace reservoir_drainage_l2244_224487

/- Given conditions -/
def initial_drainage_rate : ℝ := 8
def initial_drain_time : ℝ := 6
def max_drainage_capacity : ℝ := 12

/- Theorem statement -/
theorem reservoir_drainage :
  let reservoir_volume : ℝ := initial_drainage_rate * initial_drain_time
  let drainage_relation (Q t : ℝ) : Prop := Q = reservoir_volume / t
  let min_drainage_5hours : ℝ := reservoir_volume / 5
  let min_time_max_capacity : ℝ := reservoir_volume / max_drainage_capacity
  
  (reservoir_volume = 48) ∧
  (∀ Q t, drainage_relation Q t ↔ Q = 48 / t) ∧
  (min_drainage_5hours = 9.6) ∧
  (min_time_max_capacity = 4) :=
by sorry

end reservoir_drainage_l2244_224487


namespace barry_average_proof_l2244_224403

def barry_yards : List ℕ := [98, 107, 85, 89, 91]
def next_game_target : ℕ := 130
def total_games : ℕ := 6

theorem barry_average_proof :
  (barry_yards.sum + next_game_target) / total_games = 100 := by
  sorry

end barry_average_proof_l2244_224403


namespace apollonius_circle_minimum_l2244_224418

/-- Given points A, B, D, and a moving point P in a 2D plane, 
    prove that the minimum value of 2|PD|+|PB| is 2√10 when |PA|/|PB| = 1/2 -/
theorem apollonius_circle_minimum (A B D P : EuclideanSpace ℝ (Fin 2)) :
  A = ![(1 : ℝ), 0] →
  B = ![4, 0] →
  D = ![0, 3] →
  dist P A / dist P B = (1 : ℝ) / 2 →
  ∃ (P : EuclideanSpace ℝ (Fin 2)), 2 * dist P D + dist P B ≥ 2 * Real.sqrt 10 :=
by sorry

end apollonius_circle_minimum_l2244_224418


namespace squares_containing_a_l2244_224417

/-- Represents a square in a grid -/
structure Square where
  size : Nat
  contains_a : Bool

/-- Represents a 4x4 grid -/
def Grid := Array (Array Square)

/-- Creates a 4x4 grid with A in one cell -/
def create_grid : Grid := sorry

/-- Counts the total number of squares in the grid -/
def total_squares (grid : Grid) : Nat := sorry

/-- Counts the number of squares containing A -/
def squares_with_a (grid : Grid) : Nat := sorry

/-- Theorem stating that there are 13 squares containing A in a 4x4 grid with A in one cell -/
theorem squares_containing_a (grid : Grid) :
  total_squares grid = 20 → squares_with_a grid = 13 := by sorry

end squares_containing_a_l2244_224417


namespace perpendicular_vectors_imply_n_equals_three_l2244_224437

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (3, 2) and b = (2, n), if a is perpendicular to b, then n = 3 -/
theorem perpendicular_vectors_imply_n_equals_three :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ → ℝ × ℝ := λ n => (2, n)
  ∀ n : ℝ, perpendicular a (b n) → n = 3 := by
sorry


end perpendicular_vectors_imply_n_equals_three_l2244_224437


namespace car_trading_profit_l2244_224470

theorem car_trading_profit (P : ℝ) (h : P > 0) : 
  let discount_rate := 0.20
  let increase_rate := 0.55
  let buying_price := P * (1 - discount_rate)
  let selling_price := buying_price * (1 + increase_rate)
  let profit := selling_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 24 := by sorry

end car_trading_profit_l2244_224470


namespace min_value_on_interval_l2244_224451

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = -15 ∧ ∀ y ∈ Set.Icc 0 3, f y ≥ f x := by
  sorry

end min_value_on_interval_l2244_224451


namespace arithmetic_calculation_l2244_224462

theorem arithmetic_calculation : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end arithmetic_calculation_l2244_224462


namespace leadership_configurations_count_l2244_224471

-- Define the number of members in the society
def society_size : ℕ := 12

-- Define the number of positions to be filled
def chief_count : ℕ := 1
def supporting_chief_count : ℕ := 2
def inferior_officers_A_count : ℕ := 3
def inferior_officers_B_count : ℕ := 2

-- Define the function to calculate the number of ways to establish the leadership configuration
def leadership_configurations : ℕ := 
  society_size * (society_size - 1) * (society_size - 2) * 
  (Nat.choose (society_size - 3) inferior_officers_A_count) * 
  (Nat.choose (society_size - 3 - inferior_officers_A_count) inferior_officers_B_count)

-- Theorem statement
theorem leadership_configurations_count : leadership_configurations = 1663200 := by
  sorry

end leadership_configurations_count_l2244_224471


namespace radio_contest_winner_l2244_224460

theorem radio_contest_winner (n : ℕ) 
  (h1 : 35 % 5 = 0)
  (h2 : 35 % n = 0)
  (h3 : ∀ m : ℕ, m > 0 ∧ m < 35 → ¬(m % 5 = 0 ∧ m % n = 0)) : 
  n = 7 := by
sorry

end radio_contest_winner_l2244_224460


namespace backpack_price_l2244_224405

theorem backpack_price (t_shirt_price cap_price discount total_after_discount : ℕ) 
  (ht : t_shirt_price = 30)
  (hc : cap_price = 5)
  (hd : discount = 2)
  (ht : total_after_discount = 43) :
  ∃ backpack_price : ℕ, 
    t_shirt_price + backpack_price + cap_price - discount = total_after_discount ∧ 
    backpack_price = 10 := by
  sorry

end backpack_price_l2244_224405


namespace consecutive_numbers_sum_l2244_224445

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  sorry

end consecutive_numbers_sum_l2244_224445


namespace max_triangle_area_l2244_224436

theorem max_triangle_area (a b c : Real) (h1 : 0 < a) (h2 : a ≤ 1) (h3 : 1 ≤ b) 
  (h4 : b ≤ 2) (h5 : 2 ≤ c) (h6 : c ≤ 3) :
  ∃ (area : Real), area ≤ 1 ∧ 
    ∀ (A : Real), (∃ (α : Real), A = 1/2 * a * b * Real.sin α ∧ 
      a + b > c ∧ b + c > a ∧ c + a > b) → A ≤ area :=
by sorry

end max_triangle_area_l2244_224436


namespace sale_price_ratio_l2244_224414

theorem sale_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end sale_price_ratio_l2244_224414


namespace blue_parrots_count_l2244_224421

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) (blue_parrots : ℕ) : 
  total = 160 →
  green_fraction = 5/8 →
  blue_parrots = total - (green_fraction * total).num →
  blue_parrots = 60 := by
sorry

end blue_parrots_count_l2244_224421


namespace initial_marbles_proof_l2244_224412

/-- The number of marbles Connie gave to Juan -/
def marbles_to_juan : ℕ := 73

/-- The number of marbles Connie gave to Maria -/
def marbles_to_maria : ℕ := 45

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_to_juan + marbles_to_maria + marbles_left

theorem initial_marbles_proof : initial_marbles = 188 := by
  sorry

end initial_marbles_proof_l2244_224412


namespace savings_calculation_l2244_224468

def total_expenses : ℚ := 30150
def savings_rate : ℚ := 1/5

theorem savings_calculation (salary : ℚ) (h1 : salary * savings_rate + total_expenses = salary) :
  salary * savings_rate = 7537.5 := by
  sorry

end savings_calculation_l2244_224468


namespace fishing_tournament_result_l2244_224483

def fishing_tournament (jacob_initial : ℕ) (alex_multiplier emily_multiplier : ℕ) 
  (alex_loss emily_loss : ℕ) : ℕ :=
  let alex_initial := alex_multiplier * jacob_initial
  let emily_initial := emily_multiplier * jacob_initial
  let alex_final := alex_initial - alex_loss
  let emily_final := emily_initial - emily_loss
  let target := max alex_final emily_final + 1
  target - jacob_initial

theorem fishing_tournament_result : 
  fishing_tournament 8 7 3 23 10 = 26 := by sorry

end fishing_tournament_result_l2244_224483


namespace babylonian_conversion_l2244_224419

/-- Converts a Babylonian sexagesimal number to its decimal representation -/
def babylonian_to_decimal (a b : ℕ) : ℕ :=
  60^a + 10 * 60^b

/-- The Babylonian number 60^8 + 10 * 60^7 in decimal form -/
theorem babylonian_conversion :
  babylonian_to_decimal 8 7 = 195955200000000 := by
  sorry

#eval babylonian_to_decimal 8 7

end babylonian_conversion_l2244_224419


namespace sixth_angle_measure_l2244_224489

/-- The sum of interior angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The measures of the five known angles in the hexagon -/
def known_angles : List ℝ := [130, 95, 115, 120, 110]

/-- The theorem stating that the sixth angle in the hexagon measures 150° -/
theorem sixth_angle_measure :
  hexagon_angle_sum - (known_angles.sum) = 150 := by sorry

end sixth_angle_measure_l2244_224489


namespace first_number_proof_l2244_224484

theorem first_number_proof (x : ℕ) : 
  (∃ k : ℕ, x = 144 * k + 23) ∧ 
  (∃ m : ℕ, 7373 = 144 * m + 29) ∧
  (∀ d : ℕ, d > 144 → ¬(∃ r₁ r₂ : ℕ, x = d * k + r₁ ∧ 7373 = d * m + r₂)) →
  x = 7361 :=
by sorry

end first_number_proof_l2244_224484


namespace son_age_l2244_224457

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end son_age_l2244_224457


namespace smallest_lcm_with_gcd_5_l2244_224459

theorem smallest_lcm_with_gcd_5 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 5 ∧
    Nat.lcm k l = 203010 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 5 →
      Nat.lcm m n ≥ 203010 :=
by sorry

end smallest_lcm_with_gcd_5_l2244_224459


namespace range_of_a_l2244_224477

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -3 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 3*x + 10}
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = 5 - x}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (B a ∩ C a = C a) → (-2/3 ≤ a ∧ a ≤ 4) :=
by sorry

end range_of_a_l2244_224477


namespace five_dice_not_same_l2244_224438

theorem five_dice_not_same (n : ℕ) (h : n = 8) : 
  (1 - (n : ℚ)/(n^5 : ℚ)) = 4095/4096 := by
  sorry

end five_dice_not_same_l2244_224438


namespace problem_solution_l2244_224443

noncomputable def f (x : ℝ) := x + 1 + abs (3 - x)

theorem problem_solution :
  (∀ x ≥ -1, f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 4) ∧
  (∀ x ≥ -1, f x ≥ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 8 * a * b = a + 2 * b → 2 * a + b ≥ 9/8) :=
by sorry

end problem_solution_l2244_224443


namespace bat_wings_area_is_three_and_half_l2244_224402

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  /-- The width of the rectangle -/
  width : ℝ
  /-- The height of the rectangle -/
  height : ℝ
  /-- The length of segments DC, CB, and BA -/
  segment_length : ℝ
  /-- Width is 3 -/
  width_is_three : width = 3
  /-- Height is 4 -/
  height_is_four : height = 4
  /-- Segment length is 1 -/
  segment_is_one : segment_length = 1

/-- The area of the "bat wings" in the special rectangle -/
def batWingsArea (r : SpecialRectangle) : ℝ := sorry

/-- Theorem stating that the area of the "bat wings" is 3 1/2 -/
theorem bat_wings_area_is_three_and_half (r : SpecialRectangle) :
  batWingsArea r = 3.5 := by sorry

end bat_wings_area_is_three_and_half_l2244_224402


namespace square_plus_inverse_square_value_l2244_224439

theorem square_plus_inverse_square_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  x^2 + 1/x^2 = 7 := by
  sorry

end square_plus_inverse_square_value_l2244_224439


namespace part_a_part_b_l2244_224488

-- Define the main equation
def main_equation (x p : ℝ) : Prop := x^2 + p = -x/4

-- Define the condition for part a
def condition_a (x₁ x₂ : ℝ) : Prop := x₁/x₂ + x₂/x₁ = -9/4

-- Define the condition for part b
def condition_b (x₁ x₂ : ℝ) : Prop := x₂ = x₁^2 - 1

-- Theorem for part a
theorem part_a (x₁ x₂ p : ℝ) :
  main_equation x₁ p ∧ main_equation x₂ p ∧ condition_a x₁ x₂ → p = -1/23 := by
  sorry

-- Theorem for part b
theorem part_b (x₁ x₂ p : ℝ) :
  main_equation x₁ p ∧ main_equation x₂ p ∧ condition_b x₁ x₂ →
  p = -3/8 ∨ p = -15/8 := by
  sorry

end part_a_part_b_l2244_224488


namespace tan_45_degrees_l2244_224463

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l2244_224463


namespace opposite_of_2023_l2244_224453

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end opposite_of_2023_l2244_224453


namespace gcd_8a_plus_3_5a_plus_2_is_1_l2244_224469

theorem gcd_8a_plus_3_5a_plus_2_is_1 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 := by
  sorry

end gcd_8a_plus_3_5a_plus_2_is_1_l2244_224469


namespace algorithm_output_l2244_224413

def algorithm (x y : Int) : (Int × Int) :=
  let x' := if x < 0 then y + 3 else x
  let y' := if x < 0 then y else y - 3
  (x' - y', y' + x')

theorem algorithm_output : algorithm (-5) 15 = (3, 33) := by
  sorry

end algorithm_output_l2244_224413


namespace daisies_per_bouquet_l2244_224407

/-- Represents a flower shop with rose and daisy bouquets -/
structure FlowerShop where
  roses_per_bouquet : ℕ
  total_bouquets : ℕ
  rose_bouquets : ℕ
  daisy_bouquets : ℕ
  total_flowers : ℕ

/-- Theorem stating the number of daisies in each daisy bouquet -/
theorem daisies_per_bouquet (shop : FlowerShop) 
  (h1 : shop.roses_per_bouquet = 12)
  (h2 : shop.total_bouquets = 20)
  (h3 : shop.rose_bouquets = 10)
  (h4 : shop.daisy_bouquets = 10)
  (h5 : shop.total_flowers = 190)
  (h6 : shop.rose_bouquets + shop.daisy_bouquets = shop.total_bouquets) :
  (shop.total_flowers - shop.roses_per_bouquet * shop.rose_bouquets) / shop.daisy_bouquets = 7 := by
  sorry

end daisies_per_bouquet_l2244_224407


namespace specific_pairings_probability_eva_tom_june_leo_probability_l2244_224493

/-- The probability of two specific pairings in a class -/
theorem specific_pairings_probability (n : ℕ) (h : n ≥ 28) :
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 2) = 1 / ((n - 1) * (n - 2)) :=
sorry

/-- The probability of Eva being paired with Tom and June being paired with Leo -/
theorem eva_tom_june_leo_probability :
  (1 : ℚ) / 27 * (1 : ℚ) / 26 = 1 / 702 :=
sorry

end specific_pairings_probability_eva_tom_june_leo_probability_l2244_224493


namespace ratio_is_zero_l2244_224424

-- Define the rectangle JKLM
def Rectangle (J L M : ℝ × ℝ) : Prop :=
  (L.1 - J.1 = 8) ∧ (L.2 - M.2 = 6) ∧ (J.2 = L.2) ∧ (J.1 = M.1)

-- Define points N, P, Q
def PointN (J L N : ℝ × ℝ) : Prop :=
  N.2 = J.2 ∧ L.1 - N.1 = 2

def PointP (L M P : ℝ × ℝ) : Prop :=
  P.1 = L.1 ∧ M.2 - P.2 = 2

def PointQ (M J Q : ℝ × ℝ) : Prop :=
  Q.2 = J.2 ∧ M.1 - Q.1 = 3

-- Define the intersection points R and S
def IntersectionR (J P N Q R : ℝ × ℝ) : Prop :=
  (R.2 - J.2) / (R.1 - J.1) = (P.2 - J.2) / (P.1 - J.1) ∧
  R.2 = N.2

def IntersectionS (J M N Q S : ℝ × ℝ) : Prop :=
  (S.2 - J.2) / (S.1 - J.1) = (M.2 - J.2) / (M.1 - J.1) ∧
  S.2 = N.2

-- Theorem statement
theorem ratio_is_zero
  (J K L M N P Q R S : ℝ × ℝ)
  (h_rect : Rectangle J L M)
  (h_N : PointN J L N)
  (h_P : PointP L M P)
  (h_Q : PointQ M J Q)
  (h_R : IntersectionR J P N Q R)
  (h_S : IntersectionS J M N Q S) :
  (R.1 - S.1) / (N.1 - Q.1) = 0 :=
sorry

end ratio_is_zero_l2244_224424


namespace union_when_m_is_3_union_equals_A_iff_l2244_224464

def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem union_when_m_is_3 : A ∪ B 3 = {x | -3 ≤ x ∧ x ≤ 5} := by sorry

theorem union_equals_A_iff (m : ℝ) : A ∪ B m = A ↔ m ≤ 5/2 := by sorry

end union_when_m_is_3_union_equals_A_iff_l2244_224464


namespace dart_board_partitions_l2244_224400

def partition_count (n : ℕ) (k : ℕ) : ℕ := 
  sorry

theorem dart_board_partitions : partition_count 5 3 = 5 := by
  sorry

end dart_board_partitions_l2244_224400


namespace xixi_cards_problem_l2244_224498

theorem xixi_cards_problem (x y : ℕ) :
  (x + 3 = 3 * (y - 3) ∧ y + 4 = 4 * (x - 4)) ∨
  (x + 3 = 3 * (y - 3) ∧ x + 5 = 5 * (y - 5)) ∨
  (y + 4 = 4 * (x - 4) ∧ x + 5 = 5 * (y - 5)) →
  x = 15 := by
  sorry

end xixi_cards_problem_l2244_224498


namespace equation_solutions_no_solutions_l2244_224473

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem equation_solutions (n k : ℕ) :
  (∃ A : ℕ, A = 7 ∧ factorial n + A * n = n^k) ↔ (n = 2 ∧ k = 4) ∨ (n = 3 ∧ k = 3) :=
sorry

theorem no_solutions (n k : ℕ) :
  ¬(∃ A : ℕ, A = 2012 ∧ factorial n + A * n = n^k) :=
sorry

end equation_solutions_no_solutions_l2244_224473


namespace work_completion_time_l2244_224430

theorem work_completion_time (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 0) :
  1 / a + 1 / b = 1 / t → b = 18 → t = 7.2 → a = 2 := by
  sorry

end work_completion_time_l2244_224430


namespace min_value_theorem_l2244_224420

theorem min_value_theorem (x A B C : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hxA : x^2 + 1/x^2 = A)
  (hxB : x - 1/x = B)
  (hxC : x^3 - 1/x^3 = C) :
  ∃ (m : ℝ), m = 6.4 ∧ ∀ (A' B' C' x' : ℝ), 
    x' > 0 → A' > 0 → B' > 0 → C' > 0 →
    x'^2 + 1/x'^2 = A' →
    x' - 1/x' = B' →
    x'^3 - 1/x'^3 = C' →
    A'^3 / C' ≥ m := by
  sorry

end min_value_theorem_l2244_224420


namespace pentagon_perimeter_eq_sum_of_coefficients_l2244_224444

/-- The perimeter of a pentagon with specified vertices -/
def pentagon_perimeter : ℝ := sorry

/-- Theorem stating that the perimeter of the specified pentagon equals 2 + 2√10 -/
theorem pentagon_perimeter_eq : pentagon_perimeter = 2 + 2 * Real.sqrt 10 := by sorry

/-- Corollary showing that when expressed as p + q√10 + r√13, p + q + r = 4 -/
theorem sum_of_coefficients : ∃ (p q r : ℤ), 
  pentagon_perimeter = ↑p + ↑q * Real.sqrt 10 + ↑r * Real.sqrt 13 ∧ p + q + r = 4 := by sorry

end pentagon_perimeter_eq_sum_of_coefficients_l2244_224444


namespace rational_floor_equality_l2244_224408

theorem rational_floor_equality :
  ∃ (c d : ℤ), d < 100 ∧ d > 0 ∧
    ∀ k : ℕ, k ∈ Finset.range 100 → k > 0 →
      ⌊k * (c : ℚ) / d⌋ = ⌊k * (73 : ℚ) / 100⌋ := by
  sorry

end rational_floor_equality_l2244_224408


namespace M_invertible_iff_square_free_l2244_224461

def M (n : ℕ+) : Matrix (Fin n) (Fin n) ℤ :=
  Matrix.of (fun i j => if (i.val + 1) % j.val = 0 then 1 else 0)

def square_free (m : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → m % (k * k) ≠ 0

theorem M_invertible_iff_square_free (n : ℕ+) :
  IsUnit (M n).det ↔ square_free (n + 1) :=
sorry

end M_invertible_iff_square_free_l2244_224461


namespace parabola_tangent_hyperbola_l2244_224496

theorem parabola_tangent_hyperbola (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 5 ∧ y^2 - m*x^2 = 4 ∧ 
   ∀ x' y' : ℝ, y' = x'^2 + 5 → y'^2 - m*x'^2 ≥ 4) →
  (m = 10 + 2*Real.sqrt 21 ∨ m = 10 - 2*Real.sqrt 21) :=
by sorry

end parabola_tangent_hyperbola_l2244_224496


namespace right_triangle_legs_sum_l2244_224472

theorem right_triangle_legs_sum : ∀ a b : ℕ,
  (a + 1 = b) →                 -- legs are consecutive whole numbers
  (a ^ 2 + b ^ 2 = 41 ^ 2) →    -- Pythagorean theorem with hypotenuse 41
  (a + b = 57) :=               -- sum of legs is 57
by
  sorry

end right_triangle_legs_sum_l2244_224472


namespace marbles_remaining_proof_l2244_224499

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of marbles remaining after sales -/
def remaining_marbles (initial : ℕ) (customers : ℕ) : ℕ :=
  initial - sum_to_n customers

theorem marbles_remaining_proof :
  remaining_marbles 2500 50 = 1225 := by
  sorry

end marbles_remaining_proof_l2244_224499


namespace sisters_contribution_l2244_224416

/-- The amount of money Miranda's sister gave her to buy heels -/
theorem sisters_contribution (months_saved : ℕ) (monthly_savings : ℕ) (total_cost : ℕ) : 
  months_saved = 3 → monthly_savings = 70 → total_cost = 260 →
  total_cost - (months_saved * monthly_savings) = 50 := by
  sorry

end sisters_contribution_l2244_224416


namespace percentage_of_sum_l2244_224446

theorem percentage_of_sum (x y : ℝ) (P : ℝ) 
  (h1 : 0.2 * (x - y) = P / 100 * (x + y))
  (h2 : y = 14.285714285714285 / 100 * x) :
  P = 15 := by
  sorry

end percentage_of_sum_l2244_224446


namespace square_expression_equals_289_l2244_224485

theorem square_expression_equals_289 (x : ℝ) (h : x = 5) : 
  (2 * x + 5 + 2)^2 = 289 := by sorry

end square_expression_equals_289_l2244_224485


namespace largest_even_digit_multiple_of_9_under_1000_l2244_224454

/-- A function that checks if a number has only even digits -/
def hasOnlyEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

/-- The largest positive integer with only even digits that is less than 1000 and is a multiple of 9 -/
def largestEvenDigitMultipleOf9Under1000 : ℕ := 864

theorem largest_even_digit_multiple_of_9_under_1000 :
  (largestEvenDigitMultipleOf9Under1000 < 1000) ∧
  (largestEvenDigitMultipleOf9Under1000 % 9 = 0) ∧
  (hasOnlyEvenDigits largestEvenDigitMultipleOf9Under1000) ∧
  (∀ n : ℕ, n < 1000 → n % 9 = 0 → hasOnlyEvenDigits n → n ≤ largestEvenDigitMultipleOf9Under1000) :=
by sorry

#eval largestEvenDigitMultipleOf9Under1000

end largest_even_digit_multiple_of_9_under_1000_l2244_224454


namespace tomato_land_theorem_l2244_224478

def farmer_problem (total_land : Real) (cleared_percentage : Real) 
                   (grapes_percentage : Real) (potato_percentage : Real) : Real :=
  let cleared_land := total_land * cleared_percentage
  let grapes_land := cleared_land * grapes_percentage
  let potato_land := cleared_land * potato_percentage
  cleared_land - (grapes_land + potato_land)

theorem tomato_land_theorem :
  farmer_problem 3999.9999999999995 0.90 0.60 0.30 = 360 := by
  sorry

end tomato_land_theorem_l2244_224478


namespace remaining_region_area_l2244_224429

/-- Represents a rectangle divided into five regions -/
structure DividedRectangle where
  total_area : ℝ
  region1_area : ℝ
  region2_area : ℝ
  region3_area : ℝ
  region4_area : ℝ
  region5_area : ℝ
  area_sum : total_area = region1_area + region2_area + region3_area + region4_area + region5_area

/-- The theorem stating that one of the remaining regions has an area of 27 square units -/
theorem remaining_region_area (rect : DividedRectangle) 
    (h1 : rect.total_area = 72)
    (h2 : rect.region1_area = 15)
    (h3 : rect.region2_area = 12)
    (h4 : rect.region3_area = 18) :
    rect.region4_area = 27 ∨ rect.region5_area = 27 :=
  sorry

end remaining_region_area_l2244_224429


namespace afternoon_campers_l2244_224475

theorem afternoon_campers (evening_campers : ℕ) (afternoon_evening_difference : ℕ) 
  (h1 : evening_campers = 10)
  (h2 : afternoon_evening_difference = 24) :
  evening_campers + afternoon_evening_difference = 34 := by
  sorry

end afternoon_campers_l2244_224475


namespace cos_equality_proof_l2244_224410

theorem cos_equality_proof (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * π / 180) = Real.cos (317 * π / 180)) : n = 43 := by
  sorry

end cos_equality_proof_l2244_224410


namespace inequality_proof_l2244_224494

theorem inequality_proof (x y z : ℝ) 
  (hx : 2 < x ∧ x < 4) 
  (hy : 2 < y ∧ y < 4) 
  (hz : 2 < z ∧ z < 4) : 
  x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y) > 1 := by
  sorry

end inequality_proof_l2244_224494


namespace max_shareholder_percentage_l2244_224441

theorem max_shareholder_percentage (n : ℕ) (k : ℕ) (p : ℚ) (h1 : n = 100) (h2 : k = 66) (h3 : p = 1/2) :
  ∀ (f : ℕ → ℚ),
    (∀ i, 0 ≤ f i) →
    (∀ i, i < n → f i ≤ 1) →
    (∀ s : Finset ℕ, s.card = k → s.sum f ≥ p) →
    (∀ i, i < n → f i ≤ 1/4) :=
sorry

end max_shareholder_percentage_l2244_224441


namespace independence_test_type_I_error_l2244_224404

/-- Represents the observed value of the χ² statistic -/
def k : ℝ := sorry

/-- Represents the probability of making a Type I error -/
def type_I_error_prob : ℝ → ℝ := sorry

/-- States that as k decreases, the probability of Type I error increases -/
theorem independence_test_type_I_error (h : k₁ < k₂) :
  type_I_error_prob k₁ > type_I_error_prob k₂ := by sorry

end independence_test_type_I_error_l2244_224404
