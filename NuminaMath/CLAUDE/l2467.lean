import Mathlib

namespace NUMINAMATH_CALUDE_total_cakes_per_week_l2467_246786

/-- Represents the quantities of cakes served during lunch on a weekday -/
structure LunchCakes :=
  (chocolate : ℕ)
  (vanilla : ℕ)
  (cheesecake : ℕ)

/-- Represents the quantities of cakes served during dinner on a weekday -/
structure DinnerCakes :=
  (chocolate : ℕ)
  (vanilla : ℕ)
  (cheesecake : ℕ)
  (carrot : ℕ)

/-- Calculates the total number of cakes served on a weekday -/
def weekdayTotal (lunch : LunchCakes) (dinner : DinnerCakes) : ℕ :=
  lunch.chocolate + lunch.vanilla + lunch.cheesecake +
  dinner.chocolate + dinner.vanilla + dinner.cheesecake + dinner.carrot

/-- Calculates the total number of cakes served on a weekend day -/
def weekendTotal (lunch : LunchCakes) (dinner : DinnerCakes) : ℕ :=
  2 * (lunch.chocolate + lunch.vanilla + lunch.cheesecake +
       dinner.chocolate + dinner.vanilla + dinner.cheesecake + dinner.carrot)

/-- Theorem: The total number of cakes served during an entire week is 522 -/
theorem total_cakes_per_week
  (lunch : LunchCakes)
  (dinner : DinnerCakes)
  (h1 : lunch.chocolate = 6)
  (h2 : lunch.vanilla = 8)
  (h3 : lunch.cheesecake = 10)
  (h4 : dinner.chocolate = 9)
  (h5 : dinner.vanilla = 7)
  (h6 : dinner.cheesecake = 5)
  (h7 : dinner.carrot = 13) :
  5 * weekdayTotal lunch dinner + 2 * weekendTotal lunch dinner = 522 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_per_week_l2467_246786


namespace NUMINAMATH_CALUDE_calculation_proof_l2467_246772

theorem calculation_proof : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2467_246772


namespace NUMINAMATH_CALUDE_fox_rabbit_bridge_problem_l2467_246712

theorem fox_rabbit_bridge_problem (x : ℝ) : 
  (((2 * ((2 * ((2 * ((2 * x) - 50)) - 50)) - 50)) - 50) = 0) → x = 46.875 := by
  sorry

end NUMINAMATH_CALUDE_fox_rabbit_bridge_problem_l2467_246712


namespace NUMINAMATH_CALUDE_area_cubic_line_theorem_l2467_246764

noncomputable def area_cubic_line (a b c d p q α β : ℝ) : ℝ :=
  |a| / 12 * (β - α)^4

theorem area_cubic_line_theorem (a b c d p q α β : ℝ) 
  (ha : a ≠ 0) 
  (hαβ : α ≠ β) 
  (htouch : ∀ x, a * x^3 + b * x^2 + c * x + d = p * x + q → x = α → 
    (3 * a * x^2 + 2 * b * x + c = p))
  (hintersect : a * β^3 + b * β^2 + c * β + d = p * β + q) :
  area_cubic_line a b c d p q α β = 
    ∫ x in α..β, |a * x^3 + b * x^2 + c * x + d - (p * x + q)| :=
by sorry

end NUMINAMATH_CALUDE_area_cubic_line_theorem_l2467_246764


namespace NUMINAMATH_CALUDE_josh_remaining_money_l2467_246733

/-- Calculates the remaining money after Josh's shopping trip. -/
def remaining_money (initial_amount hat_cost pencil_cost cookie_cost cookie_count : ℚ) : ℚ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * cookie_count)

/-- Theorem stating that Josh has $3 left after his purchases. -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l2467_246733


namespace NUMINAMATH_CALUDE_prob_at_least_one_qualified_prob_merchant_rejects_l2467_246777

-- Define the probability of a product being qualified
def p_qualified : ℝ := 0.8

-- Define the number of products inspected by the company
def n_company_inspect : ℕ := 4

-- Define the total number of products sent to the merchant
def n_total : ℕ := 20

-- Define the number of unqualified products
def n_unqualified : ℕ := 3

-- Define the number of products inspected by the merchant
def n_merchant_inspect : ℕ := 2

-- Theorem for part I
theorem prob_at_least_one_qualified :
  1 - (1 - p_qualified) ^ n_company_inspect = 0.9984 := by sorry

-- Theorem for part II
theorem prob_merchant_rejects :
  (Nat.choose (n_total - n_unqualified) 1 * Nat.choose n_unqualified 1 +
   Nat.choose n_unqualified 2) / Nat.choose n_total 2 = 27 / 95 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_qualified_prob_merchant_rejects_l2467_246777


namespace NUMINAMATH_CALUDE_farmers_children_count_l2467_246751

/-- Represents the problem of determining the number of farmer's children based on apple collection and consumption. -/
theorem farmers_children_count :
  ∀ (n : ℕ),
  (n * 15 - 8 - 7 = 60) →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_farmers_children_count_l2467_246751


namespace NUMINAMATH_CALUDE_stating_trapezoid_equal_area_segment_length_trapezoid_floor_x_squared_div_150_l2467_246737

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  -- Length of the shorter base
  b : ℝ
  -- Height of the trapezoid
  h : ℝ
  -- The segment joining midpoints of legs divides area in 3:4 ratio
  midpoint_divides_area : (2 * b + 75) / (2 * b + 225) = 3 / 4
  -- Length of segment parallel to bases dividing area equally
  x : ℝ
  -- x divides the trapezoid into two equal areas
  x_divides_equally : x * (b + x / 2) = 150

/-- 
Theorem stating that for a trapezoid with the given properties,
the length of the segment dividing it into equal areas is 75.
-/
theorem trapezoid_equal_area_segment_length (t : Trapezoid) : t.x = 75 := by
  sorry

/-- 
Corollary stating that the greatest integer not exceeding x^2/150 is 37.
-/
theorem trapezoid_floor_x_squared_div_150 (t : Trapezoid) : 
  ⌊(t.x^2 / 150 : ℝ)⌋ = 37 := by
  sorry

end NUMINAMATH_CALUDE_stating_trapezoid_equal_area_segment_length_trapezoid_floor_x_squared_div_150_l2467_246737


namespace NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l2467_246752

/-- Represents the cost of fencing for a pentagonal field -/
def fencing_cost (side1 side2 side3 side4 side5 rate_a rate_b rate_c : ℝ) : ℝ × ℝ × ℝ :=
  let perimeter := side1 + side2 + side3 + side4 + side5
  (perimeter * rate_a, perimeter * rate_b, perimeter * rate_c)

/-- Theorem stating the correct fencing costs for the given pentagonal field -/
theorem pentagonal_field_fencing_cost :
  fencing_cost 25 35 40 45 50 3.5 2.25 1.5 = (682.5, 438.75, 292.5) := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l2467_246752


namespace NUMINAMATH_CALUDE_painting_time_theorem_l2467_246708

/-- Represents the time taken to paint a room -/
structure PaintingTime where
  alice : ℝ
  bob : ℝ
  charlie : ℝ

/-- Represents the problem setup -/
structure PaintingProblem where
  time : PaintingTime
  rooms : ℕ
  break_time : ℝ

/-- The equation that should be satisfied by the total time -/
def total_time_equation (t : ℝ) : Prop :=
  (13 / 24) * (t - 2) = 2

/-- The main theorem to be proved -/
theorem painting_time_theorem (p : PaintingProblem) (t : ℝ) 
    (h1 : p.time.alice = 4)
    (h2 : p.time.bob = 6)
    (h3 : p.time.charlie = 8)
    (h4 : p.rooms = 2)
    (h5 : p.break_time = 2)
    (h6 : t ≥ p.break_time) : 
  total_time_equation t :=
sorry

end NUMINAMATH_CALUDE_painting_time_theorem_l2467_246708


namespace NUMINAMATH_CALUDE_solution_x_l2467_246783

theorem solution_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 5) (h2 : y + 1 / x = 7 / 4) :
  x = 4 / 7 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_solution_x_l2467_246783


namespace NUMINAMATH_CALUDE_angle_sum_triangle_l2467_246755

theorem angle_sum_triangle (A B C : ℝ) (h : A + B = 110) : C = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_triangle_l2467_246755


namespace NUMINAMATH_CALUDE_min_value_expression_l2467_246767

theorem min_value_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (|a - 3*b - 2| + |3*a - b|) / Real.sqrt (a^2 + (b + 1)^2) ≥ 2 ∧
  (a = 0 ∧ b = 0 → (|a - 3*b - 2| + |3*a - b|) / Real.sqrt (a^2 + (b + 1)^2) = 2) :=
by sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l2467_246767


namespace NUMINAMATH_CALUDE_residue_mod_14_l2467_246739

theorem residue_mod_14 : (182 * 12 - 15 * 7 + 3) % 14 = 10 := by sorry

end NUMINAMATH_CALUDE_residue_mod_14_l2467_246739


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l2467_246757

theorem sqrt_27_div_sqrt_3_equals_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l2467_246757


namespace NUMINAMATH_CALUDE_sequence_general_term_l2467_246700

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n + 1) →
  (∀ n : ℕ, n ≥ 1 → a n = n^2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2467_246700


namespace NUMINAMATH_CALUDE_lottery_jackpot_probability_l2467_246750

/-- The number of balls for the MegaBall draw -/
def megaBallCount : ℕ := 30

/-- The number of balls for the WinnerBalls draw -/
def winnerBallCount : ℕ := 49

/-- The number of WinnerBalls drawn -/
def winnerBallsDraw : ℕ := 6

/-- The probability of winning the jackpot in the lottery -/
def jackpotProbability : ℚ := 1 / 419514480

/-- Theorem stating that the probability of winning the jackpot in the given lottery system
    is equal to 1/419,514,480 -/
theorem lottery_jackpot_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / (winnerBallCount.choose winnerBallsDraw) = jackpotProbability := by
  sorry


end NUMINAMATH_CALUDE_lottery_jackpot_probability_l2467_246750


namespace NUMINAMATH_CALUDE_largest_n_with_negative_sum_l2467_246717

theorem largest_n_with_negative_sum 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) 
  (h_sum : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) 
  (h_a6_neg : a 6 < 0) 
  (h_a4_a9_pos : a 4 + a 9 > 0) : 
  (∀ n > 11, S n ≥ 0) ∧ S 11 < 0 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_negative_sum_l2467_246717


namespace NUMINAMATH_CALUDE_right_triangle_area_l2467_246771

/-- Given a right triangle with one leg of length a and the ratio of its circumradius
    to inradius being 5:2, its area is either 2a²/3 or 3a²/8 -/
theorem right_triangle_area (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R / r = 5 / 2 ∧
  (∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
   (1/2 * a * b = 2*a^2/3 ∨ 1/2 * a * b = 3*a^2/8)) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_area_l2467_246771


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_range_set_l2467_246722

/-- For a real number a, if ax^2 + ax + a + 3 > 0 for all real x, then a ≥ 0 -/
theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) → a ≥ 0 := by
  sorry

/-- The set of all real numbers a satisfying the quadratic inequality for all x is [0, +∞) -/
theorem quadratic_inequality_range_set : 
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0} = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_range_set_l2467_246722


namespace NUMINAMATH_CALUDE_triangle_altitude_l2467_246724

theorem triangle_altitude (a b : ℝ) (B : ℝ) (h : ℝ) : 
  a = 2 → b = Real.sqrt 7 → B = π / 3 → h = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l2467_246724


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_theorem_l2467_246742

-- Define the ellipse T
def ellipse_T (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola S
def hyperbola_S (x y m n : ℝ) : Prop :=
  x^2 / m^2 - y^2 / n^2 = 1 ∧ m > 0 ∧ n > 0

-- Define the common focus
def common_focus (a b m n : ℝ) : Prop :=
  a^2 - b^2 = m^2 + n^2 ∧ a^2 - b^2 = 4

-- Define the asymptotic line l
def asymptotic_line (x y m n : ℝ) : Prop :=
  y = (n / m) * x

-- Define the symmetry condition
def symmetry_condition (a b m n : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola_S x y m n ∧
  ((x = m^2 - 2 ∧ y = m * n) ∨ (x = 4*b/5 ∧ y = 3*b/5))

-- Main theorem
theorem ellipse_hyperbola_theorem (a b m n : ℝ) :
  ellipse_T 0 b a b ∧
  hyperbola_S 2 0 m n ∧
  common_focus a b m n ∧
  (∃ (x y : ℝ), hyperbola_S x y m n ∧ asymptotic_line x y m n) ∧
  symmetry_condition a b m n →
  a^2 = 5 ∧ b^2 = 4 ∧ m^2 = 4/5 ∧ n^2 = 16/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_theorem_l2467_246742


namespace NUMINAMATH_CALUDE_child_tickets_sold_l2467_246730

/-- Proves the number of child tickets sold in a theater --/
theorem child_tickets_sold (total_tickets : ℕ) (adult_price child_price total_revenue : ℚ) :
  total_tickets = 80 →
  adult_price = 12 →
  child_price = 5 →
  total_revenue = 519 →
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 63 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l2467_246730


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2467_246710

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2467_246710


namespace NUMINAMATH_CALUDE_total_winter_clothing_l2467_246797

-- Define a structure for a box of winter clothing
structure WinterClothingBox where
  scarves : Nat
  mittens : Nat
  hats : Nat

-- Define the contents of each box
def box1 : WinterClothingBox := ⟨2, 3, 1⟩
def box2 : WinterClothingBox := ⟨4, 2, 2⟩
def box3 : WinterClothingBox := ⟨1, 5, 3⟩
def box4 : WinterClothingBox := ⟨3, 4, 1⟩
def box5 : WinterClothingBox := ⟨5, 3, 2⟩
def box6 : WinterClothingBox := ⟨2, 6, 0⟩
def box7 : WinterClothingBox := ⟨4, 1, 3⟩
def box8 : WinterClothingBox := ⟨3, 2, 4⟩
def box9 : WinterClothingBox := ⟨1, 4, 5⟩

-- Define a function to count items in a box
def countItems (box : WinterClothingBox) : Nat :=
  box.scarves + box.mittens + box.hats

-- Theorem statement
theorem total_winter_clothing :
  countItems box1 + countItems box2 + countItems box3 +
  countItems box4 + countItems box5 + countItems box6 +
  countItems box7 + countItems box8 + countItems box9 = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l2467_246797


namespace NUMINAMATH_CALUDE_second_divisor_problem_l2467_246781

theorem second_divisor_problem (x : ℕ) : 
  (210 % 13 = 3) → (210 % x = 7) → (x = 203) :=
by sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l2467_246781


namespace NUMINAMATH_CALUDE_y_is_function_of_x_f_a_is_constant_not_always_injective_not_always_analytic_l2467_246713

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Statement 1: y is a function of x
theorem y_is_function_of_x : ∀ x : ℝ, ∃ y : ℝ, y = f x := by sorry

-- Statement 3: f(a) represents the value of the function f(x) when x = a, which is a constant
theorem f_a_is_constant (a : ℝ) : ∃ k : ℝ, f a = k := by sorry

-- Statement 2 (negation): It is not necessarily true that for different x, the value of y is also different
theorem not_always_injective : ¬ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) := by sorry

-- Statement 4 (negation): It is not always possible to represent f(x) by a specific formula
theorem not_always_analytic : ¬ (∃ formula : ℝ → ℝ, ∀ x : ℝ, f x = formula x) := by sorry

end NUMINAMATH_CALUDE_y_is_function_of_x_f_a_is_constant_not_always_injective_not_always_analytic_l2467_246713


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_divisibility_l2467_246754

theorem prime_sum_of_squares_divisibility (p : ℕ) (h_prime : Prime p) 
  (h_sum : ∃ a : ℕ, 2 * p = a^2 + (a+1)^2 + (a+2)^2 + (a+3)^2) : 
  36 ∣ (p - 7) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_divisibility_l2467_246754


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2467_246773

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 5 ∨ d = 9

def contains_5_and_9 (n : ℕ) : Prop :=
  5 ∈ n.digits 10 ∧ 9 ∈ n.digits 10

theorem smallest_valid_number_last_four_digits :
  ∃ n : ℕ,
    (n > 0) ∧
    (n % 5 = 0) ∧
    (n % 9 = 0) ∧
    is_valid_number n ∧
    contains_5_and_9 n ∧
    (∀ m : ℕ, m > 0 ∧ m % 5 = 0 ∧ m % 9 = 0 ∧ is_valid_number m ∧ contains_5_and_9 m → n ≤ m) ∧
    (n % 10000 = 9995) :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2467_246773


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2467_246705

/-- Given two right triangles with sides 5, 12, and 13, let x be the side length of a square
    inscribed in the first triangle with one vertex at the right angle, and y be the side length
    of a square inscribed in the second triangle with one side on the hypotenuse. -/
def inscribed_squares (x y : ℝ) : Prop :=
  -- First triangle conditions
  5^2 + 12^2 = 13^2 ∧
  x * (12 - x) = 5 * x ∧
  -- Second triangle conditions
  5^2 + 12^2 = 13^2 ∧
  y * (13 - 2*y) = 5 * 12

/-- The ratio of the side lengths of the inscribed squares is 169/220. -/
theorem inscribed_squares_ratio :
  ∀ x y : ℝ, inscribed_squares x y → x / y = 169 / 220 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2467_246705


namespace NUMINAMATH_CALUDE_largest_among_expressions_l2467_246731

theorem largest_among_expressions : 
  let a := -|(-3)|^3
  let b := -(-3)^3
  let c := (-3)^3
  let d := -(3^3)
  (b ≥ a) ∧ (b ≥ c) ∧ (b ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_among_expressions_l2467_246731


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2467_246738

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 4) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 119 :=
by sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2467_246738


namespace NUMINAMATH_CALUDE_reciprocal_sum_relation_l2467_246759

theorem reciprocal_sum_relation (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_relation_l2467_246759


namespace NUMINAMATH_CALUDE_actual_distance_l2467_246714

/-- Calculates the actual distance between two cities given the map distance and scale. -/
theorem actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : 
  map_distance * (scale_miles / scale_distance) = 240 :=
  by
  -- Assuming map_distance = 20, scale_distance = 0.5, and scale_miles = 6
  have h1 : map_distance = 20 := by sorry
  have h2 : scale_distance = 0.5 := by sorry
  have h3 : scale_miles = 6 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_actual_distance_l2467_246714


namespace NUMINAMATH_CALUDE_leila_weekly_earnings_l2467_246793

/-- Represents the earnings of a vlogger over a week -/
def weekly_earnings (daily_viewers : ℕ) (earnings_per_view : ℚ) : ℚ :=
  daily_viewers * earnings_per_view * 7

/-- Proves that Leila earns $350 per week given the conditions -/
theorem leila_weekly_earnings : 
  let voltaire_viewers : ℕ := 50
  let leila_viewers : ℕ := 2 * voltaire_viewers
  let earnings_per_view : ℚ := 1/2
  weekly_earnings leila_viewers earnings_per_view = 350 := by
sorry

end NUMINAMATH_CALUDE_leila_weekly_earnings_l2467_246793


namespace NUMINAMATH_CALUDE_sqrt_750_minus_29_cube_l2467_246727

theorem sqrt_750_minus_29_cube (a b : ℕ+) :
  (Real.sqrt 750 - 29 : ℝ) = (Real.sqrt a.val - b.val : ℝ)^3 →
  a.val + b.val = 28 := by sorry

end NUMINAMATH_CALUDE_sqrt_750_minus_29_cube_l2467_246727


namespace NUMINAMATH_CALUDE_line_through_center_chord_length_l2467_246774

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 11/2

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define a line passing through P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y - point_P.2 = k * (x - point_P.1)

-- Theorem 1: Equation of line passing through P and center of circle
theorem line_through_center : 
  ∃ (x y : ℝ), line_through_P 2 x y ∧ 2*x - y - 2 = 0 := by sorry

-- Theorem 2: Length of chord AB when line slope is 1
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    (circle_C A.1 A.2) ∧ 
    (circle_C B.1 B.2) ∧ 
    (line_through_P 1 A.1 A.2) ∧ 
    (line_through_P 1 B.1 B.2) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 20) := by sorry

end NUMINAMATH_CALUDE_line_through_center_chord_length_l2467_246774


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_given_condition_l2467_246703

-- Define propositions p and q
def p (a x : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

-- Theorem for the first part of the problem
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, p 1 x ∧ q x → x ∈ Set.Ioo 2 3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_given_condition :
  (∀ a x : ℝ, a > 0 → (¬(p a x) → ¬(q x)) ∧ ∃ y : ℝ, ¬(p a y) ∧ q y) →
  ∀ a : ℝ, a ∈ Set.Ioc 1 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_given_condition_l2467_246703


namespace NUMINAMATH_CALUDE_ln_power_equality_l2467_246787

theorem ln_power_equality (x : ℝ) :
  (Real.log (x^4))^2 = (Real.log x)^6 ↔ x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2) :=
sorry

end NUMINAMATH_CALUDE_ln_power_equality_l2467_246787


namespace NUMINAMATH_CALUDE_permutation_equality_l2467_246791

-- Define the permutation function
def A (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

-- State the theorem
theorem permutation_equality (n : ℕ) :
  A (2 * n) ^ 3 = 9 * (A n) ^ 3 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equality_l2467_246791


namespace NUMINAMATH_CALUDE_bin_game_expected_value_l2467_246716

theorem bin_game_expected_value (k : ℕ) : 
  let total_balls : ℕ := 10 + k
  let prob_green : ℚ := 10 / total_balls
  let prob_purple : ℚ := k / total_balls
  let expected_value : ℚ := 3 * prob_green - 1 * prob_purple
  (expected_value = 3/4) → (k = 13) :=
by sorry

end NUMINAMATH_CALUDE_bin_game_expected_value_l2467_246716


namespace NUMINAMATH_CALUDE_circle_equation_min_distance_l2467_246744

theorem circle_equation_min_distance (x y : ℝ) :
  (x^2 + y^2 - 64 = 0) → (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_min_distance_l2467_246744


namespace NUMINAMATH_CALUDE_product_of_numbers_l2467_246736

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 16) : x * y = 836 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2467_246736


namespace NUMINAMATH_CALUDE_invitation_methods_l2467_246792

def total_teachers : ℕ := 10
def invited_teachers : ℕ := 6

theorem invitation_methods (total : ℕ) (invited : ℕ) : 
  total = total_teachers → invited = invited_teachers →
  (Nat.choose total invited) - (Nat.choose (total - 2) (invited - 2)) = 140 := by
  sorry

end NUMINAMATH_CALUDE_invitation_methods_l2467_246792


namespace NUMINAMATH_CALUDE_composition_constant_term_l2467_246775

/-- Given two functions f and g, and a condition on their composition,
    prove that the constant term in the composed function is 14. -/
theorem composition_constant_term
  (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x - 1)
  (hg : ∃ c, ∀ x, g x = 2 * c * x + 3)
  (h_comp : ∃ d, ∀ x, f (g x) = 15 * x + d) :
  ∃ d, (∀ x, f (g x) = 15 * x + d) ∧ d = 14 := by sorry

end NUMINAMATH_CALUDE_composition_constant_term_l2467_246775


namespace NUMINAMATH_CALUDE_tangent_curves_l2467_246701

/-- The value of α for which e^x is tangent to αx^2 -/
theorem tangent_curves (f g : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, g x = α * x^2) →
  (∃ x, f x = g x ∧ deriv f x = deriv g x) →
  α = Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_curves_l2467_246701


namespace NUMINAMATH_CALUDE_sarahs_number_l2467_246749

theorem sarahs_number :
  ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 144 ∣ n ∧ 45 ∣ n ∧ n = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_number_l2467_246749


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l2467_246776

theorem cos_five_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 + α) = 1 / 3) :
  Real.cos (5 * π / 6 + α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l2467_246776


namespace NUMINAMATH_CALUDE_power_2023_preserves_order_l2467_246723

theorem power_2023_preserves_order (a b : ℝ) (h : a > b) : a^2023 > b^2023 := by
  sorry

end NUMINAMATH_CALUDE_power_2023_preserves_order_l2467_246723


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2467_246798

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (5, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 21 :=
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2467_246798


namespace NUMINAMATH_CALUDE_quadratic_function_specific_points_l2467_246788

/-- A quadratic function passing through three specific points has a specific value for 3a - 2b + c -/
theorem quadratic_function_specific_points (a b c : ℤ) : 
  (1^2 * a + 1 * b + c = 6) → 
  ((-1)^2 * a + (-1) * b + c = 4) → 
  (0^2 * a + 0 * b + c = 3) → 
  3*a - 2*b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_specific_points_l2467_246788


namespace NUMINAMATH_CALUDE_product_of_differences_l2467_246794

theorem product_of_differences (m n : ℝ) 
  (hm : m = 1 / (Real.sqrt 3 + Real.sqrt 2)) 
  (hn : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) : 
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l2467_246794


namespace NUMINAMATH_CALUDE_equal_area_point_sum_l2467_246779

def P : ℝ × ℝ := (-4, 3)
def Q : ℝ × ℝ := (7, -5)
def R : ℝ × ℝ := (0, 6)

def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem equal_area_point_sum (S : ℝ × ℝ) :
  S.1 > (min P.1 (min Q.1 R.1)) ∧ 
  S.1 < (max P.1 (max Q.1 R.1)) ∧ 
  S.2 > (min P.2 (min Q.2 R.2)) ∧ 
  S.2 < (max P.2 (max Q.2 R.2)) ∧
  triangle_area P Q S = triangle_area Q R S ∧ 
  triangle_area Q R S = triangle_area R P S →
  10 * S.1 + S.2 = 34/3 := by sorry

end NUMINAMATH_CALUDE_equal_area_point_sum_l2467_246779


namespace NUMINAMATH_CALUDE_rectangle_with_border_area_l2467_246746

/-- Calculates the combined area of a rectangle and its border -/
def combinedArea (length width borderWidth : Real) : Real :=
  (length + 2 * borderWidth) * (width + 2 * borderWidth)

theorem rectangle_with_border_area :
  let length : Real := 0.6
  let width : Real := 0.35
  let borderWidth : Real := 0.05
  combinedArea length width borderWidth = 0.315 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_with_border_area_l2467_246746


namespace NUMINAMATH_CALUDE_f_2012_eq_neg_2_l2467_246790

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2012_eq_neg_2 (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1)))
  (h2 : is_even (λ x => f (x - 1)))
  (h3 : f 0 = 2) :
  f 2012 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2012_eq_neg_2_l2467_246790


namespace NUMINAMATH_CALUDE_orange_balls_count_l2467_246784

theorem orange_balls_count (total : Nat) (red : Nat) (blue : Nat) (pink : Nat) (orange : Nat) : 
  total = 50 →
  red = 20 →
  blue = 10 →
  total = red + blue + pink + orange →
  pink = 3 * orange →
  orange = 5 := by
sorry

end NUMINAMATH_CALUDE_orange_balls_count_l2467_246784


namespace NUMINAMATH_CALUDE_exactly_eighteen_pairs_l2467_246753

/-- Predicate to check if a pair of natural numbers satisfies the given conditions -/
def satisfies_conditions (a b : ℕ) : Prop :=
  (b ∣ (5 * a - 3)) ∧ (a ∣ (5 * b - 1))

/-- The number of pairs of natural numbers satisfying the conditions -/
def number_of_pairs : ℕ := 18

/-- Theorem stating that there are exactly 18 pairs satisfying the conditions -/
theorem exactly_eighteen_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = number_of_pairs ∧
    ∀ (pair : ℕ × ℕ), pair ∈ s ↔ satisfies_conditions pair.1 pair.2) :=
sorry

end NUMINAMATH_CALUDE_exactly_eighteen_pairs_l2467_246753


namespace NUMINAMATH_CALUDE_range_of_a_for_sufficient_not_necessary_l2467_246766

theorem range_of_a_for_sufficient_not_necessary (a : ℝ) : 
  (∀ x : ℝ, x < 1 → x < a) ∧ 
  (∃ x : ℝ, x < a ∧ x ≥ 1) ↔ 
  a > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_sufficient_not_necessary_l2467_246766


namespace NUMINAMATH_CALUDE_special_multiples_count_l2467_246795

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_multiples (n : ℕ) : ℕ :=
  count_multiples n 5 + count_multiples n 6 - count_multiples n 15

theorem special_multiples_count :
  count_special_multiples 3000 = 900 := by sorry

end NUMINAMATH_CALUDE_special_multiples_count_l2467_246795


namespace NUMINAMATH_CALUDE_max_min_value_of_expression_l2467_246734

theorem max_min_value_of_expression (a b : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ Real.sqrt 3) 
  (hb : 1 ≤ b ∧ b ≤ Real.sqrt 3) :
  (∃ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                y ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                (x^2 + y^2 - 1) / (x * y) = 1) ∧
  (∃ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                y ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                (x^2 + y^2 - 1) / (x * y) = Real.sqrt 3) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) → 
                y ∈ Set.Icc 1 (Real.sqrt 3) → 
                1 ≤ (x^2 + y^2 - 1) / (x * y) ∧ 
                (x^2 + y^2 - 1) / (x * y) ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_value_of_expression_l2467_246734


namespace NUMINAMATH_CALUDE_product_of_solutions_l2467_246709

theorem product_of_solutions : ∃ (x₁ x₂ : ℝ), 
  (|5 * x₁| + 4 = 44) ∧ 
  (|5 * x₂| + 4 = 44) ∧ 
  (x₁ ≠ x₂) ∧
  (x₁ * x₂ = -64) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2467_246709


namespace NUMINAMATH_CALUDE_total_time_at_least_5400_seconds_l2467_246735

/-- Represents an observer's record of lap times -/
structure Observer where
  lap_times : List Int
  time_difference : Int

/-- The proposition to be proved -/
theorem total_time_at_least_5400_seconds
  (observer1 observer2 : Observer)
  (h1 : observer1.time_difference = 1)
  (h2 : observer2.time_difference = -1)
  (h3 : observer1.lap_times.length = observer2.lap_times.length)
  (h4 : observer1.lap_times.length ≥ 29) :
  (List.sum observer1.lap_times + List.sum observer2.lap_times) ≥ 5400 :=
sorry

end NUMINAMATH_CALUDE_total_time_at_least_5400_seconds_l2467_246735


namespace NUMINAMATH_CALUDE_ram_original_price_l2467_246745

/-- Represents the price change of RAM due to market conditions --/
def ram_price_change (original_price : ℝ) : Prop :=
  let increased_price := original_price * 1.3
  let final_price := increased_price * 0.8
  final_price = 52

/-- Theorem stating that the original price of RAM was $50 --/
theorem ram_original_price : ∃ (price : ℝ), ram_price_change price ∧ price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ram_original_price_l2467_246745


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2467_246718

theorem geometric_series_sum : 
  let a : ℚ := 1 / 5
  let r : ℚ := -1 / 3
  let n : ℕ := 6
  let series := (Finset.range n).sum (fun i => a * r ^ i)
  series = 182 / 1215 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2467_246718


namespace NUMINAMATH_CALUDE_f_1991_equals_1988_l2467_246789

/-- Represents the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Represents the cumulative sum of digits up to r-digit numbers -/
def g (r : ℕ) : ℕ := r * 10^r - (10^r - 1) / 9

/-- 
f(n) represents the number of digits in the number containing the 10^nth digit 
in the sequence of natural numbers written in order without spaces
-/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(1991) = 1988 -/
theorem f_1991_equals_1988 : f 1991 = 1988 := by sorry

end NUMINAMATH_CALUDE_f_1991_equals_1988_l2467_246789


namespace NUMINAMATH_CALUDE_max_difference_reverse_digits_l2467_246707

theorem max_difference_reverse_digits (q r : ℕ) : 
  (10 ≤ q) ∧ (q < 100) ∧  -- q is a two-digit number
  (10 ≤ r) ∧ (r < 100) ∧  -- r is a two-digit number
  (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ q = 10*x + y ∧ r = 10*y + x) ∧  -- q and r have reversed digits
  (q - r < 30 ∨ r - q < 30) →  -- positive difference is less than 30
  (q - r ≤ 27 ∧ r - q ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_reverse_digits_l2467_246707


namespace NUMINAMATH_CALUDE_alice_age_l2467_246780

/-- Prove that Alice's age is 20 years old given the conditions. -/
theorem alice_age : 
  ∀ (alice_pens : ℕ) (clara_pens : ℕ) (alice_age : ℕ) (clara_age : ℕ),
  alice_pens = 60 →
  clara_pens = (2 * alice_pens) / 5 →
  alice_pens - clara_pens = clara_age - alice_age →
  clara_age > alice_age →
  clara_age + 5 = 61 →
  alice_age = 20 := by
sorry

end NUMINAMATH_CALUDE_alice_age_l2467_246780


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l2467_246728

/-- The perimeter of a T shape formed by two rectangles with given dimensions and overlap -/
theorem t_shape_perimeter (horizontal_width horizontal_height vertical_width vertical_height overlap : ℝ) :
  horizontal_width = 3 →
  horizontal_height = 5 →
  vertical_width = 2 →
  vertical_height = 4 →
  overlap = 1 →
  2 * (horizontal_width + horizontal_height) + 2 * (vertical_width + vertical_height) - 2 * overlap = 26 := by
  sorry

#check t_shape_perimeter

end NUMINAMATH_CALUDE_t_shape_perimeter_l2467_246728


namespace NUMINAMATH_CALUDE_triangle_sides_perfect_square_l2467_246741

theorem triangle_sides_perfect_square
  (a b c : ℤ)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_condition : Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs = 1)
  (int_quotient_1 : ∃ k : ℤ, k * (a + b - c) = a^2 + b^2 - c^2)
  (int_quotient_2 : ∃ k : ℤ, k * (b + c - a) = b^2 + c^2 - a^2)
  (int_quotient_3 : ∃ k : ℤ, k * (c + a - b) = c^2 + a^2 - b^2) :
  ∃ n : ℤ, (a + b - c) * (b + c - a) * (c + a - b) = n^2 ∨
           2 * (a + b - c) * (b + c - a) * (c + a - b) = n^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_perfect_square_l2467_246741


namespace NUMINAMATH_CALUDE_arrange_60402_eq_96_l2467_246719

/-- The number of ways to arrange the digits of 60,402 to form a 5-digit number not beginning with 0 -/
def arrange_60402 : ℕ :=
  let digits : List ℕ := [6, 0, 4, 0, 2]
  let non_zero_digits : List ℕ := digits.filter (· ≠ 0)
  let zero_count : ℕ := digits.count 0
  let total_digits : ℕ := digits.length
  (total_digits - 1) * (non_zero_digits.length - 1).factorial * zero_count.factorial

theorem arrange_60402_eq_96 : arrange_60402 = 96 := by
  sorry

end NUMINAMATH_CALUDE_arrange_60402_eq_96_l2467_246719


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l2467_246768

/-- Calculates the interest rate for the second year given the initial principal,
    first year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_principal : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_principal = 4000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 4368) :
  let first_year_amount := initial_principal * (1 + first_year_rate)
  let second_year_rate := (final_amount / first_year_amount) - 1
  second_year_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_l2467_246768


namespace NUMINAMATH_CALUDE_set_union_problem_l2467_246765

theorem set_union_problem (A B : Set ℝ) (m : ℝ) :
  A = {0, m} →
  B = {0, 2} →
  A ∪ B = {0, 1, 2} →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l2467_246765


namespace NUMINAMATH_CALUDE_runner_lap_time_l2467_246796

/-- Proves that given a 400-meter track, a runner completing 3 laps with the first lap in 70 seconds
    and an average speed of 5 m/s for the entire run, the time for each of the second and third laps
    is 85 seconds. -/
theorem runner_lap_time (track_length : ℝ) (num_laps : ℕ) (first_lap_time : ℝ) (avg_speed : ℝ) :
  track_length = 400 →
  num_laps = 3 →
  first_lap_time = 70 →
  avg_speed = 5 →
  ∃ (second_third_lap_time : ℝ),
    second_third_lap_time = 85 ∧
    (track_length * num_laps) / avg_speed = first_lap_time + 2 * second_third_lap_time :=
by sorry

end NUMINAMATH_CALUDE_runner_lap_time_l2467_246796


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l2467_246711

def solution_set : Set ℝ := {x : ℝ | |x - 2| - |2*x - 1| > 0}

theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l2467_246711


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2467_246720

theorem quadratic_roots_property :
  ∀ x₁ x₂ : ℝ,
  x₁^2 - 3*x₁ - 4 = 0 →
  x₂^2 - 3*x₂ - 4 = 0 →
  x₁ ≠ x₂ →
  x₁*x₂ - x₁ - x₂ = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2467_246720


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2467_246706

theorem triangle_angle_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensures angles are positive
  A + B + C = 180 →        -- Sum of angles in a triangle is 180°
  A = 90 →                 -- Given: Angle A is 90°
  B = 50 →                 -- Given: Angle B is 50°
  C = 40 :=                -- To prove: Angle C is 40°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2467_246706


namespace NUMINAMATH_CALUDE_daily_earnings_l2467_246743

/-- Calculates the daily earnings of a person who works every day, given their earnings over a 4-week period. -/
theorem daily_earnings (total_earnings : ℚ) (h : total_earnings = 1960) : 
  total_earnings / (4 * 7) = 70 := by
  sorry

end NUMINAMATH_CALUDE_daily_earnings_l2467_246743


namespace NUMINAMATH_CALUDE_subtract_negative_three_and_one_l2467_246726

theorem subtract_negative_three_and_one : -3 - 1 = -4 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_three_and_one_l2467_246726


namespace NUMINAMATH_CALUDE_last_digit_322_pow_369_l2467_246704

/-- The last digit of a number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Checks if two numbers have the same last digit -/
def sameLastDigit (a b : ℕ) : Prop := lastDigit a = lastDigit b

theorem last_digit_322_pow_369 : sameLastDigit (322^369) 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_322_pow_369_l2467_246704


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l2467_246762

theorem greatest_prime_factor_of_expression (p : Nat) :
  (p.Prime ∧ p ∣ (2^8 + 5^4 + 10^3) ∧ ∀ q : Nat, q.Prime → q ∣ (2^8 + 5^4 + 10^3) → q ≤ p) ↔ p = 19 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l2467_246762


namespace NUMINAMATH_CALUDE_number_of_true_propositions_l2467_246778

-- Define a type for propositions about polygons
inductive PolygonProposition
  | equalSidesRegular
  | regularCentrallySymmetric
  | hexagonRadiusEqualsSide
  | regularNgonNAxes

-- Function to evaluate the truth of each proposition
def isTrue (p : PolygonProposition) : Bool :=
  match p with
  | PolygonProposition.equalSidesRegular => false
  | PolygonProposition.regularCentrallySymmetric => false
  | PolygonProposition.hexagonRadiusEqualsSide => true
  | PolygonProposition.regularNgonNAxes => true

-- List of all propositions
def allPropositions : List PolygonProposition :=
  [PolygonProposition.equalSidesRegular,
   PolygonProposition.regularCentrallySymmetric,
   PolygonProposition.hexagonRadiusEqualsSide,
   PolygonProposition.regularNgonNAxes]

-- Theorem stating that the number of true propositions is 2
theorem number_of_true_propositions :
  (allPropositions.filter isTrue).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_true_propositions_l2467_246778


namespace NUMINAMATH_CALUDE_function_behavior_l2467_246770

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

-- Define the interval
def interval : Set ℝ := Set.Ioo 2 4

-- Theorem statement
theorem function_behavior (x y : ℝ) (hx : x ∈ interval) (hy : y ∈ interval) :
  (x < 3 ∧ y < 3 → f x > f y) ∧
  (x > 3 ∧ y > 3 → f x < f y) ∧
  (x < 3 ∧ y > 3 → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_function_behavior_l2467_246770


namespace NUMINAMATH_CALUDE_catherine_friends_l2467_246782

/-- The number of friends Catherine gave pens and pencils to -/
def num_friends : ℕ := sorry

/-- The initial number of pens Catherine had -/
def initial_pens : ℕ := 60

/-- The number of pens given to each friend -/
def pens_per_friend : ℕ := 8

/-- The number of pencils given to each friend -/
def pencils_per_friend : ℕ := 6

/-- The total number of pens and pencils left after giving away -/
def items_left : ℕ := 22

theorem catherine_friends :
  (initial_pens * 2 - items_left) / (pens_per_friend + pencils_per_friend) = num_friends :=
sorry

end NUMINAMATH_CALUDE_catherine_friends_l2467_246782


namespace NUMINAMATH_CALUDE_karen_has_32_quarters_l2467_246748

/-- Calculates the number of quarters Karen has given the conditions of the problem -/
def karens_quarters (christopher_quarters : ℕ) (dollar_difference : ℕ) : ℕ :=
  let christopher_value := christopher_quarters * 25  -- Value in cents
  let karen_value := christopher_value - dollar_difference * 100  -- Value in cents
  karen_value / 25  -- Convert back to quarters

/-- Proves that Karen has 32 quarters given the problem conditions -/
theorem karen_has_32_quarters :
  karens_quarters 64 8 = 32 := by sorry

end NUMINAMATH_CALUDE_karen_has_32_quarters_l2467_246748


namespace NUMINAMATH_CALUDE_chord_length_l2467_246702

/-- The length of the chord intercepted by the circle x^2 + y^2 = 4 on the line x - √3y + 2√3 = 0 is 2. -/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 4) → (x - Real.sqrt 3 * y + 2 * Real.sqrt 3 = 0) → 
  ∃ (a b c d : ℝ), (a^2 + b^2 = 4) ∧ (c^2 + d^2 = 4) ∧ 
  (a - Real.sqrt 3 * b + 2 * Real.sqrt 3 = 0) ∧ 
  (c - Real.sqrt 3 * d + 2 * Real.sqrt 3 = 0) ∧ 
  ((a - c)^2 + (b - d)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2467_246702


namespace NUMINAMATH_CALUDE_roys_weight_l2467_246729

/-- Given John's weight and the difference between John and Roy's weights, calculate Roy's weight. -/
theorem roys_weight (john_weight : ℕ) (weight_difference : ℕ) (h1 : john_weight = 81) (h2 : weight_difference = 77) :
  john_weight - weight_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_roys_weight_l2467_246729


namespace NUMINAMATH_CALUDE_six_ring_clock_interval_l2467_246725

/-- A clock that rings a certain number of times per day at equal intervals -/
structure RingingClock where
  rings_per_day : ℕ
  rings_per_day_pos : rings_per_day > 0

/-- The number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Calculate the time interval between rings in minutes -/
def interval_between_rings (clock : RingingClock) : ℚ :=
  minutes_per_day / (clock.rings_per_day - 1)

/-- Theorem: For a clock that rings 6 times a day, the interval between rings is 288 minutes -/
theorem six_ring_clock_interval :
  let clock : RingingClock := ⟨6, by norm_num⟩
  interval_between_rings clock = 288 := by sorry

end NUMINAMATH_CALUDE_six_ring_clock_interval_l2467_246725


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l2467_246763

/-- Given a pizza with 12 slices shared equally among Ron and his 2 friends, 
    prove that each person ate 4 slices. -/
theorem pizza_slices_per_person (total_slices : Nat) (num_friends : Nat) :
  total_slices = 12 →
  num_friends = 2 →
  total_slices / (num_friends + 1) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l2467_246763


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l2467_246758

theorem triangle_area_from_squares (a b : ℝ) (ha : a^2 = 25) (hb : b^2 = 144) : 
  (1/2) * a * b = 30 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l2467_246758


namespace NUMINAMATH_CALUDE_quadratic_roots_constraint_l2467_246740

theorem quadratic_roots_constraint (x y : ℤ) : 
  (∃ α β : ℝ, α^2 + β^2 < 4 ∧ ∀ t : ℝ, t^2 + x*t + y = 0 ↔ t = α ∨ t = β) →
  (x = -2 ∧ y = 1) ∨
  (x = -1 ∧ y = -1) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 0 ∧ y = -1) ∨
  (x = 0 ∧ y = 0) ∨
  (x = 1 ∧ y = 0) ∨
  (x = 1 ∧ y = -1) ∨
  (x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_constraint_l2467_246740


namespace NUMINAMATH_CALUDE_hotdog_count_l2467_246747

theorem hotdog_count (initial : ℕ) (sold : ℕ) (remaining : ℕ) : 
  sold = 2 → remaining = 97 → initial = remaining + sold :=
by sorry

end NUMINAMATH_CALUDE_hotdog_count_l2467_246747


namespace NUMINAMATH_CALUDE_part_one_part_two_l2467_246769

-- Define propositions P and Q
def P (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def Q (x : ℝ) : Prop := |x - 3| ≤ 1

-- Part 1
theorem part_one (x : ℝ) (h : P x 1 ∨ Q x) : 1 < x ∧ x ≤ 4 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(P x a) → ¬(Q x)) 
  (h_not_sufficient : ∃ x, ¬(P x a) ∧ Q x) : 4/3 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2467_246769


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l2467_246761

theorem geometric_progression_proof (a b c d : ℤ) :
  a = 7 ∧ b = -14 ∧ c = 28 ∧ d = -56 →
  (∃ r : ℚ, b = a * r ∧ c = b * r ∧ d = c * r) ∧
  a + d = -49 ∧
  b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l2467_246761


namespace NUMINAMATH_CALUDE_g_equality_l2467_246756

-- Define the function g
def g : ℝ → ℝ := fun x ↦ -4 * x^4 + 2 * x^3 - 5 * x^2 + x + 4

-- State the theorem
theorem g_equality (x : ℝ) : 4 * x^4 + 2 * x^2 - x + g x = 2 * x^3 - 3 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_g_equality_l2467_246756


namespace NUMINAMATH_CALUDE_tangent_circle_radius_double_inscribed_l2467_246785

/-- Given a right triangle ABC with legs a and b, hypotenuse c, inscribed circle radius r,
    circumscribed circle radius R, and a circle with radius ρ touching both legs and the
    circumscribed circle, prove that ρ = 2r. -/
theorem tangent_circle_radius_double_inscribed (a b c r R ρ : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → ρ > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem for right triangle
  R = c / 2 →  -- Radius of circumscribed circle is half the hypotenuse
  r = (a + b - c) / 2 →  -- Formula for inscribed circle radius
  ρ^2 - (a + b - c) * ρ = 0 →  -- Equation derived from tangency conditions
  ρ = 2 * r := by
sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_double_inscribed_l2467_246785


namespace NUMINAMATH_CALUDE_train_speed_problem_l2467_246721

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) (speed_ratio : ℝ) :
  train_length = 150 →
  crossing_time = 12 →
  speed_ratio = 3 →
  let slower_speed := (2 * train_length) / (crossing_time * (speed_ratio + 1))
  let faster_speed := speed_ratio * slower_speed
  faster_speed = 18.75 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2467_246721


namespace NUMINAMATH_CALUDE_wendy_bags_proof_l2467_246799

/-- The number of points Wendy earns per bag of cans recycled -/
def points_per_bag : ℕ := 5

/-- The number of bags Wendy didn't recycle -/
def unrecycled_bags : ℕ := 2

/-- The total points Wendy would earn if she recycled all but 2 bags -/
def total_points : ℕ := 45

/-- The initial number of bags Wendy had -/
def initial_bags : ℕ := 11

theorem wendy_bags_proof :
  points_per_bag * (initial_bags - unrecycled_bags) = total_points :=
by sorry

end NUMINAMATH_CALUDE_wendy_bags_proof_l2467_246799


namespace NUMINAMATH_CALUDE_seven_story_pagoda_top_lanterns_l2467_246760

/-- Represents a pagoda with a given number of stories and lanterns -/
structure Pagoda where
  stories : ℕ
  total_lanterns : ℕ
  lanterns_ratio : ℕ -- ratio of lanterns between adjacent stories

/-- Calculates the number of lanterns on the top story of a pagoda -/
def top_story_lanterns (p : Pagoda) : ℕ :=
  sorry

/-- Theorem: For a 7-story pagoda with a lantern ratio of 2 and 381 total lanterns,
    the number of lanterns on the top story is 3 -/
theorem seven_story_pagoda_top_lanterns :
  let p : Pagoda := { stories := 7, total_lanterns := 381, lanterns_ratio := 2 }
  top_story_lanterns p = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_story_pagoda_top_lanterns_l2467_246760


namespace NUMINAMATH_CALUDE_subtraction_minimizes_l2467_246715

-- Define the set of operators
inductive Operator : Type
  | add : Operator
  | sub : Operator
  | mul : Operator
  | div : Operator

-- Function to apply the operator
def apply_operator (op : Operator) (a b : ℤ) : ℤ :=
  match op with
  | Operator.add => a + b
  | Operator.sub => a - b
  | Operator.mul => a * b
  | Operator.div => a / b

-- Theorem statement
theorem subtraction_minimizes :
  ∀ op : Operator, apply_operator Operator.sub (-3) 1 ≤ apply_operator op (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_minimizes_l2467_246715


namespace NUMINAMATH_CALUDE_final_hair_length_l2467_246732

def hair_length (initial_length cut_length growth_length : ℕ) : ℕ :=
  initial_length - cut_length + growth_length

theorem final_hair_length :
  hair_length 16 11 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_final_hair_length_l2467_246732
