import Mathlib

namespace NUMINAMATH_CALUDE_worker_c_left_days_l2634_263439

def work_rate (days : ℕ) : ℚ := 1 / days

theorem worker_c_left_days 
  (rate_a rate_b rate_c : ℚ)
  (total_days : ℕ)
  (h1 : rate_a = work_rate 30)
  (h2 : rate_b = work_rate 30)
  (h3 : rate_c = work_rate 40)
  (h4 : total_days = 12)
  : ∃ (x : ℕ), 
    (total_days - x) * (rate_a + rate_b + rate_c) + x * (rate_a + rate_b) = 1 ∧ 
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_c_left_days_l2634_263439


namespace NUMINAMATH_CALUDE_ellipse_equation_l2634_263491

/-- An ellipse with center at the origin, coordinate axes as axes of symmetry,
    and passing through points (√6, 1) and (-√3, -√2) has the equation x²/9 + y²/3 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
    x^2 / m + y^2 / n = 1 ∧
    6 / m + 1 / n = 1 ∧
    3 / m + 2 / n = 1) →
  x^2 / 9 + y^2 / 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2634_263491


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2634_263446

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl A ∩ Set.compl B) = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2634_263446


namespace NUMINAMATH_CALUDE_first_scenario_solution_second_scenario_solution_l2634_263450

/-- Represents the purchase scenarios of a company buying noodles -/
structure NoodlePurchase where
  /-- Total cost in yuan -/
  total_cost : ℕ
  /-- Total number of portions -/
  total_portions : ℕ
  /-- Price of mixed sauce noodles in yuan -/
  mixed_sauce_price : ℕ
  /-- Price of beef noodles in yuan -/
  beef_price : ℕ

/-- Represents the updated purchase scenario -/
structure UpdatedNoodlePurchase where
  /-- Cost of mixed sauce noodles in yuan -/
  mixed_sauce_cost : ℕ
  /-- Cost of beef noodles in yuan -/
  beef_cost : ℕ
  /-- Price difference between beef and mixed sauce noodles in yuan -/
  price_difference : ℕ

/-- Theorem for the first scenario -/
theorem first_scenario_solution (purchase : NoodlePurchase)
  (h1 : purchase.total_cost = 3000)
  (h2 : purchase.total_portions = 170)
  (h3 : purchase.mixed_sauce_price = 15)
  (h4 : purchase.beef_price = 20) :
  ∃ (mixed_sauce beef : ℕ),
    mixed_sauce = 80 ∧
    beef = 90 ∧
    mixed_sauce + beef = purchase.total_portions ∧
    mixed_sauce * purchase.mixed_sauce_price + beef * purchase.beef_price = purchase.total_cost :=
  sorry

/-- Theorem for the second scenario -/
theorem second_scenario_solution (purchase : UpdatedNoodlePurchase)
  (h1 : purchase.mixed_sauce_cost = 1260)
  (h2 : purchase.beef_cost = 1200)
  (h3 : purchase.price_difference = 6) :
  ∃ (beef : ℕ),
    beef = 60 ∧
    (3 * beef : ℚ) / 2 * (purchase.beef_cost / beef - purchase.price_difference) = purchase.mixed_sauce_cost :=
  sorry

end NUMINAMATH_CALUDE_first_scenario_solution_second_scenario_solution_l2634_263450


namespace NUMINAMATH_CALUDE_average_study_time_difference_l2634_263459

def daily_differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]

def days_in_week : ℕ := 7

theorem average_study_time_difference : 
  (daily_differences.sum : ℚ) / days_in_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l2634_263459


namespace NUMINAMATH_CALUDE_min_value_cos_half_theta_times_two_minus_sin_theta_l2634_263403

theorem min_value_cos_half_theta_times_two_minus_sin_theta (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (min : Real), min = 0 ∧ ∀ θ', 0 < θ' ∧ θ' < π →
    min ≤ Real.cos (θ' / 2) * (2 - Real.sin θ') :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_half_theta_times_two_minus_sin_theta_l2634_263403


namespace NUMINAMATH_CALUDE_seat_swapping_arrangements_l2634_263479

def number_of_students : ℕ := 7
def students_to_swap : ℕ := 3

theorem seat_swapping_arrangements :
  (number_of_students.choose students_to_swap) * (students_to_swap.factorial) = 70 := by
  sorry

end NUMINAMATH_CALUDE_seat_swapping_arrangements_l2634_263479


namespace NUMINAMATH_CALUDE_probability_of_selecting_male_student_l2634_263481

theorem probability_of_selecting_male_student 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (h1 : total_students = male_students + female_students)
  (h2 : male_students = 2)
  (h3 : female_students = 3) :
  (male_students : ℚ) / total_students = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_male_student_l2634_263481


namespace NUMINAMATH_CALUDE_pole_intersection_height_l2634_263442

/-- Given two poles with heights 30 and 60 units, placed 50 units apart,
    the height of the intersection of the lines joining the top of each pole
    to the foot of the opposite pole is 20 units. -/
theorem pole_intersection_height :
  let h₁ : ℝ := 30  -- Height of the first pole
  let h₂ : ℝ := 60  -- Height of the second pole
  let d : ℝ := 50   -- Distance between the poles
  let m₁ : ℝ := (0 - h₁) / d  -- Slope of the first line
  let m₂ : ℝ := (0 - h₂) / (-d)  -- Slope of the second line
  let x : ℝ := (h₁ - 0) / (m₂ - m₁)  -- x-coordinate of intersection
  let y : ℝ := m₁ * x + h₁  -- y-coordinate of intersection
  y = 20 := by sorry

end NUMINAMATH_CALUDE_pole_intersection_height_l2634_263442


namespace NUMINAMATH_CALUDE_total_cost_of_supplies_l2634_263480

/-- Calculates the total cost of supplies for a class project -/
theorem total_cost_of_supplies (num_students : ℕ) 
  (bow_cost vinegar_cost baking_soda_cost : ℕ) : 
  num_students = 23 → 
  bow_cost = 5 → 
  vinegar_cost = 2 → 
  baking_soda_cost = 1 → 
  num_students * (bow_cost + vinegar_cost + baking_soda_cost) = 184 := by
  sorry

#check total_cost_of_supplies

end NUMINAMATH_CALUDE_total_cost_of_supplies_l2634_263480


namespace NUMINAMATH_CALUDE_min_values_theorem_l2634_263485

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + 3 = a * b) :
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → a + b ≤ x + y) ∧
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → a^2 + b^2 ≤ x^2 + y^2) ∧
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → 1/a + 1/b ≤ 1/x + 1/y) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + y + 3 = x * y ∧ a + b = x + y ∧ a^2 + b^2 = x^2 + y^2 ∧ 1/a + 1/b = 1/x + 1/y) :=
by
  sorry

#check min_values_theorem

end NUMINAMATH_CALUDE_min_values_theorem_l2634_263485


namespace NUMINAMATH_CALUDE_cost_solution_l2634_263477

/-- The cost of electronic whiteboards and projectors -/
def CostProblem (projector_cost : ℕ) (whiteboard_cost : ℕ) : Prop :=
  (whiteboard_cost = projector_cost + 4000) ∧
  (4 * whiteboard_cost + 3 * projector_cost = 44000)

/-- Theorem stating the correct costs for the whiteboard and projector -/
theorem cost_solution :
  ∃ (projector_cost whiteboard_cost : ℕ),
    CostProblem projector_cost whiteboard_cost ∧
    projector_cost = 4000 ∧
    whiteboard_cost = 8000 := by
  sorry

end NUMINAMATH_CALUDE_cost_solution_l2634_263477


namespace NUMINAMATH_CALUDE_prime_pair_perfect_square_sum_theorem_l2634_263445

/-- A pair of prime numbers (p, q) such that p^2 + 5pq + 4q^2 is a perfect square -/
def PrimePairWithPerfectSquareSum (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ ∃ k : ℕ, p^2 + 5*p*q + 4*q^2 = k^2

/-- The theorem stating that only three specific pairs of prime numbers satisfy the condition -/
theorem prime_pair_perfect_square_sum_theorem :
  ∀ p q : ℕ, PrimePairWithPerfectSquareSum p q ↔ 
    ((p = 13 ∧ q = 3) ∨ (p = 5 ∧ q = 11) ∨ (p = 7 ∧ q = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pair_perfect_square_sum_theorem_l2634_263445


namespace NUMINAMATH_CALUDE_baseball_gear_cost_l2634_263471

theorem baseball_gear_cost (initial_amount : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 67)
  (h2 : remaining_amount = 33) :
  initial_amount - remaining_amount = 34 := by
  sorry

end NUMINAMATH_CALUDE_baseball_gear_cost_l2634_263471


namespace NUMINAMATH_CALUDE_rectangle_triangle_equality_l2634_263420

theorem rectangle_triangle_equality (AB AD DC : ℝ) (h1 : AB = 4) (h2 : AD = 8) (h3 : DC = 4) :
  let ABCD_area := AB * AD
  let DCE_area := (1 / 2) * DC * CE
  let CE := 2 * ABCD_area / DC
  ABCD_area = DCE_area → DE = 4 * Real.sqrt 17 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equality_l2634_263420


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2634_263460

/-- The constant term in the expansion of (1/x + 2x)^6 is 160 -/
theorem constant_term_expansion : ∃ c : ℕ, c = 160 ∧ 
  ∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, (λ x => (1/x + 2*x)^6) = (λ x => c + x * f x)) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2634_263460


namespace NUMINAMATH_CALUDE_f_2x_l2634_263487

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem f_2x (x : ℝ) : f (2*x) = 4*x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_l2634_263487


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l2634_263418

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 3*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-4, 6)

-- Define a function to check if a point lies on a line
def point_on_line (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line x y

-- Define a function to check if a line passes through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  point_on_line x₁ y₁ line ∧ point_on_line x₂ y₂ line

-- Define a function to check if two lines are perpendicular
def perpendicular (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∃ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧
  (∀ x y : ℝ, line1 x y → y = m₁ * x + (P.2 - m₁ * P.1)) ∧
  (∀ x y : ℝ, line2 x y → y = m₂ * x + (P.2 - m₂ * P.1))

-- Theorem statement
theorem intersection_and_perpendicular_lines :
  (point_on_line P.1 P.2 l₁ ∧ point_on_line P.1 P.2 l₂) →
  (∃ line1 : ℝ → ℝ → Prop, line_through_points P.1 P.2 0 0 line1 ∧
    ∀ x y : ℝ, line1 x y ↔ 3*x + 2*y = 0) ∧
  (∃ line2 : ℝ → ℝ → Prop, line_through_points P.1 P.2 P.1 P.2 line2 ∧
    perpendicular line2 l₃ ∧
    ∀ x y : ℝ, line2 x y ↔ 3*x + y + 6 = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l2634_263418


namespace NUMINAMATH_CALUDE_smaller_pyramid_volume_l2634_263467

/-- The volume of a smaller pyramid cut from a right rectangular pyramid -/
theorem smaller_pyramid_volume
  (base_length : ℝ) (base_width : ℝ) (slant_edge : ℝ) (cut_height : ℝ)
  (h_base_length : base_length = 10 * Real.sqrt 2)
  (h_base_width : base_width = 6 * Real.sqrt 2)
  (h_slant_edge : slant_edge = 12)
  (h_cut_height : cut_height = 4) :
  ∃ (volume : ℝ),
    volume = 20 * ((2 * Real.sqrt 19 - 4) / (2 * Real.sqrt 19))^3 * (2 * Real.sqrt 19 - 4) :=
by sorry

end NUMINAMATH_CALUDE_smaller_pyramid_volume_l2634_263467


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2634_263410

theorem unique_solution_quadratic (n : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = n + 3 * x) ↔ n = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2634_263410


namespace NUMINAMATH_CALUDE_new_ratio_is_13_to_7_l2634_263402

/-- Represents the farm's animal count before and after the transaction -/
structure FarmCount where
  initialHorses : ℕ
  initialCows : ℕ
  finalHorses : ℕ
  finalCows : ℕ

/-- Checks if the given FarmCount satisfies the problem conditions -/
def validFarmCount (f : FarmCount) : Prop :=
  f.initialHorses = 4 * f.initialCows ∧
  f.finalHorses = f.initialHorses - 15 ∧
  f.finalCows = f.initialCows + 15 ∧
  f.finalHorses = f.finalCows + 30

/-- Theorem stating that the new ratio of horses to cows is 13:7 -/
theorem new_ratio_is_13_to_7 (f : FarmCount) (h : validFarmCount f) :
  13 * f.finalCows = 7 * f.finalHorses :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_13_to_7_l2634_263402


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2634_263427

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of the coordinates of the other endpoint is 12. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (6 + x) / 2 = 3 →
  (-2 + y) / 2 = 5 →
  x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2634_263427


namespace NUMINAMATH_CALUDE_profit_is_333_l2634_263492

/-- Represents the candy bar sales scenario -/
structure CandyBarSales where
  totalBars : ℕ
  firstBatchCost : ℚ
  secondBatchCost : ℚ
  firstBatchSell : ℚ
  secondBatchSell : ℚ

/-- Calculates the profit from candy bar sales -/
def calculateProfit (sales : CandyBarSales) : ℚ :=
  let costPrice := (800 / 3) + 100
  let sellingPrice := 300 + (600 * 2 / 3)
  sellingPrice - costPrice

/-- Theorem stating that the profit is $333 -/
theorem profit_is_333 (sales : CandyBarSales) 
    (h1 : sales.totalBars = 1200)
    (h2 : sales.firstBatchCost = 1/3)
    (h3 : sales.secondBatchCost = 1/4)
    (h4 : sales.firstBatchSell = 1/2)
    (h5 : sales.secondBatchSell = 2/3) :
  Int.floor (calculateProfit sales) = 333 := by
  sorry

#eval Int.floor (calculateProfit { 
  totalBars := 1200, 
  firstBatchCost := 1/3, 
  secondBatchCost := 1/4, 
  firstBatchSell := 1/2, 
  secondBatchSell := 2/3
})

end NUMINAMATH_CALUDE_profit_is_333_l2634_263492


namespace NUMINAMATH_CALUDE_k_value_for_decreasing_function_l2634_263401

theorem k_value_for_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)
  (h_domain : ∀ x, x ≤ 1 → ∃ y, f x = y)
  (h_inequality : ∀ x : ℝ, f (k - Real.sin x) ≥ f (k^2 - Real.sin x^2))
  : k = -1 :=
sorry

end NUMINAMATH_CALUDE_k_value_for_decreasing_function_l2634_263401


namespace NUMINAMATH_CALUDE_parallelepiped_net_removal_l2634_263452

/-- Represents a parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a parallelepiped -/
structure Net where
  squares : ℕ

/-- Represents the number of possible positions to remove a square from a net -/
def possible_removals (n : Net) : ℕ := sorry

theorem parallelepiped_net_removal 
  (p : Parallelepiped) 
  (n : Net) :
  p.length = 2 ∧ p.width = 1 ∧ p.height = 1 →
  n.squares = 10 →
  possible_removals { squares := n.squares - 1 } = 5 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_net_removal_l2634_263452


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2634_263440

theorem trigonometric_identity (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (Real.sin x)^4 / a^2 + (Real.cos x)^4 / b^2 = 1 / (a^2 + b^2)) :
  (Real.sin x)^100 / a^100 + (Real.cos x)^100 / b^100 = 2 / (a^2 + b^2)^100 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2634_263440


namespace NUMINAMATH_CALUDE_exchange_problem_l2634_263435

def exchange_rate : ℚ := 11 / 8
def spent_amount : ℕ := 70

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem exchange_problem (d : ℕ) :
  (exchange_rate * d : ℚ) - spent_amount = d →
  sum_of_digits d = 10 := by
  sorry

end NUMINAMATH_CALUDE_exchange_problem_l2634_263435


namespace NUMINAMATH_CALUDE_smallest_product_is_zero_l2634_263458

def S : Set Int := {-10, -6, 0, 2, 5}

theorem smallest_product_is_zero :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y = 0 ∧ 
  ∀ (a b : Int), a ∈ S → b ∈ S → x * y ≤ a * b :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_is_zero_l2634_263458


namespace NUMINAMATH_CALUDE_max_value_m_inequality_solution_l2634_263462

theorem max_value_m (a b : ℝ) (h : a ≠ b) :
  (∃ m : ℝ, ∀ M : ℝ, (∀ a b : ℝ, a ≠ b → M * |a - b| ≤ |2*a + b| + |a + 2*b|) → M ≤ m) ∧
  (∀ a b : ℝ, a ≠ b → 1 * |a - b| ≤ |2*a + b| + |a + 2*b|) :=
by sorry

theorem inequality_solution (x : ℝ) :
  |x - 1| < 1 * (2*x + 1) ↔ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_inequality_solution_l2634_263462


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l2634_263495

theorem magical_red_knights_fraction (total : ℕ) (red blue magical : ℕ) :
  total > 0 →
  red + blue = total →
  red = (3 * total) / 8 →
  magical = total / 4 →
  ∃ (p q : ℕ), q > 0 ∧ 
    red * p * 3 * blue = blue * q * 3 * red ∧
    red * p * q + blue * p * q = magical * q * 3 →
  (3 : ℚ) / 7 = p / q := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l2634_263495


namespace NUMINAMATH_CALUDE_jason_seashells_l2634_263409

/-- The number of seashells Jason has now -/
def current_seashells : ℕ := 36

/-- The number of seashells Jason gave away -/
def given_away_seashells : ℕ := 13

/-- The initial number of seashells Jason found -/
def initial_seashells : ℕ := current_seashells + given_away_seashells

theorem jason_seashells : initial_seashells = 49 := by
  sorry

end NUMINAMATH_CALUDE_jason_seashells_l2634_263409


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l2634_263470

theorem infinitely_many_perfect_squares_in_sequence :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ ∃ m : ℕ, ⌊n * Real.sqrt 2⌋ = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l2634_263470


namespace NUMINAMATH_CALUDE_fraction_increase_l2634_263499

theorem fraction_increase (m n a : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : a > 0) :
  (n + a) / (m + a) > n / m := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_l2634_263499


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2634_263448

/-- The area of a quadrilateral with a diagonal of length 40 and offsets 11 and 9 is 400 -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) : 
  diagonal = 40 → offset1 = 11 → offset2 = 9 → 
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 400 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2634_263448


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l2634_263486

/-- Represents the investment and profit structure of a partnership --/
structure Partnership where
  a_investment : ℝ
  b_investment_multiple : ℝ
  annual_gain : ℝ
  a_share : ℝ

/-- Calculates the ratio of B's investment to A's investment --/
def investment_ratio (p : Partnership) : ℝ := p.b_investment_multiple

theorem partnership_investment_ratio (p : Partnership) 
  (h1 : p.annual_gain = 18300)
  (h2 : p.a_share = 6100)
  (h3 : p.a_investment > 0) :
  investment_ratio p = 3 := by
  sorry

#check partnership_investment_ratio

end NUMINAMATH_CALUDE_partnership_investment_ratio_l2634_263486


namespace NUMINAMATH_CALUDE_expression_evaluation_l2634_263400

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -1/2
  2 * (3 * x^3 - x + 3 * y) - (x - 2 * y + 6 * x^3) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2634_263400


namespace NUMINAMATH_CALUDE_seed_packet_combinations_l2634_263475

/-- Represents the cost of a sunflower seed packet -/
def sunflower_cost : ℕ := 4

/-- Represents the cost of a lavender seed packet -/
def lavender_cost : ℕ := 1

/-- Represents the cost of a marigold seed packet -/
def marigold_cost : ℕ := 3

/-- Represents the total budget -/
def total_budget : ℕ := 60

/-- Counts the number of non-negative integer solutions to the equation -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 72 different combinations of seed packets -/
theorem seed_packet_combinations : count_solutions = 72 := by sorry

end NUMINAMATH_CALUDE_seed_packet_combinations_l2634_263475


namespace NUMINAMATH_CALUDE_dans_car_fuel_efficiency_l2634_263490

/-- Represents the fuel efficiency of Dan's car in miles per gallon. -/
def fuel_efficiency : ℝ := 32

/-- The cost of gas in dollars per gallon. -/
def gas_cost : ℝ := 4

/-- The distance Dan's car can travel on $42 of gas, in miles. -/
def distance : ℝ := 336

/-- The amount spent on gas, in dollars. -/
def gas_spent : ℝ := 42

/-- Theorem stating that Dan's car's fuel efficiency is 32 miles per gallon. -/
theorem dans_car_fuel_efficiency :
  fuel_efficiency = distance / (gas_spent / gas_cost) := by
  sorry

end NUMINAMATH_CALUDE_dans_car_fuel_efficiency_l2634_263490


namespace NUMINAMATH_CALUDE_broccoli_area_l2634_263407

theorem broccoli_area (current_production : ℕ) (increase : ℕ) : 
  current_production = 2601 →
  increase = 101 →
  ∃ (previous_side : ℕ) (current_side : ℕ),
    previous_side ^ 2 + increase = current_side ^ 2 ∧
    current_side ^ 2 = current_production ∧
    (current_side ^ 2 : ℚ) / current_production = 1 :=
by sorry

end NUMINAMATH_CALUDE_broccoli_area_l2634_263407


namespace NUMINAMATH_CALUDE_fib_70_mod_10_l2634_263466

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Fibonacci sequence modulo 10 -/
def fibMod10 (n : ℕ) : ℕ := fib n % 10

/-- Period of Fibonacci sequence modulo 10 -/
def fibMod10Period : ℕ := 60

theorem fib_70_mod_10 :
  fibMod10 70 = 5 := by sorry

end NUMINAMATH_CALUDE_fib_70_mod_10_l2634_263466


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2634_263478

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 * a 11 = 3 →
  a 3 + a 13 = 4 →
  (a 15 / a 5 = 1/3 ∨ a 15 / a 5 = 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2634_263478


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2634_263414

def I : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2634_263414


namespace NUMINAMATH_CALUDE_remainder_5031_div_28_l2634_263484

theorem remainder_5031_div_28 : 5031 % 28 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5031_div_28_l2634_263484


namespace NUMINAMATH_CALUDE_rent_calculation_l2634_263483

def monthly_budget (rent : ℚ) : Prop :=
  let food := (3/5) * rent
  let mortgage := 3 * food
  let savings := 2000
  let taxes := (2/5) * savings
  rent + food + mortgage + savings + taxes = 4840

theorem rent_calculation :
  ∃ (rent : ℚ), monthly_budget rent ∧ rent = 600 := by
sorry

end NUMINAMATH_CALUDE_rent_calculation_l2634_263483


namespace NUMINAMATH_CALUDE_A_subset_complement_B_l2634_263421

-- Define the universe set S
def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set A
def A : Finset Char := {'a', 'c'}

-- Define set B
def B : Finset Char := {'b', 'e'}

-- Theorem statement
theorem A_subset_complement_B : A ⊆ S \ B := by sorry

end NUMINAMATH_CALUDE_A_subset_complement_B_l2634_263421


namespace NUMINAMATH_CALUDE_minimum_value_of_a_l2634_263436

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_a_l2634_263436


namespace NUMINAMATH_CALUDE_thirtieth_set_sum_l2634_263415

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of consecutive integers from a to b, inclusive -/
def sum_consecutive (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ :=
  let first := triangular_number (n - 1) + 1
  let last := triangular_number n
  sum_consecutive first last

theorem thirtieth_set_sum : S 30 = 13515 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_set_sum_l2634_263415


namespace NUMINAMATH_CALUDE_tom_marble_pairs_l2634_263422

/-- Represents the set of marbles Tom has -/
structure MarbleSet where
  distinct_colors : Nat  -- Number of distinct colored marbles
  yellow_marbles : Nat   -- Number of identical yellow marbles

/-- Calculates the number of ways to choose 2 marbles from a given MarbleSet -/
def count_marble_pairs (ms : MarbleSet) : Nat :=
  let yellow_pair := if ms.yellow_marbles ≥ 2 then 1 else 0
  let distinct_pairs := Nat.choose ms.distinct_colors 2
  yellow_pair + distinct_pairs

/-- Theorem: Given Tom's marble set, the number of different groups of two marbles is 7 -/
theorem tom_marble_pairs :
  let toms_marbles : MarbleSet := ⟨4, 5⟩
  count_marble_pairs toms_marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_marble_pairs_l2634_263422


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2634_263464

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2634_263464


namespace NUMINAMATH_CALUDE_edith_books_count_edith_books_count_proof_l2634_263496

theorem edith_books_count : ℕ → Prop :=
  fun total : ℕ =>
    ∃ (x y : ℕ),
      x = (120 * 56) / 100 ∧  -- 20% more than 56
      y = (x + 56) / 2 ∧      -- half of total novels
      total = x + 56 + y ∧    -- total books
      total = 185             -- correct answer

-- The proof goes here
theorem edith_books_count_proof : edith_books_count 185 := by
  sorry

end NUMINAMATH_CALUDE_edith_books_count_edith_books_count_proof_l2634_263496


namespace NUMINAMATH_CALUDE_positive_expression_l2634_263424

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max ((a + b + c)^2 - 8*a*c) (max ((a + b + c)^2 - 8*b*c) ((a + b + c)^2 - 8*a*b)) > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l2634_263424


namespace NUMINAMATH_CALUDE_largest_unachievable_sum_l2634_263449

theorem largest_unachievable_sum (a : ℕ) (ha : Odd a) (ha_pos : 0 < a) :
  let n := (a^2 + 5*a + 4) / 2
  (∀ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z → a*x + (a+1)*y + (a+2)*z ≠ n) ∧
  (∀ m : ℕ, n < m → ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ a*x + (a+1)*y + (a+2)*z = m) :=
by sorry

end NUMINAMATH_CALUDE_largest_unachievable_sum_l2634_263449


namespace NUMINAMATH_CALUDE_perimeter_difference_is_one_l2634_263498

-- Define the figures
def figure1_width : ℕ := 4
def figure1_height : ℕ := 2
def figure1_extra_square : ℕ := 1

def figure2_width : ℕ := 6
def figure2_height : ℕ := 2

-- Define the perimeter calculation functions
def perimeter_figure1 (w h e : ℕ) : ℕ :=
  2 * (w + h) + 3 * e

def perimeter_figure2 (w h : ℕ) : ℕ :=
  2 * (w + h)

-- Theorem statement
theorem perimeter_difference_is_one :
  Int.natAbs (perimeter_figure1 figure1_width figure1_height figure1_extra_square -
              perimeter_figure2 figure2_width figure2_height) = 1 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_is_one_l2634_263498


namespace NUMINAMATH_CALUDE_two_distinct_roots_range_l2634_263455

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + (m+3)

-- Define the discriminant of the quadratic function
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m+3)

-- Theorem statement
theorem two_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m < -2 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_range_l2634_263455


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2634_263438

/-- A positive term geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1

/-- The theorem statement -/
theorem geometric_sequence_product (seq : GeometricSequence) 
  (h1 : 2 * (seq.a 1)^2 - 7 * (seq.a 1) + 6 = 0)
  (h2 : 2 * (seq.a 48)^2 - 7 * (seq.a 48) + 6 = 0) :
  seq.a 1 * seq.a 2 * seq.a 25 * seq.a 48 * seq.a 49 = 9 * Real.sqrt 3 := by
  sorry

#check geometric_sequence_product

end NUMINAMATH_CALUDE_geometric_sequence_product_l2634_263438


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l2634_263432

theorem odd_square_minus_one_div_eight (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - 1 = 8*k :=
by sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l2634_263432


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l2634_263433

theorem defective_shipped_percentage
  (total_units : ℝ)
  (defective_rate : ℝ)
  (shipped_rate : ℝ)
  (h1 : defective_rate = 0.07)
  (h2 : shipped_rate = 0.05) :
  (defective_rate * shipped_rate) * 100 = 0.35 := by
sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l2634_263433


namespace NUMINAMATH_CALUDE_hexagon_side_length_squared_l2634_263443

/-- A regular hexagon inscribed in an ellipse -/
structure InscribedHexagon where
  /-- The ellipse equation is x^2 + 9y^2 = 9 -/
  ellipse : ∀ (x y : ℝ), x^2 + 9*y^2 = 9 → ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y
  /-- One vertex of the hexagon is (0,1) -/
  vertex : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One diagonal of the hexagon is aligned along the y-axis -/
  diagonal : ∃ (d : ℝ × ℝ) (e : ℝ × ℝ), d.1 = 0 ∧ e.1 = 0 ∧ d.2 = -e.2
  /-- The hexagon is regular -/
  regular : ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ‖s1 - s2‖ = ‖s2 - s1‖

/-- The square of the length of each side of the hexagon is 729/98 -/
theorem hexagon_side_length_squared (h : InscribedHexagon) : 
  ∃ (s1 s2 : ℝ × ℝ), s1 ≠ s2 ∧ ‖s1 - s2‖^2 = 729/98 :=
sorry

end NUMINAMATH_CALUDE_hexagon_side_length_squared_l2634_263443


namespace NUMINAMATH_CALUDE_james_out_of_pocket_l2634_263494

def initial_purchase : ℝ := 3000
def returned_tv_cost : ℝ := 700
def returned_bike_cost : ℝ := 500
def sold_bike_cost_increase : ℝ := 0.2
def sold_bike_price_ratio : ℝ := 0.8
def toaster_cost : ℝ := 100

theorem james_out_of_pocket :
  let remaining_after_returns := initial_purchase - returned_tv_cost - returned_bike_cost
  let sold_bike_cost := returned_bike_cost * (1 + sold_bike_cost_increase)
  let sold_bike_price := sold_bike_cost * sold_bike_price_ratio
  let final_amount := remaining_after_returns - sold_bike_price + toaster_cost
  final_amount = 1420 := by sorry

end NUMINAMATH_CALUDE_james_out_of_pocket_l2634_263494


namespace NUMINAMATH_CALUDE_distance_specific_point_to_line_l2634_263453

/-- The distance from a point to a line in 3D space --/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The distance from (2, 3, -1) to the line passing through (3, -1, 4) and (5, 0, 1) is √3667/14 --/
theorem distance_specific_point_to_line :
  let point : ℝ × ℝ × ℝ := (2, 3, -1)
  let line_point1 : ℝ × ℝ × ℝ := (3, -1, 4)
  let line_point2 : ℝ × ℝ × ℝ := (5, 0, 1)
  distance_point_to_line point line_point1 line_point2 = Real.sqrt 3667 / 14 := by
  sorry

end NUMINAMATH_CALUDE_distance_specific_point_to_line_l2634_263453


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_prime_condition_l2634_263417

theorem arithmetic_progression_with_prime_condition :
  ∀ (a b c d : ℤ),
  (∃ (k : ℤ), b = a + k ∧ c = b + k ∧ d = c + k) →  -- arithmetic progression
  (∃ (p : ℕ), Nat.Prime p ∧ (d - c + 1 : ℤ) = p) →  -- d - c + 1 is prime
  a + b^2 + c^3 = d^2 * b →                        -- given equation
  (∃ (n : ℤ), a = n ∧ b = n + 1 ∧ c = n + 2 ∧ d = n + 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_prime_condition_l2634_263417


namespace NUMINAMATH_CALUDE_congruence_solution_count_l2634_263437

theorem congruence_solution_count : 
  ∃! (x : ℕ), x > 0 ∧ x < 50 ∧ (x + 20) % 43 = 75 % 43 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_count_l2634_263437


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2634_263489

theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), 
    (∀ k ∈ Finset.range 6, ¬ Nat.Prime (n + k + 1)) ∧ 
    Nat.Prime (n + 7) ∧
    (∀ m < n, ¬(∀ k ∈ Finset.range 6, ¬ Nat.Prime (m + k + 1)) ∨ ¬Nat.Prime (m + 7)) ∧
    n + 7 = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2634_263489


namespace NUMINAMATH_CALUDE_double_area_square_exists_l2634_263454

/-- A point on the grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A square on the grid --/
structure GridSquare where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  d : GridPoint

/-- The area of a grid square --/
def area (s : GridSquare) : ℕ := sorry

/-- A square is legal if its vertices are grid points --/
def is_legal (s : GridSquare) : Prop := sorry

theorem double_area_square_exists (n : ℕ) (h : ∃ s : GridSquare, is_legal s ∧ area s = n) :
  ∃ t : GridSquare, is_legal t ∧ area t = 2 * n := by sorry

end NUMINAMATH_CALUDE_double_area_square_exists_l2634_263454


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2634_263412

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (50 * π / 180) + 
   Real.tan (70 * π / 180) + Real.tan (80 * π / 180)) / 
  Real.sin (30 * π / 180) = 
  2 * (1 / Real.cos (50 * π / 180) + 
       1 / (2 * Real.cos (70 * π / 180) * Real.cos (80 * π / 180))) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2634_263412


namespace NUMINAMATH_CALUDE_product_of_solutions_l2634_263488

theorem product_of_solutions : ∃ (y₁ y₂ : ℝ), 
  (abs y₁ = 3 * (abs y₁ - 2)) ∧ 
  (abs y₂ = 3 * (abs y₂ - 2)) ∧ 
  (y₁ ≠ y₂) ∧ 
  (y₁ * y₂ = -9) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2634_263488


namespace NUMINAMATH_CALUDE_nails_for_smaller_planks_eq_eight_l2634_263423

/-- The number of large planks used for the walls -/
def large_planks : ℕ := 13

/-- The number of nails needed for each large plank -/
def nails_per_plank : ℕ := 17

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := 229

/-- The number of nails needed for smaller planks -/
def nails_for_smaller_planks : ℕ := total_nails - (large_planks * nails_per_plank)

theorem nails_for_smaller_planks_eq_eight :
  nails_for_smaller_planks = 8 := by
  sorry

end NUMINAMATH_CALUDE_nails_for_smaller_planks_eq_eight_l2634_263423


namespace NUMINAMATH_CALUDE_max_b_value_l2634_263431

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := true

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

-- Define the condition for not passing through lattice points
def no_lattice_intersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → is_lattice_point x y → line_equation m x ≠ y

-- State the theorem
theorem max_b_value :
  ∀ b : ℚ, (∀ m : ℚ, 1/3 < m ∧ m < b → no_lattice_intersection m) →
  b ≤ 68/203 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2634_263431


namespace NUMINAMATH_CALUDE_count_right_triangles_with_leg_15_l2634_263476

/-- The number of right triangles with integer side lengths and one leg equal to 15 -/
def rightTrianglesWithLeg15 : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (a, b, c) := t
    a = 15 ∧ a^2 + b^2 = c^2 ∧ a < b ∧ b < c) (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card

/-- Theorem stating that there are exactly 4 right triangles with integer side lengths and one leg equal to 15 -/
theorem count_right_triangles_with_leg_15 : rightTrianglesWithLeg15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_right_triangles_with_leg_15_l2634_263476


namespace NUMINAMATH_CALUDE_real_roots_condition_specific_condition_l2634_263426

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Part 1: Real roots condition
theorem real_roots_condition (m : ℝ) :
  (∃ x : ℝ, quadratic m x = 0) ↔ m ≤ 1/4 :=
sorry

-- Part 2: Specific condition leading to m = -1
theorem specific_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁*x₂ + x₁ + x₂ = 4) →
  m = -1 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_specific_condition_l2634_263426


namespace NUMINAMATH_CALUDE_g_eval_sqrt_half_l2634_263404

noncomputable def g (x : ℝ) : ℝ := Real.arccos (x^2) * Real.arcsin (x^2)

theorem g_eval_sqrt_half : g (1 / Real.sqrt 2) = π^2 / 18 := by
  sorry

end NUMINAMATH_CALUDE_g_eval_sqrt_half_l2634_263404


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2634_263434

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2634_263434


namespace NUMINAMATH_CALUDE_unpainted_cubes_5x5x5_l2634_263468

/-- Given a cube of size n x n x n, where the outer layer is painted,
    calculate the number of unpainted inner cubes. -/
def unpaintedCubes (n : ℕ) : ℕ :=
  (n - 2)^3

/-- The number of unpainted cubes in a 5x5x5 painted cube is 27. -/
theorem unpainted_cubes_5x5x5 :
  unpaintedCubes 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_5x5x5_l2634_263468


namespace NUMINAMATH_CALUDE_girls_in_classroom_l2634_263413

theorem girls_in_classroom (boys : ℕ) (ratio : ℚ) (girls : ℕ) : 
  boys = 20 → ratio = 1/2 → (girls : ℚ) / boys = ratio → girls = 10 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_classroom_l2634_263413


namespace NUMINAMATH_CALUDE_erased_numbers_l2634_263447

def has_digit (n : ℕ) (d : ℕ) : Prop := ∃ k m : ℕ, n = k * 10 + d + m * 10

theorem erased_numbers (remaining_with_one : ℕ) (remaining_with_two : ℕ) (remaining_without_one_or_two : ℕ) :
  remaining_with_one = 20 →
  remaining_with_two = 19 →
  remaining_without_one_or_two = 30 →
  (∀ n : ℕ, n ≤ 100 → (has_digit n 1 ∨ has_digit n 2 ∨ (¬ has_digit n 1 ∧ ¬ has_digit n 2))) →
  100 - (remaining_with_one + remaining_with_two + remaining_without_one_or_two - 2) = 33 := by
  sorry

end NUMINAMATH_CALUDE_erased_numbers_l2634_263447


namespace NUMINAMATH_CALUDE_wall_height_l2634_263472

-- Define the width of the wall
def wall_width : ℝ := 4

-- Define the area of the wall
def wall_area : ℝ := 16

-- Theorem: The height of the wall is 4 feet
theorem wall_height : 
  wall_area / wall_width = 4 :=
by sorry

end NUMINAMATH_CALUDE_wall_height_l2634_263472


namespace NUMINAMATH_CALUDE_triangle_sides_expression_l2634_263463

theorem triangle_sides_expression (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle_ineq_1 : a + b > c)
  (h_triangle_ineq_2 : a + c > b)
  (h_triangle_ineq_3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_expression_l2634_263463


namespace NUMINAMATH_CALUDE_at_least_one_negative_l2634_263406

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1)
  (sum_cd : c + d = 1)
  (product_sum : a * c + b * d > 1) :
  ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l2634_263406


namespace NUMINAMATH_CALUDE_product_14_sum_9_l2634_263493

theorem product_14_sum_9 :
  ∀ a b : ℕ, 
    1 ≤ a ∧ a ≤ 10 →
    1 ≤ b ∧ b ≤ 10 →
    a * b = 14 →
    a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_14_sum_9_l2634_263493


namespace NUMINAMATH_CALUDE_sum_even_factors_720_l2634_263461

def even_factor_sum (n : ℕ) : ℕ := sorry

theorem sum_even_factors_720 : even_factor_sum 720 = 2340 := by sorry

end NUMINAMATH_CALUDE_sum_even_factors_720_l2634_263461


namespace NUMINAMATH_CALUDE_min_sum_a_b_l2634_263444

theorem min_sum_a_b (a b : ℕ+) (h : (20 : ℚ) / 19 = 1 + 1 / (1 + a / b)) :
  ∃ (a' b' : ℕ+), (20 : ℚ) / 19 = 1 + 1 / (1 + a' / b') ∧ a' + b' = 19 ∧ 
  ∀ (c d : ℕ+), (20 : ℚ) / 19 = 1 + 1 / (1 + c / d) → a' + b' ≤ c + d :=
sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l2634_263444


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2634_263405

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ,
  x^50 + x^40 + x^30 + x^20 + x^10 + 1 = 
  (x^5 + x^4 + x^3 + x^2 + x + 1) * q + (-2 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2634_263405


namespace NUMINAMATH_CALUDE_total_cantaloupes_l2634_263425

def fred_cantaloupes : ℕ := 38
def tim_cantaloupes : ℕ := 44
def susan_cantaloupes : ℕ := 57
def nancy_cantaloupes : ℕ := 25

theorem total_cantaloupes : 
  fred_cantaloupes + tim_cantaloupes + susan_cantaloupes + nancy_cantaloupes = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_l2634_263425


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2634_263497

theorem average_of_six_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (subset : ℕ)
  (avg_subset : ℚ)
  (h_total : total = 10)
  (h_avg_all : avg_all = 80)
  (h_subset : subset = 4)
  (h_avg_subset : avg_subset = 113) :
  let remaining := total - subset
  let sum_all := total * avg_all
  let sum_subset := subset * avg_subset
  let sum_remaining := sum_all - sum_subset
  (sum_remaining : ℚ) / remaining = 58 := by sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2634_263497


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l2634_263408

theorem tan_sum_reciprocal (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4)
  (h3 : Real.tan x * Real.tan y = 1/3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l2634_263408


namespace NUMINAMATH_CALUDE_half_AB_equals_neg_two_two_l2634_263474

def OA : Fin 2 → ℝ := ![1, -2]
def OB : Fin 2 → ℝ := ![-3, 2]

theorem half_AB_equals_neg_two_two : 
  (1 / 2 : ℝ) • (OB - OA) = ![(-2 : ℝ), (2 : ℝ)] := by sorry

end NUMINAMATH_CALUDE_half_AB_equals_neg_two_two_l2634_263474


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l2634_263411

theorem factorization_of_polynomial (x : ℝ) : 
  x^4 - 3*x^3 - 28*x^2 = x^2 * (x - 7) * (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l2634_263411


namespace NUMINAMATH_CALUDE_race_distances_l2634_263429

/-- In a 100 m race, if B beats C by 4 m and A beats C by 28 m, then A beats B by 24 m. -/
theorem race_distances (x : ℝ) : 
  (100 : ℝ) - x - 4 = 100 - 28 → x = 24 := by sorry

end NUMINAMATH_CALUDE_race_distances_l2634_263429


namespace NUMINAMATH_CALUDE_unique_phone_number_l2634_263416

def is_valid_phone_number (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000

def first_four (n : ℕ) : ℕ := n / 10000

def last_four (n : ℕ) : ℕ := n % 10000

def first_three (n : ℕ) : ℕ := n / 100000

def last_five (n : ℕ) : ℕ := n % 100000

theorem unique_phone_number :
  ∃! n : ℕ, is_valid_phone_number n ∧
    first_four n + last_four n = 14405 ∧
    first_three n + last_five n = 16970 ∧
    n = 82616144 := by
  sorry

end NUMINAMATH_CALUDE_unique_phone_number_l2634_263416


namespace NUMINAMATH_CALUDE_triangle_ratio_l2634_263482

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: sides form an arithmetic sequence -/
def is_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Condition: C = 2(A + B) -/
def angle_condition (t : Triangle) : Prop :=
  t.C = 2 * (t.A + t.B)

/-- Theorem: If sides form an arithmetic sequence and C = 2(A + B), then b/a = 5/3 -/
theorem triangle_ratio (t : Triangle) 
    (h1 : is_arithmetic_sequence t) 
    (h2 : angle_condition t) : 
    t.b / t.a = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l2634_263482


namespace NUMINAMATH_CALUDE_limit_implies_a_and_b_l2634_263456

/-- Given that the limit of (ln(2-x))^2 / (x^2 + ax + b) as x approaches 1 is equal to 1,
    prove that a = -2 and b = 1. -/
theorem limit_implies_a_and_b (a b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((Real.log (2 - x))^2) / (x^2 + a*x + b) - 1| < ε) →
  a = -2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_limit_implies_a_and_b_l2634_263456


namespace NUMINAMATH_CALUDE_work_completion_equality_prove_men_first_group_l2634_263473

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 30

/-- The number of men in the second group -/
def men_second_group : ℕ := 10

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 36

/-- The number of men in the first group -/
def men_first_group : ℕ := 12

theorem work_completion_equality :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

theorem prove_men_first_group :
  men_first_group = (men_second_group * days_second_group) / days_first_group :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_prove_men_first_group_l2634_263473


namespace NUMINAMATH_CALUDE_num_biology_books_is_14_l2634_263419

/-- The number of ways to choose 2 books from n books -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of chemistry books -/
def num_chemistry_books : ℕ := 8

/-- The total number of ways to choose 2 biology and 2 chemistry books -/
def total_ways : ℕ := 2548

/-- The number of biology books satisfies the given conditions -/
theorem num_biology_books_is_14 : 
  ∃ (n : ℕ), n > 0 ∧ choose_two n * choose_two num_chemistry_books = total_ways ∧ n = 14 :=
sorry

end NUMINAMATH_CALUDE_num_biology_books_is_14_l2634_263419


namespace NUMINAMATH_CALUDE_monomial_combination_l2634_263428

/-- 
Given two monomials that can be combined, this theorem proves 
the values of their exponents.
-/
theorem monomial_combination (m n : ℕ) : 
  (∃ (a b : ℝ), 3 * a^(m+1) * b = -b^(n-1) * a^3) → 
  (m = 2 ∧ n = 2) := by
  sorry

#check monomial_combination

end NUMINAMATH_CALUDE_monomial_combination_l2634_263428


namespace NUMINAMATH_CALUDE_problem_2023_l2634_263465

theorem problem_2023 : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l2634_263465


namespace NUMINAMATH_CALUDE_wang_liang_age_l2634_263457

def is_valid_age (age : ℕ) : Prop :=
  ∃ (birth_year : ℕ),
    (2012 - birth_year = age) ∧
    (age = (birth_year / 1000) + ((birth_year / 100) % 10) + ((birth_year / 10) % 10) + (birth_year % 10))

theorem wang_liang_age :
  (is_valid_age 7 ∨ is_valid_age 25) ∧
  ∀ (age : ℕ), is_valid_age age → (age = 7 ∨ age = 25) :=
sorry

end NUMINAMATH_CALUDE_wang_liang_age_l2634_263457


namespace NUMINAMATH_CALUDE_bowl_score_theorem_l2634_263451

def noa_score : ℕ := 30

def phillip_score (noa : ℕ) : ℕ := 2 * noa

def total_score (noa phillip : ℕ) : ℕ := noa + phillip

theorem bowl_score_theorem : 
  total_score noa_score (phillip_score noa_score) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bowl_score_theorem_l2634_263451


namespace NUMINAMATH_CALUDE_quadratic_root_difference_squares_l2634_263441

theorem quadratic_root_difference_squares (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁^2 - x₂^2 = c^2 / a^2 → 
  b^4 - c^4 = 4 * a^3 * b * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_squares_l2634_263441


namespace NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l2634_263430

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 1)
  (h2 : Complex.abs (w^2 + z^2) = 14) :
  ∃ (min_val : ℝ), min_val = 41/2 ∧ 
    ∀ (w' z' : ℂ), Complex.abs (w' + z') = 1 → Complex.abs (w'^2 + z'^2) = 14 → 
      Complex.abs (w'^3 + z'^3) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l2634_263430


namespace NUMINAMATH_CALUDE_perfect_square_theorem_l2634_263469

theorem perfect_square_theorem (a b c d : ℤ) : 
  d = (a + Real.rpow 2 (1/3 : ℝ) * b + Real.rpow 4 (1/3 : ℝ) * c)^2 → 
  ∃ k : ℤ, d = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_theorem_l2634_263469
