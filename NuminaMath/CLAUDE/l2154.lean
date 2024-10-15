import Mathlib

namespace NUMINAMATH_CALUDE_parametric_to_cartesian_ellipse_parametric_to_cartesian_line_l2154_215461

-- Equation 1
theorem parametric_to_cartesian_ellipse (x y φ : ℝ) :
  x = 5 * Real.cos φ ∧ y = 4 * Real.sin φ ↔ x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Equation 2
theorem parametric_to_cartesian_line (x y t : ℝ) :
  x = 1 - 3 * t^2 ∧ y = 4 * t^2 ↔ 4 * x + 3 * y - 4 = 0 ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_ellipse_parametric_to_cartesian_line_l2154_215461


namespace NUMINAMATH_CALUDE_sum_equals_three_sqrt_fourteen_over_seven_l2154_215431

theorem sum_equals_three_sqrt_fourteen_over_seven
  (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = (3 * Real.sqrt 14) / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_three_sqrt_fourteen_over_seven_l2154_215431


namespace NUMINAMATH_CALUDE_gcd_360_1260_l2154_215485

theorem gcd_360_1260 : Nat.gcd 360 1260 = 180 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_1260_l2154_215485


namespace NUMINAMATH_CALUDE_lewis_earnings_theorem_l2154_215404

/-- Calculates Lewis's earnings per week without overtime during harvest season. -/
def lewis_earnings_without_overtime (weeks : ℕ) (overtime_pay : ℚ) (total_earnings : ℚ) : ℚ :=
  let total_overtime := overtime_pay * weeks
  let earnings_without_overtime := total_earnings - total_overtime
  earnings_without_overtime / weeks

/-- Proves that Lewis's earnings per week without overtime is approximately $27.61. -/
theorem lewis_earnings_theorem (weeks : ℕ) (overtime_pay : ℚ) (total_earnings : ℚ)
    (h1 : weeks = 1091)
    (h2 : overtime_pay = 939)
    (h3 : total_earnings = 1054997) :
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
    |lewis_earnings_without_overtime weeks overtime_pay total_earnings - 27.61| < ε :=
  sorry

end NUMINAMATH_CALUDE_lewis_earnings_theorem_l2154_215404


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2154_215456

/-- A quadratic function f(x) = ax² + bx + 3 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_monotonicity (a b : ℝ) :
  (∀ x ≤ -1, 0 ≤ f' a b x) →
  (∀ x ≥ -1, f' a b x ≤ 0) →
  b = 2 * a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2154_215456


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2154_215480

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 2 ∧ 
  ∀ x y : ℝ, x^2 - y^2 = 1 → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 - y^2 / b^2 = 1 ∧ 
      c^2 = a^2 + b^2 ∧ 
      e = c / a := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2154_215480


namespace NUMINAMATH_CALUDE_blue_ball_count_l2154_215442

/-- Given a bag of glass balls with yellow and blue colors -/
structure GlassBallBag where
  total : ℕ
  yellowProb : ℝ

/-- Theorem: In a bag of 80 glass balls where the probability of picking a yellow ball is 0.25,
    the number of blue balls is 60 -/
theorem blue_ball_count (bag : GlassBallBag)
    (h_total : bag.total = 80)
    (h_yellow_prob : bag.yellowProb = 0.25) :
    (bag.total : ℝ) * (1 - bag.yellowProb) = 60 := by
  sorry


end NUMINAMATH_CALUDE_blue_ball_count_l2154_215442


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2154_215490

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | y > 0}

-- Define set B
def B : Set ℝ := {-2, -1, 1, 2}

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2154_215490


namespace NUMINAMATH_CALUDE_tiffany_cans_problem_l2154_215447

theorem tiffany_cans_problem (monday_bags : ℕ) (next_day_bags : ℕ) : 
  (monday_bags = next_day_bags + 1) → (next_day_bags = 7) → (monday_bags = 8) :=
by sorry

end NUMINAMATH_CALUDE_tiffany_cans_problem_l2154_215447


namespace NUMINAMATH_CALUDE_factor_polynomial_l2154_215457

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2154_215457


namespace NUMINAMATH_CALUDE_inverse_functions_same_monotonicity_function_symmetry_origin_exists_odd_function_without_inverse_l2154_215415

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define inverse functions
def IsInverse (f g : RealFunction) : Prop :=
  ∀ x, g (f x) = x ∧ f (g x) = x

-- Define monotonicity
def IsMonotoneIncreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x < f y

def IsMonotoneDecreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define odd function
def IsOdd (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- Define symmetry with respect to the origin
def IsSymmetricToOrigin (f g : RealFunction) : Prop :=
  ∀ x, g x = -f (-x)

-- Theorem 1: Inverse functions have the same monotonicity
theorem inverse_functions_same_monotonicity (f g : RealFunction) 
  (h : IsInverse f g) : 
  (IsMonotoneIncreasing f ↔ IsMonotoneIncreasing g) ∧ 
  (IsMonotoneDecreasing f ↔ IsMonotoneDecreasing g) := by
  sorry

-- Theorem 2: Function symmetry with respect to the origin
theorem function_symmetry_origin (f : RealFunction) :
  IsSymmetricToOrigin f (λ x => -f (-x)) := by
  sorry

-- Theorem 3: Existence of an odd function without an inverse
theorem exists_odd_function_without_inverse :
  ∃ f : RealFunction, IsOdd f ∧ ¬(∃ g : RealFunction, IsInverse f g) := by
  sorry

end NUMINAMATH_CALUDE_inverse_functions_same_monotonicity_function_symmetry_origin_exists_odd_function_without_inverse_l2154_215415


namespace NUMINAMATH_CALUDE_peters_pizza_fraction_l2154_215432

theorem peters_pizza_fraction (total_slices : ℕ) (peters_solo_slices : ℕ) 
  (shared_with_paul : ℕ) (shared_with_mary : ℕ) : 
  total_slices = 18 → 
  peters_solo_slices = 3 → 
  shared_with_paul = 2 → 
  shared_with_mary = 1 → 
  (peters_solo_slices : ℚ) / total_slices + 
  (shared_with_paul : ℚ) / (2 * total_slices) + 
  (shared_with_mary : ℚ) / (2 * total_slices) = 11 / 36 := by
sorry

end NUMINAMATH_CALUDE_peters_pizza_fraction_l2154_215432


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_value_l2154_215409

theorem arithmetic_sequence_unique_value (a : ℝ) (a_n : ℕ → ℝ) : 
  a > 0 ∧ 
  (∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)) ∧ 
  a_n 1 = a ∧
  (∃! q : ℝ, a_n 2 - a_n 1 = q ∧ (a_n 2 + 2) - (a_n 1 + 1) = q ∧ (a_n 3 + 3) - (a_n 2 + 2) = q) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_value_l2154_215409


namespace NUMINAMATH_CALUDE_time_A_is_120_l2154_215414

/-- The time it takes for B to fill the tank alone (in minutes) -/
def time_B : ℝ := 40

/-- The total time to fill the tank when B is used for half the time and A and B fill it together for the other half (in minutes) -/
def total_time : ℝ := 29.999999999999993

/-- The time it takes for A to fill the tank alone (in minutes) -/
def time_A : ℝ := 120

/-- Theorem stating that the time for A to fill the tank alone is 120 minutes -/
theorem time_A_is_120 : time_A = 120 := by sorry

end NUMINAMATH_CALUDE_time_A_is_120_l2154_215414


namespace NUMINAMATH_CALUDE_fish_weight_l2154_215468

/-- Represents the weight of a fish with its components -/
structure Fish where
  head : ℝ
  body : ℝ
  tail : ℝ

/-- The fish satisfies the given conditions -/
def validFish (f : Fish) : Prop :=
  f.head = f.tail + f.body / 2 ∧
  f.body = f.head + f.tail ∧
  f.tail = 1

/-- The total weight of the fish -/
def totalWeight (f : Fish) : ℝ :=
  f.head + f.body + f.tail

/-- Theorem stating that a valid fish weighs 8 kg -/
theorem fish_weight (f : Fish) (h : validFish f) : totalWeight f = 8 := by
  sorry

#check fish_weight

end NUMINAMATH_CALUDE_fish_weight_l2154_215468


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2154_215493

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- The main theorem -/
theorem arithmetic_sequence_m_value
  (seq : ArithmeticSequence)
  (h1 : seq.S (m - 1) = -2)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 1) = 3)
  : m = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2154_215493


namespace NUMINAMATH_CALUDE_evening_emails_count_l2154_215416

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon and evening combined -/
def afternoon_and_evening_emails : ℕ := 13

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := afternoon_and_evening_emails - afternoon_emails

theorem evening_emails_count : evening_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_evening_emails_count_l2154_215416


namespace NUMINAMATH_CALUDE_power_function_increasing_l2154_215411

/-- A power function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_l2154_215411


namespace NUMINAMATH_CALUDE_paige_team_total_points_l2154_215433

def team_size : ℕ := 5
def paige_points : ℕ := 11
def other_player_points : ℕ := 6

theorem paige_team_total_points :
  (paige_points + (team_size - 1) * other_player_points) = 35 := by
  sorry

end NUMINAMATH_CALUDE_paige_team_total_points_l2154_215433


namespace NUMINAMATH_CALUDE_friend_brought_30_chocolates_l2154_215492

/-- The number of chocolates Nida's friend brought -/
def friend_chocolates (
  initial_chocolates : ℕ)  -- Nida's initial number of chocolates
  (loose_chocolates : ℕ)   -- Number of chocolates not in a box
  (filled_boxes : ℕ)       -- Number of filled boxes initially
  (extra_boxes_needed : ℕ) -- Number of extra boxes needed after friend brings chocolates
  : ℕ :=
  30

/-- Theorem stating that the number of chocolates Nida's friend brought is 30 -/
theorem friend_brought_30_chocolates :
  friend_chocolates 50 5 3 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_friend_brought_30_chocolates_l2154_215492


namespace NUMINAMATH_CALUDE_simplify_expression_l2154_215491

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    ((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 2)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 2)) = a - b * Real.sqrt c ∧
    ¬ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ k > 1 ∧ p ^ k ∣ c.val ∧
    a = 21 ∧ b = 12 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2154_215491


namespace NUMINAMATH_CALUDE_total_snails_is_294_l2154_215427

/-- The total number of snails found by a family of ducks -/
def total_snails : ℕ :=
  let total_ducklings : ℕ := 8
  let first_group_size : ℕ := 3
  let second_group_size : ℕ := 3
  let first_group_snails_per_duckling : ℕ := 5
  let second_group_snails_per_duckling : ℕ := 9
  let first_group_total : ℕ := first_group_size * first_group_snails_per_duckling
  let second_group_total : ℕ := second_group_size * second_group_snails_per_duckling
  let first_two_groups_total : ℕ := first_group_total + second_group_total
  let mother_duck_snails : ℕ := 3 * first_two_groups_total
  let remaining_ducklings : ℕ := total_ducklings - first_group_size - second_group_size
  let remaining_group_total : ℕ := remaining_ducklings * (mother_duck_snails / 2)
  first_group_total + second_group_total + mother_duck_snails + remaining_group_total

theorem total_snails_is_294 : total_snails = 294 := by
  sorry

end NUMINAMATH_CALUDE_total_snails_is_294_l2154_215427


namespace NUMINAMATH_CALUDE_sum_reciprocals_minus_products_l2154_215425

theorem sum_reciprocals_minus_products (a b c : ℚ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_minus_products_l2154_215425


namespace NUMINAMATH_CALUDE_highest_probability_A_l2154_215448

-- Define the sample space
variable (Ω : Type)
-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A, B, and C
variable (A B C : Set Ω)

-- State the theorem
theorem highest_probability_A (hCB : C ⊆ B) (hBA : B ⊆ A) :
  P A ≥ P B ∧ P A ≥ P C := by
  sorry

end NUMINAMATH_CALUDE_highest_probability_A_l2154_215448


namespace NUMINAMATH_CALUDE_max_sum_of_proportional_integers_l2154_215471

theorem max_sum_of_proportional_integers (x y z : ℤ) : 
  (x : ℚ) / 5 = 6 / (y : ℚ) → 
  (x : ℚ) / 5 = (z : ℚ) / 2 → 
  (∃ (a b c : ℤ), x = a ∧ y = b ∧ z = c) →
  (∀ (x' y' z' : ℤ), (x' : ℚ) / 5 = 6 / (y' : ℚ) → (x' : ℚ) / 5 = (z' : ℚ) / 2 → x + y + z ≥ x' + y' + z') →
  x + y + z = 43 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_proportional_integers_l2154_215471


namespace NUMINAMATH_CALUDE_eight_T_three_equals_fifty_l2154_215418

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem statement
theorem eight_T_three_equals_fifty : T 8 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_eight_T_three_equals_fifty_l2154_215418


namespace NUMINAMATH_CALUDE_max_value_is_58_l2154_215481

/-- Represents a type of stone with its weight and value -/
structure Stone where
  weight : ℕ
  value : ℕ

/-- The problem setup -/
def cave_problem :=
  let stone7 : Stone := { weight := 7, value := 16 }
  let stone3 : Stone := { weight := 3, value := 9 }
  let stone2 : Stone := { weight := 2, value := 4 }
  let max_weight : ℕ := 20
  let max_stone7 : ℕ := 2
  (stone7, stone3, stone2, max_weight, max_stone7)

/-- The function to maximize the value of stones -/
def maximize_value (p : Stone × Stone × Stone × ℕ × ℕ) : ℕ :=
  let (stone7, stone3, stone2, max_weight, max_stone7) := p
  sorry -- The actual maximization logic would go here

/-- The theorem stating that the maximum value is 58 -/
theorem max_value_is_58 : maximize_value cave_problem = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_58_l2154_215481


namespace NUMINAMATH_CALUDE_line_point_k_value_l2154_215495

/-- Given a line containing points (3,5), (-1,k), and (-7,2), prove that k = 3.8 -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (line : ℝ → ℝ), line 3 = 5 ∧ line (-1) = k ∧ line (-7) = 2) → k = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l2154_215495


namespace NUMINAMATH_CALUDE_platform_length_l2154_215459

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 20 seconds to cross a signal pole, the length of the platform is 285 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 20) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 285 := by sorry

end NUMINAMATH_CALUDE_platform_length_l2154_215459


namespace NUMINAMATH_CALUDE_fiftieth_islander_is_knight_l2154_215444

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the statement made by an islander about their right neighbor -/
inductive Statement
  | Knight
  | Liar

/-- The number of islanders around the table -/
def n : ℕ := 50

/-- Function that returns the statement made by the islander at a given position -/
def statement (pos : ℕ) : Statement :=
  if pos % 2 = 1 then Statement.Knight else Statement.Liar

/-- Function that determines the actual type of the islander at a given position -/
def islanderType (firstType : IslanderType) (pos : ℕ) : IslanderType :=
  sorry

/-- Theorem stating that the 50th islander must be a knight -/
theorem fiftieth_islander_is_knight (firstType : IslanderType) :
  islanderType firstType n = IslanderType.Knight :=
  sorry

end NUMINAMATH_CALUDE_fiftieth_islander_is_knight_l2154_215444


namespace NUMINAMATH_CALUDE_triangle_shape_l2154_215417

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ a = c ∨ b = c) ∨ (A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2154_215417


namespace NUMINAMATH_CALUDE_research_development_percentage_l2154_215466

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  utilities : ℝ
  equipment : ℝ
  supplies : ℝ
  salaries : ℝ
  research_development : ℝ

/-- The theorem stating that the research and development budget is 9% -/
theorem research_development_percentage
  (budget : BudgetAllocation)
  (h1 : budget.transportation = 15)
  (h2 : budget.utilities = 5)
  (h3 : budget.equipment = 4)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries = 234 / 360 * 100)
  (h6 : budget.transportation + budget.utilities + budget.equipment + budget.supplies + budget.salaries + budget.research_development = 100) :
  budget.research_development = 9 := by
sorry


end NUMINAMATH_CALUDE_research_development_percentage_l2154_215466


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2154_215413

/-- The x-coordinate of a point P on the x-axis that is equidistant from A(-3, 0) and B(0, 5) is 8/3 -/
theorem equidistant_point_x_coordinate : 
  ∃ x : ℝ, 
    (x^2 + 6*x + 9 = x^2 + 25) ∧ 
    (∀ y : ℝ, ((-3 - x)^2 + y^2 = x^2 + (5 - y)^2) → y = 0) ∧
    x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2154_215413


namespace NUMINAMATH_CALUDE_cosine_sine_shift_l2154_215426

theorem cosine_sine_shift :
  let f (x : ℝ) := Real.cos (2 * x + π / 3)
  let g (x : ℝ) := Real.sin (2 * x)
  ∃ (shift : ℝ), shift = 5 * π / 6 ∧
    ∀ (x : ℝ), f x = g (x + shift) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_shift_l2154_215426


namespace NUMINAMATH_CALUDE_magazine_subscription_issues_l2154_215494

/-- Proves that the number of issues in an 18-month magazine subscription is 36,
    given the normal price, promotional discount per issue, and total promotional discount. -/
theorem magazine_subscription_issues
  (normal_price : ℝ)
  (subscription_duration : ℝ)
  (discount_per_issue : ℝ)
  (total_discount : ℝ)
  (h1 : normal_price = 34)
  (h2 : subscription_duration = 18)
  (h3 : discount_per_issue = 0.25)
  (h4 : total_discount = 9) :
  (total_discount / discount_per_issue : ℝ) = 36 := by
sorry

end NUMINAMATH_CALUDE_magazine_subscription_issues_l2154_215494


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2154_215477

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  (f 0 = 0) ∧ (f 2 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2154_215477


namespace NUMINAMATH_CALUDE_product_parity_probabilities_l2154_215465

/-- The probability that the product of two arbitrary natural numbers is even -/
def prob_even_product : ℚ := 3/4

/-- The probability that the product of two arbitrary natural numbers is odd -/
def prob_odd_product : ℚ := 1/4

theorem product_parity_probabilities :
  (prob_even_product + prob_odd_product = 1) ∧
  (prob_even_product = 3/4) ∧
  (prob_odd_product = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_product_parity_probabilities_l2154_215465


namespace NUMINAMATH_CALUDE_function_domain_iff_m_range_l2154_215440

/-- The function f(x) = lg(x^2 - 2mx + m + 2) has domain R if and only if m ∈ (-1, 2) -/
theorem function_domain_iff_m_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*m*x + m + 2)) ↔ m > -1 ∧ m < 2 := by
sorry


end NUMINAMATH_CALUDE_function_domain_iff_m_range_l2154_215440


namespace NUMINAMATH_CALUDE_ellas_food_consumption_l2154_215463

/-- 
Given that:
1. Ella's dog eats 4 times as much food as Ella each day
2. Ella eats 20 pounds of food per day
3. The total food consumption for Ella and her dog over some number of days is 1000 pounds

This theorem proves that the number of days is 10.
-/
theorem ellas_food_consumption (dog_ratio : ℕ) (ella_daily : ℕ) (total_food : ℕ) :
  dog_ratio = 4 →
  ella_daily = 20 →
  total_food = 1000 →
  ∃ (days : ℕ), days = 10 ∧ total_food = (ella_daily + dog_ratio * ella_daily) * days :=
by sorry

end NUMINAMATH_CALUDE_ellas_food_consumption_l2154_215463


namespace NUMINAMATH_CALUDE_scientific_notation_of_116_million_l2154_215482

theorem scientific_notation_of_116_million :
  (116000000 : ℝ) = 1.16 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_116_million_l2154_215482


namespace NUMINAMATH_CALUDE_roses_apples_l2154_215458

/-- Rose's apple distribution problem -/
theorem roses_apples (num_friends : ℕ) (apples_per_friend : ℕ) : 
  num_friends = 3 → apples_per_friend = 3 → num_friends * apples_per_friend = 9 :=
by sorry

end NUMINAMATH_CALUDE_roses_apples_l2154_215458


namespace NUMINAMATH_CALUDE_max_y_value_l2154_215498

theorem max_y_value (x y : ℤ) (h : 3*x*y + 7*x + 6*y = 20) : 
  y ≤ 16 ∧ ∃ (x' y' : ℤ), 3*x'*y' + 7*x' + 6*y' = 20 ∧ y' = 16 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l2154_215498


namespace NUMINAMATH_CALUDE_sequence_bound_l2154_215428

def sequence_rule (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ d : ℕ, d < 10 ∧ 
    (a (2*n) = a (2*n - 1) - d) ∧
    (a (2*n + 1) = a (2*n) + d))

theorem sequence_bound (a : ℕ → ℕ) (h : sequence_rule a) : 
  ∀ n : ℕ, a n ≤ 10 * a 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_bound_l2154_215428


namespace NUMINAMATH_CALUDE_equation_has_six_roots_l2154_215464

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1)^3 / (x^2 * (x - 1)^2)

def is_root (x : ℝ) : Prop := f x = f Real.pi

theorem equation_has_six_roots :
  ∃ (r1 r2 r3 r4 r5 r6 : ℝ),
    r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
    r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
    r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
    r4 ≠ r5 ∧ r4 ≠ r6 ∧
    r5 ≠ r6 ∧
    is_root r1 ∧ is_root r2 ∧ is_root r3 ∧ is_root r4 ∧ is_root r5 ∧ is_root r6 ∧
    ∀ x : ℝ, is_root x → (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4 ∨ x = r5 ∨ x = r6) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_has_six_roots_l2154_215464


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_neg_i_l2154_215421

theorem imaginary_sum_equals_neg_i :
  let i : ℂ := Complex.I
  (1 / i) + (1 / i^3) + (1 / i^5) + (1 / i^7) + (1 / i^9) = -i :=
by sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_neg_i_l2154_215421


namespace NUMINAMATH_CALUDE_slope_product_is_four_l2154_215429

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define points A, B, and C
def point_on_parabola_and_line (p : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line x y

-- Define the vector relation
def vector_relation (p : ℝ) (xA yA xB yB xC yC : ℝ) : Prop :=
  xA + xB = (1/5) * xC ∧ yA + yB = (1/5) * yC

-- Define point M
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 2

-- Theorem statement
theorem slope_product_is_four (p : ℝ) (xA yA xB yB xC yC : ℝ) :
  point_on_parabola_and_line p xA yA →
  point_on_parabola_and_line p xB yB →
  parabola p xC yC →
  vector_relation p xA yA xB yB xC yC →
  point_M 2 2 →
  ((yA - 2) / (xA - 2)) * ((yB - 2) / (xB - 2)) = 4 :=
sorry

end NUMINAMATH_CALUDE_slope_product_is_four_l2154_215429


namespace NUMINAMATH_CALUDE_second_question_percentage_l2154_215488

theorem second_question_percentage
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 75)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 20) :
  ∃ (second_correct : ℝ),
    second_correct = 25 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
sorry

end NUMINAMATH_CALUDE_second_question_percentage_l2154_215488


namespace NUMINAMATH_CALUDE_dvd_cost_is_six_l2154_215438

/-- Represents the DVD production and sales scenario --/
structure DVDProduction where
  movieCost : ℕ
  dailySales : ℕ
  daysPerWeek : ℕ
  weeks : ℕ
  profit : ℕ
  sellingPriceFactor : ℚ

/-- Calculates the production cost of a single DVD --/
def calculateDVDCost (p : DVDProduction) : ℚ :=
  let totalSales := p.dailySales * p.daysPerWeek * p.weeks
  let revenue := p.profit + p.movieCost
  let costPerDVD := revenue / (totalSales * (p.sellingPriceFactor - 1))
  costPerDVD

/-- Theorem stating that the DVD production cost is $6 --/
theorem dvd_cost_is_six (p : DVDProduction) 
  (h1 : p.movieCost = 2000)
  (h2 : p.dailySales = 500)
  (h3 : p.daysPerWeek = 5)
  (h4 : p.weeks = 20)
  (h5 : p.profit = 448000)
  (h6 : p.sellingPriceFactor = 5/2) :
  calculateDVDCost p = 6 := by
  sorry

#eval calculateDVDCost {
  movieCost := 2000,
  dailySales := 500,
  daysPerWeek := 5,
  weeks := 20,
  profit := 448000,
  sellingPriceFactor := 5/2
}

end NUMINAMATH_CALUDE_dvd_cost_is_six_l2154_215438


namespace NUMINAMATH_CALUDE_shopping_tax_theorem_l2154_215483

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) : ℝ :=
  (clothing_percent * clothing_tax_rate + food_percent * food_tax_rate + other_percent * other_tax_rate) * 100

/-- Theorem stating that the total tax percentage is 4.8% given the specified conditions -/
theorem shopping_tax_theorem :
  total_tax_percentage 0.6 0.1 0.3 0.04 0 0.08 = 4.8 := by
  sorry

#eval total_tax_percentage 0.6 0.1 0.3 0.04 0 0.08

end NUMINAMATH_CALUDE_shopping_tax_theorem_l2154_215483


namespace NUMINAMATH_CALUDE_exam_candidates_count_l2154_215469

theorem exam_candidates_count : 
  ∀ (T P F : ℕ) (total_avg passed_avg failed_avg : ℚ),
    P = 100 →
    total_avg = 35 →
    passed_avg = 39 →
    failed_avg = 15 →
    T = P + F →
    (total_avg * T : ℚ) = (passed_avg * P : ℚ) + (failed_avg * F : ℚ) →
    T = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l2154_215469


namespace NUMINAMATH_CALUDE_irrational_approximation_l2154_215400

theorem irrational_approximation (ξ : ℝ) (h_irrational : Irrational ξ) :
  Set.Infinite {q : ℚ | ∃ (m : ℤ) (n : ℕ), q = m / n ∧ |ξ - (m / n)| < 1 / (Real.sqrt 5 * m^2)} := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l2154_215400


namespace NUMINAMATH_CALUDE_ace_in_top_probability_l2154_215478

/-- A standard deck of cards --/
def standard_deck : ℕ := 52

/-- The number of top cards we're considering --/
def top_cards : ℕ := 3

/-- The probability of the Ace of Spades being among the top cards --/
def prob_ace_in_top : ℚ := 3 / 52

theorem ace_in_top_probability :
  prob_ace_in_top = top_cards / standard_deck :=
by sorry

end NUMINAMATH_CALUDE_ace_in_top_probability_l2154_215478


namespace NUMINAMATH_CALUDE_tricycle_count_l2154_215446

theorem tricycle_count (num_bicycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 24 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 90 →
  ∃ num_tricycles : ℕ, num_tricycles = 14 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l2154_215446


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_subset_complement_iff_m_range_l2154_215476

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 3 ≤ x ∧ x ≤ m}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) → m = 5 := by
  sorry

-- Theorem 2
theorem subset_complement_iff_m_range :
  ∀ m : ℝ, A ⊆ (Set.univ \ B m) ↔ m < -2 ∨ m > 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_subset_complement_iff_m_range_l2154_215476


namespace NUMINAMATH_CALUDE_x_value_l2154_215402

theorem x_value : Real.sqrt (20 - 17 - 2 * 0 - 1 + 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2154_215402


namespace NUMINAMATH_CALUDE_area_relationship_l2154_215450

/-- Two congruent isosceles right-angled triangles with inscribed squares -/
structure TriangleWithSquare where
  /-- The side length of the triangle -/
  side : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The inscribed square's side is less than the triangle's side -/
  h_square_fits : square_side < side

/-- The theorem stating the relationship between the areas of squares P and R -/
theorem area_relationship (t : TriangleWithSquare) (h_area_p : t.square_side ^ 2 = 45) :
  ∃ (r : ℝ), r ^ 2 = 40 ∧ ∃ (t' : TriangleWithSquare), t'.square_side ^ 2 = r ^ 2 :=
sorry

end NUMINAMATH_CALUDE_area_relationship_l2154_215450


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2154_215410

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 16 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2154_215410


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2154_215499

def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : inversely_proportional x₁ y₁)
  (h2 : inversely_proportional x₂ y₂)
  (h3 : x₁ = 5)
  (h4 : y₁ = 15)
  (h5 : y₂ = 30) :
  x₂ = 5/2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2154_215499


namespace NUMINAMATH_CALUDE_closest_fraction_l2154_215423

def medals_won : ℚ := 17 / 100

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧
    ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
    f = 1/6 :=
  sorry

end NUMINAMATH_CALUDE_closest_fraction_l2154_215423


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2154_215454

/-- Given two vectors AB and CD in R², where AB is perpendicular to CD,
    prove that the y-coordinate of AB is 1. -/
theorem perpendicular_vectors (x : ℝ) : 
  let AB : ℝ × ℝ := (3, x)
  let CD : ℝ × ℝ := (-2, 6)
  (AB.1 * CD.1 + AB.2 * CD.2 = 0) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2154_215454


namespace NUMINAMATH_CALUDE_specific_building_height_l2154_215439

/-- Calculates the height of a building with specific floor heights -/
def building_height (total_stories : ℕ) (base_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := total_stories / 2
  let second_half := total_stories - first_half
  (first_half * base_height) + (second_half * (base_height + height_increase))

/-- Theorem stating the height of the specific building described in the problem -/
theorem specific_building_height :
  building_height 20 12 3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_specific_building_height_l2154_215439


namespace NUMINAMATH_CALUDE_product_digit_range_l2154_215403

theorem product_digit_range : 
  ∀ (a b : ℕ), 
    1 ≤ a ∧ a ≤ 9 → 
    100 ≤ b ∧ b ≤ 999 → 
    (100 ≤ a * b ∧ a * b ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_product_digit_range_l2154_215403


namespace NUMINAMATH_CALUDE_vector_angle_cosine_l2154_215412

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_angle_cosine (a b c : V) :
  a + b + c = 0 →
  ‖a‖ = 2 →
  ‖b‖ = 3 →
  ‖c‖ = 4 →
  ‖a‖ < ‖b‖ →
  inner a b / (‖a‖ * ‖b‖) = 1/4 := by sorry

end NUMINAMATH_CALUDE_vector_angle_cosine_l2154_215412


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2154_215484

theorem multiplication_subtraction_difference : ∃ (x : ℝ), x = 10 ∧ (3 * x) - (26 - x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2154_215484


namespace NUMINAMATH_CALUDE_larger_number_proof_l2154_215405

theorem larger_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  Nat.gcd a b = 60 → Nat.lcm a b = 9900 → max a b = 900 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2154_215405


namespace NUMINAMATH_CALUDE_alternating_square_sum_equals_5304_l2154_215467

def alternatingSquareSum (n : ℕ) : ℤ :=
  let seq := List.range n |> List.reverse |> List.map (λ i => (101 - i : ℤ)^2)
  seq.enum.foldl (λ acc (i, x) => acc + (if i % 4 < 2 then x else -x)) 0

theorem alternating_square_sum_equals_5304 :
  alternatingSquareSum 100 = 5304 := by
  sorry

end NUMINAMATH_CALUDE_alternating_square_sum_equals_5304_l2154_215467


namespace NUMINAMATH_CALUDE_mean_homeruns_is_12_08_l2154_215473

def total_hitters : ℕ := 12

def april_homeruns : List (ℕ × ℕ) := [(5, 4), (6, 4), (8, 2), (10, 1)]
def may_homeruns : List (ℕ × ℕ) := [(5, 2), (6, 2), (8, 3), (10, 2), (11, 1)]

def total_homeruns : ℕ := 
  (april_homeruns.map (λ p => p.1 * p.2)).sum + 
  (may_homeruns.map (λ p => p.1 * p.2)).sum

theorem mean_homeruns_is_12_08 : 
  (total_homeruns : ℚ) / total_hitters = 12.08 := by sorry

end NUMINAMATH_CALUDE_mean_homeruns_is_12_08_l2154_215473


namespace NUMINAMATH_CALUDE_matrix_power_result_l2154_215401

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec (![7, -3]) = ![-14, 6]) :
  (B^4).mulVec (![7, -3]) = ![112, -48] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_result_l2154_215401


namespace NUMINAMATH_CALUDE_expression_evaluation_l2154_215406

theorem expression_evaluation : 6 * 199 + 4 * 199 + 3 * 199 + 199 + 100 = 2886 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2154_215406


namespace NUMINAMATH_CALUDE_inequality_linear_iff_k_eq_two_l2154_215489

/-- The inequality (k+2)x^(|k|-1) + 5 < 0 is linear in x if and only if k = 2 -/
theorem inequality_linear_iff_k_eq_two (k : ℝ) : 
  (∃ (a b : ℝ), ∀ x, ((k + 2) * x^(|k| - 1) + 5 < 0) ↔ (a * x + b < 0)) ↔ k = 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_linear_iff_k_eq_two_l2154_215489


namespace NUMINAMATH_CALUDE_octagon_area_error_l2154_215422

theorem octagon_area_error (L : ℝ) (h : L > 0) : 
  let measured_length := 1.1 * L
  let true_area := 2 * (1 + Real.sqrt 2) * L^2 / 4
  let estimated_area := 2 * (1 + Real.sqrt 2) * measured_length^2 / 4
  (estimated_area - true_area) / true_area * 100 = 21 := by sorry

end NUMINAMATH_CALUDE_octagon_area_error_l2154_215422


namespace NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_perpendicular_lines_to_plane_implies_parallel_lines_l2154_215407

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those two planes are parallel
theorem perpendicular_line_implies_parallel_planes
  (m : Line) (α β : Plane) :
  perpendicular_line_plane m α →
  perpendicular_line_plane m β →
  parallel_plane_plane α β :=
sorry

-- Theorem 2: If two lines are both perpendicular to the same plane, then those two lines are parallel
theorem perpendicular_lines_to_plane_implies_parallel_lines
  (m n : Line) (α : Plane) :
  perpendicular_line_plane m α →
  perpendicular_line_plane n α →
  parallel_line_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_perpendicular_lines_to_plane_implies_parallel_lines_l2154_215407


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2154_215437

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) : 
  Real.tan (α + π / 4) = -1 ∨ Real.tan (α + π / 4) = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2154_215437


namespace NUMINAMATH_CALUDE_power_functions_inequality_l2154_215408

theorem power_functions_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) :
  (x₁ + x₂)^2 / 4 < (x₁^2 + x₂^2) / 2 ∧
  2 / (x₁ + x₂) < (1 / x₁ + 1 / x₂) / 2 :=
by sorry

end NUMINAMATH_CALUDE_power_functions_inequality_l2154_215408


namespace NUMINAMATH_CALUDE_sum_753_326_base8_l2154_215487

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent. -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits. -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

/-- The theorem stating that the sum of 753₈ and 326₈ in base 8 is 1301₈. -/
theorem sum_753_326_base8 :
  decimalToBase8 (base8ToDecimal [7, 5, 3] + base8ToDecimal [3, 2, 6]) = [1, 3, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_753_326_base8_l2154_215487


namespace NUMINAMATH_CALUDE_triangle_inscription_exists_l2154_215436

-- Define the triangle type
structure Triangle :=
  (A B C : Point)

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define inscribed triangle
def inscribed (inner outer : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_inscription_exists (ABC : Triangle) :
  ∃ (PQR : Triangle), ∃ (XYZ : Triangle),
    congruent XYZ ABC ∧ inscribed XYZ PQR := by sorry

end NUMINAMATH_CALUDE_triangle_inscription_exists_l2154_215436


namespace NUMINAMATH_CALUDE_square_difference_l2154_215496

theorem square_difference : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by sorry

end NUMINAMATH_CALUDE_square_difference_l2154_215496


namespace NUMINAMATH_CALUDE_escalator_travel_time_l2154_215419

/-- Calculates the time taken for a person to cover the length of a moving escalator -/
theorem escalator_travel_time 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (person_speed : ℝ) : 
  escalator_speed = 12 →
  escalator_length = 210 →
  person_speed = 2 →
  escalator_length / (escalator_speed + person_speed) = 15 := by
sorry


end NUMINAMATH_CALUDE_escalator_travel_time_l2154_215419


namespace NUMINAMATH_CALUDE_floor_subtraction_inequality_l2154_215497

theorem floor_subtraction_inequality (x y : ℝ) : ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_subtraction_inequality_l2154_215497


namespace NUMINAMATH_CALUDE_a_4_value_l2154_215441

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 4 / a 2 - a 3 = 0 →
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_a_4_value_l2154_215441


namespace NUMINAMATH_CALUDE_students_making_stars_l2154_215451

theorem students_making_stars (stars_per_student : ℕ) (total_stars : ℕ) (h1 : stars_per_student = 3) (h2 : total_stars = 372) :
  total_stars / stars_per_student = 124 := by
  sorry

end NUMINAMATH_CALUDE_students_making_stars_l2154_215451


namespace NUMINAMATH_CALUDE_natural_number_triples_l2154_215475

theorem natural_number_triples (a b c : ℕ) :
  (∃ m n p : ℕ, (a + b : ℚ) / c = m ∧ (b + c : ℚ) / a = n ∧ (c + a : ℚ) / b = p) →
  (∃ k : ℕ, (a = k ∧ b = k ∧ c = k) ∨
            (a = k ∧ b = k ∧ c = 2 * k) ∨
            (a = k ∧ b = 2 * k ∧ c = 3 * k) ∨
            (a = k ∧ c = 2 * k ∧ b = 3 * k) ∨
            (b = k ∧ a = 2 * k ∧ c = 3 * k) ∨
            (b = k ∧ c = 2 * k ∧ a = 3 * k) ∨
            (c = k ∧ a = 2 * k ∧ b = 3 * k) ∨
            (c = k ∧ b = 2 * k ∧ a = 3 * k)) :=
sorry

end NUMINAMATH_CALUDE_natural_number_triples_l2154_215475


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_l2154_215435

/-- The expression for which we need to find the coefficient of x^4 -/
def expression (x : ℝ) : ℝ := 5 * (x^4 - 2*x^3) + 3 * (2*x^2 - 3*x^4 + x^6) - (5*x^6 - 2*x^4)

/-- The coefficient of x^4 in the given expression is -2 -/
theorem coefficient_of_x_fourth : (deriv^[4] expression 0) / 24 = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_l2154_215435


namespace NUMINAMATH_CALUDE_jack_waiting_time_l2154_215470

/-- The total waiting time for Jack's trip to Canada -/
def total_waiting_time (customs_hours : ℕ) (quarantine_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  customs_hours + quarantine_days * hours_per_day

/-- Theorem stating that Jack's total waiting time is 356 hours -/
theorem jack_waiting_time :
  total_waiting_time 20 14 24 = 356 := by
  sorry

end NUMINAMATH_CALUDE_jack_waiting_time_l2154_215470


namespace NUMINAMATH_CALUDE_christopher_sugar_substitute_cost_l2154_215443

/-- Represents the cost calculation for Christopher's sugar substitute usage --/
theorem christopher_sugar_substitute_cost :
  let packets_per_coffee : ℕ := 1
  let coffees_per_day : ℕ := 2
  let packets_per_box : ℕ := 30
  let cost_per_box : ℚ := 4
  let days : ℕ := 90

  let daily_usage : ℕ := packets_per_coffee * coffees_per_day
  let total_packets : ℕ := daily_usage * days
  let boxes_needed : ℕ := (total_packets + packets_per_box - 1) / packets_per_box
  let total_cost : ℚ := cost_per_box * boxes_needed

  total_cost = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_christopher_sugar_substitute_cost_l2154_215443


namespace NUMINAMATH_CALUDE_max_rides_both_days_l2154_215486

/-- Represents the prices of rides on a given day -/
structure RidePrices where
  ferrisWheel : ℕ
  rollerCoaster : ℕ
  bumperCars : ℕ
  carousel : ℕ
  logFlume : ℕ
  hauntedHouse : Option ℕ

/-- Calculates the maximum number of rides within a budget -/
def maxRides (prices : RidePrices) (budget : ℕ) : ℕ :=
  sorry

/-- The daily budget -/
def dailyBudget : ℕ := 10

/-- Ride prices for the first day -/
def firstDayPrices : RidePrices :=
  { ferrisWheel := 4
  , rollerCoaster := 5
  , bumperCars := 3
  , carousel := 2
  , logFlume := 6
  , hauntedHouse := none }

/-- Ride prices for the second day -/
def secondDayPrices : RidePrices :=
  { ferrisWheel := 4
  , rollerCoaster := 7
  , bumperCars := 3
  , carousel := 2
  , logFlume := 6
  , hauntedHouse := some 4 }

theorem max_rides_both_days :
  maxRides firstDayPrices dailyBudget = 3 ∧
  maxRides secondDayPrices dailyBudget = 3 :=
sorry

end NUMINAMATH_CALUDE_max_rides_both_days_l2154_215486


namespace NUMINAMATH_CALUDE_lunch_gratuity_percentage_l2154_215460

/-- Given the conditions of a lunch bill, prove the gratuity percentage --/
theorem lunch_gratuity_percentage
  (total_price : ℝ)
  (num_people : ℕ)
  (avg_price_no_gratuity : ℝ)
  (h1 : total_price = 207)
  (h2 : num_people = 15)
  (h3 : avg_price_no_gratuity = 12) :
  (total_price - (↑num_people * avg_price_no_gratuity)) / (↑num_people * avg_price_no_gratuity) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lunch_gratuity_percentage_l2154_215460


namespace NUMINAMATH_CALUDE_distinct_values_of_c_l2154_215424

theorem distinct_values_of_c (c : ℂ) (p q r : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ z : ℂ, (z - p) * (z - q) * (z - r) + 1 = (z - c*p) * (z - c*q) * (z - c*r) + 1) →
  ∃ S : Finset ℂ, S.card = 4 ∧ c ∈ S ∧ ∀ x : ℂ, x ∈ S → 
    ∀ z : ℂ, (z - p) * (z - q) * (z - r) + 1 = (z - x*p) * (z - x*q) * (z - x*r) + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_values_of_c_l2154_215424


namespace NUMINAMATH_CALUDE_calculation_sum_l2154_215452

theorem calculation_sum (x : ℝ) (h : (x - 5) + 14 = 39) : (5 * x + 14) + 39 = 203 := by
  sorry

end NUMINAMATH_CALUDE_calculation_sum_l2154_215452


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2154_215434

theorem fractional_equation_solution (x a : ℝ) : 
  (2 * x + a) / (x + 1) = 1 → x < 0 → a > 1 ∧ a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2154_215434


namespace NUMINAMATH_CALUDE_segment_length_proof_l2154_215420

theorem segment_length_proof (C D R S : ℝ) : 
  C < R ∧ R < S ∧ S < D →  -- R and S are on the same side of midpoint
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 2 / 3 →  -- S divides CD in ratio 2:3
  S - R = 5 →  -- Length of RS is 5
  D - C = 200 := by  -- Length of CD is 200
sorry


end NUMINAMATH_CALUDE_segment_length_proof_l2154_215420


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l2154_215449

theorem parabola_point_comparison :
  ∀ (y₁ y₂ : ℝ),
  y₁ = (-5)^2 + 2*(-5) + 3 →
  y₂ = 2^2 + 2*2 + 3 →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l2154_215449


namespace NUMINAMATH_CALUDE_appetizer_price_l2154_215474

def total_spent : ℚ := 50
def entree_percentage : ℚ := 80 / 100
def num_entrees : ℕ := 4
def num_appetizers : ℕ := 2

theorem appetizer_price :
  let entree_cost : ℚ := total_spent * entree_percentage
  let appetizer_total : ℚ := total_spent - entree_cost
  let single_appetizer_price : ℚ := appetizer_total / num_appetizers
  single_appetizer_price = 5 := by
sorry

end NUMINAMATH_CALUDE_appetizer_price_l2154_215474


namespace NUMINAMATH_CALUDE_xiaoning_pe_score_l2154_215430

/-- Calculates the comprehensive score for physical education based on midterm and final exam scores -/
def calculate_pe_score (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  0.3 * midterm_score + 0.7 * final_score

/-- Theorem: Xiaoning's physical education comprehensive score is 87 points -/
theorem xiaoning_pe_score :
  let max_score : ℝ := 100
  let midterm_weight : ℝ := 0.3
  let final_weight : ℝ := 0.7
  let xiaoning_midterm : ℝ := 80
  let xiaoning_final : ℝ := 90
  calculate_pe_score xiaoning_midterm xiaoning_final = 87 := by
  sorry

#eval calculate_pe_score 80 90

end NUMINAMATH_CALUDE_xiaoning_pe_score_l2154_215430


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_l2154_215453

def total_students : ℕ := 5
def total_girls : ℕ := 3
def representatives : ℕ := 2

theorem probability_at_least_one_boy :
  let total_selections := Nat.choose total_students representatives
  let all_girl_selections := Nat.choose total_girls representatives
  (1 : ℚ) - (all_girl_selections : ℚ) / (total_selections : ℚ) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_l2154_215453


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_25_over_6_l2154_215479

theorem greatest_integer_less_than_negative_25_over_6 :
  Int.floor (-25 / 6 : ℚ) = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_25_over_6_l2154_215479


namespace NUMINAMATH_CALUDE_booboo_arrangements_l2154_215472

def word_arrangements (n : ℕ) (r₁ : ℕ) (r₂ : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r₁ * Nat.factorial r₂)

theorem booboo_arrangements :
  word_arrangements 6 2 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_booboo_arrangements_l2154_215472


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2154_215445

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2154_215445


namespace NUMINAMATH_CALUDE_age_difference_l2154_215455

/-- Represents the ages of three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrothersAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ∃ (y : ℕ), ages.richard + y = 2 * (ages.scott + y) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem age_difference (ages : BrothersAges) :
  problem_conditions ages →
  ∃ (s : ℕ), s < 14 ∧ ages.david - ages.scott = 14 - s :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2154_215455


namespace NUMINAMATH_CALUDE_basketball_free_throws_l2154_215462

theorem basketball_free_throws (total_players : Nat) (captains : Nat) 
  (h1 : total_players = 15)
  (h2 : captains = 2)
  (h3 : captains ≤ total_players) :
  (total_players - 1) * captains = 28 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l2154_215462
