import Mathlib

namespace NUMINAMATH_CALUDE_shaded_square_fraction_l2357_235791

theorem shaded_square_fraction :
  let large_square_side : ℝ := 6
  let small_square_side : ℝ := Real.sqrt 2
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_area : ℝ := small_square_side ^ 2
  (small_square_area / large_square_area) = (1 : ℝ) / 18 := by sorry

end NUMINAMATH_CALUDE_shaded_square_fraction_l2357_235791


namespace NUMINAMATH_CALUDE_interest_percentage_calculation_l2357_235713

/-- Calculates the interest percentage for a purchase with a payment plan -/
theorem interest_percentage_calculation (purchase_price : ℝ) (down_payment : ℝ) (monthly_payment : ℝ) (num_months : ℕ) :
  purchase_price = 110 →
  down_payment = 10 →
  monthly_payment = 10 →
  num_months = 12 →
  let total_paid := down_payment + (monthly_payment * num_months)
  let interest_paid := total_paid - purchase_price
  let interest_percentage := (interest_paid / purchase_price) * 100
  ∃ ε > 0, |interest_percentage - 18.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_interest_percentage_calculation_l2357_235713


namespace NUMINAMATH_CALUDE_largest_angle_measure_l2357_235761

/-- Represents a pentagon with angles in the ratio 3:3:3:4:5 -/
structure RatioPentagon where
  /-- The common factor for the angle measures -/
  x : ℝ
  /-- The sum of interior angles of a pentagon is 540° -/
  angle_sum : 3*x + 3*x + 3*x + 4*x + 5*x = 540

/-- Theorem: The largest angle in a RatioPentagon is 150° -/
theorem largest_angle_measure (p : RatioPentagon) : 5 * p.x = 150 := by
  sorry

#check largest_angle_measure

end NUMINAMATH_CALUDE_largest_angle_measure_l2357_235761


namespace NUMINAMATH_CALUDE_assignment_validity_l2357_235718

/-- Represents a variable in a programming language -/
structure Variable where
  name : String

/-- Represents an expression in a programming language -/
inductive Expression
  | Var : Variable → Expression
  | Product : Expression → Expression → Expression
  | Literal : Int → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Expression
  rhs : Expression

/-- Predicate to check if an expression is a single variable -/
def isSingleVariable : Expression → Prop
  | Expression.Var _ => True
  | _ => False

/-- Theorem: An assignment statement is valid if and only if its left-hand side is a single variable -/
theorem assignment_validity (a : Assignment) :
  isSingleVariable a.lhs ↔ True :=
sorry

#check assignment_validity

end NUMINAMATH_CALUDE_assignment_validity_l2357_235718


namespace NUMINAMATH_CALUDE_other_replaced_man_age_proof_l2357_235709

/-- The age of the other replaced man in a group of three men -/
def other_replaced_man_age : ℕ := 26

theorem other_replaced_man_age_proof 
  (initial_men : ℕ) 
  (replaced_men : ℕ) 
  (known_replaced_age : ℕ) 
  (new_men_avg_age : ℝ) 
  (h1 : initial_men = 3)
  (h2 : replaced_men = 2)
  (h3 : known_replaced_age = 23)
  (h4 : new_men_avg_age = 25)
  (h5 : ∀ (initial_avg new_avg : ℝ), new_avg > initial_avg) :
  other_replaced_man_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_other_replaced_man_age_proof_l2357_235709


namespace NUMINAMATH_CALUDE_unique_five_numbers_l2357_235742

theorem unique_five_numbers : 
  ∃! (a b c d e : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ d < e ∧
    a * b > 25 ∧ d * e < 75 ∧
    a = 5 ∧ b = 6 ∧ c = 7 ∧ d = 8 ∧ e = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_five_numbers_l2357_235742


namespace NUMINAMATH_CALUDE_equation_solutions_l2357_235798

theorem equation_solutions : 
  ∀ n m : ℕ+, 3 * 2^(m : ℕ) + 1 = (n : ℕ)^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2357_235798


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l2357_235724

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (U \ A) ∩ (U \ B) = ∅ := by sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l2357_235724


namespace NUMINAMATH_CALUDE_inverse_of_inverse_sixteen_l2357_235738

def f (x : ℝ) : ℝ := 5 * x + 6

theorem inverse_of_inverse_sixteen (hf : ∀ x, f x = 5 * x + 6) :
  (f ∘ f) (-4/5) = 16 :=
sorry

end NUMINAMATH_CALUDE_inverse_of_inverse_sixteen_l2357_235738


namespace NUMINAMATH_CALUDE_john_money_left_l2357_235765

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 4 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem: John has 50 - 10q dollars left after his purchases -/
theorem john_money_left (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l2357_235765


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2357_235768

theorem quadratic_solution_difference_squared :
  ∀ α β : ℝ,
  (α^2 - 5*α + 6 = 0) →
  (β^2 - 5*β + 6 = 0) →
  (α ≠ β) →
  (α - β)^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2357_235768


namespace NUMINAMATH_CALUDE_chef_pies_l2357_235757

theorem chef_pies (apple_pies pecan_pies pumpkin_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pecan_pies = 4) 
  (h3 : pumpkin_pies = 7) : 
  apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end NUMINAMATH_CALUDE_chef_pies_l2357_235757


namespace NUMINAMATH_CALUDE_income_increase_percentage_l2357_235763

/-- Proves that given the ratio of expenditure to savings is 3:2, if savings increase by 6% and expenditure increases by 21%, then the income increases by 15% -/
theorem income_increase_percentage 
  (I : ℝ) -- Initial income
  (E : ℝ) -- Initial expenditure
  (S : ℝ) -- Initial savings
  (h1 : E / S = 3 / 2) -- Ratio of expenditure to savings is 3:2
  (h2 : I = E + S) -- Income is the sum of expenditure and savings
  (h3 : S * 1.06 + E * 1.21 = I * (1 + 15/100)) -- New savings + new expenditure = new income
  : ∃ (x : ℝ), x = 15 ∧ I * (1 + x/100) = S * 1.06 + E * 1.21 :=
by sorry

end NUMINAMATH_CALUDE_income_increase_percentage_l2357_235763


namespace NUMINAMATH_CALUDE_conditional_probability_in_box_l2357_235775

/-- A box containing products of different classes -/
structure Box where
  total : ℕ
  firstClass : ℕ
  secondClass : ℕ

/-- The probability of drawing a first-class product followed by another first-class product -/
def probBothFirstClass (b : Box) : ℚ :=
  (b.firstClass : ℚ) * ((b.firstClass - 1) : ℚ) / ((b.total : ℚ) * ((b.total - 1) : ℚ))

/-- The probability of drawing a first-class product first -/
def probFirstClassFirst (b : Box) : ℚ :=
  (b.firstClass : ℚ) / (b.total : ℚ)

/-- The conditional probability of drawing a first-class product second, given that the first draw was a first-class product -/
def conditionalProbability (b : Box) : ℚ :=
  probBothFirstClass b / probFirstClassFirst b

theorem conditional_probability_in_box (b : Box) 
  (h1 : b.total = 4)
  (h2 : b.firstClass = 3)
  (h3 : b.secondClass = 1)
  (h4 : b.firstClass + b.secondClass = b.total) :
  conditionalProbability b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_in_box_l2357_235775


namespace NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l2357_235726

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Statement 1
theorem statement_1 : ∃ (a b : Line) (α : Plane),
  parallel_line a b ∧ contained_in b α ∧ 
  ¬(parallel_line_plane a α) ∧ ¬(contained_in a α) := by sorry

-- Statement 2
theorem statement_2 : ∃ (a b : Line) (α : Plane),
  parallel_line_plane a α ∧ parallel_line_plane b α ∧ 
  ¬(parallel_line a b) := by sorry

-- Statement 3
theorem statement_3 : ¬(∀ (a : Line) (α β : Plane),
  parallel_line_plane a α → parallel_line_plane a β → 
  (α = β ∨ ∃ (l : Line), parallel_line_plane l α ∧ parallel_line_plane l β)) := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l2357_235726


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_ratio_l2357_235734

/-- Given a hyperbola x²/a - y²/a = 1 with a > 0, prove that |FP|/|MN| = √2/2 where:
    F is the right focus
    M and N are intersection points of any line through F with the right branch
    P is the intersection of the perpendicular bisector of MN with the x-axis -/
theorem hyperbola_focus_distance_ratio (a : ℝ) (h : a > 0) :
  ∃ (F M N P : ℝ × ℝ),
    (∀ (x y : ℝ), x^2/a - y^2/a = 1 → 
      (∃ (t : ℝ), (x, y) = M ∨ (x, y) = N) → 
      (F.1 > 0 ∧ F.2 = 0) ∧
      (∃ (m : ℝ), (M.2 - F.2) = m * (M.1 - F.1) ∧ (N.2 - F.2) = m * (N.1 - F.1)) ∧
      (P.2 = 0 ∧ P.1 = (M.1 + N.1)/2 + (M.2 + N.2)^2 / (2 * (M.1 + N.1)))) →
    ‖F - P‖ / ‖M - N‖ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_ratio_l2357_235734


namespace NUMINAMATH_CALUDE_interesting_numbers_200_to_400_l2357_235727

/-- A natural number is interesting if there exists another natural number that satisfies certain conditions. -/
def IsInteresting (A : ℕ) : Prop :=
  ∃ B : ℕ, A > B ∧ Nat.Prime (A - B) ∧ ∃ n : ℕ, A * B = n * n

/-- The theorem stating the interesting numbers between 200 and 400. -/
theorem interesting_numbers_200_to_400 :
  ∀ A : ℕ, 200 < A → A < 400 → (IsInteresting A ↔ A = 225 ∨ A = 256 ∨ A = 361) := by
  sorry


end NUMINAMATH_CALUDE_interesting_numbers_200_to_400_l2357_235727


namespace NUMINAMATH_CALUDE_recurrence_sequence_x7_l2357_235710

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (x : ℕ → ℕ) : Prop :=
  (∀ n, x n > 0) ∧
  (∀ n ∈ ({1, 2, 3, 4} : Finset ℕ), x (n + 3) = x (n + 2) * (x (n + 1) + x n))

theorem recurrence_sequence_x7 (x : ℕ → ℕ) (h : RecurrenceSequence x) (h6 : x 6 = 144) :
  x 7 = 3456 := by
  sorry

#check recurrence_sequence_x7

end NUMINAMATH_CALUDE_recurrence_sequence_x7_l2357_235710


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l2357_235743

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Represents a cube -/
structure Cube where
  side : ℝ

/-- Theorem: The surface area of a rectangular solid remains unchanged
    when two unit cubes are removed from opposite corners -/
theorem surface_area_unchanged
  (solid : Dimensions)
  (cube : Cube)
  (h1 : solid.length = 2)
  (h2 : solid.width = 3)
  (h3 : solid.height = 4)
  (h4 : cube.side = 1) :
  surfaceArea solid = surfaceArea solid - 2 * (3 * cube.side^2) + 2 * (3 * cube.side^2) :=
by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l2357_235743


namespace NUMINAMATH_CALUDE_sequence_term_value_l2357_235788

/-- Given a finite sequence {a_n} with m terms, where S(n) represents the sum of all terms
    starting from the n-th term, prove that a_n = -2n - 1 when 1 ≤ n < m. -/
theorem sequence_term_value (m : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) 
    (h1 : 1 ≤ n) (h2 : n < m) (h3 : ∀ k, 1 ≤ k → k ≤ m → S k = k^2) :
  a n = -2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_value_l2357_235788


namespace NUMINAMATH_CALUDE_smallest_valid_N_sum_of_digits_l2357_235779

def P (N : ℕ) : ℚ := (N + 1 - Int.ceil (N / 3 : ℚ)) / (N + 1 : ℚ)

def is_valid (N : ℕ) : Prop :=
  N > 0 ∧ N % 5 = 0 ∧ N % 6 = 0 ∧ P N < 2/3

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_valid_N_sum_of_digits :
  ∃ N, is_valid N ∧
    (∀ M, is_valid M → N ≤ M) ∧
    sum_of_digits N = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_N_sum_of_digits_l2357_235779


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_shifted_l2357_235731

theorem root_sum_reciprocal_shifted (a b c : ℂ) : 
  (a^3 - 2*a - 5 = 0) → 
  (b^3 - 2*b - 5 = 0) → 
  (c^3 - 2*c - 5 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = 10) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_shifted_l2357_235731


namespace NUMINAMATH_CALUDE_interesting_pairs_characterization_l2357_235721

/-- An ordered pair (a, b) of positive integers is interesting if for any positive integer n,
    there exists a positive integer k such that a^k + b is divisible by 2^n. -/
def IsInteresting (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ+, (a.val ^ k.val + b.val) % (2^n.val) = 0

/-- Characterization of interesting pairs -/
theorem interesting_pairs_characterization (a b : ℕ+) :
  IsInteresting a b ↔ 
  (∃ (k l q : ℕ+), k ≥ 2 ∧ l.val % 2 = 1 ∧ q.val % 2 = 1 ∧
    ((a = 2^k.val * l + 1 ∧ b = 2^k.val * q - 1) ∨
     (a = 2^k.val * l - 1 ∧ b = 2^k.val * q + 1))) :=
sorry

end NUMINAMATH_CALUDE_interesting_pairs_characterization_l2357_235721


namespace NUMINAMATH_CALUDE_sliced_meat_cost_l2357_235769

/-- Given a 4 pack of sliced meat costing $40.00 with an additional 30% for rush delivery,
    the cost per type of meat is $13.00. -/
theorem sliced_meat_cost (pack_size : ℕ) (base_cost rush_percentage : ℚ) :
  pack_size = 4 →
  base_cost = 40 →
  rush_percentage = 0.3 →
  (base_cost + base_cost * rush_percentage) / pack_size = 13 :=
by sorry

end NUMINAMATH_CALUDE_sliced_meat_cost_l2357_235769


namespace NUMINAMATH_CALUDE_friends_total_score_l2357_235752

/-- Given three friends' scores in a table football game, prove their total score. -/
theorem friends_total_score (darius_score matt_score marius_score : ℕ) : 
  marius_score = darius_score + 3 →
  matt_score = darius_score + 5 →
  darius_score = 10 →
  darius_score + matt_score + marius_score = 38 := by
sorry


end NUMINAMATH_CALUDE_friends_total_score_l2357_235752


namespace NUMINAMATH_CALUDE_number_puzzle_l2357_235707

theorem number_puzzle (N A : ℝ) : N = 295 ∧ N / 5 + A = 65 → A = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2357_235707


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l2357_235745

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches. -/
theorem ten_player_tournament_matches : num_matches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l2357_235745


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2357_235746

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.20 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2357_235746


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l2357_235708

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg correct_num wrong_num : ℚ) : 
  n = 10 →
  initial_avg = 18 →
  correct_avg = 19 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = (n : ℚ) * correct_avg →
  wrong_num = 26 := by
sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l2357_235708


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l2357_235701

theorem largest_lcm_with_18 : 
  (Finset.max {lcm 18 3, lcm 18 5, lcm 18 9, lcm 18 12, lcm 18 15, lcm 18 18}) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l2357_235701


namespace NUMINAMATH_CALUDE_wendys_recycling_points_l2357_235774

/-- Wendy's recycling points calculation -/
theorem wendys_recycling_points :
  ∀ (points_per_bag : ℕ) (total_bags : ℕ) (unrecycled_bags : ℕ),
    points_per_bag = 5 →
    total_bags = 11 →
    unrecycled_bags = 2 →
    points_per_bag * (total_bags - unrecycled_bags) = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendys_recycling_points_l2357_235774


namespace NUMINAMATH_CALUDE_rain_amount_l2357_235728

theorem rain_amount (malina_initial : ℕ) (jahoda_initial : ℕ) (rain_amount : ℕ) : 
  malina_initial = 48 →
  malina_initial = jahoda_initial + 32 →
  (malina_initial + rain_amount) - (jahoda_initial + rain_amount) = 32 →
  malina_initial + rain_amount = 2 * (jahoda_initial + rain_amount) →
  rain_amount = 16 := by
sorry

end NUMINAMATH_CALUDE_rain_amount_l2357_235728


namespace NUMINAMATH_CALUDE_sock_problem_solution_l2357_235784

/-- Represents the number of pairs of socks at each price point --/
structure SockInventory where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Calculates the total number of sock pairs --/
def total_pairs (s : SockInventory) : ℕ :=
  s.two_dollar + s.four_dollar + s.five_dollar

/-- Calculates the total cost of all socks --/
def total_cost (s : SockInventory) : ℕ :=
  2 * s.two_dollar + 4 * s.four_dollar + 5 * s.five_dollar

theorem sock_problem_solution :
  ∃ (s : SockInventory),
    total_pairs s = 15 ∧
    total_cost s = 41 ∧
    s.two_dollar ≥ 1 ∧
    s.four_dollar ≥ 1 ∧
    s.five_dollar ≥ 1 ∧
    s.two_dollar = 11 :=
by sorry

#check sock_problem_solution

end NUMINAMATH_CALUDE_sock_problem_solution_l2357_235784


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l2357_235780

def f (x : ℝ) : ℝ := x * abs x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l2357_235780


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l2357_235783

/-- Given two positive integers with LCM 750 and HCF 25, prove their product is 18750 -/
theorem product_from_lcm_hcf (a b : ℕ+) : 
  Nat.lcm a b = 750 → Nat.gcd a b = 25 → a * b = 18750 := by sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l2357_235783


namespace NUMINAMATH_CALUDE_expression_evaluation_l2357_235755

theorem expression_evaluation :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  f 2 = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2357_235755


namespace NUMINAMATH_CALUDE_jenny_sweets_division_l2357_235749

theorem jenny_sweets_division :
  ∃ n : ℕ, n ≠ 5 ∧ n ≠ 12 ∧ 30 % n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_sweets_division_l2357_235749


namespace NUMINAMATH_CALUDE_employee_preference_city_y_l2357_235766

/-- Proves that given the conditions of the employee relocation problem,
    the percentage of employees preferring city Y is 40%. -/
theorem employee_preference_city_y (
  total_employees : ℕ)
  (relocated_to_x_percent : ℚ)
  (relocated_to_y_percent : ℚ)
  (max_preferred_relocation : ℕ)
  (h1 : total_employees = 200)
  (h2 : relocated_to_x_percent = 30 / 100)
  (h3 : relocated_to_y_percent = 70 / 100)
  (h4 : relocated_to_x_percent + relocated_to_y_percent = 1)
  (h5 : max_preferred_relocation = 140) :
  ∃ (prefer_y_percent : ℚ),
    prefer_y_percent = 40 / 100 ∧
    prefer_y_percent * total_employees = max_preferred_relocation - relocated_to_x_percent * total_employees :=
by sorry

end NUMINAMATH_CALUDE_employee_preference_city_y_l2357_235766


namespace NUMINAMATH_CALUDE_number_problem_l2357_235712

theorem number_problem (x : ℝ) : (6 * x) / 1.5 = 3.8 → x = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2357_235712


namespace NUMINAMATH_CALUDE_termite_ridden_collapsing_fraction_value_l2357_235714

/-- The fraction of homes on Gotham Street that are termite-ridden -/
def termite_ridden_fraction : ℚ := 1/3

/-- The fraction of homes on Gotham Street that are termite-ridden but not collapsing -/
def termite_ridden_not_collapsing_fraction : ℚ := 1/10

/-- The fraction of termite-ridden homes that are collapsing -/
def termite_ridden_collapsing_fraction : ℚ := 
  (termite_ridden_fraction - termite_ridden_not_collapsing_fraction) / termite_ridden_fraction

theorem termite_ridden_collapsing_fraction_value : 
  termite_ridden_collapsing_fraction = 7/30 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_collapsing_fraction_value_l2357_235714


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2357_235704

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a_3 = 4 and a_7 = 12, prove that a_11 = 36 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_3 : a 3 = 4) 
    (h_7 : a 7 = 12) : 
  a 11 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2357_235704


namespace NUMINAMATH_CALUDE_markup_percentage_is_40_percent_l2357_235796

/-- Proves that the markup percentage on the selling price of a desk is 40% given the specified conditions. -/
theorem markup_percentage_is_40_percent
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (markup : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 150)
  (h2 : selling_price = purchase_price + markup)
  (h3 : gross_profit = 100)
  (h4 : gross_profit = selling_price - purchase_price) :
  (markup / selling_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_is_40_percent_l2357_235796


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2357_235789

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculate the area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  ∃ ε > 0, |trapezoid_area t - 1242.425| < ε :=
sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2357_235789


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l2357_235764

def given_numbers : List Nat := [0, 2, 3, 4, 6]

def is_valid_three_digit (n : Nat) : Bool :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 ∈ given_numbers) ∧ ((n / 10) % 10 ∈ given_numbers) ∧ (n % 10 ∈ given_numbers)

def count_valid_three_digit : Nat :=
  (List.range 1000).filter is_valid_three_digit |>.length

def is_divisible_by_three (n : Nat) : Bool :=
  n % 3 = 0

def count_valid_three_digit_divisible_by_three : Nat :=
  (List.range 1000).filter (λ n => is_valid_three_digit n ∧ is_divisible_by_three n) |>.length

theorem three_digit_numbers_count :
  count_valid_three_digit = 48 ∧
  count_valid_three_digit_divisible_by_three = 20 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_numbers_count_l2357_235764


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l2357_235715

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the intersection points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

theorem perpendicular_bisector_of_intersection :
  ∃ (A B : ℝ × ℝ),
    (C₁ A.1 A.2 ∧ C₂ A.1 A.2) ∧
    (C₁ B.1 B.2 ∧ C₂ B.1 B.2) ∧
    A ≠ B ∧
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l2357_235715


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l2357_235773

theorem min_value_of_complex_expression (z : ℂ) (h : Complex.abs (z - 3 + 3*I) = 3) :
  ∃ (min_val : ℝ), min_val = 19 - 6 * Real.sqrt 2 ∧
    ∀ (w : ℂ), Complex.abs (w - 3 + 3*I) = 3 →
      Complex.abs (w + 2 - I)^2 + Complex.abs (w - 4 + 2*I)^2 ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l2357_235773


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2357_235767

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product := a.1 * b.1 + a.2 * b.2
  angle = Real.arccos (dot_product / (norm_a * norm_b)) →
  norm_a = 1 →
  norm_b = 1 / 2 →
  Real.sqrt (((a.1 - 2 * b.1) ^ 2) + ((a.2 - 2 * b.2) ^ 2)) = 1 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2357_235767


namespace NUMINAMATH_CALUDE_staff_age_calculation_l2357_235751

theorem staff_age_calculation (num_students : ℕ) (student_avg_age : ℕ) (num_staff : ℕ) (age_increase : ℕ) :
  num_students = 50 →
  student_avg_age = 25 →
  num_staff = 5 →
  age_increase = 2 →
  (num_students * student_avg_age + num_staff * ((student_avg_age + age_increase) * (num_students + num_staff) - num_students * student_avg_age)) / num_staff = 235 := by
  sorry

end NUMINAMATH_CALUDE_staff_age_calculation_l2357_235751


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l2357_235732

theorem complex_sum_of_parts (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  (z.re : ℝ) + (z.im : ℝ) = -1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l2357_235732


namespace NUMINAMATH_CALUDE_jane_dolls_l2357_235786

theorem jane_dolls (total : ℕ) (difference : ℕ) : 
  total = 32 → difference = 6 → ∃ (jane : ℕ), jane + (jane + difference) = total ∧ jane = 13 := by
sorry

end NUMINAMATH_CALUDE_jane_dolls_l2357_235786


namespace NUMINAMATH_CALUDE_edward_pen_expenses_l2357_235756

/-- Given Edward's initial money, book expenses, and remaining money, 
    calculate the amount spent on pens. -/
theorem edward_pen_expenses (initial_money : ℕ) (book_expenses : ℕ) (remaining_money : ℕ) 
    (h1 : initial_money = 41)
    (h2 : book_expenses = 6)
    (h3 : remaining_money = 19) :
  initial_money - book_expenses - remaining_money = 16 := by
  sorry

end NUMINAMATH_CALUDE_edward_pen_expenses_l2357_235756


namespace NUMINAMATH_CALUDE_max_value_and_right_triangle_l2357_235744

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^3 + x^2 else a * Real.log x

theorem max_value_and_right_triangle (a : ℝ) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc (-1 : ℝ) (Real.exp 1), f a x ≤ m ∧
    (m = max 2 a ∨ (a < 2 ∧ m = 2))) ∧
  (a > 0 → ∃ (P Q : ℝ × ℝ),
    (P.1 > 0 ∧ P.2 = f a P.1) ∧
    (Q.1 < 0 ∧ Q.2 = f a Q.1) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
    ((P.1 + Q.1) / 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_right_triangle_l2357_235744


namespace NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l2357_235730

theorem x_equals_plus_minus_fifteen (x : ℝ) :
  (x / 5) / 3 = 3 / (x / 5) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l2357_235730


namespace NUMINAMATH_CALUDE_tuesday_max_hours_l2357_235723

/-- Represents the days of the week from Monday to Friday -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Returns the number of hours Gabe spent riding his bike on a given day -/
def hours_spent (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 3
  | Weekday.Tuesday => 4
  | Weekday.Wednesday => 2
  | Weekday.Thursday => 3
  | Weekday.Friday => 1

/-- Theorem: Tuesday is the day when Gabe spent the greatest number of hours riding his bike -/
theorem tuesday_max_hours :
  ∀ (day : Weekday), hours_spent Weekday.Tuesday ≥ hours_spent day :=
by sorry

end NUMINAMATH_CALUDE_tuesday_max_hours_l2357_235723


namespace NUMINAMATH_CALUDE_opposite_roots_n_value_l2357_235733

/-- Given a rational function equal to (n-2)/(n+2) with roots of opposite signs, prove n = 2b + 2 -/
theorem opposite_roots_n_value (b d p q n : ℝ) (x : ℝ → ℝ) :
  (∀ x, (x^2 - b*x + d) / (p*x - q) = (n - 2) / (n + 2)) →
  (∃ r : ℝ, x r = r ∧ x (-r) = -r) →
  p = b + 1 →
  n = 2*b + 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_roots_n_value_l2357_235733


namespace NUMINAMATH_CALUDE_knife_percentage_after_trade_l2357_235787

/-- Represents Carolyn's silverware set -/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set -/
def total (s : SilverwareSet) : ℕ := s.knives + s.forks + s.spoons

/-- Represents the trade operation -/
def trade (s : SilverwareSet) (knivesGained spoonsTrade : ℕ) : SilverwareSet :=
  { knives := s.knives + knivesGained,
    forks := s.forks,
    spoons := s.spoons - spoonsTrade }

/-- Calculates the percentage of knives in a silverware set -/
def knifePercentage (s : SilverwareSet) : ℚ :=
  (s.knives : ℚ) / (total s : ℚ) * 100

theorem knife_percentage_after_trade :
  let initialSet : SilverwareSet := { knives := 6, forks := 12, spoons := 6 * 3 }
  let finalSet := trade initialSet 10 6
  knifePercentage finalSet = 40 := by sorry

end NUMINAMATH_CALUDE_knife_percentage_after_trade_l2357_235787


namespace NUMINAMATH_CALUDE_choose_two_cooks_from_eight_l2357_235772

theorem choose_two_cooks_from_eight (n : ℕ) (k : ℕ) :
  n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_cooks_from_eight_l2357_235772


namespace NUMINAMATH_CALUDE_next_shared_meeting_l2357_235799

/-- Represents the number of days between meetings for each group -/
def drama_club_cycle : ℕ := 3
def choir_cycle : ℕ := 5
def debate_team_cycle : ℕ := 7

/-- Theorem stating that the next shared meeting will occur in 105 days -/
theorem next_shared_meeting :
  ∃ (n : ℕ), n > 0 ∧ 
  n % drama_club_cycle = 0 ∧
  n % choir_cycle = 0 ∧
  n % debate_team_cycle = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ 
    m % drama_club_cycle = 0 ∧
    m % choir_cycle = 0 ∧
    m % debate_team_cycle = 0 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_next_shared_meeting_l2357_235799


namespace NUMINAMATH_CALUDE_sandy_siding_cost_l2357_235750

-- Define the dimensions and costs
def wall_width : ℝ := 10
def wall_height : ℝ := 8
def roof_base : ℝ := 10
def roof_height : ℝ := 6
def siding_section_size : ℝ := 100  -- 10 ft x 10 ft = 100 sq ft
def siding_section_cost : ℝ := 30

-- Theorem to prove
theorem sandy_siding_cost :
  let wall_area := wall_width * wall_height
  let roof_area := roof_base * roof_height
  let total_area := wall_area + roof_area
  let sections_needed := ⌈total_area / siding_section_size⌉
  let total_cost := sections_needed * siding_section_cost
  total_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_sandy_siding_cost_l2357_235750


namespace NUMINAMATH_CALUDE_distribute_five_books_four_bags_l2357_235748

/-- The number of ways to distribute n distinct objects into k identical containers --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 41 ways to distribute 5 books into 4 bags --/
theorem distribute_five_books_four_bags : distribute 5 4 = 41 := by sorry

end NUMINAMATH_CALUDE_distribute_five_books_four_bags_l2357_235748


namespace NUMINAMATH_CALUDE_adjacent_same_face_exists_l2357_235747

/-- Represents a coin showing either heads or tails -/
inductive CoinFace
| Heads
| Tails

/-- Checks if two coin faces are the same -/
def same_face (a b : CoinFace) : Prop :=
  (a = CoinFace.Heads ∧ b = CoinFace.Heads) ∨ (a = CoinFace.Tails ∧ b = CoinFace.Tails)

/-- A circular arrangement of 11 coins -/
def CoinArrangement := Fin 11 → CoinFace

theorem adjacent_same_face_exists (arrangement : CoinArrangement) :
  ∃ i : Fin 11, same_face (arrangement i) (arrangement ((i + 1) % 11)) :=
sorry


end NUMINAMATH_CALUDE_adjacent_same_face_exists_l2357_235747


namespace NUMINAMATH_CALUDE_quadratic_root_form_l2357_235785

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l2357_235785


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2357_235722

theorem scientific_notation_equality : 284000000 = 2.84 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2357_235722


namespace NUMINAMATH_CALUDE_blueberry_picking_total_l2357_235706

theorem blueberry_picking_total (annie kathryn ben sam : ℕ) : 
  annie = 16 ∧ 
  kathryn = 2 * annie + 2 ∧ 
  ben = kathryn / 2 - 3 ∧ 
  sam = 2 * (ben + kathryn) / 3 → 
  annie + kathryn + ben + sam = 96 := by
sorry

end NUMINAMATH_CALUDE_blueberry_picking_total_l2357_235706


namespace NUMINAMATH_CALUDE_seating_arrangements_l2357_235700

def dodgers : ℕ := 3
def astros : ℕ := 4
def mets : ℕ := 2
def marlins : ℕ := 1

def total_players : ℕ := dodgers + astros + mets + marlins

def number_of_teams : ℕ := 4

theorem seating_arrangements :
  (number_of_teams.factorial) * (dodgers.factorial) * (astros.factorial) * (mets.factorial) * (marlins.factorial) = 6912 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2357_235700


namespace NUMINAMATH_CALUDE_find_divisor_l2357_235736

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 127)
  (h2 : quotient = 5)
  (h3 : remainder = 2)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) :
  dividend / quotient = 25 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2357_235736


namespace NUMINAMATH_CALUDE_no_inscribed_sphere_after_truncation_l2357_235792

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  is_convex : Bool
  vertex_count : ℕ
  face_count : ℕ
  vertex_ge_face : vertex_count ≥ face_count

/-- Truncation operation on a convex polyhedron -/
def truncate (P : ConvexPolyhedron) : ConvexPolyhedron :=
  sorry

/-- A sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a sphere is inscribed in a polyhedron -/
def is_inscribed (S : Sphere) (P : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating that a truncated convex polyhedron cannot have an inscribed sphere -/
theorem no_inscribed_sphere_after_truncation (P : ConvexPolyhedron) :
  ¬ ∃ (S : Sphere), is_inscribed S (truncate P) :=
sorry

end NUMINAMATH_CALUDE_no_inscribed_sphere_after_truncation_l2357_235792


namespace NUMINAMATH_CALUDE_max_fraction_value_101_l2357_235795

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def max_fraction_value (n : ℕ) : ℚ :=
  (factorial n) / 4

theorem max_fraction_value_101 :
  ∀ (f : ℚ), f = max_fraction_value 101 ∨ f < max_fraction_value 101 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_value_101_l2357_235795


namespace NUMINAMATH_CALUDE_boxes_in_case_l2357_235729

/-- Given information about Maria's eggs and boxes -/
structure EggBoxes where
  num_boxes : ℕ
  eggs_per_box : ℕ
  total_eggs : ℕ

/-- Theorem stating that the number of boxes in a case is 3 -/
theorem boxes_in_case (maria : EggBoxes) 
  (h1 : maria.num_boxes = 3)
  (h2 : maria.eggs_per_box = 7)
  (h3 : maria.total_eggs = 21) :
  maria.num_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_in_case_l2357_235729


namespace NUMINAMATH_CALUDE_second_part_interest_rate_l2357_235740

-- Define the total sum and the two parts
def total_sum : ℚ := 2717
def second_part : ℚ := 1672
def first_part : ℚ := total_sum - second_part

-- Define the interest rates and time periods
def first_rate : ℚ := 3 / 100
def first_time : ℚ := 8
def second_time : ℚ := 3

-- Define the theorem
theorem second_part_interest_rate :
  ∃ (r : ℚ), 
    (first_part * first_rate * first_time = second_part * r * second_time) ∧
    (r = 5 / 100) := by
  sorry

end NUMINAMATH_CALUDE_second_part_interest_rate_l2357_235740


namespace NUMINAMATH_CALUDE_andrew_runs_two_miles_l2357_235753

/-- Andrew's daily run in miles -/
def andrew_daily_run : ℝ := 2

/-- Peter's daily run in miles -/
def peter_daily_run : ℝ := andrew_daily_run + 3

/-- Total number of days they run -/
def days : ℕ := 5

/-- Total miles run by both Peter and Andrew -/
def total_miles : ℝ := 35

theorem andrew_runs_two_miles :
  andrew_daily_run = 2 ∧
  peter_daily_run = andrew_daily_run + 3 ∧
  days * (andrew_daily_run + peter_daily_run) = total_miles :=
by sorry

end NUMINAMATH_CALUDE_andrew_runs_two_miles_l2357_235753


namespace NUMINAMATH_CALUDE_shooting_scores_l2357_235705

def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

def variance_A : ℝ := 2.25
def variance_B : ℝ := 4.41

theorem shooting_scores :
  let avg_A := (scores_A.sum) / scores_A.length
  let avg_B := (scores_B.sum) / scores_B.length
  let avg_all := ((scores_A ++ scores_B).sum) / (scores_A.length + scores_B.length)
  avg_A < avg_B ∧ avg_all = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_shooting_scores_l2357_235705


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2357_235754

/-- The equation 3y - 5x = 1 represents the line passing through points (-2, -3) and (4, 7) -/
theorem line_equation_through_points :
  let point1 : ℝ × ℝ := (-2, -3)
  let point2 : ℝ × ℝ := (4, 7)
  let line_eq (x y : ℝ) := 3 * y - 5 * x = 1
  (line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2) := by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2357_235754


namespace NUMINAMATH_CALUDE_horner_v2_value_l2357_235758

-- Define the polynomial
def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

-- Define Horner's method for the first two steps
def horner_v2 (a : ℝ) : ℝ := 
  let v0 := 2
  let v1 := v0 * a + 1
  v1 * a

-- Theorem statement
theorem horner_v2_value :
  horner_v2 2 = 10 :=
sorry

end NUMINAMATH_CALUDE_horner_v2_value_l2357_235758


namespace NUMINAMATH_CALUDE_second_concert_attendance_l2357_235739

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_people : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : additional_people = 119) : 
  first_concert + additional_people = 66018 := by
sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l2357_235739


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l2357_235759

theorem mans_speed_with_current 
  (current_speed : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : current_speed = 2.5)
  (h2 : speed_against_current = 10) : 
  ∃ (speed_with_current : ℝ), speed_with_current = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l2357_235759


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2357_235725

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 128 → aₙ = 14 → d = -3 → aₙ = a₁ + (n - 1) * d → n = 39 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2357_235725


namespace NUMINAMATH_CALUDE_a_8_equals_15_l2357_235782

-- Define the sequence S_n
def S (n : ℕ) : ℕ := n^2

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 0 then 0
  else S n - S (n-1)

-- Theorem statement
theorem a_8_equals_15 : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_15_l2357_235782


namespace NUMINAMATH_CALUDE_f_decreasing_interval_f_min_value_f_max_value_l2357_235778

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Theorem for monotonically decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x > f y :=
sorry

-- Theorem for minimum value on [-3, 3]
theorem f_min_value :
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f y ∧ f x = -49 :=
sorry

-- Theorem for maximum value on [-3, 3]
theorem f_max_value :
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f x ≥ f y ∧ f x = 59 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_f_min_value_f_max_value_l2357_235778


namespace NUMINAMATH_CALUDE_annas_weight_anna_weighs_80_l2357_235703

/-- The weight of Anna given Jack's weight and the balancing condition on a see-saw -/
theorem annas_weight (jack_weight : ℕ) (rock_weight : ℕ) (rock_count : ℕ) : ℕ :=
  jack_weight + rock_weight * rock_count

/-- Proof that Anna weighs 80 pounds given the conditions -/
theorem anna_weighs_80 :
  annas_weight 60 4 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_annas_weight_anna_weighs_80_l2357_235703


namespace NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l2357_235797

theorem ferris_wheel_ticket_cost 
  (initial_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (total_spent : ℕ) 
  (h1 : initial_tickets = 13)
  (h2 : remaining_tickets = 4)
  (h3 : total_spent = 81) :
  total_spent / (initial_tickets - remaining_tickets) = 9 := by
sorry

end NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l2357_235797


namespace NUMINAMATH_CALUDE_selection_options_count_l2357_235717

/-- Represents the number of people skilled in the first method -/
def skilled_in_first_method : ℕ := 5

/-- Represents the number of people skilled in the second method -/
def skilled_in_second_method : ℕ := 4

/-- Represents the total number of people -/
def total_people : ℕ := skilled_in_first_method + skilled_in_second_method

/-- Theorem: The number of ways to select one person from the group is equal to the total number of people -/
theorem selection_options_count : 
  (skilled_in_first_method + skilled_in_second_method) = total_people := by
  sorry

end NUMINAMATH_CALUDE_selection_options_count_l2357_235717


namespace NUMINAMATH_CALUDE_sequence_convergence_condition_l2357_235737

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The limit of a sequence -/
def HasLimit (s : Sequence) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |s n - l| < ε

/-- The condition on the sequence -/
def SequenceCondition (a b : ℝ) (x : Sequence) : Prop :=
  HasLimit (fun n => a * x (n + 1) - b * x n) 0

/-- The main theorem -/
theorem sequence_convergence_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : Sequence, SequenceCondition a b x → HasLimit x 0) ↔
  (a = 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ |b / a| < 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_condition_l2357_235737


namespace NUMINAMATH_CALUDE_circle_angle_theorem_l2357_235702

/-- The number of angles not greater than 120° in a circle with n points -/
def S (n : ℕ) : ℕ := sorry

/-- The minimum number of points required for a given S(n) -/
def n_min (s : ℕ) : ℕ := sorry

theorem circle_angle_theorem :
  (∀ n : ℕ, n ≥ 3 → 
    (∀ k : ℕ, k ≥ 1 →
      (2 * Nat.choose k 2 < S n ∧ S n ≤ Nat.choose k 2 + Nat.choose (k + 1) 2) →
        n_min (S n) = 2 * k + 1)) ∧
  (∀ n : ℕ, n ≥ 3 →
    (∀ k : ℕ, k ≥ 2 →
      (Nat.choose (k - 1) 2 + Nat.choose k 2 < S n ∧ S n ≤ 2 * Nat.choose k 2) →
        n_min (S n) = 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_circle_angle_theorem_l2357_235702


namespace NUMINAMATH_CALUDE_martha_cakes_l2357_235735

/-- The number of cakes Martha needs to buy -/
def total_cakes (num_children : Float) (cakes_per_child : Float) : Float :=
  num_children * cakes_per_child

/-- Theorem: Martha needs to buy 54 cakes -/
theorem martha_cakes : total_cakes 3.0 18.0 = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l2357_235735


namespace NUMINAMATH_CALUDE_speaking_sequences_count_l2357_235781

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def totalStudents : ℕ := 6

/-- The number of speakers to be selected -/
def speakersToSelect : ℕ := 4

/-- The number of specific students (A and B) -/
def specificStudents : ℕ := 2

theorem speaking_sequences_count :
  (choose specificStudents 1 * choose (totalStudents - specificStudents) (speakersToSelect - 1) * arrange speakersToSelect speakersToSelect) +
  (choose specificStudents 2 * choose (totalStudents - specificStudents) (speakersToSelect - 2) * arrange speakersToSelect speakersToSelect) = 336 :=
by sorry

end NUMINAMATH_CALUDE_speaking_sequences_count_l2357_235781


namespace NUMINAMATH_CALUDE_min_cuts_for_hendecagons_l2357_235760

/-- Represents a polygon on the table --/
structure Polygon :=
  (sides : ℕ)

/-- Represents the state of the table after some cuts --/
structure TableState :=
  (polygons : List Polygon)

/-- Performs a single straight cut on the table --/
def makeCut (state : TableState) : TableState :=
  sorry

/-- Checks if the table state contains at least 252 hendecagons --/
def hasEnoughHendecagons (state : TableState) : Prop :=
  (state.polygons.filter (λ p => p.sides = 11)).length ≥ 252

/-- The minimum number of cuts needed to create at least 252 hendecagons --/
def minCuts : ℕ := 2015

theorem min_cuts_for_hendecagons :
  ∀ (initialState : TableState),
    initialState.polygons = [Polygon.mk 4] →
    ∀ (n : ℕ),
      (∃ (finalState : TableState),
        (Nat.iterate makeCut n initialState = finalState) ∧
        hasEnoughHendecagons finalState) →
      n ≥ minCuts :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_hendecagons_l2357_235760


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l2357_235777

theorem difference_of_reciprocals (x y : ℝ) : 
  x = Real.sqrt 5 - 1 → y = Real.sqrt 5 + 1 → 1 / x - 1 / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l2357_235777


namespace NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_one_l2357_235770

theorem sqrt_real_implies_x_leq_one (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) → x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_one_l2357_235770


namespace NUMINAMATH_CALUDE_tv_screen_area_l2357_235771

theorem tv_screen_area (width height area : ℝ) : 
  width = 3 ∧ height = 7 ∧ area = width * height → area = 21 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_area_l2357_235771


namespace NUMINAMATH_CALUDE_product_equality_implies_sum_l2357_235794

theorem product_equality_implies_sum (g h : ℚ) : 
  (∀ d : ℚ, (8*d^2 - 5*d + g) * (2*d^2 + h*d - 9) = 16*d^4 + 21*d^3 - 73*d^2 - 41*d + 45) →
  g + h = -82/25 := by
sorry

end NUMINAMATH_CALUDE_product_equality_implies_sum_l2357_235794


namespace NUMINAMATH_CALUDE_solution_count_3x_4y_815_l2357_235762

theorem solution_count_3x_4y_815 : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 815 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 816) (Finset.range 816))).card = 68 :=
by sorry

end NUMINAMATH_CALUDE_solution_count_3x_4y_815_l2357_235762


namespace NUMINAMATH_CALUDE_gcd_lcm_3869_6497_l2357_235790

theorem gcd_lcm_3869_6497 :
  (Nat.gcd 3869 6497 = 73) ∧
  (Nat.lcm 3869 6497 = 344341) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_3869_6497_l2357_235790


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l2357_235776

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to define the specifics of a regular polygon for this statement
  -- Just the number of sides is sufficient

/-- The number of diagonals that can be drawn from a single vertex in a regular polygon -/
def diagonalsFromVertex (p : RegularPolygon n) : ℕ := n - 3

/-- Theorem: In a regular decagon, 7 diagonals can be drawn from any vertex -/
theorem decagon_diagonals_from_vertex :
  ∀ (p : RegularPolygon 10), diagonalsFromVertex p = 7 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l2357_235776


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l2357_235711

theorem sqrt_18_div_sqrt_2_equals_3 : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l2357_235711


namespace NUMINAMATH_CALUDE_unique_number_with_specific_factors_l2357_235741

theorem unique_number_with_specific_factors :
  ∀ (x n : ℕ),
  x = 7^n + 1 →
  Odd n →
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_factors_l2357_235741


namespace NUMINAMATH_CALUDE_average_increase_calculation_l2357_235793

theorem average_increase_calculation (current_matches : ℕ) (current_average : ℚ) (next_match_score : ℕ) : 
  current_matches = 10 →
  current_average = 34 →
  next_match_score = 78 →
  (current_matches + 1) * (current_average + (next_match_score - current_matches * current_average) / (current_matches + 1)) = 
  current_matches * current_average + next_match_score →
  (next_match_score - current_matches * current_average) / (current_matches + 1) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_average_increase_calculation_l2357_235793


namespace NUMINAMATH_CALUDE_negation_of_existence_square_greater_than_self_negation_l2357_235720

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬ ∃ x : ℕ, P x) ↔ (∀ x : ℕ, ¬ P x) := by sorry

theorem square_greater_than_self_negation :
  (¬ ∃ x : ℕ, x^2 ≤ x) ↔ (∀ x : ℕ, x^2 > x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_greater_than_self_negation_l2357_235720


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2357_235719

theorem sum_of_cubes : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2357_235719


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2357_235716

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k^2) → 2 ∣ n → 3 ∣ n → 5 ∣ n → n ≥ 900 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2357_235716
