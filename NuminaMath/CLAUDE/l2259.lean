import Mathlib

namespace NUMINAMATH_CALUDE_sugar_required_for_cake_l2259_225995

/-- Given a recipe for a cake, prove the amount of sugar required -/
theorem sugar_required_for_cake (total_flour : ℕ) (flour_added : ℕ) (extra_sugar : ℕ) : 
  total_flour = 9 → 
  flour_added = 4 → 
  extra_sugar = 6 → 
  (total_flour - flour_added) + extra_sugar = 11 := by
  sorry

end NUMINAMATH_CALUDE_sugar_required_for_cake_l2259_225995


namespace NUMINAMATH_CALUDE_fifteenSidedFigureArea_l2259_225974

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
structure Polygon where
  vertices : List Point

/-- The area of a polygon -/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- The fifteen-sided figure defined in the problem -/
def fifteenSidedFigure : Polygon :=
  { vertices := [
      {x := 1, y := 1}, {x := 1, y := 3}, {x := 3, y := 5}, {x := 4, y := 5},
      {x := 5, y := 4}, {x := 5, y := 3}, {x := 6, y := 3}, {x := 6, y := 2},
      {x := 5, y := 1}, {x := 4, y := 1}, {x := 3, y := 2}, {x := 2, y := 2},
      {x := 1, y := 1}
    ]
  }

/-- Theorem stating that the area of the fifteen-sided figure is 11 cm² -/
theorem fifteenSidedFigureArea : area fifteenSidedFigure = 11 := by sorry

end NUMINAMATH_CALUDE_fifteenSidedFigureArea_l2259_225974


namespace NUMINAMATH_CALUDE_rupert_age_rupert_candles_l2259_225945

-- Define Peter's age
def peter_age : ℕ := 10

-- Define the ratio of Rupert's age to Peter's age
def age_ratio : ℚ := 7/2

-- Theorem to prove Rupert's age
theorem rupert_age : ℕ := by
  -- The proof goes here
  sorry

-- Theorem to prove the number of candles on Rupert's cake
theorem rupert_candles : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_rupert_age_rupert_candles_l2259_225945


namespace NUMINAMATH_CALUDE_factor_sum_l2259_225948

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by sorry

end NUMINAMATH_CALUDE_factor_sum_l2259_225948


namespace NUMINAMATH_CALUDE_lemon_cupcakes_left_at_home_l2259_225953

/-- Proves that the number of lemon cupcakes left at home is 2 -/
theorem lemon_cupcakes_left_at_home 
  (total_baked : ℕ) 
  (boxes_given : ℕ) 
  (cupcakes_per_box : ℕ) 
  (h1 : total_baked = 53) 
  (h2 : boxes_given = 17) 
  (h3 : cupcakes_per_box = 3) : 
  total_baked - (boxes_given * cupcakes_per_box) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemon_cupcakes_left_at_home_l2259_225953


namespace NUMINAMATH_CALUDE_chip_paper_usage_l2259_225944

/-- Calculates the number of packs of paper Chip will use during the semester --/
def calculate_packs_of_paper (pages_per_pack : ℕ) (regular_weeks : ℕ) (short_weeks : ℕ) 
  (pages_per_regular_week : ℕ) (pages_per_short_week : ℕ) : ℕ :=
  let total_pages := regular_weeks * pages_per_regular_week + short_weeks * pages_per_short_week
  ((total_pages + pages_per_pack - 1) / pages_per_pack : ℕ)

/-- Theorem stating that Chip will use 6 packs of paper during the semester --/
theorem chip_paper_usage : 
  calculate_packs_of_paper 100 13 3 40 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_chip_paper_usage_l2259_225944


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l2259_225926

theorem least_five_digit_congruent_to_8_mod_17 :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (n % 17 = 8) ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 8 → m ≥ n) ∧
    n = 10004 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l2259_225926


namespace NUMINAMATH_CALUDE_worker_payment_l2259_225914

/-- Calculate the total amount paid to a worker for a week -/
theorem worker_payment (daily_wage : ℝ) (days_worked : List ℝ) : 
  daily_wage = 20 →
  days_worked = [11, 32, 31, 8.3, 4] →
  (daily_wage * (days_worked.sum)) = 1726 := by
sorry

end NUMINAMATH_CALUDE_worker_payment_l2259_225914


namespace NUMINAMATH_CALUDE_line_equation_l2259_225967

/-- Circle with center (3, 5) and radius √5 -/
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 5)^2 = 5}

/-- Line passing through the center of circle C -/
structure Line where
  k : ℝ
  eq : ℝ × ℝ → Prop := fun p => p.2 - 5 = k * (p.1 - 3)

/-- Point where the line intersects the y-axis -/
def P (l : Line) : ℝ × ℝ := (0, 5 - 3 * l.k)

/-- Intersection points of the line and the circle -/
def intersectionPoints (l : Line) : Set (ℝ × ℝ) :=
  {p ∈ C | l.eq p}

/-- A is the midpoint of PB -/
def isMidpoint (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

theorem line_equation (l : Line) (A B : ℝ × ℝ) 
  (hA : A ∈ intersectionPoints l) (hB : B ∈ intersectionPoints l)
  (hMid : isMidpoint A B (P l)) :
  (l.k = 2 ∧ l.eq = fun p => 2 * p.1 - p.2 - 1 = 0) ∨
  (l.k = -2 ∧ l.eq = fun p => 2 * p.1 + p.2 + 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2259_225967


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2259_225941

theorem unique_solution_condition (a b : ℝ) : 
  (∃! x : ℝ, 5 * x - 7 + a = 2 * b * x + 3) ↔ (a ≠ 10 ∧ b ≠ 5/2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2259_225941


namespace NUMINAMATH_CALUDE_sqrt_calculations_l2259_225909

theorem sqrt_calculations : 
  (2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2) ∧ 
  ((Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l2259_225909


namespace NUMINAMATH_CALUDE_work_completion_time_increase_l2259_225943

theorem work_completion_time_increase 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (removed_men : ℕ) 
  (h1 : initial_men = 100) 
  (h2 : initial_days = 20) 
  (h3 : removed_men = 50) : 
  (initial_men * initial_days) / (initial_men - removed_men) - initial_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_increase_l2259_225943


namespace NUMINAMATH_CALUDE_sqrt_144_divided_by_6_l2259_225936

theorem sqrt_144_divided_by_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_divided_by_6_l2259_225936


namespace NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_largest_n_where_B_leq_A_l2259_225910

-- Define A(n) for Alphonse's jumps
def A (n : ℕ) : ℕ :=
  n / 8 + n % 8

-- Define B(n) for Beryl's jumps
def B (n : ℕ) : ℕ :=
  n / 7 + n % 7

-- Part (a)
theorem exists_n_where_B_less_than_A :
  ∃ n : ℕ, n > 200 ∧ B n < A n :=
sorry

-- Part (b)
theorem largest_n_where_B_leq_A :
  ∀ n : ℕ, B n ≤ A n → n ≤ 343 :=
sorry

end NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_largest_n_where_B_leq_A_l2259_225910


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l2259_225904

theorem largest_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l2259_225904


namespace NUMINAMATH_CALUDE_cookie_problem_l2259_225935

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The additional number of cookies in boxes compared to bags -/
def additional_cookies : ℕ := 33

/-- The number of bags -/
def num_bags : ℕ := 9

theorem cookie_problem :
  cookies_per_box * num_boxes = cookies_per_bag * num_bags + additional_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l2259_225935


namespace NUMINAMATH_CALUDE_least_value_quadratic_l2259_225911

theorem least_value_quadratic (y : ℝ) : 
  (5 * y^2 + 7 * y + 3 = 6) → y ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l2259_225911


namespace NUMINAMATH_CALUDE_notebook_price_is_3_l2259_225940

-- Define the prices as real numbers
variable (pencil_price notebook_price : ℝ)

-- Define the purchase equations
def xiaohong_purchase : Prop :=
  4 * pencil_price + 5 * notebook_price = 15.8

def xiaoliang_purchase : Prop :=
  4 * pencil_price + 7 * notebook_price = 21.8

-- Theorem statement
theorem notebook_price_is_3
  (h1 : xiaohong_purchase pencil_price notebook_price)
  (h2 : xiaoliang_purchase pencil_price notebook_price) :
  notebook_price = 3 := by sorry

end NUMINAMATH_CALUDE_notebook_price_is_3_l2259_225940


namespace NUMINAMATH_CALUDE_tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3_l2259_225925

theorem tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3 (θ : Real) 
  (h : Real.tan θ = Real.sqrt 3) : 
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3_l2259_225925


namespace NUMINAMATH_CALUDE_min_value_implies_a_possible_a_set_l2259_225906

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The theorem stating the possible values of a -/
theorem min_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (a - 2) (a + 2), f x ≥ 6) ∧ 
  (∃ x ∈ Set.Icc (a - 2) (a + 2), f x = 6) →
  a = -3 ∨ a = 5 := by
  sorry

/-- The set of possible values for a -/
def possible_a : Set ℝ := {-3, 5}

/-- The theorem stating that the set of possible values for a is {-3, 5} -/
theorem possible_a_set : 
  ∀ a : ℝ, (∀ x ∈ Set.Icc (a - 2) (a + 2), f x ≥ 6) ∧ 
            (∃ x ∈ Set.Icc (a - 2) (a + 2), f x = 6) ↔ 
            a ∈ possible_a := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_possible_a_set_l2259_225906


namespace NUMINAMATH_CALUDE_triangle_side_range_l2259_225969

theorem triangle_side_range (a b c : ℝ) : 
  c = 4 → -- Given condition: one side has length 4
  a > 0 → -- Positive length
  b > 0 → -- Positive length
  a ≤ b → -- Assume a is the shorter of the two variable sides
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  a < 4 * Real.sqrt 2 -- Upper bound of the range
  ∧ a > 0 -- Lower bound of the range
  := by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2259_225969


namespace NUMINAMATH_CALUDE_quadratic_roots_l2259_225999

def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*a*x + c

theorem quadratic_roots (a c : ℝ) (h : a ≠ 0) :
  (quadratic_function a c (-1) = 0) →
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧
    ∀ x : ℝ, quadratic_function a c x = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2259_225999


namespace NUMINAMATH_CALUDE_marbles_lost_l2259_225954

/-- 
Given that Josh initially had 9 marbles and now has 4 marbles,
prove that the number of marbles he lost is 5.
-/
theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 9 → current = 4 → lost = initial - current → lost = 5 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l2259_225954


namespace NUMINAMATH_CALUDE_octal_arithmetic_equality_l2259_225912

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition operation for octal numbers --/
def octal_add : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Subtraction operation for octal numbers --/
def octal_sub : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Conversion from decimal to octal --/
def to_octal : ℕ → OctalNumber := sorry

/-- Theorem: In base 8, 5234₈ - 127₈ + 235₈ = 5344₈ --/
theorem octal_arithmetic_equality :
  octal_sub (octal_add (to_octal 5234) (to_octal 235)) (to_octal 127) = to_octal 5344 := by
  sorry

end NUMINAMATH_CALUDE_octal_arithmetic_equality_l2259_225912


namespace NUMINAMATH_CALUDE_abc_sum_l2259_225968

theorem abc_sum (a b c : ℕ+) (h : (139 : ℚ) / 22 = a + 1 / (b + 1 / c)) : 
  (a : ℕ) + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l2259_225968


namespace NUMINAMATH_CALUDE_sum_of_squares_coefficients_l2259_225952

theorem sum_of_squares_coefficients 
  (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) : 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_coefficients_l2259_225952


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l2259_225986

/-- Given five collinear points A, B, C, D, and E in that order, with specified distances between them,
    this function calculates the sum of squared distances from these points to any point P on the line. -/
def sum_of_squared_distances (x : ℝ) : ℝ :=
  x^2 + (x - 3)^2 + (x - 4)^2 + (x - 9)^2 + (x - 13)^2

/-- The theorem states that the minimum value of the sum of squared distances
    from five collinear points to any point on their line is 170.24,
    given specific distances between the points. -/
theorem min_sum_squared_distances :
  ∃ (min : ℝ), min = 170.24 ∧
  ∀ (x : ℝ), sum_of_squared_distances x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l2259_225986


namespace NUMINAMATH_CALUDE_special_sequence_growth_l2259_225907

/-- A sequence of positive integers satisfying the given condition -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ i ≥ 1, Nat.gcd (a i) (a (i + 1)) > a (i - 1))

/-- The main theorem: for any special sequence, each term is at least 2^n -/
theorem special_sequence_growth (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ n, a n ≥ 2^n := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_growth_l2259_225907


namespace NUMINAMATH_CALUDE_probability_to_reach_3_3_l2259_225937

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a direction of movement --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- Calculates the probability of reaching the target point from the start point
    in the given number of steps or fewer --/
def probability_to_reach (start : Point) (target : Point) (max_steps : Nat) : Rat :=
  sorry

/-- The main theorem stating the probability of reaching (3,3) from (0,0) in 8 or fewer steps --/
theorem probability_to_reach_3_3 :
  probability_to_reach ⟨0, 0⟩ ⟨3, 3⟩ 8 = 55 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_to_reach_3_3_l2259_225937


namespace NUMINAMATH_CALUDE_existence_of_c_l2259_225905

theorem existence_of_c (n : ℕ) (a b : Fin n → ℝ) 
  (h_n : n ≥ 2)
  (h_pos : ∀ i, a i > 0 ∧ b i > 0)
  (h_less : ∀ i, a i < b i)
  (h_sum : (Finset.sum Finset.univ b) < 1 + (Finset.sum Finset.univ a)) :
  ∃ c : ℝ, ∀ (i : Fin n) (k : ℤ), (a i + c + k) * (b i + c + k) > 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_c_l2259_225905


namespace NUMINAMATH_CALUDE_unique_number_of_children_l2259_225978

theorem unique_number_of_children : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_unique_number_of_children_l2259_225978


namespace NUMINAMATH_CALUDE_factorization_proof_l2259_225991

theorem factorization_proof (a b x y : ℝ) : x * (a + b) - 2 * y * (a + b) = (a + b) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2259_225991


namespace NUMINAMATH_CALUDE_symmetric_polynomial_value_l2259_225930

/-- Given a function f(x) = (x² + 3x)(x² + ax + b) where f(x) = f(2-x) for all real x, prove f(3) = -18 -/
theorem symmetric_polynomial_value (a b : ℝ) :
  (∀ x : ℝ, (x^2 + 3*x) * (x^2 + a*x + b) = ((2-x)^2 + 3*(2-x)) * ((2-x)^2 + a*(2-x) + b)) →
  (3^2 + 3*3) * (3^2 + a*3 + b) = -18 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_value_l2259_225930


namespace NUMINAMATH_CALUDE_total_emails_received_l2259_225921

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 5

theorem total_emails_received : morning_emails + afternoon_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_emails_received_l2259_225921


namespace NUMINAMATH_CALUDE_harpers_rubber_bands_l2259_225997

/-- Harper's rubber band problem -/
theorem harpers_rubber_bands :
  ∀ (h : ℕ),                        -- h represents Harper's number of rubber bands
  (h + (h - 6) = 24) →              -- Total rubber bands condition
  h = 15 := by                      -- Prove that Harper has 15 rubber bands
sorry


end NUMINAMATH_CALUDE_harpers_rubber_bands_l2259_225997


namespace NUMINAMATH_CALUDE_trajectory_of_Q_equation_l2259_225934

/-- The trajectory of point Q given the conditions in the problem -/
def trajectory_of_Q (x y : ℝ) : Prop :=
  2 * x - y + 5 = 0

/-- The line on which point P moves -/
def line_of_P (x y : ℝ) : Prop :=
  2 * x - y + 3 = 0

/-- Point M is fixed at (-1, 2) -/
def point_M : ℝ × ℝ := (-1, 2)

/-- Q is on the extension line of PM and PM = MQ -/
def Q_condition (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q.1 - point_M.1 = t * (point_M.1 - P.1) ∧ 
                    Q.2 - point_M.2 = t * (point_M.2 - P.2)

theorem trajectory_of_Q_equation :
  ∀ x y : ℝ, 
    (∃ P : ℝ × ℝ, line_of_P P.1 P.2 ∧ Q_condition P (x, y)) →
    trajectory_of_Q x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_equation_l2259_225934


namespace NUMINAMATH_CALUDE_rectangle_length_l2259_225960

/-- Given a rectangle with perimeter 680 meters and breadth 82 meters, its length is 258 meters. -/
theorem rectangle_length (perimeter breadth : ℝ) (h1 : perimeter = 680) (h2 : breadth = 82) :
  (perimeter / 2) - breadth = 258 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_l2259_225960


namespace NUMINAMATH_CALUDE_smallest_repeating_block_length_l2259_225987

/-- The number of digits in the smallest repeating block of the decimal expansion of 3/11 -/
def repeating_block_length : ℕ := 2

/-- The fraction we are considering -/
def fraction : ℚ := 3 / 11

theorem smallest_repeating_block_length :
  repeating_block_length = 2 ∧
  ∀ n : ℕ, n < repeating_block_length →
    ¬∃ (a b : ℕ), fraction = (a : ℚ) / (10^n : ℚ) + (b : ℚ) / (10^n - 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_length_l2259_225987


namespace NUMINAMATH_CALUDE_exists_k_all_multiples_contain_all_digits_l2259_225998

/-- For a given positive integer, check if its decimal representation contains all digits from 0 to 9 -/
def containsAllDigits (n : ℕ+) : Prop := sorry

/-- For a given positive integer k and a set of positive integers, check if k*i contains all digits for all i in the set -/
def allMultiplesContainAllDigits (k : ℕ+) (s : Set ℕ+) : Prop :=
  ∀ i ∈ s, containsAllDigits (i * k)

/-- Main theorem: For all positive integers n, there exists a positive integer k such that
    k, 2k, ..., nk all contain all digits from 0 to 9 in their decimal representations -/
theorem exists_k_all_multiples_contain_all_digits (n : ℕ+) :
  ∃ k : ℕ+, allMultiplesContainAllDigits k (Set.Icc 1 n) := by sorry

end NUMINAMATH_CALUDE_exists_k_all_multiples_contain_all_digits_l2259_225998


namespace NUMINAMATH_CALUDE_unique_x_with_rational_sums_l2259_225929

theorem unique_x_with_rational_sums (x : ℝ) :
  (∃ a : ℚ, x + Real.sqrt 3 = a) ∧ 
  (∃ b : ℚ, x^2 + Real.sqrt 3 = b) →
  x = 1/2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_rational_sums_l2259_225929


namespace NUMINAMATH_CALUDE_extremum_maximum_at_negative_one_l2259_225913

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem stating that x = -1 is the extremum maximum point of f(x) --/
theorem extremum_maximum_at_negative_one :
  ∃ (a : ℝ), a = -1 ∧ 
  (∀ x : ℝ, f x ≤ f a) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → f x < f a) :=
sorry

end NUMINAMATH_CALUDE_extremum_maximum_at_negative_one_l2259_225913


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l2259_225915

/-- A rectangular plot with given perimeter and short side length has a specific ratio of long to short sides -/
theorem rectangular_plot_ratio (perimeter : ℝ) (short_side : ℝ) 
  (h_perimeter : perimeter = 640) 
  (h_short_side : short_side = 80) : 
  (perimeter / 2 - short_side) / short_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l2259_225915


namespace NUMINAMATH_CALUDE_cut_square_equation_l2259_225972

/-- Represents the dimensions of a rectangular sheet and the side length of squares cut from its corners. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ
  cutSide : ℝ

/-- Calculates the area of the base of a box formed by cutting squares from a sheet's corners. -/
def baseArea (d : SheetDimensions) : ℝ :=
  (d.length - 2 * d.cutSide) * (d.width - 2 * d.cutSide)

/-- Calculates the original area of a rectangular sheet. -/
def originalArea (d : SheetDimensions) : ℝ :=
  d.length * d.width

/-- Theorem stating the relationship between the cut side length and the resulting box dimensions. -/
theorem cut_square_equation (d : SheetDimensions) 
    (h1 : d.length = 8)
    (h2 : d.width = 6)
    (h3 : baseArea d = (2/3) * originalArea d) :
  d.cutSide ^ 2 - 7 * d.cutSide + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cut_square_equation_l2259_225972


namespace NUMINAMATH_CALUDE_green_room_fraction_l2259_225958

theorem green_room_fraction (total_rooms : ℕ) (walls_per_room : ℕ) (purple_walls : ℕ) :
  total_rooms = 10 →
  walls_per_room = 8 →
  purple_walls = 32 →
  (total_rooms : ℚ) - (purple_walls / walls_per_room : ℚ) = 3/5 * total_rooms :=
by sorry

end NUMINAMATH_CALUDE_green_room_fraction_l2259_225958


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2259_225939

-- Define the arithmetic sequence
def arithmetic_seq (A d : ℝ) (k : ℕ) : ℝ := A + k * d

-- Define the geometric sequence
def geometric_seq (B q : ℝ) (k : ℕ) : ℝ := B * q ^ k

-- Main theorem
theorem arithmetic_geometric_sequence_property
  (a b c : ℝ) (m n p : ℕ) (A d B q : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (ha_arith : a = arithmetic_seq A d m)
  (hb_arith : b = arithmetic_seq A d n)
  (hc_arith : c = arithmetic_seq A d p)
  (ha_geom : a = geometric_seq B q m)
  (hb_geom : b = geometric_seq B q n)
  (hc_geom : c = geometric_seq B q p) :
  a ^ (b - c) * b ^ (c - a) * c ^ (a - b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2259_225939


namespace NUMINAMATH_CALUDE_S_subset_T_l2259_225955

open Set Real

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ k : ℤ, p.1^2 - p.2^2 = 2*k + 1}

def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | sin (2*π*p.1^2) - sin (2*π*p.2^2) = cos (2*π*p.1^2) - cos (2*π*p.2^2)}

theorem S_subset_T : S ⊆ T := by
  sorry

end NUMINAMATH_CALUDE_S_subset_T_l2259_225955


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2259_225901

-- Define the triangles and their sides
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Given values
def ABC : Triangle := { A := 15, B := 0, C := 24 }
def FGH : Triangle := { A := 0, B := 0, C := 18 }

-- Theorem statement
theorem similar_triangles_side_length :
  similar ABC FGH →
  FGH.A = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2259_225901


namespace NUMINAMATH_CALUDE_number_problem_l2259_225947

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 56 ∧ x = 140 := by sorry

end NUMINAMATH_CALUDE_number_problem_l2259_225947


namespace NUMINAMATH_CALUDE_triangle_inequality_ratio_l2259_225976

theorem triangle_inequality_ratio (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≥ 1 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    (a'^2 + b'^2 + c'^2) / (a'*b' + b'*c' + c'*a') = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_ratio_l2259_225976


namespace NUMINAMATH_CALUDE_periodic_odd_function_sum_l2259_225928

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_sum (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f)
  (h_def : ∀ x, 0 < x → x < 1 → f x = 4^x) :
  f (-5/2) + f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_sum_l2259_225928


namespace NUMINAMATH_CALUDE_basketball_win_rate_l2259_225963

theorem basketball_win_rate (games_won : ℕ) (first_games : ℕ) (remaining_games : ℕ) (target_win_rate : ℚ) : 
  games_won = 45 →
  first_games = 60 →
  remaining_games = 54 →
  target_win_rate = 3/4 →
  ∃ (additional_wins : ℕ), 
    (games_won + additional_wins : ℚ) / (first_games + remaining_games : ℚ) = target_win_rate ∧
    additional_wins = 41 :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l2259_225963


namespace NUMINAMATH_CALUDE_bicycle_discount_l2259_225975

theorem bicycle_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 ∧ 
  discount1 = 0.4 ∧ 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_discount_l2259_225975


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l2259_225917

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_240 :
  rectangle_area 3600 10 = 240 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l2259_225917


namespace NUMINAMATH_CALUDE_program_result_l2259_225990

theorem program_result : ∀ (x : ℕ), 
  x = 51 → 
  9 < x → 
  x < 100 → 
  let a := x / 10
  let b := x % 10
  10 * b + a = 15 := by
sorry

end NUMINAMATH_CALUDE_program_result_l2259_225990


namespace NUMINAMATH_CALUDE_no_double_application_function_l2259_225996

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l2259_225996


namespace NUMINAMATH_CALUDE_ribbon_length_proof_l2259_225979

/-- Calculates the total length of a ribbon before division, given the number of students,
    length per student, and leftover length. -/
def totalRibbonLength (numStudents : ℕ) (lengthPerStudent : ℝ) (leftover : ℝ) : ℝ :=
  (numStudents : ℝ) * lengthPerStudent + leftover

/-- Proves that for 10 students, 0.84 meters per student, and 0.50 meters leftover,
    the total ribbon length before division was 8.9 meters. -/
theorem ribbon_length_proof :
  totalRibbonLength 10 0.84 0.50 = 8.9 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_length_proof_l2259_225979


namespace NUMINAMATH_CALUDE_greatest_difference_is_nine_l2259_225949

/-- A three-digit integer in the form 84x that is a multiple of 3 -/
def ValidNumber (x : ℕ) : Prop :=
  x < 10 ∧ (840 + x) % 3 = 0

/-- The set of all valid x values -/
def ValidXSet : Set ℕ :=
  {x | ValidNumber x}

/-- The greatest possible difference between two valid x values -/
theorem greatest_difference_is_nine :
  ∃ (a b : ℕ), a ∈ ValidXSet ∧ b ∈ ValidXSet ∧
    ∀ (x y : ℕ), x ∈ ValidXSet → y ∈ ValidXSet →
      (a - b : ℤ).natAbs ≥ (x - y : ℤ).natAbs ∧
      (a - b : ℤ).natAbs = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_is_nine_l2259_225949


namespace NUMINAMATH_CALUDE_one_match_theorem_one_empty_theorem_l2259_225927

/-- The number of ways to arrange 4 balls in 4 boxes with exactly one match -/
def arrange_one_match : ℕ := 8

/-- The number of ways to arrange 4 balls in 4 boxes with exactly one empty box -/
def arrange_one_empty : ℕ := 144

/-- Theorem for the number of arrangements with exactly one match -/
theorem one_match_theorem : arrange_one_match = 8 := by sorry

/-- Theorem for the number of arrangements with exactly one empty box -/
theorem one_empty_theorem : arrange_one_empty = 144 := by sorry

end NUMINAMATH_CALUDE_one_match_theorem_one_empty_theorem_l2259_225927


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2259_225985

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3*x + 2*y - z = 12) : 
  x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2259_225985


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l2259_225966

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem age_ratio_in_two_years :
  man_age_in_two_years / son_age_in_two_years = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l2259_225966


namespace NUMINAMATH_CALUDE_brandon_lost_skittles_l2259_225946

theorem brandon_lost_skittles (initial : ℕ) (final : ℕ) (lost : ℕ) : 
  initial = 96 → final = 87 → initial = final + lost → lost = 9 := by sorry

end NUMINAMATH_CALUDE_brandon_lost_skittles_l2259_225946


namespace NUMINAMATH_CALUDE_min_marked_cells_l2259_225988

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece -/
inductive LPiece
  | mk : Fin 2 → Fin 2 → LPiece

/-- Checks if an L-piece touches a marked cell on the board -/
def touchesMarkedCell (b : Board m n) (l : LPiece) (i j : ℕ) : Prop :=
  ∃ (x y : Fin 2), b.cells ⟨i + x.val, sorry⟩ ⟨j + y.val, sorry⟩ = true

/-- A marking strategy for the board -/
def markingStrategy (b : Board m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n), i.val % 2 = 0 → b.cells i j = true

/-- The main theorem stating that 50 is the smallest number of marked cells
    that ensures any L-shaped piece touches at least one marked cell on a 10 × 11 board -/
theorem min_marked_cells :
  ∀ (b : Board 10 11),
    (∃ (k : ℕ), k < 50 ∧
      (∀ (l : LPiece) (i j : ℕ), i < 9 ∧ j < 10 →
        touchesMarkedCell b l i j)) →
    (∃ (b' : Board 10 11),
      markingStrategy b' ∧
      (∀ (l : LPiece) (i j : ℕ), i < 9 ∧ j < 10 →
        touchesMarkedCell b' l i j) ∧
      (∃ (k : ℕ), k = 50 ∧
        k = (Finset.filter (fun i => b'.cells i.1 i.2) (Finset.product (Finset.range 10) (Finset.range 11))).card)) :=
by
  sorry


end NUMINAMATH_CALUDE_min_marked_cells_l2259_225988


namespace NUMINAMATH_CALUDE_circumcircle_incircle_diameter_implies_equilateral_l2259_225920

-- Define a triangle
structure Triangle where
  -- We don't need to specify the vertices, just that it's a triangle
  is_triangle : Bool

-- Define the circumcircle and incircle of a triangle
def circumcircle (t : Triangle) : ℝ := sorry
def incircle (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop := sorry

-- State the theorem
theorem circumcircle_incircle_diameter_implies_equilateral (t : Triangle) :
  circumcircle t = 2 * incircle t → is_equilateral t := by
  sorry


end NUMINAMATH_CALUDE_circumcircle_incircle_diameter_implies_equilateral_l2259_225920


namespace NUMINAMATH_CALUDE_sum_of_distances_to_intersection_points_l2259_225956

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 3
def C₂ (x y : ℝ) : Prop := y^2 = 2*x

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem sum_of_distances_to_intersection_points :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    distance P A + distance P B = 6 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_sum_of_distances_to_intersection_points_l2259_225956


namespace NUMINAMATH_CALUDE_evaluate_expression_l2259_225980

theorem evaluate_expression (a x : ℝ) (h : x = a + 5) : x - a + 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2259_225980


namespace NUMINAMATH_CALUDE_power_division_l2259_225923

theorem power_division (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2259_225923


namespace NUMINAMATH_CALUDE_basketball_series_probability_l2259_225931

/-- The probability of at least k successes in n independent trials with probability p -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem basketball_series_probability :
  prob_at_least 9 5 (1/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_basketball_series_probability_l2259_225931


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l2259_225938

/-- Given three points A, B, and C in a plane, where C divides AB in a 1:3 ratio,
    prove that the sum of coordinates of A is 21 when B and C are known. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 10) →
  C = (5, 4) →
  A.1 + A.2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l2259_225938


namespace NUMINAMATH_CALUDE_percentage_of_160_l2259_225970

theorem percentage_of_160 : (3 / 8 : ℚ) / 100 * 160 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_percentage_of_160_l2259_225970


namespace NUMINAMATH_CALUDE_fred_total_cards_l2259_225933

def initial_cards : ℕ := 26
def cards_given_away : ℕ := 18
def new_cards_found : ℕ := 40

theorem fred_total_cards : 
  initial_cards - cards_given_away + new_cards_found = 48 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_cards_l2259_225933


namespace NUMINAMATH_CALUDE_cube_root_sixteen_over_thirtytwo_l2259_225918

theorem cube_root_sixteen_over_thirtytwo : 
  (16 / 32 : ℝ)^(1/3) = 1 / 2^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_sixteen_over_thirtytwo_l2259_225918


namespace NUMINAMATH_CALUDE_probability_two_specific_people_obtain_items_l2259_225942

-- Define the number of people and items
def num_people : ℕ := 4
def num_items : ℕ := 3

-- Define the probability function
noncomputable def probability_both_obtain (n_people n_items : ℕ) : ℚ :=
  (n_items.choose 2 * (n_people - 2).choose 1) / n_people.choose n_items

-- State the theorem
theorem probability_two_specific_people_obtain_items :
  probability_both_obtain num_people num_items = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_specific_people_obtain_items_l2259_225942


namespace NUMINAMATH_CALUDE_min_points_in_S_l2259_225977

-- Define a point in the xy-plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the set S
def S : Set Point := sorry

-- Define symmetry conditions
def symmetric_origin (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk (-p.x) (-p.y) ∈ s

def symmetric_x_axis (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk p.x (-p.y) ∈ s

def symmetric_y_axis (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk (-p.x) p.y ∈ s

def symmetric_y_eq_x (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk p.y p.x ∈ s

-- Theorem statement
theorem min_points_in_S :
  symmetric_origin S ∧
  symmetric_x_axis S ∧
  symmetric_y_axis S ∧
  symmetric_y_eq_x S ∧
  Point.mk 2 3 ∈ S →
  ∃ (points : Finset Point), points.card = 8 ∧ ↑points ⊆ S ∧
    (∀ (subset : Finset Point), ↑subset ⊆ S → subset.card < 8 → subset ≠ points) :=
sorry

end NUMINAMATH_CALUDE_min_points_in_S_l2259_225977


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l2259_225984

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (1, Real.sqrt 3, 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l2259_225984


namespace NUMINAMATH_CALUDE_work_left_after_14_days_l2259_225983

/-- The fraction of work left for the first task after 14 days -/
def first_task_left : ℚ := 11/60

/-- The fraction of work left for the second task after 14 days -/
def second_task_left : ℚ := 0

/-- A's work rate per day -/
def rate_A : ℚ := 1/15

/-- B's work rate per day -/
def rate_B : ℚ := 1/20

/-- C's work rate per day -/
def rate_C : ℚ := 1/25

/-- The number of days A and B work on the first task -/
def days_first_task : ℕ := 7

/-- The total number of days -/
def total_days : ℕ := 14

theorem work_left_after_14_days :
  let work_AB_7_days := (rate_A + rate_B) * days_first_task
  let work_C_7_days := rate_C * days_first_task
  let work_ABC_7_days := (rate_A + rate_B + rate_C) * (total_days - days_first_task)
  (1 - work_AB_7_days = first_task_left) ∧
  (max 0 (1 - work_C_7_days - work_ABC_7_days) = second_task_left) := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_14_days_l2259_225983


namespace NUMINAMATH_CALUDE_parabola_and_hyperbola_equations_l2259_225957

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the conditions
axiom parabola_vertex_origin : ∃ p > 0, parabola p 0 0
axiom axis_of_symmetry : ∃ p > 0, ∀ x, parabola p x 0 → x = 1
axiom intersection_point : ∃ p a b, parabola p (3/2) (Real.sqrt 6) ∧ hyperbola a b (3/2) (Real.sqrt 6)

-- Theorem to prove
theorem parabola_and_hyperbola_equations :
  ∃ p a b, (∀ x y, parabola p x y ↔ y^2 = 4*x) ∧
           (∀ x y, hyperbola a b x y ↔ 4*x^2 - (4/3)*y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_hyperbola_equations_l2259_225957


namespace NUMINAMATH_CALUDE_six_meter_logs_more_advantageous_l2259_225916

-- Define the length of logs and the target length of chunks
def log_length_6 : ℕ := 6
def log_length_7 : ℕ := 7
def chunk_length : ℕ := 1
def total_length : ℕ := 42

-- Define the number of cuts needed for each log type
def cuts_per_log_6 : ℕ := log_length_6 - 1
def cuts_per_log_7 : ℕ := log_length_7 - 1

-- Define the number of logs needed for each type
def logs_needed_6 : ℕ := (total_length + log_length_6 - 1) / log_length_6
def logs_needed_7 : ℕ := (total_length + log_length_7 - 1) / log_length_7

-- Define the total number of cuts for each log type
def total_cuts_6 : ℕ := logs_needed_6 * cuts_per_log_6
def total_cuts_7 : ℕ := logs_needed_7 * cuts_per_log_7

-- Theorem statement
theorem six_meter_logs_more_advantageous :
  total_cuts_6 < total_cuts_7 :=
by sorry

end NUMINAMATH_CALUDE_six_meter_logs_more_advantageous_l2259_225916


namespace NUMINAMATH_CALUDE_solution_set_equivalence_minimum_value_l2259_225973

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 - n * x

-- Part 1
theorem solution_set_equivalence
  (m n t : ℝ)
  (h1 : ∀ x, f m n x ≥ t ↔ -3 ≤ x ∧ x ≤ 2) :
  ∀ x, n * x^2 + m * x + t ≤ 0 ↔ -2 ≤ x ∧ x ≤ 3 :=
sorry

-- Part 2
theorem minimum_value
  (m n : ℝ)
  (h1 : f m n 1 > 0)
  (h2 : 1 ≤ m ∧ m ≤ 3) :
  ∃ (m₀ n₀ : ℝ), 1/(m₀-n₀) + 9/m₀ - n₀ = 2 ∧
    ∀ m n, f m n 1 > 0 → 1 ≤ m ∧ m ≤ 3 → 1/(m-n) + 9/m - n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_minimum_value_l2259_225973


namespace NUMINAMATH_CALUDE_total_red_stripes_on_ten_flags_l2259_225981

/-- Represents an American flag -/
structure AmericanFlag where
  stripes : ℕ
  firstStripeRed : Bool
  halfRemainingRed : Bool

/-- Calculates the number of red stripes on a single American flag -/
def redStripesPerFlag (flag : AmericanFlag) : ℕ :=
  if flag.firstStripeRed ∧ flag.halfRemainingRed then
    1 + (flag.stripes - 1) / 2
  else
    0

/-- Theorem stating the total number of red stripes on 10 American flags -/
theorem total_red_stripes_on_ten_flags :
  ∀ (flag : AmericanFlag),
    flag.stripes = 13 →
    flag.firstStripeRed = true →
    flag.halfRemainingRed = true →
    (redStripesPerFlag flag * 10 = 70) :=
by
  sorry

end NUMINAMATH_CALUDE_total_red_stripes_on_ten_flags_l2259_225981


namespace NUMINAMATH_CALUDE_cubic_polynomial_d_value_l2259_225971

/-- Represents a cubic polynomial of the form 3x^3 + dx^2 + ex - 6 -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ

def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  3 * x^3 + p.d * x^2 + p.e * x - 6

def CubicPolynomial.productOfZeros (p : CubicPolynomial) : ℝ := 2

def CubicPolynomial.sumOfCoefficients (p : CubicPolynomial) : ℝ :=
  3 + p.d + p.e - 6

theorem cubic_polynomial_d_value (p : CubicPolynomial) :
  p.productOfZeros = 9 →
  p.sumOfCoefficients = 9 →
  p.d = -18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_d_value_l2259_225971


namespace NUMINAMATH_CALUDE_roses_flats_is_three_l2259_225993

/-- Represents the plant shop inventory and fertilizer requirements --/
structure PlantShop where
  petunia_flats : ℕ
  petunias_per_flat : ℕ
  roses_per_flat : ℕ
  venus_flytraps : ℕ
  petunia_fertilizer : ℕ
  rose_fertilizer : ℕ
  venus_flytrap_fertilizer : ℕ
  total_fertilizer : ℕ

/-- Calculates the number of flats of roses in the shop --/
def roses_flats (shop : PlantShop) : ℕ :=
  let petunia_total := shop.petunia_flats * shop.petunias_per_flat * shop.petunia_fertilizer
  let venus_total := shop.venus_flytraps * shop.venus_flytrap_fertilizer
  let roses_total := shop.total_fertilizer - petunia_total - venus_total
  roses_total / (shop.roses_per_flat * shop.rose_fertilizer)

/-- Theorem stating that the number of rose flats is 3 --/
theorem roses_flats_is_three (shop : PlantShop)
  (h1 : shop.petunia_flats = 4)
  (h2 : shop.petunias_per_flat = 8)
  (h3 : shop.roses_per_flat = 6)
  (h4 : shop.venus_flytraps = 2)
  (h5 : shop.petunia_fertilizer = 8)
  (h6 : shop.rose_fertilizer = 3)
  (h7 : shop.venus_flytrap_fertilizer = 2)
  (h8 : shop.total_fertilizer = 314) :
  roses_flats shop = 3 := by
  sorry

end NUMINAMATH_CALUDE_roses_flats_is_three_l2259_225993


namespace NUMINAMATH_CALUDE_fraction_sum_to_decimal_l2259_225992

theorem fraction_sum_to_decimal : 3/8 + 5/32 = 0.53125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_to_decimal_l2259_225992


namespace NUMINAMATH_CALUDE_product_of_numbers_l2259_225919

theorem product_of_numbers (x y : ℝ) : x + y = 60 → x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2259_225919


namespace NUMINAMATH_CALUDE_tan_equation_solution_exists_l2259_225902

open Real

theorem tan_equation_solution_exists :
  ∃! θ : ℝ, 0 < θ ∧ θ < π/6 ∧
  tan θ + tan (θ + π/6) + tan (3*θ) = 0 ∧
  0 < tan θ ∧ tan θ < 1 := by
sorry

end NUMINAMATH_CALUDE_tan_equation_solution_exists_l2259_225902


namespace NUMINAMATH_CALUDE_vector_equation_holds_l2259_225924

variable {V : Type*} [AddCommGroup V]

/-- Given points A, B, C, M, O in a vector space, 
    prove that AB + MB + BC + OM + CO = AB --/
theorem vector_equation_holds (A B C M O : V) :
  (A - B) + (M - B) + (B - C) + (O - M) + (C - O) = A - B :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_holds_l2259_225924


namespace NUMINAMATH_CALUDE_seashell_collection_l2259_225982

theorem seashell_collection (stefan vail aiguo fatima : ℕ) : 
  stefan = vail + 16 →
  vail + 5 = aiguo →
  aiguo = 20 →
  fatima = 2 * aiguo →
  stefan + vail + aiguo + fatima = 106 := by
sorry

end NUMINAMATH_CALUDE_seashell_collection_l2259_225982


namespace NUMINAMATH_CALUDE_min_value_theorem_l2259_225994

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 27) :
  18 ≤ 3 * a + 2 * b + c ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 3 * a₀ + 2 * b₀ + c₀ = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2259_225994


namespace NUMINAMATH_CALUDE_square_shadow_not_trapezoid_l2259_225900

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a shadow as a quadrilateral
structure Shadow where
  vertices : Fin 4 → ℝ × ℝ

-- Define a uniform light source
structure UniformLight where
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Define a trapezoid
def is_trapezoid (s : Shadow) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ 
    (s.vertices i).1 - (s.vertices j).1 ≠ 0 ∧
    (s.vertices ((i + 1) % 4)).1 - (s.vertices ((j + 1) % 4)).1 ≠ 0 ∧
    ((s.vertices i).2 - (s.vertices j).2) / ((s.vertices i).1 - (s.vertices j).1) =
    ((s.vertices ((i + 1) % 4)).2 - (s.vertices ((j + 1) % 4)).2) / 
    ((s.vertices ((i + 1) % 4)).1 - (s.vertices ((j + 1) % 4)).1)

-- State the theorem
theorem square_shadow_not_trapezoid 
  (square : Square) (light : UniformLight) (shadow : Shadow) :
  (∃ (projection : Square → UniformLight → Shadow), 
    projection square light = shadow) →
  ¬ is_trapezoid shadow :=
sorry

end NUMINAMATH_CALUDE_square_shadow_not_trapezoid_l2259_225900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2259_225922

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = -7)
  (h_s3 : sum_arithmetic_sequence a 3 = -15) :
  (∀ n : ℕ, a n = 2 * n - 9) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n = (n - 4)^2 - 16) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n ≥ -16) ∧
  (sum_arithmetic_sequence a 4 = -16) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2259_225922


namespace NUMINAMATH_CALUDE_bus_trip_distance_l2259_225964

theorem bus_trip_distance (v : ℝ) (d : ℝ) : 
  v = 40 → 
  d / v - d / (v + 5) = 1 → 
  d = 360 := by sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l2259_225964


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2259_225932

theorem rectangle_dimensions (length width : ℝ) : 
  (2 * length + 2 * width = 16) →  -- Perimeter is 16 cm
  (length - width = 1) →           -- Difference between length and width is 1 cm
  (length = 4.5 ∧ width = 3.5) :=  -- Length is 4.5 cm and width is 3.5 cm
by
  sorry

#check rectangle_dimensions

end NUMINAMATH_CALUDE_rectangle_dimensions_l2259_225932


namespace NUMINAMATH_CALUDE_new_average_after_grace_marks_l2259_225962

/-- Represents the grace marks distribution for different percentile ranges -/
structure GraceMarksDistribution where
  below_25th : ℕ
  between_25th_50th : ℕ
  between_50th_75th : ℕ
  above_75th : ℕ

/-- Represents the class statistics -/
structure ClassStats where
  size : ℕ
  original_average : ℝ
  standard_deviation : ℝ
  percentile_25th : ℝ
  percentile_50th : ℝ
  percentile_75th : ℝ

def calculate_new_average (stats : ClassStats) (grace_marks : GraceMarksDistribution) : ℝ :=
  sorry

theorem new_average_after_grace_marks
  (stats : ClassStats)
  (grace_marks : GraceMarksDistribution)
  (h_size : stats.size = 35)
  (h_original_avg : stats.original_average = 37)
  (h_std_dev : stats.standard_deviation = 6)
  (h_25th : stats.percentile_25th = 32)
  (h_50th : stats.percentile_50th = 37)
  (h_75th : stats.percentile_75th = 42)
  (h_grace_below_25th : grace_marks.below_25th = 6)
  (h_grace_25th_50th : grace_marks.between_25th_50th = 4)
  (h_grace_50th_75th : grace_marks.between_50th_75th = 2)
  (h_grace_above_75th : grace_marks.above_75th = 0) :
  abs (calculate_new_average stats grace_marks - 40.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_grace_marks_l2259_225962


namespace NUMINAMATH_CALUDE_xf_inequality_solution_l2259_225950

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- When x < 0, f(x) + xf'(x) < 0
def condition_negative (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x + x * (deriv f x) < 0

-- f(3) = 0
def f_3_is_0 (f : ℝ → ℝ) : Prop := f 3 = 0

-- The solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x < -3 ∨ (0 < x ∧ x < 3)}

theorem xf_inequality_solution
  (heven : even_function f)
  (hneg : condition_negative f)
  (hf3 : f_3_is_0 f) :
  {x : ℝ | x * f x > 0} = solution_set f :=
sorry

end

end NUMINAMATH_CALUDE_xf_inequality_solution_l2259_225950


namespace NUMINAMATH_CALUDE_point_k_value_l2259_225959

theorem point_k_value (A B C K : ℝ) : 
  A = -3 → B = -5 → C = 6 → 
  (A + B + C + K = -A - B - C - K) → 
  K = 2 := by sorry

end NUMINAMATH_CALUDE_point_k_value_l2259_225959


namespace NUMINAMATH_CALUDE_number_operation_proof_l2259_225961

theorem number_operation_proof (x : ℝ) : x = 115 → (((x + 45) / 2) / 2) + 45 = 85 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_proof_l2259_225961


namespace NUMINAMATH_CALUDE_norm_scale_vector_l2259_225989

theorem norm_scale_vector (u : ℝ × ℝ) : ‖u‖ = 7 → ‖(5 : ℝ) • u‖ = 35 := by
  sorry

end NUMINAMATH_CALUDE_norm_scale_vector_l2259_225989


namespace NUMINAMATH_CALUDE_rectangle_area_l2259_225965

/-- The area of a rectangle with perimeter 60 and width 10 is 200 -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 60) (h2 : width = 10) :
  2 * (perimeter / 2 - width) * width = 200 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2259_225965


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2259_225903

theorem quadratic_always_positive (a : ℝ) (h : a > (1/2)) :
  ∀ x : ℝ, a * x^2 + x + (1/2) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2259_225903


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2259_225951

def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}

def B : Set ℝ := {x | Real.exp (x * Real.log 3) > 9}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2259_225951


namespace NUMINAMATH_CALUDE_divisibility_of_factorials_l2259_225908

theorem divisibility_of_factorials (n : ℕ+) : 
  ∃ k : ℤ, 2 * (3 * n.val).factorial = k * n.val.factorial * (n.val + 1).factorial * (n.val + 2).factorial := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_factorials_l2259_225908
