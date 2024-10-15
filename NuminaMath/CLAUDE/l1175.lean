import Mathlib

namespace NUMINAMATH_CALUDE_percentage_difference_l1175_117518

theorem percentage_difference (x y : ℝ) (h : y = 1.25 * x) : 
  x = 0.8 * y := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1175_117518


namespace NUMINAMATH_CALUDE_election_votes_l1175_117529

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) 
  (a_excess_percent : ℚ) (c_percent : ℚ) 
  (h_total : total_votes = 6800)
  (h_invalid : invalid_percent = 30 / 100)
  (h_a_excess : a_excess_percent = 18 / 100)
  (h_c : c_percent = 12 / 100) : 
  ∃ (b_votes c_votes : ℕ), 
    b_votes + c_votes = 2176 ∧ 
    b_votes + c_votes + (b_votes + (a_excess_percent * total_votes).floor) = 
      (total_votes * (1 - invalid_percent)).floor ∧
    c_votes = (c_percent * total_votes).floor := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l1175_117529


namespace NUMINAMATH_CALUDE_prob_neither_chooses_D_l1175_117534

/-- Represents the four projects --/
inductive Project : Type
  | A
  | B
  | C
  | D

/-- Represents the outcome of both students' choices --/
structure Outcome :=
  (fanfan : Project)
  (lelle : Project)

/-- The set of all possible outcomes --/
def all_outcomes : Finset Outcome :=
  sorry

/-- The set of outcomes where neither student chooses project D --/
def outcomes_without_D : Finset Outcome :=
  sorry

/-- The probability of an event is the number of favorable outcomes
    divided by the total number of outcomes --/
def probability (event : Finset Outcome) : Rat :=
  (event.card : Rat) / (all_outcomes.card : Rat)

/-- The main theorem: probability of neither student choosing D is 1/2 --/
theorem prob_neither_chooses_D :
  probability outcomes_without_D = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_prob_neither_chooses_D_l1175_117534


namespace NUMINAMATH_CALUDE_gravel_cost_is_correct_l1175_117555

/-- Represents the dimensions and cost parameters of a rectangular plot with a gravel path --/
structure PlotWithPath where
  length : Real
  width : Real
  pathWidth : Real
  gravelCost : Real

/-- Calculates the cost of gravelling the path for a given plot --/
def calculateGravellingCost (plot : PlotWithPath) : Real :=
  let outerArea := plot.length * plot.width
  let innerLength := plot.length - 2 * plot.pathWidth
  let innerWidth := plot.width - 2 * plot.pathWidth
  let innerArea := innerLength * innerWidth
  let pathArea := outerArea - innerArea
  pathArea * plot.gravelCost

/-- Theorem stating that the cost of gravelling the path for the given plot is 8.844 rupees --/
theorem gravel_cost_is_correct (plot : PlotWithPath) 
  (h1 : plot.length = 110)
  (h2 : plot.width = 0.65)
  (h3 : plot.pathWidth = 0.05)
  (h4 : plot.gravelCost = 0.8) :
  calculateGravellingCost plot = 8.844 := by
  sorry

#eval calculateGravellingCost { length := 110, width := 0.65, pathWidth := 0.05, gravelCost := 0.8 }

end NUMINAMATH_CALUDE_gravel_cost_is_correct_l1175_117555


namespace NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l1175_117538

theorem smallest_multiple_of_one_to_five : ∃ n : ℕ+, 
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 → m ∣ n) ∧
  (∀ k : ℕ+, (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 → m ∣ k) → n ≤ k) ∧
  n = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l1175_117538


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1175_117526

def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 2

theorem f_decreasing_on_interval (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a b x > f a b y) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1175_117526


namespace NUMINAMATH_CALUDE_both_reunions_count_l1175_117595

/-- The number of people attending both reunions -/
def both_reunions (total guests : ℕ) (oates hall : ℕ) : ℕ :=
  total - (oates + hall - total)

theorem both_reunions_count : both_reunions 150 70 52 = 28 := by
  sorry

end NUMINAMATH_CALUDE_both_reunions_count_l1175_117595


namespace NUMINAMATH_CALUDE_consecutive_triangle_sides_l1175_117530

theorem consecutive_triangle_sides (n : ℕ) (h : n ≥ 1) :
  ∀ (a : ℕ → ℕ), (∀ i, i < 6*n → a (i+1) = a i + 1) →
  ∃ i, i + 2 < 6*n ∧ 
    a i + a (i+1) > a (i+2) ∧
    a i + a (i+2) > a (i+1) ∧
    a (i+1) + a (i+2) > a i :=
sorry

end NUMINAMATH_CALUDE_consecutive_triangle_sides_l1175_117530


namespace NUMINAMATH_CALUDE_root_product_sum_l1175_117564

-- Define the polynomial
def f (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 17 * x - 7

-- Define the roots
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- State the theorem
theorem root_product_sum :
  f p = 0 ∧ f q = 0 ∧ f r = 0 →
  p * q + p * r + q * r = 17 / 5 := by
  sorry

end NUMINAMATH_CALUDE_root_product_sum_l1175_117564


namespace NUMINAMATH_CALUDE_inequality_proof_l1175_117501

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (2 * a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1175_117501


namespace NUMINAMATH_CALUDE_f_value_at_5pi_3_l1175_117572

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_value_at_5pi_3 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : is_periodic f π)
  (h_sin : ∀ x ∈ Set.Icc 0 (π/2), f x = Real.sin x) :
  f (5*π/3) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_3_l1175_117572


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l1175_117561

/-- Proves that given a mixture with an initial ratio of milk to water of 3:1,
    if adding 5 litres of milk changes the ratio to 4:1,
    then the initial volume of the mixture was 20 litres. -/
theorem mixture_volume_proof (V : ℝ) : 
  (3 / 4 * V) / (1 / 4 * V) = 3 / 1 →  -- Initial ratio of milk to water is 3:1
  ((3 / 4 * V + 5) / (1 / 4 * V) = 4 / 1) →  -- New ratio after adding 5 litres of milk is 4:1
  V = 20 := by  -- Initial volume is 20 litres
sorry

end NUMINAMATH_CALUDE_mixture_volume_proof_l1175_117561


namespace NUMINAMATH_CALUDE_min_moves_to_align_cups_l1175_117519

/-- Represents the state of cups on a table -/
structure CupState where
  totalCups : Nat
  upsideCups : Nat
  downsideCups : Nat

/-- Represents a move that flips exactly 3 cups -/
def flipThreeCups (state : CupState) : CupState :=
  { totalCups := state.totalCups,
    upsideCups := state.upsideCups + 3 - 2 * min 3 state.upsideCups,
    downsideCups := state.downsideCups + 3 - 2 * min 3 state.downsideCups }

/-- Predicate to check if all cups are facing the same direction -/
def allSameDirection (state : CupState) : Prop :=
  state.upsideCups = 0 ∨ state.upsideCups = state.totalCups

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_to_align_cups : 
  ∃ (n : Nat), 
    (∀ (state : CupState), 
      state.totalCups = 10 → 
      state.upsideCups = 5 → 
      state.downsideCups = 5 → 
      ∃ (moves : List (CupState → CupState)), 
        moves.length ≤ n ∧ 
        allSameDirection (moves.foldl (fun s m => m s) state) ∧
        ∀ m, m ∈ moves → m = flipThreeCups) ∧
    (∀ (k : Nat), 
      k < n → 
      ∃ (state : CupState), 
        state.totalCups = 10 ∧ 
        state.upsideCups = 5 ∧ 
        state.downsideCups = 5 ∧ 
        ∀ (moves : List (CupState → CupState)), 
          moves.length ≤ k → 
          (∀ m, m ∈ moves → m = flipThreeCups) → 
          ¬allSameDirection (moves.foldl (fun s m => m s) state)) ∧
    n = 3 :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_align_cups_l1175_117519


namespace NUMINAMATH_CALUDE_negative_five_squared_opposite_l1175_117559

-- Define opposite numbers
def are_opposite (a b : ℤ) : Prop := a = -b

-- Theorem statement
theorem negative_five_squared_opposite : are_opposite (-5^2) ((-5)^2) := by
  sorry

end NUMINAMATH_CALUDE_negative_five_squared_opposite_l1175_117559


namespace NUMINAMATH_CALUDE_count_zeros_up_to_2376_l1175_117570

/-- Returns true if the given positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- Counts the number of positive integers less than or equal to n that contain the digit 0 -/
def countZeros (n : ℕ+) : ℕ :=
  sorry

/-- The number of positive integers less than or equal to 2376 that contain the digit 0 is 578 -/
theorem count_zeros_up_to_2376 : countZeros 2376 = 578 :=
  sorry

end NUMINAMATH_CALUDE_count_zeros_up_to_2376_l1175_117570


namespace NUMINAMATH_CALUDE_investment_problem_l1175_117525

/-- Proves that Raghu's investment is 2656.25 given the conditions of the problem -/
theorem investment_problem (raghu : ℝ) : 
  let trishul := 0.9 * raghu
  let vishal := 1.1 * trishul
  let chandni := 1.15 * vishal
  (raghu + trishul + vishal + chandni = 10700) →
  raghu = 2656.25 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l1175_117525


namespace NUMINAMATH_CALUDE_expand_expression_l1175_117553

theorem expand_expression (x : ℝ) : (17 * x - 12) * (3 * x) = 51 * x^2 - 36 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1175_117553


namespace NUMINAMATH_CALUDE_file_storage_problem_l1175_117597

/-- Represents the minimum number of disks required to store files -/
def min_disks (total_files : ℕ) (disk_capacity : ℚ) 
  (files_size_1 : ℕ) (size_1 : ℚ)
  (files_size_2 : ℕ) (size_2 : ℚ)
  (size_3 : ℚ) : ℕ :=
  sorry

theorem file_storage_problem :
  let total_files : ℕ := 33
  let disk_capacity : ℚ := 1.44
  let files_size_1 : ℕ := 3
  let size_1 : ℚ := 1.1
  let files_size_2 : ℕ := 15
  let size_2 : ℚ := 0.6
  let size_3 : ℚ := 0.5
  let remaining_files : ℕ := total_files - files_size_1 - files_size_2
  min_disks total_files disk_capacity files_size_1 size_1 files_size_2 size_2 size_3 = 17 :=
by sorry

end NUMINAMATH_CALUDE_file_storage_problem_l1175_117597


namespace NUMINAMATH_CALUDE_gcd_21n_plus_4_14n_plus_3_gcd_factorial_plus_one_gcd_F_m_F_n_l1175_117576

-- Problem 1
theorem gcd_21n_plus_4_14n_plus_3 (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by sorry

-- Problem 2
theorem gcd_factorial_plus_one (n : ℕ) : Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1) = 1 := by sorry

-- Problem 3
def F (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_F_m_F_n (m n : ℕ) (h : m ≠ n) : Nat.gcd (F m) (F n) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_21n_plus_4_14n_plus_3_gcd_factorial_plus_one_gcd_F_m_F_n_l1175_117576


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1175_117521

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → (¬(24 ∣ m^2) ∨ ¬(720 ∣ m^3))) ∧ 
  24 ∣ n^2 ∧ 720 ∣ n^3 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1175_117521


namespace NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l1175_117511

/-- The probability that a randomly selected point (x, y) in the rectangle
    [0, 4] × [0, 8] satisfies x + y ≤ 6 is 3/8. -/
theorem probability_x_plus_y_leq_6 :
  let total_area : ℝ := 4 * 8
  let valid_area : ℝ := (1 / 2) * 4 * 6
  valid_area / total_area = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l1175_117511


namespace NUMINAMATH_CALUDE_integer_solutions_l1175_117551

/-- The equation whose solutions we're interested in -/
def equation (k x : ℝ) : Prop :=
  (k^2 - 2*k)*x^2 - (6*k - 4)*x + 8 = 0

/-- Predicate to check if a number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The main theorem stating the conditions for integer solutions -/
theorem integer_solutions (k : ℝ) :
  (∀ x : ℝ, equation k x → isInteger x) ↔ (k = 1 ∨ k = -2 ∨ k = 2/3) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_l1175_117551


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l1175_117590

theorem perfect_cube_units_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n : ℕ, n ∈ S ↔ ∃ m : ℕ, m^3 % 10 = n) ∧ 
    S.card = 10 :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l1175_117590


namespace NUMINAMATH_CALUDE_smallest_angle_tangent_equality_l1175_117524

theorem smallest_angle_tangent_equality (x : ℝ) : 
  (x > 0) → 
  (x * (180 / π) = 5.625) → 
  (Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x))) → 
  ∀ y : ℝ, (y > 0) → 
    (Real.tan (6 * y) = (Real.cos (2 * y) - Real.sin (2 * y)) / (Real.cos (2 * y) + Real.sin (2 * y))) → 
    (y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_tangent_equality_l1175_117524


namespace NUMINAMATH_CALUDE_tree_distribution_l1175_117592

/-- The number of ways to distribute n indistinguishable objects into k distinct groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 10 trees over 3 days with at least one tree per day -/
theorem tree_distribution : distribute 10 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tree_distribution_l1175_117592


namespace NUMINAMATH_CALUDE_local_min_iff_a_lt_one_l1175_117575

/-- The function f(x) defined as (x-1)^2 * (x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1)^2 * (x - a)

/-- x = 1 is a local minimum point of f(x) if and only if a < 1 -/
theorem local_min_iff_a_lt_one (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_local_min_iff_a_lt_one_l1175_117575


namespace NUMINAMATH_CALUDE_angle4_is_35_degrees_l1175_117549

-- Define angles as real numbers (in degrees)
variable (angle1 angle2 angle3 angle4 : ℝ)

-- State the theorem
theorem angle4_is_35_degrees
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4) :
  angle4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle4_is_35_degrees_l1175_117549


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l1175_117545

/-- Given an angle θ in the second quadrant satisfying the equation
    cos(θ/2) - sin(θ/2) = √(1 - sin(θ)), prove that θ/2 is in the third quadrant. -/
theorem angle_in_third_quadrant (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) -- θ is in the second quadrant
  (h2 : Real.cos (θ/2) - Real.sin (θ/2) = Real.sqrt (1 - Real.sin θ)) :
  π < θ/2 ∧ θ/2 < 3*π/2 := by
  sorry


end NUMINAMATH_CALUDE_angle_in_third_quadrant_l1175_117545


namespace NUMINAMATH_CALUDE_david_average_marks_l1175_117558

def david_marks : List ℝ := [96, 95, 82, 87, 92]
def num_subjects : ℕ := 5

theorem david_average_marks :
  (david_marks.sum / num_subjects : ℝ) = 90.4 := by
  sorry

end NUMINAMATH_CALUDE_david_average_marks_l1175_117558


namespace NUMINAMATH_CALUDE_xy_upper_bound_and_min_value_l1175_117557

theorem xy_upper_bound_and_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  x * y ≤ 4 ∧ ∃ (min : ℝ), min = 9/5 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 4 → 1/(a+1) + 4/b ≥ min :=
by sorry

end NUMINAMATH_CALUDE_xy_upper_bound_and_min_value_l1175_117557


namespace NUMINAMATH_CALUDE_randy_picture_count_randy_drew_five_pictures_l1175_117546

theorem randy_picture_count : ℕ → ℕ → ℕ → Prop :=
  fun randy peter quincy =>
    (peter = randy + 3) →
    (quincy = peter + 20) →
    (randy + peter + quincy = 41) →
    (randy = 5)

-- The proof of the theorem
theorem randy_drew_five_pictures : ∃ (randy peter quincy : ℕ), randy_picture_count randy peter quincy :=
  sorry

end NUMINAMATH_CALUDE_randy_picture_count_randy_drew_five_pictures_l1175_117546


namespace NUMINAMATH_CALUDE_area_at_stage_6_l1175_117596

/-- The side length of each square -/
def square_side : ℕ := 3

/-- The number of stages -/
def num_stages : ℕ := 6

/-- The area of the rectangle at a given stage -/
def rectangle_area (stage : ℕ) : ℕ :=
  stage * square_side * square_side

/-- Theorem: The area of the rectangle at Stage 6 is 54 square inches -/
theorem area_at_stage_6 : rectangle_area num_stages = 54 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_6_l1175_117596


namespace NUMINAMATH_CALUDE_lily_shopping_exceeds_budget_l1175_117520

/-- Proves that the total cost of items exceeds Lily's initial amount --/
theorem lily_shopping_exceeds_budget :
  let initial_amount : ℝ := 70
  let celery_price : ℝ := 8 * (1 - 0.2)
  let cereal_price : ℝ := 14
  let bread_price : ℝ := 10 * (1 - 0.05)
  let milk_price : ℝ := 12 * (1 - 0.15)
  let potato_price : ℝ := 2 * 8
  let cookie_price : ℝ := 15
  let tax_rate : ℝ := 0.07
  let total_cost : ℝ := (celery_price + cereal_price + bread_price + milk_price + potato_price + cookie_price) * (1 + tax_rate)
  total_cost > initial_amount := by sorry

end NUMINAMATH_CALUDE_lily_shopping_exceeds_budget_l1175_117520


namespace NUMINAMATH_CALUDE_right_triangle_area_l1175_117512

/-- The area of a right-angled triangle with sides 30 cm and 40 cm adjacent to the right angle is 600 cm². -/
theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 40) : 
  (1/2) * a * b = 600 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1175_117512


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1175_117503

theorem complex_power_modulus : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 4) = 256 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1175_117503


namespace NUMINAMATH_CALUDE_circle_properties_l1175_117571

/-- Given a circle with equation 3x^2 - 4y - 12 = -3y^2 + 8x, 
    prove its center coordinates, radius, and a + 2b + r -/
theorem circle_properties : 
  ∃ (a b r : ℝ), 
    (∀ x y : ℝ, 3 * x^2 - 4 * y - 12 = -3 * y^2 + 8 * x → 
      (x - a)^2 + (y - b)^2 = r^2) ∧ 
    a = 4/3 ∧ 
    b = 2/3 ∧ 
    r = 2 * Real.sqrt 13 / 3 ∧
    a + 2 * b + r = (8 + 2 * Real.sqrt 13) / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1175_117571


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1175_117506

/-- If the set A = {x ∈ ℝ | ax² + ax + 1 = 0} has only one element, then a = 4 -/
theorem unique_quadratic_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1175_117506


namespace NUMINAMATH_CALUDE_wilsborough_savings_l1175_117527

/-- Mrs. Wilsborough's concert ticket purchase problem -/
theorem wilsborough_savings : 
  let initial_savings : ℕ := 500
  let vip_ticket_price : ℕ := 100
  let regular_ticket_price : ℕ := 50
  let vip_tickets_bought : ℕ := 2
  let regular_tickets_bought : ℕ := 3
  let total_spent : ℕ := vip_ticket_price * vip_tickets_bought + regular_ticket_price * regular_tickets_bought
  let remaining_savings : ℕ := initial_savings - total_spent
  remaining_savings = 150 := by sorry

end NUMINAMATH_CALUDE_wilsborough_savings_l1175_117527


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l1175_117514

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_and_perpendicular_properties
  (a b c : Line) (γ : Plane) :
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perpendicular a γ ∧ perpendicular b γ → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l1175_117514


namespace NUMINAMATH_CALUDE_junior_count_l1175_117562

theorem junior_count (total : ℕ) (junior_percent : ℚ) (senior_percent : ℚ)
  (h_total : total = 40)
  (h_junior_percent : junior_percent = 1/5)
  (h_senior_percent : senior_percent = 1/10)
  (h_equal_team : ∃ (x : ℕ), x * 5 = junior_percent * total ∧ x * 10 = senior_percent * total) :
  ∃ (j : ℕ), j = 12 ∧ j + (total - j) = total ∧
  (junior_percent * j).num = (senior_percent * (total - j)).num :=
sorry

end NUMINAMATH_CALUDE_junior_count_l1175_117562


namespace NUMINAMATH_CALUDE_probability_same_color_is_half_l1175_117516

def num_red_balls : ℕ := 2
def num_white_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_white_balls

def num_possible_outcomes : ℕ := total_balls * total_balls
def num_same_color_outcomes : ℕ := num_red_balls * num_red_balls + num_white_balls * num_white_balls

def probability_same_color : ℚ := num_same_color_outcomes / num_possible_outcomes

theorem probability_same_color_is_half : probability_same_color = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_half_l1175_117516


namespace NUMINAMATH_CALUDE_problem_solution_l1175_117593

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem problem_solution :
  arithmetic_sequence 2 5 150 = 747 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1175_117593


namespace NUMINAMATH_CALUDE_james_lifting_time_l1175_117510

/-- Calculates the number of days until James can lift heavy again after an injury -/
def daysUntilHeavyLifting (painSubsideDays : ℕ) (healingMultiplier : ℕ) (waitAfterHealingDays : ℕ) (waitBeforeHeavyWeeks : ℕ) : ℕ :=
  let fullHealingDays := painSubsideDays * healingMultiplier
  let totalBeforeExercise := fullHealingDays + waitAfterHealingDays
  let waitBeforeHeavyDays := waitBeforeHeavyWeeks * 7
  totalBeforeExercise + waitBeforeHeavyDays

/-- Theorem stating that given the specific conditions, James can lift heavy after 39 days -/
theorem james_lifting_time :
  daysUntilHeavyLifting 3 5 3 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_james_lifting_time_l1175_117510


namespace NUMINAMATH_CALUDE_fifteen_factorial_digit_sum_l1175_117531

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem fifteen_factorial_digit_sum :
  ∃ (H M T : ℕ),
    H < 10 ∧ M < 10 ∧ T < 10 ∧
    factorial 15 = 1307674 * 10^6 + H * 10^5 + M * 10^3 + 776 * 10^2 + T * 10 + 80 ∧
    H + M + T = 17 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_digit_sum_l1175_117531


namespace NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l1175_117565

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (h : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, h x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The main theorem -/
theorem monic_quartic_value_at_zero 
  (h : ℝ → ℝ) 
  (monic_quartic : MonicQuarticPolynomial h)
  (h_neg_two : h (-2) = -4)
  (h_one : h 1 = -1)
  (h_three : h 3 = -9)
  (h_five : h 5 = -25) : 
  h 0 = -30 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l1175_117565


namespace NUMINAMATH_CALUDE_negative_three_and_half_equality_l1175_117581

theorem negative_three_and_half_equality : -4 + (1/2 : ℚ) = -(7/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_negative_three_and_half_equality_l1175_117581


namespace NUMINAMATH_CALUDE_ochos_friends_ratio_l1175_117560

/-- Given that Ocho has 8 friends, all boys play theater with him, and 4 boys play theater with him,
    prove that the ratio of girls to boys among Ocho's friends is 1:1 -/
theorem ochos_friends_ratio (total_friends : ℕ) (boys_theater : ℕ) 
    (h1 : total_friends = 8)
    (h2 : boys_theater = 4) :
    (total_friends - boys_theater) / boys_theater = 1 := by
  sorry

end NUMINAMATH_CALUDE_ochos_friends_ratio_l1175_117560


namespace NUMINAMATH_CALUDE_cos_555_degrees_l1175_117579

theorem cos_555_degrees : Real.cos (555 * π / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_555_degrees_l1175_117579


namespace NUMINAMATH_CALUDE_evaluate_expression_l1175_117540

theorem evaluate_expression : 3000 * (3000 ^ 3001) = 3000 ^ 3002 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1175_117540


namespace NUMINAMATH_CALUDE_log_sum_simplification_l1175_117577

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 2) +
  1 / (Real.log 2 / Real.log 8 + 2) +
  1 / (Real.log 3 / Real.log 9 + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l1175_117577


namespace NUMINAMATH_CALUDE_total_cost_is_660_l1175_117507

/-- Represents the cost of t-shirts for employees -/
structure TShirtCost where
  white_men : ℕ
  black_men : ℕ
  women_discount : ℕ
  total_employees : ℕ

/-- Calculates the total cost of t-shirts given the conditions -/
def total_cost (c : TShirtCost) : ℕ :=
  let employees_per_type := c.total_employees / 4
  let white_men_cost := c.white_men * employees_per_type
  let white_women_cost := (c.white_men - c.women_discount) * employees_per_type
  let black_men_cost := c.black_men * employees_per_type
  let black_women_cost := (c.black_men - c.women_discount) * employees_per_type
  white_men_cost + white_women_cost + black_men_cost + black_women_cost

/-- Theorem stating that the total cost of t-shirts is $660 -/
theorem total_cost_is_660 (c : TShirtCost)
  (h1 : c.white_men = 20)
  (h2 : c.black_men = 18)
  (h3 : c.women_discount = 5)
  (h4 : c.total_employees = 40) :
  total_cost c = 660 := by
  sorry

#eval total_cost { white_men := 20, black_men := 18, women_discount := 5, total_employees := 40 }

end NUMINAMATH_CALUDE_total_cost_is_660_l1175_117507


namespace NUMINAMATH_CALUDE_max_silver_medals_for_27_points_l1175_117508

/-- Represents the types of medals in the competition -/
inductive Medal
| Gold
| Silver
| Bronze

/-- Returns the point value of a given medal -/
def medal_points (m : Medal) : Nat :=
  match m with
  | Medal.Gold => 5
  | Medal.Silver => 3
  | Medal.Bronze => 1

/-- Represents a competitor's medal collection -/
structure MedalCollection where
  gold : Nat
  silver : Nat
  bronze : Nat

/-- Calculates the total points for a given medal collection -/
def total_points (mc : MedalCollection) : Nat :=
  mc.gold * medal_points Medal.Gold +
  mc.silver * medal_points Medal.Silver +
  mc.bronze * medal_points Medal.Bronze

/-- The main theorem to prove -/
theorem max_silver_medals_for_27_points :
  ∃ (mc : MedalCollection),
    total_points mc = 27 ∧
    mc.gold + mc.silver + mc.bronze ≤ 8 ∧
    mc.silver = 4 ∧
    ∀ (mc' : MedalCollection),
      total_points mc' = 27 →
      mc'.gold + mc'.silver + mc'.bronze ≤ 8 →
      mc'.silver ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_max_silver_medals_for_27_points_l1175_117508


namespace NUMINAMATH_CALUDE_money_division_l1175_117537

theorem money_division (a b c : ℚ) : 
  (4 * a = 5 * b) → 
  (5 * b = 10 * c) → 
  (c = 160) → 
  (a + b + c = 880) := by
sorry

end NUMINAMATH_CALUDE_money_division_l1175_117537


namespace NUMINAMATH_CALUDE_seq_is_bounded_l1175_117573

-- Define P(n) as the product of all digits of n
def P (n : ℕ) : ℕ := sorry

-- Define the sequence (n_k)
def seq (k : ℕ) : ℕ → ℕ
  | n₁ => match k with
    | 0 => n₁
    | k + 1 => seq k n₁ + P (seq k n₁)

-- Theorem statement
theorem seq_is_bounded (n₁ : ℕ) : ∃ M : ℕ, ∀ k : ℕ, seq k n₁ ≤ M := by sorry

end NUMINAMATH_CALUDE_seq_is_bounded_l1175_117573


namespace NUMINAMATH_CALUDE_team_size_l1175_117583

theorem team_size (first_day_per_person : ℕ) (second_day_multiplier : ℕ) (third_day_total : ℕ) (total_blankets : ℕ) :
  first_day_per_person = 2 →
  second_day_multiplier = 3 →
  third_day_total = 22 →
  total_blankets = 142 →
  ∃ team_size : ℕ, 
    team_size * first_day_per_person + 
    team_size * first_day_per_person * second_day_multiplier + 
    third_day_total = total_blankets ∧
    team_size = 15 :=
by sorry

end NUMINAMATH_CALUDE_team_size_l1175_117583


namespace NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l1175_117544

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l1175_117544


namespace NUMINAMATH_CALUDE_tan_difference_special_angle_l1175_117528

theorem tan_difference_special_angle (α : Real) :
  2 * Real.tan α = 3 * Real.tan (π / 8) →
  Real.tan (α - π / 8) = (5 * Real.sqrt 2 + 1) / 49 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_special_angle_l1175_117528


namespace NUMINAMATH_CALUDE_min_value_m_plus_n_l1175_117533

theorem min_value_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt (a * b) = 2) 
  (m n : ℝ) (h4 : m = b + 1/a) (h5 : n = a + 1/b) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) = 2 → m + n ≤ x + y + 1/x + 1/y ∧ m + n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_m_plus_n_l1175_117533


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l1175_117504

def selling_price : ℝ := 670
def original_cost : ℝ := 536

theorem profit_percentage_is_25_percent : 
  (selling_price - original_cost) / original_cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l1175_117504


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1175_117539

theorem complex_number_quadrant (z : ℂ) (h : z * (1 - Complex.I) = 4 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1175_117539


namespace NUMINAMATH_CALUDE_people_who_got_off_l1175_117505

theorem people_who_got_off (initial_people : ℕ) (remaining_people : ℕ) (h1 : initial_people = 48) (h2 : remaining_people = 31) :
  initial_people - remaining_people = 17 := by
  sorry

end NUMINAMATH_CALUDE_people_who_got_off_l1175_117505


namespace NUMINAMATH_CALUDE_zoo_layout_l1175_117535

theorem zoo_layout (tiger_enclosures : ℕ) (zebra_enclosures_per_tiger : ℕ) (giraffe_enclosure_ratio : ℕ)
  (tigers_per_enclosure : ℕ) (zebras_per_enclosure : ℕ) (total_animals : ℕ)
  (h1 : tiger_enclosures = 4)
  (h2 : zebra_enclosures_per_tiger = 2)
  (h3 : giraffe_enclosure_ratio = 3)
  (h4 : tigers_per_enclosure = 4)
  (h5 : zebras_per_enclosure = 10)
  (h6 : total_animals = 144) :
  (total_animals - (tiger_enclosures * tigers_per_enclosure + tiger_enclosures * zebra_enclosures_per_tiger * zebras_per_enclosure)) / 
  (giraffe_enclosure_ratio * tiger_enclosures * zebra_enclosures_per_tiger) = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_layout_l1175_117535


namespace NUMINAMATH_CALUDE_marble_sum_theorem_l1175_117509

theorem marble_sum_theorem (atticus jensen cruz : ℕ) : 
  atticus = 4 → 
  cruz = 8 → 
  atticus = jensen / 2 → 
  3 * (atticus + jensen + cruz) = 60 := by
  sorry

end NUMINAMATH_CALUDE_marble_sum_theorem_l1175_117509


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1175_117522

theorem solve_linear_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1175_117522


namespace NUMINAMATH_CALUDE_square_of_1023_l1175_117523

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l1175_117523


namespace NUMINAMATH_CALUDE_bird_fence_difference_l1175_117569

/-- Given the initial and additional numbers of sparrows and pigeons on a fence,
    and the fact that all starlings flew away, prove that there are 2 more sparrows
    than pigeons on the fence after these events. -/
theorem bird_fence_difference
  (initial_sparrows : ℕ)
  (initial_pigeons : ℕ)
  (additional_sparrows : ℕ)
  (additional_pigeons : ℕ)
  (h1 : initial_sparrows = 3)
  (h2 : initial_pigeons = 2)
  (h3 : additional_sparrows = 4)
  (h4 : additional_pigeons = 3) :
  (initial_sparrows + additional_sparrows) - (initial_pigeons + additional_pigeons) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_fence_difference_l1175_117569


namespace NUMINAMATH_CALUDE_complex_product_real_l1175_117552

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 + a * I
  let z₂ : ℂ := a - 3 * I
  (z₁ * z₂).im = 0 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l1175_117552


namespace NUMINAMATH_CALUDE_separate_amount_possible_l1175_117580

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | EqualGroup (value : ℚ)
  | UnequalGroups (value1 value2 : ℚ)

/-- Represents a weighing operation -/
def Weighing := List ℚ → WeighingResult

/-- The total amount of money in rubles -/
def total_amount : ℚ := 80

/-- The value of a single coin in rubles -/
def coin_value : ℚ := 1/20

/-- The target amount to be separated -/
def target_amount : ℚ := 25

/-- The maximum number of weighings allowed -/
def max_weighings : ℕ := 4

/-- 
  Proves that it's possible to separate the target amount from the total amount 
  using coins of the given value with only a balance scale in the specified number of weighings
-/
theorem separate_amount_possible : 
  ∃ (weighings : List Weighing), 
    weighings.length ≤ max_weighings ∧ 
    ∃ (result : List ℚ), 
      result.sum = target_amount ∧ 
      result.all (λ x => x ≤ total_amount) :=
sorry

end NUMINAMATH_CALUDE_separate_amount_possible_l1175_117580


namespace NUMINAMATH_CALUDE_race_earnings_theorem_l1175_117588

/-- Calculates the average earnings per minute for the race winner -/
def average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (gift_rate : ℚ) (winner_laps : ℕ) : ℚ :=
  let total_distance := winner_laps * lap_distance
  let total_earnings := (total_distance / 100) * gift_rate
  total_earnings / race_duration

/-- Theorem stating that the average earnings per minute is $7 given the race conditions -/
theorem race_earnings_theorem :
  average_earnings_per_minute 12 100 (7/2) 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_race_earnings_theorem_l1175_117588


namespace NUMINAMATH_CALUDE_batsman_score_difference_l1175_117586

/-- Given a batsman's statistics, prove the difference between highest and lowest scores -/
theorem batsman_score_difference
  (total_innings : ℕ)
  (total_runs : ℕ)
  (excluded_innings : ℕ)
  (excluded_runs : ℕ)
  (highest_score : ℕ)
  (h_total_innings : total_innings = 46)
  (h_excluded_innings : excluded_innings = 44)
  (h_total_runs : total_runs = 60 * total_innings)
  (h_excluded_runs : excluded_runs = 58 * excluded_innings)
  (h_highest_score : highest_score = 174) :
  highest_score - (total_runs - excluded_runs - highest_score) = 140 :=
by sorry

end NUMINAMATH_CALUDE_batsman_score_difference_l1175_117586


namespace NUMINAMATH_CALUDE_range_of_m_l1175_117584

theorem range_of_m : ∀ m : ℝ, 
  (¬∃ x : ℝ, 1 < x ∧ x < 3 ∧ x^2 - m*x - 1 = 0) ↔ 
  (m ≤ 0 ∨ m ≥ 8/3) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1175_117584


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l1175_117542

/-- Given a man's downstream speed and the stream speed, calculate the man's upstream speed. -/
theorem mans_upstream_speed
  (downstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : downstream_speed = 11)
  (h2 : stream_speed = 1.5) :
  downstream_speed - 2 * stream_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_speed_l1175_117542


namespace NUMINAMATH_CALUDE_problem_solution_l1175_117591

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3*a

-- Define the conditions
def condition1 (a : ℝ) (m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ 1 < x ∧ x < m

def condition2 (a : ℝ) : Prop :=
  ∀ x, f a x > 0

def condition3 (a : ℝ) (k : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → a^(k+3) < a^(x^2-k*x) ∧ a^(x^2-k*x) < a^(k-3)

-- State the theorem
theorem problem_solution (a m k : ℝ) :
  condition1 a m →
  condition2 a →
  condition3 a k →
  (a = 1 ∧ m = 3) ∧
  (-1 < k ∧ k < -2 + Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1175_117591


namespace NUMINAMATH_CALUDE_line_slope_through_point_l1175_117513

theorem line_slope_through_point (x y k : ℝ) : 
  x = 2 → y = Real.sqrt 3 → y = k * x → k = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_through_point_l1175_117513


namespace NUMINAMATH_CALUDE_right_triangles_2012_characterization_l1175_117587

/-- A right triangle with natural number side lengths where one leg is 2012 -/
structure RightTriangle2012 where
  other_leg : ℕ
  hypotenuse : ℕ
  is_right_triangle : other_leg ^ 2 + 2012 ^ 2 = hypotenuse ^ 2

/-- The set of all valid RightTriangle2012 -/
def all_right_triangles_2012 : Set RightTriangle2012 :=
  { t | t.other_leg > 0 ∧ t.hypotenuse > 0 }

/-- The four specific triangles mentioned in the problem -/
def specific_triangles : Set RightTriangle2012 :=
  { ⟨253005, 253013, by sorry⟩,
    ⟨506016, 506020, by sorry⟩,
    ⟨1012035, 1012037, by sorry⟩,
    ⟨1509, 2515, by sorry⟩ }

/-- The main theorem stating that the set of all valid right triangles with one leg 2012
    is equal to the set of four specific triangles -/
theorem right_triangles_2012_characterization :
  all_right_triangles_2012 = specific_triangles :=
sorry

end NUMINAMATH_CALUDE_right_triangles_2012_characterization_l1175_117587


namespace NUMINAMATH_CALUDE_factorization_coefficient_sum_l1175_117574

theorem factorization_coefficient_sum : 
  ∃ (a b c d e f g h i j k l m n o p : ℤ),
  (∀ x y : ℝ, 
    81 * x^8 - 256 * y^8 = 
    (a*x + b*y) * 
    (c*x^2 + d*x*y + e*y^2) * 
    (f*x^3 + g*x*y^2 + h*y^3) * 
    (i*x + j*y) * 
    (k*x^2 + l*x*y + m*y^2) * 
    (n*x^3 + o*x*y^2 + p*y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
sorry

end NUMINAMATH_CALUDE_factorization_coefficient_sum_l1175_117574


namespace NUMINAMATH_CALUDE_crane_height_theorem_l1175_117515

/-- Represents the height of a crane and the building it's working on -/
structure CraneBuilding where
  crane_height : ℝ
  building_height : ℝ

/-- The problem setup -/
def construction_problem (crane2_height : ℝ) : Prop :=
  let crane1 : CraneBuilding := ⟨228, 200⟩
  let crane2 : CraneBuilding := ⟨crane2_height, 100⟩
  let crane3 : CraneBuilding := ⟨147, 140⟩
  let cranes : List CraneBuilding := [crane1, crane2, crane3]
  let avg_height_diff : ℝ := (cranes.map (λ c => c.crane_height - c.building_height)).sum / cranes.length
  let avg_building_height : ℝ := (cranes.map (λ c => c.building_height)).sum / cranes.length
  avg_height_diff = 0.13 * avg_building_height

/-- The theorem to be proved -/
theorem crane_height_theorem : 
  ∃ (h : ℝ), construction_problem h ∧ abs (h - 122) < 1 :=
sorry

end NUMINAMATH_CALUDE_crane_height_theorem_l1175_117515


namespace NUMINAMATH_CALUDE_cinema_seating_l1175_117598

/-- The number of chairs occupied in a cinema row --/
def occupied_chairs (chairs_between : ℕ) : ℕ :=
  chairs_between + 2

theorem cinema_seating (chairs_between : ℕ) 
  (h : chairs_between = 30) : occupied_chairs chairs_between = 32 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seating_l1175_117598


namespace NUMINAMATH_CALUDE_book_difference_l1175_117547

def jungkook_initial : ℕ := 28
def seokjin_initial : ℕ := 28
def jungkook_bought : ℕ := 18
def seokjin_bought : ℕ := 11

theorem book_difference : 
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_difference_l1175_117547


namespace NUMINAMATH_CALUDE_game_result_l1175_117568

/-- A game between two players where the winner gains 2 points and the loser loses 1 point. -/
structure Game where
  total_games : ℕ
  games_won_by_player1 : ℕ
  final_score_player2 : ℤ

/-- Theorem stating that if player1 wins exactly 3 games and player2 has a final score of 5,
    then the total number of games played is 7. -/
theorem game_result (g : Game) 
  (h1 : g.games_won_by_player1 = 3)
  (h2 : g.final_score_player2 = 5) :
  g.total_games = 7 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1175_117568


namespace NUMINAMATH_CALUDE_solving_linear_equations_count_l1175_117543

/-- Given a total number of math homework problems, calculate the number of
    solving linear equations problems, knowing that 40% are Algebra problems
    and half of the Algebra problems are solving linear equations. -/
def solvingLinearEquationsProblems (total : ℕ) : ℕ :=
  (total * 40 / 100) / 2

/-- Proof that for 140 total math homework problems, the number of
    solving linear equations problems is 28. -/
theorem solving_linear_equations_count :
  solvingLinearEquationsProblems 140 = 28 := by
  sorry

#eval solvingLinearEquationsProblems 140

end NUMINAMATH_CALUDE_solving_linear_equations_count_l1175_117543


namespace NUMINAMATH_CALUDE_divisibility_problem_l1175_117578

theorem divisibility_problem (m p a : ℕ) (hp : Prime p) (hm : m > 0) 
  (h1 : p ∣ (m^2 - 2)) (h2 : ∃ a : ℕ, a > 0 ∧ p ∣ (a^2 + m - 2)) :
  ∃ b : ℕ, b > 0 ∧ p ∣ (b^2 - m - 2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1175_117578


namespace NUMINAMATH_CALUDE_root_product_theorem_l1175_117548

theorem root_product_theorem (n r : ℝ) (a b : ℝ) : 
  (a^2 - n*a + 6 = 0) → 
  (b^2 - n*b + 6 = 0) → 
  ((a + 2/b)^2 - r*(a + 2/b) + s = 0) → 
  ((b + 2/a)^2 - r*(b + 2/a) + s = 0) → 
  s = 32/3 := by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1175_117548


namespace NUMINAMATH_CALUDE_salary_calculation_l1175_117536

/-- Represents the number of turbans given as part of the salary -/
def turbans : ℕ := sorry

/-- The annual base salary in rupees -/
def base_salary : ℕ := 90

/-- The price of each turban in rupees -/
def turban_price : ℕ := 70

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The amount in rupees the servant received when leaving -/
def amount_received : ℕ := 50

/-- The total annual salary in rupees -/
def total_annual_salary : ℕ := base_salary + turbans * turban_price

/-- The fraction of the year the servant worked -/
def fraction_worked : ℚ := 3 / 4

theorem salary_calculation :
  (fraction_worked * total_annual_salary : ℚ) = (amount_received + turban_price : ℕ) → turbans = 1 :=
by sorry

end NUMINAMATH_CALUDE_salary_calculation_l1175_117536


namespace NUMINAMATH_CALUDE_palindrome_difference_l1175_117556

def is_palindrome (n : ℕ) : Prop :=
  ∃ (d : List ℕ), n = d.foldl (λ acc x => acc * 10 + x) 0 ∧ d = d.reverse

def has_9_digits (n : ℕ) : Prop :=
  999999999 ≥ n ∧ n ≥ 100000000

def starts_with_nonzero (n : ℕ) : Prop :=
  n ≥ 100000000

def consecutive_palindromes (m n : ℕ) : Prop :=
  is_palindrome m ∧ is_palindrome n ∧ n > m ∧
  ∀ k, m < k ∧ k < n → ¬is_palindrome k

theorem palindrome_difference (m n : ℕ) :
  has_9_digits m ∧ has_9_digits n ∧
  starts_with_nonzero m ∧ starts_with_nonzero n ∧
  consecutive_palindromes m n →
  n - m = 100000011 := by sorry

end NUMINAMATH_CALUDE_palindrome_difference_l1175_117556


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1175_117582

theorem polynomial_simplification (x : ℝ) :
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1175_117582


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1175_117594

/-- The curve function f(x) = x^2 + 2x - 2 --/
def f (x : ℝ) : ℝ := x^2 + 2*x - 2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), f x = y ∧ f' x = 0 ∧ x = -1 ∧ y = -3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1175_117594


namespace NUMINAMATH_CALUDE_find_divisor_l1175_117532

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 15698)
  (h2 : quotient = 89)
  (h3 : remainder = 14)
  (h4 : dividend = quotient * 176 + remainder) :
  176 = dividend / quotient :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l1175_117532


namespace NUMINAMATH_CALUDE_lifeguard_swim_speed_l1175_117567

theorem lifeguard_swim_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (front_crawl_time : ℝ) 
  (breaststroke_speed : ℝ) 
  (h1 : total_distance = 500)
  (h2 : total_time = 12)
  (h3 : front_crawl_time = 8)
  (h4 : breaststroke_speed = 35)
  : ∃ front_crawl_speed : ℝ, 
    front_crawl_speed * front_crawl_time + 
    breaststroke_speed * (total_time - front_crawl_time) = total_distance ∧ 
    front_crawl_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_lifeguard_swim_speed_l1175_117567


namespace NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l1175_117554

/-- Represents the miles run by Bill and Julia on Saturday and Sunday -/
structure WeekendRun where
  billSat : ℕ
  billSun : ℕ
  juliaSat : ℕ
  juliaSun : ℕ

/-- The conditions of the problem -/
def weekend_run_conditions (run : WeekendRun) : Prop :=
  run.billSun > run.billSat ∧
  run.juliaSat = 0 ∧
  run.juliaSun = 2 * run.billSun ∧
  run.billSat + run.billSun + run.juliaSat + run.juliaSun = 28 ∧
  run.billSun = 8

/-- The theorem to prove -/
theorem bill_sunday_saturday_difference (run : WeekendRun) 
  (h : weekend_run_conditions run) : 
  run.billSun - run.billSat = 4 := by
sorry

end NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l1175_117554


namespace NUMINAMATH_CALUDE_batsman_highest_score_l1175_117589

def batting_problem (total_innings : ℕ) (overall_average : ℚ) (score_difference : ℕ) (average_excluding_extremes : ℚ) : Prop :=
  let total_runs := total_innings * overall_average
  let runs_excluding_extremes := (total_innings - 2) * average_excluding_extremes
  let sum_of_extremes := total_runs - runs_excluding_extremes
  let highest_score := (sum_of_extremes + score_difference) / 2
  highest_score = 199

theorem batsman_highest_score :
  batting_problem 46 60 190 58 := by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l1175_117589


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l1175_117563

/-- Represents a position on the 10x10 board -/
def Position := Fin 10 × Fin 10

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a mark on the board -/
inductive Mark
| X
| O

/-- Represents the game state -/
structure GameState where
  board : Position → Option Mark
  currentPlayer : Player

/-- Checks if a position is winning -/
def isWinningPosition (board : Position → Option Mark) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (pos : Position) : GameState :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Position

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Second strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l1175_117563


namespace NUMINAMATH_CALUDE_total_blocks_l1175_117541

def num_boxes : ℕ := 2
def blocks_per_box : ℕ := 6

theorem total_blocks : num_boxes * blocks_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_l1175_117541


namespace NUMINAMATH_CALUDE_problem_solution_l1175_117500

theorem problem_solution (x y z : ℝ) 
  (sum_condition : x + y + z = 150)
  (equal_condition : x - 5 = y + 3 ∧ y + 3 = z^2) :
  y = 71 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1175_117500


namespace NUMINAMATH_CALUDE_grandfather_cake_blue_candles_l1175_117550

/-- The number of blue candles on Caleb's grandfather's birthday cake -/
def blue_candles (total_candles yellow_candles red_candles : ℕ) : ℕ :=
  total_candles - (yellow_candles + red_candles)

/-- Theorem stating the number of blue candles on the cake -/
theorem grandfather_cake_blue_candles :
  blue_candles 79 27 14 = 38 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_cake_blue_candles_l1175_117550


namespace NUMINAMATH_CALUDE_greatest_divisor_l1175_117502

def problem (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ q1 q2 : ℕ, 1255 = n * q1 + 8 ∧ 1490 = n * q2 + 11 ∧
  ∀ m : ℕ, m > 0 → (∃ r1 r2 : ℕ, 1255 = m * r1 + 8 ∧ 1490 = m * r2 + 11) → m ≤ n

theorem greatest_divisor : problem 29 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_l1175_117502


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1175_117517

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

/-- Theorem: If a_3 + a_4 + a_5 + a_6 + a_7 = 20 in an arithmetic sequence, then S_9 = 36 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h : seq.a 3 + seq.a 4 + seq.a 5 + seq.a 6 + seq.a 7 = 20) :
  seq.S 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1175_117517


namespace NUMINAMATH_CALUDE_at_least_four_same_probability_l1175_117566

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling a specific value on a single die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability that all five dice show the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def fourSameProbability : ℚ :=
  (numDice : ℚ) * (singleDieProbability ^ (numDice - 2)) * (1 - singleDieProbability)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_probability :
  allSameProbability + fourSameProbability = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_at_least_four_same_probability_l1175_117566


namespace NUMINAMATH_CALUDE_apartment_cost_difference_l1175_117599

def apartment_cost (rent : ℕ) (utilities : ℕ) (daily_miles : ℕ) : ℕ :=
  rent + utilities + (daily_miles * 58 * 20) / 100

theorem apartment_cost_difference : 
  apartment_cost 800 260 31 - apartment_cost 900 200 21 = 76 := by sorry

end NUMINAMATH_CALUDE_apartment_cost_difference_l1175_117599


namespace NUMINAMATH_CALUDE_winter_olympics_souvenir_sales_l1175_117585

/-- Daily sales volume as a function of selling price -/
def daily_sales (x : ℝ) : ℝ := -10 * x + 740

/-- Daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := daily_sales x * (x - 40)

/-- The selling price is between 44 and 52 yuan -/
def valid_price (x : ℝ) : Prop := 44 ≤ x ∧ x ≤ 52

theorem winter_olympics_souvenir_sales :
  ∃ (x : ℝ), valid_price x ∧
  (daily_profit x = 2400 → x = 50) ∧
  (∀ y, valid_price y → daily_profit y ≤ daily_profit 52) ∧
  daily_profit 52 = 2640 := by
  sorry


end NUMINAMATH_CALUDE_winter_olympics_souvenir_sales_l1175_117585
