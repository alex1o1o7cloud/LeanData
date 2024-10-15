import Mathlib

namespace NUMINAMATH_CALUDE_smallest_integer_ending_in_3_divisible_by_11_l725_72572

theorem smallest_integer_ending_in_3_divisible_by_11 : ∃ n : ℕ, 
  (n % 10 = 3) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, m < n → m % 10 = 3 → m % 11 ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_ending_in_3_divisible_by_11_l725_72572


namespace NUMINAMATH_CALUDE_lcm_18_45_l725_72569

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_45_l725_72569


namespace NUMINAMATH_CALUDE_bird_nest_babies_six_babies_in_nest_l725_72567

/-- The number of babies in a bird's nest given the worm requirements and available worms. -/
theorem bird_nest_babies (worms_per_baby_per_day : ℕ) (papa_worms : ℕ) (mama_worms : ℕ) 
  (stolen_worms : ℕ) (additional_worms_needed : ℕ) (days : ℕ) : ℕ :=
  let total_worms := papa_worms + mama_worms - stolen_worms + additional_worms_needed
  let worms_per_baby := worms_per_baby_per_day * days
  total_worms / worms_per_baby

/-- There are 6 babies in the nest given the specific conditions. -/
theorem six_babies_in_nest : 
  bird_nest_babies 3 9 13 2 34 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_babies_six_babies_in_nest_l725_72567


namespace NUMINAMATH_CALUDE_child_ticket_cost_l725_72524

/-- Calculates the cost of a child movie ticket given the following information:
  * Adult ticket cost is $9.50
  * Total group size is 7
  * Number of adults is 3
  * Total amount paid is $54.50
-/
theorem child_ticket_cost : 
  let adult_cost : ℝ := 9.50
  let total_group : ℕ := 7
  let num_adults : ℕ := 3
  let total_paid : ℝ := 54.50
  let num_children : ℕ := total_group - num_adults
  let child_cost : ℝ := (total_paid - (adult_cost * num_adults)) / num_children
  child_cost = 6.50 := by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l725_72524


namespace NUMINAMATH_CALUDE_q_div_p_equals_162_l725_72597

/-- The number of slips in the hat -/
def total_slips : ℕ := 40

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The number of slips with each number -/
def slips_per_number : ℕ := 4

/-- The probability that all four drawn slips bear the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability that two slips bear a number a and the other two bear a number b ≠ a -/
def q : ℚ := ((Nat.choose distinct_numbers 2 : ℚ) * 
              (Nat.choose slips_per_number 2 : ℚ) * 
              (Nat.choose slips_per_number 2 : ℚ)) / 
             (Nat.choose total_slips drawn_slips : ℚ)

/-- Theorem stating that q/p = 162 -/
theorem q_div_p_equals_162 : q / p = 162 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_162_l725_72597


namespace NUMINAMATH_CALUDE_prob_exact_tails_l725_72556

def coin_flips : ℕ := 8
def p_tails : ℚ := 4/5
def p_heads : ℚ := 1/5
def exact_tails : ℕ := 3

theorem prob_exact_tails :
  (Nat.choose coin_flips exact_tails : ℚ) * p_tails ^ exact_tails * p_heads ^ (coin_flips - exact_tails) = 3584/390625 := by
  sorry

end NUMINAMATH_CALUDE_prob_exact_tails_l725_72556


namespace NUMINAMATH_CALUDE_power_of_81_l725_72510

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l725_72510


namespace NUMINAMATH_CALUDE_cube_plus_three_square_plus_three_plus_one_l725_72526

theorem cube_plus_three_square_plus_three_plus_one : 101^3 + 3*(101^2) + 3*101 + 1 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_three_square_plus_three_plus_one_l725_72526


namespace NUMINAMATH_CALUDE_problem_solution_l725_72563

theorem problem_solution :
  (999 * (-13) = -12987) ∧
  (999 * 118 * (4/5) + 333 * (-3/5) - 999 * 18 * (3/5) = 99900) ∧
  (6 / (-1/2 + 1/3) = -36) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l725_72563


namespace NUMINAMATH_CALUDE_cars_sold_first_three_days_l725_72517

/-- Proves that the number of cars sold each day for the first three days is 5 --/
theorem cars_sold_first_three_days :
  let total_quota : ℕ := 50
  let cars_sold_next_four_days : ℕ := 3 * 4
  let remaining_cars_to_sell : ℕ := 23
  let cars_per_day_first_three_days : ℕ := (total_quota - cars_sold_next_four_days - remaining_cars_to_sell) / 3
  cars_per_day_first_three_days = 5 := by
  sorry

#eval (50 - 3 * 4 - 23) / 3

end NUMINAMATH_CALUDE_cars_sold_first_three_days_l725_72517


namespace NUMINAMATH_CALUDE_linear_quadratic_intersection_l725_72574

-- Define the functions f and g
def f (k b x : ℝ) : ℝ := k * x + b
def g (x : ℝ) : ℝ := x^2 - x - 6

-- State the theorem
theorem linear_quadratic_intersection (k b : ℝ) :
  (∃ A B : ℝ × ℝ, 
    f k b A.1 = 0 ∧ 
    f k b 0 = B.2 ∧ 
    B.1 - A.1 = 2 ∧ 
    B.2 - A.2 = 2) →
  (k = 1 ∧ b = 2) ∧
  (∀ x : ℝ, f k b x > g x → (g x + 1) / (f k b x) ≥ -3) ∧
  (∃ x : ℝ, f k b x > g x ∧ (g x + 1) / (f k b x) = -3) :=
by sorry

end NUMINAMATH_CALUDE_linear_quadratic_intersection_l725_72574


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l725_72525

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area :
  let h : ℝ := 8  -- height in inches
  let r : ℝ := 3  -- radius in inches
  let lateral_area : ℝ := 2 * π * r * h
  let base_area : ℝ := π * r^2
  let total_surface_area : ℝ := lateral_area + 2 * base_area
  total_surface_area = 66 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l725_72525


namespace NUMINAMATH_CALUDE_asian_art_pieces_l725_72599

theorem asian_art_pieces (total : ℕ) (egyptian : ℕ) (asian : ℕ) 
  (h1 : total = 992) 
  (h2 : egyptian = 527) 
  (h3 : total = egyptian + asian) : 
  asian = 465 := by
sorry

end NUMINAMATH_CALUDE_asian_art_pieces_l725_72599


namespace NUMINAMATH_CALUDE_min_sum_squares_l725_72550

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l725_72550


namespace NUMINAMATH_CALUDE_complex_equation_solution_l725_72579

theorem complex_equation_solution : ∃ (a b : ℝ), (Complex.mk a b) * (Complex.mk a b + Complex.I) * (Complex.mk a b + 2 * Complex.I) = 1001 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l725_72579


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l725_72506

theorem circle_x_axis_intersection_sum (c : ℝ × ℝ) (r : ℝ) : 
  c = (3, -4) → r = 7 → 
  ∃ x₁ x₂ : ℝ, 
    ((x₁ - 3)^2 + 4^2 = r^2) ∧
    ((x₂ - 3)^2 + 4^2 = r^2) ∧
    x₁ + x₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l725_72506


namespace NUMINAMATH_CALUDE_triangle_similarity_after_bisections_l725_72541

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle bisector construction process -/
def AngleBisectorProcess (T : Triangle) (n : ℕ) : Triangle :=
  sorry

/-- Similarity ratio between two triangles -/
def SimilarityRatio (T1 T2 : Triangle) : ℝ :=
  sorry

theorem triangle_similarity_after_bisections (T : Triangle) (h1 : T.a = 5) (h2 : T.b = 6) (h3 : T.c = 4) :
  let T_final := AngleBisectorProcess T 2021
  SimilarityRatio T T_final = (4/9)^2021 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_after_bisections_l725_72541


namespace NUMINAMATH_CALUDE_island_liars_count_l725_72576

/-- Represents the types of inhabitants on the island -/
inductive Inhabitant
  | Knight
  | Liar

/-- The total number of inhabitants on the island -/
def total_inhabitants : Nat := 2001

/-- A function that returns true if the statement "more than half of the others are liars" is true -/
def more_than_half_others_are_liars (num_liars : Nat) : Prop :=
  num_liars > (total_inhabitants - 1) / 2

/-- A function that determines if an inhabitant's statement is consistent with their type -/
def consistent_statement (inhabitant : Inhabitant) (num_liars : Nat) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => more_than_half_others_are_liars num_liars
  | Inhabitant.Liar => ¬(more_than_half_others_are_liars num_liars)

theorem island_liars_count :
  ∃ (num_liars : Nat),
    num_liars ≤ total_inhabitants ∧
    (∀ (i : Inhabitant), consistent_statement i num_liars) ∧
    num_liars = 1001 := by
  sorry

end NUMINAMATH_CALUDE_island_liars_count_l725_72576


namespace NUMINAMATH_CALUDE_solve_for_k_l725_72528

theorem solve_for_k (x k : ℝ) : x + k - 4 = 0 → x = 2 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l725_72528


namespace NUMINAMATH_CALUDE_tan_equality_implies_x_120_l725_72527

theorem tan_equality_implies_x_120 (x : Real) :
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 120 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_implies_x_120_l725_72527


namespace NUMINAMATH_CALUDE_pizza_theorem_l725_72588

/-- The number of pizzas ordered for a class celebration --/
def pizza_problem (num_boys : ℕ) (num_girls : ℕ) (boys_pizzas : ℕ) : Prop :=
  num_girls = 11 ∧
  num_boys > num_girls ∧
  boys_pizzas = 10 ∧
  ∃ (total_pizzas : ℚ),
    total_pizzas = boys_pizzas + (num_girls : ℚ) * (boys_pizzas : ℚ) / (2 * num_boys : ℚ) ∧
    total_pizzas = 11

theorem pizza_theorem :
  ∃ (num_boys : ℕ), pizza_problem num_boys 11 10 :=
sorry

end NUMINAMATH_CALUDE_pizza_theorem_l725_72588


namespace NUMINAMATH_CALUDE_rectangle_area_l725_72581

-- Define the rectangle
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the area function
def area (r : Rectangle) : ℝ :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

-- Theorem statement
theorem rectangle_area :
  let r : Rectangle := { x1 := 0, y1 := 0, x2 := 3, y2 := 3 }
  area r = 9 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l725_72581


namespace NUMINAMATH_CALUDE_unique_positive_number_l725_72514

theorem unique_positive_number : ∃! (n : ℝ), n > 0 ∧ (1/5 * n) * (1/7 * n) = n := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l725_72514


namespace NUMINAMATH_CALUDE_derek_savings_and_expenses_l725_72513

theorem derek_savings_and_expenses :
  let geometric_sum := (2 : ℝ) * (1 - 2^12) / (1 - 2)
  let arithmetic_sum := 12 / 2 * (2 * 3 + (12 - 1) * 2)
  geometric_sum - arithmetic_sum = 8022 := by
  sorry

end NUMINAMATH_CALUDE_derek_savings_and_expenses_l725_72513


namespace NUMINAMATH_CALUDE_divisors_of_18n_cubed_l725_72592

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_18n_cubed (n : ℕ) 
  (h_odd : Odd n) 
  (h_divisors : num_divisors n = 13) : 
  num_divisors (18 * n^3) = 222 := by sorry

end NUMINAMATH_CALUDE_divisors_of_18n_cubed_l725_72592


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_three_l725_72589

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) (x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

/-- Theorem: If l₁ and l₂ are parallel, then a = -3 -/
theorem parallel_lines_imply_a_eq_neg_three (a : ℝ) :
  (∀ x y : ℝ, line_l₁ a x y ↔ line_l₂ a x y) → a = -3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_three_l725_72589


namespace NUMINAMATH_CALUDE_homework_reading_assignment_l725_72530

theorem homework_reading_assignment (sam_pages pam_pages harrison_pages assigned_pages : ℕ) : 
  sam_pages = 100 →
  sam_pages = 2 * pam_pages →
  pam_pages = harrison_pages + 15 →
  harrison_pages = assigned_pages + 10 →
  assigned_pages = 25 := by
sorry

end NUMINAMATH_CALUDE_homework_reading_assignment_l725_72530


namespace NUMINAMATH_CALUDE_shirts_not_washed_l725_72544

theorem shirts_not_washed 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) : 
  short_sleeve + long_sleeve - washed = 1 := by
sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l725_72544


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l725_72529

def sum_of_range (a b : ℕ) : ℕ := 
  ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  let x := sum_of_range 20 30
  let y := count_even_in_range 20 30
  x + y = 281 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l725_72529


namespace NUMINAMATH_CALUDE_triangle_angle_f_l725_72505

theorem triangle_angle_f (D E F : Real) : 
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = Real.pi →
  5 * Real.sin D + 2 * Real.cos E = 8 →
  3 * Real.sin E + 5 * Real.cos D = 2 →
  Real.sin F = 43 / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_f_l725_72505


namespace NUMINAMATH_CALUDE_solution_set_correct_l725_72558

/-- The set of solutions to the system of equations:
    x^2 + y^2 + 8x - 6y = -20
    x^2 + z^2 + 8x + 4z = -10
    y^2 + z^2 - 6y + 4z = 0
-/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(-3, 1, 1), (-3, 1, -5), (-3, 5, 1), (-3, 5, -5),
   (-5, 1, 1), (-5, 1, -5), (-5, 5, 1), (-5, 5, -5)}

/-- The system of equations -/
def SystemEquations (x y z : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 6*y = -20 ∧
  x^2 + z^2 + 8*x + 4*z = -10 ∧
  y^2 + z^2 - 6*y + 4*z = 0

/-- Theorem stating that the SolutionSet contains exactly all solutions to the SystemEquations -/
theorem solution_set_correct :
  ∀ x y z, (x, y, z) ∈ SolutionSet ↔ SystemEquations x y z :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_l725_72558


namespace NUMINAMATH_CALUDE_parabola_equation_l725_72594

/-- Given a parabola y^2 = 2px where p > 0, if a line with slope 1 passing through
    the focus intersects the parabola at points A and B such that |AB| = 8,
    then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2*p*x → (∃ t, y = t ∧ x = t + p/2)) →  -- Line passing through focus
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →                -- A and B on parabola
  ‖A - B‖ = 8 →                                        -- |AB| = 8
  ∀ x y, y^2 = 2*p*x ↔ y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l725_72594


namespace NUMINAMATH_CALUDE_maddie_monday_viewing_l725_72512

/-- The number of minutes Maddie watched TV on Monday -/
def monday_minutes (total_episodes : ℕ) (episode_length : ℕ) (thursday_minutes : ℕ) (friday_episodes : ℕ) (weekend_minutes : ℕ) : ℕ :=
  total_episodes * episode_length - (thursday_minutes + friday_episodes * episode_length + weekend_minutes)

theorem maddie_monday_viewing : 
  monday_minutes 8 44 21 2 105 = 138 := by
  sorry

end NUMINAMATH_CALUDE_maddie_monday_viewing_l725_72512


namespace NUMINAMATH_CALUDE_gcf_540_196_l725_72520

theorem gcf_540_196 : Nat.gcd 540 196 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcf_540_196_l725_72520


namespace NUMINAMATH_CALUDE_sine_translation_stretch_l725_72518

/-- The transformation of the sine function -/
theorem sine_translation_stretch (x : ℝ) :
  let f := λ x : ℝ => Real.sin x
  let g := λ x : ℝ => Real.sin (x / 2 - π / 8)
  g x = (f ∘ (λ y => y - π / 8) ∘ (λ y => y / 2)) x :=
by sorry

end NUMINAMATH_CALUDE_sine_translation_stretch_l725_72518


namespace NUMINAMATH_CALUDE_linear_inequality_solution_set_l725_72521

theorem linear_inequality_solution_set 
  (a b : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 1) : 
  {x : ℝ | a * x + b < 0} = {x : ℝ | x > 1} := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_set_l725_72521


namespace NUMINAMATH_CALUDE_trigonometric_identity_l725_72531

theorem trigonometric_identity (α : ℝ) :
  -Real.cos (5 * α) * Real.cos (4 * α) - Real.cos (4 * α) * Real.cos (3 * α) + 2 * (Real.cos (2 * α))^2 * Real.cos α
  = 2 * Real.cos α * Real.sin (2 * α) * Real.sin (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l725_72531


namespace NUMINAMATH_CALUDE_symmetric_line_proof_l725_72547

/-- The fixed point M through which all lines ax+y+3a-1=0 pass -/
def M : ℝ × ℝ := (-3, 1)

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2*x + 3*y - 6 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2*x + 3*y + 12 = 0

/-- The family of lines passing through M -/
def family_line (a x y : ℝ) : Prop := a*x + y + 3*a - 1 = 0

theorem symmetric_line_proof :
  ∀ (a : ℝ), family_line a M.1 M.2 →
  ∀ (x y : ℝ), symmetric_line x y ↔ 
    (x - M.1 = M.1 - x' ∧ y - M.2 = M.2 - y' ∧ original_line x' y') :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_proof_l725_72547


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_and_remainder_l725_72511

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_and_remainder :
  let sum := arithmetic_sequence_sum 3 5 103
  sum = 1113 ∧ sum % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_and_remainder_l725_72511


namespace NUMINAMATH_CALUDE_probability_units_digit_less_than_3_l725_72519

/-- A five-digit even integer -/
def FiveDigitEven : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 ∧ n % 2 = 0 }

/-- The set of possible units digits for even numbers -/
def EvenUnitsDigits : Finset ℕ := {0, 2, 4, 6, 8}

/-- The set of units digits less than 3 -/
def UnitsDigitsLessThan3 : Finset ℕ := {0, 2}

/-- The probability of a randomly chosen five-digit even integer having a units digit less than 3 -/
theorem probability_units_digit_less_than_3 :
  (Finset.card UnitsDigitsLessThan3 : ℚ) / (Finset.card EvenUnitsDigits : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_units_digit_less_than_3_l725_72519


namespace NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l725_72554

-- Define the rounding function
def roundToNearestDollar (x : ℚ) : ℤ :=
  if x - x.floor < 1/2 then x.floor else x.ceil

-- Define the purchases
def purchase1 : ℚ := 299/100
def purchase2 : ℚ := 651/100
def purchase3 : ℚ := 1049/100

-- Theorem statement
theorem total_rounded_to_nearest_dollar :
  (roundToNearestDollar purchase1 + 
   roundToNearestDollar purchase2 + 
   roundToNearestDollar purchase3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l725_72554


namespace NUMINAMATH_CALUDE_remainder_is_224_l725_72545

/-- The polynomial f(x) = x^5 - 8x^4 + 16x^3 + 25x^2 - 50x + 24 -/
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 16*x^3 + 25*x^2 - 50*x + 24

/-- The remainder when f(x) is divided by (x - 4) -/
def remainder : ℝ := f 4

theorem remainder_is_224 : remainder = 224 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_224_l725_72545


namespace NUMINAMATH_CALUDE_periodic_function_value_l725_72562

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value (f : ℝ → ℝ) :
  is_periodic f 4 →
  (∀ x ∈ Set.Icc (-2) 2, f x = x) →
  f 7.6 = -0.4 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l725_72562


namespace NUMINAMATH_CALUDE_four_male_workers_selected_l725_72575

/-- Represents the number of male workers selected in a stratified sampling -/
def male_workers_selected (total_workers female_workers selected_workers : ℕ) : ℕ :=
  (total_workers - female_workers) * selected_workers / total_workers

/-- Theorem stating that 4 male workers are selected in the given scenario -/
theorem four_male_workers_selected :
  male_workers_selected 30 10 6 = 4 := by
  sorry

#eval male_workers_selected 30 10 6

end NUMINAMATH_CALUDE_four_male_workers_selected_l725_72575


namespace NUMINAMATH_CALUDE_min_sum_of_internally_tangent_circles_l725_72515

/-- Given two circles C₁ and C₂ with equations x² + y² + 2ax + a² - 4 = 0 and x² + y² - 2by - 1 + b² = 0 respectively, 
    where a, b ∈ ℝ, and C₁ and C₂ have only one common tangent line, 
    the minimum value of a + b is -√2. -/
theorem min_sum_of_internally_tangent_circles (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 → x^2 + y^2 - 2*b*y - 1 + b^2 = 0 → False) ∧ 
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y - 1 + b^2 = 0) →
  (a + b ≥ -Real.sqrt 2) ∧ (∃ a₀ b₀ : ℝ, a₀ + b₀ = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_internally_tangent_circles_l725_72515


namespace NUMINAMATH_CALUDE_basket_replacement_theorem_l725_72500

/-- The number of people who entered the stadium before the basket needed replacement -/
def people_entered : ℕ :=
  sorry

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The total number of placards the basket can hold -/
def basket_capacity : ℕ := 823

theorem basket_replacement_theorem :
  people_entered = 411 ∧
  people_entered * placards_per_person < basket_capacity ∧
  (people_entered + 1) * placards_per_person > basket_capacity :=
by sorry

end NUMINAMATH_CALUDE_basket_replacement_theorem_l725_72500


namespace NUMINAMATH_CALUDE_marbles_remainder_l725_72539

theorem marbles_remainder (r p : ℕ) 
  (h1 : r % 8 = 5) 
  (h2 : p % 8 = 7) 
  (h3 : (r + p) % 10 = 0) :
  (r + p) % 8 = 4 := by sorry

end NUMINAMATH_CALUDE_marbles_remainder_l725_72539


namespace NUMINAMATH_CALUDE_no_central_ring_numbers_l725_72593

/-- Definition of a central ring number -/
def is_central_ring_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (n % 11 ≠ 0) ∧              -- not divisible by 11
  ((n / 1000) % 11 = 0) ∧     -- removing thousands digit
  ((n % 1000 + (n / 10000) * 100) % 11 = 0) ∧  -- removing hundreds digit
  ((n / 100 * 10 + n % 10) % 11 = 0) ∧         -- removing tens digit
  ((n / 10) % 11 = 0)         -- removing ones digit

/-- Theorem: There are no central ring numbers -/
theorem no_central_ring_numbers : ¬∃ n, is_central_ring_number n := by
  sorry

end NUMINAMATH_CALUDE_no_central_ring_numbers_l725_72593


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l725_72578

theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 2) * x^(m^2 - 2) - m*x + 1 = a*x^2 + b*x + c) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l725_72578


namespace NUMINAMATH_CALUDE_one_fourths_in_seven_halves_l725_72551

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_seven_halves_l725_72551


namespace NUMINAMATH_CALUDE_zongzi_purchase_theorem_l725_72584

/-- Represents the properties of zongzi purchases in a supermarket. -/
structure ZongziPurchase where
  price_a : ℝ  -- Unit price of type A zongzi
  price_b : ℝ  -- Unit price of type B zongzi
  quantity_a : ℝ  -- Quantity of type A zongzi
  quantity_b : ℝ  -- Quantity of type B zongzi

/-- Theorem stating the properties of the zongzi purchase and the maximum purchase of type A zongzi. -/
theorem zongzi_purchase_theorem (z : ZongziPurchase) : 
  z.price_a * z.quantity_a = 1200 ∧ 
  z.price_b * z.quantity_b = 800 ∧ 
  z.quantity_b = z.quantity_a + 50 ∧ 
  z.price_a = 2 * z.price_b → 
  z.price_a = 8 ∧ z.price_b = 4 ∧ 
  (∀ m : ℕ, m ≤ 87 ↔ (m : ℝ) * 8 + (200 - m) * 4 ≤ 1150) :=
by sorry

#check zongzi_purchase_theorem

end NUMINAMATH_CALUDE_zongzi_purchase_theorem_l725_72584


namespace NUMINAMATH_CALUDE_pyarelal_loss_calculation_l725_72577

/-- Calculates Pyarelal's share of the loss given the total loss and the ratio of investments -/
def pyarelal_loss (total_loss : ℚ) (ashok_ratio : ℚ) (pyarelal_ratio : ℚ) : ℚ :=
  (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss

/-- Proves that Pyarelal's loss is 1080 given the conditions of the problem -/
theorem pyarelal_loss_calculation :
  let total_loss : ℚ := 1200
  let ashok_ratio : ℚ := 1
  let pyarelal_ratio : ℚ := 9
  pyarelal_loss total_loss ashok_ratio pyarelal_ratio = 1080 := by
  sorry

#eval pyarelal_loss 1200 1 9

end NUMINAMATH_CALUDE_pyarelal_loss_calculation_l725_72577


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l725_72583

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l725_72583


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l725_72509

theorem employee_pay_calculation (total_pay : ℝ) (percentage : ℝ) (y : ℝ) :
  total_pay = 880 →
  percentage = 120 →
  total_pay = y + (percentage / 100) * y →
  y = 400 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l725_72509


namespace NUMINAMATH_CALUDE_candy_distribution_l725_72571

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) (num_students : ℕ) :
  total_candies = 81 →
  candies_per_student = 9 →
  total_candies = candies_per_student * num_students →
  num_students = 9 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l725_72571


namespace NUMINAMATH_CALUDE_box_of_balls_theorem_l725_72535

theorem box_of_balls_theorem :
  ∃ (B X Y : ℝ),
    40 < X ∧ X < 50 ∧
    60 < Y ∧ Y < 70 ∧
    B - X = Y - B ∧
    B = 55 := by sorry

end NUMINAMATH_CALUDE_box_of_balls_theorem_l725_72535


namespace NUMINAMATH_CALUDE_part_one_part_two_l725_72542

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def q (x m : ℝ) : Prop := 1 - m^2 ≤ x ∧ x ≤ 1 + m^2

-- Part I: p is a necessary condition for q
theorem part_one (m : ℝ) : 
  (∀ x, q x m → p x) → -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

-- Part II: ¬p is a necessary but not sufficient condition for ¬q
theorem part_two (m : ℝ) :
  ((∀ x, ¬q x m → ¬p x) ∧ (∃ x, ¬p x ∧ q x m)) → m ≥ 3 ∨ m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l725_72542


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l725_72596

theorem compare_negative_fractions : -4/5 > -5/6 := by sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l725_72596


namespace NUMINAMATH_CALUDE_hyperbola_distance_l725_72516

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a point on the hyperbola
def P : ℝ × ℝ := sorry

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_distance :
  hyperbola P.1 P.2 → distance P F1 = 9 → distance P F2 = 17 := by sorry

end NUMINAMATH_CALUDE_hyperbola_distance_l725_72516


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l725_72585

theorem concentric_circles_radius (r R : ℝ) (h1 : r = 4) 
  (h2 : (1.5 * R)^2 - (0.75 * r)^2 = 3.6 * (R^2 - r^2)) : R = 6 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l725_72585


namespace NUMINAMATH_CALUDE_solution_water_amount_l725_72502

/-- Given a solution with an original ratio of bleach : detergent : water as 2 : 40 : 100,
    when the ratio of bleach to detergent is tripled and the ratio of detergent to water is halved,
    and the new solution contains 60 liters of detergent, prove that the amount of water
    in the new solution is 75 liters. -/
theorem solution_water_amount
  (original_ratio : Fin 3 → ℚ)
  (h_original : original_ratio = ![2, 40, 100])
  (new_detergent : ℚ)
  (h_new_detergent : new_detergent = 60)
  : ∃ (new_ratio : Fin 3 → ℚ) (water : ℚ),
    (new_ratio 0 / new_ratio 1 = 3 * (original_ratio 0 / original_ratio 1)) ∧
    (new_ratio 1 / new_ratio 2 = (original_ratio 1 / original_ratio 2) / 2) ∧
    (new_ratio 1 = new_detergent) ∧
    (water = 75) := by
  sorry

end NUMINAMATH_CALUDE_solution_water_amount_l725_72502


namespace NUMINAMATH_CALUDE_danielles_apartment_rooms_l725_72587

theorem danielles_apartment_rooms (heidi_rooms danielle_rooms grant_rooms : ℕ) : 
  heidi_rooms = 3 * danielle_rooms →
  grant_rooms = heidi_rooms / 9 →
  grant_rooms = 2 →
  danielle_rooms = 6 := by
sorry

end NUMINAMATH_CALUDE_danielles_apartment_rooms_l725_72587


namespace NUMINAMATH_CALUDE_symmetry_origin_symmetry_point_l725_72538

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the origin
def symmetricToOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

-- Define symmetry with respect to another point
def symmetricToPoint (p : Point2D) (k : Point2D) : Point2D :=
  { x := 2 * k.x - p.x, y := 2 * k.y - p.y }

-- Theorem for symmetry with respect to the origin
theorem symmetry_origin (m : Point2D) :
  symmetricToOrigin m = { x := -m.x, y := -m.y } := by
  sorry

-- Theorem for symmetry with respect to another point
theorem symmetry_point (m k : Point2D) :
  symmetricToPoint m k = { x := 2 * k.x - m.x, y := 2 * k.y - m.y } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_origin_symmetry_point_l725_72538


namespace NUMINAMATH_CALUDE_smallest_cube_ending_888_l725_72590

theorem smallest_cube_ending_888 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 888 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 888 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_888_l725_72590


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l725_72557

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + k = 0) ↔ k = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l725_72557


namespace NUMINAMATH_CALUDE_unique_two_digit_ratio_l725_72570

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem unique_two_digit_ratio :
  ∃! n : ℕ, is_two_digit n ∧ (n : ℚ) / (reverse_digits n : ℚ) = 7 / 4 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_ratio_l725_72570


namespace NUMINAMATH_CALUDE_A_divisible_by_1980_l725_72595

def A : ℕ := sorry  -- Definition of A as the concatenated number

-- Theorem statement
theorem A_divisible_by_1980 : 1980 ∣ A :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_A_divisible_by_1980_l725_72595


namespace NUMINAMATH_CALUDE_rope_length_is_35_l725_72507

/-- The length of the rope in meters -/
def rope_length : ℝ := 35

/-- The time ratio between walking with and against the tractor -/
def time_ratio : ℝ := 7

/-- The equation for walking in the same direction as the tractor -/
def same_direction_equation (x S : ℝ) : Prop :=
  x + time_ratio * S = 140

/-- The equation for walking in the opposite direction of the tractor -/
def opposite_direction_equation (x S : ℝ) : Prop :=
  x - S = 20

theorem rope_length_is_35 :
  ∃ S : ℝ, same_direction_equation rope_length S ∧ opposite_direction_equation rope_length S :=
sorry

end NUMINAMATH_CALUDE_rope_length_is_35_l725_72507


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l725_72566

/-- Given three lines that pass through the same point, prove the value of k -/
theorem intersection_of_three_lines (t : ℝ) (h_t : t = 6) :
  ∃ (x y : ℝ), (x + t * y + 8 = 0 ∧ 5 * x - t * y + 4 = 0 ∧ 3 * x - 5 * y + 1 = 0) →
  ∀ k : ℝ, (x + t * y + 8 = 0 ∧ 5 * x - t * y + 4 = 0 ∧ 3 * x - k * y + 1 = 0) →
  k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l725_72566


namespace NUMINAMATH_CALUDE_tadpole_survival_fraction_l725_72561

/-- Represents the frog pond ecosystem --/
structure FrogPond where
  num_frogs : ℕ
  num_tadpoles : ℕ
  max_capacity : ℕ
  frogs_to_relocate : ℕ

/-- Calculates the fraction of tadpoles that will survive to maturity as frogs --/
def survival_fraction (pond : FrogPond) : ℚ :=
  let surviving_tadpoles := pond.max_capacity - pond.num_frogs
  ↑surviving_tadpoles / ↑pond.num_tadpoles

/-- Theorem stating the fraction of tadpoles that will survive to maturity as frogs --/
theorem tadpole_survival_fraction (pond : FrogPond) 
  (h1 : pond.num_frogs = 5)
  (h2 : pond.num_tadpoles = 3 * pond.num_frogs)
  (h3 : pond.max_capacity = 8)
  (h4 : pond.frogs_to_relocate = 7) :
  survival_fraction pond = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tadpole_survival_fraction_l725_72561


namespace NUMINAMATH_CALUDE_particular_number_solution_l725_72522

theorem particular_number_solution (A B : ℤ) (h1 : A = 14) (h2 : B = 24) :
  ∃ x : ℚ, ((A + x) * A - B) / B = 13 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_solution_l725_72522


namespace NUMINAMATH_CALUDE_adam_magnets_l725_72523

theorem adam_magnets (peter_magnets : ℕ) (adam_remaining : ℕ) (adam_initial : ℕ) : 
  peter_magnets = 24 →
  adam_remaining = peter_magnets / 2 →
  adam_remaining = adam_initial * 2 / 3 →
  adam_initial = 18 := by
sorry

end NUMINAMATH_CALUDE_adam_magnets_l725_72523


namespace NUMINAMATH_CALUDE_circle_radius_from_chord_and_central_angle_l725_72573

theorem circle_radius_from_chord_and_central_angle (α : ℝ) (h : α > 0 ∧ α < 360) :
  let chord_length : ℝ := 10
  let radius : ℝ := 5 / Real.sin (α * π / 360)
  2 * radius * Real.sin (α * π / 360) = chord_length := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_chord_and_central_angle_l725_72573


namespace NUMINAMATH_CALUDE_functional_equation_solution_l725_72552

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) + f (x + y) = f x * f y + f x + f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l725_72552


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l725_72565

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x + 2) = 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x - 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l725_72565


namespace NUMINAMATH_CALUDE_jaspers_refreshments_l725_72548

theorem jaspers_refreshments (chips drinks : ℕ) (h1 : chips = 27) (h2 : drinks = 31) :
  let hot_dogs := drinks - 12
  chips - hot_dogs = 8 := by
  sorry

end NUMINAMATH_CALUDE_jaspers_refreshments_l725_72548


namespace NUMINAMATH_CALUDE_solve_for_a_l725_72555

theorem solve_for_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 8) 
  (eq3 : c = 4) : 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l725_72555


namespace NUMINAMATH_CALUDE_simplified_inverse_sum_l725_72598

theorem simplified_inverse_sum (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  (a * x⁻¹ + b * y⁻¹)⁻¹ = (x * y) / (a * y + b * x) := by
  sorry

end NUMINAMATH_CALUDE_simplified_inverse_sum_l725_72598


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_15_12_18_l725_72553

theorem least_five_digit_divisible_by_15_12_18 :
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    n % 15 = 0 ∧ 
    n % 12 = 0 ∧ 
    n % 18 = 0 ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < n ∧ m % 15 = 0 ∧ m % 12 = 0 ∧ m % 18 = 0 → false) ∧
    n = 10080 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_15_12_18_l725_72553


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l725_72536

-- Define the set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

-- Define the set B
def B : Set ℝ := {x : ℝ | x^2 < 4}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x : ℝ | ¬ (x ∈ B)}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ complement_B = {x : ℝ | 2 ≤ x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l725_72536


namespace NUMINAMATH_CALUDE_mady_balls_theorem_l725_72568

def to_nonary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_nonary_aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_nonary_aux (m / 9) ((m % 9) :: acc)
    to_nonary_aux n []

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.sum

theorem mady_balls_theorem (step : ℕ) (h : step = 2500) :
  sum_of_digits (to_nonary step) = 20 :=
sorry

end NUMINAMATH_CALUDE_mady_balls_theorem_l725_72568


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l725_72532

theorem factor_implies_d_value (d : ℚ) :
  (∀ x : ℚ, (x - 4) ∣ (d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72)) →
  d = -83/42 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l725_72532


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_27_l725_72534

-- Define the polynomial
def p (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 4) + 6 * (x^6 + 9 * x^3 - 8)

-- Theorem statement
theorem sum_of_coefficients_is_27 : 
  p 1 = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_27_l725_72534


namespace NUMINAMATH_CALUDE_function_property_l725_72503

def IteratedFunction (f : ℕ+ → ℕ+) : ℕ → ℕ+ → ℕ+
  | 0, n => n
  | k+1, n => f (IteratedFunction f k n)

theorem function_property (f : ℕ+ → ℕ+) :
  (∀ (a b c : ℕ+), a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 →
    IteratedFunction f (a*b*c - a) (a*b*c) + 
    IteratedFunction f (a*b*c - b) (a*b*c) + 
    IteratedFunction f (a*b*c - c) (a*b*c) = a + b + c) →
  ∀ n : ℕ+, n ≥ 3 → f n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l725_72503


namespace NUMINAMATH_CALUDE_series_sum_inequality_l725_72546

theorem series_sum_inequality (S : ℝ) (h : S = 2^(1/4)) : 
  ∃ n : ℕ, 2^n < S^2007 ∧ S^2007 < 2^(n+1) ∧ n = 501 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_inequality_l725_72546


namespace NUMINAMATH_CALUDE_find_x1_l725_72501

theorem find_x1 (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + 2*(x1 - x2)^2 + 2*(x2 - x3)^2 + x3^2 = 1/2) :
  x1 = 2/3 := by sorry

end NUMINAMATH_CALUDE_find_x1_l725_72501


namespace NUMINAMATH_CALUDE_sqrt_squared_equals_original_sqrt_529441_squared_l725_72564

theorem sqrt_squared_equals_original (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n)^2 = n := by
  sorry

theorem sqrt_529441_squared :
  (Real.sqrt 529441)^2 = 529441 := by
  apply sqrt_squared_equals_original
  norm_num

end NUMINAMATH_CALUDE_sqrt_squared_equals_original_sqrt_529441_squared_l725_72564


namespace NUMINAMATH_CALUDE_alcohol_fraction_after_water_increase_l725_72586

theorem alcohol_fraction_after_water_increase (v : ℝ) (h : v > 0) :
  let initial_alcohol := (2 / 3) * v
  let initial_water := (1 / 3) * v
  let new_water := 3 * initial_water
  let new_total := initial_alcohol + new_water
  initial_alcohol / new_total = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_alcohol_fraction_after_water_increase_l725_72586


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l725_72504

theorem quadratic_distinct_roots (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  (n < -6 ∨ n > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l725_72504


namespace NUMINAMATH_CALUDE_partner_c_profit_share_l725_72591

/-- Given the investments of partners A, B, and C, and the total profit,
    calculate C's share of the profit. -/
theorem partner_c_profit_share
  (invest_a invest_b invest_c total_profit : ℝ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_a = 2 / 3 * invest_c)
  (h3 : total_profit = 66000) :
  (invest_c / (invest_a + invest_b + invest_c)) * total_profit = (9 / 17) * 66000 :=
by sorry

end NUMINAMATH_CALUDE_partner_c_profit_share_l725_72591


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l725_72582

def initial_people : ℕ := 7
def initial_average_age : ℚ := 28
def leaving_person_age : ℕ := 20

theorem average_age_after_leaving :
  let total_age : ℚ := initial_people * initial_average_age
  let remaining_total_age : ℚ := total_age - leaving_person_age
  let remaining_people : ℕ := initial_people - 1
  remaining_total_age / remaining_people = 29.33 := by sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l725_72582


namespace NUMINAMATH_CALUDE_white_pieces_count_l725_72580

/-- The number of possible arrangements of chess pieces -/
def total_arrangements : ℕ := 144

/-- The number of black chess pieces -/
def black_pieces : ℕ := 3

/-- Function to calculate the number of arrangements given white and black pieces -/
def arrangements (white : ℕ) (black : ℕ) : ℕ :=
  (Nat.factorial white) * (Nat.factorial black)

/-- Theorem stating that there are 4 white chess pieces -/
theorem white_pieces_count :
  ∃ (w : ℕ), w > 0 ∧ 
    arrangements w black_pieces = total_arrangements ∧ 
    (w = black_pieces ∨ w = black_pieces + 1) :=
by sorry

end NUMINAMATH_CALUDE_white_pieces_count_l725_72580


namespace NUMINAMATH_CALUDE_condition_one_condition_two_condition_three_l725_72543

-- Define set A
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x = -1/(2*a)}

-- Theorem for condition ①
theorem condition_one : 
  ∀ a : ℝ, (A ∩ B a = B a) ↔ (a = 0 ∨ a = -1/2 ∨ a = 1/6) := by sorry

-- Theorem for condition ②
theorem condition_two :
  ∀ a : ℝ, ((Set.univ \ B a) ∩ A = {1}) ↔ (a = 1/6) := by sorry

-- Theorem for condition ③
theorem condition_three :
  ∀ a : ℝ, (A ∩ B a = ∅) ↔ (a ≠ 1/6 ∧ a ≠ -1/2) := by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_condition_three_l725_72543


namespace NUMINAMATH_CALUDE_proposition_validity_l725_72537

theorem proposition_validity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 - b^2 = 1) → (a - b < 1)) ∧
   ¬((1/b - 1/a = 1) → (a - b < 1)) ∧
   ((Real.exp a - Real.exp b = 1) → (a - b < 1)) ∧
   ¬((Real.log a - Real.log b = 1) → (a - b < 1))) := by
sorry

end NUMINAMATH_CALUDE_proposition_validity_l725_72537


namespace NUMINAMATH_CALUDE_green_mm_probability_l725_72540

-- Define the initial state and actions
def initial_green : ℕ := 20
def initial_red : ℕ := 20
def green_eaten : ℕ := 12
def red_eaten : ℕ := initial_red / 2
def yellow_added : ℕ := 14

-- Calculate the final numbers
def final_green : ℕ := initial_green - green_eaten
def final_red : ℕ := initial_red - red_eaten
def final_yellow : ℕ := yellow_added

-- Calculate the total number of M&Ms after all actions
def total_mms : ℕ := final_green + final_red + final_yellow

-- Define the probability of selecting a green M&M
def prob_green : ℚ := final_green / total_mms

-- Theorem statement
theorem green_mm_probability : prob_green = 1/4 := by sorry

end NUMINAMATH_CALUDE_green_mm_probability_l725_72540


namespace NUMINAMATH_CALUDE_sum_of_digits_of_big_number_l725_72559

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The number we're interested in -/
def big_number : ℕ := 10^95 - 97

/-- The theorem stating that the sum of digits of our big number is 840 -/
theorem sum_of_digits_of_big_number : sum_of_digits big_number = 840 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_big_number_l725_72559


namespace NUMINAMATH_CALUDE_car_lot_problem_l725_72560

theorem car_lot_problem (total : ℕ) (power_steering : ℕ) (power_windows : ℕ) (neither : ℕ) :
  total = 65 →
  power_steering = 45 →
  power_windows = 25 →
  neither = 12 →
  ∃ both : ℕ, both = 17 ∧
    total = power_steering + power_windows - both + neither :=
by sorry

end NUMINAMATH_CALUDE_car_lot_problem_l725_72560


namespace NUMINAMATH_CALUDE_days_until_birthday_l725_72549

/-- Proof of the number of days until Maria's birthday --/
theorem days_until_birthday (daily_savings : ℕ) (flower_cost : ℕ) (flowers_bought : ℕ) :
  daily_savings = 2 →
  flower_cost = 4 →
  flowers_bought = 11 →
  (flowers_bought * flower_cost) / daily_savings = 22 :=
by sorry

end NUMINAMATH_CALUDE_days_until_birthday_l725_72549


namespace NUMINAMATH_CALUDE_optimal_circular_sector_radius_l725_72533

/-- The radius that maximizes the area of a circular sector with given constraints -/
theorem optimal_circular_sector_radius : 
  ∀ (r : ℝ) (s : ℝ),
  -- Total perimeter is 32 meters
  2 * r + s = 32 →
  -- Ratio of radius to arc length is at least 2:3
  r / s ≥ 2 / 3 →
  -- Area of the sector is maximized
  ∀ (r' : ℝ) (s' : ℝ),
  2 * r' + s' = 32 →
  r' / s' ≥ 2 / 3 →
  r * s ≥ r' * s' →
  -- The optimal radius is 64/7
  r = 64 / 7 :=
by sorry

end NUMINAMATH_CALUDE_optimal_circular_sector_radius_l725_72533


namespace NUMINAMATH_CALUDE_zeros_in_Q_l725_72508

def R (k : ℕ+) : ℕ := (10^k.val - 1) / 9

def Q : ℕ := R 30 / R 6

def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 30 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l725_72508
