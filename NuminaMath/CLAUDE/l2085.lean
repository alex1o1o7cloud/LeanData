import Mathlib

namespace arithmetic_sequence_sum_l2085_208507

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end arithmetic_sequence_sum_l2085_208507


namespace range_of_S_l2085_208568

theorem range_of_S (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x ∧ x ≤ 1/2) 
  (h3 : S = x * y) : 
  -1/8 ≤ S ∧ S ≤ 0 := by
  sorry

end range_of_S_l2085_208568


namespace factorize_difference_of_squares_l2085_208588

variable (R : Type*) [CommRing R]
variable (a x y : R)

theorem factorize_difference_of_squares : a * x^2 - a * y^2 = a * (x + y) * (x - y) := by
  sorry

end factorize_difference_of_squares_l2085_208588


namespace even_function_implies_a_equals_four_l2085_208590

/-- Given that f(x) = (x + a)(x - 4) is an even function, prove that a = 4 --/
theorem even_function_implies_a_equals_four (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 4) = (-x + a) * (-x - 4)) →
  a = 4 := by
sorry

end even_function_implies_a_equals_four_l2085_208590


namespace correct_systematic_sample_l2085_208531

/-- Represents a systematic sample from a range of products. -/
structure SystematicSample where
  total_products : Nat
  sample_size : Nat
  start : Nat
  step : Nat

/-- Generates a systematic sample. -/
def generateSample (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.step)

/-- Checks if a sample is valid for the given total number of products. -/
def isValidSample (sample : List Nat) (total_products : Nat) : Prop :=
  sample.all (· < total_products) ∧ sample.length > 0 ∧ sample.Nodup

/-- Theorem: The correct systematic sample for 50 products with 5 samples is [1, 11, 21, 31, 41]. -/
theorem correct_systematic_sample :
  let sample := [1, 11, 21, 31, 41]
  let s : SystematicSample := {
    total_products := 50,
    sample_size := 5,
    start := 1,
    step := 10
  }
  generateSample s = sample ∧ isValidSample sample s.total_products := by
  sorry


end correct_systematic_sample_l2085_208531


namespace unique_solution_l2085_208596

/-- The vector [2, -3] -/
def v : Fin 2 → ℝ := ![2, -3]

/-- The vector [4, 7] -/
def w : Fin 2 → ℝ := ![4, 7]

/-- The equation to be solved -/
def equation (k : ℝ) : Prop :=
  ‖k • v - w‖ = 2 * Real.sqrt 13

/-- Theorem stating that k = -1 is the only solution -/
theorem unique_solution :
  ∃! k : ℝ, equation k ∧ k = -1 := by sorry

end unique_solution_l2085_208596


namespace alicia_sundae_cost_l2085_208562

/-- The cost of Alicia's peanut butter sundae given the prices of other sundaes and the final bill with tip -/
theorem alicia_sundae_cost (yvette_sundae brant_sundae josh_sundae : ℚ)
  (tip_percentage : ℚ) (final_bill : ℚ) :
  yvette_sundae = 9 →
  brant_sundae = 10 →
  josh_sundae = (17/2) →
  tip_percentage = (1/5) →
  final_bill = 42 →
  ∃ (alicia_sundae : ℚ),
    alicia_sundae = (final_bill / (1 + tip_percentage)) - (yvette_sundae + brant_sundae + josh_sundae) ∧
    alicia_sundae = (15/2) := by
  sorry

end alicia_sundae_cost_l2085_208562


namespace quadratic_expression_value_l2085_208508

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + 2 * y = 20) 
  (eq2 : 2 * x + 4 * y = 16) : 
  4 * x^2 + 12 * x * y + 12 * y^2 = 292 := by
sorry

end quadratic_expression_value_l2085_208508


namespace painted_subcubes_count_l2085_208566

def cube_size : ℕ := 4

-- Define a function to calculate the number of subcubes with at least two painted faces
def subcubes_with_two_or_more_painted_faces (n : ℕ) : ℕ :=
  8 + 12 * (n - 2)

-- Theorem statement
theorem painted_subcubes_count :
  subcubes_with_two_or_more_painted_faces cube_size = 32 := by
  sorry

end painted_subcubes_count_l2085_208566


namespace i_minus_one_squared_l2085_208510

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_minus_one_squared : (i - 1)^2 = -2*i := by
  sorry

end i_minus_one_squared_l2085_208510


namespace log_quarter_of_sixteen_eq_neg_two_l2085_208555

-- Define the logarithm function for base 1/4
noncomputable def log_quarter (x : ℝ) : ℝ := Real.log x / Real.log (1/4)

-- State the theorem
theorem log_quarter_of_sixteen_eq_neg_two :
  log_quarter 16 = -2 := by
  sorry

end log_quarter_of_sixteen_eq_neg_two_l2085_208555


namespace cauliflower_area_l2085_208505

theorem cauliflower_area (this_year_side : ℕ) (last_year_side : ℕ) 
  (h1 : this_year_side ^ 2 = 12544)
  (h2 : this_year_side ^ 2 = last_year_side ^ 2 + 223) :
  1 = 1 := by
  sorry

end cauliflower_area_l2085_208505


namespace function_passes_through_point_l2085_208574

-- Define the function f(x) = ax³ - 2x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x

-- Theorem statement
theorem function_passes_through_point (a : ℝ) :
  f a (-1) = 4 → a = -2 := by
  sorry

end function_passes_through_point_l2085_208574


namespace angle_sum_proof_l2085_208577

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = Real.sqrt 5 / 5) (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π/4 := by sorry

end angle_sum_proof_l2085_208577


namespace line_not_in_third_quadrant_l2085_208558

-- Define the line
def line (x : ℝ) : ℝ := -x + 1

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem line_not_in_third_quadrant :
  ∀ x : ℝ, ¬(third_quadrant x (line x)) :=
by
  sorry

end line_not_in_third_quadrant_l2085_208558


namespace custom_op_five_two_l2085_208592

-- Define the custom operation
def custom_op (a b : ℕ) : ℕ := 3*a + 4*b - a*b

-- State the theorem
theorem custom_op_five_two : custom_op 5 2 = 13 := by
  sorry

end custom_op_five_two_l2085_208592


namespace group_size_l2085_208527

theorem group_size (total_paise : ℕ) (h : total_paise = 7744) : 
  ∃ n : ℕ, n * n = total_paise ∧ n = 88 := by
  sorry

end group_size_l2085_208527


namespace candy_probability_theorem_l2085_208549

/-- The probability of selecting the same candy type for the first and last candy -/
def same_type_probability (lollipops chocolate jelly : ℕ) : ℚ :=
  let total := lollipops + chocolate + jelly
  let p_lollipop := (lollipops : ℚ) / total * (lollipops - 1) / (total - 1)
  let p_chocolate := (chocolate : ℚ) / total * (chocolate - 1) / (total - 1)
  let p_jelly := (jelly : ℚ) / total * (jelly - 1) / (total - 1)
  p_lollipop + p_chocolate + p_jelly

theorem candy_probability_theorem :
  same_type_probability 2 3 5 = 14 / 45 := by
  sorry

#eval same_type_probability 2 3 5

end candy_probability_theorem_l2085_208549


namespace final_peanut_count_l2085_208534

def peanut_problem (initial_peanuts : ℕ) (mary_adds : ℕ) (john_takes : ℕ) (friends : ℕ) : ℕ :=
  initial_peanuts + mary_adds - john_takes

theorem final_peanut_count :
  peanut_problem 4 4 2 2 = 6 := by sorry

end final_peanut_count_l2085_208534


namespace tangent_line_at_origin_l2085_208533

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem: The tangent line to y = x^3 - 3x at (0, 0) is y = -3x
theorem tangent_line_at_origin : 
  ∀ x : ℝ, (f' 0) * x = -3 * x := by sorry

end tangent_line_at_origin_l2085_208533


namespace f_has_three_distinct_roots_l2085_208504

/-- The polynomial function whose roots we want to count -/
def f (x : ℝ) : ℝ := (x + 5) * (x^2 + 5*x - 6)

/-- The statement that f has exactly 3 distinct real roots -/
theorem f_has_three_distinct_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end f_has_three_distinct_roots_l2085_208504


namespace invisible_dots_count_l2085_208553

/-- The sum of dots on a standard six-sided die -/
def standard_die_sum : Nat := 21

/-- The total number of dots on five standard six-sided dice -/
def total_dots (n : Nat) : Nat := n * standard_die_sum

/-- The sum of visible dots in the given configuration -/
def visible_dots : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 2 + 3 + 4 + 5 + 6 + 4 + 5 + 6

/-- The number of dice in the problem -/
def num_dice : Nat := 5

/-- The number of visible faces in the problem -/
def num_visible_faces : Nat := 14

theorem invisible_dots_count :
  total_dots num_dice - visible_dots = 49 :=
sorry

end invisible_dots_count_l2085_208553


namespace edward_spent_sixteen_l2085_208556

/-- The amount of money Edward spent -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem: Edward spent $16 -/
theorem edward_spent_sixteen :
  amount_spent 18 2 = 16 := by
  sorry

end edward_spent_sixteen_l2085_208556


namespace gina_hourly_rate_l2085_208540

/-- Gina's painting rates and order details -/
structure PaintingJob where
  rose_rate : ℕ  -- Cups with roses painted per hour
  lily_rate : ℕ  -- Cups with lilies painted per hour
  rose_order : ℕ  -- Number of rose cups ordered
  lily_order : ℕ  -- Number of lily cups ordered
  total_payment : ℕ  -- Total payment for the order in dollars

/-- Calculate Gina's hourly rate for a given painting job -/
def hourly_rate (job : PaintingJob) : ℚ :=
  job.total_payment / (job.rose_order / job.rose_rate + job.lily_order / job.lily_rate)

/-- Theorem: Gina's hourly rate for the given job is $30 -/
theorem gina_hourly_rate :
  let job : PaintingJob := {
    rose_rate := 6,
    lily_rate := 7,
    rose_order := 6,
    lily_order := 14,
    total_payment := 90
  }
  hourly_rate job = 30 := by
  sorry

end gina_hourly_rate_l2085_208540


namespace sequence_sum_theorem_l2085_208589

/-- Given a sequence {a_n} with S_n being the sum of its first n terms,
    if S_n^2 - 2S_n - a_nS_n + 1 = 0 for all positive integers n,
    then S_n = n / (n + 1) for all positive integers n. -/
theorem sequence_sum_theorem (a : ℕ+ → ℚ) (S : ℕ+ → ℚ)
    (h : ∀ n : ℕ+, S n ^ 2 - 2 * S n - a n * S n + 1 = 0) :
  ∀ n : ℕ+, S n = n / (n + 1) := by
  sorry

end sequence_sum_theorem_l2085_208589


namespace inequality_satisfied_by_five_integers_l2085_208584

theorem inequality_satisfied_by_five_integers :
  ∃! (S : Finset ℤ), (∀ n ∈ S, Real.sqrt (n + 1 : ℝ) ≤ Real.sqrt (5 * n - 7 : ℝ) ∧
                               Real.sqrt (5 * n - 7 : ℝ) < Real.sqrt (3 * n + 6 : ℝ)) ∧
                     S.card = 5 :=
by sorry

end inequality_satisfied_by_five_integers_l2085_208584


namespace lattice_point_decomposition_l2085_208519

/-- Represents a point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a parallelogram OABC where O is the origin -/
structure Parallelogram where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is in or on a triangle -/
def inTriangle (P Q R S : LatticePoint) : Prop := sorry

/-- Vector addition -/
def vecAdd (P Q : LatticePoint) : LatticePoint := sorry

theorem lattice_point_decomposition 
  (OABC : Parallelogram) 
  (P : LatticePoint) 
  (h : inTriangle P OABC.A OABC.B OABC.C) :
  ∃ (Q R : LatticePoint), 
    inTriangle Q (LatticePoint.mk 0 0) OABC.A OABC.C ∧ 
    inTriangle R (LatticePoint.mk 0 0) OABC.A OABC.C ∧
    P = vecAdd Q R := by sorry

end lattice_point_decomposition_l2085_208519


namespace max_circle_sum_l2085_208572

/-- Represents the seven regions formed by the intersection of three circles -/
inductive Region
| A  -- shared by all three circles
| B  -- shared by two circles
| C  -- shared by two circles
| D  -- shared by two circles
| E  -- in one circle only
| F  -- in one circle only
| G  -- in one circle only

/-- Assignment of integers to regions -/
def Assignment := Region → Fin 7

/-- A circle is represented by the four regions it contains -/
structure Circle :=
  (r1 r2 r3 r4 : Region)

/-- The three circles in the problem -/
def circles : Fin 3 → Circle := sorry

/-- The sum of values in a circle for a given assignment -/
def circleSum (a : Assignment) (c : Circle) : ℕ :=
  a c.r1 + a c.r2 + a c.r3 + a c.r4

/-- An assignment is valid if all values are distinct -/
def validAssignment (a : Assignment) : Prop :=
  ∀ r1 r2 : Region, r1 ≠ r2 → a r1 ≠ a r2

/-- An assignment satisfies the equal sum condition -/
def satisfiesEqualSum (a : Assignment) : Prop :=
  ∀ c1 c2 : Fin 3, circleSum a (circles c1) = circleSum a (circles c2)

/-- The maximum possible sum for each circle -/
def maxSum : ℕ := 15

theorem max_circle_sum :
  ∃ (a : Assignment), validAssignment a ∧ satisfiesEqualSum a ∧
  (∀ c : Fin 3, circleSum a (circles c) = maxSum) ∧
  (∀ (a' : Assignment), validAssignment a' ∧ satisfiesEqualSum a' →
    ∀ c : Fin 3, circleSum a' (circles c) ≤ maxSum) := by
  sorry

end max_circle_sum_l2085_208572


namespace complex_in_second_quadrant_m_range_l2085_208528

theorem complex_in_second_quadrant_m_range (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 2) (m - 1)
  (z.re < 0 ∧ z.im > 0) → (1 < m ∧ m < Real.sqrt 2) := by
  sorry

end complex_in_second_quadrant_m_range_l2085_208528


namespace cricket_team_average_age_l2085_208513

theorem cricket_team_average_age
  (n : ℕ) -- Total number of players
  (a : ℝ) -- Average age of the whole team
  (h1 : n = 11)
  (h2 : a = 28)
  (h3 : ((n * a) - (a + (a + 3))) / (n - 2) = a - 1) :
  a = 28 := by
sorry

end cricket_team_average_age_l2085_208513


namespace other_root_of_quadratic_l2085_208525

/-- Given a quadratic equation 3x^2 + mx - 7 = 0 where -1 is one root, 
    prove that the other root is 7/3 -/
theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x, 3 * x^2 + m * x - 7 = 0 ∧ x = -1) → 
  (∃ y, 3 * y^2 + m * y - 7 = 0 ∧ y = 7/3) :=
by sorry

end other_root_of_quadratic_l2085_208525


namespace topsoil_cost_theorem_l2085_208585

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The amount of topsoil in cubic yards -/
def amount_in_cubic_yards : ℝ := 7

/-- The cost of topsoil for a given amount of cubic yards -/
def topsoil_cost (amount : ℝ) : ℝ :=
  amount * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_theorem :
  topsoil_cost amount_in_cubic_yards = 1512 := by
  sorry

end topsoil_cost_theorem_l2085_208585


namespace polygon_sides_l2085_208554

/-- Given a polygon with sum of interior angles equal to 1080°, prove it has 8 sides -/
theorem polygon_sides (sum_interior_angles : ℝ) (h : sum_interior_angles = 1080) : 
  (sum_interior_angles / 180 + 2 : ℝ) = 8 := by
  sorry

end polygon_sides_l2085_208554


namespace roots_condition_l2085_208587

-- Define the quadratic function F(x)
def F (R l a x : ℝ) := 2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2

-- Define the conditions for the roots to be between 0 and 2R
def roots_between_0_and_2R (R l a : ℝ) : Prop :=
  (0 < a ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (-2 * R < a ∧ a < 0 ∧ l^2 < (2 * R - a)^2)

-- Theorem statement
theorem roots_condition (R l a : ℝ) (hR : R > 0) (hl : l > 0) (ha : a ≠ 0) :
  (∀ x, F R l a x = 0 → 0 < x ∧ x < 2 * R) ↔ roots_between_0_and_2R R l a := by
  sorry

end roots_condition_l2085_208587


namespace modular_equivalence_in_range_l2085_208501

theorem modular_equivalence_in_range (a b : ℤ) (h1 : a ≡ 54 [ZMOD 53]) (h2 : b ≡ 98 [ZMOD 53]) :
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 200 ∧ (a - b) ≡ n [ZMOD 53] ∧ n = 168 := by
  sorry

end modular_equivalence_in_range_l2085_208501


namespace polynomial_identity_l2085_208517

theorem polynomial_identity : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end polynomial_identity_l2085_208517


namespace females_together_arrangements_l2085_208597

/-- Represents the number of students of each gender -/
def num_males : ℕ := 2
def num_females : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_males + num_females

/-- The number of ways to arrange the students with females next to each other -/
def arrangements_with_females_together : ℕ := 12

/-- Theorem stating that the number of arrangements with females together is 12 -/
theorem females_together_arrangements :
  (arrangements_with_females_together = 12) ∧
  (num_males = 2) ∧
  (num_females = 2) ∧
  (total_students = 4) := by
  sorry

end females_together_arrangements_l2085_208597


namespace onion_bag_cost_l2085_208550

/-- The cost of one bag of onions -/
def cost_of_one_bag (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) : ℕ :=
  (price_per_onion * total_onions) / num_bags

/-- Theorem stating the cost of one bag of onions -/
theorem onion_bag_cost :
  let price_per_onion := 200
  let total_onions := 180
  let num_bags := 6
  cost_of_one_bag price_per_onion total_onions num_bags = 6000 := by
  sorry

end onion_bag_cost_l2085_208550


namespace james_water_storage_l2085_208522

def cask_capacity : ℕ := 20

def barrel_capacity (cask_cap : ℕ) : ℕ := 2 * cask_cap + 3

def total_storage (cask_cap barrel_cap num_barrels : ℕ) : ℕ :=
  cask_cap + num_barrels * barrel_cap

theorem james_water_storage :
  total_storage cask_capacity (barrel_capacity cask_capacity) 4 = 192 := by
  sorry

end james_water_storage_l2085_208522


namespace bicycle_time_calculation_l2085_208560

def total_distance : ℝ := 20
def bicycle_speed : ℝ := 30
def running_speed : ℝ := 8
def total_time : ℝ := 117

theorem bicycle_time_calculation (t : ℝ) 
  (h1 : t ≥ 0) 
  (h2 : t ≤ total_time) 
  (h3 : (t / 60) * bicycle_speed + ((total_time - t) / 60) * running_speed = total_distance) : 
  t = 12 := by sorry

end bicycle_time_calculation_l2085_208560


namespace average_speed_first_part_l2085_208516

def total_distance : ℝ := 250
def total_time : ℝ := 5.4
def distance_at_v : ℝ := 148
def speed_known : ℝ := 60

theorem average_speed_first_part (v : ℝ) : 
  (distance_at_v / v) + ((total_distance - distance_at_v) / speed_known) = total_time →
  v = 40 := by
sorry

end average_speed_first_part_l2085_208516


namespace section_A_average_weight_l2085_208512

/-- Proves that the average weight of section A is 40 kg given the conditions of the problem -/
theorem section_A_average_weight
  (students_A : ℕ)
  (students_B : ℕ)
  (avg_weight_B : ℝ)
  (avg_weight_total : ℝ)
  (h1 : students_A = 30)
  (h2 : students_B = 20)
  (h3 : avg_weight_B = 35)
  (h4 : avg_weight_total = 38) :
  let total_students := students_A + students_B
  let avg_weight_A := (avg_weight_total * total_students - avg_weight_B * students_B) / students_A
  avg_weight_A = 40 := by
sorry


end section_A_average_weight_l2085_208512


namespace projection_equals_negative_two_l2085_208542

def a : Fin 2 → ℝ
| 0 => 4
| 1 => -7

def b : Fin 2 → ℝ
| 0 => 3
| 1 => -4

theorem projection_equals_negative_two :
  let proj := (((a - 2 • b) • b) / (b • b)) • b
  proj = (-2 : ℝ) • b :=
by sorry

end projection_equals_negative_two_l2085_208542


namespace f_nonpositive_implies_k_geq_one_l2085_208543

open Real

/-- Given a function f(x) = ln(ex) - kx defined on (0, +∞), 
    if f(x) ≤ 0 for all x > 0, then k ≥ 1 -/
theorem f_nonpositive_implies_k_geq_one (k : ℝ) : 
  (∀ x > 0, Real.log (Real.exp 1 * x) - k * x ≤ 0) → k ≥ 1 := by
  sorry

end f_nonpositive_implies_k_geq_one_l2085_208543


namespace rungs_on_twenty_ladders_eq_1200_l2085_208567

/-- Calculates the number of rungs on 20 ladders given the following conditions:
  * There are 10 ladders with 50 rungs each
  * There are 20 additional ladders with an unknown number of rungs
  * Each rung costs $2
  * The total cost for all ladders is $3,400
-/
def rungs_on_twenty_ladders : ℕ :=
  let ladders_with_fifty_rungs : ℕ := 10
  let rungs_per_ladder : ℕ := 50
  let cost_per_rung : ℕ := 2
  let total_cost : ℕ := 3400
  let remaining_ladders : ℕ := 20
  
  let cost_of_fifty_rung_ladders : ℕ := ladders_with_fifty_rungs * rungs_per_ladder * cost_per_rung
  let remaining_cost : ℕ := total_cost - cost_of_fifty_rung_ladders
  remaining_cost / cost_per_rung

theorem rungs_on_twenty_ladders_eq_1200 : rungs_on_twenty_ladders = 1200 := by
  sorry

end rungs_on_twenty_ladders_eq_1200_l2085_208567


namespace fractional_equation_solution_l2085_208551

theorem fractional_equation_solution (k : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) ↔ k ≠ -3 ∧ k ≠ 5 :=
by sorry

end fractional_equation_solution_l2085_208551


namespace max_k_value_l2085_208565

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 22) / 2 := by
sorry

end max_k_value_l2085_208565


namespace monotonicity_intervals_two_zeros_condition_l2085_208502

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((1-a)/2) * x^2 - a*x - a

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) (h : a > 0) :
  (∀ x < -1, (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo (-1) a, (f' a x < 0)) ∧
  (∀ x > a, (f' a x > 0)) :=
sorry

-- Theorem for the range of a when f has exactly two zeros in (-2, 0)
theorem two_zeros_condition (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Ioo (-2) 0 ∧ y ∈ Set.Ioo (-2) 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   ∀ z ∈ Set.Ioo (-2) 0, f a z = 0 → (z = x ∨ z = y)) ↔
  (a > 0 ∧ a < 1/3) :=
sorry

end

end monotonicity_intervals_two_zeros_condition_l2085_208502


namespace fuel_mixture_problem_l2085_208526

/-- Proves that the amount of fuel A added to the tank is 122 gallons -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ)
  (h1 : tank_capacity = 218)
  (h2 : ethanol_a = 0.12)
  (h3 : ethanol_b = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_a : ℝ), fuel_a = 122 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end fuel_mixture_problem_l2085_208526


namespace group_average_score_l2085_208539

theorem group_average_score (class_average : ℝ) (differences : List ℝ) : 
  class_average = 80 →
  differences = [2, 3, -3, -5, 12, 14, 10, 4, -6, 4, -11, -7, 8, -2] →
  (class_average + (differences.sum / differences.length)) = 81.64 := by
sorry

end group_average_score_l2085_208539


namespace prism_24_edges_has_10_faces_l2085_208506

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular lateral faces. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges -/
def num_faces (p : Prism) : ℕ :=
  let base_edges := p.edges / 3
  base_edges + 2

theorem prism_24_edges_has_10_faces (p : Prism) (h : p.edges = 24) : num_faces p = 10 := by
  sorry

end prism_24_edges_has_10_faces_l2085_208506


namespace prime_factorization_sum_l2085_208595

theorem prime_factorization_sum (a b c d : ℕ) : 
  2^a * 3^b * 5^c * 11^d = 14850 → 3*a + 2*b + 4*c + 6*d = 23 := by
  sorry

end prime_factorization_sum_l2085_208595


namespace like_terms_value_l2085_208524

theorem like_terms_value (m n : ℕ) (a b c : ℝ) : 
  (∃ k : ℝ, 3 * a^m * b * c^2 = k * (-2 * a^3 * b^n * c^2)) → 
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 51 :=
by sorry

end like_terms_value_l2085_208524


namespace simplify_and_ratio_l2085_208579

theorem simplify_and_ratio : ∀ m : ℝ, 
  (6 * m + 12) / 3 = 2 * m + 4 ∧ 2 / 4 = (1 : ℚ) / 2 := by
  sorry

end simplify_and_ratio_l2085_208579


namespace quadratic_always_positive_implies_a_greater_than_one_l2085_208578

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
  sorry

end quadratic_always_positive_implies_a_greater_than_one_l2085_208578


namespace max_stores_visited_l2085_208503

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  stores : Nat
  total_visits : Nat
  unique_visitors : Nat
  double_visitors : Nat

/-- Theorem stating the maximum number of stores visited by any single person -/
theorem max_stores_visited (s : ShoppingScenario) 
  (h1 : s.stores = 7)
  (h2 : s.total_visits = 21)
  (h3 : s.unique_visitors = 11)
  (h4 : s.double_visitors = 7)
  (h5 : s.double_visitors ≤ s.unique_visitors)
  (h6 : s.double_visitors * 2 ≤ s.total_visits) :
  ∃ (max_visits : Nat), max_visits = 4 ∧ 
  ∀ (individual_visits : Nat), individual_visits ≤ max_visits :=
by sorry

end max_stores_visited_l2085_208503


namespace abs_minus_self_nonnegative_l2085_208563

theorem abs_minus_self_nonnegative (x : ℝ) : |x| - x ≥ 0 := by
  sorry

end abs_minus_self_nonnegative_l2085_208563


namespace word_count_correct_l2085_208594

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the word -/
def word_length : ℕ := 5

/-- The number of positions that can vary (middle letters) -/
def varying_positions : ℕ := word_length - 2

/-- The number of five-letter words where the first and last letters are the same -/
def num_words : ℕ := alphabet_size * (alphabet_size ^ varying_positions)

theorem word_count_correct : num_words = 456976 := by sorry

end word_count_correct_l2085_208594


namespace flower_bed_max_area_l2085_208546

/-- Given a rectangular flower bed with one side against a house,
    using 450 feet of total fencing with 150 feet along the house,
    the maximum area of the flower bed is 22500 square feet. -/
theorem flower_bed_max_area :
  ∀ (l w : ℝ),
  l = 150 →
  l + 2 * w = 450 →
  l * w ≤ 22500 :=
by sorry

end flower_bed_max_area_l2085_208546


namespace polygon_equal_sides_different_angles_l2085_208532

-- Define a polygon type
inductive Polygon
| Triangle
| Quadrilateral
| Pentagon

-- Function to check if a polygon can have all sides equal and all angles different
def canHaveEqualSidesAndDifferentAngles (p : Polygon) : Prop :=
  match p with
  | Polygon.Triangle => False
  | Polygon.Quadrilateral => False
  | Polygon.Pentagon => True

-- Theorem statement
theorem polygon_equal_sides_different_angles :
  (canHaveEqualSidesAndDifferentAngles Polygon.Triangle = False) ∧
  (canHaveEqualSidesAndDifferentAngles Polygon.Quadrilateral = False) ∧
  (canHaveEqualSidesAndDifferentAngles Polygon.Pentagon = True) := by
  sorry

#check polygon_equal_sides_different_angles

end polygon_equal_sides_different_angles_l2085_208532


namespace cost_for_3150_pencils_l2085_208514

/-- Calculates the total cost of pencils with a bulk discount --/
def total_cost_with_discount (pencils_per_box : ℕ) (regular_price : ℚ) 
  (discount_price : ℚ) (discount_threshold : ℕ) (total_pencils : ℕ) : ℚ :=
  let boxes := (total_pencils + pencils_per_box - 1) / pencils_per_box
  let price_per_box := if total_pencils > discount_threshold then discount_price else regular_price
  boxes * price_per_box

/-- Theorem stating the total cost for 3150 pencils --/
theorem cost_for_3150_pencils : 
  total_cost_with_discount 150 40 35 2000 3150 = 735 := by
  sorry

end cost_for_3150_pencils_l2085_208514


namespace fourth_roll_eight_prob_l2085_208547

-- Define the probabilities for the fair die
def fair_die_prob : ℚ := 1 / 8

-- Define the probabilities for the biased die
def biased_die_prob_eight : ℚ := 3 / 4
def biased_die_prob_other : ℚ := 1 / 28

-- Define the probability of selecting each die
def die_selection_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the theorem
theorem fourth_roll_eight_prob :
  let p_fair_three_eights : ℚ := fair_die_prob ^ 3
  let p_biased_three_eights : ℚ := biased_die_prob_eight ^ 3
  let p_three_eights : ℚ := die_selection_prob * p_fair_three_eights + die_selection_prob * p_biased_three_eights
  let p_fair_given_three_eights : ℚ := (die_selection_prob * p_fair_three_eights) / p_three_eights
  let p_biased_given_three_eights : ℚ := (die_selection_prob * p_biased_three_eights) / p_three_eights
  let p_fourth_eight : ℚ := p_fair_given_three_eights * fair_die_prob + p_biased_given_three_eights * biased_die_prob_eight
  p_fourth_eight = 1297 / 1736 :=
by sorry

end fourth_roll_eight_prob_l2085_208547


namespace trivia_team_distribution_l2085_208569

theorem trivia_team_distribution (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58)
  (h2 : not_picked = 10)
  (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 := by
  sorry

end trivia_team_distribution_l2085_208569


namespace number_of_factors_of_N_l2085_208571

def N : ℕ := 17^3 + 3 * 17^2 + 3 * 17 + 1

theorem number_of_factors_of_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 28 := by
  sorry

end number_of_factors_of_N_l2085_208571


namespace complex_ratio_range_l2085_208598

theorem complex_ratio_range (x y : ℝ) :
  let z : ℂ := x + y * Complex.I
  let ratio := (z + 1) / (z + 2)
  (ratio.re / ratio.im = Real.sqrt 3) →
  (y / x ∈ Set.Icc ((Real.sqrt 3 * -3 - 4 * Real.sqrt 2) / 5) ((Real.sqrt 3 * -3 + 4 * Real.sqrt 2) / 5)) :=
by sorry

end complex_ratio_range_l2085_208598


namespace broadcasting_methods_count_l2085_208552

/-- The number of different commercial advertisements -/
def num_commercial : ℕ := 3

/-- The number of different Olympic promotional advertisements -/
def num_olympic : ℕ := 2

/-- The total number of advertisements -/
def total_ads : ℕ := 5

/-- Function to calculate the number of broadcasting methods -/
def num_broadcasting_methods : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that the number of broadcasting methods is 36 -/
theorem broadcasting_methods_count :
  num_broadcasting_methods = 36 :=
sorry

end broadcasting_methods_count_l2085_208552


namespace ceiling_neg_sqrt_36_l2085_208575

theorem ceiling_neg_sqrt_36 : ⌈-Real.sqrt 36⌉ = -6 := by sorry

end ceiling_neg_sqrt_36_l2085_208575


namespace triangle_perimeter_l2085_208581

/-- Given a triangle with two sides of lengths 3 and 4, and the third side being the root
    of x^2 - 12x + 35 = 0 that satisfies the triangle inequality, the perimeter is 12. -/
theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 12*x + 35 = 0 →
  x > 0 →
  x < 3 + 4 →
  x > |3 - 4| →
  3 + 4 + x = 12 := by
  sorry


end triangle_perimeter_l2085_208581


namespace ice_skate_rental_fee_l2085_208599

/-- The rental fee for ice skates at a rink, given the admission fee, cost of new skates, and number of visits to justify buying. -/
theorem ice_skate_rental_fee 
  (admission_fee : ℚ) 
  (new_skates_cost : ℚ) 
  (visits_to_justify : ℕ) 
  (h1 : admission_fee = 5)
  (h2 : new_skates_cost = 65)
  (h3 : visits_to_justify = 26) :
  let rental_fee := (new_skates_cost + admission_fee * visits_to_justify) / visits_to_justify - admission_fee
  rental_fee = (5/2 : ℚ) := by
sorry

end ice_skate_rental_fee_l2085_208599


namespace no_odd_faced_odd_edged_polyhedron_l2085_208530

/-- Represents a face of a polyhedron -/
structure Face where
  edges : Nat
  odd_edges : Odd edges

/-- Represents a polyhedron -/
structure Polyhedron where
  faces : List Face
  odd_faces : Odd faces.length

/-- Theorem stating that a polyhedron with an odd number of faces, 
    each having an odd number of edges, cannot exist -/
theorem no_odd_faced_odd_edged_polyhedron : 
  ¬ ∃ (p : Polyhedron), True := by sorry

end no_odd_faced_odd_edged_polyhedron_l2085_208530


namespace customers_before_rush_count_l2085_208535

/-- The number of customers who didn't leave a tip -/
def no_tip : ℕ := 49

/-- The number of customers who left a tip -/
def left_tip : ℕ := 2

/-- The number of additional customers during lunch rush -/
def additional_customers : ℕ := 12

/-- The total number of customers after the lunch rush -/
def total_after_rush : ℕ := no_tip + left_tip

/-- The number of customers before the lunch rush -/
def customers_before_rush : ℕ := total_after_rush - additional_customers

theorem customers_before_rush_count : customers_before_rush = 39 := by
  sorry

end customers_before_rush_count_l2085_208535


namespace reflect_point_D_l2085_208593

/-- Reflect a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point across the line y = x - 1 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 1, p.1 - 1)

/-- The main theorem stating that reflecting D(5,0) across y-axis and then y=x-1 results in (-1,4) -/
theorem reflect_point_D : 
  let D : ℝ × ℝ := (5, 0)
  let D' := reflect_y_axis D
  let D'' := reflect_line D'
  D'' = (-1, 4) := by sorry

end reflect_point_D_l2085_208593


namespace sphere_cube_volume_comparison_l2085_208538

theorem sphere_cube_volume_comparison :
  ∀ (r a : ℝ), r > 0 → a > 0 →
  4 * π * r^2 = 6 * a^2 →
  (4/3) * π * r^3 > a^3 := by
sorry

end sphere_cube_volume_comparison_l2085_208538


namespace four_digit_count_l2085_208557

-- Define the range of four-digit numbers
def four_digit_start : ℕ := 1000
def four_digit_end : ℕ := 9999

-- Theorem statement
theorem four_digit_count : 
  (Finset.range (four_digit_end - four_digit_start + 1)).card = 9000 :=
by sorry

end four_digit_count_l2085_208557


namespace range_of_g_l2085_208511

/-- The function g(x) = ⌊2x⌋ - 2x has a range of [-1, 0] -/
theorem range_of_g : 
  let g : ℝ → ℝ := λ x => ⌊2 * x⌋ - 2 * x
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ -1 ≤ y ∧ y ≤ 0 := by
  sorry

end range_of_g_l2085_208511


namespace sum_remainder_mod_seven_l2085_208564

theorem sum_remainder_mod_seven :
  (9543 + 9544 + 9545 + 9546 + 9547) % 7 = 0 := by
  sorry

end sum_remainder_mod_seven_l2085_208564


namespace circle_construction_l2085_208583

/-- Given a circle k0 with diameter AB and center O0, and additional circles k1, k2, k3, k4, k5, k6
    constructed as described in the problem, prove that their radii are in specific ratios to r0. -/
theorem circle_construction (r0 : ℝ) (r1 r2 r3 r4 r5 r6 : ℝ) 
  (h1 : r0 > 0)
  (h2 : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ r5 > 0 ∧ r6 > 0)
  (h3 : ∃ (A B O0 : ℝ × ℝ), ‖A - B‖ = 2 * r0 ∧ O0 = (A + B) / 2)
  (h4 : ∃ (k1 k1' : Set (ℝ × ℝ)), k1 ∩ k1' = {O0}) :
  r1 = r0 / 2 ∧ r2 = r0 / 3 ∧ r3 = r0 / 6 ∧ r4 = r0 / 4 ∧ r5 = r0 / 7 ∧ r6 = r0 / 8 := by
  sorry


end circle_construction_l2085_208583


namespace B_equals_one_four_l2085_208521

def A : Set ℝ := {0, 1, 2, 3}

def B (m : ℝ) : Set ℝ := {x | x^2 - 5*x + m = 0}

theorem B_equals_one_four (m : ℝ) : 
  (A ∩ B m = {1}) → B m = {1, 4} := by
  sorry

end B_equals_one_four_l2085_208521


namespace exam_average_l2085_208544

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 110)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  (passed_avg * passed_boys + failed_avg * (total_boys - passed_boys)) / total_boys = 37 := by
  sorry

end exam_average_l2085_208544


namespace arccos_sqrt2_over_2_l2085_208580

theorem arccos_sqrt2_over_2 : Real.arccos (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end arccos_sqrt2_over_2_l2085_208580


namespace solution_set_part1_solution_set_part2_l2085_208518

-- Define the function f(x) = |x-a| + x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3*x} = {x : ℝ | x ≥ 2}) → a = 6 := by sorry

end solution_set_part1_solution_set_part2_l2085_208518


namespace homework_time_calculation_l2085_208559

/-- The time Max spent on biology homework in minutes -/
def biology_time : ℝ := 24

/-- The time Max spent on history homework in minutes -/
def history_time : ℝ := 1.5 * biology_time

/-- The time Max spent on chemistry homework in minutes -/
def chemistry_time : ℝ := biology_time * 0.7

/-- The time Max spent on English homework in minutes -/
def english_time : ℝ := 2 * (history_time + chemistry_time)

/-- The time Max spent on geography homework in minutes -/
def geography_time : ℝ := 3 * history_time + 0.75 * english_time

/-- The total time Max spent on homework in minutes -/
def total_homework_time : ℝ := biology_time + history_time + chemistry_time + english_time + geography_time

theorem homework_time_calculation :
  total_homework_time = 369.6 := by sorry

end homework_time_calculation_l2085_208559


namespace special_triangle_bc_length_l2085_208536

/-- A triangle with special properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Length of side AB is 1
  ab_length : dist A B = 1
  -- Length of side AC is 2
  ac_length : dist A C = 2
  -- Median from A to BC has same length as BC
  median_eq_bc : dist A ((B + C) / 2) = dist B C

/-- The length of BC in a SpecialTriangle is √2 -/
theorem special_triangle_bc_length (t : SpecialTriangle) : dist t.B t.C = Real.sqrt 2 := by
  sorry

end special_triangle_bc_length_l2085_208536


namespace sum_simplification_l2085_208500

theorem sum_simplification : -1^2022 + (-1)^2023 + 1^2024 - 1^2025 = -2 := by
  sorry

end sum_simplification_l2085_208500


namespace ten_digit_numbers_with_repeats_l2085_208591

theorem ten_digit_numbers_with_repeats (n : ℕ) : n = 9 * 10^9 - 9 * Nat.factorial 9 :=
  by
    sorry

end ten_digit_numbers_with_repeats_l2085_208591


namespace derivative_of_product_l2085_208548

theorem derivative_of_product (x : ℝ) :
  deriv (fun x => (3 * x^2 - 4*x) * (2*x + 1)) x = 18 * x^2 - 10 * x - 4 := by
sorry

end derivative_of_product_l2085_208548


namespace difference_of_squares_65_35_l2085_208545

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l2085_208545


namespace babysitter_earnings_correct_l2085_208576

/-- Calculates the babysitter's earnings for a given number of hours worked -/
def babysitter_earnings (regular_rate : ℕ) (regular_hours : ℕ) (overtime_rate : ℕ) (total_hours : ℕ) : ℕ :=
  let regular_pay := min regular_hours total_hours * regular_rate
  let overtime_pay := max 0 (total_hours - regular_hours) * overtime_rate
  regular_pay + overtime_pay

theorem babysitter_earnings_correct :
  let regular_rate : ℕ := 16
  let regular_hours : ℕ := 30
  let overtime_rate : ℕ := 28  -- 16 + (75% of 16)
  let total_hours : ℕ := 40
  babysitter_earnings regular_rate regular_hours overtime_rate total_hours = 760 :=
by sorry

end babysitter_earnings_correct_l2085_208576


namespace johnny_closed_days_l2085_208537

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of crab dishes Johnny makes per day -/
def dishes_per_day : ℕ := 40

/-- Represents the amount of crab meat used per dish in pounds -/
def crab_per_dish : ℚ := 3/2

/-- Represents the cost of crab meat per pound in dollars -/
def crab_cost_per_pound : ℕ := 8

/-- Represents Johnny's weekly expenditure on crab meat in dollars -/
def weekly_expenditure : ℕ := 1920

/-- Theorem stating that Johnny is closed 3 days a week -/
theorem johnny_closed_days : 
  days_in_week - (weekly_expenditure / (dishes_per_day * crab_per_dish * crab_cost_per_pound)) = 3 := by
  sorry

end johnny_closed_days_l2085_208537


namespace symmetric_points_range_l2085_208515

noncomputable section

open Real

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

def g (x : ℝ) : ℝ := exp x

def h (x : ℝ) : ℝ := log x

theorem symmetric_points_range (a : ℝ) :
  (∃ x y : ℝ, 1/e ≤ x ∧ x ≤ e ∧ 1/e ≤ y ∧ y ≤ e ∧
    f a x = g y ∧ f a y = g x) →
  1 ≤ a ∧ a ≤ e + 1/e :=
by sorry

end

end symmetric_points_range_l2085_208515


namespace fraction_evaluation_l2085_208586

theorem fraction_evaluation (a b : ℝ) (h1 : a = 7) (h2 : b = 4) :
  5 / (a - b)^2 = 5 / 9 := by sorry

end fraction_evaluation_l2085_208586


namespace pi_half_irrational_l2085_208529

theorem pi_half_irrational : Irrational (Real.pi / 2) := by
  sorry

end pi_half_irrational_l2085_208529


namespace antibiotics_cost_proof_l2085_208561

def antibiotics_problem (doses_per_day : ℕ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (doses_per_day * days)

theorem antibiotics_cost_proof (doses_per_day : ℕ) (days : ℕ) (total_cost : ℚ) 
  (h1 : doses_per_day = 3)
  (h2 : days = 7)
  (h3 : total_cost = 63) :
  antibiotics_problem doses_per_day days total_cost = 3 := by
sorry

end antibiotics_cost_proof_l2085_208561


namespace triangle_side_ratio_maximum_l2085_208541

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the area of the triangle is (1/2)c^2, the maximum value of (a^2 + b^2 + c^2) / (ab) is 2√2. -/
theorem triangle_side_ratio_maximum (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  (∃ (x : ℝ), (a^2 + b^2 + c^2) / (a * b) ≤ x) ∧
  (∀ (x : ℝ), (a^2 + b^2 + c^2) / (a * b) ≤ x → x ≥ 2 * Real.sqrt 2) :=
by sorry


end triangle_side_ratio_maximum_l2085_208541


namespace piano_lesson_cost_l2085_208523

/-- Calculate the total cost of piano lessons -/
theorem piano_lesson_cost (lesson_cost : ℝ) (lesson_duration : ℝ) (total_hours : ℝ) : 
  lesson_cost = 30 ∧ lesson_duration = 1.5 ∧ total_hours = 18 →
  (total_hours / lesson_duration) * lesson_cost = 360 := by
  sorry

end piano_lesson_cost_l2085_208523


namespace abc_sum_mod_five_l2085_208582

theorem abc_sum_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 5 = 1 →
  (3 * c) % 5 = 1 →
  (4 * b) % 5 = (1 + b) % 5 →
  (a + b + c) % 5 = 3 := by
  sorry

end abc_sum_mod_five_l2085_208582


namespace initial_sum_calculation_l2085_208509

/-- Proves that given a total amount of Rs. 15,500 after 4 years with a simple interest rate of 6% per annum, the initial sum of money (principal) is Rs. 12,500. -/
theorem initial_sum_calculation (total_amount : ℝ) (time : ℝ) (rate : ℝ) (principal : ℝ)
  (h1 : total_amount = 15500)
  (h2 : time = 4)
  (h3 : rate = 6)
  (h4 : total_amount = principal + (principal * rate * time / 100)) :
  principal = 12500 := by
sorry

end initial_sum_calculation_l2085_208509


namespace alex_age_l2085_208570

theorem alex_age (charlie_age alex_age : ℕ) : 
  charlie_age = 2 * alex_age + 8 → 
  charlie_age = 22 → 
  alex_age = 7 := by
sorry

end alex_age_l2085_208570


namespace simplify_polynomial_l2085_208573

theorem simplify_polynomial (y : ℝ) : y * (4 * y^2 + 3) - 6 * (y^2 + 3 * y - 8) = 4 * y^3 - 6 * y^2 - 15 * y + 48 := by
  sorry

end simplify_polynomial_l2085_208573


namespace f_g_minus_g_f_l2085_208520

def f (x : ℝ) : ℝ := 4 * x + 8

def g (x : ℝ) : ℝ := 2 * x - 3

theorem f_g_minus_g_f : ∀ x : ℝ, f (g x) - g (f x) = -17 := by
  sorry

end f_g_minus_g_f_l2085_208520
