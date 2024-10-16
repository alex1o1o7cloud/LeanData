import Mathlib

namespace NUMINAMATH_CALUDE_h_eq_f_reflected_and_shifted_l1394_139483

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the function h obtained from f by reflection and shift
def h (x : ℝ) : ℝ := f (6 - x)

-- Theorem stating the relationship between h and f
theorem h_eq_f_reflected_and_shifted :
  ∀ x : ℝ, h f x = f (6 - x) := by sorry

end NUMINAMATH_CALUDE_h_eq_f_reflected_and_shifted_l1394_139483


namespace NUMINAMATH_CALUDE_yearly_increase_fraction_l1394_139444

/-- 
Given an initial amount that increases by a fraction each year, 
this theorem proves that the fraction is 0.125 when the initial amount 
is 3200 and becomes 4050 after two years.
-/
theorem yearly_increase_fraction 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (f : ℝ) 
  (h1 : initial_amount = 3200) 
  (h2 : final_amount = 4050) 
  (h3 : final_amount = initial_amount * (1 + f)^2) : 
  f = 0.125 := by
sorry

end NUMINAMATH_CALUDE_yearly_increase_fraction_l1394_139444


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1394_139481

/-- Represents an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  (equation : ∀ x y : ℝ, x^2 + k*y^2 = 2)
  (is_ellipse : k ≠ 0)
  (foci_on_y : k < 1)

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1394_139481


namespace NUMINAMATH_CALUDE_exists_different_reassembled_triangle_l1394_139420

/-- A triangle represented by its three vertices in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A function that cuts a triangle into two parts -/
def cut (t : Triangle) : (Triangle × Triangle) :=
  sorry

/-- A function that reassembles two triangles into one -/
def reassemble (t1 t2 : Triangle) : Triangle :=
  sorry

/-- Theorem stating that there exists a triangle that can be cut and reassembled into a different triangle -/
theorem exists_different_reassembled_triangle :
  ∃ (t : Triangle), ∃ (t1 t2 : Triangle),
    (cut t = (t1, t2)) ∧ (reassemble t1 t2 ≠ t) := by
  sorry

end NUMINAMATH_CALUDE_exists_different_reassembled_triangle_l1394_139420


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l1394_139446

/-- The number of baseball cards Carter has -/
def carterCards : ℕ := 152

/-- The number of additional cards Marcus has compared to Carter -/
def marcusExtraCards : ℕ := 58

/-- The number of baseball cards Marcus has -/
def marcusCards : ℕ := carterCards + marcusExtraCards

theorem marcus_baseball_cards : marcusCards = 210 := by
  sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l1394_139446


namespace NUMINAMATH_CALUDE_relationship_abc_l1394_139485

/-- Given the definitions of a, b, and c, prove that a < c < b -/
theorem relationship_abc : 
  let a := (1/2) * Real.cos (80 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (80 * π / 180)
  let b := (2 * Real.tan (13 * π / 180)) / (1 - Real.tan (13 * π / 180)^2)
  let c := Real.sqrt ((1 - Real.cos (52 * π / 180)) / 2)
  a < c ∧ c < b := by
sorry


end NUMINAMATH_CALUDE_relationship_abc_l1394_139485


namespace NUMINAMATH_CALUDE_unique_box_dimensions_l1394_139458

theorem unique_box_dimensions : ∃! (a b c : ℕ+), 
  (a ≥ b) ∧ (b ≥ c) ∧ 
  (a.val * b.val * c.val = 2 * (a.val * b.val + a.val * c.val + b.val * c.val)) := by
  sorry

end NUMINAMATH_CALUDE_unique_box_dimensions_l1394_139458


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l1394_139488

theorem negation_of_forall_positive_square_plus_one :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l1394_139488


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l1394_139475

theorem sum_of_special_primes_is_prime (P Q : ℕ+) (h1 : Nat.Prime P)
  (h2 : Nat.Prime Q) (h3 : Nat.Prime (P - Q)) (h4 : Nat.Prime (P + Q)) :
  Nat.Prime (P + Q + (P - Q) + Q) :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l1394_139475


namespace NUMINAMATH_CALUDE_circle_and_quadratic_inequality_l1394_139407

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - 2*a*x + y^2 + 2*a^2 - 5*a + 4 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a-1)*x + 1 > 0

-- Theorem statement
theorem circle_and_quadratic_inequality (a : ℝ) :
  p a ∧ q a → 1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_and_quadratic_inequality_l1394_139407


namespace NUMINAMATH_CALUDE_cone_angle_bisecting_volume_l1394_139431

/-- 
Given a cone with the following properties:
- A perpendicular is dropped from the center of the base to the slant height
- This perpendicular rotates about the cone's axis
- The surface of rotation divides the cone's volume in half

The angle between the slant height and the axis is arccos(1 / 2^(1/4))
-/
theorem cone_angle_bisecting_volume (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  let α := Real.arccos ((1 : ℝ) / 2^(1/4))
  let V := (1/3) * π * R^2 * h
  let V_rotated := (1/3) * π * (R * (Real.cos α)^2)^2 * h
  V_rotated = (1/2) * V :=
by sorry

end NUMINAMATH_CALUDE_cone_angle_bisecting_volume_l1394_139431


namespace NUMINAMATH_CALUDE_blue_face_probability_is_five_eighths_l1394_139464

/-- An octahedron with blue and red faces -/
structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (total_is_sum : total_faces = blue_faces + red_faces)
  (total_is_eight : total_faces = 8)

/-- The probability of rolling a blue face on an octahedron -/
def blue_face_probability (o : Octahedron) : ℚ :=
  o.blue_faces / o.total_faces

/-- Theorem: The probability of rolling a blue face on an octahedron with 5 blue faces out of 8 total faces is 5/8 -/
theorem blue_face_probability_is_five_eighths (o : Octahedron) 
  (h : o.blue_faces = 5) : blue_face_probability o = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_is_five_eighths_l1394_139464


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1394_139476

theorem complex_equation_solution (z a b : ℂ) : 
  z = (1 - Complex.I)^2 + 1 + 3 * Complex.I →
  z^2 + a * z + b = 1 - Complex.I →
  a.im = 0 →
  b.im = 0 →
  a = -2 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1394_139476


namespace NUMINAMATH_CALUDE_correct_num_footballs_l1394_139427

/-- The number of footballs bought by the school gym -/
def num_footballs : ℕ := 22

/-- The number of basketballs bought by the school gym -/
def num_basketballs : ℕ := 6

/-- Theorem stating that the number of footballs is correct given the conditions -/
theorem correct_num_footballs : 
  (num_footballs = 3 * num_basketballs + 4) ∧ 
  (num_footballs = 4 * num_basketballs - 2) := by
  sorry

#check correct_num_footballs

end NUMINAMATH_CALUDE_correct_num_footballs_l1394_139427


namespace NUMINAMATH_CALUDE_inequality_count_l1394_139416

theorem inequality_count (x y a b : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0)
  (h_x_lt_1 : x < 1)
  (h_y_lt_1 : y < 1)
  (h_x_lt_a : x < a)
  (h_y_lt_b : y < b)
  (h_sum : x + y = a - b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ¬(∀ (x y a b : ℝ), x / y < a / b) := by
sorry

end NUMINAMATH_CALUDE_inequality_count_l1394_139416


namespace NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l1394_139421

theorem max_sum_abs_on_unit_sphere :
  ∃ (M : ℝ), M = 2 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l1394_139421


namespace NUMINAMATH_CALUDE_x_squared_divisible_by_y_l1394_139496

theorem x_squared_divisible_by_y (x y : ℕ+) 
  (h : ∃ (n : ℕ), (x.val^2 : ℚ) / y.val + (y.val^2 : ℚ) / x.val = n) : 
  y.val ∣ x.val^2 := by
sorry

end NUMINAMATH_CALUDE_x_squared_divisible_by_y_l1394_139496


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l1394_139484

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents 11101₂ as a list of bits. -/
def binary_11101 : List Bool := [true, true, true, false, true]

/-- Represents 10011₂ as a list of bits. -/
def binary_10011 : List Bool := [true, false, false, true, true]

/-- Represents 101₂ as a list of bits. -/
def binary_101 : List Bool := [true, false, true]

/-- Represents 11101100₂ as a list of bits. -/
def binary_11101100 : List Bool := [true, true, true, false, true, true, false, false]

/-- The main theorem to prove. -/
theorem binary_multiplication_division_equality :
  (binary_to_nat binary_11101 * binary_to_nat binary_10011) / binary_to_nat binary_101 =
  binary_to_nat binary_11101100 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l1394_139484


namespace NUMINAMATH_CALUDE_equipment_productivity_increase_l1394_139443

/-- Represents the productivity increase factor of the equipment -/
def productivity_increase : ℝ := 4

/-- Represents the time taken by the first worker to complete the job -/
def first_worker_time : ℝ := 8

/-- Represents the time taken by the second worker to complete the job -/
def second_worker_time : ℝ := 5

/-- Represents the setup time for the second worker -/
def setup_time : ℝ := 2

/-- Represents the time after which the second worker processes as many parts as the first worker -/
def equal_parts_time : ℝ := 1

theorem equipment_productivity_increase :
  (∃ (r : ℝ),
    r > 0 ∧
    r * first_worker_time = productivity_increase * r * (second_worker_time - setup_time) ∧
    r * (setup_time + equal_parts_time) = productivity_increase * r * equal_parts_time) :=
by
  sorry

#check equipment_productivity_increase

end NUMINAMATH_CALUDE_equipment_productivity_increase_l1394_139443


namespace NUMINAMATH_CALUDE_inequality_proof_l1394_139469

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1394_139469


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1394_139455

theorem quadratic_equation_result (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) : 
  (14 * y - 5)^2 = 333 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1394_139455


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1394_139492

-- Define sets A and B
def A : Set ℝ := {x | 2 * x - 1 > 0}
def B : Set ℝ := {x | |x| < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1394_139492


namespace NUMINAMATH_CALUDE_shaded_area_problem_l1394_139465

/-- The area of the shaded region in a square with side length 40 units, 
    where two congruent triangles with base 20 units and height 20 units 
    are removed, is equal to 1200 square units. -/
theorem shaded_area_problem (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 40 →
  triangle_base = 20 →
  triangle_height = 20 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l1394_139465


namespace NUMINAMATH_CALUDE_watch_cost_in_dollars_l1394_139498

/-- The cost of a watch in dollars when paid with dimes -/
def watch_cost (num_dimes : ℕ) (dime_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value

/-- Theorem: If Greyson paid for a watch with 50 dimes, and each dime is worth $0.10, then the cost of the watch is $5.00 -/
theorem watch_cost_in_dollars :
  watch_cost 50 (1/10) = 5 :=
sorry

end NUMINAMATH_CALUDE_watch_cost_in_dollars_l1394_139498


namespace NUMINAMATH_CALUDE_same_color_probability_l1394_139494

/-- The probability of drawing two balls of the same color from a bag containing 3 white balls
    and 2 black balls when 2 balls are randomly drawn at the same time. -/
theorem same_color_probability (total : ℕ) (white : ℕ) (black : ℕ) :
  total = 5 →
  white = 3 →
  black = 2 →
  (Nat.choose white 2 + Nat.choose black 2) / Nat.choose total 2 = 2 / 5 := by
  sorry

#eval Nat.choose 5 2  -- Total number of ways to draw 2 balls from 5
#eval Nat.choose 3 2  -- Number of ways to draw 2 white balls
#eval Nat.choose 2 2  -- Number of ways to draw 2 black balls

end NUMINAMATH_CALUDE_same_color_probability_l1394_139494


namespace NUMINAMATH_CALUDE_product_b3_b17_l1394_139418

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem product_b3_b17 (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
    (h_a8_b10 : a 8 = b 10) :
  b 3 * b 17 = 36 := by
sorry

end NUMINAMATH_CALUDE_product_b3_b17_l1394_139418


namespace NUMINAMATH_CALUDE_population_growth_prediction_l1394_139462

/-- Theorem: Population Growth and Prediction --/
theorem population_growth_prediction
  (initial_population : ℝ)
  (current_population : ℝ)
  (future_population : ℝ)
  (h1 : current_population = 3 * initial_population)
  (h2 : future_population = 1.4 * current_population)
  (h3 : future_population = 16800)
  : initial_population = 4000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_prediction_l1394_139462


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l1394_139403

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, x ≠ 2 → |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l1394_139403


namespace NUMINAMATH_CALUDE_expand_expression_l1394_139419

theorem expand_expression (x : ℝ) : (9*x + 4) * (2*x^2) = 18*x^3 + 8*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1394_139419


namespace NUMINAMATH_CALUDE_floor_times_self_110_l1394_139439

theorem floor_times_self_110 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 110 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_110_l1394_139439


namespace NUMINAMATH_CALUDE_candy_mixture_theorem_l1394_139495

def candy_mixture (initial_blue initial_red added_blue added_red final_blue : ℚ) : Prop :=
  ∃ (x y : ℚ),
    x > 0 ∧ y > 0 ∧
    initial_blue = 1/10 ∧
    initial_red = 1/4 ∧
    added_blue = 1/4 ∧
    added_red = 3/4 ∧
    (initial_blue * x + added_blue * y) / (x + y) = final_blue

theorem candy_mixture_theorem :
  ∀ initial_blue initial_red added_blue added_red final_blue,
    candy_mixture initial_blue initial_red added_blue added_red final_blue →
    final_blue = 4/25 →
    ∃ final_red : ℚ, final_red = 9/20 :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_theorem_l1394_139495


namespace NUMINAMATH_CALUDE_all_expressions_are_identities_l1394_139429

theorem all_expressions_are_identities (x y : ℝ) : 
  ((2*x - 1) * (x - 3) = 2*x^2 - 7*x + 3) ∧
  ((2*x + 1) * (x + 3) = 2*x^2 + 7*x + 3) ∧
  ((2 - x) * (1 - 3*x) = 2 - 7*x + 3*x^2) ∧
  ((2 + x) * (1 + 3*x) = 2 + 7*x + 3*x^2) ∧
  ((2*x - y) * (x - 3*y) = 2*x^2 - 7*x*y + 3*y^2) ∧
  ((2*x + y) * (x + 3*y) = 2*x^2 + 7*x*y + 3*y^2) :=
by
  sorry

#check all_expressions_are_identities

end NUMINAMATH_CALUDE_all_expressions_are_identities_l1394_139429


namespace NUMINAMATH_CALUDE_exam_average_proof_l1394_139442

theorem exam_average_proof :
  let group1_count : ℕ := 15
  let group1_average : ℚ := 75/100
  let group2_count : ℕ := 10
  let group2_average : ℚ := 95/100
  let total_count : ℕ := group1_count + group2_count
  
  (group1_count * group1_average + group2_count * group2_average) / total_count = 83/100 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_proof_l1394_139442


namespace NUMINAMATH_CALUDE_incorrect_classification_l1394_139414

/-- Represents a proof method -/
inductive ProofMethod
| Synthetic
| Analytic

/-- Represents the nature of a proof -/
inductive ProofNature
| Direct
| Indirect

/-- Defines the correct classification of proof methods -/
def correct_classification (method : ProofMethod) : ProofNature :=
  match method with
  | ProofMethod.Synthetic => ProofNature.Direct
  | ProofMethod.Analytic => ProofNature.Direct

/-- Theorem stating that the given classification is incorrect -/
theorem incorrect_classification :
  ¬(correct_classification ProofMethod.Synthetic = ProofNature.Direct ∧
    correct_classification ProofMethod.Analytic = ProofNature.Indirect) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_classification_l1394_139414


namespace NUMINAMATH_CALUDE_age_range_count_l1394_139425

/-- Calculates the number of integer ages within one standard deviation of the average age -/
def count_ages_within_std_dev (average_age : ℕ) (std_dev : ℕ) : ℕ :=
  (average_age + std_dev) - (average_age - std_dev) + 1

/-- Proves that given an average age of 31 and a standard deviation of 9, 
    the number of integer ages within one standard deviation of the average is 19 -/
theorem age_range_count : count_ages_within_std_dev 31 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_age_range_count_l1394_139425


namespace NUMINAMATH_CALUDE_remainder_problem_l1394_139417

theorem remainder_problem (x : ℤ) : 
  (∃ k : ℤ, x = 142 * k + 110) → 
  (∃ m : ℤ, x = 14 * m + 12) :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1394_139417


namespace NUMINAMATH_CALUDE_range_of_a_l1394_139411

theorem range_of_a (a : ℝ) : 2 * a ≠ a^2 ↔ a ≠ 0 ∧ a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1394_139411


namespace NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l1394_139435

/-- Represents the race between Karen and Tom -/
structure Race where
  karen_initial_speed : ℝ
  tom_initial_speed : ℝ
  karen_final_speed : ℝ
  tom_final_speed : ℝ
  karen_delay : ℝ
  winning_margin : ℝ

/-- Calculates the distance Tom drives before Karen wins the bet -/
def distance_tom_drives (race : Race) : ℝ :=
  sorry

/-- Theorem stating that Tom drives 21 miles before Karen wins the bet -/
theorem tom_distance_before_karen_wins (race : Race) 
  (h1 : race.karen_initial_speed = 60)
  (h2 : race.tom_initial_speed = 45)
  (h3 : race.karen_final_speed = 70)
  (h4 : race.tom_final_speed = 40)
  (h5 : race.karen_delay = 4/60)  -- 4 minutes converted to hours
  (h6 : race.winning_margin = 4) :
  distance_tom_drives race = 21 :=
sorry

end NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l1394_139435


namespace NUMINAMATH_CALUDE_keith_pears_l1394_139440

/-- Given that Jason picked 46 pears, Mike picked 12 pears, and the total number of pears picked was 105, prove that Keith picked 47 pears. -/
theorem keith_pears (jason_pears mike_pears total_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_pears = 12)
  (h3 : total_pears = 105) :
  total_pears - (jason_pears + mike_pears) = 47 :=
by sorry

end NUMINAMATH_CALUDE_keith_pears_l1394_139440


namespace NUMINAMATH_CALUDE_factorization_problems_l1394_139463

theorem factorization_problems :
  (∀ x : ℝ, x^2 - 16 = (x + 4) * (x - 4)) ∧
  (∀ a b : ℝ, a^3*b - 2*a^2*b + a*b = a*b*(a - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1394_139463


namespace NUMINAMATH_CALUDE_only_exponential_has_multiplicative_property_l1394_139471

-- Define the property that we're looking for
def HasMultiplicativeProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x + y) = f x * f y

-- Define the types of functions we're considering
class FunctionType (f : ℝ → ℝ) where
  isPower : Prop
  isLogarithmic : Prop
  isExponential : Prop
  isLinear : Prop

-- Theorem stating that only exponential functions have the multiplicative property
theorem only_exponential_has_multiplicative_property (f : ℝ → ℝ) [FunctionType f] :
  HasMultiplicativeProperty f ↔ FunctionType.isExponential f := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_only_exponential_has_multiplicative_property_l1394_139471


namespace NUMINAMATH_CALUDE_probability_exact_hits_l1394_139426

def probability_single_hit : ℝ := 0.7
def total_shots : ℕ := 5
def desired_hits : ℕ := 2

theorem probability_exact_hits :
  let p := probability_single_hit
  let n := total_shots
  let k := desired_hits
  let q := 1 - p
  (Nat.choose n k : ℝ) * p ^ k * q ^ (n - k) = 0.1323 := by sorry

end NUMINAMATH_CALUDE_probability_exact_hits_l1394_139426


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_intersection_equality_range_l1394_139433

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x - 2) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

-- Theorem for part 1
theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B 2) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Theorem for part 2
theorem intersection_equality_range :
  {a : ℝ | A ∩ B a = B a} = {a : ℝ | 3 ≤ a ∧ a ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_intersection_equality_range_l1394_139433


namespace NUMINAMATH_CALUDE_min_value_of_x_l1394_139474

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 3 + (1/3) * Real.log x) : x ≥ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1394_139474


namespace NUMINAMATH_CALUDE_range_of_a_l1394_139410

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1394_139410


namespace NUMINAMATH_CALUDE_problem_solution_l1394_139456

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1394_139456


namespace NUMINAMATH_CALUDE_product_of_roots_l1394_139470

theorem product_of_roots (x : ℝ) : 
  (x^3 - 9*x^2 + 27*x - 64 = 0) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 64 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 64) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1394_139470


namespace NUMINAMATH_CALUDE_triangle_centroid_theorem_l1394_139437

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Point O inside triangle ABC -/
structure PointInTriangle (t : Triangle) where
  O : Vector2D
  A : Vector2D
  B : Vector2D
  C : Vector2D

theorem triangle_centroid_theorem (t : Triangle) (p : PointInTriangle t) 
  (h1 : t.b = 6)
  (h2 : t.a * t.c * Real.cos t.B = t.a^2 - t.b^2 + (Real.sqrt 7 / 4) * t.b * t.c)
  (h3 : p.O.x + p.A.x + p.B.x + p.C.x = 0 ∧ p.O.y + p.A.y + p.B.y + p.C.y = 0)
  (h4 : Real.cos (t.A - π/6) = Real.cos t.A * Real.cos (π/6) + Real.sin t.A * Real.sin (π/6)) :
  (p.O.x - p.A.x)^2 + (p.O.y - p.A.y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_theorem_l1394_139437


namespace NUMINAMATH_CALUDE_percentage_decrease_proof_l1394_139447

def original_price : ℝ := 250
def new_price : ℝ := 200

theorem percentage_decrease_proof :
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_proof_l1394_139447


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l1394_139461

theorem chocolate_bars_per_box (total_bars : ℕ) (num_boxes : ℕ) 
  (h1 : total_bars = 442) (h2 : num_boxes = 17) :
  total_bars / num_boxes = 26 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l1394_139461


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1394_139479

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1394_139479


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1394_139424

theorem simplify_and_evaluate (x : ℝ) (h : x = 5) :
  x^2 * (x + 1) - x * (x^2 - x + 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1394_139424


namespace NUMINAMATH_CALUDE_points_on_line_value_at_2_l1394_139430

/-- A linear function passing through given points -/
def linear_function (x : ℝ) : ℝ := x - 1

/-- The given points satisfy the linear function -/
theorem points_on_line : 
  linear_function (-1) = -2 ∧ 
  linear_function 0 = -1 ∧ 
  linear_function 1 = 0 := by sorry

/-- The y-value corresponding to x = 2 is 1 -/
theorem value_at_2 : linear_function 2 = 1 := by sorry

end NUMINAMATH_CALUDE_points_on_line_value_at_2_l1394_139430


namespace NUMINAMATH_CALUDE_min_sum_abcd_l1394_139473

theorem min_sum_abcd (a b c d : ℕ) 
  (h1 : a + b = 2)
  (h2 : a + c = 3)
  (h3 : a + d = 4)
  (h4 : b + c = 5)
  (h5 : b + d = 6)
  (h6 : c + d = 7) :
  a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abcd_l1394_139473


namespace NUMINAMATH_CALUDE_bear_population_difference_l1394_139453

theorem bear_population_difference :
  ∀ (white black brown : ℕ),
    black = 2 * white →
    black = 60 →
    white + black + brown = 190 →
    brown - black = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_bear_population_difference_l1394_139453


namespace NUMINAMATH_CALUDE_exists_point_P_trajectory_G_l1394_139432

/-- Definition of the ellipse C -/
def is_on_ellipse_C (x y : ℝ) : Prop := x^2/36 + y^2/20 = 1

/-- Definition of point A -/
def point_A : ℝ × ℝ := (-6, 0)

/-- Definition of point F -/
def point_F : ℝ × ℝ := (4, 0)

/-- Definition of vector AP -/
def vector_AP (x y : ℝ) : ℝ × ℝ := (x + 6, y)

/-- Definition of vector FP -/
def vector_FP (x y : ℝ) : ℝ × ℝ := (x - 4, y)

/-- Theorem stating the existence of point P -/
theorem exists_point_P :
  ∃ (x y : ℝ), 
    is_on_ellipse_C x y ∧ 
    y > 0 ∧ 
    (vector_AP x y).1 * (vector_FP x y).1 + (vector_AP x y).2 * (vector_FP x y).2 = 0 ∧
    x = 3/2 ∧ 
    y = 5 * Real.sqrt 3 / 2 :=
sorry

/-- Definition of point M on ellipse C -/
def point_M (x₀ y₀ : ℝ) : Prop := is_on_ellipse_C x₀ y₀

/-- Definition of midpoint G of MF -/
def point_G (x y : ℝ) (x₀ y₀ : ℝ) : Prop :=
  x = (x₀ + 2) / 2 ∧ y = y₀ / 2

/-- Theorem stating the trajectory equation of G -/
theorem trajectory_G :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), point_M x₀ y₀ ∧ point_G x y x₀ y₀) ↔ 
    (x - 1)^2 / 9 + y^2 / 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_exists_point_P_trajectory_G_l1394_139432


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1394_139445

theorem unique_integer_solution : 
  ∀ m n : ℕ+, 
    (m : ℚ) + n - (3 * m * n) / (m + n) = 2011 / 3 ↔ 
    ((m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144)) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1394_139445


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l1394_139468

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 3*43*3*y + 43^2)) ∧
  (53 ∣ (3*x)^2 + 3*43*3*x + 43^2) ∧
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l1394_139468


namespace NUMINAMATH_CALUDE_round_trip_percentage_l1394_139490

theorem round_trip_percentage (total_passengers : ℝ) 
  (h1 : total_passengers > 0) :
  let round_trip_with_car := 0.15 * total_passengers
  let round_trip_without_car := 0.6 * (round_trip_with_car / 0.4)
  (round_trip_with_car + round_trip_without_car) / total_passengers = 0.375 := by
sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l1394_139490


namespace NUMINAMATH_CALUDE_cost_of_six_pens_l1394_139400

/-- Given that 3 pens cost 7.5 yuan, prove that 6 pens cost 15 yuan. -/
theorem cost_of_six_pens (cost_three_pens : ℝ) (h : cost_three_pens = 7.5) :
  let cost_one_pen := cost_three_pens / 3
  cost_one_pen * 6 = 15 := by sorry

end NUMINAMATH_CALUDE_cost_of_six_pens_l1394_139400


namespace NUMINAMATH_CALUDE_parabola_transformation_l1394_139401

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h + p.b, c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

theorem parabola_transformation (x : ℝ) :
  let p₀ : Parabola := { a := 1, b := 0, c := 0 }  -- y = x²
  let p₁ := shift_horizontal p₀ 3                  -- shift 3 units right
  let p₂ := shift_vertical p₁ 4                    -- shift 4 units up
  p₂.a * x^2 + p₂.b * x + p₂.c = (x - 3)^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1394_139401


namespace NUMINAMATH_CALUDE_monochromatic_sequence_exists_l1394_139467

def S (n : ℕ) : ℕ := (n * (n^2 + 5)) / 6

def is_valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i < n - 1, a i < a (i + 1)) ∧
  (∀ i < n - 2, a (i + 1) - a i ≤ a (i + 2) - a (i + 1))

theorem monochromatic_sequence_exists (n : ℕ) (h : n ≥ 2) :
  ∀ c : ℕ → Bool,
  ∃ a : ℕ → ℕ, ∃ color : Bool,
    (∀ i < n, a i ≤ S n) ∧
    (∀ i < n, c (a i) = color) ∧
    is_valid_sequence a n :=
sorry

end NUMINAMATH_CALUDE_monochromatic_sequence_exists_l1394_139467


namespace NUMINAMATH_CALUDE_rectangle_width_l1394_139486

/-- Given a rectangle with length 20 and perimeter 70, prove its width is 15 -/
theorem rectangle_width (length perimeter : ℝ) (h1 : length = 20) (h2 : perimeter = 70) :
  let width := (perimeter - 2 * length) / 2
  width = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1394_139486


namespace NUMINAMATH_CALUDE_coin_toss_is_classical_model_l1394_139434

structure Experiment where
  name : String
  is_finite : Bool
  is_equiprobable : Bool

def is_classical_probability_model (e : Experiment) : Prop :=
  e.is_finite ∧ e.is_equiprobable

def seed_germination : Experiment :=
  { name := "Seed germination",
    is_finite := true,
    is_equiprobable := false }

def product_measurement : Experiment :=
  { name := "Product measurement",
    is_finite := false,
    is_equiprobable := false }

def coin_toss : Experiment :=
  { name := "Coin toss",
    is_finite := true,
    is_equiprobable := true }

def target_shooting : Experiment :=
  { name := "Target shooting",
    is_finite := true,
    is_equiprobable := false }

theorem coin_toss_is_classical_model :
  is_classical_probability_model coin_toss ∧
  ¬is_classical_probability_model seed_germination ∧
  ¬is_classical_probability_model product_measurement ∧
  ¬is_classical_probability_model target_shooting :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_is_classical_model_l1394_139434


namespace NUMINAMATH_CALUDE_ball_ratio_l1394_139450

theorem ball_ratio (total : ℕ) (blue red : ℕ) (green : ℕ := 3 * blue) 
  (yellow : ℕ := total - (blue + red + green)) 
  (h1 : total = 36) (h2 : blue = 6) (h3 : red = 4) :
  yellow / red = 2 :=
by sorry

end NUMINAMATH_CALUDE_ball_ratio_l1394_139450


namespace NUMINAMATH_CALUDE_new_trailers_correct_l1394_139438

/-- Represents the trailer park scenario -/
structure TrailerPark where
  initial_count : ℕ
  initial_avg_age : ℕ
  years_passed : ℕ
  current_avg_age : ℕ

/-- Calculates the number of new trailers added -/
def new_trailers (park : TrailerPark) : ℕ :=
  13

/-- Theorem stating that the calculated number of new trailers is correct -/
theorem new_trailers_correct (park : TrailerPark) 
  (h1 : park.initial_count = 30)
  (h2 : park.initial_avg_age = 10)
  (h3 : park.years_passed = 5)
  (h4 : park.current_avg_age = 12) :
  new_trailers park = 13 := by
  sorry

#check new_trailers_correct

end NUMINAMATH_CALUDE_new_trailers_correct_l1394_139438


namespace NUMINAMATH_CALUDE_triangle_shape_l1394_139413

theorem triangle_shape (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A)) : 
  (A = B ∨ A = π / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1394_139413


namespace NUMINAMATH_CALUDE_rectangle_enclosure_l1394_139482

def rectangle_largest_side (length width : ℝ) : Prop :=
  let perimeter := 2 * (length + width)
  let area := length * width
  perimeter = 240 ∧ 
  area = 12 * perimeter ∧ 
  length ≥ width ∧
  length = 72

theorem rectangle_enclosure :
  ∃ (length width : ℝ), rectangle_largest_side length width :=
sorry

end NUMINAMATH_CALUDE_rectangle_enclosure_l1394_139482


namespace NUMINAMATH_CALUDE_divisible_by_4_6_9_less_than_300_l1394_139454

theorem divisible_by_4_6_9_less_than_300 : 
  (Finset.filter (fun n : ℕ => n % 4 = 0 ∧ n % 6 = 0 ∧ n % 9 = 0) (Finset.range 300)).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_4_6_9_less_than_300_l1394_139454


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l1394_139406

theorem geometric_arithmetic_sequence_problem 
  (b q a d : ℝ) 
  (h1 : b = a + d)
  (h2 : b * q = a + 3 * d)
  (h3 : b * q^2 = a + 6 * d)
  (h4 : b * (b * q) * (b * q^2) = 64) :
  b = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l1394_139406


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1394_139405

theorem trigonometric_identities (x : ℝ) : 
  ((Real.sqrt 3) / 2 * Real.cos x - (1 / 2) * Real.sin x = Real.cos (x + π / 6)) ∧ 
  (Real.sin x + Real.cos x = Real.sqrt 2 * Real.sin (x + π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1394_139405


namespace NUMINAMATH_CALUDE_child_ticket_price_l1394_139493

theorem child_ticket_price
  (adult_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (child_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : total_tickets = 900)
  (h3 : total_revenue = 5100)
  (h4 : child_tickets = 400)
  : ∃ (child_price : ℕ),
    child_price * child_tickets + adult_price * (total_tickets - child_tickets) = total_revenue ∧
    child_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_price_l1394_139493


namespace NUMINAMATH_CALUDE_class_average_weight_l1394_139499

theorem class_average_weight (n1 : ℕ) (n2 : ℕ) (w1 : ℝ) (w2 : ℝ) (h1 : n1 = 22) (h2 : n2 = 8) (h3 : w1 = 50.25) (h4 : w2 = 45.15) :
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1394_139499


namespace NUMINAMATH_CALUDE_divides_cubic_minus_one_l1394_139480

theorem divides_cubic_minus_one (a : ℤ) : 
  35 ∣ (a^3 - 1) ↔ a % 35 = 1 ∨ a % 35 = 11 ∨ a % 35 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divides_cubic_minus_one_l1394_139480


namespace NUMINAMATH_CALUDE_problem_solution_l1394_139466

theorem problem_solution (x y : ℝ) : (x - 1)^2 + Real.sqrt (y + 2) = 0 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1394_139466


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1394_139460

theorem quadratic_root_difference (m : ℝ) : 
  ∃ (x₁ x₂ : ℂ), x₁^2 + m*x₁ + 3 = 0 ∧ 
                 x₂^2 + m*x₂ + 3 = 0 ∧ 
                 x₁ ≠ x₂ ∧
                 Complex.abs (x₁ - x₂) = 2 → 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1394_139460


namespace NUMINAMATH_CALUDE_largest_of_consecutive_odd_divisible_by_3_l1394_139404

/-- Three consecutive odd natural numbers divisible by 3 whose sum is 72 -/
def ConsecutiveOddDivisibleBy3 (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∧
  (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0) ∧
  (b = a + 6 ∧ c = a + 12) ∧
  (a + b + c = 72)

theorem largest_of_consecutive_odd_divisible_by_3 {a b c : ℕ} 
  (h : ConsecutiveOddDivisibleBy3 a b c) : 
  max a (max b c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_consecutive_odd_divisible_by_3_l1394_139404


namespace NUMINAMATH_CALUDE_total_legs_on_farm_l1394_139436

/-- The number of legs for each animal type -/
def duck_legs : ℕ := 2
def dog_legs : ℕ := 4

/-- The farm composition -/
def total_animals : ℕ := 11
def num_ducks : ℕ := 6
def num_dogs : ℕ := total_animals - num_ducks

/-- The theorem to prove -/
theorem total_legs_on_farm : 
  num_ducks * duck_legs + num_dogs * dog_legs = 32 := by sorry

end NUMINAMATH_CALUDE_total_legs_on_farm_l1394_139436


namespace NUMINAMATH_CALUDE_problem_solution_l1394_139402

theorem problem_solution (m : ℕ) (q : ℚ) :
  m = 31 →
  ((1^m) / (5^m)) * ((1^16) / (4^16)) = 1 / (q * (10^31)) →
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1394_139402


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1394_139489

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1394_139489


namespace NUMINAMATH_CALUDE_path_length_squares_l1394_139422

/-- Given a line PQ of length 24 cm divided into six equal parts, with squares drawn on each part,
    the path following three sides of each square from P to Q is 72 cm long. -/
theorem path_length_squares (PQ : ℝ) (num_parts : ℕ) : 
  PQ = 24 →
  num_parts = 6 →
  (num_parts : ℝ) * (3 * (PQ / num_parts)) = 72 :=
by sorry

end NUMINAMATH_CALUDE_path_length_squares_l1394_139422


namespace NUMINAMATH_CALUDE_original_number_proof_l1394_139448

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1394_139448


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1394_139412

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 2 * F + 15 → 
  D + E + F = 180 → 
  F = 30 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1394_139412


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l1394_139428

theorem no_real_sqrt_negative_quadratic :
  ¬ ∃ x : ℝ, ∃ y : ℝ, y^2 = -(x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l1394_139428


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1394_139408

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1394_139408


namespace NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_directrix_and_yaxis_l1394_139451

/-- A circle centered on a parabola and tangent to its directrix and the y-axis -/
theorem circle_on_parabola_tangent_to_directrix_and_yaxis :
  ∀ (x₀ : ℝ) (y₀ : ℝ) (r : ℝ),
  x₀ = 1 ∨ x₀ = -1 →
  y₀ = (1/2) * x₀^2 →
  r = 1 →
  (∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2 →
    ∃ (t : ℝ), x = t ∧ y = (1/2) * t^2) ∧
  (∃ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2 ∧ y = -(1/2)) ∧
  (∃ (y : ℝ), x₀^2 + (y - y₀)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_directrix_and_yaxis_l1394_139451


namespace NUMINAMATH_CALUDE_intersection_point_solution_and_b_value_l1394_139441

/-- The intersection point of two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- A linear function of the form y = mx + c -/
structure LinearFunction where
  m : ℝ
  c : ℝ

/-- Given two linear functions and their intersection point, prove the solution and b value -/
theorem intersection_point_solution_and_b_value 
  (f1 : LinearFunction)
  (f2 : LinearFunction)
  (P : IntersectionPoint)
  (h1 : f1.m = 2 ∧ f1.c = -5)
  (h2 : f2.m = 3)
  (h3 : P.x = 1 ∧ P.y = -3)
  (h4 : P.y = f1.m * P.x + f1.c)
  (h5 : P.y = f2.m * P.x + f2.c) :
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ 
    y = f1.m * x + f1.c ∧
    y = f2.m * x + f2.c) ∧
  f2.c = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_solution_and_b_value_l1394_139441


namespace NUMINAMATH_CALUDE_seven_point_circle_triangle_count_l1394_139472

/-- A circle with points and chords -/
structure CircleWithChords where
  numPoints : ℕ
  noTripleIntersection : Bool

/-- Count of triangles formed by chord intersections -/
def triangleCount (c : CircleWithChords) : ℕ := sorry

/-- Theorem: For 7 points on a circle with no triple intersections, 
    the number of triangles formed by chord intersections is 7 -/
theorem seven_point_circle_triangle_count 
  (c : CircleWithChords) 
  (h1 : c.numPoints = 7) 
  (h2 : c.noTripleIntersection = true) : 
  triangleCount c = 7 := by sorry

end NUMINAMATH_CALUDE_seven_point_circle_triangle_count_l1394_139472


namespace NUMINAMATH_CALUDE_hot_dog_remainder_l1394_139409

theorem hot_dog_remainder : 25197641 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_remainder_l1394_139409


namespace NUMINAMATH_CALUDE_tshirt_socks_price_difference_l1394_139449

/-- The price difference between a t-shirt and socks -/
theorem tshirt_socks_price_difference 
  (jeans_price t_shirt_price socks_price : ℝ) 
  (h1 : jeans_price = 2 * t_shirt_price) 
  (h2 : jeans_price = 30) 
  (h3 : socks_price = 5) : 
  t_shirt_price - socks_price = 10 := by
sorry

end NUMINAMATH_CALUDE_tshirt_socks_price_difference_l1394_139449


namespace NUMINAMATH_CALUDE_min_cans_correct_l1394_139457

/-- The number of ounces in one can of soda -/
def ounces_per_can : ℕ := 12

/-- The number of ounces in a gallon -/
def ounces_per_gallon : ℕ := 128

/-- The minimum number of cans needed to provide at least a gallon of soda -/
def min_cans : ℕ := 11

/-- Theorem stating that min_cans is the minimum number of cans needed to provide at least a gallon of soda -/
theorem min_cans_correct : 
  (∀ n : ℕ, n * ounces_per_can ≥ ounces_per_gallon → n ≥ min_cans) ∧ 
  (min_cans * ounces_per_can ≥ ounces_per_gallon) :=
sorry

end NUMINAMATH_CALUDE_min_cans_correct_l1394_139457


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l1394_139459

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) (hp_mod : p % 8 = 3) :
  ∀ (x y : ℚ), p^2 * x^4 - 6*p*x^2 + 1 = y^2 ↔ (x = 0 ∧ (y = 1 ∨ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l1394_139459


namespace NUMINAMATH_CALUDE_solve_head_circumference_problem_l1394_139491

def head_circumference_problem (jack_circumference charlie_circumference bill_circumference : ℝ) : Prop :=
  jack_circumference = 12 ∧
  bill_circumference = 10 ∧
  bill_circumference = (2/3) * charlie_circumference ∧
  ∃ x, charlie_circumference = (1/2) * jack_circumference + x ∧
  x = 9

theorem solve_head_circumference_problem :
  ∀ jack_circumference charlie_circumference bill_circumference,
  head_circumference_problem jack_circumference charlie_circumference bill_circumference :=
by
  sorry

end NUMINAMATH_CALUDE_solve_head_circumference_problem_l1394_139491


namespace NUMINAMATH_CALUDE_function_composition_problem_l1394_139478

/-- Given two functions f and g satisfying certain conditions, prove that [g(9)]^4 = 81 -/
theorem function_composition_problem 
  (f g : ℝ → ℝ) 
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^4)
  (h3 : g 81 = 81) :
  (g 9)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_problem_l1394_139478


namespace NUMINAMATH_CALUDE_lydias_flowering_plants_fraction_l1394_139487

theorem lydias_flowering_plants_fraction (total_plants : ℕ) 
  (flowering_percentage : ℚ) (flowers_per_plant : ℕ) (total_flowers_on_porch : ℕ) :
  total_plants = 80 →
  flowering_percentage = 2/5 →
  flowers_per_plant = 5 →
  total_flowers_on_porch = 40 →
  (total_flowers_on_porch / flowers_per_plant) / (flowering_percentage * total_plants) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lydias_flowering_plants_fraction_l1394_139487


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1394_139423

theorem simplify_polynomial (x : ℝ) : 
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 
  4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1394_139423


namespace NUMINAMATH_CALUDE_selena_remaining_money_is_33_74_l1394_139415

/-- Calculates the amount Selena will be left with after paying for her meal including taxes. -/
def selena_remaining_money (tip : ℚ) (steak_price : ℚ) (burger_price : ℚ) (ice_cream_price : ℚ)
  (steak_tax : ℚ) (burger_tax : ℚ) (ice_cream_tax : ℚ) : ℚ :=
  let steak_total := 2 * steak_price * (1 + steak_tax)
  let burger_total := 2 * burger_price * (1 + burger_tax)
  let ice_cream_total := 3 * ice_cream_price * (1 + ice_cream_tax)
  tip - (steak_total + burger_total + ice_cream_total)

/-- Theorem stating that Selena will be left with $33.74 after paying for her meal including taxes. -/
theorem selena_remaining_money_is_33_74 :
  selena_remaining_money 99 24 3.5 2 0.07 0.06 0.08 = 33.74 := by
  sorry

end NUMINAMATH_CALUDE_selena_remaining_money_is_33_74_l1394_139415


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1394_139452

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : a ≠ 2*b) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1394_139452


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1394_139477

theorem no_solutions_for_equation : 
  ¬∃ (a b : ℕ+), 
    a ≥ b ∧ 
    a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1394_139477


namespace NUMINAMATH_CALUDE_square_perimeter_with_inscribed_circles_l1394_139497

/-- A circle with radius 4 -/
structure Circle :=
  (radius : ℝ)
  (h_radius : radius = 4)

/-- A square that inscribes four identical circles -/
structure Square :=
  (side : ℝ)
  (circles : Fin 4 → Circle)
  (h_inscribed : ∀ i, circles i = { radius := 4, h_radius := rfl })
  (h_touching : ∀ i, circles i.val = circles ((i + 1) % 4))

/-- The perimeter of a square that inscribes four identical circles with radius 4 is 32 -/
theorem square_perimeter_with_inscribed_circles (s : Square) : 
  4 * s.side = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_with_inscribed_circles_l1394_139497
