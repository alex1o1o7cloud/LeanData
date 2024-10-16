import Mathlib

namespace NUMINAMATH_CALUDE_initial_persons_count_l2877_287735

/-- Represents the number of days to complete the work initially -/
def initial_days : ℕ := 18

/-- Represents the number of days worked before adding more persons -/
def days_before_addition : ℕ := 6

/-- Represents the number of persons added -/
def persons_added : ℕ := 4

/-- Represents the number of days to complete the remaining work after adding persons -/
def remaining_days : ℕ := 9

/-- Represents the total amount of work -/
def total_work : ℚ := 1

/-- Theorem stating the initial number of persons working on the project -/
theorem initial_persons_count : 
  ∃ (P : ℕ), 
    (P * initial_days : ℚ) * total_work = 
    (P * days_before_addition + (P + persons_added) * remaining_days : ℚ) * total_work ∧ 
    P = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l2877_287735


namespace NUMINAMATH_CALUDE_equation_solution_l2877_287762

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2877_287762


namespace NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l2877_287798

/-- Given a sphere circumscribed around a cube, if the volume of the sphere is 32π/3,
    then the edge length of the cube is 4√3/3. -/
theorem cube_edge_length_from_circumscribed_sphere_volume :
  ∀ (r : ℝ) (edge : ℝ),
  r > 0 →
  edge > 0 →
  (4 / 3) * π * r^3 = 32 * π / 3 →
  edge = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l2877_287798


namespace NUMINAMATH_CALUDE_sum_after_changes_l2877_287746

theorem sum_after_changes (A B : ℤ) (h : A + B = 100) : 
  (A - 35) + (B + 15) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_changes_l2877_287746


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l2877_287723

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l2877_287723


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2877_287744

/-- Given that 50 cows eat 50 bags of husk in 50 days, 
    prove that one cow will eat one bag of husk in the same number of days. -/
theorem cow_husk_consumption (days : ℕ) 
  (h : 50 * 50 = 50 * days) : 
  1 * 1 = 1 * days :=
by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2877_287744


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l2877_287772

/-- Given a sinusoidal function y = a * sin(bx + c) + d with positive constants a, b, c, and d,
    if the function oscillates between 5 and -3, then a = 4 -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l2877_287772


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2877_287702

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ¬∃ n : ℕ, is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2877_287702


namespace NUMINAMATH_CALUDE_intersection_A_B_l2877_287738

-- Define set A
def A : Set ℝ := {x | x^2 - 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x | 1 ≤ x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : 
  ∀ x : ℝ, x ∈ A ∩ B ↔ 1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2877_287738


namespace NUMINAMATH_CALUDE_sequence_a_properties_l2877_287759

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * sequence_a (n + 1) - sequence_a n

theorem sequence_a_properties :
  (∀ n : ℕ, ∃ k : ℤ, sequence_a n = k) ∧
  (∀ n : ℕ, 3 ∣ sequence_a n ↔ 3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_properties_l2877_287759


namespace NUMINAMATH_CALUDE_certain_event_red_ball_l2877_287755

/-- A bag containing colored balls -/
structure Bag where
  yellow : ℕ
  red : ℕ

/-- The probability of drawing at least one red ball when drawing two balls from the bag -/
def prob_at_least_one_red (b : Bag) : ℚ :=
  1 - (b.yellow / (b.yellow + b.red)) * ((b.yellow - 1) / (b.yellow + b.red - 1))

/-- Theorem stating that drawing at least one red ball is a certain event 
    when drawing two balls from a bag with one yellow and three red balls -/
theorem certain_event_red_ball : 
  let b : Bag := { yellow := 1, red := 3 }
  prob_at_least_one_red b = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_event_red_ball_l2877_287755


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l2877_287704

/-- Proves that an employee's pay after a raise and subsequent cut is 75% of their pay after the raise -/
theorem employee_pay_percentage (initial_pay : ℝ) (raise_percentage : ℝ) (final_pay : ℝ) : 
  initial_pay = 10 →
  raise_percentage = 20 →
  final_pay = 9 →
  final_pay / (initial_pay * (1 + raise_percentage / 100)) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l2877_287704


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l2877_287777

theorem stratified_sampling_female_count :
  ∀ (total_male total_female : ℕ) (male_prob : ℚ),
    total_male = 28 →
    total_female = 21 →
    male_prob = 1/7 →
    (total_female : ℚ) * male_prob = 3 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l2877_287777


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2877_287721

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2 + Complex.I) :
  z = 1/2 + 3/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2877_287721


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2877_287789

theorem complex_fraction_calculation (z : ℂ) (h : z = 1 - I) : z^2 / (z - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2877_287789


namespace NUMINAMATH_CALUDE_right_triangle_median_length_l2877_287757

theorem right_triangle_median_length (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_length_l2877_287757


namespace NUMINAMATH_CALUDE_n_power_37_minus_n_divisibility_l2877_287749

theorem n_power_37_minus_n_divisibility (n : ℤ) : 
  (∃ k : ℤ, n^37 - n = 91 * k) ∧ 
  (∃ m : ℤ, n^37 - n = 3276 * m) ∧
  (∀ l : ℤ, l > 3276 → ∃ p : ℤ, ¬ (∃ q : ℤ, p^37 - p = l * q)) :=
by sorry

end NUMINAMATH_CALUDE_n_power_37_minus_n_divisibility_l2877_287749


namespace NUMINAMATH_CALUDE_intersection_equivalence_l2877_287708

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (intersect : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_intersects_plane : Line → Plane → Prop)
variable (plane_intersects_plane : Plane → Plane → Prop)

-- Define our specific objects
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem intersection_equivalence 
  (h1 : intersect l m)
  (h2 : in_plane l α)
  (h3 : in_plane m α)
  (h4 : ¬ in_plane l β)
  (h5 : ¬ in_plane m β)
  : (line_intersects_plane l β ∨ line_intersects_plane m β) ↔ plane_intersects_plane α β :=
sorry

end NUMINAMATH_CALUDE_intersection_equivalence_l2877_287708


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2877_287787

theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2877_287787


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l2877_287794

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- 
If for an arithmetic sequence, 2S₃ = 3S₂ + 6, 
then the common difference is 2 
-/
theorem arithmetic_seq_common_diff 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l2877_287794


namespace NUMINAMATH_CALUDE_unique_line_through_points_l2877_287774

-- Define a type for points in a plane
axiom Point : Type

-- Define a type for straight lines
axiom Line : Type

-- Define a relation for a point being on a line
axiom on_line : Point → Line → Prop

-- Axiom: For any two distinct points, there exists a line passing through both points
axiom line_through_points (P Q : Point) (h : P ≠ Q) : ∃ L : Line, on_line P L ∧ on_line Q L

-- Theorem: There is a unique straight line passing through any two distinct points
theorem unique_line_through_points (P Q : Point) (h : P ≠ Q) : 
  ∃! L : Line, on_line P L ∧ on_line Q L :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_points_l2877_287774


namespace NUMINAMATH_CALUDE_book_sale_pricing_l2877_287776

theorem book_sale_pricing (total_books : ℕ) (higher_price lower_price total_earnings : ℚ) :
  total_books = 10 →
  lower_price = 2 →
  total_earnings = 22 →
  (2 / 5 : ℚ) * total_books * higher_price + (3 / 5 : ℚ) * total_books * lower_price = total_earnings →
  higher_price = (5 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_book_sale_pricing_l2877_287776


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2877_287751

/-- Calculate the cost of plastering a tank's walls and bottom -/
theorem tank_plastering_cost 
  (length : ℝ) 
  (width : ℝ) 
  (depth : ℝ) 
  (cost_per_sq_m : ℝ) : 
  length = 25 → 
  width = 12 → 
  depth = 6 → 
  cost_per_sq_m = 0.75 → 
  2 * (length * depth + width * depth) + length * width = 744 ∧ 
  (2 * (length * depth + width * depth) + length * width) * cost_per_sq_m = 558 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l2877_287751


namespace NUMINAMATH_CALUDE_max_assembly_and_impossibility_of_simultaneous_completion_l2877_287711

/-- Represents the number of wooden boards available -/
structure WoodenBoards :=
  (typeA : ℕ)
  (typeB : ℕ)

/-- Represents the requirements for assembling a desk and a chair -/
structure AssemblyRequirements :=
  (deskTypeA : ℕ)
  (deskTypeB : ℕ)
  (chairTypeA : ℕ)
  (chairTypeB : ℕ)

/-- Represents the assembly time for a desk and a chair -/
structure AssemblyTime :=
  (desk : ℕ)
  (chair : ℕ)

/-- Theorem stating the maximum number of desks and chairs that can be assembled
    and the impossibility of simultaneous completion -/
theorem max_assembly_and_impossibility_of_simultaneous_completion
  (boards : WoodenBoards)
  (requirements : AssemblyRequirements)
  (students : ℕ)
  (time : AssemblyTime)
  (h1 : boards.typeA = 400)
  (h2 : boards.typeB = 500)
  (h3 : requirements.deskTypeA = 2)
  (h4 : requirements.deskTypeB = 1)
  (h5 : requirements.chairTypeA = 1)
  (h6 : requirements.chairTypeB = 2)
  (h7 : students = 30)
  (h8 : time.desk = 10)
  (h9 : time.chair = 7) :
  (∃ (desks chairs : ℕ),
    desks = 100 ∧
    chairs = 200 ∧
    desks * requirements.deskTypeA + chairs * requirements.chairTypeA ≤ boards.typeA ∧
    desks * requirements.deskTypeB + chairs * requirements.chairTypeB ≤ boards.typeB ∧
    ∀ (desks' chairs' : ℕ),
      desks' > desks ∨ chairs' > chairs →
      desks' * requirements.deskTypeA + chairs' * requirements.chairTypeA > boards.typeA ∨
      desks' * requirements.deskTypeB + chairs' * requirements.chairTypeB > boards.typeB) ∧
  (∀ (group : ℕ),
    group ≤ students →
    (desks : ℚ) * time.desk / group ≠ (chairs : ℚ) * time.chair / (students - group)) :=
by sorry

end NUMINAMATH_CALUDE_max_assembly_and_impossibility_of_simultaneous_completion_l2877_287711


namespace NUMINAMATH_CALUDE_subtraction_multiplication_l2877_287717

theorem subtraction_multiplication : (3.65 - 1.25) * 2 = 4.80 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_l2877_287717


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2877_287793

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2877_287793


namespace NUMINAMATH_CALUDE_compound_weight_is_334_13_l2877_287771

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Bromine in g/mol -/
def Br_weight : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Chlorine in g/mol -/
def Cl_weight : ℝ := 35.45

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := Al_weight + 3 * Br_weight + 2 * O_weight + Cl_weight

/-- Theorem stating that the molecular weight of the compound is 334.13 g/mol -/
theorem compound_weight_is_334_13 : compound_weight = 334.13 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_is_334_13_l2877_287771


namespace NUMINAMATH_CALUDE_one_true_proposition_l2877_287797

/-- A parabola y = ax^2 + bx + c opens downwards if a < 0 -/
def opens_downwards (a b c : ℝ) : Prop := a < 0

/-- The set of x where y < 0 for the parabola y = ax^2 + bx + c -/
def negative_y_set (a b c : ℝ) : Set ℝ := {x | a * x^2 + b * x + c < 0}

/-- The original proposition -/
def original_prop (a b c : ℝ) : Prop :=
  opens_downwards a b c → negative_y_set a b c ≠ ∅

/-- The converse of the original proposition -/
def converse_prop (a b c : ℝ) : Prop :=
  negative_y_set a b c ≠ ∅ → opens_downwards a b c

/-- The inverse of the original proposition -/
def inverse_prop (a b c : ℝ) : Prop :=
  ¬(opens_downwards a b c) → negative_y_set a b c = ∅

/-- The contrapositive of the original proposition -/
def contrapositive_prop (a b c : ℝ) : Prop :=
  negative_y_set a b c = ∅ → ¬(opens_downwards a b c)

/-- The main theorem: exactly one of the converse, inverse, and contrapositive is true -/
theorem one_true_proposition :
  ∃! p : Prop, p = ∀ a b c : ℝ, converse_prop a b c ∨
                                p = ∀ a b c : ℝ, inverse_prop a b c ∨
                                p = ∀ a b c : ℝ, contrapositive_prop a b c :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l2877_287797


namespace NUMINAMATH_CALUDE_n_squared_plus_n_plus_one_properties_l2877_287784

theorem n_squared_plus_n_plus_one_properties (n : ℕ) :
  (Odd (n^2 + n + 1)) ∧ (¬ ∃ m : ℕ, n^2 + n + 1 = m^2) := by
  sorry

end NUMINAMATH_CALUDE_n_squared_plus_n_plus_one_properties_l2877_287784


namespace NUMINAMATH_CALUDE_equation_solution_l2877_287770

theorem equation_solution (r : ℚ) (h1 : r ≠ 2) (h2 : r ≠ -1) :
  (r + 3) / (r - 2) = (r - 1) / (r + 1) ↔ r = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2877_287770


namespace NUMINAMATH_CALUDE_total_spider_legs_l2877_287713

/-- The number of spiders in Christopher's room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l2877_287713


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l2877_287726

theorem reporters_covering_local_politics
  (percent_not_covering_local : Real)
  (percent_not_covering_politics : Real)
  (h1 : percent_not_covering_local = 0.3)
  (h2 : percent_not_covering_politics = 0.6) :
  (1 - percent_not_covering_politics) * (1 - percent_not_covering_local) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l2877_287726


namespace NUMINAMATH_CALUDE_no_solution_equation_l2877_287727

theorem no_solution_equation : ¬∃ (x : ℝ), x ≠ 2 ∧ x + 5 / (x - 2) = 2 + 5 / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2877_287727


namespace NUMINAMATH_CALUDE_point_on_lines_abs_diff_zero_l2877_287730

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line passing through the origin
structure Line where
  slope : ℝ

-- Define the two lines l₁ and l₂
def l₁ : Line := { slope := 1 }
def l₂ : Line := { slope := -1 }

-- A point is on a line if it satisfies the line's equation
def pointOnLine (p : Point2D) (l : Line) : Prop :=
  p.y = l.slope * p.x

-- The lines are symmetric about the y-axis
axiom line_symmetry : l₁.slope = -l₂.slope

-- Theorem stating that for any point on either line, |x| - |y| = 0
theorem point_on_lines_abs_diff_zero (p : Point2D) :
  (pointOnLine p l₁ ∨ pointOnLine p l₂) → |p.x| - |p.y| = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_lines_abs_diff_zero_l2877_287730


namespace NUMINAMATH_CALUDE_ab_bounds_l2877_287792

theorem ab_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + b^2 - a - b + a*b = 1) : 
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 - x - y + x*y = 1 → a + b ≥ x + y) ∧
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 - x - y + x*y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (a + b ≤ 2) ∧
  (a^2 + b^2 ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_ab_bounds_l2877_287792


namespace NUMINAMATH_CALUDE_count_valid_digits_l2877_287739

theorem count_valid_digits : 
  let is_valid (A : ℕ) := 0 ≤ A ∧ A ≤ 9 ∧ 571 * 10 + A < 5716
  (Finset.filter is_valid (Finset.range 10)).card = 6 := by
sorry

end NUMINAMATH_CALUDE_count_valid_digits_l2877_287739


namespace NUMINAMATH_CALUDE_det_A_eq_one_l2877_287742

/-- The matrix A_n as defined in the problem -/
def A (n : ℕ+) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j => (i.val + j.val - 2).choose (j.val - 1)

/-- The theorem stating that the determinant of A_n is 1 for all positive integers n -/
theorem det_A_eq_one (n : ℕ+) : Matrix.det (A n) = 1 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_one_l2877_287742


namespace NUMINAMATH_CALUDE_power_station_output_scientific_notation_l2877_287782

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem power_station_output_scientific_notation :
  toScientificNotation 448000 = ScientificNotation.mk 4.48 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_power_station_output_scientific_notation_l2877_287782


namespace NUMINAMATH_CALUDE_jamie_oyster_collection_l2877_287785

/-- The proportion of oysters that have pearls -/
def pearl_ratio : ℚ := 1/4

/-- The number of dives Jamie makes -/
def num_dives : ℕ := 14

/-- The total number of pearls Jamie collects -/
def total_pearls : ℕ := 56

/-- The number of oysters Jamie can collect during each dive -/
def oysters_per_dive : ℕ := 16

theorem jamie_oyster_collection :
  oysters_per_dive = (total_pearls / num_dives) / pearl_ratio := by
  sorry

end NUMINAMATH_CALUDE_jamie_oyster_collection_l2877_287785


namespace NUMINAMATH_CALUDE_george_socks_theorem_l2877_287756

/-- The number of socks George initially had -/
def initial_socks : ℕ := 28

/-- The number of socks George threw away -/
def thrown_away : ℕ := 4

/-- The number of new socks George bought -/
def new_socks : ℕ := 36

/-- The total number of socks George would have after the transactions -/
def final_socks : ℕ := 60

/-- Theorem stating that the initial number of socks is correct -/
theorem george_socks_theorem : 
  initial_socks - thrown_away + new_socks = final_socks :=
by sorry

end NUMINAMATH_CALUDE_george_socks_theorem_l2877_287756


namespace NUMINAMATH_CALUDE_percentage_fraction_difference_l2877_287786

theorem percentage_fraction_difference : 
  (65 / 100 * 40) - (4 / 5 * 25) = 6 := by sorry

end NUMINAMATH_CALUDE_percentage_fraction_difference_l2877_287786


namespace NUMINAMATH_CALUDE_car_original_price_verify_car_price_l2877_287714

/-- Calculates the original price of a car given the final price after discounts, taxes, and fees. -/
theorem car_original_price (final_price : ℝ) (doc_fee : ℝ) 
  (discount1 discount2 discount3 tax_rate : ℝ) : ℝ :=
  let remaining_after_discounts := (1 - discount1) * (1 - discount2) * (1 - discount3)
  let price_with_tax := remaining_after_discounts * (1 + tax_rate)
  (final_price - doc_fee) / price_with_tax

/-- Proves that the calculated original price satisfies the given conditions. -/
theorem verify_car_price : 
  let original_price := car_original_price 7500 200 0.15 0.20 0.25 0.10
  0.561 * original_price + 200 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_car_original_price_verify_car_price_l2877_287714


namespace NUMINAMATH_CALUDE_quadratic_coeff_unequal_l2877_287747

/-- Given a quadratic equation 3x^2 + 7x + 2k = 0 with zero discriminant,
    prove that the coefficients 3, 7, and k are unequal -/
theorem quadratic_coeff_unequal (k : ℝ) :
  (7^2 - 4*3*(2*k) = 0) →
  (3 ≠ 7 ∧ 3 ≠ k ∧ 7 ≠ k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coeff_unequal_l2877_287747


namespace NUMINAMATH_CALUDE_bacteria_growth_l2877_287725

/-- Calculates the bacteria population after a given time interval -/
def bacteria_population (initial_population : ℕ) (doubling_time : ℕ) (total_time : ℕ) : ℕ :=
  initial_population * 2 ^ (total_time / doubling_time)

/-- Theorem: Given 20 initial bacteria that double every 3 minutes, 
    the population after 15 minutes is 640 -/
theorem bacteria_growth : bacteria_population 20 3 15 = 640 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2877_287725


namespace NUMINAMATH_CALUDE_round_table_seats_l2877_287780

/-- Represents a round table with equally spaced seats numbered clockwise -/
structure RoundTable where
  num_seats : ℕ

/-- Represents a seat at the round table -/
structure Seat where
  number : ℕ

/-- Two seats are opposite if they are half the table size apart -/
def are_opposite (t : RoundTable) (s1 s2 : Seat) : Prop :=
  (s2.number - s1.number) % t.num_seats = t.num_seats / 2

theorem round_table_seats (t : RoundTable) (s1 s2 : Seat) :
  s1.number = 10 →
  s2.number = 29 →
  are_opposite t s1 s2 →
  t.num_seats = 38 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seats_l2877_287780


namespace NUMINAMATH_CALUDE_circle_chord_and_area_l2877_287754

theorem circle_chord_and_area (r : ℝ) (d : ℝ) (h1 : r = 5) (h2 : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  let area := π * r^2
  chord_length = 6 ∧ area = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_and_area_l2877_287754


namespace NUMINAMATH_CALUDE_last_pill_time_l2877_287722

/-- Represents the number of hours passed since the start time (12 o'clock) -/
def hoursPassed (n : ℕ) : ℕ := 5 * (n - 1)

/-- Converts hours passed to time on a 12-hour clock -/
def clockTime (hours : ℕ) : ℕ := (12 + hours) % 12

theorem last_pill_time :
  let totalPills : ℕ := 150
  let lastPillHours : ℕ := hoursPassed totalPills
  clockTime lastPillHours = 1 := by sorry

end NUMINAMATH_CALUDE_last_pill_time_l2877_287722


namespace NUMINAMATH_CALUDE_percent_increase_in_sales_l2877_287734

theorem percent_increase_in_sales (sales_this_year sales_last_year : ℝ) 
  (h1 : sales_this_year = 460)
  (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_in_sales_l2877_287734


namespace NUMINAMATH_CALUDE_complement_of_A_l2877_287790

def A : Set ℝ := {x | Real.log x > 0}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2877_287790


namespace NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two_l2877_287733

theorem xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two :
  ∀ x y : ℝ,
  x = Real.sqrt 3 + Real.sqrt 2 →
  y = Real.sqrt 3 - Real.sqrt 2 →
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two_l2877_287733


namespace NUMINAMATH_CALUDE_subset_count_with_nonempty_intersection_l2877_287715

theorem subset_count_with_nonempty_intersection :
  let A : Finset ℕ := Finset.range 10
  let B : Finset ℕ := {1, 2, 3, 4}
  (Finset.filter (fun C => (C ∩ B).Nonempty) (Finset.powerset A)).card = 960 := by
  sorry

end NUMINAMATH_CALUDE_subset_count_with_nonempty_intersection_l2877_287715


namespace NUMINAMATH_CALUDE_function_symmetry_l2877_287720

/-- Given a function f(x) = ax^4 + b*cos(x) - x, if f(-3) = 7, then f(3) = 1 -/
theorem function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^4 + b * Real.cos x - x
  f (-3) = 7 → f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2877_287720


namespace NUMINAMATH_CALUDE_initial_segment_theorem_l2877_287724

theorem initial_segment_theorem (m : ℕ) : ∃ (n k : ℕ), (10^k * m : ℕ) ≤ 2^n ∧ 2^n < 10^k * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_initial_segment_theorem_l2877_287724


namespace NUMINAMATH_CALUDE_parents_age_at_marks_birth_l2877_287753

/-- The age of Mark and John's parents when Mark was born, given their current ages and age differences. -/
theorem parents_age_at_marks_birth (mark_age john_age_diff parents_age_multiplier : ℕ) : 
  mark_age = 18 → 
  john_age_diff = 10 → 
  parents_age_multiplier = 5 → 
  (mark_age - john_age_diff) * parents_age_multiplier - mark_age = 22 := by
sorry

end NUMINAMATH_CALUDE_parents_age_at_marks_birth_l2877_287753


namespace NUMINAMATH_CALUDE_initial_price_equation_l2877_287709

/-- The initial price of speakers before discount -/
def initial_price : ℝ := 475

/-- The final price paid after discount -/
def final_price : ℝ := 199

/-- The discount amount saved -/
def discount : ℝ := 276

/-- Theorem stating that the initial price is equal to the sum of the final price and the discount -/
theorem initial_price_equation : initial_price = final_price + discount := by
  sorry

end NUMINAMATH_CALUDE_initial_price_equation_l2877_287709


namespace NUMINAMATH_CALUDE_right_triangle_k_value_l2877_287764

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the vectors
def vector_AB (k : ℝ) : ℝ × ℝ := (k, 1)
def vector_AC : ℝ × ℝ := (2, 3)

-- Define the right angle condition
def is_right_angle (t : Triangle) : Prop :=
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  (t.C.1 - t.A.1) * BC.1 + (t.C.2 - t.A.2) * BC.2 = 0

-- Theorem statement
theorem right_triangle_k_value (t : Triangle) (k : ℝ) :
  is_right_angle t →
  t.B - t.A = vector_AB k →
  t.C - t.A = vector_AC →
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_k_value_l2877_287764


namespace NUMINAMATH_CALUDE_max_wellfed_pikes_l2877_287779

/-- Represents the state of pikes in a pond -/
structure PikeState where
  total : ℕ
  wellfed : ℕ
  hungry : ℕ

/-- Defines what it means for a pike state to be valid -/
def is_valid_state (s : PikeState) : Prop :=
  s.total = s.wellfed + s.hungry ∧ s.wellfed * 3 + s.hungry ≤ 40

/-- Defines what it means for a pike state to be maximal -/
def is_maximal_state (s : PikeState) : Prop :=
  is_valid_state s ∧ ∀ t : PikeState, is_valid_state t → t.wellfed ≤ s.wellfed

/-- The theorem to be proved -/
theorem max_wellfed_pikes :
  ∃ s : PikeState, is_maximal_state s ∧ s.wellfed = 13 := by
  sorry

end NUMINAMATH_CALUDE_max_wellfed_pikes_l2877_287779


namespace NUMINAMATH_CALUDE_equation_solutions_l2877_287748

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 7) * (x - 3)
  { x : ℝ | x ≠ 3 ∧ x ≠ 7 ∧ f x / g x = 1 } = { 3 + Real.sqrt 3, 3 + Real.sqrt 5, 3 - Real.sqrt 5 } :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2877_287748


namespace NUMINAMATH_CALUDE_fraction_comparison_l2877_287767

theorem fraction_comparison :
  (373737 : ℚ) / 777777 = 37 / 77 ∧ (41 : ℚ) / 61 < 411 / 611 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2877_287767


namespace NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2877_287796

/-- Calculates the cost per pouch in cents given the number of boxes, pouches per box, and total cost in dollars. -/
def cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) : ℕ :=
  (total_cost_dollars * 100) / (boxes * pouches_per_box)

/-- Proves that for 10 boxes with 6 pouches each, costing $12 in total, each pouch costs 20 cents. -/
theorem capri_sun_cost_per_pouch :
  cost_per_pouch 10 6 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2877_287796


namespace NUMINAMATH_CALUDE_inequality_proof_l2877_287760

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  a / Real.sqrt (a^2 + 1) + b / Real.sqrt (b^2 + 1) + c / Real.sqrt (c^2 + 1) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2877_287760


namespace NUMINAMATH_CALUDE_exists_z_satisfying_equation_l2877_287736

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (3 * x)^3 + 3 * x + 5

-- State the theorem
theorem exists_z_satisfying_equation :
  ∃ z : ℝ, f (3 * z) = 3 ∧ z = -2 / 729 := by
  sorry

end NUMINAMATH_CALUDE_exists_z_satisfying_equation_l2877_287736


namespace NUMINAMATH_CALUDE_product_of_terms_l2877_287778

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 5) ^ 2 - 4 * (a 5) + 3 = 0 →
  (a 7) ^ 2 - 4 * (a 7) + 3 = 0 →
  a 2 * a 10 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_terms_l2877_287778


namespace NUMINAMATH_CALUDE_expression_value_l2877_287758

theorem expression_value (x y : ℚ) (hx : x = -5/4) (hy : y = -3/2) : -2 * x - y^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2877_287758


namespace NUMINAMATH_CALUDE_cousins_distribution_l2877_287719

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- The number of ways to distribute the cousins into the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousins_distribution : num_distributions = 51 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l2877_287719


namespace NUMINAMATH_CALUDE_pencil_difference_l2877_287799

theorem pencil_difference (price : ℚ) (jamar_count sharona_count : ℕ) : 
  price > 0.01 →
  price * jamar_count = 216/100 →
  price * sharona_count = 272/100 →
  sharona_count - jamar_count = 7 := by
sorry

end NUMINAMATH_CALUDE_pencil_difference_l2877_287799


namespace NUMINAMATH_CALUDE_solution_positivity_l2877_287703

theorem solution_positivity (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m * x - 1 = 2 * x) ↔ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_positivity_l2877_287703


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2877_287775

/-- The theorem states that for all real numbers x, arccos x is greater than arctan x
    if and only if x is in the interval [-1, 1/√3), given that arccos x is defined for x in [-1,1]. -/
theorem arccos_gt_arctan_iff (x : ℝ) : 
  Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1 : ℝ) (1 / Real.sqrt 3) ∧ x ≠ 1 / Real.sqrt 3 := by
  sorry

/-- This definition ensures that arccos is only defined on [-1, 1] -/
def arccos_domain (x : ℝ) : Prop := x ∈ Set.Icc (-1 : ℝ) 1

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2877_287775


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2877_287745

/-- Given a point M with coordinates (2,3), prove that its symmetric point N 
    with respect to the x-axis has coordinates (2, -3) -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ := (2, 3)
  let N : ℝ × ℝ := (M.1, -M.2)
  N = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2877_287745


namespace NUMINAMATH_CALUDE_function_max_at_zero_implies_a_geq_three_l2877_287707

/-- Given a function f(x) = x + a / (x + 1) defined on [0, 2] with maximum at x = 0, prove a ≥ 3 -/
theorem function_max_at_zero_implies_a_geq_three (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → x + a / (x + 1) ≤ a) →
  a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_max_at_zero_implies_a_geq_three_l2877_287707


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2877_287729

theorem perfect_square_condition (a b : ℕ+) :
  (∃ k : ℕ, (Nat.gcd a.val b.val + Nat.lcm a.val b.val) = k * (a.val + 1)) →
  b.val ≤ a.val →
  ∃ m : ℕ, b.val = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2877_287729


namespace NUMINAMATH_CALUDE_integer_solution_x4_y4_eq_3x3y_l2877_287718

theorem integer_solution_x4_y4_eq_3x3y :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y ↔ x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_solution_x4_y4_eq_3x3y_l2877_287718


namespace NUMINAMATH_CALUDE_not_p_or_not_q_true_l2877_287716

theorem not_p_or_not_q_true (p q : Prop) (h : ¬(p ∧ q)) : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_true_l2877_287716


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2877_287773

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the altitude from C to AB
def altitude_equation (x y : ℝ) : Prop :=
  2 * x + 5 * y - 30 = 0

-- Define the midline parallel to AC
def midline_equation (x : ℝ) : Prop :=
  x = 4

-- Theorem statement
theorem triangle_ABC_properties :
  -- The altitude from C to AB satisfies the equation
  (∀ x y : ℝ, altitude_equation x y ↔ 
    (x - C.1) * (B.2 - A.2) = (y - C.2) * (B.1 - A.1)) ∧
  -- The midline parallel to AC satisfies the equation
  (∀ x : ℝ, midline_equation x ↔ 
    x = (B.1 + C.1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2877_287773


namespace NUMINAMATH_CALUDE_volume_removed_tetrahedra_l2877_287783

/-- The volume of tetrahedra removed from a cube with specified edge divisions -/
theorem volume_removed_tetrahedra (edge_length : ℝ) (h_edge : edge_length = 2) :
  let central_segment : ℝ := 1
  let slanted_segment : ℝ := 1 / 2
  let height : ℝ := edge_length - central_segment / Real.sqrt 2
  let base_area : ℝ := 1 / 8
  let tetrahedron_volume : ℝ := 1 / 3 * base_area * height
  let total_volume : ℝ := 8 * tetrahedron_volume
  total_volume = 4 / 3 - Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_tetrahedra_l2877_287783


namespace NUMINAMATH_CALUDE_probability_theorem_l2877_287795

/-- The probability that the straight-line distance between two randomly chosen points
    on the sides of a square with side length 2 is at least 1 -/
def probability_distance_at_least_one (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- A square with side length 2 -/
def square_side_two : Set (ℝ × ℝ) :=
  sorry

theorem probability_theorem :
  probability_distance_at_least_one square_side_two = (26 - Real.pi) / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2877_287795


namespace NUMINAMATH_CALUDE_integral_2x_plus_exp_x_l2877_287741

open Real MeasureTheory Interval

theorem integral_2x_plus_exp_x : ∫ x in (-1)..(1), (2 * x + Real.exp x) = Real.exp 1 - Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_integral_2x_plus_exp_x_l2877_287741


namespace NUMINAMATH_CALUDE_max_value_of_f_l2877_287791

/-- The function f(x) = -5x^2 + 25x - 15 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 15

/-- Theorem stating that the maximum value of f(x) is 750 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 750 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2877_287791


namespace NUMINAMATH_CALUDE_least_integer_abs_inequality_l2877_287781

theorem least_integer_abs_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y + 5| ≤ 20 → x ≤ y) ∧ |3*x + 5| ≤ 20 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_abs_inequality_l2877_287781


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2877_287766

theorem multiplication_puzzle :
  ∀ A B C K : ℕ,
    A ∈ Finset.range 10 →
    B ∈ Finset.range 10 →
    C ∈ Finset.range 10 →
    K ∈ Finset.range 10 →
    A < B →
    A ≠ B ∧ A ≠ C ∧ A ≠ K ∧ B ≠ C ∧ B ≠ K ∧ C ≠ K →
    (10 * A + C) * (10 * B + C) = 111 * K →
    K * 111 = 100 * K + 10 * K + K →
    A = 2 ∧ B = 3 ∧ C = 7 ∧ K = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2877_287766


namespace NUMINAMATH_CALUDE_max_dip_amount_l2877_287788

/-- Given the following conditions:
  * total_money: The total amount of money available to spend on artichokes
  * cost_per_artichoke: The cost of each artichoke
  * artichokes_per_batch: The number of artichokes needed to make one batch of dip
  * ounces_per_batch: The number of ounces of dip produced from one batch

  Prove that the maximum amount of dip that can be made is 20 ounces.
-/
theorem max_dip_amount (total_money : ℚ) (cost_per_artichoke : ℚ) 
  (artichokes_per_batch : ℕ) (ounces_per_batch : ℚ) 
  (h1 : total_money = 15)
  (h2 : cost_per_artichoke = 5/4)
  (h3 : artichokes_per_batch = 3)
  (h4 : ounces_per_batch = 5) :
  (total_money / cost_per_artichoke) * (ounces_per_batch / artichokes_per_batch) = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_dip_amount_l2877_287788


namespace NUMINAMATH_CALUDE_real_part_of_complex_difference_times_i_l2877_287740

theorem real_part_of_complex_difference_times_i :
  let z₁ : ℂ := 4 + 29 * Complex.I
  let z₂ : ℂ := 6 + 9 * Complex.I
  (z₁ - z₂) * Complex.I |>.re = 20 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_complex_difference_times_i_l2877_287740


namespace NUMINAMATH_CALUDE_certain_to_draw_black_ball_l2877_287752

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 6

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := black_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Theorem stating that drawing at least one black ball is certain -/
theorem certain_to_draw_black_ball : 
  drawn_balls > white_balls → drawn_balls ≤ total_balls → true := by sorry

end NUMINAMATH_CALUDE_certain_to_draw_black_ball_l2877_287752


namespace NUMINAMATH_CALUDE_one_third_vector_AB_l2877_287765

/-- Given two vectors OA and OB in 2D space, prove that 1/3 of vector AB equals the specified result. -/
theorem one_third_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (4, 8) → OB = (-7, -2) → (1 / 3 : ℝ) • (OB - OA) = (-11/3, -10/3) := by
  sorry

end NUMINAMATH_CALUDE_one_third_vector_AB_l2877_287765


namespace NUMINAMATH_CALUDE_local_min_implies_a_equals_one_l2877_287706

/-- Given a function f(x) = ax^3 - 2x^2 + a^2x, where a is a real number,
    if f has a local minimum at x = 1, then a = 1. -/
theorem local_min_implies_a_equals_one (a : ℝ) :
  let f := λ x : ℝ => a * x^3 - 2 * x^2 + a^2 * x
  (∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_local_min_implies_a_equals_one_l2877_287706


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2877_287728

/-- The line l: mx - y + 3 - m = 0 and the circle C: x^2 + (y-1)^2 = 5 have at least one common point. -/
theorem line_circle_intersection (m : ℝ) : 
  ∃ (x y : ℝ), (m * x - y + 3 - m = 0) ∧ (x^2 + (y-1)^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2877_287728


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2877_287761

theorem gcd_of_three_numbers : Nat.gcd 9125 (Nat.gcd 4257 2349) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2877_287761


namespace NUMINAMATH_CALUDE_unique_p_value_l2877_287731

-- Define the properties of p, q, and s
def is_valid_triple (p q s : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime s ∧ p * q = s + 6 ∧ 3 < p ∧ p < q

-- Theorem statement
theorem unique_p_value :
  ∃! p, ∃ q s, is_valid_triple p q s ∧ p = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_p_value_l2877_287731


namespace NUMINAMATH_CALUDE_cash_preference_factors_l2877_287701

/-- Represents an economic factor influencing payment preference --/
structure EconomicFactor where
  description : String
  favors_cash : Bool

/-- Represents a large retail chain --/
structure RetailChain where
  name : String
  payment_preference : String

/-- Theorem: There exist at least three distinct economic factors that could lead large retail chains to prefer cash payments --/
theorem cash_preference_factors :
  ∃ (f1 f2 f3 : EconomicFactor),
    f1.favors_cash ∧ f2.favors_cash ∧ f3.favors_cash ∧
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (∃ (rc : RetailChain), rc.payment_preference = "cash") :=
by sorry

/-- Definition: Efficiency of operations as an economic factor --/
def efficiency_factor : EconomicFactor :=
  { description := "Efficiency of Operations", favors_cash := true }

/-- Definition: Cost of handling transactions as an economic factor --/
def cost_factor : EconomicFactor :=
  { description := "Cost of Handling Transactions", favors_cash := true }

/-- Definition: Risk of fraud as an economic factor --/
def risk_factor : EconomicFactor :=
  { description := "Risk of Fraud", favors_cash := true }

end NUMINAMATH_CALUDE_cash_preference_factors_l2877_287701


namespace NUMINAMATH_CALUDE_euler_conjecture_counterexample_l2877_287750

theorem euler_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end NUMINAMATH_CALUDE_euler_conjecture_counterexample_l2877_287750


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l2877_287743

theorem quadratic_root_implies_m (x m : ℝ) : 
  x = -1 → x^2 + m*x = 3 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l2877_287743


namespace NUMINAMATH_CALUDE_expression_value_l2877_287700

theorem expression_value (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  3 * y^2 - x^2 + 2 * (2 * x^2 - 3 * x * y) - 3 * (x^2 + y^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2877_287700


namespace NUMINAMATH_CALUDE_polynomial_transformation_l2877_287768

theorem polynomial_transformation (x y : ℝ) (hx : x ≠ 0) :
  y = x + 1/x →
  (x^4 - x^3 - 6*x^2 - x + 1 = 0) ↔ (x^2*(y^2 - y - 8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l2877_287768


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l2877_287712

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℝ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 :=
by sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l2877_287712


namespace NUMINAMATH_CALUDE_system_solution_equation_solution_l2877_287763

-- Problem 1: System of equations
theorem system_solution (x y : ℝ) : x + 2*y = 3 ∧ 2*x - y = 1 → x = 1 ∧ y = 1 := by
  sorry

-- Problem 2: Algebraic equation
theorem equation_solution (x : ℝ) : x ≠ 1 → (1 / (x - 1) + 2 = 5 / (1 - x)) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_equation_solution_l2877_287763


namespace NUMINAMATH_CALUDE_nautical_mile_conversion_l2877_287705

/-- Proves that under given conditions, one nautical mile equals 1.15 land miles -/
theorem nautical_mile_conversion (speed_one_sail : ℝ) (speed_two_sails : ℝ) 
  (time_one_sail : ℝ) (time_two_sails : ℝ) (total_distance : ℝ) :
  speed_one_sail = 25 →
  speed_two_sails = 50 →
  time_one_sail = 4 →
  time_two_sails = 4 →
  total_distance = 345 →
  speed_one_sail * time_one_sail + speed_two_sails * time_two_sails = total_distance →
  (1 : ℝ) * (345 / 300) = 1.15 := by
  sorry

#check nautical_mile_conversion

end NUMINAMATH_CALUDE_nautical_mile_conversion_l2877_287705


namespace NUMINAMATH_CALUDE_batsman_average_l2877_287732

def average (totalRuns : ℕ) (innings : ℕ) : ℚ :=
  (totalRuns : ℚ) / (innings : ℚ)

theorem batsman_average (totalRuns18 : ℕ) (totalRuns17 : ℕ) :
  average totalRuns18 18 = 18 →
  totalRuns18 = totalRuns17 + 1 →
  average totalRuns17 17 = 19 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l2877_287732


namespace NUMINAMATH_CALUDE_cricket_innings_problem_l2877_287710

theorem cricket_innings_problem (initial_average : ℝ) (runs_next_inning : ℕ) (average_increase : ℝ) :
  initial_average = 15 ∧ runs_next_inning = 59 ∧ average_increase = 4 →
  ∃ n : ℕ, n = 10 ∧
    initial_average * n + runs_next_inning = (initial_average + average_increase) * (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_innings_problem_l2877_287710


namespace NUMINAMATH_CALUDE_problem_1_l2877_287737

theorem problem_1 : (1.5 - 0.6) * (3 - 1.8) = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2877_287737


namespace NUMINAMATH_CALUDE_dinner_bill_problem_l2877_287769

theorem dinner_bill_problem (P : ℝ) : 
  P > 0 →  -- Assuming the price is positive
  (0.9 * P + 0.15 * P) = (0.9 * P + 0.15 * 0.9 * P + 0.51) →
  P = 34 := by
  sorry

#check dinner_bill_problem

end NUMINAMATH_CALUDE_dinner_bill_problem_l2877_287769
