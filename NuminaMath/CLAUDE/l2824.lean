import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l2824_282491

/-- Given a function f(x) = x² - 2x + 2a, where the solution set of f(x) ≤ 0 is {x | -2 ≤ x ≤ m},
    prove that a = -4 and m = 4, and find the range of c where (c+a)x² + 2(c+a)x - 1 < 0 always holds for x. -/
theorem problem_solution (a m : ℝ) (f : ℝ → ℝ) (c : ℝ) : 
  (f = fun x => x^2 - 2*x + 2*a) →
  (∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) →
  (a = -4 ∧ m = 4) ∧
  (∀ x, (c + a)*x^2 + 2*(c + a)*x - 1 < 0 ↔ 13/4 < c ∧ c < 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2824_282491


namespace NUMINAMATH_CALUDE_group_5_frequency_l2824_282444

theorem group_5_frequency (total : ℕ) (group1 group2 group3 group4 : ℕ) 
  (h_total : total = 50)
  (h_group1 : group1 = 2)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 15)
  (h_group4 : group4 = 5) :
  (total - group1 - group2 - group3 - group4 : ℚ) / total = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_group_5_frequency_l2824_282444


namespace NUMINAMATH_CALUDE_nondecreasing_function_l2824_282425

-- Define the property that a sequence is nondecreasing
def IsNondecreasingSeq (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n ≤ m → s n ≤ s m

-- State the theorem
theorem nondecreasing_function 
  (f : ℝ → ℝ) 
  (hf_dom : ∀ x, 0 < x → f x ≠ 0) 
  (hf_cont : Continuous f) 
  (h_seq : ∀ x > 0, IsNondecreasingSeq (fun n ↦ f (n * x))) : 
  ∀ x y, 0 < x → 0 < y → x ≤ y → f x ≤ f y :=
sorry

end NUMINAMATH_CALUDE_nondecreasing_function_l2824_282425


namespace NUMINAMATH_CALUDE_x_value_proof_l2824_282498

theorem x_value_proof : ∃ x : ℝ, x = 70 * (1 + 11/100) ∧ x = 77.7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2824_282498


namespace NUMINAMATH_CALUDE_unique_number_property_l2824_282402

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2824_282402


namespace NUMINAMATH_CALUDE_dice_throw_probability_l2824_282443

theorem dice_throw_probability (n : ℕ) : 
  (1 / 2 : ℚ) ^ n = 1 / 4 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_dice_throw_probability_l2824_282443


namespace NUMINAMATH_CALUDE_circumcircle_equation_correct_l2824_282475

/-- The circumcircle of a triangle AOB, where O is the origin (0, 0), A is at (4, 0), and B is at (0, 3) --/
def CircumcircleAOB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 3*p.2 = 0}

/-- Point O is the origin --/
def O : ℝ × ℝ := (0, 0)

/-- Point A has coordinates (4, 0) --/
def A : ℝ × ℝ := (4, 0)

/-- Point B has coordinates (0, 3) --/
def B : ℝ × ℝ := (0, 3)

/-- The circumcircle equation is correct for the given triangle AOB --/
theorem circumcircle_equation_correct :
  O ∈ CircumcircleAOB ∧ A ∈ CircumcircleAOB ∧ B ∈ CircumcircleAOB :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_correct_l2824_282475


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l2824_282490

theorem consecutive_odd_squares_sum : ∃ x : ℤ, 
  (x - 2)^2 + x^2 + (x + 2)^2 = 5555 ∧ 
  Odd x ∧ Odd (x - 2) ∧ Odd (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l2824_282490


namespace NUMINAMATH_CALUDE_mary_flour_problem_l2824_282454

theorem mary_flour_problem (recipe_flour : ℕ) (flour_to_add : ℕ) 
  (h1 : recipe_flour = 7)
  (h2 : flour_to_add = 5) :
  recipe_flour - flour_to_add = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_problem_l2824_282454


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2824_282406

theorem hyperbola_equation (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  (∃ e : ℝ, e = k * Real.sqrt 5 ∧ 
   (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = k * x)) →
  (∃ x y : ℝ, x^2 / (4 * b^2) - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2824_282406


namespace NUMINAMATH_CALUDE_gh_length_is_60_over_77_l2824_282437

/-- Represents a right triangle with squares inscribed -/
structure RightTriangleWithSquares where
  -- Right triangle ABC
  AC : ℝ
  BC : ℝ
  -- Square DEFG
  DE : ℝ
  -- Square GHIJ
  GH : ℝ
  -- Condition that E lies on AC and I lies on BC
  E_on_AC : ℝ
  I_on_BC : ℝ
  -- J is the midpoint of DG
  DJ : ℝ

/-- The length of GH in the inscribed square configuration -/
def ghLength (t : RightTriangleWithSquares) : ℝ := t.GH

/-- Theorem stating the length of GH in the given configuration -/
theorem gh_length_is_60_over_77 (t : RightTriangleWithSquares) 
  (h1 : t.AC = 4) 
  (h2 : t.BC = 3) 
  (h3 : t.DE = 2 * t.GH) 
  (h4 : t.DJ = t.GH) 
  (h5 : t.E_on_AC + t.DE + t.GH = t.AC) 
  (h6 : t.I_on_BC + t.GH = t.BC) :
  ghLength t = 60 / 77 := by
  sorry


end NUMINAMATH_CALUDE_gh_length_is_60_over_77_l2824_282437


namespace NUMINAMATH_CALUDE_expression_value_when_a_is_three_l2824_282448

theorem expression_value_when_a_is_three :
  let a : ℝ := 3
  (2 * a⁻¹ + a⁻¹ / 3) / a = 7 / 27 := by sorry

end NUMINAMATH_CALUDE_expression_value_when_a_is_three_l2824_282448


namespace NUMINAMATH_CALUDE_car_speed_problem_l2824_282432

/-- The speed of the first car in miles per hour -/
def v : ℝ := 70

/-- The speed of the second car in miles per hour -/
def speed_second_car : ℝ := 55

/-- The time of travel in hours -/
def time : ℝ := 2

/-- The total distance between the cars after the given time -/
def total_distance : ℝ := 250

theorem car_speed_problem :
  v * time + speed_second_car * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2824_282432


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l2824_282421

theorem cube_root_equation_solutions :
  ∀ x : ℝ, (x^(1/3) = 15 / (10 - x^(1/3))) ↔ (x = 125 ∨ x = 27) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l2824_282421


namespace NUMINAMATH_CALUDE_total_ears_is_500_l2824_282480

/-- Calculates the total number of ears for a given number of puppies -/
def total_ears (total_puppies droopy_eared_puppies pointed_eared_puppies : ℕ) : ℕ :=
  2 * total_puppies

/-- Theorem stating that the total number of ears is 500 given the problem conditions -/
theorem total_ears_is_500 :
  let total_puppies : ℕ := 250
  let droopy_eared_puppies : ℕ := 150
  let pointed_eared_puppies : ℕ := 100
  total_ears total_puppies droopy_eared_puppies pointed_eared_puppies = 500 := by
  sorry


end NUMINAMATH_CALUDE_total_ears_is_500_l2824_282480


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l2824_282462

theorem degree_to_radian_conversion (π : ℝ) (h : π > 0) :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian 15 = π / 12 := by
sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l2824_282462


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2824_282427

theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + 8 * Complex.I → z = -15 + 8 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2824_282427


namespace NUMINAMATH_CALUDE_negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg_l2824_282428

-- Define the property of two angles having the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

-- State the theorem
theorem negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg :
  same_terminal_side (-50) 310 := by
  sorry

end NUMINAMATH_CALUDE_negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg_l2824_282428


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_range_l2824_282403

/-- Given a real number a, f is a function from ℝ to ℝ defined as f(x) = x^2 + 2(a - 1)x + 2 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + 2*(a - 1)*x + 2

/-- The theorem states that if f is increasing on [4, +∞), then a ≥ -3 -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ≥ 4 → y ≥ 4 → x ≤ y → f a x ≤ f a y) →
  a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_range_l2824_282403


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2824_282473

theorem complex_product_theorem :
  let Q : ℂ := 4 + 3 * Complex.I
  let E : ℂ := 2 * Complex.I
  let D : ℂ := 4 - 3 * Complex.I
  Q * E * D = 50 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2824_282473


namespace NUMINAMATH_CALUDE_removal_ways_count_l2824_282468

/-- Represents a block in the stack -/
structure Block where
  layer : Nat
  exposed : Bool

/-- Represents the stack of blocks -/
def Stack : Type := List Block

/-- The initial stack configuration -/
def initialStack : Stack := sorry

/-- Function to check if a block can be removed -/
def canRemove (b : Block) (s : Stack) : Bool := sorry

/-- Function to remove a block and update the stack -/
def removeBlock (b : Block) (s : Stack) : Stack := sorry

/-- Function to count the number of ways to remove 5 blocks -/
def countRemovalWays (s : Stack) : Nat := sorry

/-- The main theorem stating the number of ways to remove 5 blocks -/
theorem removal_ways_count : 
  countRemovalWays initialStack = 3384 := by sorry

end NUMINAMATH_CALUDE_removal_ways_count_l2824_282468


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2824_282459

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5, 7}
def B : Set Nat := {3, 4, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2824_282459


namespace NUMINAMATH_CALUDE_bus_passenger_count_l2824_282458

def bus_passengers (initial_passengers : ℕ) (new_passengers : ℕ) : ℕ :=
  initial_passengers + new_passengers

theorem bus_passenger_count : 
  bus_passengers 4 13 = 17 := by sorry

end NUMINAMATH_CALUDE_bus_passenger_count_l2824_282458


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2824_282476

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2824_282476


namespace NUMINAMATH_CALUDE_sqrt_of_2_4_3_6_5_2_l2824_282413

theorem sqrt_of_2_4_3_6_5_2 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_2_4_3_6_5_2_l2824_282413


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2824_282423

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2824_282423


namespace NUMINAMATH_CALUDE_no_real_solutions_l2824_282472

theorem no_real_solutions :
  ∀ x : ℝ, (3 * x) / (x^2 + 2*x + 4) + (4 * x) / (x^2 - 4*x + 5) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2824_282472


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l2824_282407

/-- Given a geometric sequence {a_n} with sum S_n of the first n terms,
    if 8a_2 + a_5 = 0, then S_3 / a_3 = 3/4 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →
  (8 * (a 2) + (a 5) = 0) →
  (S 3) / (a 3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l2824_282407


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l2824_282415

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n ≥ 100) :
  sum_factorials n % 30 = sum_factorials 4 % 30 := by
  sorry

#eval sum_factorials 4 % 30  -- Should output 3

end NUMINAMATH_CALUDE_factorial_sum_remainder_l2824_282415


namespace NUMINAMATH_CALUDE_jack_final_amount_approx_l2824_282431

/-- Calculates the final amount of money Jack has in dollars after currency exchanges, fees, and spending. -/
def jack_final_amount (initial_dollars : ℝ) (initial_euros : ℝ) (initial_yen : ℝ) (initial_rubles : ℝ)
  (euro_to_dollar : ℝ) (yen_to_dollar : ℝ) (ruble_to_dollar : ℝ)
  (transaction_fee_rate : ℝ) (spending_rate : ℝ) : ℝ :=
  let converted_euros := initial_euros * euro_to_dollar
  let converted_yen := initial_yen * yen_to_dollar
  let converted_rubles := initial_rubles * ruble_to_dollar
  let total_before_fees := initial_dollars + converted_euros + converted_yen + converted_rubles
  let fees := (converted_euros + converted_yen + converted_rubles) * transaction_fee_rate
  let total_after_fees := total_before_fees - fees
  let amount_spent := total_after_fees * spending_rate
  total_after_fees - amount_spent

/-- Theorem stating that Jack's final amount is approximately 132.85 dollars. -/
theorem jack_final_amount_approx (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  ∀ (initial_dollars initial_euros initial_yen initial_rubles
     euro_to_dollar yen_to_dollar ruble_to_dollar
     transaction_fee_rate spending_rate : ℝ),
  (initial_dollars = 45 ∧ 
   initial_euros = 36 ∧ 
   initial_yen = 1350 ∧ 
   initial_rubles = 1500 ∧
   euro_to_dollar = 2 ∧ 
   yen_to_dollar = 0.009 ∧ 
   ruble_to_dollar = 0.013 ∧
   transaction_fee_rate = 0.01 ∧
   spending_rate = 0.1) →
  |jack_final_amount initial_dollars initial_euros initial_yen initial_rubles
                     euro_to_dollar yen_to_dollar ruble_to_dollar
                     transaction_fee_rate spending_rate - 132.85| < ε :=
by sorry

end NUMINAMATH_CALUDE_jack_final_amount_approx_l2824_282431


namespace NUMINAMATH_CALUDE_construction_rate_calculation_l2824_282482

/-- Represents the hourly rate for construction work -/
def construction_rate : ℝ := 14.67

/-- Represents the total weekly earnings -/
def total_earnings : ℝ := 300

/-- Represents the hourly rate for library work -/
def library_rate : ℝ := 8

/-- Represents the total weekly work hours -/
def total_hours : ℝ := 25

/-- Represents the weekly hours worked at the library -/
def library_hours : ℝ := 10

theorem construction_rate_calculation :
  construction_rate = (total_earnings - library_rate * library_hours) / (total_hours - library_hours) :=
by sorry

#check construction_rate_calculation

end NUMINAMATH_CALUDE_construction_rate_calculation_l2824_282482


namespace NUMINAMATH_CALUDE_smallest_positive_solution_is_18_l2824_282400

theorem smallest_positive_solution_is_18 : 
  let f : ℝ → ℝ := fun t => -t^2 + 14*t + 40
  ∃ t : ℝ, t > 0 ∧ f t = 94 ∧ ∀ s : ℝ, s > 0 ∧ f s = 94 → t ≤ s → t = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_is_18_l2824_282400


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2824_282419

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given conditions for the geometric sequence -/
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 = 10 ∧ a 2 + a 4 = 5

theorem geometric_sequence_fifth_term (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_cond : sequence_conditions a) : 
  a 5 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2824_282419


namespace NUMINAMATH_CALUDE_division_equality_l2824_282497

theorem division_equality : (786^2 * 74) / 23592 = 1938.8 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l2824_282497


namespace NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_l2824_282478

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 18 consecutive integers ≤ 2016, one is divisible by its digit sum -/
theorem exists_divisible_by_digit_sum :
  ∀ (start : ℕ), start + 17 ≤ 2016 →
  ∃ n ∈ Finset.range 18, (start + n).mod (sum_of_digits (start + n)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_l2824_282478


namespace NUMINAMATH_CALUDE_allen_pizza_payment_l2824_282477

theorem allen_pizza_payment (num_boxes : ℕ) (cost_per_box : ℚ) (tip_fraction : ℚ) (change_received : ℚ) :
  num_boxes = 5 →
  cost_per_box = 7 →
  tip_fraction = 1 / 7 →
  change_received = 60 →
  let total_cost := num_boxes * cost_per_box
  let tip := tip_fraction * total_cost
  let total_paid := total_cost + tip
  let money_given := total_paid + change_received
  money_given = 100 := by
  sorry

end NUMINAMATH_CALUDE_allen_pizza_payment_l2824_282477


namespace NUMINAMATH_CALUDE_defective_units_shipped_l2824_282409

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.04 →
  shipped_rate = 0.04 →
  (defective_rate * shipped_rate * 100) = 0.16 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l2824_282409


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2824_282435

theorem profit_percentage_calculation (cost_price selling_price : ℚ) : 
  cost_price = 500 → selling_price = 750 → 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2824_282435


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2824_282450

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (p : ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, a (n + 1) = p * a n + 2^n) →
  geometric_sequence a →
  ∀ n : ℕ, a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2824_282450


namespace NUMINAMATH_CALUDE_frog_flies_consumption_l2824_282460

/-- Proves that each frog needs to eat 30 flies per day in a swamp ecosystem -/
theorem frog_flies_consumption
  (fish_frog_consumption : ℕ) -- Number of frogs each fish eats per day
  (gharial_fish_consumption : ℕ) -- Number of fish each gharial eats per day
  (gharial_count : ℕ) -- Number of gharials in the swamp
  (total_flies_eaten : ℕ) -- Total number of flies eaten per day
  (h1 : fish_frog_consumption = 8)
  (h2 : gharial_fish_consumption = 15)
  (h3 : gharial_count = 9)
  (h4 : total_flies_eaten = 32400) :
  total_flies_eaten / (gharial_count * gharial_fish_consumption * fish_frog_consumption) = 30 := by
  sorry


end NUMINAMATH_CALUDE_frog_flies_consumption_l2824_282460


namespace NUMINAMATH_CALUDE_first_term_to_diff_ratio_l2824_282439

/-- An arithmetic sequence with a given property -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_property : (9 * a + 36 * d) = 3 * (6 * a + 15 * d)

/-- The ratio of the first term to the common difference is 1:-1 -/
theorem first_term_to_diff_ratio (seq : ArithmeticSequence) : seq.a / seq.d = -1 := by
  sorry

#check first_term_to_diff_ratio

end NUMINAMATH_CALUDE_first_term_to_diff_ratio_l2824_282439


namespace NUMINAMATH_CALUDE_solve_for_b_l2824_282434

/-- The piecewise function f(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 3^x

/-- Theorem stating that if f(f(1/2)) = 9, then b = -1/2 -/
theorem solve_for_b :
  ∀ b : ℝ, f b (f b (1/2)) = 9 → b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l2824_282434


namespace NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l2824_282494

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  exterior_angle : ℝ
  regular : exterior_angle * (sides : ℝ) = 360

-- Theorem statement
theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  ∀ p : RegularPolygon, p.exterior_angle = 18 → p.sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l2824_282494


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2824_282429

theorem simplify_polynomial (r : ℝ) : (2*r^2 + 5*r - 7) - (r^2 + 4*r - 6) = r^2 + r - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2824_282429


namespace NUMINAMATH_CALUDE_equilateral_triangle_reflection_parity_l2824_282483

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a reflection of a triangle -/
def reflect (t : Triangle) : Triangle := sorry

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Predicate to check if two triangles coincide -/
def coincide (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: If an equilateral triangle is reflected multiple times and 
    coincides with the original, the number of reflections is even -/
theorem equilateral_triangle_reflection_parity 
  (t : Triangle) (n : ℕ) (h1 : is_equilateral t) :
  (coincide ((reflect^[n]) t) t) → Even n := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_reflection_parity_l2824_282483


namespace NUMINAMATH_CALUDE_equation_solution_l2824_282466

theorem equation_solution (x p : ℝ) : 
  (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ↔ 
  (x = (4 - p) / Real.sqrt (8 * (2 - p)) ∧ 0 ≤ p ∧ p ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2824_282466


namespace NUMINAMATH_CALUDE_only_B_is_random_event_l2824_282440

-- Define the events
inductive Event
| A : Event  -- Water boils at 100°C under standard atmospheric pressure
| B : Event  -- Buying a lottery ticket and winning a prize
| C : Event  -- A runner's speed is 30 meters per second
| D : Event  -- Drawing a red ball from a bag containing only white and black balls

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.A => false
  | Event.B => true
  | Event.C => false
  | Event.D => false

-- Theorem: Only Event B is a random event
theorem only_B_is_random_event :
  ∀ e : Event, isRandomEvent e ↔ e = Event.B :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_random_event_l2824_282440


namespace NUMINAMATH_CALUDE_parking_lot_length_l2824_282442

/-- Proves that given the conditions of the parking lot problem, the length is 500 feet -/
theorem parking_lot_length
  (width : ℝ)
  (usable_percentage : ℝ)
  (area_per_car : ℝ)
  (total_cars : ℝ)
  (h1 : width = 400)
  (h2 : usable_percentage = 0.8)
  (h3 : area_per_car = 10)
  (h4 : total_cars = 16000)
  : ∃ (length : ℝ), length = 500 ∧ width * length * usable_percentage = total_cars * area_per_car :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_length_l2824_282442


namespace NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l2824_282453

theorem cubic_equation_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x^2 - a = 0 ∧
    y^3 - 3*y^2 - a = 0 ∧
    z^3 - 3*z^2 - a = 0) ↔
  -4 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l2824_282453


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2824_282457

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (1 - 2*x) ≥ 0 ↔ 1/2 ≤ x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2824_282457


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l2824_282489

def contains_seven (n : ℕ) : Prop :=
  ∃ d k, n = 10 * k + 7 * d ∧ d ≤ 9

theorem smallest_n_with_seven_in_squares : 
  ∀ n : ℕ, n < 26 → ¬(contains_seven (n^2) ∧ contains_seven ((n+1)^2)) ∧
  (contains_seven (26^2) ∧ contains_seven (27^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l2824_282489


namespace NUMINAMATH_CALUDE_wild_weatherman_answers_l2824_282449

/-- Represents the format of the text --/
inductive TextFormat
  | Interview
  | Diary
  | NewsStory
  | Announcement

/-- Represents Sam Champion's childhood career aspiration --/
inductive ChildhoodAspiration
  | SpaceScientist
  | Weatherman
  | NewsReporter
  | Meteorologist

/-- Represents the state of present weather forecasting technology --/
structure WeatherForecastingTechnology where
  moreExact : Bool
  stillImperfect : Bool

/-- Represents the name of the study of weather science --/
inductive WeatherScienceName
  | Meteorology
  | Forecasting
  | Geography
  | EarthScience

/-- The main theorem statement --/
theorem wild_weatherman_answers 
  (text_format : TextFormat)
  (sam_aspiration : ChildhoodAspiration)
  (forecast_tech : WeatherForecastingTechnology)
  (weather_science : WeatherScienceName) :
  text_format = TextFormat.Interview ∧
  sam_aspiration = ChildhoodAspiration.NewsReporter ∧
  forecast_tech.moreExact = true ∧
  forecast_tech.stillImperfect = true ∧
  weather_science = WeatherScienceName.Meteorology :=
by sorry

end NUMINAMATH_CALUDE_wild_weatherman_answers_l2824_282449


namespace NUMINAMATH_CALUDE_max_notebooks_is_14_l2824_282420

/-- Represents the pricing options for notebooks -/
structure NotebookPricing where
  single_price : ℕ
  pack3_price : ℕ
  pack7_price : ℕ

/-- Calculates the maximum number of notebooks that can be bought with a given budget and pricing -/
def max_notebooks (budget : ℕ) (pricing : NotebookPricing) : ℕ :=
  sorry

/-- The specific pricing and budget from the problem -/
def problem_pricing : NotebookPricing :=
  { single_price := 2
  , pack3_price := 5
  , pack7_price := 10 }

def problem_budget : ℕ := 20

/-- Theorem stating that the maximum number of notebooks that can be bought is 14 -/
theorem max_notebooks_is_14 : 
  max_notebooks problem_budget problem_pricing = 14 := by sorry

end NUMINAMATH_CALUDE_max_notebooks_is_14_l2824_282420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2824_282496

/-- Given an arithmetic sequence {a_n} where a₁ + a₅ + a₉ = 8π,
    prove that cos(a₃ + a₇) = -1/2 -/
theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  (a 1 + a 5 + a 9 = 8 * Real.pi) →                     -- given sum condition
  Real.cos (a 3 + a 7) = -1/2 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2824_282496


namespace NUMINAMATH_CALUDE_M_subset_N_l2824_282488

def M : Set ℝ := {-1, 1}

def N : Set ℝ := {x | (1 / x) < 3}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l2824_282488


namespace NUMINAMATH_CALUDE_girls_in_college_l2824_282464

theorem girls_in_college (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 312 →
  boys + girls = total →
  8 * girls = 5 * boys →
  girls = 120 := by
sorry

end NUMINAMATH_CALUDE_girls_in_college_l2824_282464


namespace NUMINAMATH_CALUDE_tennis_ball_difference_l2824_282438

/-- Given the number of tennis balls for Brian, Frodo, and Lily, prove that Frodo has 8 more tennis balls than Lily. -/
theorem tennis_ball_difference (brian frodo lily : ℕ) : 
  brian = 2 * frodo → 
  lily = 3 → 
  brian = 22 → 
  frodo - lily = 8 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_difference_l2824_282438


namespace NUMINAMATH_CALUDE_movie_of_the_year_criterion_l2824_282484

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 1500

/-- The fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1/2

/-- The smallest number of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 750

theorem movie_of_the_year_criterion :
  min_lists = (academy_members : ℚ) * required_fraction :=
by sorry

end NUMINAMATH_CALUDE_movie_of_the_year_criterion_l2824_282484


namespace NUMINAMATH_CALUDE_fraction_multiplication_identity_l2824_282430

theorem fraction_multiplication_identity : (5 : ℚ) / 7 * 7 / 5 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_identity_l2824_282430


namespace NUMINAMATH_CALUDE_max_value_expression_l2824_282467

theorem max_value_expression (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0) 
  (sum_constraint : x + y + z = 3) :
  (x^3 - x*y^2 + y^3) * (x^3 - x^2*z + z^3) * (y^3 - y^2*z + z^3) ≤ 1 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀ + y₀ + z₀ = 3 ∧
    (x₀^3 - x₀*y₀^2 + y₀^3) * (x₀^3 - x₀^2*z₀ + z₀^3) * (y₀^3 - y₀^2*z₀ + z₀^3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2824_282467


namespace NUMINAMATH_CALUDE_shortest_distance_is_zero_l2824_282441

/-- Define a 3D vector -/
def Vector3D := Fin 3 → ℝ

/-- Define the first line -/
def line1 (t : ℝ) : Vector3D := fun i => 
  match i with
  | 0 => 4 + 3*t
  | 1 => 1 - t
  | 2 => 3 + 2*t

/-- Define the second line -/
def line2 (s : ℝ) : Vector3D := fun i =>
  match i with
  | 0 => 1 + 2*s
  | 1 => 2 + 3*s
  | 2 => 5 - 2*s

/-- Calculate the square of the distance between two points -/
def distanceSquared (v w : Vector3D) : ℝ :=
  (v 0 - w 0)^2 + (v 1 - w 1)^2 + (v 2 - w 2)^2

/-- Theorem: The shortest distance between the two lines is 0 -/
theorem shortest_distance_is_zero :
  ∃ (t s : ℝ), distanceSquared (line1 t) (line2 s) = 0 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_is_zero_l2824_282441


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2824_282418

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Conditions for our specific triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.A + t.C = 2 * Real.pi / 3 ∧
  t.b = 1 ∧
  0 < t.A ∧ t.A < Real.pi / 2 ∧
  0 < t.B ∧ t.B < Real.pi / 2 ∧
  0 < t.C ∧ t.C < Real.pi / 2

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 ∧
  ∃ (max_area : Real), max_area = Real.sqrt 3 / 4 ∧
    ∀ (area : Real), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2824_282418


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l2824_282433

-- Define an acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem double_acute_angle_range (θ : ℝ) (h : is_acute_angle θ) :
  0 < 2 * θ ∧ 2 * θ < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l2824_282433


namespace NUMINAMATH_CALUDE_right_triangle_shorter_side_l2824_282461

/-- A right triangle with perimeter 40 and area 30 has a shorter side of length 5.25 -/
theorem right_triangle_shorter_side : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
  a^2 + b^2 = c^2 ∧        -- right triangle (Pythagorean theorem)
  a + b + c = 40 ∧         -- perimeter is 40
  (1/2) * a * b = 30 ∧     -- area is 30
  (a = 5.25 ∨ b = 5.25) :=  -- one shorter side is 5.25
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_side_l2824_282461


namespace NUMINAMATH_CALUDE_equivalent_discount_l2824_282426

/-- Proves that a single discount of 32.5% before taxes is equivalent to a series of discounts
    (25% followed by 10%) and a 5% sales tax, given an original price of $50. -/
theorem equivalent_discount (original_price : ℝ) (first_discount second_discount tax : ℝ)
  (single_discount : ℝ) :
  original_price = 50 →
  first_discount = 0.25 →
  second_discount = 0.10 →
  tax = 0.05 →
  single_discount = 0.325 →
  original_price * (1 - single_discount) * (1 + tax) =
  original_price * (1 - first_discount) * (1 - second_discount) * (1 + tax) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l2824_282426


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2824_282405

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a = 10 →
  A = π / 4 →
  B = π / 6 →
  a / Real.sin A = b / Real.sin B →
  b = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2824_282405


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l2824_282410

theorem triangle_area_ratio (BD DC : ℝ) (area_ABD : ℝ) (area_ADC : ℝ) :
  BD / DC = 5 / 2 →
  area_ABD = 40 →
  area_ADC = area_ABD * (DC / BD) →
  area_ADC = 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l2824_282410


namespace NUMINAMATH_CALUDE_g_at_negative_one_l2824_282446

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem g_at_negative_one : g (-1) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_one_l2824_282446


namespace NUMINAMATH_CALUDE_total_distance_driven_l2824_282499

/-- Proves that driving at 55 mph for 2 hours and then 3 hours results in a total distance of 275 miles -/
theorem total_distance_driven (speed : ℝ) (time_before_lunch : ℝ) (time_after_lunch : ℝ) 
  (h1 : speed = 55)
  (h2 : time_before_lunch = 2)
  (h3 : time_after_lunch = 3) :
  speed * time_before_lunch + speed * time_after_lunch = 275 := by
  sorry

#check total_distance_driven

end NUMINAMATH_CALUDE_total_distance_driven_l2824_282499


namespace NUMINAMATH_CALUDE_jonas_tshirts_l2824_282416

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe where
  socks : ℕ
  shoes : ℕ
  pants : ℕ
  tshirts : ℕ

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  2 * w.socks + 2 * w.shoes + 2 * w.pants + w.tshirts

/-- The theorem to prove -/
theorem jonas_tshirts : 
  ∀ w : Wardrobe, 
    w.socks = 20 → 
    w.shoes = 5 → 
    w.pants = 10 → 
    totalItems w + 2 * 35 = 2 * totalItems w → 
    w.tshirts = 70 := by
  sorry


end NUMINAMATH_CALUDE_jonas_tshirts_l2824_282416


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2824_282445

/-- Proves that k = 167/3 given that the line -1/3 - 3kx = 7y passes through the point (1/3, -8) -/
theorem line_passes_through_point (k : ℚ) : 
  (-1/3 : ℚ) - 3 * k * (1/3 : ℚ) = 7 * (-8 : ℚ) → k = 167/3 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2824_282445


namespace NUMINAMATH_CALUDE_cherries_eaten_l2824_282465

theorem cherries_eaten (initial : ℕ) (remaining : ℕ) (h1 : initial = 67) (h2 : remaining = 42) :
  initial - remaining = 25 := by
  sorry

end NUMINAMATH_CALUDE_cherries_eaten_l2824_282465


namespace NUMINAMATH_CALUDE_problem_statement_l2824_282404

theorem problem_statement (a b : ℝ) (ha : a ≠ b) 
  (ha_eq : a^2 - 13*a + 1 = 0) (hb_eq : b^2 - 13*b + 1 = 0) :
  b / (1 + b) + (a^2 + a) / (a^2 + 2*a + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2824_282404


namespace NUMINAMATH_CALUDE_max_value_expression_l2824_282401

theorem max_value_expression (a b c d : ℝ) 
  (ha : -5.5 ≤ a ∧ a ≤ 5.5)
  (hb : -5.5 ≤ b ∧ b ≤ 5.5)
  (hc : -5.5 ≤ c ∧ c ≤ 5.5)
  (hd : -5.5 ≤ d ∧ d ≤ 5.5) :
  (∀ a' b' c' d' : ℝ, 
    -5.5 ≤ a' ∧ a' ≤ 5.5 →
    -5.5 ≤ b' ∧ b' ≤ 5.5 →
    -5.5 ≤ c' ∧ c' ≤ 5.5 →
    -5.5 ≤ d' ∧ d' ≤ 5.5 →
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ 132) ∧
  (∃ a' b' c' d' : ℝ, 
    -5.5 ≤ a' ∧ a' ≤ 5.5 ∧
    -5.5 ≤ b' ∧ b' ≤ 5.5 ∧
    -5.5 ≤ c' ∧ c' ≤ 5.5 ∧
    -5.5 ≤ d' ∧ d' ≤ 5.5 ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 132) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2824_282401


namespace NUMINAMATH_CALUDE_cubic_function_values_l2824_282455

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

-- State the theorem
theorem cubic_function_values (a b : ℝ) (ha : a ≠ 0) :
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = 3) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≥ -29) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = -29) →
  ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_values_l2824_282455


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2824_282470

theorem imaginary_part_of_complex_fraction (i : Complex) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (1 + i)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2824_282470


namespace NUMINAMATH_CALUDE_find_wrong_height_l2824_282417

/-- Given a class of boys with an initially miscalculated average height and the correct average height after fixing one boy's height, find the wrongly written height of that boy. -/
theorem find_wrong_height (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (correct_avg : ℝ) 
    (hn : n = 35)
    (hi : initial_avg = 181)
    (ha : actual_height = 106)
    (hc : correct_avg = 179) :
    ∃ wrong_height : ℝ,
      wrong_height = n * initial_avg - (n * correct_avg - actual_height) :=
by sorry

end NUMINAMATH_CALUDE_find_wrong_height_l2824_282417


namespace NUMINAMATH_CALUDE_hex_to_decimal_l2824_282485

/-- Given a hexadecimal number 10k5₍₆₎ where k is a positive integer,
    if this number equals 239 when converted to decimal, then k = 3. -/
theorem hex_to_decimal (k : ℕ+) : (1 * 6^3 + k * 6 + 5 = 239) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_decimal_l2824_282485


namespace NUMINAMATH_CALUDE_sqrt_form_existence_l2824_282492

def has_sqrt_form (a : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x^2 + 2*y^2 = a) ∧ (2*x*y = 12)

theorem sqrt_form_existence :
  has_sqrt_form 17 ∧
  has_sqrt_form 22 ∧
  has_sqrt_form 38 ∧
  has_sqrt_form 73 ∧
  ¬(has_sqrt_form 54) :=
sorry

end NUMINAMATH_CALUDE_sqrt_form_existence_l2824_282492


namespace NUMINAMATH_CALUDE_power_function_through_point_l2824_282408

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →
  f 27 = 3 →
  a = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2824_282408


namespace NUMINAMATH_CALUDE_expression_value_l2824_282414

theorem expression_value (a b c d x : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : x^2 = 9)  -- distance from x to origin is 3
  : (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2824_282414


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2824_282424

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 / (x - 3) = 1 / x) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2824_282424


namespace NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l2824_282479

theorem smallest_m_satisfying_conditions : ∃ m : ℕ+,
  (∀ k : ℕ+, (∃ n : ℕ, 5 * k = n^5) ∧
             (∃ n : ℕ, 6 * k = n^6) ∧
             (∃ n : ℕ, 7 * k = n^7) →
   m ≤ k) ∧
  (∃ n : ℕ, 5 * m = n^5) ∧
  (∃ n : ℕ, 6 * m = n^6) ∧
  (∃ n : ℕ, 7 * m = n^7) ∧
  m = 2^35 * 3^35 * 5^84 * 7^90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l2824_282479


namespace NUMINAMATH_CALUDE_randy_initial_amount_l2824_282493

def initial_amount (spend_per_visit : ℕ) (visits_per_month : ℕ) (months : ℕ) (remaining : ℕ) : ℕ :=
  spend_per_visit * visits_per_month * months + remaining

theorem randy_initial_amount :
  initial_amount 2 4 12 104 = 200 :=
by sorry

end NUMINAMATH_CALUDE_randy_initial_amount_l2824_282493


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2824_282411

theorem sqrt_fraction_simplification :
  Real.sqrt ((25 : ℝ) / 49 + (16 : ℝ) / 81) = (53 : ℝ) / 63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2824_282411


namespace NUMINAMATH_CALUDE_big_dig_nickel_output_l2824_282471

/-- Represents the daily mining output of Big Dig Mining Company -/
structure MiningOutput where
  copper : ℝ
  iron : ℝ
  nickel : ℝ

/-- Calculates the total daily output -/
def totalOutput (output : MiningOutput) : ℝ :=
  output.copper + output.iron + output.nickel

theorem big_dig_nickel_output :
  ∀ output : MiningOutput,
  output.copper = 360 ∧
  output.iron = 0.6 * totalOutput output ∧
  output.nickel = 0.1 * totalOutput output →
  output.nickel = 120 := by
sorry


end NUMINAMATH_CALUDE_big_dig_nickel_output_l2824_282471


namespace NUMINAMATH_CALUDE_evaluate_expression_l2824_282463

theorem evaluate_expression : (800^2 : ℚ) / (300^2 - 296^2) = 640000 / 2384 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2824_282463


namespace NUMINAMATH_CALUDE_range_of_a_l2824_282487

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + (a + 2) * x + 1 ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2824_282487


namespace NUMINAMATH_CALUDE_maria_name_rearrangement_time_l2824_282495

/-- The time in hours to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (repeated_letters : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  let total_rearrangements := (Nat.factorial name_length) / (Nat.factorial repeated_letters)
  let minutes_needed := total_rearrangements / rearrangements_per_minute
  (minutes_needed : ℚ) / 60

/-- Theorem stating that the time to write all rearrangements of Maria's name is 0.125 hours -/
theorem maria_name_rearrangement_time :
  time_to_write_rearrangements 5 1 8 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_maria_name_rearrangement_time_l2824_282495


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_two_fourth_quadrant_implies_m_lt_neg_two_m_eq_two_implies_sum_of_parts_l2824_282412

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m - 1) * (m + 2) + (m - 1) * Complex.I

-- Theorem 1: If z is a pure imaginary number, then m = -2
theorem pure_imaginary_implies_m_eq_neg_two (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by sorry

-- Theorem 2: If z is in the fourth quadrant, then m < -2
theorem fourth_quadrant_implies_m_lt_neg_two (m : ℝ) :
  (z m).re > 0 ∧ (z m).im < 0 → m < -2 := by sorry

-- Theorem 3: If m = 2, then (z+i)/(z-1) = a + bi where a + b = 8/5
theorem m_eq_two_implies_sum_of_parts (m : ℝ) :
  m = 2 →
  ∃ a b : ℝ, (z m + Complex.I) / (z m - 1) = a + b * Complex.I ∧ a + b = 8/5 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_two_fourth_quadrant_implies_m_lt_neg_two_m_eq_two_implies_sum_of_parts_l2824_282412


namespace NUMINAMATH_CALUDE_john_hats_cost_l2824_282452

def weeks : ℕ := 20
def days_per_week : ℕ := 7
def odd_day_price : ℕ := 45
def even_day_price : ℕ := 60
def discount_threshold : ℕ := 50
def discount_rate : ℚ := 1 / 10

def total_hats : ℕ := weeks * days_per_week
def odd_days : ℕ := total_hats / 2
def even_days : ℕ := total_hats / 2

def total_cost : ℕ := odd_days * odd_day_price + even_days * even_day_price
def discounted_cost : ℚ := total_cost * (1 - discount_rate)

theorem john_hats_cost : 
  total_hats ≥ discount_threshold → discounted_cost = 6615 := by
  sorry

end NUMINAMATH_CALUDE_john_hats_cost_l2824_282452


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l2824_282451

theorem smallest_product_of_factors (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  (∃ k : ℕ, k * a = 48) → 
  (∃ l : ℕ, l * b = 48) → 
  ¬(∃ m : ℕ, m * (a * b) = 48) → 
  (∀ c d : ℕ, c ≠ d → c > 0 → d > 0 → 
    (∃ k : ℕ, k * c = 48) → 
    (∃ l : ℕ, l * d = 48) → 
    ¬(∃ m : ℕ, m * (c * d) = 48) → 
    a * b ≤ c * d) → 
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l2824_282451


namespace NUMINAMATH_CALUDE_expression_simplification_l2824_282436

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ 2) (h2 : a ≠ -2) (h3 : a ≠ 3) :
  ((a + 3) / (a^2 - 4) - a / (a^2 - a - 6)) / ((2*a - 9) / (5*a - 10)) = 
  5 / (a^2 - a - 6) := by
  sorry

-- Verifying the result for a = 5
example : 
  let a : ℝ := 5
  ((a + 3) / (a^2 - 4) - a / (a^2 - a - 6)) / ((2*a - 9) / (5*a - 10)) = 
  5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2824_282436


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2824_282474

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h1 : a 5 * a 6 = 3) 
  (h2 : a 9 * a 10 = 9) : 
  a 7 * a 8 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2824_282474


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2824_282456

theorem min_value_of_function (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 2) : 
  ∃ x, x > 2 ∧ x + 4 / (x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2824_282456


namespace NUMINAMATH_CALUDE_common_difference_is_one_fourth_l2824_282447

/-- An arithmetic sequence with a_6 = 5 and a_10 = 6 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 6 = 5 ∧ a 10 = 6 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem common_difference_is_one_fourth
  (a : ℕ → ℚ) (h : arithmetic_sequence a) :
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_one_fourth_l2824_282447


namespace NUMINAMATH_CALUDE_distance_to_origin_is_sqrt2_l2824_282422

-- Define the ellipse parameters
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 1/2

-- Define the right focus
def right_focus (a c : ℝ) : Prop := c^2 = a^2 / 4

-- Define the quadratic equation and its roots
def quadratic_roots (a b c x₁ x₂ : ℝ) : Prop :=
  a * x₁^2 + 2 * b * x₁ + c = 0 ∧
  a * x₂^2 + 2 * b * x₂ + c = 0

-- Theorem statement
theorem distance_to_origin_is_sqrt2
  (a b c x₁ x₂ : ℝ)
  (h_ellipse : ellipse a b x₁ x₂)
  (h_eccentricity : eccentricity (Real.sqrt (1 - b^2 / a^2)))
  (h_focus : right_focus a c)
  (h_roots : quadratic_roots a (Real.sqrt (a^2 - c^2)) c x₁ x₂) :
  Real.sqrt (x₁^2 + x₂^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_is_sqrt2_l2824_282422


namespace NUMINAMATH_CALUDE_license_plate_increase_l2824_282486

theorem license_plate_increase : 
  let old_plates := 26 * 10^3
  let new_plates := 26^2 * 10^4
  new_plates / old_plates = 260 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2824_282486


namespace NUMINAMATH_CALUDE_prob_score_exceeds_14_is_0_3_expected_value_two_triple_jumps_is_13_6_l2824_282469

-- Define the success rates and scores
def triple_jump_success_rate : ℝ := 0.7
def quadruple_jump_success_rate : ℝ := 0.3
def successful_triple_jump_score : ℕ := 8
def failed_triple_jump_score : ℕ := 4
def successful_quadruple_jump_score : ℕ := 15
def failed_quadruple_jump_score : ℕ := 6

-- Define the probability of score exceeding 14 points for triple jump followed by quadruple jump
def prob_score_exceeds_14 : ℝ := 
  triple_jump_success_rate * quadruple_jump_success_rate + 
  (1 - triple_jump_success_rate) * quadruple_jump_success_rate

-- Define the expected value of score for two consecutive triple jumps
def expected_value_two_triple_jumps : ℝ := 
  (1 - triple_jump_success_rate)^2 * (2 * failed_triple_jump_score) +
  2 * triple_jump_success_rate * (1 - triple_jump_success_rate) * (successful_triple_jump_score + failed_triple_jump_score) +
  triple_jump_success_rate^2 * (2 * successful_triple_jump_score)

-- Theorem statements
theorem prob_score_exceeds_14_is_0_3 : 
  prob_score_exceeds_14 = 0.3 := by sorry

theorem expected_value_two_triple_jumps_is_13_6 : 
  expected_value_two_triple_jumps = 13.6 := by sorry

end NUMINAMATH_CALUDE_prob_score_exceeds_14_is_0_3_expected_value_two_triple_jumps_is_13_6_l2824_282469


namespace NUMINAMATH_CALUDE_line_equation_l2824_282481

/-- A line passing through point A(1,4) with zero sum of intercepts on coordinate axes -/
structure LineWithZeroSumIntercepts where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point A(1,4) -/
  passes_through_A : 4 = slope * 1 + y_intercept
  /-- The sum of intercepts on coordinate axes is zero -/
  zero_sum_intercepts : 1 - (4 - y_intercept) / slope + y_intercept = 0

/-- The equation of the line is either 4x-y=0 or x-y+3=0 -/
theorem line_equation (l : LineWithZeroSumIntercepts) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2824_282481
