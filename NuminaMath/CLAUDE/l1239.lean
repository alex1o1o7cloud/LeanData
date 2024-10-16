import Mathlib

namespace NUMINAMATH_CALUDE_book_arrangement_count_l1239_123990

-- Define the number of physics and history books
def num_physics_books : ℕ := 4
def num_history_books : ℕ := 6

-- Define the function to calculate the number of arrangements
def num_arrangements (p h : ℕ) : ℕ :=
  2 * (Nat.factorial p) * (Nat.factorial h)

-- Theorem statement
theorem book_arrangement_count :
  num_arrangements num_physics_books num_history_books = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1239_123990


namespace NUMINAMATH_CALUDE_range_of_a_l1239_123926

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4*x - 3) ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- State the theorem
theorem range_of_a : 
  (∀ x a : ℝ, q x a → p x) ∧ 
  (∃ x : ℝ, p x ∧ ∀ a : ℝ, ¬(q x a)) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∃ x : ℝ, q x a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1239_123926


namespace NUMINAMATH_CALUDE_kate_wands_proof_l1239_123921

/-- The number of wands Kate bought -/
def total_wands : ℕ := 3

/-- The cost of each wand -/
def wand_cost : ℕ := 60

/-- The selling price of each wand -/
def selling_price : ℕ := wand_cost + 5

/-- The total amount collected from sales -/
def total_collected : ℕ := 130

/-- Kate keeps one wand for herself -/
def kept_wands : ℕ := 1

theorem kate_wands_proof : 
  total_wands = (total_collected / selling_price) + kept_wands :=
by sorry

end NUMINAMATH_CALUDE_kate_wands_proof_l1239_123921


namespace NUMINAMATH_CALUDE_dinner_savings_l1239_123964

theorem dinner_savings (total_savings : ℝ) (individual_savings : ℝ) : 
  total_savings > 0 →
  individual_savings > 0 →
  total_savings = 2 * individual_savings →
  (3/4) * total_savings + 2 * (6 * 1.5 + 1) = total_savings →
  individual_savings = 40 := by
sorry

end NUMINAMATH_CALUDE_dinner_savings_l1239_123964


namespace NUMINAMATH_CALUDE_unique_function_theorem_l1239_123967

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ) (p : ℕ), Nat.Prime p →
    (p ∣ f (m + n) ↔ p ∣ (f m + f n))

theorem unique_function_theorem :
  ∃! f : ℕ → ℕ, is_surjective f ∧ satisfies_condition f :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l1239_123967


namespace NUMINAMATH_CALUDE_hotel_rooms_l1239_123959

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost total_revenue : ℚ) 
  (h1 : total_rooms = 260)
  (h2 : single_cost = 35)
  (h3 : double_cost = 60)
  (h4 : total_revenue = 14000) :
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    double_rooms = 196 :=
by sorry

end NUMINAMATH_CALUDE_hotel_rooms_l1239_123959


namespace NUMINAMATH_CALUDE_smallest_divisor_partition_l1239_123966

/-- A function that returns the sum of divisors of a positive integer -/
def sumOfDivisors (n : ℕ+) : ℕ := sorry

/-- A function that checks if the divisors of a number can be partitioned into three sets with equal sums -/
def canPartitionDivisors (n : ℕ+) : Prop := sorry

/-- The theorem stating that 120 is the smallest positive integer with the required property -/
theorem smallest_divisor_partition :
  (∀ m : ℕ+, m < 120 → ¬(canPartitionDivisors m)) ∧ 
  (canPartitionDivisors 120) := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_partition_l1239_123966


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1239_123930

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The sum of two line segments perpendicular to the asymptotes
    and passing through one of the foci -/
def sum_perp_segments (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_sum : sum_perp_segments h = h.a) : 
  eccentricity h = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1239_123930


namespace NUMINAMATH_CALUDE_unique_number_l1239_123902

/-- A six-digit number with leftmost digit 7 -/
def SixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 7

/-- Function to move the leftmost digit to the end -/
def moveLeftmostToEnd (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem -/
theorem unique_number : ∃! n : ℕ, SixDigitNumber n ∧ moveLeftmostToEnd n = n / 5 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l1239_123902


namespace NUMINAMATH_CALUDE_bag_probabilities_l1239_123957

/-- Definition of the bag of balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ

/-- Initial bag configuration -/
def initialBag : Bag := ⟨20, 5, 15⟩

/-- Probability of picking a ball of a certain color -/
def probability (bag : Bag) (color : ℕ) : ℚ :=
  color / bag.total

/-- Add balls to the bag -/
def addBalls (bag : Bag) (redAdd : ℕ) (yellowAdd : ℕ) : Bag :=
  ⟨bag.total + redAdd + yellowAdd, bag.red + redAdd, bag.yellow + yellowAdd⟩

theorem bag_probabilities (bag : Bag := initialBag) :
  (probability bag bag.yellow > probability bag bag.red) ∧
  (probability bag bag.red = 1/4) ∧
  (probability (addBalls bag 40 0) (bag.red + 40) = 3/4) ∧
  (probability (addBalls bag 14 4) (bag.red + 14) = 
   probability (addBalls bag 14 4) (bag.yellow + 4)) :=
by sorry

end NUMINAMATH_CALUDE_bag_probabilities_l1239_123957


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_proof_l1239_123955

theorem father_son_age_sum : ℕ → ℕ → ℕ
  | father_age, son_age =>
    father_age + 2 * son_age

theorem father_son_age_proof (father_age son_age : ℕ) 
  (h1 : father_age = 40) 
  (h2 : son_age = 15) : 
  father_son_age_sum father_age son_age = 70 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_proof_l1239_123955


namespace NUMINAMATH_CALUDE_smallest_total_books_l1239_123976

/-- Represents the number of books for each subject -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books -/
theorem smallest_total_books :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other → totalBooks other > 3000 →
      totalBooks books ≤ totalBooks other :=
sorry

end NUMINAMATH_CALUDE_smallest_total_books_l1239_123976


namespace NUMINAMATH_CALUDE_roberto_outfits_l1239_123945

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) (restricted_combinations : ℕ) : ℕ :=
  trousers * shirts * jackets - restricted_combinations

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits : 
  number_of_outfits 5 6 4 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1239_123945


namespace NUMINAMATH_CALUDE_sum_of_distinct_numbers_l1239_123984

theorem sum_of_distinct_numbers (x y u v : ℝ) : 
  x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ y ≠ u ∧ y ≠ v ∧ u ≠ v →
  (x + u) / (x + v) = (y + v) / (y + u) →
  x + y + u + v = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_numbers_l1239_123984


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1239_123968

theorem cube_volume_surface_area (V : ℝ) : 
  (∃ (x : ℝ), V = x^3 ∧ 2*V = 6*x^2) → V = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1239_123968


namespace NUMINAMATH_CALUDE_double_yellow_probability_blue_decode_probability_comparison_l1239_123991

-- Define the color transmission probabilities
structure ColorTransmission where
  α : Real
  β : Real
  γ : Real
  h_α : 0 < α ∧ α < 1
  h_β : 0 < β ∧ β < 1
  h_γ : 0 < γ ∧ γ < 1

-- Define the transmission schemes
inductive TransmissionScheme
| Single
| Double

-- Theorem for statement B
theorem double_yellow_probability (ct : ColorTransmission) :
  (ct.α ^ 2 : Real) = ct.α * ct.α := by sorry

-- Theorem for statement D
theorem blue_decode_probability_comparison (ct : ColorTransmission) :
  (1 - ct.α) ^ 2 < 1 - ct.α := by sorry

end NUMINAMATH_CALUDE_double_yellow_probability_blue_decode_probability_comparison_l1239_123991


namespace NUMINAMATH_CALUDE_equation_c_is_linear_l1239_123914

/-- Definition of a linear equation with one variable -/
def is_linear_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 3 = 5 -/
def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_c_is_linear : is_linear_one_var f :=
sorry

end NUMINAMATH_CALUDE_equation_c_is_linear_l1239_123914


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1239_123931

theorem asterisk_replacement : ∃ x : ℚ, (x / 18) * (36 / 72) = 1 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1239_123931


namespace NUMINAMATH_CALUDE_no_order_for_seven_l1239_123993

def f (x : ℕ) : ℕ := x^2 % 13

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem no_order_for_seven :
  ¬ ∃ n : ℕ, n > 0 ∧ iterate_f n 7 = 7 :=
sorry

end NUMINAMATH_CALUDE_no_order_for_seven_l1239_123993


namespace NUMINAMATH_CALUDE_larger_number_proof_l1239_123913

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 9660) :
  max a b = 460 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1239_123913


namespace NUMINAMATH_CALUDE_one_third_greater_than_decimal_l1239_123992

theorem one_third_greater_than_decimal : 
  ∃ (ε : ℚ), ε > 0 ∧ ε = 1 / (3 * 10^9) ∧ 1/3 = 0.333333333 + ε := by
  sorry

end NUMINAMATH_CALUDE_one_third_greater_than_decimal_l1239_123992


namespace NUMINAMATH_CALUDE_f_neg_one_eq_three_l1239_123919

/-- Given a function f(x) = x^2 - 2x, prove that f(-1) = 3 -/
theorem f_neg_one_eq_three (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*x) : f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_three_l1239_123919


namespace NUMINAMATH_CALUDE_julio_salary_julio_salary_is_500_l1239_123910

/-- Calculates Julio's salary for 3 weeks based on given conditions --/
theorem julio_salary (commission_per_customer : ℕ) (first_week_customers : ℕ) 
  (bonus : ℕ) (total_earnings : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_customers := first_week_customers + second_week_customers + third_week_customers
  let total_commission := total_customers * commission_per_customer
  let salary := total_earnings - (total_commission + bonus)
  salary

/-- Proves that Julio's salary for 3 weeks is $500 --/
theorem julio_salary_is_500 : 
  julio_salary 1 35 50 760 = 500 := by
  sorry

end NUMINAMATH_CALUDE_julio_salary_julio_salary_is_500_l1239_123910


namespace NUMINAMATH_CALUDE_four_white_possible_l1239_123988

/-- Represents the state of the urn -/
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

/-- Represents the possible operations on the urn -/
inductive Operation
  | removeFourBlackAddTwoBlack
  | removeThreeBlackOneWhiteAddOneBlackOneWhite
  | removeOneBlackThreeWhiteAddTwoWhite
  | removeFourWhiteAddTwoWhiteOneBlack

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.removeFourBlackAddTwoBlack => 
      ⟨state.white, state.black - 2⟩
  | Operation.removeThreeBlackOneWhiteAddOneBlackOneWhite => 
      ⟨state.white, state.black - 2⟩
  | Operation.removeOneBlackThreeWhiteAddTwoWhite => 
      ⟨state.white - 1, state.black - 1⟩
  | Operation.removeFourWhiteAddTwoWhiteOneBlack => 
      ⟨state.white - 2, state.black + 1⟩

/-- The theorem to be proved -/
theorem four_white_possible : 
  ∃ (ops : List Operation), 
    let final_state := ops.foldl applyOperation ⟨150, 150⟩
    final_state.white = 4 :=
sorry

end NUMINAMATH_CALUDE_four_white_possible_l1239_123988


namespace NUMINAMATH_CALUDE_grain_transfer_theorem_transfer_valid_l1239_123982

/-- The amount of grain to be transferred from Warehouse B to Warehouse A -/
def transfer : ℕ := 15

/-- The initial amount of grain in Warehouse A -/
def initial_A : ℕ := 540

/-- The initial amount of grain in Warehouse B -/
def initial_B : ℕ := 200

/-- Theorem stating that transferring the specified amount will result in
    Warehouse A having three times the grain of Warehouse B -/
theorem grain_transfer_theorem :
  (initial_A + transfer) = 3 * (initial_B - transfer) := by
  sorry

/-- Proof that the transfer amount is non-negative and not greater than
    the initial amount in Warehouse B -/
theorem transfer_valid :
  0 ≤ transfer ∧ transfer ≤ initial_B := by
  sorry

end NUMINAMATH_CALUDE_grain_transfer_theorem_transfer_valid_l1239_123982


namespace NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l1239_123949

theorem dot_product_of_specific_vectors :
  let a : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (25 * π / 180))
  let b : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (155 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l1239_123949


namespace NUMINAMATH_CALUDE_classroom_tables_l1239_123940

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of students in base 7 notation -/
def studentsBase7 : Nat := 321

/-- The number of students per table -/
def studentsPerTable : Nat := 3

/-- Theorem: The number of tables in the classroom is 54 -/
theorem classroom_tables :
  (base7ToBase10 studentsBase7) / studentsPerTable = 54 := by
  sorry


end NUMINAMATH_CALUDE_classroom_tables_l1239_123940


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l1239_123947

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 3

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 705

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l1239_123947


namespace NUMINAMATH_CALUDE_device_records_720_instances_l1239_123933

/-- Represents the number of instances recorded by a device in one hour -/
def instances_recorded (seconds_per_record : ℕ) : ℕ :=
  (60 * 60) / seconds_per_record

/-- Theorem stating that a device recording every 5 seconds for one hour will record 720 instances -/
theorem device_records_720_instances :
  instances_recorded 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_device_records_720_instances_l1239_123933


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1239_123969

theorem quadratic_equation_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2 * x₁^2 + 4 * m * x₁ + m = 0 ∧
    2 * x₂^2 + 4 * m * x₂ + m = 0 ∧
    x₁^2 + x₂^2 = 3/16) →
  m = -1/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1239_123969


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1239_123977

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1239_123977


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l1239_123962

theorem manuscript_typing_cost : 
  let total_pages : ℕ := 100
  let pages_revised_once : ℕ := 30
  let pages_revised_twice : ℕ := 20
  let pages_not_revised : ℕ := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost : ℕ := 5
  let revision_cost : ℕ := 3
  let total_cost : ℕ := 
    total_pages * initial_typing_cost + 
    pages_revised_once * revision_cost + 
    pages_revised_twice * revision_cost * 2
  total_cost = 710 := by
sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l1239_123962


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1239_123904

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- Theorem: The equation of the tangent line to y = 2x^3 - 3x + 1 at (1, 0) is 3x - y - 3 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 3 * x - y - 3 = 0} ↔
  y - point.2 = f' point.1 * (x - point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1239_123904


namespace NUMINAMATH_CALUDE_cassidy_profit_l1239_123942

/-- Cassidy's bread baking and selling scenario --/
def bread_scenario (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (midday_price : ℚ) (evening_price : ℚ) : Prop :=
  let morning_sold := total_loaves / 3
  let midday_remaining := total_loaves - morning_sold
  let midday_sold := midday_remaining / 2
  let evening_sold := midday_remaining - midday_sold
  let total_revenue := morning_sold * morning_price + midday_sold * midday_price + evening_sold * evening_price
  let total_cost := total_loaves * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 70

/-- Theorem stating Cassidy's profit is $70 --/
theorem cassidy_profit : 
  bread_scenario 60 1 3 2 (3/2) :=
sorry

end NUMINAMATH_CALUDE_cassidy_profit_l1239_123942


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1239_123980

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∃ (min : ℝ), min = 9/4 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 1/(1+x) + 4/(1+y) ≥ min) ∧
  a^2*b^2 + a^2 + b^2 ≥ a*b*(a+b+1) := by
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1239_123980


namespace NUMINAMATH_CALUDE_matrix_transformation_l1239_123901

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 6; 2, 7]
def matrix_B (x : ℤ) : Matrix (Fin 2) (Fin 2) ℤ := !![6, 2; 1, x]

theorem matrix_transformation (x : ℤ) : 
  Matrix.det matrix_A = Matrix.det (matrix_B x) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_matrix_transformation_l1239_123901


namespace NUMINAMATH_CALUDE_expand_product_l1239_123907

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1239_123907


namespace NUMINAMATH_CALUDE_line_parallel_plane_l1239_123915

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (not_subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane 
  (a b : Line) (α : Plane)
  (h1 : parallel_line a b)
  (h2 : parallel_line_plane b α)
  (h3 : not_subset a α) :
  parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_l1239_123915


namespace NUMINAMATH_CALUDE_p_days_correct_l1239_123906

/-- The number of days it takes for q to do the work alone -/
def q_days : ℝ := 10

/-- The fraction of work left after p and q work together for 2 days -/
def work_left : ℝ := 0.7

/-- The number of days it takes for p to do the work alone -/
def p_days : ℝ := 20

/-- Theorem stating that p_days is correct given the conditions -/
theorem p_days_correct : 
  2 * (1 / p_days + 1 / q_days) = 1 - work_left := by
  sorry

end NUMINAMATH_CALUDE_p_days_correct_l1239_123906


namespace NUMINAMATH_CALUDE_school_ratio_problem_l1239_123974

theorem school_ratio_problem (S T : ℕ) : 
  S / T = 50 →
  (S + 50) / (T + 5) = 25 →
  T = 3 := by
sorry

end NUMINAMATH_CALUDE_school_ratio_problem_l1239_123974


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_l1239_123951

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which point C lies
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Define the possible equations of the circumcircle
def circumcircle_eq1 (x y : ℝ) : Prop :=
  x^2 + y^2 - (1/2) * x - 5 * y - (3/2) = 0

def circumcircle_eq2 (x y : ℝ) : Prop :=
  x^2 + y^2 - (25/6) * x - (89/9) * y + (347/18) = 0

-- The theorem to be proved
theorem circumcircle_of_triangle :
  ∃ (C : ℝ × ℝ),
    line_C C.1 C.2 ∧
    (∀ (x y : ℝ), circumcircle_eq1 x y ∨ circumcircle_eq2 x y) :=
  sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_l1239_123951


namespace NUMINAMATH_CALUDE_square_product_extension_l1239_123932

theorem square_product_extension (a b : ℕ) 
  (h1 : ∃ x : ℕ, a * b = x ^ 2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y ^ 2) :
  ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_extension_l1239_123932


namespace NUMINAMATH_CALUDE_ratio_theorem_l1239_123978

theorem ratio_theorem (x y : ℝ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_l1239_123978


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l1239_123958

theorem cosine_sine_identity : 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l1239_123958


namespace NUMINAMATH_CALUDE_line_points_l1239_123916

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨0, -8⟩
  let p3 : Point := ⟨4, 4⟩
  let p4 : Point := ⟨2, 0⟩
  let p5 : Point := ⟨9, 19⟩
  let p6 : Point := ⟨-1, -9⟩
  let p7 : Point := ⟨-2, -10⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p4 ∧ 
  ¬collinear p1 p2 p6 ∧ 
  ¬collinear p1 p2 p7 :=
by sorry

end NUMINAMATH_CALUDE_line_points_l1239_123916


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1239_123920

/-- Given that x² is inversely proportional to y⁴, prove that x² = 2.25 when y = 4,
    given that x = 6 when y = 2. -/
theorem inverse_proportion_problem (k : ℝ) (x y : ℝ → ℝ) :
  (∀ t, t > 0 → x t ^ 2 * y t ^ 4 = k) →  -- x² is inversely proportional to y⁴
  x 2 = 6 →                               -- x = 6 when y = 2
  y 2 = 2 →                               -- y = 2 at this point
  y 4 = 4 →                               -- y = 4 at the point we're calculating
  x 4 ^ 2 = 2.25 :=                       -- x² = 2.25 when y = 4
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1239_123920


namespace NUMINAMATH_CALUDE_factors_72_l1239_123923

/-- The number of distinct positive factors of 72 -/
def num_factors_72 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 72 is 12 -/
theorem factors_72 : num_factors_72 = 12 := by sorry

end NUMINAMATH_CALUDE_factors_72_l1239_123923


namespace NUMINAMATH_CALUDE_throwers_count_l1239_123998

/-- Represents a football team with throwers and non-throwers (left-handed and right-handed) -/
structure FootballTeam where
  total_players : ℕ
  throwers : ℕ
  left_handed : ℕ
  right_handed : ℕ

/-- Conditions for the football team -/
def valid_team (team : FootballTeam) : Prop :=
  team.total_players = 70 ∧
  team.throwers > 0 ∧
  team.throwers + team.left_handed + team.right_handed = team.total_players ∧
  team.left_handed = (team.total_players - team.throwers) / 3 ∧
  team.right_handed = team.throwers + 2 * team.left_handed ∧
  team.throwers + team.right_handed = 60

/-- Theorem stating that a valid team has 40 throwers -/
theorem throwers_count (team : FootballTeam) (h : valid_team team) : team.throwers = 40 := by
  sorry

end NUMINAMATH_CALUDE_throwers_count_l1239_123998


namespace NUMINAMATH_CALUDE_point_distance_inequality_l1239_123981

theorem point_distance_inequality (x : ℝ) : 
  (|x - 0| > |x - (-1)|) → x < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_inequality_l1239_123981


namespace NUMINAMATH_CALUDE_total_days_2010_to_2015_l1239_123950

/-- Calculate the total number of days from 2010 through 2015, inclusive. -/
theorem total_days_2010_to_2015 : 
  let years := 6
  let leap_years := 1
  let common_year_days := 365
  let leap_year_days := 366
  years * common_year_days + leap_years * (leap_year_days - common_year_days) = 2191 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2015_l1239_123950


namespace NUMINAMATH_CALUDE_right_triangle_with_consecutive_legs_and_hypotenuse_31_l1239_123952

theorem right_triangle_with_consecutive_legs_and_hypotenuse_31 :
  ∃ (a b : ℕ), 
    a + 1 = b ∧ 
    a^2 + b^2 = 31^2 ∧ 
    a + b = 43 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_consecutive_legs_and_hypotenuse_31_l1239_123952


namespace NUMINAMATH_CALUDE_overlapping_squares_diagonal_l1239_123935

theorem overlapping_squares_diagonal (small_side large_side : ℝ) 
  (h1 : small_side = 1) 
  (h2 : large_side = 7) : 
  Real.sqrt ((small_side + large_side)^2 + (large_side - small_side)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_diagonal_l1239_123935


namespace NUMINAMATH_CALUDE_max_sector_area_l1239_123903

/-- The maximum area of a sector with constant perimeter -/
theorem max_sector_area (α R C : ℝ) (h_pos_C : C > 0) : 
  let perimeter := 2 * R + α * R
  let area := (1 / 2) * α * R^2
  (perimeter = C) → (∀ α' R', 2 * R' + α' * R' = C → (1 / 2) * α' * R'^2 ≤ C^2 / 16) :=
by sorry

end NUMINAMATH_CALUDE_max_sector_area_l1239_123903


namespace NUMINAMATH_CALUDE_circle_max_min_sum_l1239_123911

theorem circle_max_min_sum (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S_max S_min : ℝ),
    (∀ x' y', (x' - 1)^2 + (y' + 2)^2 = 4 → 2*x' + y' ≤ S_max) ∧
    (∀ x' y', (x' - 1)^2 + (y' + 2)^2 = 4 → 2*x' + y' ≥ S_min) ∧
    S_max = 4 + 2*Real.sqrt 5 ∧
    S_min = 4 - 2*Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_min_sum_l1239_123911


namespace NUMINAMATH_CALUDE_friends_basketball_score_l1239_123924

theorem friends_basketball_score 
  (total_points : ℕ) 
  (edward_points : ℕ) 
  (h1 : total_points = 13) 
  (h2 : edward_points = 7) : 
  total_points - edward_points = 6 := by
  sorry

end NUMINAMATH_CALUDE_friends_basketball_score_l1239_123924


namespace NUMINAMATH_CALUDE_fourth_ball_black_probability_l1239_123941

theorem fourth_ball_black_probability 
  (total_balls : Nat) 
  (black_balls : Nat) 
  (red_balls : Nat) 
  (h1 : total_balls = black_balls + red_balls)
  (h2 : black_balls = 4)
  (h3 : red_balls = 4) :
  (black_balls : ℚ) / total_balls = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_ball_black_probability_l1239_123941


namespace NUMINAMATH_CALUDE_cannon_hit_probability_l1239_123971

theorem cannon_hit_probability (P1 P2 P3 : ℝ) : 
  P1 = 0.2 →
  P3 = 0.3 →
  (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997 →
  P2 = 0.5 := by
sorry

end NUMINAMATH_CALUDE_cannon_hit_probability_l1239_123971


namespace NUMINAMATH_CALUDE_digit_product_of_24_l1239_123961

theorem digit_product_of_24 :
  ∀ x y : ℕ,
  x < 10 ∧ y < 10 →  -- Ensures x and y are single digits
  10 * x + y = 24 →  -- The number is 24
  10 * x + y + 18 = 10 * y + x →  -- When 18 is added, digits are reversed
  x * y = 8 :=  -- Product of digits is 8
by
  sorry

end NUMINAMATH_CALUDE_digit_product_of_24_l1239_123961


namespace NUMINAMATH_CALUDE_acute_angles_trigonometry_l1239_123987

open Real

theorem acute_angles_trigonometry (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_tan_α : tan α = 2)
  (h_sin_diff : sin (α - β) = -sqrt 10 / 10) :
  sin (2 * α) = 4 / 5 ∧ tan (α + β) = -9 / 13 := by
sorry


end NUMINAMATH_CALUDE_acute_angles_trigonometry_l1239_123987


namespace NUMINAMATH_CALUDE_aaron_sheep_count_l1239_123937

theorem aaron_sheep_count (beth_sheep : ℕ) (aaron_sheep : ℕ) : 
  aaron_sheep = 7 * beth_sheep →
  aaron_sheep + beth_sheep = 608 →
  aaron_sheep = 532 := by
sorry

end NUMINAMATH_CALUDE_aaron_sheep_count_l1239_123937


namespace NUMINAMATH_CALUDE_inequality_proof_l1239_123965

theorem inequality_proof (a b : ℝ) (h : 1 / a < 1 / b ∧ 1 / b < 0) :
  (a + b < a * b) ∧ (b / a + a / b > 2) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1239_123965


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1239_123975

theorem arithmetic_calculation : 
  4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1239_123975


namespace NUMINAMATH_CALUDE_cubic_root_proof_l1239_123908

theorem cubic_root_proof :
  let x : ℝ := (Real.rpow 81 (1/3) + Real.rpow 9 (1/3) + 1) / 27
  27 * x^3 - 6 * x^2 - 6 * x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_proof_l1239_123908


namespace NUMINAMATH_CALUDE_apples_per_bucket_l1239_123939

theorem apples_per_bucket (total_apples : ℕ) (num_buckets : ℕ) 
  (h1 : total_apples = 56) (h2 : num_buckets = 7) :
  total_apples / num_buckets = 8 := by
sorry

end NUMINAMATH_CALUDE_apples_per_bucket_l1239_123939


namespace NUMINAMATH_CALUDE_symmetric_function_implies_a_eq_neg_four_l1239_123912

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- Theorem: If f(2-x) = f(2+x) for all x, then a = -4 -/
theorem symmetric_function_implies_a_eq_neg_four (a : ℝ) :
  (∀ x, f a (2 - x) = f a (2 + x)) → a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_implies_a_eq_neg_four_l1239_123912


namespace NUMINAMATH_CALUDE_rectangle_opposite_vertices_distance_sum_equal_l1239_123943

/-- Theorem: The sums of the squares of the distances from any point in space to opposite vertices of a rectangle are equal to each other. -/
theorem rectangle_opposite_vertices_distance_sum_equal 
  (a b x y z : ℝ) : 
  (x^2 + y^2 + z^2) + ((x - a)^2 + (y - b)^2 + z^2) = 
  ((x - a)^2 + y^2 + z^2) + (x^2 + (y - b)^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_opposite_vertices_distance_sum_equal_l1239_123943


namespace NUMINAMATH_CALUDE_packet_weight_problem_l1239_123986

/-- Given the weights of packets a, b, c, d, e, and f, prove that the weight of packet a is 75 kg. -/
theorem packet_weight_problem (a b c d e f : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  f = (a + e) / 2 →
  (b + c + d + e + f) / 5 = 81 →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_packet_weight_problem_l1239_123986


namespace NUMINAMATH_CALUDE_HG_ratio_l1239_123954

-- Define the equation
def equation (G H x : ℝ) : Prop :=
  (G / (x + 7) + H / (x^2 - 6*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 42*x))

-- State the theorem
theorem HG_ratio (G H : ℤ) : 
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 → equation G H x) →
  (H : ℝ) / (G : ℝ) = 15 / 7 :=
sorry

end NUMINAMATH_CALUDE_HG_ratio_l1239_123954


namespace NUMINAMATH_CALUDE_expression_evaluation_l1239_123994

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := 2
  2 * x^2 - y^2 + (2 * y^2 - 3 * x^2) - (2 * y^2 + x^2) = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1239_123994


namespace NUMINAMATH_CALUDE_original_number_proof_l1239_123973

theorem original_number_proof (x : ℝ) : x * 1.5 = 135 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1239_123973


namespace NUMINAMATH_CALUDE_special_function_is_even_l1239_123979

/-- A function satisfying the given functional equation -/
structure SpecialFunction (f : ℝ → ℝ) : Prop where
  functional_eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y
  nonzero_at_zero : f 0 ≠ 0

/-- The main theorem: if f is a SpecialFunction, then it is even -/
theorem special_function_is_even (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_even_l1239_123979


namespace NUMINAMATH_CALUDE_equation_solution_l1239_123970

theorem equation_solution : ∃! x : ℝ, (1/4 : ℝ)^(4*x + 12) = 16^(x + 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1239_123970


namespace NUMINAMATH_CALUDE_velocity_at_2s_l1239_123938

-- Define the displacement function
def S (t : ℝ) : ℝ := 10 * t - t^2

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 10 - 2 * t

-- Theorem statement
theorem velocity_at_2s :
  v 2 = 6 := by sorry

end NUMINAMATH_CALUDE_velocity_at_2s_l1239_123938


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l1239_123944

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (4, 2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Theorem statement
theorem circumcircle_of_triangle_ABC :
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  circle_equation C.1 C.2 ∧
  ∀ (x y : ℝ), circle_equation x y →
    (x - A.1)^2 + (y - A.2)^2 =
    (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 =
    (x - C.1)^2 + (y - C.2)^2 :=
sorry


end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l1239_123944


namespace NUMINAMATH_CALUDE_remainder_problem_l1239_123960

theorem remainder_problem (N : ℤ) (h : ∃ k : ℤ, N = 39 * k + 18) : 
  ∃ m : ℤ, N = 13 * m + 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1239_123960


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1239_123989

theorem fraction_to_decimal : (22 : ℚ) / 160 = (1375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1239_123989


namespace NUMINAMATH_CALUDE_cylinder_ellipse_eccentricity_l1239_123995

/-- The eccentricity of an ellipse formed by intersecting a cylinder with a plane -/
theorem cylinder_ellipse_eccentricity (d : ℝ) (θ : ℝ) (h_d : d = 12) (h_θ : θ = π / 6) :
  let r := d / 2
  let b := r
  let a := r / Real.cos θ
  let c := Real.sqrt (a^2 - b^2)
  c / a = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_eccentricity_l1239_123995


namespace NUMINAMATH_CALUDE_bisection_method_structures_l1239_123996

-- Define the function for which we're finding the root
def f (x : ℝ) := x^2 - 2

-- Define the bisection method structure
structure BisectionMethod where
  sequential : Bool
  conditional : Bool
  loop : Bool

-- Theorem statement
theorem bisection_method_structures :
  ∀ (ε : ℝ) (a b : ℝ), 
    ε > 0 → a < b → f a * f b < 0 →
    ∃ (m : BisectionMethod),
      m.sequential ∧ m.conditional ∧ m.loop ∧
      ∃ (x : ℝ), a ≤ x ∧ x ≤ b ∧ |f x| < ε :=
sorry

end NUMINAMATH_CALUDE_bisection_method_structures_l1239_123996


namespace NUMINAMATH_CALUDE_smith_cycling_time_comparison_l1239_123909

/-- Proves that the time taken for the second trip is 3/4 of the time taken for the first trip -/
theorem smith_cycling_time_comparison 
  (first_distance : ℝ) 
  (second_distance : ℝ) 
  (speed_multiplier : ℝ) 
  (h1 : first_distance = 90) 
  (h2 : second_distance = 270) 
  (h3 : speed_multiplier = 4) 
  (v : ℝ) 
  (hv : v > 0) : 
  (second_distance / (speed_multiplier * v)) / (first_distance / v) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_smith_cycling_time_comparison_l1239_123909


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l1239_123972

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

def molecularWeight (comp : CompoundComposition) (weights : AtomicWeights) : ℝ :=
  comp.carbon * weights.carbon + comp.hydrogen * weights.hydrogen + comp.oxygen * weights.oxygen

theorem compound_oxygen_atoms 
  (comp : CompoundComposition)
  (weights : AtomicWeights)
  (h1 : comp.carbon = 4)
  (h2 : comp.hydrogen = 8)
  (h3 : weights.carbon = 12.01)
  (h4 : weights.hydrogen = 1.008)
  (h5 : weights.oxygen = 16.00)
  (h6 : molecularWeight comp weights = 88) :
  comp.oxygen = 2 := by
sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l1239_123972


namespace NUMINAMATH_CALUDE_simplify_product_l1239_123900

theorem simplify_product (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l1239_123900


namespace NUMINAMATH_CALUDE_trig_simplification_l1239_123918

theorem trig_simplification :
  (1 + Real.cos (20 * π / 180)) / (2 * Real.sin (20 * π / 180)) -
  Real.sin (10 * π / 180) * ((1 / Real.tan (5 * π / 180)) - Real.tan (5 * π / 180)) =
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l1239_123918


namespace NUMINAMATH_CALUDE_solve_system_of_equations_no_solution_for_inequalities_l1239_123925

-- Part 1: System of equations
theorem solve_system_of_equations :
  ∃! (x y : ℝ), x - 3 * y = -5 ∧ 2 * x + 2 * y = 6 ∧ x = 1 ∧ y = 2 :=
by sorry

-- Part 2: System of inequalities
theorem no_solution_for_inequalities :
  ¬∃ (x : ℝ), 2 * x < -4 ∧ (1/2) * x - 5 > 1 - (3/2) * x :=
by sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_no_solution_for_inequalities_l1239_123925


namespace NUMINAMATH_CALUDE_shopping_mall_escalator_problem_l1239_123922

/-- Represents the escalator and staircase system in the shopping mall -/
structure EscalatorSystem where
  escalator_speed : ℝ
  a_step_rate : ℝ
  b_step_rate : ℝ
  a_steps_up : ℕ
  b_steps_up : ℕ

/-- Represents the result of the problem -/
structure ProblemResult where
  exposed_steps : ℕ
  catchup_location : Bool  -- true if on staircase, false if on escalator
  steps_walked : ℕ

/-- The main theorem that proves the result of the problem -/
theorem shopping_mall_escalator_problem (sys : EscalatorSystem) 
  (h1 : sys.a_step_rate = 2 * sys.b_step_rate)
  (h2 : sys.a_steps_up = 24)
  (h3 : sys.b_steps_up = 16) :
  ∃ (result : ProblemResult), 
    result.exposed_steps = 48 ∧ 
    result.catchup_location = true ∧ 
    result.steps_walked = 176 :=
by
  sorry

end NUMINAMATH_CALUDE_shopping_mall_escalator_problem_l1239_123922


namespace NUMINAMATH_CALUDE_max_triangles_for_three_families_of_ten_l1239_123983

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : Nat)

/-- Represents the configuration of three families of parallel lines -/
structure LineConfiguration :=
  (family1 : LineFamily)
  (family2 : LineFamily)
  (family3 : LineFamily)

/-- Calculates the maximum number of triangles formed by the given line configuration -/
def maxTriangles (config : LineConfiguration) : Nat :=
  sorry

/-- Theorem stating the maximum number of triangles formed by three families of 10 parallel lines each -/
theorem max_triangles_for_three_families_of_ten :
  ∃ (config : LineConfiguration),
    config.family1.count = 10 ∧
    config.family2.count = 10 ∧
    config.family3.count = 10 ∧
    maxTriangles config = 150 :=
  sorry

end NUMINAMATH_CALUDE_max_triangles_for_three_families_of_ten_l1239_123983


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l1239_123927

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_seq : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l1239_123927


namespace NUMINAMATH_CALUDE_saturday_price_of_200_dollar_coat_l1239_123946

/-- Calculates the Saturday price of a coat at Ajax Outlet Store -/
def saturday_price (original_price : ℝ) : ℝ :=
  let regular_discount_rate := 0.6
  let saturday_discount_rate := 0.3
  let price_after_regular_discount := original_price * (1 - regular_discount_rate)
  price_after_regular_discount * (1 - saturday_discount_rate)

/-- Theorem: The Saturday price of a coat with original price $200 is $56 -/
theorem saturday_price_of_200_dollar_coat :
  saturday_price 200 = 56 := by
  sorry

end NUMINAMATH_CALUDE_saturday_price_of_200_dollar_coat_l1239_123946


namespace NUMINAMATH_CALUDE_cosine_roots_of_equation_l1239_123956

theorem cosine_roots_of_equation : 
  let f (t : ℝ) := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3
  (f (Real.cos (6 * π / 180)) = 0) →
  (f (Real.cos (78 * π / 180)) = 0) ∧
  (f (Real.cos (150 * π / 180)) = 0) ∧
  (f (Real.cos (222 * π / 180)) = 0) ∧
  (f (Real.cos (294 * π / 180)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_cosine_roots_of_equation_l1239_123956


namespace NUMINAMATH_CALUDE_semicircle_area_problem_l1239_123948

/-- The area of the shaded region in the semicircle problem -/
theorem semicircle_area_problem (A B C D E F G : ℝ) : 
  A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G →
  B - A = 3 →
  C - B = 3 →
  D - C = 3 →
  E - D = 3 →
  F - E = 3 →
  G - F = 6 →
  let semicircle_area (d : ℝ) := π * d^2 / 8
  let total_small_area := semicircle_area (B - A) + semicircle_area (C - B) + 
                          semicircle_area (D - C) + semicircle_area (E - D) + 
                          semicircle_area (F - E) + semicircle_area (G - F)
  let large_semicircle_area := semicircle_area (G - A)
  large_semicircle_area - total_small_area = 225 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_problem_l1239_123948


namespace NUMINAMATH_CALUDE_software_package_savings_l1239_123934

/-- Calculates the savings when choosing a more expensive software package with higher device coverage over a cheaper one with lower device coverage. -/
theorem software_package_savings
  (total_devices : ℕ)
  (package1_price package2_price : ℕ)
  (package1_coverage package2_coverage : ℕ)
  (h1 : total_devices = 50)
  (h2 : package1_price = 40)
  (h3 : package2_price = 60)
  (h4 : package1_coverage = 5)
  (h5 : package2_coverage = 10) :
  (total_devices / package1_coverage * package1_price) -
  (total_devices / package2_coverage * package2_price) = 100 :=
by
  sorry

#check software_package_savings

end NUMINAMATH_CALUDE_software_package_savings_l1239_123934


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1239_123999

/-- The area of an equilateral triangle, given specific internal perpendiculars -/
theorem equilateral_triangle_area (a b c : ℝ) (h : a = 2 ∧ b = 3 ∧ c = 4) : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    (a + b + c) * side / 2 = side * (side * Real.sqrt 3 / 2) / 2 ∧
    side * (side * Real.sqrt 3 / 2) / 2 = 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1239_123999


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1239_123985

/-- A geometric sequence of positive real numbers -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 4) →
  (a 5 + a 6 = 16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1239_123985


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1239_123917

-- Define the type for dice rolls
def DiceRoll := Fin 6

-- Define the condition for the angle being greater than 90°
def angleGreaterThan90 (m n : DiceRoll) : Prop :=
  (m.val : ℤ) - (n.val : ℤ) > 0

-- Define the probability space
def totalOutcomes : ℕ := 36

-- Define the number of favorable outcomes
def favorableOutcomes : ℕ := 15

-- State the theorem
theorem dice_roll_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1239_123917


namespace NUMINAMATH_CALUDE_total_renovation_time_is_79_5_l1239_123963

/-- Represents the renovation time for a house with specific room conditions. -/
def house_renovation_time (bedroom_time : ℝ) (bedroom_count : ℕ) (garden_time : ℝ) : ℝ :=
  let kitchen_time := 1.5 * bedroom_time
  let terrace_time := garden_time - 2
  let basement_time := 0.75 * kitchen_time
  let non_living_time := bedroom_time * bedroom_count + kitchen_time + garden_time + terrace_time + basement_time
  non_living_time + 2 * non_living_time

/-- Theorem stating that the total renovation time for the given house is 79.5 hours. -/
theorem total_renovation_time_is_79_5 :
  house_renovation_time 4 3 3 = 79.5 := by
  sorry

#eval house_renovation_time 4 3 3

end NUMINAMATH_CALUDE_total_renovation_time_is_79_5_l1239_123963


namespace NUMINAMATH_CALUDE_total_tickets_bought_l1239_123928

/-- Represents the cost of an adult ticket in dollars -/
def adult_ticket_cost : ℚ := 5.5

/-- Represents the cost of a child ticket in dollars -/
def child_ticket_cost : ℚ := 3.5

/-- Represents the total cost of all tickets bought in dollars -/
def total_cost : ℚ := 83.5

/-- Represents the number of children's tickets bought -/
def num_child_tickets : ℕ := 16

/-- Theorem stating that the total number of tickets bought is 21 -/
theorem total_tickets_bought : ℕ := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_bought_l1239_123928


namespace NUMINAMATH_CALUDE_total_watching_time_l1239_123929

/-- Calculates the total watching time for two people watching multiple videos at different speeds -/
theorem total_watching_time
  (video_length : ℝ)
  (num_videos : ℕ)
  (speed_ratio_1 : ℝ)
  (speed_ratio_2 : ℝ)
  (h1 : video_length = 100)
  (h2 : num_videos = 6)
  (h3 : speed_ratio_1 = 2)
  (h4 : speed_ratio_2 = 1) :
  (num_videos * video_length / speed_ratio_1) + (num_videos * video_length / speed_ratio_2) = 900 := by
  sorry

#check total_watching_time

end NUMINAMATH_CALUDE_total_watching_time_l1239_123929


namespace NUMINAMATH_CALUDE_sqrt_sum_diff_complex_expression_system_of_equations_l1239_123953

-- Problem 1
theorem sqrt_sum_diff (a b c : ℝ) (ha : a = 3) (hb : b = 27) (hc : c = 12) :
  Real.sqrt a + Real.sqrt b - Real.sqrt c = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem complex_expression (a b c d e : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 20) (hd : d = 15) (he : e = 5) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) - (Real.sqrt c - Real.sqrt d) / Real.sqrt e = Real.sqrt 3 - 1 := by sorry

-- Problem 3
theorem system_of_equations (x y : ℝ) (h1 : 2 * (x + 1) - y = 6) (h2 : x = y - 1) :
  x = 5 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_diff_complex_expression_system_of_equations_l1239_123953


namespace NUMINAMATH_CALUDE_car_travel_time_l1239_123936

theorem car_travel_time (actual_speed : ℝ) (actual_time : ℝ) 
  (h1 : actual_speed > 0) 
  (h2 : actual_time > 0) 
  (h3 : actual_speed * actual_time = (4/5 * actual_speed) * (actual_time + 15)) : 
  actual_time = 60 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l1239_123936


namespace NUMINAMATH_CALUDE_f_max_min_implies_m_range_l1239_123905

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- Theorem: If the maximum value of f(x) in [0,m] is 2 and the minimum is 1, then 1 ≤ m ≤ 2 -/
theorem f_max_min_implies_m_range (m : ℝ) 
  (h_max : ∀ x ∈ Set.Icc 0 m, f x ≤ 2) 
  (h_min : ∃ x ∈ Set.Icc 0 m, f x = 1) 
  (h_reaches_max : ∃ x ∈ Set.Icc 0 m, f x = 2) : 
  1 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_implies_m_range_l1239_123905


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1239_123997

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1239_123997
